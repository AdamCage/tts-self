#!/usr/bin/env python3
"""Probe Z4: backbone memory / latency profile (Track N / N09).

Measures peak GPU memory (bf16 + quantized variants) and forward latency on
typical token lengths from bank_entity / val100. Short-sequence MoE behaviour
is probed for MoE candidates.

Example (WSL)::

    cd ~/projects/tts-self && source .venv/bin/activate
    export HF_HOME=$HOME/hf_cache
    python scripts/42_probe_backbone.py \\
      --out out/probe_backbone_profile.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_MODELS = [
    "Qwen/Qwen3.5-0.8B-Base",
    "ai-sage/GigaChat3.1-10B-A1.8B",
]

# GigaChat Hub defaults to FP8; BF16 lives in a sibling repo.
GIGACHAT_BF16_ID = "ai-sage/GigaChat3.1-10B-A1.8B-bf16"
GIGACHAT_FP8_ID = "ai-sage/GigaChat3.1-10B-A1.8B"

# Refuse download if free space is below this (GiB). 10B bf16 ~22 GiB + margin.
MIN_FREE_GB_FOR_10B = 40.0
MIN_FREE_GB_FOR_SMALL = 5.0

SHORT_SEQ_LENS = [1, 4, 8, 16, 32]


def _disk_report(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path)}
    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
        out["total_gb"] = round(usage.total / (1024**3), 2)
        out["used_gb"] = round(usage.used / (1024**3), 2)
        out["free_gb"] = round(usage.free / (1024**3), 2)
    except OSError as exc:
        out["error"] = str(exc)
    try:
        proc = subprocess.run(
            ["df", "-h", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        out["df_h"] = (proc.stdout or proc.stderr or "").strip()
    except OSError as exc:
        out["df_h_error"] = str(exc)
    return out


def _nvidia_smi_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        out["gpu"] = (proc.stdout or "").strip()
        proc2 = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory,process_name",
                "--format=csv",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        out["compute_apps"] = (proc2.stdout or "").strip()
    except OSError as exc:
        out["error"] = str(exc)
    return out


def _gpu_busy(snapshot: dict[str, Any]) -> bool:
    apps = snapshot.get("compute_apps") or ""
    lines = [ln for ln in apps.splitlines() if ln.strip()]
    # header + any compute process
    if len(lines) <= 1:
        return False
    # Ignore empty pid rows
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(",")]
        if parts and parts[0] and parts[0] != "[N/A]":
            return True
    return False


def _wait_for_gpu(timeout_s: int = 7200, poll_s: int = 30) -> dict[str, Any]:
    t0 = time.time()
    while True:
        snap = _nvidia_smi_snapshot()
        if not _gpu_busy(snap):
            return {"waited_s": round(time.time() - t0, 1), "snapshot": snap}
        if time.time() - t0 > timeout_s:
            return {
                "waited_s": round(time.time() - t0, 1),
                "snapshot": snap,
                "timeout": True,
            }
        print(f"GPU busy; sleeping {poll_s}s ...", flush=True)
        time.sleep(poll_s)


def _percentile(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return int(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return int(sorted_vals[int(k)])
    return int(round(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)))


def _load_eval_set_texts(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    kind = data.get("kind") or "phrases"
    texts: list[str] = []
    if kind == "phrases":
        for item in data.get("items") or []:
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
        return texts
    # utt_ids — fill from manifest later
    return []


def _load_val100_texts(val_path: Path, manifest_path: Path) -> list[str]:
    data = json.loads(val_path.read_text(encoding="utf-8"))
    utt_ids: list[str] = []
    if data.get("items"):
        for item in data["items"]:
            uid = item.get("utt_id")
            if uid:
                utt_ids.append(str(uid))
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                # already have text
                pass
    else:
        utt_ids = [str(u) for u in (data.get("utt_ids") or [])]

    # Prefer embedded texts if present
    embedded: list[str] = []
    for item in data.get("items") or []:
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            embedded.append(text)
    if embedded and len(embedded) == len(utt_ids):
        return embedded

    by_id: dict[str, str] = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                uid = str(row.get("utt_id") or row.get("id") or "")
                if not uid:
                    continue
                text = row.get("text_stressed") or row.get("text_raw") or row.get("text")
                if isinstance(text, str) and text.strip():
                    by_id[uid] = text
    texts: list[str] = []
    for uid in utt_ids:
        if uid in by_id:
            texts.append(by_id[uid])
    return texts


def _resolve_local_qwen(model_id: str) -> str:
    local = Path.home() / "datasets/tts/models/Qwen3.5-0.8B-Base"
    if "Qwen3.5-0.8B" in model_id and local.exists():
        return str(local)
    return model_id


def _tokenizer_path_for(model_id: str) -> str:
    if "GigaChat" in model_id:
        tok_only = (
            Path.home()
            / "hf_cache/tokenizers_only/ai-sage__GigaChat3.1-10B-A1.8B"
        )
        if tok_only.exists() and (tok_only / "tokenizer.json").exists():
            return str(tok_only)
    return _resolve_local_qwen(model_id)


def _patch_config_floats(config_path: Path) -> dict[str, Any]:
    """Coerce int fields that transformers≥5 StrictDataclass rejects as float."""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    patched_keys: list[str] = []
    for key in (
        "routed_scaling_factor",
        "attention_dropout",
        "initializer_range",
        "rms_norm_eps",
    ):
        if key in cfg and isinstance(cfg[key], int):
            cfg[key] = float(cfg[key])
            patched_keys.append(key)
    rope = cfg.get("rope_scaling")
    if isinstance(rope, dict):
        for k, v in list(rope.items()):
            if isinstance(v, int) and k not in {
                "original_max_position_embeddings",
            }:
                rope[k] = float(v)
                patched_keys.append(f"rope_scaling.{k}")
        cfg["rope_scaling"] = rope
    if patched_keys:
        config_path.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {"patched_keys": patched_keys, "path": str(config_path)}


def _ensure_snapshot(
    repo_id: str,
    *,
    free_gb: float | None,
    min_free_gb: float,
    allow_download: bool,
) -> dict[str, Any]:
    """Ensure Hub weights are local. Returns path meta or skip reason."""
    meta: dict[str, Any] = {
        "repo_id": repo_id,
        "path": None,
        "downloaded": False,
        "error": None,
        "ok": False,
    }
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        meta["error"] = f"huggingface_hub missing: {exc}"
        return meta

    # Prefer local Qwen path without Hub
    if "Qwen3.5-0.8B" in repo_id:
        local = Path.home() / "datasets/tts/models/Qwen3.5-0.8B-Base"
        if local.exists() and list(local.glob("*.safetensors")):
            meta["path"] = str(local)
            meta["ok"] = True
            meta["source"] = "local_datasets"
            return meta

    if isinstance(free_gb, (int, float)) and free_gb < min_free_gb:
        meta["error"] = (
            f"insufficient disk: free_gb={free_gb} < min_free_gb={min_free_gb}; "
            "refusing download"
        )
        return meta

    if not allow_download:
        meta["error"] = "download disabled"
        return meta

    try:
        path = snapshot_download(
            repo_id,
            local_files_only=False,
        )
        meta["path"] = path
        meta["downloaded"] = True
        meta["ok"] = True
        meta["source"] = "snapshot_download"
        # Patch config in place for GigaChat / deepseek_v3 float validation
        cfg = Path(path) / "config.json"
        if cfg.exists() and "GigaChat" in repo_id:
            meta["config_patch"] = _patch_config_floats(cfg)
        return meta
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return meta


def _cuda_mem_mb() -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_allocated_mb": 0.0}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
        "peak_allocated_mb": round(
            torch.cuda.max_memory_allocated() / (1024**2), 2
        ),
        "peak_reserved_mb": round(
            torch.cuda.max_memory_reserved() / (1024**2), 2
        ),
    }


def _cleanup_cuda() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def _load_tokenizer(model_id: str) -> tuple[Any | None, dict[str, Any]]:
    from transformers import AutoTokenizer

    meta: dict[str, Any] = {"requested_id": model_id, "path": None, "ok": False}
    path = _tokenizer_path_for(model_id)
    meta["path"] = path
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        meta["ok"] = True
        meta["class_name"] = type(tok).__name__
        return tok, meta
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return None, meta


def _token_lengths(tokenizer: Any, texts: list[str]) -> list[int]:
    lengths: list[int] = []
    for text in texts:
        enc = tokenizer(text, add_special_tokens=True, return_tensors=None)
        ids = enc["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        lengths.append(len(ids))
    return lengths


def _length_stats(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"n": 0}
    s = sorted(lengths)
    return {
        "n": len(s),
        "min": s[0],
        "max": s[-1],
        "mean": round(statistics.mean(s), 3),
        "p50": _percentile(s, 50),
        "p90": _percentile(s, 90),
        "p95": _percentile(s, 95),
        "p99": _percentile(s, 99),
    }


def _make_batch(tokenizer: Any, seq_len: int, device: str) -> dict[str, Any]:
    import torch

    # Use pad/eos/unk as filler tokens for synthetic lengths
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    ids = torch.full((1, seq_len), int(pad_id), dtype=torch.long, device=device)
    # Put bos at start if available
    bos = tokenizer.bos_token_id
    if bos is not None and seq_len > 0:
        ids[0, 0] = int(bos)
    attn = torch.ones_like(ids)
    return {"input_ids": ids, "attention_mask": attn}


def _forward_once(model: Any, batch: dict[str, Any]) -> Any:
    import torch

    with torch.inference_mode():
        return model(**batch, use_cache=False)


def _benchmark_latency(
    model: Any,
    tokenizer: Any,
    seq_lens: list[int],
    *,
    device: str,
    warmup: int = 2,
    repeats: int = 5,
) -> dict[str, Any]:
    import torch

    results: dict[str, Any] = {}
    for L in seq_lens:
        entry: dict[str, Any] = {"seq_len": L}
        try:
            batch = _make_batch(tokenizer, L, device)
            # Warmup
            for _ in range(warmup):
                _ = _forward_once(model, batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times: list[float] = []
            for _ in range(repeats):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = _forward_once(model, batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            entry["latency_s"] = {
                "mean": round(statistics.mean(times), 6),
                "stdev": round(statistics.stdev(times), 6) if len(times) > 1 else 0.0,
                "min": round(min(times), 6),
                "max": round(max(times), 6),
                "n": len(times),
                "samples": [round(t, 6) for t in times],
            }
            entry["latency_ms_mean"] = round(statistics.mean(times) * 1000.0, 3)
            # MoE / logits shape sanity
            logits = getattr(out, "logits", None)
            if logits is not None:
                entry["logits_shape"] = list(logits.shape)
            entry["ok"] = True
        except Exception as exc:  # noqa: BLE001
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results[str(L)] = entry
    return results


def _probe_moe_short(
    model: Any,
    tokenizer: Any,
    *,
    device: str,
) -> dict[str, Any]:
    """Short-sequence behaviour (relevant for MoE routing)."""
    import torch

    out: dict[str, Any] = {
        "is_moe_hint": bool(
            getattr(getattr(model, "config", None), "n_routed_experts", None)
            or getattr(getattr(model, "config", None), "num_experts_per_tok", None)
        ),
        "config": {},
        "short_forward": {},
    }
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for key in (
            "model_type",
            "n_routed_experts",
            "num_experts_per_tok",
            "n_shared_experts",
            "moe_layer_freq",
            "first_k_dense_replace",
            "num_hidden_layers",
            "hidden_size",
        ):
            if hasattr(cfg, key):
                out["config"][key] = getattr(cfg, key)

    for L in SHORT_SEQ_LENS:
        entry: dict[str, Any] = {"seq_len": L}
        try:
            batch = _make_batch(tokenizer, L, device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = _forward_once(model, batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            entry["ok"] = True
            entry["latency_ms"] = round(dt * 1000.0, 3)
            logits = getattr(result, "logits", None)
            if logits is not None:
                entry["logits_shape"] = list(logits.shape)
                # Finite check — NaNs sometimes appear on degenerate MoE routes
                entry["logits_finite"] = bool(torch.isfinite(logits).all().item())
                entry["logits_abs_mean"] = round(float(logits.abs().mean().item()), 6)
        except Exception as exc:  # noqa: BLE001
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        out["short_forward"][str(L)] = entry
    return out


def _count_params(model: Any) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "total_params_b": round(total / 1e9, 4),
    }


def _load_model_variant(
    *,
    weights_path: str,
    variant: str,
    trust_remote_code: bool = True,
) -> tuple[Any | None, dict[str, Any]]:
    """Load one precision/quantization variant. Returns (model, meta)."""
    import torch
    from transformers import AutoModelForCausalLM

    meta: dict[str, Any] = {
        "variant": variant,
        "weights_path": weights_path,
        "ok": False,
        "error": None,
        "load_s": None,
        "device_map": None,
        "dtype": None,
        "mem_after_load": None,
        "peak_mem_after_load": None,
    }
    _cleanup_cuda()
    t0 = time.perf_counter()
    try:
        if variant == "bf16":
            meta["dtype"] = "bfloat16"
            meta["device_map"] = {"": 0}
            model = AutoModelForCausalLM.from_pretrained(
                weights_path,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=trust_remote_code,
            )
        elif variant == "fp8_native":
            # Hub FP8 checkpoint (GigaChat default). Let transformers pick dtype.
            meta["dtype"] = "fp8_native_or_auto"
            meta["device_map"] = {"": 0}
            model = AutoModelForCausalLM.from_pretrained(
                weights_path,
                device_map={"": 0},
                trust_remote_code=trust_remote_code,
            )
        elif variant == "bf16_auto_offload":
            # Allow CPU offload when 16 GB is insufficient for full bf16.
            meta["dtype"] = "bfloat16"
            meta["device_map"] = "auto"
            max_mem = {0: "14GiB", "cpu": "48GiB"}
            meta["max_memory"] = max_mem
            model = AutoModelForCausalLM.from_pretrained(
                weights_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_mem,
                trust_remote_code=trust_remote_code,
            )
        elif variant in {"int8_bnb", "int4_bnb"}:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:  # noqa: BLE001
                meta["error"] = f"BitsAndBytesConfig unavailable: {exc}"
                return None, meta
            try:
                import bitsandbytes as bnb  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                meta["error"] = f"bitsandbytes missing: {exc}"
                return None, meta
            if variant == "int8_bnb":
                qcfg = BitsAndBytesConfig(load_in_8bit=True)
                meta["dtype"] = "int8_bnb"
            else:
                qcfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
                meta["dtype"] = "int4_nf4_bnb"
            meta["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(
                weights_path,
                quantization_config=qcfg,
                device_map="auto",
                trust_remote_code=trust_remote_code,
            )
        else:
            meta["error"] = f"unknown variant {variant!r}"
            return None, meta

        meta["load_s"] = round(time.perf_counter() - t0, 3)
        meta["ok"] = True
        meta["mem_after_load"] = _cuda_mem_mb()
        meta["peak_mem_after_load"] = {
            "peak_allocated_mb": meta["mem_after_load"].get("peak_allocated_mb"),
            "peak_reserved_mb": meta["mem_after_load"].get("peak_reserved_mb"),
        }
        meta["params"] = _count_params(model)
        # Device of first parameter
        try:
            meta["first_param_device"] = str(next(model.parameters()).device)
        except StopIteration:
            meta["first_param_device"] = None
        return model, meta
    except Exception as exc:  # noqa: BLE001
        meta["load_s"] = round(time.perf_counter() - t0, 3)
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["mem_at_failure"] = _cuda_mem_mb()
        _cleanup_cuda()
        return None, meta


def _infer_device(model: Any) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cuda:0"


def _variants_for(model_id: str) -> list[tuple[str, str]]:
    """Return list of (weights_repo_or_path_key, variant_name).

    weights key is a logical id resolved later.
    """
    if "GigaChat" in model_id:
        return [
            (GIGACHAT_BF16_ID, "bf16"),
            (GIGACHAT_BF16_ID, "bf16_auto_offload"),
            (GIGACHAT_FP8_ID, "fp8_native"),
            (GIGACHAT_BF16_ID, "int8_bnb"),
            (GIGACHAT_BF16_ID, "int4_bnb"),
        ]
    # Small dense baseline
    return [
        (model_id, "bf16"),
        (model_id, "int8_bnb"),
        (model_id, "int4_bnb"),
    ]


def run_probe(
    *,
    models: list[str],
    bank_path: Path,
    val_path: Path,
    manifest_path: Path,
    out_path: Path,
    home_path: Path,
    allow_download: bool,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    disk = _disk_report(home_path)
    print(json.dumps({"disk_before": disk}, ensure_ascii=False, indent=2), flush=True)
    free_gb = disk.get("free_gb")
    if isinstance(free_gb, (int, float)) and free_gb < MIN_FREE_GB_FOR_SMALL:
        raise SystemExit(
            f"Refusing to proceed: only {free_gb} GiB free under {home_path}."
        )

    wait = _wait_for_gpu()
    if wait.get("timeout"):
        raise SystemExit(f"GPU still busy after wait: {wait}")

    import torch

    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if torch.cuda.is_available()
        else None
    )

    bank_texts = _load_eval_set_texts(bank_path)
    val_texts = _load_val100_texts(val_path, manifest_path)

    result: dict[str, Any] = {
        "_probe_meta": {
            "task": "N09",
            "probe": "Z4_backbone_profile",
            "disk_before": disk,
            "gpu_wait": wait,
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "models_requested": models,
            "sets": {
                "bank_entity": {"n_texts": len(bank_texts), "path": str(bank_path)},
                "val100": {"n_texts": len(val_texts), "path": str(val_path)},
            },
            "downloads": {},
            "note": (
                "Peak memory = CUDA max_memory_allocated after load + one forward. "
                "GigaChat Hub default is FP8; BF16 is sibling repo "
                f"{GIGACHAT_BF16_ID}."
            ),
        }
    }

    # Pre-resolve / download weight snapshots (once per unique repo)
    repos_needed: set[str] = set()
    for mid in models:
        for repo, _var in _variants_for(mid):
            repos_needed.add(repo)

    for repo in sorted(repos_needed):
        min_free = (
            MIN_FREE_GB_FOR_10B if "GigaChat" in repo or "10B" in repo else MIN_FREE_GB_FOR_SMALL
        )
        # Refresh free space before each potential download
        disk_now = _disk_report(home_path)
        free_now = disk_now.get("free_gb")
        print(f"=== ensure weights: {repo} (free_gb={free_now})", flush=True)
        snap = _ensure_snapshot(
            repo,
            free_gb=float(free_now) if isinstance(free_now, (int, float)) else None,
            min_free_gb=min_free,
            allow_download=allow_download,
        )
        result["_probe_meta"]["downloads"][repo] = {
            **snap,
            "disk_at_ensure": disk_now,
        }
        print(json.dumps(snap, ensure_ascii=False), flush=True)

    for model_id in models:
        print(f"\n======== candidate: {model_id}", flush=True)
        tok, tok_meta = _load_tokenizer(model_id)
        block: dict[str, Any] = {
            "tokenizer": tok_meta,
            "length_stats": {},
            "variants": {},
            "measured": False,
        }
        if tok is None:
            block["skipped"] = True
            block["reason"] = tok_meta.get("error")
            result[model_id] = block
            continue

        bank_lens = _token_lengths(tok, bank_texts) if bank_texts else []
        val_lens = _token_lengths(tok, val_texts) if val_texts else []
        block["length_stats"] = {
            "bank_entity": _length_stats(bank_lens),
            "val100": _length_stats(val_lens),
        }
        # Typical lengths for latency: union of p50/p90/p99 (+ short MoE lens separately)
        typical: list[int] = []
        for stats in block["length_stats"].values():
            for key in ("p50", "p90", "p99", "max"):
                v = stats.get(key)
                if isinstance(v, int) and v > 0:
                    typical.append(v)
        typical = sorted(set(typical))
        if not typical:
            typical = [32, 64, 128]
        block["typical_seq_lens"] = typical

        for repo, variant in _variants_for(model_id):
            dl = result["_probe_meta"]["downloads"].get(repo) or {}
            weights_path = dl.get("path")
            vkey = f"{repo}::{variant}"
            print(f"--- variant {vkey}", flush=True)
            if not dl.get("ok") or not weights_path:
                block["variants"][vkey] = {
                    "variant": variant,
                    "repo_id": repo,
                    "ok": False,
                    "skipped": True,
                    "reason": dl.get("error") or "weights unavailable",
                }
                continue

            model, load_meta = _load_model_variant(
                weights_path=str(weights_path),
                variant=variant,
            )
            entry: dict[str, Any] = {
                "variant": variant,
                "repo_id": repo,
                "load": load_meta,
            }
            if model is None:
                entry["ok"] = False
                entry["reason"] = load_meta.get("error")
                block["variants"][vkey] = entry
                print(f"LOAD FAIL: {load_meta.get('error')}", flush=True)
                continue

            device = _infer_device(model)
            # Prefer CUDA device string for synthetic batches
            if device.startswith("cpu") and torch.cuda.is_available():
                # Offloaded model: inputs on cuda:0 (HF usually handles)
                batch_device = "cuda:0"
            else:
                batch_device = device

            try:
                _cleanup_cuda()
                # One forward to establish working peak
                batch = _make_batch(
                    tok, max(typical[0], 8) if typical else 8, batch_device
                )
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                _ = _forward_once(model, batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                entry["mem_after_one_forward"] = _cuda_mem_mb()
                entry["latency"] = _benchmark_latency(
                    model,
                    tok,
                    typical,
                    device=batch_device,
                    warmup=warmup,
                    repeats=repeats,
                )
                entry["moe_short"] = _probe_moe_short(
                    model, tok, device=batch_device
                )
                entry["ok"] = True
                block["measured"] = True
            except Exception as exc:  # noqa: BLE001
                entry["ok"] = False
                entry["reason"] = f"benchmark: {type(exc).__name__}: {exc}"
                entry["mem_at_failure"] = _cuda_mem_mb()
            finally:
                del model
                _cleanup_cuda()

            block["variants"][vkey] = entry
            # Persist incrementally so a later OOM still leaves numbers
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result[model_id] = block
            out_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        if not block["measured"]:
            # Aggregate skip reason
            reasons = []
            for v in block["variants"].values():
                if not v.get("ok"):
                    reasons.append(v.get("reason") or v.get("load", {}).get("error"))
            block["skipped"] = True
            block["reason"] = " | ".join(r for r in reasons if r) or "no variant measured"

        result[model_id] = block

    result["_probe_meta"]["disk_after"] = _disk_report(home_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model ids",
    )
    p.add_argument(
        "--bank",
        default="configs/eval_sets/bank_entity.json",
    )
    p.add_argument(
        "--val100",
        default="configs/eval_sets/val100.json",
    )
    p.add_argument(
        "--manifest",
        default=str(Path.home() / "datasets/tts/ruslan/manifest.jsonl"),
    )
    p.add_argument(
        "--out",
        default="out/probe_backbone_profile.json",
    )
    p.add_argument(
        "--home",
        default=str(Path.home()),
        help="Path for df -h free-space check (default: ~)",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download missing weights; record skip reasons",
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    run_probe(
        models=models,
        bank_path=Path(args.bank),
        val_path=Path(args.val100),
        manifest_path=Path(args.manifest),
        out_path=Path(args.out),
        home_path=Path(args.home),
        allow_download=not args.no_download,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
