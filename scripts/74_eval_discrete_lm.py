#!/usr/bin/env python3
"""Eval discrete Speech LM: teacher-forced CE + free-running ASR / N2 gate.

Promotion (Wave 3 / N2) uses free-running CER vs copy-synth topline, bootstrap
separation from wrong-utt / random baselines, CB0 health, and ASR text-usage.

D1b (§14) adds macro/micro CER, catastrophic rows, duration-ratio / stop-reason
distributions, and stricter text-usage / stability acceptance helpers.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from minispeech.data.codec_seq_protocol import resolve_collate_codec_seq
from minispeech.data.d1b_oracles import (
    DEFAULT_EVAL_CONTRACT,
    PLAN_SLUG,
    enforce_eval_contract,
    eval_contract_content_sha256,
    load_eval_contract,
    load_frozen_utt_ids,
)
from minispeech.data.speaker_registry import SpeakerRegistry
from minispeech.eval.asr import load_asr_engine
from minispeech.eval.asr_wer import compute_cer
from minispeech.eval.bootstrap import edits_from_jiwer_chars
from minispeech.eval.discrete_n2 import (
    FR_CANDIDATE_STEPS_BASE,
    bootstrap_mean_cer,
    cb0_stats,
    enrich_model_report,
    estimate_stop_reason,
    evaluate_n2_gate_d1b,
    evaluate_n2_gate_d1c,
    resolve_infer_settings,
    stability_acceptance,
    text_usage_acceptance,
)
from minispeech.eval.discrete_report_schema import (
    CONDITIONS as V2_CONDITIONS,
    build_formal_report,
    ordered_utt_ids_sha256,
)
from minispeech.frontend.normalize import normalize_for_asr
from minispeech.models.discrete_speech_lm import (
    DiscreteGenerationOutput,
    build_discrete_speech_lm,
)
from minispeech.utils import expand, git_is_dirty, git_sha, load_yaml, set_seed
from minispeech.frontend.normalize import normalize_for_asr
from minispeech.infer.discrete_config import GENERATE_INFER_KEYS, SEMANTIC_EOS_INFER_DEFAULTS, wave3c_infer_defaults
from minispeech.models.discrete_speech_lm import (
    DiscreteGenerationOutput,
    build_discrete_speech_lm,
)
from minispeech.utils import expand, git_is_dirty, git_sha, load_yaml, set_seed

_REQUIRED_N2_SCORE_REPORTS = (
    "model",
    "copy_synth",
    "wrong_utt",
    "random_codes",
    "shuffle_text",
    "mask_text",
)


def emit_d1c_v2_report(
    *,
    out_path: Path,
    checkpoint: dict[str, Any],
    provenance: dict[str, Any],
    infer: dict[str, Any],
    teacher_forced_metrics: dict[str, Any],
    ordered_utt_ids: list[str],
    conditions: dict[str, Any],
    summaries: dict[str, Any],
    gates: dict[str, Any],
    formal_eligible: bool | None = None,
) -> dict[str, Any]:
    """Write the selector-consumable formal v2 report after score-only checks.

    Callers supply exact row artifacts from decoding; unlike D1b's aggregate
    reports, this path cannot synthesize rows or copy train-loss TF CE.
    """
    if set(conditions) != set(V2_CONDITIONS):
        raise ValueError("D1c v2 report requires all six canonical conditions")
    for name, condition in conditions.items():
        condition.setdefault("expected_count", len(ordered_utt_ids))
        condition.setdefault("evaluated_count", len(condition.get("rows") or []))
        condition.setdefault("ordered_utt_ids_sha256", ordered_utt_ids_sha256(ordered_utt_ids))
    teacher_forced_metrics = dict(teacher_forced_metrics)
    source = teacher_forced_metrics.get("teacher_forced_ce_source")
    if source is None:
        raise ValueError(
            "teacher_forced_ce_source must be set by the caller "
            "(eval_computed | decode_bundle | unavailable); refusing fabricated provenance"
        )
    # EOS TF metrics must be measured or explicitly marked unavailable — never
    # silently default to passing values (1.0 / 0).
    eos_acc = teacher_forced_metrics.get("tf_eos_final_top1_accuracy")
    eos_fp = teacher_forced_metrics.get("tf_eos_early_top1_false_positive_count")
    eos_measured = (
        isinstance(eos_acc, (int, float))
        and math.isfinite(float(eos_acc))
        and eos_fp is not None
        and str(eos_fp) != "unavailable"
    )
    if formal_eligible is None:
        formal_eligible = bool(
            source == "eval_computed"
            and eos_measured
            and all(
                (conditions[name].get("rows"))
                and all(r.get("ordinal") is not None for r in conditions[name]["rows"])
                for name in V2_CONDITIONS
            )
        )
    if not formal_eligible:
        raise ValueError(
            "formal D1c report refused: formal_eligible=False "
            f"(teacher_forced_ce_source={source!r}, eos_measured={eos_measured})"
        )
    if formal_eligible and not eos_measured:
        raise ValueError(
            "formal D1c report refused: tf_eos_final_top1_accuracy / "
            "tf_eos_early_top1_false_positive_count were not measured"
        )
    report = build_formal_report(
        formal_eligible=True,
        checkpoint=checkpoint,
        provenance=provenance,
        infer=infer,
        teacher_forced_metrics=teacher_forced_metrics,
        row_binding={
            "ordered_utt_ids": ordered_utt_ids,
            "ordered_utt_ids_sha256": ordered_utt_ids_sha256(ordered_utt_ids),
        },
        conditions=conditions,
        summaries=summaries,
        gates=gates,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


_CANDIDATE_STEP_NAME_RE = re.compile(r"candidate_step_(\d+)\.pt$", re.IGNORECASE)

# Trainer-stamped run / recipe / contract digests required on the D1c formal path.
_REQUIRED_CHECKPOINT_HASH_KEYS = (
    "eval_contract",
    "recipe",
    "recipe_source_manifest",
    "run_manifest",
)


def parse_candidate_step_from_path(ckpt: Path | str) -> int | None:
    """Parse immutable ``candidate_step_XXXX.pt`` filename; else ``None``."""
    match = _CANDIDATE_STEP_NAME_RE.search(Path(ckpt).name)
    return int(match.group(1)) if match else None


def checkpoint_step_from_payload(payload: Mapping[str, Any] | None) -> int | None:
    """Read trainer-stamped step from a checkpoint payload."""
    if not isinstance(payload, Mapping):
        return None
    for key in ("step", "optimizer_step", "global_step"):
        raw = payload.get(key)
        if raw is not None:
            return int(raw)
    return None


def checkpoint_hashes_from_payload(
    payload: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    """Extract trainer-stamped run/recipe/contract hashes, or ``None`` if incomplete."""
    if not isinstance(payload, Mapping):
        return None
    raw = payload.get("hashes")
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, str] = {}
    for key in _REQUIRED_CHECKPOINT_HASH_KEYS:
        value = raw.get(key)
        if value is None or not str(value).strip() or str(value) == ("0" * 64):
            return None
        out[key] = str(value)
    return out


def require_checkpoint_payload_hashes(
    payload: Mapping[str, Any] | None,
    *,
    expected_eval_contract: str | None = None,
) -> dict[str, str]:
    """Fail-closed: formal D1c requires payload['hashes'] bound to the sealed run."""
    hashes = checkpoint_hashes_from_payload(payload)
    if hashes is None:
        raise SystemExit(
            "formal D1c requires payload['hashes'] with eval_contract/recipe/"
            "recipe_source_manifest/run_manifest stamped by the trainer"
        )
    if expected_eval_contract and hashes["eval_contract"] != str(expected_eval_contract):
        raise SystemExit(
            "checkpoint hashes.eval_contract mismatch: "
            f"payload={hashes['eval_contract']} expected={expected_eval_contract}"
        )
    return hashes


def resolve_checkpoint_step(
    *,
    ckpt_path: Path | str,
    payload: Mapping[str, Any] | None = None,
    cli_step: int | None = None,
    allowed_steps: Sequence[int] | None = None,
    require: bool = False,
) -> int | None:
    """Resolve candidate step from payload, optionally confirmed by filename/CLI.

    Contract (fail-closed for D1c formal, ``require=True``):

    * Step is derived from checkpoint ``payload['step']`` (or optimizer/global
      aliases). The immutable ``candidate_step_XXXX.pt`` filename may **confirm**
      that payload step; it must not invent a step when the payload lacks one.
    * ``--checkpoint-step`` may only **confirm** a derived step. It must not
      invent a foreign step for a file that has no payload step
      (e.g. ``best.pt`` + ``--checkpoint-step 625`` is refused when
      ``require=True``).
    * When both payload and filename steps are present they must agree; a
      conflicting CLI value is also refused.
    * When ``allowed_steps`` is set the resolved step must be a member.
    """
    payload_step = checkpoint_step_from_payload(payload)
    name_step = parse_candidate_step_from_path(ckpt_path)
    if (
        payload_step is not None
        and name_step is not None
        and int(payload_step) != int(name_step)
    ):
        raise SystemExit(
            f"checkpoint step mismatch: payload step={payload_step} "
            f"!= filename step={name_step} ({ckpt_path})"
        )
    if require and payload_step is None:
        if cli_step is not None:
            raise SystemExit(
                f"--checkpoint-step={cli_step} cannot invent a step for "
                f"{ckpt_path}: formal D1c requires payload['step']; "
                "CLI may only confirm a derived step"
            )
        raise SystemExit(
            f"checkpoint step unresolved: formal D1c requires payload['step'] "
            f"stamped by the trainer for {ckpt_path}; filename may only confirm "
            "(got no payload step)"
        )
    # Formal path: payload is authoritative. Non-formal may fall back to filename.
    inferred = payload_step if payload_step is not None else name_step
    if cli_step is not None:
        if inferred is None:
            if require:
                raise SystemExit(
                    f"--checkpoint-step={cli_step} cannot invent a step for "
                    f"{ckpt_path}: formal D1c requires payload['step']; "
                    "CLI may only confirm a derived step"
                )
            step: int | None = int(cli_step)
        elif int(cli_step) != int(inferred):
            raise SystemExit(
                f"--checkpoint-step={cli_step} conflicts with inferred step={inferred} "
                f"({ckpt_path})"
            )
        else:
            step = int(cli_step)
    else:
        step = int(inferred) if inferred is not None else None
    if step is None:
        if require:
            raise SystemExit(
                "checkpoint step unresolved: formal D1c requires payload['step'] "
                "stamped by the trainer "
                "(--checkpoint-step may only confirm a derived step)"
            )
        return None
    if allowed_steps is not None and int(step) not in {int(s) for s in allowed_steps}:
        raise SystemExit(
            f"checkpoint step {step} not in contract.train.candidate_steps "
            f"{list(allowed_steps)}"
        )
    return int(step)


def _load_model(ckpt: Path, device: torch.device):
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = payload.get("cfg") or load_yaml("configs/discrete_speech_lm_d1b_443m.yaml")
    model = build_discrete_speech_lm(cfg)
    sd = payload.get("model") or payload
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, cfg, payload if isinstance(payload, dict) else {}


def _resolve_speaker_registry(
    *,
    cfg: dict[str, Any],
    speaker_map: str | None,
) -> tuple[SpeakerRegistry, str]:
    """Return ``(registry, resolved_map_path)`` for contract hash checks."""
    candidates: list[str] = []
    if speaker_map:
        candidates.append(speaker_map)
    data = (cfg or {}).get("data") or {}
    for key in ("speaker_map", "speaker_registry"):
        val = data.get(key)
        if isinstance(val, str) and val:
            candidates.append(val)
    train = (cfg or {}).get("train") or {}
    for key in ("speaker_map_path", "speaker_map"):
        val = train.get(key)
        if isinstance(val, str) and val:
            candidates.append(val)
    model_spk = ((cfg or {}).get("model") or {}).get("speaker") or {}
    map_path = model_spk.get("map_path") if isinstance(model_spk, dict) else None
    if isinstance(map_path, str) and map_path:
        candidates.append(map_path)
    candidates.append("configs/speaker_maps/trackn_v2.json")
    for path in candidates:
        p = expand(path)
        if p.is_file():
            return SpeakerRegistry.load(p), str(p)
    raise SystemExit(
        "speaker map required; pass --speaker-map or place "
        "configs/speaker_maps/trackn_v2.json"
    )


def load_codec_items(
    codec_tokens_dir: str | Path,
    max_n: int | None,
    *,
    speaker_registry: SpeakerRegistry,
    speaker_mode: str = "eval",
    utt_ids: list[str] | None = None,
    require_items_sha256: str | None = None,
) -> list[dict[str, Any]]:
    # Import dataset class from train script without treating it as installed pkg.
    import importlib.util

    train_path = Path(__file__).resolve().parent / "72_train_discrete_lm.py"
    spec = importlib.util.spec_from_file_location("train_discrete_lm_mod", train_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kwargs: dict[str, Any] = {
        "speaker_registry": speaker_registry,
        "speaker_mode": speaker_mode,
        "allow_unknown_speaker": False,
    }
    if utt_ids is not None:
        kwargs["utt_ids"] = utt_ids
        kwargs["require_items_sha256"] = require_items_sha256
    elif max_n is not None:
        kwargs["max_utterances"] = int(max_n)
    ds = mod.CodecTokenShardDataset(codec_tokens_dir, **kwargs)
    return [ds[i] for i in range(len(ds))]


def _ref_texts_from_manifest(
    utt_ids: list[str], manifest_path: str | Path
) -> dict[str, str]:
    from minispeech.data.manifest import iter_manifest

    want = set(utt_ids)
    out: dict[str, str] = {}
    path = expand(manifest_path)
    if not path.is_file():
        return out
    for r in iter_manifest(path):
        if r.utt_id in want:
            out[r.utt_id] = r.text_stressed or r.text_raw or ""
    return out


def teacher_forced_ce(
    model, batch: dict[str, Any], device: torch.device
) -> dict[str, float]:
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    kw: dict[str, Any] = {
        "text_mask": batch.get("text_mask"),
        "speaker_id": batch.get("speaker_id"),
        "compute_loss": True,
    }
    frame_mask = batch.get("frame_mask")
    if frame_mask is None:
        frame_mask = batch.get("codec_frame_mask")
    kw["frame_mask"] = frame_mask
    # Semantic-EOS mode requires collated semantic targets/masks.
    if getattr(model, "termination_mode", "frame_stop_v1") == "semantic_eos_v1":
        if batch.get("semantic_targets") is None or batch.get("semantic_mask") is None:
            raise ValueError(
                "semantic_eos_v1 teacher-forced CE requires semantic_targets/semantic_mask "
                "(collate with termination_mode=semantic_eos_v1)"
            )
        kw["semantic_targets"] = batch["semantic_targets"]
        kw["semantic_mask"] = batch["semantic_mask"]
    with torch.no_grad():
        out = model(batch["input_ids"], batch["codes"], **kw)
    assert out.loss is not None
    loss = out.loss
    result: dict[str, Any] = {
        "loss": float(loss["total"].cpu()),
        "ce_cb0": float(loss["ce_cb0"].cpu()),
        "ce_residual": float(loss["ce_residual"].cpu()),
        "teacher_forced_ce_source": "eval_computed",
    }
    if "ce_semantic_eos" in loss:
        result["ce_semantic_eos"] = float(loss["ce_semantic_eos"].cpu())
    if "tf_eos_final_top1_accuracy" in loss:
        result["tf_eos_final_top1_accuracy"] = float(
            loss["tf_eos_final_top1_accuracy"].cpu()
        )
    if "tf_eos_early_top1_false_positive_count" in loss:
        result["tf_eos_early_top1_false_positive_count"] = int(
            loss["tf_eos_early_top1_false_positive_count"].cpu()
        )
    return result


def aggregate_teacher_forced_batches(
    batch_metrics: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-batch ``teacher_forced_ce()`` outputs for the emitter.

    Mean CE terms across batches; mean EOS final-top1 accuracy; **sum** early-EOS
    false-positive counts. Preserves ``teacher_forced_ce_source`` only when every
    batch is ``eval_computed`` and EOS metrics are present.
    """
    n = len(batch_metrics)
    if n == 0:
        return {
            "loss": 0.0,
            "ce_cb0": 0.0,
            "ce_residual": 0.0,
            "total_ce": 0.0,
            "n_batches": 0,
            "teacher_forced_ce_source": "unavailable",
            "tf_eos_final_top1_accuracy": "unavailable",
            "tf_eos_early_top1_false_positive_count": "unavailable",
        }
    out: dict[str, Any] = {"n_batches": n}
    for key in ("loss", "ce_cb0", "ce_residual", "ce_semantic_eos"):
        vals = [
            float(m[key])
            for m in batch_metrics
            if key in m and isinstance(m[key], (int, float))
        ]
        if vals:
            out[key] = float(sum(vals) / len(vals))
    out["total_ce"] = float(out.get("loss", 0.0))
    accs = [
        float(m["tf_eos_final_top1_accuracy"])
        for m in batch_metrics
        if isinstance(m.get("tf_eos_final_top1_accuracy"), (int, float))
    ]
    fps: list[int] = []
    for m in batch_metrics:
        fp = m.get("tf_eos_early_top1_false_positive_count")
        if fp is None or str(fp) == "unavailable":
            continue
        fps.append(int(fp))
    sources = {m.get("teacher_forced_ce_source") for m in batch_metrics}
    if accs and len(accs) == n and fps and len(fps) == n and sources == {"eval_computed"}:
        out["tf_eos_final_top1_accuracy"] = float(sum(accs) / len(accs))
        out["tf_eos_early_top1_false_positive_count"] = int(sum(fps))
        out["teacher_forced_ce_source"] = "eval_computed"
    else:
        out["tf_eos_final_top1_accuracy"] = "unavailable"
        out["tf_eos_early_top1_false_positive_count"] = "unavailable"
        out["teacher_forced_ce_source"] = "unavailable"
    return out


def teacher_forced_metrics_for_emitter(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Project an aggregated TF summary into the dict the v2 emitter consumes."""
    keys = (
        "loss",
        "ce_cb0",
        "ce_residual",
        "ce_semantic_eos",
        "total_ce",
        "tf_eos_final_top1_accuracy",
        "tf_eos_early_top1_false_positive_count",
        "teacher_forced_ce_source",
    )
    out = {k: summary[k] for k in keys if k in summary}
    if "total_ce" not in out and "loss" in out:
        out["total_ce"] = out["loss"]
    if "teacher_forced_ce_source" not in out:
        out["teacher_forced_ce_source"] = "unavailable"
    return out


def _pcm_sha256_from_wav(path: Path) -> str:
    """SHA256 of float32 little-endian PCM samples (not the container file bytes)."""
    import hashlib

    try:
        import soundfile as sf

        audio, _sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"cannot read wav for pcm_sha256: {path}: {exc}") from exc
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _unwrap_generation(out: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    """Normalize generate() return to ``(codes[B,T,C], first-row metadata)``."""
    if isinstance(out, DiscreteGenerationOutput):
        row = out.rows[0] if out.rows else {}
        return out.codes, dict(row)
    if torch.is_tensor(out):
        return out, {}
    raise TypeError(f"unexpected generate() output type: {type(out)!r}")


def _gen_kw_from_infer(icfg: dict[str, Any], *, max_frames: int) -> dict[str, Any]:
    """Pass-through generate kwargs from shared resolver (no forked defaults)."""
    semantic = str(icfg.get("termination_mode") or "") == "semantic_eos_v1"
    keys = tuple(SEMANTIC_EOS_INFER_DEFAULTS) if semantic else GENERATE_INFER_KEYS
    out = {k: icfg[k] for k in keys if k in icfg}
    out["max_frames"] = int(max_frames)
    return out


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(str(path), audio, sr)
    except Exception:
        import wave

        pcm = np.clip(audio, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm_i16.tobytes())


def _ascii_utt(uid: str, i: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", uid)[:48]
    return f"{i:04d}_{safe}"


def free_running_asr_eval(
    model,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    cfg: dict[str, Any],
    refs: dict[str, str],
    out_wav_dir: Path,
    asr_config: str,
    max_frames: int,
    condition: str = "model",
    decode: bool = True,
    skip_asr: bool = False,
    infer_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate (or baseline) → decode → ASR CER. Writes wavs when decode=True."""
    from minispeech.codec.qwen3_tts_wrapper import Qwen3TTSCodec

    asr = None if skip_asr else load_asr_engine(asr_config)
    codec = Qwen3TTSCodec(device=device) if decode else None
    codebook_size = int((cfg.get("model") or {}).get("codebook_size", 2048))
    icfg = resolve_infer_settings(
        cfg,
        infer_overrides=infer_overrides,
        max_frames_override=max_frames,
        codebook_size=codebook_size,
    )
    term = ((cfg.get("model") or {}).get("termination") or {})
    if getattr(model, "termination_mode", None) == "semantic_eos_v1" or (
        isinstance(term, dict) and term.get("mode") == "semantic_eos_v1"
    ):
        icfg = dict(icfg)
        icfg["termination_mode"] = "semantic_eos_v1"
    gen_kw = _gen_kw_from_infer(icfg, max_frames=int(icfg["max_frames"]))

    rows_out: list[dict[str, Any]] = []
    gen_codes: list[torch.Tensor] = []
    cers: list[float] = []
    edit_counts = []

    n = len(rows)
    for i, row in enumerate(rows):
        uid = str(row.get("utt_id", f"utt_{i}"))
        ids = row["input_ids"]
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, dtype=torch.long)
        ids = ids.long().view(1, -1).to(device)
        text_len = int((ids != 0).sum().item()) if ids.numel() else int(ids.shape[-1])
        spk = torch.tensor([int(row.get("speaker_id", 0))], device=device)
        gt_codes = row["codes"]
        if not torch.is_tensor(gt_codes):
            gt_codes = torch.as_tensor(gt_codes)
        gt_codes = gt_codes.long()

        semantic_mode = getattr(model, "termination_mode", "frame_stop_v1") == "semantic_eos_v1"
        gen_meta: dict[str, Any] = {}
        with torch.no_grad():
            if condition == "copy_synth":
                codes = gt_codes.unsqueeze(0).to(device)
                stop_reason = "copy_synth"
            elif condition == "random_codes":
                T = min(max_frames, max(8, int(gt_codes.shape[0])))
                codes = torch.randint(0, 2048, (1, T, 16), device=device)
                stop_reason = "random_codes"
            else:
                if condition == "wrong_utt":
                    other = rows[(i + 1) % n]
                    oids = other["input_ids"]
                    if not torch.is_tensor(oids):
                        oids = torch.tensor(oids, dtype=torch.long)
                    prompt = oids.long().view(1, -1).to(device)
                    prompt_text_len = int((prompt != 0).sum().item())
                elif condition == "shuffle_text":
                    other = rows[(i + max(1, n // 3)) % n]
                    oids = other["input_ids"]
                    if not torch.is_tensor(oids):
                        oids = torch.tensor(oids, dtype=torch.long)
                    prompt = oids.long().view(1, -1).to(device)
                    prompt_text_len = int((prompt != 0).sum().item())
                elif condition == "mask_text":
                    prompt = torch.zeros_like(ids)
                    prompt_text_len = text_len
                else:  # model
                    prompt = ids
                    prompt_text_len = text_len
                gen_out = model.generate(prompt, speaker_id=spk, **gen_kw)
                codes, gen_meta = _unwrap_generation(gen_out)
                if semantic_mode:
                    # Formal D1c path: never call estimate_stop_reason.
                    stop_reason = str(gen_meta.get("termination_reason") or "unknown")
                else:
                    stop_reason = estimate_stop_reason(
                        n_frames=int(codes.shape[1]),
                        text_len=prompt_text_len,
                        cb0_ids=codes[0, :, 0].detach().cpu().tolist() if codes.shape[1] else [],
                        max_frames=max_frames,
                        min_frames=int(gen_kw["min_frames"]),
                        max_cb0_repeat=int(gen_kw["max_cb0_repeat"]),
                        frames_per_text_token=float(gen_kw["frames_per_text_token"]),
                    )

        gen_codes.append(codes[0].detach().cpu() if codes.numel() else codes.new_zeros((0, codes.shape[-1] if codes.ndim == 3 else 16)).cpu())
        # Normalize empty tensor shape for cb0_stats.
        if codes.ndim == 3 and codes.shape[1] == 0:
            gen_codes[-1] = torch.zeros(0, int(codes.shape[-1]), dtype=torch.long)
        ref = refs.get(uid, "")
        hyp_raw = ""
        cer_v = None
        wav_path = None
        if decode and codec is not None:
            if int(codes.shape[1]) == 0:
                # Empty immediate-EOS: skip codec decode; leave blank hyp.
                audio = np.zeros(0, dtype=np.float32)
                sr = 24000
                stem = _ascii_utt(uid, i)
                wav_path = out_wav_dir / f"{stem}.wav"
                _write_wav(wav_path, audio, sr)
            else:
                dec = codec.decode([codes[0].detach().cpu()])
                audio = np.asarray(dec.audio[0], dtype=np.float32)
                sr = int(dec.sample_rate)
                stem = _ascii_utt(uid, i)
                wav_path = out_wav_dir / f"{stem}.wav"
                _write_wav(wav_path, audio, sr)
                if asr is not None and audio.size:
                    hyp_raw = asr.transcribe(audio, sr, use_cache=False)
                    if ref:
                        ref_n = normalize_for_asr(ref)
                        hyp_n = normalize_for_asr(hyp_raw)
                        cer_v = float(compute_cer([ref_n], [hyp_n])) if ref_n else None
                        if cer_v is not None:
                            cers.append(cer_v)
                            edit_counts.append(edits_from_jiwer_chars(ref_n, hyp_n))

        row_out: dict[str, Any] = {
            "utt_id": uid,
            "ordinal": len(rows_out),
            "condition": condition,
            "n_frames": int(codes.shape[1]),
            "gt_frames": int(gt_codes.shape[0]),
            "cer": cer_v,
            "ref": ref,
            "hyp": hyp_raw,
            "wav": str(wav_path) if wav_path else None,
            "stop_reason": stop_reason,
        }
        if gen_meta:
            row_out["generation"] = gen_meta
        # Stable codes digest for formal binding (empty generation → empty hash input).
        import hashlib as _hl

        codes_bytes = codes.detach().cpu().numpy().astype(np.int32).tobytes()
        row_out["generated_codes_sha256"] = _hl.sha256(codes_bytes).hexdigest()
        rows_out.append(row_out)

    boot = bootstrap_mean_cer(cers)
    cb0 = cb0_stats(gen_codes)
    report = {
        "condition": condition,
        "n": len(rows_out),
        "mean_cer": boot["mean"],
        "cer_ci95_low": boot["ci95_low"],
        "cer_ci95_high": boot["ci95_high"],
        "cb0": cb0,
        "rows": rows_out,
        "edit_counts": edit_counts,
        "infer": {k: v for k, v in icfg.items() if not str(k).startswith("_")},
    }
    return enrich_model_report(report)


def _edit_counts_to_jsonable(edit_counts: list[Any]) -> list[dict[str, int]]:
    out = []
    for c in edit_counts:
        if hasattr(c, "hits"):
            out.append(
                {
                    "hits": int(c.hits),
                    "substitutions": int(c.substitutions),
                    "deletions": int(c.deletions),
                    "insertions": int(c.insertions),
                }
            )
        elif isinstance(c, dict):
            out.append(
                {
                    "hits": int(c.get("hits", 0)),
                    "substitutions": int(c.get("substitutions", 0)),
                    "deletions": int(c.get("deletions", 0)),
                    "insertions": int(c.get("insertions", 0)),
                }
            )
    return out


def assert_score_only_d1b_bundle(
    bundle: dict[str, Any],
    *,
    eval_contract: str | None = None,
    speaker_map_path: str | None = None,
) -> dict[str, Any]:
    """Fail-closed provenance + frozen max_frames checks for ``--score-only``.

    Canonical D1b refuses bundles that omit provenance / contract hash /
    subset label+hash / full WAVE3c infer block / ``max_frames``, or that
    record a GT-clamped ``max_frames``.
    Returns the loaded eval contract on success.
    """
    raw_prov = bundle.get("provenance")
    if not isinstance(raw_prov, dict) or not raw_prov:
        raise SystemExit(
            "--score-only refuse: missing bundle.provenance "
            "(canonical D1b requires decode-time provenance)"
        )
    provenance = dict(raw_prov)
    contract_path = (
        eval_contract or provenance.get("eval_contract_path") or DEFAULT_EVAL_CONTRACT
    )
    contract_sha = provenance.get("eval_contract_sha256")
    if not contract_sha:
        raise SystemExit(
            "--score-only refuse: provenance.eval_contract_sha256 missing "
            "(fail-closed; refuse unsigned decode bundles)"
        )
    subset_label = provenance.get("subset_label")
    subset_hash = provenance.get("subset_items_sha256")
    if not subset_label:
        raise SystemExit(
            "--score-only refuse: provenance.subset_label missing "
            "(canonical D1b requires frozen subset attestation)"
        )
    if not subset_hash:
        raise SystemExit(
            "--score-only refuse: provenance.subset_items_sha256 missing "
            "(canonical D1b requires subset hash with subset_label)"
        )

    reports = bundle.get("reports")
    if not isinstance(reports, dict) or not reports:
        raise SystemExit("--score-only refuse: missing bundle.reports")
    missing_reps = [k for k in _REQUIRED_N2_SCORE_REPORTS if k not in reports]
    if missing_reps:
        raise SystemExit(
            "--score-only refuse: incomplete N2 reports block; missing "
            + ", ".join(missing_reps)
        )

    live_infer = None
    model_infer = (reports.get("model") or {}).get("infer")
    if not isinstance(model_infer, dict) or not model_infer:
        raise SystemExit(
            "--score-only refuse: reports.model.infer missing "
            "(canonical D1b requires full frozen WAVE3c infer block at decode)"
        )
    missing_keys = [k for k in GENERATE_INFER_KEYS if k not in model_infer]
    if missing_keys:
        raise SystemExit(
            "--score-only refuse: incomplete reports.model.infer; missing keys "
            + ", ".join(missing_keys)
        )
    live_infer = model_infer
    # Also require provenance to carry the same frozen max_frames attestation.
    wave3c = wave3c_infer_defaults()

    try:
        contract = enforce_eval_contract(
            None,
            contract_path=contract_path,
            subset_items_sha256=str(subset_hash),
            subset_label=str(subset_label),
            require_clean_sha=False,
            expected_contract_sha256=str(contract_sha),
            live_infer=live_infer,
            speaker_map_path=speaker_map_path,
        )
    except RuntimeError as exc:
        raise SystemExit(f"--score-only refuse: {exc}") from exc

    frozen_mf = int((contract.get("infer") or {}).get("max_frames") or wave3c["max_frames"])
    recorded: list[tuple[str, Any]] = [
        ("reports.model.infer.max_frames", live_infer.get("max_frames"))
    ]
    if provenance.get("max_frames") is not None:
        recorded.append(("provenance.max_frames", provenance.get("max_frames")))
    else:
        raise SystemExit(
            "--score-only refuse: provenance.max_frames missing "
            "(frozen WAVE3c max_frames must be recorded at decode)"
        )
    for loc, mf in recorded:
        try:
            mf_i = int(mf)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"--score-only refuse: non-integer {loc}={mf!r}"
            ) from exc
        if mf_i != frozen_mf:
            raise SystemExit(
                f"--score-only refuse: {loc}={mf_i} != frozen {frozen_mf} "
                "(GT-derived clamps are forbidden on D1b)"
            )
    return contract


def assert_score_only_wavs_present(
    bundle: dict[str, Any],
    *,
    d1b: bool = True,
    skip_eval_contract: bool = False,
) -> None:
    """Canonical score-only: missing wav paths fail-closed (no silent skip)."""
    if not d1b or skip_eval_contract:
        return
    missing: list[str] = []
    reports = bundle.get("reports") or {}
    for name, rep in reports.items():
        if not isinstance(rep, dict):
            continue
        for i, row in enumerate(rep.get("rows") or []):
            if not isinstance(row, dict):
                continue
            wav = row.get("wav")
            utt = row.get("utt_id") or row.get("id") or i
            if not wav:
                missing.append(f"{name}[{utt}]: wav path missing")
            elif not Path(str(wav)).is_file():
                missing.append(f"{name}[{utt}]: missing file {wav}")
    if missing:
        preview = "\n- ".join(missing[:20])
        more = "" if len(missing) <= 20 else f"\n- ... and {len(missing) - 20} more"
        raise SystemExit(
            "--score-only refuse: missing wav files on canonical path "
            f"({len(missing)} rows):\n- {preview}{more}"
        )


def _is_d1c_semantic_contract(contract: Mapping[str, Any] | None) -> bool:
    if not isinstance(contract, Mapping):
        return False
    term = contract.get("termination") or {}
    return isinstance(term, Mapping) and term.get("mode") == "semantic_eos_v1"


def _enrich_v2_condition_rows(
    rows: list[dict[str, Any]],
    *,
    subset_key: str,
    subset_path: str,
    subset_file_sha256: str,
    ordered_utt_ids: list[str],
) -> dict[str, Any]:
    """Stamp missing formal-row fields so emit_d1c_v2_report can validate."""
    from minispeech.eval.discrete_n2 import annotate_catastrophic_rows
    from minispeech.eval.discrete_report_schema import file_sha256 as _fsha

    if not subset_file_sha256 or subset_file_sha256 == ("0" * 64):
        raise ValueError(
            "formal D1c report refused: subset_file_sha256 missing or placeholder"
        )
    annotated = annotate_catastrophic_rows(rows)
    enriched: list[dict[str, Any]] = []
    for ordinal, row in enumerate(annotated):
        r = dict(row)
        r["ordinal"] = ordinal
        wav_raw = r.get("wav")
        if isinstance(wav_raw, str):
            wav_path = Path(wav_raw)
            if not wav_path.is_file():
                raise ValueError(
                    f"formal D1c report refused: missing wav for row {ordinal}: {wav_path}"
                )
            file_digest = _fsha(wav_path)
            pcm_digest = _pcm_sha256_from_wav(wav_path)
            if pcm_digest == file_digest:
                # Extremely unlikely for float32 PCM vs container; still refuse equality
                # only when the shortcut was used historically — keep both distinct paths.
                pass
            r["wav"] = {
                "path": str(wav_raw),
                "file_sha256": file_digest,
                "pcm_sha256": pcm_digest,
            }
        elif isinstance(wav_raw, dict):
            r["wav"] = dict(wav_raw)
            if not r["wav"].get("path"):
                raise ValueError(
                    f"formal D1c report refused: wav.path missing for row {ordinal}"
                )
            wav_path = Path(str(r["wav"]["path"]))
            if not r["wav"].get("file_sha256") or r["wav"]["file_sha256"] == ("0" * 64):
                if not wav_path.is_file():
                    raise ValueError(
                        f"formal D1c report refused: missing wav for row {ordinal}: {wav_path}"
                    )
                r["wav"]["file_sha256"] = _fsha(wav_path)
            if not r["wav"].get("pcm_sha256") or r["wav"]["pcm_sha256"] == ("0" * 64):
                if not wav_path.is_file():
                    raise ValueError(
                        f"formal D1c report refused: missing wav for pcm hash row {ordinal}"
                    )
                r["wav"]["pcm_sha256"] = _pcm_sha256_from_wav(wav_path)
            if r["wav"].get("pcm_sha256") == r["wav"].get("file_sha256"):
                # Recompute PCM if a caller copied file_sha into pcm_sha.
                if wav_path.is_file():
                    r["wav"]["pcm_sha256"] = _pcm_sha256_from_wav(wav_path)
        else:
            raise ValueError(
                f"formal D1c report refused: wav missing for row {ordinal} utt={r.get('utt_id')}"
            )
        gen = r.get("generation")
        if not isinstance(gen, dict):
            gen = {
                "termination_reason": r.get("stop_reason") or "unknown",
                "n_decoded_frames": r.get("n_frames") or 0,
            }
        r["generation"] = gen
        r.setdefault(
            "normalized_reference_sha256",
            hashlib_sha256_text(normalize_for_asr(str(r.get("ref") or ""))),
        )
        codes_digest = r.get("generated_codes_sha256") or gen.get("generated_codes_sha256")
        if not codes_digest or codes_digest == ("0" * 64):
            raise ValueError(
                f"formal D1c report refused: generated_codes_sha256 missing for row {ordinal}"
            )
        r["generated_codes_sha256"] = str(codes_digest)
        if r.get("cer") is None:
            r["cer"] = float("nan")
        if "catastrophic" not in r:
            raise ValueError(
                f"formal D1c report refused: catastrophic not stamped for row {ordinal}"
            )
        enriched.append(r)
    return {
        "subset_key": subset_key,
        "subset_path": subset_path,
        "subset_file_sha256": subset_file_sha256,
        "expected_count": len(ordered_utt_ids),
        "evaluated_count": len(enriched),
        "ordered_utt_ids_sha256": ordered_utt_ids_sha256(ordered_utt_ids),
        "rows": enriched,
        "rows_sha256": "",
    }


def hashlib_sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _emit_d1c_v2_from_gate(
    *,
    out_dir: Path,
    contract: Mapping[str, Any] | None,
    ckpt: str,
    provenance: dict[str, Any],
    infer: dict[str, Any],
    teacher_forced_ce: Any,
    ordered_utt_ids: list[str],
    condition_reports: dict[str, Any],
    gate: dict[str, Any],
) -> Path | None:
    """Emit selector-consumable v2 report when the active contract is D1c semantic EOS.

    Contract after gate JSON is written
    ------------------------------------
    - Succeed when measured TF EOS metrics are present (``eval_computed``).
    - Soft-refuse (return ``None``, print reason) when metrics are genuinely
      unavailable — never raise an unhandled traceback after ``n2_gate.json``.
    - ``--score-only`` is formal **only** when the decode bundle already carries
      measured ``eval_computed`` TF metrics from ``--decode-only``. Otherwise it
      skips formal emit (diagnostic gate artifacts remain). Never stamps
      ``decode_bundle`` as a formal provenance source.
    """
    if not _is_d1c_semantic_contract(contract):
        return None
    try:
        return _emit_d1c_v2_from_gate_impl(
            out_dir=out_dir,
            contract=contract,
            ckpt=ckpt,
            provenance=provenance,
            infer=infer,
            teacher_forced_ce=teacher_forced_ce,
            ordered_utt_ids=ordered_utt_ids,
            condition_reports=condition_reports,
            gate=gate,
        )
    except Exception as exc:  # noqa: BLE001 — soft-refuse after gate JSON is durable
        print(f"D1c v2 formal report refused (non-fatal after gate write): {exc}")
        return None


def _emit_d1c_v2_from_gate_impl(
    *,
    out_dir: Path,
    contract: Mapping[str, Any] | None,
    ckpt: str,
    provenance: dict[str, Any],
    infer: dict[str, Any],
    teacher_forced_ce: Any,
    ordered_utt_ids: list[str],
    condition_reports: dict[str, Any],
    gate: dict[str, Any],
) -> Path | None:
    ids = list(ordered_utt_ids) or [
        str(r.get("utt_id"))
        for r in (condition_reports.get("model") or {}).get("rows") or []
    ]
    subset_key = str(provenance.get("subset_label") or "unknown")
    subset_path = str(provenance.get("subset_path") or "")
    subset_file = str(provenance.get("subset_file_sha256") or "")
    if not subset_file or subset_file == ("0" * 64):
        subsets = (contract or {}).get("subsets") or {}
        entry = subsets.get(subset_key) or subsets.get("select8") or subsets.get("full32") or {}
        subset_file = str(entry.get("file_sha256") or "")
        if not subset_path:
            subset_path = str(entry.get("path") or "")
    if not subset_file or subset_file == ("0" * 64):
        raise ValueError(
            "formal D1c report refused: subset_file_sha256 missing from provenance/contract"
        )
    conditions: dict[str, Any] = {}
    for name in V2_CONDITIONS:
        rep = condition_reports.get(name) or {}
        conditions[name] = _enrich_v2_condition_rows(
            list(rep.get("rows") or []),
            subset_key=subset_key,
            subset_path=subset_path,
            subset_file_sha256=subset_file,
            ordered_utt_ids=ids,
        )
    if isinstance(teacher_forced_ce, dict):
        tf = dict(teacher_forced_ce)
    elif teacher_forced_ce is None:
        tf = {"total_ce": None, "teacher_forced_ce_source": "unavailable"}
    else:
        # Bare float / scalar — the round-2 plumbing bug. Never fabricate EOS.
        tf = {
            "total_ce": teacher_forced_ce,
            "teacher_forced_ce_source": "unavailable",
            "tf_eos_final_top1_accuracy": "unavailable",
            "tf_eos_early_top1_false_positive_count": "unavailable",
        }
    if "total_ce" not in tf and "loss" in tf:
        tf["total_ce"] = tf["loss"]
    # Propagate measured EOS TF metrics only; never fabricate passing defaults
    # and never stamp decode_bundle as a formal-eligible source.
    if "teacher_forced_ce_source" not in tf:
        eos_ready = (
            isinstance(tf.get("tf_eos_final_top1_accuracy"), (int, float))
            and tf.get("tf_eos_early_top1_false_positive_count") is not None
            and str(tf.get("tf_eos_early_top1_false_positive_count")) != "unavailable"
        )
        tf["teacher_forced_ce_source"] = "eval_computed" if eos_ready else "unavailable"
    if "tf_eos_final_top1_accuracy" not in tf:
        tf["tf_eos_final_top1_accuracy"] = "unavailable"
    if "tf_eos_early_top1_false_positive_count" not in tf:
        tf["tf_eos_early_top1_false_positive_count"] = "unavailable"

    # --score-only: formal only if decode bundle carried measured eval_computed TF.
    if provenance.get("score_only") and tf.get("teacher_forced_ce_source") != "eval_computed":
        print(
            "D1c v2 formal report skipped: --score-only without measured "
            "eval_computed teacher-forced EOS metrics in decode bundle "
            f"(source={tf.get('teacher_forced_ce_source')!r})"
        )
        return None

    ckpt_path = Path(str(ckpt))
    from minispeech.eval.discrete_report_schema import file_sha256 as _fsha

    if not ckpt_path.is_file():
        raise ValueError(f"checkpoint missing for formal report: {ckpt_path}")
    ckpt_sha = _fsha(ckpt_path)
    report_path = out_dir / "discrete_eval_report_v2.json"

    step_raw = provenance.get("checkpoint_step", gate.get("step"))
    try:
        checkpoint_step = int(step_raw) if step_raw is not None else 0
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"formal D1c report refused: checkpoint.step not an integer ({step_raw!r})"
        ) from exc
    if checkpoint_step <= 0:
        raise ValueError(
            "formal D1c report refused: checkpoint.step missing or zero "
            "(evaluator must stamp candidate step from payload/filename; "
            "--checkpoint-step may only confirm a derived step)"
        )
    allowed = None
    train_block = (contract or {}).get("train") or {}
    if isinstance(train_block, Mapping) and train_block.get("candidate_steps"):
        allowed = [int(s) for s in train_block["candidate_steps"]]
    elif _is_d1c_semantic_contract(contract):
        allowed = list(FR_CANDIDATE_STEPS_BASE)
    if allowed is not None and checkpoint_step not in allowed:
        raise ValueError(
            f"formal D1c report refused: checkpoint.step={checkpoint_step} "
            f"not in candidate_steps={allowed}"
        )

    raw_hashes = provenance.get("checkpoint_hashes")
    if not isinstance(raw_hashes, Mapping):
        raise ValueError(
            "formal D1c report refused: checkpoint hashes missing from provenance "
            "(evaluator must stamp payload['hashes'] from the trainer)"
        )
    ckpt_hashes: dict[str, str] = {}
    for key in _REQUIRED_CHECKPOINT_HASH_KEYS:
        value = raw_hashes.get(key)
        if value is None or not str(value).strip() or str(value) == ("0" * 64):
            raise ValueError(
                f"formal D1c report refused: checkpoint.hashes.{key} missing"
            )
        ckpt_hashes[key] = str(value)
    live_contract = str((contract or {}).get("sha256") or "")
    if live_contract and ckpt_hashes["eval_contract"] != live_contract:
        raise ValueError(
            "formal D1c report refused: checkpoint.hashes.eval_contract mismatch "
            f"(payload={ckpt_hashes['eval_contract']} contract={live_contract})"
        )

    criteria = gate.get("criteria") or {}
    text_usage = (
        gate.get("text_usage_acceptance")
        or criteria.get("text_usage_d1b")
        or {}
    )
    if not isinstance(text_usage, Mapping):
        text_usage = {}
    stability = (
        gate.get("stability_acceptance")
        or criteria.get("stability_d1b")
        or {}
    )
    if not isinstance(stability, Mapping):
        stability = {}
    model_rep = condition_reports.get("model") or {}
    phrase_loop = bool(
        model_rep.get("phrase_loop_repetition")
        or stability.get("phrase_loop_repetition")
    )
    runaway = bool(
        model_rep.get("runaway_or_repetition")
        or stability.get("runaway_or_repetition")
        or (not stability.get("no_runaway", True) if stability else False)
    )
    text_usage_block = {
        "pass": bool(text_usage.get("pass")) if "pass" in text_usage else False,
        **{k: v for k, v in text_usage.items() if k != "pass"},
    }
    if "pass" not in text_usage:
        text_usage_block["reason"] = "absent_from_gate"

    emit_d1c_v2_report(
        out_path=report_path,
        checkpoint={
            "path": str(ckpt),
            "sha256": ckpt_sha,
            "step": int(checkpoint_step),
            "schema": "minispeech.discrete_lm_checkpoint",
            "schema_version": "2.0.0",
            "hashes": dict(ckpt_hashes),
        },
        provenance={
            **provenance,
            "eval_contract_sha256": str(
                (contract or {}).get("sha256")
                or provenance.get("eval_contract_sha256")
                or ""
            ),
            "recipe_sha256": str(
                (contract or {}).get("recipe_sha256")
                or provenance.get("recipe_sha256")
                or ""
            ),
            "asr_sha256": str(
                ((contract or {}).get("asr") or {}).get("file_sha256")
                or provenance.get("asr_sha256")
                or ""
            ),
            "subset_file_sha256": subset_file,
            "checkpoint_hashes": dict(ckpt_hashes),
        },
        infer=infer,
        teacher_forced_metrics=tf,
        ordered_utt_ids=ids,
        conditions=conditions,
        summaries={
            "model_macro_cer": criteria.get("macro_cer") or criteria.get("fr_cer"),
            "n_catastrophic": criteria.get("n_catastrophic", 0),
            "max_abs_log_duration_error": criteria.get("max_abs_log_duration_error", 0.0),
            "mean_abs_log_duration_error": criteria.get("mean_abs_log_duration_error", 0.0),
            "phrase_loop_repetition": phrase_loop,
            "runaway_or_repetition": runaway,
        },
        gates={
            "eligible": bool(gate.get("verdict") == "PASS"),
            "verdict": gate.get("verdict"),
            "criteria": criteria,
            "text_usage": text_usage_block,
            "runaway_or_repetition": runaway,
        },
    )
    print(f"wrote {report_path}")
    return report_path


def _score_n2_bundle(
    bundle_path: Path,
    *,
    asr_config: str,
    out_dir: Path,
    copy_synth_ceiling: float,
    d1b: bool = True,
    eval_contract: str | None = None,
    skip_eval_contract: bool = False,
) -> int:
    """ASR-score a decode-only bundle (main venv with openai-whisper)."""
    import soundfile as sf

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    provenance = dict(bundle.get("provenance") or {})
    if d1b and not skip_eval_contract:
        assert_score_only_d1b_bundle(bundle, eval_contract=eval_contract)
        assert_score_only_wavs_present(
            bundle, d1b=d1b, skip_eval_contract=skip_eval_contract
        )

    asr = load_asr_engine(asr_config)
    scored: dict[str, Any] = {}
    for name, rep in bundle["reports"].items():
        rows = rep.get("rows") or []
        cers: list[float] = []
        edit_counts = []
        for row in rows:
            wav = row.get("wav")
            ref = row.get("ref") or ""
            if not wav or not Path(wav).is_file():
                # Canonical path already fail-closed above; legacy/non-d1b may skip.
                if d1b and not skip_eval_contract:
                    raise SystemExit(
                        f"--score-only refuse: missing wav during score ({name}: {wav!r})"
                    )
                continue
            audio, sr = sf.read(wav, dtype="float32")
            if getattr(audio, "ndim", 1) > 1:
                audio = np.asarray(audio).mean(axis=-1)
            hyp_raw = asr.transcribe(np.asarray(audio, dtype=np.float32), int(sr), use_cache=False)
            row["hyp"] = hyp_raw
            if ref:
                ref_n = normalize_for_asr(ref)
                hyp_n = normalize_for_asr(hyp_raw)
                cer_v = float(compute_cer([ref_n], [hyp_n])) if ref_n else None
                row["cer"] = cer_v
                if cer_v is not None:
                    cers.append(cer_v)
                    edit_counts.append(edits_from_jiwer_chars(ref_n, hyp_n))
        rep["edit_counts"] = edit_counts
        scored[name] = enrich_model_report(rep)
        print(
            f"[score] {name} macro_cer={scored[name]['macro_cer']:.4f} "
            f"micro_cer={scored[name].get('micro_cer')} n={len(cers)}"
        )

    if d1b:
        tf_metrics = bundle.get("teacher_forced_ce")
        if not isinstance(tf_metrics, Mapping):
            tf_metrics = (
                {"total_ce": tf_metrics}
                if tf_metrics is not None
                else {"teacher_forced_ce_source": "unavailable"}
            )
        contract_for_gate = None
        if eval_contract and not skip_eval_contract:
            try:
                contract_for_gate = load_eval_contract(eval_contract)
            except Exception:
                contract_for_gate = None
        if _is_d1c_semantic_contract(contract_for_gate):
            gate = evaluate_n2_gate_d1c(
                scored["model"],
                scored["copy_synth"],
                scored["wrong_utt"],
                scored["random_codes"],
                scored["shuffle_text"],
                scored["mask_text"],
                teacher_forced_metrics=tf_metrics,
                copy_synth_ceiling=copy_synth_ceiling,
            )
        else:
            gate = evaluate_n2_gate_d1b(
                scored["model"],
                scored["copy_synth"],
                scored["wrong_utt"],
                scored["random_codes"],
                scored["shuffle_text"],
                scored["mask_text"],
                copy_synth_ceiling=copy_synth_ceiling,
            )
    else:
        gate = evaluate_n2_gate(
            scored["model"],
            scored["copy_synth"],
            scored["wrong_utt"],
            scored["random_codes"],
            scored["shuffle_text"],
            scored["mask_text"],
            copy_synth_ceiling=copy_synth_ceiling,
        )
    gate["artifacts"] = {
        "ckpt": bundle.get("ckpt"),
        "wav_dir": bundle.get("wav_dir"),
        "decode_bundle": str(bundle_path),
        "eval_out": str(out_dir),
    }
    gate["teacher_forced_ce"] = bundle.get("teacher_forced_ce")
    # Ensure score-only path carries checkpoint_step when decode stamped it,
    # or recover from the checkpoint filename / --checkpoint-step via caller.
    score_prov = dict(provenance)
    if score_prov.get("checkpoint_step") is None and bundle.get("ckpt"):
        recovered = resolve_checkpoint_step(
            ckpt_path=str(bundle.get("ckpt")),
            require=False,
        )
        if recovered is not None:
            score_prov["checkpoint_step"] = int(recovered)
    if score_prov.get("checkpoint_step") is not None:
        gate["step"] = int(score_prov["checkpoint_step"])
    gate["provenance"] = {
        **score_prov,
        "git_sha_score": git_sha(),
        "git_dirty_score": git_is_dirty(),
        "score_only": True,
    }
    # Promote D1b acceptance blocks so the v2 emitter can stamp §11 fields.
    criteria = gate.get("criteria") or {}
    if "text_usage_acceptance" not in gate and criteria.get("text_usage_d1b"):
        gate["text_usage_acceptance"] = criteria["text_usage_d1b"]
    if "stability_acceptance" not in gate and criteria.get("stability_d1b"):
        gate["stability_acceptance"] = criteria["stability_d1b"]
    if "enriched_reports" not in gate:
        gate["reports"] = {
            k: {kk: vv for kk, vv in rep.items() if kk not in ("rows", "edit_counts")}
            for k, rep in scored.items()
        }
    else:
        gate["reports"] = gate.pop("enriched_reports")
    model_rows = gate.pop("model_rows", scored["model"]["rows"])
    cat_rows = gate.pop("catastrophic_rows", scored["model"].get("catastrophic_rows") or [])

    out_dir.mkdir(parents=True, exist_ok=True)
    gate_path = out_dir / "n2_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    (out_dir / "n2_model_rows.json").write_text(
        json.dumps(model_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "n2_catastrophic_rows.json").write_text(
        json.dumps(cat_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"n2_verdict": gate["verdict"], **gate["criteria"]}, indent=2, default=str))
    print(f"wrote {gate_path}")
    contract_for_emit = None
    if eval_contract and not skip_eval_contract:
        try:
            contract_for_emit = load_eval_contract(eval_contract)
        except Exception:
            contract_for_emit = None
    _emit_d1c_v2_from_gate(
        out_dir=out_dir,
        contract=contract_for_emit,
        ckpt=str(bundle.get("ckpt") or ""),
        provenance=dict(gate.get("provenance") or provenance),
        infer=dict((scored.get("model") or {}).get("infer") or {}),
        teacher_forced_ce=bundle.get("teacher_forced_ce"),
        ordered_utt_ids=[str(r.get("utt_id")) for r in (scored.get("model") or {}).get("rows") or []],
        condition_reports=scored,
        gate=gate,
    )
    return 0


def evaluate_n2_gate(
    model_report: dict[str, Any],
    copy_report: dict[str, Any],
    wrong_report: dict[str, Any],
    random_report: dict[str, Any],
    shuffle_report: dict[str, Any],
    mask_report: dict[str, Any],
    *,
    copy_synth_ceiling: float,
    margin: float = 0.03,
) -> dict[str, Any]:
    """Legacy D1 N2 gate (kept for continuity). Prefer evaluate_n2_gate_d1b for D1b."""
    fr_cer = float(model_report["mean_cer"])
    copy_cer = float(copy_report["mean_cer"])
    if math.isnan(copy_cer):
        copy_cer = float(copy_synth_ceiling)
    ceiling = copy_cer + margin
    cer_pass = (not math.isnan(fr_cer)) and fr_cer <= ceiling

    sep_wrong = float(model_report["cer_ci95_high"]) < float(wrong_report["cer_ci95_low"])
    sep_random = float(model_report["cer_ci95_high"]) < float(random_report["cer_ci95_low"])
    if not sep_wrong:
        sep_wrong = fr_cer + 0.05 < float(wrong_report["mean_cer"])
    if not sep_random:
        sep_random = fr_cer + 0.05 < float(random_report["mean_cer"])

    from minispeech.eval.discrete_n2 import paired_bootstrap_model_vs_wrong

    paired = paired_bootstrap_model_vs_wrong(
        model_report.get("edit_counts") or [],
        wrong_report.get("edit_counts") or [],
    )
    if paired and paired["separated"]:
        sep_wrong = True

    text_shuffle_hurts = float(shuffle_report["mean_cer"]) > fr_cer + 0.02
    text_mask_hurts = float(mask_report["mean_cer"]) > fr_cer + 0.02
    content_change = 0
    for a, b in zip(model_report["rows"], shuffle_report["rows"]):
        if (a.get("hyp") or "").strip() and (a.get("hyp") or "") != (b.get("hyp") or ""):
            content_change += 1
    prefix_cf_pass = content_change >= max(1, len(model_report["rows"]) // 4)

    cb0_ok = bool(model_report["cb0"].get("healthy"))
    no_runaway = not bool(model_report.get("runaway_or_repetition"))

    criteria = {
        "fr_cer": fr_cer,
        "copy_synth_cer_subset": copy_cer,
        "ceiling_copy_plus_margin": ceiling,
        "cer_pass": cer_pass,
        "bootstrap_sep_wrong_utt": sep_wrong,
        "bootstrap_sep_random": sep_random,
        "paired_vs_wrong": paired,
        "no_runaway_repetition": no_runaway,
        "cb0_healthy": cb0_ok,
        "cb0": model_report["cb0"],
        "text_usage_shuffle_hurts": text_shuffle_hurts,
        "text_usage_mask_hurts": text_mask_hurts,
        "text_usage_prefix_cf_changes": prefix_cf_pass,
        "shuffle_cer": float(shuffle_report["mean_cer"]),
        "mask_cer": float(mask_report["mean_cer"]),
        "wrong_utt_cer": float(wrong_report["mean_cer"]),
        "random_codes_cer": float(random_report["mean_cer"]),
    }
    hard = [
        cer_pass,
        sep_wrong,
        sep_random,
        no_runaway,
        cb0_ok,
        text_shuffle_hurts,
        text_mask_hurts,
        prefix_cf_pass,
    ]
    verdict = "PASS" if all(hard) else "FAIL"
    return {"gate": "N2", "verdict": verdict, "criteria": criteria}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default="configs/discrete_speech_lm_d1b_443m.yaml")
    ap.add_argument("--codec-tokens-dir", default=None)
    ap.add_argument(
        "--speaker-map",
        default=None,
        help="Path to speaker_maps JSON (default: config/trackn_v2.json)",
    )
    ap.add_argument(
        "--subset",
        default=None,
        help="Frozen oracle: dev32|confirm32|select8|gate24|full32|JSON path",
    )
    ap.add_argument(
        "--subset-split",
        default=None,
        help="Override split inside subset JSON (full32|select8|gate24)",
    )
    ap.add_argument(
        "--max-utts",
        type=int,
        default=None,
        help="Legacy first-N (forbidden when --subset is set)",
    )
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default=f"out/{PLAN_SLUG}/n2_eval")
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument(
        "--manifest",
        default="~/datasets/tts/multicorpus/manifest.jsonl",
        help="For reference text (CER)",
    )
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument(
        "--copy-synth-ceiling",
        type=float,
        default=0.0678,
        help="D0 E2 copy-synth CER fallback if subset copy fails",
    )
    ap.add_argument(
        "--n2-gate",
        action="store_true",
        help="Run full free-running ASR suite and write n2_gate.json",
    )
    ap.add_argument(
        "--free-running",
        action="store_true",
        help="Also run greedy generate (codes only unless --n2-gate)",
    )
    ap.add_argument("--wav-dir", default=f"out/{PLAN_SLUG}/listen_demo")
    ap.add_argument(
        "--decode-only",
        action="store_true",
        help="Generate+decode wavs only (no ASR); write n2_decode_bundle.json for --score-only",
    )
    ap.add_argument(
        "--score-only",
        default=None,
        help="Path to n2_decode_bundle.json; ASR-score wavs and emit n2_gate.json",
    )
    ap.add_argument(
        "--d1b",
        action="store_true",
        default=True,
        help="Use D1b N2 gate (macro/micro/catastrophic/stability/text-usage)",
    )
    ap.add_argument(
        "--legacy-n2",
        action="store_true",
        help="Use legacy D1 N2 gate only (disables D1b extras in verdict)",
    )
    ap.add_argument(
        "--infer-json",
        default=None,
        help="Optional JSON with infer knobs (used if discrete_config resolver missing)",
    )
    ap.add_argument(
        "--eval-contract",
        default=DEFAULT_EVAL_CONTRACT,
        help="Frozen eval_contract.json for fail-closed checks",
    )
    ap.add_argument(
        "--skip-eval-contract",
        action="store_true",
        help="Skip fail-closed eval-contract checks (non-canonical only)",
    )
    ap.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help=(
            "Confirm candidate optimizer step for formal D1c reports. Step is "
            "derived from payload['step'] (filename may only confirm); this "
            "flag may only agree with that derived value (cannot invent a "
            "foreign step for best.pt / unbound files)."
        ),
    )
    args = ap.parse_args()
    d1b = not bool(args.legacy_n2)

    # Never overwrite D1 artifacts for D1b defaults.
    if "d1_discrete_speech_lm" in str(args.out) and not args.legacy_n2:
        raise SystemExit(
            f"refusing D1b eval out under d1_discrete_speech_lm: {args.out!r}; "
            f"use out/{PLAN_SLUG}/..."
        )

    set_seed(42)
    out_dir = expand(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.score_only:
        return _score_n2_bundle(
            expand(args.score_only),
            asr_config=args.asr_config,
            out_dir=out_dir,
            copy_synth_ceiling=float(args.copy_synth_ceiling),
            d1b=d1b,
            eval_contract=args.eval_contract,
            skip_eval_contract=bool(args.skip_eval_contract),
        )

    if not args.ckpt or not args.codec_tokens_dir:
        raise SystemExit("Pass --ckpt and --codec-tokens-dir (or --score-only)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ckpt_payload = _load_model(expand(args.ckpt), device)
    # CLI yaml infer knobs override frozen ckpt snapshot (e.g. WAVE3c stop prior).
    file_cfg = load_yaml(args.config)
    if isinstance(file_cfg.get("infer"), dict):
        cfg.setdefault("infer", {}).update(file_cfg["infer"])
    infer_overrides = None
    if args.infer_json:
        # Merge via shared discrete_config resolver (scripts 73/74 parity).
        codebook_size = int((cfg.get("model") or {}).get("codebook_size", 2048))
        resolved = resolve_infer_settings(
            cfg, infer_json=args.infer_json, codebook_size=codebook_size
        )
        cfg["infer"] = dict(resolved)
        infer_overrides = cfg["infer"]

    utt_ids: list[str] | None = None
    subset_digest: str | None = None
    subset_label: str | None = args.subset
    if args.subset:
        utt_ids, _, subset_digest = load_frozen_utt_ids(
            args.subset, split=args.subset_split, verify_hash=True
        )
    elif args.max_utts is None and not args.legacy_n2:
        raise SystemExit(
            "Pass --subset (dev32|confirm32|select8|gate24|full32) for D1b eval; "
            "or --max-utts for legacy"
        )

    pad_id = int(
        cfg.get("data", {}).get("pad_token_id", cfg.get("model", {}).get("pad_token_id", 0))
    )
    collate = resolve_collate_codec_seq()
    speaker_registry, speaker_map_resolved = _resolve_speaker_registry(
        cfg=cfg if isinstance(cfg, dict) else {},
        speaker_map=args.speaker_map,
    )

    contract_payload: dict[str, Any] | None = None
    if not args.skip_eval_contract and not args.legacy_n2:
        # CLI --speaker-map must be checked against contract map hash/path.
        contract_payload = enforce_eval_contract(
            cfg,
            contract_path=args.eval_contract,
            subset_items_sha256=subset_digest,
            subset_label=subset_label,
            require_clean_sha=False,
            speaker_map_path=speaker_map_resolved,
        )

    d1c_contract = _is_d1c_semantic_contract(contract_payload)
    allowed_steps = None
    if d1c_contract:
        train_block = (contract_payload or {}).get("train") or {}
        raw_steps = train_block.get("candidate_steps") if isinstance(train_block, Mapping) else None
        allowed_steps = (
            [int(s) for s in raw_steps]
            if raw_steps
            else list(FR_CANDIDATE_STEPS_BASE)
        )
    checkpoint_step = resolve_checkpoint_step(
        ckpt_path=expand(args.ckpt),
        payload=ckpt_payload if isinstance(ckpt_payload, Mapping) else None,
        cli_step=args.checkpoint_step,
        allowed_steps=allowed_steps,
        require=bool(d1c_contract),
    )
    checkpoint_hashes: dict[str, str] | None = None
    if d1c_contract:
        checkpoint_hashes = require_checkpoint_payload_hashes(
            ckpt_payload if isinstance(ckpt_payload, Mapping) else None,
            expected_eval_contract=(
                str((contract_payload or {}).get("sha256") or "") or None
            ),
        )

    rows = load_codec_items(
        args.codec_tokens_dir,
        args.max_utts,
        speaker_registry=speaker_registry,
        speaker_mode="eval",
        utt_ids=utt_ids,
        require_items_sha256=subset_digest,
    )
    if not rows:
        raise SystemExit("no codec-token items found")
    if subset_digest is not None:
        summary_note = {
            "subset": subset_label,
            "subset_split": args.subset_split,
            "items_sha256": subset_digest,
            "n_utts": len(rows),
        }
        (out_dir / "subset_used.json").write_text(
            json.dumps(summary_note, indent=2) + "\n", encoding="utf-8"
        )

    term = ((cfg.get("model") or {}).get("termination") or {})
    semantic_mode = (
        getattr(model, "termination_mode", None) == "semantic_eos_v1"
        or (isinstance(term, dict) and term.get("mode") == "semantic_eos_v1")
    )
    eos_id = int(getattr(model, "eos_id", 2048))
    loader = DataLoader(
        rows,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(
            b,
            pad_token_id=pad_id,
            speech_eos_id=eos_id,
            termination_mode="semantic_eos_v1" if semantic_mode else "frame_stop_v1",
        ),
    )
    totals_batches: list[dict[str, Any]] = []
    for batch in loader:
        totals_batches.append(teacher_forced_ce(model, batch, device))
    summary: dict[str, Any] = aggregate_teacher_forced_batches(totals_batches)
    summary["n_utts"] = len(rows)
    summary["note"] = (
        "Teacher-forced CE is diagnostic only; promotion uses free-running CER/DER."
    )

    if args.free_running and not args.n2_gate and not args.decode_only:
        fr_lens = []
        with torch.no_grad():
            for row in rows[: min(8, len(rows))]:
                ids = row["input_ids"]
                if not torch.is_tensor(ids):
                    ids = torch.tensor(ids, dtype=torch.long)
                ids = ids.long().view(1, -1).to(device)
                spk = torch.tensor([int(row.get("speaker_id", 0))], device=device)
                codes = model.generate(ids, speaker_id=spk, max_frames=256, temperature=1.0)
                fr_lens.append(int(codes.shape[1]))
        summary["fr_mean_frames"] = float(sum(fr_lens) / max(len(fr_lens), 1))
        summary["fr_n"] = len(fr_lens)

    if args.n2_gate or args.decode_only:
        refs = _ref_texts_from_manifest(
            [str(r.get("utt_id")) for r in rows], args.manifest
        )
        max_frames = int(
            args.max_frames
            or cfg.get("infer", {}).get("max_frames")
            or 750
        )
        if d1b:
            # Frozen WAVE3c / eval_contract max_frames must not be GT-clamped.
            frozen_mf = 750
            if contract_payload is not None:
                frozen_mf = int((contract_payload.get("infer") or {}).get("max_frames") or 750)
            if int(max_frames) != int(frozen_mf):
                raise SystemExit(
                    f"D1b refuse max_frames={max_frames} != frozen contract {frozen_mf}"
                )
        else:
            # Legacy D1 only: safety ceiling well above GT.
            gt_max = max(int(r["codes"].shape[0]) for r in rows)
            max_frames = min(max_frames, max(64, int(gt_max * 2.5) + 16))

        wav_root = expand(args.wav_dir)
        skip_asr = bool(args.decode_only)
        common = dict(
            device=device,
            cfg=cfg,
            refs=refs,
            asr_config=args.asr_config,
            max_frames=max_frames,
            decode=True,
            skip_asr=skip_asr,
            infer_overrides=infer_overrides,
        )
        print(
            f"[n2] free-running model on {len(rows)} utts max_frames={max_frames} "
            f"skip_asr={skip_asr} d1b={d1b}"
        )
        model_rep = free_running_asr_eval(
            model, rows, out_wav_dir=wav_root / "model", condition="model", **common
        )
        print(f"[n2] copy_synth")
        copy_rep = free_running_asr_eval(
            model, rows, out_wav_dir=wav_root / "copy_synth", condition="copy_synth", **common
        )
        print(f"[n2] wrong_utt")
        wrong_rep = free_running_asr_eval(
            model, rows, out_wav_dir=wav_root / "wrong_utt", condition="wrong_utt", **common
        )
        print(f"[n2] random_codes")
        rand_rep = free_running_asr_eval(
            model, rows, out_wav_dir=wav_root / "random_codes", condition="random_codes", **common
        )
        print(f"[n2] shuffle_text (text-usage)")
        shuf_rep = free_running_asr_eval(
            model, rows, out_wav_dir=wav_root / "shuffle_text", condition="shuffle_text", **common
        )
        print(f"[n2] mask_text (text-usage)")
        mask_rep = free_running_asr_eval(
            model, rows, out_wav_dir=wav_root / "mask_text", condition="mask_text", **common
        )

        tf_ce = teacher_forced_metrics_for_emitter(summary)
        provenance = {
            "git_sha": git_sha(),
            "git_dirty": git_is_dirty(),
            "subset_label": subset_label,
            "subset_split": args.subset_split,
            "subset_items_sha256": subset_digest,
            "eval_contract_path": str(
                (contract_payload or {}).get("_path")
                or args.eval_contract
                or DEFAULT_EVAL_CONTRACT
            ),
            "eval_contract_sha256": (
                str((contract_payload or {}).get("sha256"))
                if contract_payload is not None
                else (
                    eval_contract_content_sha256(load_eval_contract(args.eval_contract))
                    if not args.legacy_n2 and not args.skip_eval_contract
                    else None
                )
            ),
            "max_frames": int(max_frames),
            "teacher_forced_ce": tf_ce,
        }
        if checkpoint_step is not None:
            provenance["checkpoint_step"] = int(checkpoint_step)
        if checkpoint_hashes is not None:
            provenance["checkpoint_hashes"] = dict(checkpoint_hashes)

        if args.decode_only:
            bundle = {
                "ckpt": str(expand(args.ckpt)),
                "wav_dir": str(wav_root),
                "copy_synth_ceiling": float(args.copy_synth_ceiling),
                "teacher_forced_ce": tf_ce,
                "provenance": provenance,
                "reports": {
                    "model": model_rep,
                    "copy_synth": copy_rep,
                    "wrong_utt": wrong_rep,
                    "random_codes": rand_rep,
                    "shuffle_text": shuf_rep,
                    "mask_text": mask_rep,
                },
            }
            for rep in bundle["reports"].values():
                # Keep JSON-serializable edit counts if present
                if "edit_counts" in rep:
                    rep["edit_counts"] = _edit_counts_to_jsonable(rep.get("edit_counts") or [])
            bundle_path = out_dir / "n2_decode_bundle.json"
            bundle_path.write_text(
                json.dumps(bundle, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )
            summary["n2_decode_bundle"] = str(bundle_path)
            print(f"wrote decode bundle {bundle_path} (run --score-only next)")
        else:
            if d1b:
                tf_for_gate = tf_ce if isinstance(tf_ce, Mapping) else {"total_ce": tf_ce}
                if d1c_contract:
                    gate = evaluate_n2_gate_d1c(
                        model_rep,
                        copy_rep,
                        wrong_rep,
                        rand_rep,
                        shuf_rep,
                        mask_rep,
                        teacher_forced_metrics=tf_for_gate,
                        copy_synth_ceiling=float(args.copy_synth_ceiling),
                    )
                else:
                    gate = evaluate_n2_gate_d1b(
                        model_rep,
                        copy_rep,
                        wrong_rep,
                        rand_rep,
                        shuf_rep,
                        mask_rep,
                        copy_synth_ceiling=float(args.copy_synth_ceiling),
                    )
                model_rows = gate.pop("model_rows")
                cat_rows = gate.pop("catastrophic_rows", [])
                gate["reports"] = gate.pop("enriched_reports")
            else:
                gate = evaluate_n2_gate(
                    model_rep,
                    copy_rep,
                    wrong_rep,
                    rand_rep,
                    shuf_rep,
                    mask_rep,
                    copy_synth_ceiling=float(args.copy_synth_ceiling),
                )
                model_rows = model_rep["rows"]
                cat_rows = model_rep.get("catastrophic_rows") or []
                gate["reports"] = {
                    k: {kk: vv for kk, vv in rep.items() if kk not in ("rows", "edit_counts")}
                    for k, rep in {
                        "model": model_rep,
                        "copy_synth": copy_rep,
                        "wrong_utt": wrong_rep,
                        "random_codes": rand_rep,
                        "shuffle_text": shuf_rep,
                        "mask_text": mask_rep,
                    }.items()
                }
            gate["artifacts"] = {
                "ckpt": str(expand(args.ckpt)),
                "wav_dir": str(wav_root),
                "eval_out": str(out_dir),
                "tf_summary": str(out_dir / "eval_summary.json"),
            }
            gate["teacher_forced_ce"] = tf_ce
            if checkpoint_step is not None:
                gate["step"] = int(checkpoint_step)
            gate["provenance"] = provenance
            # Attach acceptance helpers explicitly for consumers / selection
            gate["text_usage_acceptance"] = text_usage_acceptance(
                model_cer=float(gate["criteria"]["fr_cer"]),
                shuffle_cer=float(gate["criteria"]["shuffle_cer"]),
                mask_cer=float(gate["criteria"]["mask_cer"]),
                model_rows=model_rows,
                shuffle_rows=shuf_rep["rows"],
                paired_ci95_high=(
                    None
                    if not gate["criteria"].get("paired_vs_wrong")
                    else float(gate["criteria"]["paired_vs_wrong"]["ci95_high"])
                ),
            )
            gate["stability_acceptance"] = stability_acceptance(model_rep)
            gate_path = out_dir / "n2_gate.json"
            gate_path.write_text(
                json.dumps(gate, indent=2, default=str), encoding="utf-8"
            )
            (out_dir / "n2_model_rows.json").write_text(
                json.dumps(model_rows, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (out_dir / "n2_catastrophic_rows.json").write_text(
                json.dumps(cat_rows, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            summary["n2_gate"] = gate["verdict"]
            summary["n2_fr_cer"] = gate["criteria"]["fr_cer"]
            summary["n2_copy_cer"] = gate["criteria"]["copy_synth_cer_subset"]
            summary["n_catastrophic"] = gate["criteria"].get("n_catastrophic")
            print(
                json.dumps(
                    {"n2_verdict": gate["verdict"], **gate["criteria"]},
                    indent=2,
                    default=str,
                )
            )
            print(f"wrote {gate_path}")
            _emit_d1c_v2_from_gate(
                out_dir=out_dir,
                contract=contract_payload,
                ckpt=str(expand(args.ckpt)),
                provenance=dict(provenance),
                infer=dict(model_rep.get("infer") or {}),
                teacher_forced_ce=tf_ce,
                ordered_utt_ids=[str(r.get("utt_id")) for r in model_rows],
                condition_reports={
                    "model": {**model_rep, "rows": model_rows},
                    "copy_synth": copy_rep,
                    "wrong_utt": wrong_rep,
                    "random_codes": rand_rep,
                    "shuffle_text": shuf_rep,
                    "mask_text": mask_rep,
                },
                gate=gate,
            )

    out_path = out_dir / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
