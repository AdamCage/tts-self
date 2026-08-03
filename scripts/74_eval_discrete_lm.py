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
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from minispeech.data.codec_seq_protocol import resolve_collate_codec_seq
from minispeech.data.d1b_oracles import (
    DEFAULT_EVAL_CONTRACT,
    PLAN_SLUG,
    enforce_eval_contract,
    load_frozen_utt_ids,
)
from minispeech.data.speaker_registry import SpeakerRegistry
from minispeech.eval.asr import load_asr_engine
from minispeech.eval.asr_wer import compute_cer
from minispeech.eval.bootstrap import edits_from_jiwer_chars
from minispeech.eval.discrete_n2 import (
    bootstrap_mean_cer,
    cb0_stats,
    enrich_model_report,
    estimate_stop_reason,
    evaluate_n2_gate_d1b,
    resolve_infer_settings,
    stability_acceptance,
    text_usage_acceptance,
)
from minispeech.frontend.normalize import normalize_for_asr
from minispeech.infer.discrete_config import GENERATE_INFER_KEYS
from minispeech.models.discrete_speech_lm import build_discrete_speech_lm
from minispeech.utils import expand, load_yaml, set_seed


def _load_model(ckpt: Path, device: torch.device):
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = payload.get("cfg") or load_yaml("configs/discrete_speech_lm_d1b_443m.yaml")
    model = build_discrete_speech_lm(cfg)
    sd = payload.get("model") or payload
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, cfg


def _resolve_speaker_registry(
    *,
    cfg: dict[str, Any],
    speaker_map: str | None,
) -> SpeakerRegistry:
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
            return SpeakerRegistry.load(p)
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
    with torch.no_grad():
        out = model(
            batch["input_ids"],
            batch["codes"],
            text_mask=batch.get("text_mask"),
            frame_mask=batch.get("frame_mask"),
            speaker_id=batch.get("speaker_id"),
            compute_loss=True,
        )
    assert out.loss is not None
    return {
        "loss": float(out.loss["total"].cpu()),
        "ce_cb0": float(out.loss["ce_cb0"].cpu()),
        "ce_residual": float(out.loss["ce_residual"].cpu()),
    }


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


def _gen_kw_from_infer(icfg: dict[str, Any], *, max_frames: int) -> dict[str, Any]:
    """Pass-through generate kwargs from shared resolver (no forked defaults)."""
    out = {k: icfg[k] for k in GENERATE_INFER_KEYS if k in icfg}
    out["max_frames"] = int(max_frames)
    return out


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

        with torch.no_grad():
            if condition == "copy_synth":
                codes = gt_codes.unsqueeze(0).to(device)
                stop_reason = "copy_synth"
            elif condition == "wrong_utt":
                other = rows[(i + 1) % n]
                oids = other["input_ids"]
                if not torch.is_tensor(oids):
                    oids = torch.tensor(oids, dtype=torch.long)
                oids_b = oids.long().view(1, -1).to(device)
                codes = model.generate(oids_b, speaker_id=spk, **gen_kw)
                stop_reason = estimate_stop_reason(
                    n_frames=int(codes.shape[1]),
                    text_len=int((oids_b != 0).sum().item()),
                    cb0_ids=codes[0, :, 0].detach().cpu().tolist(),
                    max_frames=max_frames,
                    min_frames=int(gen_kw["min_frames"]),
                    max_cb0_repeat=int(gen_kw["max_cb0_repeat"]),
                    frames_per_text_token=float(gen_kw["frames_per_text_token"]),
                )
            elif condition == "random_codes":
                T = min(max_frames, max(8, int(gt_codes.shape[0])))
                codes = torch.randint(0, 2048, (1, T, 16), device=device)
                stop_reason = "random_codes"
            elif condition == "shuffle_text":
                other = rows[(i + max(1, n // 3)) % n]
                oids = other["input_ids"]
                if not torch.is_tensor(oids):
                    oids = torch.tensor(oids, dtype=torch.long)
                oids_b = oids.long().view(1, -1).to(device)
                codes = model.generate(oids_b, speaker_id=spk, **gen_kw)
                stop_reason = estimate_stop_reason(
                    n_frames=int(codes.shape[1]),
                    text_len=int((oids_b != 0).sum().item()),
                    cb0_ids=codes[0, :, 0].detach().cpu().tolist(),
                    max_frames=max_frames,
                    min_frames=int(gen_kw["min_frames"]),
                    max_cb0_repeat=int(gen_kw["max_cb0_repeat"]),
                    frames_per_text_token=float(gen_kw["frames_per_text_token"]),
                )
            elif condition == "mask_text":
                masked = torch.zeros_like(ids)
                codes = model.generate(masked, speaker_id=spk, **gen_kw)
                stop_reason = estimate_stop_reason(
                    n_frames=int(codes.shape[1]),
                    text_len=text_len,
                    cb0_ids=codes[0, :, 0].detach().cpu().tolist(),
                    max_frames=max_frames,
                    min_frames=int(gen_kw["min_frames"]),
                    max_cb0_repeat=int(gen_kw["max_cb0_repeat"]),
                    frames_per_text_token=float(gen_kw["frames_per_text_token"]),
                )
            else:  # model
                codes = model.generate(ids, speaker_id=spk, **gen_kw)
                stop_reason = estimate_stop_reason(
                    n_frames=int(codes.shape[1]),
                    text_len=text_len,
                    cb0_ids=codes[0, :, 0].detach().cpu().tolist(),
                    max_frames=max_frames,
                    min_frames=int(gen_kw["min_frames"]),
                    max_cb0_repeat=int(gen_kw["max_cb0_repeat"]),
                    frames_per_text_token=float(gen_kw["frames_per_text_token"]),
                )

        gen_codes.append(codes[0].detach().cpu())
        ref = refs.get(uid, "")
        hyp_raw = ""
        cer_v = None
        wav_path = None
        if decode and codec is not None:
            dec = codec.decode([codes[0].detach().cpu()])
            audio = np.asarray(dec.audio[0], dtype=np.float32)
            sr = int(dec.sample_rate)
            stem = _ascii_utt(uid, i)
            wav_path = out_wav_dir / f"{stem}.wav"
            _write_wav(wav_path, audio, sr)
            if asr is not None:
                hyp_raw = asr.transcribe(audio, sr, use_cache=False)
                if ref:
                    ref_n = normalize_for_asr(ref)
                    hyp_n = normalize_for_asr(hyp_raw)
                    cer_v = float(compute_cer([ref_n], [hyp_n])) if ref_n else None
                    if cer_v is not None:
                        cers.append(cer_v)
                        edit_counts.append(edits_from_jiwer_chars(ref_n, hyp_n))

        rows_out.append(
            {
                "utt_id": uid,
                "condition": condition,
                "n_frames": int(codes.shape[1]),
                "gt_frames": int(gt_codes.shape[0]),
                "cer": cer_v,
                "ref": ref,
                "hyp": hyp_raw,
                "wav": str(wav_path) if wav_path else None,
                "stop_reason": stop_reason,
            }
        )

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


def _score_n2_bundle(
    bundle_path: Path,
    *,
    asr_config: str,
    out_dir: Path,
    copy_synth_ceiling: float,
    d1b: bool = True,
) -> int:
    """ASR-score a decode-only bundle (main venv with openai-whisper)."""
    import soundfile as sf

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
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
        )

    if not args.ckpt or not args.codec_tokens_dir:
        raise SystemExit("Pass --ckpt and --codec-tokens-dir (or --score-only)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _load_model(expand(args.ckpt), device)
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

    if not args.skip_eval_contract and not args.legacy_n2:
        enforce_eval_contract(
            cfg,
            contract_path=args.eval_contract,
            subset_items_sha256=subset_digest,
            subset_label=subset_label,
            require_clean_sha=False,
        )

    pad_id = int(
        cfg.get("data", {}).get("pad_token_id", cfg.get("model", {}).get("pad_token_id", 0))
    )
    collate = resolve_collate_codec_seq()
    speaker_registry = _resolve_speaker_registry(
        cfg=cfg if isinstance(cfg, dict) else {},
        speaker_map=args.speaker_map,
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

    loader = DataLoader(
        rows,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, pad_token_id=pad_id),
    )
    totals = {"loss": 0.0, "ce_cb0": 0.0, "ce_residual": 0.0}
    n = 0
    for batch in loader:
        m = teacher_forced_ce(model, batch, device)
        for k in totals:
            totals[k] += m[k]
        n += 1
    summary: dict[str, Any] = {k: v / max(n, 1) for k, v in totals.items()}
    summary["n_batches"] = n
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
        # Safety ceiling well above GT; real stop should fire earlier.
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

        if args.decode_only:
            bundle = {
                "ckpt": str(expand(args.ckpt)),
                "wav_dir": str(wav_root),
                "copy_synth_ceiling": float(args.copy_synth_ceiling),
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

    out_path = out_dir / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
