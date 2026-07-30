#!/usr/bin/env python3
"""E4: synth CER/WER on fixed demo / listen set for known checkpoints.

Default ckpts (if present):
  - runs/.../S2b_last.pt or similar
  - runs/v01_regression_direct/best.pt  (S3 calibrated)
  - runs/.../S4_fr_best*.pt

Also scans WAV dirs for NEAR_SILENT (peak/rms) — useful for S4 demo triage
(often wrong stats path → near-zero PCM after denormalize).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer

from minispeech.codec.stats import LatentNormalizer
from minispeech.codec.wrapper import VibeVoiceCodec
from minispeech.constants import QWEN_MODEL_ID, SR
from minispeech.eval.asr import AsrConfig, AsrEngine
from minispeech.eval.asr_wer import compute_cer, compute_wer
from minispeech.frontend.normalize import normalize_for_asr
from minispeech.infer.generate import load_checkpoint, save_wav, synthesize
from minispeech.utils import expand, load_yaml

# Same phrases as scripts/17_demo_phrases.py
DEMO_PHRASES: list[str] = [
    "привет! Я полностью самостоятельно обученная TTS-модель",
    "как у вас дела? Чем вам помочь?",
    "Сегодня хорошая погода, можно пройтись пешком.",
    "Пожалуйста, повторите ещё раз чуть медленнее.",
    "Спасибо за внимание, до скорой встречи!",
]

DEFAULT_CKPTS = {
    "S2b": [
        "runs/v01_regression_direct_s2b/last.pt",
        "runs/v01_regression_direct/S2b_last.pt",
        "runs/S2b_last.pt",
    ],
    "S3_calibrated": [
        "runs/v01_regression_direct/best.pt",
        "runs/v01_regression_direct/S3_calibrated_best.pt",
    ],
    "S4_fr_best": [
        "runs/v01_regression_direct/S4_fr_best_step3000.pt",
        "runs/v01_regression_direct/S4_fr_best.pt",
        "runs/S4_fr_best_step3000.pt",
    ],
}


def _resolve_ckpt(candidates: list[str]) -> Path | None:
    for c in candidates:
        p = expand(c)
        if p.is_file():
            return p
    return None


def wav_sanity(path: Path) -> dict:
    x, sr = sf.read(str(path), dtype="float32")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    dur = len(x) / sr if sr else 0.0
    flag = "ok"
    if peak < 1e-3 or rms < 1e-4:
        flag = "NEAR_SILENT"
    elif peak < 0.05 or rms < 0.005:
        flag = "WEAK"
    return {"path": str(path), "dur_s": dur, "peak": peak, "rms": rms, "flag": flag}


def scan_wav_dir(d: Path) -> list[dict]:
    if not d.is_dir():
        return []
    return [wav_sanity(p) for p in sorted(d.glob("*.wav"))]


def _resolve_qwen(infer_cfg: dict, qwen_config: str | None) -> str:
    if qwen_config:
        mid = load_yaml(qwen_config).get("qwen", {}).get("model_id")
        if mid:
            return str(expand(mid))
    return str(infer_cfg.get("qwen", QWEN_MODEL_ID))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--infer", default="configs/infer.yaml")
    ap.add_argument("--qwen-config", default="configs/qwen_local.yaml")
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml", help="For stats fallback")
    ap.add_argument("--phrases", default=None, help="Optional text file, one phrase per line")
    ap.add_argument("--out", default="out/eval_intelligibility.json")
    ap.add_argument("--wav-out", default="out/intelligibility_wavs")
    ap.add_argument("--skip-synth", action="store_true", help="Only scan existing WAV dirs")
    ap.add_argument(
        "--scan-dirs",
        default="",
        help="Comma-separated existing demo dirs to sanity-check (NEAR_SILENT)",
    )
    ap.add_argument("--ckpt", action="append", default=None, help="name=path (repeatable)")
    args = ap.parse_args()

    # Resolve ckpts
    ckpts: dict[str, Path] = {}
    if args.ckpt:
        for item in args.ckpt:
            if "=" not in item:
                raise SystemExit(f"--ckpt expects name=path, got {item}")
            name, path = item.split("=", 1)
            p = expand(path)
            if p.is_file():
                ckpts[name] = p
            else:
                print(f"WARN: missing ckpt {name}={path}")
    else:
        for name, cands in DEFAULT_CKPTS.items():
            p = _resolve_ckpt(cands)
            if p is not None:
                ckpts[name] = p
            else:
                print(f"WARN: no ckpt found for {name} (tried {cands})")

    phrases = list(DEMO_PHRASES)
    if args.phrases:
        phrases = [
            ln.strip()
            for ln in expand(args.phrases).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    report: dict = {
        "phrases": phrases,
        "ckpts_resolved": {k: str(v) for k, v in ckpts.items()},
        "sanity_scans": {},
        "results": {},
        "near_silent_hint": (
            "NEAR_SILENT demos usually mean denormalize used wrong/missing stats "
            "(infer.stats) or model emitted near-zero normalized latents. "
            "Compare peak/rms vs S3 calibrated demos; verify stats path equals "
            "data.stats_path from training."
        ),
    }

    # Scan existing dirs
    scan_dirs = [s.strip() for s in args.scan_dirs.split(",") if s.strip()]
    for d in scan_dirs:
        rows = scan_wav_dir(expand(d))
        report["sanity_scans"][d] = rows
        n_ns = sum(1 for r in rows if r["flag"] == "NEAR_SILENT")
        print(f"scan {d}: {n_ns}/{len(rows)} NEAR_SILENT")

    if args.skip_synth:
        out = expand(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out}")
        return 0

    if not ckpts:
        print("No checkpoints resolved; use --ckpt name=path or --skip-synth --scan-dirs")
        out = expand(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return 2

    infer_cfg = load_yaml(args.infer)["infer"]
    dcfg = load_yaml(args.data)["data"]
    stats_path = infer_cfg.get("stats") or dcfg["stats_path"]
    print(f"stats={stats_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = VibeVoiceCodec(model_id=infer_cfg.get("codec", "microsoft/VibeVoice-AcousticTokenizer"))
    normalizer = LatentNormalizer.from_file(stats_path)

    asr = None
    try:
        asr = AsrEngine(AsrConfig.from_yaml(args.asr_config), device=device)
        asr._load()
    except Exception as e:
        print(f"ASR unavailable: {e}")

    qwen_id = _resolve_qwen(infer_cfg, args.qwen_config)
    tok = AutoTokenizer.from_pretrained(qwen_id, trust_remote_code=True)
    try:
        from transformers import Qwen3_5TextModel

        text_model = Qwen3_5TextModel.from_pretrained(
            qwen_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    except Exception:
        from transformers import AutoModel

        text_model = AutoModel.from_pretrained(
            qwen_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )

    wav_root = expand(args.wav_out)
    wav_root.mkdir(parents=True, exist_ok=True)

    for name, ckpt in ckpts.items():
        print(f"=== {name}: {ckpt} ===")
        model, _ = load_checkpoint(ckpt, device)
        out_dir = wav_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        refs, hyps = [], []
        utt_rows = []
        for i, text in enumerate(phrases, start=1):
            audio, info = synthesize(
                text,
                model,
                codec,
                normalizer,
                tok,
                text_model,
                cfg_scale=infer_cfg.get("cfg_scale", 1.3),
                flow_steps=infer_cfg.get("flow_steps", 4),
                use_stress=infer_cfg.get("stress", {}).get("use_accentuator", True),
                stop=infer_cfg.get("stop"),
            )
            slug = f"{i:02d}.wav"
            wp = out_dir / slug
            save_wav(wp, audio)
            san = wav_sanity(wp)
            ref_n = normalize_for_asr(text)
            hyp_n = ""
            cer = wer = None
            if asr is not None:
                hyp_n = normalize_for_asr(asr.transcribe(audio, SR))
                cer = compute_cer([ref_n], [hyp_n])
                wer = compute_wer([ref_n], [hyp_n])
                refs.append(ref_n)
                hyps.append(hyp_n)
            utt_rows.append(
                {
                    "text": text,
                    "wav": str(wp),
                    "cer": cer,
                    "wer": wer,
                    "hyp": hyp_n,
                    "sanity": san,
                    "n_frames": int(info.get("n_frames", 0)),
                }
            )
            print(f"  [{i}] cer={cer} flag={san['flag']} peak={san['peak']:.4f}")

        report["results"][name] = {
            "ckpt": str(ckpt),
            "stats": str(stats_path),
            "mean_cer": float(compute_cer(refs, hyps)) if refs else None,
            "mean_wer": float(compute_wer(refs, hyps)) if refs else None,
            "n_near_silent": sum(1 for r in utt_rows if r["sanity"]["flag"] == "NEAR_SILENT"),
            "utterances": utt_rows,
        }
        report["sanity_scans"][str(out_dir)] = [r["sanity"] for r in utt_rows]

    out = expand(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
