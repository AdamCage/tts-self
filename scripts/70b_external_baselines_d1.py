#!/usr/bin/env python3
"""Wave 0 external baselines for D1 (same frontend + frozen eval sets).

Writes wavs under ``out/d1_discrete_speech_lm/<system>/`` and JSON summaries
beside them. Prefer ``demo5`` first; expand with ``--sets``.

Systems:
  - silero_raw / silero_frontend — via scripts/39_external_baseline_eval.py
  - track_a — via scripts/32_eval_mel_suite.py
  - qwen3_tts_0.6b / qwen3_tts_1.7b — full Qwen3-TTS (needs qwen-tts)
  - cosyvoice3 — optional; skipped with reason if not installable

Example (WSL)::

    export PYTHONPATH=/mnt/c/projects/tts-self/src
    export MINISPEECH_ALLOW_DIRTY=1
    export HF_HOME=$HOME/hf_cache
    python scripts/70b_external_baselines_d1.py --systems silero_frontend,track_a --sets demo5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from minispeech.eval.asr import AsrConfig, AsrEngine
from minispeech.eval.asr_wer import compute_cer, compute_wer
from minispeech.eval.test_sets import load_test_set
from minispeech.frontend.verbalize import normalize_for_tts, normalize_verbalized_for_asr
from minispeech.utils import expand, git_is_dirty, git_sha

PLAN_SLUG = "d1_discrete_speech_lm"
OUT_ROOT = Path(f"out/{PLAN_SLUG}")


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    return subprocess.call(cmd, env=env)


def _run_silero(sets: str, input_mode: str, out_root: Path) -> int:
    system = f"silero_{input_mode}"
    wav_dir = out_root / system
    out_json = out_root / f"{system}_metrics.json"
    script = str(Path(__file__).resolve().parent / "39_external_baseline_eval.py")
    return _run(
        [
            sys.executable,
            script,
            "--sets",
            sets,
            "--input-mode",
            input_mode,
            "--wav-dir",
            str(wav_dir),
            "--out",
            str(out_json),
            "--allow-download",
        ]
    )


def _run_track_a(sets: str, out_root: Path, ckpt: str) -> int:
    wav_dir = out_root / "track_a"
    out_json = out_root / "track_a_metrics.json"
    script = str(Path(__file__).resolve().parent / "32_eval_mel_suite.py")
    cmd = [
        sys.executable,
        script,
        "--ckpt",
        ckpt,
        "--sets",
        sets,
        "--out",
        str(out_json),
        "--wav-dir",
        str(wav_dir),
    ]
    # Some suite versions use --save-wav-dir; try primary first.
    rc = _run(cmd)
    if rc != 0:
        cmd2 = [
            sys.executable,
            script,
            "--ckpt",
            ckpt,
            "--sets",
            sets,
            "--out",
            str(out_json),
        ]
        rc = _run(cmd2)
    return rc


def _run_qwen3_tts(sets: str, out_root: Path, model_id: str, system: str) -> int:
    """Synthesize phrases with full Qwen3-TTS CustomVoice/Base if available."""
    try:
        from qwen_tts import Qwen3TTSModel
    except Exception as exc:  # noqa: BLE001
        payload = {
            "status": "blocked",
            "system": system,
            "model_id": model_id,
            "reason": f"qwen_tts import failed: {type(exc).__name__}: {exc}",
            "git_sha": git_sha(),
            "git_dirty": git_is_dirty(),
        }
        _write(out_root / f"{system}_metrics.json", payload)
        print(json.dumps(payload, indent=2))
        return 2

    import soundfile as sf
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    wav_root = out_root / system
    wav_root.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "status": "ok",
        "system": system,
        "model_id": model_id,
        "device": device,
        "frontend": "normalize_for_tts",
        "sets": {},
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
    }
    try:
        model = Qwen3TTSModel.from_pretrained(model_id, device_map=device)
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "blocked"
        meta["reason"] = f"model load failed: {type(exc).__name__}: {exc}"
        _write(out_root / f"{system}_metrics.json", meta)
        print(json.dumps(meta, indent=2))
        return 2

    speaker = None
    if hasattr(model, "get_supported_speakers"):
        try:
            speakers = list(model.get_supported_speakers() or [])
            meta["supported_speakers"] = speakers[:32]
            # Prefer a common English/neutral id if present; else first.
            for cand in ("ryan", "aiden", "vivian", "serena", "uncle_fu"):
                if any(str(s).lower() == cand for s in speakers):
                    speaker = next(s for s in speakers if str(s).lower() == cand)
                    break
            if speaker is None and speakers:
                speaker = speakers[0]
        except Exception as exc:  # noqa: BLE001
            meta["speaker_probe_error"] = f"{type(exc).__name__}: {exc}"
    meta["speaker"] = speaker

    asr = AsrEngine(AsrConfig.from_yaml("configs/asr.yaml"), device=device)
    asr._load()
    set_names = [s.strip() for s in sets.split(",") if s.strip()]
    n_errors = 0
    for set_id in set_names:
        ts = load_test_set(f"configs/eval_sets/{set_id}.json")
        set_dir = wav_root / set_id
        set_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for i, item in enumerate(ts.items, start=1):
            row: dict[str, Any] = {"utt_id": item.utt_id, "text": item.text}
            if not item.text:
                row["error"] = "missing text"
                n_errors += 1
                rows.append(row)
                continue
            text_in = normalize_for_tts(item.text)
            wav_path = set_dir / f"{i:04d}_{item.utt_id}.wav"
            try:
                if "CustomVoice" in model_id or hasattr(model, "generate_custom_voice"):
                    if not speaker:
                        raise RuntimeError("CustomVoice requires a speaker id; none available")
                    out = model.generate_custom_voice(
                        text=text_in,
                        speaker=str(speaker),
                        language="Russian",
                    )
                elif hasattr(model, "generate_voice_clone"):
                    raise RuntimeError(
                        "Base/VoiceClone path needs ref_audio; use CustomVoice for Wave 0 demo"
                    )
                else:
                    raise RuntimeError("No supported generate_* API on Qwen3TTSModel")
                if isinstance(out, tuple):
                    wavs, sr = out[0], out[1]
                    wav = np.asarray(
                        wavs[0] if isinstance(wavs, (list, tuple)) else wavs,
                        dtype=np.float32,
                    )
                elif isinstance(out, dict) and "wav" in out:
                    wav = np.asarray(out["wav"], dtype=np.float32)
                    sr = int(out.get("sr") or out.get("sample_rate") or 24000)
                else:
                    wav = np.asarray(out, dtype=np.float32)
                    sr = 24000
                wav = wav.reshape(-1)
                sf.write(str(wav_path), wav, sr)
                hyp_raw = asr.transcribe(wav, sr) or ""
                ref_n = normalize_verbalized_for_asr(item.text)
                hyp_n = normalize_verbalized_for_asr(hyp_raw)
                row.update(
                    {
                        "wav": str(wav_path),
                        "model_input_text": text_in,
                        "speaker": str(speaker) if speaker else None,
                        "hyp_raw": hyp_raw,
                        "hyp": hyp_n,
                        "ref_norm": ref_n,
                        "cer": float(compute_cer([ref_n], [hyp_n])),
                        "wer": float(compute_wer([ref_n], [hyp_n])),
                        "sr": sr,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
                n_errors += 1
            rows.append(row)
            print(f"[{system}/{set_id}] {item.utt_id} cer={row.get('cer')}", flush=True)
        cers = [r["cer"] for r in rows if r.get("cer") is not None]
        meta["sets"][set_id] = {
            "n": len(rows),
            "mean_cer": float(np.mean(cers)) if cers else None,
            "utterances": rows,
        }
    meta["status"] = "ok" if n_errors == 0 else "partial"
    meta["n_errors"] = n_errors
    _write(out_root / f"{system}_metrics.json", meta)
    return 0 if n_errors == 0 else 1


def _run_cosyvoice(sets: str, out_root: Path) -> int:
    payload = {
        "status": "blocked",
        "system": "cosyvoice3",
        "reason": (
            "CosyVoice3 not installed in this environment. "
            "Install separately if needed; Wave 0 continues without it."
        ),
        "sets_requested": sets,
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
    }
    try:
        import cosyvoice  # type: ignore  # noqa: F401

        payload["reason"] = "cosyvoice import succeeded but D1 harness not wired yet"
        payload["status"] = "blocked"
    except Exception:
        pass
    _write(out_root / "cosyvoice3_metrics.json", payload)
    print(json.dumps(payload, indent=2))
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", default="demo5")
    ap.add_argument(
        "--systems",
        default="silero_raw,silero_frontend,track_a,qwen3_tts_0.6b,cosyvoice3",
        help="Comma list of systems to run",
    )
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--mel-ckpt", default="runs/mel_stress_v01_s4/best.pt")
    ap.add_argument(
        "--qwen-0p6",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    )
    ap.add_argument(
        "--qwen-1p7",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    )
    args = ap.parse_args()
    out_root = expand(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    results: dict[str, Any] = {
        "plan_slug": PLAN_SLUG,
        "sets": args.sets,
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
        "systems": {},
    }
    rc_all = 0
    for sys_name in systems:
        if sys_name == "silero_raw":
            rc = _run_silero(args.sets, "raw", out_root)
        elif sys_name == "silero_frontend":
            rc = _run_silero(args.sets, "frontend", out_root)
        elif sys_name == "track_a":
            rc = _run_track_a(args.sets, out_root, args.mel_ckpt)
        elif sys_name in {"qwen3_tts_0.6b", "qwen3_tts_0.6B"}:
            rc = _run_qwen3_tts(args.sets, out_root, args.qwen_0p6, "qwen3_tts_0.6b")
        elif sys_name in {"qwen3_tts_1.7b", "qwen3_tts_1.7B"}:
            rc = _run_qwen3_tts(args.sets, out_root, args.qwen_1p7, "qwen3_tts_1.7b")
        elif sys_name == "cosyvoice3":
            rc = _run_cosyvoice(args.sets, out_root)
        else:
            print(f"Unknown system {sys_name}", flush=True)
            rc = 2
        results["systems"][sys_name] = {"returncode": rc}
        if rc not in (0, 2):
            rc_all = rc
    _write(out_root / "external_baselines_summary.json", results)
    print(json.dumps(results, indent=2))
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
