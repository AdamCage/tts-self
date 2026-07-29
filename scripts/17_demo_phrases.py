#!/usr/bin/env python3
"""Synthesize a fixed list of Russian demo phrases (research TTS only)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer

from minispeech.codec.stats import LatentNormalizer
from minispeech.codec.wrapper import VibeVoiceCodec
from minispeech.constants import QWEN_MODEL_ID, SR
from minispeech.infer.generate import load_checkpoint, save_wav, synthesize
from minispeech.utils import expand, load_yaml

# Everyday / neutral demo lines (research synthesis only — not impersonation).
DEMO_PHRASES: list[str] = [
    "привет! Я полностью самостоятельно обученная TTS-модель",
    "как у вас дела? Чем вам помочь?",
    "Сегодня хорошая погода, можно пройтись пешком.",
    "Пожалуйста, повторите ещё раз чуть медленнее.",
    "Спасибо за внимание, до скорой встречи!",
]


def _resolve_qwen_id(infer_cfg: dict, qwen_config: str | None) -> str:
    if qwen_config:
        qcfg = load_yaml(qwen_config).get("qwen", {})
        mid = qcfg.get("model_id")
        if mid:
            return str(expand(mid))
    return str(infer_cfg.get("qwen", QWEN_MODEL_ID))


def _slug(i: int, text: str) -> str:
    words = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower()).split()
    stub = "_".join(words[:6])[:48] or "phrase"
    return f"{i:02d}_{stub}.wav"


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo phrase inference for MiniSpeech-RU")
    ap.add_argument("--ckpt", default="runs/v01_regression_direct/best.pt")
    ap.add_argument("--config", default="configs/infer.yaml")
    ap.add_argument("--qwen-config", default="configs/qwen_local.yaml")
    ap.add_argument("--out-dir", default="out/demo_phrases")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument(
        "--windows-mirror",
        default="/mnt/c/projects/tts-self/out/demo_phrases",
        help="Also copy WAVs here for Windows playback (empty to skip)",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)["infer"]
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = expand(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device} ckpt={args.ckpt}")
    model, _ = load_checkpoint(args.ckpt, device)
    codec = VibeVoiceCodec(model_id=cfg.get("codec", "microsoft/VibeVoice-AcousticTokenizer"))
    normalizer = LatentNormalizer.from_file(cfg["stats"])

    qwen_id = _resolve_qwen_id(cfg, args.qwen_config)
    print(f"qwen={qwen_id}")
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

    written: list[Path] = []
    for i, text in enumerate(DEMO_PHRASES, start=1):
        audio, info = synthesize(
            text,
            model,
            codec,
            normalizer,
            tok,
            text_model,
            cfg_scale=cfg.get("cfg_scale", 1.3),
            flow_steps=cfg.get("flow_steps", 4),
            use_stress=cfg.get("stress", {}).get("use_accentuator", True),
            stop=cfg.get("stop"),
        )
        wav_path = out_dir / _slug(i, text)
        save_wav(wav_path, audio)
        n_frames = int(info.get("n_frames", 0))
        dur_s = float(len(audio) / SR)
        print(
            f"[{i}/{len(DEMO_PHRASES)}] {wav_path}\n"
            f"  text={text!r}\n"
            f"  stressed={info.get('text_stressed')!r}\n"
            f"  duration_s={dur_s:.3f} n_frames={n_frames}"
        )
        written.append(Path(wav_path))

    mirror = (args.windows_mirror or "").strip()
    if mirror:
        mirror_dir = Path(mirror)
        mirror_dir.mkdir(parents=True, exist_ok=True)
        for p in written:
            dest = mirror_dir / p.name
            shutil.copy2(p, dest)
            print(f"mirrored -> {dest}")

    print(f"Done: {len(written)} WAVs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
