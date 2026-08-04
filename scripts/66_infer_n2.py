#!/usr/bin/env python3
"""Minimal N2 listen-demo infer (A4 stub; A7 owns full side-by-side compare).

Wavs go under ``out/n2_first_execution/n2_s1_listen_demo/`` by default.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from minispeech.codec.stats import LatentNormalizer
from minispeech.codec.wrapper import VibeVoiceCodec
from minispeech.constants import SR, get_target
from minispeech.frontend.stress import accentuate
from minispeech.frontend.verbalize import normalize_for_tts
from minispeech.infer.generate import save_wav
from minispeech.train.n2_trainer import load_n2_checkpoint
from minispeech.utils import expand, load_yaml

DEMO_PHRASES = [
    "привет! Я полностью самостоятельно обученная TTS-модель",
    "как у вас дела? Чем вам помочь?",
    "Сегодня хорошая погода, можно пройтись пешком.",
    "Пожалуйста, повторите ещё раз чуть медленнее.",
    "Спасибо за внимание, до скорой встречи!",
]


def _slug(text: str, n: int = 48) -> str:
    s = re.sub(r"[^\w]+", "_", text.lower(), flags=re.UNICODE).strip("_")
    return (s[:n] or "utt").encode("ascii", "ignore").decode("ascii") or f"utt"


def _frontend(text: str) -> str:
    return accentuate(normalize_for_tts(text))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/n2_qwen_latent_v01.yaml")
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--text", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument(
        "--out",
        default="out/n2_first_execution/n2_s1_listen_demo",
    )
    ap.add_argument("--max-frames", type=int, default=90, help="~12 s @ 7.5 Hz")
    ap.add_argument("--frames-per-char", type=float, default=0.55)
    args = ap.parse_args()
    if not args.text and not args.demo:
        raise SystemExit("pass --text or --demo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_n2_checkpoint(args.ckpt, device)
    cfg = ckpt.get("cfg") or load_yaml(args.config)
    qcfg = cfg.get("qwen", {})
    icfg = cfg.get("infer", {})
    dcfg = load_yaml(args.data)["data"]

    tok = AutoTokenizer.from_pretrained(
        str(qcfg.get("model_id")), trust_remote_code=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    try:
        normalizer = LatentNormalizer.from_file(dcfg["stats_path"])
    except Exception:
        normalizer = None

    codec = VibeVoiceCodec()
    out_dir = expand(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = DEMO_PHRASES if args.demo else [args.text]
    meta = []
    fr = get_target(str(cfg.get("target", "vibevoice64"))).frame_rate

    for i, raw in enumerate(texts, start=1):
        spoken = _frontend(raw)
        enc = tok(
            spoken,
            return_tensors="pt",
            truncation=True,
            max_length=int(qcfg.get("max_length", 256)),
        )
        input_ids = enc["input_ids"].to(device)
        text_mask = torch.ones_like(input_ids, dtype=torch.bool)
        n_chars = max(len(spoken), 1)
        max_frames = args.max_frames or max(
            8, int(round(n_chars * args.frames_per_char))
        )
        z_pred, info = model.generate(
            input_ids,
            text_mask,
            max_frames=max_frames,
            cfg_scale=float(icfg.get("cfg_scale", 1.3)),
            sample_steps=int(icfg.get("sample_steps", 10)),
        )
        z = z_pred
        if normalizer is not None:
            z = normalizer.denormalize(z)
        audio = codec.decode(z).audio  # [B,1,N]
        wav = audio[0, 0].detach().float().cpu().numpy()
        # Prefer ASCII stems on Windows mirrors (Cyrillic often becomes underscores).
        stem = f"{i:02d}_demo" if args.demo else "synth"
        path = out_dir / f"{stem}.wav"
        save_wav(path, wav, sr=SR)
        try:
            rel = str(path.relative_to(expand(".")))
        except ValueError:
            rel = str(path)
        row = {
            "text": raw,
            "text_frontend": spoken,
            "path": rel,
            "n_frames": int(info.get("n_frames", max_frames)),
            "dur_s": float(info.get("n_frames", max_frames)) / fr,
        }
        meta.append(row)
        print(f"Wrote {path}")

    (out_dir / "listen_index.json").write_text(
        json.dumps({"ckpt": str(args.ckpt), "items": meta}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
