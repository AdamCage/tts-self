#!/usr/bin/env python3
"""Offline cache Qwen3.5 text hidden states (layers 12/18/24 → indices 11/17/23).

C2 (CV17 / Track N) — cache only inventory after R1:

  python scripts/05_cache_text_hidden.py \\
    --config configs/data_common_voice_ru17.yaml \\
    --qwen-config configs/qwen_local.yaml \\
    --trackn-encode-union
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from minispeech.constants import CACHED_LAYER_INDICES, QWEN_MODEL_ID
from minispeech.data.manifest import read_manifest, write_manifest
from minispeech.data.shards import ShardWriter
from minispeech.frontend.stress import accentuate
from minispeech.frontend.verbalize import normalize_for_tts
from minispeech.utils import expand, load_yaml, set_seed

# Reuse the same allowlist helpers as latent encode (C1).
from importlib import import_module
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))
_encode = import_module("04_encode_latents")
load_utt_id_allowlist = _encode.load_utt_id_allowlist


def _frontend_text(raw: str) -> str:
    """TN → verbalize → stress (train/infer parity for Qwen input)."""
    spoken = normalize_for_tts(raw)
    return accentuate(spoken)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_ruslan.yaml")
    ap.add_argument("--qwen-config", default="configs/qwen.yaml")
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--utt-ids-file", default=None)
    ap.add_argument("--eval-set", action="append", default=None)
    ap.add_argument(
        "--trackn-encode-union",
        action="store_true",
        help="Cache only trackn_train∪val∪test∪review utt_ids.",
    )
    ap.add_argument(
        "--apply-frontend",
        action="store_true",
        help="Rebuild text_stressed via TN→verbalize→stress before caching.",
    )
    ap.add_argument(
        "--save-frontend-sidecar",
        default=None,
        help="Optional JSONL path for utt_id + frontend string (parity checks).",
    )
    args = ap.parse_args()
    set_seed(42)
    dcfg = load_yaml(args.config)["data"]
    qcfg = load_yaml(args.qwen_config)["qwen"]
    all_rows = read_manifest(dcfg["manifest"])
    allow = load_utt_id_allowlist(
        utt_ids_file=args.utt_ids_file,
        eval_sets=args.eval_set,
        trackn_encode_union=args.trackn_encode_union,
    )
    rows = [r for r in all_rows if r.utt_id in allow] if allow is not None else list(all_rows)
    if args.max_utts:
        rows = rows[: args.max_utts]
    print(
        json.dumps(
            {
                "manifest_n": len(all_rows),
                "cache_n": len(rows),
                "filter": "trackn_union" if args.trackn_encode_union else None,
                "apply_frontend": bool(args.apply_frontend),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    model_id = qcfg.get("model_id", QWEN_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        from transformers import Qwen3_5TextModel

        model = Qwen3_5TextModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    except Exception:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    model.eval()
    layers = tuple(qcfg.get("cached_layers", list(CACHED_LAYER_INDICES)))
    writer = ShardWriter(dcfg["text_hidden_dir"], "text_hidden", shard_size=dcfg.get("shard_size", 2000))

    sidecar = None
    if args.save_frontend_sidecar:
        sidecar_path = expand(args.save_frontend_sidecar)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar = sidecar_path.open("w", encoding="utf-8")

    max_length = int(qcfg.get("max_length", 256))
    n_at_max = 0
    bs = args.batch_size
    for i in tqdm(range(0, len(rows), bs), desc="text-hidden"):
        batch = rows[i : i + bs]
        texts: list[str] = []
        for r in batch:
            if args.apply_frontend:
                src = r.text_raw or r.text_stressed or ""
                text = _frontend_text(src)
                r.text_stressed = text
            else:
                text = r.text_stressed or r.text_raw or ""
            texts.append(text)
            if sidecar is not None:
                sidecar.write(
                    json.dumps(
                        {"utt_id": r.utt_id, "frontend_text": text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        for j, r in enumerate(batch):
            length = int(enc["attention_mask"][j].sum().item())
            if length >= max_length:
                n_at_max += 1
            stacked = torch.stack([hs[idx + 1][j, :length].float().cpu() for idx in layers], dim=0)
            # [n_layers, L, H]
            writer.add(r.utt_id, stacked.half(), {"text_len": length})
            r.input_ids = enc["input_ids"][j, :length].tolist()

    if sidecar is not None:
        sidecar.close()
    writer.close()
    # Always rewrite the full manifest (filtered cache must not truncate it).
    if allow is not None:
        by_id = {r.utt_id: r for r in rows}
        for r in all_rows:
            if r.utt_id in by_id:
                src = by_id[r.utt_id]
                r.text_stressed = src.text_stressed
                r.input_ids = src.input_ids
        write_manifest(dcfg["manifest"], all_rows)
    else:
        write_manifest(dcfg["manifest"], rows)
    meta = {
        "n": len(rows),
        "layers": list(layers),
        "model_id": model_id,
        "hidden_size": int(qcfg.get("hidden_size", 1024)),
        "max_length": max_length,
        "n_at_max_length": n_at_max,
        "frac_at_max_length": (n_at_max / len(rows)) if rows else 0.0,
        "apply_frontend": bool(args.apply_frontend),
    }
    print(json.dumps(meta, indent=2))
    expand(dcfg["text_hidden_dir"]).joinpath("text_hidden_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
