#!/usr/bin/env python3
"""Offline cache Qwen3.5 text hidden states (layers 12/18/24 → indices 11/17/23)."""

from __future__ import annotations

import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from minispeech.constants import CACHED_LAYER_INDICES, QWEN_MODEL_ID
from minispeech.data.manifest import read_manifest, write_manifest
from minispeech.data.shards import ShardWriter
from minispeech.utils import expand, load_yaml, set_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_ruslan.yaml")
    ap.add_argument("--qwen-config", default="configs/qwen.yaml")
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    set_seed(42)
    dcfg = load_yaml(args.config)["data"]
    qcfg = load_yaml(args.qwen_config)["qwen"]
    rows = read_manifest(dcfg["manifest"])
    if args.max_utts:
        rows = rows[: args.max_utts]

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

    bs = args.batch_size
    for i in tqdm(range(0, len(rows), bs), desc="text-hidden"):
        batch = rows[i : i + bs]
        texts = [r.text_stressed for r in batch]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=qcfg.get("max_length", 256),
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        for j, r in enumerate(batch):
            length = int(enc["attention_mask"][j].sum().item())
            stacked = torch.stack([hs[idx + 1][j, :length].float().cpu() for idx in layers], dim=0)
            # [n_layers, L, H]
            writer.add(r.utt_id, stacked.half(), {"text_len": length})
            r.input_ids = enc["input_ids"][j, :length].tolist()

    writer.close()
    write_manifest(dcfg["manifest"], rows)
    meta = {"n": len(rows), "layers": list(layers), "model_id": model_id}
    print(json.dumps(meta, indent=2))
    expand(dcfg["text_hidden_dir"]).joinpath("text_hidden_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
