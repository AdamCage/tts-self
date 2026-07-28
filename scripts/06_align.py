#!/usr/bin/env python3
"""Build alignment priors A/B/C (non-blocking). Default: C (monotonic Gaussian)."""

from __future__ import annotations

import argparse
import json

import torch
from tqdm import tqdm

from minispeech.data.alignment import (
    AlignLevel,
    monotonic_gaussian_prior,
    project_word_to_token_prior,
    validate_token_projection,
    word_level_prior,
)
from minispeech.data.manifest import read_manifest
from minispeech.data.shards import ShardReader, ShardWriter
from minispeech.utils import expand, load_yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_ruslan.yaml")
    ap.add_argument("--level", choices=["A", "B", "C", "auto"], default="C")
    ap.add_argument("--max-utts", type=int, default=None)
    args = ap.parse_args()
    dcfg = load_yaml(args.config)["data"]
    rows = read_manifest(dcfg["manifest"])
    if args.max_utts:
        rows = rows[: args.max_utts]

    # Try load forced alignment json if present; else C
    align_json = expand(dcfg.get("align_dir", "~/datasets/tts/ruslan/align")) / "forced.json"
    forced = {}
    if align_json.exists():
        forced = json.loads(align_json.read_text(encoding="utf-8"))

    level = AlignLevel(args.level) if args.level != "auto" else AlignLevel.GAUSSIAN
    writer = ShardWriter(dcfg["align_dir"], "align", shard_size=dcfg.get("shard_size", 2000))
    defect_rates = []

    # Need text lengths from text_hidden index if available
    try:
        thr = ShardReader(dcfg["text_hidden_dir"], "text_hidden")
    except Exception:
        thr = None
    try:
        latr = ShardReader(dcfg["latent_dir"], "latents")
    except Exception:
        latr = None

    for r in tqdm(rows, desc=f"align-{level.value}"):
        T = r.n_frames
        if latr and r.utt_id in latr.index:
            T = int(latr.get(r.utt_id).shape[0])
        L = len(r.input_ids) if r.input_ids else max(1, len(r.text_stressed.split()))
        if thr and r.utt_id in thr.index:
            th = thr.get(r.utt_id)
            L = th.shape[1] if th.ndim == 3 else th.shape[0]

        if r.utt_id in forced and level in (AlignLevel.WORD, AlignLevel.TOKEN, AlignLevel.GAUSSIAN):
            info = forced[r.utt_id]
            word_spans = [tuple(x) for x in info["word_spans_frames"]]
            token_word_ids = info["token_word_ids"]
            if level == AlignLevel.WORD or (
                level == AlignLevel.TOKEN
                and not validate_token_projection(info.get("defect_rate", 0.0))
            ):
                prior = word_level_prior(word_spans, token_word_ids, T, L)
                used = "A"
            elif level == AlignLevel.TOKEN:
                prior, dr = project_word_to_token_prior(
                    word_spans,
                    info.get("token_char_lens", [1] * L),
                    token_word_ids,
                    T,
                    L,
                )
                defect_rates.append(dr)
                used = "B" if validate_token_projection(dr) else "A"
                if used == "A":
                    prior = word_level_prior(word_spans, token_word_ids, T, L)
            else:
                prior = monotonic_gaussian_prior(T, L)
                used = "C"
        else:
            prior = monotonic_gaussian_prior(T, L)
            used = "C"

        writer.add(r.utt_id, prior.half(), {"level": used, "T": T, "L": L})

    writer.close()
    meta = {
        "requested": args.level,
        "mean_defect_rate": float(sum(defect_rates) / len(defect_rates)) if defect_rates else None,
        "n": len(rows),
    }
    print(json.dumps(meta, indent=2))
    expand(dcfg["align_dir"]).joinpath("align_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
