#!/usr/bin/env python3
"""Offline encode with sample=False; write sharded latents + stats.

C1 (CV17 / Track N) — encode only inventory after R1, not the full corpus:

  python scripts/04_encode_latents.py \\
    --config configs/data_common_voice_ru17.yaml \\
    --trackn-encode-union

Or pass explicit sets / a plain id list:

  --eval-set configs/eval_sets/trackn_train.json --eval-set ...
  --utt-ids-file path/to/ids.txt   # one utt_id per line
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import soundfile as sf
import torch
from tqdm import tqdm

from minispeech.codec.stats import compute_latent_stats, save_stats
from minispeech.codec.wrapper import VibeVoiceCodec
from minispeech.data.manifest import read_manifest, write_manifest
from minispeech.data.shards import ShardWriter
from minispeech.utils import expand, load_yaml, set_seed

DEFAULT_TRACKN_EVAL_SETS = (
    "configs/eval_sets/trackn_train.json",
    "configs/eval_sets/trackn_val.json",
    "configs/eval_sets/trackn_test.json",
    "configs/eval_sets/trackn_review.json",
)


def _load_utt_ids_file(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
        if isinstance(raw, dict):
            if isinstance(raw.get("utt_ids"), list):
                return {str(x) for x in raw["utt_ids"]}
            items = raw.get("items")
            if isinstance(items, list):
                return {str(it["utt_id"]) for it in items if it.get("utt_id")}
        if isinstance(raw, list):
            return {str(x) for x in raw}
        raise ValueError(f"{path}: expected utt_ids / items JSON")
    out: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def load_utt_id_allowlist(
    *,
    utt_ids_file: str | None,
    eval_sets: Iterable[str] | None,
    trackn_encode_union: bool,
) -> set[str] | None:
    """Return allowlist of utt_ids, or None if no filter was requested."""
    paths: list[str] = []
    if utt_ids_file:
        paths.append(utt_ids_file)
    if eval_sets:
        paths.extend(eval_sets)
    if trackn_encode_union:
        paths.extend(DEFAULT_TRACKN_EVAL_SETS)
    if not paths:
        return None
    allow: set[str] = set()
    for p in paths:
        allow |= _load_utt_ids_file(expand(p))
    return allow


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Encode acoustic latents (sample=False). Optional Track N utt filter."
    )
    ap.add_argument("--config", default="configs/data_ruslan.yaml")
    ap.add_argument("--codec-config", default="configs/codec.yaml")
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument(
        "--utt-ids-file",
        default=None,
        help="Plain text (one id/line) or JSON with utt_ids; encode only these rows.",
    )
    ap.add_argument(
        "--eval-set",
        action="append",
        default=None,
        help="Frozen eval-set JSON (kind=utt_ids). Repeatable; unioned with --utt-ids-file.",
    )
    ap.add_argument(
        "--trackn-encode-union",
        action="store_true",
        help=(
            "Encode only utt_ids in trackn_train∪val∪test∪review "
            f"({', '.join(DEFAULT_TRACKN_EVAL_SETS)})."
        ),
    )
    args = ap.parse_args()
    set_seed(42)
    dcfg = load_yaml(args.config)["data"]
    ccfg = load_yaml(args.codec_config)["codec"]

    all_rows = read_manifest(dcfg["manifest"])
    allow = load_utt_id_allowlist(
        utt_ids_file=args.utt_ids_file,
        eval_sets=args.eval_set,
        trackn_encode_union=args.trackn_encode_union,
    )
    if allow is not None:
        rows = [r for r in all_rows if r.utt_id in allow]
    else:
        rows = list(all_rows)
    if args.max_utts:
        rows = rows[: args.max_utts]

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    codec = VibeVoiceCodec(model_id=ccfg["model_id"], dtype=dtype)
    print(f"latent layout: {codec._latent_layout}")
    print(
        json.dumps(
            {
                "manifest_n": len(all_rows),
                "encode_n": len(rows),
                "filter": (
                    "trackn_union"
                    if args.trackn_encode_union
                    else ("allowlist" if allow is not None else None)
                ),
                "allowlist_n": (len(allow) if allow is not None else None),
                "sample": False,
            },
            indent=2,
        )
    )

    writer = ShardWriter(dcfg["latent_dir"], "latents", shard_size=dcfg.get("shard_size", 2000))
    train_latents: list[torch.Tensor] = []
    all_encoded: list[torch.Tensor] = []
    n_frames_mismatch = 0
    sample_rates: dict[str, int] = {}
    by_id = {r.utt_id: r for r in all_rows}

    for r in tqdm(rows, desc="encode"):
        audio, sr = sf.read(r.wav, dtype="float32")
        sample_rates[str(int(sr))] = sample_rates.get(str(int(sr)), 0) + 1
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        # DC only
        audio = audio - float(audio.mean())
        enc = codec.encode(audio, sr, sample=False)
        z = enc.latents[0].float().cpu()  # [T, 64]
        # Prefer encode length when manifest n_frames disagrees (C1 frequency check).
        if r.n_frames and int(r.n_frames) != int(z.shape[0]):
            n_frames_mismatch += 1
            t_true = int(z.shape[0])
        else:
            t_true = min(z.shape[0], r.n_frames or z.shape[0])
        z = z[:t_true]
        r.n_frames = int(z.shape[0])
        if r.utt_id in by_id:
            by_id[r.utt_id].n_frames = r.n_frames
        writer.add(r.utt_id, z.half(), {"n_frames": r.n_frames})
        all_encoded.append(z)
        if r.split == "train":
            train_latents.append(z)

    writer.close()
    stats_src = train_latents if train_latents else all_encoded
    if not stats_src:
        raise SystemExit("no utterances encoded; check filter / manifest")
    stats = compute_latent_stats(stats_src)
    save_stats(stats, dcfg["stats_path"])
    # Always rewrite the full manifest (filtered encode must not truncate it).
    write_manifest(dcfg["manifest"], all_rows)
    meta = {
        "n": len(rows),
        "manifest_n": len(all_rows),
        "ac1_mean": float(stats["latent_ac1"].float().mean()),
        "layout": codec._latent_layout,
        "sample": False,
        "n_frames_mismatch": n_frames_mismatch,
        "sample_rate_hist": sample_rates,
        "stats_n": len(stats_src),
        "stats_from": "train" if train_latents else "all_encoded",
        "filter": (
            "trackn_union"
            if args.trackn_encode_union
            else ("allowlist" if allow is not None else None)
        ),
        "allowlist_n": (len(allow) if allow is not None else None),
    }
    print(json.dumps(meta, indent=2))
    expand(dcfg["latent_dir"]).joinpath("encode_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
