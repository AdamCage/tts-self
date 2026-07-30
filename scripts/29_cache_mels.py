#!/usr/bin/env python3
"""Extract / cache mel80 log-mels from corpus wavs (Track A).

No Hub downloads. Requires local wavs listed in the manifest.
"""

from __future__ import annotations

import argparse
import json

from tqdm import tqdm

from minispeech.data.manifest import filter_split, read_manifest
from minispeech.data.mel import MelCacheWriter, default_mel_dir, extract_mel_file, mel_backend_name
from minispeech.utils import expand, load_yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--split", default="train", help="train|val|test|all")
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--out", default=None, help="Override mel_dir")
    ap.add_argument(
        "--target",
        default="mel80",
        help="Registered mel target (mel80 | mel80_hifigan)",
    )
    ap.add_argument("--shard-size", type=int, default=2000)
    args = ap.parse_args()

    data_cfg = load_yaml(args.config)["data"]
    out_dir = expand(args.out or default_mel_dir(data_cfg))
    backend = mel_backend_name()
    if backend == "none":
        raise SystemExit("need librosa or torchaudio for mel extraction")

    rows = read_manifest(data_cfg["manifest"])
    if args.split != "all":
        rows = filter_split(rows, args.split)
    excl = set(data_cfg.get("exclude_tags", ("exclude",)))
    if excl:
        rows = [r for r in rows if (r.qc_tag or "clean") not in excl]
    if args.max_utts is not None:
        rows = rows[: args.max_utts]

    writer = MelCacheWriter(
        out_dir, shard_size=args.shard_size, resume=True, target=args.target
    )
    n_ok, n_skip, n_err = 0, 0, 0
    errors: list[dict] = []
    for u in tqdm(rows, desc="cache-mels"):
        if writer.has(u.utt_id):
            n_skip += 1
            continue
        try:
            mel = extract_mel_file(u.wav, target=args.target)
            writer.add(
                u.utt_id,
                mel,
                meta={"n_frames": int(mel.shape[0]), "n_mels": int(mel.shape[1])},
            )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            errors.append({"utt_id": u.utt_id, "error": str(exc)})
    idx = writer.close()
    report = {
        "out_dir": str(out_dir),
        "index": str(idx),
        "target": args.target,
        "backend": backend,
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_err": n_err,
        "errors_head": errors[:20],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if n_err == 0 or n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
