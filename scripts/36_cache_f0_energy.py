#!/usr/bin/env python3
"""Cache phone-level F0 / energy for variance adaptor (T13).

Aggregates frame F0 (librosa.pyin) and mel-frame energy onto forced spans.
Writes ``f0_energy_index.json`` + shards under ``--out``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from minispeech.constants import MEL80
from minispeech.data.forced_align import load_forced_json
from minispeech.data.frame_projection import phones_and_frame_durs_from_forced
from minispeech.data.manifest import filter_split, read_manifest
from minispeech.data.mel import MelCacheReader, default_mel_dir, load_wav_mono
from minispeech.data.shards import ShardWriter
from minispeech.utils import expand, load_yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gap-mode", default="expand_neighbor")
    args = ap.parse_args()

    import librosa

    dcfg = load_yaml(args.config)["data"]
    out_dir = expand(args.out or Path(str(dcfg.get("ruslan_dir"))).expanduser() / "f0_energy")
    mel_dir = expand(default_mel_dir(dcfg))
    forced = load_forced_json(expand(dcfg["forced_path"]))
    mels = MelCacheReader(mel_dir, lazy=True, expect_target="mel80")
    rows = filter_split(read_manifest(expand(dcfg["manifest"])), args.split)
    if args.max_utts:
        rows = rows[: args.max_utts]

    w = ShardWriter(out_dir, "f0e", shard_size=2000, resume=True)
    fr = float(MEL80.frame_rate)
    n_ok = 0
    for u in tqdm(rows, desc="f0-energy"):
        if u.utt_id not in forced or u.utt_id not in mels.index:
            continue
        if w.has(u.utt_id):
            continue
        entry = forced[u.utt_id]
        mel = mels.get(u.utt_id).float().numpy()
        phones, durs, _ = phones_and_frame_durs_from_forced(
            entry, frame_rate=fr, n_frames=mel.shape[0], gap_mode=args.gap_mode  # type: ignore[arg-type]
        )
        wav, sr = load_wav_mono(u.wav, target_sr=int(MEL80.sample_rate))
        f0, _, _ = librosa.pyin(
            wav,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=1024,
            hop_length=256,
        )
        f0 = np.asarray(f0, dtype=np.float32)
        energy = np.linalg.norm(mel, axis=-1).astype(np.float32)
        # Aggregate to phone level
        pitch_p = np.zeros(len(phones), dtype=np.float32)
        energy_p = np.zeros(len(phones), dtype=np.float32)
        t = 0
        for i, d in enumerate(durs):
            d = int(d)
            seg_e = energy[t : t + d]
            seg_f = f0[t : t + d] if t < len(f0) else np.array([])
            energy_p[i] = float(seg_e.mean()) if seg_e.size else 0.0
            voiced = seg_f[np.isfinite(seg_f)] if seg_f.size else np.array([])
            pitch_p[i] = float(voiced.mean()) if voiced.size else 0.0
            t += d
        tensor = torch.from_numpy(np.stack([pitch_p, energy_p], axis=-1))  # [P, 2]
        w.add(u.utt_id, tensor, meta={"n_phones": len(phones), "phones": phones[:32]})
        n_ok += 1
    idx = w.close()
    print(json.dumps({"out": str(out_dir), "index": str(idx), "n_ok": n_ok}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
