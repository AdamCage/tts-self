#!/usr/bin/env python3
"""Build RUSLAN manifest with contiguous split + challenge set flags."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from minispeech.constants import SR, n_frames_from_samples
from minispeech.data.manifest import Utterance, contiguous_split, write_manifest
from minispeech.frontend.normalize import normalize_text
from minispeech.frontend.stress import to_canonical_stress
from minispeech.utils import expand, load_yaml


def find_pairs(root: Path) -> list[tuple[Path, str, str]]:
    """Return list of (wav, text_raw, text_stressed). Heuristic for HF layout."""
    pairs: list[tuple[Path, str, str]] = []
    meta_csv = list(root.rglob("metadata.csv")) + list(root.rglob("*.tsv"))
    wavs = {p.stem: p for p in root.rglob("*.wav")}

    if meta_csv:
        for mp in meta_csv:
            for line in mp.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip() or line.startswith("id"):
                    continue
                parts = re.split(r"[|\t,]", line, maxsplit=2)
                if len(parts) < 2:
                    continue
                uid, text = parts[0].strip(), parts[-1].strip()
                wp = wavs.get(Path(uid).stem) or wavs.get(uid)
                if wp is None:
                    cand = mp.parent / uid
                    if cand.exists():
                        wp = cand
                if wp is None:
                    continue
                pairs.append((wp, text, text))
        if pairs:
            return pairs

    for wp in sorted(wavs.values()):
        for ext in (".txt", ".lab"):
            tp = wp.with_suffix(ext)
            if tp.exists():
                text = tp.read_text(encoding="utf-8").strip()
                pairs.append((wp, text, text))
                break
    return pairs


def pick_challenge(rows: list[Utterance], n: int, seed: int = 42) -> None:
    rng = random.Random(seed)
    patterns = [
        re.compile(r"\d"),
        re.compile(r"[А-Я]{2,}"),
        re.compile(r"ё", re.I),
        re.compile(r"[?!]"),
        re.compile(r".{40,}"),
    ]
    candidates = []
    for r in rows:
        score = sum(1 for p in patterns if p.search(r.text_raw))
        if score:
            candidates.append((score, r))
    candidates.sort(key=lambda x: -x[0])
    chosen = [r for _, r in candidates[: n // 2]]
    chosen_ids = {r.utt_id for r in chosen}
    rest = [r for r in rows if r.utt_id not in chosen_ids]
    rng.shuffle(rest)
    chosen += rest[: max(0, n - len(chosen))]
    for r in chosen:
        r.challenge = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_ruslan.yaml")
    ap.add_argument("--ruslan-dir", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)["data"]
    root = expand(args.ruslan_dir or cfg["ruslan_dir"])
    pairs = find_pairs(root)

    counters = {
        "pairs_found": len(pairs),
        "dropped_dur": 0,
        "dropped_unreadable": 0,
        "dropped_dup_text": 0,
        "dropped_utt_id_collision": 0,
    }

    if not pairs:
        try:
            from datasets import load_from_disk, load_dataset

            try:
                ds = load_from_disk(str(root))
            except Exception:
                ds = load_dataset(str(root), split="train")
            rows: list[Utterance] = []
            seen_ids: set[str] = set()
            for i, ex in enumerate(tqdm(ds, desc="manifest-hf")):
                audio = ex.get("audio") or ex.get("speech")
                text = ex.get("text_stressed") or ex.get("text") or ex.get("sentence") or ""
                text_raw = ex.get("text") or text
                if audio is None or not isinstance(audio, dict):
                    continue
                import numpy as np

                arr = np.asarray(audio["array"], dtype="float32")
                sr = int(audio["sampling_rate"])
                wav_path = root / "wav_cache" / f"{i:06d}.wav"
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                if not wav_path.exists():
                    sf.write(str(wav_path), arr, sr)
                n_24 = int(round(len(arr) * SR / sr))
                dur = n_24 / SR
                if dur < cfg.get("min_dur_s", 0.5) or dur > cfg.get("max_dur_s", 10.0):
                    counters["dropped_dur"] += 1
                    continue
                uid = f"ruslan_{wav_path.stem}"
                if uid in seen_ids:
                    counters["dropped_utt_id_collision"] += 1
                    raise ValueError(f"utt_id collision: {uid}")
                seen_ids.add(uid)
                rows.append(
                    Utterance(
                        utt_id=uid,
                        wav=str(wav_path),
                        text_raw=normalize_text(text_raw),
                        text_stressed=to_canonical_stress(text),
                        speaker=cfg.get("speaker", "ruslan"),
                        n_samples_24k=n_24,
                        n_frames=n_frames_from_samples(n_24),
                        dur_s=dur,
                    )
                )
        except Exception as e:
            print(f"Failed to build manifest: {e}")
            return 1
    else:
        rows = []
        seen_ids: set[str] = set()
        for wp, text_raw, text_st in tqdm(pairs, desc="manifest"):
            try:
                info = sf.info(str(wp))
            except Exception:
                counters["dropped_unreadable"] += 1
                continue
            n_24 = int(round(info.frames * SR / info.samplerate))
            dur = n_24 / SR
            if dur < cfg.get("min_dur_s", 0.5) or dur > cfg.get("max_dur_s", 10.0):
                counters["dropped_dur"] += 1
                continue
            try:
                audio, _ = sf.read(str(wp), dtype="float32")
                if audio.size == 0:
                    counters["dropped_unreadable"] += 1
                    continue
            except Exception:
                counters["dropped_unreadable"] += 1
                continue
            uid = f"ruslan_{wp.stem}"
            if uid in seen_ids:
                counters["dropped_utt_id_collision"] += 1
                raise ValueError(f"utt_id collision: {uid}")
            seen_ids.add(uid)
            rows.append(
                Utterance(
                    utt_id=uid,
                    wav=str(wp.resolve()),
                    text_raw=normalize_text(text_raw),
                    text_stressed=to_canonical_stress(text_st),
                    speaker=cfg.get("speaker", "ruslan"),
                    n_samples_24k=n_24,
                    n_frames=n_frames_from_samples(n_24),
                    dur_s=dur,
                )
            )

    seen = set()
    deduped = []
    for r in rows:
        key = r.text_raw.lower()
        if key in seen:
            counters["dropped_dup_text"] += 1
            continue
        seen.add(key)
        deduped.append(r)
    split_cfg = cfg.get("split", {})
    rows = contiguous_split(
        deduped,
        train=float(split_cfg.get("train", 0.9)),
        val=float(split_cfg.get("val", 0.05)),
        test=float(split_cfg.get("test", 0.05)),
    )
    pick_challenge(rows, cfg.get("challenge_set_size", 180))

    out = expand(cfg["manifest"])
    write_manifest(out, rows)
    test_ids = sorted(r.utt_id for r in rows if r.split == "test")
    h = hashlib.sha256("\n".join(test_ids).encode()).hexdigest()
    hours_by_split = {
        s: sum(r.dur_s for r in rows if r.split == s) / 3600.0 for s in ("train", "val", "test")
    }
    meta = {
        "n": len(rows),
        "train": sum(r.split == "train" for r in rows),
        "val": sum(r.split == "val" for r in rows),
        "test": sum(r.split == "test" for r in rows),
        "challenge": sum(r.challenge for r in rows),
        "hours": sum(r.dur_s for r in rows) / 3600.0,
        "hours_by_split": hours_by_split,
        "test_set_sha256": h,
        "filter_counters": counters,
        "utt_id_scheme": "ruslan_{stem}",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
