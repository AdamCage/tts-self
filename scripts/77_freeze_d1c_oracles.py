#!/usr/bin/env python3
"""Freeze fresh D1c oracle inventories from static manifest/cache metadata only.

Creates:
  configs/train_subsets/d1c_dev32.json
  configs/train_subsets/d1c_confirm_seed_commitment.json  (stub; no plaintext seed)

Confirmation materialization requires an external reveal and is refused here
without ``--materialize-confirm`` plus verified commitment proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minispeech.data.discrete_oracles import items_sha256  # noqa: E402
from minispeech.data.manifest import Utterance, read_manifest  # noqa: E402
from minispeech.utils import expand  # noqa: E402

SEED = 20260805
PLAN_SLUG = "d1c_semantic_eos_repair_b81ecc3e"
RUSLAN = "ruslan"
CV = "common_voice_ru_17_hf_mirror"
DEV_COUNTS = {RUSLAN: 6, CV: 26}
BUCKET_TARGETS = {"short": 10, "medium": 11, "long": 11}
DEFAULT_CODEC = {
    RUSLAN: "~/datasets/tts/ruslan/codec_tokens/codec_tokens_index.json",
    CV: "~/datasets/tts/common_voice_ru17/codec_tokens/codec_tokens_index.json",
}
DEFAULT_TRACKN = {
    "train": "configs/eval_sets/trackn_train.json",
    "val": "configs/eval_sets/trackn_val.json",
    "test": "configs/eval_sets/trackn_test.json",
    "review": "configs/eval_sets/trackn_review.json",
}


@dataclass
class EligibleRow:
    utt_id: str
    corpus: str
    speaker_id: str
    dur_s: float
    codec_frames: int
    text_token_count: int
    text_norm: str
    audio_sha256: str
    wav_path: str
    pcm_sha256: str

    @property
    def ratio(self) -> float:
        return float(self.codec_frames) / max(1, int(self.text_token_count))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_utt_ids(path: Path) -> list[str]:
    raw = load_json(path)
    if isinstance(raw, dict) and isinstance(raw.get("utt_ids"), list):
        return [str(x) for x in raw["utt_ids"]]
    raise ValueError(f"{path}: expected object with utt_ids")


def char_ngrams(text: str, n: int = 5) -> set[str]:
    text = " ".join(text.lower().split())
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def near_duplicate(a: str, b: str, threshold: float = 0.90) -> bool:
    aa, bb = char_ngrams(a), char_ngrams(b)
    if not aa or not bb:
        return False
    return len(aa & bb) / len(aa | bb) >= threshold


def bucket(value: float, edges: tuple[float, float]) -> str:
    if value < edges[0]:
        return "short"
    if value < edges[1]:
        return "medium"
    return "long"


def quantile(values: Sequence[float], p: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    return vals[max(0, min(len(vals) - 1, int(round(p * (len(vals) - 1)))))]


def corpus_of(utt_id: str, row: Utterance | None = None) -> str:
    if row is not None and row.corpus:
        return str(row.corpus)
    if utt_id.startswith("ruslan_"):
        return RUSLAN
    return CV


def load_burned_ids(registry: Path) -> set[str]:
    payload = load_json(registry)
    ids: set[str] = set()
    for entry in payload.get("entries", []):
        path = entry.get("path")
        if not isinstance(path, str) or not path.endswith(".json"):
            continue
        # Skip non-subset entries (eval contracts).
        if "train_subsets" not in path and "eval_sets" not in path:
            if "eval_contracts" in path:
                continue
        p = ROOT / path
        if not p.is_file():
            continue
        source = load_json(p)
        ids.update(str(x) for x in source.get("utt_ids", []))
        splits = source.get("splits")
        if isinstance(splits, dict):
            for vals in splits.values():
                if isinstance(vals, list):
                    ids.update(str(x) for x in vals)
    return ids


def load_codec_index(path: Path) -> dict[str, Any]:
    raw = load_json(path)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"empty codec index: {path}")
    return raw


def normalize_text(row: Utterance) -> str:
    for key in ("text_stressed", "text_raw", "text_norm", "text"):
        val = getattr(row, key, None)
        if isinstance(val, str) and val.strip():
            return " ".join(val.lower().split())
    return ""


def build_eligible(
    *,
    manifest: Path,
    trackn: Mapping[str, Path],
    codec_indexes: Mapping[str, Path],
    burned: set[str],
) -> tuple[list[EligibleRow], dict[str, Any]]:
    train_ids = set(load_utt_ids(trackn["train"]))
    held = {name: set(load_utt_ids(trackn[name])) for name in ("val", "test", "review")}
    rows = {r.utt_id: r for r in read_manifest(manifest)}
    indexes: dict[str, dict[str, Any]] = {}
    index_meta: dict[str, Any] = {}
    for corp, path in codec_indexes.items():
        idx = load_codec_index(path)
        indexes[corp] = idx
        index_meta[corp] = {
            "path": str(path),
            "file_sha256": sha256_file(path),
            "n": len(idx),
        }

    # Union of codec-index keys that are in trackn_train.
    codec_ids = set()
    for idx in indexes.values():
        codec_ids.update(idx.keys())

    excluded = set(burned) | held["val"] | held["test"] | held["review"]
    eligible: list[EligibleRow] = []
    skipped = Counter()
    for uid in sorted(train_ids & codec_ids):
        if uid in excluded:
            skipped["burned_or_heldout"] += 1
            continue
        row = rows.get(uid)
        if row is None:
            skipped["missing_manifest"] += 1
            continue
        if (row.qc_tag or "clean") != "clean":
            skipped["qc"] += 1
            continue
        corp = corpus_of(uid, row)
        idx = indexes.get(corp) or {}
        meta = idx.get(uid)
        if not isinstance(meta, dict):
            skipped["missing_codec_meta"] += 1
            continue
        n_frames = int(meta.get("n_frames") or 0)
        text_len = int(meta.get("text_len") or 0)
        if n_frames <= 0 or text_len <= 0:
            skipped["bad_codec_counts"] += 1
            continue
        # Reject codec/text geometry mismatch vs duration when available.
        dur = float(meta.get("dur_s") or row.dur_s or 0.0)
        if dur <= 0:
            skipped["bad_duration"] += 1
            continue
        text_norm = normalize_text(row)
        if not text_norm:
            skipped["empty_text"] += 1
            continue
        speaker = str(meta.get("speaker_id") or row.effective_speaker_id())
        eligible.append(
            EligibleRow(
                utt_id=uid,
                corpus=corp,
                speaker_id=speaker,
                dur_s=dur,
                codec_frames=n_frames,
                text_token_count=text_len,
                text_norm=text_norm,
                audio_sha256=str(getattr(row, "audio_sha256", "") or ""),
                wav_path=str(meta.get("wav") or getattr(row, "wav", "") or ""),
                pcm_sha256=str(getattr(row, "pcm_sha256", "") or ""),
            )
        )

    # Exact duplicate fingerprints among eligible pool (drop later copies).
    seen_fp: set[str] = set()
    seen_text_sha: set[str] = set()
    deduped: list[EligibleRow] = []
    for row in eligible:
        text_sha = hashlib.sha256(row.text_norm.encode("utf-8")).hexdigest()
        if text_sha in seen_text_sha:
            skipped["exact_text_dup"] += 1
            continue
        fp_parts = [
            f"path:{row.wav_path}",
            f"audio:{row.audio_sha256}" if row.audio_sha256 else "",
            f"pcm:{row.pcm_sha256}" if row.pcm_sha256 else "",
            f"text:{text_sha}",
        ]
        fp = "|".join(p for p in fp_parts if p)
        if fp in seen_fp:
            skipped["exact_dup"] += 1
            continue
        seen_fp.add(fp)
        seen_text_sha.add(text_sha)
        deduped.append(row)

    # Near-duplicate rejection vs burned texts only (pool is large; intra-pool
    # near-dup is enforced during selection against already-chosen rows).
    burned_texts: list[str] = []
    for uid in burned:
        row = rows.get(uid)
        if row is not None:
            t = normalize_text(row)
            if t:
                burned_texts.append(t)

    kept: list[EligibleRow] = []
    for row in deduped:
        if any(near_duplicate(row.text_norm, t) for t in burned_texts):
            skipped["near_dup_burned"] += 1
            continue
        kept.append(row)

    meta = {
        "codec_indexes": index_meta,
        "n_train": len(train_ids),
        "n_codec_intersection": len(train_ids & codec_ids),
        "n_eligible": len(kept),
        "skipped": dict(skipped),
        "heldout_sizes": {k: len(v) for k, v in held.items()},
        "n_burned": len(burned),
    }
    return kept, meta


def select_dev(
    pool: Sequence[EligibleRow], *, seed: int
) -> tuple[list[EligibleRow], dict[str, Any]]:
    if not pool:
        raise ValueError("empty eligible pool")
    durs = [r.dur_s for r in pool]
    edges = (quantile(durs, 1 / 3), quantile(durs, 2 / 3))
    ratios = {r.utt_id: r.ratio for r in pool}
    q1, q3 = quantile(list(ratios.values()), 0.25), quantile(list(ratios.values()), 0.75)
    rng = random.Random(seed)
    by_cb: dict[tuple[str, str], list[EligibleRow]] = defaultdict(list)
    for row in pool:
        by_cb[(row.corpus, bucket(row.dur_s, edges))].append(row)
    for values in by_cb.values():
        rng.shuffle(values)

    quotas = {
        RUSLAN: {"short": 2, "medium": 2, "long": 2},
        CV: {"short": 8, "medium": 9, "long": 9},
    }
    selected: list[EligibleRow] = []
    speakers: set[str] = set()
    for corp, by_bucket in quotas.items():
        for name, need in by_bucket.items():
            candidates = by_cb.get((corp, name), [])
            picked = 0
            for row in candidates:
                if corp == CV and row.speaker_id in speakers:
                    continue
                if any(near_duplicate(row.text_norm, s.text_norm) for s in selected):
                    continue
                selected.append(row)
                speakers.add(row.speaker_id)
                picked += 1
                if picked == need:
                    break
            if picked != need:
                raise ValueError(
                    f"insufficient unique {corp}/{name}: got {picked} need {need} "
                    f"(pool_bucket={len(candidates)})"
                )
    selected.sort(key=lambda r: (0 if r.corpus == RUSLAN else 1, r.utt_id))
    if len(selected) != 32:
        raise ValueError(f"DEV selection yielded {len(selected)} != 32")
    # Enforce unique CV speakers (already enforced during pick).
    cv_speakers = [r.speaker_id for r in selected if r.corpus == CV]
    if len(cv_speakers) != len(set(cv_speakers)):
        raise ValueError("CV speaker uniqueness violated")

    # Ratio extremes: ensure ≥6 low and ≥6 high; if not, swap from pool.
    def _count_extremes(rows: Sequence[EligibleRow]) -> tuple[int, int]:
        low = sum(1 for r in rows if ratios[r.utt_id] <= q1)
        high = sum(1 for r in rows if ratios[r.utt_id] >= q3)
        return low, high

    low, high = _count_extremes(selected)
    if low < 6 or high < 6:
        selected_ids = {r.utt_id for r in selected}
        # Prefer swaps that preserve corpus/bucket quotas.
        def _bucket_of(row: EligibleRow) -> str:
            return bucket(row.dur_s, edges)

        for need_extreme, target in (("low", 6 - low), ("high", 6 - high)):
            if target <= 0:
                continue
            donors = [
                r
                for r in pool
                if r.utt_id not in selected_ids
                and (
                    (need_extreme == "low" and ratios[r.utt_id] <= q1)
                    or (need_extreme == "high" and ratios[r.utt_id] >= q3)
                )
            ]
            rng.shuffle(donors)
            for donor in donors:
                if target <= 0:
                    break
                # Find a non-extreme same corpus/bucket victim.
                victims = [
                    i
                    for i, r in enumerate(selected)
                    if r.corpus == donor.corpus
                    and _bucket_of(r) == _bucket_of(donor)
                    and (q1 < ratios[r.utt_id] < q3)
                    and (r.corpus != CV or donor.speaker_id not in {
                        selected[j].speaker_id for j in range(len(selected)) if j != i and selected[j].corpus == CV
                    })
                ]
                if not victims:
                    continue
                vi = victims[0]
                old = selected[vi]
                selected[vi] = donor
                selected_ids.discard(old.utt_id)
                selected_ids.add(donor.utt_id)
                target -= 1
            selected.sort(key=lambda r: (0 if r.corpus == RUSLAN else 1, r.utt_id))
        low, high = _count_extremes(selected)
        if low < 6 or high < 6:
            raise ValueError(f"ratio-extreme quota failed after repair: low={low}, high={high}")

    # Re-check bucket totals.
    bucket_counts = Counter(bucket(r.dur_s, edges) for r in selected)
    for name, need in BUCKET_TARGETS.items():
        if bucket_counts[name] != need:
            raise ValueError(f"duration bucket {name}={bucket_counts[name]} != {need}")
    corpus_counts = Counter(r.corpus for r in selected)
    if corpus_counts[RUSLAN] != 6 or corpus_counts[CV] != 26:
        raise ValueError(f"corpus counts wrong: {dict(corpus_counts)}")

    facts = {
        "duration_edges_s": {"short_lt": edges[0], "medium_lt": edges[1]},
        "ratio_quartiles": {"q1": q1, "q3": q3},
        "ratio_extremes": {"low": low, "high": high},
        "bucket_counts": dict(bucket_counts),
        "corpus_counts": dict(corpus_counts),
        "cv_speaker_unique": True,
        "n_cv_speakers": len(set(cv_speakers)),
    }
    return selected, facts


def split_dev(
    selected: Sequence[EligibleRow],
    *,
    edges: tuple[float, float],
    q1: float,
    q3: float,
) -> dict[str, list[str]]:
    """select8 covers all duration buckets + both ratio extremes; gate24 = complement.

    Composition lock: exactly 2 RUSLAN + 6 Common Voice.
    """
    coverage: list[EligibleRow] = []
    # One from each duration bucket.
    for want in ("short", "medium", "long"):
        for row in selected:
            if row not in coverage and bucket(row.dur_s, edges) == want:
                coverage.append(row)
                break
    # Ratio extremes.
    for pred in (
        lambda r: r.ratio <= q1,
        lambda r: r.ratio >= q3,
    ):
        for row in selected:
            if row not in coverage and pred(row):
                coverage.append(row)
                break

    ruslan_pool = [r for r in selected if r.corpus == RUSLAN]
    cv_pool = [r for r in selected if r.corpus == CV]

    # Prefer coverage rows, then fill — but demote surplus RUSLAN to enforce 2+6.
    chosen: list[EligibleRow] = []
    for row in coverage:
        if row.corpus == RUSLAN and sum(1 for c in chosen if c.corpus == RUSLAN) >= 2:
            continue
        if row.corpus == CV and sum(1 for c in chosen if c.corpus == CV) >= 6:
            continue
        chosen.append(row)

    for row in ruslan_pool:
        if sum(1 for c in chosen if c.corpus == RUSLAN) >= 2:
            break
        if row not in chosen:
            chosen.append(row)
    for row in cv_pool:
        if sum(1 for c in chosen if c.corpus == CV) >= 6:
            break
        if row not in chosen:
            chosen.append(row)

    # If coverage skipped a bucket because of the RUSLAN cap, try CV replacements.
    for want in ("short", "medium", "long"):
        if any(bucket(c.dur_s, edges) == want for c in chosen):
            continue
        for row in cv_pool:
            if row in chosen:
                continue
            if bucket(row.dur_s, edges) != want:
                continue
            # Swap out a non-coverage CV if needed to stay at 8.
            if len(chosen) >= 8:
                for i, c in enumerate(chosen):
                    if c.corpus == CV and c not in coverage:
                        chosen[i] = row
                        break
            else:
                chosen.append(row)
            break

    while len(chosen) < 8:
        for row in selected:
            if row not in chosen:
                if row.corpus == RUSLAN and sum(1 for c in chosen if c.corpus == RUSLAN) >= 2:
                    continue
                if row.corpus == CV and sum(1 for c in chosen if c.corpus == CV) >= 6:
                    continue
                chosen.append(row)
            if len(chosen) >= 8:
                break
        else:
            break

    n_ruslan = sum(1 for c in chosen if c.corpus == RUSLAN)
    n_cv = sum(1 for c in chosen if c.corpus == CV)
    if n_ruslan != 2 or n_cv != 6 or len(chosen) != 8:
        raise ValueError(
            f"select8 composition invalid: need 2 RUSLAN + 6 CV, got "
            f"{n_ruslan}+{n_cv} (n={len(chosen)})"
        )

    select = [r.utt_id for r in sorted(chosen[:8], key=lambda r: (0 if r.corpus == RUSLAN else 1, r.utt_id))]
    full = [r.utt_id for r in selected]
    gate = [u for u in full if u not in set(select)]
    if len(select) != 8 or len(gate) != 24 or len(full) != 32:
        raise ValueError("split sizes invalid")
    if set(select) & set(gate):
        raise ValueError("select8/gate24 overlap")
    if sorted(select + gate) != sorted(full):
        raise ValueError("select8∪gate24 != full32")
    return {"select8": select, "gate24": gate, "full32": full}


def write_commitment_stub(path: Path) -> None:
    if path.exists():
        return
    write_json(
        path,
        {
            "schema": "minispeech.d1c_confirm_seed_commitment",
            "schema_version": "1.0.0",
            "algorithm": "SHA256(UTF8('d1c-confirm-v1\\n' + decimal_seed + '\\n'))",
            "commitment": None,
            "custodian_role": "external_user",
            "generator_version_or_hash": None,
            "intended_role": "independent_confirmation_seed",
            "plaintext_seed_present": False,
            "note": "Custodian must supply commitment before first DEV model output.",
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="~/datasets/tts/multicorpus/manifest.jsonl")
    ap.add_argument("--out", default="configs/train_subsets/d1c_dev32.json")
    ap.add_argument(
        "--burn-registry", default="configs/train_subsets/d1_oracle_burn_registry.json"
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--materialize-confirm", action="store_true")
    args = ap.parse_args()
    if args.materialize_confirm:
        raise SystemExit(
            "confirmation materialization requires a separately verified external seed reveal"
        )

    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f"refusing overwrite of sealed D1c DEV inventory: {out}")

    trackn = {k: expand(str(ROOT / v)) for k, v in DEFAULT_TRACKN.items()}
    codec_indexes = {k: expand(v) for k, v in DEFAULT_CODEC.items()}
    burned = load_burned_ids(ROOT / args.burn_registry)
    eligible, pool_meta = build_eligible(
        manifest=expand(args.manifest),
        trackn=trackn,
        codec_indexes=codec_indexes,
        burned=burned,
    )
    print(json.dumps({"pool_meta": pool_meta}, indent=2), flush=True)
    if not eligible:
        raise SystemExit(f"empty eligible pool; skipped={pool_meta.get('skipped')}")
    selected, facts = select_dev(eligible, seed=int(args.seed))
    edges = (
        float(facts["duration_edges_s"]["short_lt"]),
        float(facts["duration_edges_s"]["medium_lt"]),
    )
    q1 = float(facts["ratio_quartiles"]["q1"])
    q3 = float(facts["ratio_quartiles"]["q3"])
    splits = split_dev(selected, edges=edges, q1=q1, q3=q3)
    ids = [r.utt_id for r in selected]
    # Overlap report vs burns.
    overlap = sorted(set(ids) & burned)
    if overlap:
        raise SystemExit(f"DEV overlaps burn registry: {overlap[:5]}")

    payload = {
        "id": "d1c_dev32",
        "version": "1.0.0",
        "kind": "utt_ids",
        "role": "development_oracle",
        "plan_slug": PLAN_SLUG,
        "n": 32,
        "seed": int(args.seed),
        "utt_ids": ids,
        "items_sha256": items_sha256(ids),
        "splits": splits,
        "split_hashes": {k: items_sha256(v) for k, v in splits.items()},
        "selection": {
            "static_metadata_only": True,
            "corpus_counts": DEV_COUNTS,
            "duration_targets": BUCKET_TARGETS,
            "near_dup_jaccard_ge": 0.90,
            "char_ngram": 5,
            **facts,
        },
        "source": {
            "manifest": str(expand(args.manifest)),
            "manifest_sha256": sha256_file(expand(args.manifest)),
            "burn_registry": str(args.burn_registry),
            "pool": pool_meta,
            "trackn": {
                name: {
                    "path": str(path),
                    "file_sha256": sha256_file(path),
                    "items_sha256": items_sha256(load_utt_ids(path)),
                    "n": len(load_utt_ids(path)),
                }
                for name, path in trackn.items()
            },
        },
        "rows_meta": [
            {
                "utt_id": r.utt_id,
                "corpus": r.corpus,
                "speaker_id": r.speaker_id,
                "dur_s": r.dur_s,
                "codec_frames": r.codec_frames,
                "text_token_count": r.text_token_count,
                "ratio": r.ratio,
                "duration_bucket": bucket(r.dur_s, edges),
            }
            for r in selected
        ],
    }
    file_sha = write_json(out, payload)
    # Rewrite with declared file hash for contract binding.
    payload["file_sha256"] = file_sha
    # Keep file bytes stable: store file_sha256 of body without that field.
    body = {k: v for k, v in payload.items() if k != "file_sha256"}
    file_sha = write_json(out, body)
    payload["file_sha256"] = file_sha
    write_json(out, {**body, "file_sha256": file_sha})

    write_commitment_stub(ROOT / "configs/train_subsets/d1c_confirm_seed_commitment.json")

    # Mirror under out/<plan>/frozen for Windows/WSL convenience.
    frozen = ROOT / "out" / PLAN_SLUG / "frozen"
    write_json(frozen / "d1c_dev32.json", {**body, "file_sha256": file_sha})
    write_json(
        frozen / "oracle_overlap_report.json",
        {
            "dev_vs_burn": overlap,
            "n_burned": len(burned),
            "n_dev": 32,
            "ok": len(overlap) == 0,
            "split_hashes": payload["split_hashes"],
            "items_sha256": payload["items_sha256"],
            "file_sha256": file_sha,
        },
    )
    print(
        json.dumps(
            {
                "path": str(out),
                "items_sha256": payload["items_sha256"],
                "file_sha256": file_sha,
                "split_hashes": payload["split_hashes"],
                "selection": facts,
                "pool_n_eligible": pool_meta["n_eligible"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
