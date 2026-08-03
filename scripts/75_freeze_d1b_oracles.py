#!/usr/bin/env python3
"""Freeze D1b DEV32 / CONFIRM32 oracle utterance sets (CPU-only, read-only sources).

Creates:
  configs/train_subsets/d1b_dev32.json
  configs/train_subsets/d1b_confirm32.json
  out/d1b_n2_reliability/frozen/{dev32,confirm32,confirm_select8,confirm_gate24,
                                 leak_report,hashes,selection_report}.json

Never overwrites an existing versioned set without an explicit ``--version`` bump
that differs from the on-disk ``version`` field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minispeech.data.manifest import Utterance, read_manifest  # noqa: E402
from minispeech.utils import expand  # noqa: E402

PLAN_SLUG = "d1b_n2_reliability"
DEFAULT_SEED = 20260803
DEFAULT_VERSION = "1.0.0"
KNOWN_DEV_HALLUCINATION = "common_voice_ru_17_hf_mirror__ba1a4089af87b3ebc90db07d"

DEFAULT_CODEC_S1 = {
    "ruslan": "~/datasets/tts/ruslan/codec_tokens_s1_32/codec_tokens_index.json",
    "common_voice_ru_17_hf_mirror": (
        "~/datasets/tts/common_voice_ru17/codec_tokens_s1_32/codec_tokens_index.json"
    ),
}
DEFAULT_TRACKN = {
    "train": "configs/eval_sets/trackn_train.json",
    "val": "configs/eval_sets/trackn_val.json",
    "test": "configs/eval_sets/trackn_test.json",
    "review": "configs/eval_sets/trackn_review.json",
}
DEFAULT_MANIFEST = "~/datasets/tts/multicorpus/manifest.jsonl"
RUSLAN_CORPUS = "ruslan"
CV_CORPUS = "common_voice_ru_17_hf_mirror"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _items_sha256(items: Sequence[str]) -> str:
    return _sha256_bytes(_canonical_json(list(items)))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return _sha256_bytes(text.encode("utf-8"))


def _resolve_repo(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    expanded = expand(str(path))
    if expanded.is_absolute() and expanded.exists():
        return expanded
    cand = ROOT / path
    return cand if cand.exists() else expanded


def _load_utt_ids(path: Path) -> list[str]:
    raw = _load_json(path)
    if isinstance(raw, dict) and isinstance(raw.get("utt_ids"), list):
        return [str(x) for x in raw["utt_ids"]]
    raise ValueError(f"{path}: expected object with utt_ids array")


def _load_codec_index(path: Path) -> dict[str, Any]:
    raw = _load_json(path)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: empty or non-object codec index")
    return raw


def _corpus_of(utt_id: str, row: Utterance | None = None) -> str:
    if row is not None and row.corpus:
        return str(row.corpus)
    if utt_id.startswith("ruslan_"):
        return RUSLAN_CORPUS
    if utt_id.startswith("common_voice_ru_17_hf_mirror__"):
        return CV_CORPUS
    return "unknown"


def _audio_fingerprint(row: Utterance) -> str:
    if row.audio_sha256:
        return f"audio_sha256:{row.audio_sha256}"
    payload = {
        "wav": str(row.wav or ""),
        "dur_s": float(row.dur_s or 0.0),
        "n_samples_24k": int(row.n_samples_24k or 0),
        "speaker_id": row.effective_speaker_id(),
        "utt_id": row.utt_id,
    }
    return "meta:" + _sha256_bytes(_canonical_json(payload))[:32]


def _duration_bucket(dur_s: float, edges: tuple[float, float]) -> str:
    lo, hi = edges
    if dur_s < lo:
        return "short"
    if dur_s < hi:
        return "medium"
    return "long"


def _tertile_edges(durs: Sequence[float]) -> tuple[float, float]:
    if not durs:
        return (1.0, 3.0)
    xs = sorted(float(x) for x in durs)
    n = len(xs)

    def _q(p: float) -> float:
        if n == 1:
            return xs[0]
        idx = min(n - 1, max(0, int(math.floor(p * (n - 1)))))
        return xs[idx]

    return (_q(1.0 / 3.0), _q(2.0 / 3.0))


def _refuse_overwrite(path: Path, new_version: str) -> None:
    if not path.is_file():
        return
    old = _load_json(path)
    old_v = str(old.get("version") or "")
    if old_v == new_version:
        raise SystemExit(
            f"Refusing to overwrite {path} at version {old_v}. "
            "Bump --version (MAJOR.MINOR.PATCH) to rewrite."
        )


def export_dev32(
    *,
    codec_indexes: dict[str, Path],
    version: str,
    out_path: Path,
    encode_meta_path: Path | None,
) -> dict[str, Any]:
    """Export ordered DEV32 from existing codec_tokens_s1_32 shards."""
    _refuse_overwrite(out_path, version)

    per_corpus: dict[str, list[str]] = {}
    index_hashes: dict[str, dict[str, Any]] = {}
    index_orders: dict[str, list[str]] = {}
    for corpus, idx_path in codec_indexes.items():
        if not idx_path.is_file():
            raise FileNotFoundError(f"missing codec index: {idx_path}")
        index = _load_codec_index(idx_path)
        ids = list(index.keys())
        index_orders[corpus] = ids
        # Canonical D1 continuity order: lexicographic within corpus.
        per_corpus[corpus] = sorted(ids)
        index_hashes[corpus] = {
            "path": str(idx_path),
            "file_sha256": _sha256_file(idx_path),
            "n": len(ids),
        }

    # Prefer ruslan then CV to match D1 N2 row ordering.
    ordered: list[str] = []
    for corpus in (RUSLAN_CORPUS, CV_CORPUS):
        ordered.extend(per_corpus.get(corpus, []))
    for corpus, ids in per_corpus.items():
        if corpus not in (RUSLAN_CORPUS, CV_CORPUS):
            ordered.extend(ids)

    if len(ordered) != 32:
        raise SystemExit(f"DEV32 expected 32 utts from s1_32 shards, got {len(ordered)}")
    if KNOWN_DEV_HALLUCINATION not in ordered:
        raise SystemExit(f"DEV32 missing known hallucination utt {KNOWN_DEV_HALLUCINATION}")

    n_ruslan = sum(1 for u in ordered if _corpus_of(u) == RUSLAN_CORPUS)
    n_cv = sum(1 for u in ordered if _corpus_of(u) == CV_CORPUS)
    if (n_ruslan, n_cv) != (6, 26):
        raise SystemExit(f"DEV32 corpus ratio expected 6/26, got {n_ruslan}/{n_cv}")

    encode_meta_hash = None
    encode_meta_rel = None
    if encode_meta_path and encode_meta_path.is_file():
        encode_meta_hash = _sha256_file(encode_meta_path)
        try:
            encode_meta_rel = str(encode_meta_path.resolve().relative_to(ROOT))
        except ValueError:
            encode_meta_rel = str(encode_meta_path)

    payload = {
        "id": "d1b_dev32",
        "version": version,
        "description": (
            "D1b development oracle = exact codec_tokens_s1_32 inventory "
            "(D1 WAVE3c continuity). Not an independent confirmation set."
        ),
        "kind": "utt_ids",
        "role": "development_oracle",
        "n": len(ordered),
        "items_sha256": _items_sha256(ordered),
        "source": {
            "codec_tokens_subset_tag": "s1_32",
            "codec_indexes": index_hashes,
            "index_insertion_order": index_orders,
            "order": "lexicographic_within_corpus_ruslan_then_cv",
            "encode_meta": encode_meta_rel,
            "encode_meta_sha256": encode_meta_hash,
            "corpus_counts": {"ruslan": n_ruslan, "common_voice_ru_17_hf_mirror": n_cv},
            "known_hallucination_utt": KNOWN_DEV_HALLUCINATION,
        },
        "utt_ids": ordered,
        "splits": {
            "full32": ordered,
        },
    }
    file_sha = _write_json(out_path, payload)
    payload["file_sha256"] = file_sha
    return payload


def _select_confirm32(
    *,
    pool: list[Utterance],
    seed: int,
    n_ruslan: int,
    n_cv: int,
    edges: tuple[float, float],
) -> tuple[list[Utterance], dict[str, Any]]:
    rng = random.Random(seed)
    by_corpus: dict[str, list[Utterance]] = defaultdict(list)
    for row in pool:
        by_corpus[_corpus_of(row.utt_id, row)].append(row)

    def _bucketed(rows: list[Utterance]) -> dict[str, list[Utterance]]:
        buckets: dict[str, list[Utterance]] = {"short": [], "medium": [], "long": []}
        for r in rows:
            buckets[_duration_bucket(float(r.dur_s or 0.0), edges)].append(r)
        for b in buckets:
            rng.shuffle(buckets[b])
        return buckets

    selected: list[Utterance] = []
    used_cv_speakers: set[str] = set()
    report: dict[str, Any] = {
        "seed": seed,
        "duration_edges_s": {"short_lt": edges[0], "medium_lt": edges[1]},
        "targets": {"ruslan": n_ruslan, "cv": n_cv},
        "bucket_fills": {},
        "cv_speaker_unique": True,
        "notes": [],
    }

    # Target roughly even bucket coverage within each corpus quota.
    def _quota(n: int) -> dict[str, int]:
        base, rem = divmod(n, 3)
        q = {"short": base, "medium": base, "long": base}
        # Prefer medium, then long, then short for remainders.
        for b in ("medium", "long", "short"):
            if rem <= 0:
                break
            q[b] += 1
            rem -= 1
        return q

    for corpus, need in ((RUSLAN_CORPUS, n_ruslan), (CV_CORPUS, n_cv)):
        rows = list(by_corpus.get(corpus, []))
        if len(rows) < need:
            raise SystemExit(f"not enough {corpus} candidates: {len(rows)} < {need}")
        buckets = _bucketed(rows)
        quotas = _quota(need)
        picked: list[Utterance] = []
        fills = Counter()

        def _take(bucket: str, k: int) -> None:
            nonlocal picked
            avail = buckets[bucket]
            while k > 0 and avail:
                row = avail.pop()
                if corpus == CV_CORPUS:
                    spk = row.effective_speaker_id()
                    if spk in used_cv_speakers:
                        continue
                    used_cv_speakers.add(spk)
                picked.append(row)
                fills[bucket] += 1
                k -= 1

        for bucket, k in quotas.items():
            _take(bucket, k)

        # Top up if bucket scarcity / speaker skips left gaps.
        if len(picked) < need:
            leftovers: list[Utterance] = []
            for b in ("medium", "long", "short"):
                leftovers.extend(buckets[b])
            rng.shuffle(leftovers)
            for row in leftovers:
                if len(picked) >= need:
                    break
                if corpus == CV_CORPUS:
                    spk = row.effective_speaker_id()
                    if spk in used_cv_speakers:
                        continue
                    used_cv_speakers.add(spk)
                picked.append(row)
                fills[_duration_bucket(float(row.dur_s or 0.0), edges)] += 1

        if len(picked) < need:
            # Last resort: allow CV speaker reuse only if pool exhausted.
            if corpus == CV_CORPUS:
                report["cv_speaker_unique"] = False
                report["notes"].append(
                    "CV speaker uniqueness relaxed after unique-speaker pool exhausted"
                )
                rest = [r for r in rows if r not in picked]
                rng.shuffle(rest)
                while len(picked) < need and rest:
                    row = rest.pop()
                    picked.append(row)
                    fills[_duration_bucket(float(row.dur_s or 0.0), edges)] += 1

        if len(picked) != need:
            raise SystemExit(f"failed to select {need} for {corpus}, got {len(picked)}")
        report["bucket_fills"][corpus] = dict(fills)
        selected.extend(picked)

    # Stable final order: corpus (ruslan, cv) then utt_id.
    selected.sort(key=lambda r: (0 if _corpus_of(r.utt_id, r) == RUSLAN_CORPUS else 1, r.utt_id))
    return selected, report


def _split_confirm(
    utt_ids: list[str],
    *,
    seed: int,
    n_select: int = 8,
) -> dict[str, list[str]]:
    """Deterministic select8 / gate24 / full32 inside CONFIRM32."""
    if len(utt_ids) != 32:
        raise ValueError(f"CONFIRM32 must have 32 ids, got {len(utt_ids)}")
    rng = random.Random(seed ^ 0xC0FFEE)
    order = list(utt_ids)
    rng.shuffle(order)
    # Keep corpus mix in select8: take proportional from shuffled list but
    # ensure at least 1 ruslan and rest filled; prefer first 8 after shuffle
    # with a soft corpus rebalance.
    ruslan = [u for u in order if _corpus_of(u) == RUSLAN_CORPUS]
    cv = [u for u in order if _corpus_of(u) == CV_CORPUS]
    n_r_sel = max(1, round(n_select * (len(ruslan) / 32.0)))
    n_r_sel = min(n_r_sel, len(ruslan), n_select - 1)
    select = ruslan[:n_r_sel] + cv[: n_select - n_r_sel]
    select_set = set(select)
    gate = [u for u in utt_ids if u not in select_set]
    # Preserve parent full32 order for gate/full; select sorted for stability.
    select_sorted = sorted(select, key=lambda u: (0 if _corpus_of(u) == RUSLAN_CORPUS else 1, u))
    gate_ordered = [u for u in utt_ids if u in set(gate)]
    return {
        "select8": select_sorted,
        "gate24": gate_ordered,
        "full32": list(utt_ids),
    }


def freeze_confirm32(
    *,
    manifest_path: Path,
    trackn_paths: dict[str, Path],
    dev32_ids: Sequence[str],
    version: str,
    seed: int,
    out_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _refuse_overwrite(out_path, version)

    train_ids = set(_load_utt_ids(trackn_paths["train"]))
    held_out: dict[str, set[str]] = {}
    for name in ("val", "test", "review"):
        held_out[name] = set(_load_utt_ids(trackn_paths[name]))
    dev_set = set(dev32_ids)

    rows = read_manifest(manifest_path)
    by_id = {r.utt_id: r for r in rows}

    # Eligible = trackn_train minus DEV32 (val/test/review already outside train).
    pool_ids = sorted(train_ids - dev_set)
    leak_hits = {
        "dev32∩confirm_pool_before": sorted(dev_set & set(pool_ids))[:20],
        "train∩val": sorted(train_ids & held_out["val"])[:5],
        "train∩test": sorted(train_ids & held_out["test"])[:5],
        "train∩review": sorted(train_ids & held_out["review"])[:5],
    }
    if any(leak_hits[k] for k in ("train∩val", "train∩test", "train∩review")):
        raise SystemExit(f"trackn inventory leak before CONFIRM select: {leak_hits}")

    pool_rows: list[Utterance] = []
    missing = []
    for uid in pool_ids:
        row = by_id.get(uid)
        if row is None:
            missing.append(uid)
            continue
        if (row.qc_tag or "clean") != "clean":
            continue
        pool_rows.append(row)
    if missing:
        raise SystemExit(f"manifest missing {len(missing)} train ids e.g. {missing[:3]}")

    # Match S1 ratio 6 ruslan / 26 CV.
    n_ruslan, n_cv = 6, 26
    edges = _tertile_edges([float(r.dur_s or 0.0) for r in pool_rows])
    selected, sel_report = _select_confirm32(
        pool=pool_rows,
        seed=seed,
        n_ruslan=n_ruslan,
        n_cv=n_cv,
        edges=edges,
    )
    utt_ids = [r.utt_id for r in selected]
    if set(utt_ids) & dev_set:
        raise SystemExit("CONFIRM32 overlaps DEV32 after selection")

    # Fingerprint uniqueness.
    fps = [_audio_fingerprint(r) for r in selected]
    fp_counts = Counter(fps)
    dup_fps = [fp for fp, c in fp_counts.items() if c > 1]
    if dup_fps:
        raise SystemExit(f"duplicate audio fingerprints in CONFIRM32: {dup_fps[:5]}")
    if len(set(utt_ids)) != len(utt_ids):
        raise SystemExit("duplicate utt_ids in CONFIRM32")

    splits = _split_confirm(utt_ids, seed=seed)
    if set(splits["select8"]) & set(splits["gate24"]):
        raise SystemExit("select8/gate24 overlap")
    if sorted(splits["select8"] + splits["gate24"]) != sorted(utt_ids):
        raise SystemExit("select8∪gate24 != full32")

    # Leak report vs held-out inventories.
    confirm_set = set(utt_ids)
    leak_report = {
        "ok": True,
        "seed": seed,
        "n_confirm": len(utt_ids),
        "overlaps": {
            "confirm∩dev32": sorted(confirm_set & dev_set),
            "confirm∩val": sorted(confirm_set & held_out["val"]),
            "confirm∩test": sorted(confirm_set & held_out["test"]),
            "confirm∩review": sorted(confirm_set & held_out["review"]),
        },
        "duplicate_utt_ids": False,
        "duplicate_audio_fingerprints": False,
        "cv_speakers": sorted(
            {
                by_id[u].effective_speaker_id()
                for u in utt_ids
                if _corpus_of(u) == CV_CORPUS and u in by_id
            }
        ),
        "cv_speaker_unique": sel_report.get("cv_speaker_unique", True),
        "preselect_inventory_leaks": leak_hits,
    }
    for key, vals in leak_report["overlaps"].items():
        if vals:
            leak_report["ok"] = False

    source_hashes = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "trackn_sets": {
            name: {
                "path": str(path),
                "file_sha256": _sha256_file(path),
                "items_sha256": _items_sha256(_load_utt_ids(path)),
                "n": len(_load_utt_ids(path)),
            }
            for name, path in trackn_paths.items()
        },
        "dev32_items_sha256": _items_sha256(list(dev32_ids)),
    }

    hours = sum(float(r.dur_s or 0.0) for r in selected) / 3600.0
    bucket_counts = Counter(
        _duration_bucket(float(r.dur_s or 0.0), edges) for r in selected
    )
    payload = {
        "id": "d1b_confirm32",
        "version": version,
        "description": (
            "D1b confirmation oracle — disjoint from DEV32; train+score on same 32. "
            f"Frozen with seed={seed}."
        ),
        "kind": "utt_ids",
        "role": "confirmation_oracle",
        "n": len(utt_ids),
        "items_sha256": _items_sha256(utt_ids),
        "source": {
            "seed": seed,
            "selection": (
                "trackn_train minus DEV32/val/test/review; "
                f"ratio ruslan={n_ruslan} cv={n_cv}; "
                "duration tertile coverage; CV speaker diversity; "
                f"seed={seed}"
            ),
            "hashes": source_hashes,
            "speakers": len({r.effective_speaker_id() for r in selected}),
            "hours": hours,
            "corpus_counts": {
                "ruslan": sum(1 for u in utt_ids if _corpus_of(u) == RUSLAN_CORPUS),
                "common_voice_ru_17_hf_mirror": sum(
                    1 for u in utt_ids if _corpus_of(u) == CV_CORPUS
                ),
            },
            "duration_bucket_counts": dict(bucket_counts),
            "duration_edges_s": {"short_lt": edges[0], "medium_lt": edges[1]},
            "selection_report": sel_report,
        },
        "utt_ids": utt_ids,
        "splits": splits,
        "split_hashes": {
            name: _items_sha256(ids) for name, ids in splits.items()
        },
    }
    file_sha = _write_json(out_path, payload)
    payload["file_sha256"] = file_sha
    return payload, leak_report


def _mirror_frozen(
    *,
    frozen_dir: Path,
    dev: dict[str, Any],
    confirm: dict[str, Any],
    leak_report: dict[str, Any],
) -> dict[str, str]:
    frozen_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def _copy_set(name: str, payload: dict[str, Any]) -> None:
        out = frozen_dir / f"{name}.json"
        # Drop computed file_sha256 key before rewrite for stable content.
        body = {k: v for k, v in payload.items() if k != "file_sha256"}
        paths[name] = _write_json(out, body)

    _copy_set("dev32", dev)
    _copy_set("confirm32", confirm)
    for split_name in ("select8", "gate24", "full32"):
        ids = confirm["splits"][split_name]
        body = {
            "id": f"d1b_confirm_{split_name}",
            "version": confirm["version"],
            "parent": "d1b_confirm32",
            "kind": "utt_ids",
            "n": len(ids),
            "items_sha256": _items_sha256(ids),
            "utt_ids": ids,
        }
        paths[f"confirm_{split_name}"] = _write_json(
            frozen_dir / f"confirm_{split_name}.json", body
        )

    paths["leak_report"] = _write_json(frozen_dir / "leak_report.json", leak_report)
    selection_report = {
        "plan_slug": PLAN_SLUG,
        "dev32": {
            "id": dev["id"],
            "version": dev["version"],
            "n": dev["n"],
            "items_sha256": dev["items_sha256"],
            "corpus_counts": dev["source"]["corpus_counts"],
            "known_hallucination_utt": KNOWN_DEV_HALLUCINATION,
        },
        "confirm32": {
            "id": confirm["id"],
            "version": confirm["version"],
            "n": confirm["n"],
            "items_sha256": confirm["items_sha256"],
            "split_hashes": confirm["split_hashes"],
            "corpus_counts": confirm["source"]["corpus_counts"],
            "selection_report": confirm["source"]["selection_report"],
        },
        "disjoint": not bool(set(dev["utt_ids"]) & set(confirm["utt_ids"])),
    }
    paths["selection_report"] = _write_json(
        frozen_dir / "selection_report.json", selection_report
    )

    hashes = {
        "plan_slug": PLAN_SLUG,
        "dev32": {
            "path": "configs/train_subsets/d1b_dev32.json",
            "version": dev["version"],
            "n": dev["n"],
            "items_sha256": dev["items_sha256"],
            "file_sha256": dev.get("file_sha256"),
        },
        "confirm32": {
            "path": "configs/train_subsets/d1b_confirm32.json",
            "version": confirm["version"],
            "n": confirm["n"],
            "items_sha256": confirm["items_sha256"],
            "file_sha256": confirm.get("file_sha256"),
            "split_hashes": confirm["split_hashes"],
        },
        "frozen_mirrors": paths,
        "disjoint_dev_confirm": selection_report["disjoint"],
    }
    paths["hashes"] = _write_json(frozen_dir / "hashes.json", hashes)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--dev-out", default="configs/train_subsets/d1b_dev32.json")
    ap.add_argument("--confirm-out", default="configs/train_subsets/d1b_confirm32.json")
    ap.add_argument(
        "--frozen-dir",
        default=f"out/{PLAN_SLUG}/frozen",
    )
    ap.add_argument(
        "--encode-meta",
        default=".cursor/results/d1_discrete_speech_lm/artifacts/encode_meta_s1_32.json",
    )
    ap.add_argument("--dev-only", action="store_true")
    ap.add_argument("--confirm-only", action="store_true")
    args = ap.parse_args(argv)

    codec_indexes = {
        corpus: expand(path) for corpus, path in DEFAULT_CODEC_S1.items()
    }
    trackn_paths = {k: _resolve_repo(v) for k, v in DEFAULT_TRACKN.items()}
    manifest_path = expand(args.manifest)
    dev_out = _resolve_repo(args.dev_out)
    confirm_out = _resolve_repo(args.confirm_out)
    frozen_dir = _resolve_repo(args.frozen_dir)
    encode_meta = _resolve_repo(args.encode_meta)

    if not args.confirm_only:
        dev = export_dev32(
            codec_indexes=codec_indexes,
            version=args.version,
            out_path=dev_out,
            encode_meta_path=encode_meta,
        )
    else:
        dev = _load_json(dev_out)
        if "file_sha256" not in dev:
            dev["file_sha256"] = _sha256_file(dev_out)

    if not args.dev_only:
        confirm, leak = freeze_confirm32(
            manifest_path=manifest_path,
            trackn_paths=trackn_paths,
            dev32_ids=dev["utt_ids"],
            version=args.version,
            seed=int(args.seed),
            out_path=confirm_out,
        )
    else:
        confirm = _load_json(confirm_out)
        leak = {"ok": None, "note": "confirm-only skipped; leak report not regenerated"}

    mirrors = _mirror_frozen(
        frozen_dir=frozen_dir, dev=dev, confirm=confirm, leak_report=leak
    )

    print(
        json.dumps(
            {
                "dev32_path": str(dev_out),
                "dev32_items_sha256": dev["items_sha256"],
                "confirm32_path": str(confirm_out),
                "confirm32_items_sha256": confirm["items_sha256"],
                "disjoint": not bool(set(dev["utt_ids"]) & set(confirm["utt_ids"])),
                "leak_ok": leak.get("ok"),
                "frozen_dir": str(frozen_dir),
                "mirrors": mirrors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if leak.get("ok") is False:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
