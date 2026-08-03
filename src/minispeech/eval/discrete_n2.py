"""D1b discrete N2 protocol helpers: FR selection, catastrophic, stability, text-usage.

Shared by ``scripts/74_eval_discrete_lm.py`` and ``scripts/76_select_discrete_checkpoint.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from minispeech.eval.bootstrap import EditCounts, corpus_rate, paired_bootstrap_delta
from minispeech.frontend.normalize import normalize_for_asr

# Catastrophic substitution (eval_contract catastrophic.definition_version = d1b_v1).
CATASTROPHIC_CER_GE = 0.50
CATASTROPHIC_CER_AND_JACCARD_CER = 0.35
CATASTROPHIC_JACCARD_LT = 0.20

# Text-usage D1b acceptance (§14).
TEXT_USAGE_CER_MARGIN = 0.05
TEXT_USAGE_PREFIX_CHANGE_FRAC = 0.25

# Stability D1b acceptance (§14).
STABILITY_CB0_MAX_RUN = 12
STABILITY_MEAN_RATIO_LO = 0.90
STABILITY_MEAN_RATIO_HI = 1.10
STABILITY_UTT_RATIO_LO = 0.5
STABILITY_UTT_RATIO_HI = 2.0

# Immutable FR candidate optimizer steps (trainer must honor these).
FR_CANDIDATE_STEPS_BASE = (250, 375, 500, 625, 750)
FR_CANDIDATE_STEPS_DEV_EXT = (1000, 1250)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def items_sha256(items: Sequence[str]) -> str:
    """SHA256 of canonical JSON list — matches ``scripts/75_freeze_d1b_oracles.py``."""
    return hashlib.sha256(canonical_json_bytes(list(items))).hexdigest()


def word_tokens(text: str) -> list[str]:
    """Word tokens after ASR-normalization (strips stress/punct so refs match hyps)."""
    norm = normalize_for_asr(text or "")
    return [t.lower() for t in _WORD_RE.findall(norm)]


def word_token_jaccard(ref: str, hyp: str) -> float:
    """Jaccard over ASR-normalized word tokens (case-insensitive).

    Combining acute stress in refs must not fragment tokens; this matches
    ``eval_contract.cer_normalization`` (normalize_for_asr_v1).
    """
    a = set(word_tokens(ref))
    b = set(word_tokens(hyp))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


def is_catastrophic(
    cer: float | None,
    ref: str = "",
    hyp: str = "",
    *,
    cer_ge: float = CATASTROPHIC_CER_GE,
    cer_jaccard: float = CATASTROPHIC_CER_AND_JACCARD_CER,
    jaccard_lt: float = CATASTROPHIC_JACCARD_LT,
) -> bool:
    """D1b catastrophic: CER>=0.50 OR (CER>=0.35 AND word Jaccard<0.20)."""
    if cer is None or (isinstance(cer, float) and math.isnan(cer)):
        return False
    c = float(cer)
    if c >= cer_ge:
        return True
    if c >= cer_jaccard and word_token_jaccard(ref, hyp) < jaccard_lt:
        return True
    return False


def annotate_catastrophic_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return shallow-copied rows with catastrophic / jaccard fields."""
    out: list[dict[str, Any]] = []
    for r in rows:
        row = dict(r)
        cer = row.get("cer")
        ref = str(row.get("ref") or "")
        hyp = str(row.get("hyp") or "")
        jac = word_token_jaccard(ref, hyp)
        cat = is_catastrophic(cer if cer is None else float(cer), ref, hyp)
        row["word_jaccard"] = jac
        row["catastrophic"] = cat
        out.append(row)
    return out


def catastrophic_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    annotated = annotate_catastrophic_rows(rows)
    return [r for r in annotated if r.get("catastrophic")]


def bootstrap_mean_cer(
    cers: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    if not cers:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    arr = np.asarray(list(cers), dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    n = len(arr)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(arr[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def micro_cer_from_edits(edit_counts: Sequence[EditCounts] | Sequence[Mapping[str, Any]]) -> float:
    """Corpus micro CER = sum(edits) / sum(ref_units)."""
    counts: list[EditCounts] = []
    for c in edit_counts:
        if isinstance(c, EditCounts):
            counts.append(c)
        else:
            counts.append(
                EditCounts(
                    hits=int(c.get("hits", 0)),
                    substitutions=int(c.get("substitutions", 0)),
                    deletions=int(c.get("deletions", 0)),
                    insertions=int(c.get("insertions", 0)),
                )
            )
    return float(corpus_rate(counts))


def cb0_stats(code_batches: Sequence[Any], vocab: int = 2048) -> dict[str, Any]:
    """CB0 utilization / entropy / unique / max-run (+ healthy flag)."""
    counts: Counter[int] = Counter()
    total = 0
    max_run = 0
    for codes in code_batches:
        if hasattr(codes, "detach"):
            col = codes[:, 0].reshape(-1).tolist()
        else:
            arr = np.asarray(codes)
            if arr.ndim == 2:
                col = arr[:, 0].reshape(-1).tolist()
            else:
                col = list(arr)
        counts.update(int(x) for x in col)
        total += len(col)
        prev, run = None, 0
        for v in col:
            if v == prev:
                run += 1
            else:
                run = 1
                prev = v
            max_run = max(max_run, run)
    used = len(counts)
    util = used / float(vocab) if vocab else 0.0
    probs = (
        np.array([c / total for c in counts.values()], dtype=np.float64) if total else np.array([])
    )
    entropy = float(-(probs * np.log(probs + 1e-12)).sum()) if probs.size else 0.0
    return {
        "cb0_utilization": util,
        "cb0_entropy": entropy,
        "cb0_unique": used,
        "cb0_total_frames": total,
        "cb0_max_run": max_run,
        "healthy": util >= 0.05 and entropy >= 2.0 and max_run < 64,
    }


def duration_ratios_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    out: list[float] = []
    for r in rows:
        gt = int(r.get("gt_frames") or 0)
        n = int(r.get("n_frames") or 0)
        if gt > 0:
            out.append(n / float(gt))
    return out


def duration_ratio_distribution(ratios: Sequence[float]) -> dict[str, Any]:
    if not ratios:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p10": None,
            "p90": None,
            "buckets": {},
            "outliers_lt_0_5": [],
            "outliers_gt_2_0": [],
            "outlier_indices": [],
        }
    arr = np.asarray(list(ratios), dtype=np.float64)
    buckets = {
        "lt_0_5": int(np.sum(arr < 0.5)),
        "0_5_to_0_9": int(np.sum((arr >= 0.5) & (arr < 0.9))),
        "0_9_to_1_1": int(np.sum((arr >= 0.9) & (arr <= 1.1))),
        "1_1_to_2_0": int(np.sum((arr > 1.1) & (arr <= 2.0))),
        "gt_2_0": int(np.sum(arr > 2.0)),
    }
    outliers_lo = [i for i, r in enumerate(arr) if r < STABILITY_UTT_RATIO_LO]
    outliers_hi = [i for i, r in enumerate(arr) if r > STABILITY_UTT_RATIO_HI]
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "buckets": buckets,
        "outliers_lt_0_5": outliers_lo,
        "outliers_gt_2_0": outliers_hi,
        "outlier_indices": sorted(set(outliers_lo) | set(outliers_hi)),
    }


def stop_reason_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        reason = r.get("stop_reason") or "unknown"
        c[str(reason)] += 1
    return dict(sorted(c.items()))


def estimate_stop_reason(
    *,
    n_frames: int,
    text_len: int,
    cb0_ids: Sequence[int],
    max_frames: int,
    min_frames: int = 4,
    max_cb0_repeat: int = 12,
    frames_per_text_token: float = 2.5,
) -> str:
    """Best-effort stop reason when generate() does not emit metadata."""
    if n_frames <= 0:
        return "empty"
    if n_frames >= int(max_frames):
        return "max_frames"
    expected = max(int(min_frames), int(max(1, text_len) * float(frames_per_text_token)))
    hard_cap = min(int(max_frames), int(expected * 1.8 + 8))
    if n_frames >= hard_cap:
        return "hard_cap"
    # Trailing CB0 run
    run = 1
    for i in range(len(cb0_ids) - 1, 0, -1):
        if cb0_ids[i] == cb0_ids[i - 1]:
            run += 1
        else:
            break
    if max_cb0_repeat > 0 and run >= int(max_cb0_repeat):
        return "cb0_repeat"
    return "stop_head"


def text_usage_acceptance(
    *,
    model_cer: float,
    shuffle_cer: float,
    mask_cer: float,
    model_rows: Sequence[Mapping[str, Any]],
    shuffle_rows: Sequence[Mapping[str, Any]],
    paired_ci95_high: float | None,
    margin: float = TEXT_USAGE_CER_MARGIN,
    change_frac: float = TEXT_USAGE_PREFIX_CHANGE_FRAC,
) -> dict[str, Any]:
    """D1b text-usage: shuffle/mask ≥ model+0.05; ≥25% hyp change; paired CI high < 0."""
    shuffle_ok = (not math.isnan(shuffle_cer)) and shuffle_cer >= model_cer + margin
    mask_ok = (not math.isnan(mask_cer)) and mask_cer >= model_cer + margin
    n = min(len(model_rows), len(shuffle_rows))
    content_change = 0
    for i in range(n):
        a = (model_rows[i].get("hyp") or "").strip()
        b = (shuffle_rows[i].get("hyp") or "").strip()
        if a and a != b:
            content_change += 1
    need = max(1, int(math.ceil(n * change_frac))) if n else 0
    prefix_ok = content_change >= need if n else False
    paired_ok = paired_ci95_high is not None and float(paired_ci95_high) < 0.0
    return {
        "shuffle_cer": float(shuffle_cer),
        "mask_cer": float(mask_cer),
        "model_cer": float(model_cer),
        "margin": float(margin),
        "shuffle_ok": bool(shuffle_ok),
        "mask_ok": bool(mask_ok),
        "content_change_n": content_change,
        "content_change_need": need,
        "prefix_cf_ok": bool(prefix_ok),
        "paired_ci95_high": paired_ci95_high,
        "paired_ci_lt_zero": bool(paired_ok),
        "pass": bool(shuffle_ok and mask_ok and prefix_ok and paired_ok),
    }


def stability_acceptance(
    model_report: Mapping[str, Any],
    *,
    max_cb0_run: int = STABILITY_CB0_MAX_RUN,
    mean_lo: float = STABILITY_MEAN_RATIO_LO,
    mean_hi: float = STABILITY_MEAN_RATIO_HI,
    utt_lo: float = STABILITY_UTT_RATIO_LO,
    utt_hi: float = STABILITY_UTT_RATIO_HI,
) -> dict[str, Any]:
    """D1b stability: no runaway; cb0_max_run<=12; mean ratio in [0.9,1.1]; no utt extremes."""
    cb0 = model_report.get("cb0") or {}
    ratios = model_report.get("duration_ratios")
    if ratios is None:
        ratios = duration_ratios_from_rows(model_report.get("rows") or [])
    dist = model_report.get("duration_ratio_distribution") or duration_ratio_distribution(ratios)
    mean_r = dist.get("mean")
    if mean_r is None:
        mean_r = model_report.get("mean_dur_ratio")
    runaway = bool(model_report.get("runaway_or_repetition"))
    max_run = int(cb0.get("cb0_max_run", 0) or 0)
    no_runaway = not runaway
    cb0_ok = max_run <= int(max_cb0_run)
    mean_ok = (
        mean_r is not None
        and not (isinstance(mean_r, float) and math.isnan(mean_r))
        and mean_lo <= float(mean_r) <= mean_hi
    )
    outliers = list(dist.get("outlier_indices") or [])
    no_extreme = len(outliers) == 0
    return {
        "runaway_or_repetition": runaway,
        "no_runaway": no_runaway,
        "cb0_max_run": max_run,
        "cb0_max_run_limit": int(max_cb0_run),
        "cb0_max_run_ok": cb0_ok,
        "mean_dur_ratio": mean_r,
        "mean_dur_ratio_ok": bool(mean_ok),
        "mean_dur_ratio_range": [mean_lo, mean_hi],
        "duration_buckets": dist.get("buckets") or {},
        "utt_ratio_outliers": outliers,
        "utt_ratio_limits": [utt_lo, utt_hi],
        "no_extreme_utt_ratios": no_extreme,
        "duration_ratio_distribution": dist,
        "pass": bool(no_runaway and cb0_ok and mean_ok and no_extreme),
    }


def abs_log_duration_ratio(mean_dur_ratio: float | None) -> float:
    if mean_dur_ratio is None:
        return float("inf")
    r = float(mean_dur_ratio)
    if r <= 0.0 or math.isnan(r) or math.isinf(r):
        return float("inf")
    return abs(math.log(r))


def is_gate_eligible(candidate: Mapping[str, Any]) -> bool:
    """Eligible = not runaway and (optional explicit flag)."""
    if candidate.get("gate_eligible") is False:
        return False
    if bool(candidate.get("runaway_or_repetition")):
        return False
    return True


def selection_sort_key(candidate: Mapping[str, Any]) -> tuple:
    """Lexicographic FR selection key (§13).

    Prefer: gate-eligible → min macro CER → fewer catastrophics →
    smaller |log(mean_dur_ratio)| → lower teacher-forced CE.
    """
    eligible = is_gate_eligible(candidate)
    macro = candidate.get("macro_cer", candidate.get("mean_cer", float("inf")))
    try:
        macro_f = float(macro)
    except (TypeError, ValueError):
        macro_f = float("inf")
    if math.isnan(macro_f):
        macro_f = float("inf")
    n_cat = int(candidate.get("n_catastrophic", 0) or 0)
    mean_dur = candidate.get("mean_dur_ratio")
    try:
        mean_dur_f = float(mean_dur) if mean_dur is not None else None
    except (TypeError, ValueError):
        mean_dur_f = None
    abs_log = abs_log_duration_ratio(mean_dur_f)
    tf = candidate.get("teacher_forced_ce", candidate.get("tf_ce", float("inf")))
    try:
        tf_f = float(tf)
    except (TypeError, ValueError):
        tf_f = float("inf")
    if math.isnan(tf_f):
        tf_f = float("inf")
    return (
        0 if eligible else 1,
        macro_f,
        n_cat,
        abs_log,
        tf_f,
    )


def select_fr_checkpoint(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selection_subset: str | None = None,
) -> dict[str, Any]:
    """Pick best candidate by lexicographic key; never renames files to best.pt."""
    if not candidates:
        raise ValueError("no candidates to select")
    rows: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        row = dict(c)
        if selection_subset is not None:
            row.setdefault("selection_subset", selection_subset)
        key = selection_sort_key(row)
        row["selection_key"] = list(key)
        row["gate_eligible"] = is_gate_eligible(row)
        row["_idx"] = i
        rows.append(row)
    ranked = sorted(rows, key=lambda r: (tuple(r["selection_key"]), r.get("step", 10**12), r["_idx"]))
    for rank, r in enumerate(ranked):
        r["rank"] = rank
    best = ranked[0]
    # Strip private idx from public rows
    public = [{k: v for k, v in r.items() if k != "_idx"} for r in ranked]
    return {
        "protocol": "d1b_fr_lexicographic_v1",
        "selection_subset": selection_subset or best.get("selection_subset"),
        "selected": {
            "ckpt": best.get("ckpt"),
            "step": best.get("step"),
            "rank": 0,
            "selection_key": best["selection_key"],
            "macro_cer": best.get("macro_cer", best.get("mean_cer")),
            "n_catastrophic": best.get("n_catastrophic"),
            "mean_dur_ratio": best.get("mean_dur_ratio"),
            "teacher_forced_ce": best.get("teacher_forced_ce", best.get("tf_ce")),
            "gate_eligible": best.get("gate_eligible"),
            "runaway_or_repetition": best.get("runaway_or_repetition"),
        },
        "candidates": public,
        "note": (
            "Selected checkpoint must be copied/symlinked to best.pt only after this "
            "file is written. Candidates must not be named best.pt before selection. "
            "CONFIRM selection uses select8 only."
        ),
    }


def enrich_model_report(
    report: Mapping[str, Any],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Add D1b macro/micro/catastrophic/duration/stop fields to a free-running report."""
    out = dict(report)
    rows = annotate_catastrophic_rows(out.get("rows") or [])
    out["rows"] = rows
    cers = [float(r["cer"]) for r in rows if r.get("cer") is not None]
    boot = bootstrap_mean_cer(cers, n_boot=n_boot, seed=seed)
    out["macro_cer"] = boot["mean"]
    out["macro_cer_median"] = boot["median"]
    out["macro_cer_ci95_low"] = boot["ci95_low"]
    out["macro_cer_ci95_high"] = boot["ci95_high"]
    # Keep legacy aliases in sync when present
    out["mean_cer"] = boot["mean"]
    out["cer_ci95_low"] = boot["ci95_low"]
    out["cer_ci95_high"] = boot["ci95_high"]
    edits = out.get("edit_counts") or []
    if edits:
        out["micro_cer"] = micro_cer_from_edits(edits)
    else:
        out["micro_cer"] = float("nan")
    cats = [r for r in rows if r.get("catastrophic")]
    out["n_catastrophic"] = len(cats)
    out["catastrophic_rows"] = cats
    ratios = duration_ratios_from_rows(rows)
    out["duration_ratios"] = ratios
    dist = duration_ratio_distribution(ratios)
    out["duration_ratio_distribution"] = dist
    out["mean_dur_ratio"] = dist.get("mean")
    out["stop_reason_distribution"] = stop_reason_distribution(rows)
    # Runaway: duration>3x or long CB0 run (legacy) OR stability extreme
    cb0 = out.get("cb0") or {}
    runaway = any(r > 3.0 for r in ratios) or int(cb0.get("cb0_max_run", 0) or 0) >= 64
    out["runaway_or_repetition"] = bool(out.get("runaway_or_repetition", False) or runaway)
    return out


def paired_bootstrap_model_vs_wrong(
    model_edits: Sequence[Any],
    wrong_edits: Sequence[Any],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any] | None:
    if not model_edits or not wrong_edits or len(model_edits) != len(wrong_edits):
        return None

    def _as_ec(c: Any) -> EditCounts:
        if isinstance(c, EditCounts):
            return c
        return EditCounts(
            hits=int(c.get("hits", 0)),
            substitutions=int(c.get("substitutions", 0)),
            deletions=int(c.get("deletions", 0)),
            insertions=int(c.get("insertions", 0)),
        )

    m = [_as_ec(c) for c in model_edits]
    w = [_as_ec(c) for c in wrong_edits]
    # Δ = model - wrong in pp; model better ⇒ negative
    paired = paired_bootstrap_delta(w, m, n_boot=n_boot, threshold_pp=0.0, seed=seed)
    return {
        "delta_pp_model_minus_wrong": paired.delta_pp,
        "ci95_low": paired.ci95_low,
        "ci95_high": paired.ci95_high,
        "separated": paired.ci95_high < 0.0,
    }


def evaluate_n2_gate_d1b(
    model_report: Mapping[str, Any],
    copy_report: Mapping[str, Any],
    wrong_report: Mapping[str, Any],
    random_report: Mapping[str, Any],
    shuffle_report: Mapping[str, Any],
    mask_report: Mapping[str, Any],
    *,
    copy_synth_ceiling: float,
    margin: float = 0.03,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Original N2 criteria plus D1b catastrophic / text-usage / stability gates."""
    model = enrich_model_report(model_report, n_boot=n_boot, seed=seed)
    copy = enrich_model_report(copy_report, n_boot=n_boot, seed=seed)
    wrong = enrich_model_report(wrong_report, n_boot=n_boot, seed=seed)
    random = enrich_model_report(random_report, n_boot=n_boot, seed=seed)
    shuffle = enrich_model_report(shuffle_report, n_boot=n_boot, seed=seed)
    mask = enrich_model_report(mask_report, n_boot=n_boot, seed=seed)

    fr_cer = float(model["macro_cer"])
    copy_cer = float(copy["macro_cer"])
    if math.isnan(copy_cer):
        copy_cer = float(copy_synth_ceiling)
    ceiling = copy_cer + margin
    cer_pass = (not math.isnan(fr_cer)) and fr_cer <= ceiling

    sep_wrong = float(model["cer_ci95_high"]) < float(wrong["cer_ci95_low"])
    sep_random = float(model["cer_ci95_high"]) < float(random["cer_ci95_low"])
    if not sep_wrong:
        sep_wrong = fr_cer + 0.05 < float(wrong["macro_cer"])
    if not sep_random:
        sep_random = fr_cer + 0.05 < float(random["macro_cer"])

    paired = paired_bootstrap_model_vs_wrong(
        model.get("edit_counts") or [],
        wrong.get("edit_counts") or [],
        n_boot=n_boot,
        seed=seed,
    )
    if paired and paired["separated"]:
        sep_wrong = True

    # Legacy soft text-usage (methodology continuity)
    text_shuffle_hurts = float(shuffle["macro_cer"]) > fr_cer + 0.02
    text_mask_hurts = float(mask["macro_cer"]) > fr_cer + 0.02
    content_change = 0
    for a, b in zip(model["rows"], shuffle["rows"]):
        if (a.get("hyp") or "").strip() and (a.get("hyp") or "") != (b.get("hyp") or ""):
            content_change += 1
    prefix_cf_pass = content_change >= max(1, len(model["rows"]) // 4)

    text_usage = text_usage_acceptance(
        model_cer=fr_cer,
        shuffle_cer=float(shuffle["macro_cer"]),
        mask_cer=float(mask["macro_cer"]),
        model_rows=model["rows"],
        shuffle_rows=shuffle["rows"],
        paired_ci95_high=None if paired is None else float(paired["ci95_high"]),
    )
    stability = stability_acceptance(model)
    cb0_ok = bool((model.get("cb0") or {}).get("healthy"))
    no_runaway = not bool(model.get("runaway_or_repetition"))
    zero_cat = int(model.get("n_catastrophic", 0) or 0) == 0

    criteria = {
        "fr_cer": fr_cer,
        "macro_cer": fr_cer,
        "macro_cer_median": model.get("macro_cer_median"),
        "macro_cer_ci95_low": model.get("macro_cer_ci95_low"),
        "macro_cer_ci95_high": model.get("macro_cer_ci95_high"),
        "micro_cer": model.get("micro_cer"),
        "copy_synth_cer_subset": copy_cer,
        "ceiling_copy_plus_margin": ceiling,
        "cer_pass": cer_pass,
        "bootstrap_sep_wrong_utt": sep_wrong,
        "bootstrap_sep_random": sep_random,
        "paired_vs_wrong": paired,
        "no_runaway_repetition": no_runaway,
        "cb0_healthy": cb0_ok,
        "cb0": model.get("cb0"),
        "text_usage_shuffle_hurts": text_shuffle_hurts,
        "text_usage_mask_hurts": text_mask_hurts,
        "text_usage_prefix_cf_changes": prefix_cf_pass,
        "text_usage_d1b": text_usage,
        "stability_d1b": stability,
        "n_catastrophic": model.get("n_catastrophic"),
        "zero_catastrophic": zero_cat,
        "duration_ratio_distribution": model.get("duration_ratio_distribution"),
        "stop_reason_distribution": model.get("stop_reason_distribution"),
        "shuffle_cer": float(shuffle["macro_cer"]),
        "mask_cer": float(mask["macro_cer"]),
        "wrong_utt_cer": float(wrong["macro_cer"]),
        "random_codes_cer": float(random["macro_cer"]),
    }
    hard = [
        cer_pass,
        sep_wrong,
        sep_random,
        no_runaway,
        cb0_ok,
        text_usage["pass"],
        stability["pass"],
        zero_cat,
    ]
    # Also require legacy text-usage soft flags for continuity with D1 N2
    hard_legacy = [
        cer_pass,
        sep_wrong,
        sep_random,
        no_runaway,
        cb0_ok,
        text_shuffle_hurts,
        text_mask_hurts,
        prefix_cf_pass,
    ]
    verdict = "PASS" if all(hard) else "FAIL"
    return {
        "gate": "N2_D1b",
        "verdict": verdict,
        "criteria": criteria,
        "legacy_n2_verdict": "PASS" if all(hard_legacy) else "FAIL",
        "enriched_reports": {
            "model": {k: v for k, v in model.items() if k not in ("rows", "edit_counts")},
            "copy_synth": {k: v for k, v in copy.items() if k not in ("rows", "edit_counts")},
            "wrong_utt": {k: v for k, v in wrong.items() if k not in ("rows", "edit_counts")},
            "random_codes": {k: v for k, v in random.items() if k not in ("rows", "edit_counts")},
            "shuffle_text": {k: v for k, v in shuffle.items() if k not in ("rows", "edit_counts")},
            "mask_text": {k: v for k, v in mask.items() if k not in ("rows", "edit_counts")},
        },
        "model_rows": model["rows"],
        "catastrophic_rows": model.get("catastrophic_rows") or [],
    }


def load_utt_ids_from_subset(
    path: Path | str,
    *,
    split: str | None = None,
) -> list[str]:
    """Load ordered utt_ids; optional ``split`` for CONFIRM select8/gate24/full32."""
    from minispeech.data.d1b_oracles import (
        SUBSET_ALIASES,
        load_frozen_utt_ids,
        utt_ids_from_payload,
    )

    key = str(path).strip().lower().replace("-", "_")
    if key in SUBSET_ALIASES:
        ids, _, _ = load_frozen_utt_ids(path, split=split, verify_hash=True)
        return ids

    p = Path(path)
    if not p.is_file():
        # Allow bare relative paths via load_frozen_utt_ids resolve.
        ids, _, _ = load_frozen_utt_ids(path, split=split, verify_hash=True)
        return ids

    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        ids = utt_ids_from_payload(data, split=split)
        # Verify hash when frozen schema provides it.
        from minispeech.data.d1b_oracles import verify_items_sha256

        verify_items_sha256(ids, data, split=split)
        return ids
    raise ValueError(f"unrecognized subset schema: {path}")


def assert_dev_confirm_disjoint(dev_ids: Sequence[str], confirm_ids: Sequence[str]) -> None:
    overlap = set(dev_ids) & set(confirm_ids)
    if overlap:
        raise AssertionError(f"DEV/CONFIRM overlap ({len(overlap)}): {sorted(overlap)[:5]}...")


def resolve_infer_settings(
    cfg: Mapping[str, Any] | None = None,
    *,
    infer_overrides: Mapping[str, Any] | None = None,
    infer_json: Path | str | None = None,
    max_frames_override: int | None = None,
    codebook_size: int = 2048,
) -> dict[str, Any]:
    """Resolve infer knobs via shared ``discrete_config`` (scripts 73/74 parity).

    Optional ``infer_json`` is merged as overrides (``infer`` sub-object or flat).
    """
    from minispeech.infer.discrete_config import resolve_discrete_infer_settings

    overrides: dict[str, Any] = {}
    if infer_json is not None:
        path = Path(infer_json)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                if isinstance(payload.get("infer"), dict):
                    overrides.update(payload["infer"])
                else:
                    overrides.update(payload)
    if infer_overrides:
        overrides.update(dict(infer_overrides))
    return resolve_discrete_infer_settings(
        cfg,
        overrides=overrides or None,
        max_frames_override=max_frames_override,
        codebook_size=codebook_size,
    )


def candidate_from_eval_report(
    report: Mapping[str, Any],
    *,
    ckpt: str,
    step: int | None = None,
    teacher_forced_ce: float | None = None,
    selection_subset: str | None = None,
) -> dict[str, Any]:
    """Build a selection-candidate row from an enriched free-running model report."""
    enriched = enrich_model_report(report) if "macro_cer" not in report else dict(report)
    return {
        "ckpt": ckpt,
        "step": step,
        "selection_subset": selection_subset,
        "macro_cer": enriched.get("macro_cer", enriched.get("mean_cer")),
        "mean_cer": enriched.get("mean_cer", enriched.get("macro_cer")),
        "n_catastrophic": enriched.get("n_catastrophic", 0),
        "mean_dur_ratio": enriched.get("mean_dur_ratio"),
        "runaway_or_repetition": enriched.get("runaway_or_repetition", False),
        "gate_eligible": not bool(enriched.get("runaway_or_repetition", False)),
        "teacher_forced_ce": teacher_forced_ce,
        "cb0": enriched.get("cb0"),
        "stop_reason_distribution": enriched.get("stop_reason_distribution"),
    }


def fr_candidate_steps(*, include_dev_extension: bool = False) -> tuple[int, ...]:
    if include_dev_extension:
        return FR_CANDIDATE_STEPS_BASE + FR_CANDIDATE_STEPS_DEV_EXT
    return FR_CANDIDATE_STEPS_BASE
