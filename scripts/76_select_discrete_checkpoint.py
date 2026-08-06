#!/usr/bin/env python3
"""Offline free-running checkpoint selection for D1b (§13).

Reads scored candidate JSON (or a manifest listing candidates), ranks by the
lexicographic FR key, and writes ``checkpoint_selection.json``.

Candidates must NOT be named ``best.pt`` until after selection. CONFIRM runs
must score ``select8`` only for this script; ``gate24``/``full32`` are scored
after promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

from minispeech.data.d1b_oracles import (
    DEFAULT_EVAL_CONTRACT,
    eval_contract_content_sha256,
    items_sha256,
    load_eval_contract,
    load_frozen_utt_ids,
)
from minispeech.eval.discrete_n2 import (
    candidate_from_eval_report,
    fr_candidate_steps,
    parse_teacher_forced_ce,
    select_fr_checkpoint,
    teacher_forced_ce_from_ckpt_metrics,
)
from minispeech.eval.discrete_report_schema import validate_formal_report
from minispeech.utils import expand, git_sha


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


_D1C_STEPS = (250, 375, 500, 625, 750)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_state_sha256(ckpt_path: Path) -> str:
    """Digest trainer ``model`` state_dict bytes (training identity, not file padding).

    Fail-closed: missing/unloadable/empty model state is refused. File-level
    ``checkpoint.sha256`` alone is insufficient — junk padding can make five
    re-saves of one weight state look distinct.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ValueError("torch required to digest checkpoint model state") from exc
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "D1c formal selection refuses unreadable checkpoint model state at "
            f"{ckpt_path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            "D1c formal selection refuses checkpoint without dict payload "
            f"(need trainer model state_dict) at {ckpt_path}"
        )
    sd = payload.get("model")
    if not isinstance(sd, dict) or not sd:
        raise ValueError(
            "D1c formal selection refuses checkpoint without non-empty model "
            f"state_dict (training identity); path={ckpt_path}"
        )
    if any(str(k).startswith("module.") for k in sd):
        sd = {str(k).removeprefix("module."): v for k, v in sd.items()}
    h = hashlib.sha256()
    for key in sorted(sd.keys()):
        tensor = sd[key]
        if not torch.is_tensor(tensor):
            raise ValueError(
                "D1c formal selection refuses non-tensor model state entry "
                f"{key!r} at {ckpt_path}"
            )
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(str(key).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _require_checkpoint_hashes_match_provenance(reports: list[dict[str, Any]]) -> None:
    """Fail-closed: ``checkpoint.hashes.*`` must match ``provenance.*_sha256``."""
    pairs = (
        ("eval_contract", "eval_contract_sha256"),
        ("recipe", "recipe_sha256"),
    )
    for report in reports:
        ckpt = report["checkpoint"]
        step = int(ckpt["step"])
        hashes = ckpt.get("hashes")
        prov = report.get("provenance")
        if not isinstance(hashes, dict) or not isinstance(prov, dict):
            raise ValueError(
                "D1c formal selection refuses missing checkpoint.hashes or "
                f"provenance for step={step}"
            )
        for hash_key, prov_key in pairs:
            claimed = str(hashes.get(hash_key) or "").strip()
            attested = str(prov.get(prov_key) or "").strip()
            if not attested or attested == ("0" * 64):
                raise ValueError(
                    f"D1c formal selection refuses missing provenance.{prov_key} "
                    f"(step={step})"
                )
            if claimed != attested:
                raise ValueError(
                    "D1c formal selection refuses checkpoint.hashes."
                    f"{hash_key} != provenance.{prov_key} "
                    f"(step={step}; {claimed} != {attested})"
                )


def _require_distinct_checkpoint_identities(reports: list[dict[str, Any]]) -> None:
    """Plan §8.4: one distinct checkpoint hash/step match per formal report.

    Fail-closed rules:
    * five ``checkpoint.sha256`` values must be pairwise distinct;
    * basename must be ``candidate_step_<step:04d>.pt`` for that report's step;
    * ``checkpoint.path`` must be readable and its live sha256 must equal the
      claimed ``checkpoint.sha256``;
    * all five candidates must share one sealed run identity via
      ``checkpoint.hashes`` (eval_contract / recipe / recipe_source_manifest /
      run_manifest);
    * ``checkpoint.hashes`` must agree with ``provenance.*_sha256``;
    * five model ``state_dict`` digests must be pairwise distinct (forged
      payload step/hashes on clones of one weight state are refused).
    """
    sha_list = [str(r["checkpoint"]["sha256"]) for r in reports]
    if len(set(sha_list)) != len(sha_list):
        raise ValueError(
            "D1c formal selection refuses duplicate checkpoint.sha256 across "
            f"candidates (distinct={len(set(sha_list))}, n={len(sha_list)}); "
            "plan §8.4 requires one checkpoint hash/step match per report"
        )
    run_identity_keys = (
        "eval_contract",
        "recipe",
        "recipe_source_manifest",
        "run_manifest",
    )
    run_fingerprints: list[str] = []
    resolved_paths: list[str] = []
    weight_digests: list[str] = []
    for report in reports:
        ckpt = report["checkpoint"]
        step = int(ckpt["step"])
        ckpt_path = expand(ckpt["path"])
        expected_name = f"candidate_step_{step:04d}.pt"
        if ckpt_path.name.lower() != expected_name.lower():
            raise ValueError(
                f"checkpoint path basename {ckpt_path.name!r} does not match "
                f"expected {expected_name!r} for step={step}"
            )
        hashes = ckpt.get("hashes")
        if not isinstance(hashes, dict):
            raise ValueError(
                "D1c formal selection refuses checkpoint without hashes "
                f"(step={step}); payload['hashes'] must bind the candidate "
                "to one sealed training run"
            )
        identity: dict[str, str] = {}
        for key in run_identity_keys:
            value = hashes.get(key)
            if value is None or not str(value).strip() or str(value) == ("0" * 64):
                raise ValueError(
                    f"D1c formal selection refuses missing checkpoint.hashes.{key} "
                    f"(step={step})"
                )
            identity[key] = str(value)
        run_fingerprints.append(json.dumps(identity, sort_keys=True))
        if not ckpt_path.is_file():
            raise ValueError(
                f"checkpoint.path not readable for hash verification: {ckpt_path}"
            )
        live = _sha256_file(ckpt_path)
        claimed = str(ckpt["sha256"])
        if live != claimed:
            raise ValueError(
                f"checkpoint sha256 mismatch: claimed={claimed} live={live} "
                f"path={ckpt_path}"
            )
        resolved_paths.append(str(ckpt_path.resolve()))
        weight_digests.append(_model_state_sha256(ckpt_path))
    if len(set(run_fingerprints)) != 1:
        raise ValueError(
            "D1c formal selection refuses mixed checkpoint.hashes across "
            "candidates; all five must share one sealed run identity "
            f"(distinct_run_ids={len(set(run_fingerprints))}, n={len(run_fingerprints)})"
        )
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError(
            "D1c formal selection refuses duplicate checkpoint.path across "
            "candidates; plan §8.4 requires one checkpoint per report"
        )
    _require_checkpoint_hashes_match_provenance(reports)
    if len(set(weight_digests)) != len(weight_digests):
        raise ValueError(
            "D1c formal selection refuses identical model weights across "
            "candidates "
            f"(distinct_weight_digests={len(set(weight_digests))}, "
            f"n={len(weight_digests)}); forged payload step/hashes on clones "
            "of one weight state are not distinct training identities"
        )


def select_d1c_formal_reports(report_paths: list[str | Path]) -> dict[str, Any]:
    """Select only among complete evaluator-produced D1c v2 reports."""
    if len(report_paths) != len(_D1C_STEPS):
        raise ValueError("D1c formal selection requires exactly five evaluator report paths")
    reports = []
    for path in report_paths:
        resolved = expand(path)
        report = validate_formal_report(_load_json(resolved))
        report["report_path"] = str(resolved)
        reports.append(report)
    steps = [int(r["checkpoint"]["step"]) for r in reports]
    if sorted(steps) != list(_D1C_STEPS) or len(set(steps)) != len(steps):
        raise ValueError(f"D1c candidate steps must be exactly {list(_D1C_STEPS)}")
    _require_distinct_checkpoint_identities(reports)
    common_fields = (
        ("provenance", "eval_contract_sha256"),
        ("provenance", "recipe_sha256"),
        ("provenance", "asr_sha256"),
        ("infer",),
        ("row_binding", "ordered_utt_ids_sha256"),
    )
    for path in common_fields:
        values = []
        for report in reports:
            value: Any = report
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            values.append(json.dumps(value, sort_keys=True, ensure_ascii=False))
        if len(set(values)) != 1:
            raise ValueError(f"D1c reports have mixed provenance: {'.'.join(path)}")
    candidates: list[dict[str, Any]] = []
    for report in reports:
        tf = report["teacher_forced_metrics"]
        model = report["conditions"]["model"]
        rows = list(model.get("rows") or [])
        copy_rows = list((report["conditions"].get("copy_synth") or {}).get("rows") or [])
        reasons = [
            str((r.get("generation") or {}).get("termination_reason", "")) for r in rows
        ]
        # Recompute ranking metrics from bound rows (do not trust summaries alone).
        cers = [float(r["cer"]) for r in rows if r.get("cer") is not None]
        macro_cer = sum(cers) / len(cers) if cers else float("inf")
        copy_cers = [float(r["cer"]) for r in copy_rows if r.get("cer") is not None]
        copy_macro = sum(copy_cers) / len(copy_cers) if copy_cers else None
        n_cat = sum(1 for r in rows if bool(r.get("catastrophic")))
        # Duration errors: prefer per-row abs log duration when present.
        dur_errs = []
        dur_ratios = []
        for r in rows:
            n_frames = int(r.get("n_frames") or (r.get("generation") or {}).get("n_decoded_frames") or 0)
            gt = int(r.get("gt_frames") or 0)
            if gt > 0:
                ratio = float(n_frames) / float(gt)
                dur_ratios.append(ratio)
            if r.get("abs_log_duration_error") is not None:
                dur_errs.append(float(r["abs_log_duration_error"]))
            elif gt > 0 and n_frames > 0:
                dur_errs.append(abs(math.log(n_frames / gt)))
        max_abs = max(dur_errs) if dur_errs else float("inf")
        mean_abs = sum(dur_errs) / len(dur_errs) if dur_errs else float("inf")
        mean_ratio = (sum(dur_ratios) / len(dur_ratios)) if dur_ratios else float("inf")
        early_suppressed = sum(
            int((r.get("generation") or {}).get("early_eos_suppressed_count") or 0)
            for r in rows
        )
        max_cb0_run = max(
            (
                int((r.get("generation") or {}).get("max_cb0_run") or 0)
                for r in rows
            ),
            default=0,
        )
        empty_gen = any(
            int(r.get("n_frames") or (r.get("generation") or {}).get("n_decoded_frames") or 0)
            <= 0
            for r in rows
        )
        per_row_dur_ok = bool(dur_ratios) and all(0.5 <= x <= 2.0 for x in dur_ratios)
        mean_dur_ok = bool(dur_ratios) and 0.90 <= mean_ratio <= 1.10
        cer_vs_copy_ok = (
            copy_macro is None or (math.isfinite(macro_cer) and macro_cer <= copy_macro + 0.03)
        )
        runaway = bool((report.get("gates") or {}).get("runaway_or_repetition"))
        if not runaway:
            runaway = bool((report.get("summaries") or {}).get("runaway_or_repetition"))
        phrase_loop = bool((report.get("summaries") or {}).get("phrase_loop_repetition"))
        text_usage = (report.get("gates") or {}).get("text_usage") or {}
        if "pass" in text_usage:
            text_usage_ok = bool(text_usage.get("pass"))
        else:
            # §11: absent text_usage is not a vacuous pass on the formal path.
            text_usage_ok = False
        eos_acc = tf.get("tf_eos_final_top1_accuracy")
        eos_fp = tf.get("tf_eos_early_top1_false_positive_count")
        eos_tf_ok = (
            isinstance(eos_acc, (int, float))
            and float(eos_acc) == 1.0
            and eos_fp is not None
            and str(eos_fp) != "unavailable"
            and int(eos_fp) == 0
        )
        summary = report.get("summaries") or {}
        row_eligible = (
            bool(rows)
            and all(r == "learned_eos" for r in reasons)
            and n_cat == 0
            and all(math.isfinite(c) for c in cers)
            and early_suppressed == 0
            and max_cb0_run <= 12
            and not empty_gen
            and per_row_dur_ok
            and mean_dur_ok
            and cer_vs_copy_ok
            and not runaway
            and not phrase_loop
        ) if rows else False
        eligibility = (
            report["formal_eligible"] is True
            and tf.get("teacher_forced_ce_source") == "eval_computed"
            and row_eligible
            and eos_tf_ok
            and text_usage_ok
        )
        candidates.append(
            {
                "report_path": str(report.get("report_path") or ""),
                "ckpt": report["checkpoint"]["path"],
                "checkpoint_sha256": report["checkpoint"]["sha256"],
                "step": report["checkpoint"]["step"],
                "gate_eligible": eligibility,
                "macro_cer": macro_cer
                if math.isfinite(macro_cer)
                else summary.get("model_macro_cer", float("inf")),
                "n_catastrophic": n_cat,
                "max_abs_log_duration_error": max_abs
                if math.isfinite(max_abs)
                else summary.get("max_abs_log_duration_error", float("inf")),
                "mean_abs_log_duration_error": mean_abs
                if math.isfinite(mean_abs)
                else summary.get("mean_abs_log_duration_error", float("inf")),
                "teacher_forced_ce": tf["total_ce"],
            }
        )
    eligible = [c for c in candidates if c["gate_eligible"]]
    if not eligible:
        raise ValueError("D1c formal selection found no eligible candidate; refusing least-bad selection")
    selected = min(
        eligible,
        key=lambda c: (
            float(c["macro_cer"]),
            int(c["n_catastrophic"]),
            float(c["max_abs_log_duration_error"]),
            float(c["mean_abs_log_duration_error"]),
            float(c["teacher_forced_ce"]),
            int(c["step"]),
            str(c["checkpoint_sha256"]),
        ),
    )
    return {
        "schema": "minispeech.discrete_checkpoint_selection",
        "schema_version": "2.0.0",
        "mode": "d1c_formal",
        "candidate_steps": list(_D1C_STEPS),
        "selected": selected,
        "candidates": candidates,
    }


def _normalize_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept either a flat candidate row or a full eval report wrapper."""
    if "macro_cer" in raw or "mean_cer" in raw:
        # Flat candidate (possibly with nested model report)
        out = dict(raw)
        if "model" in raw and isinstance(raw["model"], dict) and "macro_cer" not in raw:
            return candidate_from_eval_report(
                raw["model"],
                ckpt=str(raw.get("ckpt") or raw.get("checkpoint") or ""),
                step=raw.get("step"),
                teacher_forced_ce=raw.get("teacher_forced_ce", raw.get("tf_ce")),
                selection_subset=raw.get("selection_subset"),
            )
        out.setdefault("ckpt", out.get("checkpoint"))
        return out
    if "reports" in raw and isinstance(raw["reports"], dict):
        model = raw["reports"].get("model") or {}
        return candidate_from_eval_report(
            model,
            ckpt=str(raw.get("ckpt") or ""),
            step=raw.get("step"),
            teacher_forced_ce=raw.get("teacher_forced_ce", raw.get("tf_ce")),
            selection_subset=raw.get("selection_subset"),
        )
    if "rows" in raw:
        return candidate_from_eval_report(
            raw,
            ckpt=str(raw.get("ckpt") or ""),
            step=raw.get("step"),
            teacher_forced_ce=raw.get("teacher_forced_ce", raw.get("tf_ce")),
            selection_subset=raw.get("selection_subset"),
        )
    raise ValueError(
        "candidate JSON must contain macro_cer/mean_cer, reports.model, or rows"
    )


def _tf_ce_from_ckpt(ckpt_path: str | Path | None) -> float | None:
    if not ckpt_path:
        return None
    path = expand(ckpt_path)
    if not path.is_file():
        return None
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    if isinstance(metrics, dict):
        return teacher_forced_ce_from_ckpt_metrics(metrics)
    return None


def _ensure_teacher_forced_ce(candidates: list[dict[str, Any]]) -> None:
    """Fill TF CE from ckpt metrics when missing (NON-CANONICAL escape only)."""
    for c in candidates:
        if parse_teacher_forced_ce(c) is not None:
            continue
        filled = _tf_ce_from_ckpt(c.get("ckpt") or c.get("checkpoint"))
        if filled is not None:
            c["teacher_forced_ce"] = filled
            c["teacher_forced_ce_source"] = "ckpt_metrics"


_NON_EVAL_TF_CE_SOURCES = frozenset({"ckpt_metrics", "ckpt", "train_metrics", "train_log"})
_EVAL_TF_CE_SOURCES = frozenset(
    {
        "eval",
        "eval_computed",
        "score",
        "score_only",
        "teacher_forced_eval",
        "n2_eval",
    }
)


def require_eval_computed_tf_ce(candidates: list[dict[str, Any]]) -> None:
    """Canonical D1b: finite TF CE needs explicit eval-source attestation.

    Unproven CE (missing source) and ckpt/train fills are refused.
    """
    bad: list[str] = []
    for i, c in enumerate(candidates):
        label = str(c.get("ckpt") or c.get("step") or f"idx={i}")
        if parse_teacher_forced_ce(c) is None:
            bad.append(f"{label}: missing/null teacher_forced_ce")
            continue
        src = c.get("teacher_forced_ce_source")
        if src is None or str(src).strip() == "":
            bad.append(
                f"{label}: teacher_forced_ce_source missing "
                "(explicit eval attestation required; refuse unproven CE)"
            )
            continue
        src_s = str(src)
        if src_s in _NON_EVAL_TF_CE_SOURCES:
            bad.append(f"{label}: teacher_forced_ce_source={src_s!r} (not eval-computed)")
            continue
        if src_s not in _EVAL_TF_CE_SOURCES:
            bad.append(
                f"{label}: teacher_forced_ce_source={src_s!r} not in "
                f"allowed eval sources {sorted(_EVAL_TF_CE_SOURCES)}"
            )
    if bad:
        raise SystemExit(
            "canonical D1b selection requires eval-computed finite teacher_forced_ce "
            "with explicit teacher_forced_ce_source attestation; "
            "refuse ckpt-metrics / train-log / unproven fills. Offenders:\n- "
            + "\n- ".join(bad)
        )


def _candidate_contract_sha(c: dict[str, Any]) -> str | None:
    got = c.get("eval_contract_sha256")
    if got:
        return str(got)
    prov = c.get("provenance")
    if isinstance(prov, dict) and prov.get("eval_contract_sha256"):
        return str(prov["eval_contract_sha256"])
    return None


def require_candidate_contract_provenance(
    candidates: list[dict[str, Any]],
    *,
    expected_contract_sha256: str,
) -> None:
    """Fail-closed: every candidate must attest the frozen eval_contract hash."""
    bad: list[str] = []
    for i, c in enumerate(candidates):
        label = str(c.get("ckpt") or c.get("step") or f"idx={i}")
        got = _candidate_contract_sha(c)
        if got is None:
            bad.append(f"{label}: missing eval_contract_sha256 provenance")
        elif got != expected_contract_sha256:
            bad.append(
                f"{label}: eval_contract_sha256={got} != expected={expected_contract_sha256}"
            )
    if bad:
        raise SystemExit(
            "canonical D1b selection requires verified eval_contract provenance "
            "on every candidate. Offenders:\n- " + "\n- ".join(bad)
        )


def _candidate_items_sha256(c: dict[str, Any]) -> str | None:
    got = c.get("items_sha256") or c.get("subset_items_sha256")
    if got is not None:
        return str(got)
    if c.get("utt_ids"):
        digest = items_sha256([str(u) for u in c["utt_ids"]])
        c["items_sha256"] = digest
        return digest
    return None


def _load_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if args.candidates_json:
        payload = _load_json(expand(args.candidates_json))
        if isinstance(payload, list):
            for item in payload:
                candidates.append(_normalize_candidate(dict(item)))
        elif isinstance(payload, dict) and "candidates" in payload:
            for item in payload["candidates"]:
                candidates.append(_normalize_candidate(dict(item)))
        else:
            raise SystemExit(
                "--candidates-json must be a list or {candidates: [...]} object"
            )
    for path in args.scored or []:
        raw = _load_json(expand(path))
        if not isinstance(raw, dict):
            raise SystemExit(f"scored file must be an object: {path}")
        if "ckpt" not in raw and args.ckpt_from_stem:
            raw = dict(raw)
            raw["ckpt"] = str(expand(path).with_suffix(".pt"))
        candidates.append(_normalize_candidate(raw))
    if args.candidates_dir:
        root = expand(args.candidates_dir)
        for path in sorted(root.glob("*.json")):
            if path.name == "checkpoint_selection.json":
                continue
            raw = _load_json(path)
            if not isinstance(raw, dict):
                continue
            candidates.append(_normalize_candidate(raw))
    if not candidates:
        raise SystemExit("No candidates loaded; pass --candidates-json, --scored, or --candidates-dir")
    if args.selection_subset:
        for c in candidates:
            c["selection_subset"] = args.selection_subset
    # Canonical path must not invent TF CE from checkpoint metrics.
    if args.allow_missing_tf_ce:
        _ensure_teacher_forced_ce(candidates)
    return candidates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--candidates-json",
        default=None,
        help="JSON list (or {candidates:[...]}) of scored candidate rows",
    )
    ap.add_argument(
        "--scored",
        nargs="*",
        default=None,
        help="One or more scored eval JSON paths (model report or flat candidate)",
    )
    ap.add_argument(
        "--candidates-dir",
        default=None,
        help="Directory of per-candidate scored JSON files",
    )
    ap.add_argument(
        "--selection-subset",
        default=None,
        help="Label written into checkpoint_selection.json (dev32 or select8)",
    )
    ap.add_argument(
        "--require-subset",
        choices=("dev32", "select8"),
        default=None,
        help="Fail if selection_subset mismatches (CONFIRM must be select8)",
    )
    ap.add_argument(
        "--require-items-sha256",
        default=None,
        help="Fail-closed: actual utterance-set items_sha256 (CONFIRM select8)",
    )
    ap.add_argument(
        "--verify-frozen-subset",
        default=None,
        help="Alias/path to load + verify items_sha256 (e.g. select8)",
    )
    ap.add_argument(
        "--subset-split",
        default=None,
        help="Split when --verify-frozen-subset points at confirm32 JSON",
    )
    ap.add_argument(
        "--out",
        default="checkpoint_selection.json",
        help="Output path for checkpoint_selection.json",
    )
    ap.add_argument(
        "--promote-best",
        action="store_true",
        help="After writing selection, copy/symlink selected ckpt → <run-dir>/best.pt",
    )
    ap.add_argument(
        "--best-path",
        default=None,
        help="Explicit best.pt destination (default: selected ckpt parent / best.pt)",
    )
    ap.add_argument(
        "--ckpt-from-stem",
        action="store_true",
        help="When using --scored, invent ckpt path as <json>.pt if missing",
    )
    ap.add_argument(
        "--print-schedule",
        action="store_true",
        help="Print required immutable candidate steps and exit",
    )
    ap.add_argument(
        "--include-dev-extension",
        action="store_true",
        help="With --print-schedule, include 1000/1250 DEV extension steps",
    )
    ap.add_argument(
        "--eval-contract",
        default=DEFAULT_EVAL_CONTRACT,
        help="Frozen eval_contract.json (hash stamped into selection JSON)",
    )
    ap.add_argument(
        "--allow-missing-tf-ce",
        action="store_true",
        help="NON-CANONICAL: allow null TF CE (Infinity tie-break). Forbidden for D1b.",
    )
    ap.add_argument(
        "--d1c-formal",
        action="store_true",
        help="Use v2 evaluator-report-only D1c selection; historical inputs are refused.",
    )
    ap.add_argument(
        "--formal-reports",
        nargs="*",
        default=None,
        help="Exactly five evaluator-produced discrete_eval_report v2 JSON paths.",
    )
    args = ap.parse_args()

    if args.print_schedule:
        steps = fr_candidate_steps(include_dev_extension=bool(args.include_dev_extension))
        print(json.dumps({"fr_candidate_steps": list(steps)}, indent=2))
        return 0

    if args.d1c_formal:
        if not args.formal_reports or args.candidates_json or args.scored or args.candidates_dir:
            raise SystemExit(
                "D1c formal mode accepts only --formal-reports; flat/manual candidate inputs are refused"
            )
        try:
            result = select_d1c_formal_reports(args.formal_reports)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        out_path = expand(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.promote_best:
            src = Path(str(result["selected"]["ckpt"]))
            dest = Path(args.best_path) if args.best_path else src.parent / "best.pt"
            if not src.is_file() or src.name == "best.pt":
                raise SystemExit(f"invalid immutable D1c candidate checkpoint: {src}")
            expected = str(result["selected"]["checkpoint_sha256"])
            if _sha256_file(src) != expected:
                raise SystemExit("candidate checkpoint SHA differs from evaluator report")
            shutil.copy2(src, dest)
            if _sha256_file(dest) != expected:
                raise SystemExit("promoted best.pt SHA verification failed")
            result["promotion"] = {"path": str(dest), "sha256": expected, "verified": True}
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"wrote": str(out_path), "selected": result["selected"]}, indent=2))
        return 0

    candidates = _load_candidates(args)
    subset = args.selection_subset or candidates[0].get("selection_subset")
    expected_sha = args.require_items_sha256
    if args.verify_frozen_subset:
        ids, _, digest = load_frozen_utt_ids(
            args.verify_frozen_subset,
            split=args.subset_split,
            verify_hash=True,
        )
        expected_sha = digest
        for c in candidates:
            c.setdefault("selection_subset", subset or args.verify_frozen_subset)
            got = _candidate_items_sha256(c)
            if got is None:
                raise SystemExit(
                    f"candidate missing items_sha256 for {c.get('ckpt')}: "
                    "refuse to stamp from verified frozen subset "
                    "(record hash at eval / pass utt_ids)"
                )
            if got != expected_sha:
                raise SystemExit(
                    f"candidate utterance-set hash mismatch for {c.get('ckpt')}: "
                    f"got={got} expected={expected_sha} (n={len(ids)})"
                )
    if args.require_subset:
        if subset != args.require_subset:
            raise SystemExit(
                f"selection_subset={subset!r} but --require-subset={args.require_subset!r}"
            )
        if args.require_subset == "select8":
            for c in candidates:
                if c.get("selection_subset") not in (None, "select8"):
                    raise SystemExit(
                        "CONFIRM selection must use select8 only; "
                        f"got selection_subset={c.get('selection_subset')!r} for {c.get('ckpt')}"
                    )
            # Always load + verify the frozen select8 digest (fail-closed).
            _, _, select8_sha = load_frozen_utt_ids("select8", verify_hash=True)
            if expected_sha is None:
                expected_sha = select8_sha
            elif expected_sha != select8_sha:
                raise SystemExit(
                    "CONFIRM select8 hash mismatch vs frozen oracle: "
                    f"require_items_sha256/verify={expected_sha} frozen={select8_sha}"
                )
            for c in candidates:
                got = _candidate_items_sha256(c)
                if got is None:
                    raise SystemExit(
                        "CONFIRM select8 requires items_sha256 on every candidate; "
                        f"missing for {c.get('ckpt')} (refuse to stamp frozen hash)"
                    )
                if got != expected_sha:
                    raise SystemExit(
                        "CONFIRM select8 items_sha256 mismatch: "
                        f"got={got} expected={expected_sha} for {c.get('ckpt')}"
                    )

    contract = load_eval_contract(args.eval_contract)
    contract_sha = str(
        contract.get("_content_sha256")
        or contract.get("sha256")
        or eval_contract_content_sha256(contract)
    )

    if not args.allow_missing_tf_ce:
        require_eval_computed_tf_ce(candidates)
        require_candidate_contract_provenance(
            candidates, expected_contract_sha256=contract_sha
        )

    try:
        result = select_fr_checkpoint(
            candidates,
            selection_subset=subset,
            require_tf_ce=not bool(args.allow_missing_tf_ce),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if expected_sha is not None:
        result["items_sha256"] = expected_sha
        result["selection_subset_verified"] = True
    result["provenance"] = {
        "git_sha": git_sha(),
        "eval_contract_path": str(contract.get("_path") or args.eval_contract),
        "eval_contract_sha256": contract_sha,
        "selection_subset": subset,
        "items_sha256": expected_sha,
        "require_teacher_forced_ce": not bool(args.allow_missing_tf_ce),
        "require_eval_computed_tf_ce": not bool(args.allow_missing_tf_ce),
        "require_candidate_contract_provenance": not bool(args.allow_missing_tf_ce),
    }
    out_path = expand(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "selected": result["selected"]}, indent=2))

    if args.promote_best:
        import os

        sel = result["selected"]
        # Do NOT use expand()/Path.resolve() here — resolve() follows symlinks, so
        # expand(best.pt) can equal the candidate file and unlink() destroys it.
        def _abspath(p: str | Path) -> Path:
            return Path(os.path.expanduser(str(p))).absolute()

        src = _abspath(str(sel["ckpt"]))
        if src.is_symlink() or not src.is_file():
            raise SystemExit(
                f"selected ckpt missing or is a symlink (refusing promote): {src}"
            )
        if src.name == "best.pt":
            raise SystemExit(
                "refusing to promote: selected path is already named best.pt "
                "(candidates must use immutable step names until after selection)"
            )
        dest = _abspath(args.best_path) if args.best_path else (src.parent / "best.pt")
        if dest == src:
            raise SystemExit(f"refusing to promote: dest == src ({dest})")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        shutil.copy2(src, dest)
        print(
            json.dumps(
                {"promoted_best": str(dest), "mode": "copy", "from": str(src)},
                indent=2,
            )
        )
        print("NOTE: run mandatory listen-demo after best.pt promotion (checkpoint-listen).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
