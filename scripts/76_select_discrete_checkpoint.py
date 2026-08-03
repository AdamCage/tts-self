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
import json
import shutil
from pathlib import Path
from typing import Any

from minispeech.data.d1b_oracles import items_sha256, load_frozen_utt_ids
from minispeech.eval.discrete_n2 import (
    candidate_from_eval_report,
    fr_candidate_steps,
    select_fr_checkpoint,
)
from minispeech.utils import expand


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    args = ap.parse_args()

    if args.print_schedule:
        steps = fr_candidate_steps(include_dev_extension=bool(args.include_dev_extension))
        print(json.dumps({"fr_candidate_steps": list(steps)}, indent=2))
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
        # Also stamp candidates lacking an utterance hash.
        for c in candidates:
            c.setdefault("selection_subset", subset or args.verify_frozen_subset)
            got = c.get("items_sha256") or c.get("subset_items_sha256")
            if got is None and c.get("utt_ids"):
                got = items_sha256([str(u) for u in c["utt_ids"]])
                c["items_sha256"] = got
            if got is not None and got != expected_sha:
                raise SystemExit(
                    f"candidate utterance-set hash mismatch for {c.get('ckpt')}: "
                    f"got={got} expected={expected_sha} (n={len(ids)})"
                )
            if got is None:
                # Accept when scorer recorded the subset label + we verified frozen file.
                c["items_sha256"] = expected_sha
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
            if expected_sha is None:
                # Default: verify against frozen select8 hash.
                _, _, expected_sha = load_frozen_utt_ids("select8", verify_hash=True)
            for c in candidates:
                got = c.get("items_sha256") or c.get("subset_items_sha256")
                if got is not None and got != expected_sha:
                    raise SystemExit(
                        "CONFIRM select8 items_sha256 mismatch: "
                        f"got={got} expected={expected_sha} for {c.get('ckpt')}"
                    )
                c["items_sha256"] = expected_sha

    result = select_fr_checkpoint(candidates, selection_subset=subset)
    if expected_sha is not None:
        result["items_sha256"] = expected_sha
        result["selection_subset_verified"] = True
    out_path = expand(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "selected": result["selected"]}, indent=2))

    if args.promote_best:
        sel = result["selected"]
        src = expand(str(sel["ckpt"]))
        if not src.is_file():
            raise SystemExit(f"selected ckpt missing on disk: {src}")
        if src.name == "best.pt":
            raise SystemExit(
                "refusing to promote: selected path is already named best.pt "
                "(candidates must use immutable step names until after selection)"
            )
        dest = expand(args.best_path) if args.best_path else src.parent / "best.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            dest.symlink_to(src.resolve())
            mode = "symlink"
        except OSError:
            shutil.copy2(src, dest)
            mode = "copy"
        print(json.dumps({"promoted_best": str(dest), "mode": mode, "from": str(src)}, indent=2))
        print("NOTE: run mandatory listen-demo after best.pt promotion (checkpoint-listen).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
