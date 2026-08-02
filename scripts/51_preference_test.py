#!/usr/bin/env python3
"""Build / score blind preference ballots (Phase M / M08 / S4).

Supports same-name matching or order matching (for demo dirs with different
filenames). Self-vs-self traps included. Does not invent listener scores.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Sequence

from minispeech.eval.preference import ballot_from_pairs, build_pairs, score_responses
from minispeech.utils import expand


def _collect_wavs(root: Path, *, recursive: bool) -> list[Path]:
    if recursive:
        files = sorted(root.rglob("*.wav"))
    else:
        files = sorted(root.glob("*.wav"))
    return [p for p in files if p.is_file()]


def _stage_matched(
    files_a: list[Path],
    files_b: list[Path],
    staging: Path,
    *,
    match_mode: str,
) -> tuple[list[Path], list[Path], list[dict]]:
    """Return same-basename lists under staging/a and staging/b."""
    if match_mode == "name":
        by_b = {p.name: p for p in files_b}
        matched = [(pa, by_b[pa.name]) for pa in files_a if pa.name in by_b]
    elif match_mode == "order":
        n = min(len(files_a), len(files_b))
        matched = list(zip(files_a[:n], files_b[:n]))
    else:
        raise ValueError(f"unknown match_mode: {match_mode}")

    sa = staging / "a"
    sb = staging / "b"
    if sa.exists():
        shutil.rmtree(sa)
    if sb.exists():
        shutil.rmtree(sb)
    sa.mkdir(parents=True)
    sb.mkdir(parents=True)
    out_a: list[Path] = []
    out_b: list[Path] = []
    mapping: list[dict] = []
    for i, (pa, pb) in enumerate(matched):
        name = f"utt_{i:03d}.wav"
        da = sa / name
        db = sb / name
        shutil.copy2(pa, da)
        shutil.copy2(pb, db)
        out_a.append(da)
        out_b.append(db)
        mapping.append(
            {
                "utt": name,
                "source_a": str(pa),
                "source_b": str(pb),
            }
        )
    return out_a, out_b, mapping


def _write_html(ballot: list[dict], out: Path) -> None:
    rows = []
    for b in ballot:
        rows.append(
            f"<tr><td>{b['pair_id']}</td>"
            f"<td><audio controls src='file:///{Path(b['left']).as_posix()}'></audio></td>"
            f"<td><audio controls src='file:///{Path(b['right']).as_posix()}'></audio></td>"
            f"<td>left / right / same</td></tr>"
        )
    html = (
        "<html><body><h1>Preference ballot</h1>"
        "<p>Choose which is more natural. Use <b>same</b> when no difference.</p>"
        "<table border=1>"
        "<tr><th>id</th><th>left</th><th>right</th><th>choice</th></tr>"
        + "".join(rows)
        + "</table></body></html>"
    )
    (out / "ballot.html").write_text(html, encoding="utf-8")


def _build_or_score(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir-a", required=True, help="Directory with wavs for system A")
    ap.add_argument("--dir-b", required=True, help="Directory with wavs for system B")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--out", default="out/preference_ballot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-traps", type=int, default=3)
    ap.add_argument(
        "--match-mode",
        choices=("name", "order"),
        default="name",
        help="name: match basenames; order: pair sorted lists by index (demo dirs)",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Collect wavs under subdirectories",
    )
    ap.add_argument("--score", default=None, help="Optional responses JSON to score")
    ap.add_argument(
        "--status",
        default=None,
        help="Write human-study status JSON (e.g. blocked_no_listeners)",
    )
    args = ap.parse_args(argv)

    da = expand(args.dir_a)
    db = expand(args.dir_b)
    files_a = _collect_wavs(da, recursive=args.recursive)
    files_b = _collect_wavs(db, recursive=args.recursive)
    out = expand(args.out)
    out.mkdir(parents=True, exist_ok=True)

    staged_a, staged_b, mapping = _stage_matched(
        files_a,
        files_b,
        out / "_staging",
        match_mode=args.match_mode,
    )
    (out / "pairing.json").write_text(
        json.dumps(
            {
                "match_mode": args.match_mode,
                "n_matched": len(mapping),
                "dir_a": str(da),
                "dir_b": str(db),
                "pairs": mapping,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pairs, key = build_pairs(
        staged_a,
        staged_b,
        seed=args.seed,
        n_traps=min(args.n_traps, len(staged_a)),
        system_a_name=args.name_a,
        system_b_name=args.name_b,
    )
    ballot = ballot_from_pairs(pairs)
    (out / "ballot.json").write_text(
        json.dumps(ballot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(ballot, out)

    n_content = sum(1 for p in pairs if not p.trap)
    n_traps = sum(1 for p in pairs if p.trap)
    summary = {
        "n_pairs_total": len(pairs),
        "n_content_comparisons": n_content,
        "n_traps": n_traps,
        "match_mode": args.match_mode,
        "system_a": args.name_a,
        "system_b": args.name_b,
        "out": str(out),
        "mos_status": "unmeasured",
        "listener_count": None,
        "human_scores": None,
    }
    if args.status:
        summary["study_status"] = args.status
        if args.status.startswith("blocked"):
            summary["listener_count"] = 0
            summary["note"] = (
                "Human listeners unavailable; preference study BLOCKED. "
                "No MOS / preference rates invented. Ballot ready for future panel."
            )
    (out / "status.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))

    if args.score:
        responses = json.loads(Path(expand(args.score)).read_text(encoding="utf-8"))
        if isinstance(responses, dict) and "responses" in responses:
            responses = responses["responses"]
        result = score_responses(responses, key)
        (out / "score.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False))
    return 0


def _self_check(argv: Sequence[str] | None = None) -> int:
    """Exercise trap and chance-level scoring without listener responses."""
    ap = argparse.ArgumentParser(
        prog="51_preference_test.py self-check",
        description="Score deterministic trap and coin-flip synthetic responses.",
    )
    ap.add_argument("--key", required=True, help="Ballot key.json")
    ap.add_argument("--out", default=None, help="Optional JSON report path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    key = json.loads(expand(args.key).read_text(encoding="utf-8"))
    pairs = key.get("pairs") or {}
    auto_left = [
        {
            "pair_id": pair_id,
            "choice": "same" if meta.get("trap") else "left",
        }
        for pair_id, meta in pairs.items()
    ]

    rng = random.Random(args.seed)
    coin_flip = []
    for pair_id, meta in pairs.items():
        if meta.get("trap"):
            coin_flip.append({"pair_id": pair_id, "choice": "same"})
            continue
        prefer_a = rng.random() < 0.5
        left_is_a = bool(meta["left_is_a"])
        choice = (
            ("left" if left_is_a else "right")
            if prefer_a
            else ("right" if left_is_a else "left")
        )
        coin_flip.append({"pair_id": pair_id, "choice": choice})

    payload = {
        "auto_left": score_responses(auto_left, key),
        "coin_flip": score_responses(coin_flip, key),
        "seed": args.seed,
        "key": str(expand(args.key)),
        "note": "Synthetic dry-run only — not human listener data.",
    }
    if args.out:
        out = expand(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _protocol(argv: Sequence[str] | None = None) -> int:
    """Write S4 preference protocol status (systems, traps, MOS, blockers)."""
    ap = argparse.ArgumentParser(prog="51_preference_test.py protocol")
    ap.add_argument("--out", default="out/paper1_figures/preference_protocol.json")
    ap.add_argument("--n2-gate", default="out/n2_gate.json")
    args = ap.parse_args(argv)

    n2 = None
    n2_path = expand(args.n2_gate)
    if n2_path.is_file():
        n2 = json.loads(n2_path.read_text(encoding="utf-8"))
    n2_fail = bool(isinstance(n2, dict) and str(n2.get("verdict", "")).upper() == "FAIL")

    systems = {
        "silero_frontend": {
            "wav_dir": "out/n2_first_execution/silero_frontend/demo5/",
            "role": "system_compare (shared frontend)",
            "include": True,
        },
        "track_a": {
            "wav_dir": "out/n2_first_execution/track_a/demo5/",
            "role": "system_compare",
            "include": True,
        },
        "n2_s1": {
            "wav_dir": "out/n2_first_execution/n2_s1_listen_demo/",
            "role": "N2 pilot listen-demo",
            "include": False,
            "reason": (
                "N2 gate FAIL (out/n2_gate.json); prefer Silero+frontend vs Track A only. "
                "N2 wavs exist for qualitative listen but are excluded from preference panel."
            ),
        },
    }
    if not n2_fail:
        systems["n2_s1"]["include"] = True
        systems["n2_s1"]["reason"] = "N2 gate not FAIL — include if wav coverage sufficient"

    comparisons = []
    if systems["silero_frontend"]["include"] and systems["track_a"]["include"]:
        comparisons.append(
            {
                "name": "silero_frontend_vs_track_a",
                "a": "silero_frontend",
                "b": "track_a",
                "match_mode": "order",
                "n_traps_self_vs_self": 3,
            }
        )

    payload = {
        "task": "S4",
        "plan": "n2_first_execution",
        "study_status": "BLOCKED",
        "blocker": "real_human_listeners_unavailable",
        "listener_count": 0,
        "n_comparisons_named": [c["name"] for c in comparisons],
        "comparisons": comparisons,
        "systems": systems,
        "n2_gate_verdict": (n2 or {}).get("verdict"),
        "n2_gate_source": str(n2_path) if n2_path.is_file() else None,
        "trap_design": "self-vs-self (same wav left and right); panel QC = trap_same_rate",
        "mos_status": "unmeasured",
        "mos_note": (
            "Absolute MOS / P.808 not run. Do not substitute UTMOS for human MOS. "
            "Instrument is pairwise preference with binomial CI95 when listeners exist."
        ),
        "dry_run": {
            "ballot_dir": "out/paper1_figures/preference_silero_fe_vs_track_a/",
            "self_check": "out/paper1_figures/preference_silero_fe_vs_track_a/self_check.json",
        },
        "acceptance": {
            "figures_have_source_paths": True,
            "listener_count_and_comparisons_named_or_blocked": True,
            "mos_marked_unmeasured": True,
        },
    }
    out = expand(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "self-check":
        return _self_check(args[1:])
    if args and args[0] == "protocol":
        return _protocol(args[1:])
    return _build_or_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
