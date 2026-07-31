#!/usr/bin/env python3
"""Build / score blind preference ballots (Phase M / M08)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minispeech.eval.preference import ballot_from_pairs, build_pairs, score_responses
from minispeech.utils import expand


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir-a", required=True, help="Directory with wavs for system A")
    ap.add_argument("--dir-b", required=True, help="Directory with wavs for system B")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--out", default="out/preference_ballot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-traps", type=int, default=3)
    ap.add_argument("--score", default=None, help="Optional responses JSON to score")
    args = ap.parse_args()

    da = expand(args.dir_a)
    db = expand(args.dir_b)
    files_a = sorted(da.glob("*.wav"))
    files_b = sorted(db.glob("*.wav"))
    pairs, key = build_pairs(
        files_a,
        files_b,
        seed=args.seed,
        n_traps=args.n_traps,
        system_a_name=args.name_a,
        system_b_name=args.name_b,
    )
    out = expand(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ballot = ballot_from_pairs(pairs)
    (out / "ballot.json").write_text(json.dumps(ballot, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    # Simple HTML listener
    rows = []
    for b in ballot:
        rows.append(
            f"<tr><td>{b['pair_id']}</td>"
            f"<td><audio controls src='file:///{Path(b['left']).as_posix()}'></audio></td>"
            f"<td><audio controls src='file:///{Path(b['right']).as_posix()}'></audio></td>"
            f"<td>left / right / same</td></tr>"
        )
    html = (
        "<html><body><h1>Preference ballot</h1><table border=1>"
        "<tr><th>id</th><th>left</th><th>right</th><th>choice</th></tr>"
        + "".join(rows)
        + "</table></body></html>"
    )
    (out / "ballot.html").write_text(html, encoding="utf-8")
    print(json.dumps({"n_pairs": len(pairs), "out": str(out)}, ensure_ascii=False))

    if args.score:
        responses = json.loads(Path(expand(args.score)).read_text(encoding="utf-8"))
        if isinstance(responses, dict) and "responses" in responses:
            responses = responses["responses"]
        result = score_responses(responses, key)
        (out / "score.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
