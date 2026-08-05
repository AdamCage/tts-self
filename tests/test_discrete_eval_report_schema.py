from __future__ import annotations

import copy

import pytest

from minispeech.eval.discrete_report_schema import CONDITIONS, build_formal_report, validate_formal_report


def _report():
    ids = ["u1", "u2"]
    conditions = {}
    for name in CONDITIONS:
        rows = [
            {
                "ordinal": i, "utt_id": uid, "normalized_reference_sha256": "a" * 64,
                "generated_codes_sha256": "b" * 64, "generation": {"termination_reason": "learned_eos"},
                "wav": {"path": f"{uid}.wav", "file_sha256": "c" * 64, "pcm_sha256": "d" * 64},
                "cer": 0.0,
            }
            for i, uid in enumerate(ids)
        ]
        conditions[name] = {
            "subset_key": "select8", "subset_path": "subset.json", "subset_file_sha256": "e" * 64,
            "expected_count": 2, "evaluated_count": 2, "ordered_utt_ids_sha256": "",
            "rows": rows,
        }
    from minispeech.eval.discrete_report_schema import ordered_utt_ids_sha256
    for value in conditions.values():
        value["ordered_utt_ids_sha256"] = ordered_utt_ids_sha256(ids)
    return build_formal_report(
        formal_eligible=True,
        checkpoint={
            "path": "candidate.pt",
            "sha256": "f" * 64,
            "step": 250,
            "schema": "x",
            "schema_version": "2",
            "hashes": {
                "eval_contract": "e" * 64,
                "recipe": "f" * 64,
                "recipe_source_manifest": "a" * 64,
                "run_manifest": "b" * 64,
            },
        },
        provenance={"eval_contract_sha256": "1" * 64},
        infer={"max_frames": 750},
        teacher_forced_metrics={"teacher_forced_ce_source": "eval_computed", "total_ce": 1.0},
        row_binding={"ordered_utt_ids": ids, "ordered_utt_ids_sha256": ordered_utt_ids_sha256(ids)},
        conditions=conditions,
    )


@pytest.mark.parametrize("mutate", [
    lambda r: r["conditions"]["model"]["rows"].pop(),
    lambda r: r["conditions"]["model"]["rows"].append(copy.deepcopy(r["conditions"]["model"]["rows"][0])),
    lambda r: r["conditions"]["model"]["rows"].reverse(),
])
def test_formal_rows_fail_closed(mutate):
    report = _report()
    mutate(report)
    with pytest.raises(ValueError):
        validate_formal_report(report)
