"""Frozen D1b oracle loading + eval-contract wiring."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from minispeech.data.d1b_oracles import (
    TRACKN_V2_SPEAKER_SHA256,
    enforce_eval_contract,
    expand_path_template,
    items_sha256,
    load_eval_contract,
    load_frozen_utt_ids,
    resolve_subset_alias,
)
from minispeech.infer.discrete_config import GENERATE_INFER_KEYS, wave3c_infer_defaults
from minispeech.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]
DEV32 = ROOT / "configs" / "train_subsets" / "d1b_dev32.json"
CONFIRM32 = ROOT / "configs" / "train_subsets" / "d1b_confirm32.json"
EVAL_CONTRACT = ROOT / "configs" / "eval_contracts" / "d1b_n2_reliability.json"
D1B_YAML = ROOT / "configs" / "discrete_speech_lm_d1b_443m.yaml"
SCRIPT_74 = ROOT / "scripts" / "74_eval_discrete_lm.py"
SCRIPT_73 = ROOT / "scripts" / "73_infer_discrete_lm.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not DEV32.is_file(), reason="dev32 missing")
def test_dev32_alias_ordered_and_hashed():
    ids, payload, digest = load_frozen_utt_ids("dev32", verify_hash=True)
    assert len(ids) == 32
    assert ids == [str(x) for x in payload["utt_ids"]]
    assert digest == payload["items_sha256"]
    assert "ba1a4089af87b3ebc90db07d" in ids[22] or any("ba1a4089" in u for u in ids)


@pytest.mark.skipif(not CONFIRM32.is_file(), reason="confirm32 missing")
def test_select8_split_hash():
    ids, payload, digest = load_frozen_utt_ids("select8", verify_hash=True)
    assert len(ids) == 8
    assert digest == payload["split_hashes"]["select8"]
    assert ids == [str(x) for x in payload["splits"]["select8"]]


@pytest.mark.skipif(not EVAL_CONTRACT.is_file() or not D1B_YAML.is_file(), reason="contract/yaml missing")
def test_eval_contract_speaker_wired_and_enforced():
    contract = load_eval_contract(EVAL_CONTRACT)
    assert contract["speaker_map"]["sha256"] == TRACKN_V2_SPEAKER_SHA256
    assert contract["speaker_map"]["status"] == "wired"
    assert contract["infer"]["stop_overrun_bias"] == 0.0
    cfg = load_yaml(D1B_YAML)
    ids, _, digest = load_frozen_utt_ids("dev32", verify_hash=True)
    enforce_eval_contract(
        cfg,
        contract_path=EVAL_CONTRACT,
        subset_items_sha256=digest,
        subset_label="dev32",
        require_clean_sha=False,
    )
    assert len(ids) == 32


def test_expand_run_name_template(tmp_path: Path):
    p = expand_path_template("runs/${run_name}", {"run_name": "discrete_speech_lm_d1b_dev_seed42"})
    assert p.name == "discrete_speech_lm_d1b_dev_seed42"


def test_items_sha256_stable():
    assert items_sha256(["a", "b"]) == items_sha256(["a", "b"])
    assert items_sha256(["a", "b"]) != items_sha256(["b", "a"])


def _full_n2_reports(infer: dict | None = None) -> dict:
    infer = infer if infer is not None else wave3c_infer_defaults()
    base_row = {"utt_id": "u0", "wav": "/nonexistent/u0.wav", "ref": "привет"}
    out = {}
    for name in (
        "model",
        "copy_synth",
        "wrong_utt",
        "random_codes",
        "shuffle_text",
        "mask_text",
    ):
        rep = {"rows": [dict(base_row)]}
        if name == "model":
            rep["infer"] = dict(infer)
        out[name] = rep
    return out


@pytest.mark.skipif(not EVAL_CONTRACT.is_file() or not SCRIPT_74.is_file(), reason="contract/script missing")
def test_score_only_refuses_missing_provenance_and_max_frames():
    mod = _load_script(SCRIPT_74, "eval_discrete_lm_score_only_74")
    contract = load_eval_contract(EVAL_CONTRACT)
    content_sha = str(contract.get("_content_sha256") or contract.get("sha256"))
    good_infer = wave3c_infer_defaults()
    ids, _, digest = load_frozen_utt_ids("select8", verify_hash=True)
    assert len(ids) == 8

    with pytest.raises(SystemExit, match="missing bundle.provenance"):
        mod.assert_score_only_d1b_bundle({"reports": {"model": {"infer": good_infer}}})

    with pytest.raises(SystemExit, match="eval_contract_sha256 missing"):
        mod.assert_score_only_d1b_bundle(
            {
                "provenance": {"max_frames": 750},
                "reports": _full_n2_reports(good_infer),
            },
            eval_contract=EVAL_CONTRACT,
        )

    with pytest.raises(SystemExit, match="subset_label missing"):
        mod.assert_score_only_d1b_bundle(
            {
                "provenance": {
                    "eval_contract_sha256": content_sha,
                    "max_frames": 750,
                    "subset_items_sha256": digest,
                },
                "reports": _full_n2_reports(good_infer),
            },
            eval_contract=EVAL_CONTRACT,
        )

    with pytest.raises(SystemExit, match="subset_items_sha256 missing"):
        mod.assert_score_only_d1b_bundle(
            {
                "provenance": {
                    "eval_contract_sha256": content_sha,
                    "max_frames": 750,
                    "subset_label": "select8",
                },
                "reports": _full_n2_reports(good_infer),
            },
            eval_contract=EVAL_CONTRACT,
        )

    with pytest.raises(SystemExit, match="reports.model.infer missing|incomplete"):
        mod.assert_score_only_d1b_bundle(
            {
                "provenance": {
                    "eval_contract_sha256": content_sha,
                    "max_frames": 750,
                    "subset_label": "select8",
                    "subset_items_sha256": digest,
                },
                "reports": {
                    "model": {"rows": []},
                    "copy_synth": {"rows": []},
                    "wrong_utt": {"rows": []},
                    "random_codes": {"rows": []},
                    "shuffle_text": {"rows": []},
                    "mask_text": {"rows": []},
                },
            },
            eval_contract=EVAL_CONTRACT,
        )


@pytest.mark.skipif(not EVAL_CONTRACT.is_file() or not SCRIPT_74.is_file(), reason="contract/script missing")
def test_score_only_refuses_mismatched_max_frames():
    mod = _load_script(SCRIPT_74, "eval_discrete_lm_score_only_74b")
    contract = load_eval_contract(EVAL_CONTRACT)
    content_sha = str(contract.get("_content_sha256") or contract.get("sha256"))
    _, _, digest = load_frozen_utt_ids("select8", verify_hash=True)
    bad_infer = wave3c_infer_defaults()
    bad_infer["max_frames"] = 271
    with pytest.raises(SystemExit, match="max_frames=271|cfg/live infer"):
        mod.assert_score_only_d1b_bundle(
            {
                "provenance": {
                    "eval_contract_sha256": content_sha,
                    "max_frames": 271,
                    "subset_label": "select8",
                    "subset_items_sha256": digest,
                },
                "reports": _full_n2_reports(bad_infer),
            },
            eval_contract=EVAL_CONTRACT,
        )


@pytest.mark.skipif(not EVAL_CONTRACT.is_file() or not SCRIPT_74.is_file(), reason="contract/script missing")
def test_score_only_accepts_frozen_max_frames_provenance():
    mod = _load_script(SCRIPT_74, "eval_discrete_lm_score_only_74c")
    contract = load_eval_contract(EVAL_CONTRACT)
    content_sha = str(contract.get("_content_sha256") or contract.get("sha256"))
    ids, _, digest = load_frozen_utt_ids("select8", verify_hash=True)
    out = mod.assert_score_only_d1b_bundle(
        {
            "provenance": {
                "eval_contract_sha256": content_sha,
                "eval_contract_path": str(EVAL_CONTRACT),
                "max_frames": 750,
                "subset_label": "select8",
                "subset_items_sha256": digest,
            },
            "reports": _full_n2_reports(wave3c_infer_defaults()),
        },
        eval_contract=EVAL_CONTRACT,
    )
    assert int((out.get("infer") or {}).get("max_frames")) == 750
    assert len(ids) == 8
    assert set(GENERATE_INFER_KEYS).issubset(set(wave3c_infer_defaults()))


@pytest.mark.skipif(not EVAL_CONTRACT.is_file() or not SCRIPT_74.is_file(), reason="contract/script missing")
def test_score_only_refuses_missing_wavs_fail_closed(tmp_path: Path):
    mod = _load_script(SCRIPT_74, "eval_discrete_lm_score_only_74d")
    contract = load_eval_contract(EVAL_CONTRACT)
    content_sha = str(contract.get("_content_sha256") or contract.get("sha256"))
    _, _, digest = load_frozen_utt_ids("select8", verify_hash=True)
    bundle = {
        "provenance": {
            "eval_contract_sha256": content_sha,
            "max_frames": 750,
            "subset_label": "select8",
            "subset_items_sha256": digest,
        },
        "reports": _full_n2_reports(),
    }
    with pytest.raises(SystemExit, match="missing wav"):
        mod.assert_score_only_wavs_present(bundle, d1b=True, skip_eval_contract=False)
    # Non-canonical escape may skip.
    mod.assert_score_only_wavs_present(bundle, d1b=True, skip_eval_contract=True)


@pytest.mark.skipif(not EVAL_CONTRACT.is_file() or not D1B_YAML.is_file(), reason="contract/yaml missing")
def test_reject_contradictory_subset_label_vs_hash():
    cfg = load_yaml(D1B_YAML)
    _, _, select8 = load_frozen_utt_ids("select8", verify_hash=True)
    _, _, dev32 = load_frozen_utt_ids("dev32", verify_hash=True)
    assert resolve_subset_alias("select8") == "select8"
    with pytest.raises(RuntimeError, match="label/hash contradiction"):
        enforce_eval_contract(
            cfg,
            contract_path=EVAL_CONTRACT,
            subset_label="select8",
            subset_items_sha256=dev32,
            require_clean_sha=False,
        )
    with pytest.raises(RuntimeError, match="label/hash contradiction"):
        enforce_eval_contract(
            cfg,
            contract_path=EVAL_CONTRACT,
            subset_label="dev32",
            subset_items_sha256=select8,
            require_clean_sha=False,
        )


@pytest.mark.skipif(not EVAL_CONTRACT.is_file() or not D1B_YAML.is_file(), reason="contract/yaml missing")
def test_cli_speaker_map_checked_against_contract(tmp_path: Path):
    from minispeech.data.speaker_registry import SpeakerRegistry

    cfg = load_yaml(D1B_YAML)
    bad_map = tmp_path / "bad_speakers.json"
    reg = SpeakerRegistry.from_mapping(
        {"spk_a": 1},
        source_manifest_sha256="x",
        train_set_sha256="y",
        notes="not trackn_v2",
    )
    reg.save(bad_map)
    assert reg.sha256 != TRACKN_V2_SPEAKER_SHA256
    with pytest.raises(RuntimeError, match="speaker map content sha256 mismatch"):
        enforce_eval_contract(
            cfg,
            contract_path=EVAL_CONTRACT,
            require_clean_sha=False,
            speaker_map_path=str(bad_map),
        )


@pytest.mark.skipif(not SCRIPT_73.is_file(), reason="infer script missing")
def test_canonical_infer_forbids_cli_max_frames_override():
    mod = _load_script(SCRIPT_73, "infer_discrete_lm_cli_73")
    with pytest.raises(SystemExit, match="--max-frames overrides frozen"):
        mod.refuse_canonical_cli_infer_overrides(
            skip_eval_contract=False, max_frames=271
        )
    # Non-canonical escape remains available.
    mod.refuse_canonical_cli_infer_overrides(skip_eval_contract=True, max_frames=271)
    mod.refuse_canonical_cli_infer_overrides(skip_eval_contract=False, max_frames=None)
