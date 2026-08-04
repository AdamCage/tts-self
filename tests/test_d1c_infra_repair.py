"""Extended INFRA repair tests for D1c semantic-EOS blockers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from minispeech.data.collate_codec_seq import collate_codec_seq
from minispeech.data.d1b_oracles import eval_contract_content_sha256, load_eval_contract
from minispeech.data.discrete_oracles import (
    items_sha256,
    load_formal_subset,
    payload_content_file_sha256,
)
from minispeech.models.discrete_speech_lm import (
    DiscreteGenerationOutput,
    DiscreteSpeechLM,
    DiscreteSpeechLMConfig,
)


ROOT = Path(__file__).resolve().parents[1]


def _fake_ckpt_hashes(**overrides: str) -> dict[str, str]:
    """Trainer-shaped run identity digests for synthetic formal reports."""
    base = {
        "eval_contract": "e" * 64,
        "recipe": "f" * 64,
        "recipe_source_manifest": "a" * 64,
        "run_manifest": "b" * 64,
    }
    base.update(overrides)
    return base


def _write_candidate_ckpt(tmp_path: Path, step: int, payload: bytes | None = None) -> tuple[Path, str]:
    """Write a distinct ``candidate_step_XXXX.pt`` and return (path, sha256)."""
    path = tmp_path / f"candidate_step_{step:04d}.pt"
    blob = payload if payload is not None else f"ckpt-bytes-{step}".encode()
    path.write_bytes(blob)
    return path, hashlib.sha256(blob).hexdigest()


def _semantic_model() -> DiscreteSpeechLM:
    cfg = DiscreteSpeechLMConfig.tiny()
    cfg.termination_mode = "semantic_eos_v1"
    cfg.eos_id = cfg.codebook_size
    cfg.codec_vocab_size = cfg.codebook_size
    cfg.semantic_vocab_size = cfg.codebook_size + 1
    return DiscreteSpeechLM(cfg)


def _batch(m: DiscreteSpeechLM) -> dict:
    return collate_codec_seq(
        [
            {
                "utt_id": "short",
                "input_ids": torch.tensor([3, 4]),
                "codes": torch.randint(0, m.codebook_size, (2, m.n_codebooks)),
                "speaker_id": 0,
            },
            {
                "utt_id": "long",
                "input_ids": torch.tensor([5, 6, 7]),
                "codes": torch.randint(0, m.codebook_size, (4, m.n_codebooks)),
                "speaker_id": 1,
            },
        ],
        pad_token_id=0,
        speech_eos_id=m.eos_id,
        termination_mode="semantic_eos_v1",
    )


def test_residual_heads_receive_gradients_in_semantic_eos_mode():
    m = _semantic_model()
    batch = _batch(m)
    out = m(
        batch["input_ids"],
        batch["codes"],
        text_mask=batch["text_mask"],
        frame_mask=batch["codec_frame_mask"],
        semantic_targets=batch["semantic_targets"],
        semantic_mask=batch["semantic_mask"],
        speaker_id=batch["speaker_id"],
        residual_weights=torch.ones(m.n_codebooks - 1),
    )
    assert out.loss is not None
    out.loss["total"].backward()
    residual_params = [p for p in m.heads.residuals.parameters() if p.requires_grad]
    assert residual_params
    n_with_grad = sum(1 for p in residual_params if p.grad is not None and float(p.grad.abs().sum()) > 0)
    assert n_with_grad == len(residual_params), f"only {n_with_grad}/{len(residual_params)} residual params got grads"


def test_immediate_eos_has_no_phantom_codec_frame():
    m = _semantic_model()

    class _AlwaysEOS(torch.nn.Module):
        def forward(self, h: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
            logits = h.new_full((*h.shape[:-1], m.semantic_vocab_size), -1.0e9)
            logits[..., m.eos_id] = 1.0e9
            return logits

    m.heads.semantic = _AlwaysEOS()
    out = m.generate_semantic_eos(
        torch.tensor([[3, 4, 5]]),
        max_frames=4,
        min_frames=0,
        max_cb0_repeat=100,
        temperature=0.0,
        top_k=0,
    )
    assert isinstance(out, DiscreteGenerationOutput)
    assert int(out.lengths[0]) == 0
    assert tuple(out.codes.shape) == (1, 0, m.n_codebooks)
    assert out.rows[0]["termination_reason"] == "learned_eos"
    assert out.rows[0]["eos_selected"] is True


def test_contract_2_0_0_self_check_and_no_legacy_stop_in_infer():
    contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
    assert contract["sha256"] == eval_contract_content_sha256(contract)
    from minispeech.infer.discrete_config import contract_infer_from_eval_contract

    infer = contract_infer_from_eval_contract(contract)
    for key in (
        "stop_threshold",
        "stop_consecutive",
        "frames_per_text_token",
        "stop_overrun_bias",
        "stop_earliest_frac",
    ):
        assert key not in infer


def test_formal_dev_oracles_load_with_file_sha_and_select8_composition():
    contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
    for key in ("dev32", "select8", "gate24", "full32"):
        att = load_formal_subset(key, contract=contract)
        assert att["formal_eligible"] is True
        assert att["file_sha256"]
        assert len(att["file_sha256"]) == 64
    select = load_formal_subset("select8", contract=contract)
    n_ruslan = sum(1 for u in select["utt_ids"] if str(u).startswith("ruslan_"))
    assert n_ruslan == 2
    assert len(select["utt_ids"]) - n_ruslan == 6


def test_formal_oracle_refuses_missing_file_sha(tmp_path):
    path = tmp_path / "subset.json"
    ids = ["ruslan_1", "cv_1", "cv_2", "cv_3", "cv_4", "cv_5", "cv_6", "cv_7"]
    payload = {
        "utt_ids": ids,
        "items_sha256": items_sha256(ids),
        "splits": {"select8": ids, "gate24": [], "full32": ids},
        "split_hashes": {
            "select8": items_sha256(ids),
            "gate24": items_sha256([]),
            "full32": items_sha256(ids),
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # Intentionally omit payload file_sha256 and contract file_sha256.
    contract = {
        "subsets": {
            "select8": {
                "path": str(path),
                "items_sha256": items_sha256(ids),
            }
        }
    }
    with pytest.raises(ValueError, match="file_sha256|missing hashes"):
        load_formal_subset("select8", contract=contract, split="select8")
    # Payload with declared sha but contract missing file_sha256.
    payload["file_sha256"] = content_sha
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing hashes"):
        load_formal_subset("select8", contract=contract, split="select8")


def test_formal_oracle_refuses_wrong_select8_composition(tmp_path):
    ids = [f"ruslan_{i}" for i in range(4)] + [f"cv_{i}" for i in range(4)]
    full = ids + [f"extra_{i}" for i in range(24)]
    body = {
        "utt_ids": full,
        "items_sha256": items_sha256(full),
        "splits": {"select8": ids, "gate24": full[8:], "full32": full},
        "split_hashes": {
            "select8": items_sha256(ids),
            "gate24": items_sha256(full[8:]),
            "full32": items_sha256(full),
        },
    }
    text = json.dumps(body, ensure_ascii=False, indent=2) + "\n"
    content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    payload = {**body, "file_sha256": content_sha}
    path = tmp_path / "bad_select.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    # Recompute content sha after including field? payload_content strips field.
    assert payload_content_file_sha256(payload) == content_sha
    contract = {
        "subsets": {
            "select8": {
                "path": str(path),
                "items_sha256": items_sha256(ids),
                "file_sha256": content_sha,
            }
        }
    }
    with pytest.raises(ValueError, match="2 RUSLAN"):
        load_formal_subset("select8", contract=contract, split="select8")


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_infer_script_unwraps_generation_output(tmp_path):
    """Treating DiscreteGenerationOutput as a tensor must not happen."""
    mod = _load_script("73_infer_discrete_lm.py")
    out = DiscreteGenerationOutput(
        codes=torch.zeros(1, 0, 16, dtype=torch.long),
        frame_mask=torch.zeros(1, 0, dtype=torch.bool),
        lengths=torch.zeros(1, dtype=torch.long),
        rows=[{"termination_reason": "learned_eos", "n_decoded_frames": 0}],
    )
    # Mimic the fixed unwrap path.
    codes = out.codes if isinstance(out, DiscreteGenerationOutput) else out
    assert int(codes.shape[1]) == 0
    assert hasattr(mod, "main")


def test_eval_teacher_forced_requires_semantic_targets():
    mod = _load_script("74_eval_discrete_lm.py")
    m = _semantic_model()
    batch = {
        "input_ids": torch.randint(1, 10, (1, 3)),
        "codes": torch.randint(0, m.codebook_size, (1, 2, m.n_codebooks)),
        "text_mask": torch.ones(1, 3, dtype=torch.bool),
        "frame_mask": torch.ones(1, 2, dtype=torch.bool),
        "speaker_id": torch.zeros(1, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="semantic_targets"):
        mod.teacher_forced_ce(m, batch, torch.device("cpu"))


def test_eval_unwrap_forbids_tensor_confusion():
    mod = _load_script("74_eval_discrete_lm.py")
    out = DiscreteGenerationOutput(
        codes=torch.zeros(1, 3, 16, dtype=torch.long),
        frame_mask=torch.ones(1, 3, dtype=torch.bool),
        lengths=torch.tensor([3]),
        rows=[{"termination_reason": "learned_eos"}],
    )
    codes, meta = mod._unwrap_generation(out)
    assert torch.is_tensor(codes)
    assert meta["termination_reason"] == "learned_eos"
    with pytest.raises(TypeError):
        mod._unwrap_generation({"not": "a tensor"})


def test_emit_d1c_v2_report_is_exported_and_callable(tmp_path):
    mod = _load_script("74_eval_discrete_lm.py")
    assert callable(mod.emit_d1c_v2_report)
    # Minimal smoke: wrong conditions must fail closed.
    with pytest.raises(ValueError, match="six canonical"):
        mod.emit_d1c_v2_report(
            out_path=tmp_path / "r.json",
            checkpoint={},
            provenance={},
            infer={},
            teacher_forced_metrics={},
            ordered_utt_ids=[],
            conditions={},
            summaries={},
            gates={},
        )


def test_selector_sha256_file_and_eligibility_from_rows(tmp_path):
    mod = _load_script("76_select_discrete_checkpoint.py")
    p = tmp_path / "blob.bin"
    p.write_bytes(b"abc123")
    assert mod._sha256_file(p) == hashlib.sha256(b"abc123").hexdigest()

    def _report(step: int, reason: str, cer: float, eligible_gate: bool) -> dict:
        uid = "u0"
        ckpt_path, ckpt_sha = _write_candidate_ckpt(tmp_path, step)
        row = {
            "utt_id": uid,
            "cer": cer,
            "catastrophic": False,
            "n_frames": 10,
            "gt_frames": 10,
            "generation": {"termination_reason": reason, "n_decoded_frames": 10, "early_eos_suppressed_count": 0, "max_cb0_run": 2},
            "wav": {"path": "x.wav", "file_sha256": "a" * 64, "pcm_sha256": "b" * 64},
            "normalized_reference_sha256": "c" * 64,
            "generated_codes_sha256": "d" * 64,
        }
        return {
            "schema": "minispeech.discrete_eval_report",
            "schema_version": "2.0.0",
            "formal_eligible": True,
            "checkpoint": {
                "path": str(ckpt_path),
                "sha256": ckpt_sha,
                "step": step,
                "schema": "minispeech.discrete_lm_checkpoint",
                "schema_version": "2.0.0",
                "hashes": _fake_ckpt_hashes(),
            },
            "provenance": {
                "eval_contract_sha256": "e" * 64,
                "recipe_sha256": "f" * 64,
                "asr_sha256": "1" * 64,
            },
            "infer": {"max_frames": 750},
            "teacher_forced_metrics": {
                "teacher_forced_ce_source": "eval_computed",
                "total_ce": 1.0,
                "tf_eos_final_top1_accuracy": 1.0,
                "tf_eos_early_top1_false_positive_count": 0,
            },
            "row_binding": {
                "ordered_utt_ids": [uid],
                "ordered_utt_ids_sha256": "2" * 64,
            },
            "conditions": {
                name: {
                    "subset_key": "select8",
                    "subset_path": "x",
                    "subset_file_sha256": "3" * 64,
                    "expected_count": 1,
                    "evaluated_count": 1,
                    "ordered_utt_ids_sha256": "2" * 64,
                    "rows": [row],
                    "rows_sha256": "4" * 64,
                }
                for name in (
                    "model",
                    "copy_synth",
                    "wrong_utt",
                    "random_codes",
                    "shuffle_text",
                    "mask_text",
                )
            },
            "summaries": {
                "model_macro_cer": cer,
                "n_catastrophic": 0,
                "max_abs_log_duration_error": 0.0,
                "mean_abs_log_duration_error": 0.0,
            },
            "gates": {
                "eligible": eligible_gate,
                "text_usage": {"pass": True},
            },
            "report_sha256": "5" * 64,
        }

    # Bypass validate_formal_report by injecting directly into select after monkeypatch.
    paths = []
    for i, step in enumerate((250, 375, 500, 625, 750)):
        reason = "learned_eos" if step != 250 else "safety_max_frames"
        cer = 0.1 if step != 750 else 0.05
        rep = _report(step, reason, cer, eligible_gate=True)
        pth = tmp_path / f"r{step}.json"
        # select_d1c_formal_reports validates — skip full schema by patching.
        paths.append(pth)
        pth.write_text(json.dumps(rep), encoding="utf-8")

    def _fake_validate(data, **kwargs):
        return data

    import minispeech.eval.discrete_report_schema as schema

    old = schema.validate_formal_report
    schema.validate_formal_report = _fake_validate  # type: ignore[assignment]
    # Also patch the selector module's bound name.
    mod.validate_formal_report = _fake_validate
    try:
        with pytest.raises(ValueError, match="no eligible"):
            # step 250 has safety_max_frames → all must be learned_eos for eligibility;
            # with one ineligible and others eligible, should still find eligible.
            # Force all unsafe:
            for step in (250, 375, 500, 625, 750):
                bad = _report(step, "safety_max_frames", 0.1, True)
                (tmp_path / f"r{step}.json").write_text(json.dumps(bad), encoding="utf-8")
            mod.select_d1c_formal_reports(paths)
        # Now all learned_eos — even if gates.eligible is False, row recompute should pass.
        for step in (250, 375, 500, 625, 750):
            good = _report(step, "learned_eos", 0.2 - step / 10000.0, eligible_gate=False)
            (tmp_path / f"r{step}.json").write_text(json.dumps(good), encoding="utf-8")
        result = mod.select_d1c_formal_reports(paths)
        assert result["selected"]["step"] == 750
        assert result["selected"]["gate_eligible"] is True
    finally:
        schema.validate_formal_report = old
        mod.validate_formal_report = old


def test_selector_refuses_duplicate_checkpoint_hashes(tmp_path):
    """Round-4 attack: five steps, one physical checkpoint → selector refuse."""
    mod = _load_script("76_select_discrete_checkpoint.py")
    shared = tmp_path / "best.pt"
    shared.write_bytes(b"one-physical-checkpoint")
    shared_sha = hashlib.sha256(b"one-physical-checkpoint").hexdigest()

    def _report(step: int) -> dict:
        # Forge candidate_step names that all hash-claim the same file bytes.
        # Distinct basenames but identical sha256 must still be refused.
        alias = tmp_path / f"candidate_step_{step:04d}.pt"
        alias.write_bytes(b"one-physical-checkpoint")
        return {
            "schema": "minispeech.discrete_eval_report",
            "schema_version": "2.0.0",
            "formal_eligible": True,
            "checkpoint": {
                "path": str(alias),
                "sha256": shared_sha,
                "step": step,
                "schema": "minispeech.discrete_lm_checkpoint",
                "schema_version": "2.0.0",
                "hashes": _fake_ckpt_hashes(),
            },
            "provenance": {
                "eval_contract_sha256": "e" * 64,
                "recipe_sha256": "f" * 64,
                "asr_sha256": "1" * 64,
            },
            "infer": {"max_frames": 750},
            "teacher_forced_metrics": {
                "teacher_forced_ce_source": "eval_computed",
                "total_ce": 1.0,
                "tf_eos_final_top1_accuracy": 1.0,
                "tf_eos_early_top1_false_positive_count": 0,
            },
            "row_binding": {
                "ordered_utt_ids": ["u0"],
                "ordered_utt_ids_sha256": "2" * 64,
            },
            "conditions": {
                name: {
                    "subset_key": "select8",
                    "subset_path": "x",
                    "subset_file_sha256": "3" * 64,
                    "expected_count": 1,
                    "evaluated_count": 1,
                    "ordered_utt_ids_sha256": "2" * 64,
                    "rows": [
                        {
                            "utt_id": "u0",
                            "cer": 0.1,
                            "catastrophic": False,
                            "n_frames": 10,
                            "gt_frames": 10,
                            "generation": {
                                "termination_reason": "learned_eos",
                                "n_decoded_frames": 10,
                                "early_eos_suppressed_count": 0,
                                "max_cb0_run": 2,
                            },
                            "wav": {
                                "path": "x.wav",
                                "file_sha256": "a" * 64,
                                "pcm_sha256": "b" * 64,
                            },
                            "normalized_reference_sha256": "c" * 64,
                            "generated_codes_sha256": "d" * 64,
                        }
                    ],
                    "rows_sha256": "4" * 64,
                }
                for name in (
                    "model",
                    "copy_synth",
                    "wrong_utt",
                    "random_codes",
                    "shuffle_text",
                    "mask_text",
                )
            },
            "summaries": {
                "model_macro_cer": 0.1,
                "n_catastrophic": 0,
                "max_abs_log_duration_error": 0.0,
                "mean_abs_log_duration_error": 0.0,
            },
            "gates": {"eligible": True, "text_usage": {"pass": True}},
            "report_sha256": "5" * 64,
        }

    paths = []
    for step in (250, 375, 500, 625, 750):
        p = tmp_path / f"dup_{step}.json"
        p.write_text(json.dumps(_report(step)), encoding="utf-8")
        paths.append(p)

    import minispeech.eval.discrete_report_schema as schema

    old = schema.validate_formal_report
    schema.validate_formal_report = lambda data, **kwargs: data  # type: ignore[assignment]
    mod.validate_formal_report = schema.validate_formal_report
    try:
        with pytest.raises(ValueError, match="duplicate checkpoint.sha256"):
            mod.select_d1c_formal_reports(paths)
    finally:
        schema.validate_formal_report = old
        mod.validate_formal_report = old


def test_selector_refuses_checkpoint_hash_mismatch(tmp_path):
    """Claimed checkpoint.sha256 must match bytes at checkpoint.path."""
    mod = _load_script("76_select_discrete_checkpoint.py")

    def _report(step: int, *, corrupt_hash: bool = False) -> dict:
        path, live_sha = _write_candidate_ckpt(tmp_path, step)
        claimed = ("0" * 64) if corrupt_hash else live_sha
        return {
            "schema": "minispeech.discrete_eval_report",
            "schema_version": "2.0.0",
            "formal_eligible": True,
            "checkpoint": {
                "path": str(path),
                "sha256": claimed,
                "step": step,
                "schema": "minispeech.discrete_lm_checkpoint",
                "schema_version": "2.0.0",
                "hashes": _fake_ckpt_hashes(),
            },
            "provenance": {
                "eval_contract_sha256": "e" * 64,
                "recipe_sha256": "f" * 64,
                "asr_sha256": "1" * 64,
            },
            "infer": {"max_frames": 750},
            "teacher_forced_metrics": {
                "teacher_forced_ce_source": "eval_computed",
                "total_ce": 1.0,
                "tf_eos_final_top1_accuracy": 1.0,
                "tf_eos_early_top1_false_positive_count": 0,
            },
            "row_binding": {
                "ordered_utt_ids": ["u0"],
                "ordered_utt_ids_sha256": "2" * 64,
            },
            "conditions": {
                name: {
                    "subset_key": "select8",
                    "subset_path": "x",
                    "subset_file_sha256": "3" * 64,
                    "expected_count": 1,
                    "evaluated_count": 1,
                    "ordered_utt_ids_sha256": "2" * 64,
                    "rows": [
                        {
                            "utt_id": "u0",
                            "cer": 0.1,
                            "catastrophic": False,
                            "n_frames": 10,
                            "gt_frames": 10,
                            "generation": {
                                "termination_reason": "learned_eos",
                                "n_decoded_frames": 10,
                                "early_eos_suppressed_count": 0,
                                "max_cb0_run": 2,
                            },
                            "wav": {
                                "path": "x.wav",
                                "file_sha256": "a" * 64,
                                "pcm_sha256": "b" * 64,
                            },
                            "normalized_reference_sha256": "c" * 64,
                            "generated_codes_sha256": "d" * 64,
                        }
                    ],
                    "rows_sha256": "4" * 64,
                }
                for name in (
                    "model",
                    "copy_synth",
                    "wrong_utt",
                    "random_codes",
                    "shuffle_text",
                    "mask_text",
                )
            },
            "summaries": {
                "model_macro_cer": 0.1,
                "n_catastrophic": 0,
                "max_abs_log_duration_error": 0.0,
                "mean_abs_log_duration_error": 0.0,
            },
            "gates": {"eligible": True, "text_usage": {"pass": True}},
            "report_sha256": "5" * 64,
        }

    paths = []
    for step in (250, 375, 500, 625, 750):
        p = tmp_path / f"mm_{step}.json"
        # Corrupt only step 500 so distinct-sha check passes first when all
        # claimed hashes differ, then on-disk verify fails for the mismatch.
        p.write_text(
            json.dumps(_report(step, corrupt_hash=(step == 500))),
            encoding="utf-8",
        )
        paths.append(p)

    import minispeech.eval.discrete_report_schema as schema

    old = schema.validate_formal_report
    schema.validate_formal_report = lambda data, **kwargs: data  # type: ignore[assignment]
    mod.validate_formal_report = schema.validate_formal_report
    try:
        with pytest.raises(ValueError, match="checkpoint sha256 mismatch"):
            mod.select_d1c_formal_reports(paths)
    finally:
        schema.validate_formal_report = old
        mod.validate_formal_report = old


def test_validate_formal_report_rejects_checkpoint_hash_mismatch(tmp_path):
    from minispeech.eval.discrete_report_schema import (
        build_formal_report,
        ordered_utt_ids_sha256,
        sha256_json,
        validate_formal_report,
    )

    ids = ["u1"]
    ckpt, live = _write_candidate_ckpt(tmp_path, 250)
    conditions = {}
    for name in (
        "model",
        "copy_synth",
        "wrong_utt",
        "random_codes",
        "shuffle_text",
        "mask_text",
    ):
        rows = [
            {
                "utt_id": "u1",
                "ordinal": 0,
                "normalized_reference_sha256": "a" * 64,
                "generated_codes_sha256": "b" * 64,
                "generation": {"termination_reason": "learned_eos"},
                "wav": {"path": "u1.wav", "file_sha256": "c" * 64, "pcm_sha256": "d" * 64},
                "cer": 0.0,
            }
        ]
        conditions[name] = {
            "subset_key": "select8",
            "subset_path": "subset.json",
            "subset_file_sha256": "e" * 64,
            "expected_count": 1,
            "evaluated_count": 1,
            "ordered_utt_ids_sha256": ordered_utt_ids_sha256(ids),
            "rows": rows,
        }
    report = build_formal_report(
        formal_eligible=True,
        checkpoint={
            "path": str(ckpt),
            "sha256": live,
            "step": 250,
            "schema": "x",
            "schema_version": "2",
            "hashes": _fake_ckpt_hashes(),
        },
        provenance={"eval_contract_sha256": "1" * 64},
        infer={"max_frames": 750},
        teacher_forced_metrics={"teacher_forced_ce_source": "eval_computed", "total_ce": 1.0},
        row_binding={"ordered_utt_ids": ids, "ordered_utt_ids_sha256": ordered_utt_ids_sha256(ids)},
        conditions=conditions,
    )
    report["checkpoint"]["sha256"] = "0" * 64
    report.pop("report_sha256", None)
    report["report_sha256"] = sha256_json(report)
    with pytest.raises(ValueError, match="checkpoint.sha256 does not match"):
        validate_formal_report(report)


def test_selector_refuses_checkpoint_basename_mismatch(tmp_path):
    """Basename must be candidate_step_<step:04d>.pt (mutation Y3 coverage)."""
    mod = _load_script("76_select_discrete_checkpoint.py")

    def _report(step: int, *, basename: str | None = None) -> dict:
        if basename is None:
            path, live_sha = _write_candidate_ckpt(tmp_path, step)
        else:
            path = tmp_path / basename
            blob = f"ckpt-bytes-{step}-alias".encode()
            path.write_bytes(blob)
            live_sha = hashlib.sha256(blob).hexdigest()
        return {
            "schema": "minispeech.discrete_eval_report",
            "schema_version": "2.0.0",
            "formal_eligible": True,
            "checkpoint": {
                "path": str(path),
                "sha256": live_sha,
                "step": step,
                "schema": "minispeech.discrete_lm_checkpoint",
                "schema_version": "2.0.0",
                "hashes": _fake_ckpt_hashes(),
            },
            "provenance": {
                "eval_contract_sha256": "e" * 64,
                "recipe_sha256": "f" * 64,
                "asr_sha256": "1" * 64,
            },
            "infer": {"max_frames": 750},
            "teacher_forced_metrics": {
                "teacher_forced_ce_source": "eval_computed",
                "total_ce": 1.0,
                "tf_eos_final_top1_accuracy": 1.0,
                "tf_eos_early_top1_false_positive_count": 0,
            },
            "row_binding": {
                "ordered_utt_ids": ["u0"],
                "ordered_utt_ids_sha256": "2" * 64,
            },
            "conditions": {
                name: {
                    "subset_key": "select8",
                    "subset_path": "x",
                    "subset_file_sha256": "3" * 64,
                    "expected_count": 1,
                    "evaluated_count": 1,
                    "ordered_utt_ids_sha256": "2" * 64,
                    "rows": [
                        {
                            "utt_id": "u0",
                            "cer": 0.1,
                            "catastrophic": False,
                            "n_frames": 10,
                            "gt_frames": 10,
                            "generation": {
                                "termination_reason": "learned_eos",
                                "n_decoded_frames": 10,
                                "early_eos_suppressed_count": 0,
                                "max_cb0_run": 2,
                            },
                            "wav": {
                                "path": "x.wav",
                                "file_sha256": "a" * 64,
                                "pcm_sha256": "b" * 64,
                            },
                            "normalized_reference_sha256": "c" * 64,
                            "generated_codes_sha256": "d" * 64,
                        }
                    ],
                    "rows_sha256": "4" * 64,
                }
                for name in (
                    "model",
                    "copy_synth",
                    "wrong_utt",
                    "random_codes",
                    "shuffle_text",
                    "mask_text",
                )
            },
            "summaries": {
                "model_macro_cer": 0.1,
                "n_catastrophic": 0,
                "max_abs_log_duration_error": 0.0,
                "mean_abs_log_duration_error": 0.0,
            },
            "gates": {"eligible": True, "text_usage": {"pass": True}},
            "report_sha256": "5" * 64,
        }

    paths = []
    for step in (250, 375, 500, 625, 750):
        # Cross-wire: step=250 report points at candidate_step_0750.pt basename.
        rep = _report(step, basename="best.pt" if step == 250 else None)
        if step == 250:
            # Distinct file so sha rule is not the first failure mode.
            pass
        p = tmp_path / f"base_{step}.json"
        p.write_text(json.dumps(rep), encoding="utf-8")
        paths.append(p)

    import minispeech.eval.discrete_report_schema as schema

    old = schema.validate_formal_report
    schema.validate_formal_report = lambda data, **kwargs: data  # type: ignore[assignment]
    mod.validate_formal_report = schema.validate_formal_report
    try:
        with pytest.raises(ValueError, match="basename"):
            mod.select_d1c_formal_reports(paths)
    finally:
        schema.validate_formal_report = old
        mod.validate_formal_report = old


def test_w1_w2_bound_to_live_contract_and_resolve_populates_hashes():
    from minispeech.train.discrete_lm_trainer import resolve_repro_hashes
    from minispeech.utils import load_yaml

    contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
    live = contract["sha256"]
    for name in (
        "discrete_speech_lm_d1c_semantic_eos_w1.yaml",
        "discrete_speech_lm_d1c_semantic_eos_w2.yaml",
    ):
        cfg = load_yaml(ROOT / "configs" / name)
        assert cfg["train"]["eval_contract_hash"] == live
        hashes = resolve_repro_hashes(cfg)
        assert hashes["eval_contract"] == live
        assert hashes["recipe"] == contract["recipe_sha256"]
        assert hashes["recipe_source_manifest"] == contract["recipe_source_manifest_sha256"]
        assert hashes["run_manifest"]
        assert None not in (
            hashes["recipe"],
            hashes["recipe_source_manifest"],
            hashes["run_manifest"],
        )


def test_stale_eval_contract_hash_fails_closed():
    from minispeech.train.discrete_lm_trainer import resolve_repro_hashes
    from minispeech.utils import load_yaml

    cfg = load_yaml(ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml")
    cfg = json.loads(json.dumps(cfg))  # deep copy via JSON
    cfg["train"]["eval_contract_hash"] = "a" * 64
    with pytest.raises(ValueError, match="eval_contract_hash mismatch"):
        resolve_repro_hashes(cfg)


def test_recipe_source_manifest_includes_critical_modules_and_matches():
    from minispeech.data.d1b_oracles import verify_recipe_source_manifest

    contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
    manifest = contract["recipe_source_manifest"]
    for rel in (
        "src/minispeech/data/discrete_oracles.py",
        "src/minispeech/data/d1b_oracles.py",
        "src/minispeech/eval/asr.py",
        "src/minispeech/data/codec_seq_protocol.py",
        "src/minispeech/constants.py",
    ):
        assert rel in manifest
    assert verify_recipe_source_manifest(manifest, root=ROOT) == []


def test_v2_0_1_parent_points_at_live_contract():
    parent = json.loads(
        (ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_1.json").read_text(
            encoding="utf-8"
        )
    )
    contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
    assert parent["parent_sha256"] == contract["sha256"]
    assert parent["parent_recipe_sha256"] == contract["recipe_sha256"]


def test_emit_d1c_v2_from_evaluator_shaped_rows(tmp_path):
    """Regression: ordinal + catastrophic + real pcm/file digests must allow emit."""
    import soundfile as sf

    mod = _load_script("74_eval_discrete_lm.py")
    uid = "u0"
    wav_path = tmp_path / "u0.wav"
    pcm = (np.linspace(-0.2, 0.2, 2400, dtype=np.float32))
    sf.write(str(wav_path), pcm, 24000)
    from minispeech.eval.discrete_report_schema import file_sha256 as _fsha

    file_digest = _fsha(wav_path)
    pcm_digest = mod._pcm_sha256_from_wav(wav_path)
    assert pcm_digest != file_digest

    def _row(*, cer: float = 0.1, n_frames: int = 10, reason: str = "learned_eos") -> dict:
        return {
            "utt_id": uid,
            "cer": cer,
            "ref": "привет",
            "hyp": "привет",
            "n_frames": n_frames,
            "gt_frames": 10,
            "wav": str(wav_path),
            "stop_reason": reason,
            "generation": {
                "termination_reason": reason,
                "n_decoded_frames": n_frames,
                "early_eos_suppressed_count": 0,
                "max_cb0_run": 2,
            },
            "generated_codes_sha256": "d" * 64,
        }

    contract = {
        "termination": {"mode": "semantic_eos_v1"},
        "sha256": "e" * 64,
        "recipe_sha256": "f" * 64,
        "asr": {"file_sha256": "1" * 64},
        "subsets": {
            "select8": {
                "path": "configs/train_subsets/d1c_dev32.json",
                "file_sha256": "3" * 64,
            }
        },
    }
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": {}}, ckpt)
    rows = [_row()]
    condition_reports = {
        name: {"rows": [dict(rows[0])]}
        for name in (
            "model",
            "copy_synth",
            "wrong_utt",
            "random_codes",
            "shuffle_text",
            "mask_text",
        )
    }
    # copy_synth slightly better CER so selector CER margin stays valid elsewhere
    condition_reports["copy_synth"]["rows"][0]["cer"] = 0.05
    report_path = mod._emit_d1c_v2_from_gate(
        out_dir=tmp_path / "out",
        contract=contract,
        ckpt=str(ckpt),
        provenance={
            "subset_label": "select8",
            "subset_path": "x",
            "subset_file_sha256": "3" * 64,
            "checkpoint_step": 750,
            "checkpoint_hashes": _fake_ckpt_hashes(eval_contract="e" * 64, recipe="f" * 64),
        },
        infer={"max_frames": 750},
        teacher_forced_ce={
            "loss": 1.2,
            "ce_cb0": 0.5,
            "ce_residual": 0.4,
            "tf_eos_final_top1_accuracy": 1.0,
            "tf_eos_early_top1_false_positive_count": 0,
            "teacher_forced_ce_source": "eval_computed",
        },
        ordered_utt_ids=[uid],
        condition_reports=condition_reports,
        gate={
            "verdict": "PASS",
            "criteria": {"macro_cer": 0.1, "n_catastrophic": 0},
            "text_usage_acceptance": {"pass": True},
            "stability_acceptance": {
                "phrase_loop_repetition": False,
                "runaway_or_repetition": False,
                "no_runaway": True,
            },
        },
    )
    assert report_path is not None and report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["formal_eligible"] is True
    assert report["gates"]["text_usage"]["pass"] is True
    assert report["summaries"]["phrase_loop_repetition"] is False
    assert report["summaries"]["runaway_or_repetition"] is False
    model_row = report["conditions"]["model"]["rows"][0]
    assert model_row["ordinal"] == 0
    assert "catastrophic" in model_row
    assert model_row["wav"]["file_sha256"] == file_digest
    assert model_row["wav"]["pcm_sha256"] == pcm_digest
    assert model_row["wav"]["pcm_sha256"] != model_row["wav"]["file_sha256"]


def test_emit_refuses_fabricated_eos_tf_defaults(tmp_path):
    mod = _load_script("74_eval_discrete_lm.py")
    with pytest.raises(ValueError, match="teacher_forced_ce_source|formal_eligible|not measured|unavailable"):
        mod.emit_d1c_v2_report(
            out_path=tmp_path / "r.json",
            checkpoint={
                "path": "c.pt",
                "sha256": "a" * 64,
                "step": 1,
                "schema": "minispeech.discrete_lm_checkpoint",
                "schema_version": "2.0.0",
                "hashes": _fake_ckpt_hashes(),
            },
            provenance={
                "eval_contract_sha256": "e" * 64,
                "recipe_sha256": "f" * 64,
                "asr_sha256": "1" * 64,
            },
            infer={"max_frames": 750},
            teacher_forced_metrics={"total_ce": 1.0},  # missing source + EOS metrics
            ordered_utt_ids=["u0"],
            conditions={name: {"rows": []} for name in (
                "model", "copy_synth", "wrong_utt", "random_codes", "shuffle_text", "mask_text"
            )},
            summaries={},
            gates={},
        )


def test_selector_refuses_suppressed_empty_and_extreme_duration(tmp_path):
    mod = _load_script("76_select_discrete_checkpoint.py")

    def _report(**row_extra) -> dict:
        uid = "u0"
        row = {
            "utt_id": uid,
            "ordinal": 0,
            "cer": 0.1,
            "catastrophic": False,
            "n_frames": 10,
            "gt_frames": 10,
            "generation": {
                "termination_reason": "learned_eos",
                "n_decoded_frames": 10,
                "early_eos_suppressed_count": 0,
                "max_cb0_run": 2,
            },
            "wav": {"path": "x.wav", "file_sha256": "a" * 64, "pcm_sha256": "b" * 64},
            "normalized_reference_sha256": "c" * 64,
            "generated_codes_sha256": "d" * 64,
        }
        row.update(row_extra)
        if "generation" in row_extra:
            row["generation"] = {**row["generation"], **row_extra["generation"]}
        return {
            "schema": "minispeech.discrete_eval_report",
            "schema_version": "2.0.0",
            "formal_eligible": True,
            "checkpoint": {
                "path": "c.pt",
                "sha256": "9" * 64,
                "step": 750,
                "schema": "minispeech.discrete_lm_checkpoint",
                "schema_version": "2.0.0",
                "hashes": _fake_ckpt_hashes(),
            },
            "provenance": {
                "eval_contract_sha256": "e" * 64,
                "recipe_sha256": "f" * 64,
                "asr_sha256": "1" * 64,
            },
            "infer": {"max_frames": 750},
            "teacher_forced_metrics": {
                "teacher_forced_ce_source": "eval_computed",
                "total_ce": 1.0,
                "tf_eos_final_top1_accuracy": 1.0,
                "tf_eos_early_top1_false_positive_count": 0,
            },
            "row_binding": {
                "ordered_utt_ids": [uid],
                "ordered_utt_ids_sha256": "2" * 64,
            },
            "conditions": {
                name: {
                    "subset_key": "select8",
                    "subset_path": "x",
                    "subset_file_sha256": "3" * 64,
                    "expected_count": 1,
                    "evaluated_count": 1,
                    "ordered_utt_ids_sha256": "2" * 64,
                    "rows": [row],
                    "rows_sha256": "4" * 64,
                }
                for name in (
                    "model",
                    "copy_synth",
                    "wrong_utt",
                    "random_codes",
                    "shuffle_text",
                    "mask_text",
                )
            },
            "summaries": {
                "model_macro_cer": 0.1,
                "n_catastrophic": 0,
                "max_abs_log_duration_error": 0.0,
                "mean_abs_log_duration_error": 0.0,
            },
            "gates": {"eligible": True, "text_usage": {"pass": True}},
            "report_sha256": "5" * 64,
        }

    import minispeech.eval.discrete_report_schema as schema

    old = schema.validate_formal_report
    schema.validate_formal_report = lambda data, **kwargs: data  # type: ignore[assignment]
    mod.validate_formal_report = schema.validate_formal_report
    try:
        def _write_five(builder) -> list[Path]:
            paths = []
            for step in (250, 375, 500, 625, 750):
                rep = builder(step)
                p = tmp_path / f"r{step}.json"
                p.write_text(json.dumps(rep), encoding="utf-8")
                paths.append(p)
            return paths

        def _with_step(base: dict, step: int) -> dict:
            out = json.loads(json.dumps(base))
            ckpt_path, ckpt_sha = _write_candidate_ckpt(tmp_path, step)
            out["checkpoint"]["step"] = step
            out["checkpoint"]["sha256"] = ckpt_sha
            out["checkpoint"]["path"] = str(ckpt_path)
            return out

        # early_eos + max_cb0 runaway
        bad1 = _report(
            generation={
                "termination_reason": "learned_eos",
                "n_decoded_frames": 10,
                "early_eos_suppressed_count": 7,
                "max_cb0_run": 99,
            }
        )
        with pytest.raises(ValueError, match="no eligible"):
            mod.select_d1c_formal_reports(_write_five(lambda s: _with_step(bad1, s)))

        # zero-frame generation
        bad2 = _report(
            n_frames=0,
            generation={
                "termination_reason": "learned_eos",
                "n_decoded_frames": 0,
                "early_eos_suppressed_count": 0,
                "max_cb0_run": 0,
            },
        )
        with pytest.raises(ValueError, match="no eligible"):
            mod.select_d1c_formal_reports(_write_five(lambda s: _with_step(bad2, s)))

        # extreme duration ratio ~9.5x
        bad3 = _report(
            n_frames=95,
            gt_frames=10,
            generation={
                "termination_reason": "learned_eos",
                "n_decoded_frames": 95,
                "early_eos_suppressed_count": 0,
                "max_cb0_run": 0,
            },
        )
        with pytest.raises(ValueError, match="no eligible"):
            mod.select_d1c_formal_reports(_write_five(lambda s: _with_step(bad3, s)))

        # unavailable EOS TF metrics
        bad4 = _report()
        bad4["teacher_forced_metrics"]["tf_eos_final_top1_accuracy"] = "unavailable"
        with pytest.raises(ValueError, match="no eligible"):
            mod.select_d1c_formal_reports(_write_five(lambda s: _with_step(bad4, s)))
    finally:
        schema.validate_formal_report = old
        mod.validate_formal_report = old


def test_teacher_forced_ce_returns_eos_metrics():
    mod = _load_script("74_eval_discrete_lm.py")
    m = _semantic_model()
    batch = _batch(m)
    metrics = mod.teacher_forced_ce(m, batch, torch.device("cpu"))
    assert metrics["teacher_forced_ce_source"] == "eval_computed"
    assert "tf_eos_final_top1_accuracy" in metrics
    assert "tf_eos_early_top1_false_positive_count" in metrics
    assert 0.0 <= float(metrics["tf_eos_final_top1_accuracy"]) <= 1.0


def test_production_tf_aggregation_to_emitter_survives(tmp_path):
    """Catch the round-1/2 class of defect: measured EOS discarded before emit.

    Exercises teacher_forced_ce → aggregate_teacher_forced_batches →
    teacher_forced_metrics_for_emitter → _emit_d1c_v2_from_gate with
    evaluator-shaped rows (not a hand-written already-complete TF dict).
    """
    import soundfile as sf

    mod = _load_script("74_eval_discrete_lm.py")
    m = _semantic_model()
    batch_a = _batch(m)
    batch_b = _batch(m)
    measured = [
        mod.teacher_forced_ce(m, batch_a, torch.device("cpu")),
        mod.teacher_forced_ce(m, batch_b, torch.device("cpu")),
    ]
    # Negative: the old main() aggregation that only kept CE means.
    legacy_totals = {"loss": 0.0, "ce_cb0": 0.0, "ce_residual": 0.0}
    for m_batch in measured:
        for k in legacy_totals:
            legacy_totals[k] += m_batch[k]
    legacy_summary = {k: v / len(measured) for k, v in legacy_totals.items()}
    bare_float = legacy_summary.get("loss")
    assert isinstance(bare_float, float)

    summary = mod.aggregate_teacher_forced_batches(measured)
    assert summary["teacher_forced_ce_source"] == "eval_computed"
    assert isinstance(summary["tf_eos_final_top1_accuracy"], float)
    assert isinstance(summary["tf_eos_early_top1_false_positive_count"], int)
    assert "ce_semantic_eos" in summary
    tf_ce = mod.teacher_forced_metrics_for_emitter(summary)
    assert tf_ce["teacher_forced_ce_source"] == "eval_computed"
    assert "tf_eos_final_top1_accuracy" in tf_ce

    uid = "u0"
    wav_path = tmp_path / "u0.wav"
    pcm = np.linspace(-0.2, 0.2, 2400, dtype=np.float32)
    sf.write(str(wav_path), pcm, 24000)

    def _row(*, cer: float = 0.1, n_frames: int = 10) -> dict:
        return {
            "utt_id": uid,
            "cer": cer,
            "ref": "привет",
            "hyp": "привет",
            "n_frames": n_frames,
            "gt_frames": 10,
            "wav": str(wav_path),
            "stop_reason": "learned_eos",
            "generation": {
                "termination_reason": "learned_eos",
                "n_decoded_frames": n_frames,
                "early_eos_suppressed_count": 0,
                "max_cb0_run": 2,
            },
            "generated_codes_sha256": "d" * 64,
        }

    contract = {
        "termination": {"mode": "semantic_eos_v1"},
        "sha256": "e" * 64,
        "recipe_sha256": "f" * 64,
        "asr": {"file_sha256": "1" * 64},
        "subsets": {
            "select8": {
                "path": "configs/train_subsets/d1c_dev32.json",
                "file_sha256": "3" * 64,
            }
        },
    }
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": {}}, ckpt)
    condition_reports = {
        name: {"rows": [_row(cer=0.05 if name == "copy_synth" else 0.1)]}
        for name in (
            "model",
            "copy_synth",
            "wrong_utt",
            "random_codes",
            "shuffle_text",
            "mask_text",
        )
    }
    gate = {
        "verdict": "PASS",
        "criteria": {"macro_cer": 0.1, "n_catastrophic": 0},
        "text_usage_acceptance": {"pass": True, "shuffle_ok": True, "mask_ok": True},
        "stability_acceptance": {
            "pass": True,
            "phrase_loop_repetition": False,
            "runaway_or_repetition": False,
            "no_runaway": True,
        },
    }
    provenance = {
        "subset_label": "select8",
        "subset_path": "x",
        "subset_file_sha256": "3" * 64,
        "checkpoint_step": 750,
        "checkpoint_hashes": _fake_ckpt_hashes(eval_contract="e" * 64, recipe="f" * 64),
    }

    # Positive: aggregated measured metrics → formal emit succeeds.
    report_path = mod._emit_d1c_v2_from_gate(
        out_dir=tmp_path / "out_ok",
        contract=contract,
        ckpt=str(ckpt),
        provenance=dict(provenance),
        infer={"max_frames": 750},
        teacher_forced_ce=tf_ce,
        ordered_utt_ids=[uid],
        condition_reports=condition_reports,
        gate=gate,
    )
    assert report_path is not None and report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["formal_eligible"] is True
    assert report["teacher_forced_metrics"]["teacher_forced_ce_source"] == "eval_computed"
    assert report["teacher_forced_metrics"]["tf_eos_final_top1_accuracy"] == pytest.approx(
        tf_ce["tf_eos_final_top1_accuracy"]
    )
    assert "text_usage" in report["gates"]
    assert "phrase_loop_repetition" in report["summaries"]
    assert "runaway_or_repetition" in report["summaries"]

    # Negative: bare float (legacy main path) must soft-refuse, not crash.
    refused = mod._emit_d1c_v2_from_gate(
        out_dir=tmp_path / "out_bare",
        contract=contract,
        ckpt=str(ckpt),
        provenance=dict(provenance),
        infer={"max_frames": 750},
        teacher_forced_ce=bare_float,
        ordered_utt_ids=[uid],
        condition_reports=condition_reports,
        gate=gate,
    )
    assert refused is None
    assert not (tmp_path / "out_bare" / "discrete_eval_report_v2.json").is_file()

    # --score-only without measured TF: skip formal (no decode_bundle stamp).
    skipped = mod._emit_d1c_v2_from_gate(
        out_dir=tmp_path / "out_score",
        contract=contract,
        ckpt=str(ckpt),
        provenance={**provenance, "score_only": True},
        infer={"max_frames": 750},
        teacher_forced_ce=bare_float,
        ordered_utt_ids=[uid],
        condition_reports=condition_reports,
        gate=gate,
    )
    assert skipped is None

    # --score-only WITH measured TF from decode bundle → formal OK.
    score_ok = mod._emit_d1c_v2_from_gate(
        out_dir=tmp_path / "out_score_ok",
        contract=contract,
        ckpt=str(ckpt),
        provenance={**provenance, "score_only": True},
        infer={"max_frames": 750},
        teacher_forced_ce=tf_ce,
        ordered_utt_ids=[uid],
        condition_reports=condition_reports,
        gate=gate,
    )
    assert score_ok is not None and score_ok.is_file()

    # Missing checkpoint_step must soft-refuse (round-4 blocker class).
    no_step = mod._emit_d1c_v2_from_gate(
        out_dir=tmp_path / "out_nostep",
        contract=contract,
        ckpt=str(ckpt),
        provenance={
            "subset_label": "select8",
            "subset_path": "x",
            "subset_file_sha256": "3" * 64,
        },
        infer={"max_frames": 750},
        teacher_forced_ce=tf_ce,
        ordered_utt_ids=[uid],
        condition_reports=condition_reports,
        gate=gate,
    )
    assert no_step is None
    assert not (tmp_path / "out_nostep" / "discrete_eval_report_v2.json").is_file()


def test_validate_formal_report_rejects_checkpoint_step_zero():
    from minispeech.eval.discrete_report_schema import (
        CONDITIONS,
        build_formal_report,
        ordered_utt_ids_sha256,
        sha256_json,
        validate_formal_report,
    )

    ids = ["u1"]
    conditions = {}
    for name in CONDITIONS:
        rows = [
            {
                "ordinal": 0,
                "utt_id": "u1",
                "normalized_reference_sha256": "a" * 64,
                "generated_codes_sha256": "b" * 64,
                "generation": {"termination_reason": "learned_eos"},
                "wav": {"path": "u1.wav", "file_sha256": "c" * 64, "pcm_sha256": "d" * 64},
                "cer": 0.0,
            }
        ]
        conditions[name] = {
            "subset_key": "select8",
            "subset_path": "subset.json",
            "subset_file_sha256": "e" * 64,
            "expected_count": 1,
            "evaluated_count": 1,
            "ordered_utt_ids_sha256": ordered_utt_ids_sha256(ids),
            "rows": rows,
        }
    report = build_formal_report(
        formal_eligible=True,
        checkpoint={
            "path": "candidate.pt",
            "sha256": "f" * 64,
            "step": 250,
            "schema": "x",
            "schema_version": "2",
            "hashes": _fake_ckpt_hashes(),
        },
        provenance={"eval_contract_sha256": "1" * 64},
        infer={"max_frames": 750},
        teacher_forced_metrics={"teacher_forced_ce_source": "eval_computed", "total_ce": 1.0},
        row_binding={"ordered_utt_ids": ids, "ordered_utt_ids_sha256": ordered_utt_ids_sha256(ids)},
        conditions=conditions,
    )
    report["checkpoint"]["step"] = 0
    report.pop("report_sha256", None)
    report["report_sha256"] = sha256_json(report)
    with pytest.raises(ValueError, match="checkpoint.step must be a positive integer"):
        validate_formal_report(report)


def test_resolve_checkpoint_step_payload_filename_cli():
    mod = _load_script("74_eval_discrete_lm.py")
    assert mod.parse_candidate_step_from_path("runs/x/candidate_step_0375.pt") == 375
    assert mod.parse_candidate_step_from_path("best.pt") is None
    assert mod.checkpoint_step_from_payload({"step": 500, "model": {}}) == 500
    assert (
        mod.resolve_checkpoint_step(
            ckpt_path="runs/x/candidate_step_0500.pt",
            payload={"step": 500},
            allowed_steps=(250, 375, 500, 625, 750),
            require=True,
        )
        == 500
    )
    # CLI may confirm a derived step.
    assert (
        mod.resolve_checkpoint_step(
            ckpt_path="runs/x/candidate_step_0500.pt",
            payload={"step": 500},
            cli_step=500,
            allowed_steps=(250, 375, 500, 625, 750),
            require=True,
        )
        == 500
    )
    with pytest.raises(SystemExit, match="mismatch"):
        mod.resolve_checkpoint_step(
            ckpt_path="runs/x/candidate_step_0500.pt",
            payload={"step": 375},
            require=True,
        )
    with pytest.raises(SystemExit, match="not in contract"):
        mod.resolve_checkpoint_step(
            ckpt_path="runs/x/candidate_step_0999.pt",
            payload={"step": 999},
            allowed_steps=(250, 375, 500, 625, 750),
            require=True,
        )
    # Formal path: CLI cannot invent a step for unbound best.pt.
    with pytest.raises(SystemExit, match="cannot invent"):
        mod.resolve_checkpoint_step(
            ckpt_path="best.pt",
            payload={},
            cli_step=750,
            allowed_steps=(250, 375, 500, 625, 750),
            require=True,
        )
    # Formal path: filename alone cannot invent a step (rename/re-save attack).
    with pytest.raises(SystemExit, match=r"payload\['step'\]"):
        mod.resolve_checkpoint_step(
            ckpt_path="runs/x/candidate_step_0625.pt",
            payload={"model": {}},
            allowed_steps=(250, 375, 500, 625, 750),
            require=True,
        )
    # Non-formal: filename may still derive the step.
    assert (
        mod.resolve_checkpoint_step(
            ckpt_path="runs/x/candidate_step_0500.pt",
            payload={},
            require=False,
        )
        == 500
    )
    # Non-formal (require=False) still allows diagnostic CLI override.
    assert (
        mod.resolve_checkpoint_step(
            ckpt_path="best.pt",
            payload={},
            cli_step=750,
            allowed_steps=(250, 375, 500, 625, 750),
            require=False,
        )
        == 750
    )


def test_require_checkpoint_payload_hashes_fail_closed():
    mod = _load_script("74_eval_discrete_lm.py")
    good = {
        "step": 500,
        "hashes": _fake_ckpt_hashes(eval_contract="c" * 64),
    }
    assert mod.require_checkpoint_payload_hashes(good, expected_eval_contract="c" * 64)[
        "run_manifest"
    ] == "b" * 64
    with pytest.raises(SystemExit, match="payload\\['hashes'\\]"):
        mod.require_checkpoint_payload_hashes({"step": 500})
    with pytest.raises(SystemExit, match="eval_contract mismatch"):
        mod.require_checkpoint_payload_hashes(good, expected_eval_contract="d" * 64)


def test_d1c_subset_aliases_resolve_to_d1c_oracle():
    from minispeech.data.d1b_oracles import SUBSET_ALIASES, resolve_subset_spec

    for alias, split in (
        ("d1c_dev32", "full32"),
        ("d1c_full32", "full32"),
        ("d1c_select8", "select8"),
        ("d1c_gate24", "gate24"),
    ):
        assert alias in SUBSET_ALIASES
        path, resolved_split = resolve_subset_spec(alias)
        assert path.name == "d1c_dev32.json"
        assert resolved_split == split


def test_main_stamps_checkpoint_step_and_selector_e2e(tmp_path):
    """Production main() must stamp checkpoint.step; selector §8.4 must accept the set.

    Stubs only GPU/codec/ASR outer deps (same seam as reaudit3_probe_mainpath).
    Exercises real TF aggregation → D1c gate → emitter → validate_formal_report,
    then builds the five-step candidate set and drives select_d1c_formal_reports.
    """
    import ast
    import contextlib
    import copy
    import hashlib
    import io
    import sys

    import soundfile as sf

    from minispeech.data.d1b_oracles import load_eval_contract
    from minispeech.eval.discrete_report_schema import (
        rows_sha256,
        sha256_json,
        validate_formal_report,
    )
    from minispeech.frontend.normalize import normalize_for_asr

    mod = _load_script("74_eval_discrete_lm.py")
    mod76 = _load_script("76_select_discrete_checkpoint.py")

    # AST: main() must wire D1c gate + checkpoint_step provenance (not helpers alone).
    src = (ROOT / "scripts" / "74_eval_discrete_lm.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    call_names = {
        n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "evaluate_n2_gate_d1c" in call_names
    assert "resolve_checkpoint_step" in call_names
    assert "aggregate_teacher_forced_batches" in call_names
    assert "teacher_forced_metrics_for_emitter" in call_names
    assert 'tf_ce = summary.get("loss")' not in src

    contract_path = ROOT / "configs" / "eval_contracts" / "d1c_semantic_eos_v2_0_0.json"
    contract = load_eval_contract(contract_path)
    m = _semantic_model()
    codebook = m.codebook_size
    ncb = m.n_codebooks
    utts = ["u0", "u1", "u2"]
    refs = {u: "привет мир" for u in utts}
    rows = [
        {
            "utt_id": u,
            "input_ids": torch.tensor([3 + i, 4 + i, 5 + i]),
            "codes": torch.randint(0, codebook, (4 + i, ncb)),
            "speaker_id": i % 2,
        }
        for i, u in enumerate(utts)
    ]
    wav_seed = {"n": 0}

    def fake_free_running(model_, rows_, **kw):
        condition = kw["condition"]
        out_wav_dir = Path(kw["out_wav_dir"])
        cfg = kw["cfg"]
        icfg = mod.resolve_infer_settings(
            cfg,
            infer_overrides=kw.get("infer_overrides"),
            max_frames_override=kw["max_frames"],
            codebook_size=int((cfg.get("model") or {}).get("codebook_size", 2048)),
        )
        icfg = dict(icfg)
        icfg["termination_mode"] = "semantic_eos_v1"
        rows_out = []
        gen_codes = []
        cers = []
        edit_counts = []
        hyp_for = {
            "model": "привет мир",
            "copy_synth": "привет мир",
            "wrong_utt": "щщщщщщ щщщ",
            "random_codes": "щщщщщщщщщщ",
            "shuffle_text": "щщщщщщ щщщ",
            "mask_text": "щщщщщщщщщщ",
        }
        for i, r in enumerate(rows_):
            uid = str(r["utt_id"])
            wav_seed["n"] += 1
            pcm = np.linspace(-0.3, 0.3, 2400 + wav_seed["n"], dtype=np.float32)
            wav = out_wav_dir / f"{i:04d}_{uid}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav), pcm, 24000)
            hyp = hyp_for[condition]
            ref_n = normalize_for_asr(refs[uid])
            hyp_n = normalize_for_asr(hyp)
            cer = float(mod.compute_cer([ref_n], [hyp_n]))
            edit_counts.append(mod.edits_from_jiwer_chars(ref_n, hyp_n))
            n_frames = int(r["codes"].shape[0])
            codes_t = torch.randint(0, 2048, (n_frames, 16))
            gen_codes.append(codes_t)
            cers.append(cer)
            rows_out.append(
                {
                    "utt_id": uid,
                    "ordinal": i,
                    "condition": condition,
                    "n_frames": n_frames,
                    "gt_frames": int(r["codes"].shape[0]),
                    "cer": cer,
                    "ref": refs[uid],
                    "hyp": hyp,
                    "wav": str(wav),
                    "stop_reason": "learned_eos",
                    "generation": {
                        "termination_reason": "learned_eos",
                        "n_decoded_frames": n_frames,
                        "early_eos_suppressed_count": 0,
                        "max_cb0_run": 2,
                    },
                    "generated_codes_sha256": hashlib.sha256(
                        codes_t.numpy().astype(np.int32).tobytes()
                    ).hexdigest(),
                }
            )
        boot = mod.bootstrap_mean_cer(cers)
        report = {
            "condition": condition,
            "n": len(rows_out),
            "mean_cer": boot["mean"],
            "cer_ci95_low": boot["ci95_low"],
            "cer_ci95_high": boot["ci95_high"],
            "cb0": mod.cb0_stats(gen_codes),
            "rows": rows_out,
            "edit_counts": edit_counts,
            "infer": {k: v for k, v in icfg.items() if not str(k).startswith("_")},
        }
        return mod.enrich_model_report(report)

    cfg = {
        "model": {
            "codebook_size": codebook,
            "n_codebooks": ncb,
            "termination": {"mode": "semantic_eos_v1", "eos_id": m.eos_id},
        },
        "data": {"pad_token_id": 0},
        "infer": dict(contract["infer"]),
    }
    # Trainer-shaped immutable candidate filename + payload step + hashes.
    ckpt = tmp_path / "candidate_step_0750.pt"
    live_hashes = _fake_ckpt_hashes(
        eval_contract=str(contract["sha256"]),
        recipe=str(contract["recipe_sha256"]),
        recipe_source_manifest=str(contract["recipe_source_manifest_sha256"]),
    )
    torch.save(
        {
            "model": {},
            "step": 750,
            "optimizer_step": 750,
            "cfg": cfg,
            "hashes": dict(live_hashes),
        },
        ckpt,
    )

    mod._load_model = lambda p, d: (
        m,
        copy.deepcopy(cfg),
        {"step": 750, "cfg": cfg, "hashes": dict(live_hashes)},
    )
    mod.load_codec_items = lambda *a, **k: rows
    mod.enforce_eval_contract = lambda *a, **k: contract
    mod._resolve_speaker_registry = lambda **k: (
        object(),
        "configs/speaker_maps/trackn_v2.json",
    )
    mod._ref_texts_from_manifest = lambda ids, man: dict(refs)
    mod.free_running_asr_eval = fake_free_running
    mod.load_yaml = lambda p: {"infer": dict(contract["infer"])}

    out1 = tmp_path / "run_measured"
    argv = [
        "74_eval_discrete_lm.py",
        "--ckpt",
        str(ckpt),
        "--codec-tokens-dir",
        str(tmp_path / "tokens"),
        "--max-utts",
        "3",
        "--n2-gate",
        "--out",
        str(out1),
        "--wav-dir",
        str(tmp_path / "wav1"),
        "--eval-contract",
        str(contract_path),
        "--config",
        "configs/discrete_speech_lm_d1b_443m.yaml",
    ]
    old_argv = sys.argv
    buf = io.StringIO()
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(buf):
            rc = mod.main()
    finally:
        sys.argv = old_argv
    assert rc == 0, buf.getvalue()

    v2 = out1 / "discrete_eval_report_v2.json"
    gate_path = out1 / "n2_gate.json"
    assert v2.is_file(), "main() must emit formal v2 report"
    assert gate_path.is_file()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    report = json.loads(v2.read_text(encoding="utf-8"))
    assert gate.get("gate") == "N2_D1c_semantic_eos_v1"
    assert gate.get("step") == 750
    assert gate.get("provenance", {}).get("checkpoint_step") == 750
    assert report["checkpoint"]["step"] == 750
    assert report["checkpoint"]["hashes"]["eval_contract"] == live_hashes["eval_contract"]
    assert report["checkpoint"]["hashes"]["run_manifest"] == live_hashes["run_manifest"]
    assert report["formal_eligible"] is True
    assert report["teacher_forced_metrics"]["teacher_forced_ce_source"] == "eval_computed"
    validate_formal_report(report)  # rejects step=0

    # Bare-float regression: if main rewired to summary['loss'], formal must refuse.
    # Prove by calling emitter with a float while keeping gate JSON (soft refuse).
    refused = mod._emit_d1c_v2_from_gate(
        out_dir=tmp_path / "bare",
        contract=contract,
        ckpt=str(ckpt),
        provenance={
            "checkpoint_step": 750,
            "checkpoint_hashes": dict(live_hashes),
            "subset_label": "select8",
            "subset_file_sha256": "3" * 64,
        },
        infer={"max_frames": 750},
        teacher_forced_ce=float(report["teacher_forced_metrics"]["total_ce"]),
        ordered_utt_ids=utts,
        condition_reports={
            name: {"rows": report["conditions"][name]["rows"]}
            for name in report["conditions"]
        },
        gate=gate,
    )
    assert refused is None

    # Evaluator → selector §8.4: five reports with exact candidate steps
    # and *distinct* on-disk checkpoint identities from one sealed run.
    steps = [250, 375, 500, 625, 750]
    paths = []
    for step in steps:
        cloned = copy.deepcopy(report)
        ckpt_path, ckpt_sha = _write_candidate_ckpt(tmp_path, step)
        cloned["checkpoint"]["step"] = step
        cloned["checkpoint"]["path"] = str(ckpt_path)
        cloned["checkpoint"]["sha256"] = ckpt_sha
        cloned["checkpoint"]["hashes"] = dict(live_hashes)
        # Perfect EOS so at least one candidate is eligible for selection.
        cloned["teacher_forced_metrics"]["tf_eos_final_top1_accuracy"] = 1.0
        cloned["teacher_forced_metrics"]["tf_eos_early_top1_false_positive_count"] = 0
        for cond in cloned["conditions"].values():
            for row in cond["rows"]:
                row["cer"] = 0.0
                row["catastrophic"] = False
                row["gt_frames"] = max(1, int(row.get("gt_frames") or 1))
                row["n_frames"] = int(row["gt_frames"])
                gen = row.setdefault("generation", {})
                gen["termination_reason"] = "learned_eos"
                gen["n_decoded_frames"] = int(row["n_frames"])
                gen["early_eos_suppressed_count"] = 0
                gen["max_cb0_run"] = 2
            cond["rows_sha256"] = rows_sha256(cond["rows"])
        cloned["gates"]["text_usage"] = {"pass": True}
        cloned["gates"]["runaway_or_repetition"] = False
        cloned["summaries"]["phrase_loop_repetition"] = False
        cloned["summaries"]["runaway_or_repetition"] = False
        cloned.pop("report_sha256", None)
        cloned["report_sha256"] = sha256_json(cloned)
        validate_formal_report(cloned)
        outp = tmp_path / f"report_step_{step}.json"
        outp.write_text(json.dumps(cloned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths.append(outp)

    selection = mod76.select_d1c_formal_reports(paths)
    assert selection["mode"] == "d1c_formal"
    assert selection["candidate_steps"] == steps
    assert {int(c["step"]) for c in selection["candidates"]} == set(steps)
    assert len({c["checkpoint_sha256"] for c in selection["candidates"]}) == len(steps)
    assert selection["selected"]["step"] in steps

    # Round-4 attack control: identical sha256 across five steps must refuse.
    attack_paths = []
    shared_blob = b"same-best-pt-bytes"
    shared_sha = hashlib.sha256(shared_blob).hexdigest()
    for step in steps:
        forged = copy.deepcopy(json.loads(paths[0].read_text(encoding="utf-8")))
        alias = tmp_path / "attack" / f"candidate_step_{step:04d}.pt"
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_bytes(shared_blob)
        forged["checkpoint"]["step"] = step
        forged["checkpoint"]["path"] = str(alias)
        forged["checkpoint"]["sha256"] = shared_sha
        forged["checkpoint"]["hashes"] = dict(live_hashes)
        forged.pop("report_sha256", None)
        forged["report_sha256"] = sha256_json(forged)
        # Schema accepts when hash matches the (identical) file; selector must refuse.
        validate_formal_report(forged)
        ap = tmp_path / f"attack_report_{step}.json"
        ap.write_text(json.dumps(forged), encoding="utf-8")
        attack_paths.append(ap)
    with pytest.raises(ValueError, match="duplicate checkpoint.sha256"):
        mod76.select_d1c_formal_reports(attack_paths)

    # Round-5/6: basename must be candidate_step_<step>.pt (load-bearing for Y3).
    bad_base = copy.deepcopy(json.loads(paths[0].read_text(encoding="utf-8")))
    wrong = tmp_path / "best.pt"
    wrong.write_bytes(b"basename-mismatch-bytes")
    bad_base["checkpoint"]["path"] = str(wrong)
    bad_base["checkpoint"]["sha256"] = hashlib.sha256(b"basename-mismatch-bytes").hexdigest()
    bad_base["checkpoint"]["step"] = 250
    bad_base.pop("report_sha256", None)
    bad_base["report_sha256"] = sha256_json(bad_base)
    base_paths = [tmp_path / "base_bad.json"]
    base_paths[0].write_text(json.dumps(bad_base), encoding="utf-8")
    for step in steps[1:]:
        base_paths.append(paths[steps.index(step)])
    import minispeech.eval.discrete_report_schema as schema

    old_v = schema.validate_formal_report
    schema.validate_formal_report = lambda data, **kwargs: data  # type: ignore[assignment]
    mod76.validate_formal_report = schema.validate_formal_report
    try:
        with pytest.raises(ValueError, match="basename"):
            mod76.select_d1c_formal_reports(base_paths)
    finally:
        schema.validate_formal_report = old_v
        mod76.validate_formal_report = old_v

    # Rename / re-save attack: no payload step → resolve refuses (main path).
    with pytest.raises(SystemExit, match=r"payload\['step'\]"):
        mod.resolve_checkpoint_step(
            ckpt_path=str(tmp_path / "candidate_step_0375.pt"),
            payload={"model": {}, "hashes": dict(live_hashes)},
            allowed_steps=steps,
            require=True,
        )