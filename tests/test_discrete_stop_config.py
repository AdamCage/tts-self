"""D1b stop/EOS contract, shared infer resolver, text-init fail-closed, YAML band."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from minispeech.infer.discrete_config import (
    DiscreteConfigError,
    WAVE3C_INFER_DEFAULTS,
    contract_infer_from_eval_contract,
    infer_settings_match_wave3c,
    resolve_discrete_infer_settings,
    validate_discrete_infer_settings,
    validate_discrete_lm_config,
    wave3c_infer_defaults,
)
from minispeech.models.discrete_speech_lm import DiscreteSpeechLM, DiscreteSpeechLMConfig
from minispeech.models.text_init import (
    TextInitError,
    _is_scratch_key,
    apply_qwen_text_init,
)
from minispeech.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]
D1B_YAML = ROOT / "configs" / "discrete_speech_lm_d1b_443m.yaml"
EVAL_CONTRACT = ROOT / "configs" / "eval_contracts" / "d1b_n2_reliability.json"


def test_stop_target_on_last_valid_frame():
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, False, False, False],
            [True, True, True, True, True],
        ]
    )
    tgt = DiscreteSpeechLM.stop_targets_from_mask(mask)
    assert tgt.shape == mask.shape
    assert tgt[0].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert tgt[1].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert tgt[2].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_padded_frames_excluded_from_stop_loss():
    """Stop BCE must mask with frame_mask (mirrors DiscreteSpeechLM.forward)."""
    frame_mask = torch.tensor([[True, True, True, False, False]])
    stop_tgt = DiscreteSpeechLM.stop_targets_from_mask(frame_mask)
    assert stop_tgt[0].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]

    logits = torch.zeros(1, 5)
    logits_pad = logits.clone()
    logits_pad[:, 3:] = 50.0  # huge on pads — must not affect masked loss
    pw = torch.tensor([5.0])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, stop_tgt, pos_weight=pw, reduction="none"
    )
    bce_pad = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_pad, stop_tgt, pos_weight=pw, reduction="none"
    )
    denom = frame_mask.float().sum().clamp_min(1.0)
    loss = (bce * frame_mask.float()).sum() / denom
    loss_pad = (bce_pad * frame_mask.float()).sum() / denom
    assert torch.allclose(loss, loss_pad)
    assert not torch.allclose(bce_pad.mean(), loss_pad)


def test_forced_stop_deterministic():
    """Forced stop_head fire → generate length == min_frames (deterministic)."""
    cfg = DiscreteSpeechLMConfig.tiny()
    m = DiscreteSpeechLM(cfg)
    m.eval()
    packed = m.build_inputs(
        torch.randint(1, 40, (1, 3)),
        torch.randint(0, cfg.codebook_size, (1, 2, cfg.n_codebooks)),
    )
    assert "position_ids" in packed
    with torch.no_grad():
        m.stop_head[-1].weight.zero_()
        m.stop_head[-1].bias.fill_(10.0)
    ids = torch.randint(1, 40, (1, 4))
    lengths = []
    for _ in range(3):
        codes = m.generate(
            ids,
            max_frames=30,
            temperature=0.0,
            top_k=0,
            stop_threshold=0.5,
            stop_consecutive=1,
            min_frames=4,
            max_cb0_repeat=100,
            stop_overrun_bias=0.0,
            frames_per_text_token=0.01,
            stop_earliest_frac=0.0,
            eos_cb0=None,
        )
        lengths.append(int(codes.shape[1]))
    assert lengths == [4, 4, 4]


def test_resolve_infer_scripts_73_74_identical():
    from minispeech.eval.discrete_n2 import resolve_infer_settings as resolve_n2

    cfg = load_yaml(D1B_YAML)
    a = resolve_discrete_infer_settings(cfg)
    b = resolve_discrete_infer_settings(cfg)
    assert a == b
    # Script 74 / discrete_n2 must call the shared resolver (no CLI stub).
    via_n2 = resolve_n2(cfg)
    assert via_n2 == a
    assert "_resolver" not in via_n2
    # Simulate script 73 (optional CLI max_frames) vs 74 (positional max_frames).
    from_73 = resolve_discrete_infer_settings(cfg, max_frames_override=None)
    from_74 = resolve_discrete_infer_settings(cfg, max_frames_override=750)
    assert from_73 == from_74
    # Empty infer block → WAVE3c defaults (fixes old 0.5 / 0.95 fork).
    bare = resolve_discrete_infer_settings({})
    assert bare["stop_overrun_bias"] == WAVE3C_INFER_DEFAULTS["stop_overrun_bias"]
    assert bare["stop_earliest_frac"] == WAVE3C_INFER_DEFAULTS["stop_earliest_frac"]
    assert resolve_n2({}) == bare


def test_wave3c_round_trip_eval_contract():
    assert EVAL_CONTRACT.is_file(), f"missing preflight contract: {EVAL_CONTRACT}"
    contract = json.loads(EVAL_CONTRACT.read_text(encoding="utf-8"))
    from_contract = contract_infer_from_eval_contract(contract)
    cfg = load_yaml(D1B_YAML)
    from_yaml = resolve_discrete_infer_settings(cfg)
    # Ignore max_frames if contract omits it; remaining WAVE3c knobs must match.
    diffs = []
    for key in WAVE3C_INFER_DEFAULTS:
        if key == "max_frames" and "max_frames" not in (contract.get("infer") or {}):
            continue
        if from_contract[key] != from_yaml[key]:
            diffs.append((key, from_contract[key], from_yaml[key]))
    assert not diffs, f"YAML vs eval_contract infer mismatch: {diffs}"
    assert infer_settings_match_wave3c(from_yaml, ignore_max_frames=False) == []


def test_eos_cb0_outside_vocab_rejected():
    with pytest.raises(DiscreteConfigError, match="eos_cb0"):
        validate_discrete_infer_settings(
            {**wave3c_infer_defaults(), "eos_cb0": 2048},
            codebook_size=2048,
        )
    with pytest.raises(DiscreteConfigError, match="eos_cb0"):
        resolve_discrete_infer_settings({"infer": {"eos_cb0": 2048}})


def test_validate_d1b_yaml():
    cfg = load_yaml(D1B_YAML)
    validate_discrete_lm_config(cfg)
    assert cfg["model"]["vocab_size"] == 248077
    assert cfg["model"]["pad_token_id"] == 248044
    assert cfg["model"]["text_init"]["enabled"] is False
    assert cfg["model"]["russian_conditioning"]["enabled"] is False
    assert cfg["train"]["step_unit"] == "optimizer_step"


def test_production_yaml_param_count_in_band():
    cfg_dict = load_yaml(D1B_YAML)
    model_cfg = DiscreteSpeechLMConfig.from_dict(cfg_dict)
    assert model_cfg.vocab_size == 248077
    assert model_cfg.max_speakers == 1089
    try:
        with torch.device("meta"):
            m = DiscreteSpeechLM(model_cfg)
        n = sum(p.numel() for p in m.parameters())
    except Exception:
        m = DiscreteSpeechLM(model_cfg)
        n = m.count_parameters()["total"]
    assert 430_000_000 <= n <= 460_000_000, f"param_count={n}"


def test_text_init_scratch_includes_stop_and_modality():
    assert _is_scratch_key("stop_head.0.weight")
    assert _is_scratch_key("stop_head")
    assert _is_scratch_key("modality_speech")
    assert not _is_scratch_key("text_emb.weight")


def test_text_init_vocab_mismatch_fails_closed():
    cfg = DiscreteSpeechLMConfig.tiny()
    cfg.vocab_size = 64
    m = DiscreteSpeechLM(cfg)
    # Fake Qwen state with larger embed rows and a mappable norm (won't reach
    # coverage if we fail earlier on vocab).
    src = {
        "model.embed_tokens.weight": torch.randn(128, cfg.d_model),
        "model.norm.weight": torch.ones(cfg.d_model),
    }
    with pytest.raises(TextInitError, match="vocab mismatch"):
        apply_qwen_text_init(
            m,
            model_id="fake-qwen",
            src_state=src,
            min_coverage=0.0,
        )


def test_text_init_silent_truncation_requires_unsafe_flag():
    cfg = DiscreteSpeechLMConfig.tiny()
    m = DiscreteSpeechLM(cfg)
    src = {
        # Bypass early vocab check by omitting embed; force shape mismatch path
        # via direct text_emb candidate with wrong rows after setting expected
        # equal... use allow path with matching expected skipped:
        "model.embed_tokens.weight": torch.randn(cfg.vocab_size + 10, cfg.d_model),
    }
    with pytest.raises(TextInitError, match="vocab mismatch"):
        apply_qwen_text_init(
            m, model_id="fake", src_state=src, min_coverage=0.0
        )
    # Explicit unsafe: truncation allowed; coverage may still fail → set floor 0.
    report = apply_qwen_text_init(
        m,
        model_id="fake",
        src_state=src,
        allow_unsafe_vocab_truncation=True,
        min_coverage=0.0,
    )
    assert any("truncated" in k or "[:" in k for k in report.mapped_keys)


def test_text_init_low_coverage_fails():
    cfg = DiscreteSpeechLMConfig.tiny()
    m = DiscreteSpeechLM(cfg)
    src = {"unrelated.weight": torch.randn(3, 3)}
    with pytest.raises(TextInitError, match="coverage"):
        apply_qwen_text_init(
            m,
            model_id="fake",
            src_state=src,
            min_coverage=0.05,
        )


def test_d1b_yaml_parses_and_has_honest_name():
    text = D1B_YAML.read_text(encoding="utf-8")
    assert "443" in text
    cfg = yaml.safe_load(text)
    assert cfg["model"]["vocab_size"] == 248077
    run_name = str(cfg["train"].get("run_name") or "")
    run_dir = str(cfg["train"].get("run_dir") or "")
    assert "d1b" in run_name.lower() or "d1b" in run_dir.lower()
    assert "${run_name}" in run_dir or "d1b" in run_dir.lower()
