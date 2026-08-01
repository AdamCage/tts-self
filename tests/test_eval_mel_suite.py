"""CPU-only tests for mel eval suite test-set loading + metric aggregation."""

from __future__ import annotations

import json
from pathlib import Path

from minispeech.eval.optional_metrics import run_optional_metrics, score_gigaam_ctc
from minispeech.eval.test_sets import (
    aggregate_cer_wer,
    load_suite_config,
    load_suite_test_sets,
    load_test_set,
)


REPO = Path(__file__).resolve().parents[1]


def test_load_demo5_phrases():
    ts = load_test_set(REPO / "configs/eval_sets/demo5.json")
    assert ts.id == "demo5"
    assert ts.kind == "phrases"
    assert ts.n == 5
    assert "TTS" in ts.items[0].text or "tts" in ts.items[0].text.lower()
    assert len(ts.texts()) == 5


def test_load_homograph_and_bank():
    h = load_test_set(REPO / "configs/eval_sets/homograph.json")
    b = load_test_set(REPO / "configs/eval_sets/bank_entity.json")
    assert h.n >= 6
    assert b.n >= 6
    assert any("\u0301" in it.text or "руб" in it.text.lower() for it in list(h.items) + list(b.items))


def test_load_val100_utt_ids():
    ts = load_test_set(REPO / "configs/eval_sets/val100.json")
    assert ts.kind == "utt_ids"
    assert ts.n == 100
    assert ts.utt_ids()[0].startswith("ruslan_")
    assert ts.utt_ids()[-1].startswith("ruslan_")


def test_suite_yaml_lists_sets():
    cfg = load_suite_config(REPO / "configs/eval_mel_suite.yaml")
    sets = load_suite_test_sets(cfg)
    assert set(sets) >= {"demo5", "val100", "homograph", "bank_entity"}
    prod = cfg["production"]
    assert prod["free_demo_cer"] < 0.1
    assert prod["val100_cer"] < 0.1
    assert prod["cfm_parked"] is True


def test_aggregate_cer_wer():
    rows = [
        {"cer": 0.0, "wer": 0.0},
        {"cer": 0.1, "wer": 0.2},
        {"cer": 0.2, "wer": 0.4},
    ]
    agg = aggregate_cer_wer(rows)
    assert agg["n"] == 3
    assert abs(float(agg["mean_cer"]) - 0.1) < 1e-9
    assert abs(float(agg["median_cer"]) - 0.1) < 1e-9
    assert abs(float(agg["mean_wer"]) - 0.2) < 1e-9


def test_aggregate_empty():
    agg = aggregate_cer_wer([])
    assert agg["n"] == 0
    assert agg["mean_cer"] is None


def test_optional_metrics_skip_missing():
    import numpy as np

    wav = np.zeros(22050, dtype=np.float32)
    out = run_optional_metrics(
        ["gigaam_ctc", "utmos_v2", "nisqa", "unknown_metric"],
        hyp_wav=wav,
        sr=22050,
        text="привет",
    )
    assert len(out) == 4
    assert [r["metric"] for r in out] == [
        "gigaam_ctc",
        "utmos_v2",
        "nisqa",
        "unknown_metric",
    ]
    by_metric = {r["metric"]: r for r in out}
    assert by_metric["gigaam_ctc"]["skipped"] is True
    assert by_metric["nisqa"]["skipped"] is True
    assert by_metric["unknown_metric"]["skipped"] is True
    utmos = by_metric["utmos_v2"]
    assert utmos["skipped"] is True or np.isfinite(float(utmos["value"]))
    g = score_gigaam_ctc(wav, 22050, text="x")
    assert g["skipped"] is True


def test_hifigan_speechbrain_key_remap():
    from minispeech.infer.hifigan import remap_speechbrain_hifigan_keys

    import torch

    raw = {
        "conv_pre.conv.weight_g": torch.zeros(1),
        "ups.0.conv.weight_v": torch.zeros(1),
        "resblocks.0.convs1.0.conv.bias": torch.zeros(1),
    }
    mapped = remap_speechbrain_hifigan_keys(raw)
    assert "conv_pre.weight_g" in mapped
    assert "ups.0.weight_v" in mapped
    assert "resblocks.0.convs1.0.bias" in mapped


def test_seed_production_script(tmp_path: Path):
    """Import runner seed path without GPU."""
    import importlib.util

    path = REPO / "scripts/32_eval_mel_suite.py"
    spec = importlib.util.spec_from_file_location("eval_mel_suite_32", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = load_suite_config(REPO / "configs/eval_mel_suite.yaml")
    out = tmp_path / "seed.json"
    # Patch expand-relative paths by chdir to repo
    import os

    prev = os.getcwd()
    try:
        os.chdir(REPO)
        payload = mod._seed_production(cfg, out)
    finally:
        os.chdir(prev)
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "seed_production"
    assert payload["production"]["val100_cer"] is not None
    assert "how_to_run" in data
