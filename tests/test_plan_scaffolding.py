"""Focused tests for plan scaffolding (no GPU / no ASR models)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch

from minispeech.data.dataset import collate_minispeech, make_stop_target
from minispeech.eval.bootstrap import EditCounts, paired_bootstrap_delta
from minispeech.eval.dtw import dtw_align
from minispeech.frontend.normalize import normalize_for_asr


ROOT = Path(__file__).resolve().parents[1]


def test_normalize_for_asr_idempotent():
    s = "Приве́т, Мир! Ёлка — test."
    once = normalize_for_asr(s)
    twice = normalize_for_asr(once)
    assert once == twice
    assert "ё" not in once
    assert "," not in once
    assert once == once.lower()


def test_bootstrap_zero_delta_covers_zero():
    # Identical edit counts → Δ=0; CI must cover 0
    counts = [
        EditCounts(hits=10, substitutions=1, deletions=0, insertions=0),
        EditCounts(hits=8, substitutions=0, deletions=1, insertions=1),
        EditCounts(hits=12, substitutions=2, deletions=0, insertions=0),
    ] * 5
    res = paired_bootstrap_delta(counts, counts, n_boot=500, seed=0)
    assert abs(res.delta_pp) < 1e-9
    assert res.ci95_low <= 0.0 <= res.ci95_high


def test_dtw_shifted_copy_near_zero():
    x = torch.randn(20, 64)
    # shift by 1 frame (pad start)
    y = torch.cat([x[:1], x[:-1]], dim=0)
    r = dtw_align(x, y, band_frac=0.25, metric="l1")
    assert r.mean_cost < 0.35  # aligned L1 much smaller than naive
    assert abs(r.path_slope - 1.0) < 0.25


def test_stop_window_target_count():
    t = make_stop_target(n_frames=10, T=12, stop_window=3)
    assert int(t.sum().item()) == 3
    assert torch.all(t[7:10] == 1.0)
    assert torch.all(t[:7] == 0.0)


def test_stop_window_soft_ramp():
    t = make_stop_target(n_frames=10, T=12, stop_window=3, soft=True)
    assert torch.allclose(t[7:10], torch.tensor([1 / 3, 2 / 3, 1.0]))
    assert torch.all(t[:7] == 0.0)
    assert torch.all(t[10:] == 0.0)


def test_collate_stop_window():
    batch = [
        {
            "utt_id": "a",
            "z_target": torch.randn(6, 64),
            "text_hidden": torch.randn(3, 4, 1024),
            "n_frames": 6,
            "text_len": 4,
        }
    ]
    out = collate_minispeech(batch, stop_window=3)
    assert int(out["stop_target"][0, :6].sum().item()) == 3
    out_soft = collate_minispeech(batch, stop_window=3, soft_stop=True)
    assert torch.allclose(out_soft["stop_target"][0, 3:6], torch.tensor([1 / 3, 2 / 3, 1.0]))


def test_gitignore_not_ignoring_minispeech_data():
    target = ROOT / "src" / "minispeech" / "data" / "dataset.py"
    assert target.exists()
    proc = subprocess.run(
        ["git", "check-ignore", "-v", str(target.relative_to(ROOT))],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    # exit 1 = not ignored; exit 0 with output = ignored
    assert proc.returncode != 0, f"still ignored: {proc.stdout}"


def test_generate_kv_cache_matches_full():
    from minispeech.models.minispeech import MiniSpeech

    torch.manual_seed(0)
    m = MiniSpeech(mode="regression", residual_head=False)
    m.eval()
    th = torch.randn(1, 3, 5, 1024)
    tm = torch.ones(1, 5, dtype=torch.bool)
    z_kv, _ = m.generate(
        th, tm, max_frames=6, stop_consecutive=99, min_frac=1.0, max_frac=1.0, use_kv_cache=True
    )
    z_full, _ = m.generate(
        th, tm, max_frames=6, stop_consecutive=99, min_frac=1.0, max_frac=1.0, use_kv_cache=False
    )
    assert z_kv.shape == z_full.shape
    assert torch.allclose(z_kv, z_full, atol=1e-4, rtol=1e-4)
