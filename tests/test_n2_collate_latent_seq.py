"""A3 acceptance: latent-seq collator masks + padding-invariant loss."""

from __future__ import annotations

import torch
import yaml

from minispeech.constants import get_target
from minispeech.data.collate_latent_seq import (
    collate_latent_seq,
    curriculum_max_dur_s,
    filter_indices_by_duration,
    items_compatible_with_multicorpus,
    masked_l1_latent_loss,
    parse_curriculum,
)


def _item(utt: str, n_text: int, n_frames: int, *, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    fr = get_target("vibevoice64").frame_rate
    return {
        "utt_id": utt,
        "input_ids": torch.randint(1, 100, (n_text,), generator=g),
        "z_target": torch.randn(n_frames, 64, generator=g),
        "dur_s": n_frames / fr,
        "sample_weight": 1.0,
        "corpus": "ruslan",
        "speaker_id": "ruslan",
    }


def test_config_batch_and_curriculum() -> None:
    with open("configs/n2_qwen_latent_v01.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    batch = cfg["train"]["batch"]
    assert batch["mode"] == "auto"
    assert batch["candidates"] == [1, 2, 4, 8]
    assert batch["target_vram_gb"] == 14.5
    stages = parse_curriculum(cfg["train"]["curriculum"])
    assert stages[0].max_dur_s == 3.0
    assert stages[-1].max_dur_s == 15.0
    assert curriculum_max_dur_s(stages, 0) == 3.0
    assert curriculum_max_dur_s(stages, 1000) == 8.0
    assert curriculum_max_dur_s(stages, 5000) == 15.0


def test_collate_masks_and_shapes() -> None:
    a = _item("a", n_text=3, n_frames=4, seed=1)
    b = _item("b", n_text=7, n_frames=2, seed=2)
    out = collate_latent_seq([a, b], pad_token_id=0)
    assert out["input_ids"].shape == (2, 7)
    assert out["z_target"].shape == (2, 4, 64)
    assert out["z_input"].shape == (2, 4, 64)
    assert out["text_mask"].shape == (2, 7)
    assert out["frame_mask"][0].sum() == 4
    assert out["frame_mask"][1].sum() == 2
    assert out["seq_mask"].shape == (2, 11)  # L+T
    assert out["is_acoustic"][0, 7:].sum() == 4
    # AR shift: z_input[0,1] == z_target[0,0]
    assert torch.allclose(out["z_input"][0, 1], out["z_target"][0, 0])


def test_loss_equal_under_batch_order() -> None:
    short = _item("short", n_text=2, n_frames=3, seed=10)
    long = _item("long", n_text=9, n_frames=12, seed=11)
    b1 = collate_latent_seq([short, long])
    b2 = collate_latent_seq([long, short])
    # Predict zeros — masked L1 must be order-invariant.
    pred1 = torch.zeros_like(b1["z_target"])
    pred2 = torch.zeros_like(b2["z_target"])
    l1 = masked_l1_latent_loss(pred1, b1["z_target"], b1["frame_mask"])
    l2 = masked_l1_latent_loss(pred2, b2["z_target"], b2["frame_mask"])
    assert torch.allclose(l1, l2, rtol=0, atol=1e-6)

    # Non-zero deterministic pred per utt (map by utt_id) also matches.
    def pred_for(batch: dict) -> torch.Tensor:
        p = torch.zeros_like(batch["z_target"])
        for i, uid in enumerate(batch["utt_ids"]):
            src = short if uid == "short" else long
            t = src["z_target"].shape[0]
            p[i, :t] = 0.5 * src["z_target"]
        return p

    l3 = masked_l1_latent_loss(pred_for(b1), b1["z_target"], b1["frame_mask"])
    l4 = masked_l1_latent_loss(pred_for(b2), b2["z_target"], b2["frame_mask"])
    assert torch.allclose(l3, l4, rtol=0, atol=1e-6)


def test_curriculum_filter() -> None:
    durs = [1.0, 4.0, 10.0, 16.0]
    assert filter_indices_by_duration(durs, 3.0) == [0]
    assert filter_indices_by_duration(durs, 15.0) == [0, 1, 2]
    assert filter_indices_by_duration(durs, 16.0) == [0, 1, 2, 3]


def test_multicorpus_adapter() -> None:
    mc = {
        "utt_id": "x",
        "z_target": torch.zeros(5, 64),
        "text_hidden": torch.zeros(3, 2, 8),
        "n_frames": 5,
        "input_ids": torch.tensor([1, 2, 3]),
        "sample_weight": 0.5,
        "corpus": "ruslan",
    }
    item = items_compatible_with_multicorpus(mc)
    out = collate_latent_seq([item])
    assert out["input_ids"].shape[1] == 3
    assert out["frame_mask"].sum() == 5
