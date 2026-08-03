"""Optimizer-step vs microbatch semantics for DiscreteLMTrainer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from minispeech.data.codec_seq_protocol import collate_codec_seq
from minispeech.train.discrete_lm_trainer import (
    STEP_UNIT_OPTIMIZER,
    DiscreteLMTrainer,
    _atomic_torch_save,
    freeze_disabled_ru_conditioning,
    resolve_candidate_steps,
    resolve_optimizer_step_budget,
)
from minispeech.train.distributed import cleanup_distributed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _discrete_lm_trainer_fakes import FakeDiscreteSpeechLM  # noqa: E402


class _FixedCodecDataset(Dataset):
    """Deterministic tiny codec batches for CPU trainer tests."""

    def __init__(self, n: int = 32, *, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for i in range(n):
            L = 4 + (i % 3)
            T = 5 + (i % 4)
            self.items.append(
                {
                    "utt_id": f"u{i:04d}",
                    "input_ids": torch.randint(1, 60, (L,), generator=g),
                    "codes": torch.randint(0, 32, (T, 16), generator=g),
                    "speaker_id": i % 3,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def _tiny_cfg(tmp_path: Path, **train_extra) -> dict:
    train = {
        "run_dir": str(tmp_path / "run"),
        "lr": 1e-3,
        "weight_decay": 0.0,
        "grad_accum": 4,
        "grad_clip": 0.0,
        "warmup_steps": 4,
        "lr_schedule": "constant",
        "min_lr_ratio": 1.0,
        "precision": "fp32",
        "log_every": 1,
        "save_every": 0,
        "eval_every": 0,
        "fr_checkpoint": {"enabled": False, "every": 0},
        "resume": False,
        "ddp": {"find_unused_parameters": False},
        "seed": 0,
    }
    train.update(train_extra)
    return {
        "model": {"d_model": 16, "n_layers": 1, "name": "fake_tiny"},
        "loss": {"cb0_weight": 1.0, "stop_weight": 1.0, "stop_pos_weight": 1.0},
        "train": train,
    }


def _make_trainer(
    tmp_path: Path,
    cfg: dict | None = None,
    *,
    model: FakeDiscreteSpeechLM | None = None,
) -> DiscreteLMTrainer:
    cleanup_distributed()
    cfg = cfg or _tiny_cfg(tmp_path)
    ds = _FixedCodecDataset(n=64, seed=0)
    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda b: collate_codec_seq(b, pad_token_id=0),
    )
    m = model or FakeDiscreteSpeechLM()
    return DiscreteLMTrainer(
        m,
        loader,
        None,
        cfg,
        torch.device("cpu"),
        enforce_clean_git=False,
    )


def test_eight_microbatches_yield_two_optimizer_updates(tmp_path: Path):
    trainer = _make_trainer(tmp_path, _tiny_cfg(tmp_path, grad_accum=4))
    trainer.train_steps(2, stage="S1")
    assert trainer.optimizer_step == 2
    assert trainer.global_step == 2
    assert trainer.microbatch_step == 8
    assert trainer._accum_counter == 0


def test_scheduler_advances_on_optimizer_step(tmp_path: Path):
    cfg = _tiny_cfg(
        tmp_path,
        grad_accum=4,
        warmup_steps=4,
        lr_schedule="constant",
        min_lr_ratio=1.0,
        lr=1.0,
    )
    trainer = _make_trainer(tmp_path, cfg)
    lrs: list[float] = []
    orig_step = trainer.opt.step

    def _step_and_record(*a, **k):
        out = orig_step(*a, **k)
        lrs.append(float(trainer.opt.param_groups[0]["lr"]))
        return out

    trainer.opt.step = _step_and_record  # type: ignore[method-assign]
    trainer.train_steps(2, stage="S1")
    assert len(lrs) == 2
    assert lrs[0] == pytest.approx(0.25)
    assert lrs[1] == pytest.approx(0.5)
    assert trainer.microbatch_step == 8


def test_checkpoints_only_on_optimizer_boundary(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path, grad_accum=4, save_every=1)
    trainer = _make_trainer(tmp_path, cfg)
    saves: list[tuple[int, int]] = []
    orig = trainer.save_checkpoint

    def _wrapped(stage, step, **kwargs):
        assert trainer._accum_counter == 0
        saves.append((int(step), int(trainer.microbatch_step)))
        return orig(stage, step, **kwargs)

    trainer.save_checkpoint = _wrapped  # type: ignore[method-assign]
    trainer.train_steps(2, stage="S1")
    assert len(saves) >= 2
    assert all(s[0] in (1, 2) for s in saves)
    ckpt = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["step_unit"] == STEP_UNIT_OPTIMIZER
    assert ckpt["optimizer_step"] == 2
    assert ckpt["microbatch_step"] == 8
    assert ckpt["grad_accum"] == 4


def test_mid_accum_save_refused(tmp_path: Path):
    trainer = _make_trainer(tmp_path)
    trainer._accum_counter = 2
    with pytest.raises(RuntimeError, match="optimizer boundary"):
        trainer.save_checkpoint("S1", 0)


def test_atomic_torch_save_success_replaces_and_cleans_tmp(tmp_path: Path, monkeypatch):
    dest = tmp_path / "last.pt"
    torch.save({"v": 0}, dest)
    payload = {"v": 1, "ok": True}
    seen_tmp: list[Path] = []
    real_save = torch.save

    def _tracking_save(obj, path, *a, **k):
        p = Path(path)
        if p.name.startswith(".last.pt.tmp."):
            seen_tmp.append(p)
        return real_save(obj, path, *a, **k)

    monkeypatch.setattr(torch, "save", _tracking_save)
    _atomic_torch_save(payload, dest)

    assert dest.is_file()
    assert seen_tmp, "expected write via pid-scoped tmp path"
    assert all(not p.exists() for p in seen_tmp)
    loaded = torch.load(dest, map_location="cpu", weights_only=False)
    assert loaded == payload


def test_atomic_torch_save_failure_preserves_dest_and_cleans_tmp(
    tmp_path: Path, monkeypatch
):
    dest = tmp_path / "last.pt"
    prior = {"v": "intact"}
    torch.save(prior, dest)
    tmp_seen: list[Path] = []

    def _boom(obj, path, *a, **k):
        p = Path(path)
        tmp_seen.append(p)
        p.write_bytes(b"partial-corrupt")
        raise OSError("simulated crash mid-save")

    monkeypatch.setattr(torch, "save", _boom)
    with pytest.raises(OSError, match="simulated crash"):
        _atomic_torch_save({"v": "new"}, dest)

    assert tmp_seen
    assert all(not p.exists() for p in tmp_seen)
    loaded = torch.load(dest, map_location="cpu", weights_only=False)
    assert loaded == prior


def test_freeze_ru_conditioning_when_disabled():
    m = FakeDiscreteSpeechLM(russian_conditioning=False)
    n = freeze_disabled_ru_conditioning(m)
    assert n > 0
    assert all(not p.requires_grad for p in m.ru_cond.parameters())


def test_resolve_optimizer_step_budget_primary_and_dev_extension():
    assert resolve_optimizer_step_budget({"train": {}}) == 750
    assert resolve_optimizer_step_budget({"train": {}}, {"steps": 750}) == 750
    assert (
        resolve_optimizer_step_budget(
            {"train": {"allow_dev_extension": True, "dev_extension_optimizer_steps": 1250}},
            {"name": "S1"},
        )
        == 1250
    )
    assert (
        resolve_optimizer_step_budget(
            {"train": {"optimizer_steps": 750}},
            {"dev_extension": True},
        )
        == 1250
    )
    assert (
        resolve_optimizer_step_budget(
            {
                "train": {
                    "allow_dev_extension": True,
                    "optimizer_steps_dev_extension": 1250,
                }
            },
            {"name": "S1"},
        )
        == 1250
    )


def test_resolve_candidate_steps_base_and_dev_ext():
    assert resolve_candidate_steps({"train": {}}) == (250, 375, 500, 625, 750)
    assert resolve_candidate_steps(
        {"train": {"candidate_steps": [250, 500, 750]}}
    ) == (250, 500, 750)
    assert resolve_candidate_steps(
        {"train": {"allow_dev_extension": True}}
    ) == (250, 375, 500, 625, 750, 1000, 1250)


def test_candidate_checkpoint_saved_at_boundary(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path, grad_accum=2, save_every=0, candidate_steps=[1, 2])
    trainer = _make_trainer(tmp_path, cfg)
    trainer.train_steps(2, stage="S1")
    assert (tmp_path / "run" / "candidate_step_0001.pt").is_file()
    assert (tmp_path / "run" / "candidate_step_0002.pt").is_file()
    assert (tmp_path / "run" / "hashes.json").is_file()
    assert (tmp_path / "run" / "resolved_config.yaml").is_file()


@pytest.mark.skipif(
    os.environ.get("MINISPEECH_DDP_SMOKE") != "1",
    reason="opt-in WORLD_SIZE=1 DDP init smoke; set MINISPEECH_DDP_SMOKE=1",
)
def test_ddp_world_size_1_smoke(tmp_path: Path):
    """WORLD_SIZE=1 distributed init (torchrun --standalone --nproc_per_node=1)."""
    cleanup_distributed()
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29611")
    try:
        cfg = _tiny_cfg(tmp_path, grad_accum=2, save_every=0)
        cfg["train"]["ddp"] = {"find_unused_parameters": False}
        trainer = _make_trainer(tmp_path, cfg)
        assert trainer.dist.distributed is True
        assert trainer.dist.world_size == 1
        trainer.train_steps(1, stage="S1")
        assert trainer.optimizer_step == 1
        assert trainer.microbatch_step == 2
    finally:
        cleanup_distributed()
        for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            os.environ.pop(k, None)
