"""Fail-closed resume + optimizer-boundary equivalence for DiscreteLMTrainer."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from minispeech.data.codec_seq_protocol import collate_codec_seq
from minispeech.train.discrete_lm_trainer import (
    STEP_UNIT_MICROBATCH,
    STEP_UNIT_OPTIMIZER,
    DiscreteLMTrainer,
    ResumeMismatchError,
)
from minispeech.train.distributed import cleanup_distributed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _discrete_lm_trainer_fakes import FakeDiscreteSpeechLM  # noqa: E402


class _FixedCodecDataset(Dataset):
    def __init__(self, n: int = 64, *, seed: int = 0):
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
        "warmup_steps": 0,
        "lr_schedule": "constant",
        "min_lr_ratio": 1.0,
        "precision": "fp32",
        "log_every": 100,
        "save_every": 0,
        "eval_every": 0,
        "fr_checkpoint": {"enabled": False, "every": 0},
        "resume": False,
        "ddp": {"find_unused_parameters": False},
        "speaker_map_hash": "spkmap_test_v1",
        "subset_hash": "subset_test_v1",
        "eval_contract_hash": "eval_test_v1",
        "seed": 0,
    }
    train.update(train_extra)
    return {
        "model": {"d_model": 16, "n_layers": 1, "name": "fake_tiny"},
        "loss": {"cb0_weight": 1.0, "stop_weight": 1.0, "stop_pos_weight": 1.0},
        "train": train,
    }


def _loader() -> DataLoader:
    return DataLoader(
        _FixedCodecDataset(n=64, seed=0),
        batch_size=2,
        shuffle=False,
        collate_fn=lambda b: collate_codec_seq(b, pad_token_id=0),
    )


def _trainer(
    tmp_path: Path,
    cfg: dict,
    *,
    model: FakeDiscreteSpeechLM | None = None,
) -> DiscreteLMTrainer:
    cleanup_distributed()
    torch.manual_seed(0)
    m = model or FakeDiscreteSpeechLM()
    return DiscreteLMTrainer(
        m,
        _loader(),
        None,
        cfg,
        torch.device("cpu"),
        enforce_clean_git=False,
    )


def _state_fingerprint(trainer: DiscreteLMTrainer) -> dict[str, torch.Tensor]:
    bare = trainer._bare()
    out = {f"m.{k}": v.detach().cpu().clone() for k, v in bare.state_dict().items()}
    for i, group in enumerate(trainer.opt.state_dict()["state"].values()):
        for k, v in group.items():
            if torch.is_tensor(v):
                out[f"o{i}.{k}"] = v.detach().cpu().clone()
    out["_optimizer_step"] = torch.tensor([trainer.optimizer_step])
    out["_microbatch_step"] = torch.tensor([trainer.microbatch_step])
    return out


def _assert_state_close(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert set(a) == set(b)
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6, rtol=1e-5), k


def test_resume_matches_uninterrupted(tmp_path: Path):
    n, k = 4, 2
    cfg = _tiny_cfg(tmp_path)
    torch.manual_seed(123)
    base_sd = FakeDiscreteSpeechLM().state_dict()

    m1 = FakeDiscreteSpeechLM()
    m1.load_state_dict(deepcopy(base_sd))
    t1 = _trainer(tmp_path / "a", cfg, model=m1)
    t1.train_steps(n, stage="S1")
    ref = _state_fingerprint(t1)

    run_b = tmp_path / "b"
    cfg_b = _tiny_cfg(run_b)
    m2 = FakeDiscreteSpeechLM()
    m2.load_state_dict(deepcopy(base_sd))
    t2 = _trainer(run_b, cfg_b, model=m2)
    t2.train_steps(k, stage="S1")
    t2.save_checkpoint("S1", t2.optimizer_step)
    ckpt_path = run_b / "run" / "last.pt"

    m3 = FakeDiscreteSpeechLM()
    cfg_c = _tiny_cfg(tmp_path / "c")
    cfg_c["train"]["resume"] = str(ckpt_path)
    t3 = _trainer(tmp_path / "c", cfg_c, model=m3)
    assert t3.optimizer_step == k
    t3.train_steps(n, stage="S1")
    got = _state_fingerprint(t3)
    _assert_state_close(ref, got)


def test_legacy_microbatch_rejected_without_opt_in(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path)
    t = _trainer(tmp_path, cfg)
    t.train_steps(1, stage="S1")
    path = tmp_path / "run" / "legacy.pt"
    payload = t._ckpt_payload("S1", t.optimizer_step)
    payload.pop("step_unit")
    payload.pop("optimizer_step")
    payload.pop("microbatch_step")
    payload["global_step"] = 12
    payload["hashes"] = dict(t.repro_hashes)
    torch.save(payload, path)

    cfg2 = _tiny_cfg(tmp_path / "r2")
    cfg2["train"]["resume"] = str(path)
    with pytest.raises(ResumeMismatchError, match="legacy checkpoint"):
        _trainer(tmp_path / "r2", cfg2)


def test_legacy_microbatch_migrates_with_opt_in(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path)
    t = _trainer(tmp_path, cfg)
    t.train_steps(1, stage="S1")
    path = tmp_path / "run" / "legacy.pt"
    payload = t._ckpt_payload("S1", t.optimizer_step)
    payload.pop("step_unit")
    payload.pop("optimizer_step")
    payload["legacy_step_unit"] = STEP_UNIT_MICROBATCH
    payload["global_step"] = 12
    payload["microbatch_step"] = 12
    payload["hashes"] = dict(t.repro_hashes)
    torch.save(payload, path)

    cfg2 = _tiny_cfg(tmp_path / "r2", allow_legacy_microbatch_resume=True)
    cfg2["train"]["resume"] = str(path)
    t2 = _trainer(tmp_path / "r2", cfg2)
    assert t2.optimizer_step == 3
    assert t2.microbatch_step == 12
    assert t2.global_step == 3


def test_resume_fail_closed_on_grad_accum(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path, grad_accum=4)
    t = _trainer(tmp_path, cfg)
    t.train_steps(1, stage="S1")
    path = tmp_path / "run" / "last.pt"
    assert path.is_file()

    cfg2 = _tiny_cfg(tmp_path / "r2", grad_accum=2)
    cfg2["train"]["resume"] = str(path)
    with pytest.raises(ResumeMismatchError, match="grad_accum"):
        _trainer(tmp_path / "r2", cfg2)


def test_resume_fail_closed_on_speaker_map_hash(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path)
    t = _trainer(tmp_path, cfg)
    t.train_steps(1, stage="S1")
    path = tmp_path / "run" / "last.pt"

    cfg2 = _tiny_cfg(tmp_path / "r2", speaker_map_hash="different_map")
    cfg2["train"]["resume"] = str(path)
    with pytest.raises(ResumeMismatchError, match="speaker_map"):
        _trainer(tmp_path / "r2", cfg2)


def test_resume_fail_closed_on_subset_hash(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path)
    t = _trainer(tmp_path, cfg)
    t.train_steps(1, stage="S1")
    path = tmp_path / "run" / "last.pt"

    cfg2 = _tiny_cfg(tmp_path / "r2", subset_hash="other_subset")
    cfg2["train"]["resume"] = str(path)
    with pytest.raises(ResumeMismatchError, match="subset"):
        _trainer(tmp_path / "r2", cfg2)


def test_resume_fail_closed_on_model_config_hash(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path)
    t = _trainer(tmp_path, cfg)
    t.train_steps(1, stage="S1")
    path = tmp_path / "run" / "last.pt"

    cfg2 = _tiny_cfg(tmp_path / "r2")
    cfg2["model"] = {"d_model": 16, "n_layers": 3, "name": "fake_tiny"}
    cfg2["train"]["resume"] = str(path)
    with pytest.raises(ResumeMismatchError, match="model_config"):
        _trainer(tmp_path / "r2", cfg2)


def test_checkpoint_stores_both_counters_and_hashes(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path, save_every=1)
    t = _trainer(tmp_path, cfg)
    t.train_steps(1, stage="S1")
    ckpt = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["step_unit"] == STEP_UNIT_OPTIMIZER
    assert ckpt["optimizer_step"] == 1
    assert ckpt["microbatch_step"] == 4
    assert ckpt["grad_accum"] == 4
    assert "rng" in ckpt
    assert "optimizer" in ckpt
    assert ckpt["hashes"]["speaker_map"] == "spkmap_test_v1"
    assert ckpt["hashes"]["subset"] == "subset_test_v1"
    assert ckpt["hashes"]["eval_contract"] == "eval_test_v1"
    assert ckpt["hashes"]["model_config"]


def _ddp_worker(rank: int, world_size: int, tmp: str, result_q) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29731"
    cleanup_distributed()
    try:
        cfg = _tiny_cfg(Path(tmp) / f"rank{rank}", grad_accum=2)
        cfg["train"]["ddp"] = {"find_unused_parameters": False}
        assert cfg["train"]["speaker_map_hash"] == "spkmap_test_v1"
        t = _trainer(Path(tmp) / f"rank{rank}", cfg)
        assert t.dist.distributed
        assert t.dist.world_size == world_size
        t.train_steps(1, stage="S1")
        result_q.put(
            {
                "rank": rank,
                "optimizer_step": t.optimizer_step,
                "microbatch_step": t.microbatch_step,
                "speaker_map": t.repro_hashes["speaker_map"],
                "ok": True,
            }
        )
    except Exception as exc:  # noqa: BLE001
        result_q.put({"rank": rank, "ok": False, "error": str(exc)})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        cleanup_distributed()
        for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            os.environ.pop(k, None)


@pytest.mark.skipif(
    os.environ.get("MINISPEECH_DDP_GLOO") != "1",
    reason="CPU/gloo two-process test is opt-in (MINISPEECH_DDP_GLOO=1); CI stability",
)
def test_ddp_gloo_two_process_smoke(tmp_path: Path):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    world = 2
    procs = [
        ctx.Process(target=_ddp_worker, args=(r, world, str(tmp_path), q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    results = [q.get(timeout=120) for _ in range(world)]
    for p in procs:
        p.join(timeout=60)
    assert all(r.get("ok") for r in results), results
    assert {r["speaker_map"] for r in results} == {"spkmap_test_v1"}
    assert all(r["optimizer_step"] == 1 for r in results)
    assert all(r["microbatch_step"] == 2 for r in results)
