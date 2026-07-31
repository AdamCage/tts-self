"""Trainer for Track A non-AR mel baseline."""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from minispeech.data.mel_dataset import MelSpeechDataset, collate_mel
from minispeech.models.mel_baseline import (
    MelBaseline,
    mel_l1_loss,
    phone_duration_loss,
    phone_scalar_loss,
)
from minispeech.train.distributed import (
    barrier,
    cleanup_distributed,
    device_for_rank,
    get_dist_info,
    init_distributed,
    is_distributed,
    is_main_process,
    unwrap_model,
    wrap_ddp,
)
from minispeech.train.ema import EMA
from minispeech.train.schedule import lr_multiplier, set_optimizer_lr
from minispeech.utils import expand, write_run_manifest


def cer_selection_key(metrics: dict[str, Any]) -> tuple:
    """Checkpoint ranking: prefer lower CER when present (higher tuple = better).

    Interface only — callers may pass ``{\"cer\": float}`` from an ASR eval hook
    even if CER is not computed yet during early scaffolding.
    Falls back to negative mel L1 when CER missing.
    """
    if metrics.get("cer") is not None:
        return (-float(metrics["cer"]), -float(metrics.get("mel_l1", 0.0)))
    return (0.0, -float(metrics.get("mel_l1", 1e9)))


def _capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_cpu_byte_tensor(value: Any) -> torch.Tensor:
    """torch.load may revive RNG bytes as plain tensors/ndarrays — normalize."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().to(dtype=torch.uint8)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return torch.frombuffer(bytearray(value), dtype=torch.uint8).clone()
    return torch.as_tensor(np.asarray(value), dtype=torch.uint8).cpu().contiguous()


def _restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_cpu_byte_tensor(state["torch"]))
    if "cuda" in state and torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([_as_cpu_byte_tensor(s) for s in state["cuda"]])


def resolve_resume_path(cfg: dict[str, Any], resume: str | Path | bool | None) -> Path | None:
    """Resolve a checkpoint path from an explicit arg or ``train.resume``."""
    tcfg = cfg.get("train", {})
    raw = resume if resume is not None else tcfg.get("resume")
    if raw is None or raw is False:
        return None
    if raw is True:
        run_dir = expand(tcfg.get("run_dir", "runs/mel_baseline_v01"))
        for name in ("S1_last.pt", "S2_last.pt", "S3_last.pt", "S4_last.pt", "last.pt"):
            cand = run_dir / name
            if cand.is_file():
                return cand
        best = run_dir / "best.pt"
        return best if best.is_file() else None
    path = expand(raw)
    return path if path.is_file() else None


class MelTrainer:
    def __init__(
        self,
        model: MelBaseline,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        cfg: dict[str, Any],
        device: torch.device,
        *,
        cer_eval_fn: Callable[[MelBaseline], dict[str, Any]] | None = None,
        resume: str | Path | bool | None = None,
    ):
        # DDP: init from env when launched under torchrun; no-op otherwise.
        self.dist = init_distributed()
        self.device = device_for_rank(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cer_eval_fn = cer_eval_fn

        bare = model.to(self.device)
        tcfg = cfg.get("train", {})
        self.base_lr = float(tcfg.get("lr", 2e-4))
        self.opt = torch.optim.AdamW(
            bare.parameters(),
            lr=self.base_lr,
            weight_decay=tcfg.get("weight_decay", 0.01),
        )
        self.ema = EMA(bare, decay=tcfg.get("ema_decay", 0.999))
        self.model = wrap_ddp(bare)

        self.run_dir = expand(tcfg.get("run_dir", "runs/mel_baseline_v01"))
        if is_main_process():
            self.run_dir.mkdir(parents=True, exist_ok=True)
        barrier()

        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.warmup_steps = int(tcfg.get("warmup_steps", 200))
        self.lr_schedule = str(tcfg.get("lr_schedule", "cosine"))
        self.min_lr_ratio = float(tcfg.get("min_lr_ratio", 0.05))
        self.lcfg = cfg.get("loss", {})
        self.log_every = int(tcfg.get("log_every", 50) or 50)
        prec = str(tcfg.get("precision", "bf16")).lower()
        self.use_bf16 = self.device.type == "cuda" and prec in ("bf16", "bfloat16")
        cer_cfg = tcfg.get("cer_checkpoint", {})
        self.cer_every = int(cer_cfg.get("every", 0) or 0)
        self.cer_select = bool(cer_cfg.get("enabled", self.cer_every > 0))
        self._best_key: tuple | None = None

        # Resume bookkeeping (single-process default: all zeros / None).
        self.global_step = 0
        self._total_steps: int | None = None
        self._epoch = 0
        self._batches_in_epoch = 0
        self._rng_at_epoch_start: dict[str, Any] | None = None
        self._data_iter = None

        if is_main_process():
            write_run_manifest(
                self.run_dir,
                {
                    **cfg,
                    "_resolved": {
                        "use_bf16": self.use_bf16,
                        "warmup_steps": self.warmup_steps,
                        "lr_schedule": self.lr_schedule,
                        "cer_select": self.cer_select,
                        "distributed": self.dist.distributed,
                        "world_size": self.dist.world_size,
                        "rank": self.dist.rank,
                        "note": (
                            "Mel baseline does not use acoustic_dropout. "
                            "Latent Talker configs may still set acoustic_dropout "
                            "for S3 resume compatibility — do not strip abruptly."
                        ),
                    },
                },
            )
        barrier()

        resume_path = resolve_resume_path(cfg, resume)
        if resume_path is not None:
            self.load_checkpoint(resume_path)

    def _autocast(self):
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _bare_model(self) -> MelBaseline:
        return unwrap_model(self.model)  # type: ignore[return-value]

    def _set_sampler_epoch(self, epoch: int) -> None:
        sampler = getattr(self.train_loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

    def _ckpt_payload(
        self,
        stage: str,
        step: int,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "model": self._bare_model().state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.opt.state_dict(),
            "cfg": self.cfg,
            "stage": stage,
            "step": step,
            "global_step": step,
            "metrics": metrics,
            "best_key": list(self._best_key) if self._best_key is not None else None,
            "rng": _capture_rng(),
            "dataloader": {
                "epoch": self._epoch,
                "batches_in_epoch": self._batches_in_epoch,
                "rng_at_epoch_start": self._rng_at_epoch_start,
            },
            # No separate torch scheduler — LR comes from lr_multiplier(step, total).
            "schedule": {
                "warmup_steps": self.warmup_steps,
                "lr_schedule": self.lr_schedule,
                "min_lr_ratio": self.min_lr_ratio,
                "base_lr": self.base_lr,
                "total_steps": self._total_steps,
            },
        }

    def save_checkpoint(
        self,
        stage: str,
        step: int,
        *,
        metrics: dict[str, Any] | None = None,
        promote_best: bool = False,
    ) -> None:
        if not is_main_process():
            return
        payload = self._ckpt_payload(stage, step, metrics=metrics)
        torch.save(payload, self.run_dir / f"{stage}_last.pt")
        torch.save(payload, self.run_dir / "last.pt")
        if promote_best:
            torch.save(payload, self.run_dir / "best.pt")
            if metrics is not None:
                (self.run_dir / "best_metrics.json").write_text(
                    json.dumps(metrics, indent=2), encoding="utf-8"
                )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Restore model, optimizer, EMA, step, RNG, and dataloader cursor."""
        path = expand(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        bare = self._bare_model()
        sd = ckpt.get("model") or ckpt
        if isinstance(sd, dict) and "state_dict" in sd and "model" not in ckpt:
            sd = sd["state_dict"]
        # Strip optional DDP prefix from older / multi-GPU dumps.
        if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
            sd = {k.removeprefix("module."): v for k, v in sd.items()}
        bare.load_state_dict(sd, strict=True)

        if ckpt.get("ema") is not None:
            self.ema.load_state_dict(ckpt["ema"])
        if ckpt.get("optimizer") is not None:
            self.opt.load_state_dict(ckpt["optimizer"])

        step = int(ckpt.get("global_step", ckpt.get("step") or 0))
        self.global_step = step

        best = ckpt.get("best_key")
        if isinstance(best, (list, tuple)) and best:
            self._best_key = tuple(best)

        dl = ckpt.get("dataloader") or {}
        self._epoch = int(dl.get("epoch", 0))
        self._batches_in_epoch = int(dl.get("batches_in_epoch", 0))
        self._rng_at_epoch_start = dl.get("rng_at_epoch_start")

        sched = ckpt.get("schedule") or {}
        if sched.get("total_steps") is not None:
            self._total_steps = int(sched["total_steps"])

        # Replay dataloader cursor, then restore post-step RNG for dropout continuity.
        if self._rng_at_epoch_start is not None:
            _restore_rng(self._rng_at_epoch_start)
        else:
            _restore_rng(ckpt.get("rng"))
        self._set_sampler_epoch(self._epoch)
        self._data_iter = iter(self.train_loader)
        for _ in range(self._batches_in_epoch):
            try:
                next(self._data_iter)
            except StopIteration:
                break
        if self._rng_at_epoch_start is not None and ckpt.get("rng") is not None:
            _restore_rng(ckpt["rng"])

        if is_main_process():
            print(
                f"[resume] loaded={path} step={self.global_step} "
                f"epoch={self._epoch} batches_in_epoch={self._batches_in_epoch} "
                f"has_optimizer={'optimizer' in ckpt} has_ema={'ema' in ckpt}"
            )
        barrier()
        return ckpt

    def _reset_data_iter(self) -> None:
        self._rng_at_epoch_start = _capture_rng()
        self._set_sampler_epoch(self._epoch)
        self._data_iter = iter(self.train_loader)
        self._batches_in_epoch = 0

    def _next_batch(self) -> dict[str, Any]:
        if self._data_iter is None:
            self._reset_data_iter()
        assert self._data_iter is not None
        while True:
            try:
                batch = next(self._data_iter)
                self._batches_in_epoch += 1
                return batch
            except StopIteration:
                self._epoch += 1
                self._reset_data_iter()

    def _val_mel_l1(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.ema.shadow.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                with self._autocast():
                    out = self.ema.shadow(
                        batch["phone_ids"],
                        phone_mask=batch["phone_mask"],
                        durations=batch["durations"],
                        proj_W=batch.get("proj_W"),
                    )
                    loss = mel_l1_loss(out.mel_pred, batch["mel_target"], out.frame_mask)
                total += float(loss.cpu())
                n += 1
                if n >= 8:
                    break
        return total / max(n, 1)

    def _maybe_promote(self, stage: str, step: int) -> None:
        if not is_main_process():
            barrier()
            return
        # Val / CER must not consume training RNG (dropout continuity across saves).
        rng_snap = _capture_rng()
        try:
            metrics: dict[str, Any] = {"mel_l1": self._val_mel_l1()}
            if self.cer_eval_fn is not None:
                try:
                    metrics.update(self.cer_eval_fn(self.ema.shadow))
                except Exception as exc:  # noqa: BLE001
                    metrics["cer_error"] = str(exc)
        finally:
            _restore_rng(rng_snap)
            self.model.train()
        key = cer_selection_key(metrics)
        improved = self._best_key is None or key > self._best_key
        meta = {
            "stage": stage,
            "step": step,
            "improved": improved,
            "selection_key": list(key),
            **metrics,
        }
        with open(self.run_dir / "val_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
        if improved or not self.cer_select:
            self._best_key = key
            self.save_checkpoint(stage, step, metrics=meta, promote_best=True)
        else:
            self.save_checkpoint(stage, step, metrics=meta, promote_best=False)
        barrier()

    def train_steps(
        self,
        steps: int,
        stage: str = "S1",
        *,
        total_steps: int | None = None,
    ) -> dict[str, float]:
        """Run training until ``steps`` completed (absolute), optionally resuming.

        ``total_steps`` sets the LR-schedule horizon (defaults to ``steps``, or
        the value restored from a checkpoint when continuing a prior run).
        """
        self.model.train()
        if total_steps is not None:
            horizon = int(total_steps)
        elif self._total_steps is not None and self.global_step > 0:
            horizon = max(int(self._total_steps), int(steps))
        else:
            horizon = int(steps)
        self._total_steps = horizon
        start = int(self.global_step)
        if start >= steps:
            if is_main_process():
                print(f"[train] already at step={start} >= target={steps}; nothing to do")
            return {}

        pbar = tqdm(
            range(start, steps),
            desc=f"mel-{stage}",
            disable=not is_main_process(),
            initial=start,
            total=steps,
        )
        last: dict[str, float] = {}
        save_every = int(self.cfg.get("train", {}).get("save_every", 500) or 0)
        w_mel = float(self.lcfg.get("mel_l1", 1.0))
        w_dur = float(self.lcfg.get("duration", 0.1))
        w_pitch = float(self.lcfg.get("pitch", 0.0))
        w_energy = float(self.lcfg.get("energy", 0.0))
        w_post = float(self.lcfg.get("mel_postnet", 0.0))

        for step in pbar:
            mult = lr_multiplier(
                step,
                horizon,
                warmup_steps=self.warmup_steps,
                schedule=self.lr_schedule,  # type: ignore[arg-type]
                min_lr_ratio=self.min_lr_ratio,
            )
            set_optimizer_lr(self.opt, self.base_lr, mult)
            batch = self._next_batch()
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            sw = batch.get("sample_weight")
            with self._autocast():
                out = self.model(
                    batch["phone_ids"],
                    phone_mask=batch["phone_mask"],
                    durations=batch["durations"],
                    proj_W=batch.get("proj_W"),
                    pitch_target=batch.get("pitch_target"),
                    energy_target=batch.get("energy_target"),
                )
                mel_loss = mel_l1_loss(
                    out.mel_pred, batch["mel_target"], out.frame_mask, sample_weight=sw
                )
                losses: dict[str, torch.Tensor] = {"mel_l1": mel_loss}
                total = w_mel * mel_loss
                mel_pre = out.extras.get("mel_pre_postnet") if out.extras else None
                if mel_pre is not None and w_post > 0 and mel_pre is not out.mel_pred:
                    pre_loss = mel_l1_loss(
                        mel_pre, batch["mel_target"], out.frame_mask, sample_weight=sw
                    )
                    losses["mel_pre"] = pre_loss
                    total = total + w_post * pre_loss
                if out.dur_pred is not None and w_dur > 0:
                    d_loss = phone_duration_loss(
                        out.dur_pred,
                        batch["durations"],
                        batch["phone_mask"],
                        sample_weight=sw,
                    )
                    losses["duration"] = d_loss
                    total = total + w_dur * d_loss
                pitch_pred = out.extras.get("pitch_pred") if out.extras else None
                if (
                    pitch_pred is not None
                    and w_pitch > 0
                    and batch.get("pitch_target") is not None
                ):
                    p_loss = phone_scalar_loss(
                        pitch_pred,
                        batch["pitch_target"],
                        batch["phone_mask"],
                        sample_weight=sw,
                    )
                    losses["pitch"] = p_loss
                    total = total + w_pitch * p_loss
                energy_pred = out.extras.get("energy_pred") if out.extras else None
                if (
                    energy_pred is not None
                    and w_energy > 0
                    and batch.get("energy_target") is not None
                ):
                    e_loss = phone_scalar_loss(
                        energy_pred,
                        batch["energy_target"],
                        batch["phone_mask"],
                        sample_weight=sw,
                    )
                    losses["energy"] = e_loss
                    total = total + w_energy * e_loss
                losses["total"] = total

            self.opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self.ema.update(self._bare_model())

            self.global_step = step + 1
            last = {k: float(v.detach().cpu()) for k, v in losses.items()}
            last["lr"] = float(self.opt.param_groups[0]["lr"])
            if is_main_process() and step % self.log_every == 0:
                pbar.set_postfix(loss=f"{last['total']:.4f}", lr=f"{last['lr']:.2e}")
                with open(self.run_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "stage": stage, **last}) + "\n")
            do_save = save_every > 0 and self.global_step > 0 and self.global_step % save_every == 0
            do_cer = self.cer_every > 0 and self.global_step > 0 and self.global_step % self.cer_every == 0
            if do_cer or do_save:
                self._maybe_promote(stage, step=self.global_step)

        self._maybe_promote(stage, step=steps)
        return last


def make_mel_loaders(
    data_cfg: dict,
    stage_cfg: dict,
    batch_size: int,
    *,
    model_cfg: dict | None = None,
    train_section: dict | None = None,
    vocab=None,
    seed: int = 42,
):
    train_section = train_section or {}
    model_cfg = model_cfg or {}
    mel_cfg = model_cfg.get("mel", {})
    data_sec = train_section.get("data", {})
    preload = bool(data_sec.get("preload", False))
    upsample = str(mel_cfg.get("upsample", "hard"))
    forced = data_cfg.get("forced_path") or (
        str(expand(data_cfg["align_dir"]) / "forced.json") if data_cfg.get("align_dir") else None
    )
    if forced is not None:
        forced = str(expand(forced))
    # Prefer train.data.mel_dir (e.g. mel_hifigan_v01) over data YAML default.
    mel_dir = data_sec.get("mel_dir") or data_cfg.get("mel_dir")
    if mel_dir is None and data_cfg.get("ruslan_dir"):
        mel_dir = str(expand(data_cfg["ruslan_dir"]) / "mels")
    if mel_dir is not None:
        mel_dir = str(expand(mel_dir))

    gap_mode = str(
        data_sec.get("gap_mode")
        or mel_cfg.get("gap_mode")
        or train_section.get("gap_mode")
        or "expand_neighbor"
    )
    keep_stress = bool(
        data_sec.get("keep_stress", mel_cfg.get("keep_stress", False))
    )
    f0_energy_dir = data_sec.get("f0_energy_dir") or data_cfg.get("f0_energy_dir")
    if f0_energy_dir is not None:
        f0_energy_dir = str(expand(f0_energy_dir))
    common = dict(
        manifest_path=str(expand(data_cfg["manifest"])),
        mel_dir=mel_dir,
        forced_path=forced,
        vocab=vocab,
        exclude_tags=tuple(data_cfg.get("exclude_tags", ("exclude",))),
        review_weight=data_cfg.get("review_weight", 0.5),
        target=mel_cfg.get("target", "mel80"),
        upsample=upsample,
        preload_mel=preload,
        allow_missing_mel=bool(data_sec.get("allow_missing_mel", False)),
        gap_mode=gap_mode,  # type: ignore[arg-type]
        keep_stress=keep_stress,
        f0_energy_dir=f0_energy_dir,
    )
    train_ds = MelSpeechDataset(
        split="train",
        max_utterances=stage_cfg.get("max_utterances"),
        max_hours=stage_cfg.get("max_hours"),
        **common,
    )
    val_ds = MelSpeechDataset(split="val", max_utterances=32, **common)
    pad_id = vocab.pad_id if vocab is not None else 0
    collate = partial(collate_mel, pad_id=pad_id)

    # Ensure process group is known before sampler construction (no-op if single-proc).
    init_distributed()
    info = get_dist_info()
    train_sampler: DistributedSampler | None = None
    if info.distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=info.world_size,
            rank=info.rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            collate_fn=collate,
            num_workers=0,
        )
    else:
        # shuffle=True uses global torch RNG at each iter(); MelTrainer
        # snapshots that RNG at epoch start so interrupt+resume can replay.
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=0,
        )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )
    return train_loader, val_loader


# Re-export for callers that tear down after multi-process jobs.
__all__ = [
    "MelTrainer",
    "cer_selection_key",
    "cleanup_distributed",
    "make_mel_loaders",
    "resolve_resume_path",
]
