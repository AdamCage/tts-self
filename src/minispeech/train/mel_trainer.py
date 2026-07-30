"""Trainer for Track A non-AR mel baseline."""

from __future__ import annotations

import json
from functools import partial
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from minispeech.data.mel_dataset import MelSpeechDataset, collate_mel
from minispeech.models.mel_baseline import (
    MelBaseline,
    mel_l1_loss,
    phone_duration_loss,
    phone_scalar_loss,
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
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.cer_eval_fn = cer_eval_fn
        tcfg = cfg.get("train", {})
        self.base_lr = float(tcfg.get("lr", 2e-4))
        self.opt = torch.optim.AdamW(
            model.parameters(),
            lr=self.base_lr,
            weight_decay=tcfg.get("weight_decay", 0.01),
        )
        self.ema = EMA(model, decay=tcfg.get("ema_decay", 0.999))
        self.run_dir = expand(tcfg.get("run_dir", "runs/mel_baseline_v01"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.warmup_steps = int(tcfg.get("warmup_steps", 200))
        self.lr_schedule = str(tcfg.get("lr_schedule", "cosine"))
        self.min_lr_ratio = float(tcfg.get("min_lr_ratio", 0.05))
        self.lcfg = cfg.get("loss", {})
        prec = str(tcfg.get("precision", "bf16")).lower()
        self.use_bf16 = device.type == "cuda" and prec in ("bf16", "bfloat16")
        cer_cfg = tcfg.get("cer_checkpoint", {})
        self.cer_every = int(cer_cfg.get("every", 0) or 0)
        self.cer_select = bool(cer_cfg.get("enabled", self.cer_every > 0))
        self._best_key: tuple | None = None
        write_run_manifest(
            self.run_dir,
            {
                **cfg,
                "_resolved": {
                    "use_bf16": self.use_bf16,
                    "warmup_steps": self.warmup_steps,
                    "lr_schedule": self.lr_schedule,
                    "cer_select": self.cer_select,
                    "note": (
                        "Mel baseline does not use acoustic_dropout. "
                        "Latent Talker configs may still set acoustic_dropout "
                        "for S3 resume compatibility — do not strip abruptly."
                    ),
                },
            },
        )

    def _autocast(self):
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        from contextlib import nullcontext

        return nullcontext()

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
        metrics: dict[str, Any] = {"mel_l1": self._val_mel_l1()}
        if self.cer_eval_fn is not None:
            try:
                metrics.update(self.cer_eval_fn(self.ema.shadow))
            except Exception as exc:  # noqa: BLE001
                metrics["cer_error"] = str(exc)
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
        payload = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "cfg": self.cfg,
            "stage": stage,
            "step": step,
            "metrics": meta,
        }
        torch.save(payload, self.run_dir / f"{stage}_last.pt")
        if improved or not self.cer_select:
            self._best_key = key
            torch.save(payload, self.run_dir / "best.pt")
            (self.run_dir / "best_metrics.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

    def train_steps(self, steps: int, stage: str = "S1") -> dict[str, float]:
        self.model.train()
        it = iter(self.train_loader)
        pbar = tqdm(range(steps), desc=f"mel-{stage}")
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
                steps,
                warmup_steps=self.warmup_steps,
                schedule=self.lr_schedule,  # type: ignore[arg-type]
                min_lr_ratio=self.min_lr_ratio,
            )
            set_optimizer_lr(self.opt, self.base_lr, mult)
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.train_loader)
                batch = next(it)
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
            self.ema.update(self.model)

            last = {k: float(v.detach().cpu()) for k, v in losses.items()}
            last["lr"] = float(self.opt.param_groups[0]["lr"])
            if step % 50 == 0:
                pbar.set_postfix(loss=f"{last['total']:.4f}", lr=f"{last['lr']:.2e}")
                with open(self.run_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "stage": stage, **last}) + "\n")
            do_save = save_every > 0 and step > 0 and (step + 1) % save_every == 0
            do_cer = self.cer_every > 0 and step > 0 and (step + 1) % self.cer_every == 0
            if do_cer or do_save:
                self._maybe_promote(stage, step=step + 1)

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
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )
    return train_loader, val_loader
