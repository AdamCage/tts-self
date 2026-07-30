"""Trainer for Track A CFM / rectified-flow mel decoder."""

from __future__ import annotations

import json
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from minispeech.models.mel_baseline import phone_duration_loss
from minispeech.models.mel_cfm import MelCFM, mel_flow_mse
from minispeech.train.ema import EMA
from minispeech.train.mel_trainer import cer_selection_key, make_mel_loaders
from minispeech.train.schedule import lr_multiplier, set_optimizer_lr
from minispeech.utils import expand, write_run_manifest


class MelCFMTrainer:
    def __init__(
        self,
        model: MelCFM,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        cfg: dict[str, Any],
        device: torch.device,
        *,
        cer_eval_fn: Callable[[MelCFM], dict[str, Any]] | None = None,
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
        self.run_dir = expand(tcfg.get("run_dir", "runs/mel_cfm_v01"))
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
                    "mode": "cfm",
                    "use_bf16": self.use_bf16,
                    "warmup_steps": self.warmup_steps,
                    "lr_schedule": self.lr_schedule,
                    "cer_select": self.cer_select,
                    "note": (
                        "Track A CFM mel decoder (plan a5). "
                        "Does not use VibeVoice / Talker / acoustic_dropout."
                    ),
                },
            },
        )

    def _autocast(self):
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        from contextlib import nullcontext

        return nullcontext()

    def _val_flow_mse(self) -> float:
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
                        mel_target=batch["mel_target"],
                        phone_mask=batch["phone_mask"],
                        durations=batch["durations"],
                        proj_W=batch.get("proj_W"),
                    )
                    assert out.v_pred is not None and out.v_target is not None
                    loss = mel_flow_mse(out.v_pred, out.v_target, out.frame_mask)
                total += float(loss.cpu())
                n += 1
                if n >= 8:
                    break
        return total / max(n, 1)

    def _maybe_promote(self, stage: str, step: int) -> None:
        metrics: dict[str, Any] = {"flow_mse": self._val_flow_mse()}
        # Alias for cer_selection_key fallback ranking (lower is better via negation).
        metrics["mel_l1"] = metrics["flow_mse"]
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
            "mode": "cfm",
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
        pbar = tqdm(range(steps), desc=f"mel-cfm-{stage}")
        last: dict[str, float] = {}
        save_every = int(self.cfg.get("train", {}).get("save_every", 500) or 0)
        w_flow = float(self.lcfg.get("flow_mse", 1.0))
        w_dur = float(self.lcfg.get("duration", 0.1))
        loss_history: list[float] = []
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
                    mel_target=batch["mel_target"],
                    phone_mask=batch["phone_mask"],
                    durations=batch["durations"],
                    proj_W=batch.get("proj_W"),
                )
                assert out.v_pred is not None and out.v_target is not None
                flow_loss = mel_flow_mse(
                    out.v_pred, out.v_target, out.frame_mask, sample_weight=sw
                )
                losses: dict[str, torch.Tensor] = {"flow_mse": flow_loss}
                total = w_flow * flow_loss
                if out.dur_pred is not None and w_dur > 0:
                    d_loss = phone_duration_loss(
                        out.dur_pred,
                        batch["durations"],
                        batch["phone_mask"],
                        sample_weight=sw,
                    )
                    losses["duration"] = d_loss
                    total = total + w_dur * d_loss
                losses["total"] = total

            self.opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self.ema.update(self.model)

            last = {k: float(v.detach().cpu()) for k, v in losses.items()}
            last["lr"] = float(self.opt.param_groups[0]["lr"])
            loss_history.append(last["total"])
            if step % 50 == 0:
                pbar.set_postfix(loss=f"{last['total']:.4f}", lr=f"{last['lr']:.2e}")
                with open(self.run_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "stage": stage, **last}) + "\n")
            do_save = save_every > 0 and step > 0 and (step + 1) % save_every == 0
            do_cer = self.cer_every > 0 and step > 0 and (step + 1) % self.cer_every == 0
            if do_cer or do_save:
                self._maybe_promote(stage, step=step + 1)

        self._maybe_promote(stage, step=steps)
        # Smoke diagnostics: early vs late mean loss
        if len(loss_history) >= 20:
            mid = max(len(loss_history) // 5, 1)
            early = sum(loss_history[:mid]) / mid
            late = sum(loss_history[-mid:]) / mid
            last["loss_early_mean"] = early
            last["loss_late_mean"] = late
            last["loss_decreased"] = float(late < early)
            (self.run_dir / "smoke_metrics.json").write_text(
                json.dumps(
                    {
                        "stage": stage,
                        "steps": steps,
                        "loss_early_mean": early,
                        "loss_late_mean": late,
                        "loss_decreased": late < early,
                        "final": last,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return last


__all__ = ["MelCFMTrainer", "make_mel_loaders"]
