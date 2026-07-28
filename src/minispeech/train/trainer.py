"""Minimal trainer for MiniSpeech regression / flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from minispeech.data.dataset import MiniSpeechDataset, collate_minispeech
from minispeech.models.minispeech import MiniSpeech
from minispeech.train.ema import EMA
from minispeech.train.guided_attn import attention_diagonality, guided_attention_loss, guided_weight_schedule
from minispeech.train.losses import flow_mse, total_regression_loss
from minispeech.utils import auto_batch_size, expand, set_seed, write_run_manifest


def build_model_from_cfg(cfg: dict[str, Any], mode: str = "regression") -> MiniSpeech:
    m = cfg.get("model", cfg)
    talker = m.get("talker", {})
    return MiniSpeech(
        mode=mode,  # type: ignore
        d_model=talker.get("d_model", 512),
        num_layers=talker.get("num_layers", 4),
        num_heads=talker.get("num_heads", 8),
        ffn_dim=talker.get("ffn_dim", 2048),
        latent_dim=talker.get("latent_dim", 64),
        dropout=talker.get("dropout", 0.1),
        residual_head=m.get("head_regression", {}).get("residual", True),
        causal=talker.get("causal", True),
        use_token_duration=m.get("duration_head", {}).get("level") == "token",
        flow_hidden=m.get("head_flow", {}).get("hidden", 512),
        flow_layers=m.get("head_flow", {}).get("layers", 4),
    )


class Trainer:
    def __init__(
        self,
        model: MiniSpeech,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        cfg: dict[str, Any],
        device: torch.device,
        normalizer=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.normalizer = normalizer
        tcfg = cfg.get("train", {})
        self.opt = torch.optim.AdamW(
            model.parameters(),
            lr=tcfg.get("lr", 2e-4),
            weight_decay=tcfg.get("weight_decay", 0.01),
        )
        self.ema = EMA(model, decay=tcfg.get("ema_decay", 0.999))
        self.run_dir = expand(tcfg.get("run_dir", "runs/default"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip = tcfg.get("grad_clip", 1.0)
        self.input_noise_max = tcfg.get("input_noise_sigma_max", 0.15)
        self.lcfg = cfg.get("loss", {})
        write_run_manifest(self.run_dir, cfg)

    def _noise_sigma(self, step: int, total: int) -> float:
        return self.input_noise_max * min(1.0, step / max(total * 0.5, 1))

    def train_steps(self, steps: int, stage: str = "S1") -> dict[str, float]:
        self.model.train()
        it = iter(self.train_loader)
        pbar = tqdm(range(steps), desc=f"train-{stage}")
        last: dict[str, float] = {}
        for step in pbar:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.train_loader)
                batch = next(it)
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            z_target = batch["z_target"]
            z_input = batch["z_input"]
            # replace BOS zeros with learned; apply input noise to acoustic frames
            sig = self._noise_sigma(step, steps)
            if sig > 0:
                noise = torch.randn_like(z_input) * sig
                noise[:, 0] = 0
                z_input = z_input + noise

            gw = guided_weight_schedule(
                step,
                steps,
                start=self.lcfg.get("guided_attn_start", 1.0),
                end_frac=self.lcfg.get("guided_attn_end_frac", 0.35),
            )
            out = self.model(
                z_input,
                batch["text_hidden"],
                frame_mask=batch["frame_mask"],
                text_mask=batch["text_mask"],
                align_prior=batch.get("align_prior"),
                guided_weight=gw,
                z_target=z_target,
                acoustic_dropout=float(self.cfg.get("train", {}).get("acoustic_dropout", 0.25)),
            )
            if self.model.mode == "regression":
                guided = None
                if out.cross_attn is not None and batch.get("align_prior") is not None and gw > 0:
                    guided = guided_attention_loss(
                        out.cross_attn, batch["align_prior"], batch["frame_mask"], batch["text_mask"]
                    )
                losses = total_regression_loss(
                    out.z_pred,
                    z_target,
                    out.stop_logits,
                    batch["stop_target"],
                    out.dur_pred,
                    batch["durations"],
                    batch["frame_mask"],
                    w_l1=self.lcfg.get("l1", 1.0),
                    w_cos=self.lcfg.get("cosine", 0.2),
                    w_stop=self.lcfg.get("stop", 0.1),
                    w_dur=self.lcfg.get("duration", 0.05),
                    stop_pos_weight=self.cfg.get("model", {})
                    .get("stop_head", {})
                    .get("positive_weight", 10.0),
                    guided=guided,
                    w_guided=gw,
                )
            else:
                losses = {
                    "flow": flow_mse(out.extras["v_pred"], out.extras["v_target"], batch["frame_mask"]),
                    "stop": total_regression_loss(
                        z_target,
                        z_target,
                        out.stop_logits,
                        batch["stop_target"],
                        out.dur_pred,
                        batch["durations"],
                        batch["frame_mask"],
                        w_l1=0,
                        w_cos=0,
                        w_stop=self.lcfg.get("stop", 0.1),
                        w_dur=self.lcfg.get("duration", 0.05),
                    )["stop"],
                }
                losses["total"] = losses["flow"] + 0.1 * losses["stop"]

            self.opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self.ema.update(self.model)

            last = {k: float(v.detach().cpu()) for k, v in losses.items()}
            if step % 50 == 0:
                pbar.set_postfix(loss=f"{last['total']:.4f}")
                with open(self.run_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "stage": stage, **last}) + "\n")

        ckpt = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "cfg": self.cfg,
            "stage": stage,
        }
        torch.save(ckpt, self.run_dir / f"{stage}_last.pt")
        torch.save(ckpt, self.run_dir / "best.pt")
        return last


def make_loaders(data_cfg: dict, stage_cfg: dict, batch_size: int):
    common = dict(
        manifest_path=data_cfg["manifest"],
        latent_dir=data_cfg["latent_dir"],
        text_hidden_dir=data_cfg["text_hidden_dir"],
        align_dir=data_cfg.get("align_dir"),
        preload=True,
    )
    train_ds = MiniSpeechDataset(
        split="train",
        max_utterances=stage_cfg.get("max_utterances"),
        max_hours=stage_cfg.get("max_hours"),
        **common,
    )
    val_ds = MiniSpeechDataset(split="val", max_utterances=32, **common)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_minispeech, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_minispeech, num_workers=0
    )
    return train_loader, val_loader
