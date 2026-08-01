"""Minimal trainer for MiniSpeech regression / flow."""

from __future__ import annotations

import json
from contextlib import nullcontext
from functools import partial
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from minispeech.data.dataset import MiniSpeechDataset, collate_minispeech
from minispeech.eval.free_running import evaluate_free_running_loader, fr_selection_key
from minispeech.models.minispeech import MiniSpeech
from minispeech.train.ema import EMA
from minispeech.train.guided_attn import guided_attention_loss, guided_weight_schedule
from minispeech.train.losses import flow_mse, total_regression_loss
from minispeech.train.schedule import lr_multiplier, set_optimizer_lr
from minispeech.utils import expand, write_run_manifest


def cer_aware_selection_key(
    *,
    cer: float | None = None,
    fr_stability: dict[str, Any] | None = None,
) -> tuple:
    """Prefer lower synth CER when provided; else free-running stability key.

    CER need not be computed yet — pass ``cer=None`` to keep FR ranking.
    Comparison convention: higher tuple wins (so CER is negated).
    """
    if cer is not None:
        fr_tail = fr_selection_key(fr_stability or {}) if fr_stability else (0.0,)
        return (-float(cer),) + tuple(fr_tail)
    return fr_selection_key(fr_stability or {})


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


def resolve_stop_pos_weight(cfg: dict[str, Any], mean_frames: float | None = None) -> float:
    stop_cfg = cfg.get("model", {}).get("stop_head", {})
    pw = stop_cfg.get("positive_weight", 10.0)
    window = int(stop_cfg.get("stop_window", 1))
    if isinstance(pw, str) and pw.lower() == "auto":
        mf = float(mean_frames or 45.0)
        # Soft / multi-frame positives dilute the end-event; keep a floor so
        # free-running stop does not collapse under a tiny BCE contribution.
        return max(15.0, mf / max(window, 1))
    return float(pw)


class Trainer:
    def __init__(
        self,
        model: MiniSpeech,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        cfg: dict[str, Any],
        device: torch.device,
        normalizer=None,
        *,
        cer_eval_fn: Callable[[Any], dict[str, Any]] | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.normalizer = normalizer
        self.cer_eval_fn = cer_eval_fn
        tcfg = cfg.get("train", {})
        self.base_lr = float(tcfg.get("lr", 2e-4))
        self.opt = torch.optim.AdamW(
            model.parameters(),
            lr=self.base_lr,
            weight_decay=tcfg.get("weight_decay", 0.01),
        )
        self.ema = EMA(model, decay=tcfg.get("ema_decay", 0.999))
        self.run_dir = expand(tcfg.get("run_dir", "runs/default"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip = tcfg.get("grad_clip", 1.0)
        self.input_noise_max = tcfg.get("input_noise_sigma_max", 0.15)
        self.warmup_steps = int(tcfg.get("warmup_steps", 0) or 0)
        self.lr_schedule = str(tcfg.get("lr_schedule", "cosine"))
        self.min_lr_ratio = float(tcfg.get("min_lr_ratio", 0.05))
        prec = str(tcfg.get("precision", "fp32")).lower()
        self.use_bf16 = device.type == "cuda" and prec in ("bf16", "bfloat16")
        self.lcfg = cfg.get("loss", {})
        self.stop_cfg = cfg.get("model", {}).get("stop_head", {})
        mean_frames = None
        try:
            ds = getattr(train_loader, "dataset", None)
            rows = getattr(ds, "rows", None)
            if rows:
                mean_frames = float(
                    sum(r.n_frames or r.dur_s * 7.5 for r in rows)
                    / max(len(rows), 1)
                )
        except Exception:
            pass
        self.stop_pos_weight = resolve_stop_pos_weight(cfg, mean_frames)
        self.stop_loss = str(self.stop_cfg.get("loss", "bce"))
        self.stop_focal_gamma = float(self.stop_cfg.get("focal_gamma", 2.0))
        self.stop_focal_alpha = float(self.stop_cfg.get("focal_alpha", 0.75))
        self.soft_stop = bool(self.stop_cfg.get("soft_labels", False))
        self.w_tok_dur = float(self.lcfg.get("token_duration", 0.0))
        fr_cfg = tcfg.get("fr_checkpoint", {})
        self.fr_eval_every = int(fr_cfg.get("every", tcfg.get("fr_eval_every", 0)) or 0)
        self.fr_eval_n = int(fr_cfg.get("n", tcfg.get("fr_eval_n", 32)) or 32)
        self.fr_select_best = bool(fr_cfg.get("enabled", self.fr_eval_every > 0))
        cer_cfg = tcfg.get("cer_checkpoint", {})
        self.cer_every = int(cer_cfg.get("every", 0) or 0)
        self.cer_select = bool(cer_cfg.get("enabled", False))
        stop_infer = (
            cfg.get("infer", {}).get("stop", {})
            if isinstance(cfg.get("infer"), dict)
            else {}
        )
        self.fr_stop = {
            "threshold": float(stop_infer.get("threshold", 0.45)),
            "consecutive": int(stop_infer.get("consecutive", 3)),
            "min_frac": float(stop_infer.get("min_frac", 0.94)),
            "max_frac": float(stop_infer.get("max_frac", 1.6)),
            "duration_scale": float(stop_infer.get("duration_scale", 1.04)),
        }
        self._best_fr_key: tuple | None = None
        # Resume FR-best selection across crash restarts (do not clobber prior best.pt).
        fr_best_path = self.run_dir / "best_fr_metrics.json"
        if self.fr_select_best and fr_best_path.is_file():
            try:
                prior = json.loads(fr_best_path.read_text(encoding="utf-8"))
                key = prior.get("selection_key")
                if isinstance(key, list) and key:
                    self._best_fr_key = tuple(key)
                    print(
                        f"[fr-best] restored prior key from {fr_best_path.name} "
                        f"step={prior.get('step')} "
                        "duration_in_band="
                        f"{prior.get('stability', {}).get('duration_in_band_rate')}"
                    )
            except Exception as exc:  # noqa: BLE001 — best-effort resume
                print(f"[fr-best] could not restore prior key: {exc}")
        write_run_manifest(
            self.run_dir,
            {
                **cfg,
                "_resolved": {
                    "stop_pos_weight": self.stop_pos_weight,
                    "stop_loss": self.stop_loss,
                    "stop_focal_gamma": self.stop_focal_gamma,
                    "stop_focal_alpha": self.stop_focal_alpha,
                    "soft_stop": self.soft_stop,
                    "mean_frames_est": mean_frames,
                    "fr_eval_every": self.fr_eval_every,
                    "fr_select_best": self.fr_select_best,
                    "fr_stop": self.fr_stop,
                    "use_bf16": self.use_bf16,
                    "warmup_steps": self.warmup_steps,
                    "lr_schedule": self.lr_schedule,
                    "w_tok_dur": self.w_tok_dur,
                    "cer_select": self.cer_select,
                    "acoustic_dropout_note": (
                        "acoustic_dropout kept for latent Talker resume (S3). "
                        "Mel baseline does not use it. Prefer removing only after "
                        "a measured alignment/duration path replaces the need."
                    ),
                },
            },
        )
        self._freeze_modules = [model.talker, model.text_proj, model.layer_mix]
        self._frozen = False

    def _autocast(self):
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _set_talker_frozen(self, frozen: bool) -> None:
        if frozen == self._frozen:
            return
        for mod in self._freeze_modules:
            for p in mod.parameters():
                p.requires_grad = not frozen
        self._frozen = frozen

    def _freeze_steps(self, total_steps: int) -> int:
        tcfg = self.cfg.get("train", {})
        if "freeze_talker_fraction" in tcfg:
            return int(float(tcfg["freeze_talker_fraction"]) * total_steps)
        return int(tcfg.get("freeze_talker_steps", 0))

    def _noise_sigma(self, step: int, total: int) -> float:
        return self.input_noise_max * min(1.0, step / max(total * 0.5, 1))

    def _ckpt_payload(
        self,
        stage: str,
        step: int | None = None,
        *,
        fr_metrics: dict | None = None,
    ) -> dict:
        payload = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "cfg": self.cfg,
            "stage": stage,
            "step": step,
        }
        if fr_metrics is not None:
            payload["fr_metrics"] = fr_metrics
        return payload

    def _save_ckpt(
        self,
        stage: str,
        step: int | None = None,
        *,
        promote_best: bool = False,
        fr_metrics: dict | None = None,
    ) -> None:
        ckpt = self._ckpt_payload(stage, step, fr_metrics=fr_metrics)
        # Rolling last always; avoid N×165MB step dumps.
        torch.save(ckpt, self.run_dir / f"{stage}_last.pt")
        if promote_best or not self.fr_select_best:
            torch.save(ckpt, self.run_dir / "best.pt")

    def _eval_free_running(self) -> dict[str, Any]:
        """EMA free-running eval on val_loader (or a short subset)."""
        if self.val_loader is None:
            return {}
        # Eval EMA shadow (matches load_checkpoint preference).
        report = evaluate_free_running_loader(
            self.ema.shadow,
            self.val_loader,
            device=self.device,
            stop_threshold=self.fr_stop["threshold"],
            stop_consecutive=self.fr_stop["consecutive"],
            min_frac=self.fr_stop["min_frac"],
            max_frac=self.fr_stop["max_frac"],
            duration_scale=self.fr_stop["duration_scale"],
            max_items=self.fr_eval_n,
        )
        return report

    def _maybe_promote_fr_best(self, stage: str, step: int) -> dict[str, Any] | None:
        if not self.fr_select_best and not self.cer_select:
            return None
        report = (
            self._eval_free_running()
            if self.fr_select_best or self.val_loader is not None
            else {}
        )
        stab = report.get("stability", {}) if report else {}
        cer_val = None
        if self.cer_select and self.cer_eval_fn is not None:
            try:
                cer_report = self.cer_eval_fn(self.ema.shadow)
                if cer_report.get("cer") is not None:
                    cer_val = float(cer_report["cer"])
            except Exception as exc:  # noqa: BLE001
                print(f"[cer-eval] skipped: {exc}")
        elif self.cer_select:
            # Interface ready: enabled but no hook yet — keep FR ranking.
            cer_val = None
        key = cer_aware_selection_key(cer=cer_val, fr_stability=stab)
        improved = self._best_fr_key is None or key > self._best_fr_key
        meta = {
            "stage": stage,
            "step": step,
            "improved": improved,
            "selection_key": list(key),
            "prev_selection_key": (
                list(self._best_fr_key) if self._best_fr_key is not None else None
            ),
            "stability": stab,
            "cer": cer_val,
            "n": report.get("n") if report else None,
            "duration_ratios": report.get("duration_ratios") if report else None,
        }
        with open(self.run_dir / "fr_eval.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
        if improved:
            self._best_fr_key = key
            self._save_ckpt(stage, step=step, promote_best=True, fr_metrics=meta)
            (self.run_dir / "best_fr_metrics.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            msg = f"[ckpt-best] step={step}"
            if cer_val is not None:
                msg += f" cer={cer_val:.4f}"
            if stab:
                msg += (
                    f" duration_in_band={stab.get('duration_in_band_rate'):.3f} "
                    f"stop={stab.get('stop_success_rate'):.3f}"
                )
            print(msg)
        else:
            # Still refresh rolling last without clobbering FR best.
            self._save_ckpt(stage, step=step, promote_best=False)
            print(f"[ckpt-eval] step={step} (kept prior best)")
        return meta

    def train_steps(self, steps: int, stage: str = "S1") -> dict[str, float]:
        self.model.train()
        it = iter(self.train_loader)
        pbar = tqdm(range(steps), desc=f"train-{stage}")
        last: dict[str, float] = {}
        freeze_steps = self._freeze_steps(steps)
        save_every = int(self.cfg.get("train", {}).get("save_every", 0) or 0)
        for step in pbar:
            self._set_talker_frozen(step < freeze_steps)
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
            z_target = batch["z_target"]
            z_input = batch["z_input"]
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
            sw = batch.get("sample_weight")
            with self._autocast():
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
                    if (
                        out.cross_attn is not None
                        and batch.get("align_prior") is not None
                        and gw > 0
                    ):
                        guided = guided_attention_loss(
                            out.cross_attn,
                            batch["align_prior"],
                            batch["frame_mask"],
                            batch["text_mask"],
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
                        stop_pos_weight=self.stop_pos_weight,
                        stop_loss=self.stop_loss,
                        stop_focal_gamma=self.stop_focal_gamma,
                        stop_focal_alpha=self.stop_focal_alpha,
                        guided=guided,
                        w_guided=gw,
                        sample_weight=sw,
                        tok_dur_pred=out.tok_dur_pred,
                        tok_dur_target=batch.get("token_durations"),
                        text_mask=batch["text_mask"],
                        w_tok_dur=self.w_tok_dur,
                    )
                else:
                    losses = {
                        "flow": flow_mse(
                            out.extras["v_pred"],
                            out.extras["v_target"],
                            batch["frame_mask"],
                            sample_weight=sw,
                        ),
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
                            stop_pos_weight=self.stop_pos_weight,
                            stop_loss=self.stop_loss,
                            stop_focal_gamma=self.stop_focal_gamma,
                            stop_focal_alpha=self.stop_focal_alpha,
                            sample_weight=sw,
                        )["stop"],
                    }
                    losses["total"] = losses["flow"] + 0.1 * losses["stop"]

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
            do_fr = self.fr_eval_every > 0 and step > 0 and (step + 1) % self.fr_eval_every == 0
            do_cer = self.cer_every > 0 and step > 0 and (step + 1) % self.cer_every == 0
            if do_fr or do_cer:
                self._maybe_promote_fr_best(stage, step=step + 1)
            elif do_save:
                self._save_ckpt(stage, step=step + 1, promote_best=not self.fr_select_best)

        self._set_talker_frozen(False)
        if self.fr_select_best or self.cer_select:
            self._maybe_promote_fr_best(stage, step=steps)
        else:
            self._save_ckpt(stage, step=steps, promote_best=True)
        return last


def make_loaders(
    data_cfg: dict,
    stage_cfg: dict,
    batch_size: int,
    *,
    model_cfg: dict | None = None,
    train_section: dict | None = None,
):
    train_section = train_section or {}
    model_cfg = model_cfg or {}
    # Default False: ShardReader lazy-opens tensors (lower host RAM). S1/S2 may
    # still set train.data.preload: true for speed on tiny subsets.
    preload = bool(train_section.get("data", {}).get("preload", False))
    stop_cfg = model_cfg.get("stop_head", {})
    stop_window = int(stop_cfg.get("stop_window", 1))
    soft_stop = bool(stop_cfg.get("soft_labels", False))

    common = dict(
        manifest_path=data_cfg["manifest"],
        latent_dir=data_cfg["latent_dir"],
        text_hidden_dir=data_cfg["text_hidden_dir"],
        align_dir=data_cfg.get("align_dir"),
        preload=preload,
        exclude_tags=tuple(data_cfg.get("exclude_tags", ("exclude", "duplicate"))),
        review_weight=data_cfg.get("review_weight", 0.5),
    )
    train_ds = MiniSpeechDataset(
        split="train",
        max_utterances=stage_cfg.get("max_utterances"),
        max_hours=stage_cfg.get("max_hours"),
        **common,
    )
    val_ds = MiniSpeechDataset(split="val", max_utterances=32, **common)
    collate = partial(collate_minispeech, stop_window=stop_window, soft_stop=soft_stop)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )
    return train_loader, val_loader
