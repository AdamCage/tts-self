"""N2 trainer: Qwen latent backbone + DDPM head (plan A4).

S1 overfit path uses RUSLAN (or multicorpus) latents with on-the-fly Qwen
tokenization. Free-running / head metrics reuse ``eval.free_running`` and
``eval.head_diagnostics`` — not reimplemented here.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from minispeech.codec.stats import LatentNormalizer
from minispeech.constants import get_target
from minispeech.data.collate_latent_seq import (
    collate_latent_seq,
    curriculum_max_dur_s,
    parse_curriculum,
)
from minispeech.data.dataset import MultiCorpusDataset
from minispeech.data.manifest import Utterance, filter_split, read_manifest
from minispeech.data.shards import ShardReader
from minispeech.eval.free_running import (
    aggregate_stability,
    evaluate_free_running_latents,
    fr_selection_key,
)
from minispeech.eval.head_diagnostics import summarize_head_diagnostics
from minispeech.models.backbone_lm import QwenLatentBackbone, load_qwen_backbone
from minispeech.models.head_diffusion import DiffusionHead, ddpm_train_step, sample_ddim
from minispeech.train.ema import EMA
from minispeech.train.schedule import lr_multiplier, set_optimizer_lr
from minispeech.train.trainer import cer_aware_selection_key
from minispeech.utils import expand, write_run_manifest


class _IdentityEMA:
    """No-copy EMA: shadow is the live model (avoids doubling frozen Qwen VRAM)."""

    def __init__(self, model: nn.Module):
        self.shadow = model

    def update(self, model: nn.Module) -> None:
        self.shadow = model

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd) -> None:
        self.shadow.load_state_dict(sd)


@dataclass
class N2TrainOutput:
    loss: torch.Tensor
    acoustic_hidden: torch.Tensor
    extras: dict[str, Any]


class N2Model(nn.Module):
    """Frozen Qwen backbone + trainable projectors + DDPM head."""

    def __init__(
        self,
        backbone: QwenLatentBackbone,
        head: DiffusionHead,
        *,
        cfg_dropout: float = 0.1,
        cfg_scale: float = 1.3,
        sample_steps: int = 10,
        eval_seed: int = 42,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.cfg_dropout = float(cfg_dropout)
        self.cfg_scale = float(cfg_scale)
        self.sample_steps = int(sample_steps)
        self.eval_seed = int(eval_seed)

    def trainable_parameters(self) -> list[nn.Parameter]:
        params = list(self.backbone.trainable_parameters())
        params.extend(p for p in self.head.parameters() if p.requires_grad)
        return params

    def forward(
        self,
        input_ids: torch.Tensor,
        z_input: torch.Tensor,
        z_target: torch.Tensor,
        *,
        text_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        mask_text: bool = False,
    ) -> N2TrainOutput:
        if mask_text:
            if text_mask is None:
                text_mask = input_ids.ne(0)
            text_mask = torch.zeros_like(text_mask)
            input_ids = torch.zeros_like(input_ids)
        bb = self.backbone(
            input_ids,
            z_input,
            text_mask=text_mask,
            frame_mask=frame_mask,
        )
        # Backbone may be bf16; head weights are typically fp32.
        head_dtype = next(self.head.parameters()).dtype
        h = bb.acoustic_hidden.to(dtype=head_dtype)
        z0 = z_target.to(dtype=head_dtype)
        _, _, loss = ddpm_train_step(
            self.head,
            z0,
            h,
            cfg_dropout=self.cfg_dropout if self.training else 0.0,
            frame_mask=bb.frame_mask if frame_mask is None else frame_mask,
        )
        return N2TrainOutput(
            loss=loss,
            acoustic_hidden=h,
            extras={
                "text_mask": bb.text_mask,
                "frame_mask": bb.frame_mask,
                "full_hidden": bb.full_hidden,
            },
        )

    @torch.inference_mode()
    def predict_teacher_forced(
        self,
        input_ids: torch.Tensor,
        z_input: torch.Tensor,
        *,
        text_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        cfg_scale: float | None = None,
        sample_steps: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """DDIM sample conditioned on teacher-forced acoustic hiddens."""
        bb = self.backbone(
            input_ids, z_input, text_mask=text_mask, frame_mask=frame_mask
        )
        head_dtype = next(self.head.parameters()).dtype
        h = bb.acoustic_hidden.to(dtype=head_dtype)
        g = torch.Generator(device=h.device)
        g.manual_seed(int(self.eval_seed if seed is None else seed))
        return sample_ddim(
            self.head,
            h,
            steps=int(self.sample_steps if sample_steps is None else sample_steps),
            cfg_scale=float(self.cfg_scale if cfg_scale is None else cfg_scale),
            generator=g,
            frame_mask=bb.frame_mask if frame_mask is None else frame_mask,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        max_frames: int,
        cfg_scale: float | None = None,
        sample_steps: int | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Autoregressive free-running latent generation (fixed length)."""
        device = input_ids.device
        B = input_ids.shape[0]
        assert B == 1, "N2 generate currently supports batch size 1"
        steps = int(self.sample_steps if sample_steps is None else sample_steps)
        scale = float(self.cfg_scale if cfg_scale is None else cfg_scale)
        g = torch.Generator(device=device)
        g.manual_seed(int(self.eval_seed if seed is None else seed))

        z_list: list[torch.Tensor] = []
        for t in range(int(max_frames)):
            T_cur = t + 1
            z_input = torch.zeros(1, T_cur, self.backbone.latent_dim, device=device)
            bos = self.backbone.bos_latent.to(device=device, dtype=z_input.dtype)
            z_input[:, 0] = bos
            for j, zj in enumerate(z_list):
                z_input[:, j + 1] = zj
            frame_mask = torch.ones(1, T_cur, dtype=torch.bool, device=device)
            bb = self.backbone(
                input_ids, z_input, text_mask=text_mask, frame_mask=frame_mask
            )
            head_dtype = next(self.head.parameters()).dtype
            h_t = bb.acoustic_hidden[:, t : t + 1, :].to(dtype=head_dtype)
            z_t = sample_ddim(
                self.head,
                h_t,
                steps=steps,
                cfg_scale=scale,
                generator=g,
                frame_mask=torch.ones(1, 1, dtype=torch.bool, device=device),
            )
            z_list.append(z_t[:, 0, :])

        z_pred = torch.stack(z_list, dim=1)  # [1, T, D]
        info = {
            "stop_triggered": True,
            "hit_max_frames": False,
            "n_frames": int(max_frames),
            "dur_pred": float(max_frames) / float(self.backbone.frame_rate),
        }
        return z_pred, info


def build_n2_model_from_cfg(
    cfg: dict[str, Any],
    *,
    device: torch.device,
    load_pretrained: bool = True,
    backbone: nn.Module | None = None,
) -> N2Model:
    qcfg = cfg.get("qwen", {})
    mcfg = cfg.get("model", {})
    lio = mcfg.get("latent_io", {})
    dcfg = mcfg.get("diffusion", {})
    icfg = cfg.get("infer", {})
    target = str(cfg.get("target", "vibevoice64"))
    hidden = int(qcfg.get("hidden_size", get_target(target).dim and 1024))
    dtype_name = str(qcfg.get("dtype", "bfloat16"))
    dtype = getattr(torch, dtype_name) if hasattr(torch, dtype_name) else torch.bfloat16

    bb = QwenLatentBackbone(
        target=target,
        hidden_size=hidden,
        model_id=str(qcfg.get("model_id", "")),
        backbone=backbone,
        freeze_backbone=bool(mcfg.get("backbone", {}).get("freeze", qcfg.get("freeze_backbone", True))),
        use_acoustic_marker=bool(lio.get("use_acoustic_marker", True)),
        projector_mlp_hidden=lio.get("projector_mlp_hidden"),
        dropout=float(lio.get("dropout", 0.0)),
        dtype=dtype,
        device=device if load_pretrained and backbone is None else None,
        load_pretrained=load_pretrained and backbone is None,
        local_files_only=bool(qcfg.get("local_files_only", False)),
    )
    if backbone is None and load_pretrained and bb.backbone is None:
        bb.attach_backbone(
            load_qwen_backbone(
                str(qcfg.get("model_id")),
                dtype=dtype,
                device=device,
                local_files_only=bool(qcfg.get("local_files_only", False)),
            )
        )
    head = DiffusionHead(
        d_model=hidden,
        latent_dim=get_target(target).dim,
        hidden=int(dcfg.get("hidden", hidden)),
        layers=int(dcfg.get("layers", 4)),
        time_embed_dim=int(dcfg.get("time_embed_dim", 256)),
        timesteps=int(dcfg.get("timesteps", 1000)),
        beta_start=float(dcfg.get("beta_start", 1e-4)),
        beta_end=float(dcfg.get("beta_end", 0.02)),
        target=target,
    )
    return N2Model(
        bb,
        head,
        cfg_dropout=float(dcfg.get("cfg_dropout", 0.1)),
        cfg_scale=float(icfg.get("cfg_scale", dcfg.get("cfg_scale", 1.3))),
        sample_steps=int(icfg.get("sample_steps", dcfg.get("sample_steps", 10))),
        eval_seed=int(icfg.get("eval_seed", dcfg.get("eval_seed", 42))),
    ).to(device)


class N2LatentTextDataset(Dataset):
    """Latents + Qwen ``input_ids`` from manifest text (RUSLAN S1 path).

    Does not require ``text_hidden`` for the model; latents come from the
    existing offline encode. Optional ``require_text_hidden`` validates cache
    presence for inventory checks.
    """

    def __init__(
        self,
        *,
        manifest_path: str,
        latent_dir: str,
        tokenizer,
        split: str = "train",
        max_utterances: int | None = 32,
        max_hours: float | None = None,
        max_length: int = 256,
        exclude_tags: Sequence[str] = ("exclude", "duplicate"),
        text_hidden_dir: str | None = None,
        require_text_hidden: bool = False,
        use_stressed: bool = True,
    ):
        rows = filter_split(read_manifest(manifest_path), split)
        if exclude_tags:
            excl = set(exclude_tags)
            rows = [r for r in rows if (r.qc_tag or "clean") not in excl]
        if max_utterances is not None:
            rows = rows[: int(max_utterances)]
        if max_hours is not None:
            budget = float(max_hours) * 3600.0
            acc, kept = 0.0, []
            for r in rows:
                if acc + float(r.dur_s) > budget:
                    break
                kept.append(r)
                acc += float(r.dur_s)
            rows = kept
        self.rows: list[Utterance] = rows
        self.latents = ShardReader(latent_dir, "latents")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.use_stressed = use_stressed
        if require_text_hidden:
            if not text_hidden_dir:
                raise ValueError("require_text_hidden needs text_hidden_dir")
            th = ShardReader(text_hidden_dir, "text_hidden")
            for r in self.rows:
                if r.utt_id not in th:
                    raise KeyError(f"missing text_hidden for {r.utt_id}")
        missing = [r.utt_id for r in self.rows if r.utt_id not in self.latents]
        if missing:
            raise KeyError(f"missing latents for {missing[:5]}... ({len(missing)} total)")

    def __len__(self) -> int:
        return len(self.rows)

    def _text(self, u: Utterance) -> str:
        if self.use_stressed and u.text_stressed:
            return u.text_stressed
        return u.text_raw or ""

    def __getitem__(self, idx: int) -> dict[str, Any]:
        u = self.rows[idx]
        z = self.latents.get(u.utt_id).float()
        enc = self.tokenizer(
            self._text(u),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        ids = enc["input_ids"].view(-1).long()
        return {
            "utt_id": u.utt_id,
            "input_ids": ids,
            "z_target": z,
            "dur_s": float(u.dur_s or z.shape[0] / get_target().frame_rate),
            "sample_weight": float(u.sample_weight if u.sample_weight is not None else 1.0),
            "n_frames": int(z.shape[0]),
            "corpus": u.corpus or "ruslan",
            "speaker_id": u.effective_speaker_id(),
        }


class N2MultiCorpusTokenDataset(Dataset):
    """Attach ``input_ids`` to MultiCorpusDataset items via manifest text."""

    def __init__(
        self,
        base: MultiCorpusDataset,
        *,
        manifest_path: str,
        tokenizer,
        max_length: int = 256,
        use_stressed: bool = True,
    ):
        self.base = base
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.use_stressed = use_stressed
        texts: dict[str, str] = {}
        for r in read_manifest(manifest_path):
            t = r.text_stressed if (use_stressed and r.text_stressed) else r.text_raw
            texts[r.utt_id] = t or ""
        self._texts = texts

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base[idx]
        uid = item["utt_id"]
        text = self._texts.get(uid, "")
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        item = dict(item)
        item["input_ids"] = enc["input_ids"].view(-1).long()
        if "dur_s" not in item:
            item["dur_s"] = float(item.get("n_frames", 1)) / get_target().frame_rate
        return item


class _NormCollate:
    def __init__(
        self,
        normalizer: LatentNormalizer | None,
        *,
        pad_token_id: int,
        bos_latent: torch.Tensor | None,
        target: str,
    ):
        self.normalizer = normalizer
        self.pad_token_id = pad_token_id
        self.bos_latent = bos_latent
        self.target = target

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        items = []
        for b in batch:
            z = b["z_target"]
            if self.normalizer is not None:
                z = self.normalizer.normalize(z.unsqueeze(0)).squeeze(0)
            items.append({**b, "z_target": z})
        return collate_latent_seq(
            items,
            pad_token_id=self.pad_token_id,
            bos_latent=self.bos_latent,
            target=self.target,
        )


def resolve_data_cfg(cfg: dict[str, Any], data_override: str | None = None) -> dict[str, Any]:
    """Prefer ``data.config`` in N2 yaml; CLI ``--data`` overrides."""
    from minispeech.utils import load_yaml

    if data_override:
        return load_yaml(data_override)["data"]
    dblock = cfg.get("data", {}) or {}
    path = dblock.get("config") or dblock.get("data_config")
    if path:
        return load_yaml(path)["data"]
    raise KeyError("N2 config missing data.config (or pass --data)")


def make_n2_loaders(
    cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    stage_cfg: dict[str, Any],
    batch_size: int,
    *,
    tokenizer,
    bos_latent: torch.Tensor | None = None,
    normalizer: LatentNormalizer | None = None,
) -> tuple[DataLoader, DataLoader | None, Dataset]:
    target = str(cfg.get("target", "vibevoice64"))
    qcfg = cfg.get("qwen", {})
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or 0)
    source = str((cfg.get("data") or {}).get("source", "ruslan")).lower()
    max_len = int(qcfg.get("max_length", 256))
    collate = _NormCollate(
        normalizer, pad_token_id=pad_id, bos_latent=bos_latent, target=target
    )

    if source == "multicorpus":
        from minispeech.utils import load_yaml

        mc_path = (cfg.get("data") or {}).get("multicorpus_config") or "configs/data_multicorpus.yaml"
        mcfg = load_yaml(mc_path)["data"]
        base_train = MultiCorpusDataset(
            manifest_path=mcfg["manifest"],
            corpus_caches=mcfg["corpus_caches"],
            split="train",
            trackn_sets=mcfg.get("trackn_sets"),
            max_utterances=stage_cfg.get("max_utterances"),
            max_hours=stage_cfg.get("max_hours"),
            exclude_tags=tuple(mcfg.get("exclude_tags", ("exclude", "duplicate"))),
        )
        train_ds: Dataset = N2MultiCorpusTokenDataset(
            base_train,
            manifest_path=mcfg["manifest"],
            tokenizer=tokenizer,
            max_length=max_len,
        )
        try:
            base_val = MultiCorpusDataset(
                manifest_path=mcfg["manifest"],
                corpus_caches=mcfg["corpus_caches"],
                split="val",
                trackn_sets=mcfg.get("trackn_sets"),
                max_utterances=32,
                exclude_tags=tuple(mcfg.get("exclude_tags", ("exclude", "duplicate"))),
            )
            val_ds: Dataset | None = N2MultiCorpusTokenDataset(
                base_val,
                manifest_path=mcfg["manifest"],
                tokenizer=tokenizer,
                max_length=max_len,
            )
        except Exception:
            val_ds = None
    else:
        common = dict(
            manifest_path=data_cfg["manifest"],
            latent_dir=data_cfg["latent_dir"],
            tokenizer=tokenizer,
            max_length=max_len,
            exclude_tags=tuple(data_cfg.get("exclude_tags", ("exclude", "duplicate"))),
            text_hidden_dir=data_cfg.get("text_hidden_dir"),
            require_text_hidden=bool((cfg.get("data") or {}).get("require_text_hidden", False)),
        )
        train_ds = N2LatentTextDataset(
            split="train",
            max_utterances=stage_cfg.get("max_utterances"),
            max_hours=stage_cfg.get("max_hours"),
            **common,
        )
        val_ds = N2LatentTextDataset(
            split="val",
            max_utterances=min(32, stage_cfg.get("max_utterances") or 32),
            **common,
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
        if val_ds is not None and len(val_ds) > 0
        else None
    )
    return train_loader, val_loader, train_ds


@torch.inference_mode()
def evaluate_n2_free_running(
    model: N2Model,
    loader: DataLoader,
    *,
    device: torch.device,
    max_items: int = 32,
    use_gt_length: bool = True,
    duration_scale: float = 1.0,
    sample_steps: int | None = 4,
    max_frames_cap: int | None = 64,
) -> dict[str, Any]:
    """FR DTW/duration via ``evaluate_free_running_latents`` + head diagnostics."""
    was = model.training
    model.eval()
    rows = []
    preds, gts, masks = [], [], []
    n = 0
    for batch in loader:
        if n >= max_items:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        B = batch["input_ids"].shape[0]
        for b in range(B):
            if n >= max_items:
                break
            n_gt = int(batch["frame_mask"][b].sum().item())
            max_frames = n_gt if use_gt_length else max(1, int(round(n_gt * duration_scale)))
            if max_frames_cap is not None:
                max_frames = min(max_frames, int(max_frames_cap))
            z_pred, info = model.generate(
                batch["input_ids"][b : b + 1],
                batch["text_mask"][b : b + 1],
                max_frames=max_frames,
                sample_steps=sample_steps,
            )
            z_gt = batch["z_target"][b : b + 1, :max_frames]
            m = evaluate_free_running_latents(
                z_pred,
                z_gt,
                stop_triggered=bool(info.get("stop_triggered")),
                hit_max_frames=bool(info.get("hit_max_frames"))
                or (max_frames_cap is not None and n_gt > max_frames),
            )
            rows.append(m)
            t = min(int(z_pred.shape[1]), int(z_gt.shape[1]))
            preds.append(z_pred[:, :t].cpu())
            gts.append(z_gt[:, :t].cpu())
            masks.append(torch.ones(1, t, dtype=torch.bool))
            n += 1
    if was:
        model.train()
    stab = aggregate_stability(rows)
    diag: dict[str, Any] = {}
    if preds:
        # Pad variable-length sequences (do not torch.cat on dim=0 with unequal T).
        T = max(p.shape[1] for p in preds)
        D = preds[0].shape[-1]
        Bp = len(preds)
        P = torch.zeros(Bp, T, D)
        G = torch.zeros(Bp, T, D)
        M = torch.zeros(Bp, T, dtype=torch.bool)
        for i, (p, g, _m) in enumerate(zip(preds, gts, masks)):
            t = p.shape[1]
            P[i, :t] = p[0]
            G[i, :t] = g[0]
            M[i, :t] = True
        diag = summarize_head_diagnostics(P, G, M)
    return {
        "n": len(rows),
        "stability": stab,
        "selection_key": list(fr_selection_key(stab)),
        "head_diagnostics": {
            "r2_overall": diag.get("r2_overall"),
            "var_ratio": diag.get("var_ratio"),
            "r2_per_dim_mean": diag.get("r2_per_dim_mean"),
            "corr_gt_t": diag.get("corr_gt_t"),
            "corr_gt_tm1": diag.get("corr_gt_tm1"),
            "hf_power_ratio": diag.get("hf_power_ratio"),
        },
        "mean_dtw_l1": stab.get("mean_dtw_l1"),
        "mean_duration_ratio": stab.get("mean_duration_ratio"),
    }


@torch.inference_mode()
def evaluate_n2_teacher_forced_diagnostics(
    model: N2Model,
    loader: DataLoader,
    *,
    device: torch.device,
    max_items: int = 32,
) -> dict[str, Any]:
    was = model.training
    model.eval()
    preds, gts, masks = [], [], []
    losses: list[float] = []
    n = 0
    for batch in loader:
        if n >= max_items:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(
            batch["input_ids"],
            batch["z_input"],
            batch["z_target"],
            text_mask=batch["text_mask"],
            frame_mask=batch["frame_mask"],
        )
        losses.append(float(out.loss.item()))
        z_hat = model.predict_teacher_forced(
            batch["input_ids"],
            batch["z_input"],
            text_mask=batch["text_mask"],
            frame_mask=batch["frame_mask"],
        )
        preds.append(z_hat.cpu())
        gts.append(batch["z_target"].cpu())
        masks.append(batch["frame_mask"].cpu())
        n += batch["input_ids"].shape[0]
    if was:
        model.train()
    if not preds:
        return {"tf_loss_mean": float("nan")}
    T = max(p.shape[1] for p in preds)
    D = preds[0].shape[-1]
    Bp = sum(p.shape[0] for p in preds)
    P = torch.zeros(Bp, T, D)
    G = torch.zeros(Bp, T, D)
    M = torch.zeros(Bp, T, dtype=torch.bool)
    row = 0
    for p, g, m in zip(preds, gts, masks):
        b, t, _ = p.shape
        P[row : row + b, :t] = p
        G[row : row + b, :t] = g[:, :t]
        M[row : row + b, :t] = m[:, :t]
        row += b
    diag = summarize_head_diagnostics(P, G, M)
    return {
        "tf_loss_mean": float(sum(losses) / max(len(losses), 1)),
        "r2_overall": diag["r2_overall"],
        "var_ratio": diag["var_ratio"],
        "r2_per_dim_mean": diag["r2_per_dim_mean"],
        **{k: diag[k] for k in ("corr_gt_t", "corr_gt_tm1", "hf_power_ratio") if k in diag},
    }


class N2Trainer:
    def __init__(
        self,
        model: N2Model,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        cfg: dict[str, Any],
        device: torch.device,
        *,
        train_dataset: Dataset | None = None,
        cer_eval_fn: Callable[[N2Model], dict[str, Any]] | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset
        self.cfg = cfg
        self.device = device
        self.cer_eval_fn = cer_eval_fn
        tcfg = cfg.get("train", {})
        self.base_lr = float(tcfg.get("lr", 2e-4))
        self.opt = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=self.base_lr,
            weight_decay=float(tcfg.get("weight_decay", 0.01)),
        )
        ema_decay = float(tcfg.get("ema_decay", 0.0))
        # Default 0: do not deepcopy frozen Qwen (~2× VRAM). Set ema_decay>0 to enable.
        self.ema = (
            EMA(model, decay=ema_decay) if ema_decay > 0 else _IdentityEMA(model)
        )
        self.run_dir = expand(tcfg.get("run_dir", "runs/n2_s1_default"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.warmup_steps = int(tcfg.get("warmup_steps", 0) or 0)
        self.lr_schedule = str(tcfg.get("lr_schedule", "cosine"))
        self.min_lr_ratio = float(tcfg.get("min_lr_ratio", 0.05))
        prec = str(tcfg.get("precision", "fp32")).lower()
        self.use_bf16 = device.type == "cuda" and prec in ("bf16", "bfloat16")
        self.log_every = int(tcfg.get("log_every", 50))
        self.eval_every = int(tcfg.get("eval_every", 250))
        self.save_every = int(tcfg.get("save_every", 1000))
        fr_cfg = tcfg.get("fr_checkpoint", {})
        self.fr_eval_every = int(fr_cfg.get("every", tcfg.get("fr_eval_every", 0)) or 0)
        self.fr_eval_n = int(fr_cfg.get("n", tcfg.get("fr_eval_n", 32)) or 32)
        self.fr_select_best = bool(fr_cfg.get("enabled", self.fr_eval_every > 0))
        self.curriculum = parse_curriculum(tcfg.get("curriculum"))
        self._best_fr_key: tuple | None = None
        self._loss_curve: list[dict[str, float]] = []
        fr_best_path = self.run_dir / "best_fr_metrics.json"
        if self.fr_select_best and fr_best_path.is_file():
            try:
                prior = json.loads(fr_best_path.read_text(encoding="utf-8"))
                key = prior.get("selection_key")
                if isinstance(key, list) and key:
                    self._best_fr_key = tuple(key)
            except Exception as exc:  # noqa: BLE001
                print(f"[fr-best] could not restore: {exc}")
        write_run_manifest(
            self.run_dir,
            {
                **cfg,
                "_resolved": {
                    "use_bf16": self.use_bf16,
                    "fr_eval_every": self.fr_eval_every,
                    "fr_select_best": self.fr_select_best,
                    "n_train": len(train_dataset) if train_dataset is not None else None,
                    "trainable_params": sum(
                        p.numel() for p in model.trainable_parameters()
                    ),
                    "backbone_trainable": model.backbone.backbone_trainable_parameter_count(),
                },
            },
        )

    def _autocast(self):
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _ckpt_payload(self, stage: str, step: int | None, fr_metrics: dict | None = None) -> dict:
        payload = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "cfg": self.cfg,
            "stage": stage,
            "step": step,
            "kind": "n2_qwen_latent",
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
        torch.save(ckpt, self.run_dir / f"{stage}_last.pt")
        torch.save(ckpt, self.run_dir / "last.pt")
        if promote_best or not self.fr_select_best:
            torch.save(ckpt, self.run_dir / "best.pt")

    def _eval_loader(self) -> DataLoader | None:
        return self.val_loader or self.train_loader

    def _maybe_promote_fr_best(self, stage: str, step: int) -> dict[str, Any] | None:
        if not self.fr_select_best:
            return None
        loader = self._eval_loader()
        if loader is None:
            return None
        try:
            report = evaluate_n2_free_running(
                self.ema.shadow,
                loader,
                device=self.device,
                max_items=self.fr_eval_n,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[fr-eval] failed, falling back to TF diagnostics: {exc}")
            tf = evaluate_n2_teacher_forced_diagnostics(
                self.ema.shadow, loader, device=self.device, max_items=self.fr_eval_n
            )
            # Higher is better: prefer higher r2, lower tf loss via negated loss in key.
            r2 = float(tf.get("r2_overall") or -1e9)
            loss = float(tf.get("tf_loss_mean") or 1e9)
            report = {
                "n": self.fr_eval_n,
                "stability": {
                    "duration_in_band_rate": 0.0,
                    "stop_success_rate": 0.0,
                    "runaway_rate": 0.0,
                    "nan_rate": 0.0,
                    "mean_dtw_l1": loss,
                },
                "head_diagnostics": tf,
                "tf_fallback": True,
                "selection_key": [r2, -loss],
            }
        stab = report.get("stability", {})
        cer_val = None
        if self.cer_eval_fn is not None:
            try:
                cer_report = self.cer_eval_fn(self.ema.shadow)
                if cer_report.get("cer") is not None:
                    cer_val = float(cer_report["cer"])
            except Exception as exc:  # noqa: BLE001
                print(f"[cer-eval] skipped: {exc}")
        if report.get("tf_fallback"):
            key = tuple(report["selection_key"])
        else:
            key = cer_aware_selection_key(cer=cer_val, fr_stability=stab)
        improved = self._best_fr_key is None or key > self._best_fr_key
        meta = {
            "stage": stage,
            "step": step,
            "improved": improved,
            "selection_key": list(key),
            "stability": stab,
            "cer": cer_val,
            "head_diagnostics": report.get("head_diagnostics"),
            "n": report.get("n"),
        }
        with open(self.run_dir / "fr_eval.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
        if improved:
            self._best_fr_key = key
            self._save_ckpt(stage, step=step, promote_best=True, fr_metrics=meta)
            (self.run_dir / "best_fr_metrics.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            print(
                f"[ckpt-best] step={step} dtw={stab.get('mean_dtw_l1')} "
                f"r2={report.get('head_diagnostics', {}).get('r2_overall')}"
            )
        else:
            self._save_ckpt(stage, step=step, promote_best=False)
            print(f"[ckpt-eval] step={step} (kept prior best)")
        return meta

    def _filter_batch_by_curriculum(
        self, batch: dict[str, Any], max_dur_s: float
    ) -> dict[str, Any] | None:
        durs = batch.get("durations")
        if durs is None:
            return batch
        keep = [i for i, d in enumerate(durs.tolist()) if float(d) <= max_dur_s + 1e-9]
        if not keep:
            # Nothing in-batch fits curriculum — keep full batch so S1 cannot stall.
            return batch
        if len(keep) == durs.shape[0]:
            return batch
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.shape[0] == durs.shape[0]:
                out[k] = v[keep]
            elif isinstance(v, list) and len(v) == durs.shape[0]:
                out[k] = [v[i] for i in keep]
            else:
                out[k] = v
        return out

    def train_steps(self, steps: int, stage: str = "S1") -> dict[str, Any]:
        self.model.train()
        it = iter(self.train_loader)
        pbar = tqdm(range(steps), desc=f"n2-train-{stage}")
        last: dict[str, Any] = {}
        running = 0.0
        n_logged = 0
        for step in pbar:
            mult = lr_multiplier(
                step,
                steps,
                warmup_steps=self.warmup_steps,
                schedule=self.lr_schedule,  # type: ignore[arg-type]
                min_lr_ratio=self.min_lr_ratio,
            )
            set_optimizer_lr(self.opt, self.base_lr, mult)
            max_dur = curriculum_max_dur_s(self.curriculum, step)
            batch = None
            for _ in range(32):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(self.train_loader)
                    batch = next(it)
                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()
                }
                batch = self._filter_batch_by_curriculum(batch, max_dur)
                if batch is not None:
                    break
            if batch is None:
                continue

            self.opt.zero_grad(set_to_none=True)
            with self._autocast():
                out = self.model(
                    batch["input_ids"],
                    batch["z_input"],
                    batch["z_target"],
                    text_mask=batch["text_mask"],
                    frame_mask=batch["frame_mask"],
                )
                loss = out.loss
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.trainable_parameters(), self.grad_clip
                )
            self.opt.step()
            self.ema.update(self.model)

            lv = float(loss.detach().float().item())
            running += lv
            n_logged += 1
            last = {"loss": lv, "lr": self.base_lr * mult, "max_dur_s": max_dur}
            pbar.set_postfix(loss=f"{lv:.4f}", dur=f"{max_dur:.1f}")

            if (step + 1) % self.log_every == 0:
                avg = running / max(n_logged, 1)
                self._loss_curve.append({"step": step + 1, "loss": avg})
                with open(self.run_dir / "loss.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step + 1, "loss": avg}) + "\n")
                running = 0.0
                n_logged = 0

            do_fr = self.fr_eval_every > 0 and (step + 1) % self.fr_eval_every == 0
            do_eval = self.eval_every > 0 and (step + 1) % self.eval_every == 0
            if do_fr:
                self._maybe_promote_fr_best(stage, step + 1)
            elif do_eval:
                loader = self._eval_loader()
                if loader is not None:
                    tf = evaluate_n2_teacher_forced_diagnostics(
                        self.ema.shadow, loader, device=self.device, max_items=min(16, self.fr_eval_n)
                    )
                    with open(self.run_dir / "tf_eval.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": step + 1, **tf}) + "\n")
                    print(
                        f"[tf-eval] step={step+1} loss={tf.get('tf_loss_mean'):.4f} "
                        f"r2={tf.get('r2_overall')} var_ratio={tf.get('var_ratio')}"
                    )
                self._save_ckpt(stage, step=step + 1, promote_best=not self.fr_select_best)

            if self.save_every > 0 and (step + 1) % self.save_every == 0:
                self._save_ckpt(stage, step=step + 1, promote_best=False)

        # Final FR eval + promote
        if self.fr_select_best:
            self._maybe_promote_fr_best(stage, steps)
        else:
            self._save_ckpt(stage, step=steps, promote_best=True)

        (self.run_dir / "loss_curve.json").write_text(
            json.dumps(self._loss_curve, indent=2), encoding="utf-8"
        )
        last["loss_curve_points"] = len(self._loss_curve)
        last["run_dir"] = str(self.run_dir)
        return last


def load_n2_checkpoint(
    path: str | Path,
    device: torch.device,
    *,
    load_pretrained_backbone: bool = True,
) -> tuple[N2Model, dict]:
    ckpt = torch.load(expand(path), map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") or {}
    model = build_n2_model_from_cfg(
        cfg, device=device, load_pretrained=load_pretrained_backbone
    )
    sd = ckpt.get("ema") or ckpt["model"]
    # EMA stores shadow state_dict of full module
    if any(k.startswith("backbone.") or k.startswith("head.") for k in sd):
        missing, unexpected = model.load_state_dict(sd, strict=False)
    else:
        missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"load_n2_checkpoint: missing={len(missing)} unexpected={len(unexpected)}"
    )
    model.to(device).eval()
    return model, ckpt


# Listen-demo recipe (documented for A4 / checkpoint-listen):
LISTEN_DEMO_RECIPE = """
# After runs/n2_s1_*/best.pt appears:
cd ~/projects/tts-self && source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
export PYTHONPATH=/mnt/c/projects/tts-self/src
export MINISPEECH_ALLOW_DIRTY=1
python /mnt/c/projects/tts-self/scripts/66_infer_n2.py \\
  --ckpt runs/n2_s1_ruslan/best.pt \\
  --demo \\
  --out out/n2_first_execution/n2_s1_listen_demo
"""
