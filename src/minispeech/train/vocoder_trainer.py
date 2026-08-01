"""Dataset and trainer for fine-tuning a HiFi-GAN generator on RUSLAN."""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from minispeech.constants import get_target
from minispeech.data.manifest import filter_split, read_manifest
from minispeech.data.mel import MelCacheReader, extract_mel_file, load_wav_mono
from minispeech.infer.hifigan import (
    HifiGanGenerator,
    extract_generator_state_dict,
    remap_speechbrain_hifigan_keys,
)
from minispeech.models.hifigan_disc import HiFiGanDiscriminators
from minispeech.train.schedule import lr_multiplier, set_optimizer_lr
from minispeech.utils import expand, write_run_manifest


def _load_checkpoint(path: str | Path) -> Any:
    try:
        return torch.load(expand(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(expand(path), map_location="cpu")


def load_generator_checkpoint(
    path: str | Path, *, inference_padding: int = 0
) -> HifiGanGenerator:
    """Load SpeechBrain/jik876 generator weights without stripping weight norm."""
    obj = _load_checkpoint(path)
    state = extract_generator_state_dict(obj)
    if state is None:
        raise RuntimeError(f"could not find a HiFi-GAN generator state_dict in {expand(path)}")
    if any(".conv." in key for key in state):
        state = remap_speechbrain_hifigan_keys(state)
    generator = HifiGanGenerator(inference_padding=inference_padding)
    has_weight_norm = any(key.endswith(("weight_g", "weight_v")) for key in state)
    if not has_weight_norm:
        generator.remove_weight_norm()
    missing, unexpected = generator.load_state_dict(state, strict=False)
    matched = len(generator.state_dict()) - len(missing)
    if matched < 20:
        raise RuntimeError(
            f"HiFi-GAN checkpoint mismatch for {expand(path)}: matched={matched}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    print(
        f"initialized generator from {expand(path)}: matched={matched} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    return generator


class VocoderPairDataset(Dataset[dict[str, Any]]):
    """Lazy RUSLAN mel↔waveform pairs for the manifest's existing split labels."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        target: str = "mel80_hifigan",
        mel_dir: str | Path | None = None,
        max_utterances: int | None = None,
        max_hours: float | None = None,
        exclude_tags: tuple[str, ...] = ("exclude", "duplicate"),
    ):
        self.spec = get_target(target)
        if self.spec.kind != "mel":
            raise ValueError(f"vocoder target must be mel, got {self.spec.name!r}")
        rows = filter_split(read_manifest(expand(manifest_path)), split)
        excluded = set(exclude_tags)
        rows = [row for row in rows if (row.qc_tag or "clean") not in excluded]
        if max_utterances is not None:
            rows = rows[: int(max_utterances)]
        if max_hours is not None:
            budget, elapsed, kept = float(max_hours) * 3600.0, 0.0, []
            for row in rows:
                if elapsed + float(row.dur_s) > budget:
                    break
                kept.append(row)
                elapsed += float(row.dur_s)
            rows = kept
        if not rows:
            raise ValueError(f"no usable {split!r} rows in manifest {manifest_path}")
        self.rows = rows
        self.mels: MelCacheReader | None = None
        if mel_dir is not None:
            cache_path = expand(mel_dir)
            if (cache_path / "mels_index.json").is_file():
                self.mels = MelCacheReader(cache_path, lazy=True, expect_target=self.spec.name)
                self.rows = [row for row in self.rows if row.utt_id in self.mels.index]
        if not self.rows:
            raise ValueError(f"no {split!r} rows have mels in {mel_dir}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        mel = (
            self.mels.get(row.utt_id).float()
            if self.mels is not None
            else extract_mel_file(row.wav, target=self.spec)
        )
        if mel.ndim != 2 or mel.shape[1] != self.spec.dim:
            raise ValueError(
                f"{row.utt_id}: expected [T, {self.spec.dim}] mel, "
                f"got {tuple(mel.shape)}"
            )
        wav_np, _ = load_wav_mono(row.wav, target_sr=self.spec.sample_rate)
        wav = torch.from_numpy(wav_np).float()
        target_samples = int(mel.shape[0]) * self.spec.samples_per_frame
        wav = F.pad(wav, (0, max(0, target_samples - wav.numel())))[:target_samples]
        return {"utt_id": row.utt_id, "mel": mel, "wav": wav}


def collate_vocoder_pairs(
    batch: list[dict[str, Any]],
    *,
    segment_frames: int,
    hop_length: int,
    random_crop: bool = True,
) -> dict[str, Any]:
    """Randomly crop aligned fixed-size mel/waveform segments."""
    if segment_frames < 2:
        raise ValueError("segment_frames must be at least 2")
    mels, wavs, utt_ids = [], [], []
    n_samples = segment_frames * hop_length
    for item in batch:
        mel, wav = item["mel"], item["wav"]
        frames = int(mel.shape[0])
        if frames < segment_frames:
            mel = F.pad(mel, (0, 0, 0, segment_frames - frames))
            wav = F.pad(wav, (0, max(0, n_samples - wav.numel())))
            start = 0
        else:
            start = random.randint(0, frames - segment_frames) if random_crop else 0
        sample_start = start * hop_length
        wav_seg = wav[sample_start : sample_start + n_samples]
        wav_seg = F.pad(wav_seg, (0, max(0, n_samples - wav_seg.numel())))
        mels.append(mel[start : start + segment_frames])
        wavs.append(wav_seg[:n_samples])
        utt_ids.append(str(item["utt_id"]))
    return {
        "utt_ids": utt_ids,
        "mel": torch.stack(mels).transpose(1, 2).contiguous(),
        "wav": torch.stack(wavs).unsqueeze(1).contiguous(),
    }


def discriminator_hinge_loss(
    real_scores: list[torch.Tensor], fake_scores: list[torch.Tensor]
) -> torch.Tensor:
    return sum(F.relu(1.0 - score).mean() + F.relu(1.0 + fake).mean()
               for score, fake in zip(real_scores, fake_scores))


def generator_adversarial_loss(fake_scores: list[torch.Tensor]) -> torch.Tensor:
    return sum((score - 1.0).square().mean() for score in fake_scores)


def feature_matching_loss(
    real_features: list[list[torch.Tensor]], fake_features: list[list[torch.Tensor]]
) -> torch.Tensor:
    return sum(
        F.l1_loss(fake_feature, real_feature.detach())
        for real_maps, fake_maps in zip(real_features, fake_features)
        for real_feature, fake_feature in zip(real_maps, fake_maps)
    )


class VocoderTrainer:
    """HiFi-GAN fine-tuner with mel reconstruction, hinge GAN, and feature matching."""

    def __init__(
        self,
        generator: HifiGanGenerator,
        discriminators: HiFiGanDiscriminators,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict[str, Any],
        device: torch.device,
        *,
        target: str = "mel80_hifigan",
        resume: str | Path | None = None,
    ):
        self.cfg, self.device, self.target = cfg, device, get_target(target)
        self.generator = generator.to(device)
        self.discriminators = discriminators.to(device)
        self.train_loader, self.val_loader = train_loader, val_loader
        tcfg, lcfg = cfg.get("train", {}), cfg.get("loss", {})
        self.run_dir = expand(tcfg.get("run_dir", "runs/vocoder_ft_v01"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.g_opt = torch.optim.AdamW(
            self.generator.parameters(),
            lr=float(tcfg.get("generator_lr", 2e-4)),
            betas=tuple(tcfg.get("betas", (0.8, 0.99))),
            weight_decay=float(tcfg.get("weight_decay", 0.0)),
        )
        self.d_opt = torch.optim.AdamW(
            self.discriminators.parameters(),
            lr=float(tcfg.get("discriminator_lr", 2e-4)),
            betas=tuple(tcfg.get("betas", (0.8, 0.99))),
            weight_decay=float(tcfg.get("weight_decay", 0.0)),
        )
        self.base_g_lr = float(self.g_opt.param_groups[0]["lr"])
        self.base_d_lr = float(self.d_opt.param_groups[0]["lr"])
        self.mel_weight = float(lcfg.get("mel", 45.0))
        self.adversarial_weight = float(lcfg.get("adversarial", 1.0))
        self.feature_weight = float(lcfg.get("feature_matching", 2.0))
        self.grad_clip = float(tcfg.get("grad_clip", 10.0))
        self.warmup_steps = int(tcfg.get("warmup_steps", 0))
        self.lr_schedule = str(tcfg.get("lr_schedule", "constant"))
        self.min_lr_ratio = float(tcfg.get("min_lr_ratio", 0.1))
        self.log_every = int(tcfg.get("log_every", 50))
        self.save_every = int(tcfg.get("save_every", 1000))
        self.val_batches = int(tcfg.get("val_batches", 8))
        precision = str(tcfg.get("precision", "bf16")).lower()
        self.use_bf16 = device.type == "cuda" and precision in {"bf16", "bfloat16"}
        self.global_step = 0
        self.best_val_mel = float("inf")
        self._data_iter: Any = None
        self._mel_transform = self._make_mel_transform()
        write_run_manifest(
            self.run_dir,
            cfg,
            extra={
                "target": self.target.name,
                "target_geometry": {
                    "sample_rate": self.target.sample_rate,
                    "hop_length": self.target.hop_length,
                    "fmax": self.target.fmax,
                },
                "generator_init": str(tcfg.get("generator_init")),
                "use_bf16": self.use_bf16,
            },
        )
        if resume is not None:
            self.load_checkpoint(resume)

    def _make_mel_transform(self) -> nn.Module:
        import torchaudio

        assert self.target.n_fft and self.target.hop_length and self.target.win_length
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target.sample_rate,
            n_fft=self.target.n_fft,
            win_length=self.target.win_length,
            hop_length=self.target.hop_length,
            n_mels=self.target.n_mels,
            f_min=self.target.fmin,
            f_max=self.target.fmax,
            power=1.0,
            norm=self.target.mel_norm,
            mel_scale=self.target.mel_scale or "slaney",
        ).to(self.device)

    def _autocast(self):
        return (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_bf16
            else nullcontext()
        )

    def _mel_loss(self, real_wav: torch.Tensor, fake_wav: torch.Tensor) -> torch.Tensor:
        real_mel = torch.log(self._mel_transform(real_wav.squeeze(1)).clamp_min(1e-5))
        fake_mel = torch.log(self._mel_transform(fake_wav.squeeze(1)).clamp_min(1e-5))
        return F.l1_loss(fake_mel, real_mel)

    def _next_batch(self) -> dict[str, Any]:
        if self._data_iter is None:
            self._data_iter = iter(self.train_loader)
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.train_loader)
            return next(self._data_iter)

    def _payload(self, metrics: dict[str, float]) -> dict[str, Any]:
        return {
            "generator": self.generator.state_dict(),
            "discriminators": self.discriminators.state_dict(),
            "generator_optimizer": self.g_opt.state_dict(),
            "discriminator_optimizer": self.d_opt.state_dict(),
            "cfg": self.cfg,
            "global_step": self.global_step,
            "best_val_mel": self.best_val_mel,
            "metrics": metrics,
        }

    def save_checkpoint(self, *, best: bool, metrics: dict[str, float]) -> None:
        payload = self._payload(metrics)
        torch.save(payload, self.run_dir / "last.pt")
        if best:
            torch.save(payload, self.run_dir / "best.pt")
            (self.run_dir / "best_metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = _load_checkpoint(path)
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminators.load_state_dict(checkpoint["discriminators"])
        self.g_opt.load_state_dict(checkpoint["generator_optimizer"])
        self.d_opt.load_state_dict(checkpoint["discriminator_optimizer"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_val_mel = float(checkpoint.get("best_val_mel", float("inf")))
        print(f"resumed {expand(path)} at step={self.global_step}")

    @torch.no_grad()
    def validate(self) -> float:
        self.generator.eval()
        total, count = 0.0, 0
        for batch in self.val_loader:
            mel, wav = batch["mel"].to(self.device), batch["wav"].to(self.device)
            generated = self.generator(mel)
            total += float(self._mel_loss(wav, generated).cpu())
            count += 1
            if count >= self.val_batches:
                break
        self.generator.train()
        return total / max(count, 1)

    def train_steps(self, steps: int) -> dict[str, float]:
        self.generator.train()
        self.discriminators.train()
        last: dict[str, float] = {}
        for _ in tqdm(range(self.global_step, steps), desc="vocoder-ft"):
            multiplier = lr_multiplier(
                self.global_step,
                steps,
                warmup_steps=self.warmup_steps,
                schedule=self.lr_schedule,  # type: ignore[arg-type]
                min_lr_ratio=self.min_lr_ratio,
            )
            set_optimizer_lr(self.g_opt, self.base_g_lr, multiplier)
            set_optimizer_lr(self.d_opt, self.base_d_lr, multiplier)
            batch = self._next_batch()
            mel, real_wav = batch["mel"].to(self.device), batch["wav"].to(self.device)

            with self._autocast():
                fake_wav = self.generator(mel)
            self.d_opt.zero_grad(set_to_none=True)
            real_scores, _ = self.discriminators(real_wav)
            fake_scores, _ = self.discriminators(fake_wav.detach().float())
            d_loss = discriminator_hinge_loss(real_scores, fake_scores)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminators.parameters(), self.grad_clip)
            self.d_opt.step()

            self.g_opt.zero_grad(set_to_none=True)
            with self._autocast():
                fake_wav = self.generator(mel)
            fake_wav = fake_wav.float()
            fake_scores, fake_features = self.discriminators(fake_wav)
            with torch.no_grad():
                _, real_features = self.discriminators(real_wav)
            mel_loss = self._mel_loss(real_wav, fake_wav)
            adv_loss = generator_adversarial_loss(fake_scores)
            fm_loss = feature_matching_loss(real_features, fake_features)
            g_loss = (
                self.mel_weight * mel_loss
                + self.adversarial_weight * adv_loss
                + self.feature_weight * fm_loss
            )
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip)
            self.g_opt.step()

            self.global_step += 1
            last = {
                "generator": float(g_loss.detach().cpu()),
                "discriminator": float(d_loss.detach().cpu()),
                "mel": float(mel_loss.detach().cpu()),
                "adversarial": float(adv_loss.detach().cpu()),
                "feature_matching": float(fm_loss.detach().cpu()),
                "lr": float(self.g_opt.param_groups[0]["lr"]),
            }
            if self.global_step % self.log_every == 0:
                with open(self.run_dir / "train_log.jsonl", "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": self.global_step, **last}) + "\n")
            if self.global_step % self.save_every == 0 or self.global_step == steps:
                val_mel = self.validate()
                metrics = {**last, "val_mel": val_mel}
                improved = val_mel < self.best_val_mel
                if improved:
                    self.best_val_mel = val_mel
                with open(self.run_dir / "val_log.jsonl", "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": self.global_step, **metrics}) + "\n")
                self.save_checkpoint(best=improved, metrics=metrics)
        return last


def make_vocoder_loaders(
    data_cfg: dict[str, Any],
    cfg: dict[str, Any],
    *,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Build loaders using manifest train/val book-range splits (never random split)."""
    tcfg, dcfg = cfg.get("train", {}), cfg.get("data", {})
    target_name = str(dcfg.get("target", "mel80_hifigan"))
    mel_dir = dcfg.get("mel_dir")
    if mel_dir is None:
        ruslan_dir = data_cfg.get("ruslan_dir")
        mel_dir = str(expand(ruslan_dir) / "mels_hifigan") if ruslan_dir else None
    common = {
        "manifest_path": data_cfg["manifest"],
        "target": target_name,
        "mel_dir": mel_dir,
        "exclude_tags": tuple(data_cfg.get("exclude_tags", ("exclude", "duplicate"))),
    }
    train_dataset = VocoderPairDataset(
        split="train",
        max_utterances=dcfg.get("max_utterances"),
        max_hours=dcfg.get("max_hours"),
        **common,
    )
    val_dataset = VocoderPairDataset(
        split="val",
        max_utterances=dcfg.get("val_max_utterances"),
        **common,
    )
    spec = get_target(target_name)
    segment_frames = int(dcfg.get("segment_frames", 32))
    train_collate = partial(
        collate_vocoder_pairs,
        segment_frames=segment_frames,
        hop_length=spec.samples_per_frame,
        random_crop=True,
    )
    val_collate = partial(
        collate_vocoder_pairs,
        segment_frames=segment_frames,
        hop_length=spec.samples_per_frame,
        random_crop=False,
    )
    workers = int(tcfg.get("num_workers", 2))
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            collate_fn=train_collate,
        ),
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            collate_fn=val_collate,
        ),
    )
