"""Reconstruction losses (DAC-style) + discriminator hook for later stages.

T0 trains with reconstruction only. DiscriminatorProtocol is the interface for
T1+ adversarial training; TokenizerLossBundle.adversarial path returns zeros
until a concrete discriminator is attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F


@runtime_checkable
class DiscriminatorProtocol(Protocol):
    """Minimal DAC-style discriminator interface (multi-scale / multi-period)."""

    def forward(
        self, wav: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """Return (logits per scale, feature-maps per scale)."""
        ...


def _safe_log(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))


def _stft_mag(
    wav: torch.Tensor,
    n_fft: int,
    hop: int,
    win: int,
) -> torch.Tensor:
    # cuFFT has no bf16 kernels; always run STFT in float32.
    wav32 = wav.float()
    window = torch.hann_window(win, device=wav32.device, dtype=wav32.dtype)
    spec = torch.stft(
        wav32,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        return_complex=True,
        center=True,
        normalized=False,
    )
    return spec.abs()


def multi_resolution_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    fft_sizes: tuple[int, ...] = (512, 1024, 2048),
    hop_sizes: tuple[int, ...] = (128, 256, 512),
    win_sizes: tuple[int, ...] = (512, 1024, 2048),
) -> tuple[torch.Tensor, torch.Tensor]:
    """DAC-style multi-resolution spectral convergence + log-mag L1."""
    if pred.dim() == 3:
        pred = pred.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    t = min(pred.shape[-1], target.shape[-1])
    pred = pred[..., :t]
    target = target[..., :t]

    sc_acc = pred.new_zeros(())
    mag_acc = pred.new_zeros(())
    n = 0
    for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes):
        p = _stft_mag(pred, n_fft, hop, win)
        y = _stft_mag(target, n_fft, hop, win)
        sc = torch.norm(y - p, p="fro") / torch.norm(y, p="fro").clamp_min(1e-5)
        mag = F.l1_loss(_safe_log(p), _safe_log(y))
        sc_acc = sc_acc + sc
        mag_acc = mag_acc + mag
        n += 1
    return sc_acc / n, mag_acc / n



_MEL_CACHE: dict[tuple, torch.nn.Module] = {}


def _mel_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop: int = 256,
    n_mels: int = 80,
) -> torch.Tensor:
    """Log-mel L1; runs in float32 for cuFFT. Mel filterbank is cached."""
    if pred.dim() == 3:
        pred = pred.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    t = min(pred.shape[-1], target.shape[-1])
    pred = pred[..., :t].float()
    target = target[..., :t].float()
    try:
        import torchaudio
    except ImportError:
        return pred.new_zeros(())
    key = (str(pred.device), int(sample_rate), int(n_fft), int(hop), int(n_mels))
    mel = _MEL_CACHE.get(key)
    if mel is None:
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            power=1.0,
        ).to(pred.device)
        _MEL_CACHE[key] = mel
    p = torch.log(mel(pred).clamp_min(1e-5))
    y = torch.log(mel(target).clamp_min(1e-5))
    return F.l1_loss(p, y)


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    wave_l1_weight: float = 0.1,
    stft_sc_weight: float = 1.0,
    stft_mag_weight: float = 1.0,
    mel_l1_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Weighted waveform + multi-resolution STFT reconstruction (no GAN)."""
    if pred.dim() == 3:
        p = pred.squeeze(1)
    else:
        p = pred
    if target.dim() == 3:
        y = target.squeeze(1)
    else:
        y = target
    t = min(p.shape[-1], y.shape[-1])
    p = p[..., :t]
    y = y[..., :t]
    wave = F.l1_loss(p, y)
    sc, mag = multi_resolution_stft_loss(p, y)
    mel = p.new_zeros(())
    if mel_l1_weight > 0:
        mel = _mel_l1(p, y)
    total = (
        float(wave_l1_weight) * wave
        + float(stft_sc_weight) * sc
        + float(stft_mag_weight) * mag
        + float(mel_l1_weight) * mel
    )
    return {
        "loss": total,
        "wave_l1": wave.detach(),
        "stft_sc": sc.detach(),
        "stft_mag": mag.detach(),
        "mel_l1": mel.detach() if torch.is_tensor(mel) else p.new_zeros(()),
    }


def kl_sigma_vae(
    mean: torch.Tensor,
    *,
    vae_std: float = 0.625,
) -> torch.Tensor:
    """KL of N(mean, vae_std^2 I) to N(0, I), averaged over batch/time/dim."""
    var = float(vae_std) ** 2
    kl = 0.5 * (mean.pow(2) + var - 1.0 - torch.log(torch.tensor(var, device=mean.device)))
    return kl.mean()


@dataclass
class TokenizerLossBundle:
    """Combines recon (+ optional KL) with a pluggable discriminator."""

    wave_l1_weight: float = 0.1
    stft_sc_weight: float = 1.0
    stft_mag_weight: float = 1.0
    mel_l1_weight: float = 0.0
    kl_weight: float = 1e-4
    latent_l2_weight: float = 0.0
    adv_weight: float = 0.0
    feat_weight: float = 0.0
    discriminator: DiscriminatorProtocol | None = None

    def generator_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        mean: torch.Tensor | None = None,
        vae_std: float = 0.625,
    ) -> dict[str, torch.Tensor]:
        out = reconstruction_loss(
            pred,
            target,
            wave_l1_weight=self.wave_l1_weight,
            stft_sc_weight=self.stft_sc_weight,
            stft_mag_weight=self.stft_mag_weight,
            mel_l1_weight=self.mel_l1_weight,
        )
        total = out["loss"]
        if mean is not None and self.kl_weight > 0:
            kl = kl_sigma_vae(mean, vae_std=vae_std)
            out["kl"] = kl.detach()
            total = total + float(self.kl_weight) * kl
        else:
            out["kl"] = pred.new_zeros(())
        if mean is not None and self.latent_l2_weight > 0:
            lat_l2 = mean.pow(2).mean()
            out["latent_l2"] = lat_l2.detach()
            total = total + float(self.latent_l2_weight) * lat_l2
        else:
            out["latent_l2"] = pred.new_zeros(())

        adv = pred.new_zeros(())
        feat = pred.new_zeros(())
        if self.discriminator is not None and (self.adv_weight > 0 or self.feat_weight > 0):
            fake_logits, fake_fmaps = self.discriminator(pred)
            with torch.no_grad():
                _, real_fmaps = self.discriminator(target)
            adv = torch.stack([-l.mean() for l in fake_logits]).mean()
            fm_terms = []
            for f_fake_scales, f_real_scales in zip(fake_fmaps, real_fmaps):
                for ff, fr in zip(f_fake_scales, f_real_scales):
                    fm_terms.append(F.l1_loss(ff, fr.detach()))
            if fm_terms:
                feat = torch.stack(fm_terms).mean()
            total = total + float(self.adv_weight) * adv + float(self.feat_weight) * feat
        out["adv"] = adv.detach() if torch.is_tensor(adv) else pred.new_zeros(())
        out["feat"] = feat.detach() if torch.is_tensor(feat) else pred.new_zeros(())
        out["loss"] = total
        return out

    def discriminator_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Hinge discriminator loss. No-op (zeros) when no discriminator attached."""
        if self.discriminator is None or self.adv_weight <= 0:
            z = pred.new_zeros(())
            return {"loss": z, "d_real": z, "d_fake": z}
        fake_logits, _ = self.discriminator(pred.detach())
        real_logits, _ = self.discriminator(target)
        d_real = torch.stack([F.relu(1.0 - l).mean() for l in real_logits]).mean()
        d_fake = torch.stack([F.relu(1.0 + l).mean() for l in fake_logits]).mean()
        return {"loss": d_real + d_fake, "d_real": d_real.detach(), "d_fake": d_fake.detach()}
