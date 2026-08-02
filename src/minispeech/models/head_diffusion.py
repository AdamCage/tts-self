"""DDPM diffusion head over backbone hidden states (N2 / A2).

Reuses CFG ``null_h`` / time-embed patterns from ``heads_flow.py``. Predicts
noise ε; conditioned on Qwen acoustic hidden ``h`` (dim = Qwen hidden size).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minispeech.constants import QWEN_HIDDEN_SIZE, get_target
from minispeech.models.heads_flow import SinusoidalTimeEmbed


class DiffusionBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.ada = nn.Linear(cond_dim, dim * 2)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.ada(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        h = self.linear2(self.act(self.linear1(h)))
        return x + h


@dataclass(slots=True)
class NoiseSchedule:
    """Precomputed DDPM coefficients (registered as buffers by the head)."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])


def make_linear_schedule(
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> NoiseSchedule:
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device, dtype=dtype)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # posterior q(x_{t-1}|x_t,x_0) variance
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return NoiseSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=alphas_cumprod.sqrt(),
        sqrt_one_minus_alphas_cumprod=(1.0 - alphas_cumprod).sqrt(),
        sqrt_recip_alphas=alphas.rsqrt(),
        posterior_variance=posterior_variance,
    )


def _extract(coef: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Gather schedule values for timestep tensor ``t`` [B] or [B,T] → broadcast."""
    if t.ndim == 1:
        out = coef.gather(0, t.clamp(0, coef.shape[0] - 1))
        return out.view(-1, *([1] * (len(shape) - 1)))
    # [B, T]
    B, T = t.shape
    flat = t.reshape(-1).clamp(0, coef.shape[0] - 1)
    out = coef.gather(0, flat).view(B, T, *([1] * (len(shape) - 2)))
    return out


class DiffusionHead(nn.Module):
    """ε-prediction DDPM head with classifier-free guidance."""

    def __init__(
        self,
        d_model: int = QWEN_HIDDEN_SIZE,
        latent_dim: int | None = None,
        hidden: int = 1024,
        layers: int = 4,
        time_embed_dim: int = 256,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        target: str = "vibevoice64",
    ):
        super().__init__()
        latent_dim = int(latent_dim if latent_dim is not None else get_target(target).dim)
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.timesteps = int(timesteps)

        self.in_proj = nn.Linear(latent_dim, hidden)
        self.time = SinusoidalTimeEmbed(time_embed_dim)
        self.cond_proj = nn.Linear(d_model + time_embed_dim, hidden)
        self.blocks = nn.ModuleList([DiffusionBlock(hidden, hidden) for _ in range(layers)])
        self.out = nn.Linear(hidden, latent_dim)
        self.null_h = nn.Parameter(torch.zeros(d_model))

        sched = make_linear_schedule(timesteps, beta_start, beta_end)
        self.register_buffer("betas", sched.betas, persistent=True)
        self.register_buffer("alphas", sched.alphas, persistent=True)
        self.register_buffer("alphas_cumprod", sched.alphas_cumprod, persistent=True)
        self.register_buffer("sqrt_alphas_cumprod", sched.sqrt_alphas_cumprod, persistent=True)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            sched.sqrt_one_minus_alphas_cumprod,
            persistent=True,
        )
        self.register_buffer("sqrt_recip_alphas", sched.sqrt_recip_alphas, persistent=True)
        self.register_buffer("posterior_variance", sched.posterior_variance, persistent=True)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        *,
        cfg_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t, h:
            ``[B, T, D]``
        t:
            discrete timesteps ``[B]``, ``[B, T]``, or continuous in ``[0, 1]``
            (auto-scaled to ``[0, timesteps-1]`` when ``t.dtype`` is floating
            and ``t.max() <= 1``).
        """
        t_emb_in = t
        if t.dtype.is_floating_point and float(t.max()) <= 1.0 + 1e-6:
            # continuous u in [0,1] → embed directly
            t_cont = t if t.ndim >= 2 else t.unsqueeze(-1).expand(x_t.shape[0], x_t.shape[1])
            if t_cont.ndim == 1:
                t_cont = t_cont.unsqueeze(-1).expand_as(x_t[..., 0])
            t_emb = self.time(t_cont.float())
        else:
            t_long = t.long()
            # normalize to [0,1] for sinusoidal embed
            t_cont = t_long.float() / max(self.timesteps - 1, 1)
            if t_cont.ndim == 1:
                t_cont = t_cont.unsqueeze(-1).expand(x_t.shape[0], x_t.shape[1])
            t_emb = self.time(t_cont)

        h_use = h
        if cfg_drop_mask is not None:
            null = self.null_h.to(dtype=h.dtype, device=h.device)
            mask = cfg_drop_mask.view(-1, 1, 1).to(dtype=h.dtype)
            h_use = h * (1 - mask) + null * mask
        cond = self.cond_proj(torch.cat([h_use, t_emb], dim=-1))
        x = self.in_proj(x_t)
        for blk in self.blocks:
            x = blk(x, cond)
        return self.out(x)

    def q_sample(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: ``x_t = √ᾱ z0 + √(1-ᾱ) ε``."""
        if noise is None:
            noise = torch.randn_like(z0)
        if t.dtype.is_floating_point and float(t.max()) <= 1.0 + 1e-6:
            t = (t * (self.timesteps - 1)).long()
        else:
            t = t.long()
        sa = _extract(self.sqrt_alphas_cumprod, t, z0.shape)
        so = _extract(self.sqrt_one_minus_alphas_cumprod, t, z0.shape)
        return sa * z0 + so * noise, noise


def ddpm_train_step(
    head: DiffusionHead,
    z0: torch.Tensor,
    h: torch.Tensor,
    *,
    cfg_dropout: float = 0.1,
    frame_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One training step: sample t, noise z0, predict ε. Returns (eps_pred, eps, loss)."""
    B, T, _ = z0.shape
    t = torch.randint(0, head.timesteps, (B, T), device=z0.device)
    x_t, eps = head.q_sample(z0, t)
    drop = None
    if cfg_dropout > 0:
        drop = torch.rand(B, device=z0.device) < cfg_dropout
    eps_pred = head(x_t, t, h, cfg_drop_mask=drop)
    per = (eps_pred - eps).pow(2).mean(dim=-1)  # [B, T]
    if frame_mask is not None:
        m = frame_mask.to(dtype=per.dtype)
        loss = (per * m).sum() / m.sum().clamp_min(1.0)
    else:
        loss = per.mean()
    return eps_pred, eps, loss


def _cfg_eps(
    head: DiffusionHead,
    x: torch.Tensor,
    t: torch.Tensor,
    h: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    eps_cond = head(x, t, h, cfg_drop_mask=None)
    if abs(cfg_scale - 1.0) < 1e-8:
        return eps_cond
    B = h.shape[0]
    null_mask = torch.ones(B, device=h.device, dtype=torch.bool)
    eps_uncond = head(x, t, h, cfg_drop_mask=null_mask)
    return eps_uncond + cfg_scale * (eps_cond - eps_uncond)


@torch.inference_mode()
def sample_ddim(
    head: DiffusionHead,
    h: torch.Tensor,
    *,
    steps: int = 10,
    cfg_scale: float = 1.3,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
    frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deterministic (η=0) DDIM sample from noise → z0. ``h``: ``[B,T,H]``."""
    B, T, _ = h.shape
    device, dtype = h.device, h.dtype
    if generator is not None:
        x = torch.randn(
            B, T, head.latent_dim, device=device, dtype=dtype, generator=generator
        )
    else:
        x = torch.randn(B, T, head.latent_dim, device=device, dtype=dtype)

    # Uniform stride over [0, timesteps)
    ts = torch.linspace(head.timesteps - 1, 0, steps, device=device).long()
    for i, t_val in enumerate(ts):
        t = torch.full((B, T), int(t_val.item()), device=device, dtype=torch.long)
        eps = _cfg_eps(head, x, t, h, cfg_scale)
        a_bar = _extract(head.alphas_cumprod, t, x.shape)
        sqrt_a = a_bar.sqrt()
        sqrt_one = (1.0 - a_bar).sqrt()
        x0 = (x - sqrt_one * eps) / sqrt_a.clamp_min(1e-8)

        if i + 1 == len(ts):
            x = x0
            break
        t_prev = torch.full(
            (B, T), int(ts[i + 1].item()), device=device, dtype=torch.long
        )
        a_bar_prev = _extract(head.alphas_cumprod, t_prev, x.shape)
        if eta > 0:
            # stochastic DDIM; rare for eval
            sigma = (
                eta
                * (
                    (1 - a_bar_prev)
                    / (1 - a_bar).clamp_min(1e-8)
                    * (1 - a_bar / a_bar_prev.clamp_min(1e-8))
                ).sqrt()
            )
            noise = torch.randn(
                B, T, head.latent_dim, device=device, dtype=dtype, generator=generator
            )
            x = (
                a_bar_prev.sqrt() * x0
                + (1.0 - a_bar_prev - sigma**2).clamp_min(0).sqrt() * eps
                + sigma * noise
            )
        else:
            x = a_bar_prev.sqrt() * x0 + (1.0 - a_bar_prev).sqrt() * eps

    if frame_mask is not None:
        x = x * frame_mask.unsqueeze(-1).to(dtype=x.dtype)
    return x


@torch.inference_mode()
def sample_ddpm(
    head: DiffusionHead,
    h: torch.Tensor,
    *,
    steps: int | None = None,
    cfg_scale: float = 1.3,
    generator: torch.Generator | None = None,
    frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ancestral DDPM sampling (slow). Prefer ``sample_ddim`` for eval."""
    steps = int(steps if steps is not None else head.timesteps)
    return sample_ddim(
        head,
        h,
        steps=steps,
        cfg_scale=cfg_scale,
        eta=1.0,
        generator=generator,
        frame_mask=frame_mask,
    )


def reconstruction_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    frame_mask: torch.Tensor | None = None,
) -> float:
    err = (pred - target).pow(2).mean(dim=-1)
    if frame_mask is not None:
        m = frame_mask.to(dtype=err.dtype)
        return float((err * m).sum().item() / m.sum().clamp_min(1.0).item())
    return float(err.mean().item())


def constant_mean_baseline_l2(
    target: torch.Tensor,
    frame_mask: torch.Tensor | None = None,
) -> float:
    """L2 of predicting the masked batch-mean latent everywhere."""
    if frame_mask is not None:
        m = frame_mask.unsqueeze(-1).to(dtype=target.dtype)
        mean = (target * m).sum(dim=(0, 1)) / m.sum().clamp_min(1.0)
        pred = mean.view(1, 1, -1).expand_as(target)
    else:
        mean = target.mean(dim=(0, 1))
        pred = mean.view(1, 1, -1).expand_as(target)
    return reconstruction_l2(pred, target, frame_mask)


def micro_overfit_sanity(
    *,
    steps: int = 200,
    B: int = 4,
    T: int = 8,
    d_model: int = 64,
    latent_dim: int = 64,
    cfg_scale: float = 1.3,
    sample_steps: int = 10,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Train head briefly on synthetic (h, z0); compare to constant-mean baseline.

    Primary metric is one-step ``x0`` reconstruction from a mid-noise ``x_t``
    (stable). Also reports multi-step DDIM L2 for the A2 report.
    """
    device = device or torch.device("cpu")
    g = torch.Generator(device="cpu").manual_seed(seed)
    z0 = torch.randn(B, T, latent_dim, generator=g)
    # Structured context: linear map of z0 (oracle-like conditioning).
    W = torch.randn(latent_dim, d_model, generator=g) / math.sqrt(latent_dim)
    h = z0 @ W
    z0 = z0.to(device)
    h = h.to(device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)

    head = DiffusionHead(
        d_model=d_model,
        latent_dim=latent_dim,
        hidden=128,
        layers=2,
        time_embed_dim=64,
        timesteps=100,
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=2e-3)
    loss = torch.tensor(0.0, device=device)

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        _, _, loss = ddpm_train_step(head, z0, h, cfg_dropout=0.05, frame_mask=mask)
        loss.backward()
        opt.step()

    head.eval()
    # One-step x0 hat at fixed mid-t (true context vs mean baseline).
    t_mid = torch.full((B, T), head.timesteps // 2, device=device, dtype=torch.long)
    g_noise = torch.Generator(device="cpu").manual_seed(seed + 2)
    noise = torch.randn(B, T, latent_dim, generator=g_noise).to(device)
    x_t, _ = head.q_sample(z0, t_mid, noise=noise)
    with torch.no_grad():
        eps_hat = _cfg_eps(head, x_t, t_mid, h, cfg_scale)
    sa = _extract(head.sqrt_alphas_cumprod, t_mid, x_t.shape)
    so = _extract(head.sqrt_one_minus_alphas_cumprod, t_mid, x_t.shape)
    x0_hat = (x_t - so * eps_hat) / sa.clamp_min(1e-8)
    rec = reconstruction_l2(x0_hat, z0, mask)
    base = constant_mean_baseline_l2(z0, mask)

    g_dev = torch.Generator(device=device if device.type == "cpu" else "cpu")
    g_dev.manual_seed(seed + 1)
    if device.type != "cpu":
        # Generator device must match randn device on CUDA.
        g_dev = torch.Generator(device=device)
        g_dev.manual_seed(seed + 1)
    pred = sample_ddim(
        head, h, steps=sample_steps, cfg_scale=cfg_scale, generator=g_dev, frame_mask=mask
    )
    ddim_l2 = reconstruction_l2(pred, z0, mask)
    return {
        "reconstruction_l2": rec,
        "constant_mean_l2": base,
        "ratio": rec / max(base, 1e-12),
        "ddim_l2": ddim_l2,
        "final_train_loss": float(loss.detach().item()),
    }
