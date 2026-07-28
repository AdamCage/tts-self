"""Rectified-flow acoustic head (v0.2)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t in [0,1], shape [B,T] or [B]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = t.unsqueeze(-1) * freqs
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


class FlowBlock(nn.Module):
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


class FlowHead(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        latent_dim: int = 64,
        hidden: int = 512,
        layers: int = 4,
        time_embed_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_proj = nn.Linear(latent_dim, hidden)
        self.time = SinusoidalTimeEmbed(time_embed_dim)
        self.cond_proj = nn.Linear(d_model + time_embed_dim, hidden)
        self.blocks = nn.ModuleList([FlowBlock(hidden, hidden) for _ in range(layers)])
        self.out = nn.Linear(hidden, latent_dim)
        # CFG null conditioning
        self.null_h = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        *,
        cfg_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x_t, h: [B, T, D]; t: [B, T] or [B, T, 1]
        returns velocity v_pred [B, T, latent_dim]
        """
        if t.ndim == 2:
            t_emb = self.time(t)
        else:
            t_emb = self.time(t.squeeze(-1))
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


def sample_logit_normal(shape, device, dtype, m: float = 0.0, s: float = 1.0) -> torch.Tensor:
    u = torch.randn(shape, device=device, dtype=dtype) * s + m
    return torch.sigmoid(u)


def rectified_flow_train_step(
    head: FlowHead,
    z0: torch.Tensor,
    h: torch.Tensor,
    *,
    cfg_dropout: float = 0.1,
    time_sampling: str = "logit_normal",
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, _ = z0.shape
    if time_sampling == "logit_normal":
        t = sample_logit_normal((B, T), z0.device, z0.dtype)
    else:
        t = torch.rand(B, T, device=z0.device, dtype=z0.dtype)
    eps = torch.randn_like(z0)
    t_ = t.unsqueeze(-1)
    x_t = (1 - t_) * z0 + t_ * eps
    v_target = eps - z0
    drop = None
    if cfg_dropout > 0:
        drop = (torch.rand(B, device=z0.device) < cfg_dropout)
    v_pred = head(x_t, t, h, cfg_drop_mask=drop)
    return v_pred, v_target


@torch.inference_mode()
def sample_rectified_flow(
    head: FlowHead,
    h: torch.Tensor,
    *,
    steps: int = 4,
    method: str = "heun",
    cfg_scale: float = 1.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Integrate from t=1 -> 0. Returns z0 estimate [B,T,64]."""
    B, T, _ = h.shape
    device, dtype = h.device, h.dtype
    x = torch.randn(B, T, head.latent_dim, device=device, dtype=dtype, generator=generator)
    dt = 1.0 / steps
    for i in range(steps):
        t_val = 1.0 - i * dt
        t = torch.full((B, T), t_val, device=device, dtype=dtype)

        def velocity(xx, tt):
            v_cond = head(xx, tt, h, cfg_drop_mask=None)
            if cfg_scale == 1.0:
                return v_cond
            null_mask = torch.ones(B, device=device, dtype=torch.bool)
            v_uncond = head(xx, tt, h, cfg_drop_mask=null_mask)
            return v_uncond + cfg_scale * (v_cond - v_uncond)

        v = velocity(x, t)
        if method == "euler" or i == steps - 1:
            x = x + (-dt) * v  # going t:1→0 so Δt negative in t
        else:
            x_euler = x + (-dt) * v
            t2 = torch.full((B, T), t_val - dt, device=device, dtype=dtype)
            v2 = velocity(x_euler, t2)
            x = x + (-dt) * 0.5 * (v + v2)
    return x
