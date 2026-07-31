"""Causal depth-wise conv blocks for the sigma-VAE acoustic tokenizer.

Self-attention is replaced by 1D depth-wise *causal* convolutions so the
encoder/decoder stay streamable. All temporal convolutions left-pad and never
look ahead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Conv1d with left-only padding (strictly causal)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.pad = self.dilation * (self.kernel_size - 1)
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=groups,
            bias=bias,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class DepthwiseCausalConv(nn.Module):
    """Depth-wise causal conv + pointwise (channel mixing)."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dw = CausalConv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            groups=channels,
            bias=True,
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.pw(self.dw(x)))


class ConvTransformerBlock(nn.Module):
    """Pre-norm block: depth-wise causal conv instead of self-attention + FFN."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 7,
        dilation: int = 1,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.mix = DepthwiseCausalConv(
            channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout
        )
        self.norm2 = nn.GroupNorm(1, channels)
        hidden = channels * ffn_mult
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CausalDownsample(nn.Module):
    """Stride downsample via causal conv (no future frames)."""

    def __init__(self, in_ch: int, out_ch: int, factor: int):
        super().__init__()
        if factor < 2:
            raise ValueError("downsample factor must be >= 2")
        self.factor = int(factor)
        self.conv = CausalConv1d(in_ch, out_ch, kernel_size=factor, stride=factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CausalUpsample(nn.Module):
    """Strided transposed conv with causal trim (mirror of CausalDownsample)."""

    def __init__(self, in_ch: int, out_ch: int, factor: int):
        super().__init__()
        if factor < 2:
            raise ValueError("upsample factor must be >= 2")
        self.factor = int(factor)
        # kernel=factor, stride=factor → exact *factor length before trim
        self.conv = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=factor, stride=factor, padding=0, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        # Remove future-leaning tail so output length == T * factor exactly
        # and keep causality relative to encoder stride alignment.
        expect = x.shape[-1] * self.factor
        if y.shape[-1] > expect:
            y = y[..., :expect]
        elif y.shape[-1] < expect:
            y = F.pad(y, (0, expect - y.shape[-1]))
        return y



def build_stage_blocks(
    channels: int,
    n_blocks: int,
    *,
    kernel_size: int = 7,
    ffn_mult: int = 4,
    dropout: float = 0.0,
    dilations: tuple[int, ...] | None = None,
) -> nn.Sequential:
    if n_blocks <= 0:
        return nn.Sequential()
    if dilations is None:
        dilations = tuple(2 ** (i % 4) for i in range(n_blocks))
    blocks = []
    for i in range(n_blocks):
        blocks.append(
            ConvTransformerBlock(
                channels,
                kernel_size=kernel_size,
                dilation=int(dilations[i % len(dilations)]),
                ffn_mult=ffn_mult,
                dropout=dropout,
            )
        )
    return nn.Sequential(*blocks)
