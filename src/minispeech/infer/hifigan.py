"""Compact HiFi-GAN V1 generator + state_dict remappers.

Matches the common jik876 / SpeechBrain V1 layout used for 22.05 kHz,
``n_mels=80``, upsample ``[8,8,2,2]`` (×256 hop). Used by
``infer/vocoder.py`` when a local checkpoint is a bare state_dict.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm


LRELU_SLOPE = 0.1


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilations: tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=_get_padding(kernel_size, d),
                    )
                )
                for d in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=_get_padding(kernel_size, 1),
                    )
                )
                for _ in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class HifiGanGenerator(nn.Module):
    """jik876-style HiFi-GAN generator (V1 defaults)."""

    def __init__(
        self,
        *,
        in_channels: int = 80,
        upsample_rates: tuple[int, ...] = (8, 8, 2, 2),
        upsample_kernel_sizes: tuple[int, ...] = (16, 16, 4, 4),
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        inference_padding: int = 5,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.inference_padding = int(inference_padding)
        self.conv_pre = weight_norm(nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, tuple(d)))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_mels, T]
        if self.inference_padding > 0:
            x = F.pad(x, (self.inference_padding, self.inference_padding), mode="reflect")
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs: torch.Tensor | None = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x)
            assert xs is not None
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self) -> None:
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def remap_speechbrain_hifigan_keys(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Map SpeechBrain ``*.conv.weight_*`` keys → jik876-style weight_norm keys."""
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        nk = k.replace(".conv.weight_g", ".weight_g").replace(".conv.weight_v", ".weight_v")
        nk = nk.replace(".conv.bias", ".bias")
        out[nk] = v
    return out


def extract_generator_state_dict(obj: Any) -> dict[str, torch.Tensor] | None:
    """Pull a flat generator state_dict from common checkpoint layouts."""
    if isinstance(obj, nn.Module):
        return {k: v.detach().cpu() for k, v in obj.state_dict().items()}
    if not isinstance(obj, dict):
        return None

    # Already a flat state_dict (tensor values)
    tensor_items = {k: v for k, v in obj.items() if torch.is_tensor(v)}
    if tensor_items and any(
        k.startswith(("conv_pre", "ups.", "resblocks.", "conv_post")) for k in tensor_items
    ):
        # SpeechBrain nested .conv.* vs jik876
        if any(".conv." in k for k in tensor_items):
            return remap_speechbrain_hifigan_keys(tensor_items)
        return tensor_items

    for key in ("generator", "model", "vocoder", "state_dict", "model_state_dict"):
        if key not in obj:
            continue
        sub = obj[key]
        if isinstance(sub, nn.Module):
            return {k: v.detach().cpu() for k, v in sub.state_dict().items()}
        if isinstance(sub, dict):
            tensors = {k: v for k, v in sub.items() if torch.is_tensor(v)}
            if not tensors:
                continue
            # Strip common prefixes
            cleaned: dict[str, torch.Tensor] = {}
            for k, v in tensors.items():
                nk = k
                for pref in ("generator.", "module.", "model."):
                    if nk.startswith(pref):
                        nk = nk[len(pref) :]
                cleaned[nk] = v
            if any(".conv." in k for k in cleaned):
                cleaned = remap_speechbrain_hifigan_keys(cleaned)
            return cleaned
    return None


def load_hifigan_generator(
    state: dict[str, torch.Tensor],
    *,
    device: torch.device | None = None,
) -> HifiGanGenerator:
    """Build V1 generator and load ``state`` (strict=False for minor key drift)."""
    model = HifiGanGenerator()
    has_wn = any(k.endswith("weight_g") or k.endswith("weight_v") for k in state)
    if not has_wn:
        # State already had weight_norm removed — strip parametrization first.
        try:
            model.remove_weight_norm()
        except Exception:
            pass
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 20:
        raise RuntimeError(
            f"HiFi-GAN state_dict mismatch: missing={len(missing)} unexpected={len(unexpected)}; "
            f"sample missing={missing[:5]}"
        )
    model.eval()
    if has_wn:
        try:
            model.remove_weight_norm()
        except Exception:
            pass
    if device is not None:
        model = model.to(device)
    return model
