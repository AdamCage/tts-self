"""Dataset + collate with shifted latent inputs."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from minispeech.constants import LATENT_DIM
from minispeech.data.manifest import Utterance, filter_split, read_manifest
from minispeech.data.shards import ShardReader


class MiniSpeechDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        latent_dir: str,
        text_hidden_dir: str,
        split: str = "train",
        align_dir: str | None = None,
        max_utterances: int | None = None,
        max_hours: float | None = None,
        preload: bool = True,
    ):
        rows = filter_split(read_manifest(manifest_path), split)
        if max_utterances is not None:
            rows = rows[:max_utterances]
        if max_hours is not None:
            budget = max_hours * 3600.0
            acc, kept = 0.0, []
            for r in rows:
                if acc + r.dur_s > budget:
                    break
                kept.append(r)
                acc += r.dur_s
            rows = kept
        self.rows = rows
        self.latents = ShardReader(latent_dir, "latents")
        self.text_h = ShardReader(text_hidden_dir, "text_hidden")
        self.align = ShardReader(align_dir, "align") if align_dir else None
        self._lat_cache = self.latents.preload_all() if preload else None
        self._txt_cache = self.text_h.preload_all() if preload else None

    def __len__(self) -> int:
        return len(self.rows)

    def _get_lat(self, utt_id: str) -> torch.Tensor:
        if self._lat_cache is not None:
            return self._lat_cache[utt_id]
        return self.latents.get(utt_id)

    def _get_txt(self, utt_id: str) -> torch.Tensor:
        if self._txt_cache is not None:
            return self._txt_cache[utt_id]
        return self.text_h.get(utt_id)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        u: Utterance = self.rows[idx]
        z = self._get_lat(u.utt_id).float()  # [T, 64]
        # text_hidden stored as [n_layers, L, H] or [L, H]
        th = self._get_txt(u.utt_id).float()
        item = {
            "utt_id": u.utt_id,
            "z_target": z,
            "text_hidden": th,
            "n_frames": z.shape[0],
            "text_len": th.shape[-2] if th.ndim == 3 else th.shape[0],
        }
        if self.align is not None and u.utt_id in self.align.index:
            item["align_prior"] = self.align.get(u.utt_id).float()
        return item


def collate_minispeech(batch: list[dict[str, Any]]) -> dict[str, Any]:
    B = len(batch)
    T = max(b["n_frames"] for b in batch)
    # text_hidden: [n_layers, L, H] or [L, H]
    sample_th = batch[0]["text_hidden"]
    multi_layer = sample_th.ndim == 3
    if multi_layer:
        n_layers, _, H = sample_th.shape
        L = max(b["text_len"] for b in batch)
        text_hidden = torch.zeros(B, n_layers, L, H)
    else:
        H = sample_th.shape[-1]
        L = max(b["text_len"] for b in batch)
        text_hidden = torch.zeros(B, L, H)

    z_target = torch.zeros(B, T, LATENT_DIM)
    z_input = torch.zeros(B, T, LATENT_DIM)
    frame_mask = torch.zeros(B, T, dtype=torch.bool)
    text_mask = torch.zeros(B, L, dtype=torch.bool)
    stop_target = torch.zeros(B, T)
    durations = torch.zeros(B)
    align = None
    if "align_prior" in batch[0]:
        align = torch.zeros(B, T, L)

    bos = torch.zeros(LATENT_DIM)  # real BOS is model param; zeros as placeholder in data

    for i, b in enumerate(batch):
        t = b["n_frames"]
        l = b["text_len"]
        z = b["z_target"]
        z_target[i, :t] = z
        # shifted: [BOS, z_0, ..., z_(T-2)]
        z_input[i, 0] = bos
        if t > 1:
            z_input[i, 1:t] = z[: t - 1]
        frame_mask[i, :t] = True
        text_mask[i, :l] = True
        stop_target[i, t - 1] = 1.0
        durations[i] = float(t)
        th = b["text_hidden"]
        if multi_layer:
            text_hidden[i, :, :l] = th[:, :l]
        else:
            text_hidden[i, :l] = th[:l]
        if align is not None and "align_prior" in b:
            ap = b["align_prior"]
            tt, ll = min(t, ap.shape[0]), min(l, ap.shape[1])
            align[i, :tt, :ll] = ap[:tt, :ll]

    out = {
        "utt_ids": [b["utt_id"] for b in batch],
        "z_target": z_target,
        "z_input": z_input,
        "text_hidden": text_hidden,
        "frame_mask": frame_mask,
        "text_mask": text_mask,
        "stop_target": stop_target,
        "durations": durations,
    }
    if align is not None:
        out["align_prior"] = align
    return out
