"""Dataset + collate with shifted latent inputs."""

from __future__ import annotations

from typing import Any, Sequence

import torch
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
        exclude_tags: Sequence[str] = ("exclude",),
        review_weight: float | None = 0.5,
    ):
        rows = filter_split(read_manifest(manifest_path), split)
        if exclude_tags:
            excl = set(exclude_tags)
            rows = [r for r in rows if (r.qc_tag or "clean") not in excl]
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
        self.review_weight = review_weight
        self.latents = ShardReader(latent_dir, "latents")
        self.text_h = ShardReader(text_hidden_dir, "text_hidden")
        self.align = ShardReader(align_dir, "align") if align_dir else None
        # Preload only this split/subset — never the full shard index (S3/S4 OOM).
        if preload:
            ids = [u.utt_id for u in self.rows]
            self._lat_cache = {i: self.latents.get(i) for i in ids}
            self._txt_cache = {i: self.text_h.get(i) for i in ids}
        else:
            self._lat_cache = None
            self._txt_cache = None

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
        th = self._get_txt(u.utt_id).float()
        weight = 1.0
        if u.sample_weight is not None:
            weight = float(u.sample_weight)
        elif self.review_weight is not None and u.qc_tag == "review":
            weight = float(self.review_weight)
        item = {
            "utt_id": u.utt_id,
            "z_target": z,
            "text_hidden": th,
            "n_frames": z.shape[0],
            "text_len": th.shape[-2] if th.ndim == 3 else th.shape[0],
            "sample_weight": weight,
        }
        if self.align is not None and u.utt_id in self.align.index:
            item["align_prior"] = self.align.get(u.utt_id).float()
        return item


def make_stop_target(
    n_frames: int,
    T: int,
    stop_window: int = 1,
    *,
    soft: bool = False,
) -> torch.Tensor:
    """Positive stop labels on the last `stop_window` valid frames.

    Hard mode: 1.0 on the window. Soft mode: linear ramp ``1/w … 1.0`` so the
    final frame is strongest (matches triple-stop inference better).
    """
    target = torch.zeros(T)
    if n_frames <= 0:
        return target
    w = max(1, int(stop_window))
    start = max(0, n_frames - w)
    if soft and w > 1:
        ramp = torch.arange(1, n_frames - start + 1, dtype=torch.float32) / float(w)
        target[start:n_frames] = ramp
    else:
        target[start:n_frames] = 1.0
    return target


def collate_minispeech(
    batch: list[dict[str, Any]],
    *,
    stop_window: int = 1,
    soft_stop: bool = False,
) -> dict[str, Any]:
    B = len(batch)
    T = max(b["n_frames"] for b in batch)
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
    sample_weight = torch.ones(B)
    align = None
    if "align_prior" in batch[0]:
        align = torch.zeros(B, T, L)

    bos = torch.zeros(LATENT_DIM)

    for i, b in enumerate(batch):
        t = b["n_frames"]
        l = b["text_len"]
        z = b["z_target"]
        z_target[i, :t] = z
        z_input[i, 0] = bos
        if t > 1:
            z_input[i, 1:t] = z[: t - 1]
        frame_mask[i, :t] = True
        text_mask[i, :l] = True
        stop_target[i] = make_stop_target(t, T, stop_window=stop_window, soft=soft_stop)
        durations[i] = float(t)
        sample_weight[i] = float(b.get("sample_weight", 1.0))
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
        "sample_weight": sample_weight,
    }
    if align is not None:
        out["align_prior"] = align
    return out
