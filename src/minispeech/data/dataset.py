"""Dataset + collate with shifted latent inputs (single- and multi-corpus)."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from minispeech.constants import LATENT_DIM
from minispeech.data.manifest import Utterance, filter_split, iter_manifest, read_manifest
from minispeech.data.shards import MultiRootShardReader, ShardReader
from minispeech.utils import expand


@dataclass(slots=True, frozen=True)
class MultiCorpusRow:
    """Compact row kept in RAM (avoids full Utterance / input_ids blow-up)."""

    utt_id: str
    corpus: str
    speaker_id: str
    qc_tag: str | None
    sample_weight: float | None
    dur_s: float

DEFAULT_TRACKN_SET_PATHS: dict[str, str] = {
    "train": "configs/eval_sets/trackn_train.json",
    "val": "configs/eval_sets/trackn_val.json",
    "test": "configs/eval_sets/trackn_test.json",
    "review": "configs/eval_sets/trackn_review.json",
}

# Cap RUSLAN share so one studio speaker does not dominate CV17 (config key:
# data.sampling.max_ruslan_batch_fraction).
DEFAULT_MAX_RUSLAN_BATCH_FRACTION = 0.25
RUSLAN_CORPUS_ID = "ruslan"


def load_utt_ids_from_eval_set(path: str | Path) -> list[str]:
    """Load ``utt_ids`` from a frozen Track N / eval-set JSON."""
    raw = json.loads(expand(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("utt_ids"), list):
        return [str(x) for x in raw["utt_ids"]]
    if isinstance(raw, dict) and isinstance(raw.get("items"), list):
        return [str(it["utt_id"]) for it in raw["items"] if it.get("utt_id")]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    raise ValueError(f"{path}: expected utt_ids / items")


def load_trackn_utt_ids(
    set_paths: Mapping[str, str] | None = None,
    *,
    splits: Sequence[str] = ("train", "val", "test", "review"),
) -> dict[str, list[str]]:
    """Load Track N frozen lists keyed by split name."""
    paths = dict(DEFAULT_TRACKN_SET_PATHS if set_paths is None else set_paths)
    out: dict[str, list[str]] = {}
    for split in splits:
        if split not in paths:
            raise KeyError(f"no trackn set path for split {split!r}")
        out[split] = load_utt_ids_from_eval_set(paths[split])
    return out


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
        exclude_tags: Sequence[str] = ("exclude", "duplicate"),
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


class MultiCorpusDataset(Dataset):
    """RUSLAN + CV17 (etc.) over Track N frozen utt lists with lazy shard IO.

    Parameters
    ----------
    corpus_caches:
        ``{corpus_id: {"latent_dir": ..., "text_hidden_dir": ..., "align_dir"?}}``
    trackn_sets / utt_ids:
        Prefer frozen Track N JSON paths (``trackn_train`` / ``val`` / ``test``).
        ``split`` selects which list to materialize.
    preload:
        Must stay False for large multicorpus runs (memory gate).
    max_ruslan_batch_fraction:
        Stored for :class:`RuslanCappedBatchSampler` (config-driven).
    """

    def __init__(
        self,
        manifest_path: str,
        corpus_caches: Mapping[str, Mapping[str, str]],
        *,
        split: str = "train",
        trackn_sets: Mapping[str, str] | None = None,
        utt_ids: Sequence[str] | None = None,
        max_utterances: int | None = None,
        max_hours: float | None = None,
        preload: bool = False,
        exclude_tags: Sequence[str] = ("exclude", "duplicate"),
        review_weight: float | None = 0.5,
        max_ruslan_batch_fraction: float = DEFAULT_MAX_RUSLAN_BATCH_FRACTION,
        require_caches: bool = True,
    ):
        if preload:
            raise ValueError(
                "MultiCorpusDataset refuses preload=True; use lazy MultiRootShardReader"
            )

        if utt_ids is not None:
            wanted_list = [str(u) for u in utt_ids]
        else:
            lists = load_trackn_utt_ids(trackn_sets, splits=(split,))
            wanted_list = lists[split]
        wanted = set(wanted_list)

        excl = set(exclude_tags) if exclude_tags else set()
        found: dict[str, MultiCorpusRow] = {}
        for r in iter_manifest(manifest_path):
            if r.utt_id not in wanted:
                continue
            if excl and (r.qc_tag or "clean") in excl:
                continue
            corpus = str(r.corpus or RUSLAN_CORPUS_ID)
            found[r.utt_id] = MultiCorpusRow(
                utt_id=r.utt_id,
                corpus=corpus,
                speaker_id=r.effective_speaker_id(),
                qc_tag=r.qc_tag,
                sample_weight=r.sample_weight,
                dur_s=float(r.dur_s or 0.0),
            )

        rows = [found[uid] for uid in wanted_list if uid in found]
        missing = len(wanted_list) - len(rows)
        found.clear()
        wanted.clear()
        if max_utterances is not None:
            rows = rows[: int(max_utterances)]
        if max_hours is not None:
            budget = float(max_hours) * 3600.0
            acc, kept = 0.0, []
            for r in rows:
                if acc + r.dur_s > budget:
                    break
                kept.append(r)
                acc += r.dur_s
            rows = kept

        self.rows = rows
        self.split = split
        self.review_weight = review_weight
        self.max_ruslan_batch_fraction = float(max_ruslan_batch_fraction)
        self.missing_utt_ids = missing
        self.utt_to_corpus = {r.utt_id: r.corpus for r in self.rows}

        latent_dirs = {
            corpus: cfg["latent_dir"]
            for corpus, cfg in corpus_caches.items()
            if cfg.get("latent_dir")
        }
        text_dirs = {
            corpus: cfg["text_hidden_dir"]
            for corpus, cfg in corpus_caches.items()
            if cfg.get("text_hidden_dir")
        }
        align_dirs = {
            corpus: cfg["align_dir"]
            for corpus, cfg in corpus_caches.items()
            if cfg.get("align_dir")
        }
        self.latents = MultiRootShardReader(
            latent_dirs, "latents", self.utt_to_corpus, lazy=True
        )
        self.text_h = MultiRootShardReader(
            text_dirs, "text_hidden", self.utt_to_corpus, lazy=True
        )
        self.align: MultiRootShardReader | None = None
        if align_dirs:
            self.align = MultiRootShardReader(
                align_dirs, "align", self.utt_to_corpus, lazy=True
            )

        if require_caches:
            for r in self.rows:
                if r.utt_id not in self.latents:
                    raise KeyError(
                        f"missing latents for {r.utt_id!r} "
                        f"(corpus={self.utt_to_corpus[r.utt_id]!r})"
                    )
                if r.utt_id not in self.text_h:
                    raise KeyError(
                        f"missing text_hidden for {r.utt_id!r} "
                        f"(corpus={self.utt_to_corpus[r.utt_id]!r})"
                    )

        self._lat_cache = None
        self._txt_cache = None

    def __len__(self) -> int:
        return len(self.rows)

    def corpus_of(self, idx: int) -> str:
        return self.utt_to_corpus[self.rows[idx].utt_id]

    def is_ruslan(self, idx: int) -> bool:
        return self.corpus_of(idx) == RUSLAN_CORPUS_ID

    def __getitem__(self, idx: int) -> dict[str, Any]:
        u = self.rows[idx]
        z = self.latents.get(u.utt_id).float()
        th = self.text_h.get(u.utt_id).float()
        weight = 1.0
        if u.sample_weight is not None:
            weight = float(u.sample_weight)
        elif self.review_weight is not None and u.qc_tag == "review":
            weight = float(self.review_weight)
        item = {
            "utt_id": u.utt_id,
            "corpus": u.corpus,
            "speaker_id": u.speaker_id,
            "z_target": z,
            "text_hidden": th,
            "n_frames": z.shape[0],
            "text_len": th.shape[-2] if th.ndim == 3 else th.shape[0],
            "sample_weight": weight,
        }
        if self.align is not None and u.utt_id in self.align:
            item["align_prior"] = self.align.get(u.utt_id).float()
        return item


class RuslanCappedBatchSampler(Sampler[list[int]]):
    """Batch sampler that caps the RUSLAN fraction per batch.

    Config key: ``data.sampling.max_ruslan_batch_fraction`` (default 0.25).
    Remaining slots are filled from non-RUSLAN corpora when available. If only
    RUSLAN indices remain (tail of epoch), the cap is relaxed so the epoch can
    finish.
    """

    def __init__(
        self,
        dataset: MultiCorpusDataset,
        batch_size: int,
        *,
        max_ruslan_batch_fraction: float | None = None,
        drop_last: bool = False,
        seed: int = 42,
        shuffle: bool = True,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.max_ruslan_batch_fraction = float(
            max_ruslan_batch_fraction
            if max_ruslan_batch_fraction is not None
            else dataset.max_ruslan_batch_fraction
        )
        if not 0.0 <= self.max_ruslan_batch_fraction <= 1.0:
            raise ValueError("max_ruslan_batch_fraction must be in [0, 1]")
        self.drop_last = drop_last
        self.seed = int(seed)
        self.shuffle = shuffle
        self._epoch = 0
        self._ruslan = [i for i in range(len(dataset)) if dataset.is_ruslan(i)]
        self._other = [i for i in range(len(dataset)) if not dataset.is_ruslan(i)]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def max_ruslan_per_batch(self) -> int:
        return int(math.floor(self.batch_size * self.max_ruslan_batch_fraction))

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size) if n else 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        ruslan = list(self._ruslan)
        other = list(self._other)
        if self.shuffle:
            rng.shuffle(ruslan)
            rng.shuffle(other)

        max_ru = self.max_ruslan_per_batch
        ri = 0
        oi = 0

        while ri < len(ruslan) or oi < len(other):
            batch: list[int] = []
            other_exhausted = oi >= len(other)
            # Pure-RUSLAN tail: allow a full batch. Otherwise enforce the cap.
            ru_budget = self.batch_size if other_exhausted else max_ru

            for _ in range(ru_budget):
                if ri >= len(ruslan) or len(batch) >= self.batch_size:
                    break
                batch.append(ruslan[ri])
                ri += 1
            while len(batch) < self.batch_size and oi < len(other):
                batch.append(other[oi])
                oi += 1

            if not batch:
                break
            if len(batch) < self.batch_size and self.drop_last:
                break
            if self.shuffle:
                rng.shuffle(batch)
            yield batch



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
        text_len = b["text_len"]
        z = b["z_target"]
        z_target[i, :t] = z
        z_input[i, 0] = bos
        if t > 1:
            z_input[i, 1:t] = z[: t - 1]
        frame_mask[i, :t] = True
        text_mask[i, :text_len] = True
        stop_target[i] = make_stop_target(t, T, stop_window=stop_window, soft=soft_stop)
        durations[i] = float(t)
        sample_weight[i] = float(b.get("sample_weight", 1.0))
        th = b["text_hidden"]
        if multi_layer:
            text_hidden[i, :, :text_len] = th[:, :text_len]
        else:
            text_hidden[i, :text_len] = th[:text_len]
        if align is not None and "align_prior" in b:
            ap = b["align_prior"]
            tt, ll = min(t, ap.shape[0]), min(text_len, ap.shape[1])
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
    if "corpus" in batch[0]:
        out["corpora"] = [b.get("corpus") for b in batch]
    if "speaker_id" in batch[0]:
        out["speaker_ids"] = [b.get("speaker_id") for b in batch]
    if align is not None:
        out["align_prior"] = align
    return out
