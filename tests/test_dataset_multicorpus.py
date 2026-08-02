"""C4: multicorpus dataset — Track N lists, lazy shards, no train↔test leak."""

from __future__ import annotations

import gc
import json
import tracemalloc
from pathlib import Path

import torch

from minispeech.constants import LATENT_DIM
from minispeech.data.dataset import (
    MultiCorpusDataset,
    RuslanCappedBatchSampler,
    load_utt_ids_from_eval_set,
)
from minispeech.data.manifest import Utterance, assert_no_speaker_leak, write_manifest
from minispeech.data.shards import ShardWriter

CV = "common_voice_ru_17_hf_mirror"


def _utt(
    utt_id: str,
    *,
    corpus: str,
    speaker: str,
    split: str,
    qc_tag: str = "clean",
) -> Utterance:
    return Utterance(
        utt_id=utt_id,
        wav=f"/tmp/{utt_id}.wav",
        text_raw="текст",
        text_stressed="те́кст",
        speaker=f"{corpus}:{speaker}" if corpus != "ruslan" else "ruslan",
        corpus=corpus,
        speaker_id=f"{corpus}:{speaker}" if corpus != "ruslan" else "ruslan",
        channel="studio" if corpus == "ruslan" else "crowd",
        sample_rate=22050 if corpus == "ruslan" else 48000,
        n_frames=4,
        dur_s=0.5,
        split=split,
        qc_tag=qc_tag,
    ).materialize_defaults()


def _write_set(path: Path, utt_ids: list[str], set_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "id": set_id,
                "version": "0.0.0-test",
                "kind": "utt_ids",
                "n": len(utt_ids),
                "utt_ids": utt_ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_cache(root: Path, utt_ids: list[str], *, frames: int = 4) -> None:
    lat = ShardWriter(root / "latents", "latents", shard_size=10_000, resume=False)
    txt = ShardWriter(root / "text_hidden", "text_hidden", shard_size=10_000, resume=False)
    for uid in utt_ids:
        lat.add(uid, torch.zeros(frames, LATENT_DIM), {"n_frames": frames})
        txt.add(uid, torch.zeros(3, 2, 8), {"text_len": 2})
    lat.close()
    txt.close()


def _build_fixture(tmp: Path, *, n_train: int = 40, n_test: int = 10) -> dict[str, object]:
    rows: list[Utterance] = []
    train_ids: list[str] = []
    test_ids: list[str] = []
    for i in range(n_train):
        if i % 2 == 0:
            u = _utt(f"ruslan_{i:06d}", corpus="ruslan", speaker="ruslan", split="train")
        else:
            spk = f"cv_train_{i // 2}"
            u = _utt(
                f"{CV}__train_{i:06d}",
                corpus=CV,
                speaker=spk,
                split="train",
            )
        rows.append(u)
        train_ids.append(u.utt_id)
    for i in range(n_test):
        u = _utt(
            f"{CV}__test_{i:06d}",
            corpus=CV,
            speaker=f"cv_test_{i}",
            split="test",
        )
        rows.append(u)
        test_ids.append(u.utt_id)

    man = tmp / "manifest.jsonl"
    write_manifest(man, rows)
    sets = tmp / "sets"
    sets.mkdir()
    train_set = sets / "trackn_train.json"
    test_set = sets / "trackn_test.json"
    _write_set(train_set, train_ids, "trackn_train")
    _write_set(test_set, test_ids, "trackn_test")

    ruslan_root = tmp / "ruslan"
    cv_root = tmp / "cv"
    _write_cache(ruslan_root, [u for u in train_ids if u.startswith("ruslan_")])
    _write_cache(cv_root, [u for u in train_ids if not u.startswith("ruslan_")] + test_ids)

    caches = {
        "ruslan": {
            "latent_dir": str(ruslan_root / "latents"),
            "text_hidden_dir": str(ruslan_root / "text_hidden"),
        },
        CV: {
            "latent_dir": str(cv_root / "latents"),
            "text_hidden_dir": str(cv_root / "text_hidden"),
        },
    }
    return {
        "manifest": str(man),
        "caches": caches,
        "trackn_sets": {"train": str(train_set), "test": str(test_set)},
        "train_ids": train_ids,
        "test_ids": test_ids,
        "rows": rows,
    }


def test_load_utt_ids_from_eval_set(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    _write_set(p, ["a", "b"], "s")
    assert load_utt_ids_from_eval_set(p) == ["a", "b"]


def test_no_utt_or_speaker_leak_train_test(tmp_path: Path) -> None:
    fx = _build_fixture(tmp_path)
    train_ds = MultiCorpusDataset(
        fx["manifest"],  # type: ignore[arg-type]
        fx["caches"],  # type: ignore[arg-type]
        split="train",
        trackn_sets=fx["trackn_sets"],  # type: ignore[arg-type]
        preload=False,
    )
    test_ds = MultiCorpusDataset(
        fx["manifest"],  # type: ignore[arg-type]
        fx["caches"],  # type: ignore[arg-type]
        split="test",
        trackn_sets=fx["trackn_sets"],  # type: ignore[arg-type]
        preload=False,
    )
    train_ids = {u.utt_id for u in train_ds.rows}
    test_ids = {u.utt_id for u in test_ds.rows}
    assert train_ids.isdisjoint(test_ids)

    train_spk = {u.speaker_id for u in train_ds.rows}
    test_spk = {u.speaker_id for u in test_ds.rows}
    assert train_spk.isdisjoint(test_spk)

    labeled = []
    for u in train_ds.rows:
        labeled.append(
            Utterance(
                utt_id=u.utt_id,
                wav="",
                text_raw="",
                text_stressed="",
                corpus=u.corpus,
                speaker_id=u.speaker_id,
                speaker=u.speaker_id,
                split="train",
            )
        )
    for u in test_ds.rows:
        labeled.append(
            Utterance(
                utt_id=u.utt_id,
                wav="",
                text_raw="",
                text_stressed="",
                corpus=u.corpus,
                speaker_id=u.speaker_id,
                speaker=u.speaker_id,
                split="test",
            )
        )
    assert_no_speaker_leak(labeled)

    item = train_ds[0]
    assert item["z_target"].shape[-1] == LATENT_DIM
    assert item["text_hidden"].ndim == 3


def test_ruslan_batch_fraction_capped(tmp_path: Path) -> None:
    fx = _build_fixture(tmp_path, n_train=80, n_test=8)
    ds = MultiCorpusDataset(
        fx["manifest"],  # type: ignore[arg-type]
        fx["caches"],  # type: ignore[arg-type]
        split="train",
        trackn_sets=fx["trackn_sets"],  # type: ignore[arg-type]
        max_ruslan_batch_fraction=0.25,
        preload=False,
    )
    batch_size = 8
    sampler = RuslanCappedBatchSampler(
        ds, batch_size=batch_size, max_ruslan_batch_fraction=0.25, drop_last=True
    )
    max_ru = sampler.max_ruslan_per_batch
    assert max_ru == 2
    for batch in sampler:
        n_ru = sum(1 for i in batch if ds.is_ruslan(i))
        n_other = len(batch) - n_ru
        if n_other > 0:
            assert n_ru <= max_ru
        assert len(batch) == batch_size


def _peak_construct(n: int, tmp: Path, monkeypatch) -> tuple[int, int]:
    """Peak traced memory + tensor-bytes loaded for a lazy dataset of size n."""
    import minispeech.data.dataset as ds_mod

    root = tmp / f"n{n}"
    root.mkdir()
    rows = []
    ids = []
    for i in range(n):
        corpus = "ruslan" if i % 5 == 0 else CV
        spk = "ruslan" if corpus == "ruslan" else f"spk_{i % 50}"
        u = _utt(
            f"{corpus}__m_{i:06d}",
            corpus=corpus,
            speaker=spk,
            split="train",
        )
        rows.append(u)
        ids.append(u.utt_id)
    man = root / "manifest.jsonl"
    write_manifest(man, rows)
    set_path = root / "trackn_train.json"
    _write_set(set_path, ids, "trackn_train")
    # Empty dirs — shard IO is mocked so real safetensors are not required.
    for corp in ("ruslan", "cv"):
        (root / corp / "latents").mkdir(parents=True)
        (root / corp / "text_hidden").mkdir(parents=True)
    caches = {
        "ruslan": {
            "latent_dir": str(root / "ruslan" / "latents"),
            "text_hidden_dir": str(root / "ruslan" / "text_hidden"),
        },
        CV: {
            "latent_dir": str(root / "cv" / "latents"),
            "text_hidden_dir": str(root / "cv" / "text_hidden"),
        },
    }

    loaded_bytes = {"n": 0}

    # Latents vs text_hidden use separate reader instances; return matching shapes.
    class _FakeLat:
        def __init__(self, *_a, **_k):
            pass

        def __contains__(self, utt_id: str) -> bool:
            return True

        def get(self, utt_id: str) -> torch.Tensor:
            t = torch.zeros(4, LATENT_DIM)
            loaded_bytes["n"] += int(t.numel() * t.element_size())
            return t

    class _FakeTxt:
        def __init__(self, *_a, **_k):
            pass

        def __contains__(self, utt_id: str) -> bool:
            return True

        def get(self, utt_id: str) -> torch.Tensor:
            t = torch.zeros(3, 2, 8)
            loaded_bytes["n"] += int(t.numel() * t.element_size())
            return t

    state = {"n": 0}

    def _factory(*_a, **_k):
        state["n"] += 1
        return _FakeLat() if state["n"] % 2 == 1 else _FakeTxt()

    monkeypatch.setattr(ds_mod, "MultiRootShardReader", _factory)

    gc.collect()
    tracemalloc.start()
    ds = MultiCorpusDataset(
        str(man),
        caches,
        split="train",
        trackn_sets={"train": str(set_path)},
        preload=False,
    )
    assert ds._lat_cache is None
    for i in (0, len(ds) // 2, len(ds) - 1):
        _ = ds[i]
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tensor_bytes = loaded_bytes["n"]
    del ds
    gc.collect()
    return peak, tensor_bytes


def test_peak_memory_not_linear_in_n(tmp_path: Path, monkeypatch) -> None:
    peak_1k, bytes_1k = _peak_construct(1000, tmp_path, monkeypatch)
    peak_10k, bytes_10k = _peak_construct(10000, tmp_path, monkeypatch)

    # Tensor IO: only the touched items are materialized (not O(n)).
    assert bytes_1k == bytes_10k
    one_item = 4 * LATENT_DIM * 4 + 3 * 2 * 8 * 4  # float32 zeros from mock
    assert bytes_10k == 3 * one_item

    # Peak RAM may grow with compact row metadata, but must not track full
    # fp16 latent+text_hidden payload for the 9k extra utts.
    full_payload_delta = 9000 * (4 * LATENT_DIM * 2 + 3 * 2 * 8 * 2)
    assert peak_10k > 0 and peak_1k > 0
    assert (peak_10k - peak_1k) < full_payload_delta
    # Recorded for out/c4_dataset_report.json consumers.
    print(f"\npeak_bytes n=1000:{peak_1k} n=10000:{peak_10k} tensor_bytes:{bytes_10k}")
