"""Track A: target registry, frame projection, duration stub, uniform packing smoke."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from minispeech.constants import (
    FRAME_RATE,
    LATENT_DIM,
    MEL80,
    SAMPLES_PER_FRAME,
    SR,
    TARGETS,
    VIBEVOICE64,
    get_target,
    n_frames_from_samples,
)
from minispeech.data.forced_align import (
    build_uniform_corpus_entries,
    load_forced_json,
    mfa_crosscheck_stub,
    pack_forced_entry,
    text_to_ctc_char_labels,
    try_ctc_align_emissions,
    write_forced_json,
)
from minispeech.data.frame_projection import (
    TimeSegment,
    hard_overlap_projection,
    load_forced_segments,
    n_frames_for_duration,
    normalize_columns,
    soft_overlap_projection,
)
from minispeech.frontend.durations import MeanDurationPredictor, duration_stats_from_forced


# ---------------------------------------------------------------------------
# A6 — target registry
# ---------------------------------------------------------------------------


def test_vibevoice_aliases_unchanged():
    assert LATENT_DIM == 64
    assert FRAME_RATE == 7.5
    assert SAMPLES_PER_FRAME == 3200
    assert SR == 24_000
    assert VIBEVOICE64.dim == LATENT_DIM
    assert VIBEVOICE64.frame_rate == FRAME_RATE
    assert n_frames_from_samples(3200) == 1


def test_mel80_geometry():
    assert MEL80.name == "mel80"
    assert MEL80.kind == "mel"
    assert MEL80.dim == 80
    assert MEL80.sample_rate == 22_050
    assert MEL80.hop_length == 256
    assert MEL80.n_fft == 1024
    assert MEL80.win_length == 1024
    assert MEL80.n_mels == 80
    assert MEL80.samples_per_frame == 256
    assert abs(MEL80.frame_rate - (22_050 / 256)) < 1e-9
    assert abs(MEL80.frame_rate - 86.1328125) < 1e-9
    assert "vibevoice64" in TARGETS and "mel80" in TARGETS
    assert get_target("mel80") is MEL80
    assert get_target(None) is VIBEVOICE64
    with pytest.raises(KeyError):
        get_target("nope")


# ---------------------------------------------------------------------------
# A3 — projection math (toy intervals)
# ---------------------------------------------------------------------------


def test_soft_hard_projection_toy_7_5hz():
    # Two phones: [0, 0.2) and [0.2, 0.4) at 7.5 Hz → frame_dur ≈ 0.1333 s
    # Frame 0: [0, 0.1333) → only phone0, overlap 0.1333 → weight 1.0
    # Frame 1: [0.1333, 0.2667) → phone0: 0.0667, phone1: 0.0667 → 0.5 each
    # Frame 2: [0.2667, 0.4) → only phone1 → 1.0
    segs = [
        TimeSegment(0.0, 0.2, "a"),
        TimeSegment(0.2, 0.4, "b"),
    ]
    fr = 7.5
    soft = soft_overlap_projection(segs, frame_rate=fr, n_frames=3)
    assert soft.shape == (2, 3)
    np.testing.assert_allclose(soft[0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(soft[1, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(soft[0, 1], 0.5, atol=1e-6)
    np.testing.assert_allclose(soft[1, 1], 0.5, atol=1e-6)
    np.testing.assert_allclose(soft[0, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(soft[1, 2], 1.0, atol=1e-6)

    hard = hard_overlap_projection(segs, frame_rate=fr, n_frames=3)
    assert hard[0, 0] == 1.0 and hard[1, 0] == 0.0
    # tie on frame 1 → lowest index (phone a)
    assert hard[0, 1] == 1.0 and hard[1, 1] == 0.0
    assert hard[0, 2] == 0.0 and hard[1, 2] == 1.0

    col = normalize_columns(soft)
    np.testing.assert_allclose(col.sum(axis=0), [1.0, 1.0, 1.0], atol=1e-6)


def test_mel_frame_rate_projection():
    # Exact mel hop: 1 frame = 256/22050 s ≈ 0.01161 s
    # Phone covering exactly 2 mel frames: [0, 2/fr)
    fr = MEL80.frame_rate
    segs = [TimeSegment(0.0, 2.0 / fr, "x")]
    soft = soft_overlap_projection(segs, frame_rate="mel80", n_frames=2)
    assert soft.shape == (1, 2)
    np.testing.assert_allclose(soft[0], [1.0, 1.0], atol=1e-6)
    assert n_frames_for_duration(1.0, fr) == int(np.ceil(fr - 1e-12))


def test_segments_from_forced_phones_and_predictor(tmp_path: Path):
    entry = pack_forced_entry(
        words=["раз", "два"],
        dur_s=0.4,
        n_frames=3,
        method="uniform",
        extra={
            "words": ["раз", "два"],
            "phones": [
                {"start": 0.0, "end": 0.1, "phone": "r"},
                {"start": 0.1, "end": 0.25, "phone": "a"},
                {"start": 0.25, "end": 0.4, "phone": "d"},
            ],
        },
    )
    path = write_forced_json(tmp_path / "forced.json", {"u0": entry})
    segs = load_forced_segments(path, "u0")
    assert [s.label for s in segs] == ["r", "a", "d"]
    soft = soft_overlap_projection(segs, frame_rate=7.5)
    assert soft.shape[0] == 3
    assert soft.sum() > 0

    pred = MeanDurationPredictor.from_forced_json(path)
    assert abs(pred.means["r"] - 0.1) < 1e-9
    assert abs(pred.means["a"] - 0.15) < 1e-9
    durs = pred.predict_seconds(["r", "a", "unknown"])
    assert abs(durs[0] - 0.1) < 1e-9
    assert durs[2] == pred.default_s
    frames = pred.predict_frames(["r", "a"], frame_rate=7.5)
    assert all(f >= 1 for f in frames)
    stats = duration_stats_from_forced(path)
    assert stats["n_labels"] == 3


# ---------------------------------------------------------------------------
# A2 — uniform packing smoke (no corpus audio)
# ---------------------------------------------------------------------------


def test_uniform_packing_synthetic_fixture(tmp_path: Path):
    rows = [
        SimpleNamespace(
            utt_id="syn_001",
            text_stressed="привет мир",
            text_raw="привет мир",
            dur_s=2.0,
            n_frames=15,  # 2 s @ 7.5 Hz
            input_ids=[1, 2],
            split="train",
        ),
        SimpleNamespace(
            utt_id="syn_002",
            text_stressed="один",
            text_raw="один",
            dur_s=0.8,
            n_frames=6,
            input_ids=[3],
            split="train",
        ),
    ]
    forced = build_uniform_corpus_entries(rows, max_utts=2, use_accentuator=False)
    assert set(forced) == {"syn_001", "syn_002"}
    assert forced["syn_001"]["method"] == "uniform"
    assert forced["syn_001"]["granularity"] == "phone"
    # Phone-level: more units than whitespace words for multi-char Russian
    assert forced["syn_001"]["n_phones"] >= 2
    assert len(forced["syn_001"]["phone_labels"]) == forced["syn_001"]["n_phones"]
    assert len(forced["syn_001"]["phones"]) == forced["syn_001"]["n_phones"]
    # Mel / A3 prefer phones over words (grapheme fallback or espeak phone split)
    segs = load_forced_segments(write_forced_json(tmp_path / "forced.json", forced), "syn_001")
    assert [s.label for s in segs] == forced["syn_001"]["phone_labels"]
    assert len(segs) >= len(forced["syn_001"]["words"])
    # True phone/grapheme inventory should exceed whitespace word count when G2P splits
    if forced["syn_001"].get("g2p_backend") == "grapheme_fallback":
        assert len(segs) > len(forced["syn_001"]["words"])
    # Word spans still cover full frame range for Talker priors
    assert forced["syn_001"]["word_spans_frames"][-1][1] == 15
    reloaded = load_forced_json(tmp_path / "forced.json")
    assert "syn_002" in reloaded
    assert reloaded["syn_002"]["granularity"] == "phone"
    # MFA stub never requires install
    stub = mfa_crosscheck_stub(corpus_dir="/tmp/fake")
    assert stub["status"] == "not_run"
    assert "mfa_on_path" in stub


def test_phone_level_uniform_packing_equal_slices():
    from minispeech.data.forced_align import (
        pack_forced_entry,
        proportional_unit_spans_seconds,
        text_to_phones_for_align,
        uniform_unit_spans_seconds,
    )
    from minispeech.data.frame_projection import segments_from_forced_entry

    phones, meta = text_to_phones_for_align("приве́т мир", already_stressed=True, use_accentuator=False)
    assert meta["g2p_backend"] in {"phonemizer_espeak_ru", "grapheme_fallback"}
    # Phone/grapheme split: more than whitespace words (espeak may emit IPA phones or fallback chars)
    assert len(phones) >= 2
    if meta["g2p_backend"] == "grapheme_fallback" or any(len(p) == 1 for p in phones):
        assert len(phones) >= 4
    spans = uniform_unit_spans_seconds(len(phones), 1.0)
    assert abs(spans[0][1] - spans[0][0] - 1.0 / len(phones)) < 1e-9
    assert abs(spans[-1][1] - 1.0) < 1e-9
    entry = pack_forced_entry(
        words=["привет", "мир"],
        dur_s=1.0,
        n_frames=8,
        phones=phones,
        method="uniform",
    )
    segs = segments_from_forced_entry(entry)
    assert len(segs) == len(phones)
    assert all(s.label for s in segs)
    # Proportional path sums to dur_s
    prop = proportional_unit_spans_seconds([1.0, 2.0, 1.0], 4.0)
    assert abs(prop[1][1] - prop[1][0] - 2.0) < 1e-9
    assert abs(prop[-1][1] - 4.0) < 1e-9


def test_ctc_char_labels_strip_stress_and_punct():
    labels, meta = text_to_ctc_char_labels("Приве́т, мир!", already_stressed=True)
    assert "|" in labels
    assert "," not in labels and "!" not in labels
    assert "\u0301" not in "".join(labels)
    assert meta["granularity"] == "char"
    # "привет|мир" style
    joined = "".join(labels)
    assert joined.startswith("привет")
    assert "мир" in joined


def test_try_ctc_align_emissions_mocked():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    # blank=0, targets=[1, 2]; clear path with intervening blank
    T, C = 8, 4
    log_probs = torch.full((T, C), -20.0)
    for t in range(T):
        log_probs[t, 0] = 0.0
    log_probs[2:4, 1] = 0.0
    log_probs[2:4, 0] = -20.0
    log_probs[5:7, 2] = 0.0
    log_probs[5:7, 0] = -20.0
    log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)
    targets = torch.tensor([1, 2], dtype=torch.int32)
    spans = try_ctc_align_emissions(log_probs, targets, blank=0)
    assert spans is not None
    assert len(spans) == 2
    assert spans[0] == (2, 4)
    assert spans[1] == (5, 7)


def test_ctc_forced_entry_pack_char_granularity(tmp_path: Path):
    labels = list("привет") + ["|"] + list("мир")
    # Unequal spans (not uniform)
    spans = []
    t = 0.0
    for i, _ in enumerate(labels):
        step = 0.05 if labels[i] != "|" else 0.08
        spans.append((t, t + step))
        t += step
    entry = pack_forced_entry(
        words=["привет", "мир"],
        dur_s=t,
        n_frames=max(1, int(round(t * FRAME_RATE))),
        phones=labels,
        phone_spans_seconds=spans,
        method="ctc",
        granularity="char",
        extra={"align_unit": "char", "g2p_backend": "wav2vec2_char_ctc"},
    )
    assert entry["method"] == "ctc"
    assert entry["granularity"] == "char"
    assert entry["n_phones"] == len(labels)
    path = write_forced_json(tmp_path / "forced_ctc.json", {"u0": entry})
    segs = load_forced_segments(path, "u0")
    assert [s.label for s in segs] == labels
    # Durations must follow CTC spans, not equal slices
    assert abs(segs[0].duration - 0.05) < 1e-9
    assert abs(segs[6].duration - 0.08) < 1e-9  # "|"
    from minispeech.data.mel_dataset import build_vocab_from_forced

    vocab = build_vocab_from_forced(path)
    assert vocab.unit == "char"
    assert "|" in vocab.stoi


def test_ctc_gap_fill_cover_ratio_full_timeline():
    """Synthetic gappy CTC spans → sil fill → packed durs cover full utt."""
    from minispeech.data.frame_projection import (
        SIL_LABEL,
        fill_segment_gaps,
        phones_and_frame_durs_from_forced,
        segment_cover_ratio,
        segments_from_forced_entry,
        snap_frame_durations,
    )

    # Mimic S1c: phone active spans leave ~62% blank gaps; utt_end = 1.0 s
    labels = ["с", "|", "т", "р"]
    # active total 0.38 s; gaps make cover_ratio ≈ 0.38 vs end=1.0
    raw_spans = [
        (0.05, 0.12),  # lead gap 0.05
        (0.20, 0.28),  # gap 0.08
        (0.40, 0.55),  # gap 0.12
        (0.70, 0.88),  # trail to 1.0
    ]
    entry = pack_forced_entry(
        words=["с", "тр"],
        dur_s=1.0,
        n_frames=8,
        phones=labels,
        phone_spans_seconds=raw_spans,
        method="ctc",
        granularity="char",
        extra={"dur_s": 1.0, "align_unit": "char"},
    )
    raw = segments_from_forced_entry(entry)
    cover_raw = segment_cover_ratio(raw, utt_end=1.0)
    assert cover_raw < 0.5  # gappy like production CTC

    filled = fill_segment_gaps(
        raw, utt_end=1.0, gap_label=SIL_LABEL, mode="sil_fill"
    )
    assert any(s.label == SIL_LABEL for s in filled)
    assert abs(filled[0].start - 0.0) < 1e-9
    assert abs(filled[-1].end - 1.0) < 1e-9
    # Contiguous: each end == next start
    for a, b in zip(filled, filled[1:]):
        assert abs(a.end - b.start) < 1e-9
    cover_filled = segment_cover_ratio(filled, utt_end=1.0)
    assert cover_filled == pytest.approx(1.0, abs=1e-6)

    fr = MEL80.frame_rate
    n_mel = int(round(1.0 * fr))
    phones, durs, meta = phones_and_frame_durs_from_forced(
        entry,
        frame_rate=fr,
        n_frames=n_mel,
        fill_gaps=True,
        gap_mode="sil_fill",
    )
    assert SIL_LABEL in phones
    assert sum(durs) == n_mel
    assert meta["cover_ratio_vs_end"] == pytest.approx(1.0, abs=1e-6)
    assert meta["cover_ratio_raw"] < 0.5
    assert abs(sum(durs) - n_mel) <= 0  # exact after snap

    # snap helper alone
    assert sum(snap_frame_durations([3, 3, 3], 10)) == 10
    assert sum(snap_frame_durations([10, 10], 5)) == 5


def test_expand_neighbor_cover_ratio_full_timeline(tmp_path: Path):
    """expand_neighbor absorbs CTC gaps into phones; cover=1, no sil."""
    from minispeech.data.frame_projection import (
        SIL_LABEL,
        fill_segment_gaps,
        phones_and_frame_durs_from_forced,
        segment_cover_ratio,
        segments_from_forced_entry,
    )
    from minispeech.data.mel_dataset import build_vocab_from_forced
    from minispeech.frontend.durations import (
        MeanDurationPredictor,
        pack_infer_phones_and_durs,
    )

    labels = ["с", "|", "т", "р"]
    raw_spans = [
        (0.05, 0.12),
        (0.20, 0.28),
        (0.40, 0.55),
        (0.70, 0.88),
    ]
    entry = pack_forced_entry(
        words=["с", "тр"],
        dur_s=1.0,
        n_frames=8,
        phones=labels,
        phone_spans_seconds=raw_spans,
        method="ctc",
        granularity="char",
        extra={"dur_s": 1.0, "align_unit": "char"},
    )
    raw = segments_from_forced_entry(entry)
    assert segment_cover_ratio(raw, utt_end=1.0) < 0.5

    expanded = fill_segment_gaps(raw, utt_end=1.0, mode="expand_neighbor")
    assert len(expanded) == len(raw)
    assert all(s.label != SIL_LABEL for s in expanded)
    assert abs(expanded[0].start - 0.0) < 1e-9
    assert abs(expanded[-1].end - 1.0) < 1e-9
    for a, b in zip(expanded, expanded[1:]):
        assert abs(a.end - b.start) < 1e-9
    assert segment_cover_ratio(expanded, utt_end=1.0) == pytest.approx(1.0, abs=1e-6)
    # Mid-gap split between first two phones
    mid01 = 0.5 * (0.12 + 0.20)
    assert expanded[0].end == pytest.approx(mid01, abs=1e-9)
    assert expanded[1].start == pytest.approx(mid01, abs=1e-9)

    fr = MEL80.frame_rate
    n_mel = int(round(1.0 * fr))
    phones, durs, meta = phones_and_frame_durs_from_forced(
        entry,
        frame_rate=fr,
        n_frames=n_mel,
        fill_gaps=True,
        gap_mode="expand_neighbor",
    )
    assert SIL_LABEL not in phones
    assert phones == labels
    assert sum(durs) == n_mel
    assert meta["cover_ratio_vs_end"] == pytest.approx(1.0, abs=1e-6)
    assert meta["n_sil"] == 0
    assert meta["gap_mode"] == "expand_neighbor"

    path = write_forced_json(tmp_path / "forced_expand.json", {"u0": entry})
    vocab = build_vocab_from_forced(path, gap_mode="expand_neighbor")
    assert SIL_LABEL not in vocab.stoi
    assert "|" in vocab.stoi

    pred = MeanDurationPredictor.from_forced_json(path, gap_mode="expand_neighbor")
    assert SIL_LABEL not in pred.means
    free_phones, free_durs, free_meta = pack_infer_phones_and_durs(
        labels,
        pred,
        mode="mean_stats",
        frame_rate=fr,
        sil_policy="none",
        gap_mode="expand_neighbor",
        auto_target=True,
    )
    assert SIL_LABEL not in free_phones
    assert free_meta["n_sil"] == 0
    assert free_phones == labels
    assert sum(free_durs) == free_meta["sum_dur_frames"]
    assert free_meta["cover_ratio_vs_end"] == pytest.approx(1.0)


def test_free_infer_sil_pack_full_timeline(tmp_path: Path):
    """Free-infer label+dur packing covers a contiguous sil/char/| timeline."""
    from minispeech.data.forced_align import text_to_ctc_char_labels
    from minispeech.frontend.durations import (
        MeanDurationPredictor,
        SilGapModel,
        interleave_sil_labels,
        pack_infer_phones_and_durs,
    )
    from minispeech.frontend.phone_vocab import SIL

    # Build mean stats from a gappy CTC entry (sil_fill train path)
    labels = ["п", "р", "и", "|", "м", "и", "р"]
    raw_spans = [
        (0.04, 0.10),
        (0.14, 0.20),
        (0.24, 0.30),
        (0.36, 0.42),
        (0.50, 0.56),
        (0.60, 0.66),
        (0.70, 0.78),
    ]
    entry = pack_forced_entry(
        words=["при", "мир"],
        dur_s=1.0,
        n_frames=8,
        phones=labels,
        phone_spans_seconds=raw_spans,
        method="ctc",
        granularity="char",
        extra={"dur_s": 1.0, "align_unit": "char"},
    )
    # Second utt reinforces contig р→и vs gap и→|
    entry2 = pack_forced_entry(
        words=["при", "мир"],
        dur_s=1.0,
        n_frames=8,
        phones=labels,
        phone_spans_seconds=[
            (0.02, 0.08),
            (0.08, 0.14),  # п→р contig (no gap)
            (0.20, 0.26),
            (0.32, 0.38),
            (0.46, 0.52),
            (0.52, 0.58),  # м→и contig
            (0.66, 0.74),
        ],
        method="ctc",
        granularity="char",
        extra={"dur_s": 1.0, "align_unit": "char"},
    )
    path = write_forced_json(tmp_path / "forced_infer.json", {"u0": entry, "u1": entry2})
    pred = MeanDurationPredictor.from_forced_json(path, gap_mode="sil_fill")
    gap = SilGapModel.from_forced_json(path, frame_rate=MEL80.frame_rate)
    assert SIL in pred.means or SIL in pred.counts or pred.default_s > 0
    assert gap.frames_per_content > 0

    raw, _ = text_to_ctc_char_labels("при мир", already_stressed=True)
    assert "|" in raw
    assert SIL not in raw  # char CTC alone has no sil

    interleaved = interleave_sil_labels(raw, policy="interleave")
    assert interleaved[0] == SIL and interleaved[-1] == SIL
    assert interleaved.count(SIL) == len(raw) + 1

    bigram_packed = interleave_sil_labels(raw, policy="bigram", gap_model=gap)
    assert bigram_packed[0] == SIL and bigram_packed[-1] == SIL
    assert "|" in bigram_packed
    # Bigram policy should not always exceed interleave sil count
    assert bigram_packed.count(SIL) <= interleaved.count(SIL)

    phones, durs, meta = pack_infer_phones_and_durs(
        raw,
        pred,
        mode="mean_stats",
        frame_rate=MEL80.frame_rate,
        sil_policy="bigram",
        gap_mode="sil_fill",
        gap_model=gap,
        auto_target=True,
    )
    assert phones[0] == SIL and phones[-1] == SIL
    assert "|" in phones
    assert meta["n_sil"] >= 2
    assert len(phones) == len(durs)
    assert all(d >= 1 for d in durs)
    assert sum(durs) == meta["sum_dur_frames"]
    assert meta["cover_ratio_vs_end"] == pytest.approx(1.0)
    assert meta["target_frames"] == sum(durs)

    # Explicit snap target still full coverage
    _phones2, durs2, meta2 = pack_infer_phones_and_durs(
        raw,
        pred,
        mode="mean_stats",
        frame_rate=MEL80.frame_rate,
        sil_policy="bigram",
        gap_mode="sil_fill",
        gap_model=gap,
        target_frames=200,
        snap=True,
        auto_target=False,
    )
    assert sum(durs2) == 200
    assert meta2["cover_ratio_vs_end"] == pytest.approx(1.0)

    # uniform mode
    _, durs_u, meta_u = pack_infer_phones_and_durs(
        raw,
        None,
        mode="uniform",
        frame_rate=MEL80.frame_rate,
        sil_policy="interleave",
        gap_mode="sil_fill",
        uniform_frames_per_unit=3,
        target_frames=None,
        snap=False,
        auto_target=False,
    )
    assert all(d == 3 for d in durs_u)
    assert meta_u["n_sil"] >= 2

    # oracle_forced uses filled forced entry
    phones_o, durs_o, meta_o = pack_infer_phones_and_durs(
        raw,
        mode="oracle_forced",
        frame_rate=MEL80.frame_rate,
        forced_entry=entry,
        gap_mode="sil_fill",
        target_frames=int(round(1.0 * MEL80.frame_rate)),
    )
    assert SIL in phones_o
    assert sum(durs_o) == int(round(1.0 * MEL80.frame_rate))
    assert meta_o["cover_ratio_vs_end"] == pytest.approx(1.0, abs=1e-6)

    # word policy: sil around | and edges only
    word_packed = interleave_sil_labels(raw, policy="word")
    assert word_packed[0] == SIL and word_packed[-1] == SIL
    assert word_packed.count(SIL) < interleaved.count(SIL)

    # forced_lookup: exact content match reuses gap-filled oracle timeline
    forced = {"u0": entry, "u1": entry2}
    phones_l, durs_l, meta_l = pack_infer_phones_and_durs(
        labels,  # same content as entry (no sil)
        pred,
        mode="mean_stats",
        frame_rate=MEL80.frame_rate,
        gap_mode="sil_fill",
        forced_index=forced,
        allow_forced_lookup=True,
        auto_target=False,
    )
    assert meta_l.get("sil_policy") == "forced_lookup"
    assert meta_l.get("forced_lookup_utt_id") in {"u0", "u1"}
    assert SIL in phones_l
    assert sum(durs_l) >= 1
