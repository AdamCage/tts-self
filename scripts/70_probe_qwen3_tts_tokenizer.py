#!/usr/bin/env python3
"""Wave 0 / D0: frozen Qwen3-TTS Tokenizer-12Hz GO/NO-GO probe.

Measures:
  - bit-identical re-encode
  - copy-synth CER on E2-compatible RUSLAN val[:n]
  - bank_entity DER/EER (Track A source → codec roundtrip)
  - codebook utilization
  - telephony degradation (G.711 µ-law sim)

Hard gate D0 (written to ``out/d1_discrete_speech_lm/d0_gate.json``):
  - copy-synth mean CER ≤ 0.073 (E2 continuous VibeVoice anchor)
  - bank_entity roundtrip DER not worse than continuous codec anchor
    (``--continuous-anchor-json`` or ``--continuous-anchor-only`` pass)
  - bit-identical re-encode PASS

Also writes decode wavs under ``out/d1_discrete_speech_lm/<system>/``.

Env note: ``qwen-tts`` needs ``transformers~=4.57``; VibeVoice continuous
anchor needs ``transformers>=5``. Run continuous-anchor pass first, then
switch pins and run the Qwen probe.

Example (WSL)::

    export PYTHONPATH=/mnt/c/projects/tts-self/src
    export MINISPEECH_ALLOW_DIRTY=1
    export HF_HOME=$HOME/hf_cache

    # Pass A — continuous VibeVoice anchor (transformers>=5)
    python scripts/70_probe_qwen3_tts_tokenizer.py --continuous-anchor-only

    # Pass B — Qwen D0 (transformers~=4.57 + qwen-tts)
    python scripts/70_probe_qwen3_tts_tokenizer.py \\
      --continuous-anchor-json out/d1_discrete_speech_lm/continuous_codec_anchor.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from minispeech.constants import MEL80, QWEN3TTS_12HZ, SR
from minispeech.data.channel import apply_telephony_channel, available_codecs
from minispeech.data.forced_align import text_to_ctc_char_labels, text_to_phones_for_align
from minispeech.data.manifest import filter_split, read_manifest
from minispeech.eval.asr import AsrConfig, AsrEngine
from minispeech.eval.asr_wer import compute_cer, compute_wer
from minispeech.eval.critical_spans import (
    aggregate_der_eer,
    parse_critical_spans,
    score_utterance_critical,
)
from minispeech.eval.test_sets import load_test_set
from minispeech.frontend.durations import MeanDurationPredictor, pack_infer_phones_and_durs
from minispeech.frontend.normalize import normalize_for_asr
from minispeech.frontend.phone_vocab import PhoneVocab
from minispeech.frontend.verbalize import normalize_verbalized_for_asr
from minispeech.infer.vocoder import mel_to_wav
from minispeech.models.mel_baseline import MelBaseline
from minispeech.utils import (
    expand,
    git_is_dirty,
    git_sha,
    load_yaml,
    set_seed,
    write_run_manifest,
)

PLAN_SLUG = "d1_discrete_speech_lm"
E2_ANCHOR_CER = 0.073
DEFAULT_OUT_ROOT = f"out/{PLAN_SLUG}"


def _load_e2_rows(dcfg: dict, n: int, split: str):
    rows = filter_split(read_manifest(dcfg["manifest"]), split)
    excl = set(dcfg.get("exclude_tags", ("exclude",)))
    rows = [r for r in rows if (r.qc_tag or "clean") not in excl]
    return rows[:n]


def _wav_path(row, dcfg: dict) -> Path:
    root = expand(dcfg.get("ruslan_dir") or dcfg.get("root") or "")
    for attr in ("wav", "wav_path", "audio_path", "path"):
        v = getattr(row, attr, None)
        if v:
            p = Path(str(v))
            if not p.is_absolute():
                p = root / p
            p = expand(p)
            if p.is_file():
                return p
            alt = root / "wavs" / Path(str(v)).name
            if alt.is_file():
                return alt
    cand = root / "wavs" / f"{row.utt_id}.wav"
    if cand.is_file():
        return cand
    for p in (root / "wavs").glob(f"{row.utt_id}*"):
        if p.suffix.lower() in {".wav", ".flac"}:
            return p
    raise FileNotFoundError(f"No wav for {row.utt_id}")


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _load_mel_model(ckpt_path: Path, device: torch.device) -> MelBaseline:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") or {}
    mcfg = cfg.get("model", {}).get("mel", {})
    sd = ckpt.get("ema") or ckpt["model"]
    emb_w = sd.get("emb.weight")
    vocab_size = int(emb_w.shape[0]) if emb_w is not None else int(mcfg.get("vocab_size", 512))
    predict_pitch = bool(mcfg.get("predict_pitch", False)) or any(
        k.startswith("pitch_head.") for k in sd
    )
    predict_energy = bool(mcfg.get("predict_energy", False)) or any(
        k.startswith("energy_head.") for k in sd
    )
    use_postnet = bool(mcfg.get("use_postnet", False)) or any(
        k.startswith("postnet.") for k in sd
    )
    model = MelBaseline(
        vocab_size=vocab_size,
        d_model=int(mcfg.get("d_model", 256)),
        n_mels=int(mcfg.get("n_mels", 80)),
        enc_layers=int(mcfg.get("enc_layers", 3)),
        dec_layers=int(mcfg.get("dec_layers", 4)),
        n_heads=int(mcfg.get("n_heads", 4)),
        ffn_dim=int(mcfg.get("ffn_dim", 1024)),
        dropout=float(mcfg.get("dropout", 0.1)),
        decoder=str(mcfg.get("decoder", "conv")),  # type: ignore[arg-type]
        upsample=str(mcfg.get("upsample", "hard")),  # type: ignore[arg-type]
        predict_duration=bool(mcfg.get("predict_duration", True)),
        predict_pitch=predict_pitch,
        predict_energy=predict_energy,
        use_postnet=use_postnet,
    )
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def _raw_labels(text: str, vocab: PhoneVocab):
    if getattr(vocab, "unit", "phone") == "char" or "|" in vocab.stoi:
        from minispeech.constants import STRESS_CHAR

        keep_stress = any(STRESS_CHAR in t for t in vocab.stoi)
        return text_to_ctc_char_labels(text, already_stressed=False, keep_stress=keep_stress)
    return text_to_phones_for_align(text, already_stressed=False)


def _synth_track_a(
    *,
    model: MelBaseline,
    vocab: PhoneVocab,
    predictor: MeanDurationPredictor,
    text: str,
    device: torch.device,
    vocoder_ckpt: str | None,
) -> tuple[np.ndarray, int]:
    raw, _tn = _raw_labels(text, vocab)
    phones, durs, _pmeta = pack_infer_phones_and_durs(
        raw,
        predictor,
        mode="mean_stats",
        frame_rate=float(MEL80.frame_rate),
        sil_policy="none",
        gap_mode="expand_neighbor",
        snap=True,
        auto_target=True,
    )
    ids = vocab.encode(phones)
    phone_ids = torch.tensor([ids], dtype=torch.long, device=device)
    phone_mask = torch.ones_like(phone_ids, dtype=torch.bool)
    durations = torch.tensor([durs], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model.infer(phone_ids, phone_mask=phone_mask, durations=durations)
    mel = out.mel_pred[0, out.frame_mask[0]].cpu()
    result = mel_to_wav(
        mel,
        vocoder_ckpt=vocoder_ckpt,
        prefer="neural",
        device=device,
    )
    if result.wav is None:
        raise RuntimeError(f"vocoder failed: {result.note}")
    return np.asarray(result.wav, dtype=np.float32), int(result.sample_rate)


def _select_bank_items(items: list[Any], n: int) -> list[Any]:
    by_cat: dict[str, list[Any]] = defaultdict(list)
    for it in items:
        meta = getattr(it, "meta", {}) or {}
        cat = str(meta.get("category") or "unknown")
        by_cat[cat].append(it)
    cats = sorted(by_cat.keys())
    out: list[Any] = []
    idx = {c: 0 for c in cats}
    while len(out) < n and cats:
        progressed = False
        for c in cats:
            i = idx[c]
            if i < len(by_cat[c]):
                out.append(by_cat[c][i])
                idx[c] = i + 1
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    return out


def _score_cer(ref_text: str, hyp_raw: str, *, verbalized: bool) -> dict[str, Any]:
    if verbalized:
        ref_n = normalize_verbalized_for_asr(ref_text)
        hyp_n = normalize_verbalized_for_asr(hyp_raw)
    else:
        ref_n = normalize_for_asr(ref_text)
        hyp_n = normalize_for_asr(hyp_raw)
    return {
        "cer": compute_cer([ref_n], [hyp_n]) if ref_n else None,
        "wer": compute_wer([ref_n], [hyp_n]) if ref_n else None,
        "ref_norm": ref_n,
        "hyp_norm": hyp_n,
        "hyp_raw": hyp_raw,
    }


def _domain_summary(rows: list[dict[str, Any]], *, with_critical: bool) -> dict[str, Any]:
    cers = [r["cer"] for r in rows if r.get("cer") is not None]
    wers = [r["wer"] for r in rows if r.get("wer") is not None]
    out: dict[str, Any] = {
        "n": len(rows),
        "n_scored": len(cers),
        "mean_cer": float(np.mean(cers)) if cers else None,
        "mean_wer": float(np.mean(wers)) if wers else None,
        "median_cer": float(np.median(cers)) if cers else None,
    }
    if with_critical:
        crit = [r["critical"] for r in rows if r.get("critical")]
        agg = aggregate_der_eer(crit)
        out["der"] = agg.get("der")
        out["eer"] = agg.get("eer")
        out["n_digits"] = agg.get("n_digits")
        out["n_digit_errors"] = agg.get("n_digit_errors")
        out["n_spans"] = agg.get("n_spans")
        out["n_span_errors"] = agg.get("n_span_errors")
    return out


def _codebook_utilization(code_batches: list[torch.Tensor], n_codebooks: int, vocab: int) -> dict:
    per_cb: list[dict[str, Any]] = []
    for cb in range(n_codebooks):
        counts: Counter[int] = Counter()
        total = 0
        for codes in code_batches:
            col = codes[:, cb].reshape(-1).tolist()
            counts.update(int(x) for x in col)
            total += len(col)
        used = len(counts)
        probs = np.array([counts[i] / total for i in range(vocab) if counts[i]], dtype=np.float64)
        entropy = float(-(probs * np.log(probs + 1e-12)).sum()) if probs.size else 0.0
        per_cb.append(
            {
                "codebook": cb,
                "n_used": used,
                "vocab": vocab,
                "utilization": used / vocab if vocab else None,
                "entropy_nats": entropy,
                "n_tokens": total,
            }
        )
    return {
        "n_codebooks": n_codebooks,
        "codebook_size": vocab,
        "per_codebook": per_cb,
        "mean_utilization": float(np.mean([c["utilization"] for c in per_cb])),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prepare_bank_sources(
    *,
    dcfg: dict,
    bank_set: str,
    n_bank: int,
    mel_ckpt: str,
    vocoder_ckpt: str,
    device: torch.device,
    src_dir: Path,
) -> list[dict[str, Any]]:
    ts = load_test_set(bank_set)
    items = _select_bank_items(list(ts.items), n_bank)
    ckpt_path = expand(mel_ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Track A ckpt missing: {ckpt_path}")
    model = _load_mel_model(ckpt_path, device)
    vocab = PhoneVocab.load(ckpt_path.parent / "phone_vocab.json")
    forced_path = expand(dcfg["forced_path"])
    predictor = MeanDurationPredictor.from_forced_json(forced_path, max_utts=2000)
    vocoder = str(expand(vocoder_ckpt)) if expand(vocoder_ckpt).is_file() else None
    src_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    for it in tqdm(items, desc="bank_src_synth"):
        wav, sr = _synth_track_a(
            model=model,
            vocab=vocab,
            predictor=predictor,
            text=it.text,
            device=device,
            vocoder_ckpt=vocoder,
        )
        out_wav = src_dir / f"{it.utt_id}.wav"
        sf.write(str(out_wav), wav, sr)
        out.append(
            {
                "utt_id": it.utt_id,
                "text": it.text,
                "critical_spans": (it.meta or {}).get("critical_spans") or [],
                "wav_path": str(out_wav),
                "sr": sr,
            }
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _run_continuous_anchor(args: argparse.Namespace) -> int:
    """Pass A: VibeVoice continuous codec metrics used as D0 DER anchor."""
    from minispeech.codec.wrapper import VibeVoiceCodec

    dcfg = load_yaml(args.data)["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA required for continuous-anchor pass; abort.")
        return 2

    out_root = expand(args.out_root)
    wav_dir = out_root / "vibevoice_continuous_copysynth"
    wav_dir.mkdir(parents=True, exist_ok=True)
    bank_src_dir = out_root / "bank_src_track_a"

    report: dict[str, Any] = {
        "probe": "continuous_codec_anchor",
        "codec": "vibevoice64",
        "role": "D0_DER_anchor",
        "e2_anchor_cer_reference": E2_ANCHOR_CER,
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
        "minispeech_allow_dirty": os.environ.get("MINISPEECH_ALLOW_DIRTY"),
    }

    bank_sources = _prepare_bank_sources(
        dcfg=dcfg,
        bank_set=args.bank_set,
        n_bank=args.n_bank,
        mel_ckpt=args.mel_ckpt,
        vocoder_ckpt=args.vocoder_ckpt,
        device=device,
        src_dir=bank_src_dir,
    )
    report["bank_sources_n"] = len(bank_sources)
    report["bank_src_dir"] = str(bank_src_dir)

    codec = VibeVoiceCodec(dtype=torch.bfloat16, device=device)
    asr = AsrEngine(AsrConfig.from_yaml(args.asr_config), device=device)
    asr._load()

    e2_rows = _load_e2_rows(dcfg, args.n_e2, args.split)
    e2_rt: list[dict[str, Any]] = []
    for u in tqdm(e2_rows, desc="vv_e2_roundtrip"):
        wp = _wav_path(u, dcfg)
        audio, sr = _read_mono(wp)
        enc = codec.encode(audio, sr, sample=False)
        dec = codec.decode(enc.latents).audio.squeeze().float().cpu().numpy()
        aud = np.asarray(dec, dtype=np.float32)
        sf.write(str(wav_dir / f"e2_{u.utt_id}.wav"), aud, SR)
        ref = u.text_stressed or u.text_raw
        hyp = asr.transcribe(aud, SR, use_cache=False)
        e2_rt.append({"utt_id": u.utt_id, **_score_cer(ref, hyp, verbalized=False)})

    bank_rt: list[dict[str, Any]] = []
    bank_wav = out_root / "vibevoice_continuous_bank"
    bank_wav.mkdir(parents=True, exist_ok=True)
    for src in tqdm(bank_sources, desc="vv_bank_roundtrip"):
        audio, sr = _read_mono(Path(src["wav_path"]))
        enc = codec.encode(audio, sr, sample=False)
        dec = codec.decode(enc.latents).audio.squeeze().float().cpu().numpy()
        aud = np.asarray(dec, dtype=np.float32)
        sf.write(str(bank_wav / f"{src['utt_id']}.wav"), aud, SR)
        hyp = asr.transcribe(aud, SR, use_cache=False)
        sc = _score_cer(src["text"], hyp, verbalized=True)
        spans = parse_critical_spans(src["critical_spans"])
        crit = score_utterance_critical(src["text"], spans, sc["hyp_norm"], hyp_raw=hyp)
        bank_rt.append({"utt_id": src["utt_id"], **sc, "critical": crit})

    e2_sum = _domain_summary(e2_rt, with_critical=False)
    bank_sum = _domain_summary(bank_rt, with_critical=True)
    report["e2_copy_synth"] = e2_sum
    report["bank_entity_roundtrip"] = bank_sum
    report["bank_entity_der"] = bank_sum.get("der")
    report["bank_entity_eer"] = bank_sum.get("eer")
    report["rows_e2"] = e2_rt
    report["rows_bank"] = bank_rt

    out_path = expand(args.continuous_anchor_out)
    _write_json(out_path, report)
    write_run_manifest(
        out_root / "continuous_anchor_run",
        {"probe": "continuous_codec_anchor", "args": vars(args)},
        extra={"artifact": str(out_path)},
        enforce_clean=True,
    )
    print(json.dumps({"wrote": str(out_path), "e2": e2_sum, "bank": bank_sum}, indent=2))
    return 0


def _run_qwen_probe(args: argparse.Namespace) -> int:
    from minispeech.codec.qwen3_tts_wrapper import Qwen3TTSCodec

    dcfg = load_yaml(args.data)["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA required for D0 Qwen probe; abort.")
        return 2

    out_root = expand(args.out_root)
    system = "qwen3tts_12hz_copysynth"
    wav_dir = out_root / system
    wav_dir.mkdir(parents=True, exist_ok=True)
    bank_src_dir = out_root / "bank_src_track_a"

    report: dict[str, Any] = {
        "probe": "d0_qwen3_tts_tokenizer",
        "plan_slug": PLAN_SLUG,
        "target": QWEN3TTS_12HZ.name,
        "geometry": {
            "sample_rate": QWEN3TTS_12HZ.sample_rate,
            "frame_rate": QWEN3TTS_12HZ.frame_rate,
            "samples_per_frame": QWEN3TTS_12HZ.samples_per_frame,
            "n_codebooks": QWEN3TTS_12HZ.n_codebooks,
            "codebook_size": QWEN3TTS_12HZ.codebook_size,
            "code_dtype": QWEN3TTS_12HZ.code_dtype,
        },
        "e2_cer_ceiling": E2_ANCHOR_CER,
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
        "minispeech_allow_dirty": os.environ.get("MINISPEECH_ALLOW_DIRTY"),
        "telephony": {
            "codec": args.telephony_codec,
            "available": available_codecs(),
        },
        "domains": {},
    }

    if bank_src_dir.is_dir() and any(bank_src_dir.glob("*.wav")) and not args.resynth_bank:
        # reuse sources from continuous-anchor pass for fair DER compare
        ts = load_test_set(args.bank_set)
        by_id = {it.utt_id: it for it in ts.items}
        bank_sources = []
        for p in sorted(bank_src_dir.glob("*.wav")):
            utt_id = p.stem
            it = by_id.get(utt_id)
            if it is None:
                continue
            bank_sources.append(
                {
                    "utt_id": utt_id,
                    "text": it.text,
                    "critical_spans": (it.meta or {}).get("critical_spans") or [],
                    "wav_path": str(p),
                    "sr": 22050,
                }
            )
            if len(bank_sources) >= args.n_bank:
                break
        report["bank_sources"] = {
            "kind": "reused_track_a",
            "dir": str(bank_src_dir),
            "n": len(bank_sources),
        }
    elif not args.skip_bank:
        bank_sources = _prepare_bank_sources(
            dcfg=dcfg,
            bank_set=args.bank_set,
            n_bank=args.n_bank,
            mel_ckpt=args.mel_ckpt,
            vocoder_ckpt=args.vocoder_ckpt,
            device=device,
            src_dir=bank_src_dir,
        )
        report["bank_sources"] = {
            "kind": "track_a_mel_synth",
            "dir": str(bank_src_dir),
            "n": len(bank_sources),
        }
    else:
        bank_sources = []

    print("Loading Qwen3-TTS Tokenizer-12Hz + ASR...")
    codec = Qwen3TTSCodec(device_map="cuda:0")
    report["codec_fingerprint"] = codec.weight_fingerprint()
    asr = AsrEngine(AsrConfig.from_yaml(args.asr_config), device=device)
    asr._load()

    # Bit-identical re-encode smoke on first E2 wav
    e2_rows = _load_e2_rows(dcfg, args.n_e2, args.split)
    bit_identical = None
    if e2_rows:
        wp0 = _wav_path(e2_rows[0], dcfg)
        a0, sr0 = _read_mono(wp0)
        c1 = codec.encode(a0, sr0, sample=False).codes[0]
        c2 = codec.encode(a0, sr0, sample=False).codes[0]
        bit_identical = bool(torch.equal(c1, c2))
    report["bit_identical_reencode"] = bit_identical

    # E2 copy-synth + telephony
    e2_rt: list[dict[str, Any]] = []
    e2_tel: list[dict[str, Any]] = []
    code_store: list[torch.Tensor] = []
    e2_wav = wav_dir / "e2"
    e2_wav.mkdir(parents=True, exist_ok=True)
    e2_tel_wav = wav_dir / "e2_telephony"
    e2_tel_wav.mkdir(parents=True, exist_ok=True)

    for u in tqdm(e2_rows, desc="qwen_e2_copysynth"):
        wp = _wav_path(u, dcfg)
        audio, sr = _read_mono(wp)
        enc = codec.encode(audio, sr, sample=False)
        code_store.append(enc.codes[0])
        dec = codec.decode(enc.codes).audio[0]
        sf.write(str(e2_wav / f"{u.utt_id}.wav"), dec, codec.sample_rate)
        ref = u.text_stressed or u.text_raw
        hyp = asr.transcribe(dec, codec.sample_rate, use_cache=False)
        e2_rt.append(
            {
                "utt_id": u.utt_id,
                "n_frames": enc.n_frames_true[0],
                "wav": str(e2_wav / f"{u.utt_id}.wav"),
                **_score_cer(ref, hyp, verbalized=False),
            }
        )

        tel, tel_sr = apply_telephony_channel(
            audio, sr, codec=args.telephony_codec, restore_sr=True, seed=args.seed
        )
        enc_t = codec.encode(tel, tel_sr, sample=False)
        dec_t = codec.decode(enc_t.codes).audio[0]
        sf.write(str(e2_tel_wav / f"{u.utt_id}.wav"), dec_t, codec.sample_rate)
        hyp_t = asr.transcribe(dec_t, codec.sample_rate, use_cache=False)
        e2_tel.append({"utt_id": u.utt_id, **_score_cer(ref, hyp_t, verbalized=False)})

    e2_sum = _domain_summary(e2_rt, with_critical=False)
    e2_tel_sum = _domain_summary(e2_tel, with_critical=False)
    report["domains"]["e2_copy_synth"] = {
        "status": "ok",
        "selection": f"{args.split}[:{args.n_e2}] clean (E2-compatible)",
        "summary": e2_sum,
        "rows": e2_rt,
        "wav_dir": str(e2_wav),
    }
    report["domains"]["e2_telephony"] = {
        "status": "ok",
        "summary": e2_tel_sum,
        "delta_cer_vs_clean": (
            float(e2_tel_sum["mean_cer"] - e2_sum["mean_cer"])
            if e2_tel_sum["mean_cer"] is not None and e2_sum["mean_cer"] is not None
            else None
        ),
        "wav_dir": str(e2_tel_wav),
        "rows": e2_tel,
    }

    # bank_entity
    bank_sum = None
    if bank_sources:
        bank_rt: list[dict[str, Any]] = []
        bank_wav = wav_dir / "bank_entity"
        bank_wav.mkdir(parents=True, exist_ok=True)
        for src in tqdm(bank_sources, desc="qwen_bank_roundtrip"):
            audio, sr = _read_mono(Path(src["wav_path"]))
            enc = codec.encode(audio, sr, sample=False)
            code_store.append(enc.codes[0])
            dec = codec.decode(enc.codes).audio[0]
            sf.write(str(bank_wav / f"{src['utt_id']}.wav"), dec, codec.sample_rate)
            hyp = asr.transcribe(dec, codec.sample_rate, use_cache=False)
            sc = _score_cer(src["text"], hyp, verbalized=True)
            spans = parse_critical_spans(src["critical_spans"])
            crit = score_utterance_critical(src["text"], spans, sc["hyp_norm"], hyp_raw=hyp)
            bank_rt.append({"utt_id": src["utt_id"], **sc, "critical": crit})
        bank_sum = _domain_summary(bank_rt, with_critical=True)
        report["domains"]["bank_entity"] = {
            "status": "ok",
            "summary": bank_sum,
            "wav_dir": str(bank_wav),
            "rows": bank_rt,
        }

    report["codebook_utilization"] = _codebook_utilization(
        code_store,
        n_codebooks=int(QWEN3TTS_12HZ.n_codebooks or 16),
        vocab=int(QWEN3TTS_12HZ.codebook_size or 2048),
    )

    # Continuous anchor compare
    anchor_path = expand(args.continuous_anchor_json) if args.continuous_anchor_json else None
    cont_der = None
    cont_meta: dict[str, Any] = {"status": "missing"}
    if anchor_path is not None and anchor_path.is_file():
        cont = json.loads(anchor_path.read_text(encoding="utf-8"))
        cont_der = cont.get("bank_entity_der")
        if cont_der is None:
            cont_der = (cont.get("bank_entity_roundtrip") or {}).get("der")
        cont_meta = {
            "status": "ok",
            "path": str(anchor_path),
            "bank_entity_der": cont_der,
            "e2_mean_cer": (cont.get("e2_copy_synth") or {}).get("mean_cer"),
        }
    report["continuous_codec_anchor"] = cont_meta

    qwen_cer = e2_sum.get("mean_cer")
    qwen_der = bank_sum.get("der") if bank_sum else None
    cer_pass = qwen_cer is not None and float(qwen_cer) <= E2_ANCHOR_CER + 1e-12
    bit_pass = bit_identical is True
    if cont_der is None or qwen_der is None:
        der_pass = None
        der_note = "DER compare skipped (missing continuous anchor or bank metrics)"
    else:
        der_pass = float(qwen_der) <= float(cont_der) + 1e-12
        der_note = (
            f"qwen_der={qwen_der:.6f} vs continuous_der={float(cont_der):.6f} "
            f"({'PASS' if der_pass else 'FAIL'})"
        )

    if der_pass is False or cer_pass is False or bit_pass is False:
        verdict = "FAIL"
    elif der_pass is None:
        # CER+bit required; DER missing → FAIL closed (cannot claim D0 GO)
        verdict = "FAIL"
    else:
        verdict = "PASS"

    gate = {
        "gate": "D0",
        "verdict": verdict,
        "criteria": {
            "copy_synth_cer_max": E2_ANCHOR_CER,
            "copy_synth_cer": qwen_cer,
            "copy_synth_cer_pass": cer_pass,
            "bit_identical_reencode": bit_identical,
            "bit_identical_pass": bit_pass,
            "bank_entity_der": qwen_der,
            "continuous_bank_entity_der": cont_der,
            "bank_der_pass": der_pass,
            "bank_der_note": der_note,
        },
        "artifacts": {
            "probe_report": str(out_root / "d0_probe_report.json"),
            "gate": str(out_root / "d0_gate.json"),
            "wav_dir": str(wav_dir),
            "continuous_anchor": str(anchor_path) if anchor_path else None,
        },
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
    }
    report["d0_gate"] = gate

    probe_path = out_root / "d0_probe_report.json"
    gate_path = out_root / "d0_gate.json"
    _write_json(probe_path, report)
    _write_json(gate_path, gate)
    write_run_manifest(
        out_root / "d0_probe_run",
        {"probe": "d0_qwen3_tts_tokenizer", "args": {k: str(v) for k, v in vars(args).items()}},
        extra={"gate": gate},
        enforce_clean=True,
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"WROTE {gate_path}")
    return 0 if verdict == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-e2", type=int, default=50)
    ap.add_argument("--n-bank", type=int, default=40)
    ap.add_argument("--bank-set", default="configs/eval_sets/bank_entity.json")
    ap.add_argument("--mel-ckpt", default="runs/mel_stress_v01_s4/best.pt")
    ap.add_argument("--vocoder-ckpt", default="~/models/hifigan/generator.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--telephony-codec", default="mulaw", choices=["mulaw", "alaw", "none"])
    ap.add_argument("--skip-bank", action="store_true")
    ap.add_argument("--resynth-bank", action="store_true")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument(
        "--continuous-anchor-only",
        action="store_true",
        help="Pass A: measure VibeVoice continuous anchor (needs transformers>=5)",
    )
    ap.add_argument(
        "--continuous-anchor-out",
        default=f"{DEFAULT_OUT_ROOT}/continuous_codec_anchor.json",
    )
    ap.add_argument(
        "--continuous-anchor-json",
        default=f"{DEFAULT_OUT_ROOT}/continuous_codec_anchor.json",
        help="Pass B: path to continuous anchor JSON for DER compare",
    )
    args = ap.parse_args()
    set_seed(args.seed)

    if args.continuous_anchor_only:
        return _run_continuous_anchor(args)
    return _run_qwen_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
