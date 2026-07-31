#!/usr/bin/env python3
"""N08 / Probe Z2: reference VibeVoice tokenizer on domains (Track N).

Copy-synthesis: audio → encoder (sample=False) → latents → decoder → audio
→ ASR → CER (+ DER/EER on bank_entity via critical_spans).

VibeVoice weights are used ONLY as a reference / upper-bound probe — never as
product code. Every JSON report marks this explicitly.

Domains:
  - e2_anchor: RUSLAN val[:n] (same selection as scripts/21 E2)
  - e2_telephony: same utts after G.711 µ-law telephony sim (channel.py)
  - bank_entity: phrases from configs/eval_sets/bank_entity.json — no GT wavs,
    so source audio is Track A mel synth (labeled); then tokenizer roundtrip
  - bank_entity_telephony: same bank source through telephony sim
  - multi_speaker: skipped unless local multi-speaker corpus exists (N03)

Example (WSL)::

    cd ~/projects/tts-self && source .venv/bin/activate
    export HF_HOME=$HOME/hf_cache
    python scripts/41_probe_ref_tokenizer.py \\
      --out out/probe_ref_tokenizer_domains.json
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from minispeech.codec.wrapper import VibeVoiceCodec
from minispeech.constants import MEL80, SR
from minispeech.data.channel import apply_telephony_channel, available_codecs
from minispeech.data.forced_align import text_to_ctc_char_labels, text_to_phones_for_align
from minispeech.data.manifest import filter_split, read_manifest
from minispeech.data.shards import ShardReader
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
from minispeech.utils import expand, load_yaml, set_seed

E2_ANCHOR_CER = 0.073
# Absolute CER tolerance for "reproduced within reasonable allowance"
ANCHOR_ABS_TOL = 0.04


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
            # Sometimes stored relative to ruslan_dir/wavs
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
) -> tuple[np.ndarray, int, dict[str, Any]]:
    raw, tn_meta = _raw_labels(text, vocab)
    phones, durs, pmeta = pack_infer_phones_and_durs(
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
    meta = {**pmeta, **tn_meta, "vocoder_backend": result.backend}
    return np.asarray(result.wav, dtype=np.float32), int(result.sample_rate), meta


def _select_bank_items(items: list[Any], n: int) -> list[Any]:
    """Stratified take: round-robin by category until n."""
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


def _roundtrip(
    codec: VibeVoiceCodec,
    audio: np.ndarray,
    sr: int,
) -> np.ndarray:
    enc = codec.encode(audio, sr, sample=False)
    dec = codec.decode(enc.latents).audio.squeeze().float().cpu().numpy()
    return np.asarray(dec, dtype=np.float32)


def _asr_raw(asr: AsrEngine, audio: np.ndarray, sr: int) -> str:
    return asr.transcribe(audio, sr, use_cache=False)


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


def _free_cuda(*objs: Any) -> None:
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _domain_summary(
    rows: list[dict[str, Any]],
    *,
    with_critical: bool,
) -> dict[str, Any]:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-e2", type=int, default=50, help="E2-style val utterances")
    ap.add_argument("--n-bank", type=int, default=40, help="bank_entity subset size")
    ap.add_argument("--bank-set", default="configs/eval_sets/bank_entity.json")
    ap.add_argument(
        "--mel-ckpt",
        default="runs/mel_stress_v01_s4/best.pt",
        help="Track A ckpt for bank_entity source audio (phrases have no GT wav)",
    )
    ap.add_argument("--vocoder-ckpt", default="~/models/hifigan/generator.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--telephony-codec", default="mulaw", choices=["mulaw", "alaw", "none"])
    ap.add_argument("--skip-bank", action="store_true")
    ap.add_argument("--skip-gt-latent-check", action="store_true")
    ap.add_argument("--out", default="out/probe_ref_tokenizer_domains.json")
    args = ap.parse_args()
    set_seed(args.seed)

    dcfg = load_yaml(args.data)["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA required for N08; abort.")
        return 2

    report: dict[str, Any] = {
        "probe_id": "N08",
        "probe_name": "Z2_reference_tokenizer_domains",
        "reference_tokenizer": {
            "name": "VibeVoice Acoustic Tokenizer",
            "role": "reference_only",
            "product_use": False,
            "encode_sample": False,
            "note": (
                "Weights used solely as a measurable upper bound / temporary "
                "stand-in for Track N probes. Not embedded in product code."
            ),
        },
        "e2_anchor_target_cer": E2_ANCHOR_CER,
        "anchor_abs_tol": ANCHOR_ABS_TOL,
        "telephony": {
            "codec": args.telephony_codec,
            "available_codecs": available_codecs(),
            "source": "minispeech.data.channel.apply_telephony_channel",
            "note": "Native 8 kHz telephony corpus not local (N03); simulation only.",
        },
        "multi_speaker": {
            "status": "skipped",
            "reason": (
                "N03 inventory: no multi-speaker corpus downloaded locally; "
                "only RUSLAN (1 speaker) is on disk. Not downloading in N08."
            ),
        },
        "domains": {},
    }

    # --- bank_entity source synth (Track A) before loading heavy codec+ASR ---
    bank_sources: list[dict[str, Any]] = []
    if not args.skip_bank:
        ts = load_test_set(args.bank_set)
        items = _select_bank_items(list(ts.items), args.n_bank)
        ckpt_path = expand(args.mel_ckpt)
        if not ckpt_path.is_file():
            report["domains"]["bank_entity"] = {
                "status": "blocked",
                "reason": f"Track A ckpt missing for phrase synth: {ckpt_path}",
            }
            report["domains"]["bank_entity_telephony"] = {
                "status": "blocked",
                "reason": "depends on bank_entity source audio",
            }
        else:
            print(f"Synthesizing bank_entity source via Track A ({len(items)} utts)...")
            model = _load_mel_model(ckpt_path, device)
            vocab = PhoneVocab.load(ckpt_path.parent / "phone_vocab.json")
            forced_path = expand(dcfg["forced_path"])
            predictor = MeanDurationPredictor.from_forced_json(forced_path, max_utts=2000)
            vocoder = str(expand(args.vocoder_ckpt)) if expand(args.vocoder_ckpt).is_file() else None
            src_dir = expand("out/n08_bank_src_wavs")
            src_dir.mkdir(parents=True, exist_ok=True)
            for it in tqdm(items, desc="bank_src_synth"):
                wav, sr, meta = _synth_track_a(
                    model=model,
                    vocab=vocab,
                    predictor=predictor,
                    text=it.text,
                    device=device,
                    vocoder_ckpt=vocoder,
                )
                out_wav = src_dir / f"{it.utt_id}.wav"
                sf.write(str(out_wav), wav, sr)
                bank_sources.append(
                    {
                        "utt_id": it.utt_id,
                        "text": it.text,
                        "critical_spans": (it.meta or {}).get("critical_spans") or [],
                        "wav_path": str(out_wav),
                        "sr": sr,
                        "synth_meta": meta,
                    }
                )
            _free_cuda(model)
            report["bank_entity_audio_source"] = {
                "kind": "track_a_mel_synth",
                "ckpt": str(ckpt_path),
                "vocoder_ckpt": vocoder,
                "n": len(bank_sources),
                "note": (
                    "bank_entity items are text phrases without GT RUSLAN wavs; "
                    "Track A synth provides source audio for tokenizer roundtrip. "
                    "Absolute DER mixes TTS+codec; compare source vs roundtrip DER."
                ),
            }

    # --- codec + ASR ---
    print("Loading VibeVoice codec (reference) + ASR...")
    try:
        codec = VibeVoiceCodec(dtype=torch.bfloat16, device=device)
    except Exception as exc:
        report["blocker"] = {
            "type": "reference_tokenizer_weights_unavailable",
            "error": str(exc),
            "note": "Do not substitute a discrete RVQ codec; N08 aborted.",
        }
        out = expand(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report["blocker"], indent=2))
        return 3

    asr = AsrEngine(AsrConfig.from_yaml(args.asr_config), device=device)
    asr._load()

    # --- E2 anchor: GT audio full roundtrip ---
    e2_rows = _load_e2_rows(dcfg, args.n_e2, args.split)
    e2_rt: list[dict[str, Any]] = []
    e2_tel: list[dict[str, Any]] = []
    e2_gt_latent: list[dict[str, Any]] = []
    lat_reader = None
    if not args.skip_gt_latent_check:
        try:
            lat_reader = ShardReader(dcfg["latent_dir"], "latents")
        except Exception as exc:
            report["gt_latent_check_error"] = str(exc)

    for u in tqdm(e2_rows, desc="e2_roundtrip"):
        wp = _wav_path(u, dcfg)
        audio, sr = _read_mono(wp)
        ref = u.text_stressed or u.text_raw
        rt = _roundtrip(codec, audio, sr)
        hyp = _asr_raw(asr, rt, SR)
        scored = _score_cer(ref, hyp, verbalized=False)
        e2_rt.append({"utt_id": u.utt_id, "wav": str(wp), **scored})

        tel, tel_sr = apply_telephony_channel(
            audio, sr, codec=args.telephony_codec, restore_sr=True, seed=args.seed
        )
        rt_t = _roundtrip(codec, tel, tel_sr)
        hyp_t = _asr_raw(asr, rt_t, SR)
        scored_t = _score_cer(ref, hyp_t, verbalized=False)
        e2_tel.append({"utt_id": u.utt_id, **scored_t})

        if lat_reader is not None:
            try:
                z = lat_reader.get(u.utt_id).float().unsqueeze(0).to(device)
                aud_gt = codec.decode(z).audio.squeeze().float().cpu().numpy()
                hyp_g = _asr_raw(asr, np.asarray(aud_gt, dtype=np.float32), SR)
                e2_gt_latent.append(
                    {"utt_id": u.utt_id, **_score_cer(ref, hyp_g, verbalized=False)}
                )
            except Exception as exc:
                e2_gt_latent.append({"utt_id": u.utt_id, "error": str(exc)})

    e2_sum = _domain_summary(e2_rt, with_critical=False)
    e2_tel_sum = _domain_summary(e2_tel, with_critical=False)
    gt_lat_sum = _domain_summary(
        [r for r in e2_gt_latent if r.get("cer") is not None], with_critical=False
    )
    anchor_cer = e2_sum["mean_cer"]
    gt_lat_cer = gt_lat_sum.get("mean_cer")
    # Prefer GT-latent decode CER for anchor match (historical E2 procedure);
    # still require full roundtrip numbers.
    anchor_compare_cer = gt_lat_cer if gt_lat_cer is not None else anchor_cer
    anchor_delta = (
        abs(float(anchor_compare_cer) - E2_ANCHOR_CER)
        if anchor_compare_cer is not None
        else None
    )
    anchor_ok = anchor_delta is not None and anchor_delta <= ANCHOR_ABS_TOL

    report["domains"]["e2_anchor"] = {
        "status": "ok",
        "selection": f"{args.split}[:{args.n_e2}] clean (same as scripts/21_latent_error_calibration)",
        "procedure": "audio→encode(sample=False)→decode→ASR→CER",
        "summary": e2_sum,
        "gt_latent_decode_summary": gt_lat_sum if e2_gt_latent else None,
        "gt_latent_decode_note": (
            "Historical E2 used cached GT latents→decode (not re-encode). "
            "This check reproduces that path when latents exist."
        ),
        "anchor_target_cer": E2_ANCHOR_CER,
        "anchor_compared_cer": anchor_compare_cer,
        "anchor_compared_source": "gt_latent_decode" if gt_lat_cer is not None else "full_roundtrip",
        "anchor_abs_delta": anchor_delta,
        "anchor_reproduced": anchor_ok,
        "rows": e2_rt,
    }
    report["domains"]["e2_telephony"] = {
        "status": "ok",
        "procedure": (
            f"audio→telephony({args.telephony_codec})→encode(sample=False)"
            "→decode→ASR→CER"
        ),
        "summary": e2_tel_sum,
        "delta_cer_vs_clean_roundtrip": (
            float(e2_tel_sum["mean_cer"] - e2_sum["mean_cer"])
            if e2_tel_sum["mean_cer"] is not None and e2_sum["mean_cer"] is not None
            else None
        ),
        "rows": e2_tel,
    }

    # --- bank_entity ---
    if bank_sources:
        bank_rt: list[dict[str, Any]] = []
        bank_src: list[dict[str, Any]] = []
        bank_tel: list[dict[str, Any]] = []
        for src in tqdm(bank_sources, desc="bank_roundtrip"):
            audio, sr = _read_mono(Path(src["wav_path"]))
            spans = parse_critical_spans(src["critical_spans"])
            text = src["text"]

            hyp_s = _asr_raw(asr, audio, sr)
            sc_s = _score_cer(text, hyp_s, verbalized=True)
            crit_s = score_utterance_critical(text, spans, sc_s["hyp_norm"], hyp_raw=hyp_s)
            bank_src.append(
                {
                    "utt_id": src["utt_id"],
                    **sc_s,
                    "critical": crit_s,
                }
            )

            rt = _roundtrip(codec, audio, sr)
            hyp = _asr_raw(asr, rt, SR)
            sc = _score_cer(text, hyp, verbalized=True)
            crit = score_utterance_critical(text, spans, sc["hyp_norm"], hyp_raw=hyp)
            bank_rt.append(
                {
                    "utt_id": src["utt_id"],
                    **sc,
                    "critical": crit,
                }
            )

            tel, tel_sr = apply_telephony_channel(
                audio, sr, codec=args.telephony_codec, restore_sr=True, seed=args.seed
            )
            rt_t = _roundtrip(codec, tel, tel_sr)
            hyp_t = _asr_raw(asr, rt_t, SR)
            sc_t = _score_cer(text, hyp_t, verbalized=True)
            crit_t = score_utterance_critical(text, spans, sc_t["hyp_norm"], hyp_raw=hyp_t)
            bank_tel.append(
                {
                    "utt_id": src["utt_id"],
                    **sc_t,
                    "critical": crit_t,
                }
            )

        src_sum = _domain_summary(bank_src, with_critical=True)
        rt_sum = _domain_summary(bank_rt, with_critical=True)
        tel_sum = _domain_summary(bank_tel, with_critical=True)
        report["domains"]["bank_entity"] = {
            "status": "ok",
            "audio_source": "track_a_mel_synth",
            "procedure": (
                "TrackA_synth→ASR (source) and "
                "TrackA_synth→encode(sample=False)→decode→ASR (roundtrip); "
                "CER via normalize_verbalized_for_asr; DER/EER via critical_spans"
            ),
            "source_summary": src_sum,
            "roundtrip_summary": rt_sum,
            "delta_cer_roundtrip_minus_source": (
                float(rt_sum["mean_cer"] - src_sum["mean_cer"])
                if rt_sum["mean_cer"] is not None and src_sum["mean_cer"] is not None
                else None
            ),
            "delta_der_roundtrip_minus_source": (
                float(rt_sum["der"] - src_sum["der"])
                if rt_sum.get("der") is not None and src_sum.get("der") is not None
                else None
            ),
            "rows_roundtrip": bank_rt,
            "rows_source": bank_src,
        }
        report["domains"]["bank_entity_telephony"] = {
            "status": "ok",
            "procedure": (
                f"TrackA_synth→telephony({args.telephony_codec})→encode→decode→ASR"
            ),
            "summary": tel_sum,
            "delta_cer_vs_bank_roundtrip": (
                float(tel_sum["mean_cer"] - rt_sum["mean_cer"])
                if tel_sum["mean_cer"] is not None and rt_sum["mean_cer"] is not None
                else None
            ),
            "delta_der_vs_bank_roundtrip": (
                float(tel_sum["der"] - rt_sum["der"])
                if tel_sum.get("der") is not None and rt_sum.get("der") is not None
                else None
            ),
            "rows": bank_tel,
        }

    # Verdict block
    report["verdict"] = {
        "anchor_reproduced": report["domains"]["e2_anchor"].get("anchor_reproduced"),
        "anchor_compared_cer": report["domains"]["e2_anchor"].get("anchor_compared_cer"),
        "e2_full_roundtrip_cer": e2_sum.get("mean_cer"),
        "e2_telephony_cer": e2_tel_sum.get("mean_cer"),
        "telephony_delta_cer": report["domains"]["e2_telephony"].get(
            "delta_cer_vs_clean_roundtrip"
        ),
        "bank_entity_der_roundtrip": (
            report["domains"].get("bank_entity", {}).get("roundtrip_summary", {}).get("der")
        ),
        "bank_entity_eer_roundtrip": (
            report["domains"].get("bank_entity", {}).get("roundtrip_summary", {}).get("eer")
        ),
        "bank_entity_cer_roundtrip": (
            report["domains"].get("bank_entity", {}).get("roundtrip_summary", {}).get("mean_cer")
        ),
    }

    out = expand(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"wrote": str(out), "verdict": report["verdict"]}, indent=2))
    return 0 if report["domains"]["e2_anchor"].get("anchor_reproduced") else 1


if __name__ == "__main__":
    raise SystemExit(main())
