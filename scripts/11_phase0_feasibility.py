#!/usr/bin/env python3
"""Phase 0 codec feasibility: roundtrip WER/CER, cache-equivalence, sensitivity, RTF.

Three-level gate: GO exit 0 / REVIEW exit 3 / NO-GO exit 2.
Without --skip-asr, unavailable ASR → exit 3 (not GO).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from minispeech.codec.cache_check import (
    cached_vs_full_stats,
    decode_chunked,
    evaluate_cache_gate,
)
from minispeech.codec.stats import LatentNormalizer
from minispeech.codec.wrapper import VibeVoiceCodec
from minispeech.constants import SR
from minispeech.eval.asr import AsrConfig, AsrEngine
from minispeech.eval.asr_wer import compute_cer, compute_wer, delta_wer
from minispeech.eval.bootstrap import (
    BootstrapResult,
    edits_from_jiwer_chars,
    edits_from_jiwer_words,
    paired_bootstrap_delta,
)
from minispeech.eval.latent_perturb import add_ar1_noise, add_white_noise
from minispeech.frontend.normalize import normalize_for_asr
from minispeech.utils import expand, load_yaml, set_seed


def load_metadata_texts(ruslan_dir: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for mp in list(ruslan_dir.rglob("metadata.csv")) + list(ruslan_dir.rglob("*.tsv")):
        sample = mp.read_text(encoding="utf-8", errors="ignore")[:4096]
        delim = "|" if sample.count("|") > sample.count(",") else ("\t" if "\t" in sample[:200] else ",")
        with open(mp, encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter=delim)
            for row in reader:
                if not row or row[0].lower() in {"id", "file", "filename", "wav"}:
                    continue
                key = Path(row[0].strip()).stem
                texts[key] = row[-1].strip()
    return texts


def load_sample_utterances(ruslan_dir: Path, n: int, seed: int = 42):
    wavs = sorted(ruslan_dir.rglob("*.wav"))
    rng = random.Random(seed)
    if not wavs:
        return []
    rng.shuffle(wavs)
    wavs = wavs[:n]
    meta = load_metadata_texts(ruslan_dir)
    out = []
    for wp in wavs:
        text = ""
        for ext in (".txt", ".lab"):
            tp = wp.with_suffix(ext)
            if tp.exists():
                text = tp.read_text(encoding="utf-8", errors="ignore").strip()
                break
        if not text:
            text = meta.get(wp.stem, "")
        out.append((wp, text))
    return out


def _gate_decision(
    structural_ok: bool,
    dwer: float | None,
    ci_high: float | None,
    ci_low: float | None,
    skip_asr: bool,
    asr_available: bool,
    max_go: float = 2.0,
    max_review: float = 4.0,
) -> tuple[str, int]:
    """Return (status, exit_code). status in GO|REVIEW|NO-GO|STRUCTURAL_FAIL."""
    if not structural_ok:
        return "NO-GO", 2
    if skip_asr:
        return "GO", 0  # structural-only when explicitly skipped
    if not asr_available or dwer is None:
        return "REVIEW", 3
    assert ci_high is not None and ci_low is not None
    if dwer <= max_go and ci_high <= max_review:
        return "GO", 0
    if dwer > max_review and ci_low > max_review:
        return "NO-GO", 2
    return "REVIEW", 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/codec.yaml")
    ap.add_argument("--data", default="configs/data_ruslan.yaml")
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--ruslan-dir", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--sweep-n", type=int, default=10, help="Utterances for sensitivity ASR sweep")
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--report", default=None)
    ap.add_argument("--records", default="docs/phase0_asr_records.jsonl")
    ap.add_argument(
        "--stats",
        default=None,
        help="Latent stats for normalized sensitivity (default: data.stats_path)",
    )
    args = ap.parse_args()
    ccfg = load_yaml(args.config)
    dcfg = load_yaml(args.data)["data"]
    p0 = ccfg["phase0"]
    set_seed(p0.get("seed", 42))

    # Sensitivity must use *normalized* latent units (see scripts/21_*; recovery plan E1).
    norm: LatentNormalizer | None = None
    stats_path = args.stats or dcfg.get("stats_path")
    if stats_path:
        try:
            norm = LatentNormalizer.from_file(stats_path)
            print(f"sensitivity units=normalized stats={stats_path}")
        except Exception as e:
            print(f"WARNING: stats unavailable ({e}); sensitivity falls back to raw latents")

    ruslan = expand(args.ruslan_dir or dcfg["ruslan_dir"])
    n = args.n or p0.get("n_utterances", 300)
    samples = load_sample_utterances(ruslan, n)
    if len(samples) < 5:
        print("WARNING: few/no RUSLAN wavs; using synthetic audio for structural checks only")
        samples = []
        for i in range(min(n, 16)):
            audio = (np.random.randn(SR * 2).astype(np.float32) * 0.05)
            path = expand("out/phase0_synth") / f"{i:03d}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), audio, SR)
            samples.append((path, "синтетический тест речи"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = VibeVoiceCodec(dtype=torch.bfloat16, device=device)
    print(f"layout={codec._latent_layout} device={device}")

    a0, _ = sf.read(str(samples[0][0]), dtype="float32")
    if a0.ndim > 1:
        a0 = a0.mean(-1)
    sr0 = sf.info(str(samples[0][0])).samplerate
    z1 = codec.encode(a0, sr0, sample=False).latents
    z2 = codec.encode(a0, sr0, sample=False).latents
    det_ok = bool(torch.equal(z1, z2))

    chunk_frames = int(p0.get("cache_chunk_frames", 4))
    boundary_ratio_max = float(p0.get("cache_boundary_ratio_max", 10.0))
    cache_check_dtype = str(p0.get("cache_check_dtype", "float32")).lower()
    gate_dtype = torch.float32 if cache_check_dtype in {"float32", "fp32"} else torch.bfloat16
    gate_limit = float(p0.get("cache_max_abs_diff", 1e-3))
    bf16_soft_limit = float(p0.get("cache_max_abs_diff_bf16", 2e-2))

    # Operating-dtype diagnostic (bf16): report only; soft-fail if absurdly large.
    enc_bf16 = codec.encode(a0, sr0, sample=False)
    full_bf16 = codec.decode(enc_bf16.latents).audio
    cached_bf16 = decode_chunked(codec, enc_bf16.latents, chunk_frames=chunk_frames)
    cache_stats_bf16 = cached_vs_full_stats(full_bf16, cached_bf16, chunk_frames=chunk_frames)
    bf16_ok, bf16_reason = evaluate_cache_gate(
        cache_stats_bf16,
        max_abs_limit=bf16_soft_limit,
        boundary_ratio_max=boundary_ratio_max,
    )

    # Structural gate: fp32 decode isolates real cache/seam bugs from bf16 noise.
    if gate_dtype == torch.bfloat16:
        codec_gate = codec
        enc_gate = enc_bf16
        full_gate = full_bf16
        cached_gate = cached_bf16
        cache_stats = cache_stats_bf16
    else:
        codec_gate = VibeVoiceCodec(dtype=torch.float32, device=device)
        enc_gate = codec_gate.encode(a0, sr0, sample=False)
        full_gate = codec_gate.decode(enc_gate.latents).audio
        cached_gate = decode_chunked(codec_gate, enc_gate.latents, chunk_frames=chunk_frames)
        cache_stats = cached_vs_full_stats(full_gate, cached_gate, chunk_frames=chunk_frames)
        del codec_gate
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cache_ok, cache_reason = evaluate_cache_gate(
        cache_stats,
        max_abs_limit=gate_limit,
        boundary_ratio_max=boundary_ratio_max,
    )
    # Still fail hard on a localized seam even if only seen under bf16.
    if not bf16_ok and "boundary_spike" in bf16_reason:
        cache_ok = False
        cache_reason = f"bf16 {bf16_reason}"
    max_abs = float(cache_stats["max_abs"])
    max_abs_bf16 = float(cache_stats_bf16["max_abs"])

    t0 = time.perf_counter()
    _ = codec.decode(enc_bf16.latents)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dec_s = time.perf_counter() - t0
    audio_s = full_bf16.shape[-1] / SR
    rtf = dec_s / max(audio_s, 1e-6)
    rtf_ok = rtf < float(p0.get("decoder_rtf_max", 0.3))
    structural_ok = det_ok and cache_ok and rtf_ok
    print(
        f"cache_gate dtype={cache_check_dtype} max_abs={max_abs:.6g} "
        f"boundary_ratio={float(cache_stats['boundary_ratio']):.3g} ok={cache_ok} ({cache_reason}); "
        f"bf16_max_abs={max_abs_bf16:.6g} bf16_soft_ok={bf16_ok}"
    )

    sigmas = list(p0.get("sensitivity_sigmas", [0.0, 0.1, 0.3, 0.625, 1.0]))
    rho = float(p0.get("ar1_rho", 0.8))
    sweep: dict = {"white": [], "ar1": []}
    mu = enc_bf16.latents

    asr: AsrEngine | None = None
    asr_available = False
    asr_error: str | None = None
    if not args.skip_asr:
        try:
            asr = AsrEngine(AsrConfig.from_yaml(args.asr_config), device=device)
            asr._load()
            asr_available = True
        except Exception as e:
            asr_error = str(e)
            print(f"ASR unavailable: {e}")
            asr = None

    records: list[dict] = []
    refs, hyps_orig, hyps_codec = [], [], []
    word_orig, word_codec = [], []
    char_orig, char_codec = [], []

    for wp, text in tqdm(samples[: min(len(samples), n)], desc="roundtrip"):
        audio, sr = sf.read(str(wp), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(-1)
        enc_i = codec.encode(audio, sr, sample=False)
        dec_i = codec.decode(enc_i.latents).audio.squeeze().float().cpu().numpy()
        if asr is not None and text:
            ref_n = normalize_for_asr(text)
            hyp_o = normalize_for_asr(asr.transcribe(audio, sr))
            hyp_c = normalize_for_asr(asr.transcribe(dec_i, SR))
            refs.append(ref_n)
            hyps_orig.append(hyp_o)
            hyps_codec.append(hyp_c)
            ew_o = edits_from_jiwer_words(ref_n, hyp_o)
            ew_c = edits_from_jiwer_words(ref_n, hyp_c)
            ec_o = edits_from_jiwer_chars(ref_n, hyp_o)
            ec_c = edits_from_jiwer_chars(ref_n, hyp_c)
            word_orig.append(ew_o)
            word_codec.append(ew_c)
            char_orig.append(ec_o)
            char_codec.append(ec_c)
            rec = {
                "utt_id": wp.stem,
                "dur_s": float(len(audio) / sr),
                "ref_norm": ref_n,
                "hyp_orig": hyp_o,
                "hyp_codec": hyp_c,
                "n_ref_words": ew_o.ref_units,
                "n_ref_chars": ec_o.ref_units,
                "edits_orig": {
                    "sub": ew_o.substitutions,
                    "del": ew_o.deletions,
                    "ins": ew_o.insertions,
                    "hits": ew_o.hits,
                },
                "edits_codec": {
                    "sub": ew_c.substitutions,
                    "del": ew_c.deletions,
                    "ins": ew_c.insertions,
                    "hits": ew_c.hits,
                },
                "cer_orig": ec_o.edits / max(ec_o.ref_units, 1),
                "cer_codec": ec_c.edits / max(ec_c.ref_units, 1),
                "abs_char_errors_delta": ec_c.edits - ec_o.edits,
            }
            records.append(rec)

    wer_orig = compute_wer(refs, hyps_orig) if refs else None
    wer_codec = compute_wer(refs, hyps_codec) if refs else None
    cer_orig = compute_cer(refs, hyps_orig) if refs else None
    cer_codec = compute_cer(refs, hyps_codec) if refs else None
    dwer = delta_wer(wer_orig, wer_codec) if wer_orig is not None else None
    dcer = delta_wer(cer_orig, cer_codec) if cer_orig is not None else None

    boot_wer: BootstrapResult | None = None
    boot_cer: BootstrapResult | None = None
    if word_orig:
        boot_wer = paired_bootstrap_delta(word_orig, word_codec, threshold_pp=4.0)
        boot_cer = paired_bootstrap_delta(char_orig, char_codec, threshold_pp=4.0)
        dwer = boot_wer.delta_pp
        dcer = boot_cer.delta_pp

    # Sensitivity sweep on sweep-n utterances with texts (normalized latent σ)
    sweep_samples = [(wp, t) for wp, t in samples if t][: max(1, args.sweep_n)]
    if not sweep_samples:
        sweep_samples = samples[:1]
    sweep_refs = [normalize_for_asr(t) for _, t in sweep_samples if t]
    for kind in ("white", "ar1"):
        for sigma in sigmas:
            hyps_s = []
            rms_acc = []
            for wp, text in sweep_samples:
                audio, sr = sf.read(str(wp), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(-1)
                enc_s = codec.encode(audio, sr, sample=False)
                mu_s = enc_s.latents  # raw [B,T,D]
                if norm is not None:
                    z_unit = norm.normalize(mu_s.float())
                    if sigma == 0:
                        z_p = z_unit
                    elif kind == "white":
                        z_p = add_white_noise(z_unit, float(sigma))
                    else:
                        z_p = add_ar1_noise(z_unit, float(sigma), rho=rho)
                    z_dec = norm.denormalize(z_p).to(dtype=mu_s.dtype)
                else:
                    # Legacy raw-space path (not comparable to head MAE)
                    if sigma == 0:
                        z_dec = mu_s
                    elif kind == "white":
                        z_dec = mu_s + float(sigma) * torch.randn_like(mu_s)
                    else:
                        z_dec = add_ar1_noise(mu_s.float(), float(sigma), rho=rho).to(mu_s.dtype)
                aud = codec.decode(z_dec).audio.squeeze().float().cpu().numpy()
                rms_acc.append(float(np.sqrt(np.mean(aud**2))))
                if asr is not None and text:
                    # use_cache=True is fine: distinct PCM → distinct hash
                    hyps_s.append(normalize_for_asr(asr.transcribe(aud, SR, use_cache=True)))
            entry: dict = {
                "sigma": sigma,
                "units": "normalized" if norm is not None else "raw",
                "rms": float(np.mean(rms_acc)) if rms_acc else 0.0,
                "n": len(sweep_samples),
            }
            if hyps_s and sweep_refs:
                refs_s = sweep_refs[: len(hyps_s)]
                entry["wer"] = compute_wer(refs_s, hyps_s)
                entry["cer"] = compute_cer(refs_s, hyps_s)
            sweep[kind].append(entry)

    ci_high = boot_wer.ci95_high if boot_wer else None
    ci_low = boot_wer.ci95_low if boot_wer else None
    status, exit_code = _gate_decision(
        structural_ok,
        dwer,
        ci_high,
        ci_low,
        skip_asr=args.skip_asr,
        asr_available=asr_available and bool(refs),
    )

    worst_cer = sorted(records, key=lambda r: r["cer_codec"] - r["cer_orig"], reverse=True)[:10]
    worst_abs = sorted(records, key=lambda r: r["abs_char_errors_delta"], reverse=True)[:10]

    report = {
        "decision": status,
        "go": status == "GO",
        "deterministic_sample_false": det_ok,
        "cache_check_dtype": cache_check_dtype,
        "cache_max_abs_diff": max_abs,
        "cache_max_abs_diff_bf16": max_abs_bf16,
        "cache_stats": cache_stats,
        "cache_stats_bf16": cache_stats_bf16,
        "cache_ok": cache_ok,
        "cache_reason": cache_reason,
        "decoder_rtf": rtf,
        "rtf_ok": rtf_ok,
        "wer_status": (
            "skipped"
            if args.skip_asr
            else ("ok" if asr_available and refs else "unavailable")
        ),
        "asr_error": asr_error,
        "wer_orig": wer_orig if wer_orig is None else boot_wer.metric_orig if boot_wer else wer_orig,
        "wer_codec": wer_codec if wer_codec is None else boot_wer.metric_codec if boot_wer else wer_codec,
        "delta_wer_pp": dwer,
        "cer_orig": cer_orig if cer_orig is None else boot_cer.metric_orig if boot_cer else cer_orig,
        "cer_codec": cer_codec if cer_codec is None else boot_cer.metric_codec if boot_cer else cer_codec,
        "delta_cer_pp": dcer,
        "bootstrap_wer": (
            {
                "ci95_low": boot_wer.ci95_low,
                "ci95_high": boot_wer.ci95_high,
                "p_delta_gt_4pp": boot_wer.p_delta_gt_threshold,
            }
            if boot_wer
            else None
        ),
        "bootstrap_cer": (
            {
                "ci95_low": boot_cer.ci95_low,
                "ci95_high": boot_cer.ci95_high,
                "p_delta_gt_4pp": boot_cer.p_delta_gt_threshold,
            }
            if boot_cer
            else None
        ),
        "n_samples": len(samples),
        "n_with_text": len(refs),
        "latent_layout": codec._latent_layout,
        "sensitivity": sweep,
        "worst_by_cer_delta": worst_cer,
        "worst_by_abs_char_errors": worst_abs,
        "note": (
            "vae_std=0.625 is NOT a guaranteed noise budget; see sensitivity curves. "
            "Sensitivity σ is in normalized latent units when stats_path loads "
            "(comparable to head MAE); prefer scripts/21_latent_error_calibration.py "
            "for CER(σ)/CER(α)."
        ),
        "sensitivity_units": "normalized" if norm is not None else "raw",
        "gate": {
            "GO": "ΔWER ≤ 2 pp and CI95 high ≤ 4 pp",
            "REVIEW": "ΔWER in 2–4 pp or CI crosses 4 pp; or ASR unavailable",
            "NO-GO": "ΔWER > 4 pp and CI95 low > 4 pp",
        },
    }

    records_path = expand(args.records)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with open(records_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report_path = expand(args.report or p0.get("report_path", "docs/phase0_report.md"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Phase 0 — Codec feasibility report", ""]
    md.append(f"**Decision: {status}** (exit {exit_code})")
    md.append("")
    md.append("```json")
    md.append(json.dumps({k: v for k, v in report.items() if k not in ("sensitivity", "worst_by_cer_delta", "worst_by_abs_char_errors")}, indent=2))
    md.append("```")
    md.append("")
    md.append("## Decoder sensitivity sweep")
    md.append("")
    md.append(f"Sweep ASR on n={args.sweep_n} utterances. Treat as empirical sensitivity — do **not** assume σ=0.625 is safe a priori.")
    md.append("")
    for kind in ("white", "ar1"):
        md.append(f"### {kind}")
        md.append("| sigma | rms | wer | cer |")
        md.append("|------:|----:|----:|----:|")
        for e in sweep[kind]:
            md.append(f"| {e['sigma']} | {e['rms']:.4f} | {e.get('wer')} | {e.get('cer')} |")
        md.append("")
    md.append(f"Per-utterance records: `{records_path}`")
    report_path.write_text("\n".join(md), encoding="utf-8")
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("decision", "delta_wer_pp", "delta_cer_pp", "decoder_rtf", "cache_ok", "wer_status")}, indent=2))
    print(f"Wrote {report_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
