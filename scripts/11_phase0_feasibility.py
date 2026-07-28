#!/usr/bin/env python3
"""Phase 0 codec feasibility: roundtrip WER, cache-equivalence, sensitivity sweep, RTF.

Writes docs/phase0_report.md. Exit 0 = GO, 2 = NO-GO.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from minispeech.codec.wrapper import VibeVoiceCodec
from minispeech.constants import SR
from minispeech.eval.asr_wer import compute_wer, delta_wer
from minispeech.frontend.stress import strip_stress
from minispeech.utils import expand, load_yaml, set_seed


def load_sample_utterances(ruslan_dir: Path, n: int, seed: int = 42):
    wavs = sorted(ruslan_dir.rglob("*.wav"))
    rng = random.Random(seed)
    if not wavs:
        return []
    rng.shuffle(wavs)
    wavs = wavs[:n]
    out = []
    for wp in wavs:
        text = ""
        for ext in (".txt", ".lab"):
            tp = wp.with_suffix(ext)
            if tp.exists():
                text = tp.read_text(encoding="utf-8", errors="ignore").strip()
                break
        out.append((wp, text))
    return out


def ar1_noise(shape, rho: float, sigma: float, device, dtype):
    eps = torch.randn(*shape, device=device, dtype=dtype)
    out = torch.zeros_like(eps)
    out[:, 0] = eps[:, 0]
    for t in range(1, shape[1]):
        out[:, t] = rho * out[:, t - 1] + (1 - rho**2) ** 0.5 * eps[:, t]
    return out * sigma


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/codec.yaml")
    ap.add_argument("--data", default="configs/data_ruslan.yaml")
    ap.add_argument("--ruslan-dir", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    ccfg = load_yaml(args.config)
    dcfg = load_yaml(args.data)["data"]
    p0 = ccfg["phase0"]
    set_seed(p0.get("seed", 42))

    ruslan = expand(args.ruslan_dir or dcfg["ruslan_dir"])
    n = args.n or p0.get("n_utterances", 300)
    samples = load_sample_utterances(ruslan, n)
    if len(samples) < 5:
        # Synthetic fallback for CI / early scaffold
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

    # Determinism
    a0, _ = sf.read(str(samples[0][0]), dtype="float32")
    if a0.ndim > 1:
        a0 = a0.mean(-1)
    z1 = codec.encode(a0, sf.info(str(samples[0][0])).samplerate, sample=False).latents
    z2 = codec.encode(a0, sf.info(str(samples[0][0])).samplerate, sample=False).latents
    det_ok = bool(torch.equal(z1, z2))

    # Cache equivalence on first sample
    enc = codec.encode(a0, sf.info(str(samples[0][0])).samplerate, sample=False)
    full = codec.decode(enc.latents).audio
    cache = None
    chunks = []
    z = enc.latents
    for i in range(0, z.shape[1], 4):
        out = codec.decode(z[:, i : i + 4], use_cache=True, padding_cache=cache)
        cache = out.padding_cache
        chunks.append(out.audio)
    # lengths may differ slightly; compare overlap
    cached = torch.cat(chunks, dim=-1)
    nmin = min(full.shape[-1], cached.shape[-1])
    max_abs = float((full[..., :nmin] - cached[..., :nmin]).abs().max().cpu())
    cache_ok = max_abs < float(p0.get("cache_max_abs_diff", 1e-3))

    # RTF decoder
    t0 = time.perf_counter()
    _ = codec.decode(enc.latents)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dec_s = time.perf_counter() - t0
    audio_s = full.shape[-1] / SR
    rtf = dec_s / max(audio_s, 1e-6)
    rtf_ok = rtf < float(p0.get("decoder_rtf_max", 0.3))

    # Sensitivity sweep
    sigmas = list(p0.get("sensitivity_sigmas", [0.0, 0.1, 0.3, 0.625, 1.0]))
    rho = float(p0.get("ar1_rho", 0.8))
    sweep = {"white": [], "ar1": []}
    mu = enc.latents
    # Use a few utterances for sweep decode+optional ASR
    asr = None
    if not args.skip_asr:
        try:
            import whisper

            asr = whisper.load_model("large-v3", device=str(device))
        except Exception as e:
            print(f"ASR unavailable: {e}")

    refs, hyps_orig, hyps_codec = [], [], []
    for wp, text in tqdm(samples[: min(len(samples), n)], desc="roundtrip"):
        audio, sr = sf.read(str(wp), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(-1)
        enc_i = codec.encode(audio, sr, sample=False)
        dec_i = codec.decode(enc_i.latents).audio.squeeze().float().cpu().numpy()
        if asr is not None and text:
            # resample to 16k for whisper
            import torchaudio

            def to16(x, s):
                t = torch.from_numpy(x).float()
                if s != 16000:
                    t = torchaudio.functional.resample(t.unsqueeze(0), s, 16000).squeeze(0)
                return t.numpy()

            o = asr.transcribe(to16(audio, sr), language="ru", fp16=device.type == "cuda")
            c = asr.transcribe(to16(dec_i, SR), language="ru", fp16=device.type == "cuda")
            refs.append(strip_stress(text))
            hyps_orig.append(o.get("text", "").strip())
            hyps_codec.append(c.get("text", "").strip())

    wer_orig = compute_wer(refs, hyps_orig) if refs else None
    wer_codec = compute_wer(refs, hyps_codec) if refs else None
    dwer = delta_wer(wer_orig, wer_codec) if wer_orig is not None else None

    for kind in ("white", "ar1"):
        for sigma in sigmas:
            if sigma == 0:
                z_n = mu
            elif kind == "white":
                z_n = mu + sigma * torch.randn_like(mu)
            else:
                z_n = mu + ar1_noise(mu.shape, rho, sigma, mu.device, mu.dtype)
            aud = codec.decode(z_n).audio.squeeze().float().cpu().numpy()
            entry = {"sigma": sigma, "rms": float(np.sqrt(np.mean(aud**2)))}
            if asr is not None and refs:
                import torchaudio

                t = torch.from_numpy(aud).float()
                t16 = torchaudio.functional.resample(t.unsqueeze(0), SR, 16000).squeeze(0).numpy()
                hyp = asr.transcribe(t16, language="ru", fp16=device.type == "cuda").get("text", "").strip()
                entry["wer"] = compute_wer(refs[:1], [hyp]) if refs else None
            sweep[kind].append(entry)

    wer_ok = dwer is None or dwer <= float(p0.get("delta_wer_pp_max", 4.0))
    go = det_ok and cache_ok and rtf_ok and wer_ok

    report = {
        "go": go,
        "deterministic_sample_false": det_ok,
        "cache_max_abs_diff": max_abs,
        "cache_ok": cache_ok,
        "decoder_rtf": rtf,
        "rtf_ok": rtf_ok,
        "wer_orig": wer_orig,
        "wer_codec": wer_codec,
        "delta_wer_pp": dwer,
        "wer_ok": wer_ok,
        "n_samples": len(samples),
        "latent_layout": codec._latent_layout,
        "sensitivity": sweep,
        "note": "vae_std=0.625 is NOT treated as a guaranteed noise budget; see sensitivity curves.",
    }

    report_path = expand(args.report or p0.get("report_path", "docs/phase0_report.md"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Phase 0 — Codec feasibility report", ""]
    md.append(f"**Decision: {'GO' if go else 'NO-GO'}**")
    md.append("")
    md.append("```json")
    md.append(json.dumps({k: v for k, v in report.items() if k != "sensitivity"}, indent=2))
    md.append("```")
    md.append("")
    md.append("## Decoder sensitivity sweep")
    md.append("")
    md.append("Treat as empirical sensitivity of the decoder — do **not** assume σ=0.625 is safe a priori.")
    md.append("")
    md.append("### White noise")
    md.append("| sigma | rms | wer |")
    md.append("|------:|----:|----:|")
    for e in sweep["white"]:
        md.append(f"| {e['sigma']} | {e['rms']:.4f} | {e.get('wer')} |")
    md.append("")
    md.append("### AR(1) noise")
    md.append("| sigma | rms | wer |")
    md.append("|------:|----:|----:|")
    for e in sweep["ar1"]:
        md.append(f"| {e['sigma']} | {e['rms']:.4f} | {e.get('wer')} |")
    md.append("")
    report_path.write_text("\n".join(md), encoding="utf-8")
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("go", "delta_wer_pp", "decoder_rtf", "cache_ok", "deterministic_sample_false")}, indent=2))
    print(f"Wrote {report_path}")
    return 0 if go else 2


if __name__ == "__main__":
    raise SystemExit(main())
