#!/usr/bin/env python3
"""E1/E2: latent error → ASR CER calibration + GT-latent copy-synthesis topline.

All perturbations are in *normalized* latent units (via LatentNormalizer),
then denormalized before VibeVoice decode. Fixes the Phase-0 sensitivity unit
bug (raw-space σ with ASR cache returning identical hyps).

Requires WSL GPU + codec + ASR. Use --topline-only for E2 GT latents→decode CER.
GT-mel→vocoder topline is documented as a follow-up stub (Track A).
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from minispeech.codec.stats import LatentNormalizer
from minispeech.codec.wrapper import VibeVoiceCodec
from minispeech.constants import SR
from minispeech.data.manifest import filter_split, read_manifest
from minispeech.data.shards import ShardReader
from minispeech.eval.asr import AsrConfig, AsrEngine
from minispeech.eval.asr_wer import compute_cer, compute_wer
from minispeech.eval.latent_perturb import (
    add_ar1_noise,
    add_white_noise,
    temporal_smooth,
    variance_collapse,
)
from minispeech.frontend.normalize import normalize_for_asr
from minispeech.utils import expand, load_yaml, set_seed


def _load_val_utts(dcfg: dict, n: int, split: str):
    rows = filter_split(read_manifest(dcfg["manifest"]), split)
    excl = set(dcfg.get("exclude_tags", ("exclude",)))
    rows = [r for r in rows if (r.qc_tag or "clean") not in excl]
    rows = rows[:n]
    latents = ShardReader(dcfg["latent_dir"], "latents")
    return rows, latents


def _asr_scores(asr: AsrEngine | None, audio: np.ndarray, ref: str) -> dict:
    if asr is None or not ref:
        return {"cer": None, "wer": None, "hyp": ""}
    hyp = normalize_for_asr(asr.transcribe(audio, SR, use_cache=True))
    ref_n = normalize_for_asr(ref)
    return {
        "cer": compute_cer([ref_n], [hyp]),
        "wer": compute_wer([ref_n], [hyp]),
        "hyp": hyp,
        "ref": ref_n,
    }


def _decode_norm(
    codec: VibeVoiceCodec,
    norm: LatentNormalizer,
    z_norm: torch.Tensor,
) -> np.ndarray:
    z_raw = norm.denormalize(z_norm.unsqueeze(0) if z_norm.ndim == 2 else z_norm)
    aud = codec.decode(z_raw).audio.squeeze().float().cpu().numpy()
    return np.asarray(aud, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--codec-config", default="configs/codec.yaml")
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=50, help="Val utterances for calibration")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument(
        "--topline-only",
        action="store_true",
        help="E2: only GT latents → codec decode CER (no noise sweep)",
    )
    ap.add_argument("--out", default="out/latent_error_calibration.json")
    ap.add_argument("--save-wav-dir", default=None, help="Optional dir for example WAVs")
    args = ap.parse_args()
    set_seed(args.seed)

    dcfg = load_yaml(args.data)["data"]
    norm = LatentNormalizer.from_file(dcfg["stats_path"])
    rows, lat_reader = _load_val_utts(dcfg, args.n, args.split)
    if not rows:
        print("No utterances; abort.")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = VibeVoiceCodec(dtype=torch.bfloat16, device=device)
    asr: AsrEngine | None = None
    if not args.skip_asr:
        try:
            asr = AsrEngine(AsrConfig.from_yaml(args.asr_config), device=device)
            asr._load()
        except Exception as e:
            print(f"ASR unavailable: {e}")
            asr = None

    wav_dir = expand(args.save_wav_dir) if args.save_wav_dir else None
    if wav_dir:
        wav_dir.mkdir(parents=True, exist_ok=True)

    # --- E2 topline: GT latents → decode ---
    topline_cers, topline_wers = [], []
    topline_rows = []
    for u in tqdm(rows, desc="gt_topline"):
        z = lat_reader.get(u.utt_id).float()
        z_n = norm.normalize(z.unsqueeze(0))[0]
        aud = _decode_norm(codec, norm, z_n)
        ref = u.text_stressed or u.text_raw
        scores = _asr_scores(asr, aud, ref)
        if scores["cer"] is not None:
            topline_cers.append(scores["cer"])
            topline_wers.append(scores["wer"])
        topline_rows.append({"utt_id": u.utt_id, **{k: scores[k] for k in ("cer", "wer", "hyp")}})
        if wav_dir is not None:
            sf.write(str(wav_dir / f"topline_{u.utt_id}.wav"), aud, SR)

    report: dict = {
        "n": len(rows),
        "split": args.split,
        "units": "normalized_latent",
        "note": (
            "Perturbations applied after LatentNormalizer.normalize; "
            "decode uses denormalize. Phase-0 raw-space σ sweeps are not comparable."
        ),
        "gt_latent_topline": {
            "mean_cer": float(np.mean(topline_cers)) if topline_cers else None,
            "mean_wer": float(np.mean(topline_wers)) if topline_wers else None,
            "n_scored": len(topline_cers),
            "rows": topline_rows,
        },
        "gt_mel_vocoder_topline": {
            "status": "stub",
            "note": (
                "Follow-up (Track A): GT mel → BigVGAN-v2/Vocos → ASR CER. "
                "Not implemented until mel pipeline exists."
            ),
        },
    }

    if args.topline_only:
        summary = {
            "n": report["n"],
            "units": report["units"],
            "gt_latent_topline": {
                "mean_cer": report["gt_latent_topline"]["mean_cer"],
                "mean_wer": report["gt_latent_topline"]["mean_wer"],
                "n_scored": report["gt_latent_topline"]["n_scored"],
            },
            "gt_mel_vocoder_topline": report["gt_mel_vocoder_topline"],
        }
        print(json.dumps(summary, indent=2))
        out = expand(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
        return 0 if topline_cers or args.skip_asr else 3

    white_sigmas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    alphas = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
    smooth_ks = [1, 3, 5, 9]

    def _curve(kind: str, values: list[float], apply_fn) -> list[dict]:
        out_rows = []
        for v in values:
            cers, wers, rms = [], [], []
            for u in rows:
                z = lat_reader.get(u.utt_id).float()
                z_n = norm.normalize(z.unsqueeze(0))[0]
                z_p = apply_fn(z_n, v)
                aud = _decode_norm(codec, norm, z_p)
                rms.append(float(np.sqrt(np.mean(aud**2))))
                ref = u.text_stressed or u.text_raw
                scores = _asr_scores(asr, aud, ref)
                if scores["cer"] is not None:
                    cers.append(scores["cer"])
                    wers.append(scores["wer"])
            entry = {
                "kind": kind,
                "value": v,
                "mean_cer": float(np.mean(cers)) if cers else None,
                "mean_wer": float(np.mean(wers)) if wers else None,
                "mean_rms": float(np.mean(rms)) if rms else None,
                "n_scored": len(cers),
            }
            out_rows.append(entry)
            print(f"{kind}={v}: CER={entry['mean_cer']} WER={entry['mean_wer']} rms={entry['mean_rms']:.4f}")
        return out_rows

    report["curves"] = {
        "white": _curve(
            "white_sigma",
            white_sigmas,
            lambda z, s: add_white_noise(z, s),
        ),
        "ar1": _curve(
            "ar1_sigma",
            white_sigmas,
            lambda z, s: add_ar1_noise(z, s, rho=0.8),
        ),
        "variance_collapse": _curve(
            "alpha",
            alphas,
            lambda z, a: variance_collapse(z, a),
        ),
        "temporal_smooth": _curve(
            "kernel",
            [float(k) for k in smooth_ks],
            lambda z, k: temporal_smooth(z, int(k)),
        ),
    }

    out = expand(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Compact print
    summary = {
        "n": report["n"],
        "gt_latent_topline_cer": report["gt_latent_topline"]["mean_cer"],
        "curves": {
            k: [{kk: vv for kk, vv in e.items() if kk != "kind"} for e in v]
            for k, v in report["curves"].items()
        },
    }
    print(json.dumps(summary, indent=2))
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if report["gt_latent_topline"]["mean_cer"] is not None or args.skip_asr else 3


if __name__ == "__main__":
    raise SystemExit(main())
