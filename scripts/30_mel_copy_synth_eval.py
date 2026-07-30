#!/usr/bin/env python3
"""Diagnose mel S1c intelligibility: GT-mel griffin topline + oracle-duration synth.

Writes:
  out/intelligibility_gt_mel_griffin.json
  out/intelligibility_mel_s1c_oracle_dur.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from minispeech.constants import MEL80
from minispeech.data.forced_align import load_forced_json
from minispeech.data.frame_projection import phones_and_frame_durs_from_forced
from minispeech.data.manifest import filter_split, read_manifest
from minispeech.data.mel import MelCacheReader
from minispeech.eval.asr import AsrConfig, AsrEngine
from minispeech.eval.asr_wer import compute_cer, compute_wer
from minispeech.frontend.normalize import normalize_for_asr
from minispeech.frontend.phone_vocab import PhoneVocab
from minispeech.infer.vocoder import mel_to_wav, vocoder_status
from minispeech.models.mel_baseline import MelBaseline
from minispeech.utils import expand, load_yaml


def _wav_sanity(path: Path) -> dict:
    x, sr = sf.read(str(path), dtype="float32")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    dur = len(x) / sr if sr else 0.0
    flag = "ok"
    if peak < 1e-3 or rms < 1e-4:
        flag = "NEAR_SILENT"
    elif peak < 0.05 or rms < 0.005:
        flag = "WEAK"
    return {"path": str(path), "dur_s": dur, "peak": peak, "rms": rms, "flag": flag, "sr": int(sr)}


def _load_model(ckpt_path: Path, device: torch.device) -> tuple[MelBaseline, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") or {}
    mcfg = cfg.get("model", {}).get("mel", {})
    sd = ckpt.get("ema") or ckpt["model"]
    emb_w = sd.get("emb.weight")
    vocab_size = int(emb_w.shape[0]) if emb_w is not None else int(mcfg.get("vocab_size", 512))
    model = MelBaseline(
        vocab_size=vocab_size,
        d_model=int(mcfg.get("d_model", 256)),
        n_mels=int(mcfg.get("n_mels", 80)),
        enc_layers=int(mcfg.get("enc_layers", 3)),
        dec_layers=int(mcfg.get("dec_layers", 4)),
        n_heads=int(mcfg.get("n_heads", 4)),
        ffn_dim=int(mcfg.get("ffn_dim", 1024)),
        dropout=float(mcfg.get("dropout", 0.1)),
        decoder=str(mcfg.get("decoder", "conv")),
        upsample=str(mcfg.get("upsample", "hard")),
        predict_duration=bool(mcfg.get("predict_duration", True)),
    )
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, cfg


def _phones_durs(
    entry: dict,
    frame_rate: float,
    *,
    n_frames: int | None = None,
    fill_gaps: bool = True,
    gap_mode: str = "expand_neighbor",
) -> tuple[list[str], list[int], dict]:
    return phones_and_frame_durs_from_forced(
        entry,
        frame_rate=frame_rate,
        n_frames=n_frames,
        fill_gaps=fill_gaps,
        gap_mode=gap_mode,  # type: ignore[arg-type]
    )


def _asr_score(engine: AsrEngine, wav: np.ndarray, sr: int, text: str) -> dict:
    hyp_raw = engine.transcribe(wav, sr=sr)
    ref = normalize_for_asr(text)
    hyp = normalize_for_asr(hyp_raw or "")
    return {
        "cer": float(compute_cer([ref], [hyp])),
        "wer": float(compute_wer([ref], [hyp])),
        "hyp": hyp,
        "hyp_raw": hyp_raw or "",
        "ref_norm": ref,
    }


def eval_gt_mel(
    *,
    utt_ids: list[str],
    id_to_text: dict[str, str],
    mels: MelCacheReader,
    engine: AsrEngine,
    out_wav_dir: Path,
    n: int,
) -> dict:
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for uid in utt_ids:
        if len(rows) >= n:
            break
        if uid not in mels.index or uid not in id_to_text:
            continue
        mel = mels.get(uid).float()
        result = mel_to_wav(mel, prefer="griffin_lim")
        if result.wav is None:
            rows.append({"utt_id": uid, "error": result.note})
            continue
        wav_path = out_wav_dir / f"{uid}_gt_gl.wav"
        sf.write(str(wav_path), result.wav, result.sample_rate)
        score = _asr_score(engine, result.wav, result.sample_rate, id_to_text[uid])
        rows.append(
            {
                "utt_id": uid,
                "text": id_to_text[uid],
                "wav": str(wav_path),
                "n_mel_frames": int(mel.shape[0]),
                "backend": result.backend,
                "sanity": _wav_sanity(wav_path),
                **score,
            }
        )
        print(f"GT {uid} cer={rows[-1]['cer']:.3f} hyp={rows[-1]['hyp'][:60]!r}", flush=True)
    cers = [r["cer"] for r in rows if "cer" in r]
    wers = [r["wer"] for r in rows if "wer" in r]
    return {
        "label": "gt_mel_griffin",
        "vocoder": vocoder_status(None),
        "n": len(cers),
        "mean_cer": float(np.mean(cers)) if cers else None,
        "mean_wer": float(np.mean(wers)) if wers else None,
        "utterances": rows,
    }


def eval_oracle_dur(
    *,
    utt_ids: list[str],
    id_to_text: dict[str, str],
    forced: dict,
    vocab: PhoneVocab,
    model: MelBaseline,
    device: torch.device,
    engine: AsrEngine,
    out_wav_dir: Path,
    n: int,
    frame_rate: float,
    gap_mode: str = "expand_neighbor",
) -> dict:
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cover_ratios = []
    for uid in utt_ids:
        if len(rows) >= n:
            break
        entry = forced.get(uid)
        if not isinstance(entry, dict) or uid not in id_to_text:
            continue
        phones, durs, dmeta = _phones_durs(entry, frame_rate, gap_mode=gap_mode)
        cover_ratios.append(dmeta["cover_ratio_vs_end"])
        ids = vocab.encode(phones)
        phone_ids = torch.tensor([ids], dtype=torch.long, device=device)
        phone_mask = torch.ones_like(phone_ids, dtype=torch.bool)
        durations = torch.tensor([durs], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model.infer(phone_ids, phone_mask=phone_mask, durations=durations)
        mel = out.mel_pred[0, out.frame_mask[0]].cpu()
        result = mel_to_wav(mel, prefer="griffin_lim")
        if result.wav is None:
            rows.append({"utt_id": uid, "error": result.note, **dmeta})
            continue
        wav_path = out_wav_dir / f"{uid}_oracle_dur_gl.wav"
        sf.write(str(wav_path), result.wav, result.sample_rate)
        score = _asr_score(engine, result.wav, result.sample_rate, id_to_text[uid])
        rows.append(
            {
                "utt_id": uid,
                "text": id_to_text[uid],
                "wav": str(wav_path),
                "n_pred_frames": int(mel.shape[0]),
                "backend": result.backend,
                "sanity": _wav_sanity(wav_path),
                "align_meta": dmeta,
                **score,
            }
        )
        print(
            f"ORACLE {uid} cer={rows[-1]['cer']:.3f} cover={dmeta['cover_ratio_vs_end']:.3f} "
            f"hyp={rows[-1]['hyp'][:60]!r}",
            flush=True,
        )
    cers = [r["cer"] for r in rows if "cer" in r]
    wers = [r["wer"] for r in rows if "wer" in r]
    return {
        "label": "mel_oracle_dur",
        "ckpt": None,
        "vocoder": vocoder_status(None),
        "n": len(cers),
        "mean_cer": float(np.mean(cers)) if cers else None,
        "mean_wer": float(np.mean(wers)) if wers else None,
        "mean_cover_ratio_vs_end": float(np.mean(cover_ratios)) if cover_ratios else None,
        "gap_mode": gap_mode,
        "note": (
            "Oracle phone durations from forced.json via phones_and_frame_durs_from_forced "
            f"(gap_mode={gap_mode}; cover_ratio≈1)."
        ),
        "utterances": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--ckpt", default="runs/mel_baseline_v01_s1c/best.pt")
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--forced", default=None, help="default: data.forced_path")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--n-oracle", type=int, default=5)
    ap.add_argument("--out-gt", default="out/intelligibility_gt_mel_griffin.json")
    ap.add_argument("--out-oracle", default="out/intelligibility_mel_s1c_oracle_dur.json")
    ap.add_argument("--wav-gt", default="out/gt_mel_griffin_wavs")
    ap.add_argument("--wav-oracle", default="out/mel_s1c_oracle_dur_wavs")
    ap.add_argument("--skip-gt", action="store_true")
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument(
        "--gap-mode",
        default="expand_neighbor",
        choices=["expand_neighbor", "sil_fill"],
        help="CTC gap packing for oracle durs (must match train)",
    )
    args = ap.parse_args()

    dcfg = load_yaml(args.data).get("data", {})
    mel_dir = expand(dcfg["mel_dir"])
    manifest = expand(dcfg["manifest"])
    forced_path = expand(args.forced or dcfg.get("forced_path"))
    frame_rate = float(MEL80.frame_rate)

    rows = filter_split(read_manifest(manifest), args.split)
    id_to_text = {u.utt_id: (u.text_stressed or u.text_raw) for u in rows}
    # also allow text from any split for forced-only fallback
    all_rows = read_manifest(manifest)
    for u in all_rows:
        id_to_text.setdefault(u.utt_id, u.text_stressed or u.text_raw)

    mels = MelCacheReader(mel_dir, lazy=True)
    forced = load_forced_json(forced_path)

    candidates = [
        u.utt_id
        for u in rows
        if u.utt_id in mels.index and u.utt_id in forced
    ]
    if not candidates:
        candidates = [uid for uid in forced if uid in mels.index]
        print(f"warn: no {args.split} overlap; using any mel+forced ({len(candidates)})")
    print(f"candidates={len(candidates)} split={args.split} mel={len(mels.index)} forced={len(forced)}")

    asr_cfg = AsrConfig.from_yaml(args.asr_config)
    engine = AsrEngine(asr_cfg)

    if not args.skip_gt:
        gt = eval_gt_mel(
            utt_ids=candidates,
            id_to_text=id_to_text,
            mels=mels,
            engine=engine,
            out_wav_dir=expand(args.wav_gt),
            n=args.n,
        )
        out_gt = expand(args.out_gt)
        out_gt.parent.mkdir(parents=True, exist_ok=True)
        out_gt.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"GT mel griffin mean_cer={gt['mean_cer']} n={gt['n']} -> {out_gt}")

    if not args.skip_oracle:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = expand(args.ckpt)
        model, _cfg = _load_model(ckpt_path, device)
        vocab_path = expand(args.vocab) if args.vocab else ckpt_path.parent / "phone_vocab.json"
        vocab = PhoneVocab.load(vocab_path)
        oracle = eval_oracle_dur(
            utt_ids=candidates,
            id_to_text=id_to_text,
            forced=forced,
            vocab=vocab,
            model=model,
            device=device,
            engine=engine,
            out_wav_dir=expand(args.wav_oracle),
            n=args.n_oracle,
            frame_rate=frame_rate,
            gap_mode=args.gap_mode,
        )
        oracle["ckpt"] = str(ckpt_path)
        oracle["forced_path"] = str(forced_path)
        oracle["vocab"] = str(vocab_path)
        out_o = expand(args.out_oracle)
        out_o.parent.mkdir(parents=True, exist_ok=True)
        out_o.write_text(json.dumps(oracle, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"oracle-dur mean_cer={oracle['mean_cer']} n={oracle['n']} "
            f"cover={oracle['mean_cover_ratio_vs_end']} -> {out_o}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
