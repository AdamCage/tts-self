#!/usr/bin/env python3
"""Formal mel Gate 3: free text-only infer + ASR on val (forced+mel)."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from minispeech.constants import MEL80
from minispeech.data.forced_align import load_forced_json, text_to_ctc_char_labels
from minispeech.data.manifest import filter_split, read_manifest
from minispeech.eval.asr import AsrConfig, AsrEngine
from minispeech.eval.asr_wer import compute_cer, compute_wer
from minispeech.frontend.durations import MeanDurationPredictor, pack_infer_phones_and_durs
from minispeech.frontend.normalize import normalize_for_asr
from minispeech.frontend.phone_vocab import PhoneVocab
from minispeech.infer.vocoder import mel_to_wav, vocoder_status
from minispeech.models.mel_baseline import MelBaseline
from minispeech.utils import expand, load_yaml


def _load_model(ckpt_path: Path, device: torch.device) -> MelBaseline:
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
    return model


def _synth(model, vocab, phones, durs, device):
    ids = vocab.encode(phones)
    phone_ids = torch.tensor([ids], dtype=torch.long, device=device)
    phone_mask = torch.ones_like(phone_ids, dtype=torch.bool)
    durations = torch.tensor([durs], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model.infer(phone_ids, phone_mask=phone_mask, durations=durations)
    mel = out.mel_pred[0, out.frame_mask[0]].cpu()
    result = mel_to_wav(mel, prefer="griffin_lim")
    return result.wav, result.sample_rate, int(mel.shape[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="runs/mel_baseline_v01_s3/best.pt")
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="out/intelligibility_mel_s3_val100.json")
    ap.add_argument("--wav-dir", default="out/mel_s3_val100_wavs")
    ap.add_argument("--gap-mode", default="expand_neighbor")
    ap.add_argument("--max-stats-utts", type=int, default=2000)
    args = ap.parse_args()

    dcfg = load_yaml(args.data).get("data", {})
    forced_path = expand(dcfg["forced_path"])
    manifest = expand(dcfg["manifest"])
    mel_index_path = expand(dcfg["mel_dir"]) / "mels_index.json"
    frame_rate = float(MEL80.frame_rate)

    mel_ids = set(json.loads(mel_index_path.read_text(encoding="utf-8")).keys())
    forced = load_forced_json(forced_path)
    rows_val = filter_split(read_manifest(manifest), "val")
    excl = set(dcfg.get("exclude_tags", ("exclude",)))
    rows_val = [u for u in rows_val if (u.qc_tag or "clean") not in excl]

    candidates = [
        u for u in rows_val if u.utt_id in forced and u.utt_id in mel_ids
    ][: args.n]
    if not candidates:
        raise SystemExit("no val utts with forced+mel")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = expand(args.ckpt)
    model = _load_model(ckpt_path, device)
    vocab = PhoneVocab.load(ckpt_path.parent / "phone_vocab.json")
    pred = MeanDurationPredictor.from_forced_json(
        forced_path, max_utts=args.max_stats_utts
    )

    engine = AsrEngine(AsrConfig.from_yaml(args.asr_config))
    wav_dir = expand(args.wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, u in enumerate(candidates, start=1):
        text = u.text_stressed or u.text_raw
        raw, _ = text_to_ctc_char_labels(text, already_stressed=False)
        phones, durs, meta = pack_infer_phones_and_durs(
            raw,
            pred,
            mode="mean_stats",
            frame_rate=frame_rate,
            sil_policy="none",
            gap_model=None,
            forced_index=None,
            allow_forced_lookup=False,
            auto_target=True,
        )
        wav, sr, n_frames = _synth(model, vocab, phones, durs, device)
        if wav is None:
            print(f"SKIP synth {u.utt_id}", flush=True)
            continue
        wav_path = wav_dir / f"{u.utt_id}_free.wav"
        sf.write(str(wav_path), wav, sr)

        hyp_raw = engine.transcribe(wav, sr=sr)
        ref = normalize_for_asr(text)
        hyp = normalize_for_asr(hyp_raw or "")
        peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
        rms = float(np.sqrt(np.mean(np.square(wav)))) if len(wav) else 0.0
        flag = "ok"
        if peak < 1e-3 or rms < 1e-4:
            flag = "NEAR_SILENT"
        elif peak < 0.05 or rms < 0.005:
            flag = "WEAK"
        cer = float(compute_cer([ref], [hyp]))
        wer = float(compute_wer([ref], [hyp]))
        row = {
            "utt_id": u.utt_id,
            "text": text,
            "wav": str(wav_path),
            "cer": cer,
            "wer": wer,
            "hyp": hyp,
            "hyp_raw": hyp_raw or "",
            "ref_norm": ref,
            "n_frames": n_frames,
            "n_phones": len(phones),
            "n_sil": meta.get("n_sil"),
            "sum_dur_s": meta.get("sum_dur_s"),
            "sanity": {"peak": peak, "rms": rms, "flag": flag, "sr": int(sr)},
        }
        rows.append(row)
        print(
            f"[{i}/{len(candidates)}] {u.utt_id} cer={cer:.3f} wer={wer:.3f} "
            f"flag={flag} hyp={hyp[:60]!r}",
            flush=True,
        )

    cers = [r["cer"] for r in rows]
    wers = [r["wer"] for r in rows]
    n_near = sum(1 for r in rows if r["sanity"]["flag"] == "NEAR_SILENT")
    mean_cer = float(np.mean(cers)) if cers else None
    mean_wer = float(np.mean(wers)) if wers else None
    med_cer = float(statistics.median(cers)) if cers else None
    med_wer = float(statistics.median(wers)) if wers else None

    # Gate 3: val100 CER << 0.15 (protocol / next_intelligibility)
    thr = 0.15
    gate = "PASS" if (mean_cer is not None and mean_cer < thr and n_near == 0) else "FAIL"
    out = {
        "label": "mel_s3_val100_free",
        "gate": "Gate 3",
        "gate_target": "mean_cer < 0.15 on val free text-only (expand_neighbor, no forced-lookup)",
        "gate_verdict": gate,
        "ckpt": str(ckpt_path),
        "duration_mode": "mean_stats",
        "gap_mode": args.gap_mode,
        "sil_policy": "none",
        "forced_lookup": False,
        "forced_path": str(forced_path),
        "vocoder": vocoder_status(None),
        "packing": "expand_neighbor + MeanDurationPredictor; text-only free",
        "split": "val",
        "n_requested": args.n,
        "n_available_forced_mel": sum(
            1 for u in rows_val if u.utt_id in forced and u.utt_id in mel_ids
        ),
        "n": len(rows),
        "mean_cer": mean_cer,
        "median_cer": med_cer,
        "mean_wer": mean_wer,
        "median_wer": med_wer,
        "n_near_silent": n_near,
        "selection": {
            "strategy": "contiguous_book_range_val_first_n_with_forced_and_mel",
            "first_utt": candidates[0].utt_id if candidates else None,
            "last_utt": candidates[-1].utt_id if candidates else None,
        },
        "utterances": rows,
    }
    out_path = expand(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {
        "ckpt": str(ckpt_path),
        "duration_mode": "mean_stats",
        "gap_mode": args.gap_mode,
        "sil_policy": "none",
        "forced_path": str(forced_path),
        "utterances": [
            {"utt_id": r["utt_id"], "text": r["text"], "wav": r["wav"], "n_frames": r["n_frames"]}
            for r in rows
        ],
    }
    (wav_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"n={out['n']} mean_cer={mean_cer} median_cer={med_cer} "
        f"mean_wer={mean_wer} n_near_silent={n_near} verdict={gate} -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
