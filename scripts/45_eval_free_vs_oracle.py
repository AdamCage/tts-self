#!/usr/bin/env python3
"""Compare free mean_stats(+bigram sil) vs oracle-dur on the same train utts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from minispeech.constants import MEL80
from minispeech.data.forced_align import load_forced_json, text_to_ctc_char_labels
from minispeech.data.frame_projection import phones_and_frame_durs_from_forced
from minispeech.data.manifest import filter_split, read_manifest
from minispeech.eval.asr import AsrConfig, AsrEngine
from minispeech.eval.asr_wer import compute_cer, compute_wer
from minispeech.frontend.durations import MeanDurationPredictor, SilGapModel, pack_infer_phones_and_durs
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


def _score(engine: AsrEngine, wav: np.ndarray, sr: int, text: str) -> dict:
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
    ap.add_argument("--ckpt", default="runs/mel_baseline_v01_s1d/best.pt")
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="out/intelligibility_mel_s1d_free_fixed.json")
    ap.add_argument("--wav-dir", default="out/mel_s1d_free_fixed_train_wavs")
    ap.add_argument("--sil-policy", default="bigram")
    ap.add_argument("--forced-lookup", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    dcfg = load_yaml(args.data).get("data", {})
    forced_path = expand(dcfg["forced_path"])
    manifest = expand(dcfg["manifest"])
    frame_rate = float(MEL80.frame_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = expand(args.ckpt)
    model = _load_model(ckpt_path, device)
    vocab = PhoneVocab.load(ckpt_path.parent / "phone_vocab.json")
    forced = load_forced_json(forced_path)
    pred = MeanDurationPredictor.from_forced_json(forced_path, max_utts=32)
    gap = SilGapModel.from_forced_json(forced_path, frame_rate=frame_rate, max_utts=32)

    rows_m = filter_split(read_manifest(manifest), "train")
    id_to_text = {u.utt_id: (u.text_stressed or u.text_raw) for u in rows_m}
    candidates = [u.utt_id for u in rows_m if u.utt_id in forced][: args.n]

    engine = AsrEngine(AsrConfig.from_yaml(args.asr_config))
    wav_dir = expand(args.wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)

    free_rows = []
    oracle_rows = []
    for uid in candidates:
        text = id_to_text[uid]
        entry = forced[uid]

        o_phones, o_durs, o_meta = phones_and_frame_durs_from_forced(
            entry, frame_rate=frame_rate, fill_gaps=True
        )
        o_wav, o_sr, o_frames = _synth(model, vocab, o_phones, o_durs, device)
        if o_wav is not None:
            o_path = wav_dir / f"{uid}_oracle.wav"
            sf.write(str(o_path), o_wav, o_sr)
            o_score = _score(engine, o_wav, o_sr, text)
            oracle_rows.append({"utt_id": uid, "text": text, "wav": str(o_path),
                                "n_frames": o_frames, "n_sil": o_meta["n_sil"],
                                "n_phones": len(o_phones), **o_score})
            print(f"ORACLE {uid} cer={o_score['cer']:.3f} hyp={o_score['hyp'][:50]!r}", flush=True)

        raw, _ = text_to_ctc_char_labels(text, already_stressed=False)
        f_phones, f_durs, f_meta = pack_infer_phones_and_durs(
            raw,
            pred,
            mode="mean_stats",
            frame_rate=frame_rate,
            sil_policy=args.sil_policy,  # type: ignore[arg-type]
            gap_model=gap,
            forced_index=forced if args.forced_lookup else None,
            allow_forced_lookup=args.forced_lookup,
            auto_target=True,
        )
        f_wav, f_sr, f_frames = _synth(model, vocab, f_phones, f_durs, device)
        if f_wav is not None:
            f_path = wav_dir / f"{uid}_free.wav"
            sf.write(str(f_path), f_wav, f_sr)
            f_score = _score(engine, f_wav, f_sr, text)
            # sil mask overlap vs oracle
            o_content_pos = 0
            f_content_pos = 0
            # Compare n_sil and frames
            free_rows.append({
                "utt_id": uid,
                "text": text,
                "wav": str(f_path),
                "n_frames": f_frames,
                "n_sil": f_meta["n_sil"],
                "n_phones": len(f_phones),
                "oracle_n_sil": o_meta["n_sil"],
                "oracle_n_frames": sum(o_durs),
                "sum_dur_s": f_meta.get("sum_dur_s"),
                **f_score,
            })
            print(
                f"FREE   {uid} cer={f_score['cer']:.3f} frames={f_frames}/{sum(o_durs)} "
                f"sil={f_meta['n_sil']}/{o_meta['n_sil']} hyp={f_score['hyp'][:50]!r}",
                flush=True,
            )
            _ = (o_content_pos, f_content_pos)

    def _mean(rows, key="cer"):
        xs = [r[key] for r in rows if key in r]
        return float(np.mean(xs)) if xs else None

    out = {
        "label": "mel_s1d_free_fixed",
        "ckpt": str(ckpt_path),
        "duration_mode": "mean_stats",
        "sil_policy": args.sil_policy,
        "forced_path": str(forced_path),
        "vocoder": vocoder_status(None),
        "packing": "SilGapModel bigram sil + MeanDurationPredictor; snap frames_per_content",
        "n": len(free_rows),
        "mean_cer": _mean(free_rows, "cer"),
        "mean_cer_free": _mean(free_rows, "cer"),
        "mean_cer_oracle": _mean(oracle_rows, "cer"),
        "mean_wer": _mean(free_rows, "wer"),
        "mean_wer_oracle": _mean(oracle_rows, "wer"),
        "before_mean_cer_demo": 0.8743961352657005,
        "gap_model": {
            "default_p": gap.default_p,
            "default_sil_frames": gap.default_sil_frames,
            "frames_per_content": gap.frames_per_content,
            "threshold": gap.threshold,
            "n_bigrams": len(gap.counts),
        },
        "note": (
            "Fair free vs oracle on train utts (S1d overfit). Demo-phrase CER is OOD. "
            "Retrain not required — infer-only sil/duration packing."
        ),
        "oracle": oracle_rows,
        "free": free_rows,
        "utterances": free_rows,
    }
    out_path = expand(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"FREE_CER={out['mean_cer_free']} ORACLE={out['mean_cer_oracle']} -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
