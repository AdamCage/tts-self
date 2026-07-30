#!/usr/bin/env python3
"""Unified mel eval suite runner (plan c).

Runs free-infer + Whisper CER/WER on frozen test sets (demo5, val100 ids,
homograph, bank_entity). Optional GigaAM / UTMOSv2 / NISQA / MCD-DTW / F0 RMSE
hooks skip gracefully when deps are missing.

Examples (WSL)::

    # Seed results JSON from known S4 production numbers (no GPU)
    python scripts/32_eval_mel_suite.py --seed-production --out out/eval_mel_suite_s4.json

    # Full re-score demo5 + bank + homograph with griffin-lim
    python scripts/32_eval_mel_suite.py \\
      --ckpt runs/mel_baseline_v01_s4/best.pt \\
      --sets demo5,homograph,bank_entity \\
      --out out/eval_mel_suite_s4_live.json

    # Neural vocoder (once ~/models/hifigan/generator.pt exists)
    python scripts/32_eval_mel_suite.py \\
      --ckpt runs/mel_baseline_v01_s4/best.pt --sets demo5 \\
      --prefer-vocoder neural --out out/eval_mel_suite_s4_neural_demo5.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minispeech.constants import MEL80
from minispeech.data.forced_align import text_to_ctc_char_labels, text_to_phones_for_align
from minispeech.data.manifest import read_manifest
from minispeech.eval.asr import AsrConfig, AsrEngine
from minispeech.eval.asr_wer import compute_cer, compute_wer
from minispeech.eval.optional_metrics import run_optional_metrics
from minispeech.eval.test_sets import (
    EvalTestSet,
    aggregate_cer_wer,
    load_suite_config,
    load_suite_test_sets,
)
from minispeech.frontend.durations import MeanDurationPredictor, pack_infer_phones_and_durs
from minispeech.frontend.phone_vocab import PhoneVocab
from minispeech.frontend.verbalize import normalize_verbalized_for_asr
from minispeech.infer.vocoder import mel_to_wav, vocoder_status
from minispeech.models.mel_baseline import MelBaseline
from minispeech.utils import expand, load_yaml


def _slug(text: str, n: int = 48) -> str:
    s = re.sub(r"\s+", "_", text.strip().lower())
    s = re.sub(r"[^\w\-]+", "", s, flags=re.UNICODE)
    return (s[:n] or "utt").rstrip("_")


def _load_mel_model(ckpt_path: Path, device: torch.device) -> MelBaseline:
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
        decoder=str(mcfg.get("decoder", "conv")),  # type: ignore[arg-type]
        upsample=str(mcfg.get("upsample", "hard")),  # type: ignore[arg-type]
        predict_duration=bool(mcfg.get("predict_duration", True)),
    )
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def _raw_labels(text: str, vocab: PhoneVocab) -> tuple[list[str], dict[str, Any]]:
    if getattr(vocab, "unit", "phone") == "char" or "|" in vocab.stoi:
        phones, meta = text_to_ctc_char_labels(text, already_stressed=False)
        return phones, meta
    phones, meta = text_to_phones_for_align(text, already_stressed=False)
    return phones, meta


def _fill_utt_texts(ts: EvalTestSet, data_cfg: str) -> EvalTestSet:
    """Attach manifest text for utt_ids sets that lack text."""
    if ts.kind != "utt_ids" or all(i.text for i in ts.items):
        return ts
    dcfg = load_yaml(data_cfg).get("data", {})
    rows = {u.utt_id: u for u in read_manifest(expand(dcfg["manifest"]))}
    items = []
    for it in ts.items:
        row = rows.get(it.utt_id)
        text = it.text or (row.text_norm or row.text_raw if row else "")
        items.append(type(it)(utt_id=it.utt_id, text=text, note=it.note, meta=it.meta))
    return EvalTestSet(
        id=ts.id,
        kind=ts.kind,
        items=tuple(items),
        description=ts.description,
        path=ts.path,
    )


def _synth_one(
    *,
    model: MelBaseline,
    vocab: PhoneVocab,
    predictor: MeanDurationPredictor,
    text: str,
    device: torch.device,
    gap_mode: str,
    prefer_vocoder: str | None,
    vocoder_ckpt: str | None,
) -> tuple[np.ndarray | None, int, dict[str, Any]]:
    raw, tn_meta = _raw_labels(text, vocab)
    phones, durs, pmeta = pack_infer_phones_and_durs(
        raw,
        predictor,
        mode="mean_stats",
        frame_rate=float(MEL80.frame_rate),
        sil_policy="none",
        gap_mode=gap_mode,  # type: ignore[arg-type]
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
    prefer = prefer_vocoder  # type: ignore[assignment]
    result = mel_to_wav(
        mel,
        vocoder_ckpt=vocoder_ckpt,
        prefer=prefer,  # type: ignore[arg-type]
        device=device,
    )
    meta = {
        **pmeta,
        **tn_meta,
        "n_frames": int(mel.shape[0]),
        "n_phones": len(phones),
        "vocoder_backend": result.backend,
        "vocoder_note": result.note,
    }
    return result.wav, result.sample_rate, meta


def _seed_production(suite_cfg: dict[str, Any], out_path: Path) -> dict[str, Any]:
    prod = suite_cfg.get("production") or {}
    demo_path = expand("out/intelligibility_mel_s4_demo.json")
    val_path = expand("out/intelligibility_mel_s4_val100.json")
    summary_path = expand(prod.get("summary_json") or "out/intelligibility_mel_s4.json")
    demo = json.loads(demo_path.read_text(encoding="utf-8")) if demo_path.is_file() else {}
    val = json.loads(val_path.read_text(encoding="utf-8")) if val_path.is_file() else {}
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    payload = {
        "suite": (suite_cfg.get("suite") or {}).get("name", "mel_eval_suite_v01"),
        "mode": "seed_production",
        "ckpt": prod.get("ckpt") or summary.get("ckpt"),
        "label": prod.get("label", "mel_s4_production"),
        "cfm_parked": bool(prod.get("cfm_parked", True)),
        "vocoder": prod.get("vocoder", "griffin_lim"),
        "production": {
            "free_demo_cer": prod.get("free_demo_cer", summary.get("free_demo_cer")),
            "val100_cer": prod.get("val100_cer", summary.get("val100_cer")),
            "summary_json": str(summary_path),
            "production_reason": summary.get("production_reason"),
        },
        "sets": {
            "demo5": {
                "source": str(demo_path) if demo else None,
                "mean_cer": demo.get("mean_cer", prod.get("free_demo_cer")),
                "mean_wer": demo.get("mean_wer"),
                "n": demo.get("n", 5),
                "skipped_live": True,
            },
            "val100": {
                "source": str(val_path) if val else None,
                "mean_cer": val.get("mean_cer", prod.get("val100_cer")),
                "median_cer": val.get("median_cer"),
                "mean_wer": val.get("mean_wer"),
                "n": val.get("n", 100),
                "gate_verdict": val.get("gate_verdict"),
                "skipped_live": True,
            },
            "homograph": {"n": 8, "skipped_live": True, "note": "frozen phrases; run live to score"},
            "bank_entity": {"n": 8, "skipped_live": True, "note": "frozen phrases; run live to score"},
        },
        "optional_metrics": {
            "status": "not_run",
            "hooks": suite_cfg.get("optional_metrics") or [],
        },
        "how_to_run": {
            "seed": "python scripts/32_eval_mel_suite.py --seed-production --out out/eval_mel_suite_s4.json",
            "live_demo": (
                "python scripts/32_eval_mel_suite.py "
                "--ckpt runs/mel_baseline_v01_s4/best.pt --sets demo5 "
                "--out out/eval_mel_suite_s4_demo5.json"
            ),
            "neural": (
                "python scripts/32_eval_mel_suite.py "
                "--ckpt runs/mel_baseline_v01_s4/best.pt --sets demo5 "
                "--prefer-vocoder neural --out out/eval_mel_suite_s4_neural_demo5.json"
            ),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="configs/eval_mel_suite.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--sets", default="demo5", help="Comma list: demo5,val100,homograph,bank_entity")
    ap.add_argument("--out", default="out/eval_mel_suite.json")
    ap.add_argument("--wav-dir", default=None, help="Write wavs here (default out/<label>_wavs)")
    ap.add_argument("--asr-config", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--gap-mode", default=None)
    ap.add_argument(
        "--prefer-vocoder",
        default=None,
        choices=["neural", "griffin_lim", "save_mel"],
        help="Force vocoder backend (default: auto neural→griffin)",
    )
    ap.add_argument("--vocoder-ckpt", default=None)
    ap.add_argument("--seed-production", action="store_true", help="Write known S4 numbers; no GPU")
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument(
        "--optional",
        default="gigaam_ctc,utmos_v2,nisqa",
        help="Comma list of optional metrics (skipped if missing). Empty to disable.",
    )
    ap.add_argument("--max-utts", type=int, default=None, help="Cap per set (debug)")
    args = ap.parse_args()

    suite_cfg = load_suite_config(args.suite)
    out_path = expand(args.out)

    if args.seed_production:
        payload = _seed_production(suite_cfg, out_path)
        print(json.dumps({"wrote": str(out_path), "mode": "seed_production",
                          "demo_cer": payload["production"]["free_demo_cer"],
                          "val100_cer": payload["production"]["val100_cer"]},
                         ensure_ascii=False))
        return 0

    infer_defs = suite_cfg.get("infer_defaults") or {}
    prod = suite_cfg.get("production") or {}
    ckpt_path = expand(args.ckpt or prod.get("ckpt") or "runs/mel_baseline_v01_s4/best.pt")
    data_cfg = args.data or infer_defs.get("data") or "configs/data_ruslan_local.yaml"
    gap_mode = args.gap_mode or infer_defs.get("gap_mode") or "expand_neighbor"
    asr_cfg_path = args.asr_config or (suite_cfg.get("suite") or {}).get("asr_config") or "configs/asr.yaml"
    set_names = [s.strip() for s in args.sets.split(",") if s.strip()]
    optional_names = [s.strip() for s in (args.optional or "").split(",") if s.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_mel_model(ckpt_path, device)
    vocab = PhoneVocab.load(ckpt_path.parent / "phone_vocab.json")
    dcfg = load_yaml(data_cfg).get("data", {})
    forced_path = expand(dcfg["forced_path"])
    predictor = MeanDurationPredictor.from_forced_json(forced_path, max_utts=2000)

    test_sets = load_suite_test_sets(suite_cfg, only=set_names)
    for k in list(test_sets):
        test_sets[k] = _fill_utt_texts(test_sets[k], data_cfg)

    wav_root = expand(args.wav_dir) if args.wav_dir else expand(f"out/eval_suite_{ckpt_path.parent.name}_wavs")
    wav_root.mkdir(parents=True, exist_ok=True)

    engine: AsrEngine | None = None
    if not args.skip_asr:
        engine = AsrEngine(AsrConfig.from_yaml(asr_cfg_path))

    prefer = args.prefer_vocoder
    results: dict[str, Any] = {
        "suite": (suite_cfg.get("suite") or {}).get("name"),
        "mode": "live",
        "ckpt": str(ckpt_path),
        "device": str(device),
        "gap_mode": gap_mode,
        "prefer_vocoder": prefer,
        "vocoder_status": vocoder_status(args.vocoder_ckpt),
        "sets": {},
    }

    for set_id, ts in test_sets.items():
        set_dir = wav_root / set_id
        set_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        items = list(ts.items)
        if args.max_utts is not None:
            items = items[: args.max_utts]
        for idx, item in enumerate(items):
            if not item.text:
                rows.append({"utt_id": item.utt_id, "error": "missing text", "cer": None, "wer": None})
                continue
            wav, sr, meta = _synth_one(
                model=model,
                vocab=vocab,
                predictor=predictor,
                text=item.text,
                device=device,
                gap_mode=gap_mode,
                prefer_vocoder=prefer,
                vocoder_ckpt=args.vocoder_ckpt,
            )
            wav_path = set_dir / f"{idx+1:02d}_{_slug(item.text)}.wav"
            if wav is not None:
                import soundfile as sf

                sf.write(str(wav_path), wav, sr)
            hyp = ""
            cer = wer = None
            hyp_raw = ""
            if engine is not None and wav is not None:
                hyp_raw = engine.transcribe(wav, sr=sr) or ""
                # Verbalize both sides so Whisper digit hyps ("1234 руб") score
                # fairly against spoken-form bank/date refs ("тысячу … рублей").
                ref = normalize_verbalized_for_asr(item.text)
                hyp = normalize_verbalized_for_asr(hyp_raw)
                cer = float(compute_cer([ref], [hyp]))
                wer = float(compute_wer([ref], [hyp]))
            else:
                ref = hyp = ""
            opt = run_optional_metrics(
                optional_names,
                hyp_wav=wav,
                sr=sr if wav is not None else int(MEL80.sample_rate),
                text=item.text,
            )
            row = {
                "utt_id": item.utt_id,
                "text": item.text,
                "text_tn": meta.get("text_tn"),
                "wav": str(wav_path) if wav is not None else None,
                "cer": cer,
                "wer": wer,
                "hyp": hyp,
                "hyp_raw": hyp_raw or None,
                "ref_norm": ref or None,
                "n_frames": meta.get("n_frames"),
                "vocoder_backend": meta.get("vocoder_backend"),
                "optional_metrics": opt,
            }
            rows.append(row)
            print(
                f"[{set_id}] {item.utt_id} cer={cer} backend={meta.get('vocoder_backend')}",
                flush=True,
            )
        agg = aggregate_cer_wer(rows)
        results["sets"][set_id] = {
            "description": ts.description,
            "kind": ts.kind,
            **agg,
            "utterances": rows,
        }

    # Attach known production reference numbers
    results["production_reference"] = {
        "free_demo_cer": prod.get("free_demo_cer"),
        "val100_cer": prod.get("val100_cer"),
        "ckpt": prod.get("ckpt"),
        "cfm_parked": prod.get("cfm_parked", True),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {out_path}", flush=True)
    for sid, block in results["sets"].items():
        print(f"  {sid}: mean_cer={block.get('mean_cer')} n={block.get('n')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
