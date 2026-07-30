#!/usr/bin/env python3
"""Infer mel baseline → wav (neural vocoder / griffin-lim / save-mel).

Duration modes (free infer must match train gap packing):
  mean_stats     — content labels + mean frame durs (default; expand_neighbor train)
  oracle_forced  — phones/durs from forced entry (requires --utt-id)
  uniform        — same label packing, equal frames per unit
  pred           — legacy model duration head

Default free path: ``--gap-mode expand_neighbor --sil-policy none`` (no sil).
S1d ablation: ``--gap-mode sil_fill --sil-policy bigram [--forced-lookup]``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from minispeech.constants import MEL80
from minispeech.data.forced_align import load_forced_json, text_to_ctc_char_labels, text_to_phones_for_align
from minispeech.frontend.durations import (
    MeanDurationPredictor,
    SilGapModel,
    pack_infer_phones_and_durs,
)
from minispeech.frontend.phone_vocab import PhoneVocab
from minispeech.infer.vocoder import mel_to_wav, vocoder_status
from minispeech.models.mel_baseline import MelBaseline
from minispeech.models.mel_cfm import MelCFM
from minispeech.utils import expand, load_yaml


def _is_cfm_ckpt(ckpt: dict, sd: dict) -> bool:
    mode = ckpt.get("mode") or (ckpt.get("cfg") or {}).get("mode")
    if mode == "cfm":
        return True
    return any(k.startswith("flow.") or k.startswith("backbone.") for k in sd)


def _load_model(
    ckpt_path: Path, device: torch.device
) -> tuple[MelBaseline | MelCFM, dict, str]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") or {}
    mcfg = cfg.get("model", {}).get("mel", {})
    sd = ckpt.get("ema") or ckpt["model"]
    if _is_cfm_ckpt(ckpt, sd):
        emb_w = sd.get("backbone.emb.weight")
        if emb_w is None:
            emb_w = sd.get("emb.weight")
        vocab_size = int(emb_w.shape[0]) if emb_w is not None else int(mcfg.get("vocab_size", 512))
        model: MelBaseline | MelCFM = MelCFM(
            vocab_size=vocab_size,
            d_model=int(mcfg.get("d_model", 256)),
            n_mels=int(mcfg.get("n_mels", 80)),
            enc_layers=int(mcfg.get("enc_layers", 3)),
            n_heads=int(mcfg.get("n_heads", 4)),
            ffn_dim=int(mcfg.get("ffn_dim", 1024)),
            dropout=float(mcfg.get("dropout", 0.1)),
            upsample=str(mcfg.get("upsample", "hard")),  # type: ignore[arg-type]
            predict_duration=bool(mcfg.get("predict_duration", True)),
            flow_hidden=int(mcfg.get("flow_hidden", 512)),
            flow_layers=int(mcfg.get("flow_layers", 4)),
            time_embed_dim=int(mcfg.get("time_embed_dim", 256)),
            cfg_dropout=0.0,
            time_sampling=str(mcfg.get("time_sampling", "logit_normal")),
        )
        model.load_state_dict(sd, strict=False)
        model.to(device).eval()
        return model, cfg, "cfm"

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
    return model, cfg, "regression"


def _slug(text: str, n: int = 48) -> str:
    s = re.sub(r"\s+", "_", text.strip().lower())
    s = re.sub(r"[^\w\-]+", "", s, flags=re.UNICODE)
    return (s[:n] or "utt").rstrip("_")


def _resolve_forced(args: argparse.Namespace, ckpt_path: Path) -> Path | None:
    if args.forced:
        return expand(args.forced)
    if args.data:
        dcfg = load_yaml(args.data).get("data", {})
        fp = dcfg.get("forced_path")
        if fp:
            return expand(fp)
    # Beside ckpt / common relative
    for cand in (
        ckpt_path.parent / "forced.json",
        Path("~/datasets/tts/ruslan/align/forced.json"),
    ):
        p = expand(cand) if not isinstance(cand, Path) or str(cand).startswith("~") else cand
        p = expand(p) if not p.is_absolute() else p
        if p.is_file():
            return p
    return None


def _raw_labels(text: str, vocab: PhoneVocab) -> tuple[list[str], dict]:
    if getattr(vocab, "unit", "phone") == "char" or "|" in vocab.stoi:
        phones, meta = text_to_ctc_char_labels(text, already_stressed=False)
        meta = {**meta, "g2p_backend": "wav2vec2_char_ctc"}
        return phones, meta
    phones, meta = text_to_phones_for_align(text, already_stressed=False)
    return phones, meta


def build_infer_phones_durs(
    *,
    text: str,
    vocab: PhoneVocab,
    duration_mode: str,
    sil_policy: str,
    forced_path: Path | None,
    utt_id: str | None,
    duration_stats: Path | None,
    max_stats_utts: int | None,
    target_seconds: float | None,
    uniform_frames: int,
    frame_rate: float,
    gap_model: SilGapModel | None = None,
    sil_threshold: float = 0.5,
    forced_lookup: bool = False,
    gap_mode: str = "expand_neighbor",
) -> tuple[list[str], list[int], dict]:
    """Pack labels + frame durations for one utterance."""
    raw, g2p_meta = _raw_labels(text, vocab)
    meta_base = {
        "text": text,
        "raw_labels": list(raw),
        "align_unit": getattr(vocab, "unit", "phone"),
        "g2p_backend": g2p_meta.get("g2p_backend"),
        "text_stressed": g2p_meta.get("text_stressed"),
        "duration_mode": duration_mode,
        "sil_policy": sil_policy,
        "gap_mode": gap_mode,
    }

    forced_index = None
    if forced_path and forced_path.is_file():
        forced_index = load_forced_json(forced_path)

    if duration_mode == "oracle_forced":
        if forced_index is None:
            raise SystemExit("oracle_forced requires --forced (or --data with forced_path)")
        if not utt_id:
            raise SystemExit("oracle_forced requires --utt-id")
        if utt_id not in forced_index:
            raise SystemExit(f"utt_id {utt_id!r} not in {forced_path}")
        phones, durs, pmeta = pack_infer_phones_and_durs(
            raw,
            mode="oracle_forced",
            frame_rate=frame_rate,
            forced_entry=forced_index[utt_id],
            sil_policy=sil_policy,  # ignored for oracle
            gap_mode=gap_mode,  # type: ignore[arg-type]
        )
        return phones, durs, {**meta_base, **pmeta, "utt_id": utt_id, "forced_path": str(forced_path)}

    predictor: MeanDurationPredictor | None = None
    if duration_mode in {"mean_stats", "uniform"}:
        if duration_mode == "mean_stats":
            if duration_stats and expand(duration_stats).is_file():
                predictor = MeanDurationPredictor.load(expand(duration_stats))
                meta_base["duration_stats"] = str(expand(duration_stats))
            elif forced_path and forced_path.is_file():
                predictor = MeanDurationPredictor.from_forced_json(
                    forced_path,
                    max_utts=max_stats_utts,
                    gap_mode=gap_mode,  # type: ignore[arg-type]
                )
                meta_base["forced_path"] = str(forced_path)
                meta_base["max_stats_utts"] = max_stats_utts
            else:
                raise SystemExit(
                    "mean_stats needs --forced / --data forced_path or --duration-stats"
                )
        if (
            gap_model is None
            and sil_policy == "bigram"
            and gap_mode == "sil_fill"
            and forced_path
            and forced_path.is_file()
        ):
            gap_model = SilGapModel.from_forced_json(
                forced_path,
                frame_rate=frame_rate,
                max_utts=max_stats_utts,
                threshold=sil_threshold,
            )
            meta_base["sil_gap_model"] = "from_forced_json"
            meta_base["sil_threshold"] = sil_threshold

    phones, durs, pmeta = pack_infer_phones_and_durs(
        raw,
        predictor,
        mode="uniform" if duration_mode == "uniform" else "mean_stats",
        frame_rate=frame_rate,
        sil_policy=sil_policy,  # type: ignore[arg-type]
        target_seconds=target_seconds,
        uniform_frames_per_unit=uniform_frames,
        gap_model=gap_model,
        gap_mode=gap_mode,  # type: ignore[arg-type]
        forced_index=forced_index if forced_lookup else None,
        allow_forced_lookup=forced_lookup,
        snap=True,
        auto_target=True,
    )
    unk = sum(1 for p in phones if p not in vocab.stoi)
    return phones, durs, {**meta_base, **pmeta, "n_unk_labels": unk}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", default=None, help="Stressed Russian text (single utt)")
    ap.add_argument(
        "--demo",
        action="store_true",
        help="Synthesize the 5 fixed demo phrases into --out",
    )
    ap.add_argument("--vocab", default=None, help="phone_vocab.json (default: beside ckpt)")
    ap.add_argument("--out", default="out/mel_infer")
    ap.add_argument("--config", default="configs/mel_baseline_v01.yaml")
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--vocoder-ckpt", default=None)
    ap.add_argument("--save-mel-only", action="store_true")
    ap.add_argument(
        "--duration-mode",
        default="mean_stats",
        choices=["mean_stats", "oracle_forced", "uniform", "pred"],
        help="mean_stats|oracle_forced|uniform|pred (pred=legacy dur_head)",
    )
    ap.add_argument(
        "--gap-mode",
        default="expand_neighbor",
        choices=["expand_neighbor", "sil_fill"],
        help="Match train packing (default: expand_neighbor, no sil)",
    )
    ap.add_argument(
        "--sil-policy",
        default="none",
        choices=["bigram", "interleave", "word", "none"],
        help="sil insertion for free infer (default: none; use bigram with --gap-mode sil_fill)",
    )
    ap.add_argument("--sil-threshold", type=float, default=0.5, help="P(sil|a→b) threshold for bigram")
    ap.add_argument(
        "--forced-lookup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse forced.json timeline when content labels match (default: off)",
    )
    ap.add_argument("--forced", default=None, help="forced.json for mean_stats / oracle")
    ap.add_argument("--utt-id", default=None, help="Required for oracle_forced")
    ap.add_argument("--duration-stats", default=None, help="Cached MeanDurationPredictor JSON")
    ap.add_argument("--max-stats-utts", type=int, default=None)
    ap.add_argument("--target-seconds", type=float, default=None)
    ap.add_argument("--uniform-frames", type=int, default=4)
    ap.add_argument(
        "--flow-steps",
        type=int,
        default=None,
        help="CFM sampling steps (default: ckpt infer.flow_steps or 16)",
    )
    ap.add_argument(
        "--flow-method",
        default=None,
        choices=["heun", "euler"],
        help="CFM ODE integrator (default: heun)",
    )
    ap.add_argument("--cfg-scale", type=float, default=None, help="CFM CFG scale (default: 1.0)")
    args = ap.parse_args()

    if not args.text and not args.demo:
        raise SystemExit("pass --text or --demo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = expand(args.ckpt)
    model, cfg, model_mode = _load_model(ckpt_path, device)
    icfg = cfg.get("infer") or load_yaml(args.config).get("infer", {})
    frame_rate = float(MEL80.frame_rate)
    flow_steps = int(args.flow_steps or icfg.get("flow_steps") or 16)
    flow_method = str(args.flow_method or icfg.get("flow_method") or "heun")
    cfg_scale = float(args.cfg_scale if args.cfg_scale is not None else icfg.get("cfg_scale", 1.0))

    vocab_path = expand(args.vocab) if args.vocab else ckpt_path.parent / "phone_vocab.json"
    if not vocab_path.is_file():
        raise SystemExit(f"missing vocab {vocab_path}; pass --vocab")
    vocab = PhoneVocab.load(vocab_path)

    forced_path = _resolve_forced(args, ckpt_path)
    out_dir = expand(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_phrases = [
        "привет! Я полностью самостоятельно обученная TTS-модель",
        "как у вас дела? Чем вам помочь?",
        "Сегодня хорошая погода, можно пройтись пешком.",
        "Пожалуйста, повторите ещё раз чуть медленнее.",
        "Спасибо за внимание, до скорой встречи!",
    ]
    texts = demo_phrases if args.demo else [args.text]

    prefer = "save_mel" if args.save_mel_only else None
    all_meta: list[dict] = []

    for i, text in enumerate(texts, start=1):
        if args.duration_mode == "pred":
            # Legacy: char-only / phone-only + model duration head (no sil pack).
            phones, g2p_meta = _raw_labels(text, vocab)
            durs = None
            pack_meta = {
                "text": text,
                "phones": phones,
                "duration_mode": "pred",
                "sil_policy": "none",
                "align_unit": getattr(vocab, "unit", "phone"),
                "g2p_backend": g2p_meta.get("g2p_backend"),
                "note": "legacy dur_head; no sil timeline",
            }
        else:
            phones, durs_list, pack_meta = build_infer_phones_durs(
                text=text,
                vocab=vocab,
                duration_mode=args.duration_mode,
                sil_policy=args.sil_policy,
                forced_path=forced_path,
                utt_id=args.utt_id,
                duration_stats=expand(args.duration_stats) if args.duration_stats else None,
                max_stats_utts=args.max_stats_utts,
                target_seconds=args.target_seconds,
                uniform_frames=args.uniform_frames,
                frame_rate=frame_rate,
                sil_threshold=args.sil_threshold,
                forced_lookup=args.forced_lookup,
                gap_mode=args.gap_mode,
            )
            durs = durs_list

        ids = vocab.encode(phones)
        phone_ids = torch.tensor([ids], dtype=torch.long, device=device)
        phone_mask = torch.ones_like(phone_ids, dtype=torch.bool)
        durations = (
            torch.tensor([durs], dtype=torch.float32, device=device) if durs is not None else None
        )

        with torch.no_grad():
            if model_mode == "cfm":
                out = model.infer(  # type: ignore[call-arg]
                    phone_ids,
                    phone_mask=phone_mask,
                    durations=durations,
                    steps=flow_steps,
                    method=flow_method,
                    cfg_scale=cfg_scale,
                )
            else:
                out = model.infer(phone_ids, phone_mask=phone_mask, durations=durations)
        assert out.mel_pred is not None
        mel = out.mel_pred[0, out.frame_mask[0]].cpu()

        stem = f"{i:02d}_{_slug(text)}" if args.demo else "synth"
        result = mel_to_wav(
            mel,
            vocoder_ckpt=args.vocoder_ckpt or icfg.get("vocoder_ckpt"),
            prefer=prefer,  # type: ignore[arg-type]
            save_mel_path=out_dir / f"{stem}_mel",
            device=device,
        )
        meta = {
            **pack_meta,
            "phones": phones,
            "durs": durs,
            "n_frames": int(mel.shape[0]),
            "model_mode": model_mode,
            "flow_steps": flow_steps if model_mode == "cfm" else None,
            "flow_method": flow_method if model_mode == "cfm" else None,
            "cfg_scale": cfg_scale if model_mode == "cfm" else None,
            "vocoder": vocoder_status(args.vocoder_ckpt or icfg.get("vocoder_ckpt")),
            "backend": result.backend,
            "note": result.note,
            "mel_path": result.mel_path,
            "ckpt": str(ckpt_path),
        }
        if result.wav is not None:
            wav_path = out_dir / f"{stem}.wav"
            try:
                import soundfile as sf

                sf.write(str(wav_path), result.wav, result.sample_rate)
            except ImportError:
                import torchaudio

                torchaudio.save(
                    str(wav_path),
                    torch.from_numpy(result.wav).unsqueeze(0),
                    result.sample_rate,
                )
            meta["wav"] = str(wav_path)
            print(
                f"wrote {wav_path} backend={result.backend} "
                f"mode={args.duration_mode} n_frames={meta['n_frames']} "
                f"n_sil={pack_meta.get('n_sil')}",
                flush=True,
            )
        else:
            print(f"no wav ({result.note}); mel={result.mel_path}", flush=True)

        (out_dir / f"{stem}_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        all_meta.append(meta)

    summary = {
        "ckpt": str(ckpt_path),
        "model_mode": model_mode,
        "flow_steps": flow_steps if model_mode == "cfm" else None,
        "duration_mode": args.duration_mode,
        "gap_mode": args.gap_mode,
        "sil_policy": args.sil_policy,
        "forced_lookup": bool(args.forced_lookup),
        "forced_path": str(forced_path) if forced_path else None,
        "n": len(all_meta),
        "utterances": [
            {
                "text": m.get("text"),
                "wav": m.get("wav"),
                "n_frames": m.get("n_frames"),
                "n_sil": m.get("n_sil"),
                "n_phones": m.get("n_phones"),
                "sum_dur_s": m.get("sum_dur_s"),
            }
            for m in all_meta
        ],
    }
    (out_dir / "meta.json").write_text(
        json.dumps(summary if args.demo else all_meta[0], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
