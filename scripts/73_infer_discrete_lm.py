#!/usr/bin/env python3
"""Free-running inference for discrete Speech LM → wav via frozen Qwen3-TTS 12Hz.

Writes wavs under ``out/<plan_slug>/...`` (default plan slug ``d1b_n2_reliability``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import torch

from minispeech.data.d1b_oracles import (
    DEFAULT_EVAL_CONTRACT,
    PLAN_SLUG,
    enforce_eval_contract,
    load_frozen_utt_ids,
)
from minispeech.data.speaker_registry import SpeakerRegistry
from minispeech.infer.discrete_config import resolve_discrete_infer_settings
from minispeech.models.discrete_speech_lm import DiscreteSpeechLM, build_discrete_speech_lm
from minispeech.utils import expand, load_yaml, set_seed


def _resolve_speaker_registry(
    *,
    cfg: dict,
    ckpt_payload: dict | None,
    speaker_map: str | None,
) -> SpeakerRegistry | None:
    """Load speaker map from CLI / ckpt / config when present."""
    candidates: list[str] = []
    if speaker_map:
        candidates.append(speaker_map)
    if ckpt_payload:
        for key in ("speaker_map_path", "speaker_map"):
            val = ckpt_payload.get(key)
            if isinstance(val, str) and val:
                candidates.append(val)
        extra = ckpt_payload.get("speaker_registry") or {}
        if isinstance(extra, dict) and extra.get("path"):
            candidates.append(str(extra["path"]))
    data = (cfg or {}).get("data") or {}
    for key in ("speaker_map", "speaker_registry"):
        val = data.get(key)
        if isinstance(val, str) and val:
            candidates.append(val)
    train = (cfg or {}).get("train") or {}
    for key in ("speaker_map_path", "speaker_map"):
        val = train.get(key)
        if isinstance(val, str) and val:
            candidates.append(val)
    model_spk = ((cfg or {}).get("model") or {}).get("speaker") or {}
    map_path = model_spk.get("map_path") if isinstance(model_spk, dict) else None
    if isinstance(map_path, str) and map_path:
        candidates.append(map_path)
    candidates.append("configs/speaker_maps/trackn_v2.json")
    for path in candidates:
        p = expand(path)
        if p.is_file():
            return SpeakerRegistry.load(p)
    return None


def _load_model(
    ckpt: Path, device: torch.device
) -> tuple[DiscreteSpeechLM, dict, dict]:
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = payload.get("cfg") or {}
    model = build_discrete_speech_lm(cfg)
    sd = payload.get("model") or payload
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, cfg, payload if isinstance(payload, dict) else {}


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(str(path), audio, sr)
    except Exception:
        import wave

        pcm = np.clip(audio, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm_i16.tobytes())


def _ascii_utt(uid: str, i: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", uid)[:48]
    return f"{i:04d}_{safe}"


def _load_codec_items(
    codec_tokens_dir: str,
    max_utts: int | None,
    *,
    speaker_registry: SpeakerRegistry | None,
    utt_ids: list[str] | None = None,
    require_items_sha256: str | None = None,
) -> list[dict]:
    train_path = Path(__file__).resolve().parent / "72_train_discrete_lm.py"
    spec = importlib.util.spec_from_file_location("train_discrete_lm_mod", train_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if speaker_registry is None:
        raise SystemExit(
            "speaker map required for --codec-tokens-dir; pass --speaker-map "
            "or place configs/speaker_maps/trackn_v2.json"
        )
    kwargs: dict = {
        "speaker_registry": speaker_registry,
        "speaker_mode": "infer",
        "allow_unknown_speaker": True,
    }
    if utt_ids is not None:
        kwargs["utt_ids"] = utt_ids
        kwargs["require_items_sha256"] = require_items_sha256
    elif max_utts is not None:
        kwargs["max_utterances"] = int(max_utts)
    ds = mod.CodecTokenShardDataset(codec_tokens_dir, **kwargs)
    return [ds[i] for i in range(len(ds))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/discrete_speech_lm_d1b_443m.yaml")
    ap.add_argument("--text", default=None, help="Single utterance text (tokenized externally)")
    ap.add_argument(
        "--input-ids",
        default=None,
        help="Path to .pt / .json list of token ids (preferred over --text)",
    )
    ap.add_argument(
        "--codec-tokens-dir",
        default=None,
        help="Wave-1 safetensors root(s), comma-separated; batch listen from cached text_ids",
    )
    ap.add_argument(
        "--subset",
        default=None,
        help="Frozen oracle: dev32|confirm32|select8|gate24|full32|JSON path",
    )
    ap.add_argument("--subset-split", default=None)
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--speaker-id", type=int, default=0)
    ap.add_argument(
        "--speaker-map",
        default=None,
        help="Path to speaker_maps JSON (default: ckpt/config/trackn_v2.json)",
    )
    ap.add_argument("--out", default=f"out/{PLAN_SLUG}/listen_demo")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--eval-contract",
        default=DEFAULT_EVAL_CONTRACT,
        help="Frozen eval_contract.json",
    )
    ap.add_argument("--skip-eval-contract", action="store_true")
    ap.add_argument(
        "--decode",
        action="store_true",
        help="Decode codes with frozen Qwen3-TTS tokenizer (needs qwen-tts extra)",
    )
    ap.add_argument("--codes-only", action="store_true", help="Skip wav decode; save codes.pt")
    args = ap.parse_args()

    if "d1_discrete_speech_lm" in str(args.out):
        raise SystemExit(
            f"refusing infer out under d1_discrete_speech_lm for D1b defaults: {args.out!r}"
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(args.config)
    model, ckpt_cfg, ckpt_payload = _load_model(expand(args.ckpt), device)
    if ckpt_cfg:
        cfg = ckpt_cfg
    file_cfg = load_yaml(args.config)
    if isinstance(file_cfg.get("infer"), dict):
        cfg.setdefault("infer", {}).update(file_cfg["infer"])

    speaker_registry = _resolve_speaker_registry(
        cfg=cfg if isinstance(cfg, dict) else {},
        ckpt_payload=ckpt_payload,
        speaker_map=args.speaker_map,
    )

    utt_ids = None
    subset_digest = None
    if args.subset:
        utt_ids, _, subset_digest = load_frozen_utt_ids(
            args.subset, split=args.subset_split, verify_hash=True
        )
    if not args.skip_eval_contract:
        enforce_eval_contract(
            cfg if isinstance(cfg, dict) else {},
            contract_path=args.eval_contract,
            subset_items_sha256=subset_digest,
            subset_label=args.subset,
            require_clean_sha=False,
        )

    codebook_size = int(cfg.get("model", {}).get("codebook_size", 2048))
    gen_kw = resolve_discrete_infer_settings(
        cfg,
        max_frames_override=args.max_frames,
        codebook_size=codebook_size,
    )
    out_dir = expand(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, torch.Tensor, int]] = []
    if args.codec_tokens_dir:
        items = _load_codec_items(
            args.codec_tokens_dir,
            args.max_utts if args.max_utts is not None else 8,
            speaker_registry=speaker_registry,
            utt_ids=utt_ids,
            require_items_sha256=subset_digest,
        )
        for row in items:
            ids = row["input_ids"]
            if not torch.is_tensor(ids):
                ids = torch.tensor(ids, dtype=torch.long)
            jobs.append(
                (
                    str(row.get("utt_id", "utt")),
                    ids.long().view(-1),
                    int(row.get("speaker_id", args.speaker_id)),
                )
            )
    elif args.input_ids:
        p = expand(args.input_ids)
        if p.suffix == ".json":
            ids = torch.tensor(json.loads(p.read_text(encoding="utf-8")), dtype=torch.long)
        else:
            obj = torch.load(p, map_location="cpu", weights_only=False)
            ids = obj["input_ids"] if isinstance(obj, dict) else obj
            ids = ids.long().view(-1)
        jobs.append(("utt0", ids, args.speaker_id))
    elif args.text:
        vocab = int(cfg.get("model", {}).get("vocab_size", 151936))
        ids = torch.tensor(
            [(ord(ch) % (vocab - 2)) + 1 for ch in args.text], dtype=torch.long
        )
        print(
            "WARNING: --text uses a demo char→id map; pass --input-ids or "
            "--codec-tokens-dir from the same packer as training for real eval."
        )
        jobs.append(("utt0", ids, args.speaker_id))
    else:
        raise SystemExit("Pass --codec-tokens-dir, --input-ids, or --text")

    codec = None
    if args.decode and not args.codes_only:
        from minispeech.codec.qwen3_tts_wrapper import Qwen3TTSCodec

        codec = Qwen3TTSCodec(device=device)

    index: list[dict] = []
    for i, (uid, ids, spk) in enumerate(jobs):
        input_ids = ids.unsqueeze(0).to(device)
        speaker_id = torch.tensor([spk], device=device)
        with torch.no_grad():
            codes = model.generate(input_ids, speaker_id=speaker_id, **gen_kw)
        stem = _ascii_utt(uid, i)
        codes_path = out_dir / f"{stem}_codes.pt"
        torch.save({"codes": codes.cpu(), "input_ids": ids.cpu(), "utt_id": uid}, codes_path)
        entry: dict = {
            "utt_id": uid,
            "codes": str(codes_path),
            "n_frames": int(codes.shape[1]),
        }
        if codec is not None:
            dec = codec.decode([codes[0].cpu()])
            wav_path = out_dir / f"{stem}.wav"
            _write_wav(wav_path, np.asarray(dec.audio[0], dtype=np.float32), dec.sample_rate)
            entry["wav"] = str(wav_path)
            print(f"wrote {wav_path} frames={codes.shape[1]}")
        else:
            print(f"wrote {codes_path} shape={tuple(codes.shape)}")
        index.append(entry)

    (out_dir / "infer_index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    if codec is None and not args.codes_only:
        print("Skipping wav decode (pass --decode to run frozen codec).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
