#!/usr/bin/env python3
"""Wave 1: deterministically encode Track N audio to Qwen3-TTS 12Hz ``[T,16]`` shards.

Does **not** reuse VibeVoice latent / text-hidden caches. Requires a
``qwen-tts``-compatible env (``transformers~=4.57``); prefer a dedicated
venv (see WAVE1_REPORT resume recipe).

Examples (WSL)::

    export PYTHONPATH=/mnt/c/projects/tts-self/src
    export MINISPEECH_ALLOW_DIRTY=1
    export HF_HOME=$HOME/hf_cache_qwen

    # Leak / inventory check only (no GPU encode)
    python scripts/71_encode_codec_tokens.py \\
      --config configs/data_multicorpus.yaml --verify-only

    # S1 (32 utts) + write under subset tag
    python scripts/71_encode_codec_tokens.py \\
      --config configs/data_multicorpus.yaml \\
      --eval-set configs/eval_sets/trackn_train.json \\
      --max-utts 32 --subset-tag s1_32

    # ~1 hour from train (resume-safe)
    python scripts/71_encode_codec_tokens.py \\
      --config configs/data_multicorpus.yaml \\
      --eval-set configs/eval_sets/trackn_train.json \\
      --max-hours 1.0 --subset-tag h1

    # Full Track N train (~72.98 h) — long job; resume by re-running same cmd
    python scripts/71_encode_codec_tokens.py \\
      --config configs/data_multicorpus.yaml \\
      --eval-set configs/eval_sets/trackn_train.json \\
      --subset-tag train_full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf
import torch
from tqdm import tqdm

from minispeech.constants import QWEN3_TTS_TOKENIZER_ID, QWEN3TTS_12HZ, QWEN_MODEL_ID
from minispeech.codec.qwen3_tts_wrapper import Qwen3TTSCodec, resolve_qwen3_tts_tokenizer_id
from minispeech.data.codec_tokens import (
    SPEECH_EOS_ID,
    CodecTokenExample,
    CodecTokenShardWriter,
    assert_pack_train_infer_equal,
    build_ru_text_features,
    tokenize_text_with_offsets,
    verify_trackn_no_leak,
    write_encode_meta,
)
from minispeech.data.dataset import DEFAULT_TRACKN_SET_PATHS, load_utt_ids_from_eval_set
from minispeech.data.manifest import read_manifest
from minispeech.eval.critical_spans import parse_critical_spans
from minispeech.frontend.phone_vocab import PhoneVocab
from minispeech.frontend.stress import accentuate, to_canonical_stress
from minispeech.frontend.verbalize import normalize_for_tts
from minispeech.utils import (
    enforce_clean_git_tree,
    expand,
    git_sha,
    load_yaml,
    set_seed,
    write_run_manifest,
)

# Reuse allowlist helpers from latent encode.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))
from importlib import import_module  # noqa: E402

_encode = import_module("04_encode_latents")
load_utt_id_allowlist = _encode.load_utt_id_allowlist

PLAN_SLUG = "d1_discrete_speech_lm"
DEFAULT_TEXT_TOKENIZER = QWEN_MODEL_ID


def _frontend_text(raw: str, *, apply_frontend: bool) -> str:
    if not apply_frontend:
        return to_canonical_stress(raw or "")
    spoken = normalize_for_tts(raw or "")
    return accentuate(spoken)


def _resolve_text_tokenizer_id(explicit: str | None, qwen_cfg: dict[str, Any] | None) -> str:
    if explicit:
        return explicit
    if qwen_cfg and qwen_cfg.get("model_id"):
        return str(qwen_cfg["model_id"])
    local = expand("~/datasets/tts/models/Qwen3.5-0.8B-Base")
    if local.is_dir():
        return str(local)
    return DEFAULT_TEXT_TOKENIZER


def _load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _corpus_out_dir(dcfg: dict[str, Any], corpus: str, subset_tag: str | None) -> Path:
    corpora = dcfg.get("corpora") or {}
    cfg = corpora.get(corpus) or {}
    base = cfg.get("codec_tokens_dir")
    if base:
        root = expand(base)
    else:
        data_root = expand(dcfg.get("root") or "~/datasets/tts")
        # Prefer per-corpus folders under datasets/tts/<corpus>/codec_tokens
        slug = corpus.replace("/", "_")
        root = data_root / slug / "codec_tokens"
    if subset_tag:
        root = root.parent / f"{root.name}_{subset_tag}"
    return root


def _select_rows(
    all_rows: list,
    allow: set[str] | None,
    *,
    max_utts: int | None,
    max_hours: float | None,
    exclude_tags: Iterable[str],
):
    excl = set(exclude_tags)
    rows = []
    for r in all_rows:
        if allow is not None and r.utt_id not in allow:
            continue
        if excl and (r.qc_tag or "clean") in excl:
            continue
        rows.append(r)
    # Preserve allowlist order when filtering by Track N set.
    if allow is not None:
        order = {uid: i for i, uid in enumerate(allow)}
        rows.sort(key=lambda r: order.get(r.utt_id, 10**12))
    if max_utts is not None:
        rows = rows[: int(max_utts)]
    if max_hours is not None:
        budget = float(max_hours) * 3600.0
        acc, kept = 0.0, []
        for r in rows:
            if acc + float(r.dur_s or 0.0) > budget:
                break
            kept.append(r)
            acc += float(r.dur_s or 0.0)
        rows = kept
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data_multicorpus.yaml")
    ap.add_argument("--qwen-config", default="configs/qwen_local.yaml")
    ap.add_argument("--text-tokenizer", default=None, help="HF id / path for BPE text prefix")
    ap.add_argument("--codec-model-id", default=None, help="Override Qwen3-TTS tokenizer id/path")
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--utt-ids-file", default=None)
    ap.add_argument("--eval-set", action="append", default=None)
    ap.add_argument(
        "--trackn-encode-union",
        action="store_true",
        help="Encode trackn train∪val∪test∪review (usually prefer train-only).",
    )
    ap.add_argument(
        "--subset-tag",
        default=None,
        help="Suffix for output dirs (s1_32 / h1 / train_full). Omit for canonical dirs.",
    )
    ap.add_argument("--shard-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply-frontend", action="store_true")
    ap.add_argument("--max-text-length", type=int, default=256)
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="Speaker-disjoint / no-leak check + inventory; no encode.",
    )
    ap.add_argument(
        "--pack-check-every",
        type=int,
        default=64,
        help="Run train/infer pack equality every N encoded utts (0=disable).",
    )
    ap.add_argument(
        "--meta-out",
        default=None,
        help="encode_meta.json path (default: out/d1_discrete_speech_lm/encode_meta_<tag>.json)",
    )
    args = ap.parse_args()
    set_seed(int(args.seed))
    dirty = enforce_clean_git_tree()

    dcfg = load_yaml(args.config)["data"]
    try:
        qcfg = load_yaml(args.qwen_config).get("qwen") or {}
    except FileNotFoundError:
        qcfg = {}

    trackn_sets = dcfg.get("trackn_sets") or DEFAULT_TRACKN_SET_PATHS
    leak = verify_trackn_no_leak(dcfg["manifest"], trackn_sets)
    print(json.dumps({"trackn_leak_check": leak}, ensure_ascii=False, indent=2), flush=True)
    if not leak["ok"]:
        raise SystemExit(
            "Track N speaker/utt leak detected — refuse encode. "
            f"overlaps={leak['utt_overlaps']} speaker_leaks={leak['speaker_leaks_sample']}"
        )

    if args.verify_only:
        meta = {
            "plan_slug": PLAN_SLUG,
            "verify_only": True,
            "trackn_leak_check": leak,
            "git_sha": git_sha(),
            "git_dirty": dirty,
        }
        out = expand(args.meta_out or f"out/{PLAN_SLUG}/encode_verify.json")
        write_encode_meta(out, meta)
        print(f"wrote {out}", flush=True)
        return 0

    all_rows = read_manifest(dcfg["manifest"])
    allow = load_utt_id_allowlist(
        utt_ids_file=args.utt_ids_file,
        eval_sets=args.eval_set,
        trackn_encode_union=args.trackn_encode_union,
    )
    # Default: Track N train only when no filter given.
    if allow is None and not args.utt_ids_file and not args.eval_set and not args.trackn_encode_union:
        train_path = trackn_sets.get("train") or DEFAULT_TRACKN_SET_PATHS["train"]
        allow = set(load_utt_ids_from_eval_set(train_path))
        print(f"default filter: trackn_train ({len(allow)} ids)", flush=True)

    exclude_tags = tuple(dcfg.get("exclude_tags") or ("exclude", "duplicate"))
    rows = _select_rows(
        all_rows,
        allow,
        max_utts=args.max_utts,
        max_hours=args.max_hours,
        exclude_tags=exclude_tags,
    )
    hours = sum(float(r.dur_s or 0.0) for r in rows) / 3600.0
    print(
        json.dumps(
            {
                "manifest_n": len(all_rows),
                "encode_n": len(rows),
                "encode_hours": hours,
                "subset_tag": args.subset_tag,
                "sample": False,
                "target": QWEN3TTS_12HZ.name,
            },
            indent=2,
        ),
        flush=True,
    )
    if not rows:
        raise SystemExit("no utterances selected")

    text_tok_id = _resolve_text_tokenizer_id(args.text_tokenizer, qcfg)
    tokenizer = _load_tokenizer(text_tok_id)
    phone_vocab = PhoneVocab()

    codec_id = resolve_qwen3_tts_tokenizer_id(args.codec_model_id or QWEN3_TTS_TOKENIZER_ID)
    print(f"loading codec {codec_id} …", flush=True)
    codec = Qwen3TTSCodec(model_id=codec_id)
    fp = codec.weight_fingerprint()
    codec_hash = str(fp.get("partial_sha256_16") or "unavailable")

    # Per-corpus writers (resume-safe).
    shard_size = int(args.shard_size or dcfg.get("shard_size") or 2000)
    writers: dict[str, CodecTokenShardWriter] = {}
    by_corpus_counts: dict[str, int] = defaultdict(int)
    skipped_existing = 0
    pack_checked = 0
    t0 = time.time()

    for r in tqdm(rows, desc="encode_codec_tokens"):
        corpus = str(r.corpus or "ruslan")
        if corpus not in writers:
            out_dir = _corpus_out_dir(dcfg, corpus, args.subset_tag)
            writers[corpus] = CodecTokenShardWriter(
                out_dir, prefix="codec_tokens", shard_size=shard_size, resume=True
            )
        writer = writers[corpus]
        if writer.has(r.utt_id):
            skipped_existing += 1
            continue

        wav_path = expand(r.wav)
        if not wav_path.is_file():
            raise FileNotFoundError(f"missing wav for {r.utt_id}: {wav_path}")
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=-1)
        enc = codec.encode(audio, int(sr), sample=False)
        codes = enc.codes[0].long().cpu()
        n_frames = int(codes.shape[0])

        raw_text = r.text_raw or r.text_stressed or ""
        text = _frontend_text(raw_text if args.apply_frontend else (r.text_stressed or raw_text),
                              apply_frontend=bool(args.apply_frontend))
        if not args.apply_frontend:
            text = to_canonical_stress(r.text_stressed or raw_text)

        text_ids, offsets = tokenize_text_with_offsets(
            tokenizer,
            text,
            max_length=int(args.max_text_length),
            add_special_tokens=True,
        )
        spans = parse_critical_spans(r.extra.get("critical_spans") if r.extra else None)
        feats = build_ru_text_features(
            text, offsets, phone_vocab=phone_vocab, critical_spans=spans
        )
        example = CodecTokenExample(
            utt_id=r.utt_id,
            codes=codes,
            text_ids=text_ids,
            features=feats,
            corpus=corpus,
            speaker_id=r.effective_speaker_id(),
            license=r.license,
            codec_id=QWEN3TTS_12HZ.name,
            codec_hash=codec_hash,
            n_frames=n_frames,
            text_len=int(text_ids.numel()),
            dur_s=float(r.dur_s or (n_frames / codec.frame_rate)),
            text_stressed=text,
        )
        if args.pack_check_every and (pack_checked % int(args.pack_check_every) == 0):
            assert_pack_train_infer_equal(example, speech_eos_id=SPEECH_EOS_ID)
        pack_checked += 1

        writer.add(
            example,
            meta={
                "sample_rate_src": int(sr),
                "text_tokenizer_id": text_tok_id,
                "speech_eos_id": SPEECH_EOS_ID,
                "wav": str(wav_path),
            },
        )
        by_corpus_counts[corpus] += 1

    index_paths = {}
    for corpus, w in writers.items():
        index_paths[corpus] = str(w.close())

    # Persist phone vocab next to multicorpus root or first writer.
    vocab_path = expand(
        (dcfg.get("root") or "~/datasets/tts")
    ) / "multicorpus" / f"phone_vocab_codec_{args.subset_tag or 'default'}.json"
    phone_vocab.save(vocab_path)

    elapsed = time.time() - t0
    meta = {
        "plan_slug": PLAN_SLUG,
        "target": QWEN3TTS_12HZ.name,
        "codec_model_id": codec_id,
        "codec_fingerprint": fp,
        "codec_hash": codec_hash,
        "text_tokenizer_id": text_tok_id,
        "speech_eos_id": SPEECH_EOS_ID,
        "subset_tag": args.subset_tag,
        "encode_n": len(rows),
        "encode_hours": hours,
        "newly_encoded": dict(by_corpus_counts),
        "skipped_existing": skipped_existing,
        "index_paths": index_paths,
        "phone_vocab": str(vocab_path),
        "trackn_leak_check": leak,
        "sample": False,
        "apply_frontend": bool(args.apply_frontend),
        "elapsed_s": elapsed,
        "git_sha": git_sha(),
        "git_dirty": dirty,
        "resume": (
            "Re-run the same command; ShardWriter skips utt_ids already in "
            "codec_tokens_index.json."
        ),
    }
    meta_path = expand(
        args.meta_out
        or f"out/{PLAN_SLUG}/encode_meta_{args.subset_tag or 'default'}.json"
    )
    write_encode_meta(meta_path, meta)
    run_dir = meta_path.parent / f"encode_run_{args.subset_tag or 'default'}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        run_dir,
        {
            "script": "71_encode_codec_tokens.py",
            "meta": str(meta_path),
            "index_paths": index_paths,
            "encode_n": len(rows),
            "encode_hours": hours,
        },
        enforce_clean=False,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote meta {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
