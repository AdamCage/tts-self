#!/usr/bin/env python3
"""Probe Z1: text-tokenizer fertility on Russian (Track N / N02).

Loads AutoTokenizer only (no model weights). Measures tokens/word,
tokens/char, over-split words, U+0301 stress integrity, latin/digit
behaviour, and vocabulary Cyrillic coverage.

Example (WSL)::

    cd ~/projects/tts-self && source .venv/bin/activate
    export HF_HOME=$HOME/hf_cache
    python scripts/40_probe_tokenizers.py \\
      --models Qwen/Qwen3.5-0.8B-Base,ai-sage/GigaChat3.1-10B-A1.8B \\
      --out out/probe_tokenizer_fertility.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

COMBINING_ACUTE = "\u0301"
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
WORD_RE = re.compile(r"\S+")
STRESS_CLUSTER_RE = re.compile(r"([АаЕеЁёИиОоУуЫыЭэЮюЯяA-Za-z])\u0301")

DEFAULT_MODELS = [
    "Qwen/Qwen3.5-0.8B-Base",
    "ai-sage/GigaChat3.1-10B-A1.8B",
]

LATIN_PROBES = ["TTS", "Google", "Apple", "SMS", "Play"]
DIGIT_PROBES = ["1234", "0.5%", "16:30", "23.07.2026", "4082"]


def _disk_report(path: Path) -> dict[str, Any]:
    """Free-space check via shutil + optional ``df -h`` stdout."""
    out: dict[str, Any] = {"path": str(path)}
    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
        out["total_gb"] = round(usage.total / (1024**3), 2)
        out["used_gb"] = round(usage.used / (1024**3), 2)
        out["free_gb"] = round(usage.free / (1024**3), 2)
    except OSError as exc:
        out["error"] = str(exc)
    try:
        proc = subprocess.run(
            ["df", "-h", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        out["df_h"] = (proc.stdout or proc.stderr or "").strip()
    except OSError as exc:
        out["df_h_error"] = str(exc)
    return out


def _load_eval_set_texts(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items") or []
    texts: list[str] = []
    for item in items:
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts


def _load_manifest_sample(path: Path, n: int, seed: int = 0) -> list[str]:
    """Deterministic every-k-th sample of n utterances with stressed text."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return []
    if n >= len(rows):
        chosen = rows
    else:
        step = max(1, len(rows) // n)
        # seed shifts the phase so sampling is reproducible but not always [0]
        offset = seed % step
        chosen = [rows[i] for i in range(offset, len(rows), step)][:n]
    texts: list[str] = []
    for row in chosen:
        text = row.get("text_stressed") or row.get("text_raw") or row.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts


def _token_strings(tokenizer: Any, text: str) -> list[str]:
    """Return surface token strings for ``text`` (no specials when possible)."""
    try:
        pieces = tokenizer.tokenize(text)
        if pieces is not None:
            return [str(p) for p in pieces]
    except Exception:  # noqa: BLE001
        pass
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is not None:
        return [str(t) for t in convert(ids)]
    return [str(i) for i in ids]


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def _word_token_counts(tokenizer: Any, text: str) -> list[int]:
    counts: list[int] = []
    for word in WORD_RE.findall(text):
        counts.append(len(_encode_ids(tokenizer, word)))
    return counts


def _decode_piece(tokenizer: Any, token_id: int) -> str:
    """Decode one token id to a unicode string (best-effort)."""
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    decode = getattr(tokenizer, "decode", None)
    piece = ""
    if convert is not None:
        try:
            piece = str(convert(int(token_id)))
        except Exception:  # noqa: BLE001
            piece = ""
    # Prefer round-trip decode — convert_ids_to_tokens may return byte-display forms
    if decode is not None:
        try:
            decoded = decode([int(token_id)], skip_special_tokens=True)
            if isinstance(decoded, str) and decoded:
                piece = decoded
        except Exception:  # noqa: BLE001
            pass
    return piece


def _word_keeps_stress_cluster(tokenizer: Any, word: str, cluster: str) -> bool:
    """True iff some single token's decoded surface contains ``cluster`` intact."""
    ids = _encode_ids(tokenizer, word)
    for tid in ids:
        surface = _decode_piece(tokenizer, tid)
        if cluster in surface:
            return True
    # Also accept: encoding the cluster alone is exactly one id AND that id
    # appears in the word encoding contiguous with the base letter (no split).
    alone_ids = _encode_ids(tokenizer, cluster)
    if len(alone_ids) == 1 and alone_ids[0] in ids:
        # Still require that acute is not a separate neighbouring token of the word
        acute_ids = _encode_ids(tokenizer, COMBINING_ACUTE)
        if acute_ids and any(a in ids for a in acute_ids):
            return False
        return True
    return False


def _stress_integrity(tokenizer: Any, texts: list[str]) -> dict[str, Any]:
    """Whether letter+U+0301 survives as a single tokenizer piece in words."""
    clusters: list[str] = []
    for text in texts:
        clusters.extend(m.group(0) for m in STRESS_CLUSTER_RE.finditer(text))
    unique_clusters = sorted(set(clusters))

    alone_results: list[dict[str, Any]] = []
    alone_single = 0
    for cluster in unique_clusters:
        toks = _token_strings(tokenizer, cluster)
        ids = _encode_ids(tokenizer, cluster)
        is_single = len(ids) == 1
        if is_single:
            alone_single += 1
        alone_results.append(
            {
                "cluster": cluster,
                "n_tokens": len(ids),
                "tokens": toks,
                "decoded": [_decode_piece(tokenizer, i) for i in ids],
                "single_token": is_single,
            }
        )

    in_word_checked = 0
    in_word_intact = 0
    examples: list[dict[str, Any]] = []
    for text in texts:
        for match in STRESS_CLUSTER_RE.finditer(text):
            cluster = match.group(0)
            start, end = match.span()
            left = text.rfind(" ", 0, start) + 1
            right = text.find(" ", end)
            if right < 0:
                right = len(text)
            word = text[left:right].strip(".,!?;:—–-\"'«»()[]")
            if not word:
                continue
            intact = _word_keeps_stress_cluster(tokenizer, word, cluster)
            in_word_checked += 1
            if intact:
                in_word_intact += 1
            if len(examples) < 12:
                word_ids = _encode_ids(tokenizer, word)
                examples.append(
                    {
                        "word": word,
                        "cluster": cluster,
                        "intact_in_word": intact,
                        "cluster_n_tokens_alone": len(_encode_ids(tokenizer, cluster)),
                        "cluster_tokens_alone": _token_strings(tokenizer, cluster),
                        "word_tokens": _token_strings(tokenizer, word),
                        "word_decoded_pieces": [
                            _decode_piece(tokenizer, i) for i in word_ids
                        ],
                        "word_n_tokens": len(word_ids),
                    }
                )

    n_unique = len(unique_clusters)
    alone_all_single = bool(n_unique > 0 and alone_single == n_unique)
    # Primary criterion: in real words, letter+acute stay inside one token.
    # Alone-encode("а́")==1 is necessary but not sufficient (acute may split off).
    if in_word_checked == 0:
        stress_vowel_single_token = alone_all_single if n_unique else None
    else:
        stress_vowel_single_token = in_word_intact == in_word_checked

    answer = (
        "unknown"
        if stress_vowel_single_token is None
        else ("yes" if stress_vowel_single_token else "no")
    )

    return {
        "n_stress_occurrences": len(clusters),
        "n_unique_stress_clusters": n_unique,
        "alone_single_token_count": alone_single,
        "alone_single_token_share": (
            round(alone_single / n_unique, 6) if n_unique else None
        ),
        "in_word_checked": in_word_checked,
        "in_word_intact_count": in_word_intact,
        "in_word_intact_share": (
            round(in_word_intact / in_word_checked, 6) if in_word_checked else None
        ),
        "stress_vowel_single_token": stress_vowel_single_token,
        "stress_vowel_single_token_answer": answer,
        "alone_cluster_details": alone_results,
        "examples": examples,
    }


def _probe_specials(tokenizer: Any) -> dict[str, Any]:
    latin: dict[str, Any] = {}
    for s in LATIN_PROBES:
        ids = _encode_ids(tokenizer, s)
        latin[s] = {
            "n_tokens": len(ids),
            "tokens": _token_strings(tokenizer, s),
        }
    digits: dict[str, Any] = {}
    for s in DIGIT_PROBES:
        ids = _encode_ids(tokenizer, s)
        digits[s] = {
            "n_tokens": len(ids),
            "tokens": _token_strings(tokenizer, s),
        }
    return {"latin_probes": latin, "digit_probes": digits}


def _vocab_stats(tokenizer: Any) -> dict[str, Any]:
    """Cyrillic share via decode(id): raw BPE piece strings are often latin-1 mojibake."""
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if callable(get_vocab):
            vocab_size = len(get_vocab())

    cyrillic_count = 0
    scanned = 0
    decode = getattr(tokenizer, "decode", None)
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    for i in range(vocab_size):
        surface = ""
        if decode is not None:
            try:
                surface = decode([i], skip_special_tokens=False)
            except Exception:  # noqa: BLE001
                surface = ""
        if not surface and convert is not None:
            try:
                surface = str(convert(i))
            except Exception:  # noqa: BLE001
                surface = ""
        scanned += 1
        if CYRILLIC_RE.search(surface):
            cyrillic_count += 1
    share = round(cyrillic_count / scanned, 6) if scanned else None
    return {
        "vocab_size": vocab_size,
        "vocab_entries_scanned": scanned,
        "cyrillic_token_count": cyrillic_count,
        "cyrillic_token_share": share,
    }


def _set_metrics(tokenizer: Any, texts: list[str]) -> dict[str, Any]:
    if not texts:
        return {
            "n_texts": 0,
            "n_words": 0,
            "n_chars": 0,
            "n_tokens": 0,
            "tokens_per_word": None,
            "tokens_per_char": None,
            "share_words_gt_2_tokens": None,
            "error": "empty text set",
        }

    total_tokens = 0
    total_chars = 0
    word_counts: list[int] = []
    for text in texts:
        # NFC so combining acute is in canonical form
        text = unicodedata.normalize("NFC", text)
        total_chars += len(text)
        total_tokens += len(_encode_ids(tokenizer, text))
        word_counts.extend(_word_token_counts(tokenizer, text))

    n_words = len(word_counts)
    gt2 = sum(1 for c in word_counts if c > 2)
    stress = _stress_integrity(tokenizer, [unicodedata.normalize("NFC", t) for t in texts])
    specials = _probe_specials(tokenizer)

    return {
        "n_texts": len(texts),
        "n_words": n_words,
        "n_chars": total_chars,
        "n_tokens": total_tokens,
        "tokens_per_word": round(total_tokens / n_words, 6) if n_words else None,
        "tokens_per_char": round(total_tokens / total_chars, 6) if total_chars else None,
        "share_words_gt_2_tokens": round(gt2 / n_words, 6) if n_words else None,
        "words_gt_2_tokens_count": gt2,
        "mean_tokens_per_word_isolated": (
            round(sum(word_counts) / n_words, 6) if n_words else None
        ),
        "stress_integrity": stress,
        "stress_vowel_single_token": stress["stress_vowel_single_token"],
        "stress_vowel_single_token_answer": stress["stress_vowel_single_token_answer"],
        **specials,
    }


def _resolve_model_id(model_id: str) -> str:
    """Prefer local Qwen path when the Hub id matches the project default."""
    local = Path("/home/adam/datasets/tts/models/Qwen3.5-0.8B-Base")
    if model_id in {"Qwen/Qwen3.5-0.8B-Base", "Qwen3.5-0.8B-Base"} and local.is_dir():
        if (local / "tokenizer.json").exists() or (local / "tokenizer_config.json").exists():
            return str(local)
    return model_id


def _fetch_tokenizer_files_only(model_id: str) -> Path:
    """Download tokenizer artifacts only (never model weights / full config)."""
    from huggingface_hub import snapshot_download

    safe = model_id.replace("/", "__")
    local = Path.home() / "hf_cache" / "tokenizers_only" / safe
    local.mkdir(parents=True, exist_ok=True)
    path = Path(
        snapshot_download(
            repo_id=model_id,
            allow_patterns=[
                "tokenizer*",
                "vocab*",
                "merges*",
                "special_tokens*",
                "*.model",
            ],
            local_dir=str(local),
        )
    )
    # Some hubs ship config.json next to tokenizer; AutoTokenizer may then
    # validate MoE model fields and fail. Keep tokenizer-only tree clean.
    cfg = path / "config.json"
    if cfg.exists():
        cfg.rename(cfg.with_suffix(".json.aside_from_probe"))
    return path


def _load_tokenizer(model_id: str) -> tuple[Any | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "requested_id": model_id,
        "resolved_id": None,
        "load_ok": False,
        "error": None,
        "tokenizer_only": True,
        "load_mode": None,
    }
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        meta["error"] = f"transformers missing: {exc}"
        return None, meta

    resolved = _resolve_model_id(model_id)
    meta["resolved_id"] = resolved
    errors: list[str] = []

    # 1) Direct load (local path or full repo)
    try:
        tok = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
        meta["load_ok"] = True
        meta["load_mode"] = "from_pretrained"
        meta["class_name"] = type(tok).__name__
        return tok, meta
    except Exception as exc:  # noqa: BLE001
        errors.append(f"from_pretrained({resolved}): {type(exc).__name__}: {exc}")

    # 2) Fallback: same Hub id, tokenizer files only (still this candidate)
    if model_id.startswith("/") or Path(model_id).exists():
        meta["error"] = " | ".join(errors)
        return None, meta
    try:
        tok_dir = _fetch_tokenizer_files_only(model_id)
        tok = AutoTokenizer.from_pretrained(str(tok_dir), trust_remote_code=True)
        meta["load_ok"] = True
        meta["load_mode"] = "tokenizer_files_only_snapshot"
        meta["resolved_id"] = str(tok_dir)
        meta["class_name"] = type(tok).__name__
        meta["direct_load_error"] = errors[0] if errors else None
        return tok, meta
    except Exception as exc:  # noqa: BLE001
        errors.append(
            f"tokenizer_files_only({model_id}): {type(exc).__name__}: {exc}"
        )
        meta["error"] = " | ".join(errors)
        return None, meta


def run_probe(
    *,
    models: list[str],
    bank_path: Path,
    homo_path: Path,
    manifest_path: Path,
    ruslan_n: int,
    out_path: Path,
    home_path: Path,
) -> dict[str, Any]:
    disk = _disk_report(home_path)
    print(json.dumps({"disk_before": disk}, ensure_ascii=False, indent=2), flush=True)

    free_gb = disk.get("free_gb")
    if isinstance(free_gb, (int, float)) and free_gb < 2.0:
        raise SystemExit(
            f"Refusing downloads: only {free_gb} GiB free under {home_path}. "
            "Need more space before fetching tokenizers."
        )

    sets: dict[str, list[str]] = {
        "bank_entity": _load_eval_set_texts(bank_path),
        "homograph": _load_eval_set_texts(homo_path),
        "ruslan_200": _load_manifest_sample(manifest_path, ruslan_n, seed=0),
    }
    set_meta = {name: {"n_texts": len(texts)} for name, texts in sets.items()}

    result: dict[str, Any] = {
        "_probe_meta": {
            "task": "N02",
            "probe": "Z1_tokenizer_fertility",
            "disk_before": disk,
            "sets": set_meta,
            "models_requested": models,
            "skipped": {},
            "note": "AutoTokenizer only; model weights are never loaded.",
        }
    }

    for model_id in models:
        print(f"=== loading tokenizer: {model_id}", flush=True)
        tok, load_meta = _load_tokenizer(model_id)
        if tok is None:
            result["_probe_meta"]["skipped"][model_id] = load_meta
            result[model_id] = {
                "_load": load_meta,
                "skipped": True,
                "reason": load_meta.get("error"),
            }
            print(f"SKIP {model_id}: {load_meta.get('error')}", flush=True)
            continue

        vocab = _vocab_stats(tok)
        model_block: dict[str, Any] = {
            "_load": load_meta,
            "_tokenizer": {
                **vocab,
                # model-level unambiguous stress answer aggregated over all sets
            },
        }
        stress_answers: list[bool] = []
        for set_name, texts in sets.items():
            print(f"  measuring {set_name} ({len(texts)} texts)...", flush=True)
            metrics = _set_metrics(tok, texts)
            model_block[set_name] = metrics
            if metrics.get("stress_vowel_single_token") is not None:
                stress_answers.append(bool(metrics["stress_vowel_single_token"]))

        # Unambiguous model-level answer: yes only if every non-empty stress
        # set reports single-token integrity (homograph/ruslan typically).
        # bank_entity may have zero U+0301 — ignore nulls.
        if stress_answers:
            model_stress = all(stress_answers)
            model_block["_tokenizer"]["stress_vowel_single_token"] = model_stress
            model_block["_tokenizer"]["stress_vowel_single_token_answer"] = (
                "yes" if model_stress else "no"
            )
        else:
            model_block["_tokenizer"]["stress_vowel_single_token"] = None
            model_block["_tokenizer"]["stress_vowel_single_token_answer"] = "unknown"
            model_block["_tokenizer"]["stress_note"] = (
                "no U+0301 clusters found in any set"
            )

        result[model_id] = model_block

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated HF model ids (tokenizer only)",
    )
    p.add_argument(
        "--bank",
        type=Path,
        default=Path("configs/eval_sets/bank_entity.json"),
    )
    p.add_argument(
        "--homograph",
        type=Path,
        default=Path("configs/eval_sets/homograph.json"),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path.home() / "datasets/tts/ruslan/manifest.jsonl",
    )
    p.add_argument("--ruslan-n", type=int, default=200)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("out/probe_tokenizer_fertility.json"),
    )
    p.add_argument(
        "--home",
        type=Path,
        default=Path.home(),
        help="Path for df -h free-space check (default: ~)",
    )
    args = p.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) < 1:
        print("need at least one --models id", file=sys.stderr)
        return 2

    for path in (args.bank, args.homograph, args.manifest):
        if not path.exists():
            print(f"missing input: {path}", file=sys.stderr)
            return 2

    run_probe(
        models=models,
        bank_path=args.bank,
        homo_path=args.homograph,
        manifest_path=args.manifest,
        ruslan_n=args.ruslan_n,
        out_path=args.out,
        home_path=args.home,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
