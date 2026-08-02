#!/usr/bin/env python3
"""Evaluate Silero TTS on the frozen suite with the Track A metric path.

The script synthesizes every requested utterance, saves the waveforms, runs
the same ASR and verbalized-reference normalization as
``scripts/32_eval_mel_suite.py``, and computes CER, WER, DER, EER, and SA.
It never substitutes placeholder numbers when the model is unavailable.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from minispeech.constants import STRESS_CHAR
from minispeech.data.manifest import read_manifest
from minispeech.eval.asr import AsrConfig, AsrEngine
from minispeech.eval.asr_wer import compute_cer, compute_wer
from minispeech.eval.critical_spans import compute_set_critical_metrics
from minispeech.eval.test_sets import (
    EvalTestSet,
    aggregate_cer_wer,
    load_suite_config,
    load_suite_test_sets,
)
from minispeech.frontend.verbalize import normalize_for_tts, normalize_verbalized_for_asr
from minispeech.utils import expand, load_yaml

SILERO_REPO = "snakers4/silero-models"
SILERO_MODEL = "v4_ru"
SILERO_LICENSE = "CC-BY-NC-4.0"
SILERO_SOURCE_URL = "https://github.com/snakers4/silero-models"


def _disk_free_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
        return usage.free / (1024**3)
    except Exception:
        return 0.0


def _slug(value: str) -> str:
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё_-]+", "_", value).strip("_") or "utt"


def _silero_text(text: str) -> str:
    """Translate canonical combining accents to Silero's ``+vowel`` syntax."""
    vowels = "АЕЁИОУЫЭЮЯаеёиоуыэюя"
    converted = re.sub(
        rf"([{vowels}]){STRESS_CHAR}",
        r"+\1",
        text,
    )
    return converted.replace(STRESS_CHAR, "")


def _cached_silero_repo() -> Path | None:
    hub = Path.home() / ".cache" / "torch" / "hub"
    matches = sorted(hub.glob("snakers4_silero-models_*"))
    return next((path for path in matches if (path / "hubconf.py").is_file()), None)


def _load_silero(
    *,
    local_repo: str | None,
    allow_download: bool,
    model_id: str,
    device: str,
) -> tuple[Any | None, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        return None, {"status": "blocked", "reason": f"torch missing: {exc}"}

    repo_path = expand(local_repo) if local_repo else _cached_silero_repo()
    if repo_path is not None and (repo_path / "hubconf.py").is_file():
        repo_or_dir = str(repo_path)
        source = "local"
    elif allow_download:
        repo_or_dir = SILERO_REPO
        source = "github"
    else:
        return None, {
            "status": "blocked",
            "reason": (
                "Silero is not cached locally. Re-run with --allow-download "
                "or provide --local-repo."
            ),
        }
    try:
        loaded = torch.hub.load(
            repo_or_dir=repo_or_dir,
            model="silero_tts",
            language="ru",
            speaker=model_id,
            source=source,
            trust_repo=True,
            force_reload=False,
        )
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        # Silero's packaged TTS module mutates in place and returns ``None``
        # from ``to()``, unlike a regular ``torch.nn.Module``. Some packaged
        # releases also expose no ``eval()`` method.
        model.to(device)
        if hasattr(model, "eval"):
            model.eval()
    except Exception as exc:  # noqa: BLE001
        return None, {
            "status": "blocked",
            "reason": f"Silero load failed: {type(exc).__name__}: {exc}",
        }
    return model, {
        "status": "ok",
        "repo": repo_or_dir,
        "source": source,
        "model_id": model_id,
    }


def _synth_one(
    model: Any,
    text: str,
    *,
    voice: str,
    sample_rate: int,
) -> np.ndarray:
    audio = model.apply_tts(
        text=text,
        speaker=voice,
        sample_rate=sample_rate,
        put_accent=True,
        put_yo=True,
    )
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    wav = np.asarray(audio, dtype=np.float32).squeeze()
    if wav.ndim != 1 or wav.size == 0:
        raise ValueError(f"Silero returned invalid waveform shape {wav.shape}")
    return wav


def _prepare_model_input(text: str, *, input_mode: str) -> str:
    """Build Silero input text for ``raw`` or shared ``frontend`` conditions."""
    if input_mode == "frontend":
        return _silero_text(normalize_for_tts(text))
    if input_mode == "raw":
        return _silero_text(text)
    raise ValueError(f"unsupported input_mode={input_mode!r}; use raw|frontend")


def _synth_with_frontend_fallback(
    model: Any,
    text: str,
    *,
    voice: str,
    sample_rate: int,
    input_mode: str = "raw",
) -> tuple[np.ndarray, str, str | None]:
    """Synthesize with explicit input mode.

    ``frontend`` always applies the shared TN+verbalize path used by Track A.
    ``raw`` feeds the original text (Silero accent markup only); on ValueError
    it may retry once with the frontend path and records that fallback.
    """
    primary_text = _prepare_model_input(text, input_mode=input_mode)
    try:
        return (
            _synth_one(
                model,
                primary_text,
                voice=voice,
                sample_rate=sample_rate,
            ),
            primary_text,
            None,
        )
    except ValueError as exc:
        if input_mode == "frontend":
            raise
        fallback_text = _prepare_model_input(text, input_mode="frontend")
        if fallback_text == primary_text:
            raise
        wav = _synth_one(
            model,
            fallback_text,
            voice=voice,
            sample_rate=sample_rate,
        )
        return wav, fallback_text, f"{type(exc).__name__}: {exc}"


def _fill_utt_texts(ts: EvalTestSet, manifest_path: str) -> EvalTestSet:
    if ts.kind != "utt_ids" or all(item.text for item in ts.items):
        return ts
    rows = {row.utt_id: row for row in read_manifest(expand(manifest_path))}
    items = []
    for item in ts.items:
        row = rows.get(item.utt_id)
        text = item.text
        if not text and row is not None:
            text = row.text_stressed or row.text_raw
        items.append(
            type(item)(
                utt_id=item.utt_id,
                text=text,
                note=item.note,
                meta=item.meta,
            )
        )
    return EvalTestSet(
        id=ts.id,
        kind=ts.kind,
        items=tuple(items),
        description=ts.description,
        path=ts.path,
    )


def _write(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sets",
        default="demo5,val100,bank_entity,homograph,latin",
    )
    ap.add_argument("--suite", default="configs/eval_mel_suite.yaml")
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--model-id", default=SILERO_MODEL)
    ap.add_argument("--voice", default="xenia")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--local-repo", default=None)
    ap.add_argument("--out", default="out/external_baseline_silero_v2.json")
    ap.add_argument("--wav-dir", default="out/external_baseline_silero_v2_wavs")
    ap.add_argument(
        "--input-mode",
        choices=("raw", "frontend"),
        default="raw",
        help=(
            "raw: feed source text (legacy unequal baseline); "
            "frontend: apply shared TN+verbalize before Silero (fair system compare)"
        ),
    )
    ap.add_argument("--max-utts", type=int, default=None, help="Debug cap per set")
    ap.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit a torch.hub download after the disk-space check",
    )
    ap.add_argument(
        "--no-acoustic-sa",
        action="store_true",
        help="Use ASR stress scoring instead of the shared acoustic SA diagnostic",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    free = _disk_free_gb(Path.home())
    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    set_names = [name.strip() for name in args.sets.split(",") if name.strip()]
    meta: dict[str, Any] = {
        "suite": load_suite_config(args.suite).get("suite", {}).get("name"),
        "mode": "external_baseline",
        "backend": "silero_tts",
        "model_id": args.model_id,
        "voice": args.voice,
        "device": device,
        "sample_rate": args.sample_rate,
        "license": SILERO_LICENSE,
        "source_url": SILERO_SOURCE_URL,
        "allow_download": args.allow_download,
        "disk_free_gb_before": round(free, 2),
        "asr_config": args.asr_config,
        "input_mode": args.input_mode,
        "sets": {},
    }

    if args.dry_run:
        meta["status"] = "dry_run"
        _write(meta, expand(args.out))
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return 0

    if args.allow_download and free < 5.0:
        meta["status"] = "blocked"
        meta["reason"] = f"only {free:.2f} GiB free; refusing model download"
        _write(meta, expand(args.out))
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return 2

    model, load_info = _load_silero(
        local_repo=args.local_repo,
        allow_download=args.allow_download,
        model_id=args.model_id,
        device=device,
    )
    meta["model_load"] = load_info
    if model is None:
        meta["status"] = "blocked"
        meta["reason"] = load_info["reason"]
        _write(meta, expand(args.out))
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return 2

    suite_cfg = load_suite_config(args.suite)
    test_sets = load_suite_test_sets(suite_cfg, only=set_names)
    data_cfg = load_yaml(args.data).get("data", {})
    manifest_path = str(data_cfg.get("manifest") or "")
    for set_id in list(test_sets):
        test_sets[set_id] = _fill_utt_texts(test_sets[set_id], manifest_path)

    asr = AsrEngine(AsrConfig.from_yaml(args.asr_config), device=device)
    wav_root = expand(args.wav_dir)
    wav_root.mkdir(parents=True, exist_ok=True)
    n_errors = 0
    for set_id, test_set in test_sets.items():
        items = list(test_set.items)
        if args.max_utts is not None:
            items = items[: args.max_utts]
        set_dir = wav_root / set_id
        set_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            row: dict[str, Any] = {
                "utt_id": item.utt_id,
                "text": item.text,
                "model_input_text": None,
                "cer": None,
                "wer": None,
                "hyp": "",
                "hyp_raw": None,
                "ref_norm": None,
                "wav": None,
            }
            if not item.text:
                row["error"] = "missing text"
                n_errors += 1
                rows.append(row)
                continue
            wav_path = set_dir / f"{index:04d}_{_slug(item.utt_id)}.wav"
            try:
                import soundfile as sf

                wav, model_input_text, primary_input_error = _synth_with_frontend_fallback(
                    model,
                    item.text,
                    voice=args.voice,
                    sample_rate=args.sample_rate,
                    input_mode=args.input_mode,
                )
                sf.write(str(wav_path), wav, args.sample_rate)
                hyp_raw = asr.transcribe(wav, sr=args.sample_rate) or ""
                ref_norm = normalize_verbalized_for_asr(item.text)
                hyp = normalize_verbalized_for_asr(hyp_raw)
                row.update(
                    {
                        "wav": str(wav_path),
                        "model_input_text": model_input_text,
                        "hyp_raw": hyp_raw,
                        "hyp": hyp,
                        "ref_norm": ref_norm,
                        "cer": float(compute_cer([ref_norm], [hyp])),
                        "wer": float(compute_wer([ref_norm], [hyp])),
                    }
                )
                if primary_input_error is not None:
                    row["model_input_fallback"] = "normalize_for_tts"
                    row["primary_input_error"] = primary_input_error
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
                n_errors += 1
            rows.append(row)
            print(
                f"[{set_id}] {item.utt_id} cer={row.get('cer')}",
                flush=True,
            )

        aggregate = aggregate_cer_wer(rows)
        set_block: dict[str, Any] = {
            "description": test_set.description,
            "kind": test_set.kind,
            **aggregate,
            "n_expected": len(items),
            "n_errors": sum("error" in row for row in rows),
            "utterances": rows,
        }
        critical = compute_set_critical_metrics(
            items,
            rows,
            acoustic_sa=not args.no_acoustic_sa,
            acoustic_device=device,
        )
        if critical.get("has_critical_spans") or critical.get("has_stress_target"):
            set_block.update(critical)
        meta["sets"][set_id] = set_block

    meta["status"] = "ok" if n_errors == 0 else "partial"
    meta["n_errors"] = n_errors
    meta["disk_free_gb_after"] = round(_disk_free_gb(Path.home()), 2)
    out_path = expand(args.out)
    _write(meta, out_path)
    print(
        json.dumps(
            {
                "status": meta["status"],
                "backend": meta["backend"],
                "model_id": meta["model_id"],
                "voice": meta["voice"],
                "n_errors": n_errors,
                "sets": {
                    set_id: {
                        key: block.get(key)
                        for key in (
                            "n",
                            "mean_cer",
                            "median_cer",
                            "cer_ci95_low",
                            "cer_ci95_high",
                            "n_cer_gt_1",
                            "der",
                            "eer",
                            "sa",
                        )
                        if key in block
                    }
                    for set_id, block in meta["sets"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"WROTE {out_path}")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
