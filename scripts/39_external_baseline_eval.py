#!/usr/bin/env python3
"""External RU TTS baseline harness for Paper1 T16 (Silero / local only).

Refuses Hub downloads unless ``--allow-download`` is set after a free-disk check.
Does **not** invent MOS/CER — writes ``skipped`` when the backend is absent.

Example (WSL, local Silero)::

    python scripts/39_external_baseline_eval.py \\
      --backend silero --sets demo5,val100 \\
      --out out/external_baseline_silero.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from minispeech.data.manifest import read_manifest
from minispeech.eval.test_sets import load_test_set
from minispeech.utils import expand, load_yaml, write_run_manifest


def _disk_free_gb(path: Path) -> float | None:
    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
        return usage.free / (1024**3)
    except Exception:
        return None


def _synth_silero(texts: list[str], *, speaker: str, sample_rate: int, device: str) -> dict[str, Any]:
    """Attempt local torch.hub Silero; return status + wav tensors or skip reason."""
    try:
        import torch
    except ImportError as exc:
        return {"available": False, "skipped": True, "reason": f"torch missing: {exc}"}

    try:
        model, example = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker=speaker,
            trust_repo=True,
        )
        model.to(device)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "skipped": True,
            "reason": (
                f"Silero load failed ({exc}). Install offline or pass a local repo path. "
                "Do not treat this as a measured baseline."
            ),
        }

    wavs = []
    for t in texts:
        try:
            audio = model.apply_tts(text=t, speaker=speaker, sample_rate=sample_rate)
            wavs.append(audio.detach().cpu())
        except Exception as exc:  # noqa: BLE001
            wavs.append(None)
            _ = exc
    return {
        "available": True,
        "skipped": False,
        "backend": "silero_tts",
        "speaker": speaker,
        "sample_rate": sample_rate,
        "n_ok": sum(1 for w in wavs if w is not None),
        "n": len(wavs),
        "example_keys": list(example.keys()) if isinstance(example, dict) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["silero"], default="silero")
    ap.add_argument("--sets", default="demo5")
    ap.add_argument("--data", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--eval-config", default="configs/eval_mel_suite.yaml")
    ap.add_argument("--speaker", default="xenia")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="out/external_baseline_silero.json")
    ap.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit torch.hub fetch (check disk first; prefer offline cache)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only write skip/status without loading models")
    args = ap.parse_args()

    free = _disk_free_gb(Path.home())
    meta: dict[str, Any] = {
        "backend": args.backend,
        "allow_download": args.allow_download,
        "disk_free_gb": None if free is None else round(free, 2),
        "sets": {},
        "note": (
            "External baseline harness. CER/UTMOS must be filled by piping wavs through "
            "the same ASR/UTMOS path as scripts/32_eval_mel_suite.py — not fabricated."
        ),
    }

    if not args.allow_download and not args.dry_run:
        meta["status"] = "refused_no_allow_download"
        meta["reason"] = (
            "Refusing torch.hub / Hub fetch without --allow-download. "
            "Re-run with --allow-download after confirming disk_free_gb, or point to a local clone."
        )
        out = expand(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return 0

    if args.dry_run:
        meta["status"] = "dry_run"
        out = expand(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return 0

    texts: list[str] = []
    for sid in args.sets.split(","):
        sid = sid.strip()
        if not sid:
            continue
        p = expand(f"configs/eval_sets/{sid}.json")
        if not p.is_file():
            continue
        try:
            ts = load_test_set(p)
            texts.extend([t for t in ts.texts() if t])
        except Exception:
            raw = json.loads(p.read_text(encoding="utf-8"))
            for it in raw.get("items") or []:
                if isinstance(it, dict) and it.get("text"):
                    texts.append(str(it["text"]))

    if not texts:
        dcfg = load_yaml(args.data)["data"]
        rows = list(read_manifest(expand(dcfg["manifest"])))[:5]
        texts = [r.text_stressed or r.text_raw for r in rows]

    if args.backend == "silero":
        result = _synth_silero(
            texts[:20],
            speaker=args.speaker,
            sample_rate=args.sample_rate,
            device=args.device,
        )
        meta["synthesis"] = result
        meta["status"] = "ok" if result.get("available") else "skipped"
        meta["n_texts"] = len(texts[:20])

    out = expand(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_run_manifest(out.parent, {"script": "39_external_baseline_eval", "out": str(out)}, extra={"status": meta.get("status")})
    print(json.dumps({k: meta[k] for k in ("status", "backend", "disk_free_gb", "synthesis") if k in meta}, indent=2))
    print(f"WROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
