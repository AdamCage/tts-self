#!/usr/bin/env python3
"""Build ``forced.json`` for ``scripts/06_align.py`` (Track A).

Methods
-------
uniform
    Phone-level equal-time spans: TN → stress → G2P → allocate wav duration
    across phones. No model download. Preferred S1 smoke packing (not word-
    level). Graphemefallback is OK when espeak is missing; prefer phonemizer
    + espeak-ng ``ru`` in WSL.
ctc
    Real ``torchaudio.functional.forced_align`` with a local Russian CTC
    acoustic model (typically char-CTC wav2vec2). Refuses Hub download unless
    ``--allow-download`` is set. Prefer ``--model ~/models/...``. Failed
    utterances are skipped (never silent-uniform without labeling).
mfa
    Optional Montreal Forced Aligner cross-check stub. Never requires MFA to be
    installed; prints a status dict and exits unless ``--mfa-write-uniform`` is
    set (then writes phone-level uniform packing tagged ``method=mfa_pending``).

WSL CTC runbook
---------------
Author copy is Windows; GPU/audio work is WSL::

    cd ~/projects/tts-self && source .venv/bin/activate
    export HF_HOME=$HOME/hf_cache
    git pull && git rev-parse --short HEAD

    python scripts/26_build_forced_align.py --discover-ctc

    python scripts/26_build_forced_align.py \\
      --config configs/data_ruslan_local.yaml --method ctc \\
      --model ~/models/wav2vec2-large-xlsr-53-russian \\
      --split train --max-utts 32 \\
      --out ~/datasets/tts/ruslan/align/forced.json

    python scripts/06_align.py --config configs/data_ruslan_local.yaml --level A
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from minispeech.data.forced_align import (
    build_ctc_corpus_entries,
    build_mfa_corpus_entries,
    build_uniform_corpus_entries,
    ctc_forced_align_available,
    discover_local_ctc_models,
    load_forced_json,
    merge_forced,
    mfa_available,
    mfa_crosscheck_stub,
    write_forced_json,
)
from minispeech.data.manifest import read_manifest
from minispeech.frontend.g2p import backend_name
from minispeech.utils import expand, load_yaml


def _disk_free_gb(path: Path) -> float | None:
    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
        return usage.free / (1024**3)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--method", choices=["uniform", "ctc", "mfa"], default="uniform")
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--split", default=None, help="Optional manifest split filter")
    ap.add_argument(
        "--out",
        default=None,
        help="Output forced.json (default: <align_dir>/forced.json)",
    )
    ap.add_argument("--model", default=None, help="Local CTC model dir or Hub id")
    ap.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hub download when --model is a remote id (check disk first)",
    )
    ap.add_argument("--merge", action="store_true", help="Merge into existing forced.json")
    ap.add_argument(
        "--mfa-corpus",
        default=None,
        help="MFA corpus dir (wav+lab); default <align_dir>/mfa_corpus",
    )
    ap.add_argument(
        "--mfa-out",
        default=None,
        help="MFA TextGrid output dir; default <align_dir>/mfa_out",
    )
    ap.add_argument("--mfa-dictionary", default="russian_mfa")
    ap.add_argument("--mfa-acoustic", default="russian_mfa")
    ap.add_argument(
        "--mfa-run",
        action="store_true",
        help="Actually run `mfa align` (requires mfa on PATH)",
    )
    ap.add_argument(
        "--mfa-write-uniform",
        action="store_true",
        help="With --method mfa and no --mfa-run, write uniform packing tagged mfa_pending",
    )
    ap.add_argument(
        "--discover-ctc",
        action="store_true",
        help="Scan ~/datasets, ~/models, HF_HOME for local ForCTC checkpoints and exit",
    )
    ap.add_argument(
        "--proportional",
        action="store_true",
        help="Allocate phone time by label length instead of equal slices (uniform only)",
    )
    ap.add_argument(
        "--no-accentuator",
        action="store_true",
        help="Skip ruaccent when stress marks are missing (passthrough)",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="CTC device (default: cuda if available else cpu)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="With --merge, skip utt_ids already present in the output forced.json",
    )
    args = ap.parse_args()

    if args.discover_ctc:
        found = discover_local_ctc_models()
        print(
            json.dumps(
                {
                    "n": len(found),
                    "candidates": found,
                    "torchaudio_forced_align": ctc_forced_align_available(),
                    "note": (
                        "No Hub download performed. Pass a candidate path as --model "
                        "with --method ctc. If empty, keep --method uniform (phone-level)."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    dcfg = load_yaml(args.config)["data"]
    align_dir = expand(dcfg.get("align_dir", "~/datasets/tts/ruslan/align"))
    out_path = expand(args.out) if args.out else align_dir / "forced.json"

    rows = read_manifest(dcfg["manifest"])
    if args.split:
        rows = [r for r in rows if r.split == args.split]
    excl = set(dcfg.get("exclude_tags", ("exclude",)))
    if excl:
        rows = [r for r in rows if (r.qc_tag or "clean") not in excl]

    if args.skip_existing and out_path.exists():
        existing_ids = set(load_forced_json(out_path).keys())
        rows = [r for r in rows if r.utt_id not in existing_ids]

    meta: dict = {
        "method": args.method,
        "config": args.config,
        "n_manifest": len(rows),
        "out": str(out_path),
        "g2p_backend": backend_name(),
        "granularity": "phone",
        "skip_existing": bool(args.skip_existing),
        "exclude_tags": sorted(excl) if excl else [],
    }

    pack_kw = dict(
        max_utts=args.max_utts,
        use_accentuator=not args.no_accentuator,
        proportional=bool(args.proportional),
    )

    if args.method == "uniform":
        forced = build_uniform_corpus_entries(rows, **pack_kw)
        meta["n_written"] = len(forced)
        n_phones = [int(v.get("n_phones", 0)) for v in forced.values()]
        meta["phone_stats"] = {
            "mean_n_phones": (sum(n_phones) / len(n_phones)) if n_phones else 0.0,
            "min_n_phones": min(n_phones) if n_phones else 0,
            "max_n_phones": max(n_phones) if n_phones else 0,
        }
        meta["note"] = (
            "Phone-level uniform spans (TN→stress→G2P). Replace with --method ctc "
            "on WSL when a RU CTC model is local."
        )
    elif args.method == "mfa":
        mfa_corpus = expand(args.mfa_corpus) if args.mfa_corpus else align_dir / "mfa_corpus"
        mfa_out = expand(args.mfa_out) if args.mfa_out else align_dir / "mfa_out"
        info = mfa_crosscheck_stub(
            corpus_dir=mfa_corpus,
            dictionary=args.mfa_dictionary,
            acoustic_model=args.mfa_acoustic,
        )
        meta["mfa"] = info
        if args.mfa_run or mfa_available():
            # Prefer real MFA when CLI is present or user asked --mfa-run.
            forced, mfa_stats = build_mfa_corpus_entries(
                rows,
                corpus_dir=mfa_corpus,
                output_dir=mfa_out,
                dictionary=args.mfa_dictionary,
                acoustic_model=args.mfa_acoustic,
                max_utts=args.max_utts,
                run_align=bool(args.mfa_run or mfa_available()),
            )
            meta["mfa"].update(mfa_stats)
            meta["n_written"] = len(forced)
            meta["granularity"] = "phone"
            meta["align_unit"] = "phone"
            if not forced:
                print(json.dumps(meta, indent=2, ensure_ascii=False))
                print(
                    "\nMFA produced 0 alignments. Install MFA user-local "
                    "(micromamba -c conda-forge montreal-forced-aligner), "
                    "download russian_mfa models, then re-run with --mfa-run. "
                    "Or use --mfa-write-uniform as a labeled pending fallback."
                )
                if not args.mfa_write_uniform:
                    return 3
                forced = build_uniform_corpus_entries(rows, **pack_kw)
                for v in forced.values():
                    v["method"] = "mfa_pending"
                meta["n_written"] = len(forced)
                meta["mfa"]["status"] = "fallback_uniform_phone"
        elif args.mfa_write_uniform:
            forced = build_uniform_corpus_entries(rows, **pack_kw)
            for v in forced.values():
                v["method"] = "mfa_pending"
            meta["n_written"] = len(forced)
            meta["mfa"]["status"] = "fallback_uniform_phone"
        else:
            print(json.dumps(meta, indent=2, ensure_ascii=False))
            print(
                "\nMFA not on PATH. Install user-local (no sudo), e.g.:\n"
                "  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ~/bin bin/micromamba\n"
                "  micromamba create -y -n mfa -c conda-forge montreal-forced-aligner\n"
                "  micromamba run -n mfa mfa model download acoustic russian_mfa\n"
                "  micromamba run -n mfa mfa model download dictionary russian_mfa\n"
                "Then: python scripts/26_build_forced_align.py --method mfa --mfa-run ..."
            )
            return 0
    else:
        model = args.model
        if not model:
            local = discover_local_ctc_models()
            if local:
                model = local[0]["path"]
                meta["auto_model"] = model
        hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        free = _disk_free_gb(Path(hf_home)) if hf_home else _disk_free_gb(Path.home())
        meta["ctc_preflight"] = {
            "torchaudio_forced_align": ctc_forced_align_available(),
            "model": model,
            "allow_download": args.allow_download,
            "HF_HOME": hf_home,
            "disk_free_gb": None if free is None else round(free, 2),
        }
        if not model:
            print(json.dumps(meta, indent=2, ensure_ascii=False))
            print(
                "\nRefusing CTC run without --model and no local ForCTC checkpoint found. "
                "Use --method uniform for phone-level packing dry-run, or place a RU CTC "
                "checkpoint under ~/models or ~/datasets."
            )
            return 2
        model_path = Path(str(model)).expanduser()
        if not model_path.exists() and not args.allow_download:
            print(json.dumps(meta, indent=2, ensure_ascii=False))
            print(
                "\nModel path missing and --allow-download not set. "
                "Copy a RU CTC checkpoint into WSL or set HF_HOME + --allow-download "
                "after checking disk_free_gb. No forced.json written."
            )
            return 2
        if not ctc_forced_align_available():
            print(json.dumps(meta, indent=2, ensure_ascii=False))
            print("\ntorchaudio.functional.forced_align unavailable; aborting CTC (no silent fallback).")
            return 2

        forced, ctc_stats = build_ctc_corpus_entries(
            rows,
            model_path=model_path if model_path.exists() else model,
            max_utts=args.max_utts,
            device=args.device,
            allow_download=args.allow_download,
        )
        meta["ctc"] = ctc_stats
        meta["n_written"] = len(forced)
        meta["granularity"] = "char"
        meta["align_unit"] = "char"
        if not forced:
            print(json.dumps(meta, indent=2, ensure_ascii=False))
            print("\nCTC produced 0 successful alignments; not writing empty fallback.")
            return 3

    if args.merge and out_path.exists():
        forced = merge_forced(load_forced_json(out_path), forced)

    write_forced_json(out_path, forced)
    meta["bytes"] = out_path.stat().st_size
    labels: set[str] = set()
    methods: dict[str, int] = {}
    for v in forced.values():
        for lab in v.get("phone_labels") or []:
            labels.add(str(lab))
        m = str(v.get("method", "?"))
        methods[m] = methods.get(m, 0) + 1
    meta["n_unique_phone_labels"] = len(labels)
    meta["methods"] = methods
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
