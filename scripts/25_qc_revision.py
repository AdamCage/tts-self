#!/usr/bin/env python3
"""E6: recompute corpus QC CER with verbalized/normalized references.

Does **not** re-run full-corpus ASR by default. Modes:
  - ``--from-qc``: re-score existing corpus_qc.jsonl hyps against verbalized refs
    (cheap; no ASR). Estimates how many hours move clean/review/exclude.
  - ``--max-utts N``: optional live ASR re-decode on a sample (WSL + ASR).

CTC backend note: GigaAM-CTC is preferred for deterministic RU ASR without
Whisper hallucinations; wire via ``--backend`` when a CTC engine is available.
Currently ``AsrEngine`` supports openai_whisper / faster_whisper; CTC is a
documented follow-up (see docs/next_intelligibility.md).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from minispeech.data.manifest import read_manifest
from minispeech.eval.asr import AsrConfig, AsrEngine
from minispeech.eval.asr_wer import compute_cer
from minispeech.frontend.normalize import normalize_for_asr
from minispeech.frontend.verbalize import normalize_verbalized_for_asr, verbalize_text
from minispeech.utils import expand, load_yaml


def tag_for_cer(cer: float, clean_max: float, review_max: float) -> str:
    if cer <= clean_max:
        return "clean"
    if cer <= review_max:
        return "review"
    return "exclude"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data_ruslan_local.yaml")
    ap.add_argument("--qc", default=None, help="Existing corpus_qc.jsonl")
    ap.add_argument("--from-qc", action="store_true", help="Rescore hyps vs verbalized refs")
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--backend", default=None, help="ASR backend override (whisper family for now)")
    ap.add_argument("--max-utts", type=int, default=None, help="Live ASR sample size")
    ap.add_argument("--clean-max", type=float, default=0.15)
    ap.add_argument("--review-max", type=float, default=0.25)
    ap.add_argument("--out", default="out/qc_revision.json")
    ap.add_argument("--out-jsonl", default=None, help="Optional rewritten qc jsonl")
    args = ap.parse_args()

    cfg = load_yaml(args.config)["data"]
    rows = {r.utt_id: r for r in read_manifest(cfg["manifest"])}
    qc_path = expand(
        args.qc
        or str(Path(cfg.get("ruslan_dir", "~/datasets/tts/ruslan")).expanduser() / "corpus_qc.jsonl")
    )

    if not args.from_qc and args.max_utts is None:
        print("Specify --from-qc and/or --max-utts. See docs/next_intelligibility.md.")
        return 2

    report: dict = {
        "qc_path": str(qc_path),
        "clean_max": args.clean_max,
        "review_max": args.review_max,
        "ctc_backend_note": (
            "Preferred revision uses a deterministic CTC ASR (e.g. GigaAM-CTC) "
            "to avoid Whisper insertion hallucinations. AsrEngine currently ships "
            "Whisper backends only; add a ctc backend in eval/asr.py when the "
            "model is available in WSL, then re-run with --backend ctc."
        ),
        "modes": {},
    }

    # --- Mode A: rescore existing hyps with verbalized reference ---
    if args.from_qc:
        if not qc_path.is_file():
            print(f"Missing QC file: {qc_path}")
            return 2
        old_tags = Counter()
        new_tags = Counter()
        old_hours = Counter()
        new_hours = Counter()
        improved = 0
        n = 0
        rewritten: list[dict] = []
        with open(qc_path, encoding="utf-8") as f:
            for line in tqdm(f, desc="rescore_qc"):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                uid = obj.get("utt_id")
                hyp = obj.get("hyp") or obj.get("asr_hyp") or obj.get("hyp_norm")
                if not uid or hyp is None:
                    continue
                u = rows.get(uid)
                if u is None:
                    continue
                ref_raw = u.text_raw or ""
                ref_old = normalize_for_asr(ref_raw)
                ref_new = normalize_verbalized_for_asr(ref_raw)
                hyp_n = normalize_for_asr(hyp)
                cer_old = float(obj.get("asr_cer", compute_cer([ref_old], [hyp_n])))
                cer_new = float(compute_cer([ref_new], [hyp_n]))
                tag_old = tag_for_cer(cer_old, args.clean_max, args.review_max)
                tag_new = tag_for_cer(cer_new, args.clean_max, args.review_max)
                dur = float(u.dur_s or obj.get("dur_s") or 0.0)
                old_tags[tag_old] += 1
                new_tags[tag_new] += 1
                old_hours[tag_old] += dur / 3600.0
                new_hours[tag_new] += dur / 3600.0
                if cer_new + 1e-9 < cer_old:
                    improved += 1
                n += 1
                out_obj = dict(obj)
                out_obj.update(
                    {
                        "ref_verbalized": verbalize_text(ref_raw),
                        "ref_norm_verbalized": ref_new,
                        "asr_cer_raw_ref": cer_old,
                        "asr_cer_verbalized_ref": cer_new,
                        "qc_tag_raw_ref": tag_old,
                        "qc_tag_verbalized_ref": tag_new,
                    }
                )
                rewritten.append(out_obj)

        recovered_h = new_hours.get("clean", 0.0) + new_hours.get("review", 0.0)
        excluded_old_h = old_hours.get("exclude", 0.0)
        excluded_new_h = new_hours.get("exclude", 0.0)
        report["modes"]["from_qc"] = {
            "n": n,
            "improved_cer_count": improved,
            "tags_old": dict(old_tags),
            "tags_new": dict(new_tags),
            "hours_old": {k: round(v, 3) for k, v in old_hours.items()},
            "hours_new": {k: round(v, 3) for k, v in new_hours.items()},
            "exclude_hours_delta": round(excluded_new_h - excluded_old_h, 3),
            "usable_hours_new": round(recovered_h, 3),
        }
        if args.out_jsonl:
            op = expand(args.out_jsonl)
            op.parent.mkdir(parents=True, exist_ok=True)
            with open(op, "w", encoding="utf-8") as f:
                for obj in rewritten:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            print(f"Wrote {op}")

    # --- Mode B: live ASR sample ---
    if args.max_utts is not None:
        asr_cfg = AsrConfig.from_yaml(args.asr_config)
        if args.backend:
            asr_cfg.backend = args.backend
        asr = AsrEngine(asr_cfg)
        sample = list(rows.values())[: args.max_utts]
        cers_raw, cers_verb = [], []
        for u in tqdm(sample, desc="live_asr"):
            try:
                audio, sr = sf.read(str(expand(u.wav)), dtype="float32")
            except Exception as e:
                print(f"skip {u.utt_id}: {e}")
                continue
            if audio.ndim > 1:
                audio = audio.mean(-1)
            hyp = asr.transcribe(audio, sr)
            hyp_n = normalize_for_asr(hyp)
            ref_raw = normalize_for_asr(u.text_raw)
            ref_v = normalize_verbalized_for_asr(u.text_raw)
            cers_raw.append(compute_cer([ref_raw], [hyp_n]))
            cers_verb.append(compute_cer([ref_v], [hyp_n]))
        report["modes"]["live_asr"] = {
            "n": len(cers_raw),
            "backend": asr_cfg.backend,
            "mean_cer_raw_ref": float(np.mean(cers_raw)) if cers_raw else None,
            "mean_cer_verbalized_ref": float(np.mean(cers_verb)) if cers_verb else None,
        }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = expand(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
