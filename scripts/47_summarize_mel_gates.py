#!/usr/bin/env python3
"""Summarize T04 baseline vs T07 hifigan suites for Gate V2."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


def _utmos_vals(utts: list[dict]) -> list[float]:
    out: list[float] = []
    for u in utts:
        for m in u.get("optional_metrics") or []:
            if m.get("metric") == "utmos_v2" and m.get("value") is not None and not m.get("skipped"):
                out.append(float(m["value"]))
    return out


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    res: dict = {"path": str(path), "ckpt": data.get("ckpt")}
    for name, block in (data.get("sets") or {}).items():
        utts = block.get("utterances") or []
        u = _utmos_vals(utts)
        res[name] = {
            "n": block.get("n"),
            "mean_cer": block.get("mean_cer"),
            "median_cer": block.get("median_cer"),
            "mean_wer": block.get("mean_wer"),
            "mean_utmos": statistics.fmean(u) if u else None,
            "median_utmos": statistics.median(u) if u else None,
            "n_utmos": len(u),
        }
    return res


def main() -> None:
    root = Path.home() / "projects/tts-self/out"
    base = summarize(root / "eval_mel_suite_s4_baseline_full.json")
    hifi = summarize(root / "eval_mel_suite_hifigan_s4.json")
    d_cer = None
    d_utmos = None
    if base["val100"]["mean_cer"] is not None and hifi["val100"]["mean_cer"] is not None:
        d_cer = float(hifi["val100"]["mean_cer"]) - float(base["val100"]["mean_cer"])
    if base["val100"]["mean_utmos"] is not None and hifi["val100"]["mean_utmos"] is not None:
        d_utmos = float(hifi["val100"]["mean_utmos"]) - float(base["val100"]["mean_utmos"])
    cer_ok = hifi["val100"]["mean_cer"] is not None and float(hifi["val100"]["mean_cer"]) <= 0.06
    utmos_up = d_utmos is not None and d_utmos > 0
    if cer_ok and utmos_up:
        verdict = "GO"
    elif cer_ok and d_utmos is not None:
        verdict = "CER_OK_UTMOS_NOT_UP"
    elif cer_ok:
        verdict = "CER_OK_UTMOS_MISSING"
    else:
        verdict = "FAIL"
    payload = {
        "baseline": base,
        "hifigan": hifi,
        "delta_val100_cer": d_cer,
        "delta_val100_utmos": d_utmos,
        "cer_ok": cer_ok,
        "utmos_up": utmos_up,
        "gate_v2": verdict,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    out = root / "gate_v2_summary.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    win = Path("/mnt/c/projects/tts-self/out/gate_v2_summary.json")
    win.parent.mkdir(parents=True, exist_ok=True)
    win.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
