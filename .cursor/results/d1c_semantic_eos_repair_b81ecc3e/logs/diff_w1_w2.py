#!/usr/bin/env python3
"""Prove W1/W2 configs differ only by eos_loss_weight (+ run_name/run_dir)."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path("/mnt/c/projects/tts-self-d1c")


def flatten(d, prefix=""):
    out = {}
    if not isinstance(d, dict):
        return {prefix: d}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def main() -> int:
    w1 = yaml.safe_load((ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml").read_text())
    w2 = yaml.safe_load((ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml").read_text())
    f1, f2 = flatten(w1), flatten(w2)
    keys = sorted(set(f1) | set(f2))
    diffs = {k: {"w1": f1.get(k), "w2": f2.get(k)} for k in keys if f1.get(k) != f2.get(k)}
    # Effective recipe fields: ignore run naming.
    recipe_diffs = {k: v for k, v in diffs.items() if k not in {"train.run_name", "train.run_dir"}}
    report = {
        "all_diffs": diffs,
        "recipe_effective_diffs": recipe_diffs,
        "pass": list(recipe_diffs.keys()) == ["loss.eos_loss_weight"]
        and recipe_diffs["loss.eos_loss_weight"]["w1"] == 1.0
        and recipe_diffs["loss.eos_loss_weight"]["w2"] == 2.0,
    }
    out = ROOT / ".cursor/results/d1c_semantic_eos_repair_b81ecc3e/artifacts/config_diff_w1_w2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
