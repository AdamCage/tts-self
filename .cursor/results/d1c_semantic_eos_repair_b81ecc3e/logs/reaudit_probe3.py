"""Independent adversarial probe, part 3: canonical W1/W2 config <-> contract binding."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

RESULTS: list[tuple[str, str, str]] = []


def rec(blocker: str, status: str, detail: str) -> None:
    RESULTS.append((blocker, status, detail))
    print(f"REAUDIT {blocker} {status} {detail}", flush=True)


def main() -> int:
    import yaml

    from minispeech.data.d1b_oracles import load_eval_contract
    from minispeech.train.discrete_lm_trainer import resolve_repro_hashes

    contract_path = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json"
    contract = load_eval_contract(contract_path)
    true_sha = str(contract.get("sha256"))
    true_recipe = str(contract.get("recipe_sha256"))
    true_srcman = str(contract.get("recipe_source_manifest_sha256"))
    print(f"contract sha256              = {true_sha}")
    print(f"contract recipe_sha256       = {true_recipe}")
    print(f"contract recipe_source_man   = {true_srcman}")

    for name in (
        "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml",
        "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml",
    ):
        cfg = yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))
        tcfg = cfg.get("train") or {}
        declared = str(tcfg.get("eval_contract_hash") or "")
        tag = Path(name).stem[-2:]
        rec(
            "B2",
            "PASS" if declared == true_sha else "FAIL",
            f"{tag}: train.eval_contract_hash={declared} vs live contract sha256={true_sha} "
            f"match={declared == true_sha}",
        )
        hashes = resolve_repro_hashes(cfg)
        rec(
            "B7",
            "INFO",
            f"{tag}: resolve_repro_hashes -> "
            + json.dumps({k: (v[:16] + "…" if isinstance(v, str) and len(v) > 20 else v)
                          for k, v in hashes.items()}),
        )
        inert = [k for k in ("recipe", "recipe_source_manifest", "run_manifest") if hashes.get(k) is None]
        rec(
            "B7",
            "PASS" if not inert else "FAIL",
            f"{tag}: resume hash checks that are INERT (None) for the canonical config: {inert}",
        )
        rec(
            "B7",
            "PASS" if hashes.get("eval_contract") == true_sha else "FAIL",
            f"{tag}: stamped eval_contract hash == live contract sha: "
            f"{hashes.get('eval_contract') == true_sha}",
        )
        # subset hash sanity
        want_subset = ((contract.get("subsets") or {}).get("full32") or {}).get("items_sha256")
        rec(
            "B2",
            "PASS" if str(tcfg.get("subset_hash")) == str(want_subset) else "FAIL",
            f"{tag}: train.subset_hash matches contract full32 items_sha256: "
            f"{str(tcfg.get('subset_hash')) == str(want_subset)}",
        )
        # eos_loss_weight is the only effective diff
        rec("B2", "INFO", f"{tag}: loss={cfg.get('loss')}")

    # W1 vs W2 effective diff must be exactly eos_loss_weight
    w1 = yaml.safe_load((ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml").read_text(encoding="utf-8"))
    w2 = yaml.safe_load((ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml").read_text(encoding="utf-8"))

    def flat(d, pre=""):
        out = {}
        for k, v in (d or {}).items():
            key = f"{pre}{k}"
            if isinstance(v, dict):
                out.update(flat(v, key + "."))
            else:
                out[key] = v
        return out

    f1, f2 = flat(w1), flat(w2)
    diff = {k: (f1.get(k), f2.get(k)) for k in set(f1) | set(f2) if f1.get(k) != f2.get(k)}
    allowed = {"loss.eos_loss_weight", "train.run_name", "train.run_dir"}
    extra = {k: v for k, v in diff.items() if k not in allowed}
    rec(
        "B2",
        "PASS" if not extra else "FAIL",
        f"W1 vs W2 diff={sorted(diff)} unexpected={sorted(extra)}",
    )

    print("\n=== REAUDIT PROBE3 SUMMARY ===")
    for b, s, d in RESULTS:
        print(f"{s:5s} {b:4s} {d}")
    fails = [r for r in RESULTS if r[1] in ("FAIL", "ERROR")]
    print(f"\nTOTAL checks={len(RESULTS)} fail_or_error={len(fails)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
