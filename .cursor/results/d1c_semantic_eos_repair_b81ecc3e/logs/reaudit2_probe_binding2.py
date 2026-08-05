"""REAUDIT ROUND2 probe A2 — corrected binding + resume probes.

Fixes probe-A artifacts: contract content sha excludes {sha256, recipe_sha256, _*};
checkpoint step_unit literal is "optimizer".
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path("/mnt/c/projects/tts-self-d1c")
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from minispeech.data import d1b_oracles as O  # noqa: E402
from minispeech.train import discrete_lm_trainer as T  # noqa: E402

W1 = ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml"
W2 = ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml"
C200 = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json"

out: list[tuple[str, str, str]] = []


def rec(tag, ok, msg):
    out.append((tag, "PASS" if ok else "FAIL", msg))
    print(f"B{tag} {'PASS' if ok else 'FAIL'} {msg}", flush=True)


def info(tag, msg):
    out.append((tag, "INFO", msg))
    print(f"B{tag} INFO {msg}", flush=True)


def canon(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


raw = json.loads(C200.read_text(encoding="utf-8"))
declared = str(raw["sha256"])

# ---- B1 independent content-sha recompute (matching documented exclusion rule)
body = {k: v for k, v in raw.items() if k not in {"sha256", "recipe_sha256"} and not str(k).startswith("_")}
indep = hashlib.sha256(canon(body)).hexdigest()
rec("1a", indep == declared, f"independent content recompute={indep} == declared sha256 ({indep == declared})")
info("1b", f"contract top keys: {sorted(raw.keys())}")
info("1c", f"repro block: {json.dumps(raw.get('repro'), sort_keys=True)}")
info("1d", f"termination: {json.dumps(raw.get('termination'), sort_keys=True)}")
info("1e", f"subsets: {sorted((raw.get('subsets') or {}).keys())}")

# recipe_sha256 is EXCLUDED from the content hash -> not cryptographically bound
probe = dict(raw)
probe["recipe_sha256"] = "f" * 64
tampered = hashlib.sha256(
    canon({k: v for k, v in probe.items() if k not in {"sha256", "recipe_sha256"} and not str(k).startswith("_")})
).hexdigest()
info(
    "1f",
    f"contract.recipe_sha256 is EXCLUDED from sha256 -> tampering it keeps sha valid (same={tampered == declared})",
)

# tamper a bound field -> sha must change
probe2 = copy.deepcopy(raw)
probe2.setdefault("infer", {})
if isinstance(probe2.get("infer"), dict):
    probe2["infer"]["max_frames"] = 999
t2 = hashlib.sha256(
    canon({k: v for k, v in probe2.items() if k not in {"sha256", "recipe_sha256"} and not str(k).startswith("_")})
).hexdigest()
rec("1g", t2 != declared, f"tamper(infer.max_frames=999) changes content sha ({t2 != declared})")

# ---- B2 enforce_eval_contract behaviour (the path scripts/72 uses)
w1cfg = yaml.safe_load(W1.read_text(encoding="utf-8"))
stale = "a59ada9c1e0a64f067b27dd2b4d93ca270702bd41881746d1bb9a437d2f33772"
subset_hash = str(w1cfg["train"]["subset_hash"])

try:
    c = O.enforce_eval_contract(
        w1cfg,
        contract_path=str(C200),
        subset_items_sha256=subset_hash,
        subset_label="configs/train_subsets/d1c_dev32.json",
        require_clean_sha=False,
        expected_contract_sha256=declared,
    )
    rec("2a", True, f"enforce_eval_contract accepts correct declared hash (sha={str(c.get('sha256'))[:16]})")
except Exception as e:  # noqa: BLE001
    rec("2a", False, f"enforce_eval_contract REJECTED the canonical config: {e}")

try:
    O.enforce_eval_contract(
        w1cfg,
        contract_path=str(C200),
        subset_items_sha256=subset_hash,
        subset_label="configs/train_subsets/d1c_dev32.json",
        expected_contract_sha256=stale,
    )
    rec("2b", False, "enforce_eval_contract ACCEPTED a stale expected hash")
except Exception as e:  # noqa: BLE001
    rec("2b", True, f"enforce_eval_contract refuses stale expected hash: {str(e)[:110]}")

try:
    O.enforce_eval_contract(
        w1cfg,
        contract_path=str(C200),
        subset_items_sha256=subset_hash,
        subset_label="configs/train_subsets/d1c_dev32.json",
        require_clean_sha=True,
        expected_contract_sha256=declared,
    )
    rec("2c", False, "require_clean_sha=True ACCEPTED despite no clean committed SHA")
except Exception as e:  # noqa: BLE001
    rec("2c", True, f"require_clean_sha=True refuses (no clean SHA yet): {str(e)[:110]}")

# ---- B3 resume validation with the correct step_unit literal
REQ = ("model_config", "speaker_map", "subset", "eval_contract", "recipe", "recipe_source_manifest", "run_manifest", "config")
good = T.resolve_repro_hashes(w1cfg)
term = dict(w1cfg["model"]["termination"])


class Shim:
    _validate_resume = T.DiscreteLMTrainer._validate_resume

    def __init__(self, cfg, repro, termination):
        self.cfg = cfg
        self.repro_hashes = repro
        self.termination = termination
        self.grad_accum = 4
        self.allow_legacy_microbatch_resume = False


def mk_ckpt():
    return {
        "schema": "minispeech.discrete_lm_checkpoint",
        "schema_version": "2.0.0",
        "termination": copy.deepcopy(term),
        "model": {},
        "step_unit": T.STEP_UNIT_OPTIMIZER,
        "grad_accum": 4,
        "optimizer_step": 10,
        "microbatch_step": 40,
        "global_step": 10,
        "hashes": dict(good),
    }


def run(cur, ckpt, termination=None):
    try:
        Shim(w1cfg, cur, termination or term)._validate_resume(ckpt)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


e = run(good, mk_ckpt())
rec("3a", e is None, f"happy-path D1c resume accepted (err={e})")

for k in REQ:
    cur = dict(good)
    cur[k] = None
    e = run(cur, mk_ckpt())
    ok = e is not None and ("None" in str(e) or "missing" in str(e))
    rec(f"3b-{k}", ok, f"current {k}=None refused -> {str(e)[:95]}")

for k in REQ:
    ck = mk_ckpt()
    ck["hashes"][k] = "deadbeef"
    e = run(good, ck)
    rec(f"3c-{k}", e is not None and k in str(e), f"ckpt {k} mismatch refused -> {str(e)[:80]}")

for k in ("recipe", "recipe_source_manifest", "run_manifest"):
    ck = mk_ckpt()
    ck["hashes"].pop(k)
    e = run(good, ck)
    rec(f"3d-{k}", e is not None and k in str(e), f"ckpt missing {k} refused -> {str(e)[:80]}")

# B3e residual fail-open probe for the non-D1c (frame_stop_v1) path
legacy_term = {"mode": "frame_stop_v1"}
ck = mk_ckpt()
ck["termination"] = legacy_term
ck["hashes"].pop("run_manifest")
cur = dict(good)
cur["run_manifest"] = None
e = run(cur, ck, termination=legacy_term)
info("3e", f"non-D1c frame_stop_v1 with run_manifest=None -> {('SKIPPED (fail-open)' if e is None else e)[:110]}")

# B3f drop hashes from the YAML entirely -> D1c must still resolve non-None
for drop in ("run_manifest_hash", "recipe_hash", "recipe_source_manifest_hash", "eval_contract_hash"):
    cfg = yaml.safe_load(W1.read_text(encoding="utf-8"))
    cfg["train"].pop(drop, None)
    h = T.resolve_repro_hashes(cfg)
    nones = [k for k in REQ if h.get(k) is None]
    rec(f"3f-{drop}", not nones, f"dropping train.{drop} still resolves all keys (None={nones})")

# B3g contract file unreadable -> must NOT silently leave recipe None and pass
cfg = yaml.safe_load(W1.read_text(encoding="utf-8"))
cfg["train"]["eval_contract_path"] = "configs/eval_contracts/__does_not_exist__.json"
cfg["train"].pop("recipe_hash", None)
cfg["train"].pop("recipe_source_manifest_hash", None)
h = T.resolve_repro_hashes(cfg)
nones = [k for k in REQ if h.get(k) is None]
e = run(h, mk_ckpt())
rec(
    "3g",
    bool(nones) and e is not None,
    f"missing contract -> unresolvable keys {nones}; D1c resume refuses: {str(e)[:90]}",
)

# ---- B4: is the frozen source manifest enforced anywhere at runtime?
import subprocess

grep = subprocess.run(
    ["grep", "-rn", "verify_recipe_source_manifest", "scripts", "src"],
    capture_output=True,
    text=True,
    cwd=str(ROOT),
)
callers = [ln for ln in grep.stdout.splitlines() if "d1b_oracles.py" not in ln]
info("4a", f"runtime callers of verify_recipe_source_manifest outside its own module: {callers or 'NONE'}")

print("\n=== SUMMARY B ===")
nf = 0
for tag, v, msg in out:
    if v == "FAIL":
        nf += 1
        print(f"FAIL B{tag}: {msg}")
print(f"B_total={len(out)} B_fail={nf}")
