"""REAUDIT ROUND2 probe A — FAIL #2 (contract/recipe binding) + FAIL #7 (resume hashes).

Read-only against the worktree. Writes nothing outside tempfiles.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path("/mnt/c/projects/tts-self-d1c")
sys.path.insert(0, str(ROOT / "src"))

from minispeech.data import d1b_oracles as O  # noqa: E402
from minispeech.train import discrete_lm_trainer as T  # noqa: E402

W1 = ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml"
W2 = ROOT / "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml"
C200 = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json"
C201 = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_1.json"

results: list[tuple[str, str, str]] = []


def rec(tag: str, ok: bool, msg: str) -> None:
    results.append((tag, "PASS" if ok else "FAIL", msg))
    print(f"A{tag} {'PASS' if ok else 'FAIL'} {msg}", flush=True)


def info(tag: str, msg: str) -> None:
    results.append((tag, "INFO", msg))
    print(f"A{tag} INFO {msg}", flush=True)


def canon(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------- A1 contract self-check
raw200 = json.loads(C200.read_text(encoding="utf-8"))
declared = str(raw200.get("sha256"))
# Independent recompute: sha over contract body with self-hash fields removed.
body = {k: v for k, v in raw200.items() if k != "sha256"}
indep = hashlib.sha256(canon(body)).hexdigest()
rec("1a", indep == declared, f"independent raw-file recompute={indep[:16]} declared={declared[:16]} match={indep == declared}")

live = O.load_eval_contract(C200)
live_sha = str(live.get("sha256"))
rec("1b", live_sha == declared, f"load_eval_contract sha={live_sha[:16]} == declared file sha ({declared[:16]})")
rec(
    "1c",
    str(raw200.get("recipe_sha256")) == declared,
    f"recipe_sha256={str(raw200.get('recipe_sha256'))[:16]} == sha256={declared[:16]}",
)
info("1d", f"full live contract sha256={live_sha}")
info("1e", f"recipe_source_manifest_sha256={raw200.get('recipe_source_manifest_sha256')}")

stale = "a59ada9c1e0a64f067b27dd2b4d93ca270702bd41881746d1bb9a437d2f33772"
rec("1f", declared != stale, f"live contract sha != stale a59ada9c ({declared[:16]})")

# ---------------------------------------------------------------- A2 W1/W2 declared == live
for name, path in (("w1", W1), ("w2", W2)):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    t = cfg["train"]
    dec = str(t.get("eval_contract_hash") or "")
    rec(f"2-{name}-contract", dec == live_sha, f"{name} eval_contract_hash={dec[:16]} live={live_sha[:16]}")
    rec(
        f"2-{name}-recipe",
        str(t.get("recipe_hash") or "") == str(raw200.get("recipe_sha256")),
        f"{name} recipe_hash={str(t.get('recipe_hash'))[:16]} contract.recipe_sha256={str(raw200.get('recipe_sha256'))[:16]}",
    )
    rec(
        f"2-{name}-manifest",
        str(t.get("recipe_source_manifest_hash") or "") == str(raw200.get("recipe_source_manifest_sha256")),
        f"{name} recipe_source_manifest_hash={str(t.get('recipe_source_manifest_hash'))[:16]}",
    )
    info(f"2-{name}-runmanifest", f"{name} run_manifest_hash={t.get('run_manifest_hash')}")

# ---------------------------------------------------------------- A3 v2_0_1 parent pointer
raw201 = json.loads(C201.read_text(encoding="utf-8"))
rec(
    "3a",
    str(raw201.get("parent_sha256")) == live_sha,
    f"v2_0_1.parent_sha256={str(raw201.get('parent_sha256'))[:16]} live200={live_sha[:16]}",
)
rec(
    "3b",
    str(raw201.get("parent_recipe_sha256")) == str(raw200.get("recipe_sha256")),
    f"v2_0_1.parent_recipe_sha256={str(raw201.get('parent_recipe_sha256'))[:16]}",
)
# 2.0.1 self-consistency
d201 = str(raw201.get("sha256") or "")
b201 = {k: v for k, v in raw201.items() if k != "sha256"}
i201 = hashlib.sha256(canon(b201)).hexdigest()
rec("3c", (not d201) or i201 == d201, f"v2_0_1 self sha declared={d201[:16]} recomputed={i201[:16]}")
rec(
    "3d",
    str(raw201.get("recipe_sha256") or "") == str(raw200.get("recipe_sha256")),
    f"v2_0_1.recipe_sha256={str(raw201.get('recipe_sha256'))[:16]} preserves 2.0.0 recipe_sha256 (plan §7)",
)

# ---------------------------------------------------------------- A4 recipe_source_manifest
man = raw200.get("recipe_source_manifest") or {}
errs = O.verify_recipe_source_manifest(man, root=ROOT)
rec("4a", not errs, f"verify_recipe_source_manifest n_locked={len(man)} mismatches={errs}")
man_sha = hashlib.sha256(canon(man)).hexdigest()
rec(
    "4b",
    man_sha == str(raw200.get("recipe_source_manifest_sha256")),
    f"manifest dict sha={man_sha[:16]} declared={str(raw200.get('recipe_source_manifest_sha256'))[:16]}",
)
want = [
    "src/minispeech/data/discrete_oracles.py",
    "src/minispeech/data/d1b_oracles.py",
    "src/minispeech/eval/asr.py",
    "src/minispeech/data/codec_seq_protocol.py",
    "src/minispeech/constants.py",
]
missing = [w for w in want if w not in man]
rec("4c", not missing, f"previously-unlocked modules now locked: missing={missing}")
info("4d", f"locked set ({len(man)}): {sorted(man)}")

# A4e: is the W1/W2 stripped hash actually sensitive to non-binding content?
tmp = Path(tempfile.mkdtemp())
w1_copy = tmp / W1.name
w1_copy.write_text(W1.read_text(encoding="utf-8"), encoding="utf-8")
base = O.recipe_source_file_sha256(w1_copy)
rec("4e", base == man.get("configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml"), f"stripped hash of pristine copy == manifest entry ({base[:16]})")
cfg1 = yaml.safe_load(w1_copy.read_text(encoding="utf-8"))
cfg1["train"]["seed"] = 999
w1_copy.write_text(yaml.safe_dump(cfg1, allow_unicode=True, sort_keys=False), encoding="utf-8")
mut = O.recipe_source_file_sha256(w1_copy)
rec("4f", mut != base, f"stripped hash IS sensitive to train.seed change ({mut[:16]} != {base[:16]})")
# and the binding keys are deliberately invisible
w1_copy.write_text(W1.read_text(encoding="utf-8"), encoding="utf-8")
cfg2 = yaml.safe_load(w1_copy.read_text(encoding="utf-8"))
cfg2["train"]["eval_contract_hash"] = stale
w1_copy.write_text(yaml.safe_dump(cfg2, allow_unicode=True, sort_keys=False), encoding="utf-8")
blind = O.recipe_source_file_sha256(w1_copy)
info("4g", f"stripped hash is BLIND to eval_contract_hash tamper: same={blind == base} (by design; 72 must catch it)")
# subset_hash / speaker_map_hash must NOT be blind
w1_copy.write_text(W1.read_text(encoding="utf-8"), encoding="utf-8")
cfg3 = yaml.safe_load(w1_copy.read_text(encoding="utf-8"))
cfg3["train"]["subset_hash"] = "0" * 64
w1_copy.write_text(yaml.safe_dump(cfg3, allow_unicode=True, sort_keys=False), encoding="utf-8")
sh = O.recipe_source_file_sha256(w1_copy)
rec("4h", sh != base, f"stripped hash catches subset_hash tamper: {sh != base}")

# ---------------------------------------------------------------- A5 resolve_repro_hashes on real W1/W2
REQ = ("config", "model_config", "speaker_map", "subset", "eval_contract", "recipe", "recipe_source_manifest", "run_manifest")
resolved: dict[str, dict] = {}
import os

os.chdir(ROOT)
for name, path in (("w1", W1), ("w2", W2)):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    h = T.resolve_repro_hashes(cfg)
    resolved[name] = h
    nones = [k for k in REQ if h.get(k) is None]
    rec(f"5-{name}", not nones, f"{name} resolve_repro_hashes inert(None) keys={nones}")
    info(f"5-{name}-vals", json.dumps({k: (str(v)[:16] if v else v) for k, v in h.items()}, sort_keys=True))
    rec(
        f"5-{name}-contract",
        str(h.get("eval_contract")) == live_sha,
        f"{name} resolved eval_contract == live contract sha",
    )

# A5c: stale declared contract hash must raise inside resolve_repro_hashes
cfg_stale = yaml.safe_load(W1.read_text(encoding="utf-8"))
cfg_stale["train"]["eval_contract_hash"] = stale
try:
    T.resolve_repro_hashes(cfg_stale)
    rec("5c", False, "stale eval_contract_hash did NOT raise in resolve_repro_hashes")
except ValueError as e:
    rec("5c", True, f"stale eval_contract_hash raises: {str(e)[:90]}")

# A5d: is the declared run_manifest_hash reproducible / meaningful?
cfg_norm = yaml.safe_load(W1.read_text(encoding="utf-8"))
declared_rm = str(cfg_norm["train"].get("run_manifest_hash") or "")
cfg_norm["train"].pop("run_manifest_hash", None)
h_derived = T.resolve_repro_hashes(cfg_norm)
rec(
    "5d",
    str(h_derived.get("run_manifest")) == declared_rm,
    f"w1 declared run_manifest_hash reproducible from derived identity: declared={declared_rm[:16]} derived={str(h_derived.get('run_manifest'))[:16]}",
)

# ---------------------------------------------------------------- A6 _validate_resume fail-closed (D1c)
class _FakeTrainer:
    """Minimal shim exposing only what _validate_resume touches."""

    def __init__(self, cfg, repro, termination, grad_accum=4):
        self.cfg = cfg
        self.repro_hashes = repro
        self.termination = termination
        self.grad_accum = grad_accum
        self.allow_legacy_microbatch_resume = False

    _validate_resume = T.DiscreteLMTrainer._validate_resume


w1cfg = yaml.safe_load(W1.read_text(encoding="utf-8"))
term = dict(w1cfg["model"]["termination"])
good = dict(resolved["w1"])
base_ckpt = {
    "schema": "minispeech.discrete_lm_checkpoint",
    "schema_version": "2.0.0",
    "termination": term,
    "model": {},
    "step_unit": "optimizer_step",
    "grad_accum": 4,
    "optimizer_step": 10,
    "microbatch_step": 40,
    "global_step": 10,
    "hashes": dict(good),
}


def resume_try(label, cur, ckpt):
    tr = _FakeTrainer(w1cfg, cur, term)
    try:
        tr._validate_resume(ckpt)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


err = resume_try("happy", good, base_ckpt)
rec("6a", err is None, f"matched hashes resume OK (err={err})")

for key in REQ:
    cur = dict(good)
    cur[key] = None
    e = resume_try(key, cur, base_ckpt)
    rec(f"6b-{key}", e is not None and "None" in str(e), f"D1c resume refuses current {key}=None -> {str(e)[:80]}")

for key in REQ:
    ck = copy.deepcopy(base_ckpt)
    ck["hashes"][key] = "deadbeef"
    e = resume_try(key, good, ck)
    rec(f"6c-{key}", e is not None, f"D1c resume refuses {key} mismatch -> {str(e)[:70]}")

for key in ("run_manifest", "recipe", "recipe_source_manifest"):
    ck = copy.deepcopy(base_ckpt)
    ck["hashes"].pop(key)
    e = resume_try(key, good, ck)
    rec(f"6d-{key}", e is not None, f"D1c resume refuses ckpt missing {key} -> {str(e)[:70]}")

# A6e: the *previous* fail-open — drop the key from cfg entirely
cfg_drop = yaml.safe_load(W1.read_text(encoding="utf-8"))
cfg_drop["train"].pop("run_manifest_hash", None)
h_drop = T.resolve_repro_hashes(cfg_drop)
e = resume_try("drop", h_drop, base_ckpt)
rec(
    "6e",
    (h_drop.get("run_manifest") is not None) and (e is None or "run_manifest" in str(e)),
    f"dropping run_manifest_hash from cfg no longer yields None (got={str(h_drop.get('run_manifest'))[:16]}); resume err={str(e)[:70]}",
)

# A6f: non-D1c path (frame_stop_v1) still skips None -> record as residual fail-open
term_legacy = {"mode": "frame_stop_v1"}
cur_legacy = {k: (None if k == "run_manifest" else v) for k, v in good.items()}
ck_legacy = copy.deepcopy(base_ckpt)
ck_legacy["termination"] = term_legacy
tr = _FakeTrainer(w1cfg, cur_legacy, term_legacy)
try:
    tr._validate_resume(ck_legacy)
    info("6f", "non-D1c (frame_stop_v1) still SKIPS a None current hash (residual fail-open, out of D1c scope)")
except Exception as e:  # noqa: BLE001
    info("6f", f"non-D1c also refuses: {str(e)[:80]}")

print("\n=== SUMMARY A ===")
for tag, verdict, msg in results:
    if verdict == "FAIL":
        print(f"FAIL A{tag}: {msg}")
n_fail = sum(1 for _, v, _ in results if v == "FAIL")
print(f"A_total={len(results)} A_fail={n_fail}")
