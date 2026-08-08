# REPAIR_ROUND7 — D1c INFRA repair (identical-weights / forged stamps)

**Plan:** `d1c_semantic_eos_repair_b81ecc3e`  
**Worktree:** `C:\projects\tts-self-d1c` (`d1c/semantic-eos`)  
**Branch tip at start:** `2fc2e8d` (WIP commit present; working tree still dirty)  
**Input:** `REAUDIT_ROUND6.md` (verdict **INFRA_READY_FOR_HUMAN_AUDIT**; MEDIUM: fabricated
`payload['step']` + shared `payload['hashes']` on identical weights still selects — `I1` / CLI `K7`)  
**Date:** 2026-08-06  
**Constraints:** no commit, no push, no GPU (`CUDA_VISIBLE_DEVICES=`), no seed invented, no
methodology edit, no Wave 0–2 PASS claim. Wave 3 remains **NO-GO**.

---

## Verdict

> **INFRA_REPAIR_ROUND7_COMPLETE_AWAITING_HUMAN_REAUDIT**
>
> The Round-6 MEDIUM identical-weights / forged-stamp gap is closed fail-closed on the formal
> selector path: five candidates must be distinct **training identities** (pairwise-distinct
> model `state_dict` digests), not merely distinct files with fabricated payload steps and a
> shared hashes block. Live probe: attack refused at selector + production CLI (`I1`/`K7`);
> legitimate five distinct-weight steps accepted. Also closed J1
> (`checkpoint.hashes.*` vs `provenance.*_sha256`). Pytest
> `408 passed, 3 skipped` (411 collected). Contract resealed after `scripts/76` edit.
>
> Waves 0–2 remain **not PASS**. Wave 3 remains **NO-GO**.

---

## Contract decision (identical-weights / forged stamps)

**Chosen: refuse colliding model `state_dict` digests across the five formal candidates.**

| Situation | Behavior |
|---|---|
| Five files, distinct `checkpoint.sha256` (junk padding), **identical** `model` weights, fabricated distinct `payload['step']`, shared sealed `payload['hashes']` | **refuse** (`identical model weights` / `distinct_weight_digests`) |
| Five real distinct training steps with distinct model weights, shared sealed run hashes, basename + payload step consistent | **accept** |
| Missing/empty/unloadable `model` state_dict on a candidate | **refuse** (fail-closed; file padding alone is not a training identity) |
| `checkpoint.hashes.eval_contract` / `recipe` ≠ `provenance.eval_contract_sha256` / `recipe_sha256` | **refuse** (J1) |

Rationale: Round-6 already binds filename ↔ payload step ↔ sealed run fingerprint. That still
allows cloning one weight state five times and forging trainer stamps. File-level sha256 is not
enough (junk padding). Trainer already stamps `step` + `hashes`; the selector must additionally
prove the five weight tensors are distinct training identities.

Not chosen for this round: requiring distinct `run_manifest` across candidates (contradicts the
one-sealed-run rule), or requiring a new trainer-signed per-step ledger (heavier; not needed once
weight digests collide-refuse).

---

## Fixes

### 1. Identical-weights forged-stamp attack closed — **FIXED** (`I1` / `K7`)

`scripts/76_select_discrete_checkpoint.py`:

- `_model_state_sha256(path)` — digest sorted non-empty `payload['model']` tensors
- `_require_distinct_checkpoint_identities` — after distinct file sha / basename / shared run
  fingerprint, require pairwise-distinct model digests

### 2. Hashes ↔ provenance cross-check — **FIXED** (`J1`)

`_require_checkpoint_hashes_match_provenance` — `checkpoint.hashes.eval_contract` must equal
`provenance.eval_contract_sha256`; same for `recipe` / `recipe_sha256`.

### 3. Tests — **ADDED**

- `test_selector_refuses_identical_model_weights_forged_stamps` (I1/K7)
- `test_selector_refuses_hashes_provenance_skew` (J1)
- `_write_candidate_ckpt` now writes trainer-shaped torch checkpoints with step-unique weights
  by default (so legit five-sets keep distinct training identities)

### 4. Contract reseal — **DONE**

Locked sources include `scripts/76`; resealed via `logs/repair_round7_rebind_hashes.py`.
`WAVE1_REPORT.md` updated to live round-7 digests.

---

## Live probe

`logs/repair_round7_probe_identical_weights.py` →
`logs/repair_round7_probe_identical_weights.log`:

```text
I0 PASS attack setup weight_digests=1/5 file_sha=5/5
I1 PASS selector refused: identical model weights (distinct_weight_digests=1, n=5)
K7 PASS CLI rc=1 wrote=False (scripts/76 --d1c-formal)
L0 PASS legit distinct_weight_digests=5/5
L1 PASS selector ACCEPTS legit set selected_step=250
L2 PASS CLI rc=0 wrote=True
J1 PASS selector refused hashes/provenance skew
SUMMARY pass=7 fail=0
```

---

## Pytest

| Check | Result |
|---|---|
| Full suite | `408 passed, 3 skipped, 2 warnings` (411 collected) |
| Log | `logs/repair_round7_full_pytest.log` |
| Focused (identity + J1 + related) | `8 passed` — `logs/repair_round7_focused_pytest.log` |
| GPU | masked (`CUDA_VISIBLE_DEVICES=`) |
| GIT_DIR | `/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c` (Windows worktree pointer under `/mnt/c`) |

Arithmetic: round-6 `406 passed` / 409 collected → round-7 `408 passed` / 411 collected (+2 new tests).

---

## Sealed digests (post-edit reseal)

| Field | Value |
|---|---|
| `contract_sha256` / `recipe_sha256` | `181acd0274b66253facb9277a9473336d0c52cf87d5fe23dd39199d31518baad` |
| `recipe_source_manifest_sha256` | `ebb91df5f1574d11e30017eae870642aae6373777d123741e25add62726551fc` |
| Locked sources | 19 (resealed after `scripts/76`) |
| Artifact | `artifacts/repair_round7_hashes.json` |

Round-6 digests `c9fd265a…` / `85b896a9…` are superseded.

---

## Remaining issues (not fixed this round)

1. **[LOW]** Mutation coverage holes `M2`/`M3`/`M7`/`M8`/`M9` (layered hash guards live but not
   unit-mutation-covered).
2. **[LOW]** `--promote-best` D1c branch lacks symlink guard (`RY2`).
3. **[LOW]** Redundant full-file hashing; weight EOS aggregate; pin
   `verify_recipe_source_manifest` to repo root; assert vs overwrite `recipe_hash` in `72`;
   non-D1c resume fail-open; CER margin tolerance — deferred as before.

Standing non-code blockers: no clean committed D1c SHA; no confirmation-seed commitment; Wave 3/4
NO-GO pending human sign-off.

---

## Scope confirmations

- No commit / push. Tip remains `2fc2e8d`; working tree dirty (includes this repair).
- No GPU; no training / decode / synthesis → no listen-demo obligation.
- No seed invented; no methodology edit; no Wave 0–2 PASS claim.
- Key files: `scripts/76_select_discrete_checkpoint.py`, `tests/test_d1c_infra_repair.py`,
  `configs/eval_contracts/d1c_semantic_eos_v2_0_0.json` (+ v2_0_1), W1/W2 YAML binding hashes,
  `WAVE1_REPORT.md`, this report + probe/rebind logs.
