# REAUDIT_ROUND7 — independent adversarial re-verification of the D1c INFRA repair (round 7)

**Plan:** `.cursor/plans/d1c_semantic_eos_repair_b81ecc3e.plan.md`  
**Slug:** `d1c_semantic_eos_repair_b81ecc3e`  
**Worktree audited:** `C:\projects\tts-self-d1c`, branch `d1c/semantic-eos`  
**Branch tip:** `2fc2e8d6155e8fd9e235fbf8cb8318b2255f6a56` (dirty tree; round-7 not committed)  
**Input under audit:** `REPAIR_ROUND7.md` (claims Round-6 MEDIUM I1/K7 identical-weights / forged-stamp
closed via pairwise-distinct model `state_dict` digests; J1 hashes↔provenance cross-check; live
probe 7/7; pytest `408 passed, 3 skipped`, 411 collected)  
**Prior audits:** `REAUDIT_ROUND6.md` (verdict INFRA_READY_FOR_HUMAN_AUDIT; MEDIUM residual I1/K7;
LOW J1)  
**Auditor:** independent re-audit agent. Every claim in `REPAIR_ROUND7.md` was treated as an
unverified assertion and re-tested with live probes that drive production
`select_d1c_formal_reports` and `scripts/76 --d1c-formal` as a subprocess. Repair-agent logs were
**not** treated as evidence.  
**Date:** 2026-08-06  
**Constraints honoured:** no commit, no push, no GPU (`CUDA_VISIBLE_DEVICES=`), no seed invented,
no methodology edit, no training / inference / synthesis, no Wave 0–2 PASS claim.

---

## Overall verdict

> ## INFRA_READY_FOR_HUMAN_AUDIT
>
> **The Round-6 MEDIUM identical-weights / forged-stamp gap is genuinely closed on the formal
> selector path.** Five byte-distinct re-saves of one `model` state_dict with fabricated distinct
> `payload['step']` values and a shared sealed `hashes` block are refused at
> `select_d1c_formal_reports` **and** at production CLI `scripts/76 --d1c-formal` with
> `rc≠0` and **no** `checkpoint_selection.json` written (`I1`, `K7`). A legitimate five-candidate
> set with pairwise-distinct weight digests is accepted end-to-end (`L0`–`L2`). J1
> (`checkpoint.hashes.*` ≠ `provenance.*_sha256`) refuses at selector and CLI. Bypass probes
> (empty/missing `model`, forged report-level weight-digest fields, partial 4+1 collision) all
> fail-closed. Prior blockers (rename, CLI invent, distinct file sha, FAIL #2/#7 surface, residual
> grads, select8 2+6, immediate EOS, WAVE1 live digests) hold. Independent pytest:
> **`408 passed, 3 skipped`** (411 collected) against the current dirty tree; production sources
> **pre-date** the repair pytest log and match this re-audit reproduction.
>
> **NEW findings (not verdict-blocking):** (1) **[LOW]** J1 only requires hashes↔provenance
> *equality*; a consistent dual forgery of both fields to the same wrong digest still passes
> selector (evaluator live-contract check at emit is the remaining layer). (2) **[LOW/INFO]**
> pairwise-distinct weight digests close *clone* attacks but do not bind digests to a
> trainer-signed per-step ledger — five fabricated *distinct* dummy weight tensors with forged
> stamps would still satisfy distinctness (out of stated Round-7 scope; same class as Round-6's
> "needs a ledger" note).
>
> Waves 0–2 remain **not PASS**. Wave 3 remains **NO-GO**.

---

## Per-item verdicts

| # | Check | Round-7 claim | This audit | Evidence |
|---|---|---|---|---|
| 1 | I1 identical weights + forged steps/hashes → refuse; no selection file | FIXED | **PASS** — selector refuse `identical model weights` (1/5 digests) | `reaudit7_probe_attack.log` I0–I1 |
| 1b | K7 same attack at `scripts/76 --d1c-formal` | FIXED | **PASS** — `rc=1`, `wrote=False` | K7 |
| 2 | Legitimate five distinct weight digests + valid payload → accept E2E | asserted | **PASS** — selector + CLI write selection, `selected_step=250` | L0–L2 |
| 3 | J1 hashes ≠ provenance → refuse | FIXED | **PASS** — selector + CLI | J1, J1c |
| 4 | Bypass: omit/forge weight digest field in report | (implied live digest) | **PASS** — no trusted report field; forged fields ignored; attack still refuses | B3, B3b |
| 4b | Bypass: empty / missing `model` state_dict | claimed refuse | **PASS** | B1, B2 |
| 4c | Partial collision (4 distinct + 1 clone) | (pairwise rule) | **PASS** — refuse at 4/5 digests | B4 |
| 5 | Rename (no payload step) still closed | not regressed | **PASS** | R1 |
| 5b | CLI invent (`best.pt` + `--checkpoint-step`) still closed | not regressed | **PASS** | R2 |
| 5c | Distinct file sha / duplicate sha still closed | not regressed | **PASS** | R3 |
| 5d | FAIL #2 reseal after round-7 | resealed | **PASS** — live `181acd02…` / `ebb91df5…`, 19 locked, mismatches=[], r6 superseded | rest R2a–R2j |
| 5e | FAIL #7 resume hashes | not regressed | **PASS** (+ round-6 digest now refused) | R7a–R7d, R7b2 |
| 5f | Residual grads / select8 2+6 / immediate EOS / aliases | not regressed | **all hold** | RXa–RXe |
| 5g | WAVE1 (+ authoring mirror) live round-7 digests | claimed | **PASS** | H-WAVE1_* |
| 6 | Pytest `408 passed, 3 skipped` (411 collected) | claimed | **VALID — exact independent reproduction** | `reaudit7_full_pytest.log` |
| — | Consistent dual hashes+provenance forgery | not claimed closed | **LOW residual** | code review + J1 scope |
| — | Fabricated distinct dummy weights (non-clone) | out of r7 contract | **LOW/INFO residual** | design note |

Totals: attack probe **16 PASS / 0 FAIL**; rest **26 PASS / 0 FAIL**; pytest exact match.

---

## Method — what I did differently from the repair agent

| Artifact | What it does |
|---|---|
| `logs/reaudit7_probe_attack.py` / `.log` | Independent I1/K7 attack, legit accept, J1 (+CLI), empty/missing model, forged report digests, partial collision, rename, CLI invent, duplicate sha — production selector + `scripts/76 --d1c-formal` |
| `logs/reaudit7_probe_rest.py` / `.log` | Independent reseal recompute to `181acd02…` / `ebb91df5…`, WAVE1 both copies, r5/r6 supersession, resume hashes, residual grads / EOS / select8 |
| `logs/reaudit7_full_pytest.log` | Independent full-suite reproduction (`408 passed, 3 skipped` in 74.80s) |
| `logs/reaudit7_new_tests.log` | Direct run of the two new unit tests |
| `logs/reaudit7_collect_only.log` | Collect-only → **411** |
| `logs/reaudit7_post_probe_hashes.txt` | Post-probe digests of key sources |

Repair-agent `repair_round7_probe_identical_weights.log` was read only after independent probes passed, for claim cross-check — not as primary evidence.

---

## 1. Identical-weights / forged-stamp attack (`I1` / `K7`) — **PASS**

```text
I0 PASS attack setup weight_digests=1/5 file_sha=5/5
I1 PASS selector refused: … identical model weights … (distinct_weight_digests=1, n=5)
K7 PASS CLI rc=1 wrote=False
```

Mechanism verified in `scripts/76_select_discrete_checkpoint.py`:
`_model_state_sha256` digests sorted non-empty `payload['model']` tensors; 
`_require_distinct_checkpoint_identities` requires `len(set(weight_digests)) == n` after file-sha /
basename / shared-run / J1 checks. Junk padding still yields distinct file sha256; that is no
longer sufficient.

## 2. Legitimate five-candidate accept — **PASS**

```text
L0 PASS legit distinct_weight_digests=5/5
L1 PASS selector ACCEPTS legit set selected_step=250
L2 PASS CLI rc=0 wrote=True
```

## 3. J1 hashes ↔ provenance — **PASS** (with LOW residual)

```text
J1  PASS selector refused checkpoint.hashes.eval_contract != provenance.eval_contract_sha256
J1c PASS CLI rc=1 wrote=False
```

**Residual [LOW]:** equality-only. Editing *both* `checkpoint.hashes.eval_contract` and
`provenance.eval_contract_sha256` to the same fabricated digest still satisfies J1. Live contract
binding remains an evaluator/emit-time concern, not a selector re-check against disk.

## 4. Alternate bypass surfaces — **PASS** (no bypass found for I1)

| Probe | Result |
|---|---|
| B1 empty `model={}` | refuse non-empty state_dict |
| B2 missing `model` key | refuse |
| B3 forge `checkpoint.model_state_sha256` / `weight_digest` on identical-weight set | ignored; live digest still refuses |
| B3b same fields on legit set | ignored; still accepts |
| B4 4 distinct + 1 clone of step-250 weights | refuse (`distinct_weight_digests=4`) |

There is **no** omitable weight-digest report field on the formal path — digests are computed from
checkpoint bytes at select time.

## 5. Prior regressions — **PASS**

```text
R1 PASS rename/no-step refused (payload['step'] required)
R2 PASS CLI invent refused (cannot invent via --checkpoint-step)
R3 PASS duplicate checkpoint.sha256 refused
RXa PASS residual grads 45/45 (cb0_w=0, eos_w=0)
RXb PASS immediate EOS codes.shape=(1, 0, 16)
RXc PASS select8 = 2 RUSLAN + 6 CV
RXd PASS select8 ∪ gate24 = full32, disjoint
RXe PASS d1c_select8 alias registered
R2* PASS reseal 181acd02… / ebb91df5…; 19 locked; w1/w2 bound; v2.0.1 parent ok
R7* PASS resume hashes; round-6 digest refused as stale pin
```

FAIL #4 (binding / recipe assert surface): `scripts/72` still contains fail-closed
`eval_contract_hash mismatch`; sealed manifest includes `scripts/76` after round-7 reseal.

## 6. Pytest attestation — **VALID**

| Check | Result |
|---|---|
| Claimed | `408 passed, 3 skipped`, 411 collected |
| Independent reproduction | `408 passed, 3 skipped, 2 warnings in 74.80s` — **exact match** |
| Collect-only | **411** |
| New tests | `test_selector_refuses_identical_model_weights_forged_stamps`, `test_selector_refuses_hashes_provenance_skew` — both pass |
| Arithmetic | round-6 409 collected → round-7 411; +2 matches claimed new tests |
| GPU | masked |
| Source vs repair log mtimes | `scripts/76` + `tests/test_d1c_infra_repair.py` **05:07**; `repair_round7_full_pytest.log` **05:14** — sources **do not** post-date the repair log; this re-audit's full pytest also matches |

Post-probe source digests (`logs/reaudit7_post_probe_hashes.txt`):

```text
254d30474e1bbe27…  scripts/76_select_discrete_checkpoint.py
878652ca74d2c903…  tests/test_d1c_infra_repair.py
e8d087c0e7ed5e90…  configs/eval_contracts/d1c_semantic_eos_v2_0_0.json
```

## 7. Remaining defects (fix order)

1. **[LOW — NEW]** Selector J1 is equality-only; consistent dual forgery of
   `checkpoint.hashes.*` + `provenance.*_sha256` still passes. Optionally bind selector provenance
   to the live eval-contract file hash.
2. **[LOW/INFO — NEW]** Distinct weight digests ≠ trainer step ledger. Fabricated *distinct*
   dummy weights with forged stamps remain out of Round-7's chosen contract (collision-refuse).
3. **[LOW]** Prior mutation coverage holes `M2`/`M3`/`M7`/`M8`/`M9` (unchanged; layered guards
   live but not unit-mutation-covered). New I1/J1 guards **do** have dedicated tests.
4. **[LOW]** `--promote-best` D1c branch still lacks symlink guard (`RY2`).
5. **[LOW]** Deferred items from prior rounds (redundant hashing, weight EOS aggregate, pin
   `verify_recipe_source_manifest` to repo root, assert-vs-overwrite `recipe_hash` in `72`,
   non-D1c resume fail-open, CER margin).

Standing non-code blockers: no clean committed D1c SHA; no confirmation-seed commitment;
Wave 3/4 NO-GO pending human sign-off.

---

## Scope confirmations and post-audit integrity

- No commit / push / branch mutation. Tip remains `2fc2e8d`; working tree dirty (includes round-7
  repair + this re-audit artifacts).
- No GPU; `CUDA_VISIBLE_DEVICES=` on every run. No training, decode, or model synthesis → **no
  listen-demo obligation**.
- No seed invented; no methodology edit; no Wave 0–2 PASS claim.
- Probe scratch under `/tmp` and `.cursor/results/.../logs/` only.
- Authoring mirror copy of this report:  
  `C:\projects\tts-self\.cursor\results\d1c_semantic_eos_repair_b81ecc3e\REAUDIT_ROUND7.md`
