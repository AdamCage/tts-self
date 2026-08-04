# REAUDIT_ROUND6 — independent adversarial re-verification of the D1c INFRA repair (round 6)

**Plan:** `.cursor/plans/d1c_semantic_eos_repair_b81ecc3e.plan.md`
**Slug:** `d1c_semantic_eos_repair_b81ecc3e`
**Worktree audited:** `C:\projects\tts-self-d1c`, branch `d1c/semantic-eos`
**Branch tip:** `0661d10eb616362e74dc1750e03cf25c33a7943e` (baseline `origin/master`; the branch still
carries **no commits of its own** — all D1c work remains uncommitted working-tree state)
**Input under audit:** `REPAIR_ROUND6.md` (claims the round-5 MEDIUM rename/re-save gap is closed
fail-closed: `payload['step']` required on formal path; `payload['hashes']` required + verified;
basename dedicated test; WAVE1 synced to `c9fd265a…` / `85b896a9…`; live probe 10/10; pytest
`406 passed, 3 skipped`, 409 collected)
**Prior audits:** `REAUDIT.md` … `REAUDIT_ROUND5.md` (r5 — verdict INFRA_READY_FOR_HUMAN_AUDIT,
NEW MEDIUM: filename-only rename/re-save; MEDIUM–LOW: stale WAVE1)
**Auditor:** independent re-audit agent. Every claim in `REPAIR_ROUND6.md` was treated as an
unverified assertion and re-tested with live probes that drive the **production entry points**,
including `scripts/76 --d1c-formal` as a subprocess. A previous re-audit attempt for this round was
interrupted mid-start and left incomplete probe scripts; this audit **re-executed** those probes
fresh and wrote its own rest/digest probe. Do not treat the repair agent's logs as evidence.
**Date:** 2026-08-04
**Constraints honoured:** no commit, no push, no GPU (`CUDA_VISIBLE_DEVICES=` on every run), no seed
invented, no methodology edit, no training / inference / synthesis, no Wave 0–2 PASS claim.

---

## Overall verdict

> ## INFRA_READY_FOR_HUMAN_AUDIT
>
> **The round-5 MEDIUM rename/re-save gap is genuinely closed, and I attacked it harder than the
> repair agent did.** Five byte-distinct re-saves with **no** `payload['step']` produce **0/5**
> formal reports (evaluator refuses citing `payload['step']`); wrong payload-vs-filename steps are
> refused 4/5; missing / placeholder / foreign `payload['hashes']` refuse at the evaluator; the
> production CLI `scripts/76 --d1c-formal` refuses mixed run identity, missing hashes, basename
> alias, and duplicate sha, and **accepts** a legitimate five-candidate set. Basename
> `candidate_step_<step:04d>.pt` is now load-bearing (mutation goes red via the dedicated test —
> round-5's Y3 hole is closed). `WAVE1_REPORT.md` (worktree **and** authoring mirror) publishes the
> live round-6 digests `c9fd265a…` / `85b896a9…`. Pytest attests **exactly**
> `406 passed, 3 skipped` (409 collected). Prior blockers (round-4 CLI invent, distinct sha, FAIL
> #2/#4/#7 surface, residual grads, select8 2+6, immediate EOS) hold.
>
> **NEW finding (MEDIUM, not verdict-blocking under the stated rules):** closing filename invent
> closed one door, not every semantic-duplicate path. An operator who **fabricates**
> `payload['step']` + a shared `payload['hashes']` block and re-saves one weight state five times
> still gets five formal reports the selector accepts (`I1`, CLI `K7`). Round-6 proves the five
> *training-step stamps* are present and mutually consistent with filenames and one sealed-run
> fingerprint; it does **not** prove the five *weight tensors* came from five distinct trainer
> steps. That residual requires forging trainer-stamped fields, not a rename.
>
> **NEW finding (LOW):** five layered round-6 guards (selector hashes presence, mixed-identity,
> emitter synthetic-hashes refuse, schema hashes require, schema cross-wired basename) are **live**
> but **not** mutation-covered by the unit suite (`M2`/`M3`/`M7`/`M8`/`M9`). The three guards that
> round-6 was asked to add tests for (payload-step require, `require_checkpoint_payload_hashes`,
> basename dedicated) **are** covered (`M1`/`M5`/`M6`).
>
> Waves 0–2 remain **not PASS**. Wave 3 remains **NO-GO**.

Round 5 closed CLI invent and distinct-file identity, but left filename invent open. Round 6 binds
candidates to trainer-stamped `step` + `hashes`. The residual is one layer further out: the stamps
themselves are not bound to the weight bytes.

**Scope note on the verdict rule.** The stated rule flips to STILL_INVALID_RUN if "rename attack
still works, hashes optional, WAVE1 still stale critically, or critical regression". The rename
attack as defined (filename steps, **missing/wrong** payload step) does **not** work. Hashes are
enforced. WAVE1 is synced. If the parent reads "rename/re-save" as covering fabricated trainer
stamps (`I1`), the verdict flips; on the mechanism round 6 was asked to close, it is closed.

## Per-item verdicts

| # | Check | Round-6 claim | This audit | Evidence |
|---|---|---|---|---|
| 1 | Rename/re-save (no payload step) → refuse; 0 formal reports | FIXED | **PASS** — 0/5 reports, no gate leak | `reaudit6_probe_attack.log` C1–C1b |
| 1b | Wrong payload step vs filename | FIXED | **PASS** — 4/5 mismatch refuse; only self-consistent survives | C2–C2a |
| 2 | Missing `payload['hashes']` / mismatch → refuse | FIXED | **PASS** at helper, main path, schema, CLI | B1–B6, C3–C5, F5, K3 |
| 3 | Legitimate five `candidate_step_XXXX.pt` with step+hashes → accept | asserted | **PASS** — main() stamps, selector selects, CLI writes selection | D1–D3; CLI K0–K1 |
| 4 | Basename rule load-bearing | FIXED | **PASS** — dedicated test + e2e catch mutation | M1, M1a; live E1 |
| 5 | WAVE1 (+ authoring mirror) live digests | FIXED | **PASS** — `c9fd265a…` / `85b896a9…` both copies | rest H-WAVE1_* |
| 6 | Round-4 CLI invent still closed | not regressed | **PASS** — 0/5 | C6–C6a |
| 6 | Distinct sha / forged sha / unreachable path | not regressed | **PASS** | G2, H1–H2; CLI K5 |
| 6 | FAIL #2 reseal after round-6 | resealed | **PASS** — 19 locked, mismatches=[], r4/r5 superseded | rest R2a–R2k |
| 6 | FAIL #7 resume hashes | not regressed | **PASS** (+ round-5 digest now refused) | R7a–R7d, R7b2 |
| 6 | Residual grads / select8 2+6 / immediate EOS / aliases | not regressed | **all hold** | RXa–RXe |
| 6 | Real `_load_model` returns step+hashes | implied | **PASS** (round-5 LOW #6 covered live) | F0–F1 |
| 7 | Pytest `406 passed, 3 skipped` (409 collected) | claimed | **VALID — exact reproduction** | `reaudit6_full_pytest.log` |
| — | **NEW: fabricated payload step+hashes, same weights** | not disclosed | **INFO/MEDIUM residual** | I1; CLI K7 |
| — | **NEW: mutation holes on layered hash guards** | not disclosed | **LOW coverage holes** | M2, M3, M7, M8, M9 |

Totals: attack probe **37 PASS / 0 FAIL / 5 INFO**; CLI **10 PASS / 0 FAIL / 1 INFO**; rest
**33 PASS / 0 FAIL / 1 INFO**; mutation **9 PASS / 5 FAIL (coverage) / 2 INFO**; pytest exact match.

## Method — what I did differently from the repair agent

Everything below was executed, not read. Probe scripts and logs sit next to this report.

| Artifact | What it does |
|---|---|
| `logs/reaudit6_probe_attack.py` / `.log` | **decisive probe.** Real `torch.save` → **real `_load_model`** (payload read from file bytes — not a factory seam) → real `resolve_checkpoint_step` → real gate → real emitter → **real unpatched schema** → real selector. Rename, wrong step, missing/foreign hashes, round-4 invent, legit set, basename, mixed identity, duplicate sha, **and** the fabricated-stamp residual |
| `logs/reaudit6_probe_cli.py` / `.log` | same refusals / acceptance through `python scripts/76_select_discrete_checkpoint.py --d1c-formal` as a **subprocess** |
| `logs/reaudit6_probe_mutation.py` / `.log` | mutation testing of round-6 (+ round-5) guards spliced out in memory; every test in `test_d1c_infra_repair.py` re-run |
| `logs/reaudit6_probe_rest.py` / `.log` | independent reseal recompute, WAVE1 both copies, r4/r5 supersession, resume hashes, residual grads / EOS / select8 |
| `logs/reaudit6_full_pytest.log` | independent full-suite reproduction |
| `logs/reaudit6_post_probe_hashes.json` | post-probe digests of attested sources |

Test environment: WSL2 Ubuntu, venv `/home/adam/projects/tts-self/.venv`, Python 3.11,
rootdir `/mnt/c/projects/tts-self-d1c`, `PYTHONPATH` pinned to the worktree `src`,
`CUDA_VISIBLE_DEVICES=` on every run, `GIT_DIR`/`GIT_WORK_TREE` override (same caveat as rounds 4–5).

**Why the repair agent's own probe was not sufficient.**
`logs/repair_round6_probe_rename.py` injects a payload factory into `_load_model` and, for the
basename check (E1), monkeypatches `validate_formal_report` to a pass-through. It never exercised
wrong payload-vs-filename on the full main path, never drove `scripts/76` as a CLI, never tried
mixed run identity through the real schema, and never asked whether fabricated trainer stamps with
identical weights still select. My probes close those gaps.

---

## 1. Rename / re-save attack (missing payload step) — **PASS**

```text
C1  PASS rename attack (no payload step): 0/5 reports
C1a PASS all five refusals cite payload['step']
C1b PASS no n2_gate.json leaked from refused runs: []
C2  PASS wrong payload step: only the self-consistent 250 survives; emitted_steps=[250]
C2a PASS 4/5 refused with payload-vs-filename mismatch
```

Unit-level matrix (`A1`–`A4`) confirms `require=True` refuses filename-only, CLI invent, and
payload/filename mismatch; `require=False` still allows filename derive (`A5`, documented).

## 2. Payload hashes fail-closed — **PASS**

```text
B1–B5 PASS missing / incomplete / placeholder hashes refused
B6    PASS eval_contract mismatch refused
B7    PASS legit hashes accepted (not a blanket refusal)
C3–C5 PASS main() path: missing / foreign / placeholder → 0 reports or refuse
F5    PASS schema refuses report without checkpoint.hashes
F3–F4 PASS selector refuses mixed / 1-of-5 foreign run identity
K2–K3 PASS production CLI refuses mixed identity and missing hashes
```

## 3. Legitimate five-candidate set — **PASS**

```text
D1  PASS legit main() stamped steps=[250, 375, 500, 625, 750]
D1a PASS five legit candidate files are byte-distinct
D2  PASS selector ACCEPTS legit set -> step=250
D3  PASS 5 distinct checkpoint sha256
K0–K1 PASS CLI accepts the legit set: rc=0 wrote=True step=250
```

## 4. Basename rule is load-bearing — **PASS**

Round-5 Y3 hole is closed:

```text
M1  PASS MUTATION drop basename rule -> caught by
        test_main_stamps_checkpoint_step_and_selector_e2e,
        test_selector_refuses_checkpoint_basename_mismatch
M1a PASS dedicated test is the guard's owner (caught=True)
E1  PASS live selector refused best.pt alias
K4  PASS CLI refuses best.pt basename alias
```

Note (`E0`): the schema still accepts a `best.pt` alias when the path does not match
`candidate_step_*.pt` (no step digit to cross-check); the **selector** is the enforcing layer.
That layering is real and CLI-confirmed.

## 5. WAVE1 digest sync — **PASS**

Independent reseal:

```text
R2a–R2b PASS live contract sha == c9fd265ac715b3d6 == REPAIR_ROUND6 claim
R2e     PASS n_locked=19 mismatches=[]
R2f     PASS manifest sha == 85b896a907769af4
R2c2/3  PASS round-4 and round-5 contract digests superseded
R2f2/3  PASS round-4 and round-5 manifest digests superseded
R2g     PASS scripts/74, scripts/76, discrete_report_schema.py inside locked manifest
R2h-w1/w2 PASS both recipes bind to live digests
R2i/j   PASS v2.0.1 parent pointer on live digest; no r4/r5 leftovers
H-WAVE1_REPORT.md           PASS live digests (worktree)
H-WAVE1_REPORT.md_authoring PASS live digests (C:\projects\tts-self mirror)
H-d1c_reliability_gate.json PASS
```

## 6. No critical regression — **PASS**

| Prior | Result |
|---|---|
| Round-4 CLI invent (`--checkpoint-step` on unbound file) | C6 PASS — 0/5 |
| Distinct `checkpoint.sha256` | G2 PASS; CLI K5 PASS |
| Forged / unreachable path | H1–H2 PASS |
| FAIL #2 binding | R2* PASS |
| FAIL #7 resume | R7* PASS (r5 digest refused as R7b2) |
| Residual grads `cb0_w=0, eos_w=0` | RXa PASS 45/45 |
| Immediate EOS | RXb PASS `codes.shape=(1,0,16)` |
| select8 = 2 RUSLAN + 6 CV; disjoint gate24 | RXc–RXd PASS |
| D1c subset aliases | RXe PASS |
| Real `_load_model` stamps step+hashes | F0–F1 PASS |

## 7. Pytest attestation — **VALID**

| Check | Result |
|---|---|
| Claimed | `406 passed, 3 skipped`, 409 collected |
| Independent reproduction | `406 passed, 3 skipped, 2 warnings in 54.53s` — **exact match** |
| Collect-only | **409** |
| Arithmetic | round-5 407 collected → round-6 409; +2 matches claimed new tests |
| New tests collected | `test_require_checkpoint_payload_hashes_fail_closed`, `test_selector_refuses_checkpoint_basename_mismatch` |
| Skip identities | unchanged (opt-in gloo DDP ×2, CUDA required ×1) |
| GPU | masked |
| GIT_DIR | worktree override required under `/mnt/c` (disclosed) |

Log: `logs/reaudit6_full_pytest.log`.

## 8. NEW residual — fabricated trainer stamps (`I1`) — **MEDIUM**

```text
I1 INFO RESIDUAL: five re-saves of ONE weight state with fabricated payload steps
       ACCEPTED (selected step=250, distinct_weight_digests=1/5)
K7 INFO RESIDUAL at production CLI: … -> rc=0 wrote=True
```

Mechanism: write five `candidate_step_XXXX.pt` files that share identical `model` weights but carry
distinct fabricated `payload['step']` values and a **shared** non-placeholder `payload['hashes']`
block matching the live contract. Evaluator accepts (stamps present and consistent); schema
accepts; selector accepts (distinct file sha from junk padding, shared run fingerprint, basename
OK).

This is **not** the round-5 rename attack (which had no payload step). It requires deliberately
forging trainer-stamped fields. Closing it would need something the current design does not have:
a content commitment from the training run (e.g. trainer-signed step ledger, or hashing weights
into a run-manifest entry the selector checks). Documented for the next repair round; **not**
verdict-blocking under the stated round-6 rules.

Related observation (`J1`): selector does not cross-check `checkpoint.hashes.eval_contract`
against `provenance.eval_contract_sha256`. A consistent report-level forgery of both fields still
passes. LOW — the evaluator already verifies payload hashes against the live contract at emit time;
this is report-hand-edit territory after the fact.

## 9. Mutation coverage — critical guards covered; layered holes remain

```text
M1/M1a PASS basename dedicated test is load-bearing
M5/M5a PASS payload['step'] require is load-bearing
M6/M6a PASS require_checkpoint_payload_hashes is load-bearing
M4     PASS duplicate-sha (round-5) still load-bearing
M2 FAIL selector missing-hashes guard → caught by NOTHING
M3 FAIL selector mixed-identity guard → caught by NOTHING
M7 FAIL emitter synthetic-hashes refuse → caught by NOTHING
M8 FAIL schema hashes require → caught by NOTHING
M9 FAIL schema cross-wired basename → caught by NOTHING
```

`M2`/`M3`/`M7`/`M8`/`M9` are **coverage holes**, not live defects — attack/CLI probes prove the
shipped guards work. Same class as round-5's Y3 (now fixed). **[LOW]** Add dedicated selector /
schema / emitter negatives so these five mutations go red.

`M10` INFO: five test sites still monkeypatch `validate_formal_report` to a pass-through; acceptable
because `test_main_stamps_checkpoint_step_and_selector_e2e` and this audit's probes run the real
schema.

## 10. Remaining defects, in fix order

1. **[MEDIUM — NEW] Bind weight identity to training step**, not only stamped fields. Options:
   trainer writes a per-step content digest into `payload['hashes']` / run manifest; selector
   refuses five candidates whose model-state digests collide. (`I1`, `K7`)
2. **[LOW — NEW] Cover the five layered hash/basename mutations** (`M2`/`M3`/`M7`/`M8`/`M9`).
3. **[LOW] Cross-check `checkpoint.hashes.*` vs `provenance.*_sha256`** in the selector. (`J1`)
4. **[LOW] `--promote-best` D1c branch lacks symlink guard** (pre-existing; `RY2` still `no`).
5. **[LOW]** Redundant full-file hashing across schema/selector/emit (deferred).
6. **[LOW]** Weight EOS accuracy aggregate by utterance count (deferred).
7. **[LOW]** Pin `verify_recipe_source_manifest` to repo root (deferred).
8. **[LOW]** Assert rather than overwrite `recipe_hash` in `scripts/72` (deferred).
9. **[LOW]** Non-D1c resume fail-open (deferred).
10. **[LOW]** CER margin tolerance at selector (deferred).

Standing non-code blockers (unchanged): no clean committed D1c SHA; no confirmation-seed
commitment; Wave 3/4 NO-GO pending human sign-off.

## Scope confirmations and post-audit integrity

- No commit / push / branch mutation. `HEAD` = `0661d10eb616…` before and after.
- No GPU; `CUDA_VISIBLE_DEVICES=` on every run. No training, decode, or model synthesis → **no
  listen-demo obligation**.
- No seed invented; no methodology edit; no Wave 0–2 PASS claim.
- Probe scratch under `/tmp` and `.cursor/results/.../logs/` only.
- Post-probe source digests: `logs/reaudit6_post_probe_hashes.json`

```text
8a94899ce9637691  scripts/74_eval_discrete_lm.py
b2513ec49b141bab  scripts/76_select_discrete_checkpoint.py
37fac4058b99eae8  src/minispeech/eval/discrete_report_schema.py
7d85475e6ab3db19  src/minispeech/eval/discrete_n2.py
8b665b8c9faf8304  src/minispeech/train/discrete_lm_trainer.py
3601418703b51e56  tests/test_d1c_infra_repair.py
b990bcec076fef5e  configs/eval_contracts/d1c_semantic_eos_v2_0_0.json
0e0b31797b9479fd  configs/eval_contracts/d1c_semantic_eos_v2_0_1.json
9475be54a60521b6  configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml
786225e5b40741dd  configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml
```

- Frozen manifest still verifies after probes: `n_locked=19`, `mismatches=[]`, digests
  `c9fd265a…` / `85b896a9…`.

## Report paths

- Worktree: `.cursor/results/d1c_semantic_eos_repair_b81ecc3e/REAUDIT_ROUND6.md`
- Authoring mirror: `C:\projects\tts-self\.cursor\results\d1c_semantic_eos_repair_b81ecc3e\REAUDIT_ROUND6.md`
