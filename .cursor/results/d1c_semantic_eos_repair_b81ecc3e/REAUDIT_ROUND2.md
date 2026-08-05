# REAUDIT_ROUND2 — independent adversarial re-verification of the D1c INFRA repair (round 2)

**Plan:** `.cursor/plans/d1c_semantic_eos_repair_b81ecc3e.plan.md`
**Slug:** `d1c_semantic_eos_repair_b81ecc3e`
**Worktree audited:** `C:\projects\tts-self-d1c`, branch `d1c/semantic-eos`
**Branch tip:** `0661d10eb616362e74dc1750e03cf25c33a7943e` (baseline `origin/master`; the branch still
carries **no commits of its own** — all D1c work remains uncommitted working-tree state)
**Input under audit:** `REPAIR_ROUND2.md` (claims FAILs #2/#4/#7 FIXED + four gaps closed,
fresh pytest `397 passed, 2 skipped`)
**Prior audit:** `REAUDIT.md` (verdict STILL_INVALID_RUN, 3 of 9 blockers live)
**Auditor:** independent re-audit agent. Every claim in `REPAIR_ROUND2.md` was treated as an
unverified assertion and re-tested with live probes.
**Date:** 2026-08-04
**Constraints honoured:** no commit, no push, no GPU work, no seed invented, no methodology edit,
no training/inference/synthesis (no new `best.pt`/`last.pt`, so no listen-demo obligation arises).
The one exception is disclosed in the pytest section: an early full-suite run inherited a visible
GPU and executed the tiny `test_bf16_forward_cuda` unit forward. All later runs pin
`CUDA_VISIBLE_DEVICES=`.

## Overall verdict

> ## STILL_INVALID_RUN
>
> FAIL #2 and FAIL #7 are genuinely fixed and I confirmed them with live probes.
> **FAIL #4 is not fixed end-to-end.** The v2 report emitter itself now works, but the
> evaluator that must feed it still discards the teacher-forced EOS metrics it measures, so
> **both** `_emit_d1c_v2_from_gate` call sites raise an unhandled `ValueError` on a real D1c
> eval run. The evaluator→selector formal interface (plan §8.4) has still never executed
> end-to-end.
>
> Waves 0–2 remain **not PASS**. Wave 3 remains **NO-GO**.

The round-2 repair is substantially real: the contract/recipe/manifest seal is now mutually
consistent and fails closed, all eight resume hashes resolve and are enforced for D1c, the §11
selector predicate is implemented and I could not defeat it, `scripts/72` has working D1c
guards, and the pytest attestation reproduces. But the single blocking defect is the
same *class* of defect that round 1 was rejected for: the emitter is fixed in isolation and
tested in isolation, while the production call path was never exercised.

## Per-item verdicts

| Item | Round-2 claim | This audit | Evidence |
|---|---|---|---|
| **FAIL #2** — contract/recipe binding | FIXED | **PASS** | `logs/reaudit2_probe_binding2.log`, `logs/reaudit2_probe_72cli*.log` |
| **FAIL #4** — v2 emitter E2E | FIXED | **FAIL** | `logs/reaudit2_probe_realpath.log` |
| **FAIL #7** — resume hashes | FIXED | **PASS** | `logs/reaudit2_probe_binding2.log` |
| Gap — selector §11 eligibility | Fixed | **PASS** (3 terms structurally vacuous) | `logs/reaudit2_probe_selector.log` |
| Gap — source manifest coverage | Fixed | **PASS** (never verified at runtime) | `logs/reaudit2_probe_binding2.log` |
| Gap — `72` D1c awareness | Fixed | **PASS** (no D1c subset alias) | `logs/reaudit2_probe_72cli2.log` |
| Gap — pytest attestation header/GIT_DIR | Fixed | **PASS** | `logs/reaudit2_full_pytest_gitenv.log`, `logs/reaudit2_full_pytest_noenv.log` |
| Prior PASSes 1/3/5/6/8/9 | — | **all still hold** | `logs/reaudit2_probe_regression.log` |

## Method

Everything below was executed, not read. Probe scripts and their logs are stored alongside this
report.

| Artifact | What it does |
|---|---|
| `logs/reaudit2_probe_binding.py` / `.log` | first binding pass (contains two probe-authoring artifacts, superseded by `binding2`) |
| `logs/reaudit2_probe_binding2.py` / `.log` | corrected FAIL #2 + #7: independent contract recompute, `enforce_eval_contract` tamper cases, 37 resume fail-closed cases |
| `logs/reaudit2_probe_72cli.sh` / `.log` | live `scripts/72` runs: canonical, `--skip-eval-contract`, no `--subset`, stale-hash injection, `--allow-dirty` |
| `logs/reaudit2_probe_72cli2.sh` / `.log` | same, against a synthetic `clean_sha_available=true` contract so the deeper D1c guards are reachable |
| `logs/reaudit2_probe_emitter.py` / `.log` | FAIL #4: emitter driven with evaluator-shaped rows, real wavs, real model TF measurement, plus artifact-refusal negatives |
| `logs/reaudit2_probe_selector.py` / `.log` | §11 eligibility against a **genuinely eligible** baseline (18 negative cases + 2 boundary cases) |
| `logs/reaudit2_probe_realpath.py` / `.log` | **decisive probe**: reproduces exactly what `scripts/74 main()` passes to the emitter |
| `logs/reaudit2_probe_regression.py` / `.log` | prior-PASS spot checks + post-probe manifest integrity |
| `logs/reaudit2_probe_docs.py` / `.log` | published numbers vs on-disk artifacts; confirmation-seed commitment |
| `logs/reaudit2_full_pytest_gitenv_gpu.log` | independent full-suite reproduction, `GIT_DIR` override, GPU visible — **exact match to the claim** |
| `logs/reaudit2_run_full_pytest_gitenv.sh` → `logs/reaudit2_full_pytest_gitenv.log` | same, GPU masked (confirming variant) |
| `logs/reaudit2_full_pytest_noenv.log` | full suite in the default environment (documents the GIT_DIR caveat) |

Test environment: WSL2 Ubuntu, venv `/home/adam/projects/tts-self/.venv`, Python 3.11.15,
rootdir `/mnt/c/projects/tts-self-d1c`. `pyproject.toml` sets `pythonpath = ["src"]`, and every
probe additionally pins `PYTHONPATH` to the **worktree** `src/minispeech`. Post-probe integrity
(`logs/reaudit2_probe_regression.log`):

```text
F6a PASS post-probe recipe_source_manifest mismatches=[] (n_locked=19)
```

`git status --porcelain -- src scripts configs tests` is byte-identical to the pre-audit
snapshot, so the probes mutated no attested source.

---

## FAIL #2 — contract / recipe / manifest binding — **PASS**

Everything the prior audit rejected now holds, and I verified it without using the repair
agent's own helpers where that was possible.

**Independent recompute.** I recomputed the contract content hash from the raw file myself,
applying the documented exclusion rule (`sha256`, `recipe_sha256`, `_`-prefixed keys):

```text
B1a PASS independent content recompute=32f0ea2b16fabb0a16086754d28e4f8986efa78735658de43769c172d13aac5c == declared sha256 (True)
B1g PASS tamper(infer.max_frames=999) changes content sha (True)
```

**Recipes are bound to the live contract.** Both W1 and W2 declare the live digests, and the
stale `a59ada9…` value appears nowhere in the repository outside the two audit narratives:

| Field | W1 | W2 |
|---|---|---|
| `train.eval_contract_hash` | `32f0ea2b…` (= live) | `32f0ea2b…` (= live) |
| `train.recipe_hash` | `32f0ea2b…` (= contract `recipe_sha256`) | same |
| `train.recipe_source_manifest_hash` | `cb697d4f…` | same |
| `train.run_manifest_hash` | `e0e7b95c…` | `f4a62dae…` (differs correctly — identity includes `run_name`/`eos_loss_weight`) |

```text
A2-w1-contract PASS w1 eval_contract_hash=32f0ea2b16fabb0a live=32f0ea2b16fabb0a
A2-w2-contract PASS w2 eval_contract_hash=32f0ea2b16fabb0a live=32f0ea2b16fabb0a
A3a PASS v2_0_1.parent_sha256=32f0ea2b16fabb0a live200=32f0ea2b16fabb0a
A3b PASS v2_0_1.parent_recipe_sha256=32f0ea2b16fabb0a
A4a PASS verify_recipe_source_manifest n_locked=19 mismatches=[]
A4b PASS manifest dict sha=cb697d4f35f6a250 declared=cb697d4f35f6a250
A5d PASS w1 declared run_manifest_hash reproducible from derived identity: declared=e0e7b95cf5367dd9 derived=e0e7b95cf5367dd9
```

The `run_manifest_hash` values are not arbitrary: dropping the field from the YAML and letting
`resolve_repro_hashes` derive it reproduces the declared value exactly.

**v2.0.1 parent pointer.** `configs/eval_contracts/d1c_semantic_eos_v2_0_1.json` is a
`status: not_materialized` stub carrying only `parent_contract` / `parent_sha256` /
`parent_recipe_sha256`, all repointed at the live 2.0.0 digests. It has no `sha256` or
`recipe_sha256` of its own yet, which is correct for a contract that will only be minted after
the confirmation-seed reveal; plan §7's "must preserve `recipe_sha256` from 2.0.0" is carried by
`parent_recipe_sha256` plus the `note` field. Recorded as satisfied-at-this-stage rather than
as a defect.

**The cyclic-seal resolution is sound.** W1/W2 are hashed into `recipe_source_manifest` with the
six train binding-hash fields stripped (`d1b_oracles.py:54-98`). I verified this is a *narrow*
blindness, not a general one:

```text
A4e PASS stripped hash of pristine copy == manifest entry (91125b5abfed71c8)
A4f PASS stripped hash IS sensitive to train.seed change (36ced27b46d7ee58 != 91125b5abfed71c8)
A4g INFO stripped hash is BLIND to eval_contract_hash tamper: same=True (by design; 72 must catch it)
A4h PASS stripped hash catches subset_hash tamper: True
```

**It fails closed — proven live, not statically.** I ran `scripts/72_train_discrete_lm.py`
end-to-end for five cases (`logs/reaudit2_probe_72cli.log`, `logs/reaudit2_probe_72cli2.log`):

| Case | Result |
|---|---|
| stale `eval_contract_hash` injected (temp copy) | `RuntimeError: eval_contract hash mismatch: expected=a59ada9c… got=32f0ea2b…` |
| stale `eval_contract_hash` = `a…a` against a clean-SHA contract | `RuntimeError: eval_contract hash mismatch: expected=aaaa… got=2ec7060e…` |
| `eval_contract_hash` removed entirely | `D1c recipe must declare train.eval_contract_hash; refusing silent overwrite` |
| declared hash correct | proceeds past provenance, stops only at the missing codec-token dir |

The round-1 silent overwrite (`tcfg["eval_contract_hash"] = contract.get("sha256") or …`) is
gone; `scripts/72:476-482` now asserts equality before stamping.

### Residual (low) — `recipe_sha256` is declared but not bound

`eval_contract_content_sha256` (`d1b_oracles.py:250-256`) excludes `recipe_sha256` from the
content hash, so that field can be edited without invalidating the contract `sha256`:

```text
B1f INFO contract.recipe_sha256 is EXCLUDED from sha256 -> tampering it keeps sha valid (same=True)
```

and `scripts/72:483-488` still *silently overwrites* `tcfg["recipe_hash"]` and
`tcfg["recipe_source_manifest_hash"]` from the contract instead of asserting equality with the
declared values — the exact pattern that was fixed for `eval_contract_hash`. The practical risk
is contained (`recipe_source_manifest_sha256` **is** inside the content hash, and the stamped
value always converges to the contract), but a recipe YAML can publish a wrong `recipe_hash`
that no gate reports.

---

## FAIL #4 — v2 emitter end-to-end — **FAIL**

### What is genuinely fixed

Fed rows shaped exactly as `free_running_asr_eval` builds them (`scripts/74:484-503`), with real
`.wav` files on disk and a teacher-forced measurement produced by a real semantic-EOS model
forward, the emitter works and every fail-open the prior audit found is closed
(`logs/reaudit2_probe_emitter.log`, 38 of 40 checks PASS; the 2 non-PASS entries were probe
artifacts, re-run correctly in `logs/reaudit2_probe_selector.log`):

```text
D1b PASS teacher_forced_ce returns MEASURED EOS metrics (acc=0.0, fp=0)
D1c PASS untrained tiny model does NOT report a perfect 1.0/0 EOS result
D2a PASS _emit_d1c_v2_from_gate SUCCEEDED on evaluator-shaped rows -> discrete_eval_report_v2.json
D2b PASS ordinal stamped 0..n-1 on every emitted row
D2c PASS catastrophic stamped on every emitted row
D2d PASS pcm_sha256 != file_sha256 for all rows (offenders=[])
D2e PASS pcm_sha256 independently reproduces from float32 samples
D2f PASS file_sha256 independently reproduces from container bytes
D2h PASS emitted EOS TF accuracy equals the MEASURED value (not a 1.0 default)
D3a PASS emitter refuses unmeasured EOS TF metrics
D3b PASS emitter refuses decode_bundle provenance
D3c PASS caller-forced formal_eligible=True still refused
D4a PASS missing wav refused
D4b PASS placeholder wav hashes refused
D4c PASS pcm_sha256==file_sha256 shortcut is recomputed from real PCM
D4d PASS placeholder generated_codes_sha256 refused
```

The `ordinal must equal 0` crash is gone, `formal_eligible` is derived rather than hardcoded,
the 1.0/0 EOS defaults are gone, and PCM is hashed from decoded float32 samples. `D1c` is the
decisive control: an untrained model reports accuracy `0.0`, which a hardcoded default could not
produce.

### The blocking defect: the evaluator throws the measurements away before the emitter sees them

`teacher_forced_ce()` was repaired to return the EOS metrics, but `main()` never keeps them.
`logs/reaudit2_probe_realpath.log`:

```text
H2a INFO teacher_forced_ce() per-batch keys = ['ce_cb0', 'ce_residual', 'ce_semantic_eos', 'loss',
         'teacher_forced_ce_source', 'tf_eos_early_top1_false_positive_count', 'tf_eos_final_top1_accuracy']
H1a INFO scripts/74:1335 -> totals = {"loss": 0.0, "ce_cb0": 0.0, "ce_residual": 0.0}
H2c FAIL aggregated summary retains the measured EOS TF metrics
         (dropped keys: ['ce_semantic_eos', 'teacher_forced_ce_source',
                         'tf_eos_early_top1_false_positive_count', 'tf_eos_final_top1_accuracy'])
H1d INFO scripts/74:1426 -> tf_ce = summary.get("loss")
H2d INFO tf_ce passed to the emitter = 10.645967483520508 (type=float)
```

The aggregation loop at `scripts/74:1335-1342` initialises `totals` with only
`{loss, ce_cb0, ce_residual}` and sums only those keys, so the four newly-returned fields are
discarded every batch. `scripts/74:1426` then reduces the summary to a **bare float**. That
float is what both call sites hand to the emitter (`scripts/74:1045` and `scripts/74:1564`).

Driving the real shapes through the real emitter:

```text
H3a FAIL MAIN eval path (tf_ce=summary['loss'] float) emits a v2 report ->
         ValueError: formal D1c report refused: formal_eligible=False
                     (teacher_forced_ce_source='unavailable', eos_measured=False)
H3b FAIL --score-only path (bundle tf float + score_only=True) emits a v2 report ->
         ValueError: formal D1c report refused: formal_eligible=False
                     (teacher_forced_ce_source='decode_bundle', eos_measured=False)
H3c PASS CONTROL: same rows + full measured TF dict emits fine -> OK
```

`H3c` isolates the defect precisely: identical rows, identical contract, identical checkpoint —
the only difference is whether the teacher-forced metrics survived the plumbing.

**Neither call site is guarded.** `scripts/74:1039` sits at four-space indentation, outside the
`try:` at 1035-1038 that guards only `load_eval_contract`; `scripts/74:1558` has no `try` at all
(the automated check `H4-1039` reported "guarded" on a crude text heuristic — I confirmed by
reading the source that it is **not** guarded):

```python
# scripts/74_eval_discrete_lm.py:1033-1039
    contract_for_emit = None
    if eval_contract and not skip_eval_contract:
        try:
            contract_for_emit = load_eval_contract(eval_contract)
        except Exception:
            contract_for_emit = None
    _emit_d1c_v2_from_gate(          # <- 4-space indent: OUTSIDE the try above
```

So the first real D1c eval run aborts with a traceback **after** writing `n2_gate.json`,
`n2_model_rows.json` and `n2_catastrophic_rows.json`, and produces no
`discrete_eval_report_v2.json`. `--score-only` fails the same way for a second reason: it stamps
`teacher_forced_ce_source="decode_bundle"` (`scripts/74:848-850`) while
`validate_formal_report` requires `eval_computed` (`discrete_report_schema.py:140-141`), so that
path can never yield a formal report even once the plumbing is repaired.

**Why the round-2 test suite did not catch this.**
`test_emit_d1c_v2_from_evaluator_shaped_rows` (`tests/test_d1c_infra_repair.py:454-542`) hands
the emitter a hand-written dict that already contains `tf_eos_final_top1_accuracy: 1.0`,
`tf_eos_early_top1_false_positive_count: 0` and `teacher_forced_ce_source: "eval_computed"` —
i.e. it tests the emitter with the very inputs the evaluator fails to produce. No test drives
`teacher_forced_ce` → aggregation → `tf_ce` → emitter as one chain.

### Fix required

1. Extend `totals` at `scripts/74:1335` to carry `ce_semantic_eos`, and aggregate
   `tf_eos_final_top1_accuracy` (weighted mean or min) and
   `tf_eos_early_top1_false_positive_count` (sum) across batches.
2. Replace `tf_ce = summary.get("loss")` (`scripts/74:1426`) with the full metrics dict,
   including `teacher_forced_ce_source: "eval_computed"` and `total_ce`.
3. Make the decode-only bundle carry the same dict so `--score-only` can be made formal, or
   state explicitly that `--score-only` is non-formal for D1c and skip the emit instead of
   crashing.
4. Add a regression test that runs the real chain (a tiny model through the aggregation into the
   emitter). The existing test class cannot catch this defect.

---

## FAIL #7 — resume hash checks — **PASS**

`resolve_repro_hashes` now always attempts the contract load for backfill and verifies the
declared hash, so nothing is inert for the canonical configs
(`logs/reaudit2_probe_binding2.log`):

```text
A5-w1 PASS w1 resolve_repro_hashes inert(None) keys=[]
A5-w1-vals INFO {"config": "8b3597866ed41a6c", "eval_contract": "32f0ea2b16fabb0a",
  "model_config": "0fb389c327c5301a", "recipe": "32f0ea2b16fabb0a",
  "recipe_source_manifest": "cb697d4f35f6a250", "run_manifest": "e0e7b95cf5367dd9",
  "speaker_map": "64e7cdc5c3447a43", "subset": "501330481c07d928"}
A5-w2 PASS w2 resolve_repro_hashes inert(None) keys=[]
A5c PASS stale eval_contract_hash raises: eval_contract_hash mismatch: declared=a59ada9c…
```

All eight keys are enforced, in all three failure directions (37 live cases):

```text
B3a PASS happy-path D1c resume accepted (err=None)
B3b-{model_config,speaker_map,subset,eval_contract,recipe,recipe_source_manifest,run_manifest,config}
     PASS current <key>=None refused -> "resume fail-closed: D1c requires <key> but current repro hash is missing (None)"
B3c-<all eight> PASS ckpt <key> mismatch refused
B3d-{recipe,recipe_source_manifest,run_manifest} PASS ckpt missing <key> refused
```

The round-1 fail-open is closed. Dropping a hash from the YAML no longer disables its check,
because the value is backfilled from the contract or derived:

```text
B3f-run_manifest_hash            PASS dropping train.run_manifest_hash still resolves all keys (None=[])
B3f-recipe_hash                  PASS
B3f-recipe_source_manifest_hash  PASS
B3f-eval_contract_hash           PASS
```

And when backfill genuinely cannot happen, it fails closed rather than skipping:

```text
B3g PASS missing contract -> unresolvable keys ['recipe', 'recipe_source_manifest'];
        D1c resume refuses: "resume fail-closed: D1c requires recipe but current repro hash is missing"
```

### Residual (low) — non-D1c path unchanged

`discrete_lm_trainer.py:784-792` still `continue`s past a `None` current value when
`termination.mode != "semantic_eos_v1"`:

```text
B3e INFO non-D1c frame_stop_v1 with run_manifest=None -> SKIPPED (fail-open)
```

Out of scope for FAIL #7 (which is D1c-scoped) but worth closing before the same trainer is
reused for a non-semantic run.

---

## Gap — selector §11 eligibility — **PASS**

The prior audit's four selector escapes are closed. Probe D's first attempt produced *vacuous*
refusals because its baseline carried `tf_eos_final_top1_accuracy=0.0` from an untrained model,
so I rebuilt the baseline to satisfy §11 in full and re-ran; every negative case is now
attributable to the single field it changes (`logs/reaudit2_probe_selector.log`):

```text
E1  PASS §11-clean baseline IS selectable -> SELECTED step=250
E2  PASS early_eos_suppressed_count=7                      -> REFUSED
E3  PASS max_cb0_run=99                                    -> REFUSED
E4  PASS max_cb0_run=13 (boundary >12)                     -> REFUSED
E5  PASS zero-frame generation                             -> REFUSED
E6  PASS per-row duration ratio 9.5x                       -> REFUSED
E7  PASS mean duration ratio 1.43                          -> REFUSED
E8  PASS catastrophic row                                  -> REFUSED
E9  PASS termination_reason=safety_max_frames              -> REFUSED
E10 PASS tf_eos_final_top1_accuracy=0.9                    -> REFUSED
E11 PASS tf_eos_early_top1_false_positive_count=3          -> REFUSED
E12 PASS tf_eos metrics = 'unavailable'                    -> REFUSED
E13 PASS model CER 0.40 vs copy 0.05                       -> REFUSED
E14 PASS model CER 0.09 vs copy 0.05 (+0.04 > +0.03)       -> REFUSED
E15 PASS text_usage.pass=False                             -> REFUSED
E16 PASS phrase_loop_repetition=True                       -> REFUSED
E17 PASS runaway_or_repetition=True                        -> REFUSED
E18 PASS formal_eligible=False                             -> REFUSED (schema)
E19 PASS teacher_forced_ce_source=decode_bundle            -> REFUSED (schema)
E20 PASS max_cb0_run=12 (at the allowed bound)             -> SELECTED
```

All reports were consistently re-hashed before each run, so no refusal can be attributed to
tamper detection — and tamper detection itself is still live (`D5n`: stale `rows_sha256`
refused). The `E21` boundary case (model CER exactly copy + 0.03) refuses because
`0.08 - 0.05 = 0.030000000000000006` in binary floating point; that is conservative, not a
defect, but the comparison should use a tolerance if the plan intends the bound to be inclusive.

### Residual (medium) — three §11 terms are structurally vacuous

The selector *can* enforce text-usage, phrase-loop and runaway, but the emitter never writes
those fields, so the terms never fire on a real report
(`logs/reaudit2_probe_regression.log`):

```text
F7a INFO emitted report gates keys=['criteria', 'eligible', 'verdict']
F7b INFO emitted report summaries keys=['max_abs_log_duration_error', 'mean_abs_log_duration_error',
                                        'model_macro_cer', 'n_catastrophic']
F7c INFO §11 terms never populated by the emitter -> always vacuous:
         gates.text_usage missing=True, summaries.phrase_loop_repetition missing=True,
         gates/summaries.runaway_or_repetition missing=True
```

`scripts/76:144-146` deliberately treats an absent `text_usage` block as a pass. Since
`_emit_d1c_v2_from_gate` (`scripts/74:901-916`) never populates it, plan §11's text-usage
requirement is currently unenforceable end-to-end. `E23` confirms an absent block still selects.

---

## Gap — frozen source manifest — **PASS**

The locked set grew from 14 to 19 and every declared digest matches on disk:

```text
A4a PASS verify_recipe_source_manifest n_locked=19 mismatches=[]
A4c PASS previously-unlocked modules now locked: missing=[]
```

The five modules the prior audit named — `discrete_oracles.py`, `d1b_oracles.py`,
`eval/asr.py`, `codec_seq_protocol.py`, `constants.py` — are all present.

### Residual (medium) — the manifest is never verified at runtime

```text
B4a INFO runtime callers of verify_recipe_source_manifest outside its own module: NONE
```

`enforce_eval_contract` checks the contract hash, speaker map, infer knobs, subsets and
`clean_sha_available`, but never calls `verify_recipe_source_manifest`. The only callers are
`tests/test_d1c_infra_repair.py:428-440` and the repair agent's own verify script. A source file
can therefore be mutated between the seal and the run without any fail-closed check firing.
`scripts/72` should verify the manifest on the D1c formal path.

---

## Gap — `scripts/72` D1c awareness — **PASS**

All four guards fire on live runs, and the D1c branch is reached both by config filename and by
`model.termination.mode == "semantic_eos_v1"` (`scripts/72:436-441`):

| Guard | Live result |
|---|---|
| `--skip-eval-contract` forbidden | `D1c formal path forbids --skip-eval-contract; eval-contract enforcement is mandatory` |
| silent first-N forbidden | `is_formal_cfg and utt_ids is None` → refuses; W1/W2 resolve `train.subset` from the recipe, so a canonical run binds the frozen oracle rather than first-N |
| `require_clean_sha` | without `--allow-dirty`: `eval_contract.repro.clean_sha_available is false` → refuses |
| `--allow-dirty` warning | `WARNING: --allow-dirty set with D1c config; canonical gates forbid MINISPEECH_ALLOW_DIRTY=1` |

Note that `contract.repro` has **no** `clean_sha_available` key at all; the check treats absent
as falsy, so a canonical D1c run currently cannot start without `--allow-dirty`. That is the
correct posture while no clean committed SHA exists.

### New defect (medium) — there is no D1c subset alias

`SUBSET_ALIASES` (`d1b_oracles.py:24-36`) has no D1c entry, so every documented alias resolves to
a **D1b** oracle (`logs/reaudit2_probe_72cli2.log`):

```text
alias 'dev32'   -> configs/train_subsets/d1b_dev32.json        items_sha256=c886ba31… n=32
alias 'select8' -> configs/train_subsets/d1b_confirm32_v2.json items_sha256=6667da2f… n=8
alias 'gate24'  -> configs/train_subsets/d1b_confirm32_v2.json items_sha256=68f8d987… n=24
alias 'full32'  -> configs/train_subsets/d1b_confirm32_v2.json items_sha256=63d7e351… n=32
d1c aliases present: NONE
```

The natural operator command `--subset dev32` against a D1c recipe therefore loads the wrong
corpus. It **does** fail closed (`subset dev32 items_sha256 mismatch: got=c886ba31… contract=50133048…`),
so this is a foot-gun rather than a silent-corruption path, but the only working invocations are
`--subset configs/train_subsets/d1c_dev32.json` or omitting `--subset` entirely. Add
`d1c_dev32` / `d1c_select8` / `d1c_gate24` aliases before writing the run instructions.

---

## New defect (high) — the results pack publishes stale DEV32 provenance

`REPAIR_ROUND2.md` corrected the contract block of `WAVE1_REPORT.md` (lines 41-43 now carry the
live `32f0ea2b…` / `cb697d4f…`) but left the DEV32 block at pre-refreeze values. Three of the
four hashes in that table do not match the on-disk artifact
(`logs/reaudit2_probe_docs.log`):

| `WAVE1_REPORT.md` | Published | Live |
|---|---|---|
| line 28 `items_sha256` / full32 | `501330481c07d928…` | `501330481c07d928…` ✓ |
| line 29 `file_sha256` | `4f57a222f4738464…` | `f9f82b708b23ab61…` ✗ |
| line 30 `select8` | `e32d70aebe34284e…` | `882691c9fa7e799b…` ✗ |
| line 31 `gate24` | `832ce5fafe2cf9ee…` | `5485d37be12342d1…` ✗ |

The same three stale values appear in `artifacts/oracle_overlap_report.json:7,8,12`. They come
from the original freeze log (`logs/w1_freeze_dev32.log:33-36`), which predates the blocker-3
repair that changed the select8/gate24 composition. `RESULTS.md:31-33`, `REPAIR_AUDIT.md:28-31`,
`artifacts/repair_oracle_contract_hashes.json` and `d1c_reliability_gate.json` all carry the
**correct** live values, so the pack contradicts itself.

This matters for the human audit specifically: a reviewer recomputing `d1c_dev32.json` from
`WAVE1_REPORT.md` gets three mismatches and cannot distinguish "stale document" from "tampered
artifact". It also violates the workspace rule that published numbers must cite on-disk
artifacts. Fix `WAVE1_REPORT.md:29-31` and regenerate `artifacts/oracle_overlap_report.json`.

For completeness, the raw byte digest of `configs/train_subsets/d1c_dev32.json` is
`88518b178f39185c…` while the declared/verified `file_sha256` is `f9f82b708b23ab61…` (a hash of
the canonically re-serialised body with the self-referential field removed). That redefinition
was already noted as a caveat in `REAUDIT.md §3` and is unchanged; it should be stated
explicitly wherever `file_sha256` is published.

---

## Prior-PASS spot checks — all still hold

`logs/reaudit2_probe_regression.log`, 18/18 checks PASS:

```text
F1a PASS residual params with live grads (cb0_w=0, eos_w=0): 45/45
F1b PASS loss['total'].requires_grad=True
F2a PASS generate() immediate EOS -> codes.shape=(1, 0, 16) lengths=[0]
F2b PASS lengths==[0] (no phantom frame)
F3-dev32/select8/gate24/full32 PASS formal load, formal_eligible=True, file_sha256 present
F3-comp PASS select8 composition = 2 RUSLAN + 6 CV
F3-part PASS select8 | gate24 == full32 and disjoint (|s|=8 |g|=24 |f|=32)
F4a PASS _sha256_file matches hashlib on a 3 MiB+ blob (chunk loop iterates)
F5a PASS contract self-check holds
F5b PASS contract infer keys=['max_cb0_repeat','max_frames','min_frames','repetition_penalty',
                              'temperature','top_k'] legacy_present=[]
F6a PASS post-probe recipe_source_manifest mismatches=[] (n_locked=19)
```

The `cb0_weight=0, eos_loss_weight=0` variant of the gradient test is the decisive one: all 45
residual parameters still receive non-zero gradients, which is only possible if the residual CE
term is genuinely in the backward graph. No regression.

---

## Pytest attestation — **VALID**

| Check | Result |
|---|---|
| Claimed result | `397 passed, 2 skipped, 2 warnings in 53.22s` (`repair_round2_full_pytest.log:425`) |
| Independent reproduction | `397 passed, 2 skipped, 2 warnings in 60.11s` — **exact match** (`logs/reaudit2_full_pytest_gitenv_gpu.log`, same `GIT_DIR` override, GPU visible, `git rev-parse HEAD = 0661d10e…`) |
| Confirming variant, GPU masked | `396 passed, 3 skipped, 2 warnings in 51.13s` (`logs/reaudit2_full_pytest_gitenv.log`), reconfirmed at 52.40s with `-rs` |
| Collection | **399 items in all three runs**; zero failures in all three |
| One-test delta between my two runs | fully explained (below) — **not** a discrepancy in the claim |
| Log header | now records date, rootdir, `GIT_DIR`, `GIT_WORK_TREE`, full and short git SHA, and the pytest platform/plugin banner — the round-1 caveat is fixed |
| Runner script archived | `logs/repair_round2_run_pytest.sh`; my runner is `logs/reaudit2_run_full_pytest_gitenv.sh` |
| Log mtime | `2026-08-04 09:48:01` (run started 09:47:08) |
| Newest attested source | `tests/test_d1c_infra_repair.py` at `09:46:12` — **no source post-dates the log** |
| Claimed evidence tests | all 11 named tests exist and are `PASSED` in the log |

**The one-test delta between my two runs.** The first reproduction matched the claim exactly. The
second masks the GPU (`CUDA_VISIBLE_DEVICES=`) to honour the no-GPU constraint, so the single
`@requires_cuda` test moves from passed to skipped:

```text
SKIPPED [1] tests/test_discrete_lm_resume.py:378: CPU/gloo two-process test is opt-in (MINISPEECH_DDP_GLOO=1)
SKIPPED [1] tests/test_discrete_lm_trainer_steps.py:275: opt-in WORLD_SIZE=1 DDP init smoke; set MINISPEECH_DDP_SMOKE=1
SKIPPED [1] tests/test_discrete_speech_lm.py:257: CUDA required
```

`tests/test_discrete_speech_lm.py:257` is `test_bf16_forward_cuda`, `PASSED` in the repair
agent's log (`repair_round2_full_pytest.log:159`) and in my GPU-visible run. So
397 + 2 = 396 + 3 = 399 collected, and the runs differ by exactly that one test and nothing else.
**The claim reproduces exactly when the environment matches.** The other two skips are the same
pre-existing DDP opt-ins the claim reports.

Standing caveat (unchanged, correctly disclosed): the result requires `GIT_DIR`/`GIT_WORK_TREE`
pointing at `/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c`. In the default environment
(`logs/reaudit2_full_pytest_noenv.log`):

```text
FAILED tests/test_mel_baseline.py::test_git_sha_and_run_manifest - AssertionError
1 failed, 396 passed, 2 skipped, 2 warnings in 21.00s
```

(that run was made before I pinned `CUDA_VISIBLE_DEVICES=`, hence 2 skips rather than 3; it
executed only the tiny `test_bf16_forward_cuda` unit forward, no training, decode or synthesis.)
This is an artifact of running a Windows worktree through `/mnt/c`; it would not occur in the
canonical WSL clone. The round-2 log now records the override explicitly, which resolves the
round-1 complaint.

**Attestation caveat:** a green suite is not evidence that the D1c formal path works. The suite
passes while the production evaluator path crashes (FAIL #4). The new tests are real and
non-tautological, but `test_selector_refuses_suppressed_empty_and_extreme_duration`
(`tests/test_d1c_infra_repair.py:655-659`) monkeypatches `validate_formal_report` away; my
probes exercised the unpatched validator instead.

---

## Critical remaining defects, in fix order

1. **[BLOCKER] Plumb the measured teacher-forced EOS metrics to the emitter.** Extend `totals`
   at `scripts/74:1335`, aggregate `tf_eos_final_top1_accuracy` /
   `tf_eos_early_top1_false_positive_count` / `ce_semantic_eos`, and replace
   `tf_ce = summary.get("loss")` (`scripts/74:1426`) with the full dict including
   `teacher_forced_ce_source`. Add a regression test covering the whole chain.
2. **[BLOCKER] Decide and implement the `--score-only` contract.** Either propagate a formal TF
   dict through the decode bundle, or make `_emit_d1c_v2_from_gate` return `None` for
   non-formal provenance instead of raising. Either way, guard both call sites
   (`scripts/74:1039`, `scripts/74:1558`) so a refusal is reported rather than a traceback.
3. **[HIGH] Correct the stale published DEV32 provenance** in `WAVE1_REPORT.md:29-31` and
   `artifacts/oracle_overlap_report.json:7,8,12`, and state the `file_sha256` redefinition
   explicitly wherever it is published.
4. **[MEDIUM] Populate the three vacuous §11 fields** — `gates.text_usage`,
   `summaries.phrase_loop_repetition`, `gates/summaries.runaway_or_repetition` — in
   `_emit_d1c_v2_from_gate`, and make the selector refuse an absent `text_usage` block rather
   than pass it.
5. **[MEDIUM] Verify `recipe_source_manifest` at runtime** on the D1c formal path in
   `scripts/72` / `enforce_eval_contract`.
6. **[MEDIUM] Add D1c subset aliases** (`d1c_dev32`, `d1c_select8`, `d1c_gate24`) so the
   documented `--subset` invocations bind the D1c oracle.
7. **[LOW] Assert rather than overwrite** `recipe_hash` / `recipe_source_manifest_hash` at
   `scripts/72:483-488`, and consider including `recipe_sha256` in the contract content hash.
8. **[LOW] Close the non-D1c resume fail-open** at `discrete_lm_trainer.py:784-792`.
9. **[LOW] Use a tolerance** for the `macro_cer <= copy_macro + 0.03` comparison at
   `scripts/76:133-135` if the bound is meant to be inclusive.

## Standing non-code blockers (unchanged, correctly disclosed)

1. No clean committed D1c source SHA. Branch `d1c/semantic-eos` is still at baseline `0661d10`
   with all D1c work uncommitted. `contract.repro` has no `clean_sha_available` key, so
   `require_clean_sha` refuses — correctly. Requires explicit user commit/push authorization.
2. External confirmation-seed commitment still absent and correctly **not** invented:
   `configs/train_subsets/d1c_confirm_seed_commitment.json` has `commitment: null`,
   `plaintext_seed_present: false`, and the contract records
   `confirm_seed_commitment.status = "awaiting_external_commitment"`.
3. Wave 3 GPU smoke and Wave 4 DEV remain NO-GO.

## Scope confirmations

- No commit created, no push, no branch mutation.
- No GPU work; no training, inference, decode, or synthesis. Every `scripts/72` probe aborted in
  the provenance/guard section or at the missing codec-token directory. No new `best.pt` /
  `last.pt` was produced, so no listen-demo obligation arises.
- No seed invented; no methodology file touched.
- Probe scratch files were written only under `/tmp` and
  `.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/`; the frozen source manifest still
  verifies with 0 mismatches after all probes.
- Waves 0–2 remain **not PASS**. Wave 3 remains **NO-GO**.
