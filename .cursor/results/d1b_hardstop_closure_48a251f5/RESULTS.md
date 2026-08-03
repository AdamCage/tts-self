# D1b hardstop closure — RESULTS

**Plan:** `.cursor/plans/d1b_hardstop_closure_48a251f5.plan.md`  
**Slug:** `d1b_hardstop_closure_48a251f5`  
**Date:** 2026-08-03  
**Clean SHA (normative):** `a05e4b758636bb21d1501781f330ba93057756dc` (`a05e4b7`)  
**Status:** **NO-GO for D1c start** (closure tooling complete); scientific D1b remains **HARD-STOP**  
**D1c / 1h / 5h / full:** **not started**

## Verdict

D1b scientific HARD-STOP stands (confirm_v2 seed43 clean FAIL). This plan closed remaining **formal/technical** fail-closed holes so the HARD-STOP is reproducible and cannot be accidentally re-opened by score-only / select / v1-PASS artifacts.

User verdict **NO-GO for D1c** is respected: no D1c training, no ladder scale-up, no methodology edits.

## Tasks

| Plan todo | Status |
|-----------|--------|
| freeze-hardstop | completed (seed43 clean FAIL basis; seed42 DIAGNOSTIC) |
| harden-fail-closed | completed (score-only subset/hash/infer/wavs; TF-CE attestation; speaker-map; label/hash contradiction) |
| restore-test-gate | completed (`latent_io.py` restored; atomic-save tests; `n2_qwen_latent_v01.yaml` in package) |
| reconcile-closure | completed (formal gate + V1_SUPERSEDED; plan todos marked completed) |
| verify-and-package | completed (focused + full pytest + real torchrun smoke + HASH_CHECK_OK on clean SHA) |

## Fail-closed fixes (code)

| Area | Change |
|------|--------|
| `scripts/74_eval_discrete_lm.py` | Score-only requires subset label+hash, full WAVE3c infer block, complete N2 reports; missing wavs fail-closed on canonical path |
| `src/minispeech/data/d1b_oracles.py` | Reject contradictory subset label vs hash; CLI `speaker_map_path` checked vs contract hash |
| `scripts/76_select_discrete_checkpoint.py` | Explicit `teacher_forced_ce_source` attestation; candidate `eval_contract_sha256` provenance required |
| `scripts/73_infer_discrete_lm.py` | Canonical CLI `--max-frames` refuse; speaker-map path passed into contract enforce |
| Verify harness | `.cursor/results/d1b_hardstop_closure_48a251f5/logs/hardstop_closure_verify.sh` — ignores superseded v1 PASS; asserts confirm_v2 seed43 FAIL + formal HARD-STOP; real `torchrun` smoke |

## Test evidence

Pre-commit (WSL Ubuntu, dirty sync for development only):

| Suite | Result | Log |
|-------|--------|-----|
| Focused D1b (8 files) | **96 passed, 2 skipped** | `logs/precommit_focused_full.log` |
| Real torchrun smoke | **TORCHRUN_SMOKE_OK** (nccl, world_size=1) | `logs/precommit_torchrun.log` |

Normative clean-SHA verify (WSL Ubuntu after `git reset --hard origin/master`, porcelain empty, **no** `MINISPEECH_ALLOW_DIRTY=1`):

| Suite | Result | Log |
|-------|--------|-----|
| Focused D1b | **96 passed, 2 skipped** | `logs/hardstop_closure_focused.log` |
| Full `pytest -q` | **362 passed, 2 skipped** | `logs/hardstop_closure_pytest.log` |
| `torchrun --standalone --nproc_per_node=1` | **TORCHRUN_SMOKE_OK** (nccl) | `logs/hardstop_closure_torchrun.log` |
| Hash / HARD-STOP provenance | **HASH_CHECK_OK** + **HARDSTOP_PROVENANCE_OK** | `logs/hardstop_closure_hashes.log` |
| Aggregate | **AGGREGATE_FAIL=0** | `logs/hardstop_closure_verify_master.log` |

Push: `origin/master` at `a05e4b7` (Groups 1–2). Methodology left uncommitted.

## Related results pack

Canonical scientific HARD-STOP pack (unchanged frozen confirm_v2 metrics):

- `.cursor/results/d1b_n2_reliability/RESULTS.md`
- `.cursor/results/d1b_n2_reliability/d1b_reliability_gate.json` → `HARD-STOP`
- `.cursor/results/d1b_n2_reliability/artifacts/V1_SUPERSEDED.md`
- `.cursor/results/d1b_n2_reliability/artifacts/COMMIT_PACKAGE.md`

## Methodology

Read-only during this implementation. **No edits** under `.cursor/methodology/**`.

## Boundary / follow-ups

- Next stage may be planned as **D1c termination-repair only** when the user authorizes a new plan — **not** started here.
- **1h / 5h / full** remain forbidden until a future N2 PASS under a new plan.
- Do not cite superseded `artifacts/confirm_seed*` PASS JSON as current status.
