# D1b hardstop closure — commit package

**Status:** commit/push done; clean-SHA verify green.  
**Do not** mix methodology or `configs/_tmp_*` into this package.  
**Never** use `MINISPEECH_ALLOW_DIRTY=1` for the normative verify.

## Verification snapshot (authoritative after clean pull)

| Item | Result |
|------|--------|
| Clean SHA | `a05e4b758636bb21d1501781f330ba93057756dc` (`a05e4b7`) |
| Tree | porcelain empty on WSL at verify start |
| Focused D1b suite | **96 passed, 2 skipped** |
| Full `pytest -q` | **362 passed, 2 skipped** |
| Real `torchrun --standalone --nproc_per_node=1` | **TORCHRUN_SMOKE_OK** |
| Frozen hashes / HARD-STOP provenance | **HASH_CHECK_OK** (v1 PASS ignored) |
| Formal gate | **HARD-STOP** (seed43 clean FAIL basis) |
| Push | `origin/master` updated |

## CRLF hazard

Prefer path-scoped `git add -- <paths>`; never `git add -A`. Confirm with `git diff -w --stat` before commit.

## Group 1 — hardstop fail-closed + test-gate restore

Required for full pytest (includes collate latent_seq):

- `configs/n2_qwen_latent_v01.yaml`  ← **Group 1 required**
- `src/minispeech/models/latent_io.py`
- `src/minispeech/data/d1b_oracles.py`
- `scripts/73_infer_discrete_lm.py`
- `scripts/74_eval_discrete_lm.py`
- `scripts/76_select_discrete_checkpoint.py`
- `tests/test_d1b_oracles.py`
- `tests/test_discrete_n2_protocol.py`
- `tests/test_discrete_lm_trainer_steps.py`

Suggested message:

```text
fix(d1b): fail-closed hardstop closure and restore latent_io test gate

Lock score-only subset/hash/infer/wavs, select TF-CE attestation +
contract provenance, speaker-map checks, and subset label/hash
contradictions; restore latent_io + n2_qwen config for full pytest.
```

## Group 2 — results / protocol / plans

- `.cursor/results/d1b_n2_reliability/`
- `.cursor/results/d1b_hardstop_closure_48a251f5/`
- `.cursor/plans/d1b_n2_reliability_7bd71c65.plan.md`
- `.cursor/plans/d1b_hardstop_closure_48a251f5.plan.md`

Suggested message:

```text
docs(d1b): publish HARD-STOP closure results and verify harness

Record seed43 clean formal HARD-STOP, fail-closed verify scripts,
torchrun smoke logs, and completed hardstop-closure plan todos.
```

## Group 3 — explicit excludes

- `.cursor/methodology/**`
- `configs/_tmp_*`
- Unrelated plans/results (`d1_discrete_speech_lm/`, `p1_*`, …)
- `runs/`, `out/`, `data/`, checkpoints, wavs, HF caches

## Normative verify (after push)

```bash
cd ~/projects/tts-self && git pull && git status --porcelain   # must be empty
# never: export MINISPEECH_ALLOW_DIRTY=1
bash .cursor/results/d1b_hardstop_closure_48a251f5/logs/hardstop_closure_verify.sh
```
