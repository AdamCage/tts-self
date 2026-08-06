# WAVE3_REPORT — CPU/full tests and 1-utterance GPU smoke

**Plan:** `.cursor/plans/d1c_semantic_eos_repair_b81ecc3e.plan.md`  
**Slug:** `d1c_semantic_eos_repair_b81ecc3e`  
**Date:** 2026-08-06  
**Worktree:** `C:\projects\tts-self-d1c` (`d1c/semantic-eos`)  
**WSL runtime:** `~/projects/tts-self`  
**GPU tip SHA:** `74d4caa535fb34e12a804f07a09308151c1f7a19` (`74d4caa`)  
**Commitment commit:** `c15ed40`  
**Smoke subset / runner commits:** `c71b453`, `a602593`, `74d4caa`

## Overall verdict: **PASS (Wave 3 complete)**

| Gate | Verdict | Evidence |
|------|---------|----------|
| Clean committed SHA for formal run | **PASS** | WSL clean @ `74d4caa` before train |
| Confirm-seed commitment non-null | **PASS** | `4c46d418caf9bcabc088864da5bf0b3de0475a9d9225f50383314a2c5e3bb3f2` |
| Focused + full pytest (prior CPU) | **PASS** | `408 passed, 3 skipped` — prior @ `27164af` |
| `torchrun` world-size-1 smoke | **PASS** | `TORCHRUN_SMOKE_OK backend=nccl` — `logs/wave3_torchrun_smoke.log` |
| 1-utt EOS overfit | **PASS** | `runs/d1c_eos_smoke_1utt/` 80 steps; loss `22.9 → 0.000565` |
| FR learned EOS + no EOS in codes | **PASS** | `termination_reason=learned_eos`, `codes_max=2034 < 2048` |
| Checkpoint-listen WAVs | **PASS** | `out/.../smoke_1utt/listen_demo/*.wav` (+ Windows mirrors) |
| Wave 4 / DEV / 1h | **NOT STARTED** | hard-stop before Wave 4 by plan default |

## Confirm-seed commitment

| Field | Value |
|-------|-------|
| Path | `configs/train_subsets/d1c_confirm_seed_commitment.json` |
| `commitment` | `4c46d418caf9bcabc088864da5bf0b3de0475a9d9225f50383314a2c5e3bb3f2` |
| `generator_version_or_hash` | `custodian-agent-wave3-51779b7` |
| `plaintext_seed_present` | `false` |
| Commit | `c15ed40` — `chore(d1c): mint confirm-seed commitment for Wave 3` |

Plaintext decimal seed is **chat-only** (not in git / RESULTS / this report).

## GPU smoke metrics (from artifacts)

Train (`logs/wave3_gpu_console.log` / `runs/d1c_eos_smoke_1utt/`):

- `n_train=1` utt `ruslan_000842_RUSLAN` via `configs/train_subsets/d1c_smoke_1utt.json`
- `optimizer_steps=80`, `batch=1`, `grad_accum=4`
- Final: `loss=0.0005649451049976051`, `ce_cb0=0.00041432815487496555`, `ce_residual=0.00011461576650617644`

Free-running generation (`logs/wave3_1utt_metrics.json` / codes sidecar):

| Field | Value |
|-------|-------|
| `termination_mode` | `semantic_eos_v1` |
| `termination_reason` | `learned_eos` |
| `termination_step` / `n_decoded_frames` | 22 |
| `eos_selected` | true |
| `eos_probability_at_stop` | 0.9999634027481079 |
| `eos_margin_to_best_real_cb0` | 14.951213836669922 |
| `codes_max` | 2034 (decoder-safe; no EOS id 2048) |
| `generated_codes_sha256` | `95ac9b36c44b6702648cabb1f5496f6d35f2d850ad37a8d429e88f550db67ea1` |

Note: tqdm train log lines do not print `ce_semantic_eos` scalars; EOS success is evidenced by FR `learned_eos` + TF loss collapse, not by the regex probe (`n_eos_loss_points=0` in `wave3_1utt_eos_probe.json`).

## WAVs / listen paths

Linux primary:

- `~/projects/tts-self/out/d1c_semantic_eos_repair_b81ecc3e/smoke_1utt/listen_demo/0000_ruslan_000842_RUSLAN.wav`

Windows mirrors (Cursor playback):

- `C:\projects\tts-self\out\d1c_semantic_eos_repair_b81ecc3e\smoke_1utt\listen_demo\0000_ruslan_000842_RUSLAN.wav`
- `C:\projects\tts-self-d1c\out\d1c_semantic_eos_repair_b81ecc3e\smoke_1utt\listen_demo\0000_ruslan_000842_RUSLAN.wav`

Index: `.cursor/results/d1c_semantic_eos_repair_b81ecc3e/listen_index.json`

## Known non-blockers / caveats for Wave 4

1. Smoke used `--allow-dirty` because `eval_contract.repro.clean_sha_available` is absent (known gap). Tree was actually clean; flag only relaxes `require_clean_sha`. **Wave 4 formal DEV should set `clean_sha_available: true` and reseal** before refusing `--allow-dirty`.
2. SoX missing in qwen venv (warning only); decode still wrote wav.
3. No N2 claim. Smoke is infrastructure-only.

## Next action

Wave 3 exit criteria met → proceed to **Wave 4 DEV W1 seed101** on clean tip (after optional `clean_sha_available` reseal). Do **not** start 1h/5h/full.

## Commits this resume

| SHA | Message |
|-----|---------|
| `c15ed40` | mint confirm-seed commitment |
| `c71b453` | smoke_1utt subset + GPU runner |
| `a602593` / `74d4caa` | keep GPU logs outside worktree + probe path |
