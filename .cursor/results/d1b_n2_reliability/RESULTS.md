# D1b N2 reliability — RESULTS

**Plan:** `.cursor/plans/d1b_n2_reliability_7bd71c65.plan.md`  
**Slug:** `d1b_n2_reliability`  
**Date:** 2026-08-03 (confirm_v2 reconfirm complete)  
**Formal verdict:** **HARD-STOP** — prior **PASS withdrawn / INVALID**; confirm_v2 did **not** restore PASS

## Prior PASS withdrawn

The previous formal D1b RELIABILITY **PASS** remains **INVALID** (null TF-CE selection, score-only contract bypass, GT max_frames clamp, hard_cap/phrase-loop gaps, burned CONFIRM v1). See `artifacts/V1_SUPERSEDED.md` and `d1b_reliability_gate.json`.

**Do not proceed to 1h / 5h / full.** Next stage: `D1c_termination_repair_only`.

## Environment recovery (this turn)

| Check | Result |
|-------|--------|
| Windows `C:` free | Recovered (~120 GB after deleting Windows-mirrored `runs/discrete_speech_lm_300m_s1_smoke` ~14.8 GB; was ~6 MB) |
| WSL Ubuntu | Recovered after `wsl --shutdown` (earlier E_FAIL / docker-desktop default) |
| Linux root (`~/`) | ~558 GB free on `/dev/sdd` |
| Artifact policy | Multi-GB ckpts/wavs on Linux FS only (`~/projects/tts-self/runs|out/...`); Windows mirrors small JSON + few listen wavs |

## GPU code SHA (clean vs dirty)

Aligned with `d1b_reliability_gate.json`:

| Role | SHA | Notes |
|------|-----|-------|
| Selection/provenance/stability fixes + confirm32_v2 | `bee3f009661bb30472fc02682f429787c38b46cb` | Pre-atomic-save train/select lineage |
| Atomic `torch.save` (ckpt I/O); confirm_v2 eval provenance tip | `0db69d608eb3d3d2f8fd9ba535fb92eb457fc06b` | Recorded in seed42/43 gate provenance |
| Stale v1 “canonical” (`dc749376…`) | — | **Superseded**; must not be cited as PASS authority |

### confirm_v2 seed classification

| Seed | Classification | `git_sha` (eval provenance) | `git_dirty` | Formal weight |
|------|----------------|-----------------------------|-------------|---------------|
| **43** | `FORMAL_HARD_STOP_BASIS` | `0db69d608eb3d3d2f8fd9ba535fb92eb457fc06b` | **false (clean)** | Alone establishes HARD-STOP |
| **42** | `DIAGNOSTIC` | `0db69d608eb3d3d2f8fd9ba535fb92eb457fc06b` | **true (dirty)** | Corroborating only; not sole HARD-STOP basis |

Never used `MINISPEECH_ALLOW_DIRTY=1` for intentional dirty gates. Seed42 dirty flag is recorded in primary confirm_v2 eval artifacts and demotes that seed to diagnostic. Editable install was re-pointed from `/mnt/c/...` → `/home/adam/projects/tts-self`.

## confirm_v2 oracle (uncontaminated)

| Field | Value |
|-------|-------|
| Path | `configs/train_subsets/d1b_confirm32_v2.json` |
| items_sha256 | `63d7e351602e62d26196d944cd1f7a9e928a241e70d968f6eb58c0d1b7a746b9` |
| select8 | `6667da2f8fc2c3706deff48a79489103f739419041f3fb492f5822647075a0ed` |
| gate24 | `68f8d98738114761cee86bc336be5313f3ad550519a4b6764adbdc0c4b32eda8` |
| Eval contract | 1.2.0 / `d04295d3ac9864afe5002a8470baead8f5f57a3c8f8d38831fe134aa7feada92` |

Protocol: FR checkpoint selection on **select8 only** (TF CE required) → one-shot gate24 + full32 → listen wavs. No infer retuning. No 1h.

## confirm_v2 results

Selection correctly chose **step 750** on both seeds (real TF CE, not Infinity/min-step).

| Seed | Class | Selected | select8 CER | TF CE | gate24 CER | gate24 | full32 CER | full32 | Fail mode |
|------|-------|----------|-------------|-------|------------|--------|------------|--------|-----------|
| 42 | DIAGNOSTIC (`git_dirty`) | 750 | 0.01203 | 0.00654 | 0.02052 | **FAIL** | 0.01840 | **FAIL** | phrase_loop + hard_cap (3/4) |
| 43 | FORMAL_HARD_STOP_BASIS (clean) | 750 | 0.01203 | 0.00648 | 0.02119 | **FAIL** | 0.01890 | **FAIL** | hard_cap (3/4); runaway |

CER and text-usage pass on gate24/full32 for both seeds; **stability_d1b fails** (`hard_cap_ok=false`, `runaway_or_repetition=true`). Under fixed D1b rules this is an honest N2 FAIL — not an infra abort. Seed43 clean FAIL alone sets terminal HARD-STOP.

### Artifact paths (Linux)

- Runs: `~/projects/tts-self/runs/discrete_speech_lm_d1b_confirm_v2_seed{42,43}/`
- Gates: `~/projects/tts-self/out/d1b_n2_reliability/confirm_v2_seed{42,43}/best_{gate24,full32}_n2/n2_gate.json`
- Selection: `.../runs/.../checkpoint_selection.json` and Windows `out/d1b_n2_reliability/metrics/confirm_v2_seed*_selection.json`
- Summary JSON: `.cursor/results/d1b_n2_reliability/metrics/confirm_v2_summary.json`
- Primary confirm_v2 metric JSONs are **frozen** (contract 1.2.0) — do not mutate

### Listen wavs

Linux (full):

- `~/projects/tts-self/out/d1b_n2_reliability/confirm_v2_seed42/listen_demo/`
- `~/projects/tts-self/out/d1b_n2_reliability/confirm_v2_seed43/listen_demo/`

Windows mirror (first 4 ASCII names each):

- `C:\projects\tts-self\out\d1b_n2_reliability\confirm_v2_seed42\listen_demo\utt_1.wav` …
- `C:\projects\tts-self\out\d1b_n2_reliability\confirm_v2_seed43\listen_demo\utt_1.wav` …

See `listen_index.json` (includes checkpoint path, contract/subset hashes, condition, clean/dirty).

## Infra notes from resume

1. Truncated seed42 `candidate_step_0750.pt` (2.4 GB) rebuilt from valid `S1_last.pt` @ step 750 with TF CE `0.00679` from train log; seed42 FR/gates completed without retrain.
2. First seed43 attempt aborted: dirty WSL tree (accidental `.cursor/results/...`) + editable import from `/mnt/c`; fixed and relaunched.
3. Seed43 train saved five healthy ~5.0 GB candidates on Linux ext4 with atomic save — no Bus error.

## Remediation code (earlier + this turn)

| Fix | Where / SHA |
|-----|-------------|
| Fail-closed null TF CE; real CE tie-break | `bee3f00` — `discrete_n2.py`, `76_select_*.py` |
| score-only contract + frozen max_frames; no GT clamp | `bee3f00` |
| hard_cap + phrase-loop → stability fail | `bee3f00` |
| Atomic checkpoint save | `0db69d6` — `discrete_lm_trainer._atomic_torch_save` |

## Formal gate

`.cursor/results/d1b_n2_reliability/d1b_reliability_gate.json` → **HARD-STOP**  
`prior_pass_status`: **INVALID_WITHDRAWN**  
`hard_stop_basis.establishing_seed`: **43** (clean)  
`confirm_v2.42.classification`: **DIAGNOSTIC** (`git_dirty=true`)  
`do_not_proceed_to`: `1h`, `5h`, `full`  
`next_stage`: `D1c_termination_repair_only`

## Superseded v1 metrics

On-disk PASS JSON under `artifacts/confirm_seed*` / `artifacts/dev_seed42_*` (and `out/.../metrics/confirm_seed*`) is **superseded/invalid**. Index: `artifacts/V1_SUPERSEDED.md`. Not deleted.

## Methodology

Read-only during implementation (`.cursor/methodology/**` not edited).

## Follow-ups (not started)

- Stability failures (hard_cap / phrase-loop) are the confirm_v2 blocker under fixed rules; any next stage must be a **new plan** (D1c termination-repair), not knob retuning on confirm32_v2.
- Keep large artifacts on Linux FS; avoid `/mnt/c` multi-GB writes.
