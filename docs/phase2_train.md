# Phase 2 — Training ladder (Workstream C)

Date: 2026-07-29  
Git (S2 run): `4d4282b`  
Config: `configs/talker_v01_direct.yaml` (direct head, `acoustic_dropout: 0.35`)  
Data: `configs/data_ruslan_local.yaml` — QC `exclude` filtered (`exclude_tags` default)  
Corpus: 21642 utts / 29.61 h; train non-exclude ≈ 12490 / 15.4 h  
Best ckpt: `runs/v01_regression_direct/best.pt` (WSL; not committed)

## C0 — Sync / filter

- WSL reset to `origin/master`, then pulled ladder fixes.
- `MiniSpeechDataset(exclude_tags=("exclude",))`: train 12490 with **0** exclude rows.
- S1 = 32 utts; S2 ≈ 1236 utts / 1.0 h.

## C1 — S1 reconfirm — **PASS**

Train: 3000 steps, auto `batch_size=64`, ~1.5 min.  
Final TF: `l1≈0.070`, `total≈0.119`.

Text-usage on **train** (32 overfit utts; required for S1):

| Test | relative_rise / ratio | Pass |
|------|----------------------:|:----:|
| shuffle | 1.80 | ✓ |
| mask | 1.66 | ✓ |
| prefix | ≫ 3 | ✓ |
| attention_diagonality | 0.152 | (trend) |

`s1_text_usage_pass: true`.

Note: `09_eval` previously defaulted to **val**, where TF rises are ~0 after S1 overfit (expected). Fixed to `--split train` default; S1 gate must use train.

## C2 — S2 (1 h) — **PARTIAL FAIL** (stop gate)

Init: `--init-from runs/v01_regression_direct/best.pt` (new CLI; fresh weights would wipe S1).  
Train: 12000 steps, batch 64, ~10 min.  
Final TF (train loader): `l1≈0.47`, `total≈0.53`.

### Text-usage

| Split | shuffle rise | mask rise | prefix | s2_pass (mask≥0.35) |
|-------|-------------:|----------:|:------:|:-------------------:|
| train (first 32 of 1 h) | 2.64 | 1.71 | ✓ | ✓ |
| val (32) | 0.003 | −0.002 | ✓ | ✗ |

Prefix (free-running) passes on val; teacher-forced shuffle/mask on val still near-zero (model still leans on acoustics under TF on unseen text).

### Free-running val (n=32)

| Metric | Value | Gate | Result |
|--------|------:|------|:------:|
| stop_success_rate | 0.844 | ≥ 0.95 | **FAIL** |
| runaway_rate | 0.156 | ≤ 0.02 | **FAIL** |
| nan_rate | 0.0 | = 0 | PASS |
| mean DTW L1 | 0.786 | — | baseline |
| DTW / TF L1 | 0.942 | ≤ 1.5× | PASS* |
| duration_in_band (0.8–1.2) | 0.281 | (≥0.9 is S3) | weak |

\*TF L1 on val is high (~0.83); ratio is not a quality win by itself.

### Listen set (8 val → `out/listen/S2/`)

- silence_collapse: **0/8**
- stop_triggered: **8/8**, runaway: **0/8**
- RMS ≈ 0.009–0.032 (audible, not collapsed)

## C3 — S3 — **NOT STARTED**

Blocked by S2 hard stop/runaway gate. Do not start Flow (Workstream D) until S3 is green.

## Blockers / next

1. **Stop head / duration** — raise stop success ≥95% and runaway ≤2% on fixed 32-val set (check `stop_pos_weight`, threshold, duration head bias).
2. Re-eval S2 after stop fix: `09_eval --split val --mask-min-rise 0.35` + free-running JSON.
3. Only then: S3 20k steps `--init-from best.pt`, DTW/duration/ASR gates.
4. Workstream D (Flow A/B) remains **blocked**.

### Commands (WSL)

```bash
cd ~/projects/tts-self && source .venv/bin/activate && export HF_HOME=$HOME/hf_cache
# after stop fix:
python scripts/07_train.py --config configs/talker_v01_direct.yaml \
  --data configs/data_ruslan_local.yaml --mode regression --stage S3 \
  --init-from runs/v01_regression_direct/best.pt \
  2>&1 | tee runs/v01_regression_direct/console_S3.log
```
