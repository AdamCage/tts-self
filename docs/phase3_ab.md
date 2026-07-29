# Phase 3 — Flow vs regression A/B (Workstream D)

Date: 2026-07-29. Runtime: WSL2 + RTX 4080 Super. Git authoring SHA at Flow S1 start: `8dce2c1` (WSL clone often rsynced; run manifests record local SHAs).

## Setup

| Item | Value |
|------|-------|
| Regression ckpt | `runs/v01_regression_direct/best.pt` (S3 PASS, calibrated infer) |
| Flow config | `configs/flow_v02.yaml` (`freeze_talker_fraction: 0.05`, Heun 4, CFG default 1.3) |
| Flow init | S1 from regression (`loaded=100` talker/stop/dur keys); S2/S3 from flow FR-best |
| Infer stop (both) | `thr=0.45`, `min_frac=0.94`, `duration_scale=1.04`, consec=3 |
| Val set | first 32 val utts (same FR gate set as S3 regression) |

## Objective metrics (val32, calibrated stop)

| System | stop | runaway | duration_in_band | mean_dtw_L1 | mean_dur_ratio | notes |
|--------|-----:|--------:|-----------------:|------------:|---------------:|-------|
| **Regression S3** (calibrated) | **1.0** | **0** | **0.906** | 0.808 | 0.990 | hard gates PASS |
| Flow S1 best (step 1k) | 1.0 | 0 | 0.656 | 0.801 | 0.885 | stop green; duration short |
| Flow S2 best (step 1k) | 1.0 | 0 | 0.75–0.781 | 0.88–0.90 | 0.93–0.94 | best mid-run; later collapse |
| Flow S2 late (step 8k) | 1.0 | 0 | **0.062** | — | — | FR selection kept step-1k best |

Artifacts:

- `runs/v01_regression_direct/eval_free_running_S3_calibrated.json`
- `runs/v02_flow/eval_free_running_S1_calibrated.json`
- `runs/v02_flow/eval_free_running_S2_calibrated.json`
- `runs/v02_flow/cfg_sweep_S2_best.json`
- `runs/v02_flow/best_fr_metrics.json` / `fr_eval.jsonl`

### CFG sweep (Flow S2 FR-best, Heun 4)

| cfg_scale | duration_in_band | mean_dtw_L1 | stop | runaway |
|----------:|-----------------:|------------:|-----:|--------:|
| 1.0 | 0.781 | 0.904 | 1.0 | 0 |
| 1.3 | 0.750 | 0.900 | 1.0 | 0 |
| 1.6 | 0.781 | **0.875** | 1.0 | 0 |

CFG does not close the gap to regression’s 0.906 in-band; 1.6 slightly helps DTW.

### Text-usage (Flow S1, train subset via `09_eval.py`)

- Shuffle / mask rises pass (~0.30 / ~0.28).
- Prefix counterfactual fail; overall `s1_text_usage_pass: false` (soft — same family of issues as regression val soft-fail).

### Listen set

16 val pairs under `out/listen_ab_flow/{reg,flow,gt}/` (Windows mirror: `C:\projects\tts-self\out\listen_ab_flow\`). Both heads stop reliably; flow tends to slightly shorter `dur_pred` / `T`.

ASR CER/WER not re-run in this pass (codec+ASR cost); latent FR gates + listen pairs are the A/B decision basis for now.

## Training notes

1. Flow S1 completed 3k steps without crash; FR-best at 1k (`in_band=0.625`), later steps worsened duration while DTW edged down.
2. Flow S2 (1 h, 8k) improved early FR to **0.781** then **collapsed** (`in_band→0.062` by 8k). FR-aware `best.pt` correctly retained step 1k — same failure mode class as pre-calibration regression duration drift.
3. Partial init from regression works (`init_from` talker/stop/dur); flow head trains from scratch under `freeze_talker_fraction=0.05`.

## Recommendation

**Keep regression as the production / demo head.**

Reasons:

1. Hard duration gate: regression **0.906** vs best flow subset **≤0.781** (CFG sweep does not fix it).
2. Stop / runaway already matched (both green); flow adds no stability win.
3. DTW is comparable or slightly worse for flow on the same val32.
4. Flow mid-run FR collapse on S2 argues against promoting flow before a duration-aware train recipe (shorter schedule, stronger freeze, or dur-focused FT) is proven on ≥5 h.

**Do not start full S4 for both heads yet.** Optional next experiments (not blocking demos):

- Flow S3 (5 h) with FR selection only — check whether early peak exceeds 0.85–0.90.
- Duration-head FT on flow FR-best (mirror regression dur-ft) + re-calibrate `min_frac` / `duration_scale`.
- If flow ever clears `duration_in_band≥0.90` with stop green on val32, re-open A/B with ASR + full S4 for both.

## Demo

Winning head for demos: **regression** `runs/v01_regression_direct/best.pt` → `out/demo_phrases/` (calibrated stop).

## Commands (WSL)

```bash
cd ~/projects/tts-self && source .venv/bin/activate && export HF_HOME=$HOME/hf_cache

python scripts/09b_eval_free_running.py \
  --ckpt runs/v02_flow/best.pt \
  --data configs/data_ruslan_local.yaml --infer configs/infer.yaml \
  --split val --n 32 --out runs/v02_flow/eval_free_running_S2_calibrated.json

python scripts/17_demo_phrases.py \
  --ckpt runs/v01_regression_direct/best.pt \
  --out-dir out/demo_phrases
```
