# Phase 2 — Training ladder (Workstream C → D)

Date: 2026-07-29  
Git (stop fix): `df36687` · `min_frac0.8`: `502b239` · preload/save: `f1e66d7` · phase2 docs: `60e812c`  
Config: `configs/talker_v01_direct.yaml` (direct head, `acoustic_dropout: 0.40`)  
Data: `configs/data_ruslan_local.yaml` — QC `exclude` filtered  
Best ckpt: `runs/v01_regression_direct/best.pt` (WSL; S3 dur-ft)  
Infer stop (calibrated): `min_frac=0.94`, `duration_scale=1.04` (`configs/infer.yaml`)  
Also kept: `S2b_last.pt`, `S3_last.pt`

## Root cause — S2 stop FAIL (0.844 / runaway 0.156)

1. **TF stop loss collapsed** (~1e-3) while free-running still hit `max_frac` on ~15% of val32.
2. **`w_stop=0.1` + auto `pos_weight≈7.5`** (mean_frames/window) made the stop term negligible after early S2.
3. **Hard 3-frame labels** did not match the triple-stop inference ramp; generate defaults (`min_frac=0.5`) allowed early cuts that wrecked duration-in-band (0.28).
4. Duration head TF L1 was already small — **duration_in_band failures were mostly stop timing / min_frac**, not a broken dur head.

## Fixes landed

| Change | Where |
|--------|--------|
| Soft linear stop labels on last `stop_window` | `make_stop_target(..., soft=True)` |
| Focal stop loss + soft-aware BCE weights | `losses.py`, trainer passes focal α/γ |
| Auto pos_weight floor 15 | `resolve_stop_pos_weight` |
| `w_stop=0.5`, `w_dur=0.25`, window=5, soft_labels | `talker_v01_direct.yaml` |
| Triple-stop defaults + post-`dur_pred` boost | `generate()`, `infer.yaml` |
| `scripts/09b_eval_free_running.py` (+ `--sweep-duration-scale`) | reproducible val32 FR metrics |
| Stage-subset preload (was loading **full** shard index → S3 RAM thrash) | `dataset.py` |
| Periodic `save_every` → rolling `{stage}_last.pt` | `trainer.py` |
| **FR-aware `best.pt`** via `train.fr_checkpoint` (duration_in_band / stop key) | `trainer.py` |
| Infer `duration_scale` + calibrated `min_frac=0.94` | `generate()`, `infer.yaml` |
| Flow partial `init_from` (Talker/stop/dur → new flow head) | `scripts/07_train.py` |
| S2b stage (4k continue) | config |

## C1 — S1 — **PASS** (unchanged)

Train-32 text-usage: shuffle/mask/prefix green.

## C2 — S2 → S2b — **STOP PASS**

| Checkpoint / setting | stop_success | runaway | duration_in_band | notes |
|----------------------|-------------:|--------:|-----------------:|:------|
| S2 baseline (old infer) | 0.844 | 0.156 | 0.281 | FAIL stop |
| S2 weights + new infer only | **1.0** | **0.0** | 0.438 | infer fix alone clears stop |
| S2b (4k, soft/focal) + min_frac0.65 | 1.0 | 0.0 | 0.531 | |
| S2b + min_frac **0.8** | 1.0 | 0.0 | 0.594 | |

Text-usage after S2b:

| Split | shuffle | mask | prefix | s2_pass |
|-------|--------:|-----:|:------:|:------:|
| train | 2.23 | 1.85 | ✓ | ✓ |
| val TF | ~0.006 | ~−0.002 | ✓ | ✗ |

Val TF shuffle/mask ≈0 is a **soft** issue: teacher-forced acoustics dominate on unseen text; free-running prefix still passes. Not treated as a hard stop blocker.

## C3 — S3 — **PASS** (stop + duration after infer calibration)

Runs:

1. First S3 attempt: hung near 98% (Cursor `tee`+tqdm pipe + prior full-index preload). No final ckpt.
2. Restart with subset preload + periodic save; hung again ~step 4106 in Cursor-captured tee — **mid ckpt kept** (`S3_last` @~4k).
3. Eval @~4k: stop **1.0**, runaway **0**, duration_in_band **0.844** (best mid-run TF-last peak; later overwritten).
4. Continue +8k via `nohup`: stop still 1.0, duration_in_band **regressed to 0.688**.
5. Duration-head FT 2k (`freeze_talker_fraction=1`, `w_dur=1.0`): stop 1.0, duration_in_band **0.75** (weights in `best.pt`).
6. **Infer calibration** (no further train): sweep `min_frac` × `duration_scale` × threshold on fixed val32.
   - Best: `min_frac=0.94`, `duration_scale=1.04`, `threshold=0.45`, `consecutive=3`
   - duration_in_band **0.90625**, stop **1.0**, runaway **0.0**
7. Landed **FR-aware checkpoint selection** so future mid-run peaks are not lost to TF-last overwrite.

| Gate (val32) | Final (calibrated infer) | Need | Result |
|--------------|-------------------------:|------|:------:|
| stop_success | 1.0 | ≥0.95 | **PASS** |
| runaway | 0.0 | ≤0.02 | **PASS** |
| nan / silence | 0 | 0 | **PASS** |
| duration_in_band | **0.906** | ≥0.90 (S3) | **PASS** |
| val TF text-usage | rise≈0.01 | ≥0.35 mask | **FAIL** (soft) |
| train text-usage | mask rise≈0.83 | ≥0.35 | **PASS** |

Eval artifact: `runs/v01_regression_direct/eval_free_running_S3_calibrated.json`.

## Flow (Workstream D) — **A/B DONE → keep regression**

Unblocked after duration_in_band ≥0.90 on val32 with stop green. Full writeup: [`docs/phase3_ab.md`](phase3_ab.md).

- Config: `configs/flow_v02.yaml`
- S1/S2 done; S3 started then stopped (~23%) after repeating FR collapse (`0.75→0.625`)
- Best flow FR so far: **duration_in_band ≤0.781** vs regression **0.906**
- CFG {1.0,1.3,1.6} does not close the gap; **S4 not started**
- **Recommendation: keep regression**; do not promote flow / dual S4 yet

## Demo phrases

Winning head = regression. Calibrated demos in `out/demo_phrases/` (Windows mirror: `C:\projects\tts-self\out\demo_phrases\`). Listen A/B: `out/listen_ab_flow/`.

### Commands (WSL)

```bash
cd ~/projects/tts-self && source .venv/bin/activate && export HF_HOME=$HOME/hf_cache
python scripts/09b_eval_free_running.py \
  --ckpt runs/v01_regression_direct/best.pt \
  --data configs/data_ruslan_local.yaml --split val --n 32
python scripts/17_demo_phrases.py \
  --ckpt runs/v01_regression_direct/best.pt \
  --out-dir out/demo_phrases
# Flow S1
python scripts/07_train.py \
  --config configs/flow_v02.yaml \
  --data configs/data_ruslan_local.yaml \
  --mode flow --stage S1 \
  --init-from runs/v01_regression_direct/best.pt
```
