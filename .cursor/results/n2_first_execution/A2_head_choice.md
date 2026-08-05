# A2 — Head choice: DDPM over Qwen states

**Date:** 2026-08-01  
**Config:** `configs/n2_qwen_latent_v01.yaml` → `model.diffusion`  
**Module:** `src/minispeech/models/head_diffusion.py`

## Decision

**DDPM (ε-prediction) + DDIM sampler** conditioned on Qwen acoustic hidden
states. Rectified flow over Qwen states was considered but not selected for the
N2 pilot.

## Why DDPM (not flow)

| Criterion | DDPM (chosen) | Flow over Qwen `h` |
|-----------|---------------|--------------------|
| Methodology target | Explicit in `03_architecture.md` (4-layer DDPM + CFG) | Used in Part 1 Talker path (`heads_flow.py`) |
| Part 1 failure mode | Part 1 failed with **from-scratch Talker** + flow/regression — not evidence against DDPM on a **frozen pretrained** backbone | Same Part 1 stack; restarting flow-on-Talker is out of scope (`AGENTS.md`) |
| CFG pattern | Same `null_h` dropout scheme as `FlowHead` | Already proven in-repo |
| Eval determinism | DDIM η=0 + fixed `torch.Generator` seed → bit-identical | Heun also seedable; no advantage |
| Fast eval | `sample_steps: 10` DDIM (config) | 4-step Heun typical |

No trained N2 checkpoint exists yet, so this choice is **architectural /
methodological**, not a CER bake-off. A later A6 measurement may revisit the
head class if N2 FAIL points at the generative head.

## Config knobs (geometry via TARGETS)

- Latent dim: `TARGETS["vibevoice64"].dim` = 64  
- Cond dim: `qwen.hidden_size` = 1024  
- `timesteps: 1000`, linear β, `cfg_dropout: 0.1`, `cfg_scale: 1.3`  
- Eval: `sampler: ddim`, `sample_steps: 10`, `eval_seed: 42`

## Sanity (unit test, not CER)

`micro_overfit_sanity` + `test_sanity_true_context_beats_constant_mean`: after a
short synthetic overfit, one-step `x0` reconstruction L2 with true context is
**substantially below** constant-mean prediction. Full guidance→CER table needs
A4 decode + ASR; proxy L2 vs CFG scale is recorded in the A2 §6 report.
