# §6 Report — A2 (Diffusion head)

1. **Task ID:** A2 — DDPM head over backbone hidden states with CFG and
   deterministic DDIM eval. Choice documented in `A2_head_choice.md`.

2. **Acceptance criteria**

   | Criterion | Result |
   |-----------|--------|
   | Fixed seed → bit-identical samples | **Met.** `test_deterministic_eval_bit_identical` (`torch.equal`). |
   | CFG implemented; table guidance → CER (≥3 scales) | **CFG met.** Full CER **not met** (needs A4 decode+ASR). Proxy after micro-overfit (L2, seed=99, steps=10): scale 1.0 → **1.163**; 1.3 → **1.100**; 2.0 → **0.981**. Artifact: this report. |
   | Sanity: true context beats constant-mean | **Met.** One-step x0 L2 = **0.0690** vs constant-mean **1.010**; ratio **0.068** (`micro_overfit_sanity`, 250 steps). |
   | Sampler steps + time per utterance | **Met.** Config `sample_steps: 10`; measured DDIM-10 on CPU (T=8, tiny head) ≈ **0.0084 s** (not a GPU RTF claim). |
   | No `vae_std=0.625` as error budget | **Met.** Sanity uses L2 vs mean baseline only. |

3. **Files created / changed**

   - `src/minispeech/models/head_diffusion.py` (new)
   - `configs/n2_qwen_latent_v01.yaml` (`model.diffusion` block)
   - `tests/test_n2_diffusion_head.py` (new)
   - `.cursor/results/n2_first_execution/A2_head_choice.md` (new)

4. **Artifacts**

   - `A2_head_choice.md` — DDPM vs flow rationale
   - This report
   - Param count (config-sized head): **43 950 400**

5. **Reproduce**

   ```text
   git rev-parse --short HEAD   # 2c4c4bd (dirty WIP)
   set PYTHONPATH=src
   python -m pytest -q tests/test_n2_diffusion_head.py
   # → 6 passed, 1 skipped (gpu)
   ```

6. **Found nearby (not fixed)**

   - Guidance→CER table requires trained N2 + frozen codec decode + ASR (A4/A6).
   - DDIM L2 after micro-overfit still high vs one-step x0 hat; expected for
     short synthetic train — not an e2e quality claim.

7. **Blockers**

   - CER-vs-CFG table blocked until A4 produces a synthesizable checkpoint.
