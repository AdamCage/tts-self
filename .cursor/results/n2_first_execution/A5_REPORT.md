# §6 Report — A5 (N2 text-usage gate)

1. **Task ID:** A5 — Port text-usage gate keys to N2; emit `out/n2_text_usage.json`.

2. **Acceptance criteria**

   | Criterion | Result |
   |-----------|--------|
   | JSON keys match Part 1 (`text_shuffle`, `text_mask`, `text_prefix_counterfactual`, `attention_diagonality`, `s1_text_usage_pass`, `s2_text_usage_pass`) | **Met.** |
   | Exit code 2 on S1 fail | **Met.** Process RC **2**. |
   | Shuffle CER delta with CI95 substantially positive | **Partial.** Measured on **diffusion train-loss** (not ASR CER): `relative_rise=0.428` (base 0.090 → shuffled 0.128). **Pass** vs min_rise 0.20. No ASR CI95 (CER harness not attached). |
   | Mask same | **Met (loss).** `relative_rise=1.631` (base 0.064 → masked 0.170). **Pass.** |
   | Prefix counterfactual | **Fail.** `ratio=1.100` &lt; min 3.0 → `pass=false`. |
   | Attention diagonality decision documented | **Met.** Last transformer layer, mean heads, acoustic-query→text-key; reuse `guided_attn.attention_diagonality`. **Value `null`** — Qwen3.5 path did not return usable `output_attentions` in this run. |

3. **Files created / changed**

   - `scripts/67_eval_n2_text_usage.py` (new)
   - `src/minispeech/eval/text_usage.py` — `n2_text_*` + `n2_attention_diagonality`
   - `scripts/65_train_n2.py` — `--text-usage` flag (optional post-train)

4. **Artifacts**

   | Artifact | Path |
   |----------|------|
   | Gate JSON | `out/n2_text_usage.json` |
   | Sidecar | `runs/n2_s1_ruslan/eval_text_usage.json` |
   | Log | `.cursor/results/n2_first_execution/logs/a5_text_usage.log` |

   **Headline numbers** (`out/n2_text_usage.json`):

   - `text_shuffle.relative_rise` = **0.428** → pass
   - `text_mask.relative_rise` = **1.631** → pass
   - `text_prefix_counterfactual.ratio` = **1.100** → fail
   - `s1_text_usage_pass` = **false**
   - `s2_text_usage_pass` = **false**

5. **Reproduce**

   ```bash
   cd ~/projects/tts-self && source .venv/bin/activate
   export HF_HOME=$HOME/hf_cache
   export PYTHONPATH=/mnt/c/projects/tts-self/src
   export MINISPEECH_ALLOW_DIRTY=1
   python /mnt/c/projects/tts-self/scripts/67_eval_n2_text_usage.py \
     --ckpt runs/n2_s1_ruslan/best.pt \
     --config /mnt/c/projects/tts-self/configs/n2_qwen_latent_v01.yaml \
     --data /mnt/c/projects/tts-self/configs/data_ruslan_local.yaml \
     --out out/n2_text_usage.json
   # expect exit 2 until prefix passes
   ```

6. **Found nearby (not fixed)**

   - Gate currently uses ε-prediction loss rise, not free-running CER deltas with bootstrap CI95 (plan wording). Wiring ASR CER A/B would strengthen A6 readiness.
   - Prefix test is AR+DDIM and high-variance; may need more steps / CFG tuning before ratio≥3.

7. **Blockers**

   - **`s1_text_usage_pass=false`** (prefix fail) → **A6 blocked** per plan.
   - Attention diagonality numeric value unavailable (`null`) on this backbone/attention API.
