# §6 Report — A4 (S1 overfit, N2 trainer)

1. **Task ID:** A4 — N2 trainer + S1 overfit on 32 RUSLAN utterances
   (CV17 caches not required; `data.source: ruslan`).

2. **Acceptance criteria**

   | Criterion | Result |
   |-----------|--------|
   | Trainer with `run_manifest` (`git_sha`, `git_dirty`), logs, ckpts, FR-aware selection | **Met.** `runs/n2_s1_ruslan/run_manifest.json`: `git_sha=2c4c4bd5…`, `git_dirty=false`. FR selection via `fr_eval.jsonl` / `best_fr_metrics.json`. |
   | Train 32 utts until overfit | **Met (TF loss).** Diffusion train loss **1.15 → 0.097** (step 50→3000; `loss.jsonl`). Final step loss **0.056**. |
   | Free-running metrics via existing modules | **Met (latent).** `mean_dtw_l1=2.076` (n=16), `duration_ratio=1.0` (GT-length AR). FR `r2_overall=-6.64`, `var_ratio=6.72`. TF diagnostics: `r2_overall=-0.281`, `var_ratio=0.332`. |
   | FR CER vs copy-synth **0.073** / trivial baselines | **Not met / blocked.** ASR CER not wired in this A4 final eval (`cer: null` in FR best). `out/trivial_baselines.json` absent on this host. Latent FR metrics reported instead. |
   | Listen wavs under `out/n2_first_execution/n2_s1_listen_demo/` | **Met.** Five demo wavs + `listen_index.json`. |
   | Gate R² ≥ 0.28 | **Fail on FR/TF R².** TF R² **−0.281**; FR R² **−6.64**. Overfit is on ε-loss, not content R². |

3. **Files created / changed**

   - `src/minispeech/train/n2_trainer.py` (new)
   - `scripts/65_train_n2.py` (new)
   - `scripts/66_infer_n2.py` (new; A4 stub, A7 owns full compare)
   - `configs/n2_qwen_latent_v01.yaml` (`data.source=ruslan`, S1 `run_dir`, FR knobs)
   - `src/minispeech/models/backbone_lm.py` (bf16 embed cast; pad-aligned masks)
   - `src/minispeech/models/latent_io.py` (`max_text_len` / `max_acoustic_len` on masks)

4. **Artifacts**

   | Artifact | Path |
   |----------|------|
   | Run dir | `~/projects/tts-self/runs/n2_s1_ruslan/` (`best.pt`, `last.pt`, manifests) |
   | Metrics JSON | `out/n2_s1_ruslan.json` |
   | Loss curve | `runs/n2_s1_ruslan/loss_curve.json` (+ copy under results) |
   | Listen wavs | `out/n2_first_execution/n2_s1_listen_demo/0{1..5}_demo.wav` |
   | Train log | `.cursor/results/n2_first_execution/logs/a4_train_s1.log` |

   **Loss curve (train, 50-step avg):** 50→1.152; 500→0.299; 1000→0.240; 1500→0.179; 3000→0.097.

   **Batch / VRAM:** `batch_size=2` (manual; auto skipped). Peak train ~5.3 GB (`nvidia-smi` mid-run).

5. **Reproduce**

   ```bash
   cd ~/projects/tts-self && source .venv/bin/activate
   export HF_HOME=$HOME/hf_cache
   export PYTHONPATH=/mnt/c/projects/tts-self/src
   export MINISPEECH_ALLOW_DIRTY=1
   python /mnt/c/projects/tts-self/scripts/65_train_n2.py \
     --config /mnt/c/projects/tts-self/configs/n2_qwen_latent_v01.yaml \
     --data /mnt/c/projects/tts-self/configs/data_ruslan_local.yaml \
     --stage S1 --max-utts 32 --batch-size 2 --run-dir runs/n2_s1_ruslan
   python /mnt/c/projects/tts-self/scripts/66_infer_n2.py \
     --ckpt runs/n2_s1_ruslan/best.pt --demo \
     --out out/n2_first_execution/n2_s1_listen_demo
   ```

6. **Found nearby (not fixed)**

   - Free-running R²/var_ratio are poor despite low ε-loss — expected gap until AR+DDIM content locks; A6 CER gate still required.
   - Qwen load reports unexpected visual weights (text-only submodule from multimodal checkpoint) — benign.
   - WSL launcher must use `start_new_session=True` or SIGHUP kills the train job.

7. **Blockers**

   - **FR ASR CER** not measured (no ASR hook in A4 final JSON) → cannot numerically compare to copy-synth **0.073**.
   - Content overfit (R² / intelligibility) **not** achieved; only diffusion train-loss overfit. Treat A4 as **partial pass** for scaffolding + runnable S1; content gate remains open for A5/A6.
