# §6 Report — A1 (Qwen latent backbone)

1. **Task ID:** A1 — Qwen3.5 sequential backbone over `vibevoice64` latents.
   Implemented `latent_io`, `backbone_lm`, `configs/n2_qwen_latent_v01.yaml`, and
   acceptance tests (tiny causal stack; real Qwen load path available via
   `load_pretrained=True`).

2. **Acceptance criteria**

   | Criterion | Result |
   |-----------|--------|
   | Forward runs; I/O shapes named | **Met.** Batch=2: `input_ids [2,L]`, `z_input [2,T,64]`, `acoustic_hidden [2,T,H]` (`H=64` in unit tests; config `H=1024`). |
   | Trainable list; backbone trainable = 0 at start | **Met.** Tiny: trainable projections = **4288**, backbone trainable = **0**. Config-sized projectors (64→1024 + marker + BOS): trainable = **67648**, backbone = **0**. Grads only on `latent_in` / `acoustic_marker` / `bos_latent`. |
   | Causality: change latent at t does not change outputs &lt; t | **Met.** `test_causality_latent_t_does_not_affect_earlier` (max Δ before t &lt; 1e-5). |
   | VRAM on config batch ≤ 14.5 GB | **Not measured on this host** (no CUDA). `test_vram_smoke_batch2` marked `@pytest.mark.gpu` and skipped. Re-run in WSL with real Qwen for the number. |
   | Geometry from `TARGETS` only | **Met.** `target: vibevoice64` → `get_target`; no new dim constants. |

3. **Files created / changed**

   - `src/minispeech/models/latent_io.py` (new)
   - `src/minispeech/models/backbone_lm.py` (new)
   - `configs/n2_qwen_latent_v01.yaml` (new; shared with A2/A3)
   - `tests/test_n2_backbone.py` (new)
   - `pyproject.toml` (register `gpu` pytest mark)

4. **Artifacts**

   - Config: `configs/n2_qwen_latent_v01.yaml`
   - This report: `.cursor/results/n2_first_execution/A1_REPORT.md`
   - No wav / no training run (A4+)

5. **Reproduce**

   ```text
   git rev-parse --short HEAD   # 2c4c4bd at authoring (tree dirty with Wave-2 WIP)
   set PYTHONPATH=src
   python -m pytest -q tests/test_n2_backbone.py
   # → 5 passed, 1 skipped (gpu)
   ```

6. **Found nearby (not fixed)**

   - Real Qwen weights live under WSL (`qwen.model_id` in config); Windows unit
     tests use `_TinyCausalLM`.
   - `MultiCorpusDataset` still returns `text_hidden`, not `input_ids` (A4 wiring).

7. **Blockers**

   - Peak VRAM with frozen Qwen3.5-0.8B + batch from config not measured here
     (no CUDA on authoring host).
