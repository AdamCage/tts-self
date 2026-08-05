# §6 Report — A3 (Curriculum collator + batch packing)

1. **Task ID:** A3 — Mixed text+latent collator; curriculum short→15 s and
   `train.batch` auto/14.5 GB knobs in config (not hard-coded).

2. **Acceptance criteria**

   | Criterion | Result |
   |-----------|--------|
   | Chosen batch size + actual VRAM named | **Not measured on this host** (no CUDA / no A4 trainer probe). Config declares `mode: auto`, `candidates: [1,2,4,8]`, `target_vram_gb: 14.5` for A4 `auto_batch_size`. |
   | Curriculum schedule in config, not code | **Met.** `train.curriculum.stages`: 3.0 s / 8.0 s / **15.0 s** with step budgets; `parse_curriculum` / `curriculum_max_dur_s` only read YAML. |
   | Collator masks OK; padding does not affect loss (order test) | **Met.** `test_loss_equal_under_batch_order` — masked L1 identical for `[short,long]` vs `[long,short]` (atol 1e-6). |

3. **Files created / changed**

   - `src/minispeech/data/collate_latent_seq.py` (new)
   - `configs/n2_qwen_latent_v01.yaml` (`train.batch`, `train.curriculum`)
   - `tests/test_n2_collate_latent_seq.py` (new)

4. **Artifacts**

   - This report
   - Adapter `items_compatible_with_multicorpus` ready when items expose
     `input_ids` (does not edit C4 `dataset.py`)

5. **Reproduce**

   ```text
   git rev-parse --short HEAD   # 2c4c4bd (dirty WIP)
   set PYTHONPATH=src
   python -m pytest -q tests/test_n2_collate_latent_seq.py
   # → 5 passed
   ```

6. **Found nearby (not fixed)**

   - `MultiCorpusDataset.__getitem__` does not yet attach `input_ids` (C4 /
     A4). Collator is ready; trainer must supply token ids.
   - Auto-batch VRAM probe belongs in A4 `n2_trainer` using
     `minispeech.utils.auto_batch_size`.

7. **Blockers**

   - Measured auto-selected batch size + peak VRAM deferred to A4 GPU run.
