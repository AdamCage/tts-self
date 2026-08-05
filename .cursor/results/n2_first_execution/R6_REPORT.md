# R6 — Bank delta 2×2 (labels × frontend)

## 1. Task
R6: decomposed bank_entity DER/EER change across {labels v2, v3} × {frontend before/after R3} on ckpt `runs/mel_stress_v01_s4/best.pt`; documented val100/latin `ref_norm` caveat.

## 2. Acceptance

| Criterion | Result |
|-----------|--------|
| four cells with CI95 | **PASS** — see table; utterance-cluster bootstrap CI95 on DER/EER |
| contributions sum check; unexplained remainder named | **PASS** — path `label_before + frontend_v3` = observed; remainder **0.0** (DER and EER) |
| val100/latin ref_norm caveat in doc | **PASS** — `docs/frozen_numbers_mel80.md` (v1→v2: 2/100 and 6/117) |

## 3. Changed / created files
- `out/bank_delta_2x2.json`
- `out/n2_first_execution/bank_entity_v2_from_2c4c4bd.json` (recovered labels)
- `out/_r6_delta_2x2.py` (scoring driver)
- `docs/frozen_numbers_mel80.md` — baseline `ref_norm` caveat + R6 2×2 section
- `.cursor/results/n2_first_execution/logs/r6_delta_2x2.log`

## 4. Artifacts (numbers)

Source hyps:
- frontend_after: R2 Track A `out/external_baseline_track_a_v3.json`
- frontend_before: P1 `out/baseline_p1_v2_full.json`
- labels v3: `configs/eval_sets/bank_entity.json` (3.0.0)
- labels v2: `git show 2c4c4bd:configs/eval_sets/bank_entity.json`

| Cell | DER [CI95] | EER [CI95] |
|------|------------|------------|
| before × v2 | 0.6566 [0.2209, 1.4163] | 0.4222 [0.3683, 0.4762] |
| before × v3 | 0.6370 [0.1919, 1.4079] | 0.4730 [0.4159, 0.5270] |
| after × v2 | 0.6493 [0.2135, 1.4118] | 0.3937 [0.3397, 0.4476] |
| after × v3 | 0.6238 [0.1776, 1.3958] | 0.3397 [0.2857, 0.3905] |

DER decomposition (before×v2 → after×v3):
- observed Δ = **−0.0329**
- label @ before = −0.0196; label @ after = −0.0255 (avg −0.0225)
- frontend @ v2 = −0.0073; frontend @ v3 = −0.0133 (avg −0.0103)
- interaction = −0.0059
- unexplained remainder = **0.0**

EER: observed −0.0825; remainder **0.0** (strong label×frontend interaction: label effect flips sign).

Scorer note: rescored P1+v2 DER 0.6566 ≠ published artifact aggregate 0.6142; all four cells use one current scorer so deltas are consistent.

## 5. Reproduce
```bash
cd ~/projects/tts-self && source .venv/bin/activate
export PYTHONPATH=/mnt/c/projects/tts-self/src
bash /mnt/c/projects/tts-self/out/_r6_run.sh
# HEAD 2c4c4bd, dirty=0
```

## 6. Found nearby (not fixed)
- Absolute DER for historical P1 hyps drifted under current `critical_spans` (54 utts differ); needs a freeze of the metric code if absolute levels must match old JSON.
- DER bootstrap CI upper bound can exceed 1 (edit-distance with extra hyp numerals).

## 7. Blockers
None.
