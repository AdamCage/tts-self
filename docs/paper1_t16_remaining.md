# Paper 1 — T16 remaining checklist

Full draft: [paper1_proxy_metrics_trap.md](paper1_proxy_metrics_trap.md).  
Figures: [paper1_figures.md](paper1_figures.md).  
Preferences: [paper1_preference_status.md](paper1_preference_status.md).

## Done in-repo (automation)

1. ~~Seeds + bootstrap CI on mel CTC-gap arms~~ — `out/paper2_seeds_summary.json` (T08).
2. ~~Figure stub generator~~ — superseded by S4 real figures: `scripts/68_paper1_figures.py` → `out/paper1_figures/` (each PNG has source JSON path).
3. ~~External baseline harness + fair three conditions~~ — R2: `out/external_baseline_silero_v3.json` (raw / +frontend / Track A).
4. P.808 **protocol** — [p808_pilot_protocol.md](p808_pilot_protocol.md) (no scores until humans listen). **MOS remains unmeasured.**
5. Paper2 cover-vs-CER figure — `docs/figures/paper2_cover_vs_cer.png`.
6. ~~Pairwise preference ballot dry-run~~ — `scripts/51_preference_test.py` + `out/paper1_figures/preference_silero_fe_vs_track_a/` (self-vs-self traps). Study **BLOCKED** without listeners.

## Still needed (human / external assets)

| Item | Blocker | Command / next step |
|------|---------|---------------------|
| Human pairwise preferences | Listener panel | Score ballot with `--score responses.json`; record listener_count + CI95 + trap_same_rate |
| P.808 pilot MOS | Human listeners | Follow [p808_pilot_protocol.md](p808_pilot_protocol.md); store CSV under `out/p808_pilot/` (gitignored). **MOS currently unmeasured.** |
| Optional: attention heatmaps / trap schematic | Camera-ready polish | Not required for S4 acceptance |

**Do not invent listening scores or external CER.**

## Quality ladder snapshot (for paper1 context; mel Track A)

| Checkpoint | val100 CER | val100 UTMOS | Note |
|------------|-----------:|-------------:|------|
| mel80 S4 baseline | 0.056 | 1.272 | pre-stress |
| stress S4 | **0.031** | **1.277** | Gate V4 **GO** — current production |
| variance S4 | 0.430 | 1.304 | Gate V5 **FAIL** (CER collapse) |

Artifacts: `out/eval_mel_suite_s4_baseline_full.json`, `out/eval_mel_suite_stress_s4.json`, `out/eval_mel_suite_variance_s4.json`, `out/gate_v4_v5_summary.json`.  
UTMOS columns are **not** MOS.
