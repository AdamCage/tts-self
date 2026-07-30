# Paper 1 — T16 remaining checklist

Full draft: [paper1_proxy_metrics_trap.md](paper1_proxy_metrics_trap.md).

## Done in-repo (automation)

1. ~~Seeds + bootstrap CI on mel CTC-gap arms~~ — `out/paper2_seeds_summary.json` (T08).
2. Figure stub generator — `scripts/38_paper1_figure_stubs.py` → `docs/figures/paper1_*` (skips missing inputs; does not invent curves).
3. External baseline **harness** — `scripts/39_external_baseline_eval.py` (refuses Hub download without `--allow-download`; writes `skipped` if Silero absent).
4. P.808 **protocol** — [p808_pilot_protocol.md](p808_pilot_protocol.md) (no scores until humans listen).
5. Paper2 cover-vs-CER figure — `docs/figures/paper2_cover_vs_cer.png`.

## Still needed (human / external assets)

| Item | Blocker | Command / next step |
|------|---------|---------------------|
| Render CER(σ)/CER(α) from real sweeps | Need Phase-0 / `21_latent_error_calibration` JSON on disk | `python scripts/38_paper1_figure_stubs.py` after artifacts exist |
| Silero (or other) CER on frozen sets | Local model or explicit `--allow-download` + disk check | `python scripts/39_external_baseline_eval.py --backend silero --sets demo5 --allow-download` then score wavs with ASR suite |
| P.808 pilot MOS | Human listeners | Follow [p808_pilot_protocol.md](p808_pilot_protocol.md); store CSV under `out/p808_pilot/` (gitignored) |
| Latent-side seed CI | Optional if paper keeps Part-1 latent numbers | Re-run Part-1 diagnostics seeds |

Depends on freeze tag `paper2-frozen-mel80` (local; **do not push** without OK).

**Do not invent listening scores or external CER.**

## Quality ladder snapshot (for paper1 context; mel Track A)

| Checkpoint | val100 CER | val100 UTMOS | Note |
|------------|-----------:|-------------:|------|
| mel80 S4 baseline | 0.056 | 1.272 | pre-stress |
| stress S4 | **0.031** | **1.277** | Gate V4 **GO** — current production |
| variance S4 | 0.430 | 1.304 | Gate V5 **FAIL** (CER collapse) |

Artifacts: `out/eval_mel_suite_s4_baseline_full.json`, `out/eval_mel_suite_stress_s4.json`, `out/eval_mel_suite_variance_s4.json`, `out/gate_v4_v5_summary.json`.
