# Paper 2 draft — CTC blank gaps silently destroy mel timelines

**Status:** data draft (T15). Freeze tag: `paper2-frozen-mel80` @ `56c8182`. Numbers: `docs/frozen_numbers_mel80.md`.

## Thesis (one paragraph)

Character-level CTC forced alignment leaves blank gaps covering ~62% of utterance duration. Naively using `end − start` phone spans silently crops the training mel timeline to ~38% of its true length. The model then fits a truncated prefix with healthy loss values while free synthesis remains unintelligible (CER ≈ 0.86). Standard proxy metrics (val mel L1, duration heads, stop rules) do not flag the defect. Filling gaps (`sil_fill` or `expand_neighbor`) restores cover_ratio → 1.0 and drops oracle-duration CER from ~0.61 to ~0.03–0.05; scaling data then brings free CER to production levels (~0.048 on val100).

## Measured facts (frozen domain)

| Condition | cover_ratio | Oracle CER | Free CER | Artifact |
|-----------|------------:|-----------:|---------:|----------|
| Raw CTC (S1c) | 0.318 | 0.611 | 0.860 | `out/cover_ratio_s1d.json`, `out/intelligibility_mel_s1c*.json` |
| sil_fill (S1d) | 1.000 | 0.052 | ~0.87 free / 0.008 forced-lookup | `out/cover_ratio_s1d.json` |
| expand_neighbor (S1e) | 1.000 | 0.031 | 0.211 train-free | `out/cover_ratio_s1e.json` |
| GT mel → Griffin-Lim | — | — | 0.027 | `out/intelligibility_gt_mel_griffin.json` |
| S4 production free val100 | 1.000 | — | **0.048** | `out/intelligibility_mel_s4_val100.json` |

## Ablation arms (T08 / T10) — measured 2026-07-30

Artifacts: `out/paper2_seeds_summary.json`, `out/paper2_threeway.json`, figure `docs/figures/paper2_cover_vs_cer.png`.

S1 (32 utts) × 3 seeds; oracle-duration CER with bootstrap CI95:

| Arm | mean cover | mean oracle CER | CI95 |
|-----|-----------:|----------------:|------|
| raw CTC (blank gaps) | 0.335 | **0.501** | [0.427, 0.542] |
| sil_fill | 1.000 | 0.092 | [0.090, 0.096] |
| expand_neighbor | 1.000 | **0.012** | [0.008, 0.019] |

**One-sentence result:** Restoring timeline cover from ~0.34 → 1.0 cuts oracle CER by ~40× (`expand_neighbor`), while an independent MFA phone aligner already covers ~0.93 without gap surgery.

### MFA cross-check (T09, measured 2026-07-30)

| Metric | Value |
|--------|------:|
| MFA version | 3.4.1 (micromamba env `mfa`, user-local) |
| Models | `russian_mfa` acoustic + dictionary |
| Subset | 32 utts (same S1 scale) |
| TextGrids | **32 / 32** |
| `forced_mfa.json` | `~/datasets/tts/ruslan/align/forced_mfa.json` |
| Success rate | 1.00 |
| Mean cover_ratio (phone spans / dur_s) | **0.946** (threeway parse ≈0.925) |

MFA models silence explicitly; cover ≈1 without `expand_neighbor`. Full MFA-trained mel arm (train on MFA phones) is optional follow-up — cover evidence alone already separates MFA from raw CTC.

## Related open-source handling (to expand)

Survey TODO: how ESPnet / Coqui / Piper / Matcha / FastSpeech2 recipes treat CTC blank gaps when building duration targets — do they expand, insert silence, or silently crop?

## Reproducibility

```bash
git checkout paper2-frozen-mel80
# WSL
cd ~/projects/tts-self && source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
python scripts/35_paper2_seeds.py --seeds 11,22,33
python scripts/26_build_forced_align.py --method mfa --mfa-run --max-utts 32 \
  --out ~/datasets/tts/ruslan/align/forced_mfa.json
```

## Venues

SSW, reproducibility / negative-results workshops, ICASSP short.
