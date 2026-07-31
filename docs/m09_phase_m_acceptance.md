# M09 — Phase M baseline and preference acceptance

## Artifacts

| Artifact | Role |
|----------|------|
| `out/baseline_m_critical.json` | New metrics baseline (M04–M06 + acoustic SA) |
| `docs/frozen_numbers_mel80.md` § Phase M | Frozen table |
| `out/m09_pref_dur/` | Duration mean_stats vs pred ballot (M02/M08) |
| `runs/vocoder_ft_v01/` | Short HiFi-GAN FT (M07, 2000 steps) |
| `out/m07_best_metrics.json` | Val mel / GAN losses |

## Why P0 DER/EER/SA are invalid

- DER collapsed to digit-weighted EER when Whisper emitted no Arabic digits.
- SA skipped all 895 homographs because stress-stripped lexemes are identical.

## Baseline snapshot (`--max-utts 50` except demo5)

See frozen table. Notable: bank DER **0.115** vs EER **0.240** (now distinct);
homograph acoustic SA **0.34** with **0** skipped on the scored subset.

## Blind A/B status

- **Duration:** ballot ready (`out/m09_pref_dur`). Trap check passes
  (`trap_same_rate=1.0` when traps answered `same`). Human listening not run
  in this session — preference CI requires listeners.
- **Vocoder:** FT checkpoint exists; copy-synth + preference ballot vs stock
  LibriTTS generator still needs a listening pass before claiming audible win.

## Incomplete by design / honesty

- Full-set (n=315/895/117) recompute without `--max-utts` not finished.
- Acoustic SA calibration first pass failed gate (≥0.90); word-local recal queued.
- Do not promote on UTMOS; do not retrain variance until M03 fixes land.
