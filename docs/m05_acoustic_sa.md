# M05 — Acoustic stress accuracy (SA)

## Goal

Replace ASR-lexeme SA (always `skipped` on stress-only homographs) with an
acoustic detector: CTC align → vowel spans → duration / energy / F0 score.

## Code

- `src/minispeech/eval/stress_acoustic.py`
- `scripts/53_calibrate_acoustic_sa.py`
- Suite flag: `scripts/32_eval_mel_suite.py --acoustic-sa`
- Tests: `tests/test_stress_acoustic.py`

## Calibration result (2026-07-31)

Artifact: `out/m05_acoustic_sa_calibration.json`

| Pass | n_scored | accuracy | pass ≥0.90 |
|------|---------:|---------:|:----------:|
| v1 utterance-global gold | 110 | 0.109 | no |
| v2 word-local gold | **779** words / 110 utts | **0.358** | **no** |

Best weights (both passes): `w_dur=0.5`, `w_energy=0.5`, `w_f0=1.5`.

**Verdict:** detector is **not fit as a promotion gate** (need ≥0.90). Suite may
still emit acoustic SA for diagnostics (`--acoustic-sa`); on M09 subset
homograph SA=0.34 with 0 skipped — better than ASR-lexeme all-skipped, but not
trustworthy enough to block/ship. Further work: better vowel windowing from
forced-align phones, or supervised stress classifier.

## Decision rule

Among vowels in the target word window, pick argmax of
`w_dur·z(dur) + w_energy·z(energy) + w_f0·z(f0)`.
Label `correct` / `incorrect` / `skipped` (align failure or <2 vowels).
