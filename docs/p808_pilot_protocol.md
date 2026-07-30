# ITU-T P.808 listening pilot protocol (Paper1 T16)

**Status:** protocol only — **no listening scores have been collected**. Do not invent MOS.

## Goal

Small-n Absolute Category Rating (ACR) pilot to check whether UTMOS rankings agree with human preference on MiniSpeech-RU mel arms.

## Conditions (freeze set)

Use the same text set for all systems (prefer `configs/eval_sets/demo5.json` + 10 val utts):

| ID | System | Wav source |
|----|--------|------------|
| A | GT recording | RUSLAN wav @ 22.05 kHz |
| B | mel80 S4 + HiFi-GAN | `runs/mel_baseline_v01_s4` suite wavs |
| C | stress S4 (if Gate V4 run) | `runs/mel_stress_v01_s4` |
| D | Optional: Griffin-Lim of B | same mel, GL vocoder |

Do **not** include impersonation / celebrity / third-party voice cloning.

## Procedure (summary)

1. Loudness-normalize to −23 LUFS (or peak −1 dBFS) with the same tool for all files.
2. Randomize order per listener; blind condition labels.
3. ACR 1–5 (P.808 Absolute Category Rating); optional CMOS pairwise vs GT.
4. Target **n ≥ 10** naïve RU listeners for the pilot (not publication-grade).
5. Report mean / CI only after collection; store anonymized CSVs under `out/p808_pilot/` (gitignored).

## Acceptance for “pilot done”

- Spreadsheet or CSV with listener id, item id, condition, score.
- Computed mean MOS ± 95% CI per condition.
- Explicit comparison to UTMOS on the **same wavs** (no cross-set mixing).

## Blocker

Requires human listeners and ethics/consent for a research pilot. Automation cannot complete this item.
