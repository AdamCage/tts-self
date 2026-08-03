# FR candidate schedule (D1b §13)

Plan slug: `d1b_n2_reliability`  
Owner note from `build-fr-protocol` (selection/eval agent).

## Required immutable candidate steps

Trainer must save **immutable** candidate checkpoints at optimizer-step boundaries
(not named `best.pt`):

| Phase | Optimizer steps |
|-------|-----------------|
| Base (required) | `250`, `375`, `500`, `625`, `750` |
| DEV extension (optional) | `+1000`, `+1250` |

Suggested filenames: `candidate_step_0250.pt`, … (any stable non-`best` name).

## Selection contract

- Offline selection: `scripts/76_select_discrete_checkpoint.py`
- Ranking key (lexicographic): gate-eligible / no runaway → min macro CER →
  fewer catastrophic → smaller `|log(mean_dur_ratio)|` → teacher-forced CE
- Writes `checkpoint_selection.json` with all candidate metric rows
- **Only after selection** copy/symlink the winner to `best.pt`
- CONFIRM: score **`select8` only** for selection; do not decode `gate24`/`full32`
  until the checkpoint is frozen
- DEV: may score candidates on DEV32

## Code constants

See `minispeech.eval.discrete_n2.FR_CANDIDATE_STEPS_BASE` and
`FR_CANDIDATE_STEPS_DEV_EXT`.

If the trainer agent later wires the schedule into config YAML, this note remains
the normative checklist for D1b Wave A/B/C.
