# Wave 2 — semantic EOS implementation

Status: **INVALID_RUN → infra repaired** (not PASS until human re-audit).  
Prior claim withdrawn: residual CE was detached, immediate-EOS left a phantom frame, and CLI/eval wiring was incomplete. See `REPAIR_AUDIT.md` blockers 1, 4, 5.

CPU implementation and focused compatibility tests complete. No GPU
training, synthesis, checkpoint promotion, or methodology edits were performed.

## Implemented contract

- Added `semantic_eos_v1` alongside explicit legacy `frame_stop_v1`.
- CB0 semantic head is 2049 classes only in EOS mode; `speech_emb` and all
  residual heads remain 2048 real-code rows/classes.
- Collation emits per-row `semantic_targets`, `semantic_mask`, `eos_positions`,
  `codec_frame_mask`, and `residual_mask`; codec frames remain real-only.
- EOS teacher forcing uses one extra semantic slot, with the final real CB0 as
  its input. EOS has no residual target, no residual prediction, and no
  embedding/decoder frame.
- Added EOS loss/diagnostic keys and structured canonical generation metadata.
  The decoder returns only real frames; safety termination is marked explicitly.
- Added D1c W1/W2 configs. W2's only effective recipe difference is
  `loss.eos_loss_weight: 2.0`.
- D1c checkpoints store `minispeech.discrete_lm_checkpoint/2.0.0` and a
  termination/vocabulary contract; semantic resume rejects legacy schema,
  mismatched contracts, and a legacy `stop_head` state.

## Verification

Executed in Ubuntu WSL with:

```text
PYTHONPATH=/mnt/c/projects/tts-self-d1c/src python -m pytest -q \
  tests/test_d1c_semantic_eos.py \
  tests/test_discrete_stop_config.py \
  tests/test_discrete_speech_lm.py
```

Result: **34 passed** in 3.52 seconds.

## Scope notes

This wave intentionally did not run the Wave 3 GPU smoke, make a checkpoint,
or synthesize WAVs. Those tasks remain gated on the subsequent clean committed
source workflow.
