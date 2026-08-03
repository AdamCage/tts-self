# D1b trainer config keys (repair-trainer)

Owned by `repair-trainer`. The D1b YAML (`configs/discrete_speech_lm_d1b_443m.yaml`)
is created by another agent; these keys must be present / honored.

## Step budget (optimizer updates)

| Key | Meaning |
|-----|---------|
| `train.stages[].steps` / `optimizer_steps` | Horizon in **optimizer updates** |
| `train.optimizer_steps` / `max_optimizer_steps` | Default when stage omits steps (fallback **750**) |
| `train.allow_dev_extension` | Opt-in DEV-only extension |
| `train.dev_extension_optimizer_steps` | Extension horizon (default **1250**) |
| `train.stages[].dev_extension` | Per-stage opt-in for the extension |

Primary D1b budget: **750 optimizer updates** (legacy D1 `steps: 3000` × `grad_accum: 4`).
Do **not** auto-run 3000 optimizer updates.

## DDP

| Key | Meaning |
|-----|---------|
| `train.ddp.find_unused_parameters` | Passed to `wrap_ddp` (default `false`) |

RU conditioning off ⇒ trainer freezes `ru_cond` params (`requires_grad=False`) so
they are not trainable-unused under DDP.

## Resume / repro hashes

| Key | Meaning |
|-----|---------|
| `train.speaker_map_hash` or `train.speaker_map_path` | Stored + checked on resume |
| `train.subset_hash` or `train.subset_path` | Stored + checked on resume |
| `train.eval_contract_hash` or `train.eval_contract_path` | Stored + checked on resume |
| `train.allow_legacy_microbatch_resume` | Migrate D1 ckpts (`legacy_step_unit=microbatch`) |

Checkpoint `step_unit` is always `optimizer`. Mid-accumulation saves are refused;
interruptions resume from the last boundary checkpoint.

## Smoke paths

```bash
# Single-process DDP init smoke (no promotion train)
torchrun --standalone --nproc_per_node=1 scripts/72_train_discrete_lm.py \
  --config <d1b_or_300m_yaml> --stage S1 --steps 1 --allow-dirty \
  --codec-tokens-dir <codec_tokens_root>

# Optional CPU/gloo two-process unit test
MINISPEECH_DDP_GLOO=1 pytest -q tests/test_discrete_lm_resume.py::test_ddp_gloo_two_process_smoke
```

RESULTS wording later: “DDP logic tested, multi-GPU hardware not measured”.
