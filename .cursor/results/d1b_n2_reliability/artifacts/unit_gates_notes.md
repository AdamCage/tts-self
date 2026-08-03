# Unit/integration gate — pass-unit-gates

Date: 2026-08-03
Plan slug: d1b_n2_reliability
HEAD (dirty): 9b53d0e

## Command

```
PYTHONPATH=src python -m pytest -q \
  tests/test_qwen3_tts_geometry.py \
  tests/test_codec_tokens_pack.py \
  tests/test_discrete_speech_lm.py \
  tests/test_speaker_registry.py \
  tests/test_discrete_lm_trainer_steps.py \
  tests/test_discrete_lm_resume.py \
  tests/test_discrete_n2_protocol.py \
  tests/test_discrete_stop_config.py
```

Result: **78 passed, 3 skipped**

Skips (expected):
- `test_discrete_speech_lm.py` CUDA-only case
- `test_ddp_world_size_1_smoke` (opt-in) — ran separately with `MINISPEECH_DDP_SMOKE=1` → **passed**
- `test_ddp_gloo_two_process_smoke` (opt-in `MINISPEECH_DDP_GLOO=1`) — not required for green gate

Full tee: `logs/unit_gates.log`; DDP smoke: `logs/unit_gates_ddp_smoke.log`

## Integration seams fixed

- `discrete_n2.resolve_infer_settings` now calls `resolve_discrete_infer_settings` (removed CLI/JSON stub)
- `scripts/74_eval_discrete_lm.py` uses shared GENERATE keys / resolver; `--speaker-map` + `model.speaker.map_path`
- `scripts/72` / `73` speaker-map path hooks include train + model.speaker.map_path
- Trainer: run-dir copy of speaker_map + hashes/resolved_config; immutable `candidate_step_XXXX.pt`
- YAML: `max_speakers: 1089`, optimizer-step budget keys, FR candidates 250/375/500/625/750 (+ext), WAVE3c infer
- `test_forced_stop_deterministic` re-enabled (RoPE `position_ids` assert, no skip)

## GPU phase blockers (outside unit gate)

- Working tree is **not clean** — D1b sources are largely untracked; methodology + plan diffs present.
- Hard-stop rule: no clean reproducible SHA until a deliberate commit of D1b code/config (not done here).
- Do not start `run-dev-confirm` until that clean SHA exists.
