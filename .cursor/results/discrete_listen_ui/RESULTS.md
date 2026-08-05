# Results: discrete_listen_ui

- **Plan:** `.cursor/plans/discrete_listen_ui_08f504c8.plan.md` (Cursor plans)
- **Date:** 2026-08-04
- **Git SHA (author copy):** `0661d10`
- **Status:** GO (tooling delivered; GPU listen not run in this session)

## What shipped

Portable directory [`listen_service/`](../../listen_service/) — Gradio free-text listen for D1b discrete Speech LM without Docker.

| Piece | Path |
|-------|------|
| Entry / UI | `listen_service/app.py` |
| Frontend | `listen_service/runtime/text.py` |
| Load bundle | `listen_service/runtime/bundle.py` |
| Timed synth | `listen_service/runtime/synthesize.py` |
| Env template | `listen_service/.env.example` |
| Launch | `listen_service/run.sh` |
| Docs | `listen_service/README.md` |
| Infer defaults | `listen_service/configs/infer_defaults.yaml` |
| Speaker defaults | `listen_service/configs/speaker_ruslan.json` |
| Unit test | `listen_service/tests/test_text.py` |

Default ckpt target: `runs/discrete_speech_lm_d1b_confirm_v2_seed43/best.pt` (or `CKPT_PATH` / `artifacts/best.pt`).

## Tests

```text
pytest listen_service/tests/test_text.py -q
→ 2 passed
```

Log: [logs/pytest_text.log](logs/pytest_text.log)

## Inference

Not run in this session (Windows authoring host; GPU path is WSL). After install on WSL:

```bash
cd ~/projects/tts-self && source .venv/bin/activate
pip install -r listen_service/requirements.txt
cp listen_service/.env.example listen_service/.env
# set CKPT_PATH / TOKENIZER_PATH
./listen_service/run.sh
```

Wavs will land under `listen_service/out/`.

## Methodology

Read-only during implementation (tooling only; no methodology edits). Aligns with Wave-1 text_ids tokenizer (Qwen3.5-0.8B-Base) and frozen Qwen3-TTS 12Hz decode.

## Follow-ups

- WSL smoke: `--preload` + one Generate on CUDA and CPU
- Optional: mirror listen wavs to `out/discrete_listen_ui/` if tying into plan-slug playback habit
