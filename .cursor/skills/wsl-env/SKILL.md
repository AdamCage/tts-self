---
name: wsl-env
description: >-
  Sets up and verifies MiniSpeech-RU WSL2/CUDA environment (uv venv, HF_HOME,
  local model paths, editable install). Use when onboarding, env check fails,
  CUDA/import errors, Hugging Face cache permission issues, or the user
  mentions WSL, venv, or setup.
---

# WSL environment

## Canonical layout

```text
Windows:  C:\projects\tts-self
WSL:      /mnt/c/projects/tts-self
venv:     .venv (Python 3.11)
data:     ~/datasets/tts/
HF cache: ~/hf_cache   # avoid root-owned ~/.cache/huggingface
```

## Bootstrap

```bash
wsl -d Ubuntu
cd /mnt/c/projects/tts-self
source ~/.local/bin/env   # uv
uv venv --python 3.11 .venv
source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
uv pip install -e ".[dev]"
python scripts/00_env_check.py
```

## Common failures

| Symptom | Fix |
|---------|-----|
| `Permission denied` under `~/.cache/huggingface` | `export HF_HOME=$HOME/hf_cache` |
| Code edits not visible | `uv pip install -e . --no-deps` |
| Missing `Qwen3_5TextModel` / VibeVoice | upgrade `transformers` |
| Slow Qwen DeltaNet | install `causal_conv1d` + `fla` (Phase 5+) |
| PowerShell eats quotes | use `wsl -d Ubuntu --cd /mnt/c/projects/tts-self -e bash -lc 'cmd'` |

## Models

```bash
hf download microsoft/VibeVoice-AcousticTokenizer \
  --local-dir $HOME/datasets/tts/models/VibeVoice-AcousticTokenizer
hf download Qwen/Qwen3.5-0.8B-Base \
  --local-dir $HOME/datasets/tts/models/Qwen3.5-0.8B-Base
```

Point configs: `configs/qwen_local.yaml`, codec wrapper prefers local tokenizer dir.
