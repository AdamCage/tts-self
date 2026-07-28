# MiniSpeech-RU — Russian TTS research prototype

Architecture (v0.2 target):

```text
stressed RU text → offline Qwen3.5-0.8B text hidden cache
  → Talker (causal + cross-attn) → regression | rectified-flow head
  → VibeVoice acoustic latents [64] @ 7.5 Hz → cached decoder
```

Hardware target: RTX 4080 / 4080 Super 16 GB (WSL2 Ubuntu).

This is a research prototype. Do not use for impersonation or fraud.
See `docs/` for conversation notes and the VibeVoice technical report.

**Agents:** start at [AGENTS.md](AGENTS.md).

## Quick start (WSL2)

```bash
cd /mnt/c/projects/tts-self
source ~/.local/bin/env
uv sync
uv run python scripts/00_env_check.py
```

## Agent scaffolding

| Path | Purpose |
|------|---------|
| [AGENTS.md](AGENTS.md) | Entry point for coding agents |
| `.cursor/rules/` | Persistent Cursor rules |
| `.cursor/skills/` | Phase/workflow skills |
| `.cursor/hooks.json` | sessionStart + shell safety hooks |

## Phase scripts

| Script | Purpose |
|--------|---------|
| `00_env_check.py` | CUDA / versions / FA kernels |
| `01_download.sh` | Models + RUSLAN |
| `02_build_manifest.py` | JSONL manifest + split |
| `03_verify_text_asr.py` | ASR text verification |
| `04_encode_latents.py` | Offline `sample=False` encode |
| `05_cache_text_hidden.py` | Offline Qwen hidden cache |
| `06_align.py` | Alignment prior A/B/C |
| `07_train.py` | Training entrypoint |
| `08_infer.py` | Inference |
| `09_eval.py` | WER / latency / text-usage gates |
| `10_smoke_e2e.py` | End-to-end smoke |
| `11_phase0_feasibility.py` | Codec GO/NO-GO |

## License / data

- Code: Apache-2.0
- Qwen3.5-0.8B-Base: Apache-2.0
- VibeVoice Acoustic Tokenizer: MIT
- RUSLAN Stressed: CC BY 4.0 (attribution required)

Raw datasets are not committed. Generated speech can be misused; use responsibly.
