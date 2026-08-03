# WSL disk audit + cleanup — 2026-08-03

## Before → After

| FS | Before Used | After Used | Before Avail | After Avail |
|----|-------------|------------|--------------|-------------|
| Ubuntu / | **401G** | **~219G** | 555G | **~738G** |
| Windows C: | 1.6T (~219G free) | 1.6T (**~309G** free) | — | improved ~90G |

Approx **~182G freed on WSL root** (401→219).

## Deleted (Linux) — major categories

| Category | ~GB freed |
|----------|-----------|
| 
uns/tokenizer_t0 + _small | ~58.5 |
| 
uns/discrete_speech_lm_300m_s1* ×4 | ~60 |
| GigaChat HF hubs (~/hf_cache) | ~34 |
| other old runs (n2/mel/paper2/vocoder/v01/v02) | ~19 |
| ~/.cache/uv (+ whisper) | ~15 |
| micromamba/pkgs | ~7.5 |
| ancient out/ listen demos | ~3.7 |
| Windows mirrored old 
uns/ | ~1.2 (helps C:) |

## Preserved (active D1b)

- 
uns/discrete_speech_lm_d1b_confirm_v2_seed42 (ACTIVE eval continuing)
- 
uns/discrete_speech_lm_d1b_confirm_seed42/43, d1b_dev_seed42
- out/d1b_n2_reliability/ (+ frozen)
- ~/datasets/tts codec_tokens
- .venv, .venv_qwen_tts
- VibeVoice / Qwen-TTS-Tokenizer / faster-whisper in ~/hf_cache

## Skipped (still eats space)

| Path | Size | Reason |
|------|------|--------|
| d1b confirm* + dev runs | ~150G | concurrent reconfirm — preserved by request |
| ~/datasets/tts | 39G | needed for train/eval |
| ~/.cache/huggingface (root-owned) | 3.0G | needs sudo rm (no passwordless sudo) |
| apt/journal | ~1.3G | needs sudo |
| Windows C:/Users/AdamCage/.cache/huggingface | **87G** | other-project models; not touched (careful on /mnt/c) |
| ~/models wav2vec2 | 4.3G | likely ASR/eval utility |

## Optional next (manual)

1. sudo rm -rf ~/.cache/huggingface and sudo apt-get clean && sudo journalctl --vacuum-size=100M
2. Review Windows HF hub (87G) for unused models (MOSS-SFX 11G, embeddings, duplicate TTS checkpoints)
3. After confirm_v2 finishes: optionally thin confirm_seed42/43 candidate *.pt (keep best + metrics) for another ~60–70G
4. Compact WSL VHDX from PowerShell (admin) so Windows sees reclaimed space: wsl --shutdown then Optimize-VHD / diskpart compact
