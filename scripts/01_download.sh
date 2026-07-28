#!/usr/bin/env bash
# Download models + RUSLAN stressed dataset.
set -euo pipefail
ROOT="${MINISPEECH_DATA:-$HOME/datasets/tts}"
mkdir -p "$ROOT"

export PATH="$HOME/.local/bin:$PATH"

echo "Data root: $ROOT"

# Hugging Face downloads
hf download microsoft/VibeVoice-AcousticTokenizer --local-dir "$ROOT/models/VibeVoice-AcousticTokenizer"
hf download Qwen/Qwen3.5-0.8B-Base --local-dir "$ROOT/models/Qwen3.5-0.8B-Base"

# Dry-run then download RUSLAN
hf download stilletto/ruslan-stressed --repo-type dataset --dry-run || true
hf download stilletto/ruslan-stressed --repo-type dataset --local-dir "$ROOT/ruslan"

echo "Done. RUSLAN at $ROOT/ruslan"
