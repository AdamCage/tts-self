#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/projects/tts-self
echo "SHA=$(git rev-parse HEAD)"
echo "SHORT=$(git rev-parse --short HEAD)"
echo "AHEAD=$(git rev-list --count origin/master..HEAD)"
echo "PORCELAIN=$(git status --porcelain | wc -l)"
echo "=== D1b-related ==="
git status --porcelain | grep -E 'discrete|d1b|codec_tokens|codec_seq|multi_codebook|russian_conditioning|text_init|latent_io|speaker_registry|qwen3_tts|train_subsets/d1b|eval_contracts/d1b|speaker_maps/trackn|scripts/7[0-6]|test_d1b|test_discrete|test_codec|test_qwen3|test_n2_collate|test_speaker|models/__init__|train/__init__|codec/__init__|data/__init__|constants\.py|pyproject' || true
echo "=== EXCLUDE methodology/tmp/danger ==="
git status --porcelain | grep -E '_tmp_|methodology/|/runs/|^.. out/|\.pt$|\.wav$|\.safetensors|hf_cache|C:Users|snapshots/' || true
echo "=== results d1b ==="
git status --porcelain | grep 'd1b_n2_reliability' || true
echo "=== plans ==="
git status --porcelain | grep '\.cursor/plans/' || true
echo "=== untracked count by prefix ==="
git status --porcelain | awk '/^\?\?/ {print $2}' | awk -F/ '{print $1"/"$2}' | sort | uniq -c | sort -nr | head -40
