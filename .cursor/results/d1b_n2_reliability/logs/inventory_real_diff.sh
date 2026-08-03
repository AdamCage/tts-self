#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/projects/tts-self
echo "COUNT_ALL=$(git diff --name-only HEAD | wc -l)"
echo "COUNT_REAL_W=$(git diff -w --name-only HEAD | wc -l)"
echo "=== real -w names under D1b paths ==="
git diff -w --name-only HEAD -- \
  src/minispeech/codec \
  src/minispeech/data/codec_tokens.py \
  src/minispeech/data/codec_seq_protocol.py \
  src/minispeech/data/collate_codec_seq.py \
  src/minispeech/data/d1b_oracles.py \
  src/minispeech/data/speaker_registry.py \
  src/minispeech/eval/discrete_n2.py \
  src/minispeech/infer/discrete_config.py \
  src/minispeech/models/discrete_speech_lm.py \
  src/minispeech/models/latent_io.py \
  src/minispeech/models/multi_codebook_head.py \
  src/minispeech/models/russian_conditioning.py \
  src/minispeech/models/text_init.py \
  src/minispeech/models/__init__.py \
  src/minispeech/train/discrete_lm_trainer.py \
  src/minispeech/train/__init__.py \
  scripts/70_probe_qwen3_tts_tokenizer.py \
  scripts/70b_external_baselines_d1.py \
  scripts/71_encode_codec_tokens.py \
  scripts/72_train_discrete_lm.py \
  scripts/73_infer_discrete_lm.py \
  scripts/74_eval_discrete_lm.py \
  scripts/75_freeze_d1b_oracles.py \
  scripts/76_select_discrete_checkpoint.py \
  tests/test_d1b_oracles.py \
  tests/test_discrete_n2_protocol.py \
  tests/test_discrete_speech_lm.py \
  tests/test_discrete_lm_trainer_steps.py \
  tests/test_discrete_lm_resume.py \
  tests/test_discrete_stop_config.py \
  tests/test_n2_collate_latent_seq.py \
  tests/test_speaker_registry.py \
  tests/test_codec_tokens_pack.py \
  tests/test_qwen3_tts_geometry.py \
  tests/_discrete_lm_trainer_fakes.py \
  configs/discrete_speech_lm_300m.yaml \
  configs/discrete_speech_lm_d1b_443m.yaml \
  configs/data_multicorpus.yaml \
  configs/eval_contracts \
  pyproject.toml
echo "=== -w stat for those ==="
git diff -w --stat HEAD -- \
  scripts/73_infer_discrete_lm.py \
  scripts/74_eval_discrete_lm.py \
  scripts/76_select_discrete_checkpoint.py \
  src/minispeech/models/latent_io.py \
  tests/test_d1b_oracles.py \
  tests/test_discrete_n2_protocol.py \
  tests/test_discrete_lm_trainer_steps.py \
  tests/test_n2_collate_latent_seq.py \
  src/minispeech/eval/discrete_n2.py \
  src/minispeech/data/d1b_oracles.py \
  src/minispeech/train/discrete_lm_trainer.py
echo "=== line-ending-only candidates (equal numstat, zero -w) ==="
while IFS=$'\t' read -r add del path; do
  [[ "$add" == "$del" ]] || continue
  if [[ -z "$(git diff -w --name-only HEAD -- "$path")" ]]; then
    echo "EOL_ONLY $path ($add/$del)"
  fi
done < <(git diff --numstat HEAD | awk -F'\t' 'NF==3 && $1!="-" && $2!="-" {print}')
echo "EOL_ONLY_COUNT done"
