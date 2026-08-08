# PROTOCOL — D1c Semantic EOS Repair

## Isolation

- Clean worktree: `C:\projects\tts-self-d1c`
- Branch: `d1c/semantic-eos` from `origin/master` @ `0661d10`
- Authoring dirty tree `C:\projects\tts-self` left untouched (foreign inventory in `logs/wave0_foreign_inventory.txt`)

## Frozen inventories

```bash
# DEV32 (already done)
python scripts/77_freeze_d1c_oracles.py \
  --manifest ~/datasets/tts/multicorpus/manifest.jsonl \
  --out configs/train_subsets/d1c_dev32.json
```

## Tests (pre-GPU)

```bash
cd ~/projects/tts-self
source .venv/bin/activate
export PYTHONPATH=$PWD/src
python -m pytest -q
```

## Resume status

1. Commit + push `d1c/semantic-eos` — **done** (through `74d4caa` + results pack).
2. Custodian commitment — **done** (`c15ed40`):  
   `commitment=4c46d418caf9bcabc088864da5bf0b3de0475a9d9225f50383314a2c5e3bb3f2`  
   plaintext seed chat-only; `plaintext_seed_present=false`.
3. Wave 3 GPU — **PASS** @ `74d4caa` (torchrun + 1-utt + WAVs). See `WAVE3_REPORT.md`.
4. Next: Wave 4 DEV W1 seed101 (after optional `clean_sha_available` reseal).

## WSL sync template

```bash
cd ~/projects/tts-self
unset GIT_DIR GIT_WORK_TREE
git fetch origin
git checkout d1c/semantic-eos
git pull --ff-only origin d1c/semantic-eos
test -z "$(git status --porcelain)"
source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
uv pip install -e . --no-deps
```

## Canonical train template (Wave 4+)

```bash
python scripts/72_train_discrete_lm.py \
  --config configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml \
  --eval-contract configs/eval_contracts/d1c_semantic_eos_v2_0_0.json \
  --subset configs/train_subsets/d1c_dev32.json \
  --subset-split full32 \
  --run-name d1c_eos_dev_w1_seed101 \
  --seed 101 \
  --stage S1 \
  --codec-tokens-dir "$HOME/datasets/tts/ruslan/codec_tokens,$HOME/datasets/tts/common_voice_ru17/codec_tokens"
```

Do **not** pass `--allow-dirty` on formal DEV once `repro.clean_sha_available=true` is sealed.
