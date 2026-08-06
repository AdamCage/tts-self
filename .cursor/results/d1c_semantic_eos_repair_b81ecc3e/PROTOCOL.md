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
cd /mnt/c/projects/tts-self-d1c   # or Linux clone after push
export GIT_DIR=/mnt/c/projects/tts-self/.git/worktrees/tts-self-d1c
export GIT_WORK_TREE=/mnt/c/projects/tts-self-d1c
source ~/projects/tts-self/.venv/bin/activate
export PYTHONPATH=$PWD/src
python -m pytest -q
```

## Resume after authorization

1. Commit + push `d1c/semantic-eos` from clean worktree. (**done:** `ff9d8c7`, `27164af`)
2. Custodian fills `configs/train_subsets/d1c_confirm_seed_commitment.json` commitment field.
   **STILL BLOCKING:** `"commitment": null` — Wave 3 GPU / DEV hard-stopped (plan §18).
3. WSL:

```bash
cd ~/projects/tts-self
git fetch origin
git checkout d1c/semantic-eos
git pull --ff-only origin d1c/semantic-eos
test -z "$(git status --porcelain)"
source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
uv pip install -e . --no-deps
```

4. After commitment lands: finish Wave 3 GPU (`torchrun` smoke + 1-utt) then Wave 4 DEV W1 seed101 per plan §14–16.
   Wave 3 CPU gates already PASS @ `27164af` — see `WAVE3_REPORT.md`.


## Canonical train template (after resume)

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

Exact `--formal` flags must be confirmed against the landed CLI before the first formal run.
