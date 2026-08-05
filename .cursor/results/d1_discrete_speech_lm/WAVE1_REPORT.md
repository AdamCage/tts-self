# Wave 1 — discrete codec-token caches (`build-discrete-caches`)

**Date:** 2026-08-02  
**Plan:** `.cursor/plans/d1-discrete-speech-lm_860c29d5.plan.md`  
**Slug:** `d1_discrete_speech_lm`  
**Git (authoring / D0 base):** `dce4f45` (dirty authoring tree; `MINISPEECH_ALLOW_DIRTY=1`)  
**Status:** **DONE** — code, tests, S1/1h subsets, and full Track N train encode

## Verdict

Wave 1 deliverables for packing / I/O / provenance / speaker-disjoint checks are
landed and unit-tested. Deterministic Qwen3-TTS 12Hz encode completed for the
full speaker-disjoint Track N train (**51234** utts / **72.9836** h) under a
dedicated `qwen-tts` venv (`transformers==4.57.3`), plus verified `s1_32` and
`h1` subset shards for the oracle ladder.

## Code / config added

| Path | Role |
|------|------|
| `src/minispeech/data/codec_tokens.py` | `[T,16]` shard I/O, RU features, train/infer pack + EOS, Track N leak check, `CodecTokenDataset` |
| `src/minispeech/data/collate_codec_seq.py` | Batch collate for text∥codes∥EOS + pack-equality helpers |
| `scripts/71_encode_codec_tokens.py` | Frozen encode CLI (resume-safe, multicorpus, provenance meta) |
| `configs/data_multicorpus.yaml` | Per-corpus `codec_tokens_dir` |
| `tests/test_codec_tokens_pack.py` | Pack / collate / shard / leak unit tests |

Does **not** reuse VibeVoice latent or text-hidden caches.

## Env (two-pass / dedicated venv)

| Env | Path | Pin |
|-----|------|-----|
| Main (VibeVoice) | `~/projects/tts-self/.venv` | `transformers==5.14.1` |
| **Encode (this wave)** | `~/projects/tts-self/.venv_qwen_tts` | `transformers==4.57.3` + `qwen-tts==0.1.1` |

```bash
export HF_HOME=$HOME/hf_cache_qwen   # symlink → Tokenizer-12Hz snapshot
export PYTHONPATH=/mnt/c/projects/tts-self/src
export MINISPEECH_ALLOW_DIRTY=1
source ~/projects/tts-self/.venv_qwen_tts/bin/activate
```

Codec fingerprint (partial): `partial_sha256_16=7030d98a18ae5968`,
`n_state_keys=496`, license Apache-2.0, target `qwen3tts_12hz`.

## Track N integrity

Artifact: `out/d1_discrete_speech_lm/encode_verify.json`  
(also `artifacts/encode_verify.json`)

| Check | Result |
|-------|--------|
| utt train∩val / train∩test / val∩test | **0 / 0 / 0** |
| speaker leaks | **none** |
| train | **51234** utts / **72.9836** h |
| val / test / review | 2961 / 3276 / 2430 utts |

## Encoded subsets (verified)

| Subset | n utts | hours | Meta | Shard indexes |
|--------|------:|------:|------|---------------|
| `s1_32` | 32 | 0.0494 | `out/d1_discrete_speech_lm/encode_meta_s1_32.json` | `~/datasets/tts/{ruslan,common_voice_ru17}/codec_tokens_s1_32/codec_tokens_index.json` |
| `h1` | 691 | 0.9990 | `out/d1_discrete_speech_lm/encode_meta_h1.json` | `~/datasets/tts/{ruslan,common_voice_ru17}/codec_tokens_h1/codec_tokens_index.json` |
| `train_full` (canonical) | **51234** | **72.9836** | `out/d1_discrete_speech_lm/encode_meta_train_full.json` | `~/datasets/tts/{ruslan,common_voice_ru17}/codec_tokens/codec_tokens_index.json` |

Coverage (canonical full train):

| Corpus | n utts | hours |
|--------|------:|------:|
| `common_voice_ru_17_hf_mirror` | 39199 | 57.9533 |
| `ruslan` | 12035 | 15.0303 |
| **total** | **51234** | **72.9836** |

`h1` breakdown: CV17 **534** + RUSLAN **157**.  
Text tokenizer: `/home/adam/datasets/tts/models/Qwen3.5-0.8B-Base`.  
`speech_eos_id`: **2048** (outside codebook 0..2047).  
`codec_hash`: `7030d98a18ae5968`.  
Full-train wall time: **1640.7 s** (~27.3 min) ≈ 31 utt/s on RTX 4080 Super.

## Tests

```text
pytest tests/test_codec_tokens_pack.py -q
7 passed  (Windows + WSL .venv)
```

Covers: pack shapes/EOS/AR-shift, train≡infer pack equality, collate order
invariance, shard roundtrip + provenance, RU stress/span projection, Track N
no-leak fixture.

## Full encode — complete / resume recipe

**Finished** in tmux `d1_encode` (session closed).  
**log:** `.cursor/results/d1_discrete_speech_lm/logs/w1_encode_train_full.log`  
**meta:** `out/d1_discrete_speech_lm/encode_meta_train_full.json`

Re-run is idempotent (skips indexed utt_ids) if shards ever need repair:

```bash
wsl -d Ubuntu -- bash -lc '
  cd ~/projects/tts-self && source .venv_qwen_tts/bin/activate
  export HF_HOME=$HOME/hf_cache_qwen
  export PYTHONPATH=/mnt/c/projects/tts-self/src
  export MINISPEECH_ALLOW_DIRTY=1
  python /mnt/c/projects/tts-self/scripts/71_encode_codec_tokens.py \
    --config /mnt/c/projects/tts-self/configs/data_multicorpus.yaml \
    --qwen-config /mnt/c/projects/tts-self/configs/qwen_local.yaml \
    --eval-set /mnt/c/projects/tts-self/configs/eval_sets/trackn_train.json \
    --meta-out /mnt/c/projects/tts-self/out/d1_discrete_speech_lm/encode_meta_train_full.json
'
```

## Packing contract (train == infer)

```text
[ text_ids (L)  ||  codes [T,16]  ||  EOS ]
```

- Parallel RU channels (`stress`, `phoneme`, `span_type`, `word_bound`,
  `phrase_bound`) align to the **text prefix only** via BPE offset mapping.
- Acoustic AR shift: `codes_input[0]=0`, `codes_input[t]=codes[t-1]`.
- Shard fields per utt: `codes`, `text_ids`, + five feature tensors; index meta
  stores corpus / speaker / license / codec_id / codec_hash / dur_s.

## Blockers for oracle ladder (`pass-oracle-n2`)

1. **Speech LM code** — parallel todo `implement-speech-lm` (not this wave);
   point trainer at canonical `codec_tokens` dirs (or `codec_tokens_s1_32` for
   the 32-utt oracle).
2. **Text-init tokenizer choice** — caches used Qwen3.5-0.8B-Base BPE; if LM
   text-init is Qwen3-0.6B, re-tokenize text ids with that tokenizer (codes stay
   valid; only `text_ids` + RU feature alignment need refresh).
3. **Env split** — keep encode on `.venv_qwen_tts`; do not mix with VibeVoice
   `transformers>=5` train env unless pins are reconciled.
4. No 500 h Golos/YODAS in D1 (by plan).
5. Optional: encode Track N val/test the same way before held-out eval (train
   only was required for this wave).

## Logs

| Log | Path |
|-----|------|
| verify | `.cursor/results/d1_discrete_speech_lm/logs/w1_verify.log` |
| S1 | `.cursor/results/d1_discrete_speech_lm/logs/w1_encode_s1_32.log` |
| 1h | `.cursor/results/d1_discrete_speech_lm/logs/w1_encode_h1.log` |
| full | `.cursor/results/d1_discrete_speech_lm/logs/w1_encode_train_full.log` |

## Methodology

Read-only during this todo. Aligns with
`.cursor/methodology/{04_tokenizers,03_architecture,09_roadmap}.md` Wave 1
cache contract. No methodology edits.
