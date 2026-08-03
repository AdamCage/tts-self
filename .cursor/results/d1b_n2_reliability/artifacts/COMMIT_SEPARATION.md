# Commit separation plan (no commit created)

Snapshot (recoverable): `.cursor/snapshots/d1b_repro_preflight_20260803_081443/`  
Moved out of tree: corrupted HF-cache path `C:UsersAdamCage/` → snapshot `moved_from_repo/`.

**Explicit user commit permission was not given.** Nothing was committed.

## Recommended commit groups (when user authorizes)

### A — D1 discrete Speech LM code (needed for D1b repairs)

- `src/minispeech/codec/qwen3_tts_wrapper.py` (+ `codec/__init__.py`)
- `src/minispeech/constants.py`
- `src/minispeech/data/codec_tokens.py`
- `src/minispeech/data/codec_seq_protocol.py`
- `src/minispeech/data/collate_codec_seq.py`
- `src/minispeech/models/{discrete_speech_lm,multi_codebook_head,russian_conditioning,text_init}.py` (+ `models/__init__.py`)
- `src/minispeech/train/discrete_lm_trainer.py` (+ `train/__init__.py`)
- `scripts/70_probe_qwen3_tts_tokenizer.py`
- `scripts/70b_external_baselines_d1.py`
- `scripts/71_encode_codec_tokens.py`
- `scripts/72_train_discrete_lm.py`
- `scripts/73_infer_discrete_lm.py`
- `scripts/74_eval_discrete_lm.py`
- `configs/discrete_speech_lm_300m.yaml`
- `configs/data_multicorpus.yaml` (codec_tokens_dir paths)
- `pyproject.toml` (if dependency pins for D1)
- `tests/test_{codec_tokens_pack,discrete_speech_lm,qwen3_tts_geometry}.py`

### B — D1b preflight freezes (this phase)

- `scripts/75_freeze_d1b_oracles.py`
- `configs/train_subsets/d1b_dev32.json`
- `configs/train_subsets/d1b_confirm32.json`

Optional (large / generated; keep under `out/` or results, usually not required in git):

- `out/d1b_n2_reliability/frozen/**`
- `.cursor/results/d1b_n2_reliability/**`

### C — Do **not** mix into promotion SHA without separate review

- `.cursor/methodology/**` (dirty planning edits; **do not edit during D1b execution**)
- `.cursor/plans/**`
- `.cursor/results/d1_discrete_speech_lm/**` (D1 results pack; OK as docs but separate)
- `.cursor/results/n2_first_execution/**`
- `configs/_tmp_discrete_s1*.yaml` (local train overrides)
- HF cache / lock junk (already moved from authoring copy)
- `tmp_*` scratch scripts at repo root

## Hard-stop note for later phases

| Requirement | Status |
|-------------|--------|
| Clean reproducible `git rev-parse HEAD` for WSL `git pull` | **BLOCKED** — tree dirty; no authorized commit |
| `MINISPEECH_ALLOW_DIRTY=1` for canonical gates | **Forbidden** |
| Speaker map hash in eval contract | **pending** `repair-speaker-rope` |
| Shared infer resolver | **pending** `unify-stop-config` |

Until a clean SHA containing at least groups A+B (plus subsequent repair commits) exists and is pulled into `~/projects/tts-self`, promotion GPU waves (`run-dev-confirm`) must hard-stop.
