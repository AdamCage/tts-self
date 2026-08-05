# Discrete listen service

Portable **FastAPI** UI for manual free-text listening of the D1b discrete Speech LM
checkpoint. **No Docker** — copy this directory + checkpoint artifact, install
`minispeech` from the parent repo, run.

## Pipeline

```text
RU text → TN/stress → Qwen3.5-0.8B tokenizer → DiscreteSpeechLM.generate
        → frozen Qwen3-TTS Tokenizer-12Hz decode → 24 kHz wav + metrics
```

Default speaker id: **1088** (`ruslan`). This is a smoke listen tool — it does
**not** enforce the D1b N2 eval contract.

## Layout on a VM

```text
/opt/tts-listen/
  tts-self/                         # git clone of this repo
  artifacts/best.pt                 # seed43 confirm_v2 checkpoint
  models/Qwen3.5-0.8B-Base/         # text tokenizer (or HF path)
  .venv/                            # optional dedicated venv
```

Suggested checkpoint source (WSL / training host):

`runs/discrete_speech_lm_d1b_confirm_v2_seed43/best.pt`

## Install (no Docker)

```bash
cd /path/to/tts-self
python -m venv listen_service/.venv
source listen_service/.venv/bin/activate
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"

# Parent package (model + codec wrapper + frontend)
pip install -e ".[qwen_tts,stress]"

# Service UI deps
pip install -r listen_service/requirements.txt

cp listen_service/.env.example listen_service/.env
# Edit CKPT_PATH / TOKENIZER_PATH / DEVICE

# Optional: copy checkpoint into the portable artifacts dir
# cp /path/to/best.pt listen_service/artifacts/best.pt
```

Use the same environment where `scripts/73_infer_discrete_lm.py --decode`
already works (`qwen-tts` pulls `transformers~=4.57`, which conflicts with the
VibeVoice path).

## Run

```bash
uv pip install -r listen_service/requirements.txt --python listen_service/.venv/Scripts/python.exe
listen_service\.venv\Scripts\python.exe listen_service\app.py --host 127.0.0.1 --port 7860
```

Model **preloads at startup** (stdout: `model loaded on {device}`). Device:
CUDA if available, else CPU. Open `http://127.0.0.1:7860` — simple HTML form
posts to `/synthesize` and plays wav + shows metrics JSON.

Wavs are also written under `listen_service/out/` (ASCII filenames).

## Environment

| Variable | Meaning |
|----------|---------|
| `CKPT_PATH` | Path to `best.pt` |
| `TOKENIZER_PATH` | Qwen3.5-0.8B-Base (vocab 248077) |
| `CODEC_MODEL_ID` | Optional local Qwen3-TTS Tokenizer-12Hz |
| `DEVICE` | `cuda` or `cpu` |
| `SPEAKER_ID` | Default `1088` |
| `HOST` / `PORT` | FastAPI bind |
| `HF_HOME` | Hugging Face cache |

## What to copy for another VM

1. This `listen_service/` tree (or full `tts-self` clone).
2. The checkpoint `best.pt`.
3. Text tokenizer directory (or ensure HF cache has it).
4. Qwen3-TTS 12Hz weights (local dir or HF cache).

Do **not** commit `*.pt`, HF weights, or `.env`.
