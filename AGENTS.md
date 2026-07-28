# AGENTS.md — MiniSpeech-RU

Instructions for coding agents working in this repository.

## Project

Russian TTS research prototype on **RTX 4080 / 4080 Super 16 GB (WSL2)**:

```text
stressed RU text → offline Qwen3.5-0.8B text hidden
  → Talker (causal + cross-attn) → regression | rectified-flow head
  → VibeVoice latents [64] @ 7.5 Hz → frozen cached decoder
```

Primary docs: [README.md](README.md), [docs/conversation.md](docs/conversation.md), [docs/vibevoice.pdf](docs/vibevoice.pdf), [docs/phase0_report.md](docs/phase0_report.md).

## Environment

- Prefer **WSL2 Ubuntu**, project at `/mnt/c/projects/tts-self`, venv `.venv`.
- Data/models: `~/datasets/tts/` (not committed). Set `HF_HOME=~/hf_cache` (host cache may be root-owned).
- After editing `src/`, reinstall if needed: `uv pip install -e . --no-deps` (editable may copy into site-packages).
- Run GPU work from WSL, not native Windows Python.

## Non-negotiables

1. **Never train the codec.** `sample=False` only for TTS targets.
2. **Do not treat `vae_std=0.625` as a head error budget** — use empirical sensitivity curves.
3. **Qwen stays out of the train loop** until Phase 5 LoRA; use offline text-hidden caches.
4. **Feed stressed natural Russian into Qwen**, not phoneme strings.
5. **Latent geometry** lives in [`src/minispeech/constants.py`](src/minispeech/constants.py) only (`LATENT_DIM=64`, `FRAME_RATE=7.5`, `SAMPLES_PER_FRAME=3200`).
6. **Stress format** is combining acute `U+0301` via [`frontend/stress.py`](src/minispeech/frontend/stress.py) — train and infer must match.
7. Research-only speech synthesis — no impersonation / fraud tooling.

## Phase order

Do not skip gates:

| Phase | Gate |
|-------|------|
| 0 | Codec GO/NO-GO (`scripts/11_phase0_feasibility.py`) |
| 1 | Latent + text-hidden shards + stats |
| 1b | Align prior A/B/C (C is enough to start; not a hard blocker) |
| 2 | Ladder S1→S4; **text-usage gate on S1** |
| 3 | Flow head + A/B vs regression |
| 4 | Cached streaming decode + measured TTFB |
| 5 | LoRA only if measurable win |
| 6 | Multi-speaker / emotion |

Training ladder: **32 utts → 1 h → 5 h → full corpus**. Overfit S1 before scaling.

## Commands (WSL)

```bash
cd /mnt/c/projects/tts-self && source .venv/bin/activate
export HF_HOME=$HOME/hf_cache

python scripts/00_env_check.py
python scripts/11_phase0_feasibility.py --n 100   # include ASR when ready
python scripts/02_build_manifest.py --config configs/data_ruslan_local.yaml
python scripts/04_encode_latents.py --config configs/data_ruslan_local.yaml
python scripts/05_cache_text_hidden.py --config configs/data_ruslan_local.yaml --qwen-config configs/qwen_local.yaml
python scripts/06_align.py --config configs/data_ruslan_local.yaml --level C
python scripts/07_train.py --config configs/talker_v01_direct.yaml --data configs/data_ruslan_local.yaml --mode regression --stage S1
python scripts/09_eval.py --ckpt runs/v01_regression_direct/best.pt --data configs/data_ruslan_local.yaml
pytest -q
```

## Architecture pitfalls (learned)

- Residual acoustic head can **ignore text**; prefer direct head + `acoustic_dropout` until text-usage passes.
- Teacher-forced val loss hides AR drift — use free-running metrics.
- Stop labels are sparse — use `pos_weight` / focal + triple stop rule at infer.
- Contiguous book-range splits, not random utterance splits.
- Batch: `mode: auto`, candidates `[8,16,32,64]`, budget ~14.5 GB VRAM.

## Agent scaffolding

| Path | Role |
|------|------|
| [AGENTS.md](AGENTS.md) | This file |
| [.cursor/rules/](.cursor/rules/) | Always-on + scoped coding rules |
| [.cursor/skills/](.cursor/skills/) | Workflow skills (see below) |
| [.cursor/hooks.json](.cursor/hooks.json) | sessionStart context + shell guard |

### Skills

| Skill | When |
|-------|------|
| `phase0-feasibility` | Codec GO/NO-GO |
| `offline-caches` | Manifest / latents / text hiddens / align |
| `train-ladder` | S1→S4 regression & flow |
| `text-usage-gate` | Shuffle/mask failures |
| `streaming-infer` | Synth + TTFB streaming |
| `wsl-env` | Venv / CUDA / HF_HOME |
| `eval-gates` | pytest, smoke, WER, promote checks |

## Code style

- Python 3.11+, type hints, small focused modules under `src/minispeech/`.
- Configs in YAML under `configs/`; every run should write `run_manifest.json`.
- Prefer editing existing modules over new parallel frameworks.
- Do not commit datasets, checkpoints, or raw audio.
