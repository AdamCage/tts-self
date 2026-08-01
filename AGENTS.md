# AGENTS.md — MiniSpeech-RU

Instructions for coding agents working in this repository.

## Project

Russian TTS research prototype on **RTX 4080 / 4080 Super 16 GB (WSL2)**:

```text
stressed RU text → offline Qwen3.5-0.8B text hidden
  → Talker (causal + cross-attn) → regression | rectified-flow head
  → VibeVoice latents [64] @ 7.5 Hz → frozen cached decoder
```

Primary docs: [README.md](README.md), [docs/conversation.md](docs/conversation.md), [docs/vibevoice.pdf](docs/vibevoice.pdf), [docs/phase0_report.md](docs/phase0_report.md), [docs/experiment_protocol.md](docs/experiment_protocol.md), [docs/part1_diagnosis.md](docs/part1_diagnosis.md) (Part 1 measured), [docs/next_intelligibility.md](docs/next_intelligibility.md) (Track A next). **Intelligibility work is mel Track A** (phoneme→mel baseline); do not resume latent S4/Flow until Gate 0 passes.

## Environment

- **Authoring copy (Windows / Cursor):** `C:\projects\tts-self` — edit code and configs here.
- **GPU runtime (WSL2 Ubuntu):** `~/projects/tts-self` — `git clone` / `git pull` of the same repo; venv `.venv` on the Linux filesystem.
- Data/models: `~/datasets/tts/` (not committed). Set `HF_HOME=~/hf_cache` (host cache may be root-owned).
- After editing `src/` on Windows, sync via git then in WSL reinstall if needed: `uv pip install -e . --no-deps`.
- Run GPU work from WSL, not native Windows Python. Do **not** run training from `/mnt/c/...`.

### WSL sync protocol

1. Edit on Windows → `git commit` / `git push` from the author copy.
2. Before every GPU run in WSL: `cd ~/projects/tts-self && git pull && git rev-parse --short HEAD` (record SHA in logs; `run_manifest.json` already stores `git_sha`).
3. Never hand-edit source in the WSL clone; pull only.
4. Artifacts (`runs/`, latents, ASR cache) stay in Linux FS. Numbers that must travel back to Windows go into `docs/*.md` (commit from WSL, `git pull` on Windows).
5. Long jobs: `tmux` with `2>&1 | tee runs/<name>/console.log`.

```bash
# One-time WSL setup
mkdir -p ~/projects && cd ~/projects
git clone <remote-url> tts-self   # or: git pull if already cloned
cd ~/projects/tts-self
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev,eval,stress]"
export HF_HOME=$HOME/hf_cache
python scripts/00_env_check.py && pytest -q
```

## Non-negotiables

1. **Never train the codec.** `sample=False` only for TTS targets.
2. **Do not treat `vae_std=0.625` as a head error budget** — use normalized CER(σ)/CER(α) from `scripts/21_latent_error_calibration.py` (Phase 0 sensitivity is secondary).
3. **Qwen stays out of the train loop** until a measured win justifies LoRA; use offline text-hidden caches.
4. **Feed stressed natural Russian into Qwen**, not phoneme strings.
5. **Target geometry** lives in [`src/minispeech/constants.py`](src/minispeech/constants.py) only — use the `TARGETS` registry (`get_target`). VibeVoice branch keeps module aliases `LATENT_DIM=64`, `FRAME_RATE=7.5`, `SAMPLES_PER_FRAME=3200`, `SR=24000` (`vibevoice64`). Track A adds registered mel target `mel80` (22.05 kHz, `n_fft=1024`, `hop=256` → ≈86.13 Hz, 80 bins). Do not invent parallel constants ad hoc.
6. **Stress format** is combining acute `U+0301` via [`frontend/stress.py`](src/minispeech/frontend/stress.py) — train and infer must match.
7. Research-only speech synthesis — no impersonation / fraud tooling.

## Phase order

Do not skip gates. **Promotion gates 0–6** are defined in [`docs/experiment_protocol.md`](docs/experiment_protocol.md) (R² / CER first; stop/duration alone is not enough). Part 1 is **done** — see [`docs/part1_diagnosis.md`](docs/part1_diagnosis.md) (Gate 0 FAIL). Next: Track A frontend + forced align ([`docs/next_intelligibility.md`](docs/next_intelligibility.md)).

| Phase | Gate |
|-------|------|
| 0 | Codec GO/NO-GO (`scripts/11_phase0_feasibility.py`) — ASR ΔWER with bootstrap CI |
| 1 | Latent + text-hidden shards + stats |
| 1b | Align prior A/B/C — prefer `forced.json` from `scripts/26_build_forced_align.py` then `06_align.py --level A`; C only as scaffold |
| 2 | Ladder only after Gate 0 (R²); **text-usage on S1** + synth CER |
| 3 | Generative head A/B only after intelligibility baseline exists |
| 4 | Cached streaming decode + measured TTFB (Gate 6) |
| 5 | LoRA only if measurable win |
| 6 | Multi-speaker / emotion |

Training ladder: **32 utts → 1 h → 5 h → full corpus**. Overfit S1 before scaling. Do **not** restart abandoned S4/Flow until Gate 0 (currently FAIL per Part 1).

## Commands (WSL)

```bash
cd ~/projects/tts-self && source .venv/bin/activate
export HF_HOME=$HOME/hf_cache

python scripts/00_env_check.py
python scripts/11_phase0_feasibility.py --n 120 --sweep-n 10 --asr-config configs/asr.yaml
python scripts/02_build_manifest.py --config configs/data_ruslan_local.yaml
python scripts/03_verify_text_asr.py --config configs/data_ruslan_local.yaml
python scripts/03b_apply_qc.py --config configs/data_ruslan_local.yaml
python scripts/04_encode_latents.py --config configs/data_ruslan_local.yaml
python scripts/05_cache_text_hidden.py --config configs/data_ruslan_local.yaml --qwen-config configs/qwen_local.yaml
python scripts/26_build_forced_align.py --config configs/data_ruslan_local.yaml --method uniform --max-utts 32
python scripts/06_align.py --config configs/data_ruslan_local.yaml --level A
python scripts/07_train.py --config configs/talker_v01_direct.yaml --data configs/data_ruslan_local.yaml --mode regression --stage S1
python scripts/09_eval.py --ckpt runs/v01_regression_direct/best.pt --data configs/data_ruslan_local.yaml
pytest -q
```

Frontend order for Track A: **TN (`normalize`/`verbalize`) → stress (`accentuate`) → G2P (`text_to_phonemes`)**. Optional: `uv pip install -e ".[g2p,stress]"` plus system `espeak-ng` in WSL.

## Architecture pitfalls (learned)

- Residual acoustic head can **ignore text**; prefer direct head + `acoustic_dropout` until text-usage passes.
- Teacher-forced val loss hides AR drift — use free-running metrics (DTW L1, duration ratio, ASR).
- Stop labels are sparse — use `pos_weight` / focal + positive window + triple stop rule at infer.
- Contiguous book-range splits, not random utterance splits.
- Batch: `mode: auto`, candidates `[8,16,32,64]`, budget ~14.5 GB VRAM.
- Root-anchored `.gitignore` (`/data/`) so `src/minispeech/data/` stays trackable.

## Agent scaffolding

| Path | Role |
|------|------|
| [AGENTS.md](AGENTS.md) | This file |
| [.cursor/rules/](.cursor/rules/) | Always-on + scoped coding rules |
| [.cursor/skills/](.cursor/skills/) | Workflow skills (see below) |
| [.cursor/methodology/](.cursor/methodology/) | Track N methodology (**read during implementation; edit only when planning a new stage**) |
| [.cursor/plans/](.cursor/plans/) | Execution plans (`<slug>.plan.md`) |
| [.cursor/results/](.cursor/results/) | Per-plan `RESULTS.md` + `logs/` (`<slug>/`) |
| [.cursor/hooks.json](.cursor/hooks.json) | sessionStart context + shell guard |

### Plan slug, wavs, results

- Plan file `p1_data_execution.plan.md` → slug `p1_data_execution`.
- **Any inference** must write wavs under `out/<plan_slug>/...` (fallback `out/<run>_listen_demo/` if no plan).
- After plan implementation: `.cursor/results/<plan_slug>/RESULTS.md` + `logs/` + `listen_index.json`.
- Do **not** edit `.cursor/methodology/**` while executing a plan.

### Skills

| Skill | When |
|-------|------|
| `phase0-feasibility` | Codec GO/NO-GO |
| `offline-caches` | Manifest / latents / text hiddens / align |
| `train-ladder` | S1→S4 regression & flow |
| `text-usage-gate` | Shuffle/mask/prefix failures |
| `streaming-infer` | Synth + TTFB streaming |
| `checkpoint-listen` | **Mandatory** listen-demo after any new `best.pt` / FT ckpt / infer |
| `plan-results` | **Mandatory** detailed results pack after a plan finishes |
| `wsl-env` | Venv / CUDA / HF_HOME |
| `eval-gates` | pytest, smoke, WER, promote checks |

## Code style

- Python 3.11+, type hints, small focused modules under `src/minispeech/`.
- Configs in YAML under `configs/`; every run should write `run_manifest.json`.
- Prefer editing existing modules over new parallel frameworks.
- Do not commit datasets, checkpoints, or raw audio.
