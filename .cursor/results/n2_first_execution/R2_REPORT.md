# R2 — Fair Silero three conditions

## 1. Task
R2: measured Silero raw, Silero+shared frontend, and Track A on v3 sets with the same ASR/metrics.

## 2. Acceptance

| Criterion | Result |
|-----------|--------|
| three conditions on same v3 positions, `n` matches | **PASS** — all three: demo5=5, val100=100, bank_entity=315, homograph=719, latin=117 |
| bank_entity DER delta Silero raw vs Silero+frontend named | **PASS** — 0.9734 → 0.1030, Δ = **−0.8704** |
| no Track A win claim without condition | **PASS** — docs state system compare = Silero+frontend vs Track A; Silero+frontend wins bank/latin |
| wav paths named | **PASS** — see artifacts |
| record if Silero+frontend beats Track A | **PASS** — bank DER 0.1030 vs 0.6238; latin CER 0.0418 vs 0.2386 |

## 3. Changed / created files
- `scripts/39_external_baseline_eval.py` — already had `--input-mode` (no edit this run)
- `tests/test_external_baseline.py` — already covered input-mode (6/6 pass)
- `docs/frozen_numbers_mel80.md` — Silero section rewritten (historical unequal v2 + fair v3 triple)
- `out/n2_first_execution/eval_mel_suite_win.yaml` — suite with absolute Windows v3 paths
- `out/external_baseline_silero_v3_raw.json`
- `out/external_baseline_silero_v3_frontend.json`
- `out/external_baseline_track_a_v3.json`
- `out/external_baseline_silero_v3.json` (combined)
- wav trees under `out/n2_first_execution/{silero_raw,silero_frontend,track_a}/`
- `.cursor/results/n2_first_execution/logs/r2_*.log`
- `.cursor/results/n2_first_execution/listen_index.json`

## 4. Artifacts (numbers)

Combined: `out/external_baseline_silero_v3.json`

| Set | Track A CER | Silero raw CER | Silero+FE CER | Track A DER | Silero raw DER | Silero+FE DER |
|-----|------------:|---------------:|--------------:|------------:|---------------:|--------------:|
| demo5 | 0.0935 | 0.0305 | **0.0169** | — | — | — |
| val100 | 0.0319 | **0.0078** | **0.0078** | — | — | — |
| bank_entity v3 | 0.1818 | 0.5730 | **0.0488** | 0.6238 | 0.9734 | **0.1030** |
| homograph v3 | 0.0655 | **0.0068** | **0.0068** | — | — | — |
| latin | 0.2386 | 0.3338 | **0.0418** | — | — | — |

Frontend contribution (bank DER): **−0.8704**.
System compare: Silero+frontend **beats** Track A on bank_entity and latin.

Wav (Linux): `~/projects/tts-self/out/n2_first_execution/{silero_raw,silero_frontend,track_a}/` (1256 wav each).
Windows demo5 mirror: `C:\projects\tts-self\out\n2_first_execution\{…}\demo5\` (5 wav each).

## 5. Reproduce
```bash
cd ~/projects/tts-self && source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
export PYTHONPATH=/mnt/c/projects/tts-self/src
export MINISPEECH_ALLOW_DIRTY=1
# HEAD was 2c4c4bd, dirty=0; code/configs from Windows tree via PYTHONPATH + suite YAML
bash /mnt/c/projects/tts-self/out/_r2_full_run.sh
```
Unit tests: `pytest -q /mnt/c/projects/tts-self/tests/test_external_baseline.py` → 6 passed.
Logs: `.cursor/results/n2_first_execution/logs/r2_silero_raw.log`, `r2_silero_frontend.log`, `r2_track_a.log`.

## 6. Found nearby (not fixed)
- WSL clone still has bank_entity/homograph **v2**; R2 used absolute Windows v3 paths via `eval_mel_suite_win.yaml`.
- Silero raw still had 2 ValueError fallbacks (`bank_156`, `bank_162`); frontend mode had 0.

## 7. Blockers
None.
