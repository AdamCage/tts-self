---
name: checkpoint-listen
description: >-
  After any new MiniSpeech-RU checkpoint (Talker, mel, variance, vocoder,
  tokenizer), run listen-demo inference and write wavs under
  out/<plan_slug>/... when a plan is active, else out/*_listen_demo. Use when
  training finishes, a best.pt/last.pt appears, promoting a stage, vocoder FT
  completes, or the user asks to hear / synthesize a new run.
---

# Checkpoint listen-demo (mandatory after new ckpt)

## When

Trigger whenever any of these is true:

- Train/FT job wrote `runs/*/best.pt` or `last.pt`
- User asks “что слушать”, “синтезируй”, “демо”, “hear the new model”
- Comparing systems (duration modes, vocoders, stages)
- **Any** inference / external baseline synth (always persist wavs)

Do **not** treat val loss / CER alone as completion of a train task.

## Defaults

Plan slug = plan filename without `.plan.md` (e.g. `p1_data_execution`).

| Kind | Infer entry | Default out |
|------|-------------|-------------|
| Mel Talker / stress / variance | `scripts/28_infer_mel.py --demo` | `out/<plan_slug>/<run>_listen_demo/` or `out/<run>_listen_demo/` |
| Latent Talker (legacy) | `scripts/08_infer.py` | same pattern |
| Vocoder only | same mel infer + `--vocoder-ckpt` | `.../{stock,ft}/` |
| External (Silero, …) | `scripts/39_external_baseline_eval.py` | `out/<plan_slug>/silero/` |

- GPU: WSL `~/projects/tts-self`, `source .venv/bin/activate`, `HF_HOME=~/hf_cache`
- After WSL jobs: copy wavs to `C:\projects\tts-self\out\...` for Cursor playback
- Write `.cursor/results/<plan_slug>/listen_index.json` (and/or local
  `listen_index.json`) with ckpt paths and file pairs

## Mel Talker recipe

```bash
cd ~/projects/tts-self && source .venv/bin/activate
export HF_HOME=$HOME/hf_cache

RUN=mel_stress_v01_s4   # or the new run dir name
PLAN=p1_data_execution  # active plan slug, if any
OUT="out/${PLAN}/${RUN}_listen_demo"
# no active plan: OUT="out/${RUN}_listen_demo"
python scripts/28_infer_mel.py \
  --ckpt "runs/${RUN}/best.pt" \
  --demo \
  --duration-mode mean_stats \
  --gap-mode expand_neighbor \
  --out "$OUT"
```

Optional duration A/B (same ckpt):

```bash
python scripts/28_infer_mel.py --ckpt "runs/${RUN}/best.pt" --demo \
  --duration-mode pred --out "out/${PLAN}/${RUN}_listen_demo_pred"
```

## Vocoder FT recipe

Same Talker ckpt; only `--vocoder-ckpt` changes:

```bash
STOCK=~/models/hifigan/generator.pt
FT=runs/vocoder_ft_v01/best.pt   # trainer ckpt; loader reads key "generator"

python scripts/28_infer_mel.py --ckpt runs/mel_stress_v01_s4/best.pt --demo \
  --duration-mode mean_stats --vocoder-ckpt "$STOCK" \
  --out out/m07_listen_demo/stock

python scripts/28_infer_mel.py --ckpt runs/mel_stress_v01_s4/best.pt --demo \
  --duration-mode mean_stats --vocoder-ckpt "$FT" \
  --out out/m07_listen_demo/ft_v01
```

## Checklist

- [ ] wavs exist and are non-empty
- [ ] `listen_index.json` lists stock/new paths
- [ ] User message includes concrete play paths (Windows `out/...` preferred)
- [ ] If comparing: two folders with matching filenames

## Related

- Batch/streaming details: `streaming-infer` skill
- Eval CER suite: `eval-gates` / `scripts/32_eval_mel_suite.py` (does not replace listen demo)
