---
name: checkpoint-listen
description: >-
  After any new MiniSpeech-RU checkpoint (Talker, mel, variance, vocoder,
  tokenizer), run listen-demo inference and write wavs under out/*_listen_demo.
  Use when training finishes, a best.pt/last.pt appears, promoting a stage,
  vocoder FT completes, or the user asks to hear / synthesize a new run.
---

# Checkpoint listen-demo (mandatory after new ckpt)

## When

Trigger whenever any of these is true:

- Train/FT job wrote `runs/*/best.pt` or `last.pt`
- User asks “что слушать”, “синтезируй”, “демо”, “hear the new model”
- Comparing systems (duration modes, vocoders, stages)

Do **not** treat val loss / CER alone as completion of a train task.

## Defaults

| Kind | Infer entry | Default out |
|------|-------------|-------------|
| Mel Talker / stress / variance | `scripts/28_infer_mel.py --demo` | `out/<run_name>_listen_demo/` |
| Latent Talker (legacy) | `scripts/08_infer.py` | `out/<run_name>_listen_demo/` |
| Vocoder only | same mel infer + `--vocoder-ckpt` | `out/<run>_listen_demo/{stock,ft}/` |

- GPU: WSL `~/projects/tts-self`, `source .venv/bin/activate`, `HF_HOME=~/hf_cache`
- After WSL jobs: copy wavs to `C:\projects\tts-self\out\...` for Cursor playback
- Write `listen_index.json` with ckpt paths and file pairs

## Mel Talker recipe

```bash
cd ~/projects/tts-self && source .venv/bin/activate
export HF_HOME=$HOME/hf_cache

RUN=mel_stress_v01_s4   # or the new run dir name
python scripts/28_infer_mel.py \
  --ckpt "runs/${RUN}/best.pt" \
  --demo \
  --duration-mode mean_stats \
  --gap-mode expand_neighbor \
  --out "out/${RUN}_listen_demo"
```

Optional duration A/B (same ckpt):

```bash
python scripts/28_infer_mel.py --ckpt "runs/${RUN}/best.pt" --demo \
  --duration-mode pred --out "out/${RUN}_listen_demo_pred"
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
