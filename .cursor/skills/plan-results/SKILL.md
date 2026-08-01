---
name: plan-results
description: >-
  After implementing a MiniSpeech-RU execution plan, write a detailed RESULTS.md
  plus logs under .cursor/results/<plan_slug>/. Use when a plan finishes, a phase
  gate closes, or the user asks for plan results / conclusions / logs.
---

# Plan results pack (mandatory after plan implementation)

## When

- An execution plan under `.cursor/plans/` finishes, fails a gate, or is
  hard-stopped with partial completion.
- User asks for phase conclusions, results dump, or “что сделано в плане”.

## Layout

Plan file `*.plan.md` → slug without suffix.

```text
.cursor/results/<plan_slug>/
  RESULTS.md
  logs/
    *.log
  listen_index.json          # if any inference ran
  artifacts/                   # optional small JSON copies
```

Wavs stay in `out/<plan_slug>/` (gitignored); `listen_index.json` points to them.

## RESULTS.md must include

1. Plan path + date + git SHA(s) used
2. Status: GO / NO-GO / partial + which tasks done/skipped/blocked
3. Tables of key metrics with **artifact paths** for every number
4. Code/config changes (commit SHA or file list)
5. Inference listen paths (Windows `out/<plan_slug>/...`)
6. Methodology alignment (read-only during implementation; cite docs, do not edit)
7. Follow-ups / open blockers

## Logs

- Tee long GPU/CLI jobs into `.cursor/results/<plan_slug>/logs/`
- Prefer copying final console logs from `runs/*/console.log` or `out/*.log`
- Keep logs text-only; do not commit large audio

## Methodology rule

While writing results for an **implemented** plan: **do not edit**
`.cursor/methodology/**`. Propose methodology updates only when drafting the
**next** stage plan.

## Related

- Rules: `methodology-and-results.mdc`, `inference-wavs.mdc`
- Example tone: `.cursor/results/track_n_p0_conclusion.md` (legacy flat file)
