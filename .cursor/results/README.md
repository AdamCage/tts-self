# Plan results

Per-plan packs live in directories named after the plan slug
(`p1_data_execution.plan.md` → `p1_data_execution/`).

```text
.cursor/results/<plan_slug>/
  RESULTS.md
  logs/
  listen_index.json
  artifacts/          # optional small JSON copies
```

Wavs: `out/<plan_slug>/` (gitignored). Methodology under `.cursor/methodology/`
is read during implementation and edited only when planning a new stage.

Legacy flat file: `track_n_p0_conclusion.md` (pre-convention).
