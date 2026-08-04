# Reproducibility — MiniSpeech-RU

How to keep run provenance trustworthy across the Windows author copy and the
WSL2 GPU clone.

## Two copies, one commit

| Role | Path | Rule |
|------|------|------|
| Authoring (Cursor) | `C:\projects\tts-self` | Edit code/configs here; commit here |
| GPU runtime (WSL2) | `~/projects/tts-self` | **Only** `git pull` / `git fetch` + checkout; never hand-edit |

Canonical sync:

```text
Windows:  edit → git commit → (optional) git push
WSL:      cd ~/projects/tts-self && git pull --ff-only
          git rev-parse --short HEAD   # must match Windows HEAD
          git status --porcelain       # must be empty
```

Record the SHA in logs / `run_manifest.json` (already written by trainers).

## Never copy-instead-of-pull

Do **not** rsync / `cp -a` / Explorer-copy source trees from `/mnt/c/...` into
the Linux clone. That produces:

- hundreds of “dirty” or `UNTRACKED` files (often CRLF),
- `git_sha` that does not describe the code that actually ran,
- extra worktrees that diverge from `master`.

If the WSL tree is contaminated: discard local source changes
(`git reset --hard` + `git clean -fd`, **never** `git clean -fdx` — that would
delete gitignored `runs/` / caches), then `git pull --ff-only`. Keep
`~/datasets`, `runs/`, latents, and audio.

One worktree only for day-to-day work (`git worktree list`). Extra worktrees
are for deliberate experiments, not a substitute for pull.

## Clean tree required for training

Policy (enforced in `minispeech.utils.write_run_manifest`):

1. Every new run writes `git_sha` and `git_dirty` into `run_dir/run_manifest.json`.
2. If `git status --porcelain` is non-empty, **start is refused** with
   `RuntimeError` (ignored paths such as `runs/` / `out/` do not count).
3. Emergency escape hatch only: `MINISPEECH_ALLOW_DIRTY=1` — prints a warning,
   continues, and records `git_dirty: true`. Do not use for gate / paper runs.

Unit tests that call `write_run_manifest` may pass `enforce_clean=False`.

## Eval-set content-hash provenance

Frozen RU-Critical sets live under `configs/eval_sets/*.json`. The hash
registry is version-controlled at:

```text
configs/eval_set_hashes.json
```

(`--hash-file` remains available for overrides / temp dirs.)

```bash
# Recompute after an audited set change (via scripts/54_eval_sets.py bump)
python scripts/54_eval_sets.py hash

# Prove on-disk sets match the committed registry
python scripts/54_eval_sets.py verify
```

Contract (see registry `hash_contract`):

- `items_sha256` — SHA-256 of canonical JSON of the frozen item array
  (`items`, or `utt_ids` for legacy `kind: utt_ids` sets). Metadata such as
  `version` is outside this digest; bumping version without item edits still
  changes `file_sha256`.
- `file_sha256` — SHA-256 of exact UTF-8 file bytes.

Post-hoc recovery when a historical `run_manifest.json` has a wrong `git_sha`
(e.g. P1 worktree / dirty clone): recompute content hashes of the eval sets and
checkpoints that still exist on disk and compare to this registry / recorded
artifact digests. Do not invent SHAs.

## Quick checklist before a GPU job

```bash
cd ~/projects/tts-self && source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
git pull --ff-only
git rev-parse --short HEAD          # == Windows
test -z "$(git status --porcelain)" # clean
python scripts/54_eval_sets.py verify
# then train / eval …
```
