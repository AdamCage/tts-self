#!/usr/bin/env bash
# Launch Wave 4 from a copy outside the git worktree (keeps tree clean).
set -euo pipefail
cd ~/projects/tts-self
unset GIT_DIR GIT_WORK_TREE MINISPEECH_ALLOW_DIRTY
EXT_LOGDIR="$HOME/datasets/tts/d1c_wave4_logs"
mkdir -p "$EXT_LOGDIR"

git fetch origin
git checkout d1c/semantic-eos
git reset --hard origin/d1c/semantic-eos
# Clear leftover untracked noise under results that would fail clean checks
if [ -n "$(git status --porcelain)" ]; then
  while IFS= read -r line; do
    path="${line:3}"
    case "$path" in
      .cursor/results/*)
        mkdir -p "$EXT_LOGDIR/pre_wave4_bak/$(dirname "$path")"
        mv -f "$path" "$EXT_LOGDIR/pre_wave4_bak/$path" 2>/dev/null || rm -rf "$path"
        ;;
    esac
  done < <(git status --porcelain)
fi
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY_AFTER_CLEANUP
  git status -sb
  exit 2
fi
TIP=$(git rev-parse HEAD)
echo "CLEAN_TIP=$TIP"

# Copy runner outside repo so sed/chmod cannot dirty the tree
cp -f .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave4_w1_seed101_run.sh \
  "$EXT_LOGDIR/wave4_w1_seed101_run.sh"
sed -i 's/\r$//' "$EXT_LOGDIR/wave4_w1_seed101_run.sh"
chmod +x "$EXT_LOGDIR/wave4_w1_seed101_run.sh"

# Verify still clean after copy
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY_UNEXPECTED
  git status -sb
  exit 2
fi

SESSION=d1c_wave4_w1
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION" || true
# Preserve script exit code through tee
tmux new-session -d -s "$SESSION" \
  "bash '$EXT_LOGDIR/wave4_w1_seed101_run.sh' > >(tee '$EXT_LOGDIR/wave4_w1_tmux.outer.log') 2>&1; ec=\$?; echo EXIT=\$ec | tee -a '$EXT_LOGDIR/wave4_w1_tmux.outer.log'; exit \$ec"
echo "TMUX_STARTED session=$SESSION tip=$TIP"
tmux ls | grep "$SESSION" || true
