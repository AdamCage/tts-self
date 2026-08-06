#!/usr/bin/env python3
"""Write LF launchers under /tmp and exec Wave 3 GPU."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

LAUNCH = r"""#!/usr/bin/env bash
set -euo pipefail
unset GIT_DIR GIT_WORK_TREE MINISPEECH_ALLOW_DIRTY
cd ~/projects/tts-self
git fetch origin
git checkout d1c/semantic-eos
git pull --ff-only origin d1c/semantic-eos
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY_WORKTREE
  git status -sb
  exit 2
fi
echo LAUNCH_TIP=$(git rev-parse --short HEAD)
SCRIPT_REPO=.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_gpu_run.sh
SCRIPT_TMP=/tmp/wave3_gpu_run_$$.sh
python3 - <<'PY'
from pathlib import Path
src = Path(".cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_gpu_run.sh")
import os
dst = Path(os.environ["SCRIPT_TMP"])
dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
PY
chmod +x "$SCRIPT_TMP"
export SMOKE_STEPS="${SMOKE_STEPS:-80}"
bash "$SCRIPT_TMP"
rc=$?
rm -f "$SCRIPT_TMP"
exit $rc
"""

# Fix: SCRIPT_TMP must be exported for the nested python. Rewrite launch cleanly.
LAUNCH = r"""#!/usr/bin/env bash
set -euo pipefail
unset GIT_DIR GIT_WORK_TREE MINISPEECH_ALLOW_DIRTY
cd ~/projects/tts-self
git fetch origin
git checkout d1c/semantic-eos
git pull --ff-only origin d1c/semantic-eos
if [ -n "$(git status --porcelain)" ]; then
  echo DIRTY_WORKTREE
  git status -sb
  exit 2
fi
echo LAUNCH_TIP=$(git rev-parse --short HEAD)
export SCRIPT_TMP=/tmp/wave3_gpu_run_$$.sh
python3 - <<'PY'
from pathlib import Path
import os
src = Path(".cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/wave3_gpu_run.sh")
dst = Path(os.environ["SCRIPT_TMP"])
dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
PY
chmod +x "$SCRIPT_TMP"
export SMOKE_STEPS="${SMOKE_STEPS:-80}"
bash "$SCRIPT_TMP"
rc=$?
rm -f "$SCRIPT_TMP"
exit $rc
"""

out = Path("/tmp/wave3_gpu_launch.sh")
out.write_text(LAUNCH, encoding="utf-8", newline="\n")
out.chmod(0o755)
print("wrote", out)
os.execv("/bin/bash", ["bash", str(out)])
