#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, "/mnt/c/projects/tts-self-d1c/src")
from minispeech.utils import git_sha
print("cwd", Path.cwd())
print("git_sha", repr(git_sha()))
print("git_file", Path("/mnt/c/projects/tts-self-d1c/.git").read_text()[:200])
