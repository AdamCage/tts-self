#!/usr/bin/env python3
from pathlib import Path

w42 = Path("/mnt/c/projects/tts-self/out/d1b_n2_reliability/confirm_seed42/best_full32_wavs/model")
w43 = Path("/mnt/c/projects/tts-self/out/d1b_n2_reliability/confirm_seed43/best_full32_wavs/model")
same = diff = hard = 0
for a in sorted(w42.glob("*.wav")):
    b = w43 / a.name
    sa, sb = a.stat(), b.stat()
    if sa.st_ino == sb.st_ino and sa.st_dev == sb.st_dev:
        hard += 1
    if a.read_bytes() == b.read_bytes():
        same += 1
    else:
        diff += 1
print("same_bytes", same, "diff", diff, "same_inode", hard, "n", same + diff)
print("sample mtime42", sorted(w42.glob("*.wav"))[0].stat().st_mtime)
print("sample mtime43", sorted(w43.glob("*.wav"))[0].stat().st_mtime)
