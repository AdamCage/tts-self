#!/usr/bin/env python3
import sys
from pathlib import Path

for arg in sys.argv[1:]:
    p = Path(arg)
    data = p.read_bytes()
    fixed = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    before_crlf = data.count(b"\r\n")
    after_cr = fixed.count(b"\r")
    p.write_bytes(fixed)
    print(f"fixed {p} crlf={before_crlf} -> {after_cr}")
