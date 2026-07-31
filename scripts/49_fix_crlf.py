#!/usr/bin/env python3
from pathlib import Path
import sys

for arg in sys.argv[1:]:
    p = Path(arg)
    data = p.read_bytes()
    fixed = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    p.write_bytes(fixed)
    print(f"fixed {p} crlf={data.count(b'\\r\\n')} -> {fixed.count(b'\\r')}")
