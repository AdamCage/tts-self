#!/usr/bin/env python3
from pathlib import Path
import json
import sys
p = Path(sys.argv[1])
print(p.read_text(encoding="utf-8")[:4000])
