#!/usr/bin/env python3
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
g = json.loads(p.read_text(encoding="utf-8"))
print("verdict:", g.get("verdict"))
crit = g.get("criteria") or {}
for k, v in crit.items():
    if isinstance(v, (dict, list)) and k not in ("failures",):
        continue
    print(f"  {k}: {v}")
fails = crit.get("failures") or g.get("failures") or []
if fails:
    print("failures:", fails)
cat = g.get("criteria", {}).get("n_catastrophic")
print("n_catastrophic field:", cat)
rep = (g.get("reports") or {}).get("model") or {}
print("model macro_cer:", rep.get("macro_cer") or rep.get("mean_cer"))
print("model n_catastrophic:", rep.get("n_catastrophic"))
