#!/usr/bin/env python3
import sys
import torch

path = sys.argv[1]
ck = torch.load(path, map_location="cpu", weights_only=False)
print("keys", sorted(ck.keys())[:30])
for k in ("step", "global_step", "optimizer_step", "epoch"):
    if k in ck:
        print(k, ck[k])
m = ck.get("metrics") or {}
print("metrics", m)
