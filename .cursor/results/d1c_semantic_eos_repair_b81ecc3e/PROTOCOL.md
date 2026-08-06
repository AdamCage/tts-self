# PROTOCOL — D1c Semantic EOS (closure)

## Terminal state

**HARD-STOP** after Wave 5 seed102 formal FAIL.

## Binding rules applied

1. Formal DEV requires clean SHA + sealed contract (`clean_sha_available=true`, content SHA `fc07b643…`).
2. Wave 4 W1 seed101 PASS does not unlock CONFIRM; Wave 5 seed102 replication is mandatory.
3. Seed102 model FAIL → terminal HARD-STOP; no W2 weight change; no confirm reveal; no 1h.
4. Plaintext confirm seed never entered git; commitment remains  
   `4c46d418caf9bcabc088864da5bf0b3de0475a9d9225f50383314a2c5e3bb3f2`.
5. Methodology read-only during execution.

## Formal tips

| Wave | Tip | Role |
|------|-----|------|
| 4 | `c7e3ee0…` | W1 seed101 formal decode/score |
| 5 | `5bf319b…` | seed102 train+eval (docs commit on tip; recipe hashes match) |

## Artifact index

- Wave reports: `WAVE3_REPORT.md`, `WAVE4_REPORT.md`, `WAVE5_REPORT.md`
- Gates: `artifacts/gate_w1_seed101_*.json`, `artifacts/gate_seed102_*.json`
- Final gate: `d1c_reliability_gate.json`
- Listen: `listen_index.json`
