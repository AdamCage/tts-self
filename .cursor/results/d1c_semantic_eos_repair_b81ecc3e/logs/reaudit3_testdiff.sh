#!/usr/bin/env bash
# REAUDIT round3: which tests are new vs the round-2 attestation?
set -euo pipefail
cd /mnt/c/projects/tts-self-d1c/.cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs
grep -oE '^tests/[^[:space:]]+' repair_round2_full_pytest.log | sort -u > /tmp/r2.txt
grep -oE '^tests/[^[:space:]]+' repair_round3_full_pytest.log | sort -u > /tmp/r3.txt
grep -oE '^tests/[^[:space:]]+' reaudit3_full_pytest.log | sort -u > /tmp/rr3.txt
echo "=== NEW IN ROUND 3 (vs round 2) ==="
comm -13 /tmp/r2.txt /tmp/r3.txt
echo "=== REMOVED IN ROUND 3 ==="
comm -23 /tmp/r2.txt /tmp/r3.txt
echo "=== round3 claim vs my independent run: differences ==="
diff /tmp/r3.txt /tmp/rr3.txt && echo "IDENTICAL test id sets"
echo "=== counts (r2 / r3 / mine) ==="
wc -l /tmp/r2.txt /tmp/r3.txt /tmp/rr3.txt
