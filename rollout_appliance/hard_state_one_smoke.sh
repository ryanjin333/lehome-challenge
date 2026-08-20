#!/usr/bin/env bash
# One-snapshot hard-state smoke. Do not boot the GPU unless this is the plan.
# Uses a fresh ledger so a previous burned hard-state campaign cannot
# immediately return lease_next -> None.
set -euo pipefail
WORKSPACE="${LEHOME_WORKSPACE:-/mnt/lehome}"
SRC_MATRIX="${WORKSPACE}/eval/campaign-12k-round-3/hard-state-nearmiss.json"
SMOKE_ROOT="${WORKSPACE}/eval/campaign-hard-state-nearmiss-smoke"
if [ ! -f "${SRC_MATRIX}" ]; then
  echo "missing ranked hard-state matrix: ${SRC_MATRIX}" >&2
  exit 2
fi
sudo mkdir -p "${SMOKE_ROOT}"
python3 - <<'PY'
import json
from pathlib import Path
src=Path("/mnt/lehome/eval/campaign-12k-round-3/hard-state-nearmiss.json")
out=Path("/mnt/lehome/eval/campaign-hard-state-nearmiss-smoke/matrix-1.json")
rows=json.loads(src.read_text())
if not rows:
    raise SystemExit("hard-state matrix is empty")
one=[rows[0]]
out.write_text(json.dumps(one, indent=2, sort_keys=True) + "\n")
print("smoke_snapshot", one[0]["garment"], one[0]["seed"], one[0]["restore_snapshot"])
PY
sudo rm -f "${SMOKE_ROOT}/ledger.sqlite3" "${SMOKE_ROOT}/ledger.sqlite3-wal" "${SMOKE_ROOT}/ledger.sqlite3-shm"
sudo mkdir -p "${SMOKE_ROOT}/worker-2"
sudo chown -R 1234:1234 "${SMOKE_ROOT}"
echo "fresh ledger + 1-row matrix ready at ${SMOKE_ROOT}"
echo "Next: start only worker-2 against"
echo "  --database ${SMOKE_ROOT}/ledger.sqlite3"
echo "  --attempt-matrix ${SMOKE_ROOT}/matrix-1.json"
echo "  --max-attempts 1 --target-accepted 1"

# Worker uid 1234 must be able to read the policy ready file.
chmod 0644 "${WORKSPACE}/eval/receipts/original_baseline/ready.json" 2>/dev/null || true
chmod 0755 "${WORKSPACE}/eval/receipts/original_baseline" 2>/dev/null || true
