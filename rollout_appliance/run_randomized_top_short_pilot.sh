#!/usr/bin/env bash
# Start with one geometry-randomized top-short smoke. Re-run with
# LEHOME_WORKER_COUNT=4 only after its receipt proves the sampled variation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${LEHOME_WORKSPACE:-/mnt/lehome}"
SOURCE_MATRIX="${SCRIPT_DIR}/campaign_top_short_geometry_pilot.json"
SOURCE_SHA_FILE="${SCRIPT_DIR}/campaign_top_short_geometry_pilot.json.sha256"
WORKER_COUNT="${LEHOME_WORKER_COUNT:-1}"

case "${WORKER_COUNT}" in
  "1"|"4") ;;
  *) echo "LEHOME_WORKER_COUNT must be 1 for smoke or 4 for production" >&2; exit 2 ;;
esac

if [ ! -f "${SOURCE_MATRIX}" ] || [ ! -f "${SOURCE_SHA_FILE}" ]; then
  echo "missing pinned top-short geometry pilot matrix" >&2
  exit 2
fi
EXPECTED_SHA256="$(tr -d '[:space:]' < "${SOURCE_SHA_FILE}")"
printf '%s  %s\n' "${EXPECTED_SHA256}" "${SOURCE_MATRIX}" | sha256sum -c -

if [ "${WORKER_COUNT}" = "1" ]; then
  CAMPAIGN_ROOT="${WORKSPACE}/eval/campaign-top-short-geometry-smoke-1"
  MATRIX="${CAMPAIGN_ROOT}/matrix-smoke-1.json"
  mkdir -p "${CAMPAIGN_ROOT}"
  MATRIX_EXPECTED_SHA256="$(python3 - "${SOURCE_MATRIX}" "${MATRIX}" <<'PY'
import json
import os
import sys
from hashlib import sha256
from pathlib import Path

source, target = map(Path, sys.argv[1:])
rows = json.loads(source.read_text(encoding="utf-8"))
encoded = (json.dumps(rows[:1], indent=2, sort_keys=True) + "\n").encode("utf-8")
if target.exists() or target.is_symlink():
    if target.is_symlink() or not target.is_file() or target.read_bytes() != encoded:
        raise SystemExit("existing smoke matrix does not match the pinned source-derived smoke matrix")
else:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, target)
    except FileExistsError:
        if target.is_symlink() or not target.is_file() or target.read_bytes() != encoded:
            raise SystemExit("existing smoke matrix does not match the pinned source-derived smoke matrix")
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
print(sha256(encoded).hexdigest())
PY
)"
  MAX_ATTEMPTS=1
  TARGET_ACCEPTED=1
else
  CAMPAIGN_ROOT="${WORKSPACE}/eval/campaign-top-short-geometry-pilot-1"
  MATRIX="${CAMPAIGN_ROOT}/matrix-20.json"
  mkdir -p "${CAMPAIGN_ROOT}"
  if [ ! -f "${MATRIX}" ]; then
    cp "${SOURCE_MATRIX}" "${MATRIX}"
  fi
  printf '%s  %s\n' "${EXPECTED_SHA256}" "${MATRIX}" | sha256sum -c -
  MAX_ATTEMPTS=20
  TARGET_ACCEPTED="${LEHOME_TARGET_ACCEPTED:-8}"
  MATRIX_EXPECTED_SHA256="${EXPECTED_SHA256}"
fi

export LEHOME_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
export LEHOME_ATTEMPT_MATRIX="${MATRIX}"
export LEHOME_MATRIX_TEMPLATE="${MATRIX}"
export LEHOME_ATTEMPT_MATRIX_SHA256="${MATRIX_EXPECTED_SHA256}"
export LEHOME_WORKER_COUNT="${WORKER_COUNT}"
export LEHOME_MAX_ATTEMPTS="${MAX_ATTEMPTS}"
export LEHOME_TARGET_ACCEPTED="${TARGET_ACCEPTED}"
export LEHOME_INITIAL_GARMENT="Top_Short_Seen_0"

exec "${SCRIPT_DIR}/run_12k_campaign.sh"
