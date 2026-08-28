#!/usr/bin/env bash
# Isolated one-plus-three Top_Short_Seen_2 CPU-cloth fidelity diagnostic.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_RUNNER="${SCRIPT_DIR}/run_12k_campaign.sh"
DIAGNOSTIC_ROOT="${LEHOME_FIDELITY_DIAGNOSTIC_ROOT:-}"
HOST_CODE_ROOT="${LEHOME_HOST_CODE_ROOT:-}"
ROLLOUT_IMAGE="${LEHOME_ROLLOUT_IMAGE:-}"
VALIDATE_ONLY="${LEHOME_FIDELITY_DIAGNOSTIC_VALIDATE_ONLY:-0}"

case "${VALIDATE_ONLY}" in
  "0"|"1") ;;
  *) echo "LEHOME_FIDELITY_DIAGNOSTIC_VALIDATE_ONLY must be exactly 0 or 1" >&2; exit 2 ;;
esac
if [ -z "${DIAGNOSTIC_ROOT}" ] || [ -z "${HOST_CODE_ROOT}" ]; then
  echo "fidelity diagnostic requires explicit root and reviewed host code root" >&2
  exit 2
fi
if ! [[ "${ROLLOUT_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "fidelity diagnostic requires a digest-pinned rollout image" >&2
  exit 2
fi

python3 - "${DIAGNOSTIC_ROOT}" "${HOST_CODE_ROOT}" "${SCRIPT_DIR}" <<'PY' || exit 2
import re
import sys
from pathlib import Path

root, code_root, script_dir = map(Path, sys.argv[1:])
if not root.is_absolute() or root.is_symlink() or root.exists():
    raise SystemExit("fidelity diagnostic root must be a new absolute path")
if any(ancestor.is_symlink() for ancestor in root.parents):
    raise SystemExit("fidelity diagnostic root has a symlink ancestor")
if re.fullmatch(r"fidelity-diagnostic-[a-z0-9][a-z0-9-]{7,63}", root.name) is None:
    raise SystemExit("fidelity diagnostic root name is not allow-listed")
if any(part.startswith("simple-curriculum") or part.startswith("campaign-12k") for part in root.parts):
    raise SystemExit("fidelity diagnostic refuses real campaign roots")
if not code_root.is_absolute() or code_root.is_symlink() or code_root.resolve(strict=True) / "rollout_appliance" != script_dir:
    raise SystemExit("fidelity diagnostic host code root does not own this wrapper")
PY

mkdir -p "${DIAGNOSTIC_ROOT}/stage-a" "${DIAGNOSTIC_ROOT}/stage-b"
python3 - "${DIAGNOSTIC_ROOT}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
contracts = {
    "A": (2026082789,),
    "B": (2026082709, 2026082749, 2026082789),
}
for stage, seeds in contracts.items():
    rows = []
    for index, seed in enumerate(seeds, start=1):
        attempt_id = f"fidelity-diagnostic-{stage.lower()}-{index}"
        rows.append({
            "campaign_kind": "fidelity_diagnostic_v1",
            "diagnostic_stage": stage,
            "attempt_id": attempt_id,
            "trial_id": attempt_id,
            "garment": "Top_Short_Seen_2",
            "garment_name": "Top_Short_Seen_2",
            "category": "top_short",
            "release_stage": "seen",
            "seed": seed,
            "source_seed": seed,
            "strategy": "canonical",
        })
    path = root / f"stage-{stage.lower()}" / "matrix.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(rows, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
PY

run_stage() {
  local stage="$1"
  local stage_lower count matrix matrix_sha campaign_root
  stage_lower="$(printf '%s' "${stage}" | tr '[:upper:]' '[:lower:]')"
  if [ "${stage}" = "A" ]; then count=1; else count=3; fi
  campaign_root="${DIAGNOSTIC_ROOT}/stage-${stage_lower}"
  matrix="${campaign_root}/matrix.json"
  matrix_sha="$(sha256sum "${matrix}" | awk '{print $1}')"
  LEHOME_WORKSPACE="${LEHOME_WORKSPACE:-/mnt/lehome}" \
  LEHOME_CAMPAIGN_ROOT="${campaign_root}" \
  LEHOME_ATTEMPT_MATRIX="${matrix}" \
  LEHOME_ATTEMPT_MATRIX_SHA256="${matrix_sha}" \
  LEHOME_FIDELITY_DIAGNOSTIC="1" \
  LEHOME_FIDELITY_DIAGNOSTIC_STAGE="${stage}" \
  LEHOME_SIMPLE_CURRICULUM_COLLECTION="0" \
  LEHOME_SUCCESS_REPLAY_CAMPAIGN="0" \
  LEHOME_HARD_STATE_CAMPAIGN="0" \
  LEHOME_EVALUATION_TERMINAL_UPLOAD="0" \
  LEHOME_SNAPSHOT_SOURCE_BOOTSTRAP="0" \
  LEHOME_CONTROLLED_RECOVERY_SMOKE="0" \
  LEHOME_RESUME_PREEMPTED_ROLLOUT="0" \
  LEHOME_FRESH_GARMENT_WAVES="0" \
  LEHOME_SIMULATOR_DEVICE="cpu" \
  LEHOME_WORKER_COUNT="1" \
  LEHOME_ENABLE_HF_UPLOAD="0" \
  LEHOME_SKIP_ROUND_SEAL="1" \
  LEHOME_MAX_WORKER_RESTARTS="0" \
  LEHOME_COMPLETION_METRIC="terminal_outcomes" \
  LEHOME_MAX_ATTEMPTS="${count}" \
  LEHOME_TARGET_ACCEPTED="${count}" \
  LEHOME_PARTITION_ID="fidelity-diagnostic-${stage_lower}" \
  LEHOME_INITIAL_GARMENT="Top_Short_Seen_2" \
  LEHOME_POLICY_SHA256="e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa" \
  LEHOME_POLICY_REPO="ryanjin333/lehome-groot-n17-models" \
  LEHOME_POLICY_REVISION="30ac1a84da67b099e115ad147bcd61e9d60046d3" \
  LEHOME_POLICY_STEP="12000" \
  LEHOME_POLICY_ARTIFACT_SHA256="3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06" \
  LEHOME_TRAINER_IMAGE="ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746" \
  LEHOME_ROLLOUT_IMAGE="${ROLLOUT_IMAGE}" \
  LEHOME_HOST_CODE_ROOT="${HOST_CODE_ROOT}" \
  LEHOME_ROLLOUT_PREEMPTION_CONTEXT="${campaign_root}/preemption.json" \
  LEHOME_RUN_ID="fidelity-diagnostic-${stage_lower}" \
  LEHOME_ROUND_ID="fidelity-diagnostic-${stage_lower}" \
  LEHOME_VALIDATE_MATRIX_ONLY="${VALIDATE_ONLY}" \
    bash "${CAMPAIGN_RUNNER}"
}

verify_stage() {
  local stage="$1"
  local expected_count="$2"
  python3 - "${DIAGNOSTIC_ROOT}/stage-${stage}" "${expected_count}" <<'PY'
import json
import sqlite3
import sys
from pathlib import Path

root, expected_count = Path(sys.argv[1]), int(sys.argv[2])
connection = sqlite3.connect(root / "ledger.sqlite3")
try:
    rows = connection.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type").fetchall()
finally:
    connection.close()
counts = dict(rows)
if counts.get("infrastructure_abort", 0) != 0:
    raise SystemExit("fidelity diagnostic gate found infrastructure or fidelity evidence")
if any(counts.get(event_type, 0) for event_type in (
    "interrupted", "retryable", "campaign_paused", "campaign_resumed",
)):
    raise SystemExit("fidelity diagnostic gate found non-ordinary retry or resume evidence")
if counts.get("accepted", 0) + counts.get("rejected", 0) != expected_count:
    raise SystemExit("fidelity diagnostic gate requires exact ordinary terminal outcomes")
if counts.get("terminal_pending_validation", 0) != expected_count or counts.get("campaign_ended", 0) != 1:
    raise SystemExit("fidelity diagnostic gate found an incomplete ledger")
receipts = []
for path in sorted(root.glob("**/worker-receipt.json")):
    receipts.append(json.loads(path.read_text(encoding="utf-8")))
if len(receipts) != expected_count:
    raise SystemExit("fidelity diagnostic gate requires one worker receipt per outcome")
generations = sorted(receipt.get("episode_generation") for receipt in receipts)
if generations != list(range(1, expected_count + 1)) or len({receipt.get("session_id") for receipt in receipts}) != 1:
    raise SystemExit("fidelity diagnostic stage did not use one fresh persistent session")
PY
}

run_stage A
if [ "${VALIDATE_ONLY}" = "1" ]; then
  run_stage B
  exit 0
fi
verify_stage a 1
run_stage B
verify_stage b 3
