#!/usr/bin/env bash
set -Eeuo pipefail

CHECKPOINT_PATH=""
CYCLE_ID=${CYCLE_ID:-}
RUN_STATUS=success
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT_PATH=$2; shift 2 ;;
    --cycle) CYCLE_ID=$2; shift 2 ;;
    --status) RUN_STATUS=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Usage: $0 [--checkpoint PATH] --cycle cycle-NNN [--status success|failed] [--dry-run]" >&2; exit 2 ;;
  esac
done

MODEL_REPOSITORY=${MODEL_REPOSITORY:-ryanjin333/lehome-groot-n17-models}
B1K_DATA_ROOT=${B1K_DATA_ROOT:-/workspace/datasets/2026-challenge-demos}
LOG_DIR=${LOG_DIR:-/workspace/logs/b1k}
HANDOFF_ROOT=${HANDOFF_ROOT:-/workspace/handoff}
UPLOAD_VERIFIED_MARKER=${UPLOAD_VERIFIED_MARKER:-${LOG_DIR}/UPLOAD_VERIFIED}
TRAINER_CLI=${TRAINER_CLI:-/opt/runtime/bin/lehome-train}
if [[ -z "${GROOT_PYTHON:-}" ]]; then
  if [[ -x /opt/runtime/bin/python ]]; then
    GROOT_PYTHON=/opt/runtime/bin/python
  else
    GROOT_PYTHON=python3
  fi
fi

if [[ -z "${CYCLE_ID}" ]]; then
  echo "CYCLE_ID is required." >&2
  exit 1
fi
if [[ "${RUN_STATUS}" != "success" && "${RUN_STATUS}" != "failed" ]]; then
  echo "--status must be success or failed." >&2
  exit 2
fi
if [[ "${RUN_STATUS}" == "success" && ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "A valid --checkpoint directory is required for a successful run." >&2
  exit 1
fi

CHECKPOINT_ARCHIVE=""
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_ARCHIVE="$(basename "${CHECKPOINT_PATH}").tar.zst"
fi

if (( DRY_RUN )); then
  [[ -n "${CHECKPOINT_PATH}" ]] && echo "archive checkpoint ${CHECKPOINT_PATH} as checkpoints/${CHECKPOINT_ARCHIVE}"
  echo "include ${B1K_DATA_ROOT}/meta/stats.json and meta/modality.json fingerprints in provenance"
  echo "include ${LOG_DIR}/dataset-revision.txt and groot-commit.txt in provenance"
  echo "archive ${LOG_DIR} as logs/logs.tar.zst"
  echo "write resolved-config.json, provenance.json, and reports/training-report.json"
  echo "${TRAINER_CLI} sync --request <sync-request.json>"
  echo "require disposable=true and every entry remotely_verified=true"
  echo "write verified Hugging Face revision to ${UPLOAD_VERIFIED_MARKER}"
  exit 0
fi

if [[ ! -x "${TRAINER_CLI}" ]]; then
  echo "Trainer CLI is unavailable: ${TRAINER_CLI}" >&2
  exit 1
fi

RUN_METADATA=(
  "${B1K_DATA_ROOT}/meta/stats.json"
  "${B1K_DATA_ROOT}/meta/modality.json"
  "${LOG_DIR}/dataset-revision.txt"
  "${LOG_DIR}/groot-commit.txt"
)
if [[ "${RUN_STATUS}" == "success" ]]; then
  for required in "${RUN_METADATA[@]}"; do
    if [[ ! -s "${required}" ]]; then
      echo "Required run metadata is missing or empty: ${required}" >&2
      exit 1
    fi
  done
fi

mkdir -p "${HANDOFF_ROOT}" "${LOG_DIR}"
EXPERIMENT_ROOT=$(mktemp -d "${HANDOFF_ROOT}/.${CYCLE_ID}.XXXXXX")
SYNC_STAGING_ROOT=${HANDOFF_ROOT}/sync-staging
SYNC_REQUEST=${HANDOFF_ROOT}/${CYCLE_ID}-sync-request.json
SYNC_RESULT=${HANDOFF_ROOT}/${CYCLE_ID}-sync-result.json
mkdir -p \
  "${EXPERIMENT_ROOT}/checkpoints" \
  "${EXPERIMENT_ROOT}/logs" \
  "${EXPERIMENT_ROOT}/reports" \
  "${SYNC_STAGING_ROOT}"
rm -f "${UPLOAD_VERIFIED_MARKER}" "${SYNC_RESULT}"

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  tar --zstd -cf "${EXPERIMENT_ROOT}/checkpoints/${CHECKPOINT_ARCHIVE}" \
    -C "$(dirname "${CHECKPOINT_PATH}")" "$(basename "${CHECKPOINT_PATH}")"
else
  printf 'No recoverable checkpoint was produced.\n' > "${EXPERIMENT_ROOT}/checkpoints/no-checkpoint.txt"
fi
if [[ -d "${LOG_DIR}" ]]; then
  tar --zstd -cf "${EXPERIMENT_ROOT}/logs/logs.tar.zst" \
    -C "$(dirname "${LOG_DIR}")" "$(basename "${LOG_DIR}")"
else
  printf 'No run log directory was produced.\n' > "${EXPERIMENT_ROOT}/logs/no-logs.txt"
fi

export CYCLE_ID RUN_STATUS CHECKPOINT_ARCHIVE MODEL_REPOSITORY EXPERIMENT_ROOT
export B1K_DATA_ROOT LOG_DIR SYNC_REQUEST SYNC_RESULT SYNC_STAGING_ROOT UPLOAD_VERIFIED_MARKER
"${GROOT_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

from lehome_train.io import canonical_json_sha256, sha256_file


def read_optional(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


experiment_root = Path(os.environ["EXPERIMENT_ROOT"])
data_root = Path(os.environ["B1K_DATA_ROOT"])
log_root = Path(os.environ["LOG_DIR"])
config = {
    "schema_version": 1,
    "cycle_id": os.environ["CYCLE_ID"],
    "status": os.environ["RUN_STATUS"],
    "checkpoint_archive": os.environ["CHECKPOINT_ARCHIVE"] or None,
    "dataset_root": str(data_root),
}
provenance = {
    "schema_version": 1,
    "dataset_revision": read_optional(log_root / "dataset-revision.txt"),
    "groot_commit": read_optional(log_root / "groot-commit.txt"),
    "stats_sha256": sha256_file(data_root / "meta" / "stats.json")
    if (data_root / "meta" / "stats.json").is_file()
    else None,
    "modality_sha256": sha256_file(data_root / "meta" / "modality.json")
    if (data_root / "meta" / "modality.json").is_file()
    else None,
}
report = {
    "schema_version": 1,
    "cycle_id": os.environ["CYCLE_ID"],
    "status": os.environ["RUN_STATUS"],
}
for name, payload in (
    ("resolved-config.json", config),
    ("provenance.json", provenance),
    ("reports/training-report.json", report),
):
    (experiment_root / name).write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
request = {
    "schema_version": 1,
    "command": "sync",
    "arguments": {
        "experiment_root": str(experiment_root),
        "experiment_id": os.environ["CYCLE_ID"],
        "experiment_config_sha256": canonical_json_sha256(config),
        "repository": os.environ["MODEL_REPOSITORY"],
        "revision": "main",
        "staging_root": os.environ["SYNC_STAGING_ROOT"],
        "timeout_seconds": 120,
        "max_attempts": 5,
        "output": os.environ["SYNC_RESULT"],
    },
}
Path(os.environ["SYNC_REQUEST"]).write_text(
    json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY

"${TRAINER_CLI}" sync --request "${SYNC_REQUEST}"
"${GROOT_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["SYNC_RESULT"]).read_text(encoding="utf-8"))
entries = result.get("manifest", {}).get("entries", [])
if result.get("disposable") is not True or not entries:
    raise SystemExit("Hugging Face sync did not produce disposable evidence")
if not all(entry.get("remotely_verified") is True for entry in entries):
    raise SystemExit("one or more run artifacts failed immutable readback")
marker = Path(os.environ["UPLOAD_VERIFIED_MARKER"])
marker.write_text(
    f"{result['repository']}@{result['immutable_revision']}\n",
    encoding="utf-8",
)
PY
chmod 600 "${UPLOAD_VERIFIED_MARKER}"
echo "Verified complete Hugging Face run bundle: $(<"${UPLOAD_VERIFIED_MARKER}")"
