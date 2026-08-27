#!/usr/bin/env bash
# Next-round collection after winner-gate failure.
# Parent is original step-12K. Do not collect from this-run 2K.
# Runtime inputs can select an immutable matrix/campaign root. One worker is
# the smoke default; four is the only production width on the 24-vCPU shape.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=worker_supervisor.sh
source "${SCRIPT_DIR}/worker_supervisor.sh"

WORKSPACE="${LEHOME_WORKSPACE:-/mnt/lehome}"
POLICY_SHA256="${LEHOME_POLICY_SHA256:-e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa}"
POLICY_REPO="${LEHOME_POLICY_REPO:-ryanjin333/lehome-groot-n17-models}"
POLICY_REVISION="${LEHOME_POLICY_REVISION:-30ac1a84da67b099e115ad147bcd61e9d60046d3}"
POLICY_STEP="${LEHOME_POLICY_STEP:-12000}"
POLICY_ARTIFACT_SHA256="${LEHOME_POLICY_ARTIFACT_SHA256:-3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06}"
TRAINER_IMAGE="${LEHOME_TRAINER_IMAGE:-ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746}"
CHECKPOINT_DIR="${LEHOME_CHECKPOINT_DIR:-${WORKSPACE}/eval/policies/original_baseline}"
CAMPAIGN_ROOT="${LEHOME_CAMPAIGN_ROOT:-${WORKSPACE}/eval/campaign-12k-round-3}"
RECEIPT_DIR="${LEHOME_RECEIPT_DIR:-${CAMPAIGN_ROOT}/policy-receipts}"
LEDGER="${CAMPAIGN_ROOT}/ledger.sqlite3"
MATRIX="${LEHOME_ATTEMPT_MATRIX:-${CAMPAIGN_ROOT}/matrix-400.json}"
MATRIX_TEMPLATE="${LEHOME_MATRIX_TEMPLATE:-/opt/lehome/rollout_appliance/campaign_400_balanced_geometry_v1.json}"
DEFAULT_MATRIX_SHA256="58160960d05153076693e127176c1c012e2c882f5e10bd031e0178d106db874b"
if [ -n "${LEHOME_ATTEMPT_MATRIX:-}" ] || [ -n "${LEHOME_MATRIX_TEMPLATE:-}" ]; then
  MATRIX_EXPECTED_SHA256="${LEHOME_ATTEMPT_MATRIX_SHA256:-}"
  if [ -z "${MATRIX_EXPECTED_SHA256}" ]; then
    echo "custom attempt matrix/template requires LEHOME_ATTEMPT_MATRIX_SHA256" >&2
    exit 2
  fi
else
  MATRIX_EXPECTED_SHA256="${DEFAULT_MATRIX_SHA256}"
fi
WORKER_COUNT="${LEHOME_WORKER_COUNT:-1}"
MAX_ATTEMPTS="${LEHOME_MAX_ATTEMPTS:-400}"
TARGET_ACCEPTED="${LEHOME_TARGET_ACCEPTED:-150}"
PREPARATION_TIMEOUT_SECONDS="${LEHOME_PREPARATION_TIMEOUT_SECONDS:-180}"
SOURCE_FINALIZATION_TIMEOUT_SECONDS="${LEHOME_SOURCE_FINALIZATION_TIMEOUT_SECONDS:-300}"
MAX_WORKER_RESTARTS="${LEHOME_MAX_WORKER_RESTARTS:-2}"
MAX_STEPS="${LEHOME_MAX_STEPS:-600}"
INITIAL_GARMENT="${LEHOME_INITIAL_GARMENT:-Top_Long_Seen_0}"
SIMULATOR_DEVICE="${LEHOME_SIMULATOR_DEVICE:-cuda:0}"
ROLLOUT_IMAGE="${LEHOME_ROLLOUT_IMAGE:-lehome-rollout:build}"
KIT_SEED="${LEHOME_KIT_SEED:-${WORKSPACE}/eval/kit/w0}"
HF_TOKEN_FILE="${LEHOME_HF_TOKEN_FILE:-${WORKSPACE}/secrets/hf_token}"
ROLLOUT_REPOSITORY="${LEHOME_ROLLOUT_REPOSITORY:-ryanjin333/lehome-groot-n17-rollouts}"
ROUND_ID="${LEHOME_ROUND_ID:-round-3}"
HF_REVISION="${LEHOME_HF_REVISION:-main}"
ENABLE_HF_UPLOAD="${LEHOME_ENABLE_HF_UPLOAD:-}"
RUN_ID="${LEHOME_RUN_ID:-lehome-rft-70-30-v1}"
PREEMPTION_CONTEXT="${LEHOME_ROLLOUT_PREEMPTION_CONTEXT:-/run/lehome-rollout/rollout-preemption.json}"
SKIP_ROUND_SEAL="${LEHOME_SKIP_ROUND_SEAL:-0}"
CONTROLLED_RECOVERY_SMOKE="${LEHOME_CONTROLLED_RECOVERY_SMOKE:-0}"
CONTROLLED_RECOVERY_SMOKE_RUN_ID="${LEHOME_CONTROLLED_RECOVERY_SMOKE_RUN_ID:-}"
CONTROLLED_RECOVERY_SMOKE_MATRIX_SHA256="${LEHOME_CONTROLLED_RECOVERY_MATRIX_SHA256:-}"
CONTROLLED_RECOVERY_SMOKE_MATERIALIZATION_SHA256="${LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION_SHA256:-}"
CONTROLLED_RECOVERY_SMOKE_ROW_INDEX="${LEHOME_CONTROLLED_RECOVERY_SMOKE_ROW_INDEX:-}"
SNAPSHOT_SOURCE_BOOTSTRAP="${LEHOME_SNAPSHOT_SOURCE_BOOTSTRAP:-0}"
SUCCESS_REPLAY_CAMPAIGN="${LEHOME_SUCCESS_REPLAY_CAMPAIGN:-0}"
HARD_STATE_CAMPAIGN="${LEHOME_HARD_STATE_CAMPAIGN:-0}"
RESUME_PREEMPTED_ROLLOUT="${LEHOME_RESUME_PREEMPTED_ROLLOUT:-0}"
EVALUATION_TERMINAL_UPLOAD="${LEHOME_EVALUATION_TERMINAL_UPLOAD:-0}"
FRESH_GARMENT_WAVES="${LEHOME_FRESH_GARMENT_WAVES:-${EVALUATION_TERMINAL_UPLOAD}}"
LEDGER_MAX_ATTEMPTS="${MAX_ATTEMPTS}"
if [ "${EVALUATION_TERMINAL_UPLOAD}" = "1" ]; then
  # The frozen evaluation still contains exactly MAX_ATTEMPTS distinct rows.
  # Extra ledger lease capacity exists only so an infrastructure-invalid row
  # can be retried from a clean Isaac process without dropping it from the 80.
  LEDGER_MAX_ATTEMPTS=400
fi

case "${WORKER_COUNT}" in
  "1"|"4") ;;
  *) echo "LEHOME_WORKER_COUNT must be exactly 1 (smoke) or 4 (production)" >&2; exit 2 ;;
esac
if [ -z "${ENABLE_HF_UPLOAD}" ]; then
  if [ "${WORKER_COUNT}" = "4" ]; then ENABLE_HF_UPLOAD=1; else ENABLE_HF_UPLOAD=0; fi
fi
case "${ENABLE_HF_UPLOAD}" in
  "0"|"1") ;;
  *) echo "LEHOME_ENABLE_HF_UPLOAD must be exactly 0 or 1" >&2; exit 2 ;;
esac
case "${SKIP_ROUND_SEAL}" in
  "0"|"1") ;;
  *) echo "LEHOME_SKIP_ROUND_SEAL must be exactly 0 or 1" >&2; exit 2 ;;
esac
case "${CONTROLLED_RECOVERY_SMOKE}" in
  "0"|"1") ;;
  *) echo "LEHOME_CONTROLLED_RECOVERY_SMOKE must be exactly 0 or 1" >&2; exit 2 ;;
esac
case "${SNAPSHOT_SOURCE_BOOTSTRAP}" in
  "0"|"1") ;;
  *) echo "LEHOME_SNAPSHOT_SOURCE_BOOTSTRAP must be exactly 0 or 1" >&2; exit 2 ;;
esac
case "${SUCCESS_REPLAY_CAMPAIGN}" in
  "0"|"1") ;;
  *) echo "LEHOME_SUCCESS_REPLAY_CAMPAIGN must be exactly 0 or 1" >&2; exit 2 ;;
esac
case "${HARD_STATE_CAMPAIGN}" in
  "0"|"1") ;;
  *) echo "LEHOME_HARD_STATE_CAMPAIGN must be exactly 0 or 1" >&2; exit 2 ;;
esac
if (( 10#${SUCCESS_REPLAY_CAMPAIGN} + 10#${HARD_STATE_CAMPAIGN} + 10#${EVALUATION_TERMINAL_UPLOAD} > 1 )); then
  echo "CPU campaign mode markers are mutually exclusive" >&2
  exit 2
fi
case "${SIMULATOR_DEVICE}" in
  "cuda:0") ;;
  "cpu")
    if [ "${EVALUATION_TERMINAL_UPLOAD}" = "1" ]; then
      if [ "${WORKER_COUNT}" != "4" ] || [ "${ENABLE_HF_UPLOAD}" != "1" ] \
          || [ "${SKIP_ROUND_SEAL}" != "0" ] || [ "${CONTROLLED_RECOVERY_SMOKE}" != "0" ] \
          || [ "${SNAPSHOT_SOURCE_BOOTSTRAP}" != "0" ] || [ "${RESUME_PREEMPTED_ROLLOUT}" != "0" ]; then
        echo "CPU cloth requires the exact terminal-evaluation tuple" >&2
        exit 2
      fi
    elif [ "${SUCCESS_REPLAY_CAMPAIGN}" = "1" ]; then
      if [ "${WORKER_COUNT}" != "4" ] || [ "${ENABLE_HF_UPLOAD}" != "1" ] \
          || [ "${SKIP_ROUND_SEAL}" != "0" ] || [ "${CONTROLLED_RECOVERY_SMOKE}" != "0" ] \
          || [ "${SNAPSHOT_SOURCE_BOOTSTRAP}" != "0" ] || [ "${RESUME_PREEMPTED_ROLLOUT}" != "0" ] \
          || ! [[ "${MAX_ATTEMPTS}" =~ ^([1-9]|[1-9][0-9]|[1-3][0-9][0-9]|400)$ ]] \
          || ! [[ "${TARGET_ACCEPTED}" =~ ^([1-9]|[1-9][0-9]|1[0-4][0-9]|150)$ ]] \
          || (( 10#${TARGET_ACCEPTED} > 10#${MAX_ATTEMPTS} )); then
        echo "CPU cloth requires the exact four-worker success-replay tuple" >&2
        exit 2
      fi
    elif [ "${HARD_STATE_CAMPAIGN}" = "1" ]; then
      if [ "${WORKER_COUNT}" != "4" ] || [ "${ENABLE_HF_UPLOAD}" != "1" ] \
          || [ "${SKIP_ROUND_SEAL}" != "0" ] || [ "${CONTROLLED_RECOVERY_SMOKE}" != "0" ] \
          || [ "${SNAPSHOT_SOURCE_BOOTSTRAP}" != "0" ] || [ "${RESUME_PREEMPTED_ROLLOUT}" != "0" ] \
          || ! [[ "${MAX_ATTEMPTS}" =~ ^([1-9]|[1-9][0-9]|[1-3][0-9][0-9]|400)$ ]] \
          || ! [[ "${TARGET_ACCEPTED}" =~ ^([1-9]|[1-9][0-9]|1[0-4][0-9]|150)$ ]] \
          || (( 10#${TARGET_ACCEPTED} > 10#${MAX_ATTEMPTS} )); then
        echo "CPU cloth requires the exact four-worker hard-state tuple" >&2
        exit 2
      fi
    elif [ "${CONTROLLED_RECOVERY_SMOKE}" = "1" ] && [ "${SNAPSHOT_SOURCE_BOOTSTRAP}" = "0" ]; then
      : # The controlled-smoke allow-list below validates its remaining exact invariants.
    elif [ "${SNAPSHOT_SOURCE_BOOTSTRAP}" != "1" ] || ! [[ "${WORKER_COUNT}" =~ ^(1|4)$ ]] \
        || ! [[ "${MAX_ATTEMPTS}" =~ ^([1-9]|[1-9][0-9]|[1-3][0-9][0-9]|400)$ ]] \
        || ! [[ "${TARGET_ACCEPTED}" =~ ^([1-9]|[1-9][0-9]|1[0-4][0-9]|150)$ ]] \
        || (( 10#${TARGET_ACCEPTED} > 10#${MAX_ATTEMPTS} )) \
        || [ "${ENABLE_HF_UPLOAD}" != "1" ] || [ "${SKIP_ROUND_SEAL}" != "1" ] \
        || [ "${RESUME_PREEMPTED_ROLLOUT}" != "0" ] \
        || [ "${CONTROLLED_RECOVERY_SMOKE}" != "0" ]; then
      echo "CPU cloth requires the exact unsealed snapshot-source bootstrap tuple" >&2
      exit 2
    fi
    ;;
  *) echo "LEHOME_SIMULATOR_DEVICE must be exactly cpu or cuda:0" >&2; exit 2 ;;
esac
case "${RESUME_PREEMPTED_ROLLOUT}" in
  "0"|"1") ;;
  *) echo "LEHOME_RESUME_PREEMPTED_ROLLOUT must be exactly 0 or 1" >&2; exit 2 ;;
esac
case "${EVALUATION_TERMINAL_UPLOAD}" in
  "0"|"1") ;;
  *) echo "LEHOME_EVALUATION_TERMINAL_UPLOAD must be exactly 0 or 1" >&2; exit 2 ;;
esac
case "${FRESH_GARMENT_WAVES}" in
  "0"|"1") ;;
  *) echo "LEHOME_FRESH_GARMENT_WAVES must be exactly 0 or 1" >&2; exit 2 ;;
esac
if [ "${EVALUATION_TERMINAL_UPLOAD}" = "1" ]; then
  if [ "${SIMULATOR_DEVICE}" != "cpu" ]; then
    echo "terminal evaluation requires LEHOME_SIMULATOR_DEVICE=cpu" >&2
    exit 2
  fi
  if [ "${ENABLE_HF_UPLOAD}" != "1" ] || [ "${WORKER_COUNT}" != "4" ] \
      || [ "${CONTROLLED_RECOVERY_SMOKE}" != "0" ] || [ "${SKIP_ROUND_SEAL}" != "0" ] \
      || [ "${SNAPSHOT_SOURCE_BOOTSTRAP}" != "0" ] || [ "${RESUME_PREEMPTED_ROLLOUT}" != "0" ] \
      || [ "${FRESH_GARMENT_WAVES}" != "1" ]; then
    echo "evaluation terminal publication requires the exact four-worker evaluation tuple" >&2
    exit 2
  fi
fi
if [ "${FRESH_GARMENT_WAVES}" = "1" ]; then
  if [ "${SIMULATOR_DEVICE}" != "cpu" ] \
      || { [ "${EVALUATION_TERMINAL_UPLOAD}" = "1" ] && [ "${WORKER_COUNT}" != "4" ]; } \
      || { [ "${EVALUATION_TERMINAL_UPLOAD}" != "1" ] && [ "${SNAPSHOT_SOURCE_BOOTSTRAP}" != "1" ]; }; then
    echo "fresh garment waves require four-worker terminal evaluation or one/four-worker snapshot-source discovery on CPU" >&2
    exit 2
  fi
fi
case "${MAX_WORKER_RESTARTS}" in
  ''|*[!0-9]*) echo "LEHOME_MAX_WORKER_RESTARTS must be a non-negative integer" >&2; exit 2 ;;
esac
case "${MAX_STEPS}" in
  ''|*[!0-9]*) echo "LEHOME_MAX_STEPS must be a positive integer" >&2; exit 2 ;;
  0) echo "LEHOME_MAX_STEPS must be a positive integer" >&2; exit 2 ;;
esac
if ! python3 - "${SOURCE_FINALIZATION_TIMEOUT_SECONDS}" <<'PY'
import math
import sys
try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value > 0 else 1)
PY
then
  echo "LEHOME_SOURCE_FINALIZATION_TIMEOUT_SECONDS must be positive and finite" >&2
  exit 2
fi
if ! [[ "${MATRIX_EXPECTED_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "LEHOME_ATTEMPT_MATRIX_SHA256 must be a lowercase 64-character SHA-256" >&2
  exit 2
fi
if [[ "${POLICY_REPO}" =~ [[:space:]] ]] || [ -z "${POLICY_REPO}" ]; then
  echo "LEHOME_POLICY_REPO must be a non-empty repository without whitespace" >&2
  exit 2
fi
if ! [[ "${POLICY_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LEHOME_POLICY_REVISION must be an immutable 40-character commit" >&2
  exit 2
fi
case "${POLICY_STEP}" in
  ''|*[!0-9]*) echo "LEHOME_POLICY_STEP must be a positive integer" >&2; exit 2 ;;
  0) echo "LEHOME_POLICY_STEP must be a positive integer" >&2; exit 2 ;;
esac
if ! [[ "${POLICY_ARTIFACT_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "LEHOME_POLICY_ARTIFACT_SHA256 must be a lowercase 64-character SHA-256" >&2
  exit 2
fi
if [ ! -f "${MATRIX}" ]; then
  if [ -f "${MATRIX_TEMPLATE}" ]; then
    cp "${MATRIX_TEMPLATE}" "${MATRIX}"
  else
    echo "missing 400-attempt matrix" >&2
    exit 2
  fi
fi
MATRIX_ACTUAL_SHA256="$(sha256sum "${MATRIX}" | awk '{print $1}')"
if [ "${MATRIX_ACTUAL_SHA256}" != "${MATRIX_EXPECTED_SHA256}" ]; then
  echo "attempt matrix SHA-256 mismatch: expected ${MATRIX_EXPECTED_SHA256}, got ${MATRIX_ACTUAL_SHA256}" >&2
  exit 2
fi
if [ "${SKIP_ROUND_SEAL}" = "1" ] && [ "${SNAPSHOT_SOURCE_BOOTSTRAP}" = "1" ]; then
  if [ "${CONTROLLED_RECOVERY_SMOKE}" != "0" ] || ! [[ "${WORKER_COUNT}" =~ ^(1|4)$ ]] \
      || [ "${ENABLE_HF_UPLOAD}" != "1" ] || [ "${RESUME_PREEMPTED_ROLLOUT}" != "0" ] \
      || ! [[ "${MAX_WORKER_RESTARTS}" =~ ^(0|8)$ ]] \
      || ! [[ "${RUN_ID}" =~ ^[0-9a-f]{32}$ ]] \
      || ! [[ "${ROUND_ID}" =~ ^snapshot-source-bootstrap-[0-9a-f]{20}-unsealed-source$ ]]; then
    echo "LEHOME_SKIP_ROUND_SEAL is reserved for the exact snapshot-source bootstrap tuple" >&2
    exit 2
  fi
  python3 - "${MATRIX}" "${MATRIX_EXPECTED_SHA256}" "${RUN_ID}" "${ROUND_ID}" "${MAX_ATTEMPTS}" "${TARGET_ACCEPTED}" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
path, expected, run_id, round_id, max_attempts, target_accepted = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
try: rows = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError) as error: raise SystemExit(f"snapshot source descriptor is malformed: {error}")
try:
    max_attempts, target_accepted = int(max_attempts), int(target_accepted)
except ValueError: raise SystemExit("snapshot source discovery attempt bounds are invalid")
if not isinstance(rows, list) or not 1 <= len(rows) <= 400 or not all(isinstance(row, dict) for row in rows): raise SystemExit("snapshot source descriptor must contain 1..400 rows")
if max_attempts != len(rows) or not 1 <= target_accepted <= min(150, len(rows)): raise SystemExit("snapshot source discovery attempt bounds are invalid")
legacy_single = len(rows) == 1 and rows[0].get("replay_kind") in {
    "verified_success_reset_v1", "verified_success_early_snapshot_v1",
}
if not legacy_single:
    seeds = set()
    allowed_fields = {"snapshot_source_bootstrap", "snapshot_source_descriptor_sha256", "category", "garment", "garment_name", "seed", "source_seed"}
    for row in rows:
        category = row.get("category")
        if (row.get("snapshot_source_bootstrap") is not True
                or row.get("garment_name") not in (None, row.get("garment"))
                or any(key not in allowed_fields for key in row)):
            raise SystemExit("snapshot source descriptor must be ordinary autonomous collection")
        garment = row.get("garment")
        canonical_seen = {
            "top_long": r"Top_Long_Seen_[0-9]+",
            "top_short": r"Top_Short_Seen_[0-9]+",
            "pant_long": r"Pant_Long_Seen_[0-9]+",
            "pant_short": r"Pant_Short_Seen_[0-9]+",
        }
        if category not in canonical_seen or not isinstance(garment, str) or re.fullmatch(canonical_seen[category], garment) is None:
            raise SystemExit("snapshot source descriptor garment identity is not a canonical seen garment")
        seed, source_seed = row.get("seed"), row.get("source_seed")
        if (type(seed) is not int or seed < 0 or type(source_seed) is not int
                or source_seed < 0 or source_seed != seed or seed in seeds):
            raise SystemExit("snapshot source descriptor seed binding is invalid")
        seeds.add(seed)
else:
    row = rows[0]
    if row.get("snapshot_source_bootstrap") is not True or row.get("recovery_kind") is not None:
        raise SystemExit("snapshot source descriptor lineage is invalid")
    if (type(row.get("seed")) is not int or row["seed"] < 0
            or type(row.get("source_seed")) is not int or row["source_seed"] < 0
            or row["seed"] != row["source_seed"]):
        raise SystemExit("snapshot source descriptor seed binding is invalid")
for row in rows:
    if row.get("snapshot_source_descriptor_sha256") not in (None, expected): raise SystemExit("snapshot source descriptor hash is invalid")
identity = hashlib.sha256(f"{run_id}:{expected}".encode("ascii")).hexdigest()[:20]
if round_id != f"snapshot-source-bootstrap-{identity}-unsealed-source": raise SystemExit("snapshot source descriptor does not bind the active run identity")
PY
elif [ "${SKIP_ROUND_SEAL}" = "1" ]; then
  # This is intentionally an allow-list, not a general skip switch.  A
  # staging smoke has one attempt and never creates a trainable round seal.
  if [ "${CONTROLLED_RECOVERY_SMOKE}" != "1" ] || [ "${WORKER_COUNT}" != "1" ] \
      || [ "${MAX_ATTEMPTS}" != "1" ] || [ "${TARGET_ACCEPTED}" != "1" ] \
      || [ "${ENABLE_HF_UPLOAD}" != "1" ] \
      || ! [[ "${CONTROLLED_RECOVERY_SMOKE_RUN_ID}" =~ ^[0-9a-f]{32}$ ]] \
      || ! [[ "${CONTROLLED_RECOVERY_SMOKE_MATRIX_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
      || ! [[ "${CONTROLLED_RECOVERY_SMOKE_MATERIALIZATION_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
      || ! [[ "${CONTROLLED_RECOVERY_SMOKE_ROW_INDEX}" =~ ^[0-9]+$ ]] \
      || ! [[ "${ROUND_ID}" =~ ^controlled-recovery-smoke-[0-9a-f]{20}-unsealed-staging$ ]]; then
    echo "LEHOME_SKIP_ROUND_SEAL is reserved for the exact controlled-recovery smoke tuple" >&2
    exit 2
  fi
  python3 - "${MATRIX}" "${MATRIX_EXPECTED_SHA256}" "${ROUND_ID}" "${CONTROLLED_RECOVERY_SMOKE_RUN_ID}" "${CONTROLLED_RECOVERY_SMOKE_MATRIX_SHA256}" "${CONTROLLED_RECOVERY_SMOKE_MATERIALIZATION_SHA256}" "${CONTROLLED_RECOVERY_SMOKE_ROW_INDEX}" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
path, expected, round_id, run_id, full_matrix, full_materialization, row_index = Path(sys.argv[1]), sys.argv[2], *sys.argv[3:]
try: rows = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError) as error: raise SystemExit(f"controlled smoke descriptor is malformed: {error}")
if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict): raise SystemExit("controlled smoke descriptor must contain exactly one row")
row = rows[0]
for key in ("controlled_smoke", "controlled_smoke_matrix_sha256", "controlled_smoke_materialization_sha256"):
    if key not in row: raise SystemExit("controlled smoke descriptor lineage is incomplete")
if row.get("controlled_smoke") is not True or row.get("recovery_kind") != "controlled_success_recovery_snapshot_v3": raise SystemExit("controlled smoke descriptor kind is invalid")
if (row.get("controlled_smoke_zero_perturbation") is not True
        or row.get("controlled_smoke_teacher_probe") is not True
        or row.get("controlled_smoke_perturbation_mode") != "zero_perturbation_teacher_continuation_probe_v1"):
    raise SystemExit("controlled smoke descriptor mode is invalid")
for key in ("controlled_smoke_matrix_sha256", "controlled_smoke_materialization_sha256"):
    if not isinstance(row.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", row[key]) is None: raise SystemExit("controlled smoke descriptor hash is invalid")
identity = hashlib.sha256(f"{run_id}:{full_matrix}:{full_materialization}".encode("ascii")).hexdigest()[:20]
mode_identity = hashlib.sha256(f"{identity}:zero_perturbation_teacher_continuation_probe_v1".encode("ascii")).hexdigest()[:20]
if (row.get("controlled_smoke_run_id") != run_id or row.get("controlled_smoke_row_index") != int(row_index)
        or row.get("controlled_smoke_matrix_sha256") != full_matrix
        or row.get("controlled_smoke_materialization_sha256") != full_materialization
        or row.get("controlled_smoke_identity") != identity
        or row.get("controlled_smoke_mode_identity") != mode_identity
        or round_id != f"controlled-recovery-smoke-{identity}-unsealed-staging"):
    raise SystemExit("controlled smoke descriptor does not bind the active run identity")
if row.get("controlled_smoke_descriptor_sha256") not in (None, expected): raise SystemExit("controlled smoke descriptor hash is invalid")
PY
elif [ "${CONTROLLED_RECOVERY_SMOKE}" = "1" ]; then
  echo "LEHOME_CONTROLLED_RECOVERY_SMOKE requires LEHOME_SKIP_ROUND_SEAL=1" >&2
  exit 2
fi
if [ "${RESUME_PREEMPTED_ROLLOUT}" = "1" ] && [ "${CONTROLLED_RECOVERY_SMOKE}" != "1" ]; then
  echo "preemption resume is reserved for the exact controlled-recovery smoke tuple" >&2
  exit 2
fi
mkdir -p "${CAMPAIGN_ROOT}"
evaluation_garments=()
if [ "${FRESH_GARMENT_WAVES}" = "1" ]; then
  while IFS= read -r garment; do
    evaluation_garments+=("${garment}")
  done < <(python3 - "${MATRIX}" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1], encoding="utf-8"))
seen = set()
for row in rows:
    garment = row.get("garment_name") or row.get("garment")
    if not isinstance(garment, str) or not garment:
        raise SystemExit("terminal evaluation garment identity is invalid")
    if garment not in seen:
        print(garment)
        seen.add(garment)
PY
  )
  if [ "${#evaluation_garments[@]}" -eq 0 ]; then
    echo "fresh garment evaluation requires at least one garment" >&2
    exit 2
  fi
  if [ "${EVALUATION_TERMINAL_UPLOAD}" = "1" ] && [ $(( ${#evaluation_garments[@]} % 4 )) -ne 0 ]; then
    echo "terminal fresh garment evaluation requires a multiple of four garments" >&2
    exit 2
  fi
fi
if [ "${LEHOME_VALIDATE_MATRIX_ONLY:-0}" = "1" ]; then
  exit 0
fi

if [ "${RESUME_PREEMPTED_ROLLOUT}" = "1" ]; then
  PYTHONPATH="/opt/lehome/source/lehome:/opt/lehome/trainer/src:/opt/lehome${PYTHONPATH:+:${PYTHONPATH}}" python3 - "${LEDGER}" "${MATRIX}" "${MAX_ATTEMPTS}" "${TARGET_ACCEPTED}" <<'PY'
import sys
from pathlib import Path
from lehome.flywheel.recovery_collection import load_attempt_matrix
from lehome.flywheel.task_ledger import TaskLedger
database, matrix = Path(sys.argv[1]), Path(sys.argv[2])
if database.is_symlink() or not database.is_file(): raise SystemExit("controlled smoke resume ledger is missing or unsafe")
ledger = TaskLedger(database, attempt_matrix=load_attempt_matrix(matrix), max_attempts=int(sys.argv[3]), target_accepted=int(sys.argv[4]))
try:
    if ledger.is_terminal: raise SystemExit("controlled smoke resume refuses an already terminal campaign")
    ledger.resume_after_preemption("explicit-controlled-recovery-smoke-resume")
finally:
    ledger.close()
PY
fi

write_preemption_context() {
  local active="$1"
  local preemption_context_dir
  preemption_context_dir="$(dirname "${PREEMPTION_CONTEXT}")"
  if [ ! -d "${preemption_context_dir}" ]; then
    install -d -m 0770 "${preemption_context_dir}"
  fi
  python3 - \
    "${PREEMPTION_CONTEXT}" "${RUN_ID}" "${CAMPAIGN_ROOT}" "${LEDGER}" \
    "${MATRIX}" "${MATRIX_ACTUAL_SHA256}" "${MAX_ATTEMPTS}" "${TARGET_ACCEPTED}" "${active}" \
    "${CONTROLLED_RECOVERY_SMOKE}" "${CONTROLLED_RECOVERY_SMOKE_RUN_ID}" \
    "${CONTROLLED_RECOVERY_SMOKE_MATRIX_SHA256}" "${CONTROLLED_RECOVERY_SMOKE_MATERIALIZATION_SHA256}" \
    "${CONTROLLED_RECOVERY_SMOKE_ROW_INDEX}" <<'PY'
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "kind": "lehome_rollout_preemption_context",
    "active": sys.argv[9] == "true",
    "run_id": sys.argv[2],
    "run_root": sys.argv[3],
    "database": sys.argv[4],
    "attempt_matrix": sys.argv[5],
    "attempt_matrix_sha256": sys.argv[6],
    "max_attempts": int(sys.argv[7]),
    "target_accepted": int(sys.argv[8]),
}
if sys.argv[10] == "1":
    payload.update({
        "controlled_recovery_smoke": True,
        "controlled_recovery_smoke_run_id": sys.argv[11],
        "controlled_recovery_smoke_matrix_sha256": sys.argv[12],
        "controlled_recovery_smoke_materialization_sha256": sys.argv[13],
        "controlled_recovery_smoke_row_index": int(sys.argv[14]),
    })
temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
finally:
    if temporary.exists():
        temporary.unlink()
PY
}

write_preemption_context true
if [ "${ENABLE_HF_UPLOAD}" = "1" ]; then
  if [ -L "${HF_TOKEN_FILE}" ] || [ ! -f "${HF_TOKEN_FILE}" ]; then
    echo "production Hub upload requires a real LEHOME_HF_TOKEN_FILE" >&2
    exit 2
  fi
  if [ "$(stat -c '%u:%a' "${HF_TOKEN_FILE}")" != "1234:600" ]; then
    echo "LEHOME_HF_TOKEN_FILE must be owned by uid 1234 with mode 0600" >&2
    exit 2
  fi
fi
mkdir -p "${RECEIPT_DIR}" /eval/logs /kitcache
if [ "${ENABLE_HF_UPLOAD}" = "1" ]; then
  mkdir -p "${CAMPAIGN_ROOT}/accepted" "${CAMPAIGN_ROOT}/hf-sync-receipts" "${CAMPAIGN_ROOT}/hf-readback"
  if [ "${EVALUATION_TERMINAL_UPLOAD}" = "1" ]; then
    mkdir -p "${CAMPAIGN_ROOT}/evaluation-terminal"
  fi
  mkdir -p "${CAMPAIGN_ROOT}/hf-cache"
fi
mkdir -p /kitcache/home /kitcache/xdg /kitcache/config /kitcache/ov
# Preserve the useful Isaac/Kit warm state on the shared disk while keeping
# each active worker's hot cache on the boot SSD. This turns a preemption into
# a small local copy instead of a cold extension/cache rebuild.
if [ ! -e /kitcache/home/.nvidia-omniverse ] && [ -d "${KIT_SEED}" ]; then
  for entry in .nvidia-omniverse .nv; do
    if [ -e "${KIT_SEED}/${entry}" ]; then
      cp -a "${KIT_SEED}/${entry}" /kitcache/home/
    fi
  done
  if [ -d "${KIT_SEED}/xdg" ]; then cp -a "${KIT_SEED}/xdg/." /kitcache/xdg/; fi
  if [ -d "${KIT_SEED}/config" ]; then cp -a "${KIT_SEED}/config/." /kitcache/config/; fi
  if [ -d "${KIT_SEED}/data" ]; then cp -a "${KIT_SEED}/data/." /kitcache/ov/; fi
fi
if [ -x /opt/lehome/rollout_appliance/prepare-merged-lehome.sh ]; then
  /opt/lehome/rollout_appliance/prepare-merged-lehome.sh || true
fi
chown -R 1234:1234 "${CAMPAIGN_ROOT}" /eval/logs /kitcache || true
# The campaign tree belongs to Isaac workers, except for the policy gateway's
# campaign-scoped receipt directory. Set this last so the recursive worker
# ownership cannot take it back from the unprivileged policy-server UID.
chown -R 10001:10001 "${RECEIPT_DIR}" || true

# The CPU-only finalizer follows each worker's immutable ledger handoff. It
# must start before workers so accepted successes advance the 150-episode cap
# while free workers continue leasing without wave barriers.
FINALIZER_NAME="lehome-campaign-finalizer"
UPLOADER_NAME="lehome-campaign-uploader"
FINALIZER_PID=""
UPLOADER_PID=""
POLICY_PID=""

cleanup_campaign() {
  if [[ "${FINALIZER_PID:-}" =~ ^[0-9]+$ ]]; then
    kill "${FINALIZER_PID}" 2>/dev/null || true
  fi
  if [[ "${UPLOADER_PID:-}" =~ ^[0-9]+$ ]]; then
    kill "${UPLOADER_PID}" 2>/dev/null || true
  fi
  docker rm -f "${FINALIZER_NAME}" >/dev/null 2>&1 || true
  docker rm -f "${UPLOADER_NAME}" >/dev/null 2>&1 || true
  lehome_cleanup_policy "${POLICY_PID:-}"
}
trap cleanup_campaign EXIT

docker rm -f "${FINALIZER_NAME}" >/dev/null 2>&1 || true
FINALIZER_SMOKE_FLAG=()
if [ "${CONTROLLED_RECOVERY_SMOKE}" = "1" ]; then
  FINALIZER_SMOKE_FLAG=(--controlled-recovery-smoke)
fi
FINALIZER_EVALUATION_FLAG=()
if [ "${EVALUATION_TERMINAL_UPLOAD}" = "1" ]; then
  FINALIZER_EVALUATION_FLAG=(--evaluation-terminal)
fi
docker run --rm --user 1234:1234 --network none \
  --name "${FINALIZER_NAME}" \
  -v "${WORKSPACE}:/mnt/lehome" \
  -v /opt/lehome/scripts:/opt/lehome/scripts:ro \
  -v /opt/lehome/source/lehome:/opt/lehome/source/lehome:ro \
  -v /opt/lehome/trainer/src:/opt/lehome/trainer/src:ro \
  -v /opt/lehome/rollout_appliance:/opt/lehome/rollout_appliance:ro \
  -e PYTHONPATH=/opt/lehome/source/lehome:/opt/lehome/trainer/src:/opt/lehome \
  --entrypoint /opt/lehome-challenge/.venv/bin/python \
  "${ROLLOUT_IMAGE}" \
  /opt/lehome/scripts/run_groot_artifact_sync.py \
    --role finalizer \
    --database "${LEDGER}" \
    --attempt-matrix "${MATRIX}" \
    --run-root "${CAMPAIGN_ROOT}" \
    --max-attempts "${LEDGER_MAX_ATTEMPTS}" \
    --target-accepted "${TARGET_ACCEPTED}" \
    "${FINALIZER_SMOKE_FLAG[@]}" \
    "${FINALIZER_EVALUATION_FLAG[@]}" &
FINALIZER_PID=$!

if [ "${ENABLE_HF_UPLOAD}" = "1" ]; then
  UPLOADER_ROLE="uploader"
  UPLOADER_ROOT_FLAG=(--accepted-root "${CAMPAIGN_ROOT}/accepted")
  if [ "${EVALUATION_TERMINAL_UPLOAD}" = "1" ]; then
    UPLOADER_ROLE="evaluation-uploader"
    UPLOADER_ROOT_FLAG=(--terminal-root "${CAMPAIGN_ROOT}/evaluation-terminal")
  fi
  docker rm -f "${UPLOADER_NAME}" >/dev/null 2>&1 || true
  docker run --rm --user 1234:1234 \
    --name "${UPLOADER_NAME}" \
    -v "${WORKSPACE}:/mnt/lehome" \
    -v /opt/lehome/scripts:/opt/lehome/scripts:ro \
    -v /opt/lehome/source/lehome:/opt/lehome/source/lehome:ro \
    -v /opt/lehome/trainer/src:/opt/lehome/trainer/src:ro \
    -v "${HF_TOKEN_FILE}:/run/secrets/hf_token:ro" \
    -e PYTHONPATH=/opt/lehome/source/lehome:/opt/lehome/trainer/src:/opt/lehome \
    -e "HOME=${CAMPAIGN_ROOT}/hf-cache" \
    -e "HF_HOME=${CAMPAIGN_ROOT}/hf-cache" \
    -e "XDG_CACHE_HOME=${CAMPAIGN_ROOT}/hf-cache" \
    --entrypoint /opt/lehome-challenge/.venv/bin/python \
    "${ROLLOUT_IMAGE}" \
    /opt/lehome/scripts/run_groot_artifact_sync.py \
      --role "${UPLOADER_ROLE}" \
      "${UPLOADER_ROOT_FLAG[@]}" \
      --receipts-root "${CAMPAIGN_ROOT}/hf-sync-receipts" \
      --readback-root "${CAMPAIGN_ROOT}/hf-readback" \
      --repository "${ROLLOUT_REPOSITORY}" \
      --round-id "${ROUND_ID}" \
      --revision "${HF_REVISION}" \
      --token-file /run/secrets/hf_token \
      --poll-seconds 10 \
      --failure-backoff-seconds 300 &
  UPLOADER_PID=$!
fi

# Policy server owns CUDA. Staged gateway evicts idle leftover sessions.
docker rm -f lehome-12k-policy >/dev/null 2>&1 || true
# A resumed campaign must never admit workers against the prior gateway's
# readiness receipt. The immutable policy.jsonl history remains preserved.
rm -f "${RECEIPT_DIR}/ready.json" "${RECEIPT_DIR}/metrics.json"
docker run --rm --gpus all --user 10001:10001 --network host --ipc=host \
  --name lehome-12k-policy \
  -w /cache/models \
  -v "${CHECKPOINT_DIR}:/policy:ro" \
  -v /opt/lehome/scripts:/opt/lehome/scripts:ro \
  -v /opt/lehome/source/lehome:/opt/lehome-src:ro \
  -v "${RECEIPT_DIR}:/receipts" \
  -v "${WORKSPACE}/cache:/cache" \
  -v "${WORKSPACE}/cache/isaac-groot-overlay/nvidia:/opt/isaac-groot/nvidia:ro" \
  -v "${WORKSPACE}/cache/isaac-groot-overlay/exclude:/opt/isaac-groot/.git/info/exclude:ro" \
  -e PYTHONPATH=/opt/isaac-groot:/opt/lehome-src:/opt/lehome \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint /opt/runtime/bin/python \
  "${TRAINER_IMAGE}" \
  /opt/lehome/scripts/run_groot_batched_policy_server.py \
    --model-path /policy \
    --policy-sha256 "${POLICY_SHA256}" \
    --host 127.0.0.1 \
    --port 15555 \
    --device cuda:0 \
    --seed 12000 \
    --ready-file /receipts/ready.json \
    --metrics-file /receipts/metrics.json \
    --receipt-file /receipts/policy.jsonl &
POLICY_PID=$!

ready_file="${RECEIPT_DIR}/ready.json"
for _ in $(seq 1 180); do
  if python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); raise SystemExit(0 if data.get("ready") is True else 1)' "${ready_file}" 2>/dev/null; then
    break
  fi
  sleep 2
done
python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); raise SystemExit(0 if data.get("ready") is True else 1)' "${ready_file}"
chmod 0644 "${ready_file}" || true
chmod 0755 "${RECEIPT_DIR}" || true

launch_worker() {
  local index="$1"
  local worker_garment="${2:-${INITIAL_GARMENT}}"
  local worker_identity="${3:-worker-${index}}"
  local session_id="camp12k-w${index}-$(uuidgen | tr '[:upper:]' '[:lower:]')"
  local kit="/kitcache/w${index}"
  mkdir -p "${kit}/home" "${kit}/tmp" "${kit}/xdg" "${kit}/config" "${kit}/ov"
  # Seed isolated cache from the already-populated shared kitcache if empty.
  if [ ! -e "${kit}/home/.nvidia-omniverse" ] && [ -d /kitcache/home ]; then
    cp -a /kitcache/home/. "${kit}/home/" 2>/dev/null || true
    cp -a /kitcache/ov/. "${kit}/ov/" 2>/dev/null || true
    cp -a /kitcache/xdg/. "${kit}/xdg/" 2>/dev/null || true
    cp -a /kitcache/config/. "${kit}/config/" 2>/dev/null || true
  fi
  chown -R 1234:1234 "${kit}" || true
  docker rm -f "lehome-camp12k-w${index}" >/dev/null 2>&1 || true
  # Each Isaac process needs its own POSIX shared-memory namespace. Sharing
  # host IPC makes concurrent Kit instances collide on the global messageBus
  # queue and segfault during extension startup.
  local docker_status=0
  docker run --rm --gpus all --user 1234:1234 --init --network host --shm-size=8g \
    --name "lehome-camp12k-w${index}" \
    -w /opt/lehome-challenge \
    -v "${WORKSPACE}:/mnt/lehome" \
    -v "${WORKSPACE}/eval/assets:/opt/lehome-challenge/Assets:ro" \
    -v /opt/lehome:/opt/lehome:ro \
    -v /opt/lehome/merged/lehome:/opt/lehome-challenge/source/lehome/lehome:ro \
    -v /opt/lehome/scripts:/opt/lehome-challenge/scripts:ro \
    -v /opt/lehome/pydeps:/pydeps:ro \
    -v /eval/logs:/eval/logs \
    -v /eval/logs:/opt/lehome-challenge/logs \
    -v "${kit}:/kitcache" \
    -e PYTHONEXE=/opt/lehome-challenge/.venv/bin/python \
    -e PYTHONPATH=/pydeps:/opt/lehome:/opt/lehome-challenge/source/lehome:/opt/lehome-challenge/third_party/IsaacLab/source/isaaclab:/opt/lehome-challenge/third_party/IsaacLab/source/isaaclab_tasks \
    -e HOME=/kitcache/home \
    -e TMPDIR=/kitcache/tmp \
    -e XDG_CACHE_HOME=/kitcache/xdg \
    -e XDG_DATA_HOME=/kitcache/xdg \
    -e XDG_CONFIG_HOME=/kitcache/config \
    -e OMNI_DATA_PATH=/kitcache/ov \
    -e OMNI_USER_DIR=/kitcache/ov \
    -e LEHOME_DISABLE_KEYBOARD=1 \
    -e LEHOME_EVALUATION_TERMINAL_UPLOAD="${EVALUATION_TERMINAL_UPLOAD}" \
    -e LEHOME_EVALUATION_GARMENT_AFFINITY="${FRESH_GARMENT_WAVES}" \
    -e LEHOME_SUCCESS_REPLAY_CAMPAIGN="${SUCCESS_REPLAY_CAMPAIGN}" \
    -e LEHOME_HARD_STATE_CAMPAIGN="${HARD_STATE_CAMPAIGN}" \
    --entrypoint /isaac-sim/python.sh \
    "${ROLLOUT_IMAGE}" \
    /opt/lehome/scripts/run_groot_persistent_worker.py \
      --headless \
      --database "${LEDGER}" \
      --attempt-matrix "${MATRIX}" \
      --worker-id "${worker_identity}" \
      --session-id "${session_id}" \
      --output-root "${CAMPAIGN_ROOT}/${worker_identity}" \
      --renderer-device cuda:0 \
      --policy-device cuda:0 \
      --simulator-device "${SIMULATOR_DEVICE}" \
      --policy-gateway-endpoint tcp://127.0.0.1:15555 \
      --policy-sha256 "${POLICY_SHA256}" \
      --policy-repo "${POLICY_REPO}" \
      --policy-revision "${POLICY_REVISION}" \
      --policy-step "${POLICY_STEP}" \
      --policy-artifact-sha256 "${POLICY_ARTIFACT_SHA256}" \
      --policy-timeout-seconds 180 \
      --preparation-timeout-seconds "${PREPARATION_TIMEOUT_SECONDS}" \
      --source-finalization-timeout-seconds "${SOURCE_FINALIZATION_TIMEOUT_SECONDS}" \
      --policy-ready-file "${RECEIPT_DIR}/ready.json" \
      --initial-garment "${worker_garment}" \
      --seed 101 \
      --garment_name Top_Long_Seen_0 \
      --max_steps "${MAX_STEPS}" \
      --max-attempts "${MAX_ATTEMPTS}" \
      --target-accepted "${TARGET_ACCEPTED}" \
      --save_video || docker_status=$?
  if [ "${docker_status}" -ne 0 ]; then
    return "${docker_status}"
  fi
  # Isaac's outer launcher has occasionally returned zero after an inner
  # exception. A process exit is clean only when it left no live lease behind.
  if ! lehome_worker_exit_is_clean "${LEDGER}" "${worker_identity}"; then
    echo "worker ${index} exited while still owning an active lease" >&2
    return 71
  fi
  return 0
}

# Production remains exactly four workers; lower width is an explicit smoke.
worker_status=0
if [ "${FRESH_GARMENT_WAVES}" = "1" ]; then
  run_garment_slot() {
    local index="$1"
    local garment_index=$((index - 1))
    local worker_identity
    while [ "${garment_index}" -lt "${#evaluation_garments[@]}" ]; do
      worker_identity="worker-$((garment_index + 1))-${index}"
      lehome_supervise_worker "${index}" "${MAX_WORKER_RESTARTS}" launch_worker \
        "${evaluation_garments[${garment_index}]}" "${worker_identity}" &
      local supervised_pid="$!"
      if ! wait "${supervised_pid}"; then
        return 1
      fi
      garment_index=$((garment_index + WORKER_COUNT))
    done
  }
  worker_pids=()
  for index in $(seq 1 "${WORKER_COUNT}"); do
    run_garment_slot "${index}" &
    worker_pids+=("$!")
  done
  for worker_pid in "${worker_pids[@]}"; do
    if ! wait "${worker_pid}"; then
      worker_status=1
    fi
  done
else
  worker_pids=()
  for index in $(seq 1 "${WORKER_COUNT}"); do
    lehome_supervise_worker "${index}" "${MAX_WORKER_RESTARTS}" launch_worker &
    worker_pids+=("$!")
  done
  for worker_pid in "${worker_pids[@]}"; do
    if ! wait "${worker_pid}"; then
      worker_status=1
    fi
  done
fi

pending_terminal_count() {
  python3 - "${LEDGER}" <<'PY'
import sqlite3
import sys

connection = sqlite3.connect(sys.argv[1])
count = connection.execute(
    "SELECT COUNT(*) FROM events AS pending "
    "WHERE pending.event_type = 'terminal_pending_validation' "
    "AND NOT EXISTS ("
    "SELECT 1 FROM events AS settled "
    "WHERE settled.attempt_id = pending.attempt_id "
    "AND settled.event_id > pending.event_id "
    "AND settled.event_type IN ('accepted', 'rejected', 'infrastructure_abort')"
    ")"
).fetchone()[0]
print(count)
PY
}

# Workers can finish just before the finalizer's next poll. Keep the CPU-only
# service alive until every durable terminal handoff is settled.
pending_count="$(pending_terminal_count)"
for _ in $(seq 1 120); do
  if [ "${pending_count}" = "0" ]; then
    break
  fi
  if [ "$(docker inspect --format '{{.State.Running}}' "${FINALIZER_NAME}" 2>/dev/null || true)" != "true" ]; then
    echo "finalizer exited with ${pending_count} terminal handoffs still pending" >&2
    worker_status=1
    break
  fi
  sleep 1
  pending_count="$(pending_terminal_count)"
done
if [ "${pending_count}" != "0" ]; then
  echo "finalizer did not drain ${pending_count} terminal handoffs" >&2
  worker_status=1
fi

pending_upload_count() {
  local episode_root="${CAMPAIGN_ROOT}/accepted"
  if [ "${EVALUATION_TERMINAL_UPLOAD}" = "1" ]; then
    episode_root="${CAMPAIGN_ROOT}/evaluation-terminal"
  fi
  python3 - "${episode_root}" "${CAMPAIGN_ROOT}/hf-sync-receipts" <<'PY'
import json
from pathlib import Path
import sys

accepted, receipts = map(Path, sys.argv[1:])
pending = 0
if accepted.is_dir():
    for episode in accepted.iterdir():
        if not episode.is_dir() or episode.is_symlink():
            continue
        receipt = receipts / f"{episode.name}.sync.json"
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pending += 1
            continue
        if payload.get("readback_verified") is not True:
            pending += 1
print(pending)
PY
}

if [ "${ENABLE_HF_UPLOAD}" = "1" ]; then
  upload_pending="$(pending_upload_count)"
  for _ in $(seq 1 1800); do
    if [ "${upload_pending}" = "0" ]; then
      break
    fi
    if [ "$(docker inspect --format '{{.State.Running}}' "${UPLOADER_NAME}" 2>/dev/null || true)" != "true" ]; then
      echo "Hub uploader exited with ${upload_pending} terminal episodes still local-only" >&2
      worker_status=1
      break
    fi
    sleep 1
    upload_pending="$(pending_upload_count)"
  done
  if [ "${upload_pending}" != "0" ]; then
    echo "Hub uploader did not readback-verify ${upload_pending} terminal episodes" >&2
    worker_status=1
  fi
  docker stop --time 10 "${UPLOADER_NAME}" >/dev/null 2>&1 || true
  if [[ "${UPLOADER_PID}" =~ ^[0-9]+$ ]]; then
    wait "${UPLOADER_PID}" 2>/dev/null || true
  fi
  UPLOADER_PID=""

  # A completed campaign is not a trainable immutable round until the exact
  # accepted ledger set is bound to readback-verified Hub receipts. Seal it
  # before tearing down the CPU control plane so no manual post-run step can
  # accidentally select a partial or different episode set.
  if [ "${worker_status}" = "0" ] && [ "${SKIP_ROUND_SEAL}" = "0" ] \
      && [ "${EVALUATION_TERMINAL_UPLOAD}" = "0" ]; then
    docker run --rm --user 1234:1234 --network none \
      -v "${WORKSPACE}:/mnt/lehome" \
      -v /opt/lehome/scripts:/opt/lehome/scripts:ro \
      -v /opt/lehome/source/lehome:/opt/lehome/source/lehome:ro \
      -v /opt/lehome/trainer/src:/opt/lehome/trainer/src:ro \
      -v /opt/lehome/rollout_appliance:/opt/lehome/rollout_appliance:ro \
      -e PYTHONPATH=/opt/lehome/source/lehome:/opt/lehome/trainer/src:/opt/lehome \
      --entrypoint /opt/lehome-challenge/.venv/bin/python \
      "${ROLLOUT_IMAGE}" \
      /opt/lehome/scripts/run_groot_artifact_sync.py \
        --role sealer \
        --once \
        --database "${LEDGER}" \
        --attempt-matrix "${MATRIX}" \
        --receipts-root "${CAMPAIGN_ROOT}/hf-sync-receipts" \
        --round-id "${ROUND_ID}" \
        --seal-receipt "${CAMPAIGN_ROOT}/${ROUND_ID}.strict.seal.json" \
        --max-attempts "${MAX_ATTEMPTS}" \
        --target-accepted "${TARGET_ACCEPTED}"
  fi
fi

docker stop --time 10 "${FINALIZER_NAME}" >/dev/null 2>&1 || true
if [[ "${FINALIZER_PID}" =~ ^[0-9]+$ ]]; then
  wait "${FINALIZER_PID}" 2>/dev/null || true
fi
FINALIZER_PID=""
write_preemption_context false
exit "${worker_status}"
