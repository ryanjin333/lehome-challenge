#!/usr/bin/env bash
# Run a hash-bound portable schedule through its local source hydration.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CAMPAIGN="${SCRIPT_DIR}/run_12k_campaign.sh"
WORKSPACE="${LEHOME_WORKSPACE:-/mnt/lehome}"
MATRIX="${LEHOME_CONTROLLED_RECOVERY_MATRIX:-}"
EXPECTED_SHA256="${LEHOME_CONTROLLED_RECOVERY_MATRIX_SHA256:-}"
MATERIALIZATION="${LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION:-}"
MATERIALIZATION_SHA256="${LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION_SHA256:-}"
DEPLOYMENT_GATE="${LEHOME_DEPLOYMENT_GATE_PATH:-/etc/lehome/experiment-deployment-gate.json}"
DEPLOYMENT_GATE_SHA256="${LEHOME_DEPLOYMENT_GATE_SHA256:-}"
TRAINER_SRC="${LEHOME_TRAINER_SRC:-/opt/lehome/trainer/src}"
WORKER_COUNT="${LEHOME_WORKER_COUNT:-4}"
MAX_ATTEMPTS="${LEHOME_MAX_ATTEMPTS:-96}"
TARGET_ACCEPTED="${LEHOME_TARGET_ACCEPTED:-8}"
CAMPAIGN_ROOT="${LEHOME_CAMPAIGN_ROOT:-${WORKSPACE}/eval/campaign-controlled-recovery-v1}"
# Each readback-verified accepted episode uploads immediately for preemptible
# durability. This staging prefix is not a sealed/training-ready round; only
# the base campaign's sealer can promote its exact 4/1/3 ledger set.
ROUND_ID="${LEHOME_ROUND_ID:-controlled-recovery-v1-unsealed-staging}"
ORIGINAL_12K_POLICY_SHA256="e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa"
ORIGINAL_12K_CHECKPOINT="${LEHOME_ORIGINAL_12K_CHECKPOINT:-${WORKSPACE}/eval/policies/original_baseline}"

if [ ! -f "${BASE_CAMPAIGN}" ] || [ -z "${MATRIX}" ] || [ -z "${EXPECTED_SHA256}" ] || [ -z "${MATERIALIZATION}" ] || [ -z "${MATERIALIZATION_SHA256}" ]; then echo "controlled recovery requires appliance, matrix, materialization, and pinned SHA-256 values" >&2; exit 2; fi
if [ "${WORKER_COUNT}" != "4" ]; then echo "LEHOME_WORKER_COUNT must be exactly 4" >&2; exit 2; fi
if [ "${TARGET_ACCEPTED}" != "8" ]; then echo "LEHOME_TARGET_ACCEPTED must be exactly 8" >&2; exit 2; fi
if ! [[ "${MAX_ATTEMPTS}" =~ ^([8-9]|[1-8][0-9]|9[0-6])$ ]]; then echo "LEHOME_MAX_ATTEMPTS must be in 8..96" >&2; exit 2; fi
for path in "${MATRIX}" "${MATERIALIZATION}"; do if [[ "${path}" != /* ]] || [ -L "${path}" ] || [ ! -f "${path}" ] || [ -L "${path}.sha256" ] || [ ! -f "${path}.sha256" ]; then echo "controlled recovery artifact is unsafe" >&2; exit 2; fi; done
for expected in "${EXPECTED_SHA256}" "${MATERIALIZATION_SHA256}"; do if ! [[ "${expected}" =~ ^[0-9a-f]{64}$ ]]; then echo "controlled recovery SHA-256 is invalid" >&2; exit 2; fi; done
if [[ "${DEPLOYMENT_GATE}" != /* ]] || [ -L "${DEPLOYMENT_GATE}" ] || [ ! -f "${DEPLOYMENT_GATE}" ] || ! [[ "${DEPLOYMENT_GATE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "controlled recovery requires an immutable deployment gate path and SHA-256" >&2
  exit 2
fi
PYTHONPATH="${TRAINER_SRC}${PYTHONPATH:+:${PYTHONPATH}}" python3 - "${DEPLOYMENT_GATE}" "${DEPLOYMENT_GATE_SHA256}" <<'PY'
import sys
from lehome_train.groot.experiment_deployment_gate import (
    load_deployment_gate,
    require_recovery_collection_admission,
)

require_recovery_collection_admission(load_deployment_gate(sys.argv[1], sys.argv[2]))
PY
if [ "$(sha256sum "${MATRIX}" | awk '{print $1}')" != "${EXPECTED_SHA256}" ] || [ "$(tr -d '\r\n' < "${MATRIX}.sha256")" != "${EXPECTED_SHA256}" ] || [ "$(sha256sum "${MATERIALIZATION}" | awk '{print $1}')" != "${MATERIALIZATION_SHA256}" ] || [ "$(tr -d '\r\n' < "${MATERIALIZATION}.sha256")" != "${MATERIALIZATION_SHA256}" ]; then echo "controlled recovery artifact SHA-256 mismatch" >&2; exit 2; fi

python3 - "${MATRIX}" "${MATERIALIZATION}" "${EXPECTED_SHA256}" "${MAX_ATTEMPTS}" <<'PY'
import hashlib, json, re, sys
from collections import Counter
from pathlib import Path
try:
    matrix, local = json.loads(Path(sys.argv[1]).read_text()), json.loads(Path(sys.argv[2]).read_text())
except (OSError, ValueError) as error: raise SystemExit(f"controlled recovery artifact is malformed: {error}")
expected, attempts = sys.argv[3], int(sys.argv[4]); caps = {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}
if not isinstance(matrix, dict) or matrix.get("schema_version") != 3 or matrix.get("kind") != "controlled_success_recovery_matrix_v3" or matrix.get("target_accepted") != 8 or matrix.get("category_acceptance_caps") != caps or not isinstance(matrix.get("rows"), list) or len(matrix["rows"]) != attempts: raise SystemExit("controlled recovery matrix acceptance contract is invalid")
rows = matrix["rows"]
if not all(isinstance(row, dict) for row in rows) or any(Counter(row.get("category") for row in rows).get(category, 0) < cap for category, cap in caps.items() if cap): raise SystemExit("controlled recovery matrix retry schedule is invalid")
if not isinstance(local, dict) or local.get("schema_version") != 3 or local.get("kind") != "controlled_success_recovery_materialization_v3" or local.get("matrix_sha256") != expected or local.get("target_accepted") != 8 or local.get("category_acceptance_caps") != caps or not isinstance(local.get("rows"), list) or len(local["rows"]) != attempts: raise SystemExit("controlled recovery materialization does not bind matrix")
seen, attempt_ids, trial_ids, seeds, perturbations = set(), set(), set(), set(), set()
for row, hydrated in zip(rows, local["rows"], strict=True):
    if not isinstance(row, dict) or row.get("recovery_kind") != "controlled_success_recovery_snapshot_v3" or row.get("category_acceptance_cap") != caps.get(row.get("category")) or row.get("strategy") != "canonical": raise SystemExit("controlled recovery row is invalid")
    if any(isinstance(value, str) and value.startswith("/") for value in row.values()): raise SystemExit("portable controlled recovery matrix contains a local path")
    if not isinstance(hydrated, dict) or hydrated.get("controlled_matrix_sha256") != expected or any(not isinstance(hydrated.get(key), str) or not hydrated[key].startswith("/") or Path(hydrated[key]).is_symlink() or not Path(hydrated[key]).is_file() for key in ("source_reset", "source_annotations", "source_continuation_snapshot")): raise SystemExit("controlled recovery materialization paths are unsafe")
    if {key: value for key, value in hydrated.items() if key not in {"source_reset", "source_annotations", "source_continuation_snapshot", "controlled_matrix_sha256"}} != row: raise SystemExit("controlled recovery materialization differs from matrix")
    if not isinstance(row.get("prefix_stop"), int) or not isinstance(row.get("source_first_success_step"), int) or not 0 < row["prefix_stop"] < row["source_first_success_step"] or row["prefix_stop"] % 16: raise SystemExit("controlled recovery continuation boundary is invalid")
    for key in ("source_reset_sha256", "source_annotations_sha256", "source_continuation_snapshot_sha256", "source_state_fingerprint", "perturbation_fingerprint", "source_state_perturbation_fingerprint", "source_episode_digest"):
        if not isinstance(row.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", row[key]) is None: raise SystemExit("controlled recovery immutable digest is invalid")
    if hashlib.sha256(Path(hydrated["source_reset"]).read_bytes()).hexdigest() != row["source_reset_sha256"] or hashlib.sha256(Path(hydrated["source_annotations"]).read_bytes()).hexdigest() != row["source_annotations_sha256"] or hashlib.sha256(Path(hydrated["source_continuation_snapshot"]).read_bytes()).hexdigest() != row["source_continuation_snapshot_sha256"]: raise SystemExit("controlled recovery materialization source hashes are invalid")
    if not isinstance(row.get("attempt_id"), str) or not row["attempt_id"] or row["attempt_id"] in attempt_ids or not isinstance(row.get("trial_id"), str) or not row["trial_id"] or row["trial_id"] in trial_ids or type(row.get("perturbation_seed")) is not int or row["perturbation_seed"] < 0 or row["perturbation_seed"] in seeds or row["perturbation_fingerprint"] in perturbations: raise SystemExit("controlled recovery rows do not have unique attempt identities")
    attempt_ids.add(row["attempt_id"]); trial_ids.add(row["trial_id"]); seeds.add(row["perturbation_seed"]); perturbations.add(row["perturbation_fingerprint"])
    if row["source_state_perturbation_fingerprint"] in seen: raise SystemExit("controlled recovery reuses a source-state/perturbation fingerprint")
    seen.add(row["source_state_perturbation_fingerprint"])
PY

# The controlled-recovery smoke wrapper invokes this deliberately narrow
# validation-only gate before deriving its single immutable row.  It never
# changes the production four-worker execution tuple below.
if [ "${LEHOME_CONTROLLED_RECOVERY_VALIDATE_ONLY:-0}" = "1" ]; then
  exit 0
fi

exec env LEHOME_POLICY_SHA256="${ORIGINAL_12K_POLICY_SHA256}" LEHOME_CHECKPOINT_DIR="${ORIGINAL_12K_CHECKPOINT}" LEHOME_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" LEHOME_ATTEMPT_MATRIX="${MATERIALIZATION}" LEHOME_ATTEMPT_MATRIX_SHA256="${MATERIALIZATION_SHA256}" LEHOME_WORKER_COUNT="4" LEHOME_MAX_ATTEMPTS="${MAX_ATTEMPTS}" LEHOME_TARGET_ACCEPTED="8" LEHOME_ROUND_ID="${ROUND_ID}" bash "${BASE_CAMPAIGN}"
