#!/usr/bin/env bash
set -euo pipefail
ENV_FILE=/etc/lehome/experiment-evaluator.env

fail() {
  echo "experiment evaluator: $*" >&2
  exit 2
}

file_mode() {
  stat -c '%a' "$1" 2>/dev/null || stat -f '%Lp' "$1" 2>/dev/null
}

require_regular_0600() {
  local path="$1"
  local label="$2"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "${label} is missing or unsafe"
  [[ "$(file_mode "${path}")" == "600" ]] || fail "${label} must be mode 0600"
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

require_immutable_file() {
  local path="$1"
  local expected_sha256="$2"
  local label="$3"
  [[ "${path}" == /* ]] || fail "${label} path must be absolute"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "${label} is missing or unsafe"
  [[ "$(file_mode "${path}")" == "444" ]] || fail "${label} must be immutable mode 0444"
  [[ "${expected_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "${label} SHA-256 is invalid"
  [[ "$(sha256_file "${path}")" == "${expected_sha256}" ]] || fail "${label} SHA-256 mismatch"
}

require_immutable_directory() {
  local path="$1"
  local label="$2"
  [[ "${path}" == /* ]] || fail "${label} path must be absolute"
  [[ -d "${path}" && ! -L "${path}" ]] || fail "${label} is missing or unsafe"
  [[ "$(file_mode "${path}")" == "555" ]] || fail "${label} must be immutable mode 0555"
}

[[ -f "${ENV_FILE}" && ! -L "${ENV_FILE}" ]] || fail "intentionally disabled"

# Configuration must come from the root-managed file, never from an ambient
# service or login environment left behind by an earlier evaluator mode.
unset \
  LEHOME_CONTROLLER_URL \
  LEHOME_CONTROLLER_CA_FILE \
  LEHOME_PROMOTION_MATRIX \
  LEHOME_PROMOTION_MATRIX_SHA256 \
  LEHOME_FINAL_MATRIX \
  LEHOME_FINAL_MATRIX_SHA256 \
  LEHOME_EVALUATION_MATRIX \
  LEHOME_EVALUATION_MATRIX_SHA256 \
  LEHOME_MANIFEST_SET_SHA256 \
  LEHOME_CONTROLLER_TOKEN_FILE \
  LEHOME_HF_TOKEN_FILE \
  LEHOME_EVALUATION_ROOT \
  LEHOME_EVALUATION_MODE \
  LEHOME_PROMOTION_BASELINE_POLICY \
  LEHOME_PROMOTION_BASELINE_POLICY_SHA256 \
  LEHOME_PROMOTION_BASELINE_EVIDENCE \
  LEHOME_PROMOTION_BASELINE_EVIDENCE_SHA256 \
  LEHOME_FINAL_REPORT_REPOSITORY \
  LEHOME_FINAL_REPORT_PREFIX \
  LEHOME_SEEN_REGRESSION_EVIDENCE \
  LEHOME_FINAL_SEEN_REGRESSION_HANDOFF_ROOT
source "${ENV_FILE}"

for key in \
  LEHOME_CONTROLLER_URL \
  LEHOME_CONTROLLER_CA_FILE \
  LEHOME_MANIFEST_SET_SHA256 \
  LEHOME_CONTROLLER_TOKEN_FILE \
  LEHOME_HF_TOKEN_FILE \
  LEHOME_EVALUATION_ROOT \
  LEHOME_EVALUATION_MODE; do
  [[ -n "${!key:-}" ]] || fail "${key} is required"
done

[[ "${LEHOME_CONTROLLER_URL}" == https://* ]] || fail "controller URL must use HTTPS"
[[ "${LEHOME_CONTROLLER_CA_FILE}" == /* ]] || fail "controller CA path must be absolute"
[[ -f "${LEHOME_CONTROLLER_CA_FILE}" && ! -L "${LEHOME_CONTROLLER_CA_FILE}" ]] || fail "controller CA file is missing or unsafe"
[[ "${LEHOME_EVALUATION_ROOT}" == /* && "${LEHOME_EVALUATION_ROOT}" != *".."* ]] || fail "evaluation root is unsafe"
[[ "${LEHOME_MANIFEST_SET_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "manifest-set SHA-256 is invalid"
require_regular_0600 "${LEHOME_CONTROLLER_TOKEN_FILE}" "controller token file"
require_regular_0600 "${LEHOME_HF_TOKEN_FILE}" "Hugging Face token file"
# Campaign subprocesses inherit only this credential path.  The secret value
# itself remains in the root-managed file and is never exported here.
export LEHOME_HF_TOKEN_FILE
export PYTHONPATH=/opt/lehome/trainer/src:/opt/lehome

mode_arguments=()
case "${LEHOME_EVALUATION_MODE}" in
  promotion)
    for key in LEHOME_PROMOTION_MATRIX LEHOME_PROMOTION_MATRIX_SHA256; do
      [[ -n "${!key:-}" ]] || fail "${key} is required for promotion"
    done
    evaluation_matrix="${LEHOME_PROMOTION_MATRIX}"
    evaluation_matrix_sha256="${LEHOME_PROMOTION_MATRIX_SHA256}"
    require_immutable_file "${evaluation_matrix}" "${evaluation_matrix_sha256}" "promotion matrix"

    policy_path="${LEHOME_PROMOTION_BASELINE_POLICY:-}"
    policy_sha256="${LEHOME_PROMOTION_BASELINE_POLICY_SHA256:-}"
    evidence_path="${LEHOME_PROMOTION_BASELINE_EVIDENCE:-}"
    evidence_sha256="${LEHOME_PROMOTION_BASELINE_EVIDENCE_SHA256:-}"
    policy_configured=0
    evidence_configured=0
    if [[ -n "${policy_path}" || -n "${policy_sha256}" ]]; then
      [[ -n "${policy_path}" && -n "${policy_sha256}" ]] || fail "promotion baseline policy requires path and SHA-256"
      policy_configured=1
    fi
    if [[ -n "${evidence_path}" || -n "${evidence_sha256}" ]]; then
      [[ -n "${evidence_path}" && -n "${evidence_sha256}" ]] || fail "promotion baseline evidence requires path and SHA-256"
      evidence_configured=1
    fi
    [[ "$((policy_configured + evidence_configured))" == "1" ]] || fail "promotion requires exactly one baseline policy or baseline evidence"
    if [[ "${policy_configured}" == "1" ]]; then
      require_immutable_file "${policy_path}" "${policy_sha256}" "promotion baseline policy"
      mode_arguments+=(--baseline-policy "${policy_path}")
    else
      require_immutable_file "${evidence_path}" "${evidence_sha256}" "promotion baseline evidence"
      mode_arguments+=(--baseline-evidence "${evidence_path}")
    fi
    ;;
  final-unseen80)
    for key in \
      LEHOME_FINAL_MATRIX \
      LEHOME_FINAL_MATRIX_SHA256 \
      LEHOME_FINAL_REPORT_REPOSITORY \
      LEHOME_FINAL_REPORT_PREFIX \
      LEHOME_FINAL_SEEN_REGRESSION_HANDOFF_ROOT; do
      [[ -n "${!key:-}" ]] || fail "${key} is required for final-unseen80"
    done
    evaluation_matrix="${LEHOME_FINAL_MATRIX}"
    evaluation_matrix_sha256="${LEHOME_FINAL_MATRIX_SHA256}"
    require_immutable_file "${evaluation_matrix}" "${evaluation_matrix_sha256}" "final matrix"
    [[ "${LEHOME_FINAL_REPORT_REPOSITORY}" == */* && "${LEHOME_FINAL_REPORT_REPOSITORY}" != *[[:space:]]* ]] || fail "final report repository is invalid"
    [[ "${LEHOME_FINAL_REPORT_PREFIX}" != /* && "${LEHOME_FINAL_REPORT_PREFIX}" != *".."* && "${LEHOME_FINAL_REPORT_PREFIX}" == */ ]] || fail "final report prefix is unsafe"
    require_immutable_directory "${LEHOME_FINAL_SEEN_REGRESSION_HANDOFF_ROOT}" "final seen-regression handoff root"
    mode_arguments+=(
      --final-report-repository "${LEHOME_FINAL_REPORT_REPOSITORY}"
      --final-report-prefix "${LEHOME_FINAL_REPORT_PREFIX}"
      --seen-regression-handoff-root "${LEHOME_FINAL_SEEN_REGRESSION_HANDOFF_ROOT}"
    )
    ;;
  *) fail "evaluation mode must be promotion or final-unseen80" ;;
esac

arguments=(
  /opt/lehome/scripts/run_lehome_experiment_evaluator.py
  --controller-url "${LEHOME_CONTROLLER_URL}"
  --controller-ca-file "${LEHOME_CONTROLLER_CA_FILE}"
  --matrix "${evaluation_matrix}"
  --matrix-sha256 "${evaluation_matrix_sha256}"
  --manifest-set-sha256 "${LEHOME_MANIFEST_SET_SHA256}"
  --workers 4
  --token-file "${LEHOME_CONTROLLER_TOKEN_FILE}"
  --campaign-root "${LEHOME_EVALUATION_ROOT}"
  --mode "${LEHOME_EVALUATION_MODE}"
  --final-hf-token-file "${LEHOME_HF_TOKEN_FILE}"
  "${mode_arguments[@]}"
)

exec /opt/lehome-challenge/.venv/bin/python "${arguments[@]}"
