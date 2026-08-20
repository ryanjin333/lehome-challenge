#!/usr/bin/env bash
# Stage the runtime code that the rollout Packer build copies into the
# temporary builder. Only runtime code is staged; no model weights, datasets,
# credentials, or experiment manifests. The staging directory is gitignored.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STAGE_DIR="${REPO_ROOT}/infrastructure/nebius/rollout-stage"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

cp -R "${REPO_ROOT}/source" "${STAGE_DIR}/source"
cp -R "${REPO_ROOT}/infrastructure/nebius/guest" "${STAGE_DIR}/guest"
mkdir -p "${STAGE_DIR}/scripts"
for script in run_groot_rollout_appliance.py run_groot_batched_policy_server.py \
              run_groot_rollout_controller.py run_groot_artifact_sync.py \
              run_groot_persistent_worker.py build_randomized_pilot_matrix.py \
              build_controlled_recovery_matrix.py \
              build_success_replay_matrix.py \
              materialize_finalist_seen_regression_handoff.py \
              run_lehome_experiment_evaluator.py \
              run_lehome_experiment_worker.py \
              summarize_groot_persistent_evaluation.py; do
  cp "${REPO_ROOT}/scripts/${script}" "${STAGE_DIR}/scripts/${script}"
done
cp "${REPO_ROOT}/scripts/__init__.py" "${STAGE_DIR}/scripts/__init__.py"
cp -R "${REPO_ROOT}/scripts/eval_policy" "${STAGE_DIR}/scripts/eval_policy"
cp -R "${REPO_ROOT}/scripts/utils" "${STAGE_DIR}/scripts/utils"
mkdir -p "${STAGE_DIR}/trainer"
cp -R "${REPO_ROOT}/trainer/src" "${STAGE_DIR}/trainer/src"
cp "${REPO_ROOT}/trainer/pyproject.toml" "${STAGE_DIR}/trainer/pyproject.toml"
cp -R "${REPO_ROOT}/rollout_appliance" "${STAGE_DIR}/rollout_appliance"
# Packer copies this staged tree verbatim. Keep the two success-replay entry
# points directly runnable during a local staged readback as well as in the
# captured image.
chmod 0755 "${STAGE_DIR}/scripts/build_success_replay_matrix.py"
chmod 0755 "${STAGE_DIR}/scripts/materialize_finalist_seen_regression_handoff.py"
chmod 0755 "${STAGE_DIR}/scripts/run_lehome_experiment_evaluator.py"
chmod 0755 "${STAGE_DIR}/rollout_appliance/run_success_replay_campaign.sh"

# Safety scan: no credential VALUES may enter the staged tree. Match only
# literal token values and PEM key blocks, not the key-name identifiers used
# by redaction/preflight code.
if grep -rIn --exclude-dir=.git \
    -e 'hf_[A-Za-z0-9]\{16,\}' \
    -e 'BEGIN [A-Z ]*PRIVATE KEY' \
    -e 'api_key[[:space:]]*[=:][[:space:]]*"[^"]\{8,\}"' \
    "${STAGE_DIR}"; then
  echo "refusing to stage: credential-shaped content detected" >&2
  exit 1
fi

echo "staged rollout runtime code into ${STAGE_DIR}"
