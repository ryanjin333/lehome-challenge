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
mkdir -p "${STAGE_DIR}/scripts"
for script in run_groot_rollout_appliance.py run_groot_batched_policy_server.py \
              run_groot_rollout_controller.py run_groot_artifact_sync.py \
              run_groot_persistent_worker.py; do
  cp "${REPO_ROOT}/scripts/${script}" "${STAGE_DIR}/scripts/${script}"
done
mkdir -p "${STAGE_DIR}/trainer"
cp -R "${REPO_ROOT}/trainer/src" "${STAGE_DIR}/trainer/src"
cp "${REPO_ROOT}/trainer/pyproject.toml" "${STAGE_DIR}/trainer/pyproject.toml"
cp -R "${REPO_ROOT}/rollout_appliance" "${STAGE_DIR}/rollout_appliance"

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
