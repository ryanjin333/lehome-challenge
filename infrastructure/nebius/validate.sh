#!/usr/bin/env bash
# One free validation entrypoint for the Nebius training/rollout templates.
# Runs static checks only: Python infrastructure tests, shell syntax,
# Packer init/fmt/validate, and Terraform init (no backend)/fmt/validate.
# It prints the paid next commands but NEVER executes them.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PACKER="infrastructure/nebius/.tools/packer"
TERRAFORM="infrastructure/nebius/.tools/terraform"

echo "== bootstrap pinned tools =="
bash infrastructure/nebius/tools/bootstrap.sh

echo "== shell syntax checks =="
for script in infrastructure/nebius/tools/bootstrap.sh \
              infrastructure/nebius/packer/scripts/*.sh \
              rollout_appliance/entrypoint.sh \
              rollout_appliance/smoke_one_episode.sh \
              rollout_appliance/run_12k_campaign.sh \
              rollout_appliance/run_controlled_recovery_campaign.sh \
              rollout_appliance/run_controlled_recovery_smoke.sh \
              rollout_appliance/run_snapshot_source_bootstrap.sh \
              rollout_appliance/run_randomized_top_short_pilot.sh; do
  bash -n "${script}"
done

echo "== python infrastructure tests =="
PYTHONPATH=.:source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/infrastructure

echo "== packer fmt + validate =="
"${PACKER}" init infrastructure/nebius/packer
"${PACKER}" fmt -check infrastructure/nebius/packer
echo "== stage rollout runtime code =="
bash infrastructure/nebius/tools/stage-rollout.sh
# Validate the directory so plugins.pkr.hcl, variables.pkr.hcl, and both
# sources load together. The builder plugin parses the service-account key
# even in offline validation, so point it at a synthetic throwaway PEM that
# bootstrap.sh generates in the ignored .tools directory.
VALIDATION_KEY="infrastructure/nebius/.tools/test-validation-key.pem"
if [[ ! -f "${VALIDATION_KEY}" ]]; then
  openssl genrsa -out "${VALIDATION_KEY}" 2048 2>/dev/null
fi
"${PACKER}" validate \
  -var 'project_id=test-project' \
  -var 'subnet_id=test-subnet' \
  -var 'service_account_id=test-sa' \
  -var 'service_account_public_key_id=test-key' \
  -var "service_account_private_key_file=${VALIDATION_KEY}" \
  -var 'rollout_parent_image_id=test-image' \
  -var 'image_version=0.0.0-test' \
  -var 'trainer_code_revision=0000000000000000000000000000000000000000' \
  -var 'rollout_code_revision=0000000000000000000000000000000000000000' \
  -var 'ghcr_pull_token=' \
  infrastructure/nebius/packer

echo "== terraform fmt + validate =="
"${TERRAFORM}" -chdir=infrastructure/nebius/terraform/experiment-pool init -backend=false
"${TERRAFORM}" -chdir=infrastructure/nebius/terraform/experiment-pool fmt -check -recursive
"${TERRAFORM}" -chdir=infrastructure/nebius/terraform/experiment-pool validate
for root in infrastructure/nebius/terraform/storage infrastructure/nebius/terraform/runtime; do
  "${TERRAFORM}" -chdir="${root}" init -backend=false
  "${TERRAFORM}" -chdir="${root}" fmt -check -recursive
  "${TERRAFORM}" -chdir="${root}" validate
done

echo
echo "All free validation passed. Paid next steps (operator-approved only):"
echo "  1. ${PACKER} build -var trainer_code_revision=<exact-40-character-commit> infrastructure/nebius/packer/training.pkr.hcl"
echo "  2. ${PACKER} build infrastructure/nebius/packer/rollout.pkr.hcl"
echo "  3. ${TERRAFORM} -chdir=infrastructure/nebius/terraform/storage apply"
echo "  4. ${TERRAFORM} -chdir=infrastructure/nebius/terraform/runtime apply -var-file=<role>.tfvars"
