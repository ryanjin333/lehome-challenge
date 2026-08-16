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
              rollout_appliance/entrypoint.sh; do
  bash -n "${script}"
done

echo "== python infrastructure tests =="
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/infrastructure

echo "== packer fmt + validate =="
"${PACKER}" init infrastructure/nebius/packer
"${PACKER}" fmt -check infrastructure/nebius/packer
"${PACKER}" validate \
  -var 'project_id=test-project' \
  -var 'subnet_id=test-subnet' \
  -var 'service_account_id=test-sa' \
  -var 'service_account_public_key_id=test-key' \
  -var 'service_account_private_key_file=/dev/null' \
  -var 'image_version=0.0.0-test' \
  infrastructure/nebius/packer/training.pkr.hcl
"${PACKER}" validate \
  -var 'project_id=test-project' \
  -var 'subnet_id=test-subnet' \
  -var 'service_account_id=test-sa' \
  -var 'service_account_public_key_id=test-key' \
  -var 'service_account_private_key_file=/dev/null' \
  -var 'image_version=0.0.0-test' \
  infrastructure/nebius/packer/rollout.pkr.hcl

echo "== terraform fmt + validate =="
for root in infrastructure/nebius/terraform/storage infrastructure/nebius/terraform/runtime; do
  "${TERRAFORM}" -chdir="${root}" init -backend=false
  "${TERRAFORM}" -chdir="${root}" fmt -check -recursive
  "${TERRAFORM}" -chdir="${root}" validate
done

echo
echo "All free validation passed. Paid next steps (operator-approved only):"
echo "  1. ${PACKER} build infrastructure/nebius/packer/training.pkr.hcl"
echo "  2. ${PACKER} build infrastructure/nebius/packer/rollout.pkr.hcl"
echo "  3. ${TERRAFORM} -chdir=infrastructure/nebius/terraform/storage apply"
echo "  4. ${TERRAFORM} -chdir=infrastructure/nebius/terraform/runtime apply -var-file=<role>.tfvars"
