#!/usr/bin/env bash
# Run capacity reconciliation only with the root-owned service-account key
# systemd materializes for this unit. No default Nebius CLI profile, metadata
# token, or operator shell environment is an accepted authority here.
set -euo pipefail
umask 077
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=/opt/lehome/trainer/src
unset NEBIUS_CONFIG NEBIUS_PROFILE NEBIUS_TOKEN NEBIUS_TOKEN_FILE

ENV_FILE=/etc/lehome/experiment-capacity.env
READY_FILE=/etc/lehome/experiment-bootstrap.ready
[[ -f "${ENV_FILE}" && ! -L "${ENV_FILE}" && "$(stat -c '%a:%u:%g' "${ENV_FILE}")" == "600:0:0" ]] || exit 2
[[ -f "${READY_FILE}" && ! -L "${READY_FILE}" && "$(stat -c '%a:%u:%g' "${READY_FILE}")" == "600:0:0" ]] || exit 2
for key in LEHOME_CAPACITY_CONFIG LEHOME_CAPACITY_CONTROLLER_URL LEHOME_CAPACITY_CONTROLLER_CA_FILE LEHOME_CAPACITY_RECEIPT_LOG LEHOME_CAPACITY_NEBIUS_CONFIG_FILE LEHOME_CAPACITY_NEBIUS_PROFILE LEHOME_CAPACITY_NEBIUS_SERVICE_ACCOUNT_ID LEHOME_CAPACITY_NEBIUS_PUBLIC_KEY_ID LEHOME_CAPACITY_NEBIUS_PROJECT_ID; do
  [[ -n "${!key:-}" ]] || exit 2
done
[[ "${LEHOME_CAPACITY_CONFIG}" == /etc/lehome/capacity.json ]] || exit 2
[[ "${LEHOME_CAPACITY_NEBIUS_CONFIG_FILE}" == /run/lehome-capacity/nebius-config.yaml ]] || exit 2
[[ "${LEHOME_CAPACITY_RECEIPT_LOG}" == /var/lib/lehome/controller/audit/capacity.jsonl ]] || exit 2
[[ "${LEHOME_CAPACITY_CONTROLLER_URL}" == https://* ]] || exit 2
[[ "${LEHOME_CAPACITY_CONTROLLER_CA_FILE}" == /etc/lehome/tls/controller-ca.crt ]] || exit 2
[[ "${LEHOME_CAPACITY_NEBIUS_PROFILE}" == lehome-capacity ]] || exit 2

credential_dir="${CREDENTIALS_DIRECTORY:-}"
[[ -n "${credential_dir}" && -d "${credential_dir}" && ! -L "${credential_dir}" ]] || exit 2
for name in controller-token nebius-private-key; do
  [[ -f "${credential_dir}/${name}" && ! -L "${credential_dir}/${name}" ]] || exit 2
done

runtime_dir=/run/lehome-capacity
install -d -m 0700 -o root -g root "${runtime_dir}"
runtime_token="${runtime_dir}/controller-token"
runtime_key="${runtime_dir}/nebius-private-key.pem"
install -m 0600 -o root -g root "${credential_dir}/controller-token" "${runtime_token}"
install -m 0600 -o root -g root "${credential_dir}/nebius-private-key" "${runtime_key}"
cleanup() { rm -f -- "${runtime_token}" "${runtime_key}" "${LEHOME_CAPACITY_NEBIUS_CONFIG_FILE}"; }
trap cleanup EXIT TERM INT

# The only non-compute CLI call creates an isolated local profile from the
# systemd credential. Every remote Compute request below carries --config, so
# the CLI can never fall back to a default profile or metadata authorization.
if ! nebius --config "${LEHOME_CAPACITY_NEBIUS_CONFIG_FILE}" profile create \
  --endpoint api.nebius.cloud \
  --service-account-id "${LEHOME_CAPACITY_NEBIUS_SERVICE_ACCOUNT_ID}" \
  --public-key-id "${LEHOME_CAPACITY_NEBIUS_PUBLIC_KEY_ID}" \
  --private-key-file "${runtime_key}" \
  --profile "${LEHOME_CAPACITY_NEBIUS_PROFILE}" \
  --parent-id "${LEHOME_CAPACITY_NEBIUS_PROJECT_ID}" >/dev/null 2>&1; then
  echo "Nebius capacity service-account authentication is unavailable" >&2
  exit 2
fi
[[ -f "${LEHOME_CAPACITY_NEBIUS_CONFIG_FILE}" && ! -L "${LEHOME_CAPACITY_NEBIUS_CONFIG_FILE}" && "$(stat -c '%a:%u:%g' "${LEHOME_CAPACITY_NEBIUS_CONFIG_FILE}")" == "600:0:0" ]] || exit 2

exec /usr/bin/python3 /opt/lehome/scripts/run_lehome_capacity_lifecycle.py \
  --execute \
  --config "${LEHOME_CAPACITY_CONFIG}" \
  --controller-url "${LEHOME_CAPACITY_CONTROLLER_URL}" \
  --controller-ca-file "${LEHOME_CAPACITY_CONTROLLER_CA_FILE}" \
  --token-file "${runtime_token}" \
  --receipt-log "${LEHOME_CAPACITY_RECEIPT_LOG}" \
  --nebius-config-file "${LEHOME_CAPACITY_NEBIUS_CONFIG_FILE}"
