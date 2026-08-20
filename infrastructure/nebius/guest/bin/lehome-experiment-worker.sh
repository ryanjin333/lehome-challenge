#!/usr/bin/env bash
# Fail closed: the experiment worker never relies on rollout-ledger shutdown
# hooks. Preemption asks the leased process to stop and lets the controller
# reissue only that expired lease after the guest retained its local boundary.
set -euo pipefail
umask 077
export PYTHONPATH=/opt/lehome/trainer/src${PYTHONPATH:+:${PYTHONPATH}}

ENV_FILE=/etc/lehome/experiment-worker.env
DEPLOYMENT_GATE=/etc/lehome/experiment-deployment-gate.json
TRAINING_IMAGE_MANIFEST=/etc/lehome/training-image-manifest.json
[[ -f "${ENV_FILE}" && ! -L "${ENV_FILE}" ]] || { echo "missing worker environment" >&2; exit 2; }
[[ "$(stat -c '%a:%u:%g' "${ENV_FILE}")" == "600:0:0" ]] || { echo "worker environment must be root-owned mode 0600" >&2; exit 2; }
READY_FILE=/etc/lehome/experiment-bootstrap.ready
[[ -f "${READY_FILE}" && ! -L "${READY_FILE}" && "$(stat -c '%a:%u:%g' "${READY_FILE}")" == "600:0:0" ]] || { echo "worker bootstrap is incomplete" >&2; exit 2; }
[[ -f "${DEPLOYMENT_GATE}" && ! -L "${DEPLOYMENT_GATE}" && "$(stat -c '%a:%u:%g' "${DEPLOYMENT_GATE}")" == "444:0:0" ]] || { echo "deployment gate must be root-owned mode 0444" >&2; exit 2; }
[[ -f "${TRAINING_IMAGE_MANIFEST}" && ! -L "${TRAINING_IMAGE_MANIFEST}" && "$(stat -c '%a:%u:%g' "${TRAINING_IMAGE_MANIFEST}")" == "444:0:0" ]] || { echo "training image manifest must be root-owned mode 0444" >&2; exit 2; }

# Stop does not dereference remote credentials. It must remain usable during a
# preemption even if the operator's secret revocation runs first.
if [[ "${1:-}" == "--preempt" && $# -eq 1 ]]; then
  [[ "${LEHOME_OUTPUT_ROOT:-}" == /var/lib/lehome/* ]] || { echo "unsafe worker output root" >&2; exit 2; }
  PID_FILE="${LEHOME_WORKER_PID_FILE:-/run/lehome/experiment-worker.pid}"
  REQUEST_FILE="${LEHOME_OUTPUT_ROOT}/preemption.request"
  install -d -m 0700 "${LEHOME_OUTPUT_ROOT}"
  : > "${REQUEST_FILE}"
  sync
  if [[ -f "${PID_FILE}" && ! -L "${PID_FILE}" ]]; then
    pid="$(tr -d '[:space:]' < "${PID_FILE}")"
    if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}"
      for _ in $(seq 1 45); do
        kill -0 "${pid}" 2>/dev/null || exit 0
        sleep 1
      done
      echo "experiment worker did not stop within preemption budget" >&2
      exit 1
    fi
  fi
  exit 0
fi

for key in LEHOME_CONTROLLER_URL LEHOME_CONTROLLER_CA_FILE LEHOME_WORKER_ID LEHOME_MANIFEST_SET_SHA256 LEHOME_CACHE_ROOT LEHOME_OUTPUT_ROOT LEHOME_CONTROLLER_TOKEN_FILE HF_TOKEN_FILE LEHOME_DEPLOYMENT_GATE_SHA256; do
  [[ -n "${!key:-}" ]] || { echo "missing ${key}" >&2; exit 2; }
done
case "${LEHOME_CONTROLLER_URL}" in https://*) ;; *) echo "controller endpoint must use https:// TLS proxy" >&2; exit 2;; esac
[[ "${LEHOME_CONTROLLER_TLS_PROXY_REQUIRED:-}" == "1" ]] || { echo "private TLS proxy acknowledgement is required" >&2; exit 2; }
[[ "${LEHOME_CONTROLLER_CA_FILE}" == /* && -f "${LEHOME_CONTROLLER_CA_FILE}" && ! -L "${LEHOME_CONTROLLER_CA_FILE}" ]] || { echo "private controller CA is unsafe" >&2; exit 2; }
ca_mode="$(stat -c '%a' "${LEHOME_CONTROLLER_CA_FILE}")"
[[ "${ca_mode}" =~ ^[0-7]{3,4}$ ]] && (( (8#${ca_mode} & 8#022) == 0 )) || { echo "private controller CA permissions are unsafe" >&2; exit 2; }
credential_dir=/run/lehome/experiment-credentials
install -d -m 0700 "${credential_dir}"
for credential_name in controller-token hf-token; do
  credential_source="${CREDENTIALS_DIRECTORY:-}/${credential_name}"
  [[ -f "${credential_source}" && ! -L "${credential_source}" ]] || { echo "missing system credential ${credential_name}" >&2; exit 2; }
  install -m 0600 "${credential_source}" "${credential_dir}/${credential_name}"
done
LEHOME_CONTROLLER_TOKEN_FILE="${credential_dir}/controller-token"
HF_TOKEN_FILE="${credential_dir}/hf-token"
for credential in "${LEHOME_CONTROLLER_TOKEN_FILE}" "${HF_TOKEN_FILE}"; do
  [[ -f "${credential}" && ! -L "${credential}" ]] || { echo "unsafe credential file" >&2; exit 2; }
  [[ "$(stat -c '%a' "${credential}")" == "600" ]] || { echo "credential must be 0600" >&2; exit 2; }
done
case "${LEHOME_CACHE_ROOT}:${LEHOME_OUTPUT_ROOT}" in /var/lib/lehome/*:/var/lib/lehome/*) ;; *) echo "unsafe worker roots" >&2; exit 2;; esac
[[ "${LEHOME_CACHE_ROOT}" != "${LEHOME_OUTPUT_ROOT}" ]] || { echo "worker cache and output roots must be distinct" >&2; exit 2; }
PID_FILE="${LEHOME_WORKER_PID_FILE:-/run/lehome/experiment-worker.pid}"
REQUEST_FILE="${LEHOME_OUTPUT_ROOT}/preemption.request"
[[ $# -eq 0 ]] || { echo "usage: lehome-experiment-worker [--preempt]" >&2; exit 2; }

install -d -m 0700 "${LEHOME_CACHE_ROOT}" "${LEHOME_OUTPUT_ROOT}"
install -d -m 0700 "$(dirname "${PID_FILE}")"
child_pid=""
cleanup() { rm -f -- "${PID_FILE}" "${credential_dir}/controller-token" "${credential_dir}/hf-token"; }
on_signal() {
  : > "${REQUEST_FILE}"
  sync
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill -TERM "${child_pid}" || true
    wait "${child_pid}" || true
  fi
  exit 143
}
trap cleanup EXIT
trap on_signal TERM INT

/usr/bin/python3 /opt/lehome/scripts/run_lehome_experiment_worker.py \
  --controller-url "${LEHOME_CONTROLLER_URL}" \
  --controller-ca-file "${LEHOME_CONTROLLER_CA_FILE}" \
  --worker-id "${LEHOME_WORKER_ID}" \
  --manifest-set-sha256 "${LEHOME_MANIFEST_SET_SHA256}" \
  --cache-root "${LEHOME_CACHE_ROOT}" \
  --output-root "${LEHOME_OUTPUT_ROOT}" \
  --controller-token-file "${LEHOME_CONTROLLER_TOKEN_FILE}" \
  --hf-token-file "${HF_TOKEN_FILE}" \
  --deployment-gate "${DEPLOYMENT_GATE}" \
  --deployment-gate-sha256 "${LEHOME_DEPLOYMENT_GATE_SHA256}" \
  --training-image-manifest "${TRAINING_IMAGE_MANIFEST}" &
child_pid="$!"
printf '%s\n' "$$" > "${PID_FILE}"
wait "${child_pid}"
