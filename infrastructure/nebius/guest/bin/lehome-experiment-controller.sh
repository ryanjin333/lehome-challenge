#!/usr/bin/env bash
# Run the private controller and durably close its SQLite WAL on every stop.
# The TLS reverse proxy is deliberately external to this generic image.  This
# process accepts only an exact private bind address and never a wildcard.
set -euo pipefail
umask 077
export PYTHONPATH=/opt/lehome/trainer/src${PYTHONPATH:+:${PYTHONPATH}}

ENV_FILE=/etc/lehome/experiment-controller.env
[[ -f "${ENV_FILE}" && ! -L "${ENV_FILE}" ]] || exit 2
[[ "$(stat -c '%a:%u:%g' "${ENV_FILE}")" == "600:0:0" ]] || exit 2
READY_FILE=/etc/lehome/experiment-bootstrap.ready
[[ -f "${READY_FILE}" && ! -L "${READY_FILE}" && "$(stat -c '%a:%u:%g' "${READY_FILE}")" == "600:0:0" ]] || exit 2
for key in LEHOME_CONTROLLER_DB LEHOME_CONTROLLER_TOKEN_FILE LEHOME_CONTROLLER_MANIFESTS LEHOME_CONTROLLER_AUDIT LEHOME_CONTROLLER_BIND LEHOME_CONTROLLER_TLS_PROXY_REQUIRED LEHOME_DEPLOYMENT_GATE_SHA256; do
  [[ -n "${!key:-}" ]] || exit 2
done
[[ "${LEHOME_CONTROLLER_TLS_PROXY_REQUIRED}" == "1" ]] || exit 2
DEPLOYMENT_GATE=/etc/lehome/experiment-deployment-gate.json
[[ -f "${DEPLOYMENT_GATE}" && ! -L "${DEPLOYMENT_GATE}" && "$(stat -c '%a:%u:%g' "${DEPLOYMENT_GATE}")" == "444:0:0" ]] || exit 2
for path in "${LEHOME_CONTROLLER_DB}" "${LEHOME_CONTROLLER_MANIFESTS}" "${LEHOME_CONTROLLER_AUDIT}"; do
  [[ "${path}" == /var/lib/lehome/controller/* ]] || exit 2
done
credential_source="${CREDENTIALS_DIRECTORY:-}/controller-token"
[[ -f "${credential_source}" && ! -L "${credential_source}" ]] || exit 2
credential_dir=/run/lehome-controller/credentials
install -d -m 0700 "${credential_dir}"
runtime_token="${credential_dir}/controller-token"
install -m 0600 "${credential_source}" "${runtime_token}"
LEHOME_CONTROLLER_TOKEN_FILE="${runtime_token}"
[[ -f "${LEHOME_CONTROLLER_TOKEN_FILE}" && ! -L "${LEHOME_CONTROLLER_TOKEN_FILE}" && "$(stat -c '%a' "${LEHOME_CONTROLLER_TOKEN_FILE}")" == 600 ]] || exit 2

bind_host="${LEHOME_CONTROLLER_BIND%:*}"
bind_port="${LEHOME_CONTROLLER_BIND##*:}"
case "${bind_host}" in
  ""|0.0.0.0|::|127.0.0.1|::1|localhost) exit 2 ;;
esac
[[ "${bind_port}" =~ ^[1-9][0-9]{0,4}$ ]] && (( bind_port <= 65535 )) || exit 2

controller_pid=""
cleanup_done=0
checkpoint_sqlite() {
  /usr/bin/python3 - "${LEHOME_CONTROLLER_DB}" <<'PY'
import os
import sqlite3
import sys
from pathlib import Path

database = Path(sys.argv[1])
connection = sqlite3.connect(database, timeout=5.0, isolation_level=None)
try:
    connection.execute("PRAGMA busy_timeout=5000")
    busy, _, _ = connection.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
    if busy:
        raise SystemExit("controller WAL checkpoint remained busy")
finally:
    connection.close()
for path in (database, Path(f"{database}-wal"), Path(f"{database}-shm")):
    if path.exists():
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
descriptor = os.open(database.parent, os.O_RDONLY)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
}
cleanup() {
  local status="$?"
  if [[ "${cleanup_done}" -eq 1 ]]; then
    return "${status}"
  fi
  cleanup_done=1
  trap - EXIT TERM INT
  if [[ -n "${controller_pid}" ]] && kill -0 "${controller_pid}" 2>/dev/null; then
    kill -TERM "${controller_pid}" || true
    wait "${controller_pid}" || true
  fi
  checkpoint_sqlite || status=1
  rm -f -- "${runtime_token:-}"
  touch /var/lib/lehome/controller/stopped.receipt
  sync
  return "${status}"
}
on_stop() { exit 143; }
trap cleanup EXIT
trap on_stop TERM INT

/usr/bin/python3 /opt/lehome/scripts/run_lehome_experiment_controller.py \
  --database "${LEHOME_CONTROLLER_DB}" \
  --token-file "${LEHOME_CONTROLLER_TOKEN_FILE}" \
  --manifests "${LEHOME_CONTROLLER_MANIFESTS}" \
  --audit-log "${LEHOME_CONTROLLER_AUDIT}" \
  --bind "${LEHOME_CONTROLLER_BIND}" \
  --deployment-gate "${DEPLOYMENT_GATE}" \
  --deployment-gate-sha256 "${LEHOME_DEPLOYMENT_GATE_SHA256}" &
controller_pid="$!"
wait "${controller_pid}"
