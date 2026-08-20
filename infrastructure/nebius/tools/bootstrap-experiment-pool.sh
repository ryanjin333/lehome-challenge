#!/usr/bin/env bash
# Configure the private controller TLS boundary after Terraform/Packer.  This
# script is intentionally run from an operator workstation over SSH: no
# runtime token, certificate private key, or private endpoint enters a golden
# image, Terraform state, cloud-init, or a command-line argument.
set -euo pipefail
umask 077

SSH_BIN="${LEHOME_BOOTSTRAP_SSH_BIN:-ssh}"
SCP_BIN="${LEHOME_BOOTSTRAP_SCP_BIN:-scp}"
OPENSSL_BIN="${LEHOME_BOOTSTRAP_OPENSSL_BIN:-openssl}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATRIX_FREEZER="${LEHOME_MATRIX_FREEZER:-${SCRIPT_DIR}/../../../scripts/freeze_lehome_experiment_matrices.py}"

CONTROLLER_SSH=""
CONTROLLER_IP=""
CONTROLLER_TOKEN_FILE=""
NEBIUS_PRIVATE_KEY_FILE=""
NEBIUS_SERVICE_ACCOUNT_ID=""
NEBIUS_PUBLIC_KEY_ID=""
NEBIUS_PROJECT_ID=""
HF_TOKEN_FILE=""
MANIFEST_SET_SHA256=""
TLS_PORT=8443
CONTROLLER_PORT=15555
MANIFEST_ROOT=/var/lib/lehome/controller/manifests
WORKER_ONE=""
WORKER_TWO=""
ROLLOUT_SSH=""
PROMOTION_MATRIX=""
PROMOTION_MATRIX_SHA256=""
FINAL_MATRIX=""
FINAL_MATRIX_SHA256=""
PROMOTION_BASELINE_EVIDENCE=""
PROMOTION_BASELINE_EVIDENCE_SHA256=""
FINAL_REPORT_REPOSITORY=""
DEPLOYMENT_GATE=""
DEPLOYMENT_GATE_SHA256=""

usage() {
  cat <<'USAGE'
Usage:
  bootstrap-experiment-pool.sh \
    --controller-ssh USER@HOST --controller-ip PRIVATE_IPV4 \
    --manifest-set-sha256 SHA256 \
    --controller-token-file PATH --hf-token-file PATH \
    --nebius-private-key-file PATH --nebius-service-account-id ID \
    --nebius-public-key-id ID --nebius-project-id ID \
    --rollout USER@HOST \
    --promotion-matrix PATH --promotion-matrix-sha256 SHA256 \
    --final-matrix PATH --final-matrix-sha256 SHA256 \
    --promotion-baseline-evidence PATH \
    --promotion-baseline-evidence-sha256 SHA256 \
    --deployment-gate PATH --deployment-gate-sha256 SHA256 \
    --final-report-repository OWNER/REPOSITORY \
    --worker 1=USER@HOST --worker 2=USER@HOST

The immutable controller manifest directory must already be readback-verified
at /var/lib/lehome/controller/manifests on the protected controller state disk.
This script configures only the private runtime boundary. It leaves controller,
TLS proxy, both training workers, and the rollout evaluator disabled after the
verification transaction.
USAGE
}

fail() { echo "bootstrap: $*" >&2; exit 2; }

while (($#)); do
  case "$1" in
    --controller-ssh) CONTROLLER_SSH="${2:-}"; shift 2 ;;
    --controller-ip) CONTROLLER_IP="${2:-}"; shift 2 ;;
    --manifest-set-sha256) MANIFEST_SET_SHA256="${2:-}"; shift 2 ;;
    --controller-token-file) CONTROLLER_TOKEN_FILE="${2:-}"; shift 2 ;;
    --nebius-private-key-file) NEBIUS_PRIVATE_KEY_FILE="${2:-}"; shift 2 ;;
    --nebius-service-account-id) NEBIUS_SERVICE_ACCOUNT_ID="${2:-}"; shift 2 ;;
    --nebius-public-key-id) NEBIUS_PUBLIC_KEY_ID="${2:-}"; shift 2 ;;
    --nebius-project-id) NEBIUS_PROJECT_ID="${2:-}"; shift 2 ;;
    --hf-token-file) HF_TOKEN_FILE="${2:-}"; shift 2 ;;
    --rollout) ROLLOUT_SSH="${2:-}"; shift 2 ;;
    --promotion-matrix) PROMOTION_MATRIX="${2:-}"; shift 2 ;;
    --promotion-matrix-sha256) PROMOTION_MATRIX_SHA256="${2:-}"; shift 2 ;;
    --final-matrix) FINAL_MATRIX="${2:-}"; shift 2 ;;
    --final-matrix-sha256) FINAL_MATRIX_SHA256="${2:-}"; shift 2 ;;
    --promotion-baseline-evidence) PROMOTION_BASELINE_EVIDENCE="${2:-}"; shift 2 ;;
    --promotion-baseline-evidence-sha256) PROMOTION_BASELINE_EVIDENCE_SHA256="${2:-}"; shift 2 ;;
    --deployment-gate) DEPLOYMENT_GATE="${2:-}"; shift 2 ;;
    --deployment-gate-sha256) DEPLOYMENT_GATE_SHA256="${2:-}"; shift 2 ;;
    --final-report-repository) FINAL_REPORT_REPOSITORY="${2:-}"; shift 2 ;;
    --worker)
      item="${2:-}"; shift 2
      [[ "${item}" =~ ^([12])=(.+)$ ]] || fail "worker must be 1=USER@HOST or 2=USER@HOST"
      case "${BASH_REMATCH[1]}" in
        1) [[ -z "${WORKER_ONE}" ]] || fail "worker slot is repeated"; WORKER_ONE="${BASH_REMATCH[2]}" ;;
        2) [[ -z "${WORKER_TWO}" ]] || fail "worker slot is repeated"; WORKER_TWO="${BASH_REMATCH[2]}" ;;
      esac
      ;;
    --tls-port) TLS_PORT="${2:-}"; shift 2 ;;
    --controller-port) CONTROLLER_PORT="${2:-}"; shift 2 ;;
    --manifest-root) MANIFEST_ROOT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument $1" ;;
  esac
done

[[ -n "${CONTROLLER_SSH}" && -n "${CONTROLLER_IP}" && -n "${MANIFEST_SET_SHA256}" ]] || { usage >&2; exit 2; }
[[ -n "${NEBIUS_SERVICE_ACCOUNT_ID}" && -n "${NEBIUS_PUBLIC_KEY_ID}" && -n "${NEBIUS_PROJECT_ID}" ]] || fail "Nebius service-account identity is required for exact-ID capacity actions"
for identifier in "${NEBIUS_SERVICE_ACCOUNT_ID}" "${NEBIUS_PUBLIC_KEY_ID}" "${NEBIUS_PROJECT_ID}"; do
  [[ "${identifier}" =~ ^[A-Za-z0-9._:-]+$ ]] || fail "Nebius service-account identity is invalid"
done
[[ "${MANIFEST_SET_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "manifest-set SHA-256 is invalid"
[[ "${PROMOTION_MATRIX_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "promotion matrix SHA-256 is invalid"
[[ "${FINAL_MATRIX_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "final matrix SHA-256 is invalid"
[[ "${PROMOTION_BASELINE_EVIDENCE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "promotion baseline evidence SHA-256 is invalid"
[[ "${DEPLOYMENT_GATE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "deployment gate SHA-256 is invalid"
[[ "${FINAL_REPORT_REPOSITORY}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || fail "final report repository is invalid"
[[ "${MANIFEST_ROOT}" == /var/lib/lehome/controller/* && "${MANIFEST_ROOT}" != *".."* ]] || fail "manifest root is unsafe"
for port in "${TLS_PORT}" "${CONTROLLER_PORT}"; do
  [[ "${port}" =~ ^[1-9][0-9]{0,4}$ ]] && ((port <= 65535)) || fail "port is invalid"
done
[[ "${TLS_PORT}" != "${CONTROLLER_PORT}" ]] || fail "TLS and controller ports must differ"
[[ -n "${WORKER_ONE}" && -n "${WORKER_TWO}" ]] || fail "exactly worker slots 1 and 2 are required"
[[ -n "${ROLLOUT_SSH}" ]] || fail "exactly one rollout evaluator host is required"

python3 - "${CONTROLLER_IP}" <<'PY'
import ipaddress
import sys

address = ipaddress.ip_address(sys.argv[1])
if address.version != 4 or not address.is_private:
    raise SystemExit("controller address must be an RFC1918 IPv4 address")
PY

for secret in "${CONTROLLER_TOKEN_FILE}" "${HF_TOKEN_FILE}" "${NEBIUS_PRIVATE_KEY_FILE}"; do
  [[ -f "${secret}" && ! -L "${secret}" ]] || fail "secret source is not a regular file"
  [[ "$(stat -f '%Lp' "${secret}")" == 600 || "$(stat -c '%a' "${secret}" 2>/dev/null || true)" == 600 ]] || fail "secret source must be mode 0600"
done
for artifact in "${PROMOTION_MATRIX}" "${FINAL_MATRIX}" "${PROMOTION_BASELINE_EVIDENCE}"; do
  [[ "${artifact}" == /* && -f "${artifact}" && ! -L "${artifact}" ]] || fail "evaluation artifact source is unsafe"
done
[[ "${DEPLOYMENT_GATE}" == /* && -f "${DEPLOYMENT_GATE}" && ! -L "${DEPLOYMENT_GATE}" ]] || fail "deployment gate source is unsafe"
deployment_gate_mode="$(stat -f '%Lp' "${DEPLOYMENT_GATE}" 2>/dev/null || stat -c '%a' "${DEPLOYMENT_GATE}" 2>/dev/null || true)"
[[ "${deployment_gate_mode}" == 444 ]] || fail "deployment gate source must be immutable mode 0444"
PYTHONPATH="${SCRIPT_DIR}/../../../trainer/src" python3 - "${DEPLOYMENT_GATE}" "${DEPLOYMENT_GATE_SHA256}" <<'PY'
import sys
from lehome_train.groot.experiment_deployment_gate import load_deployment_gate

load_deployment_gate(sys.argv[1], sys.argv[2])
PY
observed_promotion_sha256="$(python3 - "${PROMOTION_MATRIX}" <<'PY'
import hashlib
import pathlib
import sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
[[ "${observed_promotion_sha256}" == "${PROMOTION_MATRIX_SHA256}" ]] || fail "promotion matrix SHA-256 mismatch"
observed_final_sha256="$(python3 - "${FINAL_MATRIX}" <<'PY'
import hashlib
import pathlib
import sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
[[ "${observed_final_sha256}" == "${FINAL_MATRIX_SHA256}" ]] || fail "final matrix SHA-256 mismatch"
observed_baseline_sha256="$(python3 - "${PROMOTION_BASELINE_EVIDENCE}" <<'PY'
import hashlib
import pathlib
import sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
[[ "${observed_baseline_sha256}" == "${PROMOTION_BASELINE_EVIDENCE_SHA256}" ]] || fail "promotion baseline evidence SHA-256 mismatch"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/lehome-bootstrap.XXXXXX")"
cleanup_local() { rm -rf -- "${WORK_DIR}"; }
trap cleanup_local EXIT

[[ -f "${MATRIX_FREEZER}" && ! -L "${MATRIX_FREEZER}" ]] || fail "matrix freezer is unavailable or unsafe"
FROZEN_MATRIX_ROOT="${WORK_DIR}/frozen-evaluator-matrices"
python3 "${MATRIX_FREEZER}" \
  --promotion-source "${PROMOTION_MATRIX}" \
  --promotion-source-sha256 "${PROMOTION_MATRIX_SHA256}" \
  --final-source "${FINAL_MATRIX}" \
  --final-source-sha256 "${FINAL_MATRIX_SHA256}" \
  --output-root "${FROZEN_MATRIX_ROOT}" >/dev/null
PROMOTION_MATRIX="${FROZEN_MATRIX_ROOT}/promotion-matrix.json"
FINAL_MATRIX="${FROZEN_MATRIX_ROOT}/final-matrix.json"
PROMOTION_MATRIX_SHA256="$(tr -d '\r\n' < "${PROMOTION_MATRIX}.sha256")"
FINAL_MATRIX_SHA256="$(tr -d '\r\n' < "${FINAL_MATRIX}.sha256")"
[[ "${PROMOTION_MATRIX_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "frozen promotion matrix SHA-256 is invalid"
[[ "${FINAL_MATRIX_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "frozen final matrix SHA-256 is invalid"

# The freezer intentionally changes an organizer envelope into the exact list
# consumed by the evaluator.  Refuse bootstrap unless the already-published
# immutable controller jobs bind that frozen list hash; otherwise the first
# paid lease would train/evaluate one matrix and fail report binding afterward.
"${SSH_BIN}" "${CONTROLLER_SSH}" "sudo python3 - '${MANIFEST_ROOT}' '${MANIFEST_SET_SHA256}' '${PROMOTION_MATRIX_SHA256}'" <<'REMOTE'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest_set_sha256 = sys.argv[2]
promotion_matrix_sha256 = sys.argv[3]
campaign_path = root / "campaign.json"
if root.is_symlink() or not root.is_dir() or campaign_path.is_symlink() or not campaign_path.is_file():
    raise SystemExit("controller manifests are absent or unsafe")
try:
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"controller campaign manifest is unreadable: {error}")
job_paths = sorted(path for path in root.glob("*.json") if path.name != "campaign.json")
if not job_paths or any(path.is_symlink() or not path.is_file() for path in job_paths):
    raise SystemExit("controller job manifests are absent or unsafe")
documents = []
for path in job_paths:
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"controller job manifest is unreadable: {error}")
    evaluation = job.get("evaluation") if isinstance(job, dict) else None
    experiment_id = job.get("experiment_id") if isinstance(job, dict) else None
    if (
        not isinstance(evaluation, dict)
        or evaluation.get("matrix_sha256") != promotion_matrix_sha256
        or not isinstance(experiment_id, str)
        or len(experiment_id) != 64
    ):
        raise SystemExit("controller manifest evaluation matrix does not match the frozen promotion matrix")
    documents.append((experiment_id, job))
documents.sort(key=lambda item: item[0])
encoded = json.dumps(
    {"schema_version": 1, "jobs": documents},
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
computed = hashlib.sha256(encoded).hexdigest()
campaign_jobs = campaign.get("jobs") if isinstance(campaign, dict) else None
if (
    computed != manifest_set_sha256
    or campaign.get("manifest_set_sha256") != manifest_set_sha256
    or not isinstance(campaign_jobs, list)
    or set(campaign_jobs) != {item[0] for item in documents}
):
    raise SystemExit("controller manifest set does not match the requested immutable campaign")
REMOTE

CA_KEY="${WORK_DIR}/controller-ca.key"
CA_CERT="${WORK_DIR}/controller-ca.crt"
SERVER_KEY="${WORK_DIR}/controller-server.key"
SERVER_CSR="${WORK_DIR}/controller-server.csr"
SERVER_CERT="${WORK_DIR}/controller-server.crt"
OPENSSL_CONFIG="${WORK_DIR}/openssl.cnf"
cat > "${OPENSSL_CONFIG}" <<EOF
[req]
distinguished_name = dn
prompt = no
req_extensions = extensions
[dn]
CN = lehome-experiment-controller
[extensions]
subjectAltName = IP:${CONTROLLER_IP}
EOF
"${OPENSSL_BIN}" genrsa -out "${CA_KEY}" 4096 >/dev/null 2>&1
"${OPENSSL_BIN}" req -x509 -new -key "${CA_KEY}" -sha256 -days 30 -subj /CN=lehome-experiment-private-ca -out "${CA_CERT}" >/dev/null 2>&1
"${OPENSSL_BIN}" genrsa -out "${SERVER_KEY}" 4096 >/dev/null 2>&1
"${OPENSSL_BIN}" req -new -key "${SERVER_KEY}" -config "${OPENSSL_CONFIG}" -out "${SERVER_CSR}" >/dev/null 2>&1
"${OPENSSL_BIN}" x509 -req -in "${SERVER_CSR}" -CA "${CA_CERT}" -CAkey "${CA_KEY}" -CAcreateserial -days 30 -sha256 -extfile "${OPENSSL_CONFIG}" -extensions extensions -out "${SERVER_CERT}" >/dev/null 2>&1
chmod 0600 "${CA_KEY}" "${SERVER_KEY}"

remote_controller() { "${SSH_BIN}" "${CONTROLLER_SSH}" "$@"; }
worker_target() {
  case "$1" in
    1) printf '%s\n' "${WORKER_ONE}" ;;
    2) printf '%s\n' "${WORKER_TWO}" ;;
    *) fail "unknown worker slot" ;;
  esac
}
remote_worker() { local slot="$1"; shift; "${SSH_BIN}" "$(worker_target "${slot}")" "$@"; }
remote_rollout() { "${SSH_BIN}" "${ROLLOUT_SSH}" "$@"; }
copy_controller() { "${SCP_BIN}" -p "$1" "${CONTROLLER_SSH}:$2"; }
copy_worker() { local slot="$1"; "${SCP_BIN}" -p "$2" "$(worker_target "${slot}"):$3"; }
copy_rollout() { "${SCP_BIN}" -p "$1" "${ROLLOUT_SSH}:$2"; }

rollback_controller() {
  remote_controller 'sudo systemctl disable --now lehome-experiment-controller.service lehome-experiment-controller-proxy.service lehome-experiment-capacity.service >/dev/null 2>&1 || true; sudo rm -f /etc/lehome/experiment-bootstrap.ready' >/dev/null 2>&1 || true
  remote_rollout 'sudo systemctl disable --now lehome-experiment-evaluator.service >/dev/null 2>&1 || true; sudo rm -f /etc/lehome/experiment-bootstrap.ready' >/dev/null 2>&1 || true
  for slot in 1 2; do
    remote_worker "${slot}" 'sudo systemctl disable --now lehome-experiment-worker.service >/dev/null 2>&1 || true; sudo rm -f /etc/lehome/experiment-bootstrap.ready' >/dev/null 2>&1 || true
  done
}
trap rollback_controller ERR

copy_controller "${CA_CERT}" /tmp/lehome-controller-ca.crt
copy_controller "${SERVER_CERT}" /tmp/lehome-controller-server.crt
copy_controller "${SERVER_KEY}" /tmp/lehome-controller-server.key
copy_controller "${CONTROLLER_TOKEN_FILE}" /tmp/lehome-controller-token
copy_controller "${NEBIUS_PRIVATE_KEY_FILE}" /tmp/lehome-nebius-private-key
copy_controller "${DEPLOYMENT_GATE}" /tmp/lehome-experiment-deployment-gate.json
remote_controller "sudo bash -s -- '${CONTROLLER_IP}' '${TLS_PORT}' '${CONTROLLER_PORT}' '${MANIFEST_ROOT}' '${DEPLOYMENT_GATE_SHA256}' '${NEBIUS_SERVICE_ACCOUNT_ID}' '${NEBIUS_PUBLIC_KEY_ID}' '${NEBIUS_PROJECT_ID}'" <<'REMOTE'
set -euo pipefail
CONTROLLER_IP="$1"; TLS_PORT="$2"; CONTROLLER_PORT="$3"; MANIFEST_ROOT="$4"; DEPLOYMENT_GATE_SHA256="$5"; NEBIUS_SERVICE_ACCOUNT_ID="$6"; NEBIUS_PUBLIC_KEY_ID="$7"; NEBIUS_PROJECT_ID="$8"
install -d -m 0700 -o root -g root /etc/lehome/private /etc/lehome/tls
install -m 0600 -o root -g root /tmp/lehome-controller-token /etc/lehome/private/controller-token
install -m 0600 -o root -g root /tmp/lehome-nebius-private-key /etc/lehome/private/nebius-private-key
install -m 0600 -o root -g root /tmp/lehome-controller-server.key /etc/lehome/tls/controller.key
install -m 0600 -o root -g root /tmp/lehome-controller-server.crt /etc/lehome/tls/controller.crt
install -m 0644 -o root -g root /tmp/lehome-controller-ca.crt /etc/lehome/tls/controller-ca.crt
install -m 0444 -o root -g root /tmp/lehome-experiment-deployment-gate.json /etc/lehome/experiment-deployment-gate.json
printf '%s  %s\n' "${DEPLOYMENT_GATE_SHA256}" /etc/lehome/experiment-deployment-gate.json | sha256sum --check --strict
rm -f /tmp/lehome-controller-token /tmp/lehome-nebius-private-key /tmp/lehome-controller-server.key /tmp/lehome-controller-server.crt /tmp/lehome-controller-ca.crt /tmp/lehome-experiment-deployment-gate.json
[[ -d "${MANIFEST_ROOT}" && -f "${MANIFEST_ROOT}/campaign.json" ]] || { echo "immutable manifests are absent" >&2; exit 2; }
install -d -m 0750 -o lehome-controller -g lehome-controller /var/lib/lehome/controller/audit
cat > /etc/lehome/experiment-controller.env <<EOF
LEHOME_CONTROLLER_DB=/var/lib/lehome/controller/controller.sqlite3
LEHOME_CONTROLLER_TOKEN_FILE=/etc/lehome/private/controller-token
LEHOME_CONTROLLER_MANIFESTS=${MANIFEST_ROOT}
LEHOME_CONTROLLER_AUDIT=/var/lib/lehome/controller/audit/controller.log
LEHOME_CONTROLLER_BIND=${CONTROLLER_IP}:${CONTROLLER_PORT}
LEHOME_CONTROLLER_TLS_PROXY_REQUIRED=1
LEHOME_DEPLOYMENT_GATE_SHA256=${DEPLOYMENT_GATE_SHA256}
EOF
chown root:root /etc/lehome/experiment-controller.env
chmod 0600 /etc/lehome/experiment-controller.env
cat > /etc/lehome/experiment-capacity.env <<EOF
LEHOME_CAPACITY_CONFIG=/etc/lehome/capacity.json
LEHOME_CAPACITY_CONTROLLER_URL=https://${CONTROLLER_IP}:${TLS_PORT}
LEHOME_CAPACITY_CONTROLLER_CA_FILE=/etc/lehome/tls/controller-ca.crt
LEHOME_CAPACITY_RECEIPT_LOG=/var/lib/lehome/controller/audit/capacity.jsonl
LEHOME_CAPACITY_NEBIUS_CONFIG_FILE=/run/lehome-capacity/nebius-config.yaml
LEHOME_CAPACITY_NEBIUS_PROFILE=lehome-capacity
LEHOME_CAPACITY_NEBIUS_SERVICE_ACCOUNT_ID=${NEBIUS_SERVICE_ACCOUNT_ID}
LEHOME_CAPACITY_NEBIUS_PUBLIC_KEY_ID=${NEBIUS_PUBLIC_KEY_ID}
LEHOME_CAPACITY_NEBIUS_PROJECT_ID=${NEBIUS_PROJECT_ID}
EOF
chown root:root /etc/lehome/experiment-capacity.env
chmod 0600 /etc/lehome/experiment-capacity.env
python3 - "${DEPLOYMENT_GATE_SHA256}" <<'PY'
import json
from pathlib import Path
import sys

digest = sys.argv[1]
gate_path = Path('/etc/lehome/experiment-deployment-gate.json')
gate = json.loads(gate_path.read_text(encoding='utf-8'))
workers = gate['training_workers']
document = {
    'schema_version': 1,
    'training_workers': [
        {'instance_id': workers[0]['instance_id'], 'worker_id': workers[0]['worker_id']},
        {'instance_id': workers[1]['instance_id'], 'worker_id': workers[1]['worker_id']},
    ],
    'rollout_worker': {
        'instance_id': gate['rollout_worker']['instance_id'],
        'worker_id': gate['rollout_worker']['worker_id'],
    },
    'idle_seconds': 600,
    'operation_cap': 3,
    'deployment_gate_path': str(gate_path),
    'deployment_gate_sha256': digest,
}
target = Path('/etc/lehome/capacity.json')
target.write_text(json.dumps(document, sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8')
target.chmod(0o600)
PY
chown root:root /etc/lehome/capacity.json
cat > /etc/lehome/nginx-experiment-controller.conf <<EOF
pid /run/lehome-controller-proxy/nginx.pid;
error_log /var/log/nginx/lehome-experiment-controller.error.log warn;
events { worker_connections 64; }
http {
  access_log off;
  client_max_body_size 64k;
  server {
    listen ${CONTROLLER_IP}:${TLS_PORT} ssl;
    server_name _;
    ssl_certificate /etc/lehome/tls/controller.crt;
    ssl_certificate_key /etc/lehome/tls/controller.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    location / {
      proxy_pass http://${CONTROLLER_IP}:${CONTROLLER_PORT};
      proxy_http_version 1.1;
      proxy_set_header Host \$host;
      proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
      proxy_request_buffering off;
      proxy_buffering off;
    }
  }
}
EOF
chown root:root /etc/lehome/nginx-experiment-controller.conf
chmod 0600 /etc/lehome/nginx-experiment-controller.conf
if ! command -v nginx >/dev/null 2>&1; then
  cat > /usr/sbin/policy-rc.d <<'POLICY'
#!/bin/sh
exit 101
POLICY
  chmod 0755 /usr/sbin/policy-rc.d
  trap 'rm -f /usr/sbin/policy-rc.d' EXIT
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nginx
  rm -f /usr/sbin/policy-rc.d
  trap - EXIT
fi
systemctl daemon-reload
systemctl disable --now nginx.service lehome-experiment-controller.service lehome-experiment-controller-proxy.service lehome-experiment-capacity.service >/dev/null 2>&1 || true
rm -f /etc/lehome/experiment-bootstrap.ready
REMOTE

for slot in 1 2; do
  copy_worker "${slot}" "${CA_CERT}" /tmp/lehome-controller-ca.crt
  copy_worker "${slot}" "${CONTROLLER_TOKEN_FILE}" /tmp/lehome-controller-token
  copy_worker "${slot}" "${HF_TOKEN_FILE}" /tmp/lehome-hf-token
  copy_worker "${slot}" "${DEPLOYMENT_GATE}" /tmp/lehome-experiment-deployment-gate.json
  remote_worker "${slot}" "sudo bash -s -- '${slot}' '${CONTROLLER_IP}' '${TLS_PORT}' '${MANIFEST_SET_SHA256}' '${DEPLOYMENT_GATE_SHA256}'" <<'REMOTE'
set -euo pipefail
SLOT="$1"; CONTROLLER_IP="$2"; TLS_PORT="$3"; MANIFEST_SET_SHA256="$4"; DEPLOYMENT_GATE_SHA256="$5"
install -d -m 0700 -o root -g root /etc/lehome/private /etc/lehome/tls
install -m 0600 -o root -g root /tmp/lehome-controller-token /etc/lehome/private/controller-token
install -m 0600 -o root -g root /tmp/lehome-hf-token /etc/lehome/private/hf-token
install -m 0644 -o root -g root /tmp/lehome-controller-ca.crt /etc/lehome/tls/controller-ca.crt
install -m 0444 -o root -g root /tmp/lehome-experiment-deployment-gate.json /etc/lehome/experiment-deployment-gate.json
printf '%s  %s\n' "${DEPLOYMENT_GATE_SHA256}" /etc/lehome/experiment-deployment-gate.json | sha256sum --check --strict
rm -f /tmp/lehome-controller-token /tmp/lehome-hf-token /tmp/lehome-controller-ca.crt /tmp/lehome-experiment-deployment-gate.json
cat > /etc/lehome/experiment-worker.env <<EOF
LEHOME_CONTROLLER_URL=https://${CONTROLLER_IP}:${TLS_PORT}
LEHOME_CONTROLLER_CA_FILE=/etc/lehome/tls/controller-ca.crt
LEHOME_WORKER_ID=lehome-experiment-training-${SLOT}
LEHOME_MANIFEST_SET_SHA256=${MANIFEST_SET_SHA256}
LEHOME_CACHE_ROOT=/var/lib/lehome/cache/experiment-worker-${SLOT}
LEHOME_OUTPUT_ROOT=/var/lib/lehome/output/experiment-worker-${SLOT}
LEHOME_CONTROLLER_TOKEN_FILE=/etc/lehome/private/controller-token
HF_TOKEN_FILE=/etc/lehome/private/hf-token
LEHOME_CONTROLLER_TLS_PROXY_REQUIRED=1
LEHOME_DEPLOYMENT_GATE_SHA256=${DEPLOYMENT_GATE_SHA256}
EOF
chown root:root /etc/lehome/experiment-worker.env
chmod 0600 /etc/lehome/experiment-worker.env
systemctl daemon-reload
systemctl disable --now lehome-experiment-worker.service >/dev/null 2>&1 || true
rm -f /etc/lehome/experiment-bootstrap.ready
REMOTE
done

copy_rollout "${CA_CERT}" /tmp/lehome-controller-ca.crt
copy_rollout "${CONTROLLER_TOKEN_FILE}" /tmp/lehome-controller-token
copy_rollout "${HF_TOKEN_FILE}" /tmp/lehome-hf-token
copy_rollout "${PROMOTION_MATRIX}" /tmp/lehome-promotion-matrix.json
copy_rollout "${FINAL_MATRIX}" /tmp/lehome-final-matrix.json
copy_rollout "${PROMOTION_BASELINE_EVIDENCE}" /tmp/lehome-promotion-baseline-evidence.json
remote_rollout "sudo bash -s -- '${CONTROLLER_IP}' '${TLS_PORT}' '${MANIFEST_SET_SHA256}' '${PROMOTION_MATRIX_SHA256}' '${FINAL_MATRIX_SHA256}' '${PROMOTION_BASELINE_EVIDENCE_SHA256}' '${FINAL_REPORT_REPOSITORY}' '${DEPLOYMENT_GATE_SHA256}'" <<'REMOTE'
set -euo pipefail
CONTROLLER_IP="$1"; TLS_PORT="$2"; MANIFEST_SET_SHA256="$3"; PROMOTION_MATRIX_SHA256="$4"; FINAL_MATRIX_SHA256="$5"; PROMOTION_BASELINE_EVIDENCE_SHA256="$6"; FINAL_REPORT_REPOSITORY="$7"; DEPLOYMENT_GATE_SHA256="$8"
install -d -m 0700 -o root -g root /etc/lehome/private /etc/lehome/tls
install -d -m 0750 -o root -g root /mnt/lehome/experiment-pool/evaluation
install -d -m 0750 -o root -g root /mnt/lehome/experiment-pool/evaluation/seen-regression-handoffs
install -m 0600 -o root -g root /tmp/lehome-controller-token /etc/lehome/private/controller-token
install -m 0600 -o root -g root /tmp/lehome-hf-token /etc/lehome/private/hf-token
install -m 0644 -o root -g root /tmp/lehome-controller-ca.crt /etc/lehome/tls/controller-ca.crt
install -m 0444 -o root -g root /tmp/lehome-promotion-matrix.json /mnt/lehome/experiment-pool/evaluation/promotion-matrix.json
install -m 0444 -o root -g root /tmp/lehome-final-matrix.json /mnt/lehome/experiment-pool/evaluation/final-matrix.json
install -m 0444 -o root -g root /tmp/lehome-promotion-baseline-evidence.json /mnt/lehome/experiment-pool/evaluation/promotion-baseline-evidence.json
rm -f /tmp/lehome-controller-token /tmp/lehome-hf-token /tmp/lehome-controller-ca.crt /tmp/lehome-promotion-matrix.json /tmp/lehome-final-matrix.json /tmp/lehome-promotion-baseline-evidence.json
printf '%s  %s\n' "${PROMOTION_MATRIX_SHA256}" /mnt/lehome/experiment-pool/evaluation/promotion-matrix.json | sha256sum --check --strict
printf '%s  %s\n' "${FINAL_MATRIX_SHA256}" /mnt/lehome/experiment-pool/evaluation/final-matrix.json | sha256sum --check --strict
printf '%s  %s\n' "${PROMOTION_BASELINE_EVIDENCE_SHA256}" /mnt/lehome/experiment-pool/evaluation/promotion-baseline-evidence.json | sha256sum --check --strict
cat > /etc/lehome/experiment-evaluator.env <<EOF
LEHOME_CONTROLLER_URL=https://${CONTROLLER_IP}:${TLS_PORT}
LEHOME_CONTROLLER_CA_FILE=/etc/lehome/tls/controller-ca.crt
LEHOME_MANIFEST_SET_SHA256=${MANIFEST_SET_SHA256}
LEHOME_PROMOTION_MATRIX=/mnt/lehome/experiment-pool/evaluation/promotion-matrix.json
LEHOME_PROMOTION_MATRIX_SHA256=${PROMOTION_MATRIX_SHA256}
LEHOME_FINAL_MATRIX=/mnt/lehome/experiment-pool/evaluation/final-matrix.json
LEHOME_FINAL_MATRIX_SHA256=${FINAL_MATRIX_SHA256}
LEHOME_PROMOTION_BASELINE_EVIDENCE=/mnt/lehome/experiment-pool/evaluation/promotion-baseline-evidence.json
LEHOME_PROMOTION_BASELINE_EVIDENCE_SHA256=${PROMOTION_BASELINE_EVIDENCE_SHA256}
LEHOME_CONTROLLER_TOKEN_FILE=/run/lehome/controller-token
LEHOME_HF_TOKEN_FILE=/run/lehome/hf-token
LEHOME_EVALUATION_ROOT=/mnt/lehome/experiment-pool/evaluation/runs
LEHOME_EVALUATION_MODE=promotion
LEHOME_FINAL_REPORT_REPOSITORY=${FINAL_REPORT_REPOSITORY}
LEHOME_FINAL_REPORT_PREFIX=final-unseen80/
LEHOME_FINAL_SEEN_REGRESSION_HANDOFF_ROOT=/mnt/lehome/experiment-pool/evaluation/seen-regression-handoffs
LEHOME_CONTROLLER_TLS_PROXY_REQUIRED=1
LEHOME_DEPLOYMENT_GATE_SHA256=${DEPLOYMENT_GATE_SHA256}
EOF
chown root:root /etc/lehome/experiment-evaluator.env
chmod 0600 /etc/lehome/experiment-evaluator.env
systemctl daemon-reload
systemctl disable --now lehome-experiment-evaluator.service >/dev/null 2>&1 || true
rm -f /etc/lehome/experiment-bootstrap.ready
REMOTE

verify_controller() {
  remote_controller "printf '%s\n' '${DEPLOYMENT_GATE_SHA256}' | sudo tee /etc/lehome/experiment-bootstrap.ready >/dev/null; sudo chmod 0600 /etc/lehome/experiment-bootstrap.ready; sudo chown root:root /etc/lehome/experiment-bootstrap.ready; sudo systemctl start lehome-experiment-controller.service; sudo bash -s -- '${CONTROLLER_IP}' '${TLS_PORT}'" <<'REMOTE'
set -euo pipefail
CONTROLLER_IP="$1"; TLS_PORT="$2"
curl --silent --show-error --fail --connect-timeout 10 --cacert /etc/lehome/tls/controller-ca.crt "https://${CONTROLLER_IP}:${TLS_PORT}/health" | grep -qx '{"status":"ok"}'
test "$(stat -c '%a:%u:%g' /etc/lehome/private/nebius-private-key)" = 600:0:0
test "$(stat -c '%a:%u:%g' /etc/lehome/experiment-capacity.env)" = 600:0:0
grep -qx 'LEHOME_CAPACITY_NEBIUS_CONFIG_FILE=/run/lehome-capacity/nebius-config.yaml' /etc/lehome/experiment-capacity.env
systemctl is-enabled lehome-experiment-capacity.service | grep -qx disabled
config=/run/lehome-controller/bootstrap-capacity.curl
install -d -m 0700 -o root -g root /run/lehome-controller
{ printf '%s\n' 'silent' 'show-error' 'fail' 'connect-timeout = 10' "cacert = /etc/lehome/tls/controller-ca.crt" "url = https://${CONTROLLER_IP}:${TLS_PORT}/capacity"; printf 'header = "Authorization: Bearer '; tr -d '\r\n' < /etc/lehome/private/controller-token; printf '"\n'; } > "${config}"
chmod 0600 "${config}"
curl --config "${config}" >/dev/null
rm -f "${config}"
REMOTE
}

verify_controller
for slot in 1 2; do
  remote_worker "${slot}" "sudo bash -s -- '${slot}' '${CONTROLLER_IP}' '${TLS_PORT}' '${DEPLOYMENT_GATE_SHA256}'" <<'REMOTE'
set -euo pipefail
SLOT="$1"; CONTROLLER_IP="$2"; TLS_PORT="$3"; DEPLOYMENT_GATE_SHA256="$4"
grep -qx "LEHOME_WORKER_ID=lehome-experiment-training-${SLOT}" /etc/lehome/experiment-worker.env
grep -qx "LEHOME_CONTROLLER_URL=https://${CONTROLLER_IP}:${TLS_PORT}" /etc/lehome/experiment-worker.env
grep -qx "LEHOME_DEPLOYMENT_GATE_SHA256=${DEPLOYMENT_GATE_SHA256}" /etc/lehome/experiment-worker.env
test "$(stat -c '%a:%u:%g' /etc/lehome/experiment-deployment-gate.json)" = 444:0:0
test "$(stat -c '%a:%u:%g' /etc/lehome/training-image-manifest.json)" = 444:0:0
test "$(stat -c '%a:%u:%g' /etc/lehome/experiment-worker.env)" = 600:0:0
test "$(stat -c '%a:%u:%g' /etc/lehome/private/controller-token)" = 600:0:0
test "$(stat -c '%a:%u:%g' /etc/lehome/private/hf-token)" = 600:0:0
curl --silent --show-error --fail --connect-timeout 10 --cacert /etc/lehome/tls/controller-ca.crt "https://${CONTROLLER_IP}:${TLS_PORT}/health" | grep -qx '{"status":"ok"}'
systemctl is-enabled lehome-experiment-worker.service | grep -qx disabled
REMOTE
done

remote_rollout "sudo bash -s -- '${CONTROLLER_IP}' '${TLS_PORT}' '${MANIFEST_SET_SHA256}' '${PROMOTION_MATRIX_SHA256}' '${FINAL_MATRIX_SHA256}' '${PROMOTION_BASELINE_EVIDENCE_SHA256}' '${DEPLOYMENT_GATE_SHA256}'" <<'REMOTE'
set -euo pipefail
CONTROLLER_IP="$1"; TLS_PORT="$2"; MANIFEST_SET_SHA256="$3"; PROMOTION_MATRIX_SHA256="$4"; FINAL_MATRIX_SHA256="$5"; PROMOTION_BASELINE_EVIDENCE_SHA256="$6"; DEPLOYMENT_GATE_SHA256="$7"
grep -qx "LEHOME_CONTROLLER_URL=https://${CONTROLLER_IP}:${TLS_PORT}" /etc/lehome/experiment-evaluator.env
grep -qx "LEHOME_CONTROLLER_CA_FILE=/etc/lehome/tls/controller-ca.crt" /etc/lehome/experiment-evaluator.env
grep -qx "LEHOME_MANIFEST_SET_SHA256=${MANIFEST_SET_SHA256}" /etc/lehome/experiment-evaluator.env
grep -qx "LEHOME_PROMOTION_MATRIX_SHA256=${PROMOTION_MATRIX_SHA256}" /etc/lehome/experiment-evaluator.env
grep -qx "LEHOME_FINAL_MATRIX_SHA256=${FINAL_MATRIX_SHA256}" /etc/lehome/experiment-evaluator.env
grep -qx "LEHOME_PROMOTION_BASELINE_EVIDENCE_SHA256=${PROMOTION_BASELINE_EVIDENCE_SHA256}" /etc/lehome/experiment-evaluator.env
grep -qx "LEHOME_DEPLOYMENT_GATE_SHA256=${DEPLOYMENT_GATE_SHA256}" /etc/lehome/experiment-evaluator.env
test "$(stat -c '%a:%u:%g' /etc/lehome/experiment-evaluator.env)" = 600:0:0
test "$(stat -c '%a:%u:%g' /etc/lehome/private/controller-token)" = 600:0:0
test "$(stat -c '%a:%u:%g' /etc/lehome/private/hf-token)" = 600:0:0
curl --silent --show-error --fail --connect-timeout 10 --cacert /etc/lehome/tls/controller-ca.crt "https://${CONTROLLER_IP}:${TLS_PORT}/health" | grep -qx '{"status":"ok"}'
systemctl is-enabled lehome-experiment-evaluator.service | grep -qx disabled
REMOTE

mark_bootstrap_ready() {
  remote_controller "printf '%s\n' '${DEPLOYMENT_GATE_SHA256}' | sudo tee /etc/lehome/experiment-bootstrap.ready >/dev/null; sudo chmod 0600 /etc/lehome/experiment-bootstrap.ready; sudo chown root:root /etc/lehome/experiment-bootstrap.ready; sudo systemctl disable --now lehome-experiment-controller.service lehome-experiment-controller-proxy.service lehome-experiment-capacity.service"
  for slot in 1 2; do
    remote_worker "${slot}" "printf '%s\n' '${DEPLOYMENT_GATE_SHA256}' | sudo tee /etc/lehome/experiment-bootstrap.ready >/dev/null; sudo chmod 0600 /etc/lehome/experiment-bootstrap.ready; sudo chown root:root /etc/lehome/experiment-bootstrap.ready; sudo systemctl disable --now lehome-experiment-worker.service"
  done
  remote_rollout "printf '%s\n' '${DEPLOYMENT_GATE_SHA256}' | sudo tee /etc/lehome/experiment-bootstrap.ready >/dev/null; sudo chmod 0600 /etc/lehome/experiment-bootstrap.ready; sudo chown root:root /etc/lehome/experiment-bootstrap.ready; sudo systemctl disable --now lehome-experiment-evaluator.service"
}
mark_bootstrap_ready
trap - ERR
echo "bootstrap verified: private TLS, authenticated controller, training workers, and rollout evaluator are configured; services remain disabled"
