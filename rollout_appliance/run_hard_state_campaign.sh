#!/usr/bin/env bash
# Hard-state recovery campaign. Parent is original 12K.
# Do not collect another 12K-success 70/30 mix. Start with one isolated worker.
# Keep only successful recoveries from near-miss restored fail snapshots.
set -euo pipefail

WORKSPACE="${LEHOME_WORKSPACE:-/mnt/lehome}"
POLICY_SHA256="${LEHOME_POLICY_SHA256:-e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa}"
TRAINER_IMAGE="${LEHOME_TRAINER_IMAGE:-ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746}"
CHECKPOINT_DIR="${LEHOME_CHECKPOINT_DIR:-${WORKSPACE}/eval/policies/original_baseline}"
RECEIPT_DIR="${WORKSPACE}/eval/receipts/original_baseline"
SRC_MATRIX="${WORKSPACE}/eval/campaign-12k-round-3/hard-state-nearmiss.json"
CAMPAIGN_ROOT="${WORKSPACE}/eval/campaign-hard-state-nearmiss-1"
LEDGER="${CAMPAIGN_ROOT}/ledger.sqlite3"
MATRIX="${CAMPAIGN_ROOT}/matrix.json"
ROLLOUT_IMAGE="${LEHOME_ROLLOUT_IMAGE:-lehome-rollout:build}"
MAX_ATTEMPTS="${LEHOME_HARDSTATE_ATTEMPTS:-24}"
TARGET_ACCEPTED="${LEHOME_HARDSTATE_TARGET:-8}"

if [ ! -f "${SRC_MATRIX}" ]; then
  echo "missing ranked hard-state matrix: ${SRC_MATRIX}" >&2
  exit 2
fi

mkdir -p "${RECEIPT_DIR}" "${CAMPAIGN_ROOT}" /eval/logs /kitcache
if [ -x /opt/lehome/rollout_appliance/prepare-merged-lehome.sh ]; then
  /opt/lehome/rollout_appliance/prepare-merged-lehome.sh || true
fi
python3 - <<PY
import json
from pathlib import Path
src = Path("${SRC_MATRIX}")
out = Path("${MATRIX}")
rows = json.loads(src.read_text())
if not rows:
    raise SystemExit("hard-state matrix is empty")
out.write_text(json.dumps(rows[: int("${MAX_ATTEMPTS}")], indent=2, sort_keys=True) + "\n")
print("hard_state_rows", min(len(rows), int("${MAX_ATTEMPTS}")))
print("first", rows[0]["garment"], rows[0]["seed"])
PY
if [ "${LEHOME_FRESH_LEDGER:-1}" = "1" ] || [ ! -f "${LEDGER}" ]; then
  rm -f "${LEDGER}" "${LEDGER}-wal" "${LEDGER}-shm"
fi
mkdir -p "${CAMPAIGN_ROOT}/worker-2"
chown -R 10001:10001 "${RECEIPT_DIR}" || true
chown -R 1234:1234 "${CAMPAIGN_ROOT}" /eval/logs /kitcache || true

docker rm -f lehome-12k-policy lehome-hardstate-w2 >/dev/null 2>&1 || true
docker run --rm --gpus all --user 10001:10001 --network host --ipc=host \
  --name lehome-12k-policy \
  -w /cache/models \
  -v "${CHECKPOINT_DIR}:/policy:ro" \
  -v /opt/lehome/scripts:/opt/lehome/scripts:ro \
  -v /opt/lehome/source/lehome:/opt/lehome-src:ro \
  -v "${RECEIPT_DIR}:/receipts" \
  -v "${WORKSPACE}/cache:/cache" \
  -v "${WORKSPACE}/cache/isaac-groot-overlay/nvidia:/opt/isaac-groot/nvidia:ro" \
  -v "${WORKSPACE}/cache/isaac-groot-overlay/exclude:/opt/isaac-groot/.git/info/exclude:ro" \
  -e PYTHONPATH=/opt/isaac-groot:/opt/lehome-src:/opt/lehome \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint /opt/runtime/bin/python \
  "${TRAINER_IMAGE}" \
  /opt/lehome/scripts/run_groot_batched_policy_server.py \
    --model-path /policy \
    --policy-sha256 "${POLICY_SHA256}" \
    --host 127.0.0.1 \
    --port 15555 \
    --device cuda:0 \
    --seed 12000 \
    --ready-file /receipts/ready.json \
    --metrics-file /receipts/metrics.json \
    --receipt-file /receipts/policy.jsonl &
POLICY_PID=$!
trap 'kill "${POLICY_PID}" 2>/dev/null || true' EXIT

ready_file="${RECEIPT_DIR}/ready.json"
for _ in $(seq 1 180); do
  if python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); raise SystemExit(0 if data.get("ready") is True else 1)' "${ready_file}" 2>/dev/null; then
    break
  fi
  sleep 2
done
python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); raise SystemExit(0 if data.get("ready") is True else 1)' "${ready_file}"
# Policy uid 10001 owns these files; ubuntu cannot chmod them without sudo.
sudo chmod 0755 "${RECEIPT_DIR}" || true
sudo chmod 0644 "${ready_file}" "${RECEIPT_DIR}/metrics.json" || true

first_garment="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[0]["garment"])' "${MATRIX}")"
first_seed="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[0]["seed"])' "${MATRIX}")"
session_id="hardstate-w2-$(uuidgen | tr '[:upper:]' '[:lower:]')"
kit="/kitcache/w2"
mkdir -p "${kit}/home" "${kit}/tmp" "${kit}/xdg" "${kit}/config" "${kit}/ov"
if [ ! -e "${kit}/home/.nvidia-omniverse" ] && [ -d /kitcache/home ]; then
  cp -a /kitcache/home/. "${kit}/home/" 2>/dev/null || true
  cp -a /kitcache/ov/. "${kit}/ov/" 2>/dev/null || true
  cp -a /kitcache/xdg/. "${kit}/xdg/" 2>/dev/null || true
  cp -a /kitcache/config/. "${kit}/config/" 2>/dev/null || true
fi
chown -R 1234:1234 "${kit}" || true

docker run --rm --gpus all --user 1234:1234 --init --network host --ipc=host \
  --name lehome-hardstate-w2 \
  -w /opt/lehome-challenge \
  -v "${WORKSPACE}:/mnt/lehome" \
  -v "${WORKSPACE}/eval/assets:/opt/lehome-challenge/Assets:ro" \
  -v /opt/lehome:/opt/lehome:ro \
  -v /opt/lehome/merged/lehome:/opt/lehome-challenge/source/lehome/lehome:ro \
  -v /opt/lehome/scripts:/opt/lehome-challenge/scripts:ro \
  -v /opt/lehome/pydeps:/pydeps:ro \
  -v /eval/logs:/eval/logs \
  -v /eval/logs:/opt/lehome-challenge/logs \
  -v "${kit}:/kitcache" \
  -e PYTHONEXE=/opt/lehome-challenge/.venv/bin/python \
  -e PYTHONPATH=/pydeps:/opt/lehome:/opt/lehome-challenge/source/lehome:/opt/lehome-challenge/third_party/IsaacLab/source/isaaclab:/opt/lehome-challenge/third_party/IsaacLab/source/isaaclab_tasks \
  -e HOME=/kitcache/home \
  -e TMPDIR=/kitcache/tmp \
  -e XDG_CACHE_HOME=/kitcache/xdg \
  -e XDG_DATA_HOME=/kitcache/xdg \
  -e XDG_CONFIG_HOME=/kitcache/config \
  -e OMNI_DATA_PATH=/kitcache/ov \
  -e OMNI_USER_DIR=/kitcache/ov \
  -e LEHOME_DISABLE_KEYBOARD=1 \
  --entrypoint /isaac-sim/python.sh \
  "${ROLLOUT_IMAGE}" \
  /opt/lehome/scripts/run_groot_persistent_worker.py \
    --headless \
    --database "${LEDGER}" \
    --attempt-matrix "${MATRIX}" \
    --worker-id worker-2 \
    --session-id "${session_id}" \
    --output-root "${CAMPAIGN_ROOT}/worker-2" \
    --renderer-device cuda:0 \
    --policy-device cuda:0 \
    --policy-gateway-endpoint tcp://127.0.0.1:15555 \
    --policy-sha256 "${POLICY_SHA256}" \
    --policy-timeout-seconds 180 \
    --policy-ready-file "${RECEIPT_DIR}/ready.json" \
    --initial-garment "${first_garment}" \
    --seed "${first_seed}" \
    --garment_name "${first_garment}" \
    --max-attempts "${MAX_ATTEMPTS}" \
    --target-accepted "${TARGET_ACCEPTED}" \
    --save_video
