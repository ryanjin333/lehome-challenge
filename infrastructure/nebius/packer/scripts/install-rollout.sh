#!/usr/bin/env bash
# LeHome-specific rollout image provisioning. Downloads the exact official
# challenge tarball, verifies byte length and LFS SHA-256 BEFORE docker
# load, builds the derived four-worker layer, then removes every downloaded
# artifact so the captured image carries runtime layers only.
set -euo pipefail

CHALLENGE_REPOSITORY="${CHALLENGE_REPOSITORY:?required}"
CHALLENGE_REVISION="${CHALLENGE_REVISION:?required}"
CHALLENGE_SIZE="${CHALLENGE_SIZE:?required}"
CHALLENGE_SHA256="${CHALLENGE_SHA256:?required}"
CHALLENGE_URL="${CHALLENGE_URL:?required}"
TRAINER_IMAGE="ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
TARBALL="${WORK_DIR}/lehome-challenge.tar.gz"

systemctl start docker

# 1. Download the exact LFS object.
curl -fsSL --retry 5 --retry-delay 10 -o "${TARBALL}" "${CHALLENGE_URL}"

# 2. Verify byte length first.
OBSERVED_SIZE="$(stat -c %s "${TARBALL}")"
if [[ "${OBSERVED_SIZE}" != "${CHALLENGE_SIZE}" ]]; then
  echo "challenge tarball size mismatch: ${OBSERVED_SIZE} != ${CHALLENGE_SIZE}" >&2
  exit 1
fi

# 3. Verify SHA-256 before docker load.
echo "${CHALLENGE_SHA256}  ${TARBALL}" | sha256sum --check --strict

# 4. Load the official challenge image.
# The official tarball currently loads as lehome-challenge:latest. Docker FROM
# cannot use lehome-challenge@<git-sha>; that is not a digest. Retag the loaded
# image to a valid name:tag that still pins the exact challenge revision.
LOAD_OUTPUT="$(docker load --input "${TARBALL}")"
printf '%s\n' "${LOAD_OUTPUT}"
LOADED_IMAGE="$(printf '%s\n' "${LOAD_OUTPUT}" | sed -n 's/^Loaded image: //p' | tail -n 1)"
if [[ -z "${LOADED_IMAGE}" || "${LOADED_IMAGE}" == *":<none>" ]]; then
  echo "docker load did not produce a tagged challenge image" >&2
  exit 1
fi
PINNED_BASE="lehome-challenge:${CHALLENGE_REVISION}"
docker tag "${LOADED_IMAGE}" "${PINNED_BASE}"
docker image inspect "${PINNED_BASE}" >/dev/null

# 5. Build the derived four-worker rollout layer from the repository code.
DERIVED_TAG="lehome-rollout:build"
docker build \
  --build-arg LEHOME_BASE_IMAGE="${PINNED_BASE}" \
  --build-arg APPLIANCE_COMMIT="$(git -C /tmp/lehome-repo rev-parse HEAD 2>/dev/null || echo unknown)" \
  --tag "${DERIVED_TAG}" \
  --file /tmp/lehome-repo/rollout_appliance/Dockerfile \
  /tmp/lehome-repo

# The policy server runs from the separately portable training runtime. Keep
# this immutable image on the rollout boot disk so a replacement GPU does not
# spend paid time pulling multi-gigabyte layers before the first episode.
for pull_attempt in 1 2 3 4 5; do
  if docker pull "${TRAINER_IMAGE}"; then
    break
  fi
  if [ "${pull_attempt}" -eq 5 ]; then
    echo "failed to pull pinned trainer image after ${pull_attempt} attempts" >&2
    exit 1
  fi
  sleep "$((pull_attempt * 2))"
done

# 6. Host copies of appliance scripts so the policy-server container can
# bind-mount /opt/lehome/scripts without depending on a live git checkout.
install -d -m 0755 /opt/lehome/scripts /opt/lehome/source /opt/lehome/trainer/src
cp -a /tmp/lehome-repo/scripts/. /opt/lehome/scripts/
cp -a /tmp/lehome-repo/source/lehome /opt/lehome/source/lehome
cp -a /tmp/lehome-repo/trainer/src/. /opt/lehome/trainer/src/
install -d -m 0755 /opt/lehome/rollout_appliance
for artifact in \
  smoke_one_episode.sh \
  one_episode_smoke.py \
  prepare-merged-lehome.sh \
  run_12k_campaign.sh \
  run_success_replay_campaign.sh \
  run_controlled_recovery_campaign.sh \
  run_controlled_recovery_smoke.sh \
  run_experiment_evaluator.sh \
  worker_supervisor.sh \
  run_randomized_top_short_pilot.sh \
  eval_unseen80_smoke_v1.json \
  eval_unseen80_smoke_v1.json.sha256 \
  campaign_400_balanced_geometry_v1.json \
  campaign_400_balanced_geometry_v1.json.sha256 \
  campaign_top_short_geometry_pilot.json \
  campaign_top_short_geometry_pilot.json.sha256; do
  cp -a "/tmp/lehome-repo/rollout_appliance/${artifact}" "/opt/lehome/rollout_appliance/${artifact}"
done
chmod 0755 \
  /opt/lehome/scripts/run_lehome_experiment_evaluator.py \
  /opt/lehome/scripts/summarize_groot_persistent_evaluation.py \
  /opt/lehome/scripts/build_success_replay_matrix.py \
  /opt/lehome/rollout_appliance/smoke_one_episode.sh \
  /opt/lehome/rollout_appliance/prepare-merged-lehome.sh \
  /opt/lehome/rollout_appliance/run_12k_campaign.sh \
  /opt/lehome/rollout_appliance/run_success_replay_campaign.sh \
  /opt/lehome/rollout_appliance/run_controlled_recovery_campaign.sh \
  /opt/lehome/rollout_appliance/run_controlled_recovery_smoke.sh \
  /opt/lehome/rollout_appliance/run_experiment_evaluator.sh \
  /opt/lehome/rollout_appliance/worker_supervisor.sh \
  /opt/lehome/rollout_appliance/run_randomized_top_short_pilot.sh
install -D -m 0644 /tmp/lehome-guest/systemd/lehome-experiment-evaluator.service /etc/systemd/system/lehome-experiment-evaluator.service
systemctl daemon-reload
systemctl disable lehome-experiment-evaluator.service >/dev/null 2>&1 || true
bash -n /opt/lehome/rollout_appliance/run_success_replay_campaign.sh

# 7. Worker-side Linux wheels. Isaac's 3.11 venv has no pip; keep msgpack and
# pyzmq on PYTHONPATH so the session gateway can talk to the policy server.
install -d -m 0755 /opt/lehome/pydeps /eval/logs /kitcache/tmp /kitcache/xdg /kitcache/ov
python3 - <<'WHEELS'
from pathlib import Path
import urllib.request
import zipfile

wheels = (
    (
        "https://files.pythonhosted.org/packages/a8/a1/ad7b84b91ab5a324e707f4c9761633e357820b011a01e34ce658c1dda7cc/msgpack-1.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "msgpack-1.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "5e1da8f11a3dd397f0a32c76165cf0c4eb95b31013a94f6ecc0b280c05c91b59",
    ),
    (
        "https://files.pythonhosted.org/packages/6c/29/0652a39d4e876e0d61379047ecf7752685414ad2e253434348246f7a2a39/pyzmq-27.0.1-cp311-cp311-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl",
        "pyzmq-27.0.1-cp311-cp311-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl",
        "c512824360ea7490390566ce00bee880e19b526b312b25cc0bc30a0fe95cb67f",
    ),
)
dest = Path("/opt/lehome/pydeps")
dest.mkdir(parents=True, exist_ok=True)
import hashlib
for url, name, digest in wheels:
    target = dest / name
    urllib.request.urlretrieve(url, target)
    observed = hashlib.sha256(target.read_bytes()).hexdigest()
    if observed != digest:
        raise SystemExit(f"{name} digest mismatch: {observed}")
    with zipfile.ZipFile(target) as archive:
        archive.extractall(dest)
print("msgpack==1.1.0")
print("pyzmq==27.0.1")
print("cp311")
print("/opt/lehome/pydeps")
WHEELS
chown -R 1234:1234 /eval/logs /kitcache || true
chmod 0777 /eval/logs /kitcache /kitcache/tmp

# 8. Remove the downloaded tarball and all build caches before capture.
rm -f "${TARBALL}"
docker system prune -f
docker builder prune -f
rm -rf /var/lib/apt/lists/*
apt-get clean
