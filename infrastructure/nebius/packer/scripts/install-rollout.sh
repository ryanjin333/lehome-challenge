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
docker load --input "${TARBALL}"

# 5. Build the derived four-worker rollout layer from the repository code.
DERIVED_TAG="lehome-rollout:build"
docker build \
  --build-arg LEHOME_BASE_IMAGE="lehome-challenge@${CHALLENGE_REVISION}" \
  --build-arg APPLIANCE_COMMIT="$(git -C /tmp/lehome-repo rev-parse HEAD 2>/dev/null || echo unknown)" \
  --tag "${DERIVED_TAG}" \
  --file /tmp/lehome-repo/rollout_appliance/Dockerfile \
  /tmp/lehome-repo

# 6. Remove the downloaded tarball and all build caches before capture.
rm -f "${TARBALL}"
docker system prune -f
docker builder prune -f
rm -rf /var/lib/apt/lists/*
apt-get clean
