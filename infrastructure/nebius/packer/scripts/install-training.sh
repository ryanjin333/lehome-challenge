#!/usr/bin/env bash
# Portable training image provisioning. This image is deliberately NOT based
# on the LeHome challenge tarball and bakes no policy weights, datasets, or
# mixture. A different competition reuses it with a new immutable manifest.
set -euo pipefail

TRAINING_OCI_IMAGE="${TRAINING_OCI_IMAGE:?TRAINING_OCI_IMAGE is required}"
TRAINING_OCI_DIGEST="${TRAINING_OCI_DIGEST:?TRAINING_OCI_DIGEST is required}"
TRAINER_CODE_REVISION="${TRAINER_CODE_REVISION:?TRAINER_CODE_REVISION is required}"

systemctl start docker

# Pull the pinned trainer container and prove its digest before use.
docker pull "${TRAINING_OCI_IMAGE}"
OBSERVED_DIGEST="$(docker inspect --format '{{index .RepoDigests 0}}' "${TRAINING_OCI_IMAGE}")"
if [[ "${OBSERVED_DIGEST}" != "@${TRAINING_OCI_DIGEST#*@}" && "${OBSERVED_DIGEST}" != "${TRAINING_OCI_IMAGE}" ]]; then
  echo "training OCI digest mismatch: ${OBSERVED_DIGEST}" >&2
  exit 1
fi

# The trainer entrypoint and recovery services are already inside the OCI
# image; the host image only records identity metadata for admission checks.
install -d -m 0755 /etc/lehome
cat > /etc/lehome/training-image-manifest.json <<EOF
{
  "schema_version": 1,
  "kind": "vla-training-base-image",
  "oci_image": "${TRAINING_OCI_IMAGE}",
  "oci_digest": "${TRAINING_OCI_DIGEST}",
  "trainer_code_revision": "${TRAINER_CODE_REVISION}"
}
EOF

# Cleanup before image capture.
docker system prune -f
rm -rf /var/lib/apt/lists/*
apt-get clean
