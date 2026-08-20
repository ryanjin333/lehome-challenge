#!/usr/bin/env bash
# Portable training image provisioning. This image is deliberately NOT based
# on the LeHome challenge tarball and bakes no policy weights, datasets, or
# mixture. A different competition reuses it with a new immutable manifest.
set -euo pipefail

TRAINING_OCI_IMAGE="${TRAINING_OCI_IMAGE:?TRAINING_OCI_IMAGE is required}"
TRAINING_OCI_DIGEST="${TRAINING_OCI_DIGEST:?TRAINING_OCI_DIGEST is required}"
TRAINER_CODE_REVISION="${TRAINER_CODE_REVISION:?TRAINER_CODE_REVISION is required}"
LEHOME_ETC_DIR="${LEHOME_ETC_DIR:-/etc/lehome}"
[[ "${LEHOME_ETC_DIR}" == /* && "${LEHOME_ETC_DIR}" != *".."* ]] || { echo "unsafe LeHome configuration directory" >&2; exit 2; }
[[ "${TRAINER_CODE_REVISION}" =~ ^[0-9a-f]{40}$ ]] || { echo "trainer code revision must be an immutable 40-character commit" >&2; exit 2; }
if [[ ! "${TRAINING_OCI_IMAGE}" =~ @sha256:[0-9a-f]{64}$ || ! "${TRAINING_OCI_DIGEST}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "training OCI image and digest must be pinned SHA-256 values" >&2
  exit 2
fi
if [[ "${TRAINING_OCI_IMAGE##*@}" != "sha256:${TRAINING_OCI_DIGEST}" ]]; then
  echo "training OCI image reference and digest disagree" >&2
  exit 2
fi
TRAINING_OCI_DIGEST="sha256:${TRAINING_OCI_DIGEST}"

systemctl start docker
# The rollout image shares the guest files, but only the reusable training
# image enables the immutable-runtime transaction controller.
systemctl enable lehome-training.service
id -u lehome >/dev/null 2>&1 || useradd --system --home /var/lib/lehome --shell /usr/sbin/nologin lehome
usermod -aG docker lehome
install -d -m 0700 -o lehome -g lehome /var/lib/lehome/cache /var/lib/lehome/output
install -D -m 0755 /tmp/lehome-guest/bin/lehome-experiment-worker.sh /usr/local/bin/lehome-experiment-worker
install -D -m 0644 /tmp/lehome-guest/systemd/lehome-experiment-worker.service /etc/systemd/system/lehome-experiment-worker.service
install -D -m 0755 /tmp/run_lehome_experiment_worker.py /opt/lehome/scripts/run_lehome_experiment_worker.py
install -d -m 0755 /opt/lehome/trainer/src
cp -a /tmp/lehome_train /opt/lehome/trainer/src/lehome_train
chown -R root:root /opt/lehome/scripts /opt/lehome/trainer
python3 -m py_compile /opt/lehome/scripts/run_lehome_experiment_worker.py /opt/lehome/trainer/src/lehome_train/groot/experiment_worker.py /opt/lehome/trainer/src/lehome_train/groot/experiment_runtime_request.py
rm -rf /tmp/run_lehome_experiment_worker.py /tmp/lehome_train
systemctl daemon-reload
# The worker is installed but deliberately inert until its non-secret env and
# separate credential files are supplied by the operator.

# Optional build-time GHCR auth. The token is provided only through the
# Packer environment, is used for this pull, then discarded. It is never
# written to disk or left in the captured image.
if [[ -n "${GHCR_PULL_TOKEN:-}" ]]; then
  printf '%s' "${GHCR_PULL_TOKEN}" | docker login ghcr.io -u token --password-stdin
fi

# Pull the pinned trainer container and prove its digest before use.
docker pull "${TRAINING_OCI_IMAGE}"
OBSERVED_DIGEST="$(docker inspect --format '{{index .RepoDigests 0}}' "${TRAINING_OCI_IMAGE}")"
if [[ "${OBSERVED_DIGEST}" != "${TRAINING_OCI_IMAGE%@*}@${TRAINING_OCI_DIGEST}" ]]; then
  echo "training OCI digest mismatch: ${OBSERVED_DIGEST}" >&2
  exit 1
fi

# The trainer entrypoint and recovery services are already inside the OCI
# image; the host image only records identity metadata for admission checks.
install -d -m 0755 "${LEHOME_ETC_DIR}"
cat > "${LEHOME_ETC_DIR}/training-image-manifest.json" <<EOF
{
  "schema_version": 1,
  "kind": "vla-training-base-image",
  "oci_image": "${TRAINING_OCI_IMAGE}",
  "oci_digest": "${TRAINING_OCI_DIGEST}",
  "trainer_code_revision": "${TRAINER_CODE_REVISION}"
}
EOF
chown root:root "${LEHOME_ETC_DIR}/training-image-manifest.json"
chmod 0444 "${LEHOME_ETC_DIR}/training-image-manifest.json"

# Cleanup before image capture.
docker logout ghcr.io >/dev/null 2>&1 || true
unset GHCR_PULL_TOKEN
rm -f /root/.docker/config.json /home/ubuntu/.docker/config.json || true
docker system prune -f
rm -rf /var/lib/apt/lists/*
apt-get clean
