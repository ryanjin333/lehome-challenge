#!/usr/bin/env bash
# Build the LeHome GR00T rollout image.
#
# Steps:
#   1. Ensure the official lehome-challenge base image is loaded from the
#      Hugging Face tarball (datasets/lehome/docker).
#   2. Build the rollout layer on top (pinned Isaac-GR00T runtime + server).
#   3. Optionally push to a registry (required before Vast can pull it).
#
# NOTE: the image is linux/amd64. On Apple Silicon this builds under qemu and
# is slow; building on an x86_64 Linux host (or a cheap Vast instance) is
# recommended for the real push.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="${IMAGE_TAG:-ryanjin333/lehome-rollout:latest}"
BASE_IMAGE="${BASE_IMAGE:-lehome-challenge:latest}"
PLATFORM="linux/amd64"
PUSH="${PUSH:-0}"
TARBALL_URL="${TARBALL_URL:-https://huggingface.co/datasets/lehome/docker/resolve/main/lehome-challenge.tar.gz}"

log() { printf '[build] %s\n' "$*"; }

if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  tarball="${TMPDIR:-/tmp}/lehome-challenge.tar.gz"
  if [[ ! -s "${tarball}" ]]; then
    log "downloading official image tarball (~tens of GB): ${TARBALL_URL}"
    curl -fL --retry 3 -o "${tarball}" "${TARBALL_URL}"
  fi
  log "loading base image from ${tarball}"
  docker load -i "${tarball}"
  # Normalize whatever tag the tarball ships under.
  loaded="$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -i lehome-challenge | head -n1 || true)"
  if [[ -n "${loaded}" && "${loaded}" != "${BASE_IMAGE}" ]]; then
    log "tagging ${loaded} -> ${BASE_IMAGE}"
    docker tag "${loaded}" "${BASE_IMAGE}"
  fi
fi
docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1 || { echo "base image ${BASE_IMAGE} unavailable" >&2; exit 1; }

log "building ${IMAGE_TAG} (${PLATFORM})"
docker build \
  --platform "${PLATFORM}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  -t "${IMAGE_TAG}" \
  "${HERE}"

log "built ${IMAGE_TAG}"
docker image inspect "${IMAGE_TAG}" --format 'size: {{.Size}} bytes'

if [[ "${PUSH}" == "1" ]]; then
  if [[ -n "${REGISTRY_HOST:-}" && -n "${REGISTRY_USER:-}" && -n "${REGISTRY_TOKEN:-}" ]]; then
    log "logging in to ${REGISTRY_HOST}"
    printf '%s' "${REGISTRY_TOKEN}" | docker login "${REGISTRY_HOST}" -u "${REGISTRY_USER}" --password-stdin
  fi
  log "pushing ${IMAGE_TAG}"
  docker push "${IMAGE_TAG}"
fi
