#!/usr/bin/env bash
# Initialize and mount only the dedicated, Terraform-protected controller
# state disk. This never accepts the rollout disk or an arbitrary device.
set -euo pipefail

DEVICE=/dev/disk/by-id/virtio-controller-state
TARGET=/var/lib/lehome/controller
[[ -b "${DEVICE}" ]] || { echo "controller state disk is unavailable" >&2; exit 2; }
install -d -m 0750 -o lehome-controller -g lehome-controller "${TARGET}"
if mountpoint -q "${TARGET}"; then
  exit 0
fi
if ! blkid -o value -s TYPE "${DEVICE}" >/dev/null 2>&1; then
  mkfs.ext4 -F "${DEVICE}" >/dev/null
fi
mount -o noatime "${DEVICE}" "${TARGET}"
chown lehome-controller:lehome-controller "${TARGET}"
chmod 0750 "${TARGET}"
