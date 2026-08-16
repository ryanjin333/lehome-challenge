#!/usr/bin/env bash
# Shared provisioning for both golden images: container runtime, guest
# recovery services, and GPU host prerequisites. Runs on a temporary
# on-demand CPU builder; nothing here reads or stores credentials.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl docker.io jq systemd-container \
  nvidia-driver-570-server

# The NVIDIA container toolkit lets the runtime hand the RTX PRO 6000 to the
# trainer/Isaac containers. Installed from NVIDIA's signed Ubuntu repository.
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y --no-install-recommends nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker

# Guest services: shared-disk admission and bounded preemption shutdown.
install -d -m 0755 /opt/lehome/guest/bin /opt/lehome/guest/systemd
install -m 0644 /tmp/lehome-guest/lehome_workspace.py /opt/lehome/guest/lehome_workspace.py
install -m 0644 /tmp/lehome-guest/lehome_preempt.py /opt/lehome/guest/lehome_preempt.py
install -m 0644 /tmp/lehome-guest/systemd/lehome-workspace.service /etc/systemd/system/lehome-workspace.service
install -m 0644 /tmp/lehome-guest/systemd/lehome-preempt.service /etc/systemd/system/lehome-preempt.service
systemctl enable lehome-workspace.service lehome-preempt.service

# Python for the guest services (no third-party packages required).
apt-get install -y --no-install-recommends python3

# Cleanup before image capture.
rm -rf /tmp/lehome-guest /var/lib/apt/lists/*
apt-get clean
