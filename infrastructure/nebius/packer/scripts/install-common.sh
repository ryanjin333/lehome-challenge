#!/usr/bin/env bash
# Shared provisioning for both golden images: container runtime, guest
# recovery services, and GPU host prerequisites. Runs on a temporary
# on-demand CPU builder; nothing here reads or stores credentials.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl docker.io jq systemd-container \
  nebius-observability-agent nebius-observability-agent-updater \
  nvidia-driver-580-server-open

# The NVIDIA container toolkit lets the runtime hand the RTX PRO 6000 to the
# trainer/Isaac containers. Installed from NVIDIA's signed Ubuntu repository.
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y --no-install-recommends nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker

# Nebius GPU dashboards scrape DCGM on localhost:5555. The policy gateway
# must never bind that port. Install DCGM on both golden images so a new
# GPU VM publishes utilization without a manual host-engine install.
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb   -o /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
rm -f /tmp/cuda-keyring.deb
apt-get update
apt-get install -y --no-install-recommends   datacenter-gpu-manager-4-core   datacenter-gpu-manager-4-cuda12
systemctl enable nvidia-dcgm.service
systemctl enable \
  nebius_observability_agent.service \
  nebius_observability_agent_updater.service

# Guest services: shared-disk admission and bounded preemption shutdown.
install -d -m 0755 /opt/lehome/guest/bin /opt/lehome/guest/systemd
install -m 0644 /tmp/lehome-guest/lehome_workspace.py /opt/lehome/guest/lehome_workspace.py
install -m 0644 /tmp/lehome-guest/lehome_preempt.py /opt/lehome/guest/lehome_preempt.py
install -m 0755 /tmp/lehome-guest/bin/lehome-workspace.sh /opt/lehome/guest/bin/lehome-workspace.sh
install -m 0755 /tmp/lehome-guest/bin/lehome-preempt.sh /opt/lehome/guest/bin/lehome-preempt.sh
install -m 0755 /tmp/lehome-guest/bin/lehome-training.sh /opt/lehome/guest/bin/lehome-training.sh
install -m 0755 /tmp/lehome-guest/bin/lehome-training-control.sh /opt/lehome/guest/bin/lehome-training-control.sh
install -m 0644 /tmp/lehome-guest/systemd/lehome-workspace.service /etc/systemd/system/lehome-workspace.service
install -m 0644 /tmp/lehome-guest/systemd/lehome-preempt.service /etc/systemd/system/lehome-preempt.service
install -m 0644 /tmp/lehome-guest/systemd/lehome-training.service /etc/systemd/system/lehome-training.service
systemctl enable lehome-workspace.service lehome-preempt.service

# Python for the guest services (no third-party packages required).
apt-get install -y --no-install-recommends python3 git

# Cleanup before image capture.
rm -rf /tmp/lehome-guest /var/lib/apt/lists/*
apt-get clean
