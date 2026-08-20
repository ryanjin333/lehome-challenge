#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl python3

# Pin the official Nebius installer to a known version.  The controller never
# relies on a preinstalled or user-home CLI profile; the capacity daemon passes
# its isolated root-owned config on every Compute command.
NEBIUS_CLI_VERSION=0.12.263
NEBIUS_CLI_INSTALL_DIR=/usr/local/lib/lehome/nebius-cli
install -d -m 0755 -o root -g root "${NEBIUS_CLI_INSTALL_DIR}"
NEBIUS_CLI_VERSION="${NEBIUS_CLI_VERSION}" \
  NEBIUS_INSTALL_FOLDER="${NEBIUS_CLI_INSTALL_DIR}" \
  HOME=/var/lib/lehome/nebius-installer \
  SHELL=/bin/sh \
  bash <(curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh)
ln -sfn "${NEBIUS_CLI_INSTALL_DIR}/nebius" /usr/local/bin/nebius
test "$(/usr/local/bin/nebius version)" = "${NEBIUS_CLI_VERSION}"
rm -rf /var/lib/lehome/nebius-installer
install -d -m 0750 -o lehome-controller -g lehome-controller /var/lib/lehome/controller
install -D -m 0755 /tmp/lehome-guest/bin/lehome-experiment-controller.sh /usr/local/bin/lehome-experiment-controller
install -D -m 0755 /tmp/lehome-guest/bin/lehome-experiment-controller-state.sh /usr/local/bin/lehome-experiment-controller-state
install -D -m 0755 /tmp/lehome-guest/bin/lehome-experiment-capacity.sh /usr/local/bin/lehome-experiment-capacity
install -D -m 0644 /tmp/lehome-guest/systemd/lehome-experiment-controller.service /etc/systemd/system/lehome-experiment-controller.service
install -D -m 0644 /tmp/lehome-guest/systemd/lehome-experiment-controller-proxy.service /etc/systemd/system/lehome-experiment-controller-proxy.service
install -D -m 0644 /tmp/lehome-guest/systemd/lehome-experiment-controller-state.service /etc/systemd/system/lehome-experiment-controller-state.service
install -D -m 0644 /tmp/lehome-guest/systemd/lehome-experiment-capacity.service /etc/systemd/system/lehome-experiment-capacity.service
install -D -m 0755 /tmp/run_lehome_experiment_controller.py /opt/lehome/scripts/run_lehome_experiment_controller.py
install -D -m 0755 /tmp/run_lehome_capacity_lifecycle.py /opt/lehome/scripts/run_lehome_capacity_lifecycle.py
install -d -m 0755 /opt/lehome/trainer/src
cp -a /tmp/lehome_train /opt/lehome/trainer/src/lehome_train
chown -R root:root /opt/lehome/scripts /opt/lehome/trainer
python3 -m py_compile /opt/lehome/scripts/run_lehome_experiment_controller.py /opt/lehome/scripts/run_lehome_capacity_lifecycle.py /opt/lehome/trainer/src/lehome_train/groot/experiment_controller.py /opt/lehome/trainer/src/lehome_train/groot/experiment_service.py /opt/lehome/trainer/src/lehome_train/groot/experiment_capacity.py /opt/lehome/trainer/src/lehome_train/groot/experiment_deployment_gate.py
rm -rf /tmp/run_lehome_experiment_controller.py /tmp/run_lehome_capacity_lifecycle.py /tmp/lehome_train
systemctl daemon-reload
systemctl enable lehome-experiment-controller-state.service
