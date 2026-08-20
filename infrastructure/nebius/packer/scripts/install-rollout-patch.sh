#!/usr/bin/env bash
# Apply a code-only rollout layer on top of an existing READY rollout image.
set -euo pipefail

PINNED_BASE="lehome-challenge:a914115729bb0bfd260971b9c8d4147bff38c1fb"
DERIVED_TAG="lehome-rollout:build"
TRAINER_IMAGE="ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746"

systemctl start docker
docker image inspect "${PINNED_BASE}" >/dev/null

docker build \
  --build-arg LEHOME_BASE_IMAGE="${PINNED_BASE}" \
  --build-arg APPLIANCE_COMMIT="$(git -C /tmp/lehome-repo rev-parse HEAD 2>/dev/null || echo staged)" \
  --tag "${DERIVED_TAG}" \
  --file /tmp/lehome-repo/rollout_appliance/Dockerfile \
  /tmp/lehome-repo
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

install -d -m 0755 /opt/lehome/scripts /opt/lehome/source /opt/lehome/rollout_appliance /opt/lehome/trainer/src
cp -a /tmp/lehome-repo/scripts/. /opt/lehome/scripts/
# The parent image already has this directory. Remove that exact package root
# before copying so cp cannot nest the refreshed tree under lehome/lehome and
# leave stale modules at the canonical import path.
rm -rf -- /opt/lehome/source/lehome
cp -a /tmp/lehome-repo/source/lehome /opt/lehome/source/lehome
cp -a /tmp/lehome-repo/rollout_appliance/. /opt/lehome/rollout_appliance/
cp -a /tmp/lehome-repo/trainer/src/. /opt/lehome/trainer/src/

# Refresh the guest control plane too. The incremental parent can predate a
# first-boot fix even when its heavyweight Isaac/Docker layers are reusable.
install -d -m 0755 /opt/lehome/guest/bin /opt/lehome/guest/systemd
install -m 0644 /tmp/lehome-repo/guest/lehome_workspace.py /opt/lehome/guest/lehome_workspace.py
install -m 0644 /tmp/lehome-repo/guest/lehome_preempt.py /opt/lehome/guest/lehome_preempt.py
install -m 0755 /tmp/lehome-repo/guest/bin/lehome-workspace.sh /opt/lehome/guest/bin/lehome-workspace.sh
install -m 0755 /tmp/lehome-repo/guest/bin/lehome-preempt.sh /opt/lehome/guest/bin/lehome-preempt.sh
install -m 0644 /tmp/lehome-repo/guest/systemd/lehome-workspace.service /etc/systemd/system/lehome-workspace.service
install -m 0644 /tmp/lehome-repo/guest/systemd/lehome-preempt.service /etc/systemd/system/lehome-preempt.service
install -m 0644 /tmp/lehome-repo/guest/systemd/lehome-experiment-evaluator.service /etc/systemd/system/lehome-experiment-evaluator.service
systemctl daemon-reload
systemctl enable lehome-workspace.service lehome-preempt.service
systemctl disable lehome-experiment-evaluator.service >/dev/null 2>&1 || true
chmod 0755 \
  /opt/lehome/scripts/run_lehome_experiment_evaluator.py \
  /opt/lehome/scripts/summarize_groot_persistent_evaluation.py \
  /opt/lehome/rollout_appliance/smoke_one_episode.sh \
  /opt/lehome/rollout_appliance/run_12k_campaign.sh \
  /opt/lehome/rollout_appliance/run_controlled_recovery_campaign.sh \
  /opt/lehome/rollout_appliance/run_controlled_recovery_smoke.sh \
  /opt/lehome/rollout_appliance/run_experiment_evaluator.sh \
  /opt/lehome/rollout_appliance/worker_supervisor.sh \
  /opt/lehome/rollout_appliance/prepare-merged-lehome.sh \
  /opt/lehome/rollout_appliance/run_randomized_top_short_pilot.sh

docker builder prune -f
sync
