# syntax=docker/dockerfile:1.7
# Thin rollout appliance layer derived from the loaded official LeHome challenge
# image. Packer downloads the exact LFS tarball, verifies byte length and
# SHA-256 (see challenge-artifact.json), runs `docker load`, then builds this
# layer against it. No model checkpoint, dataset, or secret is baked in.
ARG LEHOME_BASE_IMAGE=lehome-challenge:a914115729bb0bfd260971b9c8d4147bff38c1fb
FROM ${LEHOME_BASE_IMAGE}

ARG APPLIANCE_COMMIT
LABEL org.opencontainers.image.source="https://github.com/lehome-official/lehome-challenge" \
      io.lehome.appliance="rollout-four-worker" \
      io.lehome.appliance-commit="${APPLIANCE_COMMIT}" \
      io.lehome.challenge-sha256="1a85e389962909debc4ee9988d8a8c388f905fba60686ef78b1623e6872f7123"

# Runtime code only: the four-worker appliance, the session-aware policy
# gateway, the controller/finalizer/uploader CLIs, and the LeHome runtime
# package. The base image already owns Isaac Sim, GR00T, and the trainer env.
COPY source/lehome /opt/lehome/source/lehome
COPY scripts/run_groot_rollout_appliance.py \
     scripts/run_groot_batched_policy_server.py \
     scripts/run_groot_rollout_controller.py \
     scripts/run_groot_artifact_sync.py \
     scripts/run_groot_persistent_worker.py \
     scripts/build_randomized_pilot_matrix.py \
     scripts/build_controlled_recovery_matrix.py \
     scripts/build_success_replay_matrix.py \
     /opt/lehome/scripts/
COPY scripts/__init__.py /opt/lehome/scripts/__init__.py
COPY scripts/eval_policy /opt/lehome/scripts/eval_policy
COPY scripts/utils /opt/lehome/scripts/utils
COPY trainer/src /opt/lehome/trainer/src
COPY trainer/pyproject.toml /opt/lehome/trainer/pyproject.toml
COPY rollout_appliance/entrypoint.sh /usr/local/bin/lehome-rollout-entrypoint
COPY rollout_appliance/smoke_one_episode.sh \
     rollout_appliance/one_episode_smoke.py \
     rollout_appliance/validate_fresh_replay_evidence.py \
     rollout_appliance/run_12k_campaign.sh \
     rollout_appliance/run_success_replay_campaign.sh \
     rollout_appliance/run_controlled_recovery_campaign.sh \
     rollout_appliance/run_controlled_recovery_smoke.sh \
     rollout_appliance/run_snapshot_source_bootstrap.sh \
     rollout_appliance/worker_supervisor.sh \
     rollout_appliance/run_randomized_top_short_pilot.sh \
     rollout_appliance/campaign_top_short_geometry_pilot.json \
     rollout_appliance/campaign_top_short_geometry_pilot.json.sha256 \
     /opt/lehome/rollout_appliance/

RUN chmod 0755 /usr/local/bin/lehome-rollout-entrypoint \
    && chmod 0755 /opt/lehome/scripts/build_success_replay_matrix.py \
    && bash -n /usr/local/bin/lehome-rollout-entrypoint \
    && chmod 0755 /opt/lehome/rollout_appliance/smoke_one_episode.sh \
                  /opt/lehome/rollout_appliance/validate_fresh_replay_evidence.py \
                  /opt/lehome/rollout_appliance/run_12k_campaign.sh \
                  /opt/lehome/rollout_appliance/run_success_replay_campaign.sh \
                  /opt/lehome/rollout_appliance/run_controlled_recovery_campaign.sh \
                  /opt/lehome/rollout_appliance/run_controlled_recovery_smoke.sh \
                  /opt/lehome/rollout_appliance/run_snapshot_source_bootstrap.sh \
                  /opt/lehome/rollout_appliance/worker_supervisor.sh \
                  /opt/lehome/rollout_appliance/run_randomized_top_short_pilot.sh \
    && bash -n /opt/lehome/rollout_appliance/run_12k_campaign.sh \
    && bash -n /opt/lehome/rollout_appliance/run_success_replay_campaign.sh \
    && bash -n /opt/lehome/rollout_appliance/run_controlled_recovery_campaign.sh \
    && bash -n /opt/lehome/rollout_appliance/run_controlled_recovery_smoke.sh \
    && bash -n /opt/lehome/rollout_appliance/run_snapshot_source_bootstrap.sh \
    && bash -n /opt/lehome/rollout_appliance/worker_supervisor.sh \
    && bash -n /opt/lehome/rollout_appliance/run_randomized_top_short_pilot.sh \
    && /opt/lehome-challenge/.venv/bin/python -m py_compile \
      /opt/lehome/scripts/run_groot_rollout_appliance.py \
      /opt/lehome/scripts/run_groot_batched_policy_server.py \
      /opt/lehome/scripts/run_groot_rollout_controller.py \
      /opt/lehome/scripts/run_groot_artifact_sync.py \
      /opt/lehome/scripts/run_groot_persistent_worker.py \
      /opt/lehome/scripts/build_randomized_pilot_matrix.py \
      /opt/lehome/scripts/build_controlled_recovery_matrix.py \
      /opt/lehome/scripts/build_success_replay_matrix.py \
      /opt/lehome/source/lehome/lehome/flywheel/fresh_replay_evidence.py \
      /opt/lehome/scripts/eval_policy/groot_policy.py \
      /opt/lehome/scripts/utils/evaluation.py \
      /opt/lehome/rollout_appliance/one_episode_smoke.py \
      /opt/lehome/rollout_appliance/validate_fresh_replay_evidence.py

ENV LEHOME_APPLIANCE_ROOT=/opt/lehome \
    PYTHONPATH=/opt/lehome/source/lehome:/opt/lehome/trainer/src \
    LEHOME_WORKSPACE=/mnt/lehome

USER root
WORKDIR /mnt/lehome
ENTRYPOINT ["/usr/local/bin/lehome-rollout-entrypoint"]
CMD ["supervisor"]
