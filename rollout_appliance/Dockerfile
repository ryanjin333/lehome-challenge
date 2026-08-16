# syntax=docker/dockerfile:1.7
# Thin rollout appliance layer derived from the loaded official LeHome challenge
# image. Packer downloads the exact LFS tarball, verifies byte length and
# SHA-256 (see challenge-artifact.json), runs `docker load`, then builds this
# layer against it. No model checkpoint, dataset, or secret is baked in.
ARG LEHOME_BASE_IMAGE=lehome-challenge@a914115729bb0bfd260971b9c8d4147bff38c1fb
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
     /opt/lehome/scripts/
COPY trainer/src /opt/lehome/trainer/src
COPY trainer/pyproject.toml /opt/lehome/trainer/pyproject.toml
COPY rollout_appliance/entrypoint.sh /usr/local/bin/lehome-rollout-entrypoint

RUN chmod 0755 /usr/local/bin/lehome-rollout-entrypoint \
    && bash -n /usr/local/bin/lehome-rollout-entrypoint \
    && /opt/runtime/bin/python -m py_compile \
      /opt/lehome/scripts/run_groot_rollout_appliance.py \
      /opt/lehome/scripts/run_groot_batched_policy_server.py \
      /opt/lehome/scripts/run_groot_rollout_controller.py \
      /opt/lehome/scripts/run_groot_artifact_sync.py \
      /opt/lehome/scripts/run_groot_persistent_worker.py

ENV LEHOME_APPLIANCE_ROOT=/opt/lehome \
    PYTHONPATH=/opt/lehome/source/lehome:/opt/lehome/trainer/src \
    LEHOME_WORKSPACE=/mnt/lehome

USER root
WORKDIR /mnt/lehome
ENTRYPOINT ["/usr/local/bin/lehome-rollout-entrypoint"]
CMD ["supervisor"]
