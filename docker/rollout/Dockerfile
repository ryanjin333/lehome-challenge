# syntax=docker/dockerfile:1.7
#
# LeHome Challenge rollout image: official Isaac Sim 5.1 / Isaac Lab / LeRobot
# simulator (lehome-challenge) + pinned Isaac-GR00T inference runtime serving
# the BC checkpoint over the challenge docker-policy HTTP protocol.
#
# Base image is the official challenge image loaded from
# https://huggingface.co/datasets/lehome/docker (see build.sh).
ARG BASE_IMAGE=lehome-challenge:latest
FROM ${BASE_IMAGE}

ARG UV_VERSION=0.8.22
ARG UV_SHA256=741ff1f5742c5a4a25d2f829e8395355e43f7a5ae2ebc6368e9ae2df0efb69cf
ARG PYTHON_VERSION=3.10.18
ARG ISAAC_GROOT_REVISION=23ace64f17aa5015259b8609d371eb61a357c776

LABEL org.opencontainers.image.title="lehome-groot-rollout" \
      org.opencontainers.image.description="LeHome Challenge rollout worker: Isaac Sim evaluator + GR00T N1.7 BC policy server" \
      io.lehome.isaac-groot-revision="${ISAAC_GROOT_REVISION}"

SHELL ["/bin/sh", "-ec"]
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/gr00t-runtime \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300

# Tooling needed to hydrate pinned sources at build time. Guarded so images
# that already ship these packages do not pay apt churn.
RUN if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then \
      apt-get update \
      && apt-get install -y --no-install-recommends ca-certificates curl git git-lfs \
      && rm -rf /var/lib/apt/lists/*; \
    fi \
    && curl -fsSLo /tmp/uv.tar.gz \
      "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    && echo "${UV_SHA256}  /tmp/uv.tar.gz" | sha256sum --check --strict \
    && tar -xzf /tmp/uv.tar.gz -C /tmp \
    && install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv \
    && rm -rf /tmp/uv.tar.gz /tmp/uv-x86_64-unknown-linux-gnu \
    && uv python install "${PYTHON_VERSION}"

# Pinned Isaac-GR00T checkout + frozen dependency sync into a dedicated venv,
# mirroring the proven trainer image build. LFS artifact files are skipped so
# the checkout stays lean; model weights hydrate from the Hub at runtime.
WORKDIR /opt/isaac-groot
RUN git init . \
    && git remote add origin https://github.com/NVIDIA/Isaac-GR00T.git \
    && git fetch --depth=1 origin "${ISAAC_GROOT_REVISION}" \
    && GIT_LFS_SKIP_SMUDGE=1 git checkout --detach FETCH_HEAD \
    && git ls-files -z -- '*.safetensors' '*.ckpt' '*.parquet' '*.mp4' '*.gif' '*.whl' \
      > /tmp/isaac-groot-artifacts \
    && if [ -s /tmp/isaac-groot-artifacts ]; then \
      xargs -0 git update-index --skip-worktree -- < /tmp/isaac-groot-artifacts; \
      xargs -0 rm -f -- < /tmp/isaac-groot-artifacts; \
    fi \
    && rm -f /tmp/isaac-groot-artifacts \
    && test "$(git rev-parse HEAD)" = "${ISAAC_GROOT_REVISION}" \
    && uv sync --frozen --no-dev --no-cache \
    && uv pip install --python /opt/gr00t-runtime/bin/python --no-cache-dir msgpack pyzmq \
    && git config --system --add safe.directory /opt/isaac-groot \
    && test "$(git -C /opt/isaac-groot rev-parse HEAD)" = "${ISAAC_GROOT_REVISION}"

# Rollout runtime files.
COPY --chmod=0755 groot_policy_server.py /opt/lehome-rollout/groot_policy_server.py
COPY --chmod=0755 entrypoint.sh /usr/local/bin/lehome-rollout-entrypoint
RUN /opt/gr00t-runtime/bin/python -m py_compile /opt/lehome-rollout/groot_policy_server.py \
    && mkdir -p /workspace/rollout /cache/models/nvidia

ENV LEHOME_ROOT=/opt/lehome-challenge \
    GROOT_ROOT=/opt/isaac-groot \
    GROOT_PYTHON=/opt/gr00t-runtime/bin/python \
    POLICY_SERVER_SCRIPT=/opt/lehome-rollout/groot_policy_server.py \
    POLICY_SERVER_PORT=8080 \
    GROOT_POLICY_DEVICE=cuda:0 \
    POLICY_REPO=ryanjin333/lehome-groot-n17-models \
    POLICY_REVISION=30ac1a84da67b099e115ad147bcd61e9d60046d3 \
    POLICY_INCLUDE=policies/step-12000/* \
    POLICY_MODEL_SUBDIR=policies/step-12000 \
    BACKBONE_REPO=nvidia/Cosmos-Reason2-2B \
    BACKBONE_LOCAL_PATH=/cache/models/nvidia/Cosmos-Reason2-2B \
    GROOT_ACTION_HORIZON=40 \
    ASSET_REPO=lehome/asset_challenge \
    ROLLOUT_ROOT=/workspace/rollout \
    GARMENT_TYPES="top_long top_short pant_long pant_short" \
    NUM_EPISODES=5 \
    MAX_STEPS=600 \
    SAVE_VIDEO=1 \
    HF_HOME=/workspace/hf-cache \
    HF_HUB_CACHE=/workspace/hf-cache/hub \
    PYTHONUNBUFFERED=1

WORKDIR /opt/lehome-challenge
ENTRYPOINT ["/usr/local/bin/lehome-rollout-entrypoint"]
CMD ["rollout"]
