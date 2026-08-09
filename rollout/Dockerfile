# syntax=docker/dockerfile:1.7
# The parent is deliberately required at build time: a tag is never a valid
# substitute for the resolved official BEHAVIOR/Isaac image digest.
ARG BEHAVIOR_PARENT_IMAGE=stanfordvl/behavior@sha256:b789b8d8efefda509b37404a676523d6cee81e2860558287cf6c34c2af3b79c7
FROM ${BEHAVIOR_PARENT_IMAGE} AS rollout-runtime

ARG BEHAVIOR_PARENT_IMAGE
ARG BEHAVIOR_PARENT_DIGEST=sha256:b789b8d8efefda509b37404a676523d6cee81e2860558287cf6c34c2af3b79c7
ARG BEHAVIOR_REVISION=26f2c7ef7b9cf96bd0414f81e1e751e493762779
ARG GROOT_REVISION=ace36d935b376fbf25cd56371e23877b95407c40
ARG REPOSITORY_COMMIT

LABEL org.opencontainers.image.source="https://github.com/ryanjin333/lehome-challenge" \
      org.opencontainers.image.revision="${REPOSITORY_COMMIT}" \
      io.lehome.image-role="rollout" \
      io.lehome.behavior-revision="${BEHAVIOR_REVISION}" \
      io.lehome.isaac-groot-revision="${GROOT_REVISION}" \
      io.lehome.behavior-parent-digest="${BEHAVIOR_PARENT_DIGEST}"

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    OMNI_KIT_ACCEPT_EULA=YES \
    OMNIGIBSON_DATA_PATH=/workspace/omnigibson-data \
    HEADLESS=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HF_HOME=/workspace/.cache/huggingface \
    B1K_HF_TOKEN_FILE=/workspace/.cache/huggingface/token \
    BEHAVIOR_PYTHON=/opt/conda/envs/behavior/bin/python \
    GROOT_PYTHON=/opt/isaac-groot/.venv/bin/python \
    HOME=/home/rollout \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# The parent image currently inherits /opt/isaac-groot as its working
# directory. Move away before replacing that checkout so child processes can
# always resolve their current directory.
WORKDIR /

# The official pinned BEHAVIOR source is the simulator/evaluator authority;
# only serving dependencies are added here, never trainer lifecycle packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git git-lfs python3-pip util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /behavior-src /opt/isaac-groot \
    && git init /behavior-src \
    && git -C /behavior-src remote add origin https://github.com/StanfordVL/BEHAVIOR-1K.git \
    && git -C /behavior-src fetch --depth=1 origin "${BEHAVIOR_REVISION}" \
    && git -C /behavior-src checkout --detach FETCH_HEAD \
    && test "$(git -C /behavior-src rev-parse HEAD)" = "${BEHAVIOR_REVISION}" \
    && test -f /behavior-src/OmniGibson/omnigibson/eval/r1pro.yaml \
    && git init /opt/isaac-groot \
    && git -C /opt/isaac-groot remote add origin https://github.com/wensi-ai/Isaac-GR00T.git \
    && git -C /opt/isaac-groot fetch --depth=1 origin "${GROOT_REVISION}" \
    && git -C /opt/isaac-groot checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/isaac-groot rev-parse HEAD)" = "${GROOT_REVISION}" \
    && python3 -m pip install --no-cache-dir huggingface_hub==0.36.2 uv==0.8.22

RUN [[ "${BEHAVIOR_PARENT_IMAGE}" == *@"${BEHAVIOR_PARENT_DIGEST}" ]] \
    || { echo "BEHAVIOR_PARENT_IMAGE must be pinned to BEHAVIOR_PARENT_DIGEST" >&2; exit 64; }

WORKDIR /opt/rollout
COPY rollout/pyproject.toml rollout/uv.lock ./
COPY rollout/src ./src
COPY rollout/task-manifest.json ./task-manifest.json
COPY rollout/entrypoint.sh /usr/local/bin/b1k-rollout-entrypoint
RUN /opt/conda/envs/behavior/bin/python -m pip install --no-cache-dir . \
    && cd /opt/isaac-groot && uv sync --frozen --no-dev \
    && /opt/isaac-groot/.venv/bin/python -c 'import gr00t' \
    && "$BEHAVIOR_PYTHON" -c 'import b1k_rollout.policy_server' \
    && test -f /behavior-src/OmniGibson/omnigibson/eval/r1pro.yaml \
    && test -f /opt/isaac-groot/examples/b1k/r1pro.py \
    && test -f /opt/isaac-groot/examples/b1k/r1pro.json \
    && test -f /opt/isaac-groot/scripts/b1k/serve_b1k.py \
    && test -f /opt/isaac-groot/gr00t/policy/websocket_b1k_server.py \
    && test -f /opt/isaac-groot/gr00t/eval/eval_b1k_wrapper.py \
    && groupadd --gid 10001 rollout \
    && useradd --uid 10001 --gid rollout --create-home --home-dir /home/rollout --shell /bin/bash rollout \
    && install -d -o rollout -g rollout -m 0700 /workspace /workspace/campaign /workspace/checkpoint-source /workspace/omnigibson-data /workspace/.cache/huggingface \
    && chmod 0755 /usr/local/bin/b1k-rollout-entrypoint

WORKDIR /workspace
HEALTHCHECK --interval=30s --timeout=5s --start-period=45m --retries=3 CMD /opt/conda/envs/behavior/bin/python -m b1k_rollout.cli healthcheck
ENTRYPOINT ["/usr/local/bin/b1k-rollout-entrypoint"]
CMD []
