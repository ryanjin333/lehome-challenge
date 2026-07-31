# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719

ARG UV_VERSION=0.8.22
ARG UV_SHA256=741ff1f5742c5a4a25d2f829e8395355e43f7a5ae2ebc6368e9ae2df0efb69cf
ARG PYTHON_VERSION=3.10.18
ARG ISAAC_GROOT_REVISION=23ace64f17aa5015259b8609d371eb61a357c776
ARG REPOSITORY_COMMIT

LABEL org.opencontainers.image.source="https://github.com/lehome-official/lehome-challenge" \
      org.opencontainers.image.revision="${REPOSITORY_COMMIT}" \
      io.lehome.cuda-base-digest="sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719" \
      io.lehome.isaac-groot-revision="${ISAAC_GROOT_REVISION}" \
      io.lehome.model-revision="2fc962b973bccdd5d8ce4f67cc63b264d6886495"

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/runtime \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    CUDA_HOME=/usr/local/cuda

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl ffmpeg git git-lfs libaio-dev libegl1 util-linux \
    && curl -fsSLo /tmp/uv.tar.gz \
      "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    && echo "${UV_SHA256}  /tmp/uv.tar.gz" | sha256sum --check --strict \
    && tar -xzf /tmp/uv.tar.gz -C /tmp \
    && install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv \
    && rm -rf /tmp/uv.tar.gz /tmp/uv-x86_64-unknown-linux-gnu \
    && uv python install "${PYTHON_VERSION}" \
    && test "$(uv run --no-project --python "${PYTHON_VERSION}" python -c 'import platform; print(platform.python_version())')" = "${PYTHON_VERSION}"

WORKDIR /opt/isaac-groot
RUN git init . \
    && git remote add origin https://github.com/NVIDIA/Isaac-GR00T.git \
    && git fetch --depth=1 origin "${ISAAC_GROOT_REVISION}" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${ISAAC_GROOT_REVISION}" \
    && test -z "$(git status --porcelain=v1 --untracked-files=all)" \
    && uv sync --frozen --no-dev --no-install-project --no-cache

# Validate the trainer lock independently, then install the local package into
# the exact same runtime as the pinned upstream checkout. Upstream's lock owns
# the shared CUDA/PyTorch dependency graph.
WORKDIR /opt/trainer
COPY trainer/pyproject.toml trainer/uv.lock ./
RUN UV_PROJECT_ENVIRONMENT=/tmp/trainer-lock-check \
      uv sync --frozen --no-dev --no-install-project --no-cache \
    && rm -rf /tmp/trainer-lock-check
COPY trainer/src ./src
COPY trainer/config ./config
COPY trainer/docker/entrypoint.sh /usr/local/bin/lehome-entrypoint
RUN uv pip install --python /opt/runtime/bin/python --no-deps /opt/trainer \
    && test "$(/opt/runtime/bin/python -c 'import platform; print(platform.python_version())')" = "${PYTHON_VERSION}" \
    && test "$(git -C /opt/isaac-groot rev-parse HEAD)" = "${ISAAC_GROOT_REVISION}" \
    && /opt/runtime/bin/lehome-train --help >/dev/null \
    && groupadd --gid 10001 trainer \
    && useradd --uid 10001 --gid trainer --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin trainer \
    && install -d -o trainer -g trainer -m 0750 /cache /prepared /output \
    && chmod 0755 /usr/local/bin/lehome-entrypoint

ENV PATH=/opt/runtime/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    VIRTUAL_ENV=/opt/runtime \
    PYTHONPATH=/opt/isaac-groot \
    LEHOME_GROOT_ROOT=/opt/isaac-groot \
    LEHOME_TRAIN_RUNTIME_FACTORY=lehome_train.groot.production_runtime:create \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    XDG_CACHE_HOME=/cache/xdg \
    TORCH_HOME=/cache/torch \
    WANDB_DIR=/output/wandb \
    TMPDIR=/cache/tmp \
    HOME=/nonexistent \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER trainer
WORKDIR /output
ENTRYPOINT ["/usr/local/bin/lehome-entrypoint"]
CMD ["--help"]
