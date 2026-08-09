# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719 AS training-build

ARG UV_VERSION=0.8.22
ARG UV_SHA256=741ff1f5742c5a4a25d2f829e8395355e43f7a5ae2ebc6368e9ae2df0efb69cf
ARG PYTHON_VERSION=3.10.18
ARG UBUNTU_SNAPSHOT=20260701T000000Z
ARG ISAAC_GROOT_REVISION=ace36d935b376fbf25cd56371e23877b95407c40
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
    printf '%s\n' \
      "deb [check-valid-until=no signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/ jammy main universe" \
      "deb [check-valid-until=no signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/ jammy-updates main universe" \
      "deb [check-valid-until=no signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/ jammy-security main universe" \
      > /tmp/ubuntu-snapshot.list \
    && mkdir -p /tmp/apt-lists/partial \
    && apt_options=( \
      -o Dir::Etc::sourcelist=/tmp/ubuntu-snapshot.list \
      -o Dir::Etc::sourceparts=- \
      -o Dir::State::lists=/tmp/apt-lists \
    ) \
    && apt-get "${apt_options[@]}" update \
    && apt-get "${apt_options[@]}" install -y --no-install-recommends \
      build-essential ca-certificates curl ffmpeg git git-lfs libaio-dev libegl1 util-linux \
    && curl -fsSLo /tmp/uv.tar.gz \
      "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    && echo "${UV_SHA256}  /tmp/uv.tar.gz" | sha256sum --check --strict \
    && tar -xzf /tmp/uv.tar.gz -C /tmp \
    && install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv \
    && rm -rf /tmp/uv.tar.gz /tmp/uv-x86_64-unknown-linux-gnu /tmp/apt-lists /tmp/ubuntu-snapshot.list \
    && uv python install "${PYTHON_VERSION}" \
    && test "$(uv run --no-project --python "${PYTHON_VERSION}" python -c 'import platform; print(platform.python_version())')" = "${PYTHON_VERSION}"

WORKDIR /opt/isaac-groot
RUN git init . \
    && git remote add origin https://github.com/wensi-ai/Isaac-GR00T.git \
    && git fetch --depth=1 origin "${ISAAC_GROOT_REVISION}" \
    && GIT_LFS_SKIP_SMUDGE=1 git checkout --detach FETCH_HEAD \
    && git ls-files -z -- '*.safetensors' '*.ckpt' '*.parquet' '*.mp4' '*.gif' '*.whl' \
      > /tmp/isaac-groot-artifacts \
    && if [[ -s /tmp/isaac-groot-artifacts ]]; then \
      xargs -0 git update-index --skip-worktree -- < /tmp/isaac-groot-artifacts; \
      xargs -0 rm -f -- < /tmp/isaac-groot-artifacts; \
    fi \
    && rm -f /tmp/isaac-groot-artifacts \
    && test "$(git rev-parse HEAD)" = "${ISAAC_GROOT_REVISION}" \
    && test -z "$(git status --porcelain=v1 --untracked-files=all)" \
    && test -f scripts/b1k/train_b1k.py \
    && test -f scripts/b1k/deploy_modality.py \
    && test -f examples/b1k/r1pro.py \
    && test -f examples/b1k/r1pro.json \
    && test -f gr00t/data/dataset/lerobot_episode_loader.py \
    && uv sync --frozen --no-dev --no-install-project --no-cache \
    && git config --system --add safe.directory /opt/isaac-groot

# Export the trainer's frozen dependency graph and reconcile it into the same
# runtime as the pinned upstream checkout. Shared exact pins (Hub, PyArrow, and
# Click) are aligned with upstream; pip check then rejects any incompatible
# composite environment.
WORKDIR /opt/trainer
COPY trainer/pyproject.toml trainer/uv.lock ./
RUN uv lock --check \
    && uv export --frozen --no-dev --no-emit-project --no-hashes \
      --output-file /tmp/trainer-requirements.txt \
    && uv pip install --python /opt/runtime/bin/python \
      --requirement /tmp/trainer-requirements.txt \
    && rm -f /tmp/trainer-requirements.txt
COPY trainer/src ./src
COPY trainer/config ./config
COPY trainer/scripts/verify-b1k-cli.py /usr/local/bin/verify-b1k-cli
COPY trainer/b1k_launchkit/onstart.sh /opt/b1k-launchkit/onstart.sh
COPY trainer/b1k_launchkit/token_bootstrap.py /opt/b1k-launchkit/token_bootstrap.py
COPY trainer/b1k_launchkit/training_smoke.py /opt/b1k-launchkit/training_smoke.py
COPY trainer/b1k_launchkit/bucket_helper/pyproject.toml trainer/b1k_launchkit/bucket_helper/uv.lock /opt/b1k-bucket-helper-src/
COPY trainer/b1k_launchkit/bucket_helper/b1k-bucket-helper /opt/b1k-bucket-helper-src/b1k-bucket-helper
COPY trainer/docker/entrypoint.sh /usr/local/bin/lehome-entrypoint
RUN uv pip install --python /opt/runtime/bin/python --no-deps /opt/trainer \
    && uv pip check --python /opt/runtime/bin/python \
    && test "$(/opt/runtime/bin/python -c 'import platform; print(platform.python_version())')" = "${PYTHON_VERSION}" \
    && test "$(git -C /opt/isaac-groot rev-parse HEAD)" = "${ISAAC_GROOT_REVISION}" \
    && /opt/runtime/bin/lehome-train --help >/dev/null \
    && chmod 0755 /usr/local/bin/verify-b1k-cli \
    && cd /opt/b1k-bucket-helper-src \
    && UV_PROJECT_ENVIRONMENT=/opt/b1k-bucket-helper uv sync --frozen --no-dev --no-install-project --python /opt/runtime/bin/python \
    && install -D -m 0755 /opt/b1k-bucket-helper-src/b1k-bucket-helper /opt/b1k-bucket-helper/libexec/b1k-bucket-helper \
    && printf '%s\n' '#!/bin/sh' 'exec /opt/b1k-bucket-helper/bin/python /opt/b1k-bucket-helper/libexec/b1k-bucket-helper "$@"' > /opt/b1k-bucket-helper/bin/b1k-bucket-helper \
    && chmod 0755 /opt/b1k-bucket-helper/bin/b1k-bucket-helper \
    && groupadd --gid 10001 trainer \
    && useradd --uid 10001 --gid trainer --create-home --home-dir /home/trainer --shell /bin/bash trainer \
    && install -o trainer -g trainer -m 0640 /dev/null /home/trainer/.bashrc \
    && chmod 0750 /home/trainer \
    && install -d -o trainer -g trainer -m 0750 /opt/isaac-groot/.git/lfs/tmp \
    && install -d -o trainer -g trainer -m 0750 /cache /prepared /output \
    && install -d -o trainer -g trainer -m 0750 /cache/tmp \
    && install -d -o trainer -g trainer -m 0750 /workspace/data/b1k /workspace/models/groot-upstream /workspace/models/cosmos /workspace/models/groot /workspace/checkpoints /workspace/logs /workspace/final /workspace/outputs \
    && chmod 0755 /opt/b1k-launchkit/onstart.sh /opt/b1k-launchkit/token_bootstrap.py /opt/b1k-launchkit/training_smoke.py \
    && chmod 0755 /usr/local/bin/lehome-entrypoint

FROM training-build AS training-runtime

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
    HOME=/home/trainer \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER trainer
WORKDIR /output
ENTRYPOINT ["/usr/local/bin/lehome-entrypoint"]
CMD ["--help"]
