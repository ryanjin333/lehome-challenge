#!/usr/bin/env bash
# Light Vast on-start: provision LeHome + GR00T from scratch on a CUDA base
# image. No fat registry image; everything hydrates from fast CDNs
# (GitHub, PyPI/NVIDIA pip index, Hugging Face).
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libglu1-mesa libgl1 libegl1 libxrandr2 libxinerama1 \
  libxcursor1 libxi6 libxext6 libx11-6 git curl ca-certificates
export __GLX_VENDOR_LIBRARY_NAME=nvidia
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

mkdir -p /workspace /opt/gr00t-src
cd /workspace || exit 1
git clone --depth 1 --branch ryanjin333/vast-template-sync https://github.com/ryanjin333/lehome-challenge.git lehome
cd lehome
uv sync --python 3.11
cd third_party && git clone --depth 1 https://github.com/lehome-official/IsaacLab.git && cd ..
source .venv/bin/activate
./third_party/IsaacLab/isaaclab.sh -i none
uv pip install -e ./source/lehome

# Pinned Isaac-GR00T inference runtime (same revision as the fat image).
export UV_PROJECT_ENVIRONMENT=/opt/gr00t-runtime
mkdir -p /opt/gr00t-src && cd /opt/gr00t-src
git init -q .
git remote add origin https://github.com/NVIDIA/Isaac-GR00T.git
git fetch --depth 1 origin 23ace64f17aa5015259b8609d371eb61a357c776
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach FETCH_HEAD
uv sync --frozen --no-dev --no-cache --python 3.10
uv pip install --python /opt/gr00t-runtime/bin/python --no-cache-dir msgpack pyzmq

cd /workspace/lehome
export LEHOME_ROOT=/workspace/lehome
export GROOT_PYTHON=/opt/gr00t-runtime/bin/python
export POLICY_SERVER_SCRIPT=/workspace/lehome/docker/rollout/groot_policy_server.py
exec bash docker/rollout/entrypoint.sh rollout
