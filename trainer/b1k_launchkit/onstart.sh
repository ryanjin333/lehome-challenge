#!/usr/bin/env bash
# Root-only bootstrap: persist the inherited account token once, then drop it.
set -euo pipefail

if [[ "$(id -u)" != 0 ]]; then echo "onstart must run as root" >&2; exit 64; fi
if [[ "${AUTO_DESTROY:-0}" != 0 ]]; then echo "AUTO_DESTROY must be 0" >&2; exit 64; fi
if [[ -z "${HF_TOKEN:-}" ]]; then echo "missing inherited HF_TOKEN" >&2; exit 64; fi
if [[ ! -x /opt/b1k-bucket-helper/bin/b1k-bucket-helper ]]; then echo "missing executable b1k bucket helper" >&2; exit 64; fi

install -d -o 10001 -g 10001 -m 0750 \
  /workspace/data /workspace/models /workspace/checkpoints /workspace/logs /workspace/logs/wandb /workspace/final \
  "/workspace/outputs/${RUN_ID:?RUN_ID is required}" /workspace/smoke-canary /workspace/outputs/b1k-smoke-canary
printf '%s' "$HF_TOKEN" | /opt/runtime/bin/python /opt/b1k-launchkit/token_bootstrap.py
unset HF_TOKEN
export B1K_HF_TOKEN_FILE=/workspace/.cache/huggingface/token
export B1K_LIFECYCLE_ADAPTER=lehome_train.b1k.production:build_production_controller
export WANDB_DIR=/workspace/logs/wandb

if [[ "${B1K_TRAINING_SMOKE_RUNTIME:-0}" == "1" ]]; then
  exec setpriv --reuid=10001 --regid=10001 --init-groups /bin/bash -c 'umask 077; : > /workspace/smoke-canary/training-ready; exec /bin/sleep infinity'
fi

exec setpriv --reuid=10001 --regid=10001 --init-groups \
  /bin/bash -c 'export B1K_HF_TOKEN_FILE=/workspace/.cache/huggingface/token B1K_LIFECYCLE_ADAPTER=lehome_train.b1k.production:build_production_controller; exec /opt/runtime/bin/python -m lehome_train.b1k.lifecycle >> /workspace/logs/controller.log 2>&1'
