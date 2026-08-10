#!/usr/bin/env bash
set -euo pipefail

case " $* " in
  *" hf auth login "*|*" huggingface-cli login "*)
    echo "interactive Hugging Face login is forbidden" >&2
    exit 64
    ;;
esac

if [[ "${AUTO_DESTROY:-}" != "0" ]]; then
  echo "AUTO_DESTROY must be exactly 0" >&2
  exit 64
fi
if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]]; then
  echo "OMNI_KIT_ACCEPT_EULA must be YES" >&2
  exit 64
fi
if [[ "${B1K_ACCEPT_DATASET_TOS:-}" != "YES" ]]; then echo "B1K_ACCEPT_DATASET_TOS must be YES" >&2; exit 64; fi
: "${OMNIGIBSON_DATA_PATH:?OMNIGIBSON_DATA_PATH is required}"
if [[ "$(id -u)" == 0 ]]; then
  : "${HF_TOKEN:?HF_TOKEN is required only for root bootstrap}"
  if [[ "${B1K_HF_TOKEN_FILE:-}" != "/workspace/.cache/huggingface/token" ]]; then
    echo "B1K_HF_TOKEN_FILE must use the production token path" >&2
    exit 64
  fi
  install -d -o 10001 -g 10001 -m 0700 \
    /workspace /workspace/campaign /workspace/checkpoint-source \
    /workspace/omnigibson-data /workspace/smoke-canary \
    /workspace/campaign/.cache/numba /workspace/campaign/.cache/triton \
    /workspace/campaign/.cache/matplotlib
  printf '%s' "$HF_TOKEN" | "$BEHAVIOR_PYTHON" -m b1k_rollout.token_bootstrap
  unset HF_TOKEN
  export NUMBA_CACHE_DIR=/workspace/campaign/.cache/numba
  export TRITON_CACHE_DIR=/workspace/campaign/.cache/triton
  export MPLCONFIGDIR=/workspace/campaign/.cache/matplotlib
  exec setpriv --reuid=10001 --regid=10001 --init-groups "$0" "$@"
fi
if [[ ! -f "${B1K_HF_TOKEN_FILE:-}" || -L "${B1K_HF_TOKEN_FILE:-}" ]]; then
  echo "B1K_HF_TOKEN_FILE must be a regular file" >&2
  exit 64
fi
token_mode=$(stat -c '%a' "$B1K_HF_TOKEN_FILE" 2>/dev/null || stat -f '%Lp' "$B1K_HF_TOKEN_FILE")
if [[ "${token_mode:1:1}" != "0" || "${token_mode:2:1}" != "0" ]]; then
  echo "B1K_HF_TOKEN_FILE must not be readable by group or other" >&2
  exit 64
fi

# Account-level secret material is consumed only from the mounted file.  Never
# pass an environment token into child processes or command-line arguments.
unset HF_TOKEN
"${BEHAVIOR_PYTHON:-/opt/conda/envs/behavior/bin/python}" -m b1k_rollout.cli assets-bootstrap
if [[ "${B1K_ROLLOUT_SMOKE_RUNTIME:-0}" == "1" ]]; then
  umask 077
  : > /workspace/smoke-canary/rollout-ready
  exec /bin/sleep infinity
fi
if [[ "${1:-}" == "smoke-runtime" ]]; then
  exec "${BEHAVIOR_PYTHON:-/opt/conda/envs/behavior/bin/python}" -m b1k_rollout.cli "$@"
fi
if [[ "${B1K_ROLLOUT_VERIFY_PRIVILEGE_DROP:-}" == "1" ]]; then
  test "$(id -u)" = 10001
  test "$(id -g)" = 10001
  test -O "$B1K_HF_TOKEN_FILE"
  test -z "${HF_TOKEN+x}"
  exit 0
fi
"${BEHAVIOR_PYTHON:-/opt/conda/envs/behavior/bin/python}" -m b1k_rollout.cli checkpoint-bootstrap
"${BEHAVIOR_PYTHON:-/opt/conda/envs/behavior/bin/python}" -m b1k_rollout.cli preflight
exec "${BEHAVIOR_PYTHON:-/opt/conda/envs/behavior/bin/python}" -m b1k_rollout.cli campaign "$@"
