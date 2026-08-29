#!/usr/bin/env bash
# Canonical host-side Docker boundary for every native reference gate mode.
set -euo pipefail

readonly MODE="${1:-}"
readonly OUTPUT_MODE="${2:-}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly RUNTIME_IMAGE_ID="sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7"
readonly WORKSPACE_ROOT="/mnt/lehome"
readonly ASSET_SOURCE_ROOT="$WORKSPACE_ROOT/eval/assets"
readonly CANONICAL_ASSETS_ROOT="$WORKSPACE_ROOT/reference-native/assets"
readonly FLASH_ATTENTION_WHEEL="$WORKSPACE_ROOT/reference-native/dependencies/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
readonly FLASH_ATTENTION_WHEEL_SHA256="cd1a45ebfc1731a13e55ad68e0c9ad92390ddfffba306f9222be67c6d5a805af"
readonly DM_TREE_WHEEL="$WORKSPACE_ROOT/reference-native/dependencies/dm_tree-0.1.9-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
readonly DM_TREE_WHEEL_SHA256="294dc1cecf87552a45cdd5ddb215e7f5295a5a47c46f1f0a0463c3dd02a527d7"
readonly QWEN_VL_UTILS_WHEEL="$WORKSPACE_ROOT/reference-native/dependencies/qwen_vl_utils-0.0.14-py3-none-any.whl"
readonly QWEN_VL_UTILS_WHEEL_SHA256="5e28657bfd031e56bd447c5901b58ddfc3835285ed100f4c56580e0ade054e96"
readonly TORCHDIFFEQ_WHEEL="$WORKSPACE_ROOT/reference-native/dependencies/torchdiffeq-0.2.5-py3-none-any.whl"
readonly TORCHDIFFEQ_WHEEL_SHA256="aa1db4bed13bd04952f28a53cdf4336d1ab60417c1d9698d7a239fec1cf2bcf8"
readonly RUNTIME_REVISION="${LEHOME_NATIVE_REFERENCE_RUNTIME_REVISION:-}"
readonly RUNTIME_REPO_ROOT="$WORKSPACE_ROOT/runtime-code/$RUNTIME_REVISION"
readonly LAUNCHER="$RUNTIME_REPO_ROOT/rollout_appliance/run_native_reference_evaluator_gate.sh"
readonly RUNTIME_VERIFIER="$RUNTIME_REPO_ROOT/scripts/verify_native_reference_evaluator_gate.py"

fail() { printf 'error: %s\n' "$*" >&2; exit 2; }
[[ "$MODE" == source-stage || "$MODE" == inventory-cache || "$MODE" == validate-only || "$MODE" == execute ]] \
  || fail "mode must be source-stage, inventory-cache, validate-only, or execute"
[[ -z "$OUTPUT_MODE" || "$OUTPUT_MODE" == --print-command ]] \
  || fail "the only optional argument is --print-command"
[[ $# -le 2 ]] || fail "unexpected arguments"
[[ "$RUNTIME_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || fail "LEHOME_NATIVE_REFERENCE_RUNTIME_REVISION must be an exact 40-character lowercase Git revision"

declare -a command=(docker run --rm --pull never --gpus all --init --network host --shm-size=8g)
command+=(--mount "type=bind,src=$WORKSPACE_ROOT,dst=$WORKSPACE_ROOT")
command+=(--mount "type=bind,src=$RUNTIME_REPO_ROOT,dst=$RUNTIME_REPO_ROOT,readonly")
command+=(--mount "type=bind,src=$FLASH_ATTENTION_WHEEL,dst=$FLASH_ATTENTION_WHEEL,readonly")
command+=(--mount "type=bind,src=$DM_TREE_WHEEL,dst=$DM_TREE_WHEEL,readonly")
command+=(--mount "type=bind,src=$QWEN_VL_UTILS_WHEEL,dst=$QWEN_VL_UTILS_WHEEL,readonly")
command+=(--mount "type=bind,src=$TORCHDIFFEQ_WHEEL,dst=$TORCHDIFFEQ_WHEEL,readonly")
for root in objects robots scenes textures; do
  command+=(--mount "type=bind,src=$ASSET_SOURCE_ROOT/$root,dst=$CANONICAL_ASSETS_ROOT/$root,readonly")
  command+=(--mount "type=bind,src=$ASSET_SOURCE_ROOT/$root,dst=$RUNTIME_REPO_ROOT/Assets/$root,readonly")
done
command+=(--workdir "$RUNTIME_REPO_ROOT")

append_environment() {
  local name="$1" value="$2"
  command+=(--env "$name=$value")
}

append_required_environment() {
  local name="$1" value="${!1:-}"
  [[ -n "$value" ]] || fail "$name is required for $MODE"
  append_environment "$name" "$value"
}

append_environment LEHOME_NATIVE_REFERENCE_MODE "$MODE"
append_environment LEHOME_NATIVE_REFERENCE_VALIDATE_ONLY "$([[ "$MODE" == validate-only ]] && printf 1 || printf 0)"
append_environment LEHOME_NATIVE_REFERENCE_PYTHON /isaac-sim/python.sh
append_environment PYTHONEXE /opt/lehome-challenge/.venv/bin/python
append_environment LEHOME_NATIVE_REFERENCE_SOURCE_ROOT "$WORKSPACE_ROOT/reference-native/source"
append_environment LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT "$WORKSPACE_ROOT/cache/reference-theo-d384fe0/repo/pretrained_model"
append_environment LEHOME_NATIVE_REFERENCE_METADATA_ROOT "$WORKSPACE_ROOT/cache/reference-theo-d384fe0/repo/dataset_meta"
append_environment LEHOME_NATIVE_REFERENCE_ASSETS_ROOT "$CANONICAL_ASSETS_ROOT"
append_environment LEHOME_NATIVE_REFERENCE_VM_ID computeinstance-u00t6xfqhadrcmssa2
append_environment LEHOME_NATIVE_REFERENCE_DISK_ID computedisk-u00pbe55crxy7jr56x

case "$MODE" in
  source-stage)
    ;;
  inventory-cache)
    append_required_environment LEHOME_NATIVE_REFERENCE_CACHE_MANIFEST_OUTPUT
    append_required_environment LEHOME_NATIVE_REFERENCE_CACHE_MANIFEST_PATH
    ;;
  validate-only)
    append_required_environment LEHOME_NATIVE_REFERENCE_OUTPUT_ROOT
    append_required_environment LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_REVISION
    append_required_environment LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_PATH
    ;;
  execute)
    append_required_environment LEHOME_NATIVE_REFERENCE_OUTPUT_ROOT
    append_required_environment LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_REVISION
    append_required_environment LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_PATH
    append_required_environment LEHOME_NATIVE_REFERENCE_PROVIDER_RUNNING_RECEIPT
    append_required_environment LEHOME_NATIVE_REFERENCE_RUNTIME_IMAGE_RECEIPT
    ;;
esac

command+=(--entrypoint bash "$RUNTIME_IMAGE_ID" "$LAUNCHER")

if [[ "$OUTPUT_MODE" == --print-command ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

command -v docker >/dev/null 2>&1 || fail "docker is unavailable"
[[ -d "$WORKSPACE_ROOT" && ! -L "$WORKSPACE_ROOT" ]] || fail "workspace mount root is unavailable or unsafe"
[[ -f "$LAUNCHER" && ! -L "$LAUNCHER" ]] || fail "reviewed native reference launcher is unavailable or unsafe"
[[ -f "$RUNTIME_VERIFIER" && ! -L "$RUNTIME_VERIFIER" ]] || fail "reviewed native reference verifier is unavailable or unsafe"
[[ -f "$FLASH_ATTENTION_WHEEL" && ! -L "$FLASH_ATTENTION_WHEEL" ]] || fail "fixed FlashAttention wheel is unavailable or unsafe"
[[ "$(sha256sum -- "$FLASH_ATTENTION_WHEEL" | awk '{print $1}')" == "$FLASH_ATTENTION_WHEEL_SHA256" ]] \
  || fail "fixed FlashAttention wheel digest is invalid"
for wheel_and_digest in "$DM_TREE_WHEEL:$DM_TREE_WHEEL_SHA256" "$QWEN_VL_UTILS_WHEEL:$QWEN_VL_UTILS_WHEEL_SHA256" "$TORCHDIFFEQ_WHEEL:$TORCHDIFFEQ_WHEEL_SHA256"; do
  wheel="${wheel_and_digest%%:*}"; digest="${wheel_and_digest##*:}"
  [[ -f "$wheel" && ! -L "$wheel" ]] || fail "fixed public pyproject dependency wheel is unavailable or unsafe"
  [[ "$(sha256sum -- "$wheel" | awk '{print $1}')" == "$digest" ]] \
    || fail "fixed public pyproject dependency wheel digest is invalid"
done
python3 "$RUNTIME_VERIFIER" prepare-runtime-mountpoints --runtime-root "$RUNTIME_REPO_ROOT" >/dev/null \
  || fail "reviewed runtime asset mountpoints could not be prepared safely"
for root in objects robots scenes textures; do
  [[ -d "$ASSET_SOURCE_ROOT/$root" && ! -L "$ASSET_SOURCE_ROOT/$root" ]] \
    || fail "authenticated asset source is unavailable or unsafe: $root"
  [[ -d "$RUNTIME_REPO_ROOT/Assets/$root" && ! -L "$RUNTIME_REPO_ROOT/Assets/$root" ]] \
    || fail "reviewed runtime asset mountpoint is unavailable or unsafe: $root"
done

exec "${command[@]}"
