#!/usr/bin/env bash
# Canonical host-side Docker boundary for every native reference gate mode.
set -euo pipefail

readonly MODE="${1:-}"
readonly OUTPUT_MODE="${2:-}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly RUNTIME_REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly LAUNCHER="$SCRIPT_DIR/run_native_reference_evaluator_gate.sh"
readonly RUNTIME_IMAGE_ID="sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7"
readonly WORKSPACE_ROOT="/mnt/lehome"
readonly ASSET_SOURCE_ROOT="$WORKSPACE_ROOT/eval/assets"
readonly CANONICAL_ASSETS_ROOT="$WORKSPACE_ROOT/reference-native/assets"

fail() { printf 'error: %s\n' "$*" >&2; exit 2; }
[[ "$MODE" == source-stage || "$MODE" == inventory-cache || "$MODE" == validate-only || "$MODE" == execute ]] \
  || fail "mode must be source-stage, inventory-cache, validate-only, or execute"
[[ -z "$OUTPUT_MODE" || "$OUTPUT_MODE" == --print-command ]] \
  || fail "the only optional argument is --print-command"
[[ $# -le 2 ]] || fail "unexpected arguments"

declare -a command=(docker run --rm --pull never --gpus all --init --network host --shm-size=8g)
command+=(--mount "type=bind,src=$WORKSPACE_ROOT,dst=$WORKSPACE_ROOT")
command+=(--mount "type=bind,src=$RUNTIME_REPO_ROOT,dst=$RUNTIME_REPO_ROOT,readonly")
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
append_environment LEHOME_NATIVE_REFERENCE_PYTHON /opt/lehome-challenge/.venv/bin/python
append_environment LEHOME_NATIVE_REFERENCE_SOURCE_ROOT "$WORKSPACE_ROOT/reference-native/source"
append_environment LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT "$WORKSPACE_ROOT/reference-native/pretrained_model"
append_environment LEHOME_NATIVE_REFERENCE_METADATA_ROOT "$WORKSPACE_ROOT/reference-native/dataset_meta"
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
for root in objects robots scenes textures; do
  [[ -d "$ASSET_SOURCE_ROOT/$root" && ! -L "$ASSET_SOURCE_ROOT/$root" ]] \
    || fail "authenticated asset source is unavailable or unsafe: $root"
  [[ -d "$RUNTIME_REPO_ROOT/Assets/$root" && ! -L "$RUNTIME_REPO_ROOT/Assets/$root" ]] \
    || fail "reviewed runtime asset mountpoint is unavailable or unsafe: $root"
done

exec "${command[@]}"
