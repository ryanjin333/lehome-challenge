#!/usr/bin/env bash
# Stage a clean, immutable checkout on an already-running approved VM.
# No provider lifecycle operation or credential-file access is performed here.
set -euo pipefail

usage() { echo "usage: $0 --ssh-target USER@HOST [--ssh-port PORT]" >&2; exit 2; }
target=""; port="22"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --ssh-target) target="${2:-}"; shift 2 ;;
    --ssh-port) port="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$target" =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$ ]] || usage
[[ "$port" =~ ^[0-9]{1,5}$ ]] || usage

repo_root="$(git rev-parse --show-toplevel)"; cd "$repo_root"
revision="$(git rev-parse HEAD)"
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo "clean source has no exact HEAD" >&2; exit 2; }
git diff --quiet; git diff --cached --quiet
test -z "$(git status --porcelain)" || { echo "clean source has untracked or modified files" >&2; exit 2; }
for required in source/lehome trainer/src scripts rollout_appliance; do
  test -d "$required" && test ! -L "$required" || { echo "clean source directory is missing or unsafe: $required" >&2; exit 2; }
done
test -f configs/eval_groot_n17_public_280.json && test ! -L configs/eval_groot_n17_public_280.json || { echo "clean source config is missing or unsafe" >&2; exit 2; }

bundle_dir="$(mktemp -d "${TMPDIR:-/tmp}/lehome-runtime-code.XXXXXX")"; bundle="$bundle_dir/$revision.bundle"
remote_helper="$repo_root/scripts/stage_runtime_code_remote.py"
test -f "$remote_helper" && test ! -L "$remote_helper" || { echo "remote staging helper is missing or unsafe" >&2; exit 2; }
remote_stage=""; remote_base="/mnt/lehome/runtime-code"
ssh_args=(-o ClearAllForwardings=yes -o BatchMode=yes -p "$port")
scp_args=(-o ClearAllForwardings=yes -o BatchMode=yes -P "$port")
cleanup_remote_stage() {
  if [[ "$remote_stage" =~ ^/mnt/lehome/runtime-code/\.runtime-code-stage\.[A-Za-z0-9]{8,}$ ]]; then
    ssh "${ssh_args[@]}" "$target" "test -d '$remote_stage' && test ! -L '$remote_stage' && rm -rf -- '$remote_stage'" >/dev/null 2>&1 || true
  fi
}
cleanup_local() { cleanup_remote_stage; rm -rf "$bundle_dir"; }
trap cleanup_local EXIT INT TERM
# Give the bundle a named ref; a raw object ID can otherwise produce an empty
# bundle on a one-commit operator checkout. The remote still checks out only
# the separately verified exact revision.
git bundle create "$bundle" HEAD; bundle_sha256="$(sha256sum "$bundle" | awk '{print $1}')"

ssh "${ssh_args[@]}" "$target" "mountpoint -q /mnt/lehome && mkdir -p '$remote_base' && test -d '$remote_base' && test ! -L '$remote_base'"
remote_stage="$(ssh "${ssh_args[@]}" "$target" "mountpoint -q /mnt/lehome && mktemp -d '$remote_base/.runtime-code-stage.XXXXXXXX'")"
[[ "$remote_stage" =~ ^/mnt/lehome/runtime-code/\.runtime-code-stage\.[A-Za-z0-9]{8,}$ ]] || { echo "remote staging directory is unsafe" >&2; exit 1; }
scp "${scp_args[@]}" "$bundle" "$remote_helper" "$target:$remote_stage/"
ssh "${ssh_args[@]}" "$target" \
  "LEHOME_REVIEWED_REVISION='$revision' python3 '$remote_stage/stage_runtime_code_remote.py' --revision '$revision' --bundle-sha256 '$bundle_sha256' --stage '$remote_stage'"
