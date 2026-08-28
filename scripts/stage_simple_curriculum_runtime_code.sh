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

ssh "${ssh_args[@]}" "$target" "mkdir -p '$remote_base' && test -d '$remote_base' && test ! -L '$remote_base'"
remote_stage="$(ssh "${ssh_args[@]}" "$target" "mktemp -d '$remote_base/.runtime-code-stage.XXXXXXXX'")"
[[ "$remote_stage" =~ ^/mnt/lehome/runtime-code/\.runtime-code-stage\.[A-Za-z0-9]{8,}$ ]] || { echo "remote staging directory is unsafe" >&2; exit 1; }
scp "${scp_args[@]}" "$bundle" "$target:$remote_stage/code.bundle"

ssh "${ssh_args[@]}" "$target" bash -s -- "$revision" "$bundle_sha256" "$remote_stage" <<'REMOTE_STAGE'
set -euo pipefail
revision="$1"; bundle_sha256="$2"; stage="$3"
base=/mnt/lehome/runtime-code; final="$base/$revision"; remote_bundle="$stage/code.bundle"
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$bundle_sha256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$stage" =~ ^/mnt/lehome/runtime-code/\.runtime-code-stage\.[A-Za-z0-9]{8,}$ ]] || exit 2
test -d "$base" && test ! -L "$base" && test -d "$stage" && test ! -L "$stage"
test -f "$remote_bundle" && test ! -L "$remote_bundle"
test "$(sha256sum "$remote_bundle" | awk '{print $1}')" = "$bundle_sha256" || { echo "remote bundle digest mismatch" >&2; exit 1; }
cleanup_remote() { rm -rf -- "$stage"; }; trap cleanup_remote EXIT INT TERM
verify_final() {
  local checkout="$1" required
  test -d "$checkout" && test ! -L "$checkout" && test -d "$checkout/.git" && test ! -L "$checkout/.git"
  test "$(git -C "$checkout" rev-parse HEAD)" = "$revision"; git -C "$checkout" diff --quiet; test -z "$(git -C "$checkout" status --porcelain)"
  for required in source/lehome trainer/src scripts rollout_appliance; do test -d "$checkout/$required" && test ! -L "$checkout/$required"; done
  test -f "$checkout/configs/eval_groot_n17_public_280.json" && test ! -L "$checkout/configs/eval_groot_n17_public_280.json"
}
if [ -e "$final" ]; then
  verify_final "$final" || { echo "runtime code final path already exists but is not exact" >&2; exit 1; }
  tree="$(git -C "$final" rev-parse HEAD^{tree})"
  printf '{"schema_version":1,"kind":"lehome_runtime_code_stage_v1","revision":"%s","path":"%s","bundle_sha256":"%s","tree":"%s","status":"existing_verified"}\n' "$revision" "$final" "$bundle_sha256" "$tree"
  exit 0
fi
git init -q "$stage/repository"; git -C "$stage/repository" bundle verify "$remote_bundle"; git -C "$stage/repository" fetch -q "$remote_bundle" "$revision"
git clone -q --no-checkout "$stage/repository" "$stage/checkout"; git -C "$stage/checkout" checkout --detach -q "$revision"; verify_final "$stage/checkout"
if mv -T "$stage/checkout" "$final"; then status=staged; else verify_final "$final" || { echo "runtime code final collision is not exact" >&2; exit 1; }; status=existing_verified; fi
verify_final "$final"; tree="$(git -C "$final" rev-parse HEAD^{tree})"
printf '{"schema_version":1,"kind":"lehome_runtime_code_stage_v1","revision":"%s","path":"%s","bundle_sha256":"%s","tree":"%s","status":"%s"}\n' "$revision" "$final" "$bundle_sha256" "$tree" "$status"
REMOTE_STAGE
