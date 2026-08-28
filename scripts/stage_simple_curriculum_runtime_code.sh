#!/usr/bin/env bash
# Stage this clean checkout on an already-running operator-approved VM.
# This helper never manages provider resources and never reads credentials.
set -euo pipefail

usage() { echo "usage: $0 --ssh-target USER@HOST [--ssh-port PORT] [--identity-file PATH]" >&2; exit 2; }
target=""; port="22"; identity_file=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --ssh-target) target="${2:-}"; shift 2 ;;
    --ssh-port) port="${2:-}"; shift 2 ;;
    --identity-file) identity_file="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
case "$target" in *[!A-Za-z0-9._@-]*|"" ) usage ;; esac
case "$port" in *[!0-9]*|"" ) usage ;; esac
if [ -n "$identity_file" ] && { [ ! -f "$identity_file" ] || [ -L "$identity_file" ]; }; then echo "identity file must be a regular file" >&2; exit 2; fi

repo_root="$(git rev-parse --show-toplevel)"; cd "$repo_root"
revision="$(git rev-parse HEAD)"
case "$revision" in *[!0-9a-f]*|?????????????????????????????????????????) ;; *) echo "clean source has no exact HEAD" >&2; exit 2 ;; esac
git diff --quiet; git diff --cached --quiet
test -z "$(git status --porcelain)" || { echo "clean source has untracked or modified files" >&2; exit 2; }
for required in source/lehome trainer/src scripts rollout_appliance configs/eval_groot_n17_public_280.json; do
  if [ ! -e "$required" ] || [ -L "$required" ]; then echo "clean source is missing required $required" >&2; exit 2; fi
done

bundle_dir="$(mktemp -d "${TMPDIR:-/tmp}/lehome-runtime-code.XXXXXX")"; bundle="$bundle_dir/$revision.bundle"
cleanup_local() { rm -rf "$bundle_dir"; }; trap cleanup_local EXIT INT TERM
git bundle create "$bundle" "$revision"; bundle_sha256="$(sha256sum "$bundle" | awk '{print $1}')"
ssh_args=(-o ClearAllForwardings=yes -o BatchMode=yes -o IdentitiesOnly=yes -p "$port")
scp_args=(-o ClearAllForwardings=yes -o BatchMode=yes -o IdentitiesOnly=yes -P "$port")
if [ -n "$identity_file" ]; then ssh_args=(-i "$identity_file" "${ssh_args[@]}"); scp_args=(-i "$identity_file" "${scp_args[@]}"); fi
remote_base="/mnt/lehome/runtime-code"; remote_bundle="$remote_base/.bundle-$revision-$$"
ssh "${ssh_args[@]}" "$target" "mkdir -p '$remote_base' && test ! -L '$remote_base'"
scp "${scp_args[@]}" "$bundle" "$target:$remote_bundle"

ssh "${ssh_args[@]}" "$target" bash -s -- "$revision" "$bundle_sha256" "$remote_bundle" <<'REMOTE_STAGE'
set -euo pipefail
revision="$1"; bundle_sha256="$2"; remote_bundle="$3"
base=/mnt/lehome/runtime-code; final="$base/$revision"
case "$revision" in *[!0-9a-f]*|?????????????????????????????????????????) exit 2 ;; esac
case "$bundle_sha256" in *[!0-9a-f]*|????????????????????????????????????????????????????????????????) exit 2 ;; esac
test -d "$base" && test ! -L "$base"; test -f "$remote_bundle" && test ! -L "$remote_bundle"
test "$(sha256sum "$remote_bundle" | awk '{print $1}')" = "$bundle_sha256" || { echo "remote bundle digest mismatch" >&2; exit 1; }
verify_final() {
  local checkout="$1"
  test -d "$checkout" && test ! -L "$checkout" && test -d "$checkout/.git" && test ! -L "$checkout/.git"
  test "$(git -C "$checkout" rev-parse HEAD)" = "$revision"; git -C "$checkout" diff --quiet; test -z "$(git -C "$checkout" status --porcelain)"
  for required in source/lehome trainer/src scripts rollout_appliance configs/eval_groot_n17_public_280.json; do test -e "$checkout/$required" && test ! -L "$checkout/$required"; done
}
if [ -e "$final" ]; then
  rm -f "$remote_bundle"; verify_final "$final" || { echo "runtime code final path already exists but is not exact" >&2; exit 1; }
  tree="$(git -C "$final" rev-parse HEAD^{tree})"
  printf '{"schema_version":1,"kind":"lehome_runtime_code_stage_v1","revision":"%s","path":"%s","bundle_sha256":"%s","tree":"%s","status":"existing_verified"}\n' "$revision" "$final" "$bundle_sha256" "$tree"
  exit 0
fi
tmp="$(mktemp -d "$base/.stage-$revision.XXXXXX")"
cleanup_remote() { rm -rf "$tmp"; rm -f "$remote_bundle"; }; trap cleanup_remote EXIT INT TERM
git init -q "$tmp/repository"; git -C "$tmp/repository" bundle verify "$remote_bundle"; git -C "$tmp/repository" fetch -q "$remote_bundle" "$revision"
git clone -q --no-checkout "$tmp/repository" "$tmp/checkout"; git -C "$tmp/checkout" checkout --detach -q "$revision"; verify_final "$tmp/checkout"
test ! -e "$final"; mv "$tmp/checkout" "$final"; verify_final "$final"; tree="$(git -C "$final" rev-parse HEAD^{tree})"
printf '{"schema_version":1,"kind":"lehome_runtime_code_stage_v1","revision":"%s","path":"%s","bundle_sha256":"%s","tree":"%s","status":"staged"}\n' "$revision" "$final" "$bundle_sha256" "$tree"
REMOTE_STAGE
