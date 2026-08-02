#!/usr/bin/env bash
set -euo pipefail

drop_privileges=()
if [[ "$(id -u)" -eq 0 ]]; then
  drop_privileges=(
    /usr/bin/setpriv
    --reuid=10001
    --regid=10001
    --init-groups
    --no-new-privs
  )
fi

case " $* " in
  *" hf auth login "*|*" huggingface-cli login "*)
    echo "interactive Hugging Face login is forbidden; pass HF_TOKEN only to a remote command" >&2
    exit 64
    ;;
esac

for path in /cache /prepared /output; do
  if [[ ! -d "$path" ]] || ! mountpoint -q "$path"; then
    echo "$path must be an explicit container mount" >&2
    exit 64
  fi
  probe="$path/.lehome-write-test-$$"
  if ! "${drop_privileges[@]}" /bin/bash -euo pipefail -c '
    umask 077
    : >"$1"
    rm -f "$1"
  ' _ "$probe"; then
    echo "$path must be writable by the non-root trainer user" >&2
    exit 73
  fi
done

"${drop_privileges[@]}" mkdir -p \
  /cache/tmp /cache/xdg /cache/torch /cache/huggingface/hub /output/wandb

if [[ $# -eq 0 ]]; then
  set -- --help
fi
if [[ "$1" != "lehome-train" ]]; then
  set -- lehome-train "$@"
fi

remote=false
case "${2:-}:${3:-}" in
  prepare:*|train:*|restore:*|sync:*|data:publish|data:retrieve|model:retrieve)
    remote=true
    ;;
esac

if [[ "$remote" == true ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "this remote command requires HF_TOKEN in the current process environment" >&2
    exit 64
  fi
else
  unset HF_TOKEN
fi

exec "${drop_privileges[@]}" /usr/bin/env HOME=/nonexistent "$@"
