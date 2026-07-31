#!/usr/bin/env bash
set -euo pipefail

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
  if ! (umask 077 && : >"$probe" && rm -f "$probe"); then
    echo "$path must be writable by the non-root trainer user" >&2
    exit 73
  fi
done

mkdir -p /cache/tmp /cache/xdg /cache/torch /cache/huggingface/hub /output/wandb

if [[ $# -eq 0 ]]; then
  set -- --help
fi
if [[ "$1" != "lehome-train" ]]; then
  set -- lehome-train "$@"
fi

remote=false
if [[ "${2:-}" == "sync" ]]; then
  remote=true
elif [[ "${2:-}" == "data" && "${3:-}" == "publish" ]]; then
  remote=true
fi

if [[ "$remote" == true ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "this remote command requires HF_TOKEN in the current process environment" >&2
    exit 64
  fi
else
  unset HF_TOKEN
fi

exec "$@"
