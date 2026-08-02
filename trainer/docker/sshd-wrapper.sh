#!/bin/bash
set -euo pipefail

readonly real_sshd=/usr/sbin/sshd.distrib
readonly key_dir="${LEHOME_SSH_HOST_KEY_DIR:-/workspace/.cache/lehome-ssh-hostkeys}"

test -x "$real_sshd"
install -d -o root -g root -m 0755 /run/sshd

validation_only=false
for argument in "$@"; do
  case "$argument" in
    -t|-T)
      validation_only=true
      ;;
  esac
done

if "$validation_only"; then
  # Debian's package maintainer scripts validate sshd while Vast derives its
  # managed-SSH image. Temporary build-time keys let that validation complete;
  # they are never used by a running instance.
  ssh-keygen -A
else
  # Persist a unique host identity inside this instance's workspace. The
  # public base image deliberately contains no reusable SSH private keys.
  umask 077
  install -d -o root -g root -m 0700 "$key_dir"
  for key_type in rsa ecdsa ed25519; do
    private_key="$key_dir/ssh_host_${key_type}_key"
    if [[ ! -s "$private_key" ]]; then
      ssh-keygen -q -t "$key_type" -N '' -f "$private_key"
    fi
    install -o root -g root -m 0600 "$private_key" "/etc/ssh/ssh_host_${key_type}_key"
    install -o root -g root -m 0644 "$private_key.pub" "/etc/ssh/ssh_host_${key_type}_key.pub"
  done
fi

exec "$real_sshd" "$@"
