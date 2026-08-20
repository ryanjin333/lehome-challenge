#!/usr/bin/env bash
# Run one immutable LeHome runtime-mixture training transaction on Nebius.
#
# This guest controller deliberately knows no model/data hyperparameters.  It
# only accepts a complete immutable binding from /etc/lehome/runtime.env,
# verifies it, and invokes the checked runtime CLI in the fixed admission
# order.  All durable state remains on /mnt/lehome; this script never cleans
# or rewrites shared data.
set -euo pipefail
umask 077

RUNTIME_ENV="${LEHOME_RUNTIME_ENV:-/etc/lehome/runtime.env}"
if [[ -f "${RUNTIME_ENV}" ]]; then
  # The file is provisioned by cloud-init or the approved secret injector.
  # shellcheck disable=SC1090
  source "${RUNTIME_ENV}"
fi

ROLE="${LEHOME_ROLE:?LEHOME_ROLE is required}"
RUN_ID="${LEHOME_RUN_ID:?LEHOME_RUN_ID is required}"
if [[ "${ROLE}" != "training" ]]; then
  echo "lehome training controller refuses non-training role" >&2
  exit 64
fi

# A generic base image has no experiment binding.  Staying inert here lets it
# be safely booted for image/driver inspection without accidental training.
if [[ -z "${LEHOME_EXPERIMENT_MANIFEST:-}" ]]; then
  echo "lehome training controller inert: immutable experiment manifest is absent" >&2
  exit 0
fi

WORKSPACE="${LEHOME_WORKSPACE_MOUNT:-/mnt/lehome}"
PID_FILE="${LEHOME_TRAINING_PID_FILE:-/run/lehome-training.pid}"
CODE_ROOT="${LEHOME_CODE_ROOT:-/var/lib/lehome/training-code}"
PREPARED_HOST="${LEHOME_PREPARED_ROOT:-}"
CACHE_HOST="${LEHOME_CACHE_ROOT:-}"
OUTPUT_HOST="${LEHOME_OUTPUT_ROOT:-}"
lease_owned=1
child_pid=""

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing required runtime binding: ${name}" >&2
    exit 64
  fi
}

require_regular_file() {
  local path="$1" label="$2"
  if [[ -L "${path}" || ! -f "${path}" ]]; then
    echo "${label} must be a regular non-symlink file" >&2
    exit 64
  fi
}

require_workspace_file() {
  local path="$1" label="$2" resolved workspace_resolved
  require_regular_file "${path}" "${label}"
  resolved="$(realpath -e -- "${path}")"
  workspace_resolved="$(realpath -e -- "${WORKSPACE}")"
  case "${resolved}" in
    "${workspace_resolved}"/*) ;;
    *)
      echo "${label} must live beneath the mounted shared workspace" >&2
      exit 64
      ;;
  esac
}

require_workspace_directory() {
  local path="$1" label="$2" resolved workspace_resolved
  if [[ -L "${path}" || ! -d "${path}" ]]; then
    echo "${label} must be an existing non-symlink directory" >&2
    exit 64
  fi
  resolved="$(realpath -e -- "${path}")"
  workspace_resolved="$(realpath -e -- "${WORKSPACE}")"
  case "${resolved}" in
    "${workspace_resolved}"|"${workspace_resolved}"/*) ;;
    *)
      echo "${label} must live beneath the mounted shared workspace" >&2
      exit 64
      ;;
  esac
}

require_sha256() {
  local path="$1" expected="$2" label="$3" actual
  if [[ ! "${expected}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "${label} hash must be lowercase SHA-256" >&2
    exit 64
  fi
  actual="$(sha256sum -- "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "${label} SHA-256 mismatch" >&2
    exit 64
  fi
}

release_lease() {
  if [[ "${lease_owned}" -eq 1 ]]; then
    lease_owned=0
    /opt/lehome/guest/bin/lehome-workspace.sh --release || \
      echo "warning: training role release failed; preserving workspace for recovery" >&2
  fi
}

on_signal() {
  echo "lehome training controller received SIGTERM; requesting orderly trainer stop" >&2
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill -TERM "${child_pid}" || true
    wait "${child_pid}" || true
  fi
  exit 143
}

cleanup() {
  rm -f -- "${PID_FILE}"
  release_lease
}

trap on_signal TERM INT
trap cleanup EXIT

for required in \
  LEHOME_EXPERIMENT_MANIFEST LEHOME_EXPERIMENT_MANIFEST_SHA256 \
  LEHOME_CODE_BUNDLE LEHOME_CODE_BUNDLE_SHA256 LEHOME_CODE_REVISION \
  LEHOME_HF_TOKEN_FILE LEHOME_RUNTIME_HYDRATE_REQUEST \
  LEHOME_RUNTIME_HYDRATE_REQUEST_SHA256 LEHOME_RUNTIME_PILOT_REQUEST \
  LEHOME_RUNTIME_PILOT_REQUEST_SHA256 LEHOME_RUNTIME_WARMUP_REQUEST \
  LEHOME_RUNTIME_WARMUP_REQUEST_SHA256 LEHOME_RUNTIME_TRAIN_REQUEST \
  LEHOME_RUNTIME_TRAIN_REQUEST_SHA256; do
  require_env "${required}"
done
for required in LEHOME_PREPARED_ROOT LEHOME_CACHE_ROOT LEHOME_OUTPUT_ROOT; do
  require_env "${required}"
done
require_workspace_directory "${PREPARED_HOST}" "runtime prepared directory"
require_workspace_directory "${CACHE_HOST}" "runtime cache directory"
require_workspace_directory "${OUTPUT_HOST}" "runtime output directory"
prepared_resolved="$(realpath -e -- "${PREPARED_HOST}")"
cache_resolved="$(realpath -e -- "${CACHE_HOST}")"
output_resolved="$(realpath -e -- "${OUTPUT_HOST}")"
if [[ "${prepared_resolved}" == "${cache_resolved}" || \
      "${prepared_resolved}" == "${output_resolved}" || \
      "${cache_resolved}" == "${output_resolved}" ]]; then
  echo "runtime prepared, cache, and output roots must be distinct" >&2
  exit 64
fi

if [[ ! "${LEHOME_CODE_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LEHOME_CODE_REVISION must be an immutable 40-character Git revision" >&2
  exit 64
fi

for binding in \
  "${LEHOME_EXPERIMENT_MANIFEST}:LEHOME_EXPERIMENT_MANIFEST_SHA256:experiment manifest" \
  "${LEHOME_CODE_BUNDLE}:LEHOME_CODE_BUNDLE_SHA256:reviewed code bundle" \
  "${LEHOME_RUNTIME_HYDRATE_REQUEST}:LEHOME_RUNTIME_HYDRATE_REQUEST_SHA256:hydrate request" \
  "${LEHOME_RUNTIME_PILOT_REQUEST}:LEHOME_RUNTIME_PILOT_REQUEST_SHA256:loader pilot request" \
  "${LEHOME_RUNTIME_WARMUP_REQUEST}:LEHOME_RUNTIME_WARMUP_REQUEST_SHA256:GPU warmup request" \
  "${LEHOME_RUNTIME_TRAIN_REQUEST}:LEHOME_RUNTIME_TRAIN_REQUEST_SHA256:training request"; do
  IFS=: read -r path hash_name label <<<"${binding}"
  require_workspace_file "${path}" "${label}"
  require_sha256 "${path}" "${!hash_name}" "${label}"
done

# A promoted sweep uses locally generated, canonical request bytes.  The
# compatibility-bound request set still provides hydrate/pilot/warmup; only
# the final train envelope/config/experiment vary by immutable job.  Do not
# accept a partial overlay: an absent overlay retains the legacy 2K flow.
TRAIN_REQUEST="${LEHOME_RUNTIME_TRAIN_REQUEST}"
if [[ -n "${LEHOME_SWEEP_TRAIN_OVERLAY:-}" || -n "${LEHOME_SWEEP_TRAIN_REQUEST:-}" ]]; then
  require_env LEHOME_SWEEP_TRAIN_OVERLAY
  require_env LEHOME_SWEEP_TRAIN_OVERLAY_SHA256
  require_env LEHOME_SWEEP_TRAIN_REQUEST
  require_env LEHOME_SWEEP_TRAIN_REQUEST_SHA256
  require_env LEHOME_SWEEP_RUNTIME_BINDING
  require_env LEHOME_SWEEP_RUNTIME_BINDING_SHA256
  require_workspace_file "${LEHOME_SWEEP_TRAIN_OVERLAY}" "sweep train overlay"
  require_sha256 "${LEHOME_SWEEP_TRAIN_OVERLAY}" "${LEHOME_SWEEP_TRAIN_OVERLAY_SHA256}" "sweep train overlay"
  require_workspace_file "${LEHOME_SWEEP_TRAIN_REQUEST}" "sweep train request"
  require_sha256 "${LEHOME_SWEEP_TRAIN_REQUEST}" "${LEHOME_SWEEP_TRAIN_REQUEST_SHA256}" "sweep train request"
  require_workspace_file "${LEHOME_SWEEP_RUNTIME_BINDING}" "sweep runtime binding"
  require_sha256 "${LEHOME_SWEEP_RUNTIME_BINDING}" "${LEHOME_SWEEP_RUNTIME_BINDING_SHA256}" "sweep runtime binding"
  parent_resume_values=(
    "${LEHOME_SWEEP_PARENT_ARCHIVE:-}"
    "${LEHOME_SWEEP_PARENT_DESCRIPTOR:-}"
    "${LEHOME_SWEEP_PARENT_CHECKPOINT:-}"
  )
  parent_resume_count=0
  for parent_resume_value in "${parent_resume_values[@]}"; do
    if [[ -n "${parent_resume_value}" ]]; then
      parent_resume_count=$((parent_resume_count + 1))
    fi
  done
  if [[ "${parent_resume_count}" -ne 0 && "${parent_resume_count}" -ne 3 ]]; then
    echo "promoted sweep resume bindings must be complete" >&2
    exit 64
  fi
  if [[ "${parent_resume_count}" -eq 3 ]]; then
    require_workspace_file "${LEHOME_SWEEP_PARENT_ARCHIVE}" "sweep promoted parent archive"
    require_workspace_file "${LEHOME_SWEEP_PARENT_DESCRIPTOR}" "sweep promoted parent descriptor"
    require_workspace_directory "${LEHOME_SWEEP_PARENT_CHECKPOINT}" "sweep promoted parent checkpoint"
  fi
  TRAIN_REQUEST="${LEHOME_SWEEP_TRAIN_REQUEST}"
fi

request_container_path() {
  local request="$1" resolved prepared_resolved relative
  resolved="$(realpath -e -- "${request}")"
  prepared_resolved="$(realpath -e -- "${PREPARED_HOST}")"
  case "${resolved}" in
    "${prepared_resolved}"/*)
      relative="${resolved#"${prepared_resolved}"/}"
      printf '/prepared/%s\n' "${relative}"
      ;;
    *)
      echo "runtime request must live beneath the prepared mount" >&2
      exit 64
      ;;
  esac
}

cache_container_path() {
  local path="$1" resolved cache_resolved relative
  resolved="$(realpath -e -- "${path}")"
  cache_resolved="$(realpath -e -- "${CACHE_HOST}")"
  case "${resolved}" in
    "${cache_resolved}"/*)
      relative="${resolved#"${cache_resolved}"/}"
      printf '/cache/%s\n' "${relative}"
      ;;
    *)
      echo "sweep promoted parent must live beneath the cache mount" >&2
      exit 64
      ;;
  esac
}

require_regular_file "${LEHOME_HF_TOKEN_FILE}" "Hugging Face token file"
if [[ "$(wc -l < "${LEHOME_HF_TOKEN_FILE}")" -gt 1 ]]; then
  echo "Hugging Face token file must contain exactly one line" >&2
  exit 64
fi
IFS= read -r HUGGING_FACE_HUB_TOKEN < "${LEHOME_HF_TOKEN_FILE}" || true
export HUGGING_FACE_HUB_TOKEN
if [[ ! "${HUGGING_FACE_HUB_TOKEN}" =~ ^hf_[A-Za-z0-9]{20,}$ ]]; then
  echo "Hugging Face token file does not contain a scoped token" >&2
  exit 64
fi

# The code bundle is immutable and full: clone it into the disposable boot
# disk, prove its requested revision, and mount only trainer/src read-only.
# ``git bundle verify`` itself requires repository context, so verification
# must happen inside the checkout created from the bundle.
bundle_root="${CODE_ROOT}/${LEHOME_CODE_BUNDLE_SHA256}"
checkout="${bundle_root}/checkout"
if [[ -e "${bundle_root}" && ! -d "${checkout}" ]]; then
  echo "code bundle cache has an unsafe or incomplete checkout" >&2
  exit 64
fi
if [[ ! -d "${checkout}" ]]; then
  install -d -m 0700 "${bundle_root}"
  git clone --no-checkout "${LEHOME_CODE_BUNDLE}" "${checkout}"
fi
git -C "${checkout}" bundle verify "${LEHOME_CODE_BUNDLE}" >/dev/null
git -C "${checkout}" checkout --detach "${LEHOME_CODE_REVISION}"
if [[ "$(git -C "${checkout}" rev-parse HEAD)" != "${LEHOME_CODE_REVISION}" ]] || \
   ! git -C "${checkout}" diff --quiet --ignore-submodules --; then
  echo "reviewed code bundle does not resolve to the required clean revision" >&2
  exit 64
fi
require_regular_file "${checkout}/trainer/src/lehome_train/cli.py" "trainer code entrypoint"

TRAINING_IMAGE_MANIFEST="${LEHOME_TRAINING_IMAGE_MANIFEST:-/etc/lehome/training-image-manifest.json}"
image_identity=()
while IFS= read -r image_identity_line; do
  image_identity+=("${image_identity_line}")
done < <(/usr/bin/python3 - "${TRAINING_IMAGE_MANIFEST}" <<'PY'
import json
import re
import sys
from pathlib import Path
path = Path(sys.argv[1])
if path.is_symlink() or not path.is_file():
    raise SystemExit('missing training image manifest')
value = json.loads(path.read_text(encoding='utf-8'))
image = value.get('oci_image')
digest = value.get('oci_digest')
if not isinstance(image, str) or not isinstance(digest, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
    raise SystemExit('invalid training image manifest')
print(image)
print(digest)
PY
)
if [[ "${#image_identity[@]}" -ne 2 ]]; then
  echo "invalid pinned training image identity" >&2
  exit 64
fi
TRAINER_IMAGE="${image_identity[0]}"
TRAINER_DIGEST="${image_identity[1]}"
docker image inspect "${TRAINER_IMAGE}" >/dev/null
if ! docker image inspect --format '{{join .RepoDigests "\\n"}}' "${TRAINER_IMAGE}" | grep -Fqx "${TRAINER_IMAGE%@*}@${TRAINER_DIGEST}"; then
  echo "pinned trainer image digest is unavailable locally" >&2
  exit 64
fi

run_phase() {
  local phase="$1" request="$2" container_request
  container_request="$(request_container_path "${request}")"
  echo "lehome training phase: ${phase}" >&2
  local sweep_env=()
  if [[ -n "${LEHOME_SWEEP_TRAIN_OVERLAY:-}" ]]; then
    sweep_env=(
      --env "LEHOME_SWEEP_TRAIN_OVERLAY=$(request_container_path "${LEHOME_SWEEP_TRAIN_OVERLAY}")"
      --env "LEHOME_SWEEP_TRAIN_OVERLAY_SHA256=${LEHOME_SWEEP_TRAIN_OVERLAY_SHA256}"
      --env "LEHOME_SWEEP_RUNTIME_BINDING=$(request_container_path "${LEHOME_SWEEP_RUNTIME_BINDING}")"
      --env "LEHOME_SWEEP_RUNTIME_BINDING_SHA256=${LEHOME_SWEEP_RUNTIME_BINDING_SHA256}"
    )
    if [[ -n "${LEHOME_SWEEP_PARENT_ARCHIVE:-}" ]]; then
      sweep_env+=(
        --env "LEHOME_SWEEP_PARENT_ARCHIVE=$(cache_container_path "${LEHOME_SWEEP_PARENT_ARCHIVE}")"
        --env "LEHOME_SWEEP_PARENT_DESCRIPTOR=$(cache_container_path "${LEHOME_SWEEP_PARENT_DESCRIPTOR}")"
        --env "LEHOME_SWEEP_PARENT_CHECKPOINT=$(cache_container_path "${LEHOME_SWEEP_PARENT_CHECKPOINT}")"
      )
    fi
  fi
  local docker_args=(
    run --rm --gpus all --network host --shm-size=16g
    --mount "type=bind,src=${PREPARED_HOST},dst=/prepared"
    --mount "type=bind,src=${CACHE_HOST},dst=/cache"
    --mount "type=bind,src=${OUTPUT_HOST},dst=/output"
    --mount "type=bind,src=${checkout}/trainer/src,dst=/opt/lehome/trainer/src,readonly"
    --env HUGGING_FACE_HUB_TOKEN --env HF_HOME=/cache/huggingface
  )
  # Bash nounset treats an empty array expansion as unset on some supported
  # guest images.  Append only when there are actual sweep bindings.
  if [[ "${#sweep_env[@]}" -gt 0 ]]; then
    docker_args+=("${sweep_env[@]}")
  fi
  docker_args+=(
    --env LEHOME_TRAIN_RUNTIME_FACTORY=lehome_train.groot.production_runtime:create
    --env PYTHONPATH=/opt/lehome/trainer/src --entrypoint lehome-train
    "${TRAINER_IMAGE}" "${phase}" --request "${container_request}"
  )
  docker "${docker_args[@]}" &
  child_pid="$!"
  printf '%s\n' "${child_pid}" > "${PID_FILE}"
  wait "${child_pid}"
  child_pid=""
  rm -f -- "${PID_FILE}"
}

run_phase hydrate-runtime-mixture "${LEHOME_RUNTIME_HYDRATE_REQUEST}"
run_phase pilot-runtime-mixture "${LEHOME_RUNTIME_PILOT_REQUEST}"
run_phase runtime-gpu-warmup "${LEHOME_RUNTIME_WARMUP_REQUEST}"
run_phase runtime-mixture-train "${TRAIN_REQUEST}"
