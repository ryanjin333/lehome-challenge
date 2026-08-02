#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

export PATH=/opt/runtime/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=/opt/isaac-groot
export LEHOME_GROOT_ROOT=/opt/isaac-groot
export LEHOME_TRAIN_RUNTIME_FACTORY=lehome_train.groot.production_runtime:create
export HF_HOME=/cache/huggingface
export HF_HUB_CACHE=/cache/huggingface/hub
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export XDG_CACHE_HOME=/cache/xdg
export TORCH_HOME=/cache/torch
export TMPDIR=/cache/tmp

MODEL_REPOSITORY=ryanjin333/lehome-groot-n17-models
DATA_REPOSITORY=ryanjin333/lehome-groot-n17-data

load_vast_runtime_variable() {
  local variable_name=$1
  local entry
  if [[ -n "${!variable_name:-}" || ! -r /proc/1/environ ]]; then
    return
  fi
  while IFS= read -r -d '' entry; do
    case "$entry" in
      "${variable_name}="*)
        printf -v "$variable_name" '%s' "${entry#*=}"
        export "$variable_name"
        return
        ;;
    esac
  done < /proc/1/environ
}

load_vast_runtime_variable CONTAINER_ID
load_vast_runtime_variable CONTAINER_API_KEY

SMOKE_ID=lifecycle-smoke-${CONTAINER_ID:-manual}
EXPERIMENT_ROOT=/output/${SMOKE_ID}
STAGING_ROOT=/output/${SMOKE_ID}-staging
FIRST_REQUEST_PATH=/output/${SMOKE_ID}-first-sync-request.json
FIRST_SYNC_RESULT=/output/${SMOKE_ID}-first-sync-result.json
SHUTDOWN_REQUEST_PATH=/output/${SMOKE_ID}-shutdown-sync-request.json
SHUTDOWN_SYNC_RESULT=/output/${SMOKE_ID}-shutdown-sync-result.json
UPLOAD_VERIFIED_MARKER=/output/${SMOKE_ID}-UPLOAD_VERIFIED
export UPLOAD_VERIFIED_MARKER

if [[ ! "${HF_TOKEN:-}" =~ ^hf_[A-Za-z0-9]+$ ]]; then
  echo "A token-shaped HF_TOKEN is required." >&2
  exit 1
fi
if [[ "${AUTO_DESTROY:-0}" != "1" ]]; then
  echo "AUTO_DESTROY=1 is required for this bounded lifecycle test." >&2
  exit 1
fi
if [[ ! "${CONTAINER_ID:-}" =~ ^[0-9]+$ || -z "${CONTAINER_API_KEY:-}" ]]; then
  echo "The Vast CONTAINER_ID and instance-scoped CONTAINER_API_KEY are required." >&2
  exit 1
fi

if (( DRY_RUN )); then
  echo "Verify both approved private Hugging Face repositories; no dataset or base-model download"
  echo "Run one synthetic optimizer step on exactly one CUDA GPU"
  echo "Create a tiny closed artifact set"
  echo "lehome-train sync --request ${FIRST_REQUEST_PATH}"
  echo "Require disposable=true and every entry remotely_verified=true; add first-sync-result.json to evidence"
  echo "lehome-train sync --request ${SHUTDOWN_REQUEST_PATH}"
  echo "Require a second immutable readback; write shutdown-sync-result.json marker"
  echo "${SCRIPT_DIR}/destroy_instance.sh"
  exit 0
fi

python - <<'PY'
from lehome_train.hub import HuggingFaceHubTransport, ensure_approved_private_repository

transport = HuggingFaceHubTransport(timeout_seconds=30.0)
repository_policies = {
    "ryanjin333/lehome-groot-n17-data": False,
    "ryanjin333/lehome-groot-n17-models": True,
}
for repository, require_write in repository_policies.items():
    access = ensure_approved_private_repository(
        transport=transport,
        repository=repository,
        create=False,
        timeout_seconds=30.0,
    )
    if not (access.can_read and access.private_repository):
        raise SystemExit(f"approved private repository is not readable: {repository}")
    if require_write and not access.can_write:
        raise SystemExit(f"approved model repository is not writable: {repository}")
PY

python - <<'PY'
import torch
from torch.utils.data import DataLoader, TensorDataset

assert torch.cuda.device_count() == 1, "exactly one CUDA GPU must be visible"
device = torch.device("cuda:0")
dataset = TensorDataset(torch.randn(8, 4), torch.randn(8, 2))
features, targets = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
model = torch.nn.Linear(4, 2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer.zero_grad(set_to_none=True)
loss = torch.nn.functional.mse_loss(model(features.to(device)), targets.to(device))
assert torch.isfinite(loss)
loss.backward()
optimizer.step()
print("GPU_SENTINEL:optimizer-step-complete")
PY

mkdir -p \
  "${EXPERIMENT_ROOT}/checkpoints" \
  "${EXPERIMENT_ROOT}/logs" \
  "${EXPERIMENT_ROOT}/reports" \
  "${STAGING_ROOT}"
printf 'bounded lifecycle smoke checkpoint\n' > "${EXPERIMENT_ROOT}/checkpoints/step-smoke.tar.zst"
printf 'bounded lifecycle smoke log\n' > "${EXPERIMENT_ROOT}/logs/smoke.log"
printf '{"schema_version":1,"kind":"lifecycle-smoke"}\n' > "${EXPERIMENT_ROOT}/resolved-config.json"
printf '{"schema_version":1,"source":"pinned-ghcr-image"}\n' > "${EXPERIMENT_ROOT}/provenance.json"
printf '{"schema_version":1,"status":"smoke-only"}\n' > "${EXPERIMENT_ROOT}/reports/training-report.json"

export MODEL_REPOSITORY SMOKE_ID EXPERIMENT_ROOT STAGING_ROOT

write_sync_request() {
  local request_path=$1
  local sync_result=$2
  REQUEST_PATH=${request_path} SYNC_RESULT=${sync_result} python - <<'PY'
import json
import os
from pathlib import Path

from lehome_train.io import canonical_json_sha256

experiment_root = Path(os.environ["EXPERIMENT_ROOT"])
config = json.loads((experiment_root / "resolved-config.json").read_text(encoding="utf-8"))
request = {
    "schema_version": 1,
    "command": "sync",
    "arguments": {
        "experiment_root": str(experiment_root),
        "experiment_id": os.environ["SMOKE_ID"],
        "experiment_config_sha256": canonical_json_sha256(config),
        "repository": os.environ["MODEL_REPOSITORY"],
        "revision": "main",
        "staging_root": os.environ["STAGING_ROOT"],
        "timeout_seconds": 30,
        "max_attempts": 5,
        "output": os.environ["SYNC_RESULT"],
    },
}
Path(os.environ["REQUEST_PATH"]).write_text(
    json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY
}

validate_sync_result() {
  local sync_result=$1
  SYNC_RESULT=${sync_result} python - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["SYNC_RESULT"]).read_text(encoding="utf-8"))
entries = result.get("manifest", {}).get("entries", [])
if result.get("disposable") is not True or not entries:
    raise SystemExit("Hugging Face sync did not produce disposable evidence")
if not all(entry.get("remotely_verified") is True for entry in entries):
    raise SystemExit("one or more Hugging Face artifacts failed immutable readback")
PY
}

write_sync_request "${FIRST_REQUEST_PATH}" "${FIRST_SYNC_RESULT}"
lehome-train sync --request "${FIRST_REQUEST_PATH}"
validate_sync_result "${FIRST_SYNC_RESULT}"

# Preserve the first immutable upload/readback evidence inside the artifact set,
# then sync and verify that larger closed set once more before destruction.
cp "${FIRST_SYNC_RESULT}" "${EXPERIMENT_ROOT}/reports/first-sync-result.json"
chmod 600 "${EXPERIMENT_ROOT}/reports/first-sync-result.json"

write_sync_request "${SHUTDOWN_REQUEST_PATH}" "${SHUTDOWN_SYNC_RESULT}"
lehome-train sync --request "${SHUTDOWN_REQUEST_PATH}"
validate_sync_result "${SHUTDOWN_SYNC_RESULT}"
SYNC_RESULT=${SHUTDOWN_SYNC_RESULT} python - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["SYNC_RESULT"]).read_text(encoding="utf-8"))
Path(os.environ["UPLOAD_VERIFIED_MARKER"]).write_text(
    f"{result['repository']}@{result['immutable_revision']}\n",
    encoding="utf-8",
)
PY
chmod 600 "${UPLOAD_VERIFIED_MARKER}"
if [[ "${HOLD_BEFORE_DESTROY:-0}" == "1" ]]; then
  echo "Upload/readback verification complete; holding Vast instance ${CONTAINER_ID} before destruction."
  exit 0
fi
"${SCRIPT_DIR}/destroy_instance.sh"
