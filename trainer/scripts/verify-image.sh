#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
mode=cpu
expected_release_mode=release
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --gpu) mode=gpu ;;
    --diagnostic) expected_release_mode=diagnostic-dirty ;;
    *) echo "unsupported verifier option: $1" >&2; exit 64 ;;
  esac
  shift
done

repository_commit=$(git -C "$repo_root" rev-parse HEAD)
image_ref=${1:-${IMAGE_REPOSITORY:-lehome-groot-n17-trainer}:$repository_commit}
expected_base=sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719
expected_groot=23ace64f17aa5015259b8609d371eb61a357c776
expected_model=2fc962b973bccdd5d8ce4f67cc63b264d6886495

actual_user=$(docker image inspect --format '{{.Config.User}}' "$image_ref")
if [[ "$actual_user" != "trainer" ]]; then
  echo "image must run as the named non-root trainer user, found: $actual_user" >&2
  exit 1
fi

actual_release_mode=$(docker image inspect --format '{{index .Config.Labels "io.lehome.release-mode"}}' "$image_ref")
if [[ "$actual_release_mode" != "$expected_release_mode" ]]; then
  echo "image release mode is $actual_release_mode; expected $expected_release_mode" >&2
  exit 1
fi

for pair in \
  "io.lehome.cuda-base-digest=$expected_base" \
  "io.lehome.isaac-groot-revision=$expected_groot" \
  "io.lehome.model-revision=$expected_model" \
  "org.opencontainers.image.revision=$repository_commit"; do
  key=${pair%%=*}
  expected=${pair#*=}
  actual=$(docker image inspect --format "{{index .Config.Labels \"$key\"}}" "$image_ref")
  if [[ "$actual" != "$expected" ]]; then
    echo "image label $key differs from its immutable pin" >&2
    exit 1
  fi
done

if docker image inspect --format '{{json .Config.Env}}' "$image_ref" \
  | grep -Eq 'HF_TOKEN=|hf_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{30,}'; then
  echo "image configuration contains credential material" >&2
  exit 1
fi

mount_root=$(mktemp -d)
mkdir -p "$mount_root/cache" "$mount_root/prepared" "$mount_root/output"
chmod 0777 "$mount_root/cache" "$mount_root/prepared" "$mount_root/output"

cleanup_mount_root() {
  local status=$?
  docker run --rm --platform linux/amd64 --user 0:0 \
    -v "$mount_root/cache:/cache" \
    -v "$mount_root/prepared:/prepared" \
    -v "$mount_root/output:/output" \
    --entrypoint /bin/chmod "$image_ref" \
    -R a+rwX /cache /prepared /output >/dev/null 2>&1 || true
  rm -rf "$mount_root" || true
  return "$status"
}
trap cleanup_mount_root EXIT

run=(docker run --rm --platform linux/amd64
  -v "$mount_root/cache:/cache"
  -v "$mount_root/prepared:/prepared"
  -v "$mount_root/output:/output")

"${run[@]}" "$image_ref" --help >/dev/null
"${run[@]}" --entrypoint /bin/bash "$image_ref" -euo pipefail -c '
  test "$(id -u)" -ne 0
  test "$(python -c '\''import platform; print(platform.python_version())'\'')" = 3.10.18
  python -c '\''import gr00t, lehome_train, torch'\''
  test "$(git -C /opt/isaac-groot rev-parse HEAD)" = 23ace64f17aa5015259b8609d371eb61a357c776
  test -z "$(git -C /opt/isaac-groot status --porcelain=v1 --untracked-files=all)"
  lehome-train --help >/dev/null
  test ! -e /isaac-sim
  test ! -e /IsaacLab
  test ! -e /models
  test ! -e /data
  large_file=$(find /opt/trainer /opt/isaac-groot -type f -size +50M -print -quit)
  test -z "$large_file"
  while IFS= read -r -d "" candidate; do
    if [[ "$(wc -c < "$candidate")" -gt 1024 ]] \
      || [[ "$(head -n 1 "$candidate")" != "version https://git-lfs.github.com/spec/v1" ]]; then
      echo "image contains a hydrated model or dataset artifact: $candidate" >&2
      exit 1
    fi
  done < <(find /opt/trainer /opt/isaac-groot -type f \( -iname "*.safetensors" -o -iname "*.ckpt" -o -iname "*.parquet" -o -iname "*.mp4" -o -iname "*.gif" -o -iname "*.whl" \) -print0)
  set +e
  grep -RIE "hf_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{30,}" /opt/trainer /opt/isaac-groot >/dev/null
  secret_status=$?
  set -e
  [[ "$secret_status" -eq 1 ]]
'

if [[ "$mode" == "gpu" ]]; then
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "--gpu acceptance requires a Linux NVIDIA host" >&2
    exit 69
  fi
  gpu_output=$("${run[@]}" -i --gpus device=0 -e CUDA_VISIBLE_DEVICES=0 \
    --entrypoint /opt/runtime/bin/python "$image_ref" - <<'PY'
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
  )
  printf '%s\n' "$gpu_output"
  if ! grep -Fxq 'GPU_SENTINEL:optimizer-step-complete' <<<"$gpu_output"; then
    echo "GPU optimizer-step sentinel was not returned by the container" >&2
    exit 1
  fi
fi

echo "verified $image_ref ($mode structural gate)"
