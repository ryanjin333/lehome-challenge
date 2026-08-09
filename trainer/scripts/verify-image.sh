#!/usr/bin/env bash
set -euo pipefail

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

repository_commit=${REPOSITORY_COMMIT:?REPOSITORY_COMMIT must be the immutable source revision}
if ! [[ "$repository_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "REPOSITORY_COMMIT must be a lowercase source revision" >&2
  exit 64
fi
image_ref=${1:-}
if ! [[ "$image_ref" =~ ^docker\.io/ryanjin333/behavior1k-groot-n17@sha256:[0-9a-f]{64}$ ]]; then
  echo "image must be the canonical Docker Hub digest reference" >&2
  exit 64
fi
expected_base=sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719
expected_groot=ace36d935b376fbf25cd56371e23877b95407c40
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
  "io.lehome.image-role=training" \
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

docker run --rm --platform linux/amd64 --entrypoint /bin/bash "$image_ref" -euo pipefail -c '
  test "$TMPDIR" = /cache/tmp
  test -d "$TMPDIR"
  test -w "$TMPDIR"
  probe=$(mktemp)
  [[ "$probe" == /cache/tmp/* ]]
  rm -f "$probe"
  test "$HOME" = /home/trainer
  test -d "$HOME"
  test -w "$HOME"
  test -w "$HOME/.bashrc"
  test "$(getent passwd "$(id -un)" | cut -d: -f6-7)" = "/home/trainer:/bin/bash"
  printf "\n" >> "$HOME/.bashrc"
'

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
  test "$(/opt/runtime/bin/python -c '\''import huggingface_hub; print(huggingface_hub.__version__)'\'')" = 0.36.2
  test "$(/opt/b1k-bucket-helper/bin/python -c '\''import platform; print(platform.python_version())'\'')" = 3.10.18
  test "$(/opt/b1k-bucket-helper/bin/python -c '\''import huggingface_hub; print(huggingface_hub.__version__)'\'')" = 1.24.0
  test -x /opt/b1k-bucket-helper/bin/b1k-bucket-helper
  grep -Fxq "exec /opt/b1k-bucket-helper/bin/python /opt/b1k-bucket-helper/libexec/b1k-bucket-helper \"\$@\"" /opt/b1k-bucket-helper/bin/b1k-bucket-helper
  test "$(git -C /opt/isaac-groot rev-parse HEAD)" = ace36d935b376fbf25cd56371e23877b95407c40
  test -z "$(git -C /opt/isaac-groot status --porcelain=v1 --untracked-files=all)"
  test -f /opt/isaac-groot/scripts/b1k/train_b1k.py
  test -f /opt/isaac-groot/scripts/b1k/deploy_modality.py
  test -f /opt/isaac-groot/examples/b1k/r1pro.py
  test -f /opt/isaac-groot/examples/b1k/r1pro.json
  test -f /opt/isaac-groot/gr00t/data/dataset/lerobot_episode_loader.py
  /opt/runtime/bin/python /opt/isaac-groot/scripts/b1k/train_b1k.py --help >/dev/null
  /usr/local/bin/verify-b1k-cli
  lehome-train --help >/dev/null
  test ! -e /isaac-sim
  test ! -e /IsaacLab
  test ! -e /OmniGibson
  test ! -e /models
  test ! -e /data
  large_file=$(find /opt/trainer /opt/isaac-groot -type f -size +50M -print -quit)
  test -z "$large_file"
  bundled_artifact=$(find /opt/trainer /opt/isaac-groot -type f \( -iname "*.safetensors" -o -iname "*.ckpt" -o -iname "*.parquet" -o -iname "*.mp4" -o -iname "*.gif" -o -iname "*.whl" \) -print -quit)
  test -z "$bundled_artifact"
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
