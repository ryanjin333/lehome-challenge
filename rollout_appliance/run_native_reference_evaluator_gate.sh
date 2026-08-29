#!/usr/bin/env bash
# Run the public GR00T submission through its own pinned evaluator boundary.
# This appliance accepts existing identities/caches only: no provider mutation,
# checkpoint-weight download (including pretrained_model), image build, or
# external upload is possible here.
set -euo pipefail

readonly SOURCE_REPOSITORY="theo-zhou/lehome-groot-submission-4"
readonly SOURCE_REVISION="d384fe00508acd96ab1c3c5dc265e08261f94b3b"
readonly EXPECTED_SOURCE_TREE_SHA256="eada9f80b0dda1428177fe4551efa8059fe85845d4db5b32bb673f88a50c6bb2"
readonly EXPECTED_LEROBOT_VERSION="0.4.3"
readonly EXACT_VM_ID="computeinstance-u00t6xfqhadrcmssa2"
readonly PROTECTED_DISK_ID="computedisk-u00pbe55crxy7jr56x"
readonly PROVIDER_SOURCE_IMAGE_ID="computeimage-u00zf6w3yf72gakhcy"
readonly RUNTIME_IMAGE_REFERENCE="lehome-rollout:build"
readonly RUNTIME_IMAGE_ID="sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly RUNTIME_REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly NATIVE_SITE_ROOT="$RUNTIME_REPO_ROOT/rollout_appliance/native_reference_site"
readonly ISAACLAB_ROOT="/opt/lehome-challenge/third_party/IsaacLab/source/isaaclab"
readonly ISAACLAB_TASKS_ROOT="/opt/lehome-challenge/third_party/IsaacLab/source/isaaclab_tasks"
readonly CANONICAL_CACHE_MANIFEST="$SCRIPT_DIR/native_reference_canonical_cache_manifest.json"

fail() { printf 'error: %s\n' "$*" >&2; exit 2; }
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }
require_digest() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be a SHA-256 digest"; }
require_absolute_directory() {
  [[ "$1" == /* && "$1" != *".."* ]] || fail "$2 must be an absolute path without traversal"
  [[ -d "$1" && ! -L "$1" ]] || fail "$2 is unavailable or unsafe"
}
require_new_output() {
  [[ "$1" == /* && "$1" != *".."* ]] || fail "native reference output root is unsafe"
  [[ "$(basename -- "$1")" =~ ^native-reference-[0-9]{12,14}$ ]] || fail "native reference output root must have a new native-reference timestamp name"
  [[ ! -e "$1" && ! -L "$1" ]] || fail "native reference output root already exists"
}
require_new_file() {
  [[ "$1" == /* && "$1" != *".."* ]] || fail "$2 is unsafe"
  [[ ! -e "$1" && ! -L "$1" ]] || fail "$2 already exists"
  require_absolute_directory "$(dirname -- "$1")" "$2 parent"
}

tree_sha256() {
  python3 - "$1" <<'PY'
import hashlib, stat, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(strict=True); digest = hashlib.sha256()
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root)
    if ".git" in relative.parts or path.is_dir(): continue
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode): raise SystemExit("native reference tree has an unsafe entry")
    digest.update(relative.as_posix().encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
print(digest.hexdigest())
PY
}

expected_checkpoint_digest() {
  case "$1" in
    config.json) echo b7c385bc57456eae603e929b84defb7e991194aade2aad70785e21991e37614c ;;
    model.safetensors) echo d8d91b6c11cb5aa18fe9a48e7da88eae0ec7e5a227a315ba67ee167d645cde76 ;;
    policy_preprocessor.json) echo a258dac8fa4e4e138990776e156cae36ae6cf172504a8c9e5f2d5864c9126009 ;;
    policy_postprocessor.json) echo f9e18fa7da47e2b6d7ba3459236b140e28f834ce5640ba199be1412d50672fa7 ;;
    policy_preprocessor_step_2_groot_pack_inputs_v3.safetensors|policy_postprocessor_step_0_groot_action_unpack_unnormalize_v1.safetensors) echo 74dcbba5d152b7e07c239d8cd66b19b1fd08aa37ff930aa5f2e94cd772a4a912 ;;
    train_config.json) echo 81cd0cfe2b2f70dbf55bc7739f9a1f248aebd0e281994f415964d9d0f6e3c118 ;;
    *) return 1 ;;
  esac
}

validate_checkpoint() {
  local mode="${1:-full}"
  local -a required=(config.json model.safetensors policy_preprocessor.json policy_postprocessor.json policy_preprocessor_step_2_groot_pack_inputs_v3.safetensors policy_postprocessor_step_0_groot_action_unpack_unnormalize_v1.safetensors train_config.json)
  local item observed expected
  [[ -z "$(find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type l -print -quit)" ]] || fail "native reference checkpoint contains a symlink"
  for item in "${required[@]}"; do
    [[ -f "$CHECKPOINT_ROOT/$item" && ! -L "$CHECKPOINT_ROOT/$item" ]] || fail "native reference checkpoint cache is incomplete: $item"
    if [[ "$item" != model.safetensors || "$mode" == full ]]; then
      local observed_digest
      observed_digest="$(sha256_file "$CHECKPOINT_ROOT/$item")"
      [[ "$observed_digest" == "$(expected_checkpoint_digest "$item")" ]] || fail "native reference checkpoint digest mismatch: $item"
      if [[ "$item" == model.safetensors ]]; then CHECKPOINT_MODEL_SHA256="$observed_digest"; fi
    fi
  done
  observed="$(find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type f -exec basename -- {} \; | LC_ALL=C sort)"
  expected="$(printf '%s\n' "${required[@]}" | LC_ALL=C sort)"
  [[ "$observed" == "$expected" ]] || fail "native reference checkpoint has an unexpected file set"
  "$PYTHON_BIN" - "$CHECKPOINT_ROOT/policy_preprocessor.json" "$CHECKPOINT_ROOT/policy_postprocessor.json" <<'PY'
import json, sys
from pathlib import Path
pre, post = (json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[1:])
try: horizon, dimension = pre["steps"][2]["config"]["action_horizon"], post["steps"][0]["config"]["env_action_dim"]
except (KeyError, IndexError, TypeError) as error: raise SystemExit("saved LeRobot processors have an unexpected schema") from error
if (horizon, dimension) != (16, 12): raise SystemExit("saved LeRobot processors do not preserve the 16-action/12-joint contract")
PY
}

checkpoint_tree_sha256() {
  "$PYTHON_BIN" - "$CHECKPOINT_ROOT" "$CHECKPOINT_MODEL_SHA256" <<'PY'
import hashlib, sys
from pathlib import Path
root, model_sha = Path(sys.argv[1]), sys.argv[2]
digest = hashlib.sha256()
for path in sorted(root.iterdir()):
    if not path.is_file() or path.is_symlink():
        raise SystemExit("checkpoint tree is unsafe")
    file_sha = model_sha if path.name == "model.safetensors" else hashlib.sha256(path.read_bytes()).hexdigest()
    digest.update(path.name.encode("utf-8") + b"\0" + file_sha.encode("ascii") + b"\n")
print(digest.hexdigest())
PY
}

validate_stage_integrity() {
  # Rehash the complete checkpoint, including the model, before every paid
  # simulator stage. A metadata snapshot is not sufficient evidence here.
  validate_checkpoint full
  validate_source_runtime
  validate_runtime_support
  authenticate_canonical_caches
  validate_runtime_asset_bindings
  [[ "$AUTHENTICATED_METADATA_TREE_SHA256" == "$METADATA_TREE_SHA256" ]] || fail "native reference metadata tree changed during evaluation"
  [[ "$AUTHENTICATED_ASSETS_TREE_SHA256" == "$ASSETS_TREE_SHA256" ]] || fail "native reference assets tree changed during evaluation"
}

validate_runtime_support() {
  [[ -d "$NATIVE_SITE_ROOT" && ! -L "$NATIVE_SITE_ROOT" ]] \
    || fail "reviewed native reference site support is unavailable or unsafe"
  [[ -f "$NATIVE_SITE_ROOT/sitecustomize.py" && ! -L "$NATIVE_SITE_ROOT/sitecustomize.py" ]] \
    || fail "reviewed native reference sitecustomize is unavailable or unsafe"
  require_absolute_directory "$ISAACLAB_ROOT" "trusted IsaacLab source"
  require_absolute_directory "$ISAACLAB_TASKS_ROOT" "trusted IsaacLab tasks source"
}

validate_runtime_asset_bindings() {
  PYTHONSAFEPATH=1 "$PYTHON_BIN" "$SCRIPT_DIR/../scripts/verify_native_reference_evaluator_gate.py" \
    validate-asset-bindings --assets-root "$ASSETS_ROOT" \
    --runtime-repo-root "$RUNTIME_REPO_ROOT" >/dev/null \
    || fail "native runtime asset bind mounts do not share canonical device/inode identities"
}

authenticate_canonical_caches() {
  local receipt
  receipt="$("$PYTHON_BIN" "$SCRIPT_DIR/../scripts/verify_native_reference_evaluator_gate.py" authenticate-cache \
    --metadata-root "$METADATA_ROOT" --assets-root "$ASSETS_ROOT" \
    --manifest "$CANONICAL_CACHE_MANIFEST")" || fail "canonical metadata/assets authentication failed"
  AUTHENTICATED_METADATA_TREE_SHA256="$(printf '%s' "$receipt" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["metadata_tree_sha256"])')" || fail "canonical metadata authentication receipt is invalid"
  AUTHENTICATED_ASSETS_TREE_SHA256="$(printf '%s' "$receipt" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["assets_tree_sha256"])')" || fail "canonical assets authentication receipt is invalid"
}

stage_native_source() {
  # GIT_LFS_SKIP_SMUDGE makes this source-stage-only: it must never pull LFS
  # checkpoint weights while staging public evaluator code.
  [[ -z "$(find "$SOURCE_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]] || fail "source-stage requires an empty source root"
  command -v git >/dev/null 2>&1 || fail "git is required to stage pinned native source"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$SOURCE_ROOT" init --quiet
  git -C "$SOURCE_ROOT" remote add origin "https://huggingface.co/$SOURCE_REPOSITORY"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$SOURCE_ROOT" -c filter.lfs.smudge= -c filter.lfs.required=false fetch --depth=1 origin "$SOURCE_REVISION"
  git -C "$SOURCE_ROOT" sparse-checkout init --no-cone
  git -C "$SOURCE_ROOT" sparse-checkout set --no-cone scripts source/lehome configs
  GIT_LFS_SKIP_SMUDGE=1 git -C "$SOURCE_ROOT" -c filter.lfs.smudge= -c filter.lfs.required=false checkout --quiet --detach "$SOURCE_REVISION"
  [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || fail "source-stage checkout revision mismatch"
  printf '%s\n' "{\"status\":\"source_staged\",\"source_revision\":\"$SOURCE_REVISION\"}"
}

validate_source_runtime() {
  [[ -f "$SOURCE_ROOT/scripts/eval.py" && ! -L "$SOURCE_ROOT/scripts/eval.py" ]] || fail "native reference source cache is incomplete"
  [[ -d "$SOURCE_ROOT/source/lehome" && ! -L "$SOURCE_ROOT/source/lehome" && -d "$SOURCE_ROOT/configs" && ! -L "$SOURCE_ROOT/configs" ]] || fail "native reference source cache is incomplete"
  git -C "$SOURCE_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "native reference source cache is not a pinned source checkout"
  [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || fail "native reference source revision mismatch"
  git -C "$SOURCE_ROOT" diff --quiet || fail "native reference source checkout is modified"
  [[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] || fail "native reference source checkout has untrusted files"
  [[ "$(tree_sha256 "$SOURCE_ROOT")" == "$EXPECTED_SOURCE_TREE_SHA256" ]] || fail "native reference source tree digest mismatch"
  local found
  found="$(PYTHONPATH="$SOURCE_ROOT/source/lehome:$SOURCE_ROOT:$ISAACLAB_ROOT:$ISAACLAB_TASKS_ROOT" "$PYTHON_BIN" -c 'import lerobot; print(getattr(lerobot, "__version__", ""))')"
  [[ "$found" == "$EXPECTED_LEROBOT_VERSION" ]] || fail "native reference requires LeRobot $EXPECTED_LEROBOT_VERSION, found ${found:-missing}"
}

validate_cache_trust_manifest() {
  "$PYTHON_BIN" "$SCRIPT_DIR/../scripts/verify_native_reference_evaluator_gate.py" fetch-cache-manifest \
    --revision "$CACHE_TRUST_MANIFEST_REVISION" --path "$CACHE_TRUST_MANIFEST_PATH" \
    --checkpoint-tree-sha256 "$CHECKPOINT_TREE_SHA256" \
    --metadata-tree-sha256 "$METADATA_TREE_SHA256" \
    --assets-tree-sha256 "$ASSETS_TREE_SHA256" >/dev/null || fail "immutable cache manifest readback failed"
}

write_cache_inventory_manifest() {
  "$PYTHON_BIN" - "$CACHE_INVENTORY_OUTPUT" "$CACHE_INVENTORY_REMOTE_PATH" "$CHECKPOINT_TREE_SHA256" "$METADATA_TREE_SHA256" "$ASSETS_TREE_SHA256" <<'PY'
import json, os, sys
from pathlib import Path
output = Path(sys.argv[1])
payload = {"schema_version":2,"kind":"lehome_native_reference_cache_trust_manifest_v2","source_repository":"ryanjin333/lehome-groot-n17-rollouts","path":sys.argv[2],"checkpoint_tree_sha256":sys.argv[3],"metadata_tree_sha256":sys.argv[4],"assets_tree_sha256":sys.argv[5]}
fd=os.open(output, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o444)
with os.fdopen(fd,"w",encoding="utf-8") as stream: json.dump(payload,stream,sort_keys=True,separators=(",",":")); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
PY
}

validate_running_provider_binding() {
  "$PYTHON_BIN" - "$OUTPUT_ROOT/evidence/provider-running-receipt.json" "$VM_ID" "$DISK_ID" "$PROVIDER_SOURCE_IMAGE_ID" <<'PY'
import json, sys
from pathlib import Path
receipt=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected={"vm_id":sys.argv[2],"disk_id":sys.argv[3],"provider_source_image_id":sys.argv[4],"state":"RUNNING"}
if any(receipt.get(key) != value for key,value in expected.items()):
    raise SystemExit("provider RUNNING receipt does not bind the pinned VM/disk/source image")
PY
}

validate_runtime_image_binding() {
  "$PYTHON_BIN" - "$OUTPUT_ROOT/evidence/runtime-image-receipt.json" "$RUNTIME_IMAGE_REFERENCE" "$RUNTIME_IMAGE_ID" <<'PY'
import json, sys
from pathlib import Path
receipt=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected={"runtime_image_reference":sys.argv[2],"runtime_image_id":sys.argv[3]}
if any(receipt.get(key) != value for key,value in expected.items()):
    raise SystemExit("runtime image receipt does not bind the approved local image")
PY
}

probe_cuda() {
  "$PYTHON_BIN" - "$OUTPUT_ROOT/cuda-runtime.json" <<'PY'
import json, os, sys
from pathlib import Path
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1 or not torch.version.cuda: raise SystemExit("native reference requires an available CUDA policy device")
payload = {"schema_version":1,"kind":"lehome_native_reference_cuda_probe_v1","cuda_available":True,"cuda_device_count":torch.cuda.device_count(),"cuda_runtime":torch.version.cuda,"cuda_device_name":torch.cuda.get_device_name(0),"policy_device":"cuda:0"}
fd=os.open(Path(sys.argv[1]), os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o444)
with os.fdopen(fd,"w",encoding="utf-8") as stream: json.dump(payload,stream,sort_keys=True,separators=(",",":")); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
PY
}

probe_host_runtime() {
  (cd -- "$RUNTIME_REPO_ROOT" && PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 PYTHONPATH="$SOURCE_ROOT/source/lehome:$SOURCE_ROOT:$ISAACLAB_ROOT:$ISAACLAB_TASKS_ROOT" \
    "$PYTHON_BIN" "$SCRIPT_DIR/../scripts/verify_native_reference_evaluator_gate.py" probe-host-runtime \
      --source-root "$SOURCE_ROOT" --isaaclab-root "$ISAACLAB_ROOT" \
      --receipt "$OUTPUT_ROOT/host-runtime.json" >/dev/null) \
    || fail "native reference host runtime probe failed"
}

write_identity_and_preflight() {
  "$PYTHON_BIN" - "$OUTPUT_ROOT/identity.json" "$OUTPUT_ROOT/preflight.json" "$OUTPUT_ROOT/cuda-runtime.json" "$OUTPUT_ROOT/host-runtime.json" "$CHECKPOINT_TREE_SHA256" "$METADATA_TREE_SHA256" "$ASSETS_TREE_SHA256" "$CACHE_TRUST_MANIFEST_SHA256" "$PROVIDER_RUNNING_RECEIPT_SHA256" "$RUNTIME_IMAGE_RECEIPT_SHA256" "$VM_ID" "$DISK_ID" "$RUNTIME_IMAGE_REFERENCE" "$RUNTIME_IMAGE_ID" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
identity_path, preflight_path, cuda_path, host_path = map(Path, sys.argv[1:5]); cuda=json.loads(cuda_path.read_text()); host=json.loads(host_path.read_text())
expected_host={"schema_version","kind","source_root","python_executable","python_version","torch_version","lerobot_version","lerobot_origin","scripts_eval_origin","lehome_origin","isaaclab_app_origin","app_launcher_class"}
if set(host) != expected_host or host.get("schema_version") != 1 or host.get("kind") != "lehome_native_reference_host_runtime_v1" or host.get("lerobot_version") != "0.4.3" or host.get("app_launcher_class") != "isaaclab.app.AppLauncher" or not str(host.get("isaaclab_app_origin","")).startswith("/opt/lehome-challenge/third_party/IsaacLab/source/isaaclab/"): raise SystemExit("native reference host runtime is invalid")
identity={"source_repository":"theo-zhou/lehome-groot-submission-4","source_revision":"d384fe00508acd96ab1c3c5dc265e08261f94b3b","source_tree_sha256":"eada9f80b0dda1428177fe4551efa8059fe85845d4db5b32bb673f88a50c6bb2","checkpoint_tree_sha256":sys.argv[5],"metadata_tree_sha256":sys.argv[6],"assets_tree_sha256":sys.argv[7],"cache_trust_manifest_sha256":sys.argv[8],"provider_running_receipt_sha256":sys.argv[9],"runtime_image_receipt_sha256":sys.argv[10],"provider_source_image_id":"computeimage-u00zf6w3yf72gakhcy","runtime_image_reference":sys.argv[13],"runtime_image_id":sys.argv[14],"lerobot_version":"0.4.3","policy_class":"scripts.eval_policy.lerobot_policy.LeRobotPolicy","policy_device":"cuda:0","cuda_available":cuda["cuda_available"],"cuda_device_count":cuda["cuda_device_count"],"cuda_runtime":cuda["cuda_runtime"],"vm_id":sys.argv[11],"disk_id":sys.argv[12],"simulator_device":"cpu","task_description":"fold the garment on the table","action_horizon":16,"action_dimension":12,"success_checker":"pinned_raw_success_distance_second_mesh_points",**{key:host[key] for key in ("source_root","python_executable","python_version","torch_version","lerobot_origin","scripts_eval_origin","lehome_origin")}}
preflight={"schema_version":2,"kind":"lehome_native_reference_preflight_v2","identity":identity,"cuda_probe_sha256":hashlib.sha256(cuda_path.read_bytes()).hexdigest(),"host_runtime_sha256":hashlib.sha256(host_path.read_bytes()).hexdigest(),"runtime_image_receipt_sha256":sys.argv[10]}
for path,document in ((identity_path,identity),(preflight_path,preflight)):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(fd,"w",encoding="utf-8") as stream: json.dump(document,stream,sort_keys=True,separators=(",",":")); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
PY
}

# Provider RUNNING/attachment is an external, read-only operator gate.  The
# pure-Python finalizer binds its stopped receipt; this script never calls a
# provider CLI.
MODE="${LEHOME_NATIVE_REFERENCE_MODE:-}"
VALIDATE_ONLY="${LEHOME_NATIVE_REFERENCE_VALIDATE_ONLY:-0}"
if [[ -z "$MODE" ]]; then MODE=$([[ "$VALIDATE_ONLY" == 1 ]] && printf validate-only || printf execute); fi
[[ "$MODE" == source-stage || "$MODE" == inventory-cache || "$MODE" == validate-only || "$MODE" == execute ]] || fail "LEHOME_NATIVE_REFERENCE_MODE must be source-stage, inventory-cache, validate-only, or execute"
[[ "$VALIDATE_ONLY" == 0 || "$VALIDATE_ONLY" == 1 ]] || fail "LEHOME_NATIVE_REFERENCE_VALIDATE_ONLY must be exactly 0 or 1"
SOURCE_ROOT="${LEHOME_NATIVE_REFERENCE_SOURCE_ROOT:-}"
PYTHON_BIN="${LEHOME_NATIVE_REFERENCE_PYTHON:-python3}"
require_absolute_directory "$SOURCE_ROOT" "native reference source cache"
if [[ "$MODE" == source-stage ]]; then stage_native_source; exit 0; fi
RUNTIME_IMAGE_RECEIPT="${LEHOME_NATIVE_REFERENCE_RUNTIME_IMAGE_RECEIPT:-}"
if [[ "$MODE" == execute ]]; then
  [[ "$RUNTIME_IMAGE_RECEIPT" == /* && "$RUNTIME_IMAGE_RECEIPT" != *".."* && -f "$RUNTIME_IMAGE_RECEIPT" && ! -L "$RUNTIME_IMAGE_RECEIPT" ]] || fail "runtime image receipt is unavailable or unsafe"
fi

VM_ID="${LEHOME_NATIVE_REFERENCE_VM_ID:-}"; DISK_ID="${LEHOME_NATIVE_REFERENCE_DISK_ID:-}"
CHECKPOINT_ROOT="${LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT:-}"; METADATA_ROOT="${LEHOME_NATIVE_REFERENCE_METADATA_ROOT:-}"; ASSETS_ROOT="${LEHOME_NATIVE_REFERENCE_ASSETS_ROOT:-}"; OUTPUT_ROOT="${LEHOME_NATIVE_REFERENCE_OUTPUT_ROOT:-}"
CACHE_TRUST_MANIFEST_REVISION="${LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_REVISION:-}"; CACHE_TRUST_MANIFEST_PATH="${LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_PATH:-}"; PROVIDER_RUNNING_RECEIPT="${LEHOME_NATIVE_REFERENCE_PROVIDER_RUNNING_RECEIPT:-}"
require_absolute_directory "$CHECKPOINT_ROOT" "native reference checkpoint cache"; require_absolute_directory "$METADATA_ROOT" "native reference metadata cache"; require_absolute_directory "$ASSETS_ROOT" "native reference challenge assets"
validate_checkpoint full; validate_source_runtime; validate_runtime_support
authenticate_canonical_caches
validate_runtime_asset_bindings
CHECKPOINT_TREE_SHA256="$(checkpoint_tree_sha256)"; METADATA_TREE_SHA256="$AUTHENTICATED_METADATA_TREE_SHA256"; ASSETS_TREE_SHA256="$AUTHENTICATED_ASSETS_TREE_SHA256"
CACHE_INVENTORY_OUTPUT="${LEHOME_NATIVE_REFERENCE_CACHE_MANIFEST_OUTPUT:-}"; CACHE_INVENTORY_REMOTE_PATH="${LEHOME_NATIVE_REFERENCE_CACHE_MANIFEST_PATH:-}"
if [[ "$MODE" == inventory-cache ]]; then
  require_new_file "$CACHE_INVENTORY_OUTPUT" "cache inventory manifest"
  [[ "$CACHE_INVENTORY_REMOTE_PATH" =~ ^reference-checks/native-cache-[a-z0-9][a-z0-9.-]{0,63}/cache-trust-manifest\.json$ ]] || fail "cache inventory manifest remote path is invalid"
  write_cache_inventory_manifest
  printf '%s\n' '{"status":"inventory_complete","mode":"zero-evaluation"}'
  exit 0
fi
[[ "$VM_ID" == "$EXACT_VM_ID" ]] || fail "LEHOME_NATIVE_REFERENCE_VM_ID is not the exact rollout VM"
[[ "$DISK_ID" == "$PROTECTED_DISK_ID" ]] || fail "LEHOME_NATIVE_REFERENCE_DISK_ID is not the protected shared disk"
require_new_output "$OUTPUT_ROOT"
validate_cache_trust_manifest
if [[ "$MODE" == validate-only ]]; then printf '%s\n' '{"status":"validated","mode":"no-execution"}'; exit 0; fi
[[ "$PROVIDER_RUNNING_RECEIPT" == /* && "$PROVIDER_RUNNING_RECEIPT" != *".."* && -f "$PROVIDER_RUNNING_RECEIPT" && ! -L "$PROVIDER_RUNNING_RECEIPT" ]] || fail "provider RUNNING receipt is unavailable or unsafe"

mkdir --mode=0700 -- "$OUTPUT_ROOT"; mkdir --mode=0700 -- "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/videos" "$OUTPUT_ROOT/receipts" "$OUTPUT_ROOT/evidence"
"$PYTHON_BIN" "$SCRIPT_DIR/../scripts/verify_native_reference_evaluator_gate.py" fetch-cache-manifest --revision "$CACHE_TRUST_MANIFEST_REVISION" --path "$CACHE_TRUST_MANIFEST_PATH" --checkpoint-tree-sha256 "$CHECKPOINT_TREE_SHA256" --metadata-tree-sha256 "$METADATA_TREE_SHA256" --assets-tree-sha256 "$ASSETS_TREE_SHA256" --receipt "$OUTPUT_ROOT/evidence/cache-trust-manifest.json" >/dev/null
"$PYTHON_BIN" "$SCRIPT_DIR/../scripts/verify_native_reference_evaluator_gate.py" bind-provider-receipt --state RUNNING --input "$PROVIDER_RUNNING_RECEIPT" --receipt "$OUTPUT_ROOT/evidence/provider-running-receipt.json" >/dev/null
"$PYTHON_BIN" "$SCRIPT_DIR/../scripts/verify_native_reference_evaluator_gate.py" bind-runtime-image-receipt --input "$RUNTIME_IMAGE_RECEIPT" --receipt "$OUTPUT_ROOT/evidence/runtime-image-receipt.json" >/dev/null
CACHE_TRUST_MANIFEST_SHA256="$(sha256_file "$OUTPUT_ROOT/evidence/cache-trust-manifest.json")"
PROVIDER_RUNNING_RECEIPT_SHA256="$(sha256_file "$OUTPUT_ROOT/evidence/provider-running-receipt.json")"
RUNTIME_IMAGE_RECEIPT_SHA256="$(sha256_file "$OUTPUT_ROOT/evidence/runtime-image-receipt.json")"
validate_running_provider_binding
validate_runtime_image_binding
probe_cuda; probe_host_runtime; write_identity_and_preflight
run_stage() {
  local stage="$1" category="$2" garment="$3"
  local log="$OUTPUT_ROOT/logs/stage-${stage}.log"
  local public_log_root="$OUTPUT_ROOT/public-runtime/stage-${stage}"
  validate_stage_integrity
  local -a command=("$PYTHON_BIN" -P -m scripts.eval --headless --device cpu --task "LeHome-BiSO101-Direct-Garment-v2" --policy_type lerobot --policy_path "$CHECKPOINT_ROOT" --dataset_root "$METADATA_ROOT/${category}_merged" --task_description "fold the garment on the table" --garment_type "$category" --garment_filter "$garment" --num_episodes 2 --max_steps 600 --seed 42 --save_video --video_dir "$OUTPUT_ROOT/videos/stage-${stage}" --garment_cfg_base_path "$ASSETS_ROOT/objects/Challenge_Garment" --particle_cfg_path "$SOURCE_ROOT/source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml" --ee_urdf_path "$ASSETS_ROOT/robots/so101_new_calib.urdf")
  (cd -- "$RUNTIME_REPO_ROOT" && \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONSAFEPATH=1 \
    LEHOME_NATIVE_REFERENCE_LOG_PROJECT_ROOT="$public_log_root" \
    LEHOME_NATIVE_REFERENCE_SOURCE_ROOT="$SOURCE_ROOT" \
    PYTHONPATH="$SOURCE_ROOT/source/lehome:$SOURCE_ROOT:$ISAACLAB_ROOT:$ISAACLAB_TASKS_ROOT:$NATIVE_SITE_ROOT" \
    "${command[@]}") >"$log" 2>&1 || fail "native reference stage $stage failed; inspect $log"
  "$PYTHON_BIN" "$SCRIPT_DIR/../scripts/verify_native_reference_evaluator_gate.py" compile-stage --bundle-root "$OUTPUT_ROOT" --stage "$stage" --category "$category" --garment "$garment" --identity "$OUTPUT_ROOT/identity.json" >/dev/null
}
run_stage 1 top_long Top_Long_Seen_0
if ! "$PYTHON_BIN" - "$OUTPUT_ROOT/result.json" <<'PY'
import json, sys
from pathlib import Path
rows=json.loads(Path(sys.argv[1]).read_text()).get("attempts")
raise SystemExit(0 if isinstance(rows,list) and len(rows)==2 and [row.get("success") for row in rows]==[True,True] else 1)
PY
then
  "$PYTHON_BIN" "$SCRIPT_DIR/../scripts/verify_native_reference_evaluator_gate.py" verify-execution --result "$OUTPUT_ROOT/result.json" --bundle-root "$OUTPUT_ROOT" --receipt "$OUTPUT_ROOT/execution-receipt.json" >/dev/null
  exit 0
fi
run_stage 2 top_short Top_Short_Seen_0; run_stage 3 pant_long Pant_Long_Seen_0; run_stage 4 pant_short Pant_Short_Seen_0
validate_stage_integrity
"$PYTHON_BIN" "$SCRIPT_DIR/../scripts/verify_native_reference_evaluator_gate.py" verify-execution --result "$OUTPUT_ROOT/result.json" --bundle-root "$OUTPUT_ROOT" --receipt "$OUTPUT_ROOT/execution-receipt.json" >/dev/null
