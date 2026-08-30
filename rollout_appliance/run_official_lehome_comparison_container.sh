#!/usr/bin/env bash
# Host-side boundary for the pinned official LeHome smoke/full comparison.
set -euo pipefail

readonly MODE="${1:-}"
readonly SOURCE_REVISION="a805ad2f7ab52a4583066fc4ee5180459a7f9d15"
readonly ASSET_REVISION="bea65fd960ad5a1bb3bd3fa77164b28001c08ef9"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly RUNTIME_REVISION="${LEHOME_OFFICIAL_RUNTIME_REVISION:-}"
readonly OFFICIAL_SOURCE_ROOT="${LEHOME_OFFICIAL_SOURCE_ROOT:-}"
readonly OFFICIAL_ASSETS_ROOT="${LEHOME_OFFICIAL_ASSETS_ROOT:-}"
readonly METADATA_ROOT="${LEHOME_OFFICIAL_METADATA_ROOT:-}"
readonly N17_CHECKPOINT_ROOT="${LEHOME_OFFICIAL_N17_CHECKPOINT_ROOT:-}"
readonly N17_IDENTITY_RECEIPT="${LEHOME_OFFICIAL_N17_IDENTITY_RECEIPT:-}"
readonly N17_BASE_MODEL_ROOT="/mnt/lehome/cache/models/nvidia/Cosmos-Reason2-2B"
readonly CONTROLLER_WIRE_ROOT="/mnt/lehome/eval/pydeps"
readonly COMPETITOR_CHECKPOINT_ROOT="${LEHOME_OFFICIAL_COMPETITOR_CHECKPOINT_ROOT:-}"
readonly SANITIZED_CONFIG_ROOT="${LEHOME_OFFICIAL_SANITIZED_CONFIG_ROOT:-}"
readonly COMPATIBILITY_RECEIPT="${LEHOME_OFFICIAL_COMPATIBILITY_RECEIPT:-}"
readonly SMOKE_RECEIPT="${LEHOME_OFFICIAL_SMOKE_RECEIPT:-}"
readonly OUTPUT_ROOT="${LEHOME_OFFICIAL_OUTPUT_ROOT:-}"
readonly ROLLOUT_IMAGE_ID="sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7"
readonly POLICY_IMAGE_ID="ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746"
readonly PEFT_WHEEL_PATH="/mnt/lehome/reference-native/dependencies/peft-0.18.1-py3-none-any.whl"
readonly PEFT_WHEEL_SHA256="0bf06847a3551e3019fc58c440cffc9a6b73e6e2962c95b52e224f77bbdb50f1"
readonly FLASH_ATTENTION_WHEEL_PATH="/mnt/lehome/reference-native/dependencies/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
readonly FLASH_ATTENTION_WHEEL_SHA256="cd1a45ebfc1731a13e55ad68e0c9ad92390ddfffba306f9222be67c6d5a805af"
readonly DM_TREE_WHEEL_PATH="/mnt/lehome/reference-native/dependencies/dm_tree-0.1.9-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
readonly DM_TREE_WHEEL_SHA256="294dc1cecf87552a45cdd5ddb215e7f5295a5a47c46f1f0a0463c3dd02a527d7"
readonly QWEN_VL_UTILS_WHEEL_PATH="/mnt/lehome/reference-native/dependencies/qwen_vl_utils-0.0.14-py3-none-any.whl"
readonly QWEN_VL_UTILS_WHEEL_SHA256="5e28657bfd031e56bd447c5901b58ddfc3835285ed100f4c56580e0ade054e96"
readonly TORCHDIFFEQ_WHEEL_PATH="/mnt/lehome/reference-native/dependencies/torchdiffeq-0.2.5-py3-none-any.whl"
readonly TORCHDIFFEQ_WHEEL_SHA256="aa1db4bed13bd04952f28a53cdf4336d1ab60417c1d9698d7a239fec1cf2bcf8"
readonly POLICY_PORT="15571"
readonly BRIDGE_PORT="18080"
readonly POLICY_TOKEN_ENV="LEHOME_OFFICIAL_POLICY_API_TOKEN"
readonly POLICY_CONTAINER="lehome-official-policy-$$"
readonly EVAL_CONTAINER="lehome-official-eval-$$"
# The Python harness emits the organizer command with exactly --device cpu.
readonly OFFICIAL_SIMULATOR_ARGUMENTS="--device cpu"

fail() { printf 'error: %s\n' "$*" >&2; exit 2; }
require_directory() {
  [[ "$1" == /* && "$1" != *".."* && -d "$1" && ! -L "$1" ]] || fail "$2 is unavailable or unsafe"
}
require_file() {
  [[ "$1" == /* && "$1" != *".."* && -f "$1" && ! -L "$1" ]] || fail "$2 is unavailable or unsafe"
}
cleanup() {
  docker rm -f "$EVAL_CONTAINER" >/dev/null 2>&1 || true
  docker rm -f "$POLICY_CONTAINER" >/dev/null 2>&1 || true
  if [[ -n "${SMOKE_VIEW_ROOT:-}" && -d "$SMOKE_VIEW_ROOT" ]]; then
    find "$SMOKE_VIEW_ROOT" -type f -delete 2>/dev/null || true
    rmdir "$SMOKE_VIEW_ROOT"/objects/Challenge_Garment/Release 2>/dev/null || true
    rmdir "$SMOKE_VIEW_ROOT"/objects/Challenge_Garment 2>/dev/null || true
    rmdir "$SMOKE_VIEW_ROOT"/objects 2>/dev/null || true
    rmdir "$SMOKE_VIEW_ROOT" 2>/dev/null || true
  fi
  if [[ -n "${EVIDENCE_ROOT:-}" && -d "$EVIDENCE_ROOT" ]]; then
    find "$EVIDENCE_ROOT" -type f -delete 2>/dev/null || true
    rmdir "$EVIDENCE_ROOT"/competitor-runtime 2>/dev/null || true
    rmdir "$EVIDENCE_ROOT" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

[[ "$MODE" == smoke || "$MODE" == full ]] || fail "mode must be smoke or full"
[[ $# -eq 1 ]] || fail "expected exactly one mode argument"
command -v docker >/dev/null 2>&1 || fail "docker is unavailable"
command -v git >/dev/null 2>&1 || fail "git is unavailable"
require_directory "$OFFICIAL_SOURCE_ROOT" "official source checkout"
require_directory "$OFFICIAL_ASSETS_ROOT" "official asset checkout"
require_directory "$METADATA_ROOT" "policy metadata root"
require_directory "$N17_CHECKPOINT_ROOT" "N1.7 checkpoint"
require_file "$N17_IDENTITY_RECEIPT" "N1.7 checkpoint identity receipt"
require_directory "$N17_BASE_MODEL_ROOT" "N1.7 base model"
require_directory "$CONTROLLER_WIRE_ROOT" "N1.7 controller wire dependencies"
for base_model_file_and_digest in \
  "model.safetensors:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0" \
  "config.json:bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b" \
  "preprocessor_config.json:27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516" \
  "tokenizer.json:a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7" \
  "tokenizer_config.json:c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399" \
  "chat_template.json:6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4"; do
  base_model_file="${base_model_file_and_digest%%:*}"
  base_model_digest="${base_model_file_and_digest##*:}"
  require_file "$N17_BASE_MODEL_ROOT/$base_model_file" "N1.7 base-model file"
  [[ "$(sha256sum -- "$N17_BASE_MODEL_ROOT/$base_model_file" | awk '{print $1}')" == "$base_model_digest" ]] \
    || fail "N1.7 base-model file digest mismatch"
done
require_directory "$COMPETITOR_CHECKPOINT_ROOT" "competitor checkpoint"
require_directory "$SANITIZED_CONFIG_ROOT" "competitor sanitized config view"
require_file "$COMPATIBILITY_RECEIPT" "competitor compatibility receipt"
if [[ "$MODE" == full ]]; then
  require_file "$SMOKE_RECEIPT" "full comparison requires a valid smoke receipt"
else
  [[ -z "$SMOKE_RECEIPT" ]] || fail "smoke mode does not accept a smoke receipt"
fi
require_directory "$REPO_ROOT" "runtime repository"
[[ "$RUNTIME_REVISION" =~ ^[0-9a-f]{40}$ ]] || fail "LEHOME_OFFICIAL_RUNTIME_REVISION must be an exact revision"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$RUNTIME_REVISION" ]] || fail "reviewed runtime revision mismatch"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || fail "reviewed runtime checkout is modified"
[[ "$OUTPUT_ROOT" == /* && "$OUTPUT_ROOT" != *".."* && ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] \
  || fail "output root must be a new absolute path"
require_directory "$(dirname -- "$OUTPUT_ROOT")" "output parent"
[[ "$(git -C "$OFFICIAL_SOURCE_ROOT" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
  || fail "official source revision mismatch"
[[ "$(git -C "$OFFICIAL_ASSETS_ROOT" rev-parse HEAD)" == "$ASSET_REVISION" ]] \
  || fail "official asset revision mismatch"
[[ -z "$(git -C "$OFFICIAL_SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] \
  || fail "official source checkout is modified"
[[ -z "$(git -C "$OFFICIAL_ASSETS_ROOT" status --porcelain --untracked-files=all)" ]] \
  || fail "official asset checkout is modified"
for wheel_and_digest in "$PEFT_WHEEL_PATH:$PEFT_WHEEL_SHA256" "$FLASH_ATTENTION_WHEEL_PATH:$FLASH_ATTENTION_WHEEL_SHA256" "$DM_TREE_WHEEL_PATH:$DM_TREE_WHEEL_SHA256" "$QWEN_VL_UTILS_WHEEL_PATH:$QWEN_VL_UTILS_WHEEL_SHA256" "$TORCHDIFFEQ_WHEEL_PATH:$TORCHDIFFEQ_WHEEL_SHA256"; do
  wheel="${wheel_and_digest%%:*}"; digest="${wheel_and_digest##*:}"
  require_file "$wheel" "native-reference dependency wheel"
  [[ "$(sha256sum -- "$wheel" | awk '{print $1}')" == "$digest" ]] \
    || fail "native-reference dependency wheel digest mismatch"
done

POLICY_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
[[ ${#POLICY_TOKEN} -eq 64 ]] || fail "could not create an ephemeral policy token"
EVIDENCE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/lehome-official-evidence.XXXXXX")"
python3 - "$REPO_ROOT" "$RUNTIME_REVISION" "$EVIDENCE_ROOT/runtime-identity.json" <<'PY'
import hashlib, json, os, stat, sys
from pathlib import Path
root, revision, output = Path(sys.argv[1]).resolve(strict=True), sys.argv[2], Path(sys.argv[3])
digest=hashlib.sha256()
for path in sorted(root.rglob("*")):
    relative=path.relative_to(root)
    if ".git" in relative.parts or path.is_dir(): continue
    metadata=path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode): raise SystemExit("reviewed runtime tree contains an unsafe entry")
    file_digest=hashlib.sha256(path.read_bytes()).hexdigest()
    digest.update(relative.as_posix().encode()+b"\0"+file_digest.encode("ascii")+b"\n")
payload={"revision":revision,"tree_sha256":digest.hexdigest()}
fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,"w",encoding="utf-8") as stream: json.dump(payload,stream,sort_keys=True,separators=(",",":")); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
PY

capture_image_receipt() {
  local reference="$1" output="$2" mode="$3" raw
  raw="$EVIDENCE_ROOT/image-inspect-$mode.json"
  docker image inspect -- "$reference" >"$raw" || fail "$mode image inspection failed"
  python3 - "$raw" "$output" "$reference" "$mode" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
raw_path, output, reference, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
raw = raw_path.read_bytes(); rows = json.loads(raw)
if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict): raise SystemExit("image inspection schema is invalid")
row = rows[0]; image_id = row.get("Id"); digests = row.get("RepoDigests") or []
if not isinstance(image_id, str) or not image_id.startswith("sha256:") or not isinstance(digests, list) or any(not isinstance(item, str) for item in digests): raise SystemExit("image inspection identity is invalid")
if mode == "rollout" and image_id != reference: raise SystemExit("rollout image ID mismatch")
if mode == "policy" and reference not in digests: raise SystemExit("policy image RepoDigest mismatch")
payload={"kind":"lehome_official_image_inspection_v1","reference":reference,"image_id":image_id,"repo_digests":digests,"docker_inspect_sha256":hashlib.sha256(raw).hexdigest()}
fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,"w",encoding="utf-8") as stream: json.dump(payload,stream,sort_keys=True,separators=(",",":")); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
PY
  [[ -f "$output" && ! -L "$output" ]] || fail "$mode image inspection receipt was not created"
}
capture_image_receipt "$ROLLOUT_IMAGE_ID" "$EVIDENCE_ROOT/rollout-image.json" rollout
capture_image_receipt "$POLICY_IMAGE_ID" "$EVIDENCE_ROOT/policy-image.json" policy

docker run --rm --detach --pull never --gpus all --init --network host \
  --name "$POLICY_CONTAINER" \
  --workdir /cache/models \
  --mount "type=bind,src=$N17_CHECKPOINT_ROOT,dst=$N17_CHECKPOINT_ROOT,readonly" \
  --mount "type=bind,src=$N17_BASE_MODEL_ROOT,dst=/cache/models/nvidia/Cosmos-Reason2-2B,readonly" \
  --mount "type=bind,src=$REPO_ROOT,dst=/runtime,readonly" \
  --mount "type=bind,src=$EVIDENCE_ROOT,dst=/evidence" \
  --env "$POLICY_TOKEN_ENV=$POLICY_TOKEN" \
  --env PYTHONPATH=/runtime/source/lehome:/runtime:/opt/isaac-groot \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --entrypoint /opt/runtime/bin/python \
  "$POLICY_IMAGE_ID" \
  -m scripts.run_groot_n17_public96_policy_server \
    --model-path "$N17_CHECKPOINT_ROOT" \
    --host 127.0.0.1 \
    --port "$POLICY_PORT" \
    --api-token-env "$POLICY_TOKEN_ENV" \
    --device cuda:0 \
    --seed 42 \
    --readiness-receipt /evidence/policy-server-readiness.json >/dev/null

for _ in $(seq 1 180); do
  [[ -f "$EVIDENCE_ROOT/policy-server-readiness.json" ]] && break
  docker inspect -f '{{.State.Running}}' "$POLICY_CONTAINER" 2>/dev/null | grep -qx true \
    || fail "N1.7 policy server exited before readiness"
  sleep 1
done
[[ -f "$EVIDENCE_ROOT/policy-server-readiness.json" ]] || fail "N1.7 policy server readiness timed out"
docker logs "$POLICY_CONTAINER" >"$EVIDENCE_ROOT/policy-server-startup.log" 2>&1 \
  || fail "policy server log capture failed"
docker exec -i "$POLICY_CONTAINER" /opt/runtime/bin/python - <<'PY' >"$EVIDENCE_ROOT/cuda-runtime.json"
import json, torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1 or not torch.version.cuda: raise SystemExit("CUDA unavailable")
print(json.dumps({"cuda_available":True,"cuda_device_count":torch.cuda.device_count(),"cuda_runtime":torch.version.cuda,"cuda_device_name":torch.cuda.get_device_name(0)},sort_keys=True,separators=(",",":")))
PY
chmod 0444 "$EVIDENCE_ROOT/policy-server-readiness.json" "$EVIDENCE_ROOT/policy-server-startup.log" "$EVIDENCE_ROOT/cuda-runtime.json"

declare -a mounts=(
  --mount "type=bind,src=$OFFICIAL_SOURCE_ROOT,dst=/official/lehome,readonly"
  --mount "type=bind,src=$OFFICIAL_ASSETS_ROOT,dst=/official/assets,readonly"
  --mount "type=bind,src=$OFFICIAL_ASSETS_ROOT,dst=/official/assets-view,readonly"
  --mount "type=bind,src=$METADATA_ROOT,dst=/official/metadata,readonly"
  --mount "type=bind,src=$N17_CHECKPOINT_ROOT,dst=$N17_CHECKPOINT_ROOT,readonly"
  --mount "type=bind,src=$N17_IDENTITY_RECEIPT,dst=$N17_IDENTITY_RECEIPT,readonly"
  --mount "type=bind,src=$COMPETITOR_CHECKPOINT_ROOT,dst=$COMPETITOR_CHECKPOINT_ROOT,readonly"
  --mount "type=bind,src=$SANITIZED_CONFIG_ROOT,dst=$SANITIZED_CONFIG_ROOT,readonly"
  --mount "type=bind,src=$COMPATIBILITY_RECEIPT,dst=$COMPATIBILITY_RECEIPT,readonly"
  --mount "type=bind,src=$EVIDENCE_ROOT,dst=/official/evidence"
  --mount "type=bind,src=$REPO_ROOT,dst=/runtime,readonly"
  --mount "type=bind,src=$CONTROLLER_WIRE_ROOT,dst=/official/wire,readonly"
  --mount "type=bind,src=$PEFT_WHEEL_PATH,dst=$PEFT_WHEEL_PATH,readonly"
  --mount "type=bind,src=$FLASH_ATTENTION_WHEEL_PATH,dst=$FLASH_ATTENTION_WHEEL_PATH,readonly"
  --mount "type=bind,src=$DM_TREE_WHEEL_PATH,dst=$DM_TREE_WHEEL_PATH,readonly"
  --mount "type=bind,src=$QWEN_VL_UTILS_WHEEL_PATH,dst=$QWEN_VL_UTILS_WHEEL_PATH,readonly"
  --mount "type=bind,src=$TORCHDIFFEQ_WHEEL_PATH,dst=$TORCHDIFFEQ_WHEEL_PATH,readonly"
  --mount "type=bind,src=$(dirname -- "$OUTPUT_ROOT"),dst=/official/output"
)
if [[ "$MODE" == full ]]; then
  mounts+=(--mount "type=bind,src=$SMOKE_RECEIPT,dst=$SMOKE_RECEIPT,readonly")
fi

if [[ "$MODE" == smoke ]]; then
  SMOKE_VIEW_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/lehome-official-smoke.XXXXXX")"
  mkdir -p "$SMOKE_VIEW_ROOT/objects/Challenge_Garment/Release"
  printf '%s\n' Top_Long_Seen_0 >"$SMOKE_VIEW_ROOT/objects/Challenge_Garment/Release/Release_test_list.txt"
  chmod 0444 "$SMOKE_VIEW_ROOT/objects/Challenge_Garment/Release/Release_test_list.txt"
  mounts+=(--mount "type=bind,src=$SMOKE_VIEW_ROOT/objects/Challenge_Garment/Release/Release_test_list.txt,dst=/official/assets-view/objects/Challenge_Garment/Release/Release_test_list.txt,readonly")
fi

readonly CONTAINER_OUTPUT="/official/output/$(basename -- "$OUTPUT_ROOT")"
readonly CONTAINER_EVALUATION_ASSETS="$([[ "$MODE" == full ]] && printf /official/assets || printf /official/assets-view)"
readonly CONTAINER_SCRIPT='set -euo pipefail
bridge_pid=""
cleanup_inner() { if [[ -n "$bridge_pid" ]]; then kill "$bridge_pid" >/dev/null 2>&1 || true; wait "$bridge_pid" >/dev/null 2>&1 || true; fi; }
trap cleanup_inner EXIT INT TERM
readonly PYTHON_BIN=/opt/lehome-challenge/.venv/bin/python
readonly COMPETITOR_EVIDENCE=/official/evidence/competitor-runtime
smoke_args=()
if [[ '"$MODE"' == full ]]; then smoke_args=(--smoke-receipt "$LEHOME_OFFICIAL_SMOKE_RECEIPT"); fi
mkdir --mode=0700 -- "$COMPETITOR_EVIDENCE"
git config --global --add safe.directory /official/lehome
git config --global --add safe.directory /official/assets
PYTHONSAFEPATH=1 "$PYTHON_BIN" /runtime/scripts/verify_native_reference_evaluator_gate.py validate-peft-overlay >/dev/null
PYTHONSAFEPATH=1 "$PYTHON_BIN" /runtime/scripts/verify_native_reference_evaluator_gate.py validate-flash-attention-overlay >/dev/null
PYTHONSAFEPATH=1 "$PYTHON_BIN" /runtime/scripts/verify_native_reference_evaluator_gate.py validate-public-pyproject-dependencies-overlay >/dev/null
PYTHONSAFEPATH=1 "$PYTHON_BIN" /runtime/scripts/verify_native_reference_evaluator_gate.py prepare-peft-overlay --receipt "$COMPETITOR_EVIDENCE/peft-overlay-receipt.json" >/dev/null
PYTHONSAFEPATH=1 "$PYTHON_BIN" /runtime/scripts/verify_native_reference_evaluator_gate.py prepare-flash-attention-overlay --receipt "$COMPETITOR_EVIDENCE/flash-attention-overlay-receipt.json" >/dev/null
PYTHONSAFEPATH=1 "$PYTHON_BIN" /runtime/scripts/verify_native_reference_evaluator_gate.py prepare-public-pyproject-dependencies-overlay --receipt "$COMPETITOR_EVIDENCE/public-pyproject-dependencies-overlay-receipt.json" >/dev/null
uv pip install --offline --no-deps --python /opt/lehome-challenge/.venv/bin/python '"$FLASH_ATTENTION_WHEEL_PATH"' >/dev/null
uv pip install --offline --no-deps --reinstall --python /opt/lehome-challenge/.venv/bin/python '"$DM_TREE_WHEEL_PATH"' '"$QWEN_VL_UTILS_WHEEL_PATH"' '"$TORCHDIFFEQ_WHEEL_PATH"' >/dev/null
"$PYTHON_BIN" - "$COMPETITOR_EVIDENCE/flash-attention-runtime-receipt.json" <<'"'"'PY'"'"'
import json, os, sys
from pathlib import Path
import flash_attn, torch
from flash_attn import flash_attn_func
if torch.__version__ != "2.7.0+cu128" or torch.version.cuda != "12.8" or bool(torch._C._GLIBCXX_USE_CXX11_ABI) is not True: raise SystemExit("FlashAttention torch runtime mismatch")
if not torch.cuda.is_available() or list(torch.cuda.get_device_capability(0)) != [12, 0]: raise SystemExit("FlashAttention CUDA capability mismatch")
origin=str(Path(flash_attn.__file__).resolve()); expected="/opt/lehome-challenge/.venv/lib/python3.11/site-packages/flash_attn/__init__.py"
if origin != expected or str(flash_attn.__version__) != "2.8.3": raise SystemExit("FlashAttention runtime identity mismatch")
query=torch.randn((1,2,4,64),dtype=torch.float16,device="cuda"); output=flash_attn_func(query,query,query,causal=False); torch.cuda.synchronize()
if not bool(torch.isfinite(output).all().item()): raise SystemExit("FlashAttention kernel returned non-finite values")
payload={"schema_version":1,"kind":"lehome_native_reference_flash_attention_runtime_v1","torch_version":str(torch.__version__),"torch_cuda_version":torch.version.cuda,"torch_cxx11_abi":bool(torch._C._GLIBCXX_USE_CXX11_ABI),"cuda_capability":list(torch.cuda.get_device_capability(0)),"flash_attn_version":str(flash_attn.__version__),"flash_attn_origin":origin,"kernel":{"shape":[1,2,4,64],"dtype":"float16","finite":True}}
target=Path(sys.argv[1]); fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,"w",encoding="utf-8") as stream: json.dump(payload,stream,sort_keys=True,separators=(",",":")); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
PY
"$PYTHON_BIN" - "$COMPETITOR_EVIDENCE/public-pyproject-dependencies-runtime-receipt.json" <<'"'"'PY'"'"'
import importlib.metadata, json, os, sys
from pathlib import Path
import tree, qwen_vl_utils, torchdiffeq
from lerobot.policies.groot import groot_n1
root="/opt/lehome-challenge/.venv/lib/python3.11/site-packages"; expected={"tree":("dm-tree","0.1.9",f"{root}/tree/__init__.py"),"qwen_vl_utils":("qwen-vl-utils","0.0.14",f"{root}/qwen_vl_utils/__init__.py"),"torchdiffeq":("torchdiffeq","0.2.5",f"{root}/torchdiffeq/__init__.py")}; modules={"tree":tree,"qwen_vl_utils":qwen_vl_utils,"torchdiffeq":torchdiffeq}
for module,(distribution,version,origin) in ((modules[name], values) for name,values in expected.items()):
    if importlib.metadata.version(distribution) != version or str(Path(module.__file__).resolve()) != origin: raise SystemExit("public dependency runtime identity mismatch")
if tree.map_structure(lambda left,right:left+right,{"joint":1},{"joint":2}) != {"joint":3} or not callable(qwen_vl_utils.process_vision_info) or not callable(torchdiffeq.odeint) or groot_n1.tree is not tree: raise SystemExit("public dependency runtime probe failed")
payload={"schema_version":1,"kind":"lehome_native_reference_public_pyproject_dependencies_runtime_v1","tree_version":"0.1.9","tree_origin":expected["tree"][2],"tree_map_structure":True,"qwen_vl_utils_version":"0.0.14","qwen_vl_utils_origin":expected["qwen_vl_utils"][2],"qwen_vl_utils_process_vision_info":True,"torchdiffeq_version":"0.2.5","torchdiffeq_origin":expected["torchdiffeq"][2],"torchdiffeq_odeint":True,"groot_tree_origin":expected["tree"][2],"groot_tree_is_tree_module":True}
target=Path(sys.argv[1]); fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,"w",encoding="utf-8") as stream: json.dump(payload,stream,sort_keys=True,separators=(",",":")); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
PY
PYTHONDONTWRITEBYTECODE=1 PYNPUT_BACKEND=dummy "$PYTHON_BIN" - "$COMPETITOR_EVIDENCE/pynput-backend-receipt.json" <<'"'"'PY'"'"'
import json, os, sys
from pathlib import Path
from pynput import keyboard
listener_module=keyboard.Listener.__module__; key_module=keyboard.Key.__module__; x11_loaded=any(name == "Xlib" or name.startswith("Xlib.") for name in sys.modules)
if os.environ.get("PYNPUT_BACKEND") != "dummy" or listener_module != "pynput.keyboard._base" or key_module != "pynput.keyboard._base" or x11_loaded: raise SystemExit("pynput dummy backend probe failed")
payload={"schema_version":1,"kind":"lehome_native_reference_pynput_backend_v1","pynput_backend":"dummy","keyboard_listener_module":listener_module,"keyboard_key_module":key_module,"x11_modules_loaded":False,"keyboard_control_started":False}
target=Path(sys.argv[1]); fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,"w",encoding="utf-8") as stream: json.dump(payload,stream,sort_keys=True,separators=(",",":")); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
PY
chmod 0444 "$COMPETITOR_EVIDENCE"/*.json
PYTHONPATH=/official/wire "$PYTHON_BIN" - <<'"'"'PY'"'"'
import msgpack, zmq
if msgpack.__version__ != "1.1.0" or zmq.__version__ != "27.0.1":
    raise SystemExit("controller wire dependency identity mismatch")
PY
mkdir -p /official/bridge-log
LEHOME_NATIVE_REFERENCE_LOG_PROJECT_ROOT=/official/bridge-log LEHOME_NATIVE_REFERENCE_SOURCE_ROOT=/official/lehome PYTHONPATH=/runtime/rollout_appliance/native_reference_site:/official/wire:/runtime/source/lehome:/runtime:/opt/lehome-challenge/third_party/IsaacLab/source/isaaclab:/opt/lehome-challenge/third_party/IsaacLab/source/isaaclab_tasks /isaac-sim/python.sh -m scripts.serve_official_docker_policy_bridge \
  --listen-host 127.0.0.1 --listen-port '"$BRIDGE_PORT"' \
  --policy-server-endpoint tcp://127.0.0.1:'"$POLICY_PORT"' \
  --policy-server-token-env '"$POLICY_TOKEN_ENV"' \
  --policy-server-request-timeout 600 >/official/bridge-log/bridge.log 2>&1 &
bridge_pid=$!
bridge_ready=""
for _ in $(seq 1 180); do
  kill -0 "$bridge_pid" >/dev/null 2>&1 || { cat /official/bridge-log/bridge.log >&2; exit 2; }
  if kill -0 "$bridge_pid" >/dev/null 2>&1 && "$PYTHON_BIN" - <<'"'"'PY'"'"'
import json, urllib.request
request=urllib.request.Request("http://127.0.0.1:'"$BRIDGE_PORT"'/reset",data=b"{}",headers={"Content-Type":"application/json"},method="POST")
with urllib.request.urlopen(request, timeout=2) as response:
    assert json.load(response) == {"status":"ok"}
PY
  then bridge_ready=1; break; fi
  sleep 1
done
[[ "$bridge_ready" == 1 ]] || { cat /official/bridge-log/bridge.log >&2; echo "bridge failed readiness" >&2; exit 2; }
PYTHONPATH=/runtime/source/lehome:/runtime /isaac-sim/python.sh -m scripts.run_official_lehome_comparison run \
  --mode '"$MODE"' \
  --source-root /official/lehome \
  --canonical-assets-root /official/assets \
  --evaluation-assets-root '"$CONTAINER_EVALUATION_ASSETS"' \
  --metadata-root /official/metadata \
  --n17-checkpoint '"$N17_CHECKPOINT_ROOT"' \
  --n17-identity-receipt '"$N17_IDENTITY_RECEIPT"' \
  --reference-matrix /runtime/configs/eval_groot_n17_public96_reference.json \
  --reference-matrix-sha256 /runtime/configs/eval_groot_n17_public96_reference.json.sha256 \
  --output-root '"$CONTAINER_OUTPUT"' \
  --competitor-checkpoint '"$COMPETITOR_CHECKPOINT_ROOT"' \
  --sanitized-config-root '"$SANITIZED_CONFIG_ROOT"' \
  --compatibility-receipt '"$COMPATIBILITY_RECEIPT"' \
  --runtime-root /runtime \
  --runtime-revision '"$RUNTIME_REVISION"' \
  --runtime-identity-receipt /official/evidence/runtime-identity.json \
  --rollout-image-receipt /official/evidence/rollout-image.json \
  --policy-image-receipt /official/evidence/policy-image.json \
  --cuda-receipt /official/evidence/cuda-runtime.json \
  --policy-server-readiness-receipt /official/evidence/policy-server-readiness.json \
  --policy-server-startup-log /official/evidence/policy-server-startup.log \
  --competitor-runtime-evidence-root "$COMPETITOR_EVIDENCE" \
  --native-site-root /runtime/rollout_appliance/native_reference_site \
  --isaaclab-root /opt/lehome-challenge/third_party/IsaacLab/source/isaaclab \
  --isaaclab-tasks-root /opt/lehome-challenge/third_party/IsaacLab/source/isaaclab_tasks \
  --docker-url http://127.0.0.1:'"$BRIDGE_PORT"' \
  --python-bin /opt/lehome-challenge/.venv/bin/python \
  "${smoke_args[@]}"'

docker run --rm --pull never --gpus all --init --network host --shm-size=8g \
  --name "$EVAL_CONTAINER" \
  "${mounts[@]}" \
  --env "$POLICY_TOKEN_ENV=$POLICY_TOKEN" \
  --env PYTHONEXE=/opt/lehome-challenge/.venv/bin/python \
  --env "LEHOME_OFFICIAL_SMOKE_RECEIPT=$SMOKE_RECEIPT" \
  --entrypoint bash \
  "$ROLLOUT_IMAGE_ID" -lc "$CONTAINER_SCRIPT"

[[ -f "$OUTPUT_ROOT/comparison-receipt.json" && -f "$OUTPUT_ROOT/comparison-receipt.sha256.json" ]] \
  || fail "official comparison did not produce its immutable receipts"
printf '%s\n' "$OUTPUT_ROOT/comparison-receipt.json"
