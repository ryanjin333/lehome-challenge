#!/usr/bin/env bash
# Host boundary for the native-N1.5 candidate/reference focused gate.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly RUNTIME_REVISION="${LEHOME_OFFICIAL_RUNTIME_REVISION:-}"
readonly SOURCE_ROOT="${LEHOME_OFFICIAL_SOURCE_ROOT:-}"
readonly ASSETS_ROOT="${LEHOME_OFFICIAL_ASSETS_ROOT:-}"
readonly METADATA_ROOT="${LEHOME_OFFICIAL_METADATA_ROOT:-}"
readonly CANDIDATE_CHECKPOINT="${LEHOME_N15_CANDIDATE_CHECKPOINT:-}"
readonly CANDIDATE_IDENTITY_RECEIPT="${LEHOME_N15_CANDIDATE_IDENTITY_RECEIPT:-}"
readonly CANDIDATE_SANITIZED_CONFIG="${LEHOME_N15_CANDIDATE_SANITIZED_CONFIG_ROOT:-}"
readonly CANDIDATE_COMPATIBILITY_RECEIPT="${LEHOME_N15_CANDIDATE_COMPATIBILITY_RECEIPT:-}"
readonly REFERENCE_CHECKPOINT="${LEHOME_N15_REFERENCE_CHECKPOINT:-}"
readonly REFERENCE_SANITIZED_CONFIG="${LEHOME_N15_REFERENCE_SANITIZED_CONFIG_ROOT:-}"
readonly REFERENCE_COMPATIBILITY_RECEIPT="${LEHOME_N15_REFERENCE_COMPATIBILITY_RECEIPT:-}"
readonly NATIVE_RUNTIME_EVIDENCE="${LEHOME_N15_NATIVE_RUNTIME_EVIDENCE_ROOT:-}"
readonly NATIVE_DEPENDENCIES_ROOT="${LEHOME_N15_NATIVE_DEPENDENCIES_ROOT:-/mnt/lehome/reference-native/dependencies}"
readonly HF_CACHE_ROOT="${LEHOME_N15_FOCUSED_HF_CACHE_ROOT:-}"
readonly OUTPUT_ROOT="${LEHOME_N15_FOCUSED_OUTPUT_ROOT:-}"
readonly REPOSITORY="${LEHOME_N15_FOCUSED_REPOSITORY:-}"
readonly PUBLICATION_RECEIPT="${LEHOME_N15_FOCUSED_PUBLICATION_RECEIPT:-}"
readonly PROMOTION_RECEIPT="${LEHOME_N15_FOCUSED_PROMOTION_RECEIPT:-}"
readonly ROLLOUT_IMAGE_ID="sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7"
readonly REFERENCE_MATRIX="$REPO_ROOT/configs/eval_groot_n17_public96_reference.json"
readonly REFERENCE_MATRIX_SHA256="$REPO_ROOT/configs/eval_groot_n17_public96_reference.json.sha256"
# The organizer evaluator command is pinned to --device cpu and --seed 42.
readonly OFFICIAL_SIMULATOR_ARGUMENTS="--device cpu --seed 42"
readonly EVAL_CONTAINER="lehome-n15-focused-eval-$$"

fail() { printf 'error: %s\n' "$*" >&2; exit 2; }
require_directory() {
  [[ "$1" == /* && "$1" != *".."* && -d "$1" && ! -L "$1" ]] || fail "$2 is unavailable or unsafe"
}
require_file() {
  [[ "$1" == /* && "$1" != *".."* && -f "$1" && ! -L "$1" ]] || fail "$2 is unavailable or unsafe"
}
cleanup() {
  docker rm -f "$EVAL_CONTAINER" >/dev/null 2>&1 || true
  if [[ -n "${EVIDENCE_ROOT:-}" && -d "$EVIDENCE_ROOT" ]]; then
    find "$EVIDENCE_ROOT" -type f -delete 2>/dev/null || true
    rmdir "$EVIDENCE_ROOT" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

[[ $# -eq 0 ]] || fail "this wrapper accepts no positional arguments"
command -v docker >/dev/null 2>&1 || fail "docker is unavailable"
command -v git >/dev/null 2>&1 || fail "git is unavailable"
require_directory "$REPO_ROOT" "runtime repository"
require_directory "$SOURCE_ROOT" "official source checkout"
require_directory "$ASSETS_ROOT" "official asset checkout"
require_directory "$METADATA_ROOT" "metadata root"
require_directory "$CANDIDATE_CHECKPOINT" "candidate-n15 checkpoint"
require_file "$CANDIDATE_IDENTITY_RECEIPT" "candidate-n15 identity receipt"
CANDIDATE_TRAINING_ROOT="$(PYTHONPATH="$REPO_ROOT" python3 - "$CANDIDATE_IDENTITY_RECEIPT" "$CANDIDATE_CHECKPOINT" <<'PY'
import sys
from pathlib import Path
from rollout_appliance.native_reference_site.training_identity import validate_training_identity_receipt
from source.lehome.lehome.n15_reproduction import CONTRACT
value = validate_training_identity_receipt(
    Path(sys.argv[1]), expected_contract=CONTRACT,
    expected_pretrained_root=Path(sys.argv[2]),
)
print(value["training_root"])
PY
)"
require_directory "$CANDIDATE_TRAINING_ROOT" "candidate-n15 training output"
require_directory "$REFERENCE_CHECKPOINT" "reference-n15 checkpoint"
require_directory "$REFERENCE_SANITIZED_CONFIG" "reference-n15 compatibility view"
require_file "$REFERENCE_COMPATIBILITY_RECEIPT" "reference-n15 compatibility receipt"
require_directory "$NATIVE_RUNTIME_EVIDENCE" "native N1.5 runtime evidence"
require_directory "$NATIVE_DEPENDENCIES_ROOT" "native N1.5 dependency wheels"
require_directory "$HF_CACHE_ROOT" "pinned N1.5 Hugging Face cache"
require_file "$REFERENCE_MATRIX" "frozen reference matrix"
require_file "$REFERENCE_MATRIX_SHA256" "frozen reference matrix checksum"
[[ "$OUTPUT_ROOT" == /* && "$OUTPUT_ROOT" != *".."* && ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] \
  || fail "focused output root must be a new absolute path"
require_directory "$(dirname -- "$OUTPUT_ROOT")" "focused output parent"
[[ "$CANDIDATE_SANITIZED_CONFIG" == /* && "$CANDIDATE_SANITIZED_CONFIG" != *".."* \
   && ! -e "$CANDIDATE_SANITIZED_CONFIG" && ! -L "$CANDIDATE_SANITIZED_CONFIG" ]] \
  || fail "candidate compatibility view must be a new absolute path"
[[ "$CANDIDATE_COMPATIBILITY_RECEIPT" == /* && "$CANDIDATE_COMPATIBILITY_RECEIPT" != *".."* \
   && ! -e "$CANDIDATE_COMPATIBILITY_RECEIPT" && ! -L "$CANDIDATE_COMPATIBILITY_RECEIPT" ]] \
  || fail "candidate compatibility receipt must be a new absolute path"
[[ "$PUBLICATION_RECEIPT" == /* && "$PUBLICATION_RECEIPT" != *".."* && ! -e "$PUBLICATION_RECEIPT" ]] \
  || fail "publication receipt must be a new absolute path"
[[ "$PROMOTION_RECEIPT" == /* && "$PROMOTION_RECEIPT" != *".."* && ! -e "$PROMOTION_RECEIPT" ]] \
  || fail "promotion receipt must be a new absolute path"
[[ "$(dirname -- "$PUBLICATION_RECEIPT")" == "$(dirname -- "$OUTPUT_ROOT")" \
   && "$(dirname -- "$PROMOTION_RECEIPT")" == "$(dirname -- "$OUTPUT_ROOT")" \
   && "$(dirname -- "$CANDIDATE_SANITIZED_CONFIG")" == "$(dirname -- "$OUTPUT_ROOT")" \
   && "$(dirname -- "$CANDIDATE_COMPATIBILITY_RECEIPT")" == "$(dirname -- "$OUTPUT_ROOT")" ]] \
  || fail "focused output and generated receipts/views must share one mounted parent"
[[ "$REPOSITORY" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || fail "public dataset repository is invalid"
[[ "$RUNTIME_REVISION" =~ ^[0-9a-f]{40}$ ]] || fail "runtime revision must be exact"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$RUNTIME_REVISION" ]] || fail "runtime revision mismatch"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || fail "runtime checkout is modified"
for wheel_and_digest in \
  "peft-0.18.1-py3-none-any.whl:0bf06847a3551e3019fc58c440cffc9a6b73e6e2962c95b52e224f77bbdb50f1" \
  "flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl:cd1a45ebfc1731a13e55ad68e0c9ad92390ddfffba306f9222be67c6d5a805af" \
  "dm_tree-0.1.9-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl:294dc1cecf87552a45cdd5ddb215e7f5295a5a47c46f1f0a0463c3dd02a527d7" \
  "qwen_vl_utils-0.0.14-py3-none-any.whl:5e28657bfd031e56bd447c5901b58ddfc3835285ed100f4c56580e0ade054e96" \
  "torchdiffeq-0.2.5-py3-none-any.whl:aa1db4bed13bd04952f28a53cdf4336d1ab60417c1d9698d7a239fec1cf2bcf8"; do
  wheel="${wheel_and_digest%%:*}"; digest="${wheel_and_digest##*:}"
  require_file "$NATIVE_DEPENDENCIES_ROOT/$wheel" "native N1.5 dependency wheel"
  [[ "$(sha256sum -- "$NATIVE_DEPENDENCIES_ROOT/$wheel" | awk '{print $1}')" == "$digest" ]] \
    || fail "native N1.5 dependency wheel digest mismatch"
done

EVIDENCE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/lehome-n15-focused.XXXXXX")"
python3 - "$REPO_ROOT" "$RUNTIME_REVISION" "$EVIDENCE_ROOT/runtime-identity.json" <<'PY'
import hashlib, json, os, stat, sys
from pathlib import Path
root, revision, output = Path(sys.argv[1]).resolve(strict=True), sys.argv[2], Path(sys.argv[3])
digest = hashlib.sha256()
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root)
    if ".git" in relative.parts or path.is_dir():
        continue
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit("runtime contains an unsafe entry")
    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    digest.update(relative.as_posix().encode() + b"\0" + file_digest.encode("ascii") + b"\n")
payload = {"revision": revision, "tree_sha256": digest.hexdigest()}
with output.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, separators=(",", ":")); stream.write("\n")
os.chmod(output, 0o444)
PY

docker image inspect -- "$ROLLOUT_IMAGE_ID" >"$EVIDENCE_ROOT/image-inspect.json"
python3 - "$EVIDENCE_ROOT/image-inspect.json" "$EVIDENCE_ROOT/rollout-image.json" "$ROLLOUT_IMAGE_ID" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
raw_path, output, reference = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
raw = raw_path.read_bytes(); rows = json.loads(raw)
if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("Id") != reference:
    raise SystemExit("rollout image identity mismatch")
payload = {"kind":"lehome_official_image_inspection_v1","reference":reference,"image_id":reference,"repo_digests":rows[0].get("RepoDigests") or [],"docker_inspect_sha256":hashlib.sha256(raw).hexdigest()}
with output.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, separators=(",", ":")); stream.write("\n")
os.chmod(output, 0o444)
PY

docker run --rm --pull never --gpus all --entrypoint /opt/lehome-challenge/.venv/bin/python \
  "$ROLLOUT_IMAGE_ID" -c 'import json,torch; print(json.dumps({"cuda_available":torch.cuda.is_available(),"cuda_device_count":torch.cuda.device_count(),"cuda_runtime":torch.version.cuda,"cuda_device_name":torch.cuda.get_device_name(0)}))' \
  >"$EVIDENCE_ROOT/cuda-runtime.json"
chmod 0444 "$EVIDENCE_ROOT/cuda-runtime.json"

mounts=(
  --mount "type=bind,src=$REPO_ROOT,dst=/runtime,readonly"
  --mount "type=bind,src=$SOURCE_ROOT,dst=/official/lehome,readonly"
  --mount "type=bind,src=$ASSETS_ROOT,dst=/official/assets,readonly"
  --mount "type=bind,src=$ASSETS_ROOT,dst=/official/lehome/Assets,readonly"
  --mount "type=bind,src=$METADATA_ROOT,dst=$METADATA_ROOT,readonly"
  --mount "type=bind,src=$CANDIDATE_CHECKPOINT,dst=$CANDIDATE_CHECKPOINT,readonly"
  --mount "type=bind,src=$CANDIDATE_TRAINING_ROOT,dst=$CANDIDATE_TRAINING_ROOT,readonly"
  --mount "type=bind,src=$CANDIDATE_IDENTITY_RECEIPT,dst=$CANDIDATE_IDENTITY_RECEIPT,readonly"
  --mount "type=bind,src=$REFERENCE_CHECKPOINT,dst=$REFERENCE_CHECKPOINT,readonly"
  --mount "type=bind,src=$REFERENCE_SANITIZED_CONFIG,dst=$REFERENCE_SANITIZED_CONFIG,readonly"
  --mount "type=bind,src=$REFERENCE_COMPATIBILITY_RECEIPT,dst=$REFERENCE_COMPATIBILITY_RECEIPT,readonly"
  --mount "type=bind,src=$NATIVE_RUNTIME_EVIDENCE,dst=$NATIVE_RUNTIME_EVIDENCE,readonly"
  --mount "type=bind,src=$NATIVE_DEPENDENCIES_ROOT,dst=$NATIVE_DEPENDENCIES_ROOT,readonly"
  --mount "type=bind,src=$HF_CACHE_ROOT,dst=/official/n15-hf-cache,readonly"
  --mount "type=bind,src=$EVIDENCE_ROOT,dst=/official/evidence,readonly"
  --mount "type=bind,src=$(dirname -- "$OUTPUT_ROOT"),dst=$(dirname -- "$OUTPUT_ROOT")"
)

CONTAINER_SCRIPT='set -euo pipefail
uv pip install --offline --no-deps --python /opt/lehome-challenge/.venv/bin/python \
  "$NATIVE_DEPENDENCIES_ROOT/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl" \
  "$NATIVE_DEPENDENCIES_ROOT/dm_tree-0.1.9-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
  "$NATIVE_DEPENDENCIES_ROOT/qwen_vl_utils-0.0.14-py3-none-any.whl" \
  "$NATIVE_DEPENDENCIES_ROOT/torchdiffeq-0.2.5-py3-none-any.whl" >/dev/null
/opt/lehome-challenge/.venv/bin/python /runtime/scripts/run_official_lehome_comparison.py \
  prepare-n15-candidate-compatibility \
  --candidate-checkpoint "$CANDIDATE_CHECKPOINT" \
  --training-identity-receipt "$CANDIDATE_IDENTITY_RECEIPT" \
  --sanitized-config-root "$CANDIDATE_SANITIZED_CONFIG" \
  --compatibility-receipt "$CANDIDATE_COMPATIBILITY_RECEIPT"
/isaac-sim/python.sh -m scripts.run_official_lehome_comparison run-n15-focused \
  --profile n15-focused \
  --source-root /official/lehome \
  --canonical-assets-root /official/assets \
  --metadata-root "$METADATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --candidate-checkpoint "$CANDIDATE_CHECKPOINT" \
  --candidate-identity-receipt "$CANDIDATE_IDENTITY_RECEIPT" \
  --candidate-sanitized-config-root "$CANDIDATE_SANITIZED_CONFIG" \
  --candidate-compatibility-receipt "$CANDIDATE_COMPATIBILITY_RECEIPT" \
  --reference-checkpoint "$REFERENCE_CHECKPOINT" \
  --reference-sanitized-config-root "$REFERENCE_SANITIZED_CONFIG" \
  --reference-compatibility-receipt "$REFERENCE_COMPATIBILITY_RECEIPT" \
  --reference-matrix /runtime/configs/eval_groot_n17_public96_reference.json \
  --reference-matrix-sha256 /runtime/configs/eval_groot_n17_public96_reference.json.sha256 \
  --native-site-root /runtime/rollout_appliance/native_reference_site \
  --native-runtime-evidence-root "$NATIVE_RUNTIME_EVIDENCE" \
  --runtime-root /runtime --runtime-revision "$RUNTIME_REVISION" \
  --runtime-identity-receipt /official/evidence/runtime-identity.json \
  --rollout-image-receipt /official/evidence/rollout-image.json \
  --cuda-receipt /official/evidence/cuda-runtime.json \
  --isaaclab-root /opt/lehome-challenge/third_party/IsaacLab/source/isaaclab \
  --isaaclab-tasks-root /opt/lehome-challenge/third_party/IsaacLab/source/isaaclab_tasks \
  --python-bin /opt/lehome-challenge/.venv/bin/python'

# Both policies use the same native LeRobot adapter. The Python harness executes
# candidate-n15 completely before reference-n15 and never starts a policy server.
docker run --rm --pull never --gpus all --init --network host --shm-size=8g \
  --name "$EVAL_CONTAINER" "${mounts[@]}" \
  --env PYTHONPATH=/runtime/source/lehome:/runtime \
  --env GIT_CONFIG_COUNT=2 \
  --env GIT_CONFIG_KEY_0=safe.directory --env GIT_CONFIG_VALUE_0=/official/lehome \
  --env GIT_CONFIG_KEY_1=safe.directory --env GIT_CONFIG_VALUE_1=/official/assets \
  --env "NATIVE_DEPENDENCIES_ROOT=$NATIVE_DEPENDENCIES_ROOT" \
  --env "RUNTIME_REVISION=$RUNTIME_REVISION" \
  --env "METADATA_ROOT=$METADATA_ROOT" --env "OUTPUT_ROOT=$OUTPUT_ROOT" \
  --env "CANDIDATE_CHECKPOINT=$CANDIDATE_CHECKPOINT" \
  --env "CANDIDATE_IDENTITY_RECEIPT=$CANDIDATE_IDENTITY_RECEIPT" \
  --env "CANDIDATE_SANITIZED_CONFIG=$CANDIDATE_SANITIZED_CONFIG" \
  --env "CANDIDATE_COMPATIBILITY_RECEIPT=$CANDIDATE_COMPATIBILITY_RECEIPT" \
  --env "REFERENCE_CHECKPOINT=$REFERENCE_CHECKPOINT" \
  --env "REFERENCE_SANITIZED_CONFIG=$REFERENCE_SANITIZED_CONFIG" \
  --env "REFERENCE_COMPATIBILITY_RECEIPT=$REFERENCE_COMPATIBILITY_RECEIPT" \
  --env "NATIVE_RUNTIME_EVIDENCE=$NATIVE_RUNTIME_EVIDENCE" \
  --env HF_HOME=/tmp/lehome-n15-hf-home \
  --env HF_HUB_CACHE=/official/n15-hf-cache/hub \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env PYNPUT_BACKEND=dummy \
  --entrypoint bash "$ROLLOUT_IMAGE_ID" -lc "$CONTAINER_SCRIPT"

# Publication is explicit. Its anonymous byte readback must finish before the
# separate promotion verifier is allowed to return pass.
docker run --rm --pull never \
  --mount "type=bind,src=$REPO_ROOT,dst=/runtime,readonly" \
  --mount "type=bind,src=$(dirname -- "$OUTPUT_ROOT"),dst=$(dirname -- "$OUTPUT_ROOT")" \
  --env HF_TOKEN="${HF_TOKEN:-}" --entrypoint python3 "$ROLLOUT_IMAGE_ID" \
  /runtime/scripts/run_official_lehome_comparison.py publish \
    --receipt "$OUTPUT_ROOT/comparison-receipt.json" \
    --repository "$REPOSITORY" \
    --publication-receipt "$PUBLICATION_RECEIPT"

docker run --rm --pull never \
  --mount "type=bind,src=$REPO_ROOT,dst=/runtime,readonly" \
  --mount "type=bind,src=$(dirname -- "$OUTPUT_ROOT"),dst=$(dirname -- "$OUTPUT_ROOT")" \
  --entrypoint python3 "$ROLLOUT_IMAGE_ID" \
  /runtime/scripts/run_official_lehome_comparison.py verify-n15-focused \
    --receipt "$OUTPUT_ROOT/comparison-receipt.json" \
    --publication-receipt "$PUBLICATION_RECEIPT" \
    --promotion-receipt "$PROMOTION_RECEIPT"

python3 - "$PROMOTION_RECEIPT" <<'PY'
import json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("status") != "pass" or value.get("publication_readback_verified") is not True:
    raise SystemExit("focused promotion did not pass after anonymous_byte_readback_verified")
PY
printf '%s\n' "$PROMOTION_RECEIPT"
