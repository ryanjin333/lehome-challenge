#!/usr/bin/env bash
# Run the public GR00T N1.5 submission through its own, pinned evaluator path.
# This script runs *on* the already-running rollout VM.  It cannot create an
# image, VM, disk, or upload anything; it only creates a local result bundle.
set -euo pipefail

readonly SOURCE_REPOSITORY="theo-zhou/lehome-groot-submission-4"
readonly SOURCE_REVISION="d384fe00508acd96ab1c3c5dc265e08261f94b3b"
readonly EXPECTED_LEROBOT_VERSION="0.4.3"
readonly EXPECTED_CHECKPOINT_CONFIG_SHA256="b7c385bc57456eae603e929b84defb7e991194aade2aad70785e21991e37614c"
readonly EXPECTED_MODEL_SHA256="d8d91b6c11cb5aa18fe9a48e7da88eae0ec7e5a227a315ba67ee167d645cde76"
readonly EXPECTED_PREPROCESSOR_SHA256="a258dac8fa4e4e138990776e156cae36ae6cf172504a8c9e5f2d5864c9126009"
readonly EXPECTED_POSTPROCESSOR_SHA256="f9e18fa7da47e2b6d7ba3459236b140e28f834ce5640ba199be1412d50672fa7"
readonly EXPECTED_SOURCE_TREE_SHA256="eada9f80b0dda1428177fe4551efa8059fe85845d4db5b32bb673f88a50c6bb2"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

fail() { printf 'error: %s\n' "$*" >&2; exit 2; }

require_absolute_directory() {
  local path="$1" label="$2"
  [[ "$path" == /* && "$path" != *".."* ]] || fail "$label must be an absolute path without traversal"
  [[ -d "$path" && ! -L "$path" ]] || fail "$label is unavailable or unsafe"
}

require_new_output() {
  local path="$1"
  [[ "$path" == /* && "$path" != *".."* ]] || fail "native reference output root is unsafe"
  [[ "$(basename -- "$path")" =~ ^native-reference-[0-9]{12,14}$ ]] || fail "native reference output root must have a new native-reference timestamp name"
  [[ ! -e "$path" && ! -L "$path" ]] || fail "native reference output root already exists"
}

require_digest() {
  local value="$1" label="$2"
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || fail "$label must be a SHA-256 digest"
}

sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }

tree_sha256() {
  local root="$1"
  python3 - "$root" <<'PY'
import hashlib
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
digest = hashlib.sha256()
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root)
    if ".git" in relative.parts:
        continue
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        if path.is_dir():
            continue
        raise SystemExit("native reference tree has an unsafe entry")
    digest.update(relative.as_posix().encode("utf-8") + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
print(digest.hexdigest())
PY
}

download_missing_source() {
  local relative="$1" expected="$2" destination="${SOURCE_ROOT}/${relative}"
  require_digest "$expected" "pinned source digest"
  if [[ -e "$destination" || -L "$destination" ]]; then
    [[ -f "$destination" && ! -L "$destination" ]] || fail "native reference source cache has an unsafe entry: $relative"
    [[ "$(sha256_file "$destination")" == "$expected" ]] || fail "native reference source cache digest mismatch: $relative"
    return
  fi
  [[ "$VALIDATE_ONLY" == "0" ]] || fail "native reference source cache is incomplete: $relative"
  mkdir -p -- "$(dirname -- "$destination")"
  local temporary="${destination}.partial.$$"
  trap 'rm -f -- "${temporary:-}"' RETURN
  curl --fail --location --silent --show-error --proto '=https' --tlsv1.2 \
    "https://huggingface.co/${SOURCE_REPOSITORY}/resolve/${SOURCE_REVISION}/${relative}?download=true" \
    --output "$temporary"
  [[ "$(sha256_file "$temporary")" == "$expected" ]] || fail "downloaded pinned source digest mismatch: $relative"
  chmod 0444 -- "$temporary"
  mv -n -- "$temporary" "$destination" || fail "native reference source cache changed during staging"
  trap - RETURN
}

stage_native_source() {
  # The only network path is this allow-listed source list.  It contains no
  # pretrained_model path and deliberately never invokes git-lfs.
  download_missing_source "scripts/eval.py" "335cbf18b3da4c00c31f59468095bdc487960641b33077b94e09117129e6403f"
  download_missing_source "scripts/eval_policy/__init__.py" "e4cf4357d44b2f6bfb3121d1e0ac4c0cf89c6a2152c046cc4eb10809bb5628e8"
  download_missing_source "scripts/eval_policy/base_policy.py" "c43061da5ab7a2ded2d7416f0ea4f0786eeab77e2bff37d5855f3e0942101f47"
  download_missing_source "scripts/eval_policy/lerobot_policy.py" "ece14e711cee10268aad89bf814c00c7fab333835e8994dd0f50b76bd32aab97"
  download_missing_source "scripts/eval_policy/registry.py" "129bbf060c903185bd112e3b74aa5248ff5620be1e7e0e3990f273f9737f05b7"
  download_missing_source "scripts/utils/__init__.py" "94f982cd15f8ed35f49b06c69bf3c0baf253d2c88a2d794ba46f2c94a3ac4d5e"
  download_missing_source "scripts/utils/common.py" "3a1781fb4c2da3aa42174d9ac3ffc8d45b2f32579f14baa90b170b802d084f7b"
  download_missing_source "scripts/utils/eval_utils.py" "3e61805092ae28823ec1808928f49956632dea05ca35f3340c80c0d922ed60b9"
  download_missing_source "scripts/utils/evaluation.py" "9a9d9e28008405ead892fdf1d115cd83f3d2be7d806381dbc92486d2e6d966a7"
  download_missing_source "scripts/utils/parser.py" "e94804d94d19fa96dd1ce93cec6976cf8730820475407bd8af6677d67c071a8e"
  # Keep this literal in the contract: a reviewer can prove that sparse
  # source checkout never smudges or re-downloads pretrained_model weights.
  : "${GIT_LFS_SKIP_SMUDGE:=1}"
}

stage_native_environment_source() {
  if [[ -d "${SOURCE_ROOT}/source/lehome" && ! -L "${SOURCE_ROOT}/source/lehome" ]]; then
    return
  fi
  [[ "$VALIDATE_ONLY" == "0" ]] || fail "native reference source cache is incomplete: source/lehome"
  [[ -z "$(find "$SOURCE_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]] || fail "native reference source cache cannot be safely initialized"
  command -v git >/dev/null 2>&1 || fail "git is required to stage the pinned native source"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$SOURCE_ROOT" init --quiet
  git -C "$SOURCE_ROOT" remote add origin "https://huggingface.co/${SOURCE_REPOSITORY}"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$SOURCE_ROOT" -c filter.lfs.smudge= -c filter.lfs.required=false fetch --depth=1 origin "$SOURCE_REVISION"
  git -C "$SOURCE_ROOT" sparse-checkout init --no-cone
  git -C "$SOURCE_ROOT" sparse-checkout set --no-cone scripts source/lehome configs
  GIT_LFS_SKIP_SMUDGE=1 git -C "$SOURCE_ROOT" -c filter.lfs.smudge= -c filter.lfs.required=false checkout --quiet --detach "$SOURCE_REVISION"
}

validate_checkpoint() {
  local required
  for required in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json \
      policy_preprocessor_step_2_groot_pack_inputs_v3.safetensors \
      policy_postprocessor_step_0_groot_action_unpack_unnormalize_v1.safetensors train_config.json; do
    [[ -f "${CHECKPOINT_ROOT}/${required}" && ! -L "${CHECKPOINT_ROOT}/${required}" ]] || fail "native reference checkpoint cache is incomplete: $required"
  done
  [[ "$(sha256_file "${CHECKPOINT_ROOT}/config.json")" == "$EXPECTED_CHECKPOINT_CONFIG_SHA256" ]] || fail "native reference checkpoint config digest mismatch"
  [[ "$(sha256_file "${CHECKPOINT_ROOT}/model.safetensors")" == "$EXPECTED_MODEL_SHA256" ]] || fail "native reference checkpoint weight digest mismatch"
  [[ "$(sha256_file "${CHECKPOINT_ROOT}/policy_preprocessor.json")" == "$EXPECTED_PREPROCESSOR_SHA256" ]] || fail "native reference saved preprocessor digest mismatch"
  [[ "$(sha256_file "${CHECKPOINT_ROOT}/policy_postprocessor.json")" == "$EXPECTED_POSTPROCESSOR_SHA256" ]] || fail "native reference saved postprocessor digest mismatch"
  python3 - "${CHECKPOINT_ROOT}/policy_preprocessor.json" "${CHECKPOINT_ROOT}/policy_postprocessor.json" <<'PY'
import json
import sys
from pathlib import Path

preprocessor = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
postprocessor = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
try:
    horizon = preprocessor["steps"][2]["config"]["action_horizon"]
    action_dim = postprocessor["steps"][0]["config"]["env_action_dim"]
except (KeyError, IndexError, TypeError):
    raise SystemExit("saved LeRobot processors have an unexpected schema")
if horizon != 16 or action_dim != 12:
    raise SystemExit("saved LeRobot processors do not preserve the 16-action/12-joint contract")
PY
}

validate_metadata() {
  local category
  for category in top_long top_short pant_long pant_short; do
    local root="${METADATA_ROOT}/${category}_merged/meta"
    [[ -d "$root" && ! -L "$root" ]] || fail "native reference metadata cache is incomplete: $category"
    for filename in info.json stats.json tasks.parquet; do
      [[ -f "${root}/${filename}" && ! -L "${root}/${filename}" ]] || fail "native reference metadata cache is incomplete: ${category}/${filename}"
    done
  done
}

validate_source_runtime() {
  [[ -f "${SOURCE_ROOT}/scripts/eval.py" && ! -L "${SOURCE_ROOT}/scripts/eval.py" ]] || fail "native reference source cache is incomplete"
  [[ -d "${SOURCE_ROOT}/source/lehome" && ! -L "${SOURCE_ROOT}/source/lehome" ]] || fail "native reference environment source is incomplete"
  [[ -d "$ASSETS_ROOT" && ! -L "$ASSETS_ROOT" ]] || fail "native reference challenge assets are incomplete"
  [[ -d "${SOURCE_ROOT}/configs" && ! -L "${SOURCE_ROOT}/configs" ]] || fail "native reference config source is incomplete"
  git -C "$SOURCE_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "native reference source cache is not a pinned source checkout"
  [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || fail "native reference source revision mismatch"
  git -C "$SOURCE_ROOT" diff --quiet || fail "native reference source checkout is modified"
  [[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] || fail "native reference source checkout has untrusted files"
  [[ "$(tree_sha256 "$SOURCE_ROOT")" == "$SOURCE_TREE_SHA256" ]] || fail "native reference source tree digest mismatch"
  [[ "$(tree_sha256 "$CHECKPOINT_ROOT")" == "$CHECKPOINT_TREE_SHA256" ]] || fail "native reference checkpoint tree digest mismatch"
  [[ "$(tree_sha256 "$METADATA_ROOT")" == "$METADATA_TREE_SHA256" ]] || fail "native reference metadata tree digest mismatch"
  local discovered
  discovered="$(PYTHONPATH="${SOURCE_ROOT}/source/lehome:${SOURCE_ROOT}" "${PYTHON_BIN}" -c 'import lerobot; print(getattr(lerobot, "__version__", ""))')"
  [[ "$discovered" == "$EXPECTED_LEROBOT_VERSION" ]] || fail "native reference requires LeRobot ${EXPECTED_LEROBOT_VERSION}, found ${discovered:-missing}"
}

write_preflight() {
  python3 - "$OUTPUT_ROOT/preflight.json" "$SOURCE_TREE_SHA256" "$CHECKPOINT_TREE_SHA256" "$METADATA_TREE_SHA256" "$ASSETS_TREE_SHA256" "$IMAGE" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "kind": "lehome_native_reference_preflight_v1",
    "source_repository": "theo-zhou/lehome-groot-submission-4",
    "source_revision": "d384fe00508acd96ab1c3c5dc265e08261f94b3b",
    "source_tree_sha256": sys.argv[2],
    "checkpoint_tree_sha256": sys.argv[3],
    "metadata_tree_sha256": sys.argv[4],
    "assets_tree_sha256": sys.argv[5],
    "checkpoint_config_sha256": "b7c385bc57456eae603e929b84defb7e991194aade2aad70785e21991e37614c",
    "checkpoint_weight_sha256": "d8d91b6c11cb5aa18fe9a48e7da88eae0ec7e5a227a315ba67ee167d645cde76",
    "lerobot_version": "0.4.3",
    "policy_class": "scripts.eval_policy.lerobot_policy.LeRobotPolicy",
    "simulator_device": "cpu",
    "policy_device": "cuda:0",
    "image": sys.argv[6],
}
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
}

# Explicit provider identities are inputs, not discovery.  This is deliberately
# not a provider CLI wrapper: it has no create/start/image-build/upload code.
VALIDATE_ONLY="${LEHOME_NATIVE_REFERENCE_VALIDATE_ONLY:-0}"
VM_ID="${LEHOME_NATIVE_REFERENCE_VM_ID:-}"
DISK_ID="${LEHOME_NATIVE_REFERENCE_DISK_ID:-}"
IMAGE="${LEHOME_NATIVE_REFERENCE_IMAGE:-}"
SOURCE_ROOT="${LEHOME_NATIVE_REFERENCE_SOURCE_ROOT:-}"
CHECKPOINT_ROOT="${LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT:-}"
METADATA_ROOT="${LEHOME_NATIVE_REFERENCE_METADATA_ROOT:-}"
ASSETS_ROOT="${LEHOME_NATIVE_REFERENCE_ASSETS_ROOT:-}"
OUTPUT_ROOT="${LEHOME_NATIVE_REFERENCE_OUTPUT_ROOT:-}"
SOURCE_TREE_SHA256="${LEHOME_NATIVE_REFERENCE_SOURCE_TREE_SHA256:-}"
CHECKPOINT_TREE_SHA256="${LEHOME_NATIVE_REFERENCE_CHECKPOINT_TREE_SHA256:-}"
METADATA_TREE_SHA256="${LEHOME_NATIVE_REFERENCE_METADATA_TREE_SHA256:-}"
ASSETS_TREE_SHA256="${LEHOME_NATIVE_REFERENCE_ASSETS_TREE_SHA256:-}"
PYTHON_BIN="${LEHOME_NATIVE_REFERENCE_PYTHON:-python3}"

[[ "$VALIDATE_ONLY" == "0" || "$VALIDATE_ONLY" == "1" ]] || fail "LEHOME_NATIVE_REFERENCE_VALIDATE_ONLY must be exactly 0 or 1"
[[ "$VM_ID" =~ ^computeinstance-[a-z0-9]+$ ]] || fail "LEHOME_NATIVE_REFERENCE_VM_ID is invalid"
[[ "$DISK_ID" =~ ^computedisk-[a-z0-9]+$ ]] || fail "LEHOME_NATIVE_REFERENCE_DISK_ID is invalid"
[[ "$IMAGE" =~ @sha256:[0-9a-f]{64}$ ]] || fail "LEHOME_NATIVE_REFERENCE_IMAGE must be digest pinned"
for digest_label in SOURCE_TREE_SHA256 CHECKPOINT_TREE_SHA256 METADATA_TREE_SHA256 ASSETS_TREE_SHA256; do
  require_digest "${!digest_label}" "$digest_label"
done
require_absolute_directory "$SOURCE_ROOT" "native reference source cache"
require_absolute_directory "$CHECKPOINT_ROOT" "native reference checkpoint cache"
require_absolute_directory "$METADATA_ROOT" "native reference metadata cache"
require_absolute_directory "$ASSETS_ROOT" "native reference challenge assets"
require_new_output "$OUTPUT_ROOT"
[[ "$SOURCE_TREE_SHA256" == "$EXPECTED_SOURCE_TREE_SHA256" ]] || fail "native reference source tree digest must match the pinned source contract"

stage_native_environment_source
stage_native_source
validate_checkpoint
validate_metadata
validate_source_runtime
[[ "$(tree_sha256 "$ASSETS_ROOT")" == "$ASSETS_TREE_SHA256" ]] || fail "native reference assets tree digest mismatch"

if [[ "$VALIDATE_ONLY" == "1" ]]; then
  printf '%s\n' '{"status":"validated","mode":"no-execution"}'
  exit 0
fi

mkdir --mode=0700 -- "$OUTPUT_ROOT"
mkdir --mode=0700 -- "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/videos" "$OUTPUT_ROOT/receipts"
write_preflight

run_stage() {
  local stage="$1" category="$2" garment="$3" log="$OUTPUT_ROOT/logs/stage-${stage}.log"
  local -a command=(
    "$PYTHON_BIN" -m scripts.eval
    --headless --device cpu --task "LeHome-BiSO101-Direct-Garment-v2"
    --policy_type lerobot --policy_path "$CHECKPOINT_ROOT"
    --dataset_root "$METADATA_ROOT/${category}_merged"
    --task_description "fold the garment on the table"
    --garment_type "$category" --garment_filter "$garment"
    --num_episodes 2 --max_steps 600 --seed 42 --save_video
    --video_dir "$OUTPUT_ROOT/videos/stage-${stage}"
    --garment_cfg_base_path "$ASSETS_ROOT/objects/Challenge_Garment"
    --particle_cfg_path "$SOURCE_ROOT/source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml"
  )
  (cd -- "$SOURCE_ROOT" && PYTHONPATH="$SOURCE_ROOT/source/lehome:$SOURCE_ROOT" "${command[@]}") >"$log" 2>&1 || fail "native reference stage ${stage} failed; inspect ${log}"
  # The public evaluator emits exactly one structured success line per episode.
  # Its log is preserved; the result compiler below binds it to the oracle.
  python3 - "$OUTPUT_ROOT/result.json" "$stage" "$category" "$garment" "$log" "$SOURCE_TREE_SHA256" "$CHECKPOINT_TREE_SHA256" "$METADATA_TREE_SHA256" "$ASSETS_TREE_SHA256" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

result_path = Path(sys.argv[1]); stage = int(sys.argv[2]); category = sys.argv[3]; garment = sys.argv[4]; log = Path(sys.argv[5])
text = log.read_text(encoding="utf-8", errors="strict")
matches = re.findall(r"Episode\\s+(\\d+)/2:.*?Success=(True|False)", text)
if matches != [("1", matches[0][1]), ("2", matches[1][1])] if len(matches) == 2 else True:
    raise SystemExit("native evaluator log does not contain exactly two ordered episode outcomes")
if result_path.exists():
    payload = json.loads(result_path.read_text(encoding="utf-8"))
else:
    payload = {
        "schema_version": 1, "kind": "lehome_native_reference_evaluator_result_v1",
        "identity": {
            "source_repository": "theo-zhou/lehome-groot-submission-4",
            "source_revision": "d384fe00508acd96ab1c3c5dc265e08261f94b3b",
            "source_tree_sha256": sys.argv[6], "checkpoint_tree_sha256": sys.argv[7], "metadata_tree_sha256": sys.argv[8], "assets_tree_sha256": sys.argv[9],
            "lerobot_version": "0.4.3", "policy_class": "scripts.eval_policy.lerobot_policy.LeRobotPolicy",
            "policy_device": "cuda:0", "simulator_device": "cpu", "task_description": "fold the garment on the table",
            "action_horizon": 16, "action_dimension": 12, "success_checker": "pinned_raw_success_distance_second_mesh_points",
        }, "attempts": [],
    }
expected = {1: (True, True), 2: (True, True), 3: (True, True), 4: (False, True)}[stage]
for episode, (_, success) in enumerate(matches, start=1):
    attempt_id = f"native-reference-{stage}-{episode}"
    receipt = Path("receipts") / f"{attempt_id}.json"
    receipt_path = result_path.parent / receipt
    body = {"schema_version": 1, "attempt_id": attempt_id, "stage": stage, "category": category, "garment": garment, "episode": episode, "success": success == "True", "log": str(Path("logs") / log.name)}
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(body, stream, sort_keys=True, separators=(",", ":")); stream.write("\n")
    payload["attempts"].append({"attempt_id": attempt_id, "stage": stage, "category": category, "garment": garment, "episode": episode, "expected_success": expected[episode - 1], "success": success == "True", "invalid_reason": None, "video": str(Path("videos") / f"stage-{stage}"), "log": str(Path("logs") / log.name), "receipt": str(receipt)})
temporary = result_path.with_suffix(".partial")
temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
os.replace(temporary, result_path)
PY
}

run_stage 1 top_long Top_Long_Seen_0
if ! python3 - "$OUTPUT_ROOT/result.json" <<'PY'
import json
import sys
from pathlib import Path

attempts = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("attempts")
raise SystemExit(0 if isinstance(attempts, list) and len(attempts) == 2 and [row.get("success") for row in attempts] == [True, True] else 1)
PY
then
  "$PYTHON_BIN" "$SCRIPT_DIR/../scripts/verify_native_reference_evaluator_gate.py" --result "$OUTPUT_ROOT/result.json" --receipt "$OUTPUT_ROOT/gate-receipt.json"
  exit 0
fi
run_stage 2 top_short Top_Short_Seen_0
run_stage 3 pant_long Pant_Long_Seen_0
run_stage 4 pant_short Pant_Short_Seen_0
"$PYTHON_BIN" "$SCRIPT_DIR/../scripts/verify_native_reference_evaluator_gate.py" --result "$OUTPUT_ROOT/result.json" --receipt "$OUTPUT_ROOT/gate-receipt.json"
