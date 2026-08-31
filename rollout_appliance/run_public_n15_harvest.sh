#!/usr/bin/env bash
# Exact public-N1.5 seen-garment harvest boundary. Policy execution is only
# the pinned upstream command: python -P -m scripts.eval --save_datasets.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly BUILDER="$REPO_ROOT/scripts/build_public_n15_harvest.py"
readonly EXACT_VM_ID="computeinstance-u00t6xfqhadrcmssa2"
readonly PROTECTED_DISK_ID="computedisk-u00pbe55crxy7jr56x"
readonly ROLLOUT_IMAGE_ID="sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7"
readonly RUNTIME_PYTHON="/opt/lehome-challenge/.venv/bin/python"
readonly PUBLIC_SOURCE_REVISION="d384fe00508acd96ab1c3c5dc265e08261f94b3b"
readonly RUNTIME_REVISION="${LEHOME_N15_RUNTIME_REVISION:-}"
readonly SOURCE_ROOT="${LEHOME_N15_PUBLIC_SOURCE_ROOT:-}"
readonly CHECKPOINT_ROOT="${LEHOME_N15_CHECKPOINT_ROOT:-}"
readonly TRAINING_IDENTITY_RECEIPT="${LEHOME_N15_TRAINING_IDENTITY_RECEIPT:-}"
readonly ROLLOUT_IMAGE_RECEIPT="${LEHOME_N15_ROLLOUT_IMAGE_RECEIPT:-}"
readonly HARVEST_ROOT="${LEHOME_N15_HARVEST_ROOT:-}"
readonly PUBLIC_REPOSITORY="${LEHOME_N15_PUBLIC_HF_REPOSITORY:-}"
readonly MANIFEST="${LEHOME_N15_HARVEST_MANIFEST:-$HARVEST_ROOT/manifest.json}"
readonly MANIFEST_RECEIPT="${LEHOME_N15_HARVEST_MANIFEST_RECEIPT:-$HARVEST_ROOT/manifest-receipt.json}"
readonly RUNTIME_RECEIPT="$HARVEST_ROOT/runtime-receipt.json"
readonly DOCKER_INSPECT_RECEIPT="$HARVEST_ROOT/docker-inspect.json"
readonly OBSERVATIONAL_SITE="$HARVEST_ROOT/observational-site"
readonly FOUR_ADMISSION_ROOT="$HARVEST_ROOT/admission/four"
readonly TWO_ADMISSION_ROOT="$HARVEST_ROOT/admission/two"
readonly WORKER_SELECTION="${LEHOME_N15_WORKER_SELECTION_RECEIPT:-$HARVEST_ROOT/worker-selection.json}"
readonly WORKER_PLAN="${LEHOME_N15_WORKER_PLAN:-$HARVEST_ROOT/worker-plan.json}"
readonly PROCESS_TABLE="$HARVEST_ROOT/process-table.tsv"
readonly PROCESS_STATUS_TSV="$HARVEST_ROOT/process-status.tsv"
readonly FIRST_PROCESS_STATUS="$HARVEST_ROOT/first-process-status.json"
readonly FINAL_PROCESS_STATUS="$HARVEST_ROOT/final-process-status.json"
readonly FIRST_100_OUTCOMES="${LEHOME_N15_FIRST_100_OUTCOMES:-$HARVEST_ROOT/first-100-outcomes.json}"
readonly FIRST_100_GATE="${LEHOME_N15_FIRST_100_GATE_RECEIPT:-$HARVEST_ROOT/first-100-gate.json}"
readonly FINAL_OUTCOMES="$HARVEST_ROOT/final-outcomes.json"
readonly FIRST_SUCCESS_DATASETS="$HARVEST_ROOT/first-success-datasets.json"
readonly FINAL_SUCCESS_DATASETS="$HARVEST_ROOT/final-success-datasets.json"
readonly PUBLICATION_RECEIPT="${LEHOME_N15_PUBLICATION_RECEIPT:-${HARVEST_ROOT}.publication.json}"
readonly PROVIDER_STOPPED_RECEIPT="${LEHOME_N15_PROVIDER_STOPPED_RECEIPT:-${HARVEST_ROOT}.provider-stopped.json}"
readonly TERMINAL_RECEIPT="${LEHOME_N15_TERMINAL_RECEIPT:-${HARVEST_ROOT}.terminal.json}"
readonly DEFER_PROVIDER_STOP="${LEHOME_N15_DEFER_PROVIDER_STOP:-0}"
HARVEST_TERMINAL_COMPLETE=0

fail() { printf 'error: %s\n' "$*" >&2; exit 2; }
require_dir() {
  [[ "$1" == /* && "$1" != *".."* && -d "$1" && ! -L "$1" ]] \
    || fail "$2 is unavailable or unsafe"
}
require_file() {
  [[ "$1" == /* && "$1" != *".."* && -f "$1" && ! -L "$1" ]] \
    || fail "$2 is unavailable or unsafe"
}
require_new_file() {
  [[ "$1" == /* && "$1" != *".."* && ! -e "$1" && ! -L "$1" ]] \
    || fail "$2 must be a new safe path"
}

stop_and_observe_exact_vm() {
  # validate-provider-stop requires state=STOPPED and
  # protected_disk_preserved=true for the one protected secondary disk.
  if [[ -f "$PROVIDER_STOPPED_RECEIPT" && ! -L "$PROVIDER_STOPPED_RECEIPT" ]]; then
    python3 "$BUILDER" validate-provider-stop \
      --provider-receipt "$PROVIDER_STOPPED_RECEIPT" >/dev/null
    return
  fi
  command -v nebius >/dev/null 2>&1 || return 1
  local response
  response="$(mktemp "${TMPDIR:-/tmp}/lehome-n15-provider.XXXXXX.json")" || return 1
  if nebius compute instance get --id "$EXACT_VM_ID" --format json --no-browser \
      --no-progress --no-check-update --retries 1 --timeout 60s >"$response" \
      && python3 "$BUILDER" observe-provider-stop --response "$response" \
        --output "$PROVIDER_STOPPED_RECEIPT" >/dev/null 2>&1; then
    python3 "$BUILDER" validate-provider-stop \
      --provider-receipt "$PROVIDER_STOPPED_RECEIPT" >/dev/null
    rm -f -- "$response"
    return
  fi
  nebius compute instance stop --id "$EXACT_VM_ID" --format json --no-browser \
    --no-progress --no-check-update --retries 1 --timeout 60s >/dev/null
  local attempt
  for ((attempt=0; attempt<60; attempt++)); do
    if nebius compute instance get --id "$EXACT_VM_ID" --format json --no-browser \
        --no-progress --no-check-update --retries 1 --timeout 60s >"$response" \
        && python3 "$BUILDER" observe-provider-stop --response "$response" \
          --output "$PROVIDER_STOPPED_RECEIPT" >/dev/null 2>&1; then
      python3 "$BUILDER" validate-provider-stop \
        --provider-receipt "$PROVIDER_STOPPED_RECEIPT" >/dev/null
      rm -f -- "$response"
      return
    fi
    sleep 2
  done
  rm -f -- "$response"
  return 1
}

on_exit() {
  local status=$?
  if [[ "$HARVEST_TERMINAL_COMPLETE" -ne 1 || "$status" -ne 0 ]]; then
    if ! stop_and_observe_exact_vm; then status=3; fi
  fi
  trap - EXIT
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT TERM

[[ $# -eq 0 ]] || fail "this wrapper accepts no positional arguments"
[[ "$DEFER_PROVIDER_STOP" == 0 || "$DEFER_PROVIDER_STOP" == 1 ]] || fail "defer provider stop must be exactly 0 or 1"
command -v docker >/dev/null 2>&1 || fail "docker is unavailable"
command -v git >/dev/null 2>&1 || fail "git is unavailable"
command -v nebius >/dev/null 2>&1 || fail "Nebius CLI is unavailable"
require_file "$BUILDER" "checked-in harvest contract builder"
require_dir "$SOURCE_ROOT" "pinned public N1.5 source"
require_file "$SOURCE_ROOT/scripts/eval.py" "public scripts.eval"
require_dir "$CHECKPOINT_ROOT" "accepted Task 1 checkpoint"
require_file "$TRAINING_IDENTITY_RECEIPT" "accepted Task 1 training identity receipt"
require_file "$ROLLOUT_IMAGE_RECEIPT" "rollout image identity receipt"
require_dir "$SOURCE_ROOT/Datasets/example/four_types_merged" "pinned public dataset snapshot"
[[ "$PUBLIC_REPOSITORY" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] \
  || fail "public Hugging Face dataset repository is invalid"
[[ "$RUNTIME_REVISION" =~ ^[0-9a-f]{40}$ ]] || fail "runtime revision must be exact"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$RUNTIME_REVISION" ]] \
  || fail "runtime revision mismatch"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] \
  || fail "runtime source tree is not fully clean"
[[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$PUBLIC_SOURCE_REVISION" ]] \
  || fail "public source revision mismatch"
[[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] \
  || fail "public source tree is not fully clean"
[[ "$HARVEST_ROOT" == /* && "$HARVEST_ROOT" != *".."* \
   && ! -e "$HARVEST_ROOT" && ! -L "$HARVEST_ROOT" ]] \
  || fail "harvest root must be a new absolute path"
require_dir "$(dirname -- "$HARVEST_ROOT")" "harvest parent"
[[ "$MANIFEST" == "$HARVEST_ROOT/manifest.json" \
   && "$MANIFEST_RECEIPT" == "$HARVEST_ROOT/manifest-receipt.json" ]] \
  || fail "manifest bundle paths are fixed inside the harvest root"
for output in "$PUBLICATION_RECEIPT" "$PROVIDER_STOPPED_RECEIPT" "$TERMINAL_RECEIPT"; do
  require_new_file "$output" "terminal evidence output"
  [[ "$output" != "$HARVEST_ROOT"/* ]] \
    || fail "terminal evidence output must remain outside the upload bundle"
done
mkdir -m 0700 -- "$HARVEST_ROOT"
python3 "$BUILDER" write-observational-site --output-dir "$OBSERVATIONAL_SITE" >/dev/null

docker image inspect -- "$ROLLOUT_IMAGE_ID" >"$DOCKER_INSPECT_RECEIPT"
chmod 0444 "$DOCKER_INSPECT_RECEIPT"

TRAINING_ROOT="$(PYTHONPATH="$REPO_ROOT/source/lehome:$REPO_ROOT" python3 - \
  "$TRAINING_IDENTITY_RECEIPT" "$CHECKPOINT_ROOT" <<'PY'
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
readonly TRAINING_ROOT
require_dir "$TRAINING_ROOT" "accepted Task 1 training root"

docker run --rm --pull never --network none --init \
  --mount "type=bind,src=$REPO_ROOT,dst=/runtime,readonly" \
  --mount "type=bind,src=$SOURCE_ROOT,dst=$SOURCE_ROOT,readonly" \
  --mount "type=bind,src=$TRAINING_ROOT,dst=$TRAINING_ROOT,readonly" \
  --mount "type=bind,src=$TRAINING_IDENTITY_RECEIPT,dst=$TRAINING_IDENTITY_RECEIPT,readonly" \
  --mount "type=bind,src=$ROLLOUT_IMAGE_RECEIPT,dst=$ROLLOUT_IMAGE_RECEIPT,readonly" \
  --mount "type=bind,src=$HARVEST_ROOT,dst=$HARVEST_ROOT" \
  --env PYTHONPATH=/runtime/source/lehome:/runtime \
  --env PYTHONHOME= --env PYTHONSAFEPATH=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint "$RUNTIME_PYTHON" "$ROLLOUT_IMAGE_ID" -P \
  /runtime/scripts/build_public_n15_harvest.py verify-runtime \
  --source-root "$SOURCE_ROOT" --source-revision "$PUBLIC_SOURCE_REVISION" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --training-identity-receipt "$TRAINING_IDENTITY_RECEIPT" \
  --rollout-image-receipt "$ROLLOUT_IMAGE_RECEIPT" \
  --docker-inspect-receipt "$DOCKER_INSPECT_RECEIPT" \
  --output "$RUNTIME_RECEIPT" >/dev/null

IFS=$'\t' read -r checkpoint_tree checkpoint_receipt source_tree dataset_snapshot image_sha \
  < <(python3 - "$RUNTIME_RECEIPT" <<'PY'
import json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_bytes())
print("\t".join(value[key] for key in (
    "checkpoint_tree_sha256", "training_identity_receipt_sha256",
    "source_tree_sha256", "dataset_snapshot_sha256", "rollout_image_sha256",
)))
PY
)
runtime_receipt_sha="$(sha256sum -- "$RUNTIME_RECEIPT" | awk '{print $1}')"
readonly checkpoint_tree checkpoint_receipt source_tree dataset_snapshot image_sha runtime_receipt_sha

python3 "$BUILDER" build \
  --checkpoint-tree-sha256 "$checkpoint_tree" \
  --checkpoint-receipt-sha256 "$checkpoint_receipt" \
  --runtime-receipt-sha256 "$runtime_receipt_sha" \
  --source-tree-sha256 "$source_tree" \
  --dataset-snapshot-sha256 "$dataset_snapshot" \
  --rollout-image-sha256 "$image_sha" \
  --manifest "$MANIFEST" --receipt "$MANIFEST_RECEIPT" >/dev/null

inspect_success_datasets_in_runtime() {
  local expected_attempt_count="$1" output="$2"
  docker run --rm --pull never --network none --init \
    --mount "type=bind,src=$REPO_ROOT,dst=/runtime,readonly" \
    --mount "type=bind,src=$HARVEST_ROOT,dst=$HARVEST_ROOT" \
    --env PYTHONPATH=/runtime/source/lehome:/runtime \
    --env PYTHONHOME= --env PYTHONSAFEPATH=1 --env PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint "$RUNTIME_PYTHON" "$ROLLOUT_IMAGE_ID" -P \
    /runtime/scripts/build_public_n15_harvest.py inspect-success-datasets \
    --manifest "$MANIFEST" --harvest-root "$HARVEST_ROOT" \
    --expected-attempt-count "$expected_attempt_count" --output "$output" >/dev/null
}

run_admission_eval() {
  local category="$1" garment="$2" process_seed="$3" episode_count="$4"
  local output_root="$5" container_name="$6" fidelity_path="$7"
  docker run --rm --pull never --network none --gpus all --init --shm-size=8g \
    --name "$container_name" \
    --mount "type=bind,src=$REPO_ROOT,dst=/runtime,readonly" \
    --mount "type=bind,src=$SOURCE_ROOT,dst=$SOURCE_ROOT,readonly" \
    --mount "type=bind,src=$CHECKPOINT_ROOT,dst=$CHECKPOINT_ROOT,readonly" \
    --mount "type=bind,src=$HARVEST_ROOT,dst=$HARVEST_ROOT" \
    --workdir "$SOURCE_ROOT" \
    --env "PYTHONPATH=$OBSERVATIONAL_SITE:/runtime/rollout_appliance/native_reference_site:$SOURCE_ROOT/source/lehome:$SOURCE_ROOT:/opt/lehome-challenge/third_party/IsaacLab/source/isaaclab:/opt/lehome-challenge/third_party/IsaacLab/source/isaaclab_tasks" \
    --env PYTHONHOME= --env PYTHONSAFEPATH=1 --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYNPUT_BACKEND=dummy --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT= \
    --env LEHOME_NATIVE_REFERENCE_SANITIZED_CONFIG_ROOT= \
    --env LEHOME_NATIVE_REFERENCE_CHECKPOINT_COMPATIBILITY_RECEIPT= \
    --env LEHOME_CPU_ACTION= \
    --env "LEHOME_NATIVE_REFERENCE_LOG_PROJECT_ROOT=$output_root/native-project" \
    --env "LEHOME_NATIVE_REFERENCE_SOURCE_ROOT=$SOURCE_ROOT" \
    --env "LEHOME_NATIVE_CLOTH_FIDELITY_EVIDENCE=$fidelity_path" \
    --entrypoint "$RUNTIME_PYTHON" "$ROLLOUT_IMAGE_ID" -P -m scripts.eval \
    --policy_type lerobot --policy_path "$CHECKPOINT_ROOT" \
    --garment_type "$category" --garment_filter "$garment" \
    --dataset_root "$SOURCE_ROOT/Datasets/example/four_types_merged" \
    --num_episodes "$episode_count" --seed "$process_seed" \
    --enable_cameras --headless --device cpu
}

admission_schedule() {
  python3 "$BUILDER" admission-schedule --worker-count "$1" |
    python3 -c 'import json,sys
for row in json.load(sys.stdin):
    print("\t".join(map(str, (row["worker_id"], row["smoke_id"], row["category"], row["garment"], row["process_seed"]))))'
}

run_admission_count() {
  local count="$1" root="$2" worker smoke_id category garment process_seed
  mkdir -m 0700 -p -- "$root/memory" "$root/smokes"
  mapfile -t admission_rows < <(admission_schedule "$count")
  local -a pids=()
  : >"$root/memory-status.tsv"
  local used total extra
  IFS=',' read -r used total extra < <(
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
  )
  [[ -z "${extra:-}" ]] || fail "worker admission requires exactly one GPU"
  used="${used//[[:space:]]/}"; total="${total//[[:space:]]/}"
  printf 'sample_index\tactive_process_count\tgpu_used_mib\tgpu_total_mib\n0\t0\t%s\t%s\n' \
    "$used" "$total" >"$root/memory.tsv"
  for row in "${admission_rows[@]}"; do
    IFS=$'\t' read -r worker smoke_id category garment process_seed <<<"$row"
    run_admission_eval "$category" "$garment" "$process_seed" 0 \
      "$root/memory/worker-$worker" "lehome-n15-memory-$count-$worker" "" \
      >"$root/memory/worker-$worker.log" 2>&1 &
    pids+=("$!")
  done
  local sample=1 active pid
  while :; do
    active=0
    for pid in "${pids[@]}"; do if kill -0 "$pid" 2>/dev/null; then ((active+=1)); fi; done
    IFS=',' read -r used total extra < <(
      nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    )
    [[ -z "${extra:-}" ]] || fail "worker admission requires exactly one GPU"
    used="${used//[[:space:]]/}"; total="${total//[[:space:]]/}"
    printf '%s\t%s\t%s\t%s\n' "$sample" "$active" "$used" "$total" >>"$root/memory.tsv"
    ((sample+=1))
    (( active != 0 )) || break
    sleep 1
  done
  for worker in "${!pids[@]}"; do
    if wait "${pids[$worker]}"; then code=0; else code=$?; fi
    printf '%s\t%s\n' "$worker" "$code" >>"$root/memory-status.tsv"
  done
  python3 "$BUILDER" assess-memory --evidence-root "$root" --worker-count "$count" \
    --output "$root/memory-receipt.json" >/dev/null
  local passed
  passed="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["memory_check"]["passed"]).lower())' "$root/memory-receipt.json")"
  if [[ "$passed" == true ]]; then
    pids=(); : >"$root/smoke-status.tsv"
    for row in "${admission_rows[@]}"; do
      IFS=$'\t' read -r worker smoke_id category garment process_seed <<<"$row"
      mkdir -m 0700 -- "$root/smokes/worker-$worker"
      printf '%s\n' "$smoke_id" >"$root/smokes/worker-$worker/smoke-id.txt"
      run_admission_eval "$category" "$garment" "$process_seed" 1 \
        "$root/smokes/worker-$worker" "lehome-n15-smoke-$count-$worker" \
        "$root/smokes/worker-$worker/cloth-fidelity.jsonl" \
        >"$root/smokes/worker-$worker/evaluator.log" 2>&1 &
      pids+=("$!")
    done
    for worker in "${!pids[@]}"; do
      if wait "${pids[$worker]}"; then code=0; else code=$?; fi
      printf '%s\t%s\n' "$worker" "$code" >>"$root/smoke-status.tsv"
    done
  fi
}

run_admission_count 4 "$FOUR_ADMISSION_ROOT"
python3 "$BUILDER" assess-admission --manifest "$MANIFEST" \
  --runtime-receipt "$RUNTIME_RECEIPT" --evidence-root "$FOUR_ADMISSION_ROOT" \
  --worker-count 4 --output "$FOUR_ADMISSION_ROOT/admission-receipt.json" >/dev/null
four_passed="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["passed"]).lower())' "$FOUR_ADMISSION_ROOT/admission-receipt.json")"
admission=(python3 "$BUILDER" admit-workers --manifest "$MANIFEST" \
  --runtime-receipt "$RUNTIME_RECEIPT" \
  --four-worker-evidence-root "$FOUR_ADMISSION_ROOT" --output "$WORKER_SELECTION")
if [[ "$four_passed" != true ]]; then
  run_admission_count 2 "$TWO_ADMISSION_ROOT"
  python3 "$BUILDER" assess-admission --manifest "$MANIFEST" \
    --runtime-receipt "$RUNTIME_RECEIPT" --evidence-root "$TWO_ADMISSION_ROOT" \
    --worker-count 2 --output "$TWO_ADMISSION_ROOT/admission-receipt.json" >/dev/null
  admission+=(--two-worker-evidence-root "$TWO_ADMISSION_ROOT")
fi
"${admission[@]}" >/dev/null
python3 "$BUILDER" render-worker-plan --manifest "$MANIFEST" \
  --admission "$WORKER_SELECTION" --source-root "$SOURCE_ROOT" \
  --checkpoint-root "$CHECKPOINT_ROOT" --output-root "$HARVEST_ROOT" \
  --output "$WORKER_PLAN" >/dev/null

python3 - "$MANIFEST" "$PROCESS_TABLE" <<'PY'
import json, os, sys
from pathlib import Path
manifest, output = Path(sys.argv[1]), Path(sys.argv[2])
rows = json.loads(manifest.read_bytes())["attempts"]
seen, processes = set(), []
for row in rows:
    identity = (row["category"], row["garment"], row["process_seed"])
    if identity in seen:
        continue
    seen.add(identity)
    process_id = f'{row["category"]}-{row["garment_index"]:02d}-s{row["process_seed"]:06d}'
    processes.append((*identity, process_id))
if len(processes) != 40:
    raise SystemExit("frozen process table is not exactly 40 garments")
with output.open("x", encoding="ascii") as stream:
    for row in processes:
        stream.write("\t".join(map(str, row)) + "\n")
os.chmod(output, 0o444)
PY

mapfile -t process_rows <"$PROCESS_TABLE"
worker_count="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["worker_count"])' "$WORKER_SELECTION")"
[[ "$worker_count" == 4 || "$worker_count" == 2 ]] || fail "worker selection is invalid"
: >"$PROCESS_STATUS_TSV"
chmod 0600 "$PROCESS_STATUS_TSV"

declare -a process_pids=()
declare -a process_ids=()
declare -a process_categories=()
declare -a process_garments=()
declare -a process_seeds=()

run_native_process() {
  local category="$1" garment="$2" process_seed="$3" process_id="$4"
  local process_root="$HARVEST_ROOT/processes/$process_id"
  mkdir -m 0700 -p -- "$process_root"
  docker run --rm --pull never --network none --gpus all --init --shm-size=8g \
    --mount "type=bind,src=$REPO_ROOT,dst=/runtime,readonly" \
    --mount "type=bind,src=$SOURCE_ROOT,dst=$SOURCE_ROOT,readonly" \
    --mount "type=bind,src=$CHECKPOINT_ROOT,dst=$CHECKPOINT_ROOT,readonly" \
    --mount "type=bind,src=$HARVEST_ROOT,dst=$HARVEST_ROOT" \
    --workdir "$SOURCE_ROOT" \
    --env "PYTHONPATH=$OBSERVATIONAL_SITE:/runtime/rollout_appliance/native_reference_site:$SOURCE_ROOT/source/lehome:$SOURCE_ROOT:/opt/lehome-challenge/third_party/IsaacLab/source/isaaclab:/opt/lehome-challenge/third_party/IsaacLab/source/isaaclab_tasks" \
    --env PYTHONHOME= --env PYTHONSAFEPATH=1 --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYNPUT_BACKEND=dummy --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT= \
    --env LEHOME_NATIVE_REFERENCE_SANITIZED_CONFIG_ROOT= \
    --env LEHOME_NATIVE_REFERENCE_CHECKPOINT_COMPATIBILITY_RECEIPT= \
    --env LEHOME_CPU_ACTION= \
    --env "LEHOME_NATIVE_REFERENCE_LOG_PROJECT_ROOT=$process_root/native-project" \
    --env "LEHOME_NATIVE_REFERENCE_SOURCE_ROOT=$SOURCE_ROOT" \
    --env "LEHOME_NATIVE_CLOTH_FIDELITY_EVIDENCE=$process_root/cloth-fidelity.jsonl" \
    --entrypoint "$RUNTIME_PYTHON" "$ROLLOUT_IMAGE_ID" -P -m scripts.eval \
    --policy_type lerobot --policy_path "$CHECKPOINT_ROOT" \
    --garment_type "$category" --garment_filter "$garment" \
    --dataset_root "$SOURCE_ROOT/Datasets/example/four_types_merged" \
    --num_episodes 25 --seed "$process_seed" --enable_cameras --headless \
    --device cpu --save_datasets \
    --eval_dataset_path "$process_root/dataset" --log_suffix "$process_id" \
    >"$process_root/evaluator.log" 2>&1
}

collect_batch_statuses() {
  local failed=0 index pid code
  for index in "${!process_pids[@]}"; do
    pid="${process_pids[$index]}"
    if wait "$pid"; then code=0; else code=$?; failed=1; fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "${process_ids[$index]}" "${process_categories[$index]}" \
      "${process_garments[$index]}" "${process_seeds[$index]}" "$code" \
      >>"$PROCESS_STATUS_TSV"
  done
  process_pids=()
  process_ids=()
  process_categories=()
  process_garments=()
  process_seeds=()
  (( failed == 0 )) || fail "one or more native evaluator processes failed"
}

run_wave() {
  local start="$1" end="$2" index category garment process_seed process_id
  for ((index=start; index<end; index++)); do
    IFS=$'\t' read -r category garment process_seed process_id <<<"${process_rows[$index]}"
    run_native_process "$category" "$garment" "$process_seed" "$process_id" &
    process_pids+=("$!")
    process_ids+=("$process_id")
    process_categories+=("$category")
    process_garments+=("$garment")
    process_seeds+=("$process_seed")
    if (( ${#process_pids[@]} == worker_count )); then collect_batch_statuses; fi
  done
  if (( ${#process_pids[@]} != 0 )); then collect_batch_statuses; fi
}

run_wave 0 4
python3 "$BUILDER" build-process-status --tsv "$PROCESS_STATUS_TSV" \
  --expected-process-count 4 --output "$FIRST_PROCESS_STATUS" >/dev/null
inspect_success_datasets_in_runtime 100 "$FIRST_SUCCESS_DATASETS"
python3 "$BUILDER" collect-outcomes --manifest "$MANIFEST" \
  --process-status "$FIRST_PROCESS_STATUS" --harvest-root "$HARVEST_ROOT" \
  --expected-attempt-count 100 --success-dataset-receipt "$FIRST_SUCCESS_DATASETS" \
  --output "$FIRST_100_OUTCOMES" >/dev/null
python3 "$BUILDER" first-100 --manifest "$MANIFEST" \
  --outcomes "$FIRST_100_OUTCOMES" --output "$FIRST_100_GATE" >/dev/null

run_wave 4 40
chmod 0444 "$PROCESS_STATUS_TSV"
python3 "$BUILDER" build-process-status --tsv "$PROCESS_STATUS_TSV" \
  --expected-process-count 40 --output "$FINAL_PROCESS_STATUS" >/dev/null
inspect_success_datasets_in_runtime 1000 "$FINAL_SUCCESS_DATASETS"
python3 "$BUILDER" collect-outcomes --manifest "$MANIFEST" \
  --process-status "$FINAL_PROCESS_STATUS" --harvest-root "$HARVEST_ROOT" \
  --expected-attempt-count 1000 --success-dataset-receipt "$FINAL_SUCCESS_DATASETS" \
  --output "$FINAL_OUTCOMES" >/dev/null

# publish-hf performs authenticated upload plus authenticated and anonymous
# byte-for-byte readback at the returned immutable revision.
python3 "$BUILDER" publish-hf --bundle-root "$HARVEST_ROOT" \
  --manifest "$MANIFEST" --manifest-receipt "$MANIFEST_RECEIPT" \
  --final-outcomes "$FINAL_OUTCOMES" --repository "$PUBLIC_REPOSITORY" \
  --output "$PUBLICATION_RECEIPT" >/dev/null
if [[ "$DEFER_PROVIDER_STOP" == 1 ]]; then
  # The host lifecycle owns the exact provider STOPPED observation.  Returning
  # after sealed/public readback keeps its SSH session alive until it captures
  # the compact terminal receipts, rather than allowing this guest to sever it.
  HARVEST_TERMINAL_COMPLETE=1
  printf '%s\n' "$PUBLICATION_RECEIPT"
  exit 0
fi
stop_and_observe_exact_vm || fail "exact provider VM could not be stopped and verified"
python3 "$BUILDER" verify-terminal --manifest "$MANIFEST" \
  --manifest-receipt "$MANIFEST_RECEIPT" \
  --publication-receipt "$PUBLICATION_RECEIPT" \
  --provider-receipt "$PROVIDER_STOPPED_RECEIPT" \
  --output "$TERMINAL_RECEIPT" >/dev/null
HARVEST_TERMINAL_COMPLETE=1
