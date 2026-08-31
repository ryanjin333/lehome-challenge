#!/usr/bin/env bash
# Cost-bounded single-VM controller for the pinned public GR00T N1.5 path.
# Completed paid stages resume only from their canonical immutable receipt chain.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly BUILDER="$REPO_ROOT/scripts/run_public_n15_reproduction.py"
readonly PROVIDER_VERIFIER="$REPO_ROOT/scripts/verify_native_reference_evaluator_gate.py"
readonly HARVEST_BUILDER="$REPO_ROOT/scripts/build_public_n15_harvest.py"
readonly EXACT_VM_ID="computeinstance-u00t6xfqhadrcmssa2"
readonly PROTECTED_DISK_ID="computedisk-u00pbe55crxy7jr56x"
readonly EXACT_IMAGE_ID="computeimage-u00zf6w3yf72gakhcy"
readonly RUN_ID="${LEHOME_N15_RUN_ID:-}"
readonly PIPELINE_ROOT="${LEHOME_N15_PIPELINE_ROOT:-}"
readonly SSH_TARGET="${LEHOME_N15_SSH_TARGET:-}"
readonly REMOTE_ROOT="${LEHOME_N15_REMOTE_ROOT:-}"
readonly REMOTE_PIPELINE_ROOT="${LEHOME_N15_REMOTE_PIPELINE_ROOT:-}"
readonly REMOTE_RUNS_BASE="${LEHOME_N15_REMOTE_RUNS_BASE:-/mnt/lehome/public-n15-runs}"
readonly MAX_BUDGET_USD="${LEHOME_N15_MAX_BUDGET_USD:-100}"
# Code-owned conservative ceiling: 3 USD/hour times (12h train + 4h gate +
# 8h harvest) = 72 USD. The live provider preflight must not exceed 3 USD/h.
readonly PROVIDER_HOURLY_CEILING_USD=3
readonly TRAIN_TIMEOUT_SECONDS=43200
readonly FOCUSED_TIMEOUT_SECONDS=14400
readonly HARVEST_TIMEOUT_SECONDS=28800
readonly ESTIMATED_COST_USD=72
readonly PUBLIC_REPOSITORY="${LEHOME_N15_PUBLIC_HF_REPOSITORY:-}"
readonly SOURCE_ROOT="${LEHOME_N15_PUBLIC_SOURCE_ROOT:-}"
readonly SOURCE_RECEIPT="${LEHOME_N15_SOURCE_RECEIPT:-}"
readonly SNAPSHOTS_RECEIPT="${LEHOME_N15_RESOLVED_SNAPSHOTS_RECEIPT:-}"
readonly TRAINING_ROOT="${LEHOME_N15_TRAINING_ROOT:-}"
readonly HF_TOKEN_FILE="${LEHOME_N15_HF_TOKEN_FILE:-}"
readonly RUNTIME_REVISION="${LEHOME_N15_RUNTIME_REVISION:-}"
readonly TRAINING_HF_CACHE="${LEHOME_N15_TRAINING_HF_CACHE_ROOT:-}"
readonly TRAINING_PYTHON="${LEHOME_N15_TRAINING_PYTHON:-/opt/lehome-challenge/.venv/bin/python}"
readonly LEROBOT_WHEEL="${LEHOME_N15_LEROBOT_WHEEL:-}"
readonly ASSETS_ROOT="${LEHOME_OFFICIAL_ASSETS_ROOT:-}"
readonly METADATA_ROOT="${LEHOME_OFFICIAL_METADATA_ROOT:-}"
readonly REFERENCE_CHECKPOINT="${LEHOME_N15_REFERENCE_CHECKPOINT:-}"
readonly REFERENCE_SANITIZED_CONFIG="${LEHOME_N15_REFERENCE_SANITIZED_CONFIG_ROOT:-}"
readonly REFERENCE_COMPATIBILITY="${LEHOME_N15_REFERENCE_COMPATIBILITY_RECEIPT:-}"
readonly NATIVE_RUNTIME_EVIDENCE="${LEHOME_N15_NATIVE_RUNTIME_EVIDENCE_ROOT:-}"
readonly NATIVE_DEPENDENCIES="${LEHOME_N15_NATIVE_DEPENDENCIES_ROOT:-}"
readonly FOCUSED_HF_CACHE="${LEHOME_N15_FOCUSED_HF_CACHE_ROOT:-}"
readonly ROLLOUT_IMAGE_RECEIPT="${LEHOME_N15_ROLLOUT_IMAGE_RECEIPT:-}"
readonly PLAN_RECEIPT="$PIPELINE_ROOT/lifecycle-plan.json"
readonly DEADLINE_RECEIPT="$PIPELINE_ROOT/paid-deadline.json"
readonly TRAINING_IDENTITY_RECEIPT="$TRAINING_ROOT/training-identity.json"
readonly TRAINING_PUBLICATION_RECEIPT="$TRAINING_ROOT/training-publication.json"
readonly FOCUSED_OUTPUT_ROOT="$REMOTE_PIPELINE_ROOT/focused"
readonly FOCUSED_PROMOTION_RECEIPT="$FOCUSED_OUTPUT_ROOT/promotion.json"
readonly HARVEST_ROOT="$REMOTE_PIPELINE_ROOT/harvest"
readonly HARVEST_MANIFEST_RECEIPT="$PIPELINE_ROOT/harvest-manifest-receipt.json"
readonly HARVEST_MANIFEST="$PIPELINE_ROOT/harvest-manifest.json"
readonly HARVEST_PUBLICATION_RECEIPT="$PIPELINE_ROOT/harvest-publication.json"
readonly HARVEST_TERMINAL_RECEIPT="$PIPELINE_ROOT/harvest-terminal.json"
PROVIDER_STOPPED_RECEIPT="$PIPELINE_ROOT/provider-stopped.json"
PIPELINE_COMPLETE=0

fail() { printf 'error: %s\n' "$*" >&2; exit 2; }
require_abs_dir() { [[ "$1" == /* && "$1" != *".."* && -d "$1" && ! -L "$1" ]] || fail "$2 is unavailable or unsafe"; }
require_abs_file() { [[ "$1" == /* && "$1" != *".."* && -f "$1" && ! -L "$1" ]] || fail "$2 is unavailable or unsafe"; }
provider_get() { nebius compute instance get --id "$EXACT_VM_ID" --format json --no-browser --no-progress --no-check-update --retries 1 --timeout 60s; }
capture_exact_provider_state() {
  local expected_state="$1" receipt="$2"
  # This uses the established nested Nebius parser: metadata/status/spec,
  # exact source image, and exactly the one protected secondary disk.
  PYTHONPATH="$REPO_ROOT" python3 "$PROVIDER_VERIFIER" capture-provider \
    --state "$expected_state" --receipt "$receipt" >/dev/null
}

stop_exact_vm() {
  local response="$PIPELINE_ROOT/.provider-stop.$$.json"
  local observation="$PIPELINE_ROOT/.provider-stop-observation.$$.json"
  if [[ -f "$PROVIDER_STOPPED_RECEIPT" ]]; then
    python3 "$HARVEST_BUILDER" validate-provider-stop --provider-receipt "$PROVIDER_STOPPED_RECEIPT" >/dev/null
    if capture_exact_provider_state STOPPED "$observation"; then
      rm -f -- "$observation"; return
    fi
    rm -f -- "$observation"
    PROVIDER_STOPPED_RECEIPT="$PIPELINE_ROOT/provider-stopped-${RUN_ID}-$(date +%s).json"
  fi
  if capture_exact_provider_state STOPPED "$observation"; then
    provider_get >"$response"
    python3 "$HARVEST_BUILDER" observe-provider-stop --response "$response" --output "$PROVIDER_STOPPED_RECEIPT" >/dev/null
    rm -f -- "$response" "$observation"; return
  fi
  rm -f -- "$observation"
  nebius compute instance stop --id "$EXACT_VM_ID" --format json --no-browser --no-progress --no-check-update --retries 1 --timeout 60s >/dev/null || return 1
  for _ in {1..60}; do
    observation="$PIPELINE_ROOT/.provider-stop-observation.$$.json"
    if capture_exact_provider_state STOPPED "$observation"; then
      provider_get >"$response"
      python3 "$HARVEST_BUILDER" observe-provider-stop --response "$response" --output "$PROVIDER_STOPPED_RECEIPT" >/dev/null
      rm -f -- "$response" "$observation"; return
    fi
    rm -f -- "$observation"
    sleep 2
  done
  rm -f -- "$response"; return 1
}
trap stop_exact_vm EXIT
trap 'exit 130' INT TERM

remote() { ssh -o BatchMode=yes "$SSH_TARGET" "$@"; }
initialize_deadline() {
  python3 - "$PLAN_RECEIPT" "$DEADLINE_RECEIPT" "$RUN_ID" <<'PY'
import hashlib, json, os, sys, time
from pathlib import Path
plan, output, run_id = map(Path if False else str, sys.argv[1:])
plan_bytes = Path(plan).read_bytes(); digest = hashlib.sha256(plan_bytes).hexdigest()
if Path(output).exists():
    value = json.loads(Path(output).read_bytes())
    if value != {"schema_version": 1, "kind": "lehome_public_n15_paid_deadline_v1", "run_id": run_id, "lifecycle_plan_sha256": digest, "started_unix_seconds": value.get("started_unix_seconds"), "deadline_unix_seconds": value.get("deadline_unix_seconds")} or type(value["started_unix_seconds"]) is not int or value["deadline_unix_seconds"] != value["started_unix_seconds"] + 86400:
        raise SystemExit("paid deadline receipt is invalid")
    print(value["deadline_unix_seconds"]); raise SystemExit(0)
started = int(time.time()); value = {"schema_version": 1, "kind": "lehome_public_n15_paid_deadline_v1", "run_id": run_id, "lifecycle_plan_sha256": digest, "started_unix_seconds": started, "deadline_unix_seconds": started + 86400}
with Path(output).open("x", encoding="ascii") as stream: stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
os.chmod(output, 0o444); print(value["deadline_unix_seconds"])
PY
}
run_paid_stage() {
  local label="$1" limit_seconds="$2"; shift 2
  local aggregate_deadline now stage_deadline pid status
  aggregate_deadline="$(initialize_deadline)" || fail "aggregate paid deadline is invalid"
  now="$(date +%s)"; stage_deadline=$(( now + limit_seconds )); (( stage_deadline < aggregate_deadline )) || stage_deadline="$aggregate_deadline"
  (( now < stage_deadline )) || fail "$label has no remaining paid time"
  "$@" & pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    now="$(date +%s)"
    if (( now >= stage_deadline )); then
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" || true
      fail "$label exceeded its code-owned paid timeout"
    fi
    sleep 1
  done
  wait "$pid"; status=$?
  (( status == 0 )) || fail "$label failed"
}
remote_file_exists() { remote bash -s -- "$1" <<'SH'
set -euo pipefail
test -f "$1" && test ! -L "$1"
SH
}

verify_remote_training_chain() {
  remote bash -s -- "$REMOTE_ROOT" "$SOURCE_ROOT" "$SOURCE_RECEIPT" "$SNAPSHOTS_RECEIPT" "$TRAINING_ROOT" "$EXACT_VM_ID" "$PROTECTED_DISK_ID" "$TRAINING_IDENTITY_RECEIPT" <<'SH'
set -euo pipefail
root="$1"; source_root="$2"; source_receipt="$3"; snapshots="$4"; training_root="$5"; vm_id="$6"; disk_id="$7"; receipt="$8"
test -f "$receipt" && test ! -L "$receipt"
temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/lehome-n15-verify-training.XXXXXX")"
temporary="$temporary_root/receipt.json"
trap 'rm -rf -- "$temporary_root"' EXIT
python3 "$root/scripts/run_public_n15_reproduction.py" verify-training-output --checkout "$source_root" --source-receipt "$source_receipt" --resolved-snapshots-receipt "$snapshots" --vm-id "$vm_id" --disk-id "$disk_id" --training-root "$training_root" --output "$temporary" >/dev/null
cmp -s "$temporary" "$receipt"
SH
}

verify_remote_training_publication() {
  remote bash -s -- "$TRAINING_PUBLICATION_RECEIPT" "$PUBLIC_REPOSITORY" "n15-public/$RUN_ID/training" "$TRAINING_IDENTITY_RECEIPT" <<'SH'
set -euo pipefail
python3 - "$1" "$2" "$3" "$4" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
receipt, repository, prefix, training = map(Path if False else str, sys.argv[1:])
raw = Path(receipt).read_bytes(); value = json.loads(raw)
if raw != (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(): raise SystemExit("training publication is not canonical")
if (set(value) != {"schema_version", "kind", "repository", "remote_prefix", "immutable_revision", "entries", "anonymous_byte_readback_verified"} or value["schema_version"] != 1 or value["kind"] != "lehome_public_n15_training_publication_v1" or value["repository"] != repository or value["remote_prefix"] != prefix or re.fullmatch(r"[0-9a-f]{40}", value["immutable_revision"]) is None or value["anonymous_byte_readback_verified"] is not True or not isinstance(value["entries"], list) or not value["entries"]): raise SystemExit("training publication receipt is invalid")
if not any(item.get("path") == Path(training).name + "/training-identity.json" or item.get("path") == "training-identity.json" for item in value["entries"] if isinstance(item, dict)): raise SystemExit("training publication does not bind training identity")
PY
SH
}

verify_remote_focused_chain() {
  remote bash -s -- "$REMOTE_ROOT" "$FOCUSED_OUTPUT_ROOT" "$FOCUSED_OUTPUT_ROOT/publication.json" "$FOCUSED_PROMOTION_RECEIPT" <<'SH'
set -euo pipefail
root="$1"; output="$2"; publication="$3"; promotion="$4"; temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/lehome-n15-verify-focused.XXXXXX")"; temporary="$temporary_root/receipt.json"; trap 'rm -rf -- "$temporary_root"' EXIT
python3 "$root/scripts/run_official_lehome_comparison.py" verify-n15-focused --receipt "$output/comparison-receipt.json" --publication-receipt "$publication" --promotion-receipt "$temporary" >/dev/null
cmp -s "$temporary" "$promotion"
SH
}

verify_remote_harvest_chain() {
  remote bash -s -- "$REMOTE_ROOT" "$HARVEST_ROOT" "$REMOTE_PIPELINE_ROOT/harvest.publication.json" <<'SH'
set -euo pipefail
root="$1"; harvest="$2"; publication="$3"
python3 "$root/scripts/build_public_n15_harvest.py" verify --manifest "$harvest/manifest.json" --receipt "$harvest/manifest-receipt.json" >/dev/null
test -s "$publication" && test ! -L "$publication"
SH
}

fetch_remote_immutable() {
  local remote_path="$1" local_path="$2"
  [[ ! -e "$local_path" && ! -L "$local_path" ]] || fail "local immutable receipt already exists"
  remote bash -s -- "$remote_path" <<'SH' >"$local_path"
set -euo pipefail
test -f "$1" && test ! -L "$1"
cat -- "$1"
SH
  chmod 0444 "$local_path"
}

finalize_host_harvest_terminal() {
  python3 "$HARVEST_BUILDER" verify-terminal --manifest "$HARVEST_MANIFEST" --manifest-receipt "$HARVEST_MANIFEST_RECEIPT" --publication-receipt "$HARVEST_PUBLICATION_RECEIPT" --provider-receipt "$PROVIDER_STOPPED_RECEIPT" --output "$HARVEST_TERMINAL_RECEIPT" >/dev/null
}

validate_remote_runtime() {
  remote bash -s -- "$REMOTE_ROOT" "$REMOTE_PIPELINE_ROOT" "$SOURCE_ROOT" "$SOURCE_RECEIPT" "$SNAPSHOTS_RECEIPT" "$TRAINING_ROOT" "$RUNTIME_REVISION" "$EXACT_VM_ID" "$PROTECTED_DISK_ID" <<'SH'
set -euo pipefail
root="$1"; pipeline_root="$2"; source_root="$3"; source_receipt="$4"; snapshots="$5"; training_root="$6"; revision="$7"; vm_id="$8"; disk_id="$9"
test -f /var/lib/cloud/instance/boot-finished
mount_source="$(findmnt -T "$pipeline_root" --noheadings --output SOURCE)"
[[ "$mount_source" == /dev/* ]] && lsblk -n -o TYPE "$mount_source" | grep -Eq '^(part|lvm|crypt)$'
# Cloud-init attaches the exact Nebius secondary disk with device_id=lehome.
# Prove that its stable guest device backs this run's workspace mount.
test -e /dev/disk/by-id/virtio-lehome
[[ "$(lsblk -ndo MAJ:MIN /dev/disk/by-id/virtio-lehome)" == "$(findmnt -T "$pipeline_root" --noheadings --output MAJ:MIN)" ]]
nvidia-smi -L | grep -q .
test "$(git -C "$root" rev-parse HEAD)" = "$revision"
test -z "$(git -C "$root" status --porcelain --untracked-files=all)"
verified_inputs="$training_root/verified-inputs.json"
if [[ -f "$verified_inputs" ]]; then
  temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/lehome-n15-verify-inputs.XXXXXX")"
  trap 'rm -rf -- "$temporary_root"' EXIT
  python3 "$root/scripts/run_public_n15_reproduction.py" verify-inputs --checkout "$source_root" --source-receipt "$source_receipt" --resolved-snapshots-receipt "$snapshots" --vm-id "$vm_id" --disk-id "$disk_id" --output "$temporary_root/receipt.json" >/dev/null
  cmp -s "$temporary_root/receipt.json" "$verified_inputs"
else
  python3 "$root/scripts/run_public_n15_reproduction.py" verify-inputs --checkout "$source_root" --source-receipt "$source_receipt" --resolved-snapshots-receipt "$snapshots" --vm-id "$vm_id" --disk-id "$disk_id" --output "$verified_inputs" >/dev/null
fi
SH
}

train_stage() {
  remote bash -s -- "$REMOTE_ROOT" "$SOURCE_ROOT" "$SOURCE_RECEIPT" "$SNAPSHOTS_RECEIPT" "$TRAINING_ROOT" "$EXACT_VM_ID" "$PROTECTED_DISK_ID" "$TRAINING_HF_CACHE" "$TRAINING_PYTHON" "$LEROBOT_WHEEL" <<'SH'
set -euo pipefail
root="$1"; source_root="$2"; source_receipt="$3"; snapshots="$4"; training_root="$5"; vm_id="$6"; disk_id="$7"; hf_cache="$8"; python_bin="$9"; wheel="${10}"
test ! -e "$training_root/evidence/execution-manifest.json"
mkdir -p "$training_root/evidence" "$training_root/logs"
install -m 0444 "$source_receipt" "$training_root/evidence/source-receipt.json"
install -m 0444 "$snapshots" "$training_root/evidence/resolved-snapshots-receipt.json"
install -m 0444 "$source_root/uv.lock" "$training_root/evidence/uv.lock"
test -f "$wheel" && test ! -L "$wheel" && test -d "$hf_cache" && test ! -L "$hf_cache"
test -x "$python_bin" && test -x "$(dirname -- "$python_bin")/lerobot-train"
"$python_bin" -I -c 'import lerobot; from pathlib import Path; assert Path(lerobot.__file__).is_file()'
install -m 0444 "$wheel" "$training_root/evidence/lerobot-0.4.3-py3-none-any.whl"
"$python_bin" - "$training_root/evidence/runtime-receipt.json" "$training_root/evidence/uv.lock" "$training_root/evidence/lerobot-0.4.3-py3-none-any.whl" <<'PY'
import importlib.util, json, os, sys
from pathlib import Path
output, lock, wheel = map(Path, sys.argv[1:])
package = Path(importlib.util.find_spec("lerobot").origin).parent
value = {"schema_version": 1, "kind": "lehome_public_n15_training_runtime_v1", "python_executable": sys.executable, "lerobot_wheel_path": str(wheel), "lerobot_wheel_sha256": __import__("hashlib").sha256(wheel.read_bytes()).hexdigest(), "lerobot_package_root": str(package), "dependency_lock_path": str(lock), "dependency_lock_sha256": __import__("hashlib").sha256(lock.read_bytes()).hexdigest()}
with output.open("x", encoding="ascii") as stream: stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
os.chmod(output, 0o444)
PY
python3 "$root/scripts/run_public_n15_reproduction.py" render-training --checkout "$source_root" --source-receipt "$source_receipt" --resolved-snapshots-receipt "$snapshots" --vm-id "$vm_id" --disk-id "$disk_id" --output "$training_root/evidence/execution-manifest.json" >/dev/null
# The Task1 verifier seals this exact manifest, source/snapshot receipts,
# dependency lock, runtime receipt, train log, and checkpoint—not a hand-made
# approximation of a successful training result.
cd "$source_root"; export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$hf_cache"; "$(dirname -- "$python_bin")/lerobot-train" --config_path=configs/train_groot.yaml 2>&1 | tee "$training_root/logs/train.log"
python3 "$root/scripts/run_public_n15_reproduction.py" verify-training-output --checkout "$source_root" --source-receipt "$source_receipt" --resolved-snapshots-receipt "$snapshots" --vm-id "$vm_id" --disk-id "$disk_id" --training-root "$training_root" --output "$training_root/training-identity.json" >/dev/null
SH
}

publish_training_readback() {
  # The remote publisher uses a fresh, immutable prefix and anonymous byte readback.
  remote bash -s -- "$TRAINING_ROOT" "$PUBLIC_REPOSITORY" "n15-public/$RUN_ID/training" "$HF_TOKEN_FILE" <<'SH'
set -euo pipefail
root="$1"; repository="$2"; prefix="$3"; token_file="$4"; test -f "$token_file" && test ! -L "$token_file"; export HF_TOKEN="$(cat "$token_file")"
python3 - "$root" "$repository" "$prefix" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
root, repository, prefix = sys.argv[1:]; directory = Path(root); receipt = directory / "training-publication.json"
if receipt.exists(): raise SystemExit("training publication receipt already exists")
entries = [{"path": str(path.relative_to(directory)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(directory.rglob("*")) if path.is_file() and path != receipt]
commit = HfApi(token=os.environ["HF_TOKEN"]).upload_folder(repo_id=repository, repo_type="model", folder_path=str(directory), path_in_repo=prefix, commit_message="public N1.5 training " + prefix)
revision = str(commit.oid)
for entry in entries:
    fetched = hf_hub_download(repo_id=repository, repo_type="model", filename=prefix + "/" + entry["path"], revision=revision, token=False)
    if hashlib.sha256(Path(fetched).read_bytes()).hexdigest() != entry["sha256"]: raise SystemExit("anonymous training byte readback mismatch")
value = {"schema_version": 1, "kind": "lehome_public_n15_training_publication_v1", "repository": repository, "remote_prefix": prefix, "immutable_revision": revision, "entries": entries, "anonymous_byte_readback_verified": True}
with receipt.open("x", encoding="utf-8") as stream: json.dump(value, stream, sort_keys=True, separators=(",", ":")); stream.write("\n")
os.chmod(receipt, 0o444)
PY
SH
}
focused_stage() {
  remote bash -s -- "$REMOTE_ROOT" "$HF_TOKEN_FILE" "$RUNTIME_REVISION" "$SOURCE_ROOT" "$ASSETS_ROOT" "$METADATA_ROOT" "$TRAINING_ROOT/checkpoints/012000/pretrained_model" "$TRAINING_ROOT/training-identity.json" "$REMOTE_PIPELINE_ROOT/focused/candidate-config" "$REMOTE_PIPELINE_ROOT/focused/candidate-compatibility.json" "$REFERENCE_CHECKPOINT" "$REFERENCE_SANITIZED_CONFIG" "$REFERENCE_COMPATIBILITY" "$NATIVE_RUNTIME_EVIDENCE" "$NATIVE_DEPENDENCIES" "$FOCUSED_HF_CACHE" "$REMOTE_PIPELINE_ROOT/focused" "$PUBLIC_REPOSITORY" "$REMOTE_PIPELINE_ROOT/focused/publication.json" "$REMOTE_PIPELINE_ROOT/focused/promotion.json" <<'SH'
set -euo pipefail
root="$1"; token="$2"; shift 2; test -f "$token" && test ! -L "$token"; export HF_TOKEN="$(cat "$token")"
export LEHOME_OFFICIAL_RUNTIME_REVISION="$1" LEHOME_OFFICIAL_SOURCE_ROOT="$2" LEHOME_OFFICIAL_ASSETS_ROOT="$3" LEHOME_OFFICIAL_METADATA_ROOT="$4"
export LEHOME_N15_CANDIDATE_CHECKPOINT="$5" LEHOME_N15_CANDIDATE_IDENTITY_RECEIPT="$6" LEHOME_N15_CANDIDATE_SANITIZED_CONFIG_ROOT="$7" LEHOME_N15_CANDIDATE_COMPATIBILITY_RECEIPT="$8"
export LEHOME_N15_REFERENCE_CHECKPOINT="$9" LEHOME_N15_REFERENCE_SANITIZED_CONFIG_ROOT="${10}" LEHOME_N15_REFERENCE_COMPATIBILITY_RECEIPT="${11}" LEHOME_N15_NATIVE_RUNTIME_EVIDENCE_ROOT="${12}" LEHOME_N15_NATIVE_DEPENDENCIES_ROOT="${13}" LEHOME_N15_FOCUSED_HF_CACHE_ROOT="${14}" LEHOME_N15_FOCUSED_OUTPUT_ROOT="${15}" LEHOME_N15_FOCUSED_REPOSITORY="${16}" LEHOME_N15_FOCUSED_PUBLICATION_RECEIPT="${17}" LEHOME_N15_FOCUSED_PROMOTION_RECEIPT="${18}"
exec "$root/rollout_appliance/run_public_n15_focused_gate.sh"
SH
}
harvest_stage() {
  remote bash -s -- "$REMOTE_ROOT" "$HF_TOKEN_FILE" "$RUNTIME_REVISION" "$SOURCE_ROOT" "$TRAINING_ROOT/checkpoints/012000/pretrained_model" "$TRAINING_ROOT/training-identity.json" "$ROLLOUT_IMAGE_RECEIPT" "$REMOTE_PIPELINE_ROOT/harvest" "$PUBLIC_REPOSITORY" "$REMOTE_PIPELINE_ROOT/harvest.publication.json" "$REMOTE_PIPELINE_ROOT/harvest.provider-stopped.json" "$REMOTE_PIPELINE_ROOT/harvest.terminal.json" <<'SH'
set -euo pipefail
root="$1"; token="$2"; shift 2; test -f "$token" && test ! -L "$token"; export HF_TOKEN="$(cat "$token")"
export LEHOME_N15_RUNTIME_REVISION="$1" LEHOME_N15_PUBLIC_SOURCE_ROOT="$2" LEHOME_N15_CHECKPOINT_ROOT="$3" LEHOME_N15_TRAINING_IDENTITY_RECEIPT="$4" LEHOME_N15_ROLLOUT_IMAGE_RECEIPT="$5" LEHOME_N15_HARVEST_ROOT="$6" LEHOME_N15_PUBLIC_HF_REPOSITORY="$7" LEHOME_N15_PUBLICATION_RECEIPT="$8" LEHOME_N15_PROVIDER_STOPPED_RECEIPT="$9" LEHOME_N15_TERMINAL_RECEIPT="${10}" LEHOME_N15_DEFER_PROVIDER_STOP=1
exec "$root/rollout_appliance/run_public_n15_harvest.sh"
SH
}

[[ $# -eq 0 ]] || fail "this wrapper accepts no positional arguments"
command -v nebius >/dev/null 2>&1 || fail "Nebius CLI is unavailable"
command -v ssh >/dev/null 2>&1 || fail "SSH is unavailable"
require_abs_dir "$PIPELINE_ROOT" "pipeline receipt root"; require_abs_file "$BUILDER" "checked-in lifecycle planner"
require_abs_file "$PROVIDER_VERIFIER" "checked-in exact Nebius provider parser"; require_abs_file "$HARVEST_BUILDER" "checked-in harvest provider parser"
[[ "$PUBLIC_REPOSITORY" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ && -n "$SSH_TARGET" && "$REMOTE_ROOT" == /* && "$REMOTE_PIPELINE_ROOT" == /* && -n "$ASSETS_ROOT" && -n "$METADATA_ROOT" && -n "$REFERENCE_CHECKPOINT" && -n "$REFERENCE_SANITIZED_CONFIG" && -n "$REFERENCE_COMPATIBILITY" && -n "$NATIVE_RUNTIME_EVIDENCE" && -n "$NATIVE_DEPENDENCIES" && -n "$FOCUSED_HF_CACHE" && -n "$ROLLOUT_IMAGE_RECEIPT" && -n "$TRAINING_HF_CACHE" && -n "$LEROBOT_WHEEL" ]] || fail "all canonical remote inputs are required"
[[ "$TRAINING_ROOT" == "$REMOTE_PIPELINE_ROOT/training" ]] || fail "training root must be this run's canonical remote training directory"
[[ "$REMOTE_PIPELINE_ROOT" == "$REMOTE_RUNS_BASE/$RUN_ID" ]] || fail "remote pipeline root must be the canonical run-specific directory"
# Immutable pre-start cost admission: run_public_n15_reproduction.py lifecycle-plan.
if [[ ! -e "$PLAN_RECEIPT" ]]; then python3 "$BUILDER" lifecycle-plan --run-id "$RUN_ID" --repository "$PUBLIC_REPOSITORY" --remote-pipeline-root "$REMOTE_PIPELINE_ROOT" --budget-usd "$MAX_BUDGET_USD" --estimated-cost-usd "$ESTIMATED_COST_USD" --output "$PLAN_RECEIPT" >/dev/null; fi
python3 "$BUILDER" verify-lifecycle-plan --run-id "$RUN_ID" --repository "$PUBLIC_REPOSITORY" --remote-pipeline-root "$REMOTE_PIPELINE_ROOT" --budget-usd "$MAX_BUDGET_USD" --estimated-cost-usd "$ESTIMATED_COST_USD" --output "$PLAN_RECEIPT" >/dev/null
# A complete immutable terminal chain is terminal even if a prior controller
# crashed after it.  Observe current provider state before *any* start: never
# rerun a paid stage from a completed run, and clean up a stale RUNNING VM.
if [[ -f "$HARVEST_TERMINAL_RECEIPT" ]]; then
  terminal_temp_root="$(mktemp -d "${TMPDIR:-/tmp}/lehome-n15-terminal-preflight.XXXXXX")"
  if python3 "$HARVEST_BUILDER" verify-terminal --manifest "$HARVEST_MANIFEST" --manifest-receipt "$HARVEST_MANIFEST_RECEIPT" --publication-receipt "$HARVEST_PUBLICATION_RECEIPT" --provider-receipt "$PROVIDER_STOPPED_RECEIPT" --output "$terminal_temp_root/receipt.json" >/dev/null; then
    if capture_exact_provider_state STOPPED "$terminal_temp_root/provider.json"; then
      rm -rf -- "$terminal_temp_root"; PIPELINE_COMPLETE=1; exit 0
    fi
    rm -rf -- "$terminal_temp_root"
    stop_exact_vm || fail "completed run left the exact VM running and it could not be stopped"
    PIPELINE_COMPLETE=1; exit 0
  fi
  rm -rf -- "$terminal_temp_root"
  fail "existing terminal receipt chain is invalid"
fi
response="$PIPELINE_ROOT/.provider-start.$$.json"; capture_exact_provider_state STOPPED "$response" || fail "Nebius Compute API is unavailable or exact VM is not stopped"; rm -f -- "$response"
nebius compute instance start --id "$EXACT_VM_ID" --format json --no-browser --no-progress --no-check-update --retries 1 --timeout 60s >/dev/null
for _ in {1..60}; do capture_exact_provider_state RUNNING "$response" && break; rm -f -- "$response"; sleep 2; done
capture_exact_provider_state RUNNING "$response" || fail "exact VM did not reach RUNNING"; rm -f -- "$response"
validate_remote_runtime || fail "runtime/cloud-init/workspace/GPU/upstream gate failed"
if ! remote_file_exists "$TRAINING_IDENTITY_RECEIPT"; then run_paid_stage train "$TRAIN_TIMEOUT_SECONDS" train_stage; fi
verify_remote_training_chain || fail "training receipt chain failed"
if ! remote_file_exists "$TRAINING_PUBLICATION_RECEIPT"; then publish_training_readback || fail "training publication/readback failed"; fi
verify_remote_training_publication || fail "training publication chain failed"
if ! remote_file_exists "$FOCUSED_PROMOTION_RECEIPT"; then run_paid_stage focused_gate "$FOCUSED_TIMEOUT_SECONDS" focused_stage; fi
verify_remote_focused_chain || fail "focused receipt chain failed"
if [[ ! -e "$HARVEST_TERMINAL_RECEIPT" ]]; then
  run_paid_stage harvest "$HARVEST_TIMEOUT_SECONDS" harvest_stage
  verify_remote_harvest_chain || fail "harvest pre-stop receipt chain failed"
  fetch_remote_immutable "$HARVEST_ROOT/manifest.json" "$HARVEST_MANIFEST"
  fetch_remote_immutable "$HARVEST_ROOT/manifest-receipt.json" "$HARVEST_MANIFEST_RECEIPT"
  fetch_remote_immutable "$REMOTE_PIPELINE_ROOT/harvest.publication.json" "$HARVEST_PUBLICATION_RECEIPT"
  stop_exact_vm || fail "exact VM could not be stopped"
  finalize_host_harvest_terminal || fail "host harvest terminal verification failed"
else
  python3 "$HARVEST_BUILDER" verify-terminal --manifest "$HARVEST_MANIFEST" --manifest-receipt "$HARVEST_MANIFEST_RECEIPT" --publication-receipt "$HARVEST_PUBLICATION_RECEIPT" --provider-receipt "$PROVIDER_STOPPED_RECEIPT" --output "$(mktemp -d "${TMPDIR:-/tmp}/lehome-n15-verify-terminal.XXXXXX")/receipt.json" >/dev/null || fail "existing host harvest terminal chain failed"
fi
stop_exact_vm || fail "exact VM could not be stopped"; PIPELINE_COMPLETE=1
