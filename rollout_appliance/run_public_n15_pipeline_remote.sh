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
readonly MAX_BUDGET_USD="${LEHOME_N15_MAX_BUDGET_USD:-100}"
readonly ESTIMATED_COST_USD="${LEHOME_N15_ESTIMATED_COST_USD:-}"
readonly PUBLIC_REPOSITORY="${LEHOME_N15_PUBLIC_HF_REPOSITORY:-}"
readonly SOURCE_ROOT="${LEHOME_N15_PUBLIC_SOURCE_ROOT:-}"
readonly SOURCE_RECEIPT="${LEHOME_N15_SOURCE_RECEIPT:-}"
readonly SNAPSHOTS_RECEIPT="${LEHOME_N15_RESOLVED_SNAPSHOTS_RECEIPT:-}"
readonly TRAINING_ROOT="${LEHOME_N15_TRAINING_ROOT:-}"
readonly HF_TOKEN_FILE="${LEHOME_N15_HF_TOKEN_FILE:-}"
readonly RUNTIME_REVISION="${LEHOME_N15_RUNTIME_REVISION:-}"
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
readonly TRAINING_IDENTITY_RECEIPT="$TRAINING_ROOT/training-identity.json"
readonly TRAINING_PUBLICATION_RECEIPT="$TRAINING_ROOT/training-publication.json"
readonly FOCUSED_OUTPUT_ROOT="$REMOTE_PIPELINE_ROOT/focused"
readonly FOCUSED_PROMOTION_RECEIPT="$FOCUSED_OUTPUT_ROOT/promotion.json"
readonly HARVEST_ROOT="$REMOTE_PIPELINE_ROOT/harvest"
readonly HARVEST_TERMINAL_RECEIPT="$REMOTE_PIPELINE_ROOT/harvest.terminal.json"
readonly PROVIDER_STOPPED_RECEIPT="$PIPELINE_ROOT/provider-stopped.json"
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
    capture_exact_provider_state STOPPED "$observation"
    rm -f -- "$observation"; return
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
temporary="$(mktemp "$training_root/.verify-training.XXXXXX")"
trap 'rm -f -- "$temporary"' EXIT
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
root="$1"; output="$2"; publication="$3"; promotion="$4"; temporary="$(mktemp "$output/.verify-focused.XXXXXX")"; trap 'rm -f -- "$temporary"' EXIT
python3 "$root/scripts/run_official_lehome_comparison.py" verify-n15-focused --receipt "$output/comparison-receipt.json" --publication-receipt "$publication" --promotion-receipt "$temporary" >/dev/null
cmp -s "$temporary" "$promotion"
SH
}

verify_remote_harvest_chain() {
  remote bash -s -- "$REMOTE_ROOT" "$HARVEST_ROOT" "$REMOTE_PIPELINE_ROOT/harvest.publication.json" "$REMOTE_PIPELINE_ROOT/harvest.provider-stopped.json" "$HARVEST_TERMINAL_RECEIPT" <<'SH'
set -euo pipefail
root="$1"; harvest="$2"; publication="$3"; stopped="$4"; terminal="$5"; temporary="$(mktemp "$harvest/.verify-harvest.XXXXXX")"; trap 'rm -f -- "$temporary"' EXIT
python3 "$root/scripts/build_public_n15_harvest.py" verify-terminal --manifest "$harvest/manifest.json" --manifest-receipt "$harvest/manifest-receipt.json" --publication-receipt "$publication" --provider-receipt "$stopped" --output "$temporary" >/dev/null
cmp -s "$temporary" "$terminal"
SH
}

validate_remote_runtime() {
  remote bash -s -- "$REMOTE_ROOT" "$REMOTE_PIPELINE_ROOT" "$SOURCE_ROOT" "$SOURCE_RECEIPT" "$SNAPSHOTS_RECEIPT" "$TRAINING_ROOT" "$RUNTIME_REVISION" "$EXACT_VM_ID" "$PROTECTED_DISK_ID" <<'SH'
set -euo pipefail
root="$1"; pipeline_root="$2"; source_root="$3"; source_receipt="$4"; snapshots="$5"; training_root="$6"; revision="$7"; vm_id="$8"; disk_id="$9"
test -f /var/lib/cloud/instance/boot-finished
mount_source="$(findmnt -T "$pipeline_root" --noheadings --output SOURCE)"
[[ "$mount_source" == /dev/* ]] && lsblk -n -o TYPE "$mount_source" | grep -Eq '^(part|lvm|crypt)$'
nvidia-smi -L | grep -q .
test "$(git -C "$root" rev-parse HEAD)" = "$revision"
test -z "$(git -C "$root" status --porcelain --untracked-files=all)"
python3 "$root/scripts/run_public_n15_reproduction.py" verify-inputs --checkout "$source_root" --source-receipt "$source_receipt" --resolved-snapshots-receipt "$snapshots" --vm-id "$vm_id" --disk-id "$disk_id" --output "$training_root/verified-inputs.json" >/dev/null
SH
}

train_stage() {
  remote bash -s -- "$REMOTE_ROOT" "$SOURCE_ROOT" "$SOURCE_RECEIPT" "$SNAPSHOTS_RECEIPT" "$TRAINING_ROOT" "$EXACT_VM_ID" "$PROTECTED_DISK_ID" <<'SH'
set -euo pipefail
root="$1"; source_root="$2"; source_receipt="$3"; snapshots="$4"; training_root="$5"; vm_id="$6"; disk_id="$7"
test ! -e "$training_root/training-execution.json"
python3 "$root/scripts/run_public_n15_reproduction.py" render-training --checkout "$source_root" --source-receipt "$source_receipt" --resolved-snapshots-receipt "$snapshots" --vm-id "$vm_id" --disk-id "$disk_id" --output "$training_root/training-execution.json" >/dev/null
cd "$source_root"; export HF_HUB_OFFLINE=1; lerobot-train --config_path=configs/train_groot.yaml
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
export LEHOME_N15_RUNTIME_REVISION="$1" LEHOME_N15_PUBLIC_SOURCE_ROOT="$2" LEHOME_N15_CHECKPOINT_ROOT="$3" LEHOME_N15_TRAINING_IDENTITY_RECEIPT="$4" LEHOME_N15_ROLLOUT_IMAGE_RECEIPT="$5" LEHOME_N15_HARVEST_ROOT="$6" LEHOME_N15_PUBLIC_HF_REPOSITORY="$7" LEHOME_N15_PUBLICATION_RECEIPT="$8" LEHOME_N15_PROVIDER_STOPPED_RECEIPT="$9" LEHOME_N15_TERMINAL_RECEIPT="${10}"
exec "$root/rollout_appliance/run_public_n15_harvest.sh"
SH
}

[[ $# -eq 0 ]] || fail "this wrapper accepts no positional arguments"
command -v nebius >/dev/null 2>&1 || fail "Nebius CLI is unavailable"
command -v ssh >/dev/null 2>&1 || fail "SSH is unavailable"
require_abs_dir "$PIPELINE_ROOT" "pipeline receipt root"; require_abs_file "$BUILDER" "checked-in lifecycle planner"
require_abs_file "$PROVIDER_VERIFIER" "checked-in exact Nebius provider parser"; require_abs_file "$HARVEST_BUILDER" "checked-in harvest provider parser"
[[ "$PUBLIC_REPOSITORY" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ && -n "$SSH_TARGET" && "$REMOTE_ROOT" == /* && "$REMOTE_PIPELINE_ROOT" == /* && -n "$ESTIMATED_COST_USD" && -n "$ASSETS_ROOT" && -n "$METADATA_ROOT" && -n "$REFERENCE_CHECKPOINT" && -n "$REFERENCE_SANITIZED_CONFIG" && -n "$REFERENCE_COMPATIBILITY" && -n "$NATIVE_RUNTIME_EVIDENCE" && -n "$NATIVE_DEPENDENCIES" && -n "$FOCUSED_HF_CACHE" && -n "$ROLLOUT_IMAGE_RECEIPT" ]] || fail "all canonical remote inputs and the cost estimate are required"
[[ "$TRAINING_ROOT" == "$REMOTE_PIPELINE_ROOT/training" ]] || fail "training root must be this run's canonical remote training directory"
# Immutable pre-start cost admission: run_public_n15_reproduction.py lifecycle-plan.
if [[ ! -e "$PLAN_RECEIPT" ]]; then python3 "$BUILDER" lifecycle-plan --run-id "$RUN_ID" --repository "$PUBLIC_REPOSITORY" --budget-usd "$MAX_BUDGET_USD" --estimated-cost-usd "$ESTIMATED_COST_USD" --output "$PLAN_RECEIPT" >/dev/null; fi
python3 "$BUILDER" verify-lifecycle-plan --run-id "$RUN_ID" --repository "$PUBLIC_REPOSITORY" --budget-usd "$MAX_BUDGET_USD" --estimated-cost-usd "$ESTIMATED_COST_USD" --output "$PLAN_RECEIPT" >/dev/null
response="$PIPELINE_ROOT/.provider-start.$$.json"; capture_exact_provider_state STOPPED "$response" || fail "Nebius Compute API is unavailable or exact VM is not stopped"; rm -f -- "$response"
nebius compute instance start --id "$EXACT_VM_ID" --format json --no-browser --no-progress --no-check-update --retries 1 --timeout 60s >/dev/null
for _ in {1..60}; do capture_exact_provider_state RUNNING "$response" && break; rm -f -- "$response"; sleep 2; done
capture_exact_provider_state RUNNING "$response" || fail "exact VM did not reach RUNNING"; rm -f -- "$response"
validate_remote_runtime || fail "runtime/cloud-init/workspace/GPU/upstream gate failed"
if ! remote_file_exists "$TRAINING_IDENTITY_RECEIPT"; then train_stage || fail "training failed"; fi
verify_remote_training_chain || fail "training receipt chain failed"
if ! remote_file_exists "$TRAINING_PUBLICATION_RECEIPT"; then publish_training_readback || fail "training publication/readback failed"; fi
verify_remote_training_publication || fail "training publication chain failed"
if ! remote_file_exists "$FOCUSED_PROMOTION_RECEIPT"; then focused_stage || fail "focused gate failed"; fi
verify_remote_focused_chain || fail "focused receipt chain failed"
if ! remote_file_exists "$HARVEST_TERMINAL_RECEIPT"; then harvest_stage || fail "harvest failed"; fi
verify_remote_harvest_chain || fail "harvest receipt chain failed"
stop_exact_vm || fail "exact VM could not be stopped"; PIPELINE_COMPLETE=1
