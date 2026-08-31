#!/usr/bin/env bash
# Cost-bounded single-VM controller for the pinned public GR00T N1.5 path.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly BUILDER="$REPO_ROOT/scripts/run_public_n15_reproduction.py"
readonly EXACT_VM_ID="computeinstance-u00t6xfqhadrcmssa2"
readonly PROTECTED_DISK_ID="computedisk-u00pbe55crxy7jr56x"
readonly RUN_ID="${LEHOME_N15_RUN_ID:-}"
readonly PIPELINE_ROOT="${LEHOME_N15_PIPELINE_ROOT:-}"
readonly SSH_TARGET="${LEHOME_N15_SSH_TARGET:-}"
readonly REMOTE_ROOT="${LEHOME_N15_REMOTE_ROOT:-}"
readonly EXPECTED_IMAGE_ID="${LEHOME_N15_EXPECTED_IMAGE_ID:-}"
readonly MAX_BUDGET_USD="${LEHOME_N15_MAX_BUDGET_USD:-100}"
readonly ESTIMATED_COST_USD="${LEHOME_N15_ESTIMATED_COST_USD:-}"
readonly PUBLIC_REPOSITORY="${LEHOME_N15_PUBLIC_HF_REPOSITORY:-}"
readonly SOURCE_ROOT="${LEHOME_N15_PUBLIC_SOURCE_ROOT:-}"
readonly SOURCE_RECEIPT="${LEHOME_N15_SOURCE_RECEIPT:-}"
readonly SNAPSHOTS_RECEIPT="${LEHOME_N15_RESOLVED_SNAPSHOTS_RECEIPT:-}"
readonly TRAINING_ROOT="${LEHOME_N15_TRAINING_ROOT:-}"
readonly HF_TOKEN_FILE="${LEHOME_N15_HF_TOKEN_FILE:-}"
readonly RUNTIME_REVISION="${LEHOME_N15_RUNTIME_REVISION:-}"
readonly PLAN_RECEIPT="$PIPELINE_ROOT/lifecycle-plan.json"
readonly TRAINING_IDENTITY_RECEIPT="$PIPELINE_ROOT/training-identity.json"
readonly TRAINING_PUBLICATION_RECEIPT="$PIPELINE_ROOT/training-publication.json"
readonly FOCUSED_PROMOTION_RECEIPT="$PIPELINE_ROOT/focused-promotion.json"
readonly HARVEST_TERMINAL_RECEIPT="$PIPELINE_ROOT/harvest-terminal.json"
readonly PROVIDER_STOPPED_RECEIPT="$PIPELINE_ROOT/provider-stopped.json"
PIPELINE_COMPLETE=0

fail() { printf 'error: %s\n' "$*" >&2; exit 2; }
require_abs_dir() { [[ "$1" == /* && "$1" != *".."* && -d "$1" && ! -L "$1" ]] || fail "$2 is unavailable or unsafe"; }
require_abs_file() { [[ "$1" == /* && "$1" != *".."* && -f "$1" && ! -L "$1" ]] || fail "$2 is unavailable or unsafe"; }
provider_get() { nebius compute instance get --id "$EXACT_VM_ID" --format json --no-browser --no-progress --no-check-update --retries 1 --timeout 60s; }

validate_provider_response() {
  local response="$1" expected_state="$2"
  python3 - "$response" "$expected_state" "$EXACT_VM_ID" "$PROTECTED_DISK_ID" "$EXPECTED_IMAGE_ID" <<'PY'
import json, sys
from pathlib import Path
raw, expected, vm_id, disk_id, image_id = sys.argv[1:]
value = json.loads(Path(raw).read_text(encoding="utf-8"))
def leaves(item):
    if isinstance(item, dict):
        for child in item.values(): yield from leaves(child)
    elif isinstance(item, list):
        for child in item: yield from leaves(child)
    elif isinstance(item, (str, int, float, bool)): yield str(item)
tokens = set(leaves(value))
for needed in (vm_id, disk_id, image_id, expected):
    if needed not in tokens: raise SystemExit(f"provider response lacks {needed}")
PY
}

stop_exact_vm() {
  local response="$PIPELINE_ROOT/.provider-stop.$$.json"
  if provider_get >"$response" && validate_provider_response "$response" STOPPED; then rm -f -- "$response"; return; fi
  nebius compute instance stop --id "$EXACT_VM_ID" --format json --no-browser --no-progress --no-check-update --retries 1 --timeout 60s >/dev/null || return 1
  for _ in {1..60}; do
    if provider_get >"$response" && validate_provider_response "$response" STOPPED; then
      python3 - "$response" "$PROVIDER_STOPPED_RECEIPT" <<'PY'
import json, os, sys
from pathlib import Path
raw, output = Path(sys.argv[1]), Path(sys.argv[2])
if output.exists(): raise SystemExit("provider stop receipt already exists")
value = {"schema_version": 1, "kind": "lehome_public_n15_provider_stop_v1", "state": "STOPPED", "protected_disk_preserved": True, "instance": json.loads(raw.read_text())}
with output.open("x", encoding="utf-8") as stream: json.dump(value, stream, sort_keys=True, separators=(",", ":")); stream.write("\n")
os.chmod(output, 0o444)
PY
      rm -f -- "$response"; return
    fi
    sleep 2
  done
  rm -f -- "$response"; return 1
}
trap stop_exact_vm EXIT
trap 'exit 130' INT TERM

require_stage_receipt() {
  local stage="$1" receipt="$2"
  require_abs_file "$receipt" "$stage immutable receipt"
  python3 - "$stage" "$receipt" <<'PY'
import json, sys
stage, receipt = sys.argv[1:]
value = json.load(open(receipt, encoding="utf-8"))
if not isinstance(value, dict): raise SystemExit("immutable receipt is not an object")
if stage == "train" and value.get("kind") != "lehome_public_n15_verified_training_output_v1": raise SystemExit("training receipt is not accepted")
if stage == "focused_gate" and (value.get("status") != "pass" or value.get("publication_readback_verified") is not True): raise SystemExit("focused gate has not passed readback")
if stage == "harvest" and value.get("kind") != "lehome_public_n15_harvest_terminal_v1": raise SystemExit("harvest receipt is not terminal")
PY
}
remote() { ssh -o BatchMode=yes "$SSH_TARGET" "$@"; }

validate_remote_runtime() {
  remote bash -s -- "$REMOTE_ROOT" "$SOURCE_ROOT" "$SOURCE_RECEIPT" "$SNAPSHOTS_RECEIPT" "$TRAINING_ROOT" "$RUNTIME_REVISION" "$EXACT_VM_ID" "$PROTECTED_DISK_ID" <<'SH'
set -euo pipefail
root="$1"; source_root="$2"; source_receipt="$3"; snapshots="$4"; training_root="$5"; revision="$6"; vm_id="$7"; disk_id="$8"
test -f /var/lib/cloud/instance/boot-finished
findmnt -T "$root" >/dev/null
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
focused_stage() { remote bash -lc "cd $(printf '%q' "$REMOTE_ROOT") && ./rollout_appliance/run_public_n15_focused_gate.sh"; }
harvest_stage() { remote bash -lc "cd $(printf '%q' "$REMOTE_ROOT") && ./rollout_appliance/run_public_n15_harvest.sh"; }

[[ $# -eq 0 ]] || fail "this wrapper accepts no positional arguments"
command -v nebius >/dev/null 2>&1 || fail "Nebius CLI is unavailable"
command -v ssh >/dev/null 2>&1 || fail "SSH is unavailable"
require_abs_dir "$PIPELINE_ROOT" "pipeline receipt root"; require_abs_file "$BUILDER" "checked-in lifecycle planner"
[[ "$EXPECTED_IMAGE_ID" =~ ^computeimage-[a-z0-9]+$ || "$EXPECTED_IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || fail "expected exact image identity is invalid"
[[ "$PUBLIC_REPOSITORY" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ && -n "$SSH_TARGET" && "$REMOTE_ROOT" == /* && -n "$ESTIMATED_COST_USD" ]] || fail "remote target/root, repository, and cost estimate are required"
# Immutable pre-start cost admission: run_public_n15_reproduction.py lifecycle-plan.
if [[ ! -e "$PLAN_RECEIPT" ]]; then python3 "$BUILDER" lifecycle-plan --run-id "$RUN_ID" --budget-usd "$MAX_BUDGET_USD" --estimated-cost-usd "$ESTIMATED_COST_USD" --output "$PLAN_RECEIPT" >/dev/null; fi
require_abs_file "$PLAN_RECEIPT" "immutable lifecycle plan"
response="$PIPELINE_ROOT/.provider-start.$$.json"; provider_get >"$response" || fail "Nebius Compute API is unavailable"; validate_provider_response "$response" STOPPED || fail "exact VM is not stopped with expected image/disk"; rm -f -- "$response"
nebius compute instance start --id "$EXACT_VM_ID" --format json --no-browser --no-progress --no-check-update --retries 1 --timeout 60s >/dev/null
for _ in {1..60}; do provider_get >"$response" && validate_provider_response "$response" RUNNING && break; sleep 2; done
validate_provider_response "$response" RUNNING || fail "exact VM did not reach RUNNING"; rm -f -- "$response"
validate_remote_runtime || fail "runtime/cloud-init/workspace/GPU/upstream gate failed"
if [[ ! -e "$TRAINING_IDENTITY_RECEIPT" ]]; then train_stage || fail "training failed"; fi
require_stage_receipt train "$TRAINING_IDENTITY_RECEIPT"
if [[ ! -e "$TRAINING_PUBLICATION_RECEIPT" ]]; then publish_training_readback || fail "training publication/readback failed"; fi
require_abs_file "$TRAINING_PUBLICATION_RECEIPT" "training immutable publication receipt"
if [[ ! -e "$FOCUSED_PROMOTION_RECEIPT" ]]; then focused_stage || fail "focused gate failed"; fi
require_stage_receipt focused_gate "$FOCUSED_PROMOTION_RECEIPT"
if [[ ! -e "$HARVEST_TERMINAL_RECEIPT" ]]; then harvest_stage || fail "harvest failed"; fi
require_stage_receipt harvest "$HARVEST_TERMINAL_RECEIPT"
stop_exact_vm || fail "exact VM could not be stopped"; PIPELINE_COMPLETE=1
