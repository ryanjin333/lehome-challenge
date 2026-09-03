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
readonly RUNTIME_IMAGE_ID="sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7"
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
readonly SSH_READINESS_ATTEMPTS=18
readonly SSH_READINESS_CONNECT_TIMEOUT_SECONDS=2
readonly SSH_READINESS_HARD_TIMEOUT_SECONDS=3
readonly SSH_READINESS_REAP_TIMEOUT_SECONDS=1
readonly SSH_READINESS_INTERVAL_SECONDS=2
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
readonly TRAINING_UV="${LEHOME_N15_TRAINING_UV:-}"
readonly LEROBOT_WHEEL="${LEHOME_N15_LEROBOT_WHEEL:-}"
readonly RESUME_PARTIAL="${LEHOME_N15_RESUME_PARTIAL:-0}"
readonly RESUME_CHECKPOINT="${LEHOME_N15_RESUME_CHECKPOINT:-}"
readonly RESUME_STEP="${LEHOME_N15_RESUME_STEP:-}"
readonly RESUME_ATTEMPT_ID="${LEHOME_N15_RESUME_ATTEMPT_ID:-}"
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
readonly HOST_TRAINING_STAGE_RECEIPT="$PIPELINE_ROOT/host-stage-training-complete.json"
readonly HOST_FOCUSED_STAGE_RECEIPT="$PIPELINE_ROOT/host-stage-focused-complete.json"
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
PROVIDER_CLEANUP_REQUIRED=0
ACTIVE_PAID_STAGE_PID=""
ACTIVE_PAID_STAGE_PGID=""
CONTROLLER_LOCK_FD=""
CONTROLLER_LOCK_PATH=""
PRESTART_ADMITTED_STAGE=""

fail() { printf 'error: %s\n' "$*" >&2; exit 2; }
require_abs_dir() { [[ "$1" == /* && "$1" != *".."* && -d "$1" && ! -L "$1" ]] || fail "$2 is unavailable or unsafe"; }
require_abs_file() { [[ "$1" == /* && "$1" != *".."* && -f "$1" && ! -L "$1" ]] || fail "$2 is unavailable or unsafe"; }
verify_conservative_task_budget() {
  python3 - "$MAX_BUDGET_USD" "$PROVIDER_HOURLY_CEILING_USD" "${1:-}" <<'PY'
import json, math, stat, sys, time
from pathlib import Path

budget, ceiling = map(float, sys.argv[1:3])
deadline_path = sys.argv[3]
window_seconds = 86400
if not math.isfinite(budget) or not (0 < budget <= 100):
    raise SystemExit("conservative provider budget is invalid")
maximum_cost = ceiling * window_seconds / 3600
if maximum_cost > budget:
    raise SystemExit("conservative provider budget is below the maximum paid window")
if not deadline_path:
    raise SystemExit(0)
path = Path(deadline_path)
metadata = path.lstat()
raw = path.read_bytes()
value = json.loads(raw)
canonical = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
started, deadline = value.get("started_unix_seconds"), value.get("deadline_unix_seconds")
if (
    path.is_symlink() or not stat.S_ISREG(metadata.st_mode)
    or stat.S_IMODE(metadata.st_mode) != 0o444 or raw != canonical
    or type(started) is not int or type(deadline) is not int
    or deadline != started + window_seconds
):
    raise SystemExit("conservative provider budget deadline is invalid")
elapsed_seconds = int(time.time()) - started
if elapsed_seconds < 0 or ceiling * elapsed_seconds / 3600 > budget:
    raise SystemExit("conservative provider budget deadline elapsed time exceeds the task budget")
PY
}
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
trap 'exit 130' INT TERM

acquire_controller_lock() {
  local runtime_base
  if [[ -n "${LEHOME_N15_CONTROLLER_LOCK_TEST_ROOT:-}" ]]; then
    case "${PYTEST_CURRENT_TEST:-}" in
      tests/infrastructure/test_public_n15_pipeline_remote.py::*) ;;
      *) return 1 ;;
    esac
    runtime_base="$(cd "$LEHOME_N15_CONTROLLER_LOCK_TEST_ROOT" && pwd -P)" || return 1
  else
    runtime_base="$(cd /var/tmp && pwd -P)" || return 1
  fi
  local lock_root="$runtime_base/lehome-public-n15-controller-$(id -u)"
  local lock_path="$lock_root/$EXACT_VM_ID.lock"
  CONTROLLER_LOCK_PATH="$lock_path"
  python3 - "$runtime_base" "$lock_root" "$lock_path" <<'PY' || return 1
import os
from pathlib import Path
import stat
import sys

runtime_base, lock_root, lock_path = map(Path, sys.argv[1:4])
if (
    not runtime_base.is_absolute()
    or runtime_base.is_symlink()
    or not runtime_base.is_dir()
    or runtime_base.resolve(strict=True) != runtime_base
    or lock_root.parent != runtime_base
    or lock_path.parent != lock_root
    or lock_path.name != "computeinstance-u00t6xfqhadrcmssa2.lock"
):
    raise SystemExit(73)
try:
    lock_root.mkdir(mode=0o700)
except FileExistsError:
    pass
root_metadata = lock_root.lstat()
if (
    not stat.S_ISDIR(root_metadata.st_mode)
    or root_metadata.st_uid != os.getuid()
    or stat.S_IMODE(root_metadata.st_mode) != 0o700
):
    raise SystemExit(73)
try:
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
except OSError:
    raise SystemExit(73)
try:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
    ):
        raise SystemExit(73)
finally:
    os.close(descriptor)
PY
  exec {CONTROLLER_LOCK_FD}<>"$lock_path" || return 1
  if ! python3 - "$CONTROLLER_LOCK_FD" "$lock_path" "$RUN_ID" "$SCRIPT_DIR/run_public_n15_pipeline_remote.sh" "$$" <<'PY'
import fcntl
import json
import os
from pathlib import Path
import stat
import sys

descriptor = int(sys.argv[1])
lock_path = Path(sys.argv[2])
run_id, script_path, controller_pid = sys.argv[3], sys.argv[4], int(sys.argv[5])
metadata = os.fstat(descriptor)
path_metadata = lock_path.lstat()
if (
    lock_path.is_symlink()
    or not stat.S_ISREG(metadata.st_mode)
    or metadata.st_uid != os.getuid()
    or stat.S_IMODE(metadata.st_mode) != 0o600
    or metadata.st_nlink != 1
    or (metadata.st_dev, metadata.st_ino) != (path_metadata.st_dev, path_metadata.st_ino)
):
    raise SystemExit(73)
try:
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit(73)
value = {
    "schema_version": 1,
    "kind": "lehome_public_n15_controller_lock_v1",
    "run_id": run_id,
    "controller_pid": controller_pid,
    "acquisition_helper_pid": os.getpid(),
    "script_path": script_path,
}
payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
os.lseek(descriptor, 0, os.SEEK_SET)
os.ftruncate(descriptor, 0)
remaining = memoryview(payload)
while remaining:
    remaining = remaining[os.write(descriptor, remaining):]
os.fsync(descriptor)
PY
  then
    exec {CONTROLLER_LOCK_FD}>&-
    CONTROLLER_LOCK_FD=""
    return 1
  fi
  return 0
}

release_controller_lock() {
  if [[ -n "$CONTROLLER_LOCK_FD" && -n "$CONTROLLER_LOCK_PATH" ]]; then
    python3 - "$CONTROLLER_LOCK_FD" "$CONTROLLER_LOCK_PATH" "$$" <<'PY' || true
import json
import os
from pathlib import Path
import sys

descriptor, path, controller_pid = int(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
try:
    metadata = os.stat(path, follow_symlinks=False)
    held = os.fstat(descriptor)
    if (metadata.st_dev, metadata.st_ino) != (held.st_dev, held.st_ino):
        raise SystemExit(0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(descriptor), encoding="ascii") as stream:
        value = json.load(stream)
    if value.get("controller_pid") == controller_pid:
        path.unlink()
        parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
except (FileNotFoundError, json.JSONDecodeError, OSError):
    pass
PY
    exec {CONTROLLER_LOCK_FD}>&-
    CONTROLLER_LOCK_FD=""
  fi
}

terminate_active_paid_stage() {
  local attempt discovered_pgid=""
  if [[ -z "$ACTIVE_PAID_STAGE_PGID" && -n "$ACTIVE_PAID_STAGE_PID" ]]; then
    discovered_pgid="$(python3 - "$ACTIVE_PAID_STAGE_PID" <<'PY' || true
import os
import sys

pid = int(sys.argv[1])
try:
    pgid = os.getpgid(pid)
    session = os.getsid(pid)
except ProcessLookupError:
    raise SystemExit(0)
if pgid == pid and session == pid:
    print(pgid)
PY
)"
    if [[ "$discovered_pgid" == "$ACTIVE_PAID_STAGE_PID" ]]; then
      ACTIVE_PAID_STAGE_PGID="$discovered_pgid"
    fi
  fi
  if [[ -n "$ACTIVE_PAID_STAGE_PGID" ]]; then
    kill -TERM -- "-$ACTIVE_PAID_STAGE_PGID" 2>/dev/null || true
    for (( attempt = 1; attempt <= 25; attempt++ )); do
      kill -0 -- "-$ACTIVE_PAID_STAGE_PGID" 2>/dev/null || break
      sleep 0.2
    done
    if kill -0 -- "-$ACTIVE_PAID_STAGE_PGID" 2>/dev/null; then
      kill -KILL -- "-$ACTIVE_PAID_STAGE_PGID" 2>/dev/null || true
    fi
  elif [[ -n "$ACTIVE_PAID_STAGE_PID" ]]; then
    kill -TERM "$ACTIVE_PAID_STAGE_PID" 2>/dev/null || true
  fi
  if [[ -n "$ACTIVE_PAID_STAGE_PID" ]]; then
    wait "$ACTIVE_PAID_STAGE_PID" 2>/dev/null || true
  fi
  if [[ -n "$ACTIVE_PAID_STAGE_PGID" ]] && kill -0 -- "-$ACTIVE_PAID_STAGE_PGID" 2>/dev/null; then
    return 1
  fi
  ACTIVE_PAID_STAGE_PID=""
  ACTIVE_PAID_STAGE_PGID=""
}

controller_cleanup() {
  local status=$?
  local cleanup_failed=0
  trap - EXIT
  trap '' INT TERM
  while ! terminate_active_paid_stage; do
    cleanup_failed=1
    printf 'error: paid-stage process group survived cleanup; retaining controller lock and retrying\n' >&2
    sleep 1
  done
  if [[ "$PROVIDER_CLEANUP_REQUIRED" == 1 ]]; then
    while ! stop_exact_vm; do
      cleanup_failed=1
      printf 'error: exact VM stop confirmation failed; retaining controller lock and retrying\n' >&2
      sleep 1
    done
    PROVIDER_CLEANUP_REQUIRED=0
  fi
  release_controller_lock
  if (( cleanup_failed == 1 && status == 0 )); then status=2; fi
  exit "$status"
}

remote() { ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$SSH_TARGET" "$@"; }
probe_ssh_readiness() {
  python3 - "$SSH_READINESS_CONNECT_TIMEOUT_SECONDS" "$SSH_READINESS_HARD_TIMEOUT_SECONDS" "$SSH_READINESS_REAP_TIMEOUT_SECONDS" "$SSH_TARGET" <<'PY'
import os
import signal
import subprocess
import sys

connect_timeout, hard_timeout, reap_timeout = map(int, sys.argv[1:4])
target = sys.argv[4]
probe = None

def stop_probe_group(force=False):
    if probe is None or probe.poll() is not None:
        return
    try:
        os.killpg(probe.pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass

def interrupted(signum, _frame):
    if probe is None:
        raise SystemExit(128 + signum)
    stop_probe_group()
    try:
        probe.wait(timeout=reap_timeout)
    except subprocess.TimeoutExpired:
        stop_probe_group(force=True)
        try:
            probe.wait(timeout=reap_timeout)
        except subprocess.TimeoutExpired:
            pass
    raise SystemExit(128 + signum)

signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGTERM, interrupted)
probe = subprocess.Popen(
    ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", f"ConnectTimeout={connect_timeout}", target, "true"],
    start_new_session=True,
)
try:
    raise SystemExit(probe.wait(timeout=hard_timeout))
except subprocess.TimeoutExpired:
    stop_probe_group()
    try:
        probe.wait(timeout=reap_timeout)
    except subprocess.TimeoutExpired:
        stop_probe_group(force=True)
        probe.wait(timeout=reap_timeout)
    raise SystemExit(1)
PY
}
wait_for_ssh_readiness() {
  local attempt
  # Eighteen (3s probe + at most 2s group reaping) windows and seventeen 2s
  # intervals bound this gate to 124 seconds even when post-connect SSH hangs.
  for (( attempt = 1; attempt <= SSH_READINESS_ATTEMPTS; attempt++ )); do
    if probe_ssh_readiness; then return 0; fi
    (( attempt == SSH_READINESS_ATTEMPTS )) || sleep "$SSH_READINESS_INTERVAL_SECONDS"
  done
  return 1
}
initialize_deadline() {
  python3 - "$PLAN_RECEIPT" "$DEADLINE_RECEIPT" "$RUN_ID" <<'PY'
import hashlib, json, os, stat, sys, time
from pathlib import Path
plan, output, run_id = map(Path if False else str, sys.argv[1:])
plan_bytes = Path(plan).read_bytes(); digest = hashlib.sha256(plan_bytes).hexdigest()
output_path = Path(output)
if output_path.exists() or output_path.is_symlink():
    metadata = output_path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise SystemExit("paid deadline receipt is not immutable")
    value = json.loads(output_path.read_bytes())
    if value != {"schema_version": 1, "kind": "lehome_public_n15_paid_deadline_v1", "run_id": run_id, "lifecycle_plan_sha256": digest, "started_unix_seconds": value.get("started_unix_seconds"), "deadline_unix_seconds": value.get("deadline_unix_seconds")} or type(value["started_unix_seconds"]) is not int or value["deadline_unix_seconds"] != value["started_unix_seconds"] + 86400:
        raise SystemExit("paid deadline receipt is invalid")
    print(value["deadline_unix_seconds"]); raise SystemExit(0)
started = int(time.time()); value = {"schema_version": 1, "kind": "lehome_public_n15_paid_deadline_v1", "run_id": run_id, "lifecycle_plan_sha256": digest, "started_unix_seconds": started, "deadline_unix_seconds": started + 86400}
with Path(output).open("x", encoding="ascii") as stream: stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
os.chmod(output, 0o444); print(value["deadline_unix_seconds"])
PY
}
initialize_stage_deadline() {
  local label="$1" limit_seconds="$2" aggregate_deadline="$3"
  local stage_receipt="$PIPELINE_ROOT/stage-$label-deadline.json"
  python3 - "$PLAN_RECEIPT" "$stage_receipt" "$RUN_ID" "$label" "$limit_seconds" "$aggregate_deadline" <<'PY'
import hashlib, json, os, stat, sys, time
from pathlib import Path

plan, output = map(Path, sys.argv[1:3])
run_id, stage, limit, aggregate = sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
plan_metadata = plan.lstat()
if not stat.S_ISREG(plan_metadata.st_mode) or stat.S_IMODE(plan_metadata.st_mode) != 0o444:
    raise SystemExit("lifecycle plan is not immutable")
digest = hashlib.sha256(plan.read_bytes()).hexdigest()
expected_keys = {"schema_version", "kind", "run_id", "stage", "lifecycle_plan_sha256", "started_unix_seconds", "deadline_unix_seconds"}
if output.exists() or output.is_symlink():
    metadata = output.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise SystemExit("stage deadline is not immutable")
    value = json.loads(output.read_bytes())
    started, deadline = value.get("started_unix_seconds"), value.get("deadline_unix_seconds")
    if set(value) != expected_keys or (value.get("schema_version"), value.get("kind"), value.get("run_id"), value.get("stage"), value.get("lifecycle_plan_sha256")) != (1, "lehome_public_n15_stage_deadline_v1", run_id, stage, digest) or type(started) is not int or type(deadline) is not int or deadline != min(started + limit, aggregate):
        raise SystemExit("stage deadline is invalid")
    print(deadline)
    raise SystemExit(0)
started = int(time.time()); deadline = min(started + limit, aggregate)
value = {"schema_version": 1, "kind": "lehome_public_n15_stage_deadline_v1", "run_id": run_id, "stage": stage, "lifecycle_plan_sha256": digest, "started_unix_seconds": started, "deadline_unix_seconds": deadline}
with output.open("x", encoding="ascii") as stream:
    stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
os.chmod(output, 0o444)
print(deadline)
PY
}
run_paid_stage() {
  local label="$1" limit_seconds="$2" stage_function="$3"
  local aggregate_deadline now stage_deadline pid status launcher_root launcher_ready launcher_ack launcher_acknowledged ready_pid acknowledged_pid attempt
  case "$stage_function" in train_stage|focused_stage|harvest_stage) ;; *) fail "unknown paid stage dispatcher" ;; esac
  if [[ -n "$PRESTART_ADMITTED_STAGE" && "$label" != "$PRESTART_ADMITTED_STAGE" ]]; then
    fail "$label is not the host-sealed next unfinished stage"
  fi
  aggregate_deadline="$(initialize_deadline)" || fail "aggregate paid deadline is invalid"
  stage_deadline="$(initialize_stage_deadline "$label" "$limit_seconds" "$aggregate_deadline")" || fail "$label deadline receipt is invalid"
  (( stage_deadline <= aggregate_deadline )) || fail "$label deadline exceeds aggregate deadline"
  now="$(date +%s)"
  (( now < stage_deadline )) || fail "$label has no remaining paid time"
  # macOS has no external ``setsid``. This tiny controller-owned Python
  # launcher creates the session before execing an allowlisted Bash dispatcher.
  export -f remote train_stage focused_stage harvest_stage
  export REMOTE_ROOT SSH_TARGET HF_TOKEN_FILE RUNTIME_REVISION SOURCE_ROOT SOURCE_RECEIPT SNAPSHOTS_RECEIPT TRAINING_ROOT EXACT_VM_ID PROTECTED_DISK_ID TRAINING_HF_CACHE TRAINING_PYTHON TRAINING_UV LEROBOT_WHEEL RUNTIME_IMAGE_ID RESUME_PARTIAL RESUME_CHECKPOINT RESUME_STEP RESUME_ATTEMPT_ID ASSETS_ROOT METADATA_ROOT REFERENCE_CHECKPOINT REFERENCE_SANITIZED_CONFIG REFERENCE_COMPATIBILITY NATIVE_RUNTIME_EVIDENCE NATIVE_DEPENDENCIES FOCUSED_HF_CACHE FOCUSED_OUTPUT_ROOT PUBLIC_REPOSITORY ROLLOUT_IMAGE_RECEIPT REMOTE_PIPELINE_ROOT
  launcher_root="$(mktemp -d "$PIPELINE_ROOT/.stage-launcher-${label}.XXXXXX")"
  launcher_ready="$launcher_root/ready"
  launcher_ack="$launcher_root/ack"
  launcher_acknowledged="$launcher_root/acknowledged"
  python3 - "$stage_function" "$launcher_ready" "$launcher_ack" "$launcher_acknowledged" <<'PY' &
import os
from pathlib import Path
import signal
import sys
import time

os.setsid()
ready = Path(sys.argv[2])

# Deterministically exercise the otherwise tiny post-setsid/pre-readiness
# interruption window without letting a paid-stage dispatcher run.
test_window = os.environ.get("LEHOME_N15_TEST_SETSID_WINDOW")
if (
    test_window
    and os.environ.get("PYTEST_CURRENT_TEST", "").startswith(
        "tests/infrastructure/test_public_n15_pipeline_remote.py::"
    )
):
    child = os.fork()
    if child == 0:
        stopped = Path(os.environ["DESCENDANT_STOPPED"])
        pid_path = Path(os.environ["DESCENDANT_PID"])

        def finish(_signum, _frame):
            stopped.write_text("stopped\n", encoding="ascii")
            os._exit(0)

        signal.signal(signal.SIGTERM, finish)
        pid_path.write_text(f"{os.getpid()}\n", encoding="ascii")
        while True:
            signal.pause()
    Path(test_window).write_text("post-setsid\n", encoding="ascii")
    while True:
        signal.pause()

descriptor = os.open(ready, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="ascii") as stream:
    stream.write(f"{os.getpid()}\n")
    stream.flush()
    os.fsync(stream.fileno())
ack = Path(sys.argv[3])
acknowledged = Path(sys.argv[4])
deadline = time.monotonic() + 5
while time.monotonic() < deadline:
    try:
        acknowledgement = ack.read_text(encoding="ascii")
    except FileNotFoundError:
        acknowledgement = ""
    if acknowledgement == f"{os.getpid()}\n":
        descriptor = os.open(
            acknowledged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(f"{os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        break
    if acknowledgement:
        raise SystemExit(74)
    time.sleep(0.01)
else:
    raise SystemExit(75)
os.execvpe(
    "bash",
    ["bash", "-c", 'case "$1" in train_stage) train_stage ;; focused_stage) focused_stage ;; harvest_stage) harvest_stage ;; *) exit 64 ;; esac', "bash", sys.argv[1]],
    os.environ,
)
PY
  pid=$!
  ACTIVE_PAID_STAGE_PID="$pid"
  ACTIVE_PAID_STAGE_PGID=""
  ready_pid=""
  for (( attempt = 1; attempt <= 200; attempt++ )); do
    if [[ -f "$launcher_ready" && ! -L "$launcher_ready" ]]; then
      # The launcher creates its session before publishing this handshake.
      # Track the group before parsing so even a malformed handshake cannot
      # leave detached descendants behind.
      ACTIVE_PAID_STAGE_PGID="$pid"
      IFS= read -r ready_pid < "$launcher_ready"
      [[ "$ready_pid" == "$pid" ]] || { terminate_active_paid_stage || true; rm -rf -- "$launcher_root"; fail "$label launcher handshake identity mismatch"; }
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true; ACTIVE_PAID_STAGE_PID=""; rm -rf -- "$launcher_root"; fail "$label launcher exited before setsid handshake"
    fi
    now="$(date +%s)"
    if (( now >= stage_deadline )); then
      terminate_active_paid_stage || true; rm -rf -- "$launcher_root"; fail "$label launcher handshake exceeded its paid timeout"
    fi
    sleep 0.02
  done
  [[ "$ready_pid" == "$pid" ]] || { terminate_active_paid_stage || true; rm -rf -- "$launcher_root"; fail "$label launcher setsid handshake failed"; }
  ( umask 077; set -o noclobber; printf '%s\n' "$pid" > "$launcher_ack" ) \
    || { terminate_active_paid_stage || true; rm -rf -- "$launcher_root"; fail "$label launcher acknowledgement failed"; }
  acknowledged_pid=""
  for (( attempt = 1; attempt <= 200; attempt++ )); do
    if [[ -f "$launcher_acknowledged" && ! -L "$launcher_acknowledged" ]]; then
      IFS= read -r acknowledged_pid < "$launcher_acknowledged"
      [[ "$acknowledged_pid" == "$pid" ]] || { terminate_active_paid_stage || true; rm -rf -- "$launcher_root"; fail "$label launcher acknowledgement identity mismatch"; }
      break
    fi
    if ! kill -0 -- "-$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true; ACTIVE_PAID_STAGE_PID=""; ACTIVE_PAID_STAGE_PGID=""; rm -rf -- "$launcher_root"; fail "$label launcher exited before controller acknowledgement"
    fi
    now="$(date +%s)"
    if (( now >= stage_deadline )); then
      terminate_active_paid_stage || true; rm -rf -- "$launcher_root"; fail "$label launcher acknowledgement exceeded its paid timeout"
    fi
    sleep 0.02
  done
  [[ "$acknowledged_pid" == "$pid" ]] || { terminate_active_paid_stage || true; rm -rf -- "$launcher_root"; fail "$label launcher acknowledgement handshake failed"; }
  rm -rf -- "$launcher_root"
  while kill -0 -- "-$pid" 2>/dev/null; do
    now="$(date +%s)"
    if (( now >= stage_deadline )); then
      terminate_active_paid_stage || fail "$label process group survived watchdog termination"
      fail "$label exceeded its code-owned paid timeout"
    fi
    sleep 1
  done
  if wait "$pid"; then status=0; else status=$?; fi
  ACTIVE_PAID_STAGE_PID=""
  ACTIVE_PAID_STAGE_PGID=""
  (( status == 0 )) || fail "$label failed"
  if [[ "$label" == "$PRESTART_ADMITTED_STAGE" ]]; then
    PRESTART_ADMITTED_STAGE=""
  fi
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
  remote bash -s -- "$TRAINING_PUBLICATION_RECEIPT" "$PUBLIC_REPOSITORY" "n15-public/$RUN_ID/training" "$TRAINING_ROOT" <<'SH'
set -euo pipefail
python3 - "$1" "$2" "$3" "$4" <<'PY'
import hashlib, json, re, stat, sys
from pathlib import Path, PurePosixPath
from huggingface_hub import HfApi, hf_hub_download

receipt, repository, prefix, training = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
receipt_metadata = receipt.lstat()
training_metadata = training.lstat()
if (
    receipt.is_symlink() or not stat.S_ISREG(receipt_metadata.st_mode)
    or stat.S_IMODE(receipt_metadata.st_mode) != 0o444
    or training.is_symlink() or not stat.S_ISDIR(training_metadata.st_mode)
):
    raise SystemExit("training publication receipt or root is unsafe")
raw = receipt.read_bytes(); value = json.loads(raw)
if raw != (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"):
    raise SystemExit("training publication is not canonical")
if (
    set(value) != {"schema_version", "kind", "repository", "remote_prefix", "immutable_revision", "entries", "anonymous_byte_readback_verified"}
    or value["schema_version"] != 1
    or value["kind"] != "lehome_public_n15_training_publication_v1"
    or value["repository"] != repository
    or value["remote_prefix"] != prefix
    or re.fullmatch(r"[0-9a-f]{40}", str(value["immutable_revision"])) is None
    or value["anonymous_byte_readback_verified"] is not True
    or not isinstance(value["entries"], list) or not value["entries"]
):
    raise SystemExit("training publication receipt is invalid")
expected = {}
for item in value["entries"]:
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise SystemExit("training publication entry schema is invalid")
    relative, digest = item["path"], item["sha256"]
    pure = PurePosixPath(str(relative))
    if (
        not isinstance(relative, str) or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or relative in expected or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
        or relative == "training-publication.json"
    ):
        raise SystemExit("training publication entry is invalid")
    expected[relative] = digest
current = {}
for path in sorted(training.rglob("*")):
    relative = path.relative_to(training).as_posix()
    metadata = path.lstat()
    if path.is_symlink() or stat.S_ISDIR(metadata.st_mode):
        continue
    if not stat.S_ISREG(metadata.st_mode) or path == receipt or relative.startswith(".training-publication.json."):
        if path != receipt:
            raise SystemExit("training publication tree contains an unsafe entry")
        continue
    current[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
if current != expected or expected.get("training-identity.json") != hashlib.sha256((training / "training-identity.json").read_bytes()).hexdigest():
    raise SystemExit("training publication does not bind the current verified training tree")
revision = value["immutable_revision"]
tree = HfApi(token=False).list_repo_tree(
    repo_id=repository, repo_type="model", path_in_repo=prefix,
    revision=revision, recursive=True, token=False,
)
remote_paths = {
    item.rfilename[len(prefix) + 1:]
    for item in tree
    if getattr(item, "size", None) is not None and item.rfilename.startswith(prefix + "/")
}
if remote_paths != set(expected):
    raise SystemExit("immutable publication revision tree mismatch")
for relative, digest in expected.items():
    fetched = hf_hub_download(
        repo_id=repository, repo_type="model", filename=prefix + "/" + relative,
        revision=revision, token=False,
    )
    if hashlib.sha256(Path(fetched).read_bytes()).hexdigest() != digest:
        raise SystemExit("immutable publication revision byte mismatch")
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
  local remote_path="$1" local_path="$2" temporary
  temporary="$(mktemp "$(dirname -- "$local_path")/.${local_path##*/}.XXXXXX")"
  if ! remote bash -s -- "$remote_path" <<'SH' >"$temporary"
set -euo pipefail
test -f "$1" && test ! -L "$1"
cat -- "$1"
SH
  then
    rm -f -- "$temporary"
    return 1
  fi
  if [[ ! -s "$temporary" || -L "$temporary" ]]; then
    rm -f -- "$temporary"
    return 1
  fi
  chmod 0444 "$temporary"
  if [[ -e "$local_path" || -L "$local_path" ]]; then
    if ! python3 - "$temporary" "$local_path" <<'PY'
import hashlib, os, stat, sys
from pathlib import Path

temporary, destination = map(Path, sys.argv[1:])
try:
    metadata = destination.lstat()
    remote_payload = temporary.read_bytes()
    local_payload = destination.read_bytes()
except OSError:
    raise SystemExit(73)
if (
    destination.is_symlink()
    or not stat.S_ISREG(metadata.st_mode)
    or stat.S_IMODE(metadata.st_mode) != 0o444
    or metadata.st_size == 0
    or local_payload != remote_payload
    or hashlib.sha256(local_payload).digest() != hashlib.sha256(remote_payload).digest()
):
    raise SystemExit(73)
PY
    then
      rm -f -- "$temporary"
      return 1
    fi
    rm -f -- "$temporary"
    return 0
  fi
  if ! python3 - "$temporary" "$local_path" <<'PY'
import os, sys
temporary, destination = sys.argv[1:]
descriptor = os.open(temporary, os.O_RDONLY | os.O_NOFOLLOW)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
os.link(temporary, destination)
parent = os.open(os.path.dirname(destination), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(parent)
finally:
    os.close(parent)
PY
  then
    rm -f -- "$temporary"
    return 1
  fi
  rm -f -- "$temporary"
}

host_next_unfinished_stage() {
  python3 - "$HOST_TRAINING_STAGE_RECEIPT" "$HOST_FOCUSED_STAGE_RECEIPT" "$RUN_ID" \
    "$TRAINING_IDENTITY_RECEIPT" "$TRAINING_PUBLICATION_RECEIPT" \
    "$FOCUSED_OUTPUT_ROOT/comparison-receipt.json" "$FOCUSED_OUTPUT_ROOT/publication.json" "$FOCUSED_PROMOTION_RECEIPT" <<'PY'
import json, re, stat, sys
from pathlib import Path

training, focused = map(Path, sys.argv[1:3])
run_id = sys.argv[3]
expected = {
    "training": sys.argv[4:6],
    "focused": sys.argv[6:9],
}

def validate(path, stage):
    if not path.exists() and not path.is_symlink():
        return False
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise SystemExit(f"host {stage} completion seal is unsafe")
    raw = path.read_bytes()
    value = json.loads(raw)
    canonical = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    receipts = value.get("remote_receipts") if isinstance(value, dict) else None
    if (
        raw != canonical
        or set(value) != {"schema_version", "kind", "run_id", "stage", "remote_receipts"}
        or (value.get("schema_version"), value.get("kind"), value.get("run_id"), value.get("stage"))
        != (1, "lehome_public_n15_host_stage_completion_v1", run_id, stage)
        or not isinstance(receipts, list)
        or [item.get("path") for item in receipts if isinstance(item, dict)] != expected[stage]
        or any(
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256"))) is None
            for item in receipts
        )
    ):
        raise SystemExit(f"host {stage} completion seal is invalid")
    return True

training_complete = validate(training, "training")
focused_complete = validate(focused, "focused")
if focused_complete and not training_complete:
    raise SystemExit("host focused completion seal lacks training ancestry")
print("harvest" if focused_complete else "focused_gate" if training_complete else "train")
PY
}

record_host_stage_completion() {
  local stage="$1" output="$2" temporary_root index remote_path
  shift 2
  temporary_root="$(mktemp -d "$PIPELINE_ROOT/.host-stage-${stage}.XXXXXX")"
  trap 'rm -rf -- "$temporary_root"' RETURN
  index=0
  for remote_path in "$@"; do
    fetch_remote_immutable "$remote_path" "$temporary_root/$index.receipt"
    index=$((index + 1))
  done
  python3 - "$output" "$RUN_ID" "$stage" "$temporary_root" "$@" <<'PY'
import hashlib, json, os, stat, sys, tempfile
from pathlib import Path

output, run_id, stage, temporary_root = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
remote_paths = sys.argv[5:]
receipts = []
for index, remote_path in enumerate(remote_paths):
    path = temporary_root / f"{index}.receipt"
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_size == 0:
        raise SystemExit("host stage source receipt is unsafe")
    receipts.append({"path": remote_path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
value = {
    "schema_version": 1,
    "kind": "lehome_public_n15_host_stage_completion_v1",
    "run_id": run_id,
    "stage": stage,
    "remote_receipts": receipts,
}
payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
if output.exists() or output.is_symlink():
    metadata = output.lstat()
    if output.is_symlink() or not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o444 or output.read_bytes() != payload:
        raise SystemExit("existing host stage completion seal mismatch")
    raise SystemExit(0)
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.link(temporary, output)
    parent = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
finally:
    temporary.unlink(missing_ok=True)
PY
  rm -rf -- "$temporary_root"
  trap - RETURN
}

reconcile_remote_stage_seals() {
  local training_identity=0 training_publication=0 focused=0
  remote_file_exists "$TRAINING_IDENTITY_RECEIPT" && training_identity=1
  remote_file_exists "$TRAINING_PUBLICATION_RECEIPT" && training_publication=1
  (( training_identity == training_publication )) \
    || fail "remote training completion receipts are incomplete"
  if (( training_identity == 1 )); then
    verify_remote_training_chain || fail "remote training completion is invalid"
    verify_remote_training_publication || fail "remote training publication is invalid"
    record_host_stage_completion training "$HOST_TRAINING_STAGE_RECEIPT" \
      "$TRAINING_IDENTITY_RECEIPT" "$TRAINING_PUBLICATION_RECEIPT"
  elif [[ -e "$HOST_TRAINING_STAGE_RECEIPT" || -L "$HOST_TRAINING_STAGE_RECEIPT" ]]; then
    fail "host training completion seal has no matching remote completion"
  fi

  remote_file_exists "$FOCUSED_PROMOTION_RECEIPT" && focused=1
  if (( focused == 1 )); then
    (( training_identity == 1 )) \
      || fail "remote focused completion lacks training ancestry"
    verify_remote_focused_chain || fail "remote focused completion is invalid"
    record_host_stage_completion focused "$HOST_FOCUSED_STAGE_RECEIPT" \
      "$FOCUSED_OUTPUT_ROOT/comparison-receipt.json" \
      "$FOCUSED_OUTPUT_ROOT/publication.json" "$FOCUSED_PROMOTION_RECEIPT"
  elif [[ -e "$HOST_FOCUSED_STAGE_RECEIPT" || -L "$HOST_FOCUSED_STAGE_RECEIPT" ]]; then
    fail "host focused completion seal has no matching remote completion"
  fi
}

finalize_host_harvest_terminal() {
  python3 "$HARVEST_BUILDER" verify-terminal --manifest "$HARVEST_MANIFEST" --manifest-receipt "$HARVEST_MANIFEST_RECEIPT" --publication-receipt "$HARVEST_PUBLICATION_RECEIPT" --provider-receipt "$PROVIDER_STOPPED_RECEIPT" --output "$HARVEST_TERMINAL_RECEIPT" >/dev/null
}

validate_remote_runtime() {
  remote bash -s -- "$REMOTE_ROOT" "$REMOTE_PIPELINE_ROOT" "$SOURCE_ROOT" "$SOURCE_RECEIPT" "$SNAPSHOTS_RECEIPT" "$TRAINING_ROOT" "$RUNTIME_REVISION" "$EXACT_VM_ID" "$PROTECTED_DISK_ID" <<'SH'
set -euo pipefail
root="$1"; pipeline_root="$2"; source_root="$3"; source_receipt="$4"; snapshots="$5"; training_root="$6"; revision="$7"; vm_id="$8"; disk_id="$9"
test -f /var/lib/cloud/instance/boot-finished
workspace_base="$(dirname -- "$pipeline_root")"
[[ "$workspace_base" == /mnt/lehome/public-n15-runs ]]
test -d "$workspace_base" && test ! -L "$workspace_base"
mount_source="$(findmnt -T "$workspace_base" --noheadings --output SOURCE)"
[[ "$mount_source" == /dev/* ]] && lsblk -n -o TYPE "$mount_source" | grep -Eq '^(disk|part|lvm|crypt)$'
# Cloud-init attaches the exact Nebius secondary disk with device_id=lehome.
# Prove that its stable guest device backs this run's workspace mount.
test -e /dev/disk/by-id/virtio-lehome
[[ "$(lsblk -ndo MAJ:MIN /dev/disk/by-id/virtio-lehome)" == "$(findmnt -T "$workspace_base" --noheadings --output MAJ:MIN)" ]]
if [[ ! -e "$pipeline_root" ]]; then mkdir -m 0700 -- "$pipeline_root"; fi
test -d "$pipeline_root" && test ! -L "$pipeline_root"
nvidia-smi -L | grep -q .
test "$(git -C "$root" rev-parse HEAD)" = "$revision"
test -z "$(git -C "$root" status --porcelain --untracked-files=all)"
verified_inputs="$(dirname -- "$training_root")/verified-inputs.json"
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

wait_for_remote_runtime() {
  local attempt
  # SSH can accept the controller before cloud-init creates boot-finished.
  # Re-run the whole read-only runtime gate within the existing boot-readiness
  # budget rather than stopping a guest during that short race.
  for (( attempt = 1; attempt <= SSH_READINESS_ATTEMPTS; attempt++ )); do
    if validate_remote_runtime; then return 0; fi
    (( attempt == SSH_READINESS_ATTEMPTS )) || sleep "$SSH_READINESS_INTERVAL_SECONDS"
  done
  return 1
}

train_stage() {
  remote bash -s -- "$REMOTE_ROOT" "$SOURCE_ROOT" "$SOURCE_RECEIPT" "$SNAPSHOTS_RECEIPT" "$TRAINING_ROOT" "$EXACT_VM_ID" "$PROTECTED_DISK_ID" "$TRAINING_HF_CACHE" "$TRAINING_PYTHON" "$TRAINING_UV" "$LEROBOT_WHEEL" "$RUNTIME_IMAGE_ID" "$RESUME_PARTIAL" "$RESUME_CHECKPOINT" "$RESUME_STEP" "$RESUME_ATTEMPT_ID" <<'SH'
set -euo pipefail
root="$1"; source_root="$2"; source_receipt="$3"; snapshots="$4"; training_root="$5"; vm_id="$6"; disk_id="$7"; hf_cache="$8"; python_bin="$9"; uv_bin="${10}"; wheel="${11}"; runtime_image_id="${12}"; resume_partial="${13}"; resume_checkpoint="${14}"; resume_step="${15}"; resume_attempt_id="${16}"
upstream_output="$source_root/outputs/train/groot_four_types_merged_batch64_lr2e-4"
staging_root="${training_root}.evidence-staging"
resume_name=""
resume_log="$staging_root/logs/train.log"
resume_scratch_root=""
cleanup_resume_scratch() {
  local status=$?
  if [[ -n "$resume_scratch_root" && -d "$staging_root" && ! -L "$staging_root" ]]; then
    python3 "$root/scripts/run_public_n15_reproduction.py" cleanup-resume-scratch \
      --staging-root "$staging_root" --attempt-id "$resume_attempt_id" >/dev/null || true
  fi
  return "$status"
}
trap cleanup_resume_scratch EXIT
trap 'exit 130' INT TERM
if [[ -d "${training_root}.finalizing" || -f "$staging_root/training-finalization.json" || -f "${training_root}.finalizing/evidence/training-finalization.json" ]]; then
  python3 "$root/scripts/run_public_n15_reproduction.py" finalize-training-output \
    --checkout "$source_root" --source-receipt "$source_receipt" \
    --resolved-snapshots-receipt "$snapshots" --vm-id "$vm_id" --disk-id "$disk_id" \
    --training-root "$training_root" --staging-root "$staging_root" \
    --upstream-output "$upstream_output" >/dev/null
  exit 0
fi
if [[ -d "$upstream_output" && ! -L "$upstream_output" && -d "$staging_root" && ! -L "$staging_root" ]]; then
  completed_state="$(python3 "$root/scripts/run_public_n15_reproduction.py" verify-completed-upstream \
    --checkout "$source_root" --source-receipt "$source_receipt" \
    --resolved-snapshots-receipt "$snapshots" --vm-id "$vm_id" --disk-id "$disk_id" \
    --training-root "$training_root" --staging-root "$staging_root" \
    --upstream-output "$upstream_output")"
  if [[ "$completed_state" == '{"complete":true}' ]]; then
    recovery_eagle_home="$staging_root/eagle-home"
    if [[ -e "$recovery_eagle_home" || -L "$recovery_eagle_home" ]]; then
      test -d "$recovery_eagle_home" && test ! -L "$recovery_eagle_home"
      find "$recovery_eagle_home" -depth -type f -delete
      find "$recovery_eagle_home" -depth -type d -empty -delete
      test ! -e "$recovery_eagle_home"
    fi
    python3 "$root/scripts/run_public_n15_reproduction.py" finalize-training-output \
      --checkout "$source_root" --source-receipt "$source_receipt" \
      --resolved-snapshots-receipt "$snapshots" --vm-id "$vm_id" --disk-id "$disk_id" \
      --training-root "$training_root" --staging-root "$staging_root" \
      --upstream-output "$upstream_output" >/dev/null
    exit 0
  fi
fi
if [[ "$resume_partial" == 1 ]]; then
  [[ "$resume_step" =~ ^[0-9]+$ ]] || { echo "explicit resume step is invalid" >&2; exit 2; }
  [[ "$resume_attempt_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$ ]] || { echo "explicit resume attempt identity is invalid" >&2; exit 2; }
  printf -v resume_name '%06d' "$resume_step"
  [[ "$resume_checkpoint" == "$upstream_output/checkpoints/$resume_name" ]] || { echo "explicit resume checkpoint path is not the configured boundary" >&2; exit 2; }
  test ! -e "$training_root" && test ! -L "$training_root"
  test -d "$upstream_output" && test ! -L "$upstream_output"
  test -d "$staging_root" && test ! -L "$staging_root"
  protected_device="$(findmnt -T "$(dirname -- "$training_root")" --noheadings --output MAJ:MIN)"
  [[ "$(findmnt -T "$staging_root" --noheadings --output MAJ:MIN)" == "$protected_device" ]]
  [[ "$(findmnt -T "$upstream_output" --noheadings --output MAJ:MIN)" == "$protected_device" ]]
  ! pgrep -f '/opt/lehome-challenge/.venv/bin/lerobot-train([[:space:]]|$)' >/dev/null
  resume_log="$staging_root/logs/train-resume-${resume_attempt_id}.log"
  test ! -e "$resume_log" && test ! -L "$resume_log"
  mkdir -m 0700 -p "$staging_root/evidence/resume-attempts"
  python3 "$root/scripts/run_public_n15_reproduction.py" verify-resume-checkpoint \
    --checkout "$source_root" --source-receipt "$source_receipt" \
    --resolved-snapshots-receipt "$snapshots" --vm-id "$vm_id" --disk-id "$disk_id" \
    --training-root "$training_root" --staging-root "$staging_root" \
    --upstream-output "$upstream_output" --resume-step "$resume_step" \
    --attempt-id "$resume_attempt_id" \
    --output "$staging_root/evidence/resume-attempts/${resume_attempt_id}.json" >/dev/null
  printf 'resume attempt %s admitted at checkpoint %s\n' "$resume_attempt_id" "$resume_name" >"$resume_log"
  python3 "$root/scripts/run_public_n15_reproduction.py" prepare-resume-scratch \
    --staging-root "$staging_root" --attempt-id "$resume_attempt_id" >/dev/null
  resume_scratch_root="$staging_root/.resume-scratch-${resume_attempt_id}"
else
  test ! -e "$training_root" && test ! -L "$training_root"
  test ! -e "$upstream_output" && test ! -L "$upstream_output"
  test ! -e "$staging_root" && test ! -L "$staging_root"
  mkdir -m 0700 -p "$staging_root/evidence/upstream" "$staging_root/evidence/compatibility" "$staging_root/logs"
  mkdir -p "$(dirname -- "$upstream_output")"
  install -m 0444 "$source_receipt" "$staging_root/evidence/source-receipt.json"
  install -m 0444 "$snapshots" "$staging_root/evidence/resolved-snapshots-receipt.json"
  install -m 0444 "$source_root/uv.lock" "$staging_root/evidence/uv.lock"
fi
test -f "$wheel" && test ! -L "$wheel" && test -d "$hf_cache" && test ! -L "$hf_cache"
test -x "$uv_bin" && test ! -L "$uv_bin"
test -x "$python_bin"
# Keep installer scratch and package copies on the protected disk. The VM root
# volume is intentionally small; FlashAttention is extracted into container
# tmpfs, rather than mutating the persistent venv with a cache materialization
# that proved corrupt on reboot.
export UV_CACHE_DIR="$(dirname -- "$python_bin")/.uv-cache"
export TMPDIR="$(dirname -- "$python_bin")/.uv-tmp"
export UV_LINK_MODE=copy
mkdir -m 0700 -p "$UV_CACHE_DIR" "$TMPDIR"
if [[ "$resume_partial" == 1 ]]; then
  cmp -s "$wheel" "$staging_root/evidence/upstream/lerobot-0.4.3-py3-none-any.whl"
  comparison_root="$resume_scratch_root/compatibility"
  mkdir -m 0700 -- "$comparison_root"
  python3 "$root/scripts/run_public_n15_reproduction.py" build-compatible-wheel \
    --upstream-wheel "$staging_root/evidence/upstream/lerobot-0.4.3-py3-none-any.whl" \
    --wheel-output "$comparison_root/lerobot-0.4.3-py3-none-any.whl" \
    --receipt-output "$comparison_root/lerobot-compatibility-receipt.json" >/dev/null
  cmp -s "$comparison_root/lerobot-0.4.3-py3-none-any.whl" "$staging_root/evidence/compatibility/lerobot-0.4.3-py3-none-any.whl"
  temporary_receipt="$comparison_root/lerobot-compatibility-receipt.json"
  immutable_receipt="$staging_root/evidence/compatibility/lerobot-compatibility-receipt.json"
  cmp -s "$temporary_receipt" "$immutable_receipt"
else
  install -m 0444 "$wheel" "$staging_root/evidence/upstream/lerobot-0.4.3-py3-none-any.whl"
  python3 "$root/scripts/run_public_n15_reproduction.py" build-compatible-wheel \
    --upstream-wheel "$staging_root/evidence/upstream/lerobot-0.4.3-py3-none-any.whl" \
    --wheel-output "$staging_root/evidence/compatibility/lerobot-0.4.3-py3-none-any.whl" \
    --receipt-output "$staging_root/evidence/compatibility/lerobot-compatibility-receipt.json" >/dev/null
fi
python3 "$root/scripts/run_public_n15_reproduction.py" verify-compatible-wheel \
  --upstream-wheel "$staging_root/evidence/upstream/lerobot-0.4.3-py3-none-any.whl" \
  --wheel "$staging_root/evidence/compatibility/lerobot-0.4.3-py3-none-any.whl" \
  --receipt "$staging_root/evidence/compatibility/lerobot-compatibility-receipt.json" >/dev/null
"$uv_bin" pip install --offline --no-deps --reinstall --python "$python_bin" \
  "$staging_root/evidence/compatibility/lerobot-0.4.3-py3-none-any.whl" >/dev/null
test -x "$(dirname -- "$python_bin")/lerobot-train"
"$python_bin" -I -c 'import lerobot; from pathlib import Path; assert Path(lerobot.__file__).is_file()'
runtime_comparison_root=""
runtime_inspect="$staging_root/evidence/runtime-image-inspect.json"
runtime_receipt="$staging_root/evidence/runtime-image-receipt.json"
if [[ "$resume_partial" == 1 ]]; then
  runtime_comparison_root="$resume_scratch_root/runtime-image"
  mkdir -m 0700 -- "$runtime_comparison_root"
  runtime_inspect="$runtime_comparison_root/runtime-image-inspect.json"
  runtime_receipt="$runtime_comparison_root/runtime-image-receipt.json"
fi
sudo -n docker image inspect -- "$runtime_image_id" >"$runtime_inspect"
"$python_bin" - "$runtime_inspect" "$runtime_receipt" "$runtime_image_id" <<'PY'
import json, os, sys
from pathlib import Path

source, target, expected = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rows = json.loads(source.read_text(encoding="utf-8"))
if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("Id") != expected:
    raise SystemExit("approved N1.5 training image identity mismatch")
payload = {"schema_version": 1, "kind": "lehome_public_n15_training_runtime_image_v1", "image_id": expected}
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "w", encoding="ascii") as stream:
    json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
source.unlink()
PY
if [[ "$resume_partial" == 1 ]]; then
  temporary_receipt="$runtime_receipt"
  immutable_receipt="$staging_root/evidence/runtime-image-receipt.json"
  cmp -s "$temporary_receipt" "$immutable_receipt"
fi
overlay_comparison_root=""
peft_receipt="$staging_root/evidence/peft-overlay-receipt.json"
flash_overlay_receipt="$staging_root/evidence/flash-attention-overlay-receipt.json"
flash_runtime_receipt="$staging_root/evidence/flash-attention-runtime-receipt.json"
container_runtime_receipt="$staging_root/evidence/training-container-runtime-receipt.json"
if [[ "$resume_partial" == 1 ]]; then
  overlay_comparison_root="$resume_scratch_root/overlays"
  mkdir -m 0700 -- "$overlay_comparison_root"
  peft_receipt="$overlay_comparison_root/peft-overlay-receipt.json"
  flash_overlay_receipt="$overlay_comparison_root/flash-attention-overlay-receipt.json"
  flash_runtime_receipt="$overlay_comparison_root/flash-attention-runtime-receipt.json"
  container_runtime_receipt="$overlay_comparison_root/training-container-runtime-receipt.json"
fi
sudo -n docker run --rm -i --pull never --gpus all --network none \
  --tmpfs "/flash:rw,exec,size=2g,mode=700,uid=$(id -u),gid=$(id -g)" \
  --mount "type=bind,src=$root,dst=$root,readonly" \
  --mount "type=bind,src=$staging_root,dst=$staging_root" \
  --mount "type=bind,src=$staging_root/evidence/compatibility/lerobot-0.4.3-py3-none-any.whl,dst=/runtime/lerobot-0.4.3-py3-none-any.whl,readonly" \
  --mount "type=bind,src=/mnt/lehome/reference-native/dependencies,dst=/deps,readonly" \
  --mount "type=bind,src=/mnt/lehome/reference-native/dependencies,dst=/mnt/lehome/reference-native/dependencies,readonly" \
  --entrypoint bash "$runtime_image_id" -s -- "$root" "$peft_receipt" "$flash_overlay_receipt" "$flash_runtime_receipt" "$container_runtime_receipt" "$runtime_image_id" <<'CONTAINER'
set -euo pipefail
python_bin=/opt/lehome-challenge/.venv/bin/python
pythonpath=/flash/site-packages:/deps/peft-0.18.1-py3-none-any.whl
root="$1"
peft_wheel="/mnt/lehome/reference-native/dependencies/peft-0.18.1-py3-none-any.whl"
PYTHONSAFEPATH=1 PYTHONPATH="$peft_wheel" "$python_bin" "$root/scripts/verify_native_reference_evaluator_gate.py" \
  prepare-peft-overlay --receipt "$2" >/dev/null
flash_wheel="/mnt/lehome/reference-native/dependencies/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
dm_tree_wheel="/mnt/lehome/reference-native/dependencies/dm_tree-0.1.9-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
PYTHONSAFEPATH=1 "$python_bin" "$root/scripts/verify_native_reference_evaluator_gate.py" \
  prepare-flash-attention-overlay --receipt "$3" >/dev/null
mkdir -m 0700 /flash/site-packages
"$python_bin" - "$flash_wheel" "$dm_tree_wheel" /flash/site-packages <<'PY'
import hashlib, os, sys, zipfile
from pathlib import Path

flash_wheel, dm_tree_wheel, target = map(Path, sys.argv[1:])
if hashlib.sha256(dm_tree_wheel.read_bytes()).hexdigest() != "294dc1cecf87552a45cdd5ddb215e7f5295a5a47c46f1f0a0463c3dd02a527d7":
    raise SystemExit("dm-tree wheel identity mismatch")
lerobot_wheel = Path("/runtime/lerobot-0.4.3-py3-none-any.whl")
with zipfile.ZipFile(lerobot_wheel) as archive:
    archive.extractall(target)
with zipfile.ZipFile(flash_wheel) as archive:
    for item in archive.infolist():
        top_level = item.filename.partition("/")[0]
        if item.filename.startswith(("flash_attn/", "flash_attn_2_cuda")) or (
            top_level.startswith("flash_attn-") and top_level.endswith(".dist-info")
        ):
            archive.extract(item, target)
with zipfile.ZipFile(dm_tree_wheel) as archive:
    archive.extractall(target)
PY
PYTHONPATH="$pythonpath" "$python_bin" -c 'import lerobot.scripts.lerobot_train'
PYTHONPATH="$pythonpath" "$python_bin" - "$4" "$5" "$6" <<'PY'
import importlib.metadata, importlib.util, json, os, sys
from pathlib import Path

import flash_attn, torch, tree
from flash_attn import flash_attn_func
from lerobot.policies.groot import groot_n1

if importlib.metadata.version("flash_attn") != "2.8.3":
    raise SystemExit("FlashAttention package metadata identity is invalid")
if importlib.metadata.version("dm-tree") != "0.1.9":
    raise SystemExit("dm-tree package metadata identity is invalid")
if torch.__version__ != "2.7.0+cu128" or torch.version.cuda != "12.8":
    raise SystemExit("FlashAttention requires torch 2.7.0+cu128 with CUDA 12.8")
if bool(torch._C._GLIBCXX_USE_CXX11_ABI) is not True:
    raise SystemExit("FlashAttention requires CXX11 ABI true")
if not torch.cuda.is_available() or list(torch.cuda.get_device_capability(0)) != [12, 0]:
    raise SystemExit("FlashAttention requires CUDA capability [12, 0]")
origin = str(Path(flash_attn.__file__).resolve())
expected = "/flash/site-packages/flash_attn/__init__.py"
if origin != expected:
    raise SystemExit("FlashAttention installed runtime identity is invalid")
tree_origin = str(Path(tree.__file__).resolve())
expected_tree = "/flash/site-packages/tree/__init__.py"
if tree_origin != expected_tree:
    raise SystemExit("dm-tree installed runtime identity is invalid")
if tree.map_structure(lambda left, right: left + right, {"joint": 1}, {"joint": 2}) != {"joint": 3}:
    raise SystemExit("dm-tree map_structure runtime probe failed")
if groot_n1.tree is not tree:
    raise SystemExit("LeRobot GR00T did not import the verified dm-tree module")
query = torch.randn((1, 2, 4, 64), dtype=torch.float16, device="cuda")
output = flash_attn_func(query, query, query, causal=False)
torch.cuda.synchronize()
if not bool(torch.isfinite(output).all().item()):
    raise SystemExit("FlashAttention CUDA kernel returned non-finite values")
payload = {
    "schema_version": 1,
    "kind": "lehome_native_reference_flash_attention_runtime_v1",
    "torch_version": str(torch.__version__),
    "torch_cuda_version": torch.version.cuda,
    "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    "cuda_capability": list(torch.cuda.get_device_capability(0)),
    "flash_attn_version": "2.8.3",
    "flash_attn_origin": origin,
    "kernel": {"shape": [1, 2, 4, 64], "dtype": "float16", "finite": True},
}
target = Path(sys.argv[1])
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
lerobot = importlib.util.find_spec("lerobot")
peft = importlib.util.find_spec("peft")
container_payload = {
    "schema_version": 1,
    "kind": "lehome_public_n15_training_container_runtime_v1",
    "image_id": sys.argv[3],
    "python_executable": sys.executable,
    "python_version": list(sys.version_info[:3]),
    "pythonpath": os.environ.get("PYTHONPATH"),
    "lerobot_origin": None if lerobot is None else lerobot.origin,
    "peft_origin": None if peft is None else peft.origin,
    "flash_attn_origin": origin,
    "torch_version": str(torch.__version__),
    "torch_cuda_version": torch.version.cuda,
    "cuda_capability": list(torch.cuda.get_device_capability(0)),
}
expected_container = {
    "image_id": "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7",
    "python_executable": "/opt/lehome-challenge/.venv/bin/python",
    "pythonpath": "/flash/site-packages:/deps/peft-0.18.1-py3-none-any.whl",
    "lerobot_origin": "/flash/site-packages/lerobot/__init__.py",
    "peft_origin": "/deps/peft-0.18.1-py3-none-any.whl/peft/__init__.py",
}
if any(container_payload.get(key) != value for key, value in expected_container.items()):
    raise SystemExit("N1.5 container training runtime identity mismatch")
container_target = Path(sys.argv[2])
fd = os.open(container_target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "w", encoding="ascii") as stream:
    json.dump(container_payload, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
CONTAINER
if [[ "$resume_partial" == 1 ]]; then
  for receipt_name in peft-overlay-receipt.json flash-attention-overlay-receipt.json flash-attention-runtime-receipt.json training-container-runtime-receipt.json; do
    temporary_receipt="$overlay_comparison_root/$receipt_name"
    immutable_receipt="$staging_root/evidence/$receipt_name"
    cmp -s "$temporary_receipt" "$immutable_receipt"
  done
fi
generated_runtime_receipt="$staging_root/evidence/runtime-receipt.json"
if [[ "$resume_partial" == 1 ]]; then generated_runtime_receipt="$resume_scratch_root/runtime-receipt.json"; fi
"$python_bin" - "$root" "$source_root/configs/train_groot.yaml" "$generated_runtime_receipt" "$staging_root/evidence/uv.lock" "$staging_root/evidence/upstream/lerobot-0.4.3-py3-none-any.whl" "$staging_root/evidence/compatibility/lerobot-0.4.3-py3-none-any.whl" "$staging_root/evidence/compatibility/lerobot-compatibility-receipt.json" "$training_root/evidence/uv.lock" "$training_root/evidence/upstream/lerobot-0.4.3-py3-none-any.whl" "$training_root/evidence/compatibility/lerobot-0.4.3-py3-none-any.whl" "$training_root/evidence/compatibility/lerobot-compatibility-receipt.json" <<'PY'
import hashlib, importlib.util, json, os, sys
from pathlib import Path
root, config, output, staged_lock, staged_upstream, staged_wheel, staged_compatibility, final_lock, final_upstream, final_wheel, final_compatibility = map(Path, sys.argv[1:])
sys.path.insert(0, str(root / "source/lehome"))
from lehome.n15_reproduction import resolve_groot_scheduler_from_yaml
package = Path(importlib.util.find_spec("lerobot").origin).parent
value = {"schema_version": 1, "kind": "lehome_public_n15_training_runtime_v1", "python_executable": sys.executable, "upstream_lerobot_wheel_path": str(final_upstream), "upstream_lerobot_wheel_sha256": hashlib.sha256(staged_upstream.read_bytes()).hexdigest(), "compatibility_wheel_path": str(final_wheel), "compatibility_wheel_sha256": hashlib.sha256(staged_wheel.read_bytes()).hexdigest(), "compatibility_wheel_receipt_path": str(final_compatibility), "compatibility_wheel_receipt_sha256": hashlib.sha256(staged_compatibility.read_bytes()).hexdigest(), "lerobot_package_root": str(package), "dependency_lock_path": str(final_lock), "dependency_lock_sha256": hashlib.sha256(staged_lock.read_bytes()).hexdigest(), "scheduler": resolve_groot_scheduler_from_yaml(config.read_text(encoding="utf-8"))}
with output.open("x", encoding="ascii") as stream: stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
os.chmod(output, 0o444)
PY
if [[ "$resume_partial" == 1 ]]; then
  temporary_receipt="$generated_runtime_receipt"
  immutable_receipt="$staging_root/evidence/runtime-receipt.json"
  cmp -s "$temporary_receipt" "$immutable_receipt"
fi
generated_execution_manifest="$staging_root/evidence/execution-manifest.json"
if [[ "$resume_partial" == 1 ]]; then generated_execution_manifest="$resume_scratch_root/execution-manifest.json"; fi
python3 "$root/scripts/run_public_n15_reproduction.py" render-training --checkout "$source_root" --source-receipt "$source_receipt" --resolved-snapshots-receipt "$snapshots" --vm-id "$vm_id" --disk-id "$disk_id" --output "$generated_execution_manifest" >/dev/null
if [[ "$resume_partial" == 1 ]]; then
  temporary_receipt="$generated_execution_manifest"
  immutable_receipt="$staging_root/evidence/execution-manifest.json"
  cmp -s "$temporary_receipt" "$immutable_receipt"
fi
dataset_blobs="$("$python_bin" - "$root/source/lehome" "$snapshots" "$source_root/Datasets/example/four_types_merged" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from lehome.n15_reproduction import resolve_dataset_blobs_mount

print(resolve_dataset_blobs_mount(
    resolved_snapshots_receipt=Path(sys.argv[2]),
    expected_dataset_root=Path(sys.argv[3]),
))
PY
)"
test -d "$dataset_blobs" && test ! -L "$dataset_blobs"
eagle_repository="$hf_cache/models--lerobot--eagle2hg-processor-groot-n1p5"
eagle_snapshot="$eagle_repository/snapshots/baf604d8a5caf26fda5cc545f141bc1814156237"
test -d "$eagle_snapshot" && test ! -L "$eagle_snapshot"
eagle_home="$staging_root/eagle-home"
if [[ "$resume_partial" == 1 && -e "$eagle_home" ]]; then
  test -d "$eagle_home" && test ! -L "$eagle_home"
  sudo -n chown -R --no-dereference "$(id -u):$(id -g)" "$eagle_home"
  find "$eagle_home" -depth -type f -delete
  find "$eagle_home" -depth -type d -empty -delete
fi
test ! -e "$eagle_home" && test ! -L "$eagle_home"
eagle_cache="$eagle_home/lerobot/lerobot/eagle2hg-processor-groot-n1p5"
mkdir -m 0700 -p "$eagle_cache"
for eagle_asset in vocab.json merges.txt added_tokens.json chat_template.json special_tokens_map.json config.json generation_config.json preprocessor_config.json processor_config.json tokenizer_config.json; do
  eagle_asset_source="$(readlink -f "$eagle_snapshot/$eagle_asset")"
  [[ "$eagle_asset_source" == "$eagle_repository/blobs/"* ]]
  test -f "$eagle_asset_source" && test ! -L "$eagle_asset_source"
  install -m 0444 "$eagle_asset_source" "$eagle_cache/$eagle_asset"
done
# The Task1 verifier seals this exact manifest, source/snapshot receipts,
# dependency lock, runtime receipt, train log, and checkpoint—not a hand-made
# approximation of a successful training result.
cd "$source_root"; export HF_HOME="$eagle_home" HF_HUB_OFFLINE=1 HF_HUB_CACHE="$hf_cache"
# GR00T reads this explicit LeRobot cache override at import time. Pin it as
# well as HF_HOME so the prepared offline Eagle assets cannot fall back to the
# guest's default home cache.
HF_LEROBOT_HOME="$eagle_home/lerobot"
export HF_HOME HF_LEROBOT_HOME HF_HUB_OFFLINE HF_HUB_CACHE
sudo -n docker run --rm -i --pull never --gpus all --network none \
  --shm-size "32g" \
  --tmpfs "/flash:rw,exec,size=2g,mode=700,uid=$(id -u),gid=$(id -g)" \
  --mount "type=bind,src=$source_root,dst=$source_root" \
  --mount "type=bind,src=$staging_root,dst=$staging_root" \
  --mount "type=bind,src=$hf_cache,dst=$hf_cache,readonly" \
  --mount "type=bind,src=$dataset_blobs,dst=$dataset_blobs,readonly" \
  --mount "type=bind,src=$staging_root/evidence/compatibility/lerobot-0.4.3-py3-none-any.whl,dst=/runtime/lerobot-0.4.3-py3-none-any.whl,readonly" \
  --mount "type=bind,src=/mnt/lehome/reference-native/dependencies,dst=/deps,readonly" \
  --mount "type=bind,src=/mnt/lehome/reference-native/dependencies,dst=/mnt/lehome/reference-native/dependencies,readonly" \
  --entrypoint bash "$runtime_image_id" -s -- "$source_root" "$eagle_home" "$hf_cache" "$staging_root" "$resume_partial" "$resume_checkpoint" <<'CONTAINER' 2>&1 | tee -a "$resume_log"
set -euo pipefail
source_root="$1"; eagle_home="$2"; hf_cache="$3"; staging_root="$4"; resume_partial="$5"; resume_checkpoint="$6"
python_bin=/opt/lehome-challenge/.venv/bin/python
mkdir -m 0700 /flash/site-packages
dm_tree_wheel="/mnt/lehome/reference-native/dependencies/dm_tree-0.1.9-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
"$python_bin" - /deps/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl "$dm_tree_wheel" /flash/site-packages <<'PY'
import hashlib, sys, zipfile
from pathlib import Path
flash_wheel, dm_tree_wheel, target = map(Path, sys.argv[1:])
if hashlib.sha256(dm_tree_wheel.read_bytes()).hexdigest() != "294dc1cecf87552a45cdd5ddb215e7f5295a5a47c46f1f0a0463c3dd02a527d7":
    raise SystemExit("dm-tree wheel identity mismatch")
lerobot_wheel = Path("/runtime/lerobot-0.4.3-py3-none-any.whl")
with zipfile.ZipFile(lerobot_wheel) as archive:
    archive.extractall(target)
with zipfile.ZipFile(flash_wheel) as archive:
    for item in archive.infolist():
        top_level = item.filename.partition("/")[0]
        if item.filename.startswith(("flash_attn/", "flash_attn_2_cuda")) or (
            top_level.startswith("flash_attn-") and top_level.endswith(".dist-info")
        ):
            archive.extract(item, target)
with zipfile.ZipFile(dm_tree_wheel) as archive:
    archive.extractall(target)
PY
cd "$source_root"
export HF_HOME="$eagle_home" HF_LEROBOT_HOME="$eagle_home/lerobot" HF_HUB_OFFLINE=1 HF_HUB_CACHE="$hf_cache"
PYTHONPATH="/flash/site-packages:/deps/peft-0.18.1-py3-none-any.whl" /opt/lehome-challenge/.venv/bin/python -c 'import lerobot.scripts.lerobot_train'
PYTHONPATH="/flash/site-packages:/deps/peft-0.18.1-py3-none-any.whl" /opt/lehome-challenge/.venv/bin/python - <<'PY'
import importlib.metadata
from pathlib import Path

import tree
from lerobot.policies.groot import groot_n1

if importlib.metadata.version("dm-tree") != "0.1.9":
    raise SystemExit("dm-tree package metadata identity is invalid")
expected_tree = "/flash/site-packages/tree/__init__.py"
if str(Path(tree.__file__).resolve()) != expected_tree:
    raise SystemExit("dm-tree installed runtime identity is invalid")
if tree.map_structure(lambda left, right: left + right, {"joint": 1}, {"joint": 2}) != {"joint": 3}:
    raise SystemExit("dm-tree map_structure runtime probe failed")
if groot_n1.tree is not tree:
    raise SystemExit("LeRobot GR00T did not import the verified dm-tree module")
PY
if [[ "$resume_partial" == 1 ]]; then
  PYTHONPATH="/flash/site-packages:/deps/peft-0.18.1-py3-none-any.whl" /opt/lehome-challenge/.venv/bin/lerobot-train --config_path="$resume_checkpoint/pretrained_model/train_config.json" --resume=true --wandb.mode=offline
else
  PYTHONPATH="/flash/site-packages:/deps/peft-0.18.1-py3-none-any.whl" /opt/lehome-challenge/.venv/bin/lerobot-train --config_path=configs/train_groot.yaml --wandb.mode=offline
fi
CONTAINER
test -d "$upstream_output" && test ! -L "$upstream_output"
test -d "$eagle_home" && test ! -L "$eagle_home"
test -d "$staging_root" && test ! -L "$staging_root"
sudo -n chown -R --no-dereference "$(id -u):$(id -g)" "$upstream_output" "$eagle_home" "$staging_root"
test -z "$(find "$upstream_output" "$eagle_home" "$staging_root" ! -user "$(id -u)" -print -quit)"
if [[ "${FAKE_INTERRUPT_POINT:-}" == after-trainer && "${PYTEST_CURRENT_TEST:-}" == tests/infrastructure/test_public_n15_pipeline_remote.py::* ]]; then kill -TERM "$$"; fi
find "$eagle_home" -depth -type f -delete
find "$eagle_home" -depth -type d -empty -delete
test ! -e "$eagle_home"
if [[ "$resume_partial" == 1 ]]; then
  python3 "$root/scripts/run_public_n15_reproduction.py" cleanup-resume-scratch \
    --staging-root "$staging_root" --attempt-id "$resume_attempt_id" >/dev/null
  resume_scratch_root=""
fi
python3 "$root/scripts/run_public_n15_reproduction.py" finalize-training-output \
  --checkout "$source_root" --source-receipt "$source_receipt" \
  --resolved-snapshots-receipt "$snapshots" --vm-id "$vm_id" --disk-id "$disk_id" \
  --training-root "$training_root" --staging-root "$staging_root" \
  --upstream-output "$upstream_output" >/dev/null
SH
}

publish_training_readback() {
  # The remote publisher uses a fresh, immutable prefix and anonymous byte readback.
  remote bash -s -- "$TRAINING_ROOT" "$PUBLIC_REPOSITORY" "n15-public/$RUN_ID/training" "$HF_TOKEN_FILE" <<'SH'
set -euo pipefail
root="$1"; repository="$2"; prefix="$3"; token_file="$4"; test -f "$token_file" && test ! -L "$token_file"; export HF_TOKEN="$(cat "$token_file")"
python3 - "$root" "$repository" "$prefix" <<'PY'
import hashlib, json, os, re, sys, tempfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
root, repository, prefix = sys.argv[1:]; directory = Path(root); receipt = directory / "training-publication.json"
if receipt.exists(): raise SystemExit("training publication receipt already exists")
for scratch in directory.glob(".training-publication.json.*"):
    scratch.unlink(missing_ok=True)
entries = [{"path": str(path.relative_to(directory)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(directory.rglob("*")) if path.is_file() and path != receipt]
expected = {entry["path"]: entry["sha256"] for entry in entries}
api = HfApi(token=os.environ["HF_TOKEN"])

def verify_revision(revision):
    try:
        tree = api.list_repo_tree(repo_id=repository, repo_type="model", path_in_repo=prefix, revision=revision, recursive=True)
    except EntryNotFoundError:
        return None
    remote_paths = {
        item.rfilename[len(prefix) + 1:]
        for item in tree
        if getattr(item, "size", None) is not None and item.rfilename.startswith(prefix + "/")
    }
    if remote_paths != set(expected):
        return False
    for relative, digest in expected.items():
        fetched = hf_hub_download(repo_id=repository, repo_type="model", filename=prefix + "/" + relative, revision=revision, token=False)
        if hashlib.sha256(Path(fetched).read_bytes()).hexdigest() != digest:
            return False
    return True

revision = None
prefix_seen = False
for prior in api.list_repo_commits(repo_id=repository, repo_type="model", token=os.environ["HF_TOKEN"]):
    candidate = str(prior.commit_id)
    if re.fullmatch(r"[0-9a-f]{40}", candidate) is None:
        raise SystemExit("repository commit history contains an invalid immutable revision")
    match = verify_revision(candidate)
    prefix_seen = prefix_seen or match is not None
    if match is True:
        revision = candidate
        break
if revision is None:
    if prefix_seen:
        raise SystemExit("training publication prefix already exists with different bytes")
    commit = api.upload_folder(repo_id=repository, repo_type="model", folder_path=str(directory), path_in_repo=prefix, commit_message="public N1.5 training " + prefix)
    revision = str(commit.oid)
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None or verify_revision(revision) is not True:
        raise SystemExit("uploaded training revision failed exact anonymous readback")
value = {"schema_version": 1, "kind": "lehome_public_n15_training_publication_v1", "repository": repository, "remote_prefix": prefix, "immutable_revision": revision, "entries": entries, "anonymous_byte_readback_verified": True}
payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
descriptor, temporary_name = tempfile.mkstemp(prefix=".training-publication.json.", dir=directory)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    if (
        os.environ.get("LEHOME_N15_TEST_PUBLICATION_RECEIPT_FAULT") == "after-temporary"
        and os.environ.get("PYTEST_CURRENT_TEST", "").startswith("tests/infrastructure/test_public_n15_pipeline_remote.py::")
    ):
        os._exit(86)
    os.link(temporary, receipt)
    parent = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
finally:
    temporary.unlink(missing_ok=True)
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

run_pipeline_after_runtime() {
  if ! remote_file_exists "$TRAINING_IDENTITY_RECEIPT"; then run_paid_stage train "$TRAIN_TIMEOUT_SECONDS" train_stage; fi
  verify_remote_training_chain || fail "training receipt chain failed"
  if ! remote_file_exists "$TRAINING_PUBLICATION_RECEIPT"; then publish_training_readback || fail "training publication/readback failed"; fi
  verify_remote_training_publication || fail "training publication chain failed"
  record_host_stage_completion training "$HOST_TRAINING_STAGE_RECEIPT" \
    "$TRAINING_IDENTITY_RECEIPT" "$TRAINING_PUBLICATION_RECEIPT"
  if ! remote_file_exists "$FOCUSED_PROMOTION_RECEIPT"; then run_paid_stage focused_gate "$FOCUSED_TIMEOUT_SECONDS" focused_stage; fi
  verify_remote_focused_chain || fail "focused receipt chain failed"
  record_host_stage_completion focused "$HOST_FOCUSED_STAGE_RECEIPT" \
    "$FOCUSED_OUTPUT_ROOT/comparison-receipt.json" \
    "$FOCUSED_OUTPUT_ROOT/publication.json" "$FOCUSED_PROMOTION_RECEIPT"
  if [[ ! -e "$HARVEST_TERMINAL_RECEIPT" ]]; then
    if ! remote_file_exists "$REMOTE_PIPELINE_ROOT/harvest.publication.json"; then
      run_paid_stage harvest "$HARVEST_TIMEOUT_SECONDS" harvest_stage
    fi
    verify_remote_harvest_chain || fail "harvest pre-stop receipt chain failed"
    fetch_remote_immutable "$HARVEST_ROOT/manifest.json" "$HARVEST_MANIFEST"
    if [[ "${LEHOME_N15_TEST_HARVEST_FETCH_FAULT_AFTER:-}" == 1 && "${PYTEST_CURRENT_TEST:-}" == tests/infrastructure/test_public_n15_pipeline_remote.py::* ]]; then kill -TERM "$$"; fi
    fetch_remote_immutable "$HARVEST_ROOT/manifest-receipt.json" "$HARVEST_MANIFEST_RECEIPT"
    if [[ "${LEHOME_N15_TEST_HARVEST_FETCH_FAULT_AFTER:-}" == 2 && "${PYTEST_CURRENT_TEST:-}" == tests/infrastructure/test_public_n15_pipeline_remote.py::* ]]; then kill -TERM "$$"; fi
    fetch_remote_immutable "$REMOTE_PIPELINE_ROOT/harvest.publication.json" "$HARVEST_PUBLICATION_RECEIPT"
    stop_exact_vm || fail "exact VM could not be stopped"
    finalize_host_harvest_terminal || fail "host harvest terminal verification failed"
  else
    python3 "$HARVEST_BUILDER" verify-terminal --manifest "$HARVEST_MANIFEST" --manifest-receipt "$HARVEST_MANIFEST_RECEIPT" --publication-receipt "$HARVEST_PUBLICATION_RECEIPT" --provider-receipt "$PROVIDER_STOPPED_RECEIPT" --output "$(mktemp -d "${TMPDIR:-/tmp}/lehome-n15-verify-terminal.XXXXXX")/receipt.json" >/dev/null || fail "existing host harvest terminal chain failed"
  fi
  stop_exact_vm || fail "exact VM could not be stopped"
  PIPELINE_COMPLETE=1
}

main() {
[[ $# -eq 0 ]] || fail "this wrapper accepts no positional arguments"
[[ "$RESUME_PARTIAL" == 0 || "$RESUME_PARTIAL" == 1 ]] || fail "resume-partial mode must be explicitly 0 or 1"
if [[ "$RESUME_PARTIAL" == 1 ]]; then
  [[ "$RESUME_STEP" =~ ^[0-9]+$ && "$RESUME_CHECKPOINT" == "$SOURCE_ROOT/outputs/train/groot_four_types_merged_batch64_lr2e-4/checkpoints/$(printf '%06d' "$RESUME_STEP")" ]] || fail "resume requires the exact configured checkpoint path and step"
  (( RESUME_STEP > 0 && RESUME_STEP < 12000 && RESUME_STEP % 1500 == 0 )) || fail "resume step is not a valid checkpoint boundary"
  [[ "$RESUME_ATTEMPT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$ ]] || fail "resume requires a distinct attempt identity"
else
  [[ -z "$RESUME_CHECKPOINT" && -z "$RESUME_STEP" && -z "$RESUME_ATTEMPT_ID" ]] || fail "resume checkpoint inputs require explicit resume-partial mode"
fi
command -v nebius >/dev/null 2>&1 || fail "Nebius CLI is unavailable"
command -v ssh >/dev/null 2>&1 || fail "SSH is unavailable"
require_abs_dir "$PIPELINE_ROOT" "pipeline receipt root"; require_abs_file "$BUILDER" "checked-in lifecycle planner"
require_abs_file "$PROVIDER_VERIFIER" "checked-in exact Nebius provider parser"; require_abs_file "$HARVEST_BUILDER" "checked-in harvest provider parser"
[[ "$PUBLIC_REPOSITORY" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ && -n "$SSH_TARGET" && "$REMOTE_ROOT" == /* && "$REMOTE_PIPELINE_ROOT" == /* && -n "$ASSETS_ROOT" && -n "$METADATA_ROOT" && -n "$REFERENCE_CHECKPOINT" && -n "$REFERENCE_SANITIZED_CONFIG" && -n "$REFERENCE_COMPATIBILITY" && -n "$NATIVE_RUNTIME_EVIDENCE" && -n "$NATIVE_DEPENDENCIES" && -n "$FOCUSED_HF_CACHE" && -n "$ROLLOUT_IMAGE_RECEIPT" && -n "$TRAINING_HF_CACHE" && -n "$TRAINING_UV" && -n "$LEROBOT_WHEEL" ]] || fail "all canonical remote inputs are required"
[[ "$TRAINING_ROOT" == "$REMOTE_PIPELINE_ROOT/training" ]] || fail "training root must be this run's canonical remote training directory"
[[ "$REMOTE_PIPELINE_ROOT" == "$REMOTE_RUNS_BASE/$RUN_ID" ]] || fail "remote pipeline root must be the canonical run-specific directory"
verify_conservative_task_budget || fail "conservative provider budget admission failed"
# Immutable pre-start cost admission: run_public_n15_reproduction.py lifecycle-plan.
if [[ ! -e "$PLAN_RECEIPT" ]]; then python3 "$BUILDER" lifecycle-plan --run-id "$RUN_ID" --repository "$PUBLIC_REPOSITORY" --remote-pipeline-root "$REMOTE_PIPELINE_ROOT" --budget-usd "$MAX_BUDGET_USD" --estimated-cost-usd "$ESTIMATED_COST_USD" --output "$PLAN_RECEIPT" >/dev/null; fi
python3 "$BUILDER" verify-lifecycle-plan --run-id "$RUN_ID" --repository "$PUBLIC_REPOSITORY" --remote-pipeline-root "$REMOTE_PIPELINE_ROOT" --budget-usd "$MAX_BUDGET_USD" --estimated-cost-usd "$ESTIMATED_COST_USD" --output "$PLAN_RECEIPT" >/dev/null
acquire_controller_lock || fail "another N1.5 controller already owns this run"
trap controller_cleanup EXIT
if [[ "$RESUME_PARTIAL" == 1 && -f "$HARVEST_TERMINAL_RECEIPT" ]]; then
  fail "explicit partial resume is forbidden for a completed canonical pipeline"
fi
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
aggregate_deadline="$(initialize_deadline)" || fail "aggregate paid deadline is invalid"
verify_conservative_task_budget "$DEADLINE_RECEIPT" \
  || fail "conservative provider budget admission failed"
(( $(date +%s) < aggregate_deadline )) || fail "aggregate paid deadline has expired"
PRESTART_ADMITTED_STAGE="$(host_next_unfinished_stage)" \
  || fail "host-sealed next unfinished stage is invalid"
case "$PRESTART_ADMITTED_STAGE" in
  train) admitted_timeout="$TRAIN_TIMEOUT_SECONDS" ;;
  focused_gate) admitted_timeout="$FOCUSED_TIMEOUT_SECONDS" ;;
  harvest) admitted_timeout="$HARVEST_TIMEOUT_SECONDS" ;;
  *) fail "host-sealed next unfinished stage is invalid" ;;
esac
admitted_deadline="$(initialize_stage_deadline "$PRESTART_ADMITTED_STAGE" "$admitted_timeout" "$aggregate_deadline")" \
  || fail "$PRESTART_ADMITTED_STAGE deadline receipt is invalid"
(( $(date +%s) < admitted_deadline )) \
  || fail "$PRESTART_ADMITTED_STAGE deadline has expired"
# A provider STOPPED observation is also the fail-closed proof that no trainer
# or prior controller can already be live when this controller admits resume.
# provider must be STOPPED before explicit partial resume.
response="$PIPELINE_ROOT/.provider-start.$$.json"; capture_exact_provider_state STOPPED "$response" || fail "Nebius Compute API is unavailable or exact VM is not stopped"; rm -f -- "$response"
PROVIDER_CLEANUP_REQUIRED=1
nebius compute instance start --id "$EXACT_VM_ID" --format json --no-browser --no-progress --no-check-update --retries 1 --timeout 60s >/dev/null
running_observed=0
for _ in {1..60}; do
  if capture_exact_provider_state RUNNING "$response"; then running_observed=1; break; fi
  rm -f -- "$response"; sleep 2
done
(( running_observed == 1 )) && [[ -f "$response" && ! -L "$response" ]] || fail "exact VM did not reach RUNNING"
rm -f -- "$response"
wait_for_ssh_readiness || fail "exact VM did not become SSH-ready"
wait_for_remote_runtime || fail "runtime/cloud-init/workspace/GPU/upstream gate failed"
reconcile_remote_stage_seals
reconciled_stage="$(host_next_unfinished_stage)" \
  || fail "reconciled host-sealed stage is invalid"
if [[ "$reconciled_stage" != "$PRESTART_ADMITTED_STAGE" ]]; then
  PRESTART_ADMITTED_STAGE="$reconciled_stage"
  case "$PRESTART_ADMITTED_STAGE" in
    train) admitted_timeout="$TRAIN_TIMEOUT_SECONDS" ;;
    focused_gate) admitted_timeout="$FOCUSED_TIMEOUT_SECONDS" ;;
    harvest) admitted_timeout="$HARVEST_TIMEOUT_SECONDS" ;;
    *) fail "reconciled host-sealed stage is invalid" ;;
  esac
  aggregate_deadline="$(initialize_deadline)" || fail "aggregate paid deadline is invalid"
  verify_conservative_task_budget "$DEADLINE_RECEIPT" \
    || fail "conservative provider budget admission failed"
  (( $(date +%s) < aggregate_deadline )) || fail "aggregate paid deadline has expired"
  admitted_deadline="$(initialize_stage_deadline "$PRESTART_ADMITTED_STAGE" "$admitted_timeout" "$aggregate_deadline")" \
    || fail "$PRESTART_ADMITTED_STAGE deadline receipt is invalid"
  (( $(date +%s) < admitted_deadline )) \
    || fail "$PRESTART_ADMITTED_STAGE deadline has expired"
fi
if [[ "$RESUME_PARTIAL" == 1 ]] && { remote_file_exists "$TRAINING_IDENTITY_RECEIPT" || remote_file_exists "$TRAINING_PUBLICATION_RECEIPT"; }; then
  fail "explicit partial resume is forbidden after canonical training receipts exist"
fi
run_pipeline_after_runtime
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
