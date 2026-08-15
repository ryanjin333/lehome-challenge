"""Fail-closed lifecycle for one paid public-unseen top-40 checkpoint evaluation.

The functions in this module are deliberately explicit.  They do not discover,
reuse, or retain provider resources: a lifecycle owns one newly-created
on-demand host and every post-create failure performs the same evidence-sync
and disposal path.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import shlex
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence

from lehome.flywheel.matrix import build_public_matrix, matrix_sha256


APPROVED_IMAGE_REPOSITORY = "mirror.gcr.io/ryanjin333/lehome-rollout"
# This is the immutable Docker Hub manifest used by the canonical 1K/2K runs.
APPROVED_IMAGE_DIGEST = "sha256:293c4f258f3742a7234699d706fb7088d0da8a764957bc79b244d830561abc12"
MAX_WALL_SECONDS = 4 * 60 * 60
LEASE_WATCHDOG_RESERVE_SECONDS = 5 * 60
MAX_TOTAL_DOLLARS = 3.00
MAX_PROJECTED_HOURLY_USD = 1.00  # strict: equality is rejected
OFFER_QUERY = "gpu_name=RTX_3090 num_gpus=4 reliability>=0.95 cpu_cores_effective>=64 cpu_ram>=128 disk_space>=300 duration>=4"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
VAST_SSH_IDENTITY = str(Path.home() / ".ssh" / "vast_quest")
EVALUATION_RENT_CLAIM_ROOT = Path("/private/tmp/lehome-checkpoint-evaluation-rent-claims")


def _corrective_module():
    """Load the reviewed image-native runtime helpers without provider actions."""
    name = "checkpoint_evaluation_corrective_runtime"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("run_groot_corrective_vast_lifecycle.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("reviewed corrective runtime helpers are unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _campaign_module():
    name = "checkpoint_evaluation_campaign_runtime"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("run_groot_flywheel_campaign.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("campaign parser is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


APPROVED_GROOT_ROOT = _corrective_module().APPROVED_GROOT_ROOT
APPROVED_GROOT_REVISION = _corrective_module().APPROVED_GROOT_REVISION
APPROVED_GROOT_PYTHON = _corrective_module().APPROVED_GROOT_PYTHON
APPROVED_GROOT_PYTHON_SHA256 = _corrective_module().APPROVED_GROOT_PYTHON_SHA256
APPROVED_ASSET_REVISION = _corrective_module().APPROVED_ASSET_REVISION
APPROVED_POLICY_REPOSITORY = "ryanjin333/lehome-groot-n17-models"
APPROVED_EVALUATION_REPOSITORY = "ryanjin333/lehome-groot-n17-data"
APPROVED_POLICY_REVISION = "a9076779c970f382bf0341a1015275bf15f13822"
APPROVED_POLICY_STEP = 12000
APPROVED_POLICY_ARTIFACT_SHA256 = "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def canonical_trial_ids() -> list[str]:
    return [
        trial.trial_id for trial in build_public_matrix().trials
        if trial.release_stage == "public_unseen" and trial.category in {"top_long", "top_short"}
    ]


def canonical_matrix_sha256() -> str:
    return matrix_sha256(build_public_matrix())


def _read_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is invalid")
    return value


def _write_new(path: Path, value: Mapping[str, object]) -> dict[str, object]:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_text(encoding="utf-8") != encoded:
            raise ValueError("refusing to overwrite differing lifecycle evidence")
    else:
        path.write_text(encoded, encoding="utf-8")
    return dict(value)


def _number(value: object) -> float:
    if isinstance(value, bool):
        return math.inf
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.inf
    return parsed if math.isfinite(parsed) else math.inf


def _r580(value: object) -> bool:
    if not isinstance(value, str):
        return False
    found = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?", value)
    if found is None:
        return False
    version = tuple(int(item or 0) for item in found.groups())
    return (580, 65, 6) <= version < (590, 0, 0)


def _approved_host(row: Mapping[str, object], *, require_running: bool) -> bool:
    hourly = _number(row.get("dph_total"))
    return (
        (not require_running or row.get("actual_status") == "running")
        and row.get("is_bid") is False
        and row.get("gpu_name") == "RTX 3090"
        and row.get("num_gpus") == 4
        and _number(row.get("cpu_cores_effective")) >= 64
        and _number(row.get("cpu_ram")) >= 128_000
        and _r580(row.get("driver_version"))
        and 0 < hourly < MAX_PROJECTED_HOURLY_USD
        and hourly * MAX_WALL_SECONDS / 3600 <= MAX_TOTAL_DOLLARS
    )


def _validate_invocation(invocation: object) -> dict[str, object]:
    if not isinstance(invocation, dict):
        raise ValueError("lifecycle manifest lacks invocation")
    required = {
        "kind": "public_unseen_tops_checkpoint_evaluation",
        "matrix_sha256": canonical_matrix_sha256(),
        "selected_trial_ids": canonical_trial_ids(),
        "execution_mode": "policy_server",
        "simulator_device": "cpu",
        "policy_device_pool": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        "workers": 4,
        "strategy": "canonical",
        "max_steps": 600,
    }
    if any(invocation.get(key) != value for key, value in required.items()):
        raise ValueError("lifecycle manifest invocation is not the canonical CPU policy-server top-40 run")
    for key, pattern in (("policy_revision", _COMMIT), ("code_revision", _COMMIT), ("asset_revision", _COMMIT), ("groot_revision", _COMMIT), ("policy_artifact_sha256", _SHA256)):
        if not isinstance(invocation.get(key), str) or pattern.fullmatch(str(invocation[key])) is None:
            raise ValueError(f"lifecycle manifest invocation has invalid {key}")
    if not isinstance(invocation.get("policy_repo"), str) or not invocation["policy_repo"]:
        raise ValueError("lifecycle manifest lacks immutable policy repository")
    if (invocation.get("policy_repo") != APPROVED_POLICY_REPOSITORY
            or invocation.get("policy_revision") != APPROVED_POLICY_REVISION
            or invocation.get("policy_step") != APPROVED_POLICY_STEP
            or invocation.get("policy_artifact_sha256") != APPROVED_POLICY_ARTIFACT_SHA256):
        raise ValueError("lifecycle manifest baseline checkpoint is not the approved 12K identity")
    if type(invocation.get("policy_step")) is not int or int(invocation["policy_step"]) <= 0:
        raise ValueError("lifecycle manifest lacks policy checkpoint step")
    if (invocation.get("groot_revision") != APPROVED_GROOT_REVISION
            or invocation.get("groot_root") != APPROVED_GROOT_ROOT
            or invocation.get("groot_python") != APPROVED_GROOT_PYTHON
            or invocation.get("groot_python_sha256") != APPROVED_GROOT_PYTHON_SHA256
            or invocation.get("groot_python_version") != "3.10.18"):
        raise ValueError("lifecycle manifest lacks immutable GR00T runtime")
    if invocation.get("image_identity") != APPROVED_IMAGE_DIGEST:
        raise ValueError("lifecycle manifest image identity is not the approved comparable digest")
    return dict(invocation)


def read_lifecycle_manifest(path: Path) -> dict[str, object]:
    value = _read_json(path, "evaluation lifecycle manifest")
    if value.get("schema_version") != 1 or value.get("kind") != "groot_checkpoint_evaluation_lifecycle":
        raise ValueError("evaluation lifecycle manifest is invalid")
    invocation = _validate_invocation(value.get("invocation"))
    runtime = value.get("runtime")
    if (not isinstance(runtime, dict) or runtime.get("matrix_path") != "configs/eval_groot_n17_public_280.json"
            or runtime.get("policy_path") != "policies/step-12000"
            or runtime.get("release_assets_root") != "Assets/objects/Challenge_Garment/Release"
            or runtime.get("groot_root") != APPROVED_GROOT_ROOT):
        raise ValueError("evaluation lifecycle manifest lacks exact operational runtime paths")
    if invocation.get("asset_revision") != APPROVED_ASSET_REVISION:
        raise ValueError("evaluation lifecycle manifest asset revision is not approved")
    if value.get("rollout_image") != f"{APPROVED_IMAGE_REPOSITORY}@{APPROVED_IMAGE_DIGEST}":
        raise ValueError("evaluation lifecycle manifest rollout image is not immutable and comparable")
    if not isinstance(value.get("code_bundle_sha256"), str) or _SHA256.fullmatch(str(value["code_bundle_sha256"])) is None:
        raise ValueError("evaluation lifecycle manifest lacks a clean code bundle digest")
    if value.get("hard_wall_seconds") != MAX_WALL_SECONDS or value.get("total_dollar_ceiling_usd") != MAX_TOTAL_DOLLARS:
        raise ValueError("evaluation lifecycle manifest may not widen duration or dollar caps")
    return {**value, "invocation": invocation, "runtime": dict(runtime)}


def capture_provider_evidence(*, offers: Sequence[Mapping[str, object]], instances: Sequence[Mapping[str, object]], volumes: Sequence[Mapping[str, object]], output: Path, now_unix: int) -> dict[str, object]:
    """Record a redacted pre-rent snapshot only when the account is empty."""
    if type(now_unix) is not int or now_unix <= 0:
        raise ValueError("provider evidence time is invalid")
    if instances:
        raise ValueError("evaluation lifecycle requires account-wide zero instances before rent")
    if volumes:
        raise ValueError("evaluation lifecycle requires account-wide zero volumes before rent")
    accepted = [row for row in offers if _approved_host(row, require_running=False)]
    if not accepted:
        if any(0 < _number(row.get("dph_total")) < MAX_PROJECTED_HOURLY_USD and _number(row.get("dph_total")) * MAX_WALL_SECONDS / 3600 > MAX_TOTAL_DOLLARS for row in offers):
            raise ValueError("evaluation lifecycle offer exceeds the hard total-dollar ceiling")
        if any(row.get("is_bid") is not False for row in offers):
            raise ValueError("evaluation lifecycle requires an on-demand offer")
        if any(row.get("gpu_name") == "RTX 3090" and row.get("num_gpus") == 4 and not _r580(row.get("driver_version")) for row in offers):
            raise ValueError("evaluation lifecycle requires an R580 host")
        raise ValueError("evaluation lifecycle requires a sub-$1 on-demand 4xRTX3090 offer")
    selected = min(accepted, key=lambda row: (_number(row.get("dph_total")), int(row.get("id", 2**63 - 1))))
    if type(selected.get("id")) is not int:
        raise ValueError("evaluation lifecycle offer lacks a stable ID")
    snapshot = {
        "offers": [{key: row.get(key) for key in ("id", "is_bid", "gpu_name", "num_gpus", "cpu_cores_effective", "cpu_ram", "driver_version", "dph_total")} for row in offers],
        "instances": [], "volumes": [], "captured_at_unix": now_unix,
    }
    source = output.with_name(output.stem + "-source.json")
    _write_new(source, snapshot)
    return _write_new(output, {
        "schema_version": 1, "kind": "groot_checkpoint_evaluation_provider_evidence",
        "source_snapshot_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_response_sha256": canonical_sha256(snapshot), "queried_at_unix": now_unix,
        "rental_kind": "on-demand", "offer_id": selected["id"], "gpu_name": "RTX 3090", "num_gpus": 4,
        "driver_version": selected["driver_version"], "instance_hourly_cost_usd": _number(selected["dph_total"]),
        "projected_account_hourly_total_usd": _number(selected["dph_total"]),
        "projected_total_dollars_at_wall_cap": _number(selected["dph_total"]) * MAX_WALL_SECONDS / 3600,
        "hard_wall_seconds": MAX_WALL_SECONDS, "total_dollar_ceiling_usd": MAX_TOTAL_DOLLARS,
    })


def _raw(runner: Callable[[tuple[str, ...]], object], command: tuple[str, ...]) -> object:
    result = runner(command)
    payload = result if isinstance(result, str) else getattr(result, "stdout", "")
    try:
        return json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("provider raw response is invalid") from error


def _ssh_identity() -> str:
    identity = Path(VAST_SSH_IDENTITY)
    if identity.is_symlink() or not identity.is_file():
        raise ValueError("canonical Vast SSH identity is unavailable or unsafe")
    return str(identity)


def _wait_running(instance_id: int, runner: Callable[[tuple[str, ...]], object], *, polls: int = 360, sleep: Callable[[float], None] = time.sleep) -> dict[str, object]:
    for index in range(polls):
        row = _raw(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
        if isinstance(row, dict) and row.get("id") == instance_id and _approved_host(row, require_running=True):
            if isinstance(row.get("ssh_host"), str) and row["ssh_host"] and type(row.get("ssh_port", 22)) is int:
                return row
        if index + 1 < polls:
            sleep(5.0)
    raise ValueError("new instance did not reach bounded running SSH-ready state")


def _verify_absent(instance_id: int, runner: Callable[[tuple[str, ...]], object], *, polls: int = 60, sleep: Callable[[float], None] = time.sleep) -> tuple[bool, bool, bool]:
    status = (False, False, False)
    for index in range(polls):
        exact = _raw(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
        instances = _raw(runner, ("vastai", "--raw", "show", "instances"))
        volumes = _raw(runner, ("vastai", "--raw", "show", "volumes"))
        exact_absent = exact in (None, {}, []) or exact == {"instances": None}
        status = (exact_absent, instances == [], volumes == [])
        if all(status) or index + 1 == polls:
            return status
        sleep(2.0)
    return status


def _resolve_ambiguous_create(
    runner: Callable[[tuple[str, ...]], object], *, polls: int = 60,
    sleep: Callable[[float], None] = time.sleep,
) -> int | None:
    """Identify the only new compatible host after a malformed create reply."""
    candidate: int | None = None
    for index in range(polls):
        rows = _raw(runner, ("vastai", "--raw", "show", "instances"))
        volumes = _raw(runner, ("vastai", "--raw", "show", "volumes"))
        if volumes != [] or not isinstance(rows, list):
            return None
        if len(rows) == 1:
            row = rows[0]
            observed = row.get("id") if isinstance(row, dict) else None
            if type(observed) is not int or observed <= 0:
                return None
            candidate = observed
            if _approved_host(row, require_running=False):
                return candidate
        if rows:
            if len(rows) != 1:
                return None
        if index + 1 < polls:
            sleep(2.0)
    # The account was empty immediately before create.  A sole positive-ID row
    # is therefore owned even while Vast is still populating its offer fields;
    # return it so the failure path destroys it instead of leaking the lease.
    return candidate


def _cleanup_failure(instance_id: int, lifecycle_root: Path, runner: Callable[[tuple[str, ...]], object], reason: str, *, sync_attempted: bool = False) -> None:
    """Destroy the owned instance once and preserve redacted disposal evidence."""
    destroy_issued, exact_empty, instances_empty, volumes_empty = _destroy_owned_once(instance_id, runner)
    receipt = {
        "schema_version": 1, "kind": "groot_checkpoint_evaluation_failure_disposal",
        "instance_id": instance_id, "reason": reason, "instance_absent": exact_empty,
        "account_instances_empty": instances_empty, "account_volumes_empty": volumes_empty,
        "destroy_issued": destroy_issued, "synced_available_evidence": sync_attempted, "redacted": True,
    }
    _write_new(lifecycle_root / f"failure-{instance_id}.json", receipt)
    if not all((exact_empty, instances_empty, volumes_empty)):
        raise RuntimeError("failure disposal did not verify empty provider account")


def _rent_claim_path(manifest: Mapping[str, object]) -> Path:
    claim_root = EVALUATION_RENT_CLAIM_ROOT
    if claim_root.is_symlink():
        raise ValueError("evaluation rent claim root is unsafe")
    claim_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if claim_root.is_symlink() or not claim_root.is_dir():
        raise ValueError("evaluation rent claim root is unsafe")
    # One account-wide claim prevents distinct valid manifests from racing the
    # zero-resource preflight into two simultaneous provider creates.
    return claim_root / "active.json"


def _acquire_rent_claim(path: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    claim = {
        "schema_version": 1,
        "kind": "groot_checkpoint_evaluation_rent_claim",
        "status": "claimed",
        "invocation_sha256": canonical_sha256(manifest["invocation"]),
        "rollout_image": manifest["rollout_image"],
        "code_bundle_sha256": manifest["code_bundle_sha256"],
    }
    encoded = (json.dumps(claim, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("evaluation rent claim is already held; no provider action was taken") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return claim


def _replace_rent_claim(path: Path, expected: Mapping[str, object], replacement: Mapping[str, object]) -> None:
    if _read_json(path, "evaluation rent claim") != dict(expected):
        raise RuntimeError("evaluation rent claim changed unexpectedly")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError("evaluation rent claim temporary path is not clean")
    encoded = json.dumps(replacement, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        temporary.write_text(encoded, encoding="utf-8")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _release_rent_claim(path: Path, claim: Mapping[str, object]) -> None:
    if _read_json(path, "evaluation rent claim") != dict(claim):
        raise RuntimeError("evaluation rent claim changed before release")
    path.unlink()


def _terminalize_rent_claim(path: Path, claim: Mapping[str, object], *, status: str, instance_id: int | None = None) -> None:
    replacement = dict(claim) | {"status": status}
    if instance_id is not None:
        replacement["instance_id"] = instance_id
    _replace_rent_claim(path, claim, replacement)


def _destroy_owned_once(
    instance_id: int, runner: Callable[[tuple[str, ...]], object], *, polls: int = 60,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bool, bool, bool, bool]:
    """Serialize watchdog, failure cleanup, and publication disposal."""
    root = EVALUATION_RENT_CLAIM_ROOT
    if root.is_symlink():
        raise RuntimeError("evaluation destroy lock root is unsafe")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(root / "destroy.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        exact_empty, instances_empty, volumes_empty = _verify_absent(instance_id, runner, polls=1, sleep=sleep)
        destroy_issued = False
        if not exact_empty or not instances_empty:
            runner(("vastai", "destroy", "instance", str(instance_id), "--yes"))
            destroy_issued = True
            exact_empty, instances_empty, volumes_empty = _verify_absent(instance_id, runner, polls=polls, sleep=sleep)
        return destroy_issued, exact_empty, instances_empty, volumes_empty
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _release_completed_rent_claim(instance: Mapping[str, object]) -> None:
    expected_path = EVALUATION_RENT_CLAIM_ROOT / "active.json"
    if instance.get("rent_claim_path") != str(expected_path):
        raise RuntimeError("evaluation instance does not bind the controller-owned rent claim")
    claim = _read_json(expected_path, "evaluation rent claim")
    if claim.get("status") != "succeeded" or claim.get("instance_id") != instance.get("instance_id"):
        raise RuntimeError("evaluation rent claim does not match the completed instance")
    expected_path.unlink()
    parent_descriptor = os.open(expected_path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _launch_lease_watchdog(instance_id: int, deadline_unix: int, receipt: Path) -> int:
    """Start a detached local controller that enforces the hard lease deadline."""
    process = subprocess.Popen(
        (
            "/usr/bin/caffeinate", "-dimsu", sys.executable,
            str(Path(__file__).resolve()), "watchdog-destroy",
            "--instance-id", str(instance_id), "--deadline-unix", str(deadline_unix),
            "--receipt", str(receipt), "--execute",
        ),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True, close_fds=True,
    )
    if type(process.pid) is not int or process.pid <= 0:
        raise RuntimeError("evaluation lease watchdog did not start")
    return process.pid


def enforce_lease_deadline(
    instance_id: int, deadline_unix: int, receipt: Path, *,
    runner: Callable[[tuple[str, ...]], object], sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.time,
) -> dict[str, object]:
    """Destroy the owned host at the hard deadline unless it is already absent."""
    if type(instance_id) is not int or instance_id <= 0 or type(deadline_unix) is not int or deadline_unix <= 0:
        raise ValueError("evaluation lease watchdog identity is invalid")
    if not receipt.is_absolute() or receipt.exists() or receipt.is_symlink():
        raise ValueError("evaluation lease watchdog receipt must be an absent absolute path")
    delay = max(0.0, float(deadline_unix) - now())
    if delay:
        sleep(delay)
    destroy_issued, exact_empty, instances_empty, volumes_empty = _destroy_owned_once(
        instance_id, runner, sleep=sleep,
    )
    value = {
        "schema_version": 1, "kind": "groot_checkpoint_evaluation_lease_watchdog",
        "instance_id": instance_id, "deadline_unix": deadline_unix,
        "destroy_issued": destroy_issued, "instance_absent": exact_empty,
        "account_instances_empty": instances_empty, "account_volumes_empty": volumes_empty,
        "destroyed_and_absent": all((exact_empty, instances_empty, volumes_empty)),
    }
    written = _write_new(receipt, value)
    if not written["destroyed_and_absent"]:
        raise RuntimeError("evaluation lease watchdog could not verify provider absence")
    return written


def rent_evaluation(manifest_path: Path, *, lifecycle_root: Path, runner: Callable[[tuple[str, ...]], object], now_unix: int | None = None, ambiguous_create_polls: int = 60, sleep: Callable[[float], None] = time.sleep, watchdog_launcher: Callable[[int, int, Path], int] = _launch_lease_watchdog) -> dict[str, object]:
    """Rent exactly one compatible host after a validated local-only preflight."""
    manifest = read_lifecycle_manifest(manifest_path)  # must precede every provider call
    timestamp = time.time_ns() // 1_000_000_000 if now_unix is None else now_unix
    _ssh_identity()  # fail before the irreversible provider create call
    claim_path = _rent_claim_path(manifest)
    claim = _acquire_rent_claim(claim_path, manifest)
    create_attempted = False
    instance_id: int | None = None
    try:
        instances = _raw(runner, ("vastai", "--raw", "show", "instances"))
        volumes = _raw(runner, ("vastai", "--raw", "show", "volumes"))
        if not isinstance(instances, list) or not isinstance(volumes, list):
            raise ValueError("provider account preflight response is invalid")
        if instances:
            raise ValueError("evaluation lifecycle requires account-wide zero instances before rent")
        if volumes:
            raise ValueError("evaluation lifecycle requires account-wide zero volumes before rent")
        offers = _raw(runner, ("vastai", "--raw", "search", "offers", OFFER_QUERY, "--on-demand", "--storage", "300"))
        if not isinstance(offers, list):
            raise ValueError("provider account preflight response is invalid")
        evidence = capture_provider_evidence(offers=offers, instances=instances, volumes=volumes, output=lifecycle_root / f"provider-{timestamp}.json", now_unix=timestamp)
        create_attempted = True
        created = _raw(runner, (
            "vastai", "--raw", "create", "instance", str(evidence["offer_id"]), "--image", str(manifest["rollout_image"]),
            "--env", f"-e LEHOME_FLYWHEEL_IMAGE_IDENTITY={APPROVED_IMAGE_DIGEST}", "--disk", "300", "--ssh", "--direct", "--cancel-unavail",
        ))
        candidate = created.get("new_contract") if isinstance(created, dict) else None
        if type(candidate) is not int or candidate <= 0:
            raise RuntimeError("provider create response lacks instance ID")
        instance_id = candidate
        lease_started = timestamp
        lease_deadline = lease_started + MAX_WALL_SECONDS
        watchdog_receipt = lifecycle_root / f"watchdog-{instance_id}.json"
        watchdog_pid = watchdog_launcher(instance_id, lease_deadline, watchdog_receipt)
        live = _wait_running(instance_id, runner)
        if _number(live.get("dph_total")) != evidence["instance_hourly_cost_usd"]:
            raise ValueError("new instance readback cost differs from preflight offer")
        invocation = manifest["invocation"]
        receipt = {
            "schema_version": 1, "kind": "groot_checkpoint_evaluation_instance", "instance_id": instance_id,
            "host": live["ssh_host"], "port": live.get("ssh_port", 22), "invocation_sha256": canonical_sha256(invocation),
            "provider_evidence_sha256": canonical_sha256(evidence), "provider_response_sha256": canonical_sha256(live),
            "lease_started_unix": lease_started, "lease_deadline_unix": lease_deadline,
            "watchdog_pid": watchdog_pid, "watchdog_receipt": str(watchdog_receipt),
            "rent_claim_path": str(claim_path),
            "hard_wall_seconds": MAX_WALL_SECONDS, "total_dollar_ceiling_usd": MAX_TOTAL_DOLLARS,
        }
        written = _write_new(lifecycle_root / f"instance-{instance_id}.json", receipt)
        _terminalize_rent_claim(claim_path, claim, status="succeeded", instance_id=instance_id)
        return written
    except BaseException as error:
        if instance_id is not None:
            try:
                _cleanup_failure(instance_id, lifecycle_root, runner, "post_create_failure")
            except BaseException:
                _terminalize_rent_claim(claim_path, claim, status="blocked_cleanup_unverified", instance_id=instance_id)
                raise
            _release_rent_claim(claim_path, claim)
        elif create_attempted:
            try:
                resolved = _resolve_ambiguous_create(runner, polls=ambiguous_create_polls, sleep=sleep)
            except BaseException:
                resolved = None
            if resolved is None:
                _terminalize_rent_claim(claim_path, claim, status="blocked_ambiguous_create")
            else:
                try:
                    _cleanup_failure(resolved, lifecycle_root, runner, "ambiguous_create_response")
                except BaseException:
                    _terminalize_rent_claim(claim_path, claim, status="blocked_cleanup_unverified", instance_id=resolved)
                    raise
                _release_rent_claim(claim_path, claim)
        else:
            _release_rent_claim(claim_path, claim)
        raise error


def _prelaunch_account(instance: Mapping[str, object], runner: Callable[[tuple[str, ...]], object]) -> None:
    instance_id = instance.get("instance_id")
    if type(instance_id) is not int:
        raise ValueError("instance receipt is invalid")
    rows = _raw(runner, ("vastai", "--raw", "show", "instances"))
    volumes = _raw(runner, ("vastai", "--raw", "show", "volumes"))
    exact = _raw(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
    if not isinstance(rows, list) or not isinstance(volumes, list) or len(rows) != 1 or rows[0].get("id") != instance_id or volumes != [] or not isinstance(exact, dict) or exact.get("id") != instance_id or not _approved_host(exact, require_running=True):
        raise ValueError("pre-launch provider account is not the exact zero-overlap evaluation host")


def campaign_arguments(invocation: Mapping[str, object], runtime: Mapping[str, object], *, checkout: str, output_root: str) -> list[str]:
    """Compose the literal campaign CLI and validate it with the real parser in tests."""
    return [
        "--matrix", f"{checkout}/{runtime['matrix_path']}", "--public-unseen-tops",
        "--policy-path", f"/workspace/checkpoints/{invocation['policy_revision']}/{runtime['policy_path']}",
        "--policy-revision-file", f"/workspace/checkpoints/{invocation['policy_revision']}/revision.txt",
        "--output-root", output_root, "--policy-repo", str(invocation["policy_repo"]),
        "--policy-step", str(invocation["policy_step"]), "--code-revision", str(invocation["code_revision"]),
        "--asset-revision", str(invocation["asset_revision"]),
        "--release-assets-root", f"{checkout}/{runtime['release_assets_root']}",
        "--simulator-version", str(invocation["simulator_version"]),
        "--policy-artifact-sha256", str(invocation["policy_artifact_sha256"]),
        "--image-identity", APPROVED_IMAGE_DIGEST, "--execution-mode", "policy_server", "--device", "cpu",
        "--policy-device", "cuda:0", "--workers", "4", "--groot-root", str(runtime["groot_root"]),
        "--groot-revision", str(invocation["groot_revision"]), "--groot-python", str(invocation["groot_python"]),
    ]


def stage_launch_sync_evaluation(manifest_path: Path, instance_receipt: Mapping[str, object], *, lifecycle_root: Path, runner: Callable[[tuple[str, ...]], object], code_bundle: Path, token_file: Path) -> dict[str, object]:
    """Stage a clean bundle, launch the fixed topology, then synchronize evidence."""
    manifest = read_lifecycle_manifest(manifest_path)
    invocation = manifest["invocation"]
    runtime = manifest["runtime"]
    instance_id = instance_receipt.get("instance_id")
    if (instance_receipt.get("kind") != "groot_checkpoint_evaluation_instance" or type(instance_id) is not int or instance_id <= 0
            or instance_receipt.get("invocation_sha256") != canonical_sha256(invocation)
            or not isinstance(instance_receipt.get("host"), str) or type(instance_receipt.get("port")) is not int
            or type(instance_receipt.get("lease_deadline_unix")) is not int
            or type(instance_receipt.get("watchdog_pid")) is not int or instance_receipt["watchdog_pid"] <= 0
            or not isinstance(instance_receipt.get("watchdog_receipt"), str)
            or not Path(str(instance_receipt["watchdog_receipt"])).is_absolute()
            or instance_receipt.get("rent_claim_path") != str(EVALUATION_RENT_CLAIM_ROOT / "active.json")):
        raise ValueError("instance receipt is not bound to the exact evaluation invocation")
    remote = f"root@{instance_receipt['host']}"; port = str(instance_receipt["port"])
    identity = canonical_sha256(invocation)[:16]
    remote_root = f"/workspace/checkpoint-evaluation/{identity}"
    output_remote = f"{remote_root}/evaluation"
    remaining_seconds = int(instance_receipt["lease_deadline_unix"]) - time.time_ns() // 1_000_000_000
    campaign_seconds = remaining_seconds - LEASE_WATCHDOG_RESERVE_SECONDS
    sync_attempted = False
    identity_file: str | None = None
    transport_options = (
        "-o", "IdentitiesOnly=yes", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
    )
    try:
        if campaign_seconds <= 0 or remaining_seconds > MAX_WALL_SECONDS:
            raise ValueError("instance lease deadline is invalid or exhausted")
        if code_bundle.is_symlink() or not code_bundle.is_file() or hashlib.sha256(code_bundle.read_bytes()).hexdigest() != manifest["code_bundle_sha256"]:
            raise ValueError("clean exact code bundle is unavailable")
        if token_file.is_symlink() or not token_file.is_file() or stat.S_IMODE(token_file.stat().st_mode) & 0o077:
            raise ValueError("secure publication token file is unavailable")
        _prelaunch_account(instance_receipt, runner)
        bundle_remote, token_remote = f"{remote_root}/code.bundle", f"{remote_root}/hf.token"
        identity_file = _ssh_identity()
        ssh = ("ssh", "-i", identity_file, *transport_options, "-p", port, remote)
        for command, label in ((ssh + ("mkdir", "-p", remote_root), "remote staging"),
                               (("scp", "-i", identity_file, *transport_options, "-P", port, str(code_bundle), f"{remote}:{bundle_remote}"), "code staging"),
                               (("scp", "-i", identity_file, *transport_options, "-P", port, str(token_file), f"{remote}:{token_remote}"), "token staging")):
            result = runner(command)
            if getattr(result, "returncode", 0) not in (0, None):
                raise RuntimeError(f"{label} failed")
        checkout = f"{remote_root}/checkout"
        policy_root = "/workspace/checkpoints/" + str(invocation["policy_revision"])
        policy_path = policy_root + "/" + str(runtime["policy_path"])
        corrective = _corrective_module()
        wire_target = remote_root + "/controller-wire"
        setup = [
            "test -x /opt/lehome-challenge/.venv/bin/python",
            *corrective._groot_wrapper_setup(),
            *corrective._controller_wire_setup(remote_root, checkout),
            corrective._controller_import_preflight(checkout, wire_target=wire_target),
            *corrective._asset_checkout_setup(checkout),
        ]
        arguments = " ".join(shlex.quote(value) for value in campaign_arguments(invocation, runtime, checkout=checkout, output_root=output_remote))
        command = (
            "set -eu; chmod 600 " + shlex.quote(token_remote) + "; "
            + "git clone --no-checkout " + shlex.quote(bundle_remote) + " " + shlex.quote(checkout) + "; "
            + "git -C " + shlex.quote(checkout) + " checkout --detach " + shlex.quote(str(invocation["code_revision"])) + "; "
            + "test \"$(git -C " + shlex.quote(checkout) + " rev-parse HEAD)\" = " + shlex.quote(str(invocation["code_revision"])) + "; "
            + "git -C " + shlex.quote(checkout) + " diff --quiet; "
            + "export HF_TOKEN=\"$(cat " + shlex.quote(token_remote) + ")\"; "
            + "; ".join(setup) + "; "
            + corrective._hf_download("/opt/lehome-challenge/.venv/bin/hf download " + shlex.quote(str(invocation["policy_repo"]))
            + " --revision " + shlex.quote(str(invocation["policy_revision"]))
            + " --include " + shlex.quote(str(runtime["policy_path"]) + "/*")
            + " --local-dir " + shlex.quote(policy_root)) + "; "
            + "printf '%s\\n' " + shlex.quote(str(invocation["policy_revision"])) + " > " + shlex.quote(policy_root + "/revision.txt") + "; "
            + "test -d " + shlex.quote(policy_path) + "; "
            + corrective._controller_pythonpath(checkout, wire_target=wire_target) + " /opt/lehome-challenge/.venv/bin/python -c "
            + shlex.quote("from pathlib import Path; from scripts.run_groot_flywheel_trial import policy_artifact_sha256; assert policy_artifact_sha256(Path(" + repr(policy_path) + ")) == " + repr(str(invocation["policy_artifact_sha256"]))) + "; "
            + "export CUDA_VISIBLE_DEVICES=0,1,2,3 LEHOME_POLICY_DEVICE_POOL=cuda:0,cuda:1,cuda:2,cuda:3; "
            + "test \"$LEHOME_POLICY_DEVICE_POOL\" = cuda:0,cuda:1,cuda:2,cuda:3; "
            + "timeout --signal=TERM --kill-after=20s " + str(campaign_seconds) + "s "
            + corrective._controller_pythonpath(checkout, wire_target=wire_target) + " /opt/lehome-challenge/.venv/bin/python -m scripts.run_groot_flywheel_campaign"
            + " " + arguments
        )
        launched = runner(ssh + ("sh", "-lc", command))
        sync_root = lifecycle_root / f"synced-{instance_id}"
        sync_attempted = True
        sync = runner(("scp", "-r", "-i", identity_file, *transport_options, "-P", port, f"{remote}:{output_remote}/.", str(sync_root)))
        if getattr(launched, "returncode", 0) not in (0, None) or getattr(sync, "returncode", 0) not in (0, None):
            raise RuntimeError("remote launch or evidence synchronization failed")
        return _write_new(lifecycle_root / f"launch-{instance_id}.json", {
            "schema_version": 1, "kind": "groot_checkpoint_evaluation_launch", "instance_id": instance_id,
            "invocation_sha256": canonical_sha256(invocation), "status": "synced", "sync_root": str(sync_root),
            "code_bundle_sha256": manifest["code_bundle_sha256"], "hard_wall_seconds": MAX_WALL_SECONDS,
            "total_dollar_ceiling_usd": MAX_TOTAL_DOLLARS,
        })
    except BaseException as error:
        if not sync_attempted:
            # A failed stage/preflight may still have created diagnostics.  The
            # copy is intentionally best-effort and never carries token bytes
            # into a receipt; disposal remains mandatory either way.
            sync_attempted = True
            if identity_file is not None:
                try:
                    runner(("scp", "-r", "-i", identity_file, *transport_options, "-P", port, f"{remote}:{output_remote}/.", str(lifecycle_root / f"synced-failure-{instance_id}")))
                except BaseException:
                    pass
        _cleanup_failure(instance_id, lifecycle_root, runner, "remote_launch_failure", sync_attempted=sync_attempted)
        _release_completed_rent_claim(instance_receipt)
        raise RuntimeError("remote launch failed; available evidence synchronized and instance disposed") from error


def _publication(path: Path) -> dict[str, object]:
    return _read_json(path, "publication receipt")


def destroy_after_publication(instance_id: int, publication_receipt: Path, instance_receipt: Path, *, disposal_receipt: Path, runner: Callable[[tuple[str, ...]], object], absence_polls: int = 60, sleep: Callable[[float], None] = time.sleep) -> dict[str, object]:
    """Destroy only after a private immutable, tree-listed, fresh-readback publication."""
    if not disposal_receipt.is_absolute() or disposal_receipt.exists() or disposal_receipt.is_symlink():
        raise ValueError("disposal receipt must be an absent absolute path")
    publication = _publication(publication_receipt)
    instance = _read_json(instance_receipt, "instance receipt")
    invocation = publication.get("invocation")
    expected_release_id = canonical_sha256({
        "invocation": invocation,
        "trial_ids": canonical_trial_ids(),
        "kind": "diagnostic_evaluation_not_rft",
    }) if isinstance(invocation, dict) else None
    expected_prefix = (
        f"evaluations/groot-n17-step-{invocation.get('policy_step')}/{expected_release_id}"
        if isinstance(invocation, dict) else None
    )
    if (
        publication.get("kind") != "groot_checkpoint_evaluation_publication"
        or publication.get("repository") != APPROVED_EVALUATION_REPOSITORY
        or publication.get("disposable") is not True or publication.get("repository_private") is not True
        or publication.get("tree_listing_verified") is not True or publication.get("fresh_readback_verified") is not True
        or publication.get("instance_id") != instance_id or instance.get("instance_id") != instance_id
        or instance.get("rent_claim_path") != str(EVALUATION_RENT_CLAIM_ROOT / "active.json")
        or publication.get("instance_receipt_sha256") != hashlib.sha256(instance_receipt.read_bytes()).hexdigest()
        or not isinstance(invocation, dict) or publication.get("invocation_sha256") != canonical_sha256(invocation)
        or instance.get("invocation_sha256") != canonical_sha256(invocation)
        or not isinstance(publication.get("immutable_revision"), str) or _COMMIT.fullmatch(str(publication["immutable_revision"])) is None
        or publication.get("release_id") != expected_release_id
        or publication.get("remote_prefix") != expected_prefix
    ):
        raise ValueError("publication receipt is not an exact immutable instance/invocation readback binding")
    try:
        destroy_issued, exact_empty, instances_empty, volumes_empty = _destroy_owned_once(
            instance_id, runner, polls=absence_polls, sleep=sleep,
        )
    except BaseException:
        _write_new(disposal_receipt, {
            "schema_version": 1, "kind": "groot_checkpoint_evaluation_disposal",
            "instance_id": instance_id, "destroyed_and_absent": False, "destroy_failed": True,
            "publication_receipt_sha256": hashlib.sha256(publication_receipt.read_bytes()).hexdigest(),
            "instance_receipt_sha256": hashlib.sha256(instance_receipt.read_bytes()).hexdigest(),
            "immutable_revision": publication["immutable_revision"], "remote_prefix": publication["remote_prefix"],
        })
        raise
    if not all((exact_empty, instances_empty, volumes_empty)):
        _write_new(disposal_receipt, {
            "schema_version": 1, "kind": "groot_checkpoint_evaluation_disposal",
            "instance_id": instance_id,
            "publication_receipt_sha256": hashlib.sha256(publication_receipt.read_bytes()).hexdigest(),
            "instance_receipt_sha256": hashlib.sha256(instance_receipt.read_bytes()).hexdigest(),
            "immutable_revision": publication["immutable_revision"], "remote_prefix": publication["remote_prefix"],
            "instance_absent": exact_empty, "account_instances_empty": instances_empty,
            "account_volumes_empty": volumes_empty, "destroy_issued": destroy_issued,
            "destroyed_and_absent": False,
            "absence_unverified": True,
        })
        raise ValueError("publication disposal did not empty the exact provider account")
    written = _write_new(disposal_receipt, {
        "schema_version": 1, "kind": "groot_checkpoint_evaluation_disposal",
        "instance_id": instance_id,
        "publication_receipt_sha256": hashlib.sha256(publication_receipt.read_bytes()).hexdigest(),
        "instance_receipt_sha256": hashlib.sha256(instance_receipt.read_bytes()).hexdigest(),
        "immutable_revision": publication["immutable_revision"], "remote_prefix": publication["remote_prefix"],
        "instance_absent": exact_empty, "account_instances_empty": instances_empty,
        "account_volumes_empty": volumes_empty, "destroy_issued": destroy_issued,
        "destroyed_and_absent": True,
    })
    _release_completed_rent_claim(instance)
    return written


def compare_checkpoint_invocations(one: Mapping[str, object], two: Mapping[str, object]) -> dict[str, tuple[object, object]]:
    """Return forbidden 1K/2K invocation differences; checkpoint fields may differ."""
    allowed = {"policy_step", "policy_artifact_sha256"}
    return {
        key: (one.get(key), two.get(key))
        for key in sorted(set(one) | set(two))
        if key not in allowed and one.get(key) != two.get(key)
    }


def _subprocess_runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    rent = actions.add_parser("rent"); rent.add_argument("--manifest", type=Path, required=True); rent.add_argument("--lifecycle-root", type=Path, required=True); rent.add_argument("--execute", action="store_true")
    launch = actions.add_parser("stage-launch-sync"); launch.add_argument("--manifest", type=Path, required=True); launch.add_argument("--instance-receipt", type=Path, required=True); launch.add_argument("--lifecycle-root", type=Path, required=True); launch.add_argument("--code-git-bundle", type=Path, required=True); launch.add_argument("--token-file", type=Path, required=True); launch.add_argument("--execute", action="store_true")
    destroy = actions.add_parser("destroy"); destroy.add_argument("--instance-id", type=int, required=True); destroy.add_argument("--publication-receipt", type=Path, required=True); destroy.add_argument("--instance-receipt", type=Path, required=True); destroy.add_argument("--disposal-receipt", type=Path, required=True); destroy.add_argument("--execute", action="store_true")
    watchdog = actions.add_parser("watchdog-destroy"); watchdog.add_argument("--instance-id", type=int, required=True); watchdog.add_argument("--deadline-unix", type=int, required=True); watchdog.add_argument("--receipt", type=Path, required=True); watchdog.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "dry_run", "action": args.action}, sort_keys=True))
        return 0
    if args.action == "rent": result = rent_evaluation(args.manifest, lifecycle_root=args.lifecycle_root, runner=_subprocess_runner)
    elif args.action == "stage-launch-sync": result = stage_launch_sync_evaluation(args.manifest, _read_json(args.instance_receipt, "instance receipt"), lifecycle_root=args.lifecycle_root, runner=_subprocess_runner, code_bundle=args.code_git_bundle, token_file=args.token_file)
    elif args.action == "destroy": result = destroy_after_publication(args.instance_id, args.publication_receipt, args.instance_receipt, disposal_receipt=args.disposal_receipt, runner=_subprocess_runner)
    else: result = enforce_lease_deadline(args.instance_id, args.deadline_unix, args.receipt, runner=_subprocess_runner)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
