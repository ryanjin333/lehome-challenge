"""Fail-closed lifecycle for one paid public-unseen top-40 checkpoint evaluation.

The functions in this module are deliberately explicit.  They do not discover,
reuse, or retain provider resources: a lifecycle owns one newly-created
on-demand host and every post-create failure performs the same evidence-sync
and disposal path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import stat
import shlex
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence

from lehome.flywheel.matrix import build_public_matrix, matrix_sha256


APPROVED_IMAGE_REPOSITORY = "ryanjin333/lehome-rollout"
# This is the immutable Docker Hub manifest used by the canonical 1K/2K runs.
APPROVED_IMAGE_DIGEST = "sha256:293c4f258f3742a7234699d706fb7088d0da8a764957bc79b244d830561abc12"
MAX_WALL_SECONDS = 4 * 60 * 60
MAX_TOTAL_DOLLARS = 3.00
MAX_PROJECTED_HOURLY_USD = 1.00  # strict: equality is rejected
OFFER_QUERY = "gpu_name=RTX_3090 num_gpus=4 reliability>=0.95 cpu_cores_effective>=64 cpu_ram>=128 disk_space>=300 duration>=4"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
VAST_SSH_IDENTITY = str(Path.home() / ".ssh" / "vast_quest")


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
        exact_absent = exact in (None, {}, []) or (isinstance(exact, dict) and exact.get("instances") is None)
        status = (exact_absent, instances == [], volumes == [])
        if all(status) or index + 1 == polls:
            return status
        sleep(2.0)
    return status


def _cleanup_failure(instance_id: int, lifecycle_root: Path, runner: Callable[[tuple[str, ...]], object], reason: str, *, sync_attempted: bool = False) -> None:
    """Destroy the owned instance once and preserve redacted disposal evidence."""
    runner(("vastai", "destroy", "instance", str(instance_id), "--yes"))
    exact_empty, instances_empty, volumes_empty = _verify_absent(instance_id, runner)
    receipt = {
        "schema_version": 1, "kind": "groot_checkpoint_evaluation_failure_disposal",
        "instance_id": instance_id, "reason": reason, "instance_absent": exact_empty,
        "account_instances_empty": instances_empty, "account_volumes_empty": volumes_empty,
        "synced_available_evidence": sync_attempted, "redacted": True,
    }
    _write_new(lifecycle_root / f"failure-{instance_id}.json", receipt)
    if not all((exact_empty, instances_empty, volumes_empty)):
        raise RuntimeError("failure disposal did not verify empty provider account")


def rent_evaluation(manifest_path: Path, *, lifecycle_root: Path, runner: Callable[[tuple[str, ...]], object], now_unix: int | None = None) -> dict[str, object]:
    """Rent exactly one compatible host after a validated local-only preflight."""
    manifest = read_lifecycle_manifest(manifest_path)  # must precede every provider call
    timestamp = time.time_ns() // 1_000_000_000 if now_unix is None else now_unix
    instances = _raw(runner, ("vastai", "--raw", "show", "instances"))
    volumes = _raw(runner, ("vastai", "--raw", "show", "volumes"))
    if not isinstance(instances, list) or not isinstance(volumes, list):
        raise ValueError("provider account preflight response is invalid")
    # Do not query offers, much less create a host, while any account resource
    # exists.  This makes the no-overlap check an actual provider-call gate.
    if instances:
        raise ValueError("evaluation lifecycle requires account-wide zero instances before rent")
    if volumes:
        raise ValueError("evaluation lifecycle requires account-wide zero volumes before rent")
    offers = _raw(runner, ("vastai", "--raw", "search", "offers", OFFER_QUERY, "--on-demand", "--storage", "300"))
    if not isinstance(offers, list):
        raise ValueError("provider account preflight response is invalid")
    evidence = capture_provider_evidence(offers=offers, instances=instances, volumes=volumes, output=lifecycle_root / f"provider-{timestamp}.json", now_unix=timestamp)
    _ssh_identity()  # fail before the irreversible provider create call
    created = _raw(runner, (
        "vastai", "--raw", "create", "instance", str(evidence["offer_id"]), "--image", str(manifest["rollout_image"]),
        "--env", f"-e LEHOME_FLYWHEEL_IMAGE_IDENTITY={APPROVED_IMAGE_DIGEST}", "--disk", "300", "--ssh", "--direct", "--cancel-unavail",
    ))
    instance_id = created.get("new_contract") if isinstance(created, dict) else None
    if type(instance_id) is not int or instance_id <= 0:
        raise ValueError("provider create response lacks instance ID")
    try:
        live = _wait_running(instance_id, runner)
    except BaseException:
        _cleanup_failure(instance_id, lifecycle_root, runner, "post_create_readback_failure")
        raise
    if _number(live.get("dph_total")) != evidence["instance_hourly_cost_usd"]:
        _cleanup_failure(instance_id, lifecycle_root, runner, "cost_readback_mismatch")
        raise ValueError("new instance readback cost differs from preflight offer")
    invocation = manifest["invocation"]
    lease_started = timestamp
    receipt = {
        "schema_version": 1, "kind": "groot_checkpoint_evaluation_instance", "instance_id": instance_id,
        "host": live["ssh_host"], "port": live.get("ssh_port", 22), "invocation_sha256": canonical_sha256(invocation),
        "provider_evidence_sha256": canonical_sha256(evidence), "provider_response_sha256": canonical_sha256(live),
        "lease_started_unix": lease_started, "lease_deadline_unix": lease_started + MAX_WALL_SECONDS,
        "hard_wall_seconds": MAX_WALL_SECONDS, "total_dollar_ceiling_usd": MAX_TOTAL_DOLLARS,
    }
    return _write_new(lifecycle_root / f"instance-{instance_id}.json", receipt)


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
            or type(instance_receipt.get("lease_deadline_unix")) is not int):
        raise ValueError("instance receipt is not bound to the exact evaluation invocation")
    remote = f"root@{instance_receipt['host']}"; port = str(instance_receipt["port"])
    identity = canonical_sha256(invocation)[:16]
    remote_root = f"/workspace/checkpoint-evaluation/{identity}"
    output_remote = f"{remote_root}/evaluation"
    remaining_seconds = int(instance_receipt["lease_deadline_unix"]) - time.time_ns() // 1_000_000_000
    sync_attempted = False
    try:
        if remaining_seconds <= 0 or remaining_seconds > MAX_WALL_SECONDS:
            raise ValueError("instance lease deadline is invalid or exhausted")
        if code_bundle.is_symlink() or not code_bundle.is_file() or hashlib.sha256(code_bundle.read_bytes()).hexdigest() != manifest["code_bundle_sha256"]:
            raise ValueError("clean exact code bundle is unavailable")
        if token_file.is_symlink() or not token_file.is_file() or stat.S_IMODE(token_file.stat().st_mode) & 0o077:
            raise ValueError("secure publication token file is unavailable")
        _prelaunch_account(instance_receipt, runner)
        bundle_remote, token_remote = f"{remote_root}/code.bundle", f"{remote_root}/hf.token"
        identity_file = _ssh_identity()
        ssh = ("ssh", "-i", identity_file, "-o", "IdentitiesOnly=yes", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-p", port, remote)
        for command, label in ((ssh + ("mkdir", "-p", remote_root), "remote staging"),
                               (("scp", "-i", identity_file, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", port, str(code_bundle), f"{remote}:{bundle_remote}"), "code staging"),
                               (("scp", "-i", identity_file, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", port, str(token_file), f"{remote}:{token_remote}"), "token staging")):
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
            + "timeout --signal=TERM --kill-after=20s " + str(remaining_seconds) + "s "
            + corrective._controller_pythonpath(checkout, wire_target=wire_target) + " /opt/lehome-challenge/.venv/bin/python -m scripts.run_groot_flywheel_campaign"
            + " --matrix " + shlex.quote(checkout + "/" + str(runtime["matrix_path"]))
            + " --public-unseen-tops --execution-mode policy_server --device cpu --workers 4"
            + " --policy-device cuda:0"
            + " --policy-repo " + shlex.quote(str(invocation["policy_repo"]))
            + " --policy-revision-file " + shlex.quote(policy_root + "/revision.txt")
            + " --policy-path " + shlex.quote(policy_path)
            + " --policy-step " + str(invocation["policy_step"])
            + " --policy-artifact-sha256 " + shlex.quote(str(invocation["policy_artifact_sha256"]))
            + " --code-revision " + shlex.quote(str(invocation["code_revision"]))
            + " --asset-revision " + shlex.quote(str(invocation["asset_revision"]))
            + " --release-assets-root " + shlex.quote(checkout + "/" + str(runtime["release_assets_root"]))
            + " --simulator-version " + shlex.quote(str(invocation["simulator_version"]))
            + " --image-identity " + shlex.quote(APPROVED_IMAGE_DIGEST)
            + " --groot-root " + shlex.quote(str(runtime["groot_root"]))
            + " --groot-revision " + shlex.quote(str(invocation["groot_revision"]))
            + " --groot-python " + shlex.quote(str(invocation["groot_python"]))
            + " --output-root " + shlex.quote(output_remote)
        )
        launched = runner(ssh + ("sh", "-lc", command))
        sync_root = lifecycle_root / f"synced-{instance_id}"
        sync_attempted = True
        sync = runner(("scp", "-r", "-i", identity_file, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", port, f"{remote}:{output_remote}/.", str(sync_root)))
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
            runner(("scp", "-r", "-i", _ssh_identity(), "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", port, f"{remote}:{output_remote}/.", str(lifecycle_root / f"synced-failure-{instance_id}")))
        _cleanup_failure(instance_id, lifecycle_root, runner, "remote_launch_failure", sync_attempted=sync_attempted)
        raise RuntimeError("remote launch failed; available evidence synchronized and instance disposed") from error


def _publication(path: Path) -> dict[str, object]:
    return _read_json(path, "publication receipt")


def destroy_after_publication(instance_id: int, publication_receipt: Path, instance_receipt: Path, *, runner: Callable[[tuple[str, ...]], object]) -> bool:
    """Destroy only after a private immutable, tree-listed, fresh-readback publication."""
    publication = _publication(publication_receipt)
    instance = _read_json(instance_receipt, "instance receipt")
    invocation = publication.get("invocation")
    if (
        publication.get("kind") != "groot_checkpoint_evaluation_publication"
        or publication.get("disposable") is not True or publication.get("repository_private") is not True
        or publication.get("tree_listing_verified") is not True or publication.get("fresh_readback_verified") is not True
        or publication.get("instance_id") != instance_id or instance.get("instance_id") != instance_id
        or publication.get("instance_receipt_sha256") != hashlib.sha256(instance_receipt.read_bytes()).hexdigest()
        or not isinstance(invocation, dict) or publication.get("invocation_sha256") != canonical_sha256(invocation)
        or instance.get("invocation_sha256") != canonical_sha256(invocation)
        or not isinstance(publication.get("immutable_revision"), str) or _COMMIT.fullmatch(str(publication["immutable_revision"])) is None
        or not isinstance(publication.get("remote_prefix"), str) or not publication["remote_prefix"]
    ):
        raise ValueError("publication receipt is not an exact immutable instance/invocation readback binding")
    runner(("vastai", "destroy", "instance", str(instance_id), "--yes"))
    exact_empty, instances_empty, volumes_empty = _verify_absent(instance_id, runner)
    if not all((exact_empty, instances_empty, volumes_empty)):
        raise ValueError("publication disposal did not empty the exact provider account")
    return True


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
    destroy = actions.add_parser("destroy"); destroy.add_argument("--instance-id", type=int, required=True); destroy.add_argument("--publication-receipt", type=Path, required=True); destroy.add_argument("--instance-receipt", type=Path, required=True); destroy.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "dry_run", "action": args.action}, sort_keys=True))
        return 0
    if args.action == "rent": result = rent_evaluation(args.manifest, lifecycle_root=args.lifecycle_root, runner=_subprocess_runner)
    elif args.action == "stage-launch-sync": result = stage_launch_sync_evaluation(args.manifest, _read_json(args.instance_receipt, "instance receipt"), lifecycle_root=args.lifecycle_root, runner=_subprocess_runner, code_bundle=args.code_git_bundle, token_file=args.token_file)
    else: result = {"destroyed": destroy_after_publication(args.instance_id, args.publication_receipt, args.instance_receipt, runner=_subprocess_runner)}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
