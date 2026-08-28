#!/usr/bin/env python3
"""Local-only terminal finalizer for the one approved simple-curriculum VM.

The paid controller only writes a compact handoff.  This separate process owns
the provider STOPPED observation and the subsequent zero-compute publication.
It has no create/start/delete operation.
"""
from __future__ import annotations

from hashlib import sha256
import argparse
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Protocol


EXACT_INSTANCE_ID = "computeinstance-u00t6xfqhadrcmssa2"
EXACT_INSTANCE_NAME = "lehome-rollout"
PROTECTED_DISK_ID = "computedisk-u00pbe55crxy7jr56x"


class FinalizationError(RuntimeError):
    pass


class Provider(Protocol):
    def get(self, instance_id: str) -> Mapping[str, object]: ...
    def stop(self, instance_id: str) -> None: ...


class Publisher(Protocol):
    def publish(self, root: Path, *, handoff: Mapping[str, object], stop_observation: Mapping[str, object], seal: Mapping[str, object]) -> Mapping[str, object]: ...


def _checked_absolute(value: str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts) or path.is_symlink():
        raise FinalizationError(f"{label} is unsafe")
    return path


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def validate_handoff(raw: Mapping[str, object]) -> dict[str, object]:
    payload = dict(raw); declared = payload.pop("handoff_sha256", None)
    required = {"schema_version", "kind", "run_id", "round_id", "instance_id", "terminal_outcome", "predecessor_receipt_sha256", "code_revision", "code_tree_sha256", "runtime_identity", "runtime_identity_sha256", "first_100_receipt_sha256", "evidence", "handoff_sha256"}
    if set(raw) != required or declared != _digest(payload):
        raise FinalizationError("operator handoff is malformed")
    if payload["schema_version"] != 1 or payload["kind"] != "lehome_simple_curriculum_operator_stop_handoff_v1":
        raise FinalizationError("operator handoff kind is invalid")
    if payload["instance_id"] != EXACT_INSTANCE_ID or not isinstance(payload["run_id"], str) or not isinstance(payload["round_id"], str):
        raise FinalizationError("operator handoff is not bound to the approved VM")
    if not _hex(payload["code_revision"], 40) or not _hex(payload["code_tree_sha256"], 64):
        raise FinalizationError("operator handoff code identity is invalid")
    if not isinstance(payload["runtime_identity"], Mapping) or payload["runtime_identity_sha256"] != _digest(dict(payload["runtime_identity"])):
        raise FinalizationError("operator handoff runtime identity is invalid")
    if not isinstance(payload["evidence"], list):
        raise FinalizationError("operator handoff evidence is invalid")
    for item in payload["evidence"]:
        if not isinstance(item, Mapping) or set(item) != {"stage", "receipt_sha256", "file_sha256"} or not isinstance(item["stage"], str) or not _hex(item["receipt_sha256"], 64) or not _hex(item["file_sha256"], 64):
            raise FinalizationError("operator handoff evidence is invalid")
    return dict(raw)


def _validate_instance(raw: Mapping[str, object]) -> dict[str, object]:
    value = dict(raw)
    disks = value.get("disks")
    if value.get("id") != EXACT_INSTANCE_ID or value.get("name") != EXACT_INSTANCE_NAME or not isinstance(disks, list) or PROTECTED_DISK_ID not in disks:
        raise FinalizationError("provider response is not the exact protected rollout VM")
    if value.get("state") not in {"RUNNING", "STOPPED"}:
        raise FinalizationError("provider response has an unsafe VM state")
    return value


def stop_exact_instance(provider: Provider, *, timeout_seconds: float) -> dict[str, object]:
    if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 600:
        raise FinalizationError("stop timeout is invalid")
    first = _validate_instance(provider.get(EXACT_INSTANCE_ID))
    if first["state"] == "RUNNING":
        provider.stop(EXACT_INSTANCE_ID)
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        observed = _validate_instance(provider.get(EXACT_INSTANCE_ID))
        if observed["state"] == "STOPPED":
            body = {"schema_version": 1, "kind": "lehome_simple_curriculum_stopped_observation_v1", "instance_id": EXACT_INSTANCE_ID, "instance_name": EXACT_INSTANCE_NAME, "protected_disk_id": PROTECTED_DISK_ID, "state": "STOPPED", "provider_response_sha256": _digest(observed)}
            return {**body, "observation_sha256": _digest(body)}
        if time.monotonic() >= deadline:
            raise FinalizationError("provider did not report STOPPED before timeout")
        time.sleep(0.25)


def _seal(handoff: Mapping[str, object], stopped: Mapping[str, object]) -> dict[str, object]:
    outcome = handoff["terminal_outcome"]
    # The remote handoff must have a separately validated exact-cap report;
    # without it terminal claims remain intentionally non-complete.
    complete = outcome == "complete" and any(item.get("stage") == "success-replay" for item in handoff["evidence"])
    body = {"schema_version": 1, "kind": "lehome_simple_curriculum_final_seal_v1", "run_id": handoff["run_id"], "round_id": handoff["round_id"], "terminal_outcome": outcome, "completion_claim": "caps_unverified" if complete else "not_complete", "handoff_sha256": handoff["handoff_sha256"], "stopped_observation_sha256": stopped["observation_sha256"]}
    return {**body, "seal_sha256": _digest(body)}


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FinalizationError("finalization staging path is unsafe")
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(_canonical(value)); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


class SubprocessNebiusProvider:
    """Restricted exact-instance adapter; it cannot create, start, or delete."""
    def __init__(self, command: tuple[str, ...] = ("nebius",)) -> None: self.command = command
    def _json(self, arguments: tuple[str, ...]) -> Mapping[str, object]:
        result = subprocess.run(self.command + arguments, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode:
            raise FinalizationError("Nebius exact-instance command failed")
        try: value = json.loads(result.stdout)
        except json.JSONDecodeError as error: raise FinalizationError("Nebius exact-instance response is not JSON") from error
        if not isinstance(value, Mapping): raise FinalizationError("Nebius exact-instance response is invalid")
        return value
    def get(self, instance_id: str) -> Mapping[str, object]:
        if instance_id != EXACT_INSTANCE_ID: raise FinalizationError("refusing non-approved instance")
        return self._json(("compute", "instance", "get", instance_id, "--format", "json"))
    def stop(self, instance_id: str) -> None:
        if instance_id != EXACT_INSTANCE_ID: raise FinalizationError("refusing non-approved instance")
        self._json(("compute", "instance", "stop", instance_id, "--format", "json"))


def fetch_remote_handoff(*, ssh_target: str, port: int, campaign_root: str, destination: Path) -> dict[str, object]:
    """Fetch only the compact handoff over noninteractive SSH into temp storage."""
    root = _checked_absolute(campaign_root, label="remote campaign root")
    if not isinstance(port, int) or not 1 <= port <= 65535 or not ssh_target or any(c.isspace() for c in ssh_target):
        raise FinalizationError("SSH target or port is invalid")
    remote = str(root / "reports" / "operator-stop-handoff.json")
    command = ("ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-p", str(port), ssh_target, "cat", "--", remote)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode: raise FinalizationError("remote handoff fetch failed")
    if destination.exists() or destination.is_symlink(): raise FinalizationError("handoff destination is unsafe")
    descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle: handle.write(result.stdout); handle.flush(); os.fsync(handle.fileno())
    try: value = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error: raise FinalizationError("remote handoff is not JSON") from error
    if not isinstance(value, dict): raise FinalizationError("remote handoff is invalid")
    return value


class HfFinalizerPublisher:
    """Compact two-phase public finalization using the reviewed transport."""
    repository = "ryanjin333/lehome-groot-n17-rollouts"
    def __init__(self, token_path: Path) -> None: self.token_path = token_path
    def publish(self, root: Path, *, handoff: Mapping[str, object], stop_observation: Mapping[str, object], seal: Mapping[str, object]) -> Mapping[str, object]:
        import importlib.util
        source = Path(__file__).with_name("publish_simple_curriculum_collection.py")
        spec = importlib.util.spec_from_file_location("simple_curriculum_publisher", source)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
        token = module._load_token(self.token_path)
        _atomic_json(root / "reports" / "operator-stop-handoff.json", handoff)
        _atomic_json(root / "reports" / "stopped-observation.json", stop_observation)
        _atomic_json(root / "seals" / "final-seal.json", seal)
        files = ("reports/operator-stop-handoff.json", "reports/stopped-observation.json", "seals/final-seal.json")
        bundle = module.CollectionPublicationBundle(root=root, run_id=handoff["run_id"], repository=self.repository, revision="main", files=files)
        transport = module.HuggingFacePublicDatasetTransport()
        evidence = module.publish_collection_bundle(bundle, token=token, transport=transport)
        receipt = {"schema_version": 1, "kind": "lehome_simple_curriculum_operator_finalization_receipt_v1", "run_id": handoff["run_id"], "round_id": handoff["round_id"], "evidence_revision": evidence.immutable_revision, "evidence_bundle_sha256": evidence.bundle_sha256, "final_seal_sha256": seal["seal_sha256"], "readback_verified": True, "public_readback_verified": True}
        _atomic_json(root / "reports" / "final-publication.json", receipt)
        entry = module._collect_entries(module.CollectionPublicationBundle(root=root, run_id=handoff["run_id"], repository=self.repository, revision="main", files=("reports/final-publication.json",)))[0]
        head = transport.resolve_approved_ref(repository=self.repository, ref="main", token=token)
        revision = transport.upload_files(repository=self.repository, revision="main", source=root, entries=(entry,), token=token, remote_prefix=evidence.remote_prefix, parent_commit=head)
        all_entries = tuple(evidence.entries) + (entry,)
        for auth in (token, None): module._verify_download(transport=transport, bundle=bundle, revision=revision, prefix=evidence.remote_prefix, entries=all_entries, token=auth)
        return {"immutable_revision": revision, "readback_verified": True, "public_readback_verified": True}


def finalize_operator_handoff(handoff: Mapping[str, object], *, provider: Provider, publisher: Publisher, staging_parent: Path, stop_timeout_seconds: float) -> dict[str, object]:
    """Stop in a safety boundary, then publish only if all prior checks pass."""
    validation_error: Exception | None = None
    valid: dict[str, object] | None = None
    try:
        valid = validate_handoff(handoff)
    except Exception as error:  # stop still happens below
        validation_error = error
    try:
        stopped = stop_exact_instance(provider, timeout_seconds=stop_timeout_seconds)
    except Exception as error:
        raise FinalizationError("infrastructure_stop_failure") from error
    if validation_error is not None:
        raise FinalizationError("infrastructure_stop_failure") from validation_error
    assert valid is not None
    seal = _seal(valid, stopped)
    with tempfile.TemporaryDirectory(prefix="lehome-finalizer-", dir=staging_parent) as temporary:
        receipt = publisher.publish(Path(temporary), handoff=valid, stop_observation=stopped, seal=seal)
    if receipt.get("readback_verified") is not True or receipt.get("public_readback_verified") is not True:
        raise FinalizationError("publication readback failed after STOPPED")
    return {"result": "finalized", "stop_observation": stopped, "seal": seal, "publication": dict(receipt)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-target", required=True); parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument("--remote-campaign-root", required=True); parser.add_argument("--run-id", required=True); parser.add_argument("--round-id", required=True); parser.add_argument("--hf-token-file", type=Path, required=True)
    parser.add_argument("--stop-timeout-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    provider = SubprocessNebiusProvider()
    try:
        with tempfile.TemporaryDirectory(prefix="lehome-finalizer-fetch-") as temporary:
            handoff = fetch_remote_handoff(ssh_target=args.ssh_target, port=args.ssh_port, campaign_root=args.remote_campaign_root, destination=Path(temporary) / "handoff.json")
            if handoff.get("run_id") != args.run_id or handoff.get("round_id") != args.round_id:
                raise FinalizationError("remote handoff does not match explicit invocation IDs")
            result = finalize_operator_handoff(handoff, provider=provider, publisher=HfFinalizerPublisher(args.hf_token_file), staging_parent=Path(temporary), stop_timeout_seconds=args.stop_timeout_seconds)
        print(json.dumps({"result": result["result"], "immutable_revision": result["publication"]["immutable_revision"]}, sort_keys=True))
        return 0
    except FinalizationError as error:
        # Handoff fetch/validation is inside the safety boundary too.  The
        # remote process may be dead or its disk evidence corrupt, but it is
        # never a reason to leave this exact billed VM running.
        try:
            stop_exact_instance(provider, timeout_seconds=args.stop_timeout_seconds)
        except FinalizationError:
            pass
        print(json.dumps({"result": "infrastructure_stop_failure", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__": raise SystemExit(main())
