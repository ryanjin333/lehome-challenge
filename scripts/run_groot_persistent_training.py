#!/usr/bin/env python3
"""Explicit, resume-safe Vast lifecycle for the 2K persistent RFT run.

Every action is a dry-run unless ``--execute`` is supplied.  The command runner
is injected in tests; normal execution uses the Vast CLI and SSH only after the
caller explicitly crosses that boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time
from typing import Callable, Mapping

from lehome_train.hub import HubTransport, HuggingFaceHubTransport
from lehome_train.io import atomic_write_json, sha256_file
from lehome_train.release_manifest import validate_training_capability

ORGANIZER_SOURCE = {"repository": "lehome/dataset_challenge_merged", "revision": "17e8dee8fac294ffd21d250501d3b31bf8679042", "subdir": "four_types_merged", "mirror_repository": "kunhsiang/lehome-four-types-merged", "mirror_revision": "2ebcccf528dec91cefac0c94a9214a83028ae6cc", "manifest_sha256": "bf8fbae82002a33ff304b9a70993bdfe1c678ba9e8f798c1ad370d58969435eb"}
CORRECTIVE_SOURCE = {"repository": "ryanjin333/lehome-groot-n17-data", "revision": "e6cd1c182514c15271c805d03a646e7a4f95b17c", "prefix": "corrective-rft/b96be3db22174a12dab62a8a673f7c7d083f87aa7b50c4e03ee43e064da56c35"}
PARENT_CHECKPOINT = {"repository": "ryanjin333/lehome-groot-n17-models", "revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3", "subpath": "policies/step-12000", "archive_sha256": "0ddd4e7ce351dd2172cd1edd967293a50d02c15c0f2c21ca39db94692a57e0b5", "artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06"}
# Vast's raw expression grammar does not support a portable OR form for two
# exact SKU strings.  Query only stable numeric facts, then enforce the narrow
# WS/S allowlist on raw rows in ``_offer_gpu``.
OFFER_QUERY = "gpu_ram>=96000 num_gpus=1 reliability>=0.95"
_DIGEST_PREFIX = "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:"
BOOTSTRAP_TRAINER_IMAGE = (
    _DIGEST_PREFIX
    + "b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746"
)
MAX_ACCOUNT_HOURLY_USD = 1.00
Runner = Callable[[tuple[str, ...]], str]


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _load(path: str) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError("lifecycle request must be an object")
    return value


def _read_private_token(path_value: str | None) -> str:
    if not path_value:
        raise ValueError("destroy requires --token-file")
    path = Path(path_value)
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("token file must be a private regular file")
    try:
        token = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        raise ValueError("token file is unreadable") from None
    if not token or any(character.isspace() for character in token):
        raise ValueError("token file is invalid")
    return token


def _verify_prepare_evidence(receipt: Mapping[str, object]) -> None:
    """Bind free preparation to local source/release evidence, not strings."""
    organizer = receipt.get("organizer_source")
    corrective = receipt.get("corrective_source")
    if not isinstance(organizer, Mapping) or any(
        organizer.get(key) != value for key, value in ORGANIZER_SOURCE.items()
    ):
        raise ValueError("prepare organizer evidence is not the pinned repository/subdir/manifest")
    if not isinstance(corrective, Mapping) or any(
        corrective.get(key) != value for key, value in CORRECTIVE_SOURCE.items()
    ):
        raise ValueError("prepare corrective release evidence is not the pinned repository/revision/prefix")
    release_id = corrective.get("release_id")
    if not isinstance(release_id, str) or re.fullmatch(r"[0-9a-f]{64}", release_id) is None:
        raise ValueError("prepare corrective release evidence lacks an immutable local release ID")


def _verified_corrective_release_evidence(
    roots: list[Path], release_receipt_path: object,
) -> dict[str, object]:
    """Bind local RFT inputs to a prior full-release tree readback receipt.

    The published HF identity and the snapshot's internal provenance deliberately
    differ.  Both must be present in the locally authenticated release receipt;
    callers cannot supply loose revision strings in a materialize request.
    """
    if not isinstance(release_receipt_path, str):
        raise ValueError("materialize requires a verified corrective release receipt path")
    receipt = _load_regular_json(Path(release_receipt_path), "corrective release receipt")
    published = receipt.get("published_release")
    local = receipt.get("local_snapshot")
    if not isinstance(published, Mapping) or not isinstance(local, Mapping):
        raise ValueError("corrective release receipt lacks published and local identities")
    evidence = dict(published)
    _verify_prepare_evidence({"organizer_source": ORGANIZER_SOURCE, "corrective_source": evidence})
    internal_revision, internal_release_id = local.get("source_revision"), local.get("source_release_id")
    if not isinstance(internal_revision, str) or re.fullmatch(r"[0-9a-f]{40}", internal_revision) is None or not isinstance(internal_release_id, str) or re.fullmatch(r"[0-9a-f]{64}", internal_release_id) is None:
        raise ValueError("corrective release receipt local provenance is invalid")
    expected_trees = local.get("trees")
    if not isinstance(expected_trees, Mapping) or set(expected_trees) != {str(root) for root in roots}:
        raise ValueError("corrective release receipt does not bind the local snapshot roots")
    for root in roots:
        manifest = _load_regular_json(root / "manifest.json", "corrective snapshot manifest")
        if manifest.get("source_revision") != internal_revision or manifest.get("source_release_id") != internal_release_id:
            raise ValueError("corrective snapshot provenance differs from verified release receipt")
        if expected_trees.get(str(root)) != _tree_readback_sha256(root):
            raise ValueError("corrective snapshot tree differs from verified release receipt")
    return {
        **evidence,
        "local_source_revision": internal_revision,
        "local_source_release_id": internal_release_id,
        "local_tree_sha256": dict(expected_trees),
    }


def derive_corrective_receipt(
    *, disposal_receipt: Path, snapshot_root: Path, output: Path,
) -> dict[str, object]:
    """Derive the local corrective binding from immutable disposal evidence.

    This is intentionally free and local: no Hub client is constructed.  The
    disposal receipt already records the authenticated private immutable tree;
    this action only ties that published identity to the local snapshot's own
    internal provenance and a complete safe tree digest.
    """
    disposal = _load_regular_json(disposal_receipt, "corrective disposal receipt")
    release_id = CORRECTIVE_SOURCE["prefix"].rpartition("/")[2]
    required = {
        "schema_version": 1,
        "disposable": True,
        "repository": CORRECTIVE_SOURCE["repository"],
        "immutable_revision": CORRECTIVE_SOURCE["revision"],
        "remote_prefix": CORRECTIVE_SOURCE["prefix"],
        "release_id": release_id,
        "fresh_readback_verified": True,
        "tree_listing_verified": True,
    }
    if any(disposal.get(key) != value for key, value in required.items()):
        raise ValueError("corrective disposal receipt is not the approved private immutable release")
    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
        raise ValueError("corrective snapshot root is unavailable")
    manifest = _load_regular_json(snapshot_root / "manifest.json", "corrective snapshot manifest")
    internal_revision, internal_release_id = manifest.get("source_revision"), manifest.get("source_release_id")
    if (
        manifest.get("source_format") != "verified_flywheel_rft_release"
        or manifest.get("source_repository") != CORRECTIVE_SOURCE["repository"]
        or not isinstance(internal_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", internal_revision) is None
        or not isinstance(internal_release_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", internal_release_id) is None
    ):
        raise ValueError("corrective snapshot local provenance is invalid")
    if output.exists() or output.is_symlink():
        raise ValueError("corrective release receipt output must not already exist")
    payload = {
        "schema_version": 1,
        "published_release": {
            **CORRECTIVE_SOURCE,
            "release_id": release_id,
            "disposal_receipt_sha256": hashlib.sha256(disposal_receipt.read_bytes()).hexdigest(),
        },
        "local_snapshot": {
            "source_revision": internal_revision,
            "source_release_id": internal_release_id,
            "trees": {str(snapshot_root): _tree_readback_sha256(snapshot_root)},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError("corrective release receipt output parent is unsafe")
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        if temporary.is_symlink() or not temporary.is_file():
            raise ValueError("corrective release receipt output is unsafe")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _run(command: tuple[str, ...]) -> str:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return completed.stdout


def _bounded_sleep(seconds: float) -> None:
    time.sleep(seconds)


def _json(runner: Runner, command: tuple[str, ...]) -> object:
    try: return json.loads(runner(command))
    except json.JSONDecodeError as error: raise ValueError("provider response is invalid JSON") from error


def _offer_gpu(row: Mapping[str, object]) -> bool:
    return row.get("gpu_name") in {"RTX PRO 6000 WS", "RTX PRO 6000 S"} and float(row.get("gpu_ram", 0)) >= 96000


def _project(row: Mapping[str, object], fields: tuple[str, ...]) -> dict[str, object]:
    return {field: row[field] for field in fields if field in row}


def _trainer_image(value: object) -> str:
    if not isinstance(value, str) or not value.startswith(_DIGEST_PREFIX) or len(value) != len(_DIGEST_PREFIX) + 64 or any(char not in "0123456789abcdef" for char in value[-64:]):
        raise ValueError("request must provide an accepted trainer OCI digest")
    return value


def _tree_readback_sha256(root: Path) -> str:
    """Match the remote sorted ``sha256sum`` tree receipt byte-for-byte."""
    rows: list[bytes] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("sealed generation contains a symlink")
        if path.is_file():
            rows.append(hashlib.sha256(path.read_bytes()).hexdigest().encode() + b"  ./" + path.relative_to(root).as_posix().encode() + b"\n")
    return hashlib.sha256(b"".join(rows)).hexdigest()


_STABLE_INSTANCE_FIELDS = (
    "id", "actual_status", "gpu_name", "gpu_ram", "num_gpus", "dph_total",
    "ssh_host", "ssh_port", "driver_version",
)


def _stable_instance_identity(row: Mapping[str, object]) -> str:
    """Hash only contract facts Vast does not mutate with incidental metadata."""
    return _hash(_project(row, _STABLE_INSTANCE_FIELDS))


def _require_account_cap(total: object, *, label: str) -> float:
    if type(total) not in (int, float) or not math.isfinite(float(total)) or float(total) < 0:
        raise ValueError(f"{label} account-wide hourly total is invalid")
    if float(total) >= MAX_ACCOUNT_HOURLY_USD:
        raise ValueError("account-wide instance and storage total exceeds $1/hr")
    return float(total)


def _live_account_total(*, runner: Runner) -> float:
    """Read all charged instance/volume rows once, without duplicating disk."""
    instances = _json(runner, ("vastai", "--raw", "show", "instances"))
    volumes = _json(runner, ("vastai", "--raw", "show", "volumes"))
    if not isinstance(instances, list) or not isinstance(volumes, list):
        raise ValueError("provider account listing is invalid")
    return _require_account_cap(
        sum(float(row.get("dph_total", 0)) for row in instances if isinstance(row, Mapping))
        + sum(float(row.get("storage_total_cost", 0)) for row in volumes if isinstance(row, Mapping)),
        label="live provider",
    )


def capture_offers(*, runner: Runner, now_unix: int | None = None, ttl_seconds: int = 300) -> dict[str, object]:
    # Vast produces the total hourly quote only after the requested disk size is
    # supplied.  ``dph_total`` below is consequently the single all-in 300GB
    # quote; a separately reported storage component is evidence, not a second
    # account charge.
    offers = _json(runner, ("vastai", "--raw", "search", "offers", OFFER_QUERY, "--interruptible", "--storage", "300"))
    instances = _json(runner, ("vastai", "--raw", "show", "instances"))
    volumes = _json(runner, ("vastai", "--raw", "show", "volumes"))
    if not all(isinstance(value, list) for value in (offers, instances, volumes)): raise ValueError("provider listing is invalid")
    eligible = [row for row in offers if isinstance(row, Mapping) and _offer_gpu(row) and row.get("num_gpus") == 1 and float(row.get("dph_total", 99)) < 1]
    if not eligible: raise ValueError("no interruptible RTX PRO 6000 96GB offer under $1/hr")
    offer = min(eligible, key=lambda row: float(row["dph_total"]))
    storage_unit_cost = offer.get("storage_cost", offer.get("storage_cost_per_gb"))
    if storage_unit_cost is not None and (type(storage_unit_cost) not in (int, float) or float(storage_unit_cost) < 0):
        raise ValueError("offer storage quote is invalid")
    base_hourly = offer.get("dph_base")
    requested_storage_hourly: float | None
    if type(base_hourly) in (int, float):
        requested_storage_hourly = max(0.0, float(offer["dph_total"]) - float(base_hourly))
    elif type(storage_unit_cost) in (int, float):
        # Retain an explicit breakdown if Vast supplies it, while accounting
        # from the all-in --storage quote only once.
        requested_storage_hourly = float(storage_unit_cost) * 300
    else:
        requested_storage_hourly = None
    existing_instance_total = sum(float(row.get("dph_total", 0)) for row in instances if isinstance(row, Mapping))
    existing_storage_total = sum(float(row.get("storage_total_cost", 0)) for row in volumes if isinstance(row, Mapping))
    total = existing_instance_total + existing_storage_total + float(offer["dph_total"])
    _require_account_cap(total, label="captured offer")
    captured = int(time.time()) if now_unix is None else now_unix
    safe_offer = _project(offer, ("id", "gpu_name", "gpu_ram", "num_gpus", "dph_total", "dph_base", "storage_cost", "storage_cost_per_gb", "min_bid", "driver_version", "is_bid", "image"))
    return {"schema_version": 1, "kind": "persistent_training_offer", "offer": safe_offer, "account_hourly_total_usd": total, "existing_instance_hourly_total_usd": existing_instance_total, "existing_storage_hourly_total_usd": existing_storage_total, "requested_storage_gb": 300, "requested_storage_hourly_usd": requested_storage_hourly, "storage_quote_included_in_dph_total": True, "captured_at_unix": captured, "expires_at_unix": captured + ttl_seconds, "search_mode": "interruptible"}


def rent(*, evidence: Mapping[str, object], runner: Runner, max_readiness_polls: int = 12, sleep: Callable[[float], None] = _bounded_sleep, require_capability: bool = True) -> dict[str, object]:
    offer = evidence.get("offer")
    if not isinstance(offer, Mapping) or type(offer.get("id")) is not int: raise ValueError("offer evidence is invalid")
    if evidence.get("search_mode") != "interruptible" or type(evidence.get("expires_at_unix")) is not int or evidence["expires_at_unix"] < int(time.time()): raise ValueError("offer evidence is expired or not interruptible")
    image = _trainer_image(evidence.get("trainer_image"))
    capability = evidence.get("training_capability")
    image_digest = image.rpartition("@")[2]
    if require_capability:
        if not isinstance(capability, Mapping) or capability.get("image_digest") != image_digest or not isinstance(capability.get("optimizer_step"), Mapping) or capability["optimizer_step"].get("passed") is not True or not isinstance(capability.get("nvml"), Mapping):
            raise ValueError("rent requires a matching accepted training capability receipt")
        validate_training_capability(capability)
    elif image != BOOTSTRAP_TRAINER_IMAGE:
        raise ValueError("bootstrap canary requires the historical structurally pinned trainer image")
    _require_account_cap(evidence.get("account_hourly_total_usd"), label="offer evidence")
    quoted_offer_hourly = offer.get("dph_total")
    if type(quoted_offer_hourly) not in (int, float) or float(quoted_offer_hourly) < 0:
        raise ValueError("offer evidence lacks the all-in 300GB hourly quote")
    # The capture receipt is short-lived but pre-rental account state can still
    # change.  Re-read retained charges immediately before creating the lease.
    _require_account_cap(
        _live_account_total(runner=runner) + float(quoted_offer_hourly),
        label="fresh rental projection",
    )
    bid = offer.get("min_bid", offer.get("dph_total"))
    if type(bid) not in (int, float) or float(bid) >= 1: raise ValueError("offer bid price is invalid")
    created = _json(runner, ("vastai", "--raw", "create", "instance", str(offer["id"]), "--image", image, "--disk", "300", "--bid_price", str(bid), "--ssh", "--direct", "--cancel-unavail", "--env", "-e LEHOME_TRAIN_IMAGE=" + image))
    if not isinstance(created, Mapping) or type(created.get("new_contract")) is not int: raise ValueError("provider did not return an instance ID")
    instance_id = created["new_contract"]
    live: object = {}
    for _ in range(max_readiness_polls):
        live = _json(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
        if isinstance(live, Mapping) and live.get("actual_status", "running") == "running" and live.get("ssh_host"):
            break
        sleep(5.0)
    else:
        runner(("vastai", "destroy", "instance", str(instance_id), "--yes"))
        absent = _json(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
        if absent not in ({}, None): raise ValueError("post-create cleanup absence readback failed")
        raise ValueError("instance readiness poll timed out")
    if not isinstance(live, Mapping) or live.get("id") != instance_id or not _offer_gpu(live) or live.get("num_gpus") != 1 or not live.get("ssh_host") or type(live.get("ssh_port")) is not int or float(live.get("dph_total", 99)) >= 1:
        runner(("vastai", "destroy", "instance", str(instance_id), "--yes"))
        absent = _json(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
        if absent not in ({}, None):
            raise ValueError("post-create cleanup absence readback failed")
        raise ValueError("instance readback does not match accepted offer")
    return {"schema_version": 1, "kind": "persistent_training_instance", "instance_id": instance_id, "host": live.get("ssh_host"), "port": live.get("ssh_port"), "trainer_image": image, "offer_evidence_sha256": _hash(evidence), "provider_response_sha256": _stable_instance_identity(live), "account_hourly_total_usd": _require_account_cap(evidence.get("account_hourly_total_usd"), label="offer evidence")}


def bootstrap_canary(*, evidence: Mapping[str, object], runner: Runner) -> dict[str, object]:
    """Rent the one historical image only long enough to prove its capability.

    Full 2K actions cannot call this path: they consume the resulting canonical
    receipt through ``rent`` with ``require_capability=True``.
    """
    if evidence.get("trainer_image") != BOOTSTRAP_TRAINER_IMAGE:
        raise ValueError("bootstrap canary requires the historical structurally pinned trainer image")
    instance = rent(evidence=evidence, runner=runner, require_capability=False)
    try:
        bundle = evidence.get("code_bundle")
        bundle_receipt = evidence.get("code_bundle_sha256_file")
        if not isinstance(bundle, str) or not isinstance(bundle_receipt, str):
            raise ValueError("bootstrap canary requires a clean current code bundle and receipt")
        bundle_path = Path(bundle)
        _verify_code_bundle_receipt(bundle_path, Path(bundle_receipt))
        _safe_archive(bundle_path, "bootstrap code bundle")
        remote = "/tmp/lehome-bootstrap"
        runner((*_ssh_prefix(instance), "set -eu; mkdir -p " + remote + " /prepared/bootstrap-code"))
        runner(("scp", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", str(instance["port"]), str(bundle_path), "root@" + str(instance["host"]) + ":" + remote + "/code.bundle"))
        observed = runner((*_ssh_prefix(instance), "sha256sum " + remote + "/code.bundle")).strip().split()
        expected = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        if not observed or observed[0] != expected:
            raise ValueError("bootstrap code bundle remote hash readback failed")
        command = (
            *_ssh_prefix(instance),
            "set -eu; tar --no-same-owner --no-same-permissions -xf " + remote
            + "/code.bundle -C /prepared/bootstrap-code; timeout 600 env -u HF_TOKEN "
            + "PYTHONPATH=/prepared/bootstrap-code/source/lehome:/prepared/bootstrap-code/trainer/src "
            + "python -m lehome_train.cli validate-training-capability --one-step --image-digest "
            + BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2],
        )
        capability = json.loads(runner(command))
        if not isinstance(capability, Mapping):
            raise ValueError("bootstrap canary did not emit a capability receipt")
        if capability.get("image_digest") != BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2]:
            raise ValueError("bootstrap capability image does not bind to rented image")
        validated = dict(validate_training_capability(capability))
    except BaseException:
        runner(("vastai", "destroy", "instance", str(instance["instance_id"]), "--yes"))
        absent = _json(runner, ("vastai", "--raw", "show", "instance", str(instance["instance_id"])))
        if absent not in ({}, None):
            raise ValueError("bootstrap cleanup absence readback failed")
        raise
    return {
        "schema_version": 1,
        "kind": "persistent_training_capability",
        "instance_id": instance["instance_id"],
        "trainer_image": BOOTSTRAP_TRAINER_IMAGE,
        "provider_response_sha256": instance["provider_response_sha256"],
        "instance": dict(instance),
        "training_capability": validated,
    }


def _ssh_prefix(instance: Mapping[str, object]) -> tuple[str, ...]:
    host, port = instance.get("host"), instance.get("port")
    if not isinstance(host, str) or not host or type(port) is not int or port <= 0: raise ValueError("instance SSH receipt is invalid")
    return ("ssh", "-o", "IdentitiesOnly=yes", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port), "root@" + host)


def _safe_archive(path: Path, label: str) -> None:
    """Reject traversal, links, and special files before any remote extraction."""
    try:
        with tarfile.open(path, "r:*") as archive:
            members = archive.getmembers()
    except (tarfile.TarError, OSError):
        raise ValueError(f"{label} is not a readable regular archive") from None
    if not members:
        raise ValueError(f"{label} is empty")
    for member in members:
        parts = Path(member.name).parts
        if (
            not member.name
            or member.name.startswith("/")
            or ".." in parts
            or member.issym()
            or member.islnk()
            or not (member.isfile() or member.isdir())
        ):
            raise ValueError(f"{label} has an unsafe archive member")


def _verify_code_bundle_receipt(bundle: Path, receipt: Path) -> str:
    if bundle.is_symlink() or not bundle.is_file() or receipt.is_symlink() or not receipt.is_file():
        raise ValueError("code bundle receipt must name regular files")
    try:
        fields = receipt.read_text(encoding="utf-8").strip().split()
    except (OSError, UnicodeError):
        raise ValueError("code bundle receipt is unreadable") from None
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if len(fields) != 2 or fields[0] != digest or fields[1] not in {bundle.name, "code.bundle"}:
        raise ValueError("code bundle receipt does not match bundle")
    return digest


def _stage_setup_command(parent_archive_sha256: str) -> str:
    """Static remote setup: extraction is constrained to the three mounts."""
    return (
        "set -eu; "
        "mkdir -p /prepared /prepared/code /prepared/config /cache /cache/parent /output; "
        "rm -rf /prepared/generation; "
        "mv /tmp/lehome-stage/generation /prepared/generation; "
        "mv /tmp/lehome-stage/generation.generation.json /prepared/generation.generation.json; "
        "mv /tmp/lehome-stage/launch.json /prepared/config/launch.json; "
        "mv /tmp/lehome-stage/experiment.json /prepared/config/experiment.json; "
        "mv /tmp/lehome-stage/continuous.json /prepared/config/continuous.json; "
        "mv /tmp/lehome-stage/resume.json /prepared/config/resume.json; "
        "if test -f /tmp/lehome-stage/resume-checkpoint.json; then mv /tmp/lehome-stage/resume-checkpoint.json /prepared/config/resume-checkpoint.json; fi; "
        "mv /tmp/lehome-stage/tune.json /prepared/config/tune.json; "
        "mv /tmp/lehome-stage/modality.py /prepared/config/modality.py; "
        "mv /tmp/lehome-stage/token /prepared/config/publisher.token; "
        "tar --no-same-owner --no-same-permissions -xf /tmp/lehome-stage/code.bundle -C /prepared/code; "
        "tar --no-same-owner --no-same-permissions -xf /tmp/lehome-stage/parent.tar -C /cache/parent; "
        "test \"$(sha256sum /tmp/lehome-stage/parent.tar | cut -d' ' -f1)\" = "
        + parent_archive_sha256
        + "; PYTHONPATH=/prepared/code/trainer/src python -c \"from lehome_train.groot.checkpoint_identity import policy_artifact_sha256; assert policy_artifact_sha256('/cache/parent') == '"
        + PARENT_CHECKPOINT["artifact_sha256"]
        + "'\"; chmod 600 /prepared/config/publisher.token; "
        "test ! -L /prepared/generation; test ! -L /cache/parent; test ! -L /prepared/code"
    )


def _load_regular_json(path: Path, label: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"{label} is malformed") from None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is malformed")
    return value


def _sealed_generation_identity(
    generation_root: Path,
    *,
    claimed_mix_plan_sha256: object | None = None,
    claimed_dataset_manifest_sha256: object | None = None,
) -> dict[str, str]:
    """Derive lifecycle data identity from the verified sibling receipt only."""
    from lehome_train.flywheel.mix import load_generation_receipt

    if generation_root.is_symlink() or not generation_root.is_dir():
        raise ValueError("stage generation root is unsafe")
    receipt = load_generation_receipt(generation_root)
    mix_plan = receipt.get("mix_plan_sha256")
    manifest = receipt.get("dataset_manifest_sha256")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("sealed") is not True
        or not isinstance(mix_plan, str)
        or re.fullmatch(r"[0-9a-f]{64}", mix_plan) is None
        or not isinstance(manifest, str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest) is None
    ):
        raise ValueError("stage generation receipt identity is invalid")
    if (
        claimed_mix_plan_sha256 is not None
        and claimed_mix_plan_sha256 != mix_plan
    ) or (
        claimed_dataset_manifest_sha256 is not None
        and claimed_dataset_manifest_sha256 != manifest
    ):
        raise ValueError("stage caller generation identity differs from sealed receipt")
    return {
        "mix_plan_sha256": mix_plan,
        "dataset_manifest_sha256": manifest,
        # This is deliberately a local dataset convention, not a Hub revision.
        "dataset_revision": manifest[:40],
    }


def _validate_staged_operational_requests(
    launch_path: Path,
    continuous_path: Path,
    *,
    generation_identity: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Ensure the request already names the fixed paths created by staging."""
    launch = _load_regular_json(launch_path, "launch config")
    expected_launch = {
        "base_model_path": "/cache/parent",
        "dataset_path": "/prepared/generation",
        "output_dir": "/output",
        "modality_config_path": "/prepared/config/modality.py",
    }
    labels = {
        "base_model_path": "base model path",
        "dataset_path": "generation path",
        "output_dir": "output path",
        "modality_config_path": "modality path",
    }
    for key, expected in expected_launch.items():
        if launch.get(key) != expected:
            raise ValueError(f"stage {labels[key]} is not the hydrated operational path")
    if generation_identity is not None and launch.get("dataset_revision") != generation_identity["dataset_revision"]:
        raise ValueError("stage launch dataset revision is not derived from the sealed manifest")
    continuous = _load_regular_json(continuous_path, "continuous request")
    if (
        set(continuous) != {"schema_version", "command", "arguments"}
        or continuous.get("schema_version") != 1
        or continuous.get("command") != "continuous-train"
        or not isinstance(continuous.get("arguments"), Mapping)
    ):
        raise ValueError("stage continuous request is not an executable continuous-train envelope")
    arguments = continuous["arguments"]
    expected_continuous = {
        "launch_config": "/prepared/config/launch.json",
        "experiment_config": "/prepared/config/experiment.json",
        "generation_root": "/prepared/generation",
        "publisher_token_file": "/prepared/config/publisher.token",
    }
    for key, expected in expected_continuous.items():
        if arguments.get(key) != expected:
            raise ValueError("stage continuous request is not bound to hydrated paths")
    return arguments


def _validate_staged_experiment(
    experiment_path: Path, generation_identity: Mapping[str, str],
) -> None:
    experiment = _load_regular_json(experiment_path, "experiment config")
    if (
        experiment.get("dataset_repository") != "local/sealed-mixed-generation"
        or experiment.get("dataset_manifest_sha256")
        != generation_identity["dataset_manifest_sha256"]
        or experiment.get("dataset_revision") != generation_identity["dataset_revision"]
    ):
        raise ValueError("stage experiment dataset identity is not derived from the sealed manifest")


def _validate_resume_descriptor_for_stage(
    arguments: Mapping[str, object], request: Mapping[str, object],
) -> tuple[str, str, int] | None:
    descriptor = arguments.get("resume_checkpoint")
    publication = arguments.get("resume_publication")
    source = request.get("resume_checkpoint_descriptor")
    if descriptor is None and publication is None:
        if source is not None:
            raise ValueError("stage has a resume descriptor without a resume request")
        return None
    if descriptor != "/prepared/config/resume-checkpoint.json" or not isinstance(publication, Mapping):
        raise ValueError("stage resume request is not bound to the hydrated descriptor")
    if not isinstance(source, str) or not source:
        raise ValueError("stage resume requires an authenticated descriptor source")
    required = {
        "repository", "immutable_revision", "remote_prefix", "relative_path",
        "artifact_sha256", "artifact_byte_size", "descriptor_relative_path",
        "descriptor_sha256", "descriptor_byte_size",
    }
    allowed = required | {
        "optimizer_step", "readback_verified", "generation_sha256",
        "config_sha256", "experiment_id",
    }
    if (
        not required.issubset(publication)
        or not set(publication).issubset(allowed)
        or ("readback_verified" in publication and publication.get("readback_verified") is not True)
    ):
        raise ValueError("stage resume publication is incompatible")
    descriptor_sha = publication["descriptor_sha256"]
    descriptor_size = publication["descriptor_byte_size"]
    if (
        not isinstance(descriptor_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", descriptor_sha) is None
        or type(descriptor_size) is not int
        or descriptor_size <= 0
    ):
        raise ValueError("stage resume descriptor identity is invalid")
    path = Path(source)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != descriptor_size:
        raise ValueError("stage resume descriptor source is unavailable")
    if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor_sha:
        raise ValueError("stage resume descriptor source differs from immutable publication")
    return source, descriptor_sha, descriptor_size


def _validate_replacement_stage_binding(
    arguments: Mapping[str, object], request: Mapping[str, object], *,
    instance: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Bind a staged resume to the exact authenticated replacement receipt."""
    if arguments.get("resume_checkpoint") is None and arguments.get("resume_publication") is None:
        if request.get("replacement_resume_receipt") is not None:
            raise ValueError("stage replacement receipt is present without a resume")
        return None
    receipt = request.get("replacement_resume_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("stage resume requires an authenticated replacement receipt")
    publication = arguments.get("resume_publication")
    descriptor = receipt.get("resume_checkpoint_descriptor")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "persistent_training_replacement_resume"
        or receipt.get("instance") != instance
        or receipt.get("generation_sha256") != request.get("generation_sha256")
        or receipt.get("generation_sha256") != receipt.get("resume_generation_sha256")
        or receipt.get("config_sha256") != receipt.get("resume_config_sha256")
        or not isinstance(publication, Mapping)
        or publication != receipt.get("resume_checkpoint_publication")
        or not isinstance(descriptor, Mapping)
        or descriptor.get("path") != request.get("resume_checkpoint_descriptor")
        or descriptor.get("sha256") != publication.get("descriptor_sha256")
        or descriptor.get("byte_size") != publication.get("descriptor_byte_size")
        or descriptor.get("relative_path") != publication.get("descriptor_relative_path")
    ):
        raise ValueError("stage resume replacement publication or descriptor is incompatible")
    return receipt


def stage(*, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner) -> dict[str, object]:
    _stage_command(request)
    generation_root = Path(str(request["generation_root"]))
    if Path(str(request["generation_receipt"])) != generation_root.with_name(
        generation_root.name + ".generation.json"
    ):
        raise ValueError("stage generation receipt must be the sibling sealed receipt")
    generation_identity = _sealed_generation_identity(
        generation_root,
        claimed_mix_plan_sha256=request.get("generation_sha256"),
        claimed_dataset_manifest_sha256=request.get("dataset_manifest_sha256"),
    )
    if request.get("sealed_generation_sha256") != generation_identity["mix_plan_sha256"]:
        raise ValueError("stage caller generation identity differs from sealed receipt")
    if (
        request.get("dataset_manifest_sha256")
        != generation_identity["dataset_manifest_sha256"]
        or request.get("dataset_revision") != generation_identity["dataset_revision"]
    ):
        raise ValueError("stage caller dataset identity differs from sealed receipt")
    _validate_staged_operational_requests(
        Path(str(request["launch_config"])),
        Path(str(request["continuous_request"])),
        generation_identity=generation_identity,
    )
    resume_arguments = _validate_staged_operational_requests(
        Path(str(request["launch_config"])),
        Path(str(request["resume_request"])),
        generation_identity=generation_identity,
    )
    _validate_staged_experiment(
        Path(str(request["experiment_config"])), generation_identity
    )
    resume_descriptor = _validate_resume_descriptor_for_stage(resume_arguments, request)
    _validate_replacement_stage_binding(resume_arguments, request, instance=instance)
    remote_dir = "/tmp/lehome-stage"
    runner((*_ssh_prefix(instance), "mkdir -p " + remote_dir))
    runner(("scp", "-r", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", str(instance["port"]), str(generation_root), "root@" + str(instance["host"]) + ":" + remote_dir + "/generation"))
    generation_tree = _tree_readback_sha256(generation_root)
    observed_tree = runner((*_ssh_prefix(instance), "cd " + remote_dir + "/generation && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum")).strip().split()
    if not observed_tree or observed_tree[0] != generation_tree: raise ValueError("remote sealed generation tree readback failed")
    code_bundle = Path(str(request["code_bundle"]))
    if _verify_code_bundle_receipt(code_bundle, Path(str(request["code_bundle_sha256_file"]))) != request["code_bundle_sha256"]:
        raise ValueError("staged code bundle hash differs from request")
    pairs = (
        ("code_bundle", "code.bundle"), ("code_bundle_sha256_file", "code.bundle.sha256"),
        ("generation_receipt", "generation.generation.json"), ("parent_checkpoint", "parent.tar"),
        ("launch_config", "launch.json"), ("experiment_config", "experiment.json"),
        ("continuous_request", "continuous.json"), ("resume_request", "resume.json"),
        ("tune_request", "tune.json"), ("modality_config", "modality.py"), ("token_file", "token"),
    )
    if resume_descriptor is not None:
        pairs += (("resume_checkpoint_descriptor", "resume-checkpoint.json"),)
    receipts = []
    for field, remote_name in pairs:
        source = Path(str(request[field]))
        if source.is_symlink() or not source.is_file(): raise ValueError("stage input must be a regular file")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        runner(("scp", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", str(instance["port"]), str(source), "root@" + str(instance["host"]) + ":" + remote_dir + "/" + remote_name))
        observed = runner((*_ssh_prefix(instance), "sha256sum " + remote_dir + "/" + remote_name)).strip().split()
        if not observed or observed[0] != digest: raise ValueError("remote staged hash readback failed")
        receipts.append({"name": remote_name, "sha256": digest})
    receipt_payload = json.loads(Path(str(request["generation_receipt"])).read_text(encoding="utf-8"))
    if receipt_payload.get("sealed") is not True: raise ValueError("staged generation receipt is not sealed")
    code_digest = next(item["sha256"] for item in receipts if item["name"] == "code.bundle")
    if request.get("code_bundle_sha256") != code_digest: raise ValueError("staged code bundle hash differs from request")
    _safe_archive(Path(str(request["code_bundle"])), "code bundle")
    _safe_archive(Path(str(request["parent_checkpoint"])), "parent checkpoint")
    parent_archive_sha256, _ = _parent_identities(request)
    runner((*_ssh_prefix(instance), _stage_setup_command(parent_archive_sha256)))
    return {"paid_action": True, "action": "stage", "instance_id": instance["instance_id"], "generation_tree_sha256": generation_tree, "code_bundle_sha256": code_digest, "generation_identity": generation_identity, "transfers": receipts}


def _parent_identities(request: Mapping[str, object]) -> tuple[str, str]:
    archive_sha, artifact_sha = request.get("parent_archive_sha256"), request.get("parent_checkpoint_sha256")
    if (
        archive_sha != PARENT_CHECKPOINT["archive_sha256"]
        or artifact_sha != PARENT_CHECKPOINT["artifact_sha256"]
    ):
        raise ValueError("stage parent archive and policy artifact identities are required")
    return archive_sha, artifact_sha


def _stage_command(request: Mapping[str, object]) -> str:
    required = ("code_bundle", "code_bundle_sha256", "code_bundle_sha256_file", "generation_root", "generation_receipt", "generation_sha256", "sealed_generation_sha256", "dataset_manifest_sha256", "dataset_revision", "parent_checkpoint", "parent_checkpoint_sha256", "parent_archive_sha256", "launch_config", "experiment_config", "continuous_request", "resume_request", "tune_request", "modality_config", "token_file")
    if any(not isinstance(request.get(key), str) or not request[key] for key in required):
        raise ValueError("stage requires exact code, generation, parent, config, modality, and token paths")
    if request.get("generation_sha256") != request.get("sealed_generation_sha256"):
        raise ValueError("stage generation identity is not sealed")
    if request.get("parent_checkpoint_repository") != PARENT_CHECKPOINT["repository"] or request.get("parent_checkpoint_revision") != PARENT_CHECKPOINT["revision"] or request.get("parent_checkpoint_subpath") != PARENT_CHECKPOINT["subpath"]:
        raise ValueError("stage parent checkpoint identity is not approved")
    _parent_identities(request)
    return "stage-validated"


def _require_instance_capability(instance: Mapping[str, object], request: Mapping[str, object]) -> None:
    """Promote a bootstrap result only when its identity is this live receipt."""
    capability = request.get("capability_receipt")
    if not isinstance(capability, Mapping):
        raise ValueError("full training requires an instance-bound capability receipt")
    if (
        capability.get("kind") != "persistent_training_capability"
        or capability.get("instance_id") != instance.get("instance_id")
        or capability.get("trainer_image") != instance.get("trainer_image")
        or capability.get("provider_response_sha256") != instance.get("provider_response_sha256")
    ):
        raise ValueError("capability receipt is not bound to this instance")
    training_capability = capability.get("training_capability")
    if not isinstance(training_capability, Mapping):
        raise ValueError("capability receipt is malformed")
    if training_capability.get("image_digest") != str(instance.get("trainer_image", "")).rpartition("@")[2]:
        raise ValueError("capability receipt image is incompatible")
    validate_training_capability(training_capability)


def _runtime_receipt(path_value: object, *, kind: str) -> tuple[dict[str, object], str]:
    if type(path_value) is not str:
        raise ValueError("runtime mixture receipt path is required")
    path = Path(path_value)
    value = dict(_load_regular_json(path, "runtime mixture receipt"))
    source_keys = {"repository", "immutable_revision", "remote_prefix", "fresh_readback_verified", "tree_listing_verified"}
    deployment_keys = source_keys | {"mixture_id", "pending_receipt_sha256", "artifact_entries"}
    if set(value) != (source_keys if kind in {"bc", "rollout"} else deployment_keys):
        raise ValueError("runtime mixture receipt schema is incompatible")
    prefix = value.get("remote_prefix")
    if (
        value.get("repository") != CORRECTIVE_SOURCE["repository"]
        or type(value.get("immutable_revision")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", str(value.get("immutable_revision"))) is None
        or value.get("fresh_readback_verified") is not True
        or value.get("tree_listing_verified") is not True
        or (kind == "bc" and prefix != "bc/full")
        or (kind == "rollout" and (type(prefix) is not str or re.fullmatch(r"rollouts/round-[1-9][0-9]*", prefix) is None))
        or (kind == "deployment" and (not isinstance(value.get("mixture_id"), str) or prefix != "mixtures/" + str(value.get("mixture_id")) or not isinstance(value.get("artifact_entries"), list) or not value["artifact_entries"]))
    ):
        raise ValueError("runtime mixture receipt is not an authenticated campaign binding")
    return value, sha256_file(path)


def _runtime_identity(instance: Mapping[str, object], request: Mapping[str, object]) -> dict[str, object]:
    if (
        instance.get("platform_arch") != "x86_64"
        or instance.get("trainer_image") != BOOTSTRAP_TRAINER_IMAGE
        or type(instance.get("instance_id")) is not int
        or not isinstance(request.get("code_revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", str(request.get("code_revision"))) is None
    ):
        raise ValueError("runtime mixture production requires native x86_64 and the approved pinned image")
    bc, bc_sha = _runtime_receipt(request.get("bc_readback_receipt"), kind="bc")
    rollout, rollout_sha = _runtime_receipt(request.get("rollout_readback_receipt"), kind="rollout")
    deployment, deployment_sha = _runtime_receipt(request.get("deployment_receipt"), kind="deployment")
    return {"bc": bc, "bc_receipt_sha256": bc_sha, "rollout": rollout, "rollout_receipt_sha256": rollout_sha, "deployment": deployment, "deployment_receipt_sha256": deployment_sha}


def _validated_runtime_pilot(path_value: object) -> dict[str, object]:
    if type(path_value) is not str:
        raise ValueError("runtime mixture production requires a pilot receipt")
    value = dict(_load_regular_json(Path(path_value), "runtime mixture pilot receipt"))
    evidence = value.get("authenticated_evidence")
    rows = value.get("timing_rows")
    if (
        value.get("schema_version") != 3
        or value.get("kind") != "runtime_mixture_loader_pilot"
        or value.get("model_loaded") is not False or value.get("gpu_initialized") is not False
        or value.get("native_x86_required") is not True
        or value.get("canonical_worker_counts") != [0, 4, 8, 16, 24]
        or value.get("canonical_completion") is not True or value.get("throughput_verified") is not True
        or not isinstance(evidence, Mapping)
        or set(evidence) != {"provider_instance_id", "provider_response_sha256", "platform_arch", "image_digest", "code_revision", "code_bundle_sha256", "bc_revision", "rollout_revision", "deployment_revision"}
        or type(evidence.get("provider_instance_id")) is not int
        or evidence.get("platform_arch") != "x86_64"
        or evidence.get("image_digest") != BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2]
        or any(re.fullmatch(r"[0-9a-f]{40}", str(evidence.get(key))) is None for key in ("code_revision", "bc_revision", "rollout_revision", "deployment_revision"))
        or re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("provider_response_sha256"))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("code_bundle_sha256"))) is None
        or not isinstance(rows, list) or [row.get("worker_count") if isinstance(row, Mapping) else None for row in rows] != [0, 4, 8, 16, 24]
        or any(not isinstance(row, Mapping) or set(row) != {"worker_count", "decoded_samples", "seconds", "samples_per_second", "host_cpu_seconds", "host_max_rss_mib", "latency_seconds_p50", "latency_seconds_p95"} or type(row.get("decoded_samples")) is not int or row["decoded_samples"] < 100 or any(type(row.get(key)) not in (int, float) or float(row[key]) < 0 for key in ("seconds", "samples_per_second", "host_cpu_seconds", "host_max_rss_mib", "latency_seconds_p50", "latency_seconds_p95")) for row in rows)
    ):
        raise ValueError("runtime mixture production requires an authenticated measured canonical CPU-only pilot receipt")
    return value


def runtime_mixture_pilot_provider_plan() -> dict[str, object]:
    """Describe, but never place, the separately approved CPU-pilot rental."""
    return {
        "paid_action": False, "action": "runtime-pilot-plan", "provider_action": "not_rented",
        "platform_arch": "x86_64", "purchase_option": "on_demand",
        "account_hourly_cap_usd": MAX_ACCOUNT_HOURLY_USD, "max_instances": 1,
    }


def runtime_mixture_train(*, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner) -> dict[str, object]:
    """Execute only the receipt-bound runtime-mixture trainer, never legacy RFT."""
    identity = _runtime_identity(instance, request)
    pilot = _validated_runtime_pilot(request.get("pilot_receipt"))
    pilot_evidence = pilot["authenticated_evidence"]
    if (
        type(request.get("code_bundle_sha256")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", str(request.get("code_bundle_sha256"))) is None
        or pilot_evidence.get("code_revision") != request.get("code_revision")
        or pilot_evidence.get("code_bundle_sha256") != request.get("code_bundle_sha256")
        or pilot_evidence.get("bc_revision") != identity["bc"]["immutable_revision"]
        or pilot_evidence.get("rollout_revision") != identity["rollout"]["immutable_revision"]
        or pilot_evidence.get("deployment_revision") != identity["deployment"]["immutable_revision"]
    ):
        raise ValueError("runtime mixture pilot evidence does not bind the staged code and immutable mixture")
    output = request.get("execution_receipt")
    if type(output) is not str or not Path(output).is_absolute() or Path(output).exists() or Path(output).is_symlink():
        raise ValueError("runtime mixture execution receipt must be an absent absolute path")
    command = (
        "set -eu; env -u HF_TOKEN PYTHONPATH=/prepared/code/source/lehome:/prepared/code/trainer/src "
        "lehome-train runtime-mixture-train --request /prepared/config/runtime-train.json"
    )
    runner((*_ssh_prefix(instance), command))
    receipt = {
        "schema_version": 1, "kind": "runtime_mixture_execution", "platform_arch": "x86_64",
        "trainer_image": BOOTSTRAP_TRAINER_IMAGE, "code_revision": request["code_revision"],
        "instance_id": instance["instance_id"], "bc_revision": identity["bc"]["immutable_revision"],
        "bc_tree_receipt_sha256": identity["bc_receipt_sha256"],
        "rollout_revision": identity["rollout"]["immutable_revision"],
        "rollout_tree_receipt_sha256": identity["rollout_receipt_sha256"],
        "deployment_revision": identity["deployment"]["immutable_revision"],
        "deployment_tree_receipt_sha256": identity["deployment_receipt_sha256"],
        "pilot_receipt_sha256": sha256_file(Path(str(request["pilot_receipt"]))),
        "runtime_command": "runtime-mixture-train", "throughput_verified": pilot["throughput_verified"],
    }
    atomic_write_json(Path(output), receipt)
    return {"paid_action": True, "action": "runtime-train", "instance_id": instance["instance_id"], "execution_receipt": receipt}


def runtime_mixture_stage(*, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner) -> dict[str, object]:
    """Securely stage reviewed runtime-only inputs; legacy generation is absent."""
    _runtime_identity(instance, request)
    required = {
        "code_bundle": "code.bundle", "code_bundle_sha256_file": "code.bundle.sha256",
        "launch_config": "launch.json", "experiment_config": "experiment.json",
        "runtime_train_request": "runtime-train.json", "runtime_hydrate_request": "runtime-hydrate.json",
        "modality_config": "modality.py", "token_file": "runtime.token",
        "bc_readback_receipt": "bc-readback.json", "rollout_readback_receipt": "rollout-readback.json",
        "deployment_receipt": "deployment-receipt.json",
    }
    if any(type(request.get(field)) is not str or not request[field] for field in required):
        raise ValueError("runtime mixture stage requires reviewed code, config, token, and authenticated receipts")
    code = Path(str(request["code_bundle"]))
    if _verify_code_bundle_receipt(code, Path(str(request["code_bundle_sha256_file"]))) != request.get("code_bundle_sha256"):
        raise ValueError("runtime mixture stage code bundle receipt differs from the reviewed bundle")
    _safe_archive(code, "runtime mixture code bundle")
    _read_private_token(str(request["token_file"]))
    for field in ("runtime_train_request", "runtime_hydrate_request"):
        envelope = _load_regular_json(Path(str(request[field])), "runtime mixture reviewed request")
        if envelope.get("command") not in {"runtime-mixture-train", "hydrate-runtime-mixture"} or "/prepared/generation" in json.dumps(envelope, sort_keys=True) or "continuous-train" in json.dumps(envelope, sort_keys=True):
            raise ValueError("runtime mixture stage rejects legacy generation request content")
    remote_dir = "/tmp/lehome-runtime-stage"
    runner((*_ssh_prefix(instance), "mkdir -p " + remote_dir))
    transfers: list[dict[str, str]] = []
    for field, remote_name in required.items():
        source = Path(str(request[field]))
        if source.is_symlink() or not source.is_file():
            raise ValueError("runtime mixture stage input must be a regular file")
        digest = sha256_file(source)
        runner(("scp", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", str(instance["port"]), str(source), "root@" + str(instance["host"]) + ":" + remote_dir + "/" + remote_name))
        observed = runner((*_ssh_prefix(instance), "sha256sum " + remote_dir + "/" + remote_name)).strip().split()
        if not observed or observed[0] != digest:
            raise ValueError("runtime mixture staged hash readback failed")
        transfers.append({"name": remote_name, "sha256": digest})
    runner((*_ssh_prefix(instance), "set -eu; mkdir -p /prepared/code /prepared/config /prepared/runtime /output; mv " + remote_dir + "/launch.json /prepared/config/launch.json; mv " + remote_dir + "/experiment.json /prepared/config/experiment.json; mv " + remote_dir + "/runtime-train.json /prepared/config/runtime-train.json; mv " + remote_dir + "/runtime-hydrate.json /prepared/config/runtime-hydrate.json; mv " + remote_dir + "/modality.py /prepared/config/modality.py; mv " + remote_dir + "/runtime.token /prepared/config/runtime.token; mv " + remote_dir + "/bc-readback.json /prepared/config/bc-readback.json; mv " + remote_dir + "/rollout-readback.json /prepared/config/rollout-readback.json; mv " + remote_dir + "/deployment-receipt.json /prepared/config/deployment-receipt.json; tar --no-same-owner --no-same-permissions -xf " + remote_dir + "/code.bundle -C /prepared/code; chmod 600 /prepared/config/runtime.token; test ! -L /prepared/code"))
    return {"paid_action": True, "action": "runtime-stage", "instance_id": instance["instance_id"], "code_bundle_sha256": request["code_bundle_sha256"], "transfers": transfers}


def runtime_mixture_hydrate(*, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner) -> dict[str, object]:
    identity = _runtime_identity(instance, request)
    output = runner((*_ssh_prefix(instance), "set -eu; HF_TOKEN=\"$(cat /prepared/config/runtime.token)\" PYTHONPATH=/prepared/code/source/lehome:/prepared/code/trainer/src lehome-train hydrate-runtime-mixture --request /prepared/config/runtime-hydrate.json"))
    try:
        receipt = json.loads(output)
    except json.JSONDecodeError as error:
        raise ValueError("runtime mixture hydration did not return an authenticated receipt") from error
    deployment = identity["deployment"]
    if not isinstance(receipt, Mapping) or receipt.get("kind") != "runtime_mixture_hydration" or receipt.get("immutable_revision") != deployment["immutable_revision"] or receipt.get("remote_prefix") != deployment["remote_prefix"] or receipt.get("fresh_readback_verified") is not True:
        raise ValueError("runtime mixture hydration receipt is not bound to the approved deployment")
    return {"paid_action": True, "action": "runtime-hydrate", "instance_id": instance["instance_id"], "hydration_receipt": dict(receipt)}


def promote_canary(*, capability_receipt: Mapping[str, object], runner: Runner) -> dict[str, object]:
    """Fresh-read the exact canary before allowing an operational promotion."""
    instance = capability_receipt.get("instance")
    if not isinstance(instance, Mapping):
        raise ValueError("capability receipt lacks an instance-bound SSH receipt")
    _require_instance_capability(instance, {"capability_receipt": capability_receipt})
    _require_account_cap(instance.get("account_hourly_total_usd"), label="capability instance")
    instance_id = instance.get("instance_id")
    if type(instance_id) is not int:
        raise ValueError("capability receipt instance is invalid")
    live = _json(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
    if (
        not isinstance(live, Mapping)
        or live.get("id") != instance_id
        or live.get("actual_status", "running") != "running"
        or not _offer_gpu(live)
        or live.get("num_gpus") != 1
        or float(live.get("dph_total", 99)) >= 1
        or not isinstance(live.get("ssh_host"), str)
        or type(live.get("ssh_port")) is not int
        or _stable_instance_identity(live) != instance.get("provider_response_sha256")
    ):
        raise ValueError("fresh live provider readback does not match capability instance")
    # This reads retained volumes and all current instances after the live
    # instance readback.  The promoted instance is already in that list, so
    # do not add it again.
    total = _live_account_total(runner=runner)
    return dict(instance) | {"account_hourly_total_usd": total}


def provider_interruption_terminal(
    *,
    instance_id: int,
    generation_sha256: str,
    config_sha256: str,
    experiment_id: str,
    publications: list[Mapping[str, object]],
    provider_reason: str,
) -> dict[str, object]:
    """Record only a provider-side stop as resumable work."""
    if (
        type(instance_id) is not int
        or re.fullmatch(r"[0-9a-f]{64}", generation_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", config_sha256) is None
        or not isinstance(experiment_id, str)
        or not experiment_id
        or not isinstance(provider_reason, str)
        or not provider_reason
    ):
        raise ValueError("provider interruption identity is invalid")
    verified = [
        dict(item) for item in publications
        if item.get("readback_verified") is True and type(item.get("optimizer_step")) is int
    ]
    return {
        "schema_version": 1,
        "kind": "continuous_corrective_training_terminal",
        "status": "provider_interrupted",
        "instance_id": instance_id,
        "generation_sha256": generation_sha256,
        "config_sha256": config_sha256,
        "experiment_id": experiment_id,
        "provider_reason": provider_reason,
        "immutable_checkpoint_publications": verified,
        "resumable_checkpoint_step": max((int(item["optimizer_step"]) for item in verified), default=None),
        "disposable": False,
    }


def classify_provider_interruption(
    *, instance: Mapping[str, object], runner: Runner
) -> str | None:
    """Read provider state; SSH loss alone is never enough to call resume safe."""
    instance_id = instance.get("instance_id")
    if type(instance_id) is not int:
        raise ValueError("instance receipt is invalid")
    live = _json(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
    if live in ({}, None):
        return "instance_absent"
    if not isinstance(live, Mapping) or live.get("id") != instance_id:
        raise ValueError("provider interruption readback is invalid")
    status = live.get("actual_status")
    if status in {"interrupted", "terminated", "stopped", "offline"}:
        return "provider_" + str(status)
    return None


def resume_identity(
    terminal: Mapping[str, object], *, generation_sha256: str, config_sha256: str
) -> Mapping[str, object]:
    if terminal.get("status") != "provider_interrupted":
        raise ValueError("resume requires a provider interruption terminal")
    if terminal.get("generation_sha256") != generation_sha256 or terminal.get("config_sha256") != config_sha256:
        raise ValueError("provider interruption resume identity is incompatible")
    step = terminal.get("resumable_checkpoint_step")
    publications = terminal.get("immutable_checkpoint_publications")
    if type(step) is not int or step <= 0 or not isinstance(publications, list):
        raise ValueError("provider interruption has no immutable resumable checkpoint")
    candidates = [item for item in publications if isinstance(item, Mapping) and item.get("optimizer_step") == step]
    if len(candidates) != 1:
        raise ValueError("provider interruption checkpoint discovery is ambiguous")
    candidate = candidates[0]
    if (
        candidate.get("generation_sha256") != generation_sha256
        or candidate.get("config_sha256") != config_sha256
        or candidate.get("experiment_id") != terminal.get("experiment_id")
        or not isinstance(candidate.get("immutable_revision"), str)
        or not all(
            isinstance(candidate.get(field), str) and candidate[field]
            for field in (
                "relative_path", "artifact_sha256", "descriptor_relative_path",
                "descriptor_sha256",
            )
        )
        or type(candidate.get("artifact_byte_size")) is not int
        or type(candidate.get("descriptor_byte_size")) is not int
    ):
        raise ValueError("provider interruption checkpoint is not bound to the terminal identity")
    return candidate


_RESUME_PUBLICATION_FIELDS = {
    "optimizer_step", "readback_verified", "generation_sha256", "config_sha256",
    "experiment_id", "repository", "immutable_revision", "remote_prefix",
    "relative_path", "artifact_sha256", "artifact_byte_size",
    "descriptor_relative_path", "descriptor_sha256", "descriptor_byte_size",
}
_RESUME_TERMINAL_FIELDS = {
    "schema_version", "kind", "status", "instance_id", "generation_sha256",
    "config_sha256", "experiment_id", "provider_reason",
    "immutable_checkpoint_publications", "resumable_checkpoint_step", "disposable",
}
_REPLACEMENT_RECEIPT_FIELDS = {
    "schema_version", "kind", "instance", "capability_receipt",
    "resume_checkpoint_publication", "resume_checkpoint_descriptor",
    "generation_sha256", "config_sha256", "resume_generation_sha256",
    "resume_config_sha256", "experiment_id", "replaced_instance_id",
}
_INSTANCE_RECEIPT_FIELDS = {
    "schema_version", "kind", "instance_id", "host", "port", "trainer_image",
    "offer_evidence_sha256", "provider_response_sha256", "account_hourly_total_usd",
}
_CAPABILITY_RECEIPT_FIELDS = {
    "schema_version", "kind", "instance_id", "trainer_image",
    "provider_response_sha256", "instance", "training_capability",
}
_CONTINUOUS_TRAIN_FIELDS = {
    "launch_config", "experiment_config", "generation_root", "parent_checkpoint_sha256",
    "normalization_sha256", "checkpoint_repository", "checkpoint_revision",
    "instance_id", "result_output", "status_output", "resume_checkpoint",
    "resume_publication", "publisher_token_file",
}


def _resume_sha256(value: object, label: str) -> str:
    try:
        return _sha256(value, label)
    except ValueError:
        raise ValueError(f"resume {label} is invalid") from None


def _validate_resume_instance(instance: Mapping[str, object]) -> None:
    if set(instance) != _INSTANCE_RECEIPT_FIELDS:
        raise ValueError("resume instance schema is invalid")
    if (
        instance.get("schema_version") != 1
        or instance.get("kind") != "persistent_training_instance"
        or type(instance.get("instance_id")) is not int
        or instance["instance_id"] <= 0
        or not isinstance(instance.get("host"), str)
        or not instance["host"]
        or type(instance.get("port")) is not int
        or instance["port"] <= 0
        or type(instance.get("account_hourly_total_usd")) not in (int, float)
    ):
        raise ValueError("resume instance schema is invalid")
    _trainer_image(instance.get("trainer_image"))
    _resume_sha256(instance.get("offer_evidence_sha256"), "instance offer evidence SHA-256")
    _resume_sha256(instance.get("provider_response_sha256"), "instance provider response SHA-256")
    _require_account_cap(instance["account_hourly_total_usd"], label="resume instance")


def _validate_resume_capability(
    capability: object, *, instance: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(capability, Mapping) or set(capability) != _CAPABILITY_RECEIPT_FIELDS:
        raise ValueError("resume capability schema is invalid")
    if capability.get("schema_version") != 1 or capability.get("instance") != instance:
        raise ValueError("resume capability schema is invalid")
    _require_instance_capability(instance, {"capability_receipt": capability})
    return capability


def _validate_resume_publication(
    publication: object, *, generation_sha256: str, config_sha256: str,
    experiment_id: str,
) -> Mapping[str, object]:
    if not isinstance(publication, Mapping) or set(publication) != _RESUME_PUBLICATION_FIELDS:
        raise ValueError("resume publication schema is invalid")
    if (
        publication.get("optimizer_step") is None
        or type(publication["optimizer_step"]) is not int
        or publication["optimizer_step"] <= 0
        or publication.get("readback_verified") is not True
        or publication.get("generation_sha256") != generation_sha256
        or publication.get("config_sha256") != config_sha256
        or publication.get("experiment_id") != experiment_id
        or publication.get("repository") != PARENT_CHECKPOINT["repository"]
        or not isinstance(publication.get("remote_prefix"), str)
        or not publication["remote_prefix"]
        or type(publication.get("artifact_byte_size")) is not int
        or publication["artifact_byte_size"] <= 0
        or type(publication.get("descriptor_byte_size")) is not int
        or publication["descriptor_byte_size"] <= 0
    ):
        raise ValueError("resume publication schema is invalid")
    paths = ("remote_prefix", "relative_path", "descriptor_relative_path")
    if any(
        not isinstance(publication.get(field), str)
        or not publication[field]
        or Path(publication[field]).is_absolute()
        or ".." in Path(publication[field]).parts
        for field in paths
    ):
        raise ValueError("resume publication schema is invalid")
    _resume_sha256(publication.get("artifact_sha256"), "publication artifact SHA-256")
    _resume_sha256(publication.get("descriptor_sha256"), "publication descriptor SHA-256")
    revision = publication.get("immutable_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("resume publication schema is invalid")
    return publication


def _validate_resume_terminal(
    terminal: object, *, generation_sha256: str, config_sha256: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if not isinstance(terminal, Mapping) or set(terminal) != _RESUME_TERMINAL_FIELDS:
        raise ValueError("resume terminal schema is invalid")
    experiment = terminal.get("experiment_id")
    if (
        terminal.get("schema_version") != 1
        or terminal.get("kind") != "continuous_corrective_training_terminal"
        or terminal.get("status") != "provider_interrupted"
        or type(terminal.get("instance_id")) is not int
        or terminal["instance_id"] <= 0
        or terminal.get("generation_sha256") != generation_sha256
        or terminal.get("config_sha256") != config_sha256
        or not isinstance(experiment, str)
        or not experiment
        or not isinstance(terminal.get("provider_reason"), str)
        or not terminal["provider_reason"]
        or terminal.get("disposable") is not False
        or type(terminal.get("resumable_checkpoint_step")) is not int
        or terminal["resumable_checkpoint_step"] <= 0
        or not isinstance(terminal.get("immutable_checkpoint_publications"), list)
        or not terminal["immutable_checkpoint_publications"]
    ):
        raise ValueError("resume terminal schema is invalid")
    for publication in terminal["immutable_checkpoint_publications"]:
        _validate_resume_publication(
            publication, generation_sha256=generation_sha256,
            config_sha256=config_sha256, experiment_id=experiment,
        )
    canonical_step = max(
        int(publication["optimizer_step"])
        for publication in terminal["immutable_checkpoint_publications"]
    )
    if terminal["resumable_checkpoint_step"] != canonical_step:
        raise ValueError("resume terminal resumable checkpoint is not canonical")
    return terminal, resume_identity(
        terminal, generation_sha256=generation_sha256, config_sha256=config_sha256,
    )


def _validate_replacement_resume_receipt(
    receipt: object, *, instance: Mapping[str, object], capability: Mapping[str, object],
    terminal: Mapping[str, object], publication: Mapping[str, object],
    generation_sha256: str, config_sha256: str,
) -> Mapping[str, object]:
    if not isinstance(receipt, Mapping) or set(receipt) != _REPLACEMENT_RECEIPT_FIELDS:
        raise ValueError("resume replacement schema is invalid")
    descriptor = receipt.get("resume_checkpoint_descriptor")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "persistent_training_replacement_resume"
        or receipt.get("instance") != instance
        or receipt.get("capability_receipt") != capability
        or receipt.get("resume_checkpoint_publication") != publication
        or receipt.get("generation_sha256") != generation_sha256
        or receipt.get("config_sha256") != config_sha256
        or receipt.get("resume_generation_sha256") != generation_sha256
        or receipt.get("resume_config_sha256") != config_sha256
        or receipt.get("experiment_id") != terminal.get("experiment_id")
        or receipt.get("replaced_instance_id") != terminal.get("instance_id")
        or receipt.get("replaced_instance_id") == instance.get("instance_id")
        or not isinstance(descriptor, Mapping)
        or set(descriptor) != {"path", "sha256", "byte_size", "relative_path"}
        or not isinstance(descriptor.get("path"), str)
        or not Path(descriptor["path"]).is_absolute()
        or ".." in Path(descriptor["path"]).parts
        or descriptor.get("sha256") != publication.get("descriptor_sha256")
        or descriptor.get("byte_size") != publication.get("descriptor_byte_size")
        or descriptor.get("relative_path") != publication.get("descriptor_relative_path")
    ):
        raise ValueError("resume replacement schema is invalid")
    return receipt


def discover_resume_publication(
    terminal: Mapping[str, object], *, transport: HubTransport, token: str
) -> Mapping[str, object]:
    """Fresh-read the immutable Hub publication selected by the terminal.

    A lost VM and its request JSON are never a source of checkpoint bytes.  The
    terminal identifies a published revision, but this function authenticates
    the complete tree and artifact digest again before a replacement may use it.
    """
    generation, config = terminal.get("generation_sha256"), terminal.get("config_sha256")
    if not isinstance(generation, str) or not isinstance(config, str):
        raise ValueError("provider interruption terminal is malformed")
    publication = resume_identity(terminal, generation_sha256=generation, config_sha256=config)
    _verify_publication_tree(publication=publication, transport=transport, token=token)
    return publication


def _hydrate_resume_descriptor(
    *, publication: Mapping[str, object], transport: HubTransport, token: str,
    output: Path,
) -> dict[str, object]:
    """Persist a descriptor only after the immutable publication readback."""
    descriptor = publication.get("descriptor_relative_path")
    descriptor_sha = publication.get("descriptor_sha256")
    descriptor_size = publication.get("descriptor_byte_size")
    repository = publication.get("repository")
    revision = publication.get("immutable_revision")
    prefix = publication.get("remote_prefix")
    if (
        not all(isinstance(value, str) and value for value in (descriptor, descriptor_sha, repository, revision, prefix))
        or type(descriptor_size) is not int
        or descriptor_size <= 0
        or output.parent.is_symlink()
        or not output.parent.is_dir()
    ):
        raise ValueError("resume descriptor hydration target or identity is invalid")
    if output.exists() or output.is_symlink():
        if (
            output.is_symlink()
            or not output.is_file()
            or output.stat().st_size != descriptor_size
            or hashlib.sha256(output.read_bytes()).hexdigest() != descriptor_sha
        ):
            raise ValueError("resume descriptor retry output is incompatible")
        return {
            "path": str(output),
            "sha256": descriptor_sha,
            "byte_size": descriptor_size,
            "relative_path": descriptor,
        }
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="persistent-resume-descriptor-") as temporary:
        destination = Path(temporary)
        transport.download_files(
            repository=repository,
            revision=revision,
            destination=destination,
            relative_paths=(descriptor,),
            token=token,
            remote_prefix=prefix,
        )
        observed = destination / descriptor
        if (
            not observed.is_file()
            or observed.is_symlink()
            or observed.stat().st_size != descriptor_size
            or hashlib.sha256(observed.read_bytes()).hexdigest() != descriptor_sha
        ):
            raise ValueError("resume descriptor authenticated readback mismatch")
        temporary = output.with_name(f".{output.name}.incomplete")
        if temporary.exists() or temporary.is_symlink():
            raise ValueError("resume descriptor hydration temporary path is unavailable")
        try:
            with temporary.open("xb") as handle:
                handle.write(observed.read_bytes())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    if (
        output.is_symlink()
        or not output.is_file()
        or output.stat().st_size != descriptor_size
        or hashlib.sha256(output.read_bytes()).hexdigest() != descriptor_sha
    ):
        output.unlink(missing_ok=True)
        raise ValueError("resume descriptor hydration writeback mismatch")
    return {
        "path": str(output),
        "sha256": descriptor_sha,
        "byte_size": descriptor_size,
        "relative_path": descriptor,
    }


def replacement_resume_descriptor(
    *, terminal: Mapping[str, object], capability_receipt: Mapping[str, object], runner: Runner,
    transport: HubTransport, token: str, descriptor_output: Path,
) -> dict[str, object]:
    """Promote a replacement canary and bind it to the last immutable resume.

    The caller performs the explicit bootstrap/rent first.  This refuses to
    reuse the interrupted instance, so a resume cannot accidentally target an
    old SSH endpoint or change the sealed generation/config identity.
    """
    generation = terminal.get("generation_sha256")
    config = terminal.get("config_sha256")
    old_instance_id = terminal.get("instance_id")
    if not isinstance(generation, str) or not isinstance(config, str) or type(old_instance_id) is not int:
        raise ValueError("provider interruption terminal is malformed")
    checkpoint = dict(discover_resume_publication(terminal, transport=transport, token=token))
    hydrated_descriptor = _hydrate_resume_descriptor(
        publication=checkpoint,
        transport=transport,
        token=token,
        output=descriptor_output,
    )
    replacement = promote_canary(capability_receipt=capability_receipt, runner=runner)
    if replacement.get("instance_id") == old_instance_id:
        raise ValueError("replacement resume must use a newly bound instance")
    return {
        "schema_version": 1,
        "kind": "persistent_training_replacement_resume",
        "instance": replacement,
        "capability_receipt": dict(capability_receipt),
        "resume_checkpoint_publication": checkpoint,
        "resume_checkpoint_descriptor": hydrated_descriptor,
        "generation_sha256": generation,
        "config_sha256": config,
        "resume_generation_sha256": generation,
        "resume_config_sha256": config,
        "experiment_id": terminal.get("experiment_id"),
        "replaced_instance_id": old_instance_id,
    }


def _verify_staged_resume_binding(
    *, instance: Mapping[str, object], publication: Mapping[str, object],
    descriptor: Mapping[str, object], runner: Runner,
) -> None:
    """Re-read the staged resume inputs immediately before their execution."""
    try:
        envelope = json.loads(
            runner((*_ssh_prefix(instance), "cat /prepared/config/resume.json"))
        )
    except (json.JSONDecodeError, subprocess.CalledProcessError, OSError, TimeoutError) as error:
        raise ValueError("staged resume envelope is unavailable") from error
    arguments = envelope.get("arguments") if isinstance(envelope, Mapping) else None
    if (
        not isinstance(envelope, Mapping)
        or set(envelope) != {"schema_version", "command", "arguments"}
        or envelope.get("schema_version") != 1
        or envelope.get("command") != "continuous-train"
        or not isinstance(arguments, Mapping)
        or set(arguments) != _CONTINUOUS_TRAIN_FIELDS
    ):
        raise ValueError("staged resume request schema is invalid")
    if (
        arguments.get("launch_config") != "/prepared/config/launch.json"
        or arguments.get("experiment_config") != "/prepared/config/experiment.json"
        or arguments.get("generation_root") != "/prepared/generation"
        or arguments.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT["artifact_sha256"]
        or _resume_sha256(arguments.get("normalization_sha256"), "normalization SHA-256")
        != arguments.get("normalization_sha256")
        or arguments.get("checkpoint_repository") != PARENT_CHECKPOINT["repository"]
        or arguments.get("checkpoint_revision") != "main"
        or arguments.get("instance_id") != instance.get("instance_id")
        or not all(
            isinstance(arguments.get(field), str)
            and re.fullmatch(r"/output/[A-Za-z0-9._/-]+", arguments[field]) is not None
            and ".." not in Path(arguments[field]).parts
            for field in ("result_output", "status_output")
        )
        or arguments.get("resume_checkpoint") != "/prepared/config/resume-checkpoint.json"
        or arguments.get("resume_publication") != publication
        or arguments.get("publisher_token_file") != "/prepared/config/publisher.token"
    ):
        raise ValueError("staged resume request identity is incompatible")
    if (
        set(descriptor) != {"path", "sha256", "byte_size", "relative_path"}
        or not isinstance(descriptor.get("path"), str)
        or not descriptor.get("path")
        or descriptor.get("sha256") != publication.get("descriptor_sha256")
        or descriptor.get("byte_size") != publication.get("descriptor_byte_size")
        or descriptor.get("relative_path") != publication.get("descriptor_relative_path")
    ):
        raise ValueError("staged resume descriptor metadata differs from immutable publication")
    try:
        observed = runner((
            *_ssh_prefix(instance),
            "sha256sum /prepared/config/resume-checkpoint.json && "
            "wc -c < /prepared/config/resume-checkpoint.json",
        )).strip().splitlines()
    except (subprocess.CalledProcessError, OSError, TimeoutError) as error:
        raise ValueError("staged resume descriptor is unavailable") from error
    if (
        len(observed) != 2
        or observed[0].split(maxsplit=1)[0] != descriptor["sha256"]
        or observed[1].strip() != str(descriptor["byte_size"])
    ):
        raise ValueError("staged resume descriptor differs from the authenticated replacement receipt")


def remote_action(*, action: str, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner) -> dict[str, object]:
    instance_id = instance.get("instance_id")
    if type(instance_id) is not int: raise ValueError("instance receipt is invalid")
    if action == "stage":
        return stage(instance=instance, request=request, runner=runner)
    if action == "runtime-stage":
        return runtime_mixture_stage(instance=instance, request=request, runner=runner)
    if action == "runtime-hydrate":
        return runtime_mixture_hydrate(instance=instance, request=request, runner=runner)
    if action == "runtime-train":
        return runtime_mixture_train(instance=instance, request=request, runner=runner)
    elif action in {"tune", "train", "status", "resume"}:
        if action in {"tune", "train"}:
            _require_instance_capability(instance, request)
        if action == "resume":
            generation = _resume_sha256(request.get("generation_sha256"), "identity")
            resume_generation = _resume_sha256(
                request.get("resume_generation_sha256"), "identity"
            )
            config = _resume_sha256(request.get("config_sha256"), "identity")
            resume_config = _resume_sha256(
                request.get("resume_config_sha256"), "identity"
            )
            if generation != resume_generation or config != resume_config:
                raise ValueError("resume identity is incompatible")
            _validate_resume_instance(instance)
            capability = _validate_resume_capability(
                request.get("capability_receipt"), instance=instance,
            )
            terminal, expected = _validate_resume_terminal(
                request.get("provider_interruption_terminal"),
                generation_sha256=generation, config_sha256=config,
            )
            if request.get("resume_checkpoint_publication") != expected:
                raise ValueError("resume checkpoint descriptor is not the authenticated interruption publication")
            replacement = _validate_replacement_resume_receipt(
                request.get("replacement_resume_receipt"), instance=instance,
                capability=capability, terminal=terminal, publication=expected,
                generation_sha256=generation, config_sha256=config,
            )
            if request.get("resume_checkpoint_descriptor") != replacement["resume_checkpoint_descriptor"]:
                raise ValueError("resume replacement descriptor is not freshly bound")
            _verify_staged_resume_binding(
                instance=instance,
                publication=expected,
                descriptor=replacement["resume_checkpoint_descriptor"],
                runner=runner,
            )
        terminal_path = request.get("terminal_path", "/output/persistent-training/terminal.json")
        if action == "status" and (
            type(terminal_path) is not str
            or not terminal_path.startswith("/output/")
            or ".." in Path(terminal_path).parts
            or re.fullmatch(r"/output/[A-Za-z0-9._/-]+", terminal_path) is None
        ):
            raise ValueError("status terminal path must be beneath /output")
        commands = {
            "tune": "PYTHONPATH=/prepared/code/source/lehome:/prepared/code/trainer/src lehome-train tune --request /prepared/config/tune.json",
            "train": "env -u HF_TOKEN PYTHONPATH=/prepared/code/source/lehome:/prepared/code/trainer/src lehome-train continuous-train --request /prepared/config/continuous.json",
            "resume": "env -u HF_TOKEN PYTHONPATH=/prepared/code/source/lehome:/prepared/code/trainer/src lehome-train continuous-train --request /prepared/config/resume.json",
            "status": "test -f " + str(terminal_path) + " && cat " + str(terminal_path),
        }
        command = "set -eu; " + commands[action]
    else:
        raise ValueError("unsupported remote lifecycle action")
    try:
        output = runner((*_ssh_prefix(instance), command))
    except (subprocess.CalledProcessError, OSError, TimeoutError):
        # A dead SSH connection is ambiguous.  It becomes resumable only after
        # the provider itself reports an interruption/absence; ordinary code,
        # data, or trainer failures are deliberately propagated unchanged.
        reason = classify_provider_interruption(instance=instance, runner=runner)
        if reason is None or action not in {"train", "resume"}:
            raise
        if action == "resume":
            # These were fully authenticated before the resume command ran;
            # never require caller-provided duplicates for a repeated preemption.
            experiment = terminal["experiment_id"]
            publications = terminal["immutable_checkpoint_publications"]
        else:
            generation = request.get("generation_sha256")
            config = request.get("config_sha256")
            experiment = request.get("experiment_id")
            publications = request.get("immutable_checkpoint_publications", [])
            if (
                not isinstance(generation, str)
                or not isinstance(config, str)
                or not isinstance(experiment, str)
                or not isinstance(publications, list)
            ):
                raise ValueError("provider interruption request lacks immutable training identity") from None
        terminal = provider_interruption_terminal(
            instance_id=instance_id,
            generation_sha256=generation,
            config_sha256=config,
            experiment_id=experiment,
            publications=[item for item in publications if isinstance(item, Mapping)],
            provider_reason=reason,
        )
        return {"paid_action": True, "action": action, "instance_id": instance_id, "terminal": terminal}
    if action == "status":
        try: terminal = json.loads(output)
        except json.JSONDecodeError as error: raise ValueError("remote status terminal is invalid JSON") from error
        if not isinstance(terminal, Mapping) or terminal.get("kind") != "continuous_corrective_training_terminal": raise ValueError("remote status terminal is invalid")
        return {"paid_action": True, "action": action, "instance_id": instance_id, "terminal": dict(terminal)}
    return {"paid_action": True, "action": action, "instance_id": instance_id}


def _verify_publication_tree(*, publication: Mapping[str, object], transport: HubTransport, token: str) -> None:
    """Perform the real authenticated immutable Hub readback, never a shell shim."""
    repository, revision, prefix = publication.get("repository"), publication.get("immutable_revision"), publication.get("remote_prefix")
    artifact, size = publication.get("artifact_sha256"), publication.get("artifact_byte_size")
    descriptor, descriptor_sha, descriptor_size = (
        publication.get("descriptor_relative_path"),
        publication.get("descriptor_sha256"),
        publication.get("descriptor_byte_size"),
    )
    if repository != PARENT_CHECKPOINT["repository"]:
        raise ValueError("immutable publication repository is not the approved model repository")
    if (
        not all(
            isinstance(value, str) and value
            for value in (repository, revision, prefix, artifact, descriptor, descriptor_sha)
        )
        or type(size) is not int
        or size <= 0
        or type(descriptor_size) is not int
        or descriptor_size <= 0
    ):
        raise ValueError("immutable publication binding is invalid")
    tree = transport.list_tree(repository=repository, revision=revision, token=token)
    target = str(prefix).rstrip("/") + "/" + str(publication.get("relative_path", ""))
    descriptor_target = str(prefix).rstrip("/") + "/" + str(descriptor)
    remote_files = {entry.relative_path for entry in tree if entry.entry_type == "file"}
    if target not in remote_files or descriptor_target not in remote_files:
        raise ValueError("immutable Hub tree lacks checkpoint artifact or descriptor")
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="persistent-hub-readback-") as temporary:
        destination = Path(temporary)
        transport.download_files(repository=repository, revision=revision, destination=destination, relative_paths=(str(publication.get("relative_path")), str(descriptor)), token=token, remote_prefix=prefix)
        observed = destination / str(publication.get("relative_path"))
        observed_descriptor = destination / str(descriptor)
        if (
            not observed.is_file()
            or observed.stat().st_size != size
            or hashlib.sha256(observed.read_bytes()).hexdigest() != artifact
            or not observed_descriptor.is_file()
            or observed_descriptor.stat().st_size != descriptor_size
            or hashlib.sha256(observed_descriptor.read_bytes()).hexdigest() != descriptor_sha
        ):
            raise ValueError("immutable Hub artifact readback mismatch")


def destroy(*, instance_id: int, training_receipt: Mapping[str, object], runner: Runner | None = None, transport: HubTransport | None = None, token: str | None = None) -> dict[str, object]:
    publications = training_receipt.get("immutable_checkpoint_publications")
    if training_receipt.get("kind") != "continuous_corrective_training_terminal" or training_receipt.get("instance_id") != instance_id or training_receipt.get("immutable_checkpoint_steps") != [1000, 2000] or not isinstance(publications, list) or {item.get("optimizer_step") for item in publications if isinstance(item, Mapping)} != {1000, 2000} or not all(isinstance(item, Mapping) and item.get("readback_verified") is True and isinstance(item.get("immutable_revision"), str) for item in publications):
        raise ValueError("instance-bound disposal requires two immutable checkpoints")
    terminal_identity = (
        training_receipt.get("generation_sha256"),
        training_receipt.get("config_sha256"),
        training_receipt.get("experiment_id"),
    )
    if not all(isinstance(value, str) and value for value in terminal_identity):
        raise ValueError("disposal terminal lacks generation/config/experiment identity")
    for item in publications:
        assert isinstance(item, Mapping)
        if (
            item.get("repository") != PARENT_CHECKPOINT["repository"]
            or (item.get("generation_sha256"), item.get("config_sha256"), item.get("experiment_id")) != terminal_identity
        ):
            raise ValueError("immutable publication is not bound to the approved model repository and terminal identity")
    if runner is not None:
        if transport is None or not isinstance(token, str) or not token:
            raise ValueError("destroy requires an authenticated Hub transport and token")
        for item in publications:
            assert isinstance(item, Mapping)
            _verify_publication_tree(publication=item, transport=transport, token=token)
        runner(("vastai", "destroy", "instance", str(instance_id), "--yes"))
        observed = _json(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
        if observed not in ({}, None): raise ValueError("destroy absence readback failed")
    return {"paid_action": runner is not None, "destroy_authorized": True, "instance_id": instance_id}


def _materialize(request: Mapping[str, object]) -> dict[str, object]:
    """Build the only accepted 70/30 generation through the canonical mixer."""
    organizer, corrective, destination, seed, staging_root = (
        request.get("organizer_root"),
        request.get("corrective_roots"),
        request.get("destination"),
        request.get("seed"),
        request.get("persistent_staging_root"),
    )
    if (
        not isinstance(organizer, str)
        or not isinstance(corrective, list)
        or not corrective
        or not all(isinstance(item, str) and item for item in corrective)
        or not isinstance(destination, str)
        or type(seed) is not int
        or not isinstance(staging_root, str)
        or not staging_root
    ):
        raise ValueError("materialize requires organizer, corrective roots, destination, persistent staging root, and integer seed")
    video_workers = request.get("video_workers", 4)
    if type(video_workers) is not int or not 1 <= video_workers <= 32:
        raise ValueError("materialize video_workers must be an integer from 1 to 32")
    from lehome_train.flywheel.mix import (
        build_mix_plan,
        materialize_mixed_snapshot,
        verify_generation,
    )

    organizer_root = Path(organizer)
    corrective_roots = [Path(item) for item in corrective]
    destination_root = Path(destination)
    if destination_root.is_symlink():
        raise ValueError("materialize destination must not be a symlink")
    plan = build_mix_plan(organizer_root, corrective_roots, seed=seed)
    # Organizer is pinned through its manifest contract.  Corrective roots are
    # additionally bound to a previous authenticated full-release readback.
    organizer_evidence = request.get("organizer_source_evidence")
    if not isinstance(organizer_evidence, Mapping):
        raise ValueError("materialize requires verified organizer source evidence")
    corrective_evidence = _verified_corrective_release_evidence(
        corrective_roots, request.get("corrective_release_receipt"),
    )
    evidence = {"organizer_source": dict(organizer_evidence), "corrective_source": corrective_evidence}
    _verify_prepare_evidence(evidence)
    materialize_mixed_snapshot(
        plan, organizer_root, corrective_roots, destination_root,
        persistent_source_evidence=evidence,
        persistent_staging_root=staging_root,
        video_workers=video_workers,
    )
    sealed = verify_generation(destination_root)
    if sealed["organizer_training_frames"] * 3 != sealed["rft_training_frames"] * 7:
        raise ValueError("materialized generation is not exact 70/30")
    mix_plan = _sha256(sealed.get("mix_plan_sha256"), "materialized mix plan SHA-256")
    manifest = _sha256(
        sealed.get("dataset_manifest_sha256"), "materialized dataset manifest SHA-256"
    )
    return {
        "paid_action": False,
        "action": "materialize",
        "generation_root": str(destination_root),
        "generation_receipt": str(destination_root.with_name(destination_root.name + ".generation.json")),
        "generation_sha256": mix_plan,
        "sealed_generation_sha256": mix_plan,
        "dataset_manifest_sha256": manifest,
        "dataset_revision": manifest[:40],
        "generation_receipt_sha256": _hash(sealed),
    }


def main_for_test(argv: list[str], *, runner: Runner = _run) -> dict[str, object]:
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("derive-corrective-receipt", "materialize", "prepare", "capture-offers", "bootstrap-canary", "promote", "replacement-resume", "rent", "stage", "runtime-pilot-plan", "runtime-stage", "runtime-hydrate", "runtime-train", "tune", "train", "status", "resume", "destroy")); parser.add_argument("--request", required=True); parser.add_argument("--execute", action="store_true"); parser.add_argument("--token-file")
    args = parser.parse_args(argv); request = _load(args.request)
    if args.action == "materialize":
        return _materialize(request)
    if args.action == "derive-corrective-receipt":
        fields = ("disposal_receipt", "snapshot_root", "output")
        if any(not isinstance(request.get(field), str) for field in fields):
            raise ValueError("derive-corrective-receipt requires disposal receipt, snapshot root, and output")
        return {
            "paid_action": False,
            "action": args.action,
            "receipt": derive_corrective_receipt(
                disposal_receipt=Path(str(request["disposal_receipt"])),
                snapshot_root=Path(str(request["snapshot_root"])),
                output=Path(str(request["output"])),
            ),
        }
    if args.action == "runtime-pilot-plan":
        return runtime_mixture_pilot_provider_plan()
    if args.action == "prepare":
        generation = request.get("generation_root")
        if not isinstance(generation, str): raise ValueError("prepare requires generation_root")
        root = Path(generation)
        receipt = root.with_name(root.name + ".generation.json")
        if not root.is_dir() or root.is_symlink() or not receipt.is_file() or receipt.is_symlink(): raise ValueError("prepare requires a local sealed generation")
        from lehome_train.flywheel.mix import verify_generation
        sealed = verify_generation(root)
        if sealed.get("organizer_training_frames", 0) * 3 != sealed.get("rft_training_frames", -1) * 7: raise ValueError("prepare generation is not exact 70/30")
        evidence = sealed.get("persistent_source_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("prepare sealed generation lacks source/release evidence")
        _verify_prepare_evidence(evidence)
        return {"paid_action": False, "action": "prepare", "organizer_source": ORGANIZER_SOURCE, "corrective_source": CORRECTIVE_SOURCE, "request": request}
    if not args.execute: return {"paid_action": False, "action": args.action, "dry_run": True, "request": request}
    if args.action == "capture-offers": return capture_offers(runner=runner)
    if args.action == "bootstrap-canary": return bootstrap_canary(evidence=request, runner=runner)
    if args.action == "promote": return {"paid_action": True, "action": "promote", "instance": promote_canary(capability_receipt=request, runner=runner)}
    if args.action == "replacement-resume":
        terminal = request.get("provider_interruption_terminal")
        capability = request.get("capability_receipt")
        descriptor_output = request.get("resume_descriptor_output")
        if not isinstance(terminal, Mapping) or not isinstance(capability, Mapping) or not isinstance(descriptor_output, str):
            raise ValueError("replacement-resume requires interruption terminal, replacement capability receipt, and descriptor output")
        return {"paid_action": True, "action": "replacement-resume", **replacement_resume_descriptor(terminal=terminal, capability_receipt=capability, runner=runner, transport=HuggingFaceHubTransport(timeout_seconds=30.0), token=_read_private_token(args.token_file), descriptor_output=Path(descriptor_output))}
    if args.action == "rent": return rent(evidence=request, runner=runner)
    if args.action == "destroy":
        return destroy(
            instance_id=request.get("instance_id"),  # type: ignore[arg-type]
            training_receipt=request,
            runner=runner,
            transport=HuggingFaceHubTransport(timeout_seconds=30.0),
            token=_read_private_token(args.token_file),
        )
    instance = request.get("instance")
    if not isinstance(instance, Mapping): raise ValueError("remote action requires an instance receipt")
    return remote_action(action=args.action, instance=instance, request=request, runner=runner)


if __name__ == "__main__": print(json.dumps(main_for_test(__import__("sys").argv[1:]), sort_keys=True))
