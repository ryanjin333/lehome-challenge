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
from pathlib import Path
import re
import subprocess
import tarfile
import time
from typing import Callable, Mapping

from lehome_train.hub import HubTransport, HuggingFaceHubTransport
from lehome_train.release_manifest import validate_training_capability

ORGANIZER_SOURCE = {"repository": "lehome/dataset_challenge_merged", "revision": "17e8dee8fac294ffd21d250501d3b31bf8679042", "subdir": "four_types_merged", "mirror_repository": "kunhsiang/lehome-four-types-merged", "mirror_revision": "2ebcccf528dec91cefac0c94a9214a83028ae6cc", "manifest_sha256": "bf8fbae82002a33ff304b9a70993bdfe1c678ba9e8f798c1ad370d58969435eb"}
CORRECTIVE_SOURCE = {"revision": "e6cd1c182514c15271c805d03a646e7a4f95b17c", "prefix": "corrective-rft/b96be3db22174a12dab62a8a673f7c7d083f87aa7b50c4e03ee43e064da56c35"}
PARENT_CHECKPOINT = {"repository": "ryanjin333/lehome-groot-n17-models", "revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3", "subpath": "policies/step-12000", "artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06"}
# Vast's raw expression grammar does not support a portable OR form for two
# exact SKU strings.  Query only stable numeric facts, then enforce the narrow
# WS/S allowlist on raw rows in ``_offer_gpu``.
OFFER_QUERY = "gpu_ram>=96000 num_gpus=1 reliability>=0.95"
_DIGEST_PREFIX = "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:"
BOOTSTRAP_TRAINER_IMAGE = (
    _DIGEST_PREFIX
    + "b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746"
)
Runner = Callable[[tuple[str, ...]], str]


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


def capture_offers(*, runner: Runner, now_unix: int | None = None, ttl_seconds: int = 300) -> dict[str, object]:
    offers = _json(runner, ("vastai", "--raw", "search", "offers", OFFER_QUERY, "--interruptible"))
    instances = _json(runner, ("vastai", "--raw", "show", "instances"))
    volumes = _json(runner, ("vastai", "--raw", "show", "volumes"))
    if not all(isinstance(value, list) for value in (offers, instances, volumes)): raise ValueError("provider listing is invalid")
    eligible = [row for row in offers if isinstance(row, Mapping) and _offer_gpu(row) and row.get("num_gpus") == 1 and float(row.get("dph_total", 99)) < 1]
    if not eligible: raise ValueError("no interruptible RTX PRO 6000 96GB offer under $1/hr")
    offer = min(eligible, key=lambda row: float(row["dph_total"]))
    # Some raw offer variants omit a disk quote.  Count a deliberately
    # conservative fallback rather than silently treating 300GB as free.
    storage_unit_cost = offer.get("storage_cost", offer.get("storage_cost_per_gb", 0.001))
    if type(storage_unit_cost) not in (int, float) or float(storage_unit_cost) < 0:
        raise ValueError("offer storage quote is invalid")
    requested_storage_hourly = float(storage_unit_cost) * 300
    total = sum(float(row.get("dph_total", 0)) for row in instances if isinstance(row, Mapping)) + sum(float(row.get("storage_total_cost", 0)) for row in volumes if isinstance(row, Mapping)) + float(offer["dph_total"]) + requested_storage_hourly
    if total > 2: raise ValueError("account-wide instance and storage total exceeds $2/hr")
    captured = int(time.time()) if now_unix is None else now_unix
    safe_offer = _project(offer, ("id", "gpu_name", "gpu_ram", "num_gpus", "dph_total", "min_bid", "driver_version", "is_bid", "image"))
    return {"schema_version": 1, "kind": "persistent_training_offer", "offer": safe_offer, "account_hourly_total_usd": total, "requested_storage_gb": 300, "requested_storage_hourly_usd": requested_storage_hourly, "captured_at_unix": captured, "expires_at_unix": captured + ttl_seconds, "search_mode": "interruptible"}


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
    return {"schema_version": 1, "kind": "persistent_training_instance", "instance_id": instance_id, "host": live.get("ssh_host"), "port": live.get("ssh_port"), "trainer_image": image, "offer_evidence_sha256": _hash(evidence), "provider_response_sha256": _hash(live)}


def bootstrap_canary(*, evidence: Mapping[str, object], runner: Runner) -> dict[str, object]:
    """Rent the one historical image only long enough to prove its capability.

    Full 2K actions cannot call this path: they consume the resulting canonical
    receipt through ``rent`` with ``require_capability=True``.
    """
    if evidence.get("trainer_image") != BOOTSTRAP_TRAINER_IMAGE:
        raise ValueError("bootstrap canary requires the historical structurally pinned trainer image")
    instance = rent(evidence=evidence, runner=runner, require_capability=False)
    command = (
        *_ssh_prefix(instance),
        "set -eu; timeout 600 lehome-train validate-training-capability --one-step --image-digest "
        + BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2],
    )
    try:
        capability = json.loads(runner(command))
    except json.JSONDecodeError as error:
        raise ValueError("bootstrap canary did not emit a capability receipt") from error
    if not isinstance(capability, Mapping):
        raise ValueError("bootstrap canary did not emit a capability receipt")
    if capability.get("image_digest") != BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2]:
        raise ValueError("bootstrap capability image does not bind to rented image")
    validated = dict(validate_training_capability(capability))
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


def _stage_setup_command() -> str:
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
        "mv /tmp/lehome-stage/tune.json /prepared/config/tune.json; "
        "mv /tmp/lehome-stage/modality.py /prepared/config/modality.py; "
        "mv /tmp/lehome-stage/token /prepared/config/publisher.token; "
        "tar --no-same-owner --no-same-permissions -xf /tmp/lehome-stage/code.bundle -C /prepared/code; "
        "tar --no-same-owner --no-same-permissions -xf /tmp/lehome-stage/parent.tar -C /cache/parent; "
        "test \"$(sha256sum /tmp/lehome-stage/parent.tar | cut -d' ' -f1)\" = "
        + PARENT_CHECKPOINT["artifact_sha256"]
        + "; chmod 600 /prepared/config/publisher.token; "
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


def _validate_staged_operational_requests(launch_path: Path, continuous_path: Path) -> None:
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
    continuous = _load_regular_json(continuous_path, "continuous request")
    expected_continuous = {
        "launch_config": "/prepared/config/launch.json",
        "experiment_config": "/prepared/config/experiment.json",
        "generation_root": "/prepared/generation",
        "publisher_token_file": "/prepared/config/publisher.token",
    }
    for key, expected in expected_continuous.items():
        if continuous.get(key) != expected:
            raise ValueError("stage continuous request is not bound to hydrated paths")


def stage(*, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner) -> dict[str, object]:
    _stage_command(request)
    _validate_staged_operational_requests(
        Path(str(request["launch_config"])),
        Path(str(request["continuous_request"])),
    )
    remote_dir = "/tmp/lehome-stage"
    runner((*_ssh_prefix(instance), "mkdir -p " + remote_dir))
    generation_root = Path(str(request["generation_root"]))
    if generation_root.is_symlink() or not generation_root.is_dir(): raise ValueError("stage generation root is unsafe")
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
    runner((*_ssh_prefix(instance), _stage_setup_command()))
    return {"paid_action": True, "action": "stage", "instance_id": instance["instance_id"], "generation_tree_sha256": generation_tree, "code_bundle_sha256": code_digest, "transfers": receipts}


def _stage_command(request: Mapping[str, object]) -> str:
    required = ("code_bundle", "code_bundle_sha256", "code_bundle_sha256_file", "generation_root", "generation_receipt", "parent_checkpoint", "parent_checkpoint_sha256", "launch_config", "experiment_config", "continuous_request", "resume_request", "tune_request", "modality_config", "token_file")
    if any(not isinstance(request.get(key), str) or not request[key] for key in required):
        raise ValueError("stage requires exact code, generation, parent, config, modality, and token paths")
    if request.get("generation_sha256") != request.get("sealed_generation_sha256"):
        raise ValueError("stage generation identity is not sealed")
    parent_sha = request.get("parent_checkpoint_sha256")
    if request.get("parent_checkpoint_repository") != PARENT_CHECKPOINT["repository"] or request.get("parent_checkpoint_revision") != PARENT_CHECKPOINT["revision"] or request.get("parent_checkpoint_subpath") != PARENT_CHECKPOINT["subpath"] or parent_sha != PARENT_CHECKPOINT["artifact_sha256"]:
        raise ValueError("stage parent checkpoint identity is not approved")
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


def remote_action(*, action: str, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner) -> dict[str, object]:
    instance_id = instance.get("instance_id")
    if type(instance_id) is not int: raise ValueError("instance receipt is invalid")
    if action == "resume" and request.get("generation_sha256") != request.get("resume_generation_sha256") or action == "resume" and request.get("config_sha256") != request.get("resume_config_sha256"):
        raise ValueError("resume requires exact generation/config identity")
    if action == "stage":
        return stage(instance=instance, request=request, runner=runner)
    elif action in {"tune", "train", "status", "resume"}:
        if action in {"tune", "train", "resume"}:
            _require_instance_capability(instance, request)
        if action == "resume" and request.get("generation_sha256") != request.get("resume_generation_sha256") or action == "resume" and request.get("config_sha256") != request.get("resume_config_sha256"):
            raise ValueError("resume requires exact generation/config identity")
        terminal_path = request.get("terminal_path", "/output/persistent-training/terminal.json")
        if action == "status" and (
            type(terminal_path) is not str
            or not terminal_path.startswith("/output/")
            or ".." in Path(terminal_path).parts
            or re.fullmatch(r"/output/[A-Za-z0-9._/-]+", terminal_path) is None
        ):
            raise ValueError("status terminal path must be beneath /output")
        commands = {
            "tune": "PYTHONPATH=/prepared/code/source/lehome:/prepared/code/trainer/src lehome-train smoke --request /prepared/config/tune.json",
            "train": "env -u HF_TOKEN PYTHONPATH=/prepared/code/source/lehome:/prepared/code/trainer/src lehome-train continuous-train --request /prepared/config/continuous.json",
            "resume": "env -u HF_TOKEN PYTHONPATH=/prepared/code/source/lehome:/prepared/code/trainer/src lehome-train continuous-train --request /prepared/config/resume.json",
            "status": "test -f " + str(terminal_path) + " && cat " + str(terminal_path),
        }
        command = "set -eu; " + commands[action]
    else:
        raise ValueError("unsupported remote lifecycle action")
    output = runner((*_ssh_prefix(instance), command))
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
    if not all(isinstance(value, str) and value for value in (repository, revision, prefix, artifact)) or type(size) is not int or size <= 0:
        raise ValueError("immutable publication binding is invalid")
    tree = transport.list_tree(repository=repository, revision=revision, token=token)
    target = str(prefix).rstrip("/") + "/" + str(publication.get("relative_path", ""))
    if not any(entry.relative_path == target and entry.entry_type == "file" for entry in tree):
        raise ValueError("immutable Hub tree lacks checkpoint artifact")
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="persistent-hub-readback-") as temporary:
        destination = Path(temporary)
        transport.download_files(repository=repository, revision=revision, destination=destination, relative_paths=(str(publication.get("relative_path")),), token=token, remote_prefix=prefix)
        observed = destination / str(publication.get("relative_path"))
        if not observed.is_file() or observed.stat().st_size != size or hashlib.sha256(observed.read_bytes()).hexdigest() != artifact:
            raise ValueError("immutable Hub artifact readback mismatch")


def destroy(*, instance_id: int, training_receipt: Mapping[str, object], runner: Runner | None = None, transport: HubTransport | None = None, token: str | None = None) -> dict[str, object]:
    publications = training_receipt.get("immutable_checkpoint_publications")
    if training_receipt.get("kind") != "continuous_corrective_training_terminal" or training_receipt.get("instance_id") != instance_id or training_receipt.get("immutable_checkpoint_steps") != [1000, 2000] or not isinstance(publications, list) or {item.get("optimizer_step") for item in publications if isinstance(item, Mapping)} != {1000, 2000} or not all(isinstance(item, Mapping) and item.get("readback_verified") is True and isinstance(item.get("immutable_revision"), str) for item in publications):
        raise ValueError("instance-bound disposal requires two immutable checkpoints")
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
    organizer, corrective, destination, seed = (
        request.get("organizer_root"),
        request.get("corrective_roots"),
        request.get("destination"),
        request.get("seed"),
    )
    if (
        not isinstance(organizer, str)
        or not isinstance(corrective, list)
        or not corrective
        or not all(isinstance(item, str) and item for item in corrective)
        or not isinstance(destination, str)
        or type(seed) is not int
    ):
        raise ValueError("materialize requires organizer, corrective roots, destination, and integer seed")
    from lehome_train.flywheel.mix import (
        build_mix_plan,
        materialize_mixed_snapshot,
        verify_generation,
    )

    organizer_root = Path(organizer)
    corrective_roots = [Path(item) for item in corrective]
    destination_root = Path(destination)
    if destination_root.exists() or destination_root.is_symlink():
        raise ValueError("materialize destination must not already exist")
    plan = build_mix_plan(organizer_root, corrective_roots, seed=seed)
    materialize_mixed_snapshot(plan, organizer_root, corrective_roots, destination_root)
    sealed = verify_generation(destination_root)
    if sealed["organizer_training_frames"] * 3 != sealed["rft_training_frames"] * 7:
        raise ValueError("materialized generation is not exact 70/30")
    return {
        "paid_action": False,
        "action": "materialize",
        "generation_root": str(destination_root),
        "generation_receipt": str(destination_root.with_name(destination_root.name + ".generation.json")),
        "generation_sha256": _hash(sealed),
    }


def main_for_test(argv: list[str], *, runner: Runner = _run) -> dict[str, object]:
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("materialize", "prepare", "capture-offers", "bootstrap-canary", "rent", "stage", "tune", "train", "status", "resume", "destroy")); parser.add_argument("--request", required=True); parser.add_argument("--execute", action="store_true"); parser.add_argument("--token-file")
    args = parser.parse_args(argv); request = _load(args.request)
    if args.action == "materialize":
        return _materialize(request)
    if args.action == "prepare":
        generation = request.get("generation_root")
        if not isinstance(generation, str): raise ValueError("prepare requires generation_root")
        root = Path(generation)
        receipt = root.with_name(root.name + ".generation.json")
        if not root.is_dir() or root.is_symlink() or not receipt.is_file() or receipt.is_symlink(): raise ValueError("prepare requires a local sealed generation")
        from lehome_train.flywheel.mix import verify_generation
        sealed = verify_generation(root)
        if sealed.get("organizer_training_frames", 0) * 3 != sealed.get("rft_training_frames", -1) * 7: raise ValueError("prepare generation is not exact 70/30")
        revisions = sealed.get("source_revisions")
        if not isinstance(revisions, Mapping):
            raise ValueError("prepare sealed generation has no source revisions")
        observed_revisions = set(revisions.values())
        if ORGANIZER_SOURCE["revision"] not in observed_revisions:
            raise ValueError("prepare organizer source revision is not pinned in sealed generation")
        # The corrective Hub prefix is a source contract outside the materialized
        # dataset, but the exact immutable revision must still be represented.
        if CORRECTIVE_SOURCE["revision"] not in observed_revisions:
            raise ValueError("prepare corrective source revision is not pinned in sealed generation")
        return {"paid_action": False, "action": "prepare", "organizer_source": ORGANIZER_SOURCE, "corrective_source": CORRECTIVE_SOURCE, "request": request}
    if not args.execute: return {"paid_action": False, "action": args.action, "dry_run": True, "request": request}
    if args.action == "capture-offers": return capture_offers(runner=runner)
    if args.action == "bootstrap-canary": return bootstrap_canary(evidence=request, runner=runner)
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
