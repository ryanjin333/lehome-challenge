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
import subprocess
import time
from typing import Callable, Mapping

ORGANIZER_SOURCE = {"repository": "lehome/dataset_challenge_merged", "revision": "17e8dee8fac294ffd21d250501d3b31bf8679042", "subdir": "four_types_merged", "mirror_repository": "kunhsiang/lehome-four-types-merged", "mirror_revision": "2ebcccf528dec91cefac0c94a9214a83028ae6cc", "manifest_sha256": "bf8fbae82002a33ff304b9a70993bdfe1c678ba9e8f798c1ad370d58969435eb"}
CORRECTIVE_SOURCE = {"revision": "e6cd1c182514c15271c805d03a646e7a4f95b17c", "prefix": "corrective-rft/b96be3db22174a12dab62a8a673f7c7d083f87aa7b50c4e03ee43e064da56c35"}
PARENT_CHECKPOINT = {"repository": "ryanjin333/lehome-groot-n17-models", "revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3", "subpath": "policies/step-12000", "artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06"}
# Vast's raw expression grammar does not support a portable OR form for two
# exact SKU strings.  Query only stable numeric facts, then enforce the narrow
# WS/S allowlist on raw rows in ``_offer_gpu``.
OFFER_QUERY = "gpu_ram>=96000 num_gpus=1 reliability>=0.95"
_DIGEST_PREFIX = "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:"
Runner = Callable[[tuple[str, ...]], str]


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load(path: str) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError("lifecycle request must be an object")
    return value


def _run(command: tuple[str, ...]) -> str:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return completed.stdout


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


def capture_offers(*, runner: Runner, now_unix: int | None = None, ttl_seconds: int = 300) -> dict[str, object]:
    offers = _json(runner, ("vastai", "--raw", "search", "offers", OFFER_QUERY, "--interruptible"))
    instances = _json(runner, ("vastai", "--raw", "show", "instances"))
    volumes = _json(runner, ("vastai", "--raw", "show", "volumes"))
    if not all(isinstance(value, list) for value in (offers, instances, volumes)): raise ValueError("provider listing is invalid")
    eligible = [row for row in offers if isinstance(row, Mapping) and _offer_gpu(row) and row.get("num_gpus") == 1 and float(row.get("dph_total", 99)) < 1]
    if not eligible: raise ValueError("no interruptible RTX PRO 6000 96GB offer under $1/hr")
    offer = min(eligible, key=lambda row: float(row["dph_total"]))
    total = sum(float(row.get("dph_total", 0)) for row in instances if isinstance(row, Mapping)) + sum(float(row.get("storage_total_cost", 0)) for row in volumes if isinstance(row, Mapping)) + float(offer["dph_total"])
    if total > 2: raise ValueError("account-wide instance and storage total exceeds $2/hr")
    captured = int(time.time()) if now_unix is None else now_unix
    safe_offer = _project(offer, ("id", "gpu_name", "gpu_ram", "num_gpus", "dph_total", "min_bid", "driver_version", "is_bid", "image"))
    return {"schema_version": 1, "kind": "persistent_training_offer", "offer": safe_offer, "account_hourly_total_usd": total, "captured_at_unix": captured, "expires_at_unix": captured + ttl_seconds, "search_mode": "interruptible"}


def rent(*, evidence: Mapping[str, object], runner: Runner, max_readiness_polls: int = 12) -> dict[str, object]:
    offer = evidence.get("offer")
    if not isinstance(offer, Mapping) or type(offer.get("id")) is not int: raise ValueError("offer evidence is invalid")
    if evidence.get("search_mode") != "interruptible" or type(evidence.get("expires_at_unix")) is not int or evidence["expires_at_unix"] < int(time.time()): raise ValueError("offer evidence is expired or not interruptible")
    image = _trainer_image(evidence.get("trainer_image"))
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
    else:
        runner(("vastai", "destroy", "instance", str(instance_id), "--yes")); raise ValueError("instance readiness poll timed out")
    if not isinstance(live, Mapping) or live.get("id") != instance_id or not _offer_gpu(live) or live.get("num_gpus") != 1 or not live.get("ssh_host") or type(live.get("ssh_port")) is not int or float(live.get("dph_total", 99)) >= 1:
        runner(("vastai", "destroy", "instance", str(instance_id), "--yes")); raise ValueError("instance readback does not match accepted offer")
    return {"schema_version": 1, "kind": "persistent_training_instance", "instance_id": instance_id, "host": live.get("ssh_host"), "port": live.get("ssh_port"), "trainer_image": image, "offer_evidence_sha256": _hash(evidence), "provider_response_sha256": _hash(live)}


def _ssh_prefix(instance: Mapping[str, object]) -> tuple[str, ...]:
    host, port = instance.get("host"), instance.get("port")
    if not isinstance(host, str) or not host or type(port) is not int or port <= 0: raise ValueError("instance SSH receipt is invalid")
    return ("ssh", "-o", "IdentitiesOnly=yes", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port), "root@" + host)


def stage(*, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner) -> dict[str, object]:
    _stage_command(request)
    remote_dir = "/tmp/lehome-stage"
    runner((*_ssh_prefix(instance), "mkdir -p " + remote_dir))
    generation_root = Path(str(request["generation_root"]))
    if generation_root.is_symlink() or not generation_root.is_dir(): raise ValueError("stage generation root is unsafe")
    runner(("scp", "-r", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", str(instance["port"]), str(generation_root), "root@" + str(instance["host"]) + ":" + remote_dir + "/generation"))
    generation_manifest = hashlib.sha256((generation_root / "manifest.json").read_bytes()).hexdigest()
    observed_manifest = runner((*_ssh_prefix(instance), "sha256sum " + remote_dir + "/generation/manifest.json")).strip().split()
    if not observed_manifest or observed_manifest[0] != generation_manifest: raise ValueError("remote generation manifest readback failed")
    pairs = (("code_bundle", "code.bundle"), ("generation_receipt", "generation.generation.json"), ("parent_checkpoint", "parent.tar"), ("launch_config", "launch.json"), ("modality_config", "modality.py"), ("token_file", "token"))
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
    if receipt_payload.get("dataset_manifest_sha256") != generation_manifest: raise ValueError("staged generation receipt does not bind its manifest")
    return {"paid_action": True, "action": "stage", "instance_id": instance["instance_id"], "generation_manifest_sha256": generation_manifest, "transfers": receipts}


def _stage_command(request: Mapping[str, object]) -> str:
    required = ("code_bundle", "code_bundle_sha256", "generation_root", "generation_receipt", "parent_checkpoint", "parent_checkpoint_sha256", "launch_config", "modality_config", "token_file")
    if any(not isinstance(request.get(key), str) or not request[key] for key in required):
        raise ValueError("stage requires exact code, generation, parent, config, modality, and token paths")
    if request.get("generation_sha256") != request.get("sealed_generation_sha256"):
        raise ValueError("stage generation identity is not sealed")
    parent_sha = request.get("parent_checkpoint_sha256")
    if request.get("parent_checkpoint_repository") != PARENT_CHECKPOINT["repository"] or request.get("parent_checkpoint_revision") != PARENT_CHECKPOINT["revision"] or request.get("parent_checkpoint_subpath") != PARENT_CHECKPOINT["subpath"] or parent_sha != PARENT_CHECKPOINT["artifact_sha256"]:
        raise ValueError("stage parent checkpoint identity is not approved")
    # The token is a file transport only; no value is embedded in local JSON or
    # a command string.  Remote validation is hashes/path identity, never logs.
    return "set -eu; test -f /tmp/lehome-stage/code.bundle; sha256sum -c /tmp/lehome-stage/code.bundle.sha256; test -f /tmp/lehome-stage/generation.generation.json; lehome-train continuous-train --request /tmp/lehome-stage/continuous.json"


def remote_action(*, action: str, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner) -> dict[str, object]:
    instance_id = instance.get("instance_id")
    if type(instance_id) is not int: raise ValueError("instance receipt is invalid")
    if action == "resume" and request.get("generation_sha256") != request.get("resume_generation_sha256") or action == "resume" and request.get("config_sha256") != request.get("resume_config_sha256"):
        raise ValueError("resume requires exact generation/config identity")
    if action == "stage":
        return stage(instance=instance, request=request, runner=runner)
    elif action in {"tune", "train", "status", "resume"}:
        if action == "resume" and request.get("generation_sha256") != request.get("resume_generation_sha256") or action == "resume" and request.get("config_sha256") != request.get("resume_config_sha256"):
            raise ValueError("resume requires exact generation/config identity")
        command = "set -eu; lehome-train " + ("continuous-train" if action in {"train", "resume"} else action) + " --request /tmp/lehome-stage/continuous.json"
    else:
        raise ValueError("unsupported remote lifecycle action")
    runner((*_ssh_prefix(instance), command))
    return {"paid_action": True, "action": action, "instance_id": instance_id}


def destroy(*, instance_id: int, training_receipt: Mapping[str, object], runner: Runner | None = None) -> dict[str, object]:
    publications = training_receipt.get("immutable_checkpoint_publications")
    if training_receipt.get("kind") != "continuous_corrective_training_terminal" or training_receipt.get("instance_id") != instance_id or training_receipt.get("immutable_checkpoint_steps") != [1000, 2000] or not isinstance(publications, list) or {item.get("optimizer_step") for item in publications if isinstance(item, Mapping)} != {1000, 2000} or not all(isinstance(item, Mapping) and item.get("readback_verified") is True and isinstance(item.get("immutable_revision"), str) for item in publications):
        raise ValueError("instance-bound disposal requires two immutable checkpoints")
    if runner is not None:
        runner(("vastai", "destroy", "instance", str(instance_id), "--yes"))
        observed = _json(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
        if observed not in ({}, None): raise ValueError("destroy absence readback failed")
    return {"paid_action": runner is not None, "destroy_authorized": True, "instance_id": instance_id}


def main_for_test(argv: list[str], *, runner: Runner = _run) -> dict[str, object]:
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("prepare", "capture-offers", "rent", "stage", "tune", "train", "status", "resume", "destroy")); parser.add_argument("--request", required=True); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv); request = _load(args.request)
    if args.action == "prepare":
        generation = request.get("generation_root")
        if isinstance(generation, str):
            root = Path(generation)
            receipt = root.with_name(root.name + ".generation.json")
            if not root.is_dir() or root.is_symlink() or not receipt.is_file() or receipt.is_symlink(): raise ValueError("prepare requires a local sealed generation")
            from lehome_train.flywheel.mix import verify_generation
            verify_generation(root)
        return {"paid_action": False, "action": "prepare", "organizer_source": ORGANIZER_SOURCE, "corrective_source": CORRECTIVE_SOURCE, "request": request}
    if not args.execute: return {"paid_action": False, "action": args.action, "dry_run": True, "request": request}
    if args.action == "capture-offers": return capture_offers(runner=runner)
    if args.action == "rent": return rent(evidence=request, runner=runner)
    if args.action == "destroy": return destroy(instance_id=request.get("instance_id"), training_receipt=request, runner=runner)  # type: ignore[arg-type]
    instance = request.get("instance")
    if not isinstance(instance, Mapping): raise ValueError("remote action requires an instance receipt")
    return remote_action(action=args.action, instance=instance, request=request, runner=runner)


if __name__ == "__main__": print(json.dumps(main_for_test(__import__("sys").argv[1:]), sort_keys=True))
