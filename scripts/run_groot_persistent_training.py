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
import shutil
import subprocess
import tarfile
import tempfile
import time
from typing import Callable, Mapping

from lehome_train.hub import HubTransport, HuggingFaceHubTransport
from lehome_train.io import atomic_write_json, canonical_json_bytes, sha256_file
from lehome_train.constants import MODEL_REVISION
from lehome_train.release_manifest import validate_training_capability

ORGANIZER_SOURCE = {"repository": "lehome/dataset_challenge_merged", "revision": "17e8dee8fac294ffd21d250501d3b31bf8679042", "subdir": "four_types_merged", "mirror_repository": "kunhsiang/lehome-four-types-merged", "mirror_revision": "2ebcccf528dec91cefac0c94a9214a83028ae6cc", "manifest_sha256": "bf8fbae82002a33ff304b9a70993bdfe1c678ba9e8f798c1ad370d58969435eb"}
CORRECTIVE_SOURCE = {"repository": "ryanjin333/lehome-groot-n17-data", "revision": "e6cd1c182514c15271c805d03a646e7a4f95b17c", "prefix": "corrective-rft/b96be3db22174a12dab62a8a673f7c7d083f87aa7b50c4e03ee43e064da56c35"}
PARENT_CHECKPOINT = {"repository": "ryanjin333/lehome-groot-n17-models", "revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3", "subpath": "policies/step-12000", "archive_sha256": "0ddd4e7ce351dd2172cd1edd967293a50d02c15c0f2c21ca39db94692a57e0b5", "artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06"}
# Vast's raw expression grammar does not support a portable OR form for two
# exact SKU strings.  Query only stable numeric facts, then enforce the narrow
# WS/S allowlist on raw rows in ``_offer_gpu``.
OFFER_QUERY = "gpu_ram>=96 num_gpus=1 reliability>=0.95"
RUNTIME_PILOT_OFFER_QUERY = "cpu_arch=amd64 cpu_cores_effective>=32 cpu_ram>=64 disk_space>=120 reliability>=0.98 num_gpus=1 direct_port_count>=2 duration>=1"
RUNTIME_PILOT_READINESS_POLLS = 120
# Direct PRO6000 leases observed in Vast can remain loading beyond the generic
# one-minute readiness window.  Keep this override local to the paid direct GPU
# path; legacy rentals retain ``rent``'s twelve-poll default.
RUNTIME_GPU_WARMUP_READINESS_POLLS = 120
RUNTIME_SSH_ATTESTATION_POLLS = 12
RUNTIME_ABSENCE_READBACK_POLLS = 12
RUNTIME_GPU_RECOVERY_GRACE_SECONDS = 300
RUNTIME_GPU_RECOVERY_OBSERVATION_POLLS = 12
RUNTIME_GPU_RECOVERY_POLL_SECONDS = 5.0
# The only legacy, machine-id-less blocked direct-GPU claim this recovery path
# may release was reconciled against this explicitly approved machine.  An
# absent-offer receipt has no provider-authenticated machine row, so accepting
# a caller-controlled alternative here would turn its conservative blacklist
# into an unauthenticated bypass during the next rent.
RUNTIME_GPU_LEGACY_RECOVERY_BLACKLISTED_MACHINE_ID = 140799
RUNTIME_GPU_ARCH_TIMEOUT_SECONDS = 15
RUNTIME_GPU_PROBE_TIMEOUT_SECONDS = 600
RUNTIME_GPU_PROBE_CALL_TIMEOUT_SECONDS = 620
_RUNTIME_PILOT_OFFER_FIELDS = (
    "id", "ask_contract_id", "machine_id", "cpu_arch", "cpu_cores_effective", "cpu_ram",
    "disk_space", "disk_bw", "inet_down", "reliability", "num_gpus", "dph_total",
    "storage_total_cost", "is_bid", "rentable", "rented", "gpu_name", "gpu_ram", "driver_version",
)
_DIGEST_PREFIX = "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:"
BOOTSTRAP_TRAINER_IMAGE = (
    _DIGEST_PREFIX
    + "b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746"
)
RUNTIME_CPU_PILOT_IMAGE = (
    _DIGEST_PREFIX
    + "e4c7ac02d22f46485c1e7861a9e85b85daff14283ef62cb9e16025a3d1ecf555"
)
MAX_ACCOUNT_HOURLY_USD = 1.00
RUNTIME_GPU_RENT_CLAIM_ROOT = Path("/private/tmp/lehome-runtime-gpu-rent-claims")
Runner = Callable[[tuple[str, ...]], str]
VAST_SSH_IDENTITY = Path(
    os.environ.get("LEHOME_VAST_SSH_IDENTITY", "~/.ssh/vast_quest")
).expanduser()


class RuntimeResumeAlreadyClaimed(ValueError):
    """A successful replacement may not hydrate the same cursor twice."""


class RuntimeGpuRentOutcome(ValueError):
    """Tell the claim controller whether a failed rental is safe to retry."""

    def __init__(self, error: BaseException, *, no_lease_exists: bool) -> None:
        super().__init__(str(error))
        self.no_lease_exists = no_lease_exists


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


def _run(command: tuple[str, ...], *, timeout_seconds: int | None = None) -> str:
    completed = subprocess.run(
        command, check=True, text=True, capture_output=True, timeout=timeout_seconds,
    )
    return completed.stdout


def _run_bounded(
    runner: Runner, command: tuple[str, ...], *, timeout_seconds: int,
) -> str:
    """Bound real controller subprocesses while preserving injected test runners."""
    if timeout_seconds <= 0:
        raise ValueError("bounded runner timeout must be positive")
    if runner is _run:
        return _run(command, timeout_seconds=timeout_seconds)
    return runner(command)


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
    "id", "machine_id", "actual_status", "gpu_name", "gpu_ram", "num_gpus", "dph_total",
    "ssh_host", "ssh_port", "driver_version",
)

_RUNTIME_PILOT_INSTANCE_FIELDS = (
    "id", "actual_status", "cpu_arch", "cpu_cores_effective", "cpu_ram", "disk_space",
    "machine_id", "gpu_name", "gpu_ram", "num_gpus", "dph_total", "driver_version",
    "ssh_host", "ssh_port",
)


def _stable_instance_identity(row: Mapping[str, object]) -> str:
    """Hash only contract facts Vast does not mutate with incidental metadata."""
    return _hash(_project(row, _STABLE_INSTANCE_FIELDS))


def _runtime_pilot_instance_identity(row: Mapping[str, object]) -> str:
    """Bind the CPU-pilot receipt to the native host facts it actually proved."""
    canonical = dict(row)
    canonical["cpu_cores_effective"] = _canonical_runtime_pilot_cpu_cores(
        row.get("cpu_cores_effective")
    )
    return _hash(_project(canonical, _RUNTIME_PILOT_INSTANCE_FIELDS))


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


def _canonical_runtime_pilot_cpu_cores(value: object) -> int:
    """Accept Vast's finite integral JSON number and freeze it as an int."""
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError("runtime pilot CPU core count must be an integral numeric value")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError("runtime pilot CPU core count must be an integral numeric value")
    return int(numeric)


def _runtime_pilot_reliability(value: object) -> float:
    """Require a finite numeric reliability value from paid offer evidence."""
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError("runtime pilot reliability must be a finite numeric value")
    return float(value)


def capture_offers(*, runner: Runner, now_unix: int | None = None, ttl_seconds: int = 300) -> dict[str, object]:
    # Vast produces the total hourly quote only after the requested disk size is
    # supplied.  ``dph_total`` below is consequently the single all-in 300GB
    # quote; a separately reported storage component is evidence, not a second
    # account charge.
    offers = _json(runner, ("vastai", "--raw", "search", "offers", OFFER_QUERY, "--interruptible", "--storage", "300"))
    instances = _json(runner, ("vastai", "--raw", "show", "instances"))
    volumes = _json(runner, ("vastai", "--raw", "show", "volumes"))
    if not all(isinstance(value, list) for value in (offers, instances, volumes)): raise ValueError("provider listing is invalid")
    eligible = [
        row for row in offers if isinstance(row, Mapping) and _positive_int(row.get("machine_id"))
        and _offer_gpu(row) and row.get("num_gpus") == 1
        and type(row.get("dph_total")) in (int, float) and math.isfinite(float(row["dph_total"]))
        and float(row["dph_total"]) < 1
    ]
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
    safe_offer = _project(offer, ("id", "machine_id", "gpu_name", "gpu_ram", "num_gpus", "dph_total", "dph_base", "storage_cost", "storage_cost_per_gb", "min_bid", "driver_version", "is_bid", "image"))
    return {"schema_version": 1, "kind": "persistent_training_offer", "offer": safe_offer, "account_hourly_total_usd": total, "existing_instance_hourly_total_usd": existing_instance_total, "existing_storage_hourly_total_usd": existing_storage_total, "requested_storage_gb": 300, "requested_storage_hourly_usd": requested_storage_hourly, "storage_quote_included_in_dph_total": True, "captured_at_unix": captured, "expires_at_unix": captured + ttl_seconds, "search_mode": "interruptible"}


def capture_runtime_pilot_offer(*, runner: Runner, now_unix: int | None = None) -> dict[str, object]:
    """Capture the bounded on-demand native-x86 CPU-pilot lease evidence."""
    offers = _json(runner, ("vastai", "--raw", "search", "offers", RUNTIME_PILOT_OFFER_QUERY, "--on-demand", "--storage", "120", "--order", "dph", "--raw"))
    if not isinstance(offers, list):
        raise ValueError("runtime pilot provider offer listing is invalid")
    eligible: list[tuple[Mapping[str, object], int]] = []
    for row in offers:
        if not isinstance(row, Mapping):
            continue
        try:
            cores = _canonical_runtime_pilot_cpu_cores(row.get("cpu_cores_effective"))
            reliability = _runtime_pilot_reliability(row.get("reliability"))
        except ValueError:
            continue
        if row.get("cpu_arch") == "amd64" and cores >= 32 and type(row.get("cpu_ram")) in (int, float) and float(row["cpu_ram"]) >= 64000 and type(row.get("disk_space")) in (int, float) and float(row["disk_space"]) >= 120 and reliability >= .98 and row.get("num_gpus") == 1 and row.get("is_bid") is False and row.get("rentable") is True and row.get("rented") is False and type(row.get("id")) is int and type(row.get("dph_total")) in (int, float):
            eligible.append((row, cores))
    if not eligible:
        raise ValueError("no on-demand native x86 runtime pilot offer is eligible")
    offer, cores = min(eligible, key=lambda item: (float(item[0]["dph_total"]), -float(item[0].get("disk_bw", 0)), int(item[0]["id"])))
    total = _live_account_total(runner=runner) + float(offer["dph_total"])
    _require_account_cap(total, label="runtime pilot projected")
    captured = int(time.time()) if now_unix is None else now_unix
    canonical_offer = _project(offer, _RUNTIME_PILOT_OFFER_FIELDS)
    canonical_offer["cpu_cores_effective"] = cores
    return {"schema_version": 1, "kind": "runtime_mixture_cpu_pilot_offer", "offer": canonical_offer, "raw_offer_sha256": _hash(dict(offer)), "account_hourly_total_usd": total, "captured_at_unix": captured, "expires_at_unix": captured + 300, "search_mode": "on_demand", "platform_arch": "amd64", "storage_gb": 120, "trainer_image": RUNTIME_CPU_PILOT_IMAGE, "image_digest": RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2]}


def _runtime_pilot_offer(evidence: Mapping[str, object]) -> Mapping[str, object]:
    """Validate the short-lived, on-demand CPU pilot offer receipt."""
    offer = evidence.get("offer")
    try:
        reliability = _runtime_pilot_reliability(
            offer.get("reliability") if isinstance(offer, Mapping) else None
        )
    except ValueError:
        raise ValueError("runtime CPU pilot offer evidence is invalid") from None
    if (
        evidence.get("schema_version") != 1
        or evidence.get("kind") != "runtime_mixture_cpu_pilot_offer"
        or evidence.get("search_mode") != "on_demand"
        or evidence.get("platform_arch") != "amd64"
        or evidence.get("storage_gb") != 120
        or evidence.get("trainer_image") != RUNTIME_CPU_PILOT_IMAGE
        or evidence.get("image_digest") != RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2]
        or type(evidence.get("expires_at_unix")) is not int
        or int(evidence["expires_at_unix"]) < int(time.time())
        or not isinstance(offer, Mapping)
        or type(offer.get("id")) is not int
        or offer.get("cpu_arch") != "amd64"
        or type(offer.get("cpu_cores_effective")) is not int
        or int(offer["cpu_cores_effective"]) < 32
        or type(offer.get("machine_id")) is not int
        or not isinstance(offer.get("gpu_name"), str) or not offer["gpu_name"]
        or type(offer.get("gpu_ram")) not in (int, float) or float(offer["gpu_ram"]) <= 0
        or not isinstance(offer.get("driver_version"), str) or not offer["driver_version"]
        or type(offer.get("cpu_ram")) not in (int, float)
        or float(offer["cpu_ram"]) < 64000
        or type(offer.get("disk_space")) not in (int, float)
        or float(offer["disk_space"]) < 120
        or reliability < .98
        or offer.get("num_gpus") != 1
        or offer.get("is_bid") is not False
        or offer.get("rentable") is not True
        or offer.get("rented") is not False
        or type(offer.get("dph_total")) not in (int, float)
        or re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("raw_offer_sha256"))) is None
    ):
        raise ValueError("runtime CPU pilot offer evidence is invalid")
    _require_account_cap(evidence.get("account_hourly_total_usd"), label="runtime pilot offer evidence")
    return offer


def _runtime_pilot_live_matches(*, live: Mapping[str, object], instance_id: int, offer: Mapping[str, object]) -> bool:
    try:
        cores = _canonical_runtime_pilot_cpu_cores(live.get("cpu_cores_effective"))
    except ValueError:
        return False
    reliability = live.get("reliability")
    try:
        reliability_matches = reliability is None or _runtime_pilot_reliability(reliability) >= .98
    except ValueError:
        reliability_matches = False
    driver_matches = (
        not isinstance(offer.get("driver_version"), str)
        or not offer["driver_version"]
        or live.get("driver_version") == offer["driver_version"]
    )
    return (
        live.get("id") == instance_id
        and live.get("actual_status") == "running"
        and live.get("cpu_arch") == "amd64"
        and cores >= 32
        and type(live.get("cpu_ram")) in (int, float)
        and float(live["cpu_ram"]) >= float(offer.get("cpu_ram", 64000))
        and type(live.get("disk_space")) in (int, float)
        and float(live["disk_space"]) >= 120
        and live.get("machine_id") == offer.get("machine_id")
        and live.get("gpu_name") == offer.get("gpu_name")
        and live.get("gpu_ram") == offer.get("gpu_ram")
        and live.get("num_gpus") == offer.get("num_gpus") == 1
        and live.get("dph_total") == offer.get("dph_total")
        and driver_matches and reliability_matches
        and isinstance(live.get("ssh_host"), str)
        and bool(live.get("ssh_host"))
        and type(live.get("ssh_port")) is int
        and int(live["ssh_port"]) > 0
    )


def _await_runtime_instance_absence(
    *, instance_id: int, runner: Runner, max_polls: int = RUNTIME_ABSENCE_READBACK_POLLS,
    sleep: Callable[[float], None] = _bounded_sleep,
) -> bool:
    """Wait a bounded interval for Vast's eventually-consistent destroy view."""
    if type(max_polls) is not int or max_polls <= 0:
        raise ValueError("runtime absence readback poll bound is invalid")
    for poll in range(max_polls):
        if _runtime_instance_is_absent(
            _json(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
        ):
            return True
        if poll + 1 < max_polls:
            sleep(5.0)
    return False


def _runtime_instance_is_absent(value: object) -> bool:
    """Recognize only the canonical absent forms returned by Vast raw JSON."""
    return value is None or (type(value) is dict and not value) or (
        type(value) is dict and set(value) == {"instances"} and value["instances"] is None
    )


def _runtime_pilot_cleanup(
    *, instance_id: int, runner: Runner,
    max_absence_polls: int = RUNTIME_ABSENCE_READBACK_POLLS,
    sleep: Callable[[float], None] = _bounded_sleep,
) -> None:
    """Destroy only the instance that this invocation just created."""
    runner(("vastai", "destroy", "instance", str(instance_id), "--yes"))
    if not _await_runtime_instance_absence(
        instance_id=instance_id, runner=runner, max_polls=max_absence_polls, sleep=sleep,
    ):
        raise ValueError("runtime CPU pilot cleanup absence readback failed")


def _runtime_abort_cleanup(
    *, instance: Mapping[str, object], request: Mapping[str, object], error: BaseException,
    runner: Runner, max_absence_polls: int = RUNTIME_ABSENCE_READBACK_POLLS,
    sleep: Callable[[float], None] = _bounded_sleep,
) -> None:
    """Persist redacted abort evidence, then destroy only this just-rented lease."""
    instance_id = instance.get("instance_id")
    if type(instance_id) is not int:
        return
    receipt = {
        "schema_version": 1, "kind": "runtime_mixture_abort_cleanup", "instance_id": instance_id,
        "provider_response_sha256": instance.get("provider_response_sha256"),
        "code_revision": request.get("code_revision"), "code_bundle_sha256": request.get("code_bundle_sha256"),
        "parent_checkpoint_artifact_sha256": PARENT_CHECKPOINT["artifact_sha256"],
        # Failure receipts deliberately never serialize command output or exception
        # text: either can contain a Hub token or a presigned URL.
        "error_type": type(error).__name__, "error": "redacted remote failure",
        "disposable": False,
    }
    output = _runtime_failure_receipt_path(request)
    def record(extra: Mapping[str, object]) -> None:
        if not output.exists() and not output.is_symlink():
            atomic_write_json(output, receipt | dict(extra))
    try:
        runner(("vastai", "destroy", "instance", str(instance_id), "--yes"))
    except BaseException:
        record({"cleanup_status": "destroy_failed"})
        raise RuntimeError("runtime abort cleanup could not destroy the newly rented instance") from error
    try:
        absent = _await_runtime_instance_absence(
            instance_id=instance_id, runner=runner,
            max_polls=max_absence_polls, sleep=sleep,
        )
    except BaseException:
        record({"cleanup_status": "absence_unverified"})
        raise RuntimeError("runtime abort cleanup could not verify instance absence") from error
    if not absent:
        record({"cleanup_status": "absence_unverified"})
        raise RuntimeError("runtime abort cleanup did not verify instance absence") from error
    record({"cleanup_status": "destroyed_and_absent"})


def _runtime_failure_receipt_path(request: Mapping[str, object]) -> Path:
    """Require a fresh, caller-chosen durable receipt before paid work begins."""
    output = request.get("failure_receipt")
    if type(output) is not str:
        raise ValueError("runtime paid action requires an absent absolute failure_receipt")
    path = Path(output)
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("runtime paid action requires an absent absolute failure_receipt")
    return path


def _runtime_abort_on_failure(
    *, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner,
    operation: Callable[[], dict[str, object]],
) -> dict[str, object]:
    """Run a paid runtime step and abort-clean only its freshly rented lease."""
    _runtime_failure_receipt_path(request)
    if instance.get("kind") not in {
        "runtime_mixture_cpu_pilot_instance", "runtime_mixture_gpu_warmup_instance",
    }:
        return operation()
    try:
        return operation()
    except RuntimeResumeAlreadyClaimed:
        # This is an idempotency/replay rejection for the active winner, not a
        # failed paid action.  Its instance and failure receipt must stay put.
        raise
    except BaseException as error:
        _runtime_abort_cleanup(instance=instance, request=request, error=error, runner=runner)
        raise


def rent_runtime_cpu_pilot(
    *, evidence: Mapping[str, object], runner: Runner, max_readiness_polls: int = RUNTIME_PILOT_READINESS_POLLS,
    sleep: Callable[[float], None] = _bounded_sleep,
) -> dict[str, object]:
    """Create one bounded native-x86 lease and remove it on every failed proof."""
    _require_vast_ssh_identity()
    _runtime_failure_receipt_path(evidence)
    _runtime_campaign_binding(evidence)
    offer = _runtime_pilot_offer(evidence)
    _require_account_cap(
        _live_account_total(runner=runner) + float(offer["dph_total"]),
        label="fresh runtime pilot projection",
    )
    created = _json(runner, (
        "vastai", "--raw", "create", "instance", str(offer["id"]), "--image",
        RUNTIME_CPU_PILOT_IMAGE, "--disk", "120", "--ssh", "--direct", "--cancel-unavail",
        "--env", "-e LEHOME_TRAIN_IMAGE=" + RUNTIME_CPU_PILOT_IMAGE,
    ))
    if not isinstance(created, Mapping) or type(created.get("new_contract")) is not int:
        raise ValueError("runtime CPU pilot provider did not return an instance ID")
    instance_id = int(created["new_contract"])
    try:
        live: object = {}
        for _ in range(max_readiness_polls):
            live = _json(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
            if isinstance(live, Mapping) and live.get("actual_status") == "running" and live.get("ssh_host"):
                break
            sleep(5.0)
        else:
            raise ValueError("runtime CPU pilot readiness poll timed out")
        if not isinstance(live, Mapping) or not _runtime_pilot_live_matches(
            live=live, instance_id=instance_id, offer=offer,
        ):
            raise ValueError("runtime CPU pilot instance readback does not match accepted offer")
        instance = {
            "schema_version": 1, "kind": "runtime_mixture_cpu_pilot_instance",
            "instance_id": instance_id, "host": live["ssh_host"], "port": live["ssh_port"],
            "platform_arch": "x86_64", "trainer_image": RUNTIME_CPU_PILOT_IMAGE,
            "image_digest": RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2],
            "offer_evidence_sha256": _hash(evidence),
            "provider_response_sha256": _runtime_pilot_instance_identity(live),
            "account_hourly_total_usd": _require_account_cap(
                evidence.get("account_hourly_total_usd"), label="runtime pilot offer evidence",
            ),
        }
        _await_platform_arch_attestation(instance=instance, runner=runner, sleep=sleep)
        return instance
    except BaseException as error:
        _runtime_abort_cleanup(
            instance=locals().get("instance", {
                "instance_id": instance_id,
                "provider_response_sha256": _hash(created),
            }),
            request=evidence, error=error, runner=runner, sleep=sleep,
        )
        raise


def rent(*, evidence: Mapping[str, object], runner: Runner, max_readiness_polls: int = 12, sleep: Callable[[float], None] = _bounded_sleep, require_capability: bool = True, abort_request: Mapping[str, object] | None = None) -> dict[str, object]:
    def outcome(error: BaseException, *, no_lease_exists: bool) -> None:
        if abort_request is not None:
            raise RuntimeGpuRentOutcome(error, no_lease_exists=no_lease_exists) from error
        raise error

    try:
        _require_vast_ssh_identity()
        offer = evidence.get("offer")
        if (
            not isinstance(offer, Mapping) or not _positive_int(offer.get("id"))
            or (abort_request is not None and not _positive_int(offer.get("machine_id")))
        ): raise ValueError("offer evidence is invalid")
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
        _require_account_cap(
            _live_account_total(runner=runner) + float(quoted_offer_hourly),
            label="fresh rental projection",
        )
        bid = offer.get("min_bid", offer.get("dph_total"))
        if type(bid) not in (int, float) or float(bid) >= 1: raise ValueError("offer bid price is invalid")
    except BaseException as error:
        outcome(error, no_lease_exists=True)
        raise AssertionError("unreachable")
    try:
        created = _json(runner, ("vastai", "--raw", "create", "instance", str(offer["id"]), "--image", image, "--disk", "300", "--bid_price", str(bid), "--ssh", "--direct", "--cancel-unavail", "--env", "-e LEHOME_TRAIN_IMAGE=" + image))
    except BaseException as error:
        outcome(error, no_lease_exists=False)
        raise AssertionError("unreachable")
    if not isinstance(created, Mapping) or type(created.get("new_contract")) is not int:
        outcome(ValueError("provider did not return an instance ID"), no_lease_exists=False)
        raise AssertionError("unreachable")
    instance_id = created["new_contract"]
    live: object = {}
    try:
        for _ in range(max_readiness_polls):
            live = _json(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
            if isinstance(live, Mapping) and live.get("actual_status") == "running" and live.get("ssh_host"):
                break
            sleep(5.0)
        else:
            raise ValueError("instance readiness poll timed out")
        if not isinstance(live, Mapping) or live.get("id") != instance_id or (abort_request is not None and live.get("machine_id") != offer.get("machine_id")) or not _offer_gpu(live) or live.get("num_gpus") != 1 or not live.get("ssh_host") or type(live.get("ssh_port")) is not int or float(live.get("dph_total", 99)) >= 1:
            raise ValueError("instance readback does not match accepted offer")
        return {"schema_version": 1, "kind": "persistent_training_instance", "instance_id": instance_id, "machine_id": live.get("machine_id"), "host": live.get("ssh_host"), "port": live.get("ssh_port"), "trainer_image": image, "offer_evidence_sha256": _hash(evidence), "provider_response_sha256": _stable_instance_identity(live), "account_hourly_total_usd": _require_account_cap(evidence.get("account_hourly_total_usd"), label="offer evidence")}
    except BaseException as error:
        if abort_request is not None:
            _runtime_abort_cleanup(
                instance={"instance_id": instance_id, "provider_response_sha256": _hash(live if live else created)},
                request=abort_request, error=error, runner=runner,
            )
            outcome(error, no_lease_exists=True)
            raise AssertionError("unreachable")
        runner(("vastai", "destroy", "instance", str(instance_id), "--yes"))
        absent = _json(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
        if absent not in ({}, None):
            raise ValueError("post-create cleanup absence readback failed")
        raise


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
        runner((*_scp_prefix(instance), str(bundle_path), "root@" + str(instance["host"]) + ":" + remote + "/code.bundle"))
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


def _require_vast_ssh_identity() -> Path:
    """Require the explicitly provisioned Vast private key without reading it."""
    if VAST_SSH_IDENTITY.is_symlink() or not VAST_SSH_IDENTITY.is_file():
        raise ValueError("Vast SSH identity must be a regular file")
    return VAST_SSH_IDENTITY


def _ssh_prefix(instance: Mapping[str, object]) -> tuple[str, ...]:
    host, port = instance.get("host"), instance.get("port")
    if not isinstance(host, str) or not host or type(port) is not int or port <= 0: raise ValueError("instance SSH receipt is invalid")
    identity = _require_vast_ssh_identity()
    return ("ssh", "-o", "IdentitiesOnly=yes", "-i", str(identity), "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port), "root@" + host)


def _scp_prefix(instance: Mapping[str, object], *, recursive: bool = False) -> tuple[str, ...]:
    host, port = instance.get("host"), instance.get("port")
    if not isinstance(host, str) or not host or type(port) is not int or port <= 0: raise ValueError("instance SSH receipt is invalid")
    identity = _require_vast_ssh_identity()
    return (
        "scp", *( ("-r",) if recursive else () ), "-o", "IdentitiesOnly=yes",
        "-i", str(identity), "-o", "StrictHostKeyChecking=accept-new", "-P", str(port),
    )


def _attest_platform_arch(
    instance: Mapping[str, object], *, runner: Runner,
    ssh_connection_timeout_seconds: int | None = None,
) -> str:
    """Accept native x86 only after the just-rented host reports it over SSH."""
    prefix = _ssh_prefix(instance)
    if ssh_connection_timeout_seconds is not None:
        if type(ssh_connection_timeout_seconds) is not int or ssh_connection_timeout_seconds <= 0:
            raise ValueError("platform attestation SSH connection timeout is invalid")
        prefix = (
            *prefix[:-1], "-o", f"ConnectTimeout={ssh_connection_timeout_seconds}", prefix[-1],
        )
    try:
        command = (*prefix, "set -eu; uname -m")
        if ssh_connection_timeout_seconds is None:
            arch = runner(command).strip()
        else:
            arch = _run_bounded(
                runner, command, timeout_seconds=ssh_connection_timeout_seconds,
            ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, TimeoutError) as error:
        raise ValueError("native platform attestation is unavailable") from error
    if arch not in {"x86_64", "amd64"}:
        raise ValueError("runtime mixture requires native x86_64 platform proof")
    return "x86_64"


def _await_platform_arch_attestation(
    *, instance: Mapping[str, object], runner: Runner,
    max_polls: int = RUNTIME_SSH_ATTESTATION_POLLS,
    sleep: Callable[[float], None] = _bounded_sleep,
) -> str:
    """Allow a just-running lease a short, bounded interval to accept SSH."""
    if type(max_polls) is not int or max_polls <= 0:
        raise ValueError("runtime SSH attestation poll bound is invalid")
    for poll in range(max_polls):
        try:
            return _attest_platform_arch(
                instance, runner=runner, ssh_connection_timeout_seconds=5,
            )
        except ValueError as error:
            if not isinstance(error.__cause__, (subprocess.CalledProcessError, OSError, TimeoutError)):
                raise
            if poll + 1 == max_polls:
                raise
            sleep(5.0)
    raise AssertionError("unreachable runtime SSH attestation state")


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


def _verify_reviewed_code_bundle(bundle: Path, receipt: Path, revision: object) -> str:
    """Require a real clean Git bundle at the reviewed immutable revision."""
    digest = _verify_code_bundle_receipt(bundle, receipt)
    if type(revision) is not str or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("reviewed code bundle requires an immutable Git revision")
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="runtime-code-bundle-") as temporary:
        root = Path(temporary) / "source"
        try:
            subprocess.run(("git", "clone", "--quiet", str(bundle), str(root)), check=True, text=True, capture_output=True)
            subprocess.run(("git", "-C", str(root), "checkout", "--quiet", "--detach", revision), check=True, text=True, capture_output=True)
            head = subprocess.run(("git", "-C", str(root), "rev-parse", "HEAD"), check=True, text=True, capture_output=True).stdout.strip()
            dirty = subprocess.run(("git", "-C", str(root), "status", "--porcelain"), check=True, text=True, capture_output=True).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("reviewed code bundle is not a Git bundle containing the requested revision") from error
        if head != revision or dirty:
            raise ValueError("reviewed code bundle does not resolve to a clean requested revision")
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
    runner((*_scp_prefix(instance, recursive=True), str(generation_root), "root@" + str(instance["host"]) + ":" + remote_dir + "/generation"))
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
        runner((*_scp_prefix(instance), str(source), "root@" + str(instance["host"]) + ":" + remote_dir + "/" + remote_name))
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
        or (kind == "rollout" and prefix != "rollouts/round-1")
        or (kind == "deployment" and not _runtime_deployment_receipt_is_canonical(value, prefix))
    ):
        raise ValueError("runtime mixture receipt is not an authenticated campaign binding")
    return value, sha256_file(path)


def _runtime_deployment_receipt_is_canonical(
    receipt: Mapping[str, object], prefix: object,
) -> bool:
    """Keep provider rent behind the same exact deployment receipt schema as hydration."""
    mixture_id = receipt.get("mixture_id")
    pending = receipt.get("pending_receipt_sha256")
    entries = receipt.get("artifact_entries")
    if (
        type(mixture_id) is not str
        or re.fullmatch(r"[0-9a-f]{64}", mixture_id) is None
        or prefix != "mixtures/" + mixture_id
        or type(pending) is not str
        or re.fullmatch(r"[0-9a-f]{64}", pending) is None
        or not isinstance(entries, list)
        or not entries
    ):
        return False
    paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"relative_path", "sha256", "byte_size"}:
            return False
        relative, digest, size = entry.get("relative_path"), entry.get("sha256"), entry.get("byte_size")
        path = Path(relative) if type(relative) is str else None
        if (
            type(relative) is not str or not relative or path is None
            or path.is_absolute() or ".." in path.parts or "." in path.parts
            or type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(size) is not int or size < 0 or relative in paths
        ):
            return False
        paths.add(relative)
    return {"mixture.json", "windows.json", "mixture-normalization.json"} <= paths


def _runtime_parent_checkpoint(request: Mapping[str, object]) -> Path:
    """Validate the exact local parent archive before a direct-GPU lease mutates."""
    value = request.get("parent_checkpoint")
    if type(value) is not str:
        raise ValueError("runtime parent checkpoint path is required")
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise ValueError("runtime parent checkpoint must be a regular archive")
    _safe_archive(path, "runtime parent checkpoint")
    if sha256_file(path) != PARENT_CHECKPOINT["archive_sha256"]:
        raise ValueError("runtime parent checkpoint archive is incompatible")
    return path


def _runtime_campaign_binding(request: Mapping[str, object]) -> dict[str, object]:
    """Validate immutable mixture/code evidence before any paid provider call."""
    if (
        not isinstance(request.get("code_revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", str(request.get("code_revision"))) is None
        or not isinstance(request.get("code_bundle_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(request.get("code_bundle_sha256"))) is None
    ):
        raise ValueError("runtime mixture production requires immutable code bindings")
    bc, bc_sha = _runtime_receipt(request.get("bc_readback_receipt"), kind="bc")
    rollout, rollout_sha = _runtime_receipt(request.get("rollout_readback_receipt"), kind="rollout")
    deployment, deployment_sha = _runtime_receipt(request.get("deployment_receipt"), kind="deployment")
    return {"bc": bc, "bc_receipt_sha256": bc_sha, "rollout": rollout,
            "rollout_receipt_sha256": rollout_sha, "deployment": deployment,
            "deployment_receipt_sha256": deployment_sha}


def _runtime_identity(instance: Mapping[str, object], request: Mapping[str, object]) -> dict[str, object]:
    if (
        instance.get("platform_arch") != "x86_64"
        or instance.get("trainer_image") != BOOTSTRAP_TRAINER_IMAGE
        or type(instance.get("instance_id")) is not int
    ):
        raise ValueError("runtime mixture production requires native x86_64 and the approved pinned image")
    return _runtime_campaign_binding(request)


def _runtime_cpu_pilot_identity(instance: Mapping[str, object], request: Mapping[str, object]) -> dict[str, object]:
    """Validate the CPU-only pilot lease without broadening GPU acceptance."""
    if (
        instance.get("platform_arch") != "x86_64"
        or instance.get("trainer_image") != RUNTIME_CPU_PILOT_IMAGE
        or instance.get("image_digest") != RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2]
        or type(instance.get("instance_id")) is not int
    ):
        raise ValueError("runtime CPU pilot requires native x86_64 and the approved CPU-only pinned image")
    return _runtime_campaign_binding(request)


def _runtime_hydration_identity(
    instance: Mapping[str, object], request: Mapping[str, object],
) -> dict[str, object]:
    """Select the exact lease identity admitted to runtime hydration."""
    if instance.get("kind") == "runtime_mixture_cpu_pilot_instance":
        _runtime_pilot_instance(instance)
        return _runtime_cpu_pilot_identity(instance, request)
    return _runtime_identity(instance, request)


def _validated_runtime_pilot_value(receipt: Mapping[str, object]) -> dict[str, object]:
    """Validate only the current schema4 CPU-only pilot CLI output."""
    value = dict(receipt)
    evidence = value.get("authenticated_evidence")
    rows = value.get("timing_rows")
    common_invalid = (
        value.get("kind") != "runtime_mixture_loader_pilot"
        or value.get("model_loaded") is not False or value.get("gpu_initialized") is not False
        or value.get("native_x86_required") is not True
        or value.get("canonical_worker_counts") != [0, 4, 8, 16, 24]
        or value.get("canonical_completion") is not True
        or not isinstance(evidence, Mapping)
        or set(evidence) != {"provider_instance_id", "provider_response_sha256", "platform_arch", "image_digest", "code_revision", "code_bundle_sha256", "bc_revision", "rollout_revision", "deployment_revision"}
        or type(evidence.get("provider_instance_id")) is not int
        or evidence.get("platform_arch") != "x86_64"
        or evidence.get("image_digest") != RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2]
        or any(re.fullmatch(r"[0-9a-f]{40}", str(evidence.get(key))) is None for key in ("code_revision", "bc_revision", "rollout_revision", "deployment_revision"))
        or re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("provider_response_sha256"))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("code_bundle_sha256"))) is None
        or not isinstance(rows, list) or [row.get("worker_count") if isinstance(row, Mapping) else None for row in rows] != [0, 4, 8, 16, 24]
        or any(not isinstance(row, Mapping) or set(row) != {"worker_count", "decoded_samples", "seconds", "samples_per_second", "host_cpu_seconds", "host_max_rss_mib", "latency_seconds_p50", "latency_seconds_p95"} or type(row.get("decoded_samples")) is not int or row["decoded_samples"] < 100 or any(type(row.get(key)) not in (int, float) or float(row[key]) < 0 for key in ("seconds", "samples_per_second", "host_cpu_seconds", "host_max_rss_mib", "latency_seconds_p50", "latency_seconds_p95")) for row in rows)
    )
    version = value.get("schema_version")
    current_invalid = version != 4 or (
        set(value) != {
            "schema_version", "kind", "model_loaded", "gpu_initialized", "processor_contract",
            "representative", "sample_count_per_worker", "worker_counts", "canonical_worker_counts",
            "loader_throughput", "timing_rows", "authenticated_evidence", "cache_cap",
            "native_x86_required", "timeout_seconds", "canonical_completion",
        }
        or value.get("processor_contract") != "pinned_processor_integration_required"
        or value.get("worker_counts") != [0, 4, 8, 16, 24]
        or type(value.get("sample_count_per_worker")) is not int
        or int(value["sample_count_per_worker"]) < 100
        or not isinstance(value.get("loader_throughput"), Mapping)
        or not isinstance(value.get("representative"), Mapping)
        or value["representative"].get("three_cameras") is not True
        or value["representative"].get("action_horizon") != 16
        or type(value.get("cache_cap")) is not int
        or type(value.get("timeout_seconds")) not in (int, float)
    )
    if common_invalid or current_invalid:
        raise ValueError("runtime mixture production requires an authenticated measured canonical CPU-only pilot receipt")
    return value


def _validated_runtime_pilot(path_value: object) -> dict[str, object]:
    if type(path_value) is not str:
        raise ValueError("runtime mixture production requires a pilot receipt")
    return _validated_runtime_pilot_value(
        _load_regular_json(Path(path_value), "runtime mixture pilot receipt"),
    )


def _positive_int(value: object) -> bool:
    return type(value) is int and int(value) > 0


def _runtime_gpu_rent_preflight(
    request: Mapping[str, object], *, recovery_safe: bool = False,
) -> dict[str, object]:
    """Validate direct-GPU immutable source bindings before provider create."""
    if "training_capability" in request:
        raise ValueError("direct runtime GPU rent derives capability on its single lease")
    identity = _runtime_campaign_binding(request)
    _runtime_parent_checkpoint(request)
    _require_vast_ssh_identity()
    offer = request.get("offer")
    if not isinstance(offer, Mapping) or not _positive_int(offer.get("id")):
        raise ValueError("runtime GPU rent requires a fresh concrete offer before claiming a lease")
    if (
        request.get("search_mode") != "interruptible"
        or type(request.get("expires_at_unix")) is not int
        or (not recovery_safe and request["expires_at_unix"] < int(time.time()))
        or _trainer_image(request.get("trainer_image")) != BOOTSTRAP_TRAINER_IMAGE
    ):
        raise ValueError("runtime GPU rent offer is not a fresh pinned interruptible lease")
    _require_account_cap(request.get("account_hourly_total_usd"), label="runtime GPU offer evidence")
    quote = offer.get("dph_total")
    bid = offer.get("min_bid", quote)
    if (
        type(quote) not in (int, float) or not math.isfinite(float(quote)) or float(quote) < 0
        or type(bid) not in (int, float) or not math.isfinite(float(bid)) or float(bid) >= 1
        or not recovery_safe and not _positive_int(offer.get("machine_id"))
    ):
        raise ValueError("runtime GPU rent offer price is invalid")
    bundle = request.get("code_bundle")
    receipt = request.get("code_bundle_sha256_file")
    if type(bundle) is not str or type(receipt) is not str:
        raise ValueError("runtime GPU rent requires the reviewed code bundle and checksum")
    if _verify_reviewed_code_bundle(
        Path(bundle), Path(receipt), request.get("code_revision"),
    ) != request.get("code_bundle_sha256"):
        raise ValueError("runtime GPU rent code bundle differs from its immutable binding")
    output = request.get("bootstrap_capability_receipt")
    if not recovery_safe and (
        type(output) is not str or not Path(output).is_absolute()
        or Path(output).exists() or Path(output).is_symlink()
    ):
        raise ValueError("runtime GPU rent requires an absent bootstrap capability receipt")
    claim_path = _runtime_gpu_rent_claim_path(
        request=request, identity=identity, allow_held=recovery_safe,
    )
    if not recovery_safe:
        _validate_runtime_gpu_recovery_for_new_rent(
            request=request, identity=identity, claim_path=claim_path,
        )
    return identity


def _runtime_gpu_rent_claim_path(
    *, request: Mapping[str, object], identity: Mapping[str, object],
    require_request_path: bool = True, allow_held: bool = False,
) -> Path:
    """Use one controller-owned claim namespace per immutable direct campaign."""
    identity_key = {
        "code_revision": request.get("code_revision"),
        "code_bundle_sha256": request.get("code_bundle_sha256"),
        "trainer_image": request.get("trainer_image"),
        "bc_revision": identity["bc"]["immutable_revision"],
        "bc_receipt_sha256": identity["bc_receipt_sha256"],
        "rollout_revision": identity["rollout"]["immutable_revision"],
        "rollout_prefix": identity["rollout"]["remote_prefix"],
        "rollout_receipt_sha256": identity["rollout_receipt_sha256"],
        "deployment_revision": identity["deployment"]["immutable_revision"],
        "mixture_id": identity["deployment"]["mixture_id"],
        "deployment_receipt_sha256": identity["deployment_receipt_sha256"],
        "parent_archive_sha256": PARENT_CHECKPOINT["archive_sha256"],
        "parent_checkpoint_artifact_sha256": PARENT_CHECKPOINT["artifact_sha256"],
    }
    root = RUNTIME_GPU_RENT_CLAIM_ROOT
    if root.is_symlink():
        raise ValueError("runtime GPU rent claim root is unsafe")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("runtime GPU rent claim root is unsafe")
    path = root / (_hash(identity_key) + ".json")
    if require_request_path and request.get("rent_claim_receipt") != str(path):
        raise ValueError("runtime GPU rent claim receipt must use the controller-owned canonical path")
    if not allow_held and (path.exists() or path.is_symlink()):
        raise ValueError("runtime GPU rent claim is already held; no provider action was taken")
    return path


def _runtime_gpu_rent_claim(
    *, path: Path, request: Mapping[str, object], identity: Mapping[str, object],
) -> dict[str, object]:
    """Atomically admit exactly one controller to create a direct GPU lease."""
    offer = request["offer"]
    assert isinstance(offer, Mapping)
    claim = {
        "schema_version": 1, "kind": "runtime_mixture_gpu_rent_claim", "status": "claimed",
        "request_sha256": _hash({key: value for key, value in request.items() if key != "rent_claim_receipt"}),
        "offer_sha256": _hash(dict(offer)),
        "bootstrap_capability_receipt": request["bootstrap_capability_receipt"],
        "code_revision": request["code_revision"], "code_bundle_sha256": request["code_bundle_sha256"],
        "bc_revision": identity["bc"]["immutable_revision"],
        "bc_receipt_sha256": identity["bc_receipt_sha256"],
        "rollout_revision": identity["rollout"]["immutable_revision"],
        "rollout_prefix": identity["rollout"]["remote_prefix"],
        "rollout_receipt_sha256": identity["rollout_receipt_sha256"],
        "deployment_revision": identity["deployment"]["immutable_revision"],
        "mixture_id": identity["deployment"]["mixture_id"],
        "deployment_receipt_sha256": identity["deployment_receipt_sha256"],
        "parent_archive_sha256": PARENT_CHECKPOINT["archive_sha256"],
        "parent_checkpoint_artifact_sha256": PARENT_CHECKPOINT["artifact_sha256"],
    }
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("runtime GPU rent claim is already held; no provider action was taken") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(claim))
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


def _release_runtime_gpu_rent_claim(*, path: Path, claim: Mapping[str, object]) -> None:
    """Release only this failed controller's claim after any lease is absent."""
    if dict(_load_regular_json(path, "runtime GPU rent claim")) != dict(claim):
        raise RuntimeError("runtime GPU rent claim changed before failure cleanup")
    path.unlink()


def _terminalize_runtime_gpu_rent_claim(
    *, path: Path, claim: Mapping[str, object], instance: Mapping[str, object],
    capability_sha256: str,
) -> None:
    """Keep a durable winner record so replay cannot rent another GPU."""
    if dict(_load_regular_json(path, "runtime GPU rent claim")) != dict(claim):
        raise RuntimeError("runtime GPU rent claim changed before success terminalization")
    atomic_write_json(path, dict(claim) | {
        "status": "succeeded", "instance_id": instance["instance_id"],
        "provider_response_sha256": instance["provider_response_sha256"],
        "capability_sha256": capability_sha256,
    })


def _block_runtime_gpu_rent_claim(
    *, path: Path, claim: Mapping[str, object], error: BaseException,
) -> None:
    """Retain an ambiguous create claim so a retry cannot double-rent."""
    if dict(_load_regular_json(path, "runtime GPU rent claim")) != dict(claim):
        raise RuntimeError("runtime GPU rent claim changed before blocked terminalization")
    atomic_write_json(path, dict(claim) | {
        "status": "blocked", "error_type": type(error).__name__,
    })


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_json(path: Path, value: Mapping[str, object]) -> None:
    """Create controller evidence once, and make both file and name durable."""
    _write_exclusive_bytes(path, canonical_json_bytes(dict(value)))


def _write_exclusive_bytes(path: Path, payload: bytes) -> None:
    """Create immutable byte evidence exactly as it was authenticated."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_parent(path)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _replace_json_durably(path: Path, value: Mapping[str, object]) -> None:
    """Atomically replace an already-owned recovery receipt and fsync its name."""
    temporary = path.with_name("." + path.name + ".tmp-" + hashlib.sha256(os.urandom(16)).hexdigest())
    try:
        _write_exclusive_json(temporary, value)
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_gpu_claim_immutable(
    *, claim: Mapping[str, object], identity: Mapping[str, object], request: Mapping[str, object],
) -> bool:
    expected = {
        "schema_version": 1, "kind": "runtime_mixture_gpu_rent_claim",
        "code_revision": request.get("code_revision"),
        "code_bundle_sha256": request.get("code_bundle_sha256"),
        "bc_revision": identity["bc"]["immutable_revision"],
        "bc_receipt_sha256": identity["bc_receipt_sha256"],
        "rollout_revision": identity["rollout"]["immutable_revision"],
        "rollout_prefix": identity["rollout"]["remote_prefix"],
        "rollout_receipt_sha256": identity["rollout_receipt_sha256"],
        "deployment_revision": identity["deployment"]["immutable_revision"],
        "mixture_id": identity["deployment"]["mixture_id"],
        "deployment_receipt_sha256": identity["deployment_receipt_sha256"],
        "parent_archive_sha256": PARENT_CHECKPOINT["archive_sha256"],
        "parent_checkpoint_artifact_sha256": PARENT_CHECKPOINT["artifact_sha256"],
    }
    return all(claim.get(key) == value for key, value in expected.items())


def _valid_blocked_runtime_gpu_claim(
    *, claim: Mapping[str, object], identity: Mapping[str, object], request: Mapping[str, object],
) -> bool:
    fields = {
        "schema_version", "kind", "status", "error_type", "request_sha256", "offer_sha256",
        "bootstrap_capability_receipt", "code_revision", "code_bundle_sha256",
        "bc_revision", "bc_receipt_sha256", "rollout_revision", "rollout_prefix",
        "rollout_receipt_sha256", "deployment_revision", "mixture_id",
        "deployment_receipt_sha256", "parent_archive_sha256",
        "parent_checkpoint_artifact_sha256",
    }
    return (
        set(claim) == fields
        and claim.get("status") == "blocked"
        and claim.get("error_type") == "RuntimeGpuRentOutcome"
        and isinstance(claim.get("bootstrap_capability_receipt"), str)
        and all(re.fullmatch(r"[0-9a-f]{64}", str(claim.get(key))) for key in ("request_sha256", "offer_sha256", "code_bundle_sha256", "bc_receipt_sha256", "rollout_receipt_sha256", "deployment_receipt_sha256", "parent_archive_sha256", "parent_checkpoint_artifact_sha256"))
        and _runtime_gpu_claim_immutable(claim=claim, identity=identity, request=request)
    )


def _runtime_gpu_recovery_receipt_path(claim_path: Path) -> Path:
    return claim_path.with_name(claim_path.stem + ".recovery.json")


def _runtime_gpu_recovery_archive_path(claim_path: Path, claim_bytes_sha256: str) -> Path:
    return claim_path.with_name(claim_path.stem + ".blocked-" + claim_bytes_sha256 + ".json")


_RUNTIME_GPU_LEGACY_ARCHIVE_SUFFIXES = frozenset({
    ".blocked-no-instance-offer-32602753-20260814T1525.json",
    ".blocked-no-instance-offer-38355172-20260814T1527.json",
    ".blocked-no-instance-offer-46000988-20260814T1523.json",
    ".blocked-no-instance-offer-47277315-20260814T1521.json",
    ".blocked-verified-empty-20260814T1510.json",
})


def _runtime_gpu_recovery_archive_name_is_valid(path: Path, claim_path: Path) -> bool:
    suffix = path.name.removeprefix(claim_path.stem)
    return (
        suffix in _RUNTIME_GPU_LEGACY_ARCHIVE_SUFFIXES
        or re.fullmatch(r"\.blocked-[0-9a-f]{64}\.json", suffix) is not None
    )


def _runtime_gpu_recovery_archives(
    *, claim_path: Path, identity: Mapping[str, object], request: Mapping[str, object],
    allow_recovery_receipt: bool = False,
) -> list[dict[str, object]]:
    """Authenticate every historical archive in this controller-owned namespace."""
    prefix = claim_path.stem + ".blocked-"
    rows: list[dict[str, object]] = []
    digests: set[str] = set()
    for path in sorted(claim_path.parent.iterdir()):
        if allow_recovery_receipt and path == _runtime_gpu_recovery_receipt_path(claim_path):
            continue
        if path != claim_path and path.name.startswith(claim_path.stem + ".") and not path.name.startswith(prefix):
            raise ValueError("runtime GPU recovery archive namespace has an unrelated file")
        if not path.name.startswith(prefix):
            continue
        if not _runtime_gpu_recovery_archive_name_is_valid(path, claim_path) or path.is_symlink() or not path.is_file():
            raise ValueError("runtime GPU recovery archive namespace is malformed")
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        suffix = path.name.removeprefix(claim_path.stem)
        named_digest = suffix.removeprefix(".blocked-").removesuffix(".json")
        if (
            re.fullmatch(r"[0-9a-f]{64}", named_digest) is not None
            and actual != named_digest
        ) or actual in digests:
            raise ValueError("runtime GPU recovery archive hash is ambiguous")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise ValueError("runtime GPU recovery archive is malformed") from None
        if not isinstance(value, Mapping) or not _valid_blocked_runtime_gpu_claim(
            claim=value, identity=identity, request=request,
        ):
            raise ValueError("runtime GPU recovery archive is not a matching blocked claim")
        digests.add(actual)
        rows.append({
            "relative_filename": path.name, "byte_sha256": actual,
            "request_sha256": value["request_sha256"], "offer_sha256": value["offer_sha256"],
            "reconciled_machine_id": None, "original_offer_id": None,
        })
    return rows


def _runtime_gpu_recovery_request(
    request: Mapping[str, object],
) -> tuple[Path, int, Path]:
    fields = {
        "schema_version", "kind", "blocked_rent_request", "expected_original_machine_id",
        "recovery_receipt",
    }
    if (
        set(request) != fields or request.get("schema_version") != 1
        or request.get("kind") != "runtime_mixture_gpu_rent_recovery"
        or not isinstance(request.get("blocked_rent_request"), str)
        or not _positive_int(request.get("expected_original_machine_id"))
        or not isinstance(request.get("recovery_receipt"), str)
    ):
        raise ValueError("runtime GPU recovery request schema is invalid")
    blocked = Path(str(request["blocked_rent_request"]))
    receipt = Path(str(request["recovery_receipt"]))
    if (
        not blocked.is_absolute() or blocked.is_symlink() or not blocked.is_file()
        or not receipt.is_absolute() or receipt.is_symlink()
        or receipt.exists() and not receipt.is_file()
    ):
        raise ValueError("runtime GPU recovery paths are unsafe or unavailable")
    return blocked, int(request["expected_original_machine_id"]), receipt


def _runtime_gpu_recovery_live_offer(
    *, runner: Runner, original_offer_id: int, expected_machine_id: int,
) -> dict[str, object]:
    value = _json(runner, (
        "vastai", "--raw", "search", "offers", OFFER_QUERY, "--interruptible", "--storage", "300",
    ))
    if not isinstance(value, list):
        raise ValueError("runtime GPU recovery offer search is invalid")
    matching = [row for row in value if isinstance(row, Mapping) and row.get("id") == original_offer_id]
    if len(matching) != 1:
        raise ValueError("runtime GPU recovery original offer is missing or ambiguous")
    offer = matching[0]
    if (
        not _positive_int(offer.get("machine_id")) or offer.get("machine_id") != expected_machine_id
        or not _offer_gpu(offer) or offer.get("num_gpus") != 1
        or type(offer.get("dph_total")) not in (int, float)
        or not math.isfinite(float(offer["dph_total"])) or float(offer["dph_total"]) >= 1
    ):
        raise ValueError("runtime GPU recovery original offer drifted")
    return _project(offer, ("id", "machine_id", "gpu_name", "gpu_ram", "num_gpus", "dph_total"))


def _runtime_gpu_recovery_offer_snapshot(
    *, runner: Runner, original_offer_id: int, expected_machine_id: int, timestamp_unix: int,
) -> tuple[str, dict[str, object]]:
    query = "id = " + str(original_offer_id)
    value = _json(runner, (
        "vastai", "--raw", "search", "offers", query, "--interruptible", "--storage", "300",
    ))
    if not isinstance(value, list):
        raise ValueError("runtime GPU recovery offer search is invalid")
    matching = [row for row in value if isinstance(row, Mapping) and row.get("id") == original_offer_id]
    if not matching:
        if value != []:
            raise ValueError("runtime GPU recovery narrow offer search is inconsistent")
        return "absent", {"timestamp_unix": timestamp_unix, "query_sha256": _hash(query), "response_sha256": _hash(value), "matching_count": 0}
    if len(value) != 1 or len(matching) != 1:
        raise ValueError("runtime GPU recovery original offer is ambiguous")
    # Reuse the narrow live-row validation without persisting raw provider fields.
    proof = _runtime_gpu_recovery_live_offer(
        runner=lambda _command: json.dumps(value), original_offer_id=original_offer_id,
        expected_machine_id=expected_machine_id,
    )
    return "present", proof


def _recovery_now(now_unix: int | Callable[[], int] | None) -> int:
    value = int(time.time()) if now_unix is None else (now_unix() if callable(now_unix) else now_unix)
    if type(value) is not int:
        raise ValueError("runtime GPU recovery clock is invalid")
    return value


def _runtime_gpu_recovery_crash(stage: str, crash_after: str | None) -> None:
    if crash_after == stage:
        raise RuntimeError("injected recovery crash after " + stage)


def _runtime_gpu_recovery_offer_proof_is_valid(
    proof: object, *, offer_id: object, machine_id: object,
) -> bool:
    return (
        isinstance(proof, Mapping)
        and set(proof) == {"id", "machine_id", "gpu_name", "gpu_ram", "num_gpus", "dph_total"}
        and proof.get("id") == offer_id and proof.get("machine_id") == machine_id
        and _positive_int(offer_id) and _positive_int(machine_id) and _offer_gpu(proof)
        and proof.get("num_gpus") == 1 and type(proof.get("dph_total")) in (int, float)
        and math.isfinite(float(proof["dph_total"])) and float(proof["dph_total"]) < 1
    )


def _runtime_gpu_recovery_reconciled_receipt_is_valid(
    receipt: Mapping[str, object], *, claim_path: Path, claim_sha256: str,
    original_offer_id: int, machine_id: int, archives: list[dict[str, object]],
) -> bool:
    observations = receipt.get("observations")
    expected_empty_hash = _hash([])
    observation_timestamps = [row.get("timestamp_unix") for row in observations] if isinstance(observations, list) else []
    common = (
        receipt.get("schema_version") == 1 and receipt.get("kind") == "runtime_mixture_gpu_rent_recovery_receipt"
        and receipt.get("status") == "reconciled" and receipt.get("released") is False
        and receipt.get("canonical_claim_path") == str(claim_path) and receipt.get("canonical_claim_sha256") == claim_sha256
        and receipt.get("original_offer_id") == original_offer_id and receipt.get("archive_claims") == archives
        and isinstance(observations, list) and len(observations) == RUNTIME_GPU_RECOVERY_OBSERVATION_POLLS
        and all(isinstance(row, Mapping) and set(row) == {"timestamp_unix", "instances_sha256", "volumes_sha256"} and type(row.get("timestamp_unix")) is int and row.get("instances_sha256") == expected_empty_hash and row.get("volumes_sha256") == expected_empty_hash for row in observations)
        and all(before <= after for before, after in zip(observation_timestamps, observation_timestamps[1:]))
    )
    if receipt.get("offer_proof_mode") == "absent":
        snapshot_fields = {"timestamp_unix", "query_sha256", "response_sha256", "matching_count"}
        return (
            common and set(receipt) == {"schema_version", "kind", "status", "released", "canonical_claim_path", "canonical_claim_sha256", "archive_claims", "original_offer_id", "observations", "offer_proof_mode", "blacklisted_machine_id", "start_offer_snapshot", "end_offer_snapshot"}
            and receipt.get("blacklisted_machine_id") == machine_id
            and all(isinstance(snapshot, Mapping) and set(snapshot) == snapshot_fields and type(snapshot.get("timestamp_unix")) is int and snapshot.get("query_sha256") == _hash("id = " + str(original_offer_id)) and snapshot.get("response_sha256") == expected_empty_hash and snapshot.get("matching_count") == 0 for snapshot in (receipt.get("start_offer_snapshot"), receipt.get("end_offer_snapshot")))
            and receipt["end_offer_snapshot"]["timestamp_unix"] >= receipt["start_offer_snapshot"]["timestamp_unix"] + RUNTIME_GPU_RECOVERY_OBSERVATION_POLLS * RUNTIME_GPU_RECOVERY_POLL_SECONDS
            and all(receipt["start_offer_snapshot"]["timestamp_unix"] <= row["timestamp_unix"] <= receipt["end_offer_snapshot"]["timestamp_unix"] for row in observations)
        )
    fields = {
        "schema_version", "kind", "status", "released", "canonical_claim_path",
        "canonical_claim_sha256", "archive_claims", "original_offer_id", "reconciled_machine_id",
        "observations", "start_offer_proof", "start_offer_proof_sha256", "end_offer_proof",
        "end_offer_proof_sha256", "offer_proof_mode",
    }
    return (
        common and set(receipt) == fields and receipt.get("offer_proof_mode") == "present"
        and receipt.get("reconciled_machine_id") == machine_id
        and _runtime_gpu_recovery_offer_proof_is_valid(receipt.get("start_offer_proof"), offer_id=original_offer_id, machine_id=machine_id)
        and _runtime_gpu_recovery_offer_proof_is_valid(receipt.get("end_offer_proof"), offer_id=original_offer_id, machine_id=machine_id)
        and receipt.get("start_offer_proof_sha256") == _hash(receipt.get("start_offer_proof"))
        and receipt.get("end_offer_proof_sha256") == _hash(receipt.get("end_offer_proof"))
    )


def recover_runtime_gpu_rent(
    *, request: Mapping[str, object], runner: Runner,
    now_unix: int | Callable[[], int] | None = None,
    sleep: Callable[[float], None] = _bounded_sleep,
    crash_after: str | None = None,
) -> dict[str, object]:
    """Fail-closed, read-only reconciliation before one blocked lease is released."""
    blocked_path, expected_machine_id, receipt_path = _runtime_gpu_recovery_request(request)
    original = dict(_load_regular_json(blocked_path, "blocked runtime GPU rent request"))
    identity = _runtime_gpu_rent_preflight(original, recovery_safe=True)
    claim_path = _runtime_gpu_rent_claim_path(
        request=original, identity=identity, allow_held=True,
    )
    if str(receipt_path) != str(_runtime_gpu_recovery_receipt_path(claim_path)):
        raise ValueError("runtime GPU recovery receipt must use the controller-owned canonical path")
    original_offer = original.get("offer")
    if not isinstance(original_offer, Mapping) or not _positive_int(original_offer.get("id")):
        raise ValueError("runtime GPU recovery original offer is invalid")
    claim_bytes = claim_path.read_bytes() if claim_path.is_file() and not claim_path.is_symlink() else None
    receipt = dict(_load_regular_json(receipt_path, "runtime GPU recovery receipt")) if receipt_path.exists() else None
    if claim_bytes is None:
        if receipt is None or receipt.get("canonical_claim_path") != str(claim_path) or not isinstance(receipt.get("canonical_claim_sha256"), str):
            raise ValueError("runtime GPU recovery canonical blocked claim is unavailable")
        claim_sha256 = str(receipt["canonical_claim_sha256"])
        archive_path = _runtime_gpu_recovery_archive_path(claim_path, claim_sha256)
        if archive_path.is_symlink() or not archive_path.is_file() or hashlib.sha256(archive_path.read_bytes()).hexdigest() != claim_sha256:
            raise ValueError("runtime GPU recovery canonical archive is unavailable")
        claim_bytes = archive_path.read_bytes()
    claim_sha256 = hashlib.sha256(claim_bytes).hexdigest()
    try: claim_value = json.loads(claim_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError): raise ValueError("runtime GPU recovery canonical claim is malformed") from None
    if not isinstance(claim_value, Mapping) or not _valid_blocked_runtime_gpu_claim(claim=claim_value, identity=identity, request=original) or claim_value.get("request_sha256") != _hash({key: value for key, value in original.items() if key != "rent_claim_receipt"}) or claim_value.get("offer_sha256") != _hash(dict(original_offer)):
        raise ValueError("runtime GPU recovery canonical claim does not bind the original request")
    archives = _runtime_gpu_recovery_archives(claim_path=claim_path, identity=identity, request=original, allow_recovery_receipt=receipt is not None)
    expires = original.get("expires_at_unix")
    if type(expires) is not int or _recovery_now(now_unix) < expires + RUNTIME_GPU_RECOVERY_GRACE_SECONDS:
        raise ValueError("runtime GPU recovery grace period has not elapsed")
    initial_receipt = {
        "schema_version": 1, "kind": "runtime_mixture_gpu_rent_recovery_receipt",
        "status": "observing", "released": False, "canonical_claim_sha256": claim_sha256,
    }
    if receipt is None:
        _write_exclusive_json(receipt_path, initial_receipt)
        _runtime_gpu_recovery_crash("observing", crash_after)
        receipt = initial_receipt
    if receipt == initial_receipt:
        mode, start_offer = _runtime_gpu_recovery_offer_snapshot(
            runner=runner, original_offer_id=int(original_offer["id"]), expected_machine_id=expected_machine_id,
            timestamp_unix=_recovery_now(now_unix),
        )
        observations: list[dict[str, object]] = []
        for _ in range(RUNTIME_GPU_RECOVERY_OBSERVATION_POLLS):
            timestamp = _recovery_now(now_unix)
            instances = _json(runner, ("vastai", "--raw", "show", "instances"))
            volumes = _json(runner, ("vastai", "--raw", "show", "volumes"))
            if instances != [] or volumes != []:
                raise ValueError("runtime GPU recovery provider account is not provably empty")
            observations.append({
                "timestamp_unix": timestamp,
                "instances_sha256": _hash(instances), "volumes_sha256": _hash(volumes),
            })
            sleep(RUNTIME_GPU_RECOVERY_POLL_SECONDS)
        end_mode, end_offer = _runtime_gpu_recovery_offer_snapshot(
            runner=runner, original_offer_id=int(original_offer["id"]), expected_machine_id=expected_machine_id,
            timestamp_unix=_recovery_now(now_unix),
        )
        if end_mode != mode:
            raise ValueError("runtime GPU recovery original offer changed between snapshots")
        if mode == "absent" and end_offer["timestamp_unix"] < start_offer["timestamp_unix"] + RUNTIME_GPU_RECOVERY_OBSERVATION_POLLS * RUNTIME_GPU_RECOVERY_POLL_SECONDS:
            raise ValueError("runtime GPU recovery absent offer snapshots are not sufficiently separated")
        archive_path = _runtime_gpu_recovery_archive_path(claim_path, claim_sha256)
        archived_current = {
            "relative_filename": archive_path.name, "byte_sha256": claim_sha256,
            "request_sha256": claim_value["request_sha256"], "offer_sha256": claim_value["offer_sha256"],
            "reconciled_machine_id": expected_machine_id if mode == "present" else None, "original_offer_id": original_offer["id"],
        }
        final = {
            "schema_version": 1, "kind": "runtime_mixture_gpu_rent_recovery_receipt",
            "status": "reconciled", "released": False,
            "canonical_claim_path": str(claim_path), "canonical_claim_sha256": claim_sha256,
            "archive_claims": archives + [archived_current], "original_offer_id": original_offer["id"],
            "reconciled_machine_id": expected_machine_id,
            "observations": observations, "start_offer_proof": start_offer,
            "start_offer_proof_sha256": _hash(start_offer), "end_offer_proof": end_offer,
            "end_offer_proof_sha256": _hash(end_offer), "offer_proof_mode": mode,
        }
        if mode == "absent":
            final.pop("reconciled_machine_id"); final.pop("start_offer_proof"); final.pop("start_offer_proof_sha256"); final.pop("end_offer_proof"); final.pop("end_offer_proof_sha256")
            final |= {"blacklisted_machine_id": expected_machine_id, "start_offer_snapshot": start_offer, "end_offer_snapshot": end_offer}
        _replace_json_durably(receipt_path, final); _runtime_gpu_recovery_crash("reconciled", crash_after); receipt = final
    archive_path = _runtime_gpu_recovery_archive_path(claim_path, claim_sha256)
    current_machine = expected_machine_id if receipt.get("offer_proof_mode") == "present" else None
    expected_archives = [row for row in archives if row["relative_filename"] != archive_path.name] + [{"relative_filename": archive_path.name, "byte_sha256": claim_sha256, "request_sha256": claim_value["request_sha256"], "offer_sha256": claim_value["offer_sha256"], "reconciled_machine_id": current_machine, "original_offer_id": original_offer["id"]}]
    if receipt.get("status") == "released":
        reconciled = dict(receipt); reconciled.pop("archive_path", None)
        reconciled["status"] = "reconciled"; reconciled["released"] = False
        if (
            claim_path.exists() or receipt.get("released") is not True
            or receipt.get("archive_path") != str(archive_path)
            or not _runtime_gpu_recovery_reconciled_receipt_is_valid(
                reconciled, claim_path=claim_path, claim_sha256=claim_sha256,
                original_offer_id=int(original_offer["id"]), machine_id=expected_machine_id,
                archives=expected_archives,
            )
        ):
            raise ValueError("runtime GPU recovery released receipt is not authenticated")
        return {"paid_action": False, "action": "runtime-gpu-rent-recover", "recovery_receipt": receipt}
    if not _runtime_gpu_recovery_reconciled_receipt_is_valid(receipt, claim_path=claim_path, claim_sha256=claim_sha256, original_offer_id=int(original_offer["id"]), machine_id=expected_machine_id, archives=expected_archives):
        raise ValueError("runtime GPU recovery receipt is not an authenticated reconciled state")
    if not archive_path.exists():
        if claim_path.is_symlink() or not claim_path.is_file() or claim_path.read_bytes() != claim_bytes: raise RuntimeError("runtime GPU recovery canonical claim changed before archive")
        _write_exclusive_bytes(archive_path, claim_bytes); _runtime_gpu_recovery_crash("archived", crash_after)
    if archive_path.is_symlink() or not archive_path.is_file() or archive_path.read_bytes() != claim_bytes: raise RuntimeError("runtime GPU recovery archive readback mismatches canonical claim")
    if claim_path.exists():
        if claim_path.is_symlink() or claim_path.read_bytes() != claim_bytes: raise RuntimeError("runtime GPU recovery canonical claim changed before release")
        claim_path.unlink(); _fsync_parent(claim_path); _runtime_gpu_recovery_crash("unlinked", crash_after)
    released_final = dict(receipt) | {"released": True, "status": "released", "archive_path": str(archive_path)}
    _replace_json_durably(receipt_path, released_final); _runtime_gpu_recovery_crash("released", crash_after)
    if dict(_load_regular_json(receipt_path, "runtime GPU recovery receipt")) != released_final: raise RuntimeError("runtime GPU recovery receipt readback mismatches")
    return {"paid_action": False, "action": "runtime-gpu-rent-recover", "recovery_receipt": released_final}


def _validate_runtime_gpu_recovery_for_new_rent(
    *, request: Mapping[str, object], identity: Mapping[str, object], claim_path: Path,
) -> None:
    """A recovered claim may be retried exactly once on another offer and machine."""
    recovery_path = _runtime_gpu_recovery_receipt_path(claim_path)
    namespace = list(claim_path.parent.glob(claim_path.stem + ".blocked-*.json"))
    if not namespace and not recovery_path.exists():
        return
    if request.get("recovery_receipt") != str(recovery_path):
        raise ValueError("runtime GPU rent requires the canonical recovery receipt")
    receipt = _load_regular_json(recovery_path, "runtime GPU recovery receipt")
    mode = receipt.get("offer_proof_mode")
    fields = {
        "schema_version", "kind", "status", "released", "canonical_claim_path",
        "canonical_claim_sha256", "archive_claims", "original_offer_id", "reconciled_machine_id",
        "observations", "start_offer_proof", "start_offer_proof_sha256", "end_offer_proof",
        "end_offer_proof_sha256", "offer_proof_mode", "archive_path",
    }
    if mode == "absent":
        fields -= {"reconciled_machine_id", "start_offer_proof", "start_offer_proof_sha256", "end_offer_proof", "end_offer_proof_sha256"}
        fields |= {"blacklisted_machine_id", "start_offer_snapshot", "end_offer_snapshot"}
    if set(receipt) != fields or receipt.get("schema_version") != 1 or receipt.get("kind") != "runtime_mixture_gpu_rent_recovery_receipt" or receipt.get("status") != "released" or receipt.get("released") is not True or receipt.get("canonical_claim_path") != str(claim_path):
        raise ValueError("runtime GPU recovery receipt is not a released canonical recovery")
    archive_path = _runtime_gpu_recovery_archive_path(
        claim_path, str(receipt.get("canonical_claim_sha256")),
    ) if re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("canonical_claim_sha256"))) else None
    if (
        archive_path is None or receipt.get("archive_path") != str(archive_path)
        or archive_path.is_symlink() or not archive_path.is_file()
        or hashlib.sha256(archive_path.read_bytes()).hexdigest() != receipt.get("canonical_claim_sha256")
        or not _positive_int(receipt.get("original_offer_id"))
        or not _positive_int(receipt.get("reconciled_machine_id") if mode == "present" else receipt.get("blacklisted_machine_id"))
        or (mode == "absent" and receipt.get("blacklisted_machine_id") != RUNTIME_GPU_LEGACY_RECOVERY_BLACKLISTED_MACHINE_ID)
    ):
        raise ValueError("runtime GPU recovery receipt archive is invalid")
    try:
        archived_claim = json.loads(archive_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("runtime GPU recovery receipt archive is invalid") from None
    if not isinstance(archived_claim, Mapping) or not _valid_blocked_runtime_gpu_claim(
        claim=archived_claim, identity=identity, request=request,
    ):
        raise ValueError("runtime GPU recovery receipt archive is not a matching blocked claim")
    archived_rows = _runtime_gpu_recovery_archives(
        claim_path=claim_path, identity=identity, request=request, allow_recovery_receipt=True,
    )
    current_rows = [
        row for row in archived_rows
        if row["relative_filename"] == archive_path.name
        and row["byte_sha256"] == receipt["canonical_claim_sha256"]
    ]
    if len(current_rows) != 1:
        raise ValueError("runtime GPU recovery receipt canonical archive is ambiguous")
    archived_rows = [
        row | {
            "reconciled_machine_id": receipt["reconciled_machine_id"] if mode == "present" else None,
            "original_offer_id": receipt["original_offer_id"],
        }
        if row in current_rows else row
        for row in archived_rows
    ]
    reconciled = dict(receipt)
    reconciled.pop("archive_path")
    reconciled["status"] = "reconciled"
    reconciled["released"] = False
    if not _runtime_gpu_recovery_reconciled_receipt_is_valid(
        reconciled, claim_path=claim_path, claim_sha256=str(receipt["canonical_claim_sha256"]),
        original_offer_id=int(receipt["original_offer_id"]),
        machine_id=int(receipt["reconciled_machine_id"] if mode == "present" else receipt["blacklisted_machine_id"]), archives=archived_rows,
    ):
        raise ValueError("runtime GPU recovery receipt proofs are invalid")
    offer = request.get("offer")
    if (
        not isinstance(offer, Mapping) or offer.get("id") == receipt.get("original_offer_id")
        or offer.get("machine_id") == (receipt.get("reconciled_machine_id") if mode == "present" else receipt.get("blacklisted_machine_id"))
    ):
        raise ValueError("runtime GPU recovery retry requires a different offer and machine")


def _runtime_gpu_bootstrap_capability(
    *, instance: Mapping[str, object], request: Mapping[str, object],
    identity: Mapping[str, object], runner: Runner,
) -> dict[str, object]:
    """Probe the reviewed image on this lease before GPU warm-up can proceed."""
    bundle = Path(str(request["code_bundle"]))
    receipt = Path(str(request["code_bundle_sha256_file"]))
    bundle_sha = _verify_reviewed_code_bundle(bundle, receipt, request.get("code_revision"))
    if bundle_sha != request.get("code_bundle_sha256"):
        raise ValueError("runtime GPU bootstrap code bundle differs from its immutable binding")
    remote = "/tmp/lehome-runtime-gpu-bootstrap"
    runner((*_ssh_prefix(instance), "set -eu; rm -rf " + remote + " /prepared/bootstrap-code; mkdir -p " + remote))
    runner((*_scp_prefix(instance), str(bundle), "root@" + str(instance["host"]) + ":" + remote + "/code.bundle"))
    observed = runner((*_ssh_prefix(instance), "sha256sum " + remote + "/code.bundle")).strip().split()
    if not observed or observed[0] != bundle_sha:
        raise ValueError("runtime GPU bootstrap code bundle readback failed")
    probe_prefix = _ssh_prefix(instance)
    probe_prefix = (
        *probe_prefix[:-1], "-o", f"ConnectTimeout={RUNTIME_GPU_ARCH_TIMEOUT_SECONDS}", probe_prefix[-1],
    )
    command = (
        "set -eu; git clone --quiet --no-checkout " + remote + "/code.bundle /prepared/bootstrap-code; "
        "git -C /prepared/bootstrap-code checkout --quiet --detach " + str(request["code_revision"]) + "; "
        "test \"$(git -C /prepared/bootstrap-code rev-parse HEAD)\" = " + str(request["code_revision"]) + "; "
        "test -z \"$(git -C /prepared/bootstrap-code status --porcelain)\"; "
        "timeout " + str(RUNTIME_GPU_PROBE_TIMEOUT_SECONDS) + " env -u HF_TOKEN PYTHONPATH=/prepared/bootstrap-code/source/lehome:/prepared/bootstrap-code/trainer/src "
        "python -m lehome_train.cli validate-training-capability --one-step --image-digest "
        + BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2]
    )
    try:
        capability = json.loads(_run_bounded(
            runner, (*probe_prefix, command), timeout_seconds=RUNTIME_GPU_PROBE_CALL_TIMEOUT_SECONDS,
        ))
    except (json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        raise ValueError("runtime GPU bootstrap capability probe did not return JSON") from error
    if not isinstance(capability, Mapping):
        raise ValueError("runtime GPU bootstrap capability probe did not return a receipt")
    validated = dict(validate_training_capability(capability))
    outer = {
        "schema_version": 1, "kind": "runtime_mixture_gpu_bootstrap_capability",
        "instance_id": instance["instance_id"],
        "provider_response_sha256": instance["provider_response_sha256"],
        "platform_arch": "x86_64", "trainer_image": BOOTSTRAP_TRAINER_IMAGE,
        "image_digest": BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2],
        "code_revision": request["code_revision"], "code_bundle_sha256": bundle_sha,
        "bc_revision": identity["bc"]["immutable_revision"],
        "bc_receipt_sha256": identity["bc_receipt_sha256"],
        "rollout_revision": identity["rollout"]["immutable_revision"],
        "rollout_prefix": identity["rollout"]["remote_prefix"],
        "rollout_receipt_sha256": identity["rollout_receipt_sha256"],
        "deployment_revision": identity["deployment"]["immutable_revision"],
        "mixture_id": identity["deployment"]["mixture_id"],
        "deployment_receipt_sha256": identity["deployment_receipt_sha256"],
        "parent_archive_sha256": PARENT_CHECKPOINT["archive_sha256"],
        "parent_checkpoint_artifact_sha256": PARENT_CHECKPOINT["artifact_sha256"],
        "training_capability": validated,
    }
    atomic_write_json(Path(str(request["bootstrap_capability_receipt"])), outer)
    return outer


def _runtime_gpu_bootstrap_capability_receipt(
    *, path: Path, instance: Mapping[str, object], request: Mapping[str, object],
    identity: Mapping[str, object],
) -> dict[str, object]:
    """Require the same-lease one-step probe before staging or measuring."""
    value = dict(_load_regular_json(path, "runtime GPU bootstrap capability receipt"))
    expected = {
        "schema_version": 1, "kind": "runtime_mixture_gpu_bootstrap_capability",
        "instance_id": instance.get("instance_id"),
        "provider_response_sha256": instance.get("provider_response_sha256"),
        "platform_arch": "x86_64", "trainer_image": BOOTSTRAP_TRAINER_IMAGE,
        "image_digest": BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2],
        "code_revision": request.get("code_revision"),
        "code_bundle_sha256": request.get("code_bundle_sha256"),
        "bc_revision": identity["bc"]["immutable_revision"],
        "bc_receipt_sha256": identity["bc_receipt_sha256"],
        "rollout_revision": identity["rollout"]["immutable_revision"],
        "rollout_prefix": "rollouts/round-1",
        "rollout_receipt_sha256": identity["rollout_receipt_sha256"],
        "deployment_revision": identity["deployment"]["immutable_revision"],
        "mixture_id": identity["deployment"]["mixture_id"],
        "deployment_receipt_sha256": identity["deployment_receipt_sha256"],
        "parent_archive_sha256": PARENT_CHECKPOINT["archive_sha256"],
        "parent_checkpoint_artifact_sha256": PARENT_CHECKPOINT["artifact_sha256"],
    }
    if (
        set(value) != set(expected) | {"training_capability"}
        or any(value.get(key) != expected_value for key, expected_value in expected.items())
    ):
        raise ValueError("runtime GPU bootstrap capability receipt is not bound to this lease and campaign")
    try:
        validate_training_capability(value.get("training_capability"))
    except (TypeError, ValueError) as error:
        raise ValueError("runtime GPU bootstrap capability receipt is invalid") from error
    if sha256_file(path) != instance.get("capability_sha256"):
        raise ValueError("runtime GPU instance does not bind its same-lease bootstrap capability receipt")
    return value


def _runtime_gpu_pilot_preflight(*, pilot: Mapping[str, object], request: Mapping[str, object], identity: Mapping[str, object]) -> None:
    """Cross-check the CPU receipt without accepting its image on a GPU path."""
    proof = pilot["authenticated_evidence"]
    if (
        not isinstance(proof, Mapping)
        or proof.get("image_digest") != RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2]
        or proof.get("code_revision") != request.get("code_revision")
        or proof.get("code_bundle_sha256") != request.get("code_bundle_sha256")
        or proof.get("bc_revision") != identity["bc"]["immutable_revision"]
        or proof.get("rollout_revision") != identity["rollout"]["immutable_revision"]
        or proof.get("deployment_revision") != identity["deployment"]["immutable_revision"]
    ):
        raise ValueError("runtime GPU warm-up pilot is not bound to the CPU image and immutable runtime sources")


def _runtime_pilot_instance(instance: Mapping[str, object]) -> None:
    if (
        instance.get("schema_version") != 1
        or instance.get("kind") != "runtime_mixture_cpu_pilot_instance"
        or type(instance.get("instance_id")) is not int
        or not isinstance(instance.get("host"), str)
        or type(instance.get("port")) is not int
        or instance.get("platform_arch") != "x86_64"
        or instance.get("trainer_image") != RUNTIME_CPU_PILOT_IMAGE
        or instance.get("image_digest") != RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2]
        or re.fullmatch(r"[0-9a-f]{64}", str(instance.get("provider_response_sha256"))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(instance.get("offer_evidence_sha256"))) is None
    ):
        raise ValueError("runtime CPU pilot instance receipt is invalid")


def run_runtime_cpu_pilot(
    *, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner,
) -> dict[str, object]:
    """Run the existing CPU-only loader sweep and bind its remote receipt locally."""
    _runtime_pilot_instance(instance)
    identity = _runtime_cpu_pilot_identity(instance, request)
    bundle = request.get("code_bundle_sha256")
    output = request.get("lifecycle_receipt")
    bootstrap_path = request.get("bootstrap_receipt")
    if (
        type(bundle) is not str
        or re.fullmatch(r"[0-9a-f]{64}", bundle) is None
        or type(output) is not str
        or not Path(output).is_absolute()
        or Path(output).exists()
        or Path(output).is_symlink()
        or type(bootstrap_path) is not str
    ):
        raise ValueError("runtime CPU pilot requires an absent absolute lifecycle receipt output")
    bootstrap = _runtime_bootstrap_receipt(
        path=Path(bootstrap_path), instance=instance, request=request,
    )
    if (
        bootstrap.get("instance_id") != instance.get("instance_id")
        or bootstrap.get("provider_response_sha256") != instance.get("provider_response_sha256")
        or any(bootstrap.get(key) != identity[name]["immutable_revision"] for key, name in (
            ("bc_revision", "bc"), ("rollout_revision", "rollout"), ("deployment_revision", "deployment"),
        ))
    ):
        raise ValueError("runtime CPU pilot bootstrap receipt is not bound to its instance and immutable sources")
    command = (
        "set -eu; env -u HF_TOKEN PYTHONPATH=/prepared/code/source/lehome:/prepared/code/trainer/src "
        "lehome-train pilot-runtime-mixture --request /prepared/config/runtime-pilot.json"
    )
    try:
        remote = _json(runner, (*_ssh_prefix(instance), command))
    except ValueError as error:
        raise ValueError("runtime CPU pilot did not return a measured receipt") from error
    if not isinstance(remote, Mapping):
        raise ValueError("runtime CPU pilot did not return a measured receipt")
    pilot = _validated_runtime_pilot_value(remote)
    proof = pilot["authenticated_evidence"]
    if (
        proof.get("provider_instance_id") != instance.get("instance_id")
        or proof.get("provider_response_sha256") != instance.get("provider_response_sha256")
        or proof.get("platform_arch") != "x86_64"
        or proof.get("image_digest") != RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2]
        or proof.get("code_revision") != request.get("code_revision")
        or proof.get("code_bundle_sha256") != bundle
        or proof.get("bc_revision") != identity["bc"]["immutable_revision"]
        or proof.get("rollout_revision") != identity["rollout"]["immutable_revision"]
        or proof.get("deployment_revision") != identity["deployment"]["immutable_revision"]
    ):
        raise ValueError("runtime CPU pilot measured receipt is not bound to provider, code, image, and mixture")
    lifecycle = {
        "schema_version": 1, "kind": "runtime_mixture_cpu_pilot_lifecycle",
        "instance_id": instance["instance_id"],
        "provider_response_sha256": instance["provider_response_sha256"],
        "platform_arch": "x86_64", "trainer_image": RUNTIME_CPU_PILOT_IMAGE,
        "image_digest": RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2],
        "code_revision": request["code_revision"], "code_bundle_sha256": bundle,
        "bc_revision": identity["bc"]["immutable_revision"],
        "rollout_revision": identity["rollout"]["immutable_revision"],
        "deployment_revision": identity["deployment"]["immutable_revision"],
        "deployment_receipt_sha256": identity["deployment_receipt_sha256"],
        "bootstrap_receipt_sha256": sha256_file(Path(bootstrap_path)),
        "staged_transfers": bootstrap["transfers"],
        "pilot_receipt": pilot,
    }
    atomic_write_json(Path(output), lifecycle)
    return {"paid_action": True, "action": "runtime-pilot-run", "instance_id": instance["instance_id"], "lifecycle_receipt": lifecycle}


def _runtime_gpu_warmup_instance(instance: Mapping[str, object]) -> None:
    if (
        instance.get("kind") != "runtime_mixture_gpu_warmup_instance"
        or instance.get("platform_arch") != "x86_64"
        or instance.get("trainer_image") != BOOTSTRAP_TRAINER_IMAGE
        or type(instance.get("instance_id")) is not int
        or not isinstance(instance.get("host"), str)
        or type(instance.get("port")) is not int
        or re.fullmatch(r"[0-9a-f]{64}", str(instance.get("provider_response_sha256"))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(instance.get("capability_sha256"))) is None
    ):
        raise ValueError("runtime GPU warm-up instance receipt is invalid")


def _runtime_warmup_binding(
    *, binding: Mapping[str, object], instance: Mapping[str, object], request: Mapping[str, object],
    identity: Mapping[str, object],
) -> dict[str, object]:
    from lehome_train.groot.runtime_mixture_warmup import validate_warmup_binding

    checked = validate_warmup_binding(binding)
    mixture = checked["mixture"]
    deployment = checked["deployment"]
    code = checked["code"]
    parent = checked["parent_checkpoint"]
    if (
        not isinstance(mixture, Mapping) or not isinstance(deployment, Mapping)
        or not isinstance(code, Mapping) or not isinstance(parent, Mapping)
        or mixture.get("repository") != CORRECTIVE_SOURCE["repository"]
        or mixture.get("revision") != identity["deployment"]["immutable_revision"]
        or mixture.get("mixture_id") != identity["deployment"]["mixture_id"]
        or mixture.get("source_revisions") != {
            "organizer": identity["bc"]["immutable_revision"],
            "rollout": identity["rollout"]["immutable_revision"],
        }
        or deployment.get("oci_image_digest") != BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2]
        or deployment.get("provider") != "vast"
        or deployment.get("capability_sha256") != instance.get("capability_sha256")
        or code.get("repository_revision") != request.get("code_revision")
        or code.get("bundle_sha256") != request.get("code_bundle_sha256")
        or parent != {
            "repository": PARENT_CHECKPOINT["repository"], "revision": PARENT_CHECKPOINT["revision"],
            "subpath": PARENT_CHECKPOINT["subpath"], "artifact_sha256": PARENT_CHECKPOINT["artifact_sha256"],
        }
    ):
        raise ValueError("runtime GPU warm-up binding does not match provider, code, deployment, and parent")
    return checked


def _validated_runtime_gpu_warmup_lifecycle(
    *, path: Path, instance: Mapping[str, object], request: Mapping[str, object],
) -> tuple[dict[str, object], int]:
    """Accept only a measured direct-GPU warm-up from this exact lease."""
    warmup = dict(_load_regular_json(path, "runtime GPU warm-up lifecycle receipt"))
    identity = _runtime_identity(instance, request)
    binding = _runtime_warmup_binding(
        binding=warmup.get("runtime_warmup_binding")
        if isinstance(warmup.get("runtime_warmup_binding"), Mapping) else {},
        instance=instance,
        request=request,
        identity=identity,
    )
    from lehome_train.groot.runtime_mixture_warmup import validate_gpu_warmup_receipt

    selected = validate_gpu_warmup_receipt(
        warmup.get("warmup_receipt") if isinstance(warmup.get("warmup_receipt"), Mapping) else {},
        expected_binding=binding,
    )
    if (
        warmup.get("kind") != "runtime_mixture_gpu_warmup_lifecycle"
        or warmup.get("instance_id") != instance.get("instance_id")
        or warmup.get("provider_response_sha256") != instance.get("provider_response_sha256")
        or warmup.get("capability_sha256") != instance.get("capability_sha256")
        or warmup.get("bootstrap_capability_receipt_sha256") != instance.get("capability_sha256")
        or warmup.get("code_revision") != request.get("code_revision")
        or warmup.get("code_bundle_sha256") != request.get("code_bundle_sha256")
        or warmup.get("parent_checkpoint_artifact_sha256") != PARENT_CHECKPOINT["artifact_sha256"]
        or warmup.get("deployment_revision") != identity["deployment"]["immutable_revision"]
        or warmup.get("selected_loader_workers") != selected
    ):
        raise ValueError("runtime GPU warm-up receipt is not bound to the same GPU instance")
    return warmup, selected


def run_runtime_gpu_warmup(
    *, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner,
) -> dict[str, object]:
    """Invoke the measured direct-GPU adapter after immutable identity checks."""
    _runtime_gpu_warmup_instance(instance)
    identity = _runtime_identity(instance, request)
    capability_path = request.get("bootstrap_capability_receipt")
    if type(capability_path) is not str:
        raise ValueError("runtime GPU warm-up requires the same-lease bootstrap capability receipt")
    _runtime_gpu_bootstrap_capability_receipt(
        path=Path(capability_path), instance=instance, request=request, identity=identity,
    )
    binding_path = request.get("runtime_warmup_binding")
    output = request.get("warmup_lifecycle_receipt")
    if type(binding_path) is not str or type(output) is not str or not Path(output).is_absolute() or Path(output).exists() or Path(output).is_symlink():
        raise ValueError("runtime GPU warm-up requires reviewed binding and absent absolute lifecycle receipt")
    binding = _runtime_warmup_binding(
        binding=_load_regular_json(Path(binding_path), "runtime GPU warm-up binding"),
        instance=instance, request=request, identity=identity,
    )
    command = (
        "set -eu; env -u HF_TOKEN PYTHONPATH=/prepared/code/source/lehome:/prepared/code/trainer/src "
        "lehome-train runtime-gpu-warmup --request /prepared/config/runtime-warmup.json"
    )
    remote = _json(runner, (*_ssh_prefix(instance), command))
    if not isinstance(remote, Mapping):
        raise ValueError("runtime GPU warm-up did not return a measured receipt")
    from lehome_train.groot.runtime_mixture_warmup import validate_gpu_warmup_receipt
    selected = validate_gpu_warmup_receipt(remote, expected_binding=binding)
    lifecycle = {
        "schema_version": 1, "kind": "runtime_mixture_gpu_warmup_lifecycle",
        "instance_id": instance["instance_id"], "provider_response_sha256": instance["provider_response_sha256"],
        "platform_arch": "x86_64", "trainer_image": BOOTSTRAP_TRAINER_IMAGE,
        "capability_sha256": instance["capability_sha256"], "code_revision": request["code_revision"],
        "bootstrap_capability_receipt_sha256": instance["capability_sha256"],
        "code_bundle_sha256": request["code_bundle_sha256"], "parent_checkpoint_artifact_sha256": PARENT_CHECKPOINT["artifact_sha256"],
        "deployment_revision": identity["deployment"]["immutable_revision"],
        "runtime_warmup_binding": binding, "warmup_receipt": dict(remote),
        "selected_loader_workers": selected,
    }
    atomic_write_json(Path(output), lifecycle)
    return {"paid_action": True, "action": "runtime-gpu-warmup", "instance_id": instance["instance_id"], "warmup_lifecycle_receipt": lifecycle}


def rent_runtime_gpu_warmup(*, evidence: Mapping[str, object], runner: Runner) -> dict[str, object]:
    """Promote a fresh direct PRO6000 lease after up to ten minutes of readiness."""
    _runtime_failure_receipt_path(evidence)
    identity = _runtime_gpu_rent_preflight(evidence)
    claim_path = _runtime_gpu_rent_claim_path(request=evidence, identity=identity)
    claim = _runtime_gpu_rent_claim(path=claim_path, request=evidence, identity=identity)
    rented: Mapping[str, object] | None = None
    try:
        rented = rent(
            evidence=evidence, runner=runner, require_capability=False,
            abort_request=evidence,
            max_readiness_polls=RUNTIME_GPU_WARMUP_READINESS_POLLS,
        )
        _await_platform_arch_attestation(instance=rented, runner=runner)
        outer = _runtime_gpu_bootstrap_capability(
            instance=rented, request=evidence, identity=identity, runner=runner,
        )
        capability_sha256 = sha256_file(Path(str(evidence["bootstrap_capability_receipt"])))
        _terminalize_runtime_gpu_rent_claim(
            path=claim_path, claim=claim, instance=rented,
            capability_sha256=capability_sha256,
        )
        return {
            **rented, "kind": "runtime_mixture_gpu_warmup_instance", "platform_arch": "x86_64",
            "image_digest": BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2],
            "capability_sha256": capability_sha256,
            "bootstrap_capability_receipt_sha256": capability_sha256,
        }
    except BaseException as error:
        if rented is not None:
            try:
                _runtime_abort_cleanup(instance=rented, request=evidence, error=error, runner=runner)
            except BaseException as cleanup_error:
                _block_runtime_gpu_rent_claim(path=claim_path, claim=claim, error=cleanup_error)
                raise
            _release_runtime_gpu_rent_claim(path=claim_path, claim=claim)
        elif isinstance(error, RuntimeGpuRentOutcome) and error.no_lease_exists:
            _release_runtime_gpu_rent_claim(path=claim_path, claim=claim)
        else:
            _block_runtime_gpu_rent_claim(path=claim_path, claim=claim, error=error)
        raise


def destroy_runtime_cpu_pilot(
    *, instance_id: int, lifecycle_receipt: Mapping[str, object], runner: Runner,
    max_absence_polls: int = RUNTIME_ABSENCE_READBACK_POLLS,
    sleep: Callable[[float], None] = _bounded_sleep,
) -> dict[str, object]:
    """Allow disposal only after the exact instance-bound measured pilot receipt."""
    if type(instance_id) is not int or lifecycle_receipt.get("kind") != "runtime_mixture_cpu_pilot_lifecycle":
        raise ValueError("runtime CPU pilot destroy receipt is invalid")
    pilot = lifecycle_receipt.get("pilot_receipt")
    if not isinstance(pilot, Mapping):
        raise ValueError("runtime CPU pilot destroy requires an authenticated pilot receipt")
    validated = _validated_runtime_pilot_value(pilot)
    proof = validated["authenticated_evidence"]
    expected = {
        "instance_id": instance_id, "platform_arch": "x86_64",
        "trainer_image": RUNTIME_CPU_PILOT_IMAGE,
        "image_digest": RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2],
    }
    if any(lifecycle_receipt.get(key) != value for key, value in expected.items()) or any(re.fullmatch(r"[0-9a-f]{64}", str(lifecycle_receipt.get(key))) is None for key in ("provider_response_sha256", "code_bundle_sha256", "deployment_receipt_sha256", "bootstrap_receipt_sha256")) or not isinstance(lifecycle_receipt.get("staged_transfers"), list) or not lifecycle_receipt["staged_transfers"] or any(not isinstance(item, Mapping) or set(item) != {"name", "sha256"} or not isinstance(item.get("name"), str) or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256"))) is None for item in lifecycle_receipt["staged_transfers"]) or any(re.fullmatch(r"[0-9a-f]{40}", str(lifecycle_receipt.get(key))) is None for key in ("code_revision", "bc_revision", "rollout_revision", "deployment_revision")) or proof.get("provider_instance_id") != instance_id or proof.get("provider_response_sha256") != lifecycle_receipt.get("provider_response_sha256") or proof.get("platform_arch") != "x86_64" or proof.get("image_digest") != RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2] or any(proof.get(key) != lifecycle_receipt.get(key) for key in ("code_revision", "code_bundle_sha256", "bc_revision", "rollout_revision", "deployment_revision")):
        raise ValueError("runtime CPU pilot destroy receipt is not bound to its authenticated pilot")
    runner(("vastai", "destroy", "instance", str(instance_id), "--yes"))
    if not _await_runtime_instance_absence(
        instance_id=instance_id, runner=runner, max_polls=max_absence_polls, sleep=sleep,
    ):
        raise ValueError("runtime CPU pilot destroy absence readback failed")
    return {"paid_action": True, "destroy_authorized": True, "instance_id": instance_id}


def _runtime_checkpoint_identity(instance: Mapping[str, object], request: Mapping[str, object]):
    """Construct the stable checkpoint identity from the staged runtime receipts."""
    _runtime_gpu_warmup_instance(instance)
    identity = _runtime_identity(instance, request)
    seed = request.get("schedule_seed")
    if type(seed) is not int:
        raise ValueError("runtime checkpoint lifecycle requires integer schedule_seed")
    from lehome_train.groot.runtime_checkpoint_lifecycle import RuntimeMixtureTrainingIdentity
    return RuntimeMixtureTrainingIdentity(
        mixture_id=str(identity["deployment"]["mixture_id"]),
        deployment_receipt_sha256=str(identity["deployment_receipt_sha256"]),
        source_revisions=(
            ("organizer", str(identity["bc"]["immutable_revision"]), "bc/full", str(identity["bc_receipt_sha256"])),
            ("rollout", str(identity["rollout"]["immutable_revision"]), str(identity["rollout"]["remote_prefix"]), str(identity["rollout_receipt_sha256"])),
        ),
        schedule_seed=seed, code_bundle_sha256=str(request["code_bundle_sha256"]),
        code_bundle_revision=str(request["code_revision"]),
        oci_image=BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2],
        parent_step12000_artifact_sha256=PARENT_CHECKPOINT["artifact_sha256"],
    )


def publish_runtime_checkpoint(
    *, instance: Mapping[str, object], request: Mapping[str, object], publisher: Callable[..., dict[str, object]], hub: object,
) -> dict[str, object]:
    """Publish and fresh-read one exact runtime 1K/2K checkpoint package."""
    identity = _runtime_checkpoint_identity(instance, request)
    descriptor_path, root = request.get("checkpoint_descriptor"), request.get("checkpoint_artifact_root")
    if type(descriptor_path) is not str or type(root) is not str:
        raise ValueError("runtime checkpoint publish requires descriptor and artifact root")
    from lehome_train.checkpoints import load_checkpoint_descriptor
    from lehome_train.groot.runtime_checkpoint_lifecycle import publish_runtime_mixture_checkpoint
    publication = publish_runtime_mixture_checkpoint(
        identity=identity, checkpoint=load_checkpoint_descriptor(descriptor_path), artifact_root=Path(root),
        publisher=publisher, hub=hub,  # type: ignore[arg-type]
    )
    return {"paid_action": True, "action": "runtime-checkpoint-publish", "instance_id": instance["instance_id"], "publication": publication}


def runtime_checkpoint_terminal(
    *, instance: Mapping[str, object], request: Mapping[str, object], provider_loss: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Produce only module-validated completion or provider-interruption terminals."""
    identity = _runtime_checkpoint_identity(instance, request)
    publications = request.get("runtime_checkpoint_publications")
    if not isinstance(publications, list) or not all(isinstance(item, Mapping) for item in publications):
        raise ValueError("runtime checkpoint terminal requires publications")
    from lehome_train.groot.runtime_checkpoint_lifecycle import provider_interruption_terminal, runtime_mixture_completion_terminal
    rows = tuple(publications)
    if provider_loss is None:
        terminal = runtime_mixture_completion_terminal(identity=identity, instance_id=str(instance["instance_id"]), publications=rows)
    else:
        if provider_loss.get("kind") not in {"instance_absent", "preempted"}:
            raise ValueError("runtime checkpoint replacement requires explicit provider absence or preemption")
        terminal = provider_interruption_terminal(identity=identity, instance_id=str(instance["instance_id"]), publications=rows, provider_loss=provider_loss)
    return {"paid_action": True, "action": "runtime-checkpoint-terminal", "terminal": terminal}


def classify_runtime_provider_loss(*, instance: Mapping[str, object], runner: Runner) -> dict[str, object] | None:
    """Classify only two fresh provider readbacks; SSH/API loss is never a cursor."""
    instance_id = instance.get("instance_id")
    if type(instance_id) is not int:
        raise ValueError("runtime GPU instance is invalid")
    try:
        observed = tuple(
            _json(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
            for _ in range(2)
        )
    except (subprocess.CalledProcessError, OSError, ValueError):
        return None
    if all(live in ({}, None) for live in observed):
        return {"kind": "instance_absent", "evidence_sha256": _hash({"instance_id": instance_id, "reason": "instance_absent", "readbacks": 2})}
    states: list[Mapping[str, object]] = []
    for live in observed:
        if not isinstance(live, Mapping) or live.get("id") != instance_id:
            raise ValueError("runtime provider loss readback is invalid")
        if live.get("actual_status") not in {"interrupted", "terminated", "stopped", "offline"}:
            return None
        states.append(live)
    if len(states) == 2:
        return {"kind": "preempted", "evidence_sha256": _hash([_project(live, _STABLE_INSTANCE_FIELDS) for live in states])}
    return None


def runtime_anchor_interruption_terminal(
    *, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner,
    hub: object,
) -> dict[str, object]:
    """Construct a loss terminal solely from fresh provider and durable Hub evidence."""
    required = {
        "checkpoint_repository", "checkpoint_anchor_ref", "checkpoint_experiment_id",
        "experiment_config", "runtime_source_evidence", "expected_checkpoint_steps",
        "terminal_receipt", "instance",
    }
    if set(request) != required:
        raise ValueError("runtime checkpoint interruption request has an incompatible schema")
    if request["checkpoint_repository"] != PARENT_CHECKPOINT["repository"] or request["checkpoint_anchor_ref"] != "main":
        raise ValueError("runtime checkpoint interruption anchor repository/ref is not approved")
    experiment_id = request["checkpoint_experiment_id"]
    config_path, identity_path, output = request["experiment_config"], request["runtime_source_evidence"], request["terminal_receipt"]
    if (
        not isinstance(experiment_id, str) or not experiment_id or "/" in experiment_id
        or not isinstance(config_path, str) or not isinstance(identity_path, str)
        or not isinstance(output, str) or not Path(output).is_absolute()
        or Path(output).exists() or Path(output).is_symlink()
        or not isinstance(request["expected_checkpoint_steps"], list)
        or request["expected_checkpoint_steps"] not in ([1000], [1000, 2000])
    ):
        raise ValueError("runtime checkpoint interruption request is malformed")
    loss = classify_runtime_provider_loss(instance=instance, runner=runner)
    if loss is None:
        raise ValueError("runtime checkpoint replacement requires repeated provider absence or preemption evidence")
    from lehome_train.groot.production_runtime import _runtime_checkpoint_identity_from_evidence
    from lehome_train.groot.runtime_checkpoint_lifecycle import (
        discover_runtime_checkpoint_anchor, provider_interruption_terminal,
        runtime_mixture_completion_terminal,
    )
    identity = _runtime_checkpoint_identity_from_evidence(
        _load_regular_json(Path(identity_path), "runtime checkpoint source evidence")
    )
    scratch = Path(tempfile.mkdtemp(prefix="runtime-interruption-anchor-"))
    try:
        discovered = discover_runtime_checkpoint_anchor(
            identity=identity, experiment_id=experiment_id,
            experiment_config_sha256=_hash(_load_regular_json(Path(config_path), "runtime checkpoint experiment config")),
            # Discovery owns an absent destination for checkpoint byte
            # readback.  ``mkdtemp`` has already created the parent scratch
            # directory, so use a fresh child rather than passing it itself.
            anchor_ref="main", hub=hub, destination=scratch / "checkpoint",  # type: ignore[arg-type]
        )
        step = discovered.resume.optimizer_step
        if step not in request["expected_checkpoint_steps"]:
            raise ValueError("runtime checkpoint anchor step is outside the expected interruption policy")
        if step == 2000:
            # A completed chain cannot be resumed.  Discovery has validated its
            # immutable 2K -> 1K link and freshly read both payloads, so route
            # directly into the canonical disposal-capable completion terminal.
            terminal = runtime_mixture_completion_terminal(
                identity=identity, instance_id=str(instance["instance_id"]),
                publications=discovered.publications,
            )
        else:
            terminal = provider_interruption_terminal(
                identity=identity, instance_id=str(instance["instance_id"]),
                publications=discovered.publications, provider_loss=loss,
            )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    atomic_write_json(Path(output), terminal)
    return {
        "paid_action": True, "action": "runtime-checkpoint-interrupted", "terminal": terminal,
        "immutable_anchor_revision": discovered.immutable_anchor_revision,
        "resumable_checkpoint_step": step,
    }


def resume_runtime_checkpoint(
    *, replacement: Mapping[str, object], request: Mapping[str, object], terminal: Mapping[str, object], hub: object, destination: Path,
) -> dict[str, object]:
    """Hydrate the module-authenticated cursor onto a distinct replacement lease."""
    identity = _runtime_checkpoint_identity(replacement, request)
    if terminal.get("instance_id") == str(replacement.get("instance_id")):
        raise ValueError("runtime checkpoint replacement must use a distinct instance")
    provider_loss = terminal.get("provider_loss")
    config_path = request.get("experiment_config")
    if not isinstance(provider_loss, Mapping) or not isinstance(config_path, str):
        raise ValueError("runtime checkpoint replacement requires loss evidence and experiment config")
    terminal_path = request.get("terminal_receipt")
    if type(terminal_path) is not str:
        raise ValueError("runtime checkpoint replacement requires its durable terminal path")
    terminal_file = Path(terminal_path)
    publications = terminal.get("immutable_checkpoint_publications")
    if not isinstance(publications, list) or len(publications) != 1 or not isinstance(publications[0], Mapping):
        raise ValueError("runtime checkpoint replacement requires exactly one 1K publication")
    publication_for_claim = dict(publications[0])
    if publication_for_claim.get("optimizer_step") != 1000 or not isinstance(publication_for_claim.get("runtime_cursor"), Mapping):
        raise ValueError("runtime checkpoint replacement cursor is not the 1K boundary")
    claim = {
        "schema_version": 1, "kind": "runtime_mixture_resume_claim",
        "terminal_sha256": sha256_file(terminal_file), "identity_sha256": identity.sha256,
        "cursor_sha256": _hash(dict(publication_for_claim["runtime_cursor"])),
        "publication_sha256": _hash(publication_for_claim),
        "replacement_instance_id": replacement.get("instance_id"),
        "experiment_config_sha256": sha256_file(Path(config_path)),
    }
    if type(claim["replacement_instance_id"]) is not int:
        raise ValueError("runtime checkpoint replacement instance is invalid")
    claim_path = terminal_file.with_name(terminal_file.name + ".resume-claim.json")
    if claim_path.exists() or claim_path.is_symlink():
        observed_claim = _load_regular_json(claim_path, "runtime checkpoint resume claim")
        if observed_claim == claim:
            raise RuntimeResumeAlreadyClaimed("runtime checkpoint cursor is already claimed by this replacement")
        raise ValueError("runtime checkpoint cursor is already claimed by another replacement")
    from lehome_train.groot.runtime_checkpoint_lifecycle import (
        discover_runtime_checkpoint_anchor, provider_interruption_terminal,
    )
    discovered = discover_runtime_checkpoint_anchor(
        identity=identity, experiment_id=str(request.get("checkpoint_experiment_id")),
        experiment_config_sha256=_hash(_load_regular_json(Path(config_path), "runtime checkpoint experiment config")),
        anchor_ref="main", hub=hub, destination=destination,  # type: ignore[arg-type]
    )
    step = discovered.resume.optimizer_step
    publication = {
        "schema_version": 1, "kind": "runtime_mixture_checkpoint_publication",
        **dict(discovered.anchor["checkpoint"]), "identity": identity.to_dict(),
        "identity_sha256": identity.sha256,
        "runtime_cursor": {"optimizer_step": step, "global_sample_offset": step * 64, "physical_batch_size": 64, "action_horizon": 16},
        "fresh_tree_readback_verified": True,
    }
    recovered_terminal = provider_interruption_terminal(
        identity=identity, instance_id=str(terminal.get("instance_id")),
        publications=(publication,), provider_loss=provider_loss,
    )
    cursor = discovered.resume
    # This is the exact cursor consumed by ``runtime-mixture-train`` and then
    # copied unchanged through final staging.  ``dataset_kwargs`` is an
    # entrypoint-only projection and drops the h16 binding, so it must never
    # become the lifecycle receipt.
    runtime_cursor = {
        "optimizer_step": cursor.optimizer_step,
        "global_sample_offset": cursor.global_sample_offset,
        "physical_batch_size": cursor.physical_batch_size,
        "action_horizon": 16,
    }
    # Claim only after discovery has resolved a stable immutable ref and
    # materialized both authenticated bytes.  A transient Hub readback failure
    # therefore abort-cleans its lease without consuming the durable cursor.
    if publication != publication_for_claim:
        raise ValueError("runtime checkpoint anchor no longer matches its interruption terminal")
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        observed_claim = _load_regular_json(claim_path, "runtime checkpoint resume claim")
        if observed_claim == claim:
            raise RuntimeResumeAlreadyClaimed("runtime checkpoint cursor is already claimed by this replacement")
        raise ValueError("runtime checkpoint cursor is already claimed by another replacement")
    else:
        try:
            payload = json.dumps(claim, sort_keys=True, separators=(",", ":")).encode("utf-8")
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if _load_regular_json(claim_path, "runtime checkpoint resume claim") != claim:
            raise ValueError("runtime checkpoint resume claim readback mismatches")
    return {"paid_action": True, "action": "runtime-checkpoint-resume", "instance_id": replacement["instance_id"], "runtime_cursor": runtime_cursor, "checkpoint_archive": str(cursor.checkpoint_archive), "checkpoint_descriptor": str(cursor.checkpoint_descriptor), "runtime_resume_anchor": discovered.previous_link(), "runtime_resume_publication": publication, "recovered_terminal": recovered_terminal, "immutable_anchor_revision": discovered.immutable_anchor_revision, "runtime_resume_claim": str(claim_path)}


def destroy_runtime_checkpoint_completion(
    *, instance: Mapping[str, object], request: Mapping[str, object], terminal: Mapping[str, object], hub: object,
    runner: Runner, max_absence_polls: int = RUNTIME_ABSENCE_READBACK_POLLS,
    sleep: Callable[[float], None] = _bounded_sleep,
) -> dict[str, object]:
    """Authorize both readbacks before destroying the exact completed GPU lease."""
    output = _runtime_failure_receipt_path(request)
    instance_id = instance.get("instance_id")
    if type(instance_id) is not int:
        raise ValueError("runtime checkpoint disposal instance is invalid")
    receipt = {
        "schema_version": 1, "kind": "runtime_mixture_abort_cleanup", "instance_id": instance_id,
        "provider_response_sha256": instance.get("provider_response_sha256"),
        "code_revision": request.get("code_revision"), "code_bundle_sha256": request.get("code_bundle_sha256"),
        "parent_checkpoint_artifact_sha256": PARENT_CHECKPOINT["artifact_sha256"],
        "error_type": "RuntimeError", "error": "redacted remote failure", "disposable": False,
    }
    def record(status: str) -> None:
        if not output.exists() and not output.is_symlink():
            atomic_write_json(output, receipt | {"cleanup_status": status})
    identity = _runtime_checkpoint_identity(instance, request)
    from lehome_train.groot.runtime_checkpoint_lifecycle import authorize_runtime_mixture_disposal
    authorization = authorize_runtime_mixture_disposal(instance_id=str(instance_id), terminal=terminal, identity=identity, hub=hub)  # type: ignore[arg-type]
    try:
        runner(("vastai", "destroy", "instance", str(instance_id), "--yes"))
    except BaseException as error:
        record("destroy_failed")
        raise RuntimeError("runtime checkpoint destroy command failed") from error
    try:
        absent = _await_runtime_instance_absence(
            instance_id=instance_id, runner=runner, max_polls=max_absence_polls, sleep=sleep,
        )
    except BaseException as error:
        record("absence_unverified")
        raise RuntimeError("runtime checkpoint destroy could not verify instance absence") from error
    if not absent:
        record("absence_unverified")
        raise RuntimeError("runtime checkpoint destroy did not verify instance absence")
    return {"paid_action": True, "destroy_authorized": True, "instance_id": instance_id, "authorization": authorization}


class _RuntimeCheckpointHub:
    """Token-bound adapter for the checkpoint module's deliberately small Hub API."""

    def __init__(self, *, transport: HubTransport, token: str) -> None:
        self._transport, self._token = transport, token

    def list_tree(self, *, repository: str, revision: str) -> object:
        return self._transport.list_tree(repository=repository, revision=revision, token=self._token)

    def download_files(
        self, *, repository: str, revision: str, destination: Path,
        relative_paths: tuple[str, ...], remote_prefix: str,
    ) -> object:
        return self._transport.download_files(
            repository=repository, revision=revision, destination=destination,
            relative_paths=relative_paths, remote_prefix=remote_prefix, token=self._token,
        )

    def resolve_approved_ref(self, *, repository: str, ref: str) -> str:
        return self._transport.resolve_approved_ref(
            repository=repository, ref=ref, token=self._token,
        )


def _runtime_checkpoint_terminal_output(request: Mapping[str, object], terminal: Mapping[str, object]) -> dict[str, object]:
    """Optionally persist an immutable terminal without accepting an overwrite."""
    output = request.get("terminal_receipt")
    if output is None:
        return dict(terminal)
    if type(output) is not str or not Path(output).is_absolute() or Path(output).exists() or Path(output).is_symlink():
        raise ValueError("runtime checkpoint terminal receipt must be an absent absolute path")
    atomic_write_json(Path(output), dict(terminal))
    return dict(terminal)


def _runtime_checkpoint_publisher(*, request: Mapping[str, object], token: str):
    repository, revision, experiment, root = (
        request.get("checkpoint_repository"), request.get("checkpoint_revision"),
        request.get("checkpoint_experiment_id"), request.get("checkpoint_artifact_root"),
    )
    if (
        repository != PARENT_CHECKPOINT["repository"] or type(revision) is not str
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None or type(experiment) is not str
        or not experiment or type(root) is not str
    ):
        raise ValueError("runtime checkpoint publisher identity is invalid")
    from lehome_train.groot.production_adapters import HubCheckpointUploader
    from lehome_train.groot.runtime_checkpoint_lifecycle import hub_checkpoint_publisher
    return hub_checkpoint_publisher(HubCheckpointUploader(
        repository=repository, revision=revision, experiment_id=experiment,
        artifact_root=root, token=token,
    ))


def runtime_mixture_pilot_provider_plan() -> dict[str, object]:
    """Describe, but never place, the separately approved CPU-pilot rental."""
    return {
        "paid_action": False, "action": "runtime-pilot-plan", "provider_action": "not_rented",
        "platform_arch": "x86_64", "purchase_option": "on_demand",
        "account_hourly_cap_usd": MAX_ACCOUNT_HOURLY_USD, "max_instances": 1,
    }


def runtime_mixture_train(*, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner) -> dict[str, object]:
    """Execute only the receipt-bound runtime-mixture trainer, never legacy RFT."""
    _runtime_gpu_warmup_instance(instance)
    identity = _runtime_identity(instance, request)
    warmup_path = request.get("warmup_lifecycle_receipt")
    if type(warmup_path) is not str:
        raise ValueError("runtime mixture train requires the measured GPU warm-up lifecycle receipt")
    warmup, measured_workers = _validated_runtime_gpu_warmup_lifecycle(
        path=Path(warmup_path), instance=instance, request=request,
    )
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
        "warmup_lifecycle_receipt_sha256": sha256_file(Path(warmup_path)),
        "selected_loader_workers": measured_workers,
        "runtime_command": "runtime-mixture-train", "throughput_verified": measured_workers == warmup["selected_loader_workers"],
    }
    atomic_write_json(Path(output), receipt)
    return {"paid_action": True, "action": "runtime-train", "instance_id": instance["instance_id"], "execution_receipt": receipt}


def _runtime_stage_selected_workers(*, selected_path: Path, launch_path: Path) -> tuple[int, str]:
    """Bind the immutable warm-up selection to the already-generated launch JSON.

    The lifecycle never rewrites the launch configuration after a warm-up.  The
    canonical runtime command performs the same comparison immediately before
    it starts the trainer; this preflight catches a mismatched staged artifact.
    """
    selected = _load_regular_json(selected_path, "runtime selected-workers receipt")
    workers = selected.get("selected_loader_workers")
    if type(workers) is not int or workers < 0 or workers > 64:
        raise ValueError("runtime selected-workers receipt is invalid")
    launch = _load_regular_json(launch_path, "runtime launch config")
    if launch.get("dataloader_num_workers") != workers:
        raise ValueError("runtime launch config workers do not match authenticated warm-up selection")
    return workers, sha256_file(selected_path)


def _runtime_final_launch_contract(
    launch: Mapping[str, object], *, parent: Mapping[str, object],
) -> dict[str, object]:
    """Return the immutable final-launch subset shared with direct GPU warm-up."""
    expected = {
        "base_model_revision": MODEL_REVISION,
        "physical_batch_size": 64, "global_batch_size": 64, "num_gpus": 1,
        "max_steps": 2000, "save_steps": 1000,
        "training_action_horizon": 16, "model_action_chunk_capacity": 40,
        "parent_checkpoint_repository": parent["repository"],
        "parent_checkpoint_revision": parent["revision"],
        "parent_checkpoint_subpath": parent["subpath"],
        "parent_checkpoint_artifact_sha256": parent["artifact_sha256"],
    }
    base_model_path = launch.get("base_model_path")
    path = Path(base_model_path) if type(base_model_path) is str else None
    if (
        base_model_path != "/cache/parent"
        and (path is None or path.name != "parent" or path.parent.name != "cache")
    ) or any(launch.get(key) != value for key, value in expected.items()):
        raise ValueError("runtime final launch config is not the approved direct-GPU campaign")
    return expected


def _runtime_final_experiment_contract(
    experiment: Mapping[str, object], *, mixture: Mapping[str, object],
) -> None:
    expected = {
        "container_digest": BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2],
        "model_revision": MODEL_REVISION, "physical_batch_size": 64,
        "gradient_accumulation_steps": 1, "sample_presentations": 128_000,
        "action_horizon": 16, "tune_language_backbone": False,
        "tune_visual_backbone": False, "dataset_repository": mixture["repository"],
        "dataset_revision": mixture["revision"], "dataset_manifest_sha256": mixture["mixture_id"],
    }
    if any(experiment.get(key) != value for key, value in expected.items()):
        raise ValueError("runtime final experiment is not the approved direct-GPU campaign")


def _runtime_envelope(path: Path, *, command: str, fields: set[str], label: str) -> Mapping[str, object]:
    """Reject drifted runtime command envelopes before any bytes reach a lease."""
    envelope = _load_regular_json(path, label)
    arguments = envelope.get("arguments")
    if (
        set(envelope) != {"schema_version", "command", "arguments"}
        or envelope.get("schema_version") != 1 or envelope.get("command") != command
        or not isinstance(arguments, Mapping) or set(arguments) != fields
    ):
        raise ValueError(f"{label} is not the canonical {command} envelope")
    serialized = json.dumps(envelope, sort_keys=True)
    if "/prepared/generation" in serialized or "continuous-train" in serialized:
        raise ValueError("runtime stage rejects legacy generation request content")
    return arguments


def _runtime_bootstrap_receipt(*, path: Path, instance: Mapping[str, object], request: Mapping[str, object]) -> Mapping[str, object]:
    receipt = _load_regular_json(path, "runtime bootstrap stage receipt")
    identity = _runtime_campaign_binding(request)
    expected = {
        "schema_version": 1, "kind": "runtime_mixture_bootstrap_stage",
        "code_revision": request.get("code_revision"), "code_bundle_sha256": request.get("code_bundle_sha256"),
        "trainer_image": instance.get("trainer_image"),
        "image_digest": instance.get("image_digest"),
        "bc_revision": identity["bc"]["immutable_revision"],
        "rollout_revision": identity["rollout"]["immutable_revision"],
        "deployment_revision": identity["deployment"]["immutable_revision"],
        "bc_receipt_sha256": identity["bc_receipt_sha256"],
        "rollout_receipt_sha256": identity["rollout_receipt_sha256"],
        "deployment_receipt_sha256": identity["deployment_receipt_sha256"],
    }
    if instance.get("kind") == "runtime_mixture_gpu_warmup_instance":
        capability_path = request.get("bootstrap_capability_receipt")
        if type(capability_path) is not str:
            raise ValueError("runtime bootstrap receipt lacks the same-lease capability receipt")
        _runtime_gpu_bootstrap_capability_receipt(
            path=Path(capability_path), instance=instance, request=request, identity=identity,
        )
        expected |= {
            "parent_archive_sha256": PARENT_CHECKPOINT["archive_sha256"],
            "parent_checkpoint_artifact_sha256": PARENT_CHECKPOINT["artifact_sha256"],
            "bootstrap_capability_receipt_sha256": sha256_file(Path(capability_path)),
        }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("runtime bootstrap stage is not bound to its direct runtime instance")
    if (
        receipt.get("instance_id") != instance.get("instance_id")
        or receipt.get("provider_response_sha256") != instance.get("provider_response_sha256")
        or not isinstance(receipt.get("transfers"), list)
        or not receipt["transfers"]
        or any(not isinstance(item, Mapping) or set(item) != {"name", "sha256"} or not isinstance(item.get("name"), str) or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256"))) is None for item in receipt["transfers"])
    ):
        raise ValueError("runtime bootstrap receipt lacks immutable transfer evidence")
    return receipt


def runtime_mixture_bootstrap_stage(*, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner) -> dict[str, object]:
    """Stage immutable inputs on the direct-GPU lease before warm-up."""
    identity = _runtime_hydration_identity(instance, request)
    required = {
        "code_bundle": "code.bundle", "code_bundle_sha256_file": "code.bundle.sha256",
        "token_file": "runtime.token",
        "runtime_hydrate_request": "runtime-hydrate.json",
        "bc_readback_receipt": "bc-readback.json", "rollout_readback_receipt": "rollout-readback.json",
        "deployment_receipt": "deployment-receipt.json",
    }
    is_cpu_diagnostic = instance.get("kind") == "runtime_mixture_cpu_pilot_instance"
    if is_cpu_diagnostic:
        required["runtime_pilot_request"] = "runtime-pilot.json"
    else:
        required["parent_checkpoint"] = "parent.tar"
    if any(type(request.get(field)) is not str or not request[field] for field in required):
        raise ValueError("runtime bootstrap stage requires only reviewed immutable inputs")
    output = request.get("bootstrap_receipt")
    if type(output) is not str or not Path(output).is_absolute() or Path(output).exists() or Path(output).is_symlink():
        raise ValueError("runtime bootstrap stage requires an absent absolute bootstrap_receipt")
    code = Path(str(request["code_bundle"]))
    if _verify_reviewed_code_bundle(code, Path(str(request["code_bundle_sha256_file"])), request.get("code_revision")) != request.get("code_bundle_sha256"):
        raise ValueError("runtime bootstrap code bundle differs from the reviewed bundle")
    _read_private_token(str(request["token_file"]))
    parent = None if is_cpu_diagnostic else _runtime_parent_checkpoint(request)
    bootstrap_capability = None
    bootstrap_capability_sha = None
    if parent is not None:
        capability_path = request.get("bootstrap_capability_receipt")
        if type(capability_path) is not str:
            raise ValueError("runtime bootstrap stage requires the same-lease capability receipt")
        bootstrap_capability = _runtime_gpu_bootstrap_capability_receipt(
            path=Path(capability_path), instance=instance, request=request, identity=identity,
        )
        bootstrap_capability_sha = sha256_file(Path(capability_path))
    hydrate = _runtime_envelope(Path(str(request["runtime_hydrate_request"])), command="hydrate-runtime-mixture", fields={"deployment_receipt", "source_readback_receipts", "destination", "mounts_descriptor"}, label="runtime hydration request")
    if hydrate["deployment_receipt"] != "/prepared/config/deployment-receipt.json" or hydrate["destination"] != "/prepared/runtime" or hydrate["mounts_descriptor"] != "/prepared/runtime/mounts.json":
        raise ValueError("runtime hydration request paths are not canonical")
    if is_cpu_diagnostic:
        pilot = _runtime_envelope(Path(str(request["runtime_pilot_request"])), command="pilot-runtime-mixture", fields={"mixture_manifest", "mounts_descriptor", "sample_count", "worker_counts", "timeout_seconds", "authenticated_evidence"}, label="runtime pilot request")
        if pilot["mixture_manifest"] != "/prepared/runtime/mixture.json" or pilot["mounts_descriptor"] != "/prepared/runtime/mounts.json":
            raise ValueError("runtime pilot request paths are not canonical")
    transfer_sources: list[tuple[Path, str, str]] = []
    for field, remote_name in required.items():
        source = Path(str(request[field]))
        if source.is_symlink() or not source.is_file():
            raise ValueError("runtime bootstrap input must be a regular file")
        digest = sha256_file(source)
        transfer_sources.append((source, remote_name, digest))
    remote_dir = "/tmp/lehome-runtime-bootstrap"
    runner((*_ssh_prefix(instance), "mkdir -p " + remote_dir))
    transfers: list[dict[str, str]] = []
    for source, remote_name, digest in transfer_sources:
        runner((*_scp_prefix(instance), str(source), "root@" + str(instance["host"]) + ":" + remote_dir + "/" + remote_name))
        observed = runner((*_ssh_prefix(instance), "sha256sum " + remote_dir + "/" + remote_name)).strip().split()
        if not observed or observed[0] != digest:
            raise ValueError("runtime bootstrap staged hash readback failed")
        transfers.append({"name": remote_name, "sha256": digest})
    parent_setup = ""
    if parent is not None:
        parent_setup = (
            "test ! -e /cache/parent; mkdir -p /cache/parent; "
            "test \"$(sha256sum " + remote_dir + "/parent.tar | cut -d' ' -f1)\" = " + PARENT_CHECKPOINT["archive_sha256"] + "; "
            "tar --no-same-owner --no-same-permissions -xf " + remote_dir + "/parent.tar -C /cache/parent; "
            "PYTHONPATH=/prepared/code/trainer/src python -c \"from lehome_train.groot.checkpoint_identity import policy_artifact_sha256; assert policy_artifact_sha256('/cache/parent') == '" + PARENT_CHECKPOINT["artifact_sha256"] + "'\"; "
            "test ! -L /cache/parent; "
        )
    runner((*_ssh_prefix(instance), "set -eu; mkdir -p /prepared/config /cache /output; cd " + remote_dir + "; sha256sum -c code.bundle.sha256; mv runtime-hydrate.json /prepared/config/runtime-hydrate.json; " + ("mv runtime-pilot.json /prepared/config/runtime-pilot.json; " if is_cpu_diagnostic else "") + "mv runtime.token /prepared/config/runtime.token; mv bc-readback.json /prepared/config/bc-readback.json; mv rollout-readback.json /prepared/config/rollout-readback.json; mv deployment-receipt.json /prepared/config/deployment-receipt.json; git clone --quiet --no-checkout " + remote_dir + "/code.bundle /prepared/code; git -C /prepared/code checkout --quiet --detach " + str(request["code_revision"]) + "; test \"$(git -C /prepared/code rev-parse HEAD)\" = " + str(request["code_revision"]) + "; test -z \"$(git -C /prepared/code status --porcelain)\"; " + parent_setup + "chmod 600 /prepared/config/runtime.token; test ! -L /prepared/code; test ! -e /prepared/runtime"))
    receipt = {"schema_version": 1, "kind": "runtime_mixture_bootstrap_stage", "instance_id": instance["instance_id"], "provider_response_sha256": instance["provider_response_sha256"], "platform_arch": "x86_64", "trainer_image": instance["trainer_image"], "image_digest": instance["image_digest"], "code_revision": request["code_revision"], "code_bundle_sha256": request["code_bundle_sha256"], "bc_revision": identity["bc"]["immutable_revision"], "rollout_revision": identity["rollout"]["immutable_revision"], "deployment_revision": identity["deployment"]["immutable_revision"], "bc_receipt_sha256": identity["bc_receipt_sha256"], "rollout_receipt_sha256": identity["rollout_receipt_sha256"], "deployment_receipt_sha256": identity["deployment_receipt_sha256"], "transfers": transfers}
    if parent is not None:
        assert bootstrap_capability is not None and bootstrap_capability_sha is not None
        receipt |= {
            "parent_archive_sha256": PARENT_CHECKPOINT["archive_sha256"],
            "parent_checkpoint_artifact_sha256": PARENT_CHECKPOINT["artifact_sha256"],
            "bootstrap_capability_receipt_sha256": bootstrap_capability_sha,
        }
    atomic_write_json(Path(output), receipt)
    return {"paid_action": True, "action": "runtime-bootstrap-stage", "bootstrap_receipt": receipt}


def runtime_mixture_warmup_stage(*, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner) -> dict[str, object]:
    """Stage the direct-GPU measurement slot without touching final inputs."""
    identity = _runtime_identity(instance, request)
    _runtime_bootstrap_receipt(path=Path(str(request.get("bootstrap_receipt", ""))), instance=instance, request=request)
    required = {
        "runtime_warmup_binding": "runtime-warmup-binding.json",
        "runtime_warmup_request": "runtime-warmup.json", "warmup_launch_config": "launch.json",
    }
    if any(type(request.get(key)) is not str or not request[key] for key in required):
        raise ValueError("runtime warmup stage requires binding, warmup request, and isolated launch config")
    output = request.get("warmup_stage_receipt")
    if type(output) is not str or not Path(output).is_absolute() or Path(output).exists() or Path(output).is_symlink():
        raise ValueError("runtime warmup stage requires an absent absolute warmup_stage_receipt")
    _runtime_envelope(Path(str(request["runtime_warmup_request"])), command="runtime-gpu-warmup", fields={"binding"}, label="runtime GPU warm-up request")
    launch = _load_regular_json(Path(str(request["warmup_launch_config"])), "runtime warmup launch config")
    binding = _runtime_warmup_binding(
        binding=_load_regular_json(Path(str(request["runtime_warmup_binding"])), "runtime warmup binding"),
        instance=instance, request=request, identity=identity,
    )
    launch_contract = _runtime_final_launch_contract(launch, parent=binding["parent_checkpoint"])
    if launch.get("training_action_horizon") != 16 or launch.get("model_action_chunk_capacity") != 40:
        raise ValueError("runtime warmup launch config is incompatible")
    remote_dir = "/tmp/lehome-runtime-warmup"
    runner((*_ssh_prefix(instance), "mkdir -p " + remote_dir))
    transfers: list[dict[str, str]] = []
    for field, remote_name in required.items():
        source = Path(str(request[field]))
        if source.is_symlink() or not source.is_file():
            raise ValueError("runtime warmup stage input must be a regular file")
        digest = sha256_file(source)
        runner((*_scp_prefix(instance), str(source), "root@" + str(instance["host"]) + ":" + remote_dir + "/" + remote_name))
        observed = runner((*_ssh_prefix(instance), "sha256sum " + remote_dir + "/" + remote_name)).strip().split()
        if not observed or observed[0] != digest:
            raise ValueError("runtime warmup staged hash readback failed")
        transfers.append({"name": remote_name, "sha256": digest})
    # The production warm-up adapter consumes the canonical staged launch
    # location.  Final staging later replaces this isolated warm-up config.
    runner((*_ssh_prefix(instance), "set -eu; mv " + remote_dir + "/runtime-warmup-binding.json /prepared/config/runtime-warmup-binding.json; mv " + remote_dir + "/runtime-warmup.json /prepared/config/runtime-warmup.json; mv " + remote_dir + "/launch.json /prepared/config/launch.json; test -d /prepared/code; test ! -L /prepared/code"))
    receipt = {"schema_version": 1, "kind": "runtime_mixture_warmup_stage", "instance_id": instance["instance_id"], "provider_response_sha256": instance["provider_response_sha256"], "bootstrap_receipt_sha256": sha256_file(Path(str(request["bootstrap_receipt"]))), "launch_contract": launch_contract, "transfers": transfers}
    atomic_write_json(Path(output), receipt)
    return {"paid_action": True, "action": "runtime-warmup-stage", "warmup_stage_receipt": receipt}


def runtime_mixture_stage(*, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner) -> dict[str, object]:
    """Securely stage reviewed runtime-only inputs; legacy generation is absent."""
    _runtime_identity(instance, request)
    required = {
        "bootstrap_receipt": "bootstrap.json",
        "warmup_stage_receipt": "warmup-stage.json",
        "code_bundle": "code.bundle", "code_bundle_sha256_file": "code.bundle.sha256",
        "launch_config": "launch.json", "experiment_config": "experiment.json",
        "runtime_train_request": "runtime-train.json", "runtime_hydrate_request": "runtime-hydrate.json",
        "runtime_warmup_request": "runtime-warmup.json",
        "modality_config": "modality.py", "token_file": "runtime.token",
        "gpu_warmup_receipt": "gpu-warmup.json",
        "runtime_warmup_binding": "runtime-warmup-binding.json", "selected_workers": "selected-workers.json",
        "runtime_source_evidence": "source-evidence.json",
        "bc_readback_receipt": "bc-readback.json", "rollout_readback_receipt": "rollout-readback.json",
        "deployment_receipt": "deployment-receipt.json",
    }
    if any(type(request.get(field)) is not str or not request[field] for field in required):
        raise ValueError("runtime mixture stage requires reviewed code, config, token, and authenticated receipts")
    _runtime_bootstrap_receipt(
        path=Path(str(request["bootstrap_receipt"])), instance=instance, request=request,
    )
    warmup_stage = _load_regular_json(Path(str(request["warmup_stage_receipt"])), "runtime warmup stage receipt")
    if (
        warmup_stage.get("kind") != "runtime_mixture_warmup_stage"
        or warmup_stage.get("instance_id") != instance.get("instance_id")
        or warmup_stage.get("provider_response_sha256") != instance.get("provider_response_sha256")
        or warmup_stage.get("bootstrap_receipt_sha256") != sha256_file(Path(str(request["bootstrap_receipt"])))
    ):
        raise ValueError("runtime final stage is not bound to the measured warmup slot")
    warmup_lifecycle, measured_workers = _validated_runtime_gpu_warmup_lifecycle(
        path=Path(str(request["gpu_warmup_receipt"])), instance=instance, request=request,
    )
    code = Path(str(request["code_bundle"]))
    if _verify_reviewed_code_bundle(code, Path(str(request["code_bundle_sha256_file"])), request.get("code_revision")) != request.get("code_bundle_sha256"):
        raise ValueError("runtime mixture stage code bundle receipt differs from the reviewed bundle")
    source_evidence = _load_regular_json(Path(str(request["runtime_source_evidence"])), "runtime checkpoint source evidence")
    runtime_identity = _runtime_identity(instance, request)
    if (
        source_evidence.get("mixture_id") != runtime_identity["deployment"]["mixture_id"]
        or source_evidence.get("deployment_receipt_sha256") != runtime_identity["deployment_receipt_sha256"]
        or source_evidence.get("code_bundle_sha256") != request["code_bundle_sha256"]
        or source_evidence.get("code_bundle_revision") != request["code_revision"]
        or source_evidence.get("oci_image") != BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2]
        or source_evidence.get("parent_step12000_artifact_sha256") != PARENT_CHECKPOINT["artifact_sha256"]
        or source_evidence.get("physical_batch_size") != 64 or source_evidence.get("action_horizon") != 16
        or type(source_evidence.get("schedule_seed")) is not int
        or source_evidence.get("source_revisions") != [
            {"source_id": "organizer", "immutable_revision": runtime_identity["bc"]["immutable_revision"], "prefix": "bc/full", "tree_sha256": runtime_identity["bc_receipt_sha256"]},
            {"source_id": "rollout", "immutable_revision": runtime_identity["rollout"]["immutable_revision"], "prefix": runtime_identity["rollout"]["remote_prefix"], "tree_sha256": runtime_identity["rollout_receipt_sha256"]},
        ]
    ):
        raise ValueError("runtime checkpoint source evidence is not bound to staged provider, code, and runtime artifacts")
    selected_workers, selected_workers_sha256 = _runtime_stage_selected_workers(
        selected_path=Path(str(request["selected_workers"])), launch_path=Path(str(request["launch_config"])),
    )
    final_launch = _load_regular_json(Path(str(request["launch_config"])), "runtime final launch config")
    expected_launch = _runtime_final_launch_contract(
        final_launch, parent=warmup_lifecycle["runtime_warmup_binding"]["parent_checkpoint"],
    )
    if warmup_stage.get("launch_contract") != expected_launch:
        raise ValueError("runtime final launch config drifted from the measured GPU warm-up")
    _runtime_final_experiment_contract(
        _load_regular_json(Path(str(request["experiment_config"])), "runtime final experiment config"),
        mixture=warmup_lifecycle["runtime_warmup_binding"]["mixture"],
    )
    if selected_workers != measured_workers:
        raise ValueError("runtime final stage selected workers do not match the measured GPU warm-up")
    resume_archive, resume_descriptor, resume_cursor = (
        request.get("runtime_resume_archive"), request.get("runtime_resume_descriptor"), request.get("runtime_resume_cursor"),
    )
    if (resume_archive is None and resume_descriptor is None and resume_cursor is None):
        resume_transfers: dict[str, str] = {}
    elif (
        type(resume_archive) is str and type(resume_descriptor) is str
        and isinstance(resume_cursor, Mapping) and isinstance(request.get("runtime_resume_anchor"), Mapping)
        and isinstance(request.get("runtime_resume_publication"), Mapping)
    ):
        resume_transfers = {"runtime_resume_archive": "runtime-resume.tar", "runtime_resume_descriptor": "runtime-resume.json"}
    else:
        raise ValueError("runtime stage resume requires exact archive, descriptor, and cursor together")
    _read_private_token(str(request["token_file"]))
    train = _runtime_envelope(Path(str(request["runtime_train_request"])), command="runtime-mixture-train", fields={"launch_config", "experiment_config", "runtime_manifest", "runtime_window_index", "runtime_normalization", "runtime_mounts_descriptor", "runtime_source_evidence", "warmup_receipt", "runtime_warmup_binding", "runtime_resume_archive", "runtime_resume_descriptor", "runtime_resume_cursor", "runtime_resume_anchor", "runtime_resume_publication", "checkpoint_repository", "checkpoint_revision", "publisher_token_file", "instance_id", "result_output", "status_output"}, label="runtime train request")
    expected_train_paths = {
        "launch_config": "/prepared/config/launch.json", "experiment_config": "/prepared/config/experiment.json",
        "runtime_manifest": "/prepared/runtime/mixture.json", "runtime_window_index": "/prepared/runtime/windows.json",
        "runtime_normalization": "/prepared/runtime/mixture-normalization.json", "runtime_mounts_descriptor": "/prepared/runtime/mounts.json",
        "runtime_source_evidence": "/prepared/runtime/source-evidence.json",
        "warmup_receipt": "/prepared/config/gpu-warmup.json", "runtime_warmup_binding": "/prepared/config/runtime-warmup-binding.json",
        "publisher_token_file": "/prepared/config/runtime.token",
        "result_output": "/output/runtime-train-result.json", "status_output": "/output/runtime-train-status.json",
    }
    if any(train.get(key) != value for key, value in expected_train_paths.items()):
        raise ValueError("runtime train request paths are not canonical")
    if train.get("checkpoint_repository") != PARENT_CHECKPOINT["repository"] or train.get("instance_id") != instance.get("instance_id") or train.get("checkpoint_revision") != "main":
        raise ValueError("runtime train publication provenance is not canonical")
    if resume_transfers:
        if (
            train.get("runtime_resume_archive") != "/prepared/config/runtime-resume.tar"
            or train.get("runtime_resume_descriptor") != "/prepared/config/runtime-resume.json"
            or train.get("runtime_resume_cursor") != resume_cursor
            or train.get("runtime_resume_anchor") != request.get("runtime_resume_anchor")
            or train.get("runtime_resume_publication") != request.get("runtime_resume_publication")
        ):
            raise ValueError("runtime train request resume inputs are not the staged authenticated cursor")
    elif any(train.get(key) is not None for key in ("runtime_resume_archive", "runtime_resume_descriptor", "runtime_resume_cursor", "runtime_resume_anchor", "runtime_resume_publication")):
        raise ValueError("initial runtime train request must not invent a resume cursor")
    _runtime_envelope(Path(str(request["runtime_hydrate_request"])), command="hydrate-runtime-mixture", fields={"deployment_receipt", "source_readback_receipts", "destination", "mounts_descriptor"}, label="runtime hydration request")
    _runtime_envelope(Path(str(request["runtime_warmup_request"])), command="runtime-gpu-warmup", fields={"binding"}, label="runtime GPU warm-up request")
    remote_dir = "/tmp/lehome-runtime-stage"
    runner((*_ssh_prefix(instance), "mkdir -p " + remote_dir))
    transfers: list[dict[str, str]] = []
    transfer_fields = (required | resume_transfers).copy()
    transfer_fields.pop("gpu_warmup_receipt")
    for field, remote_name in transfer_fields.items():
        source = Path(str(request[field]))
        if source.is_symlink() or not source.is_file():
            raise ValueError("runtime mixture stage input must be a regular file")
        digest = sha256_file(source)
        runner((*_scp_prefix(instance), str(source), "root@" + str(instance["host"]) + ":" + remote_dir + "/" + remote_name))
        observed = runner((*_ssh_prefix(instance), "sha256sum " + remote_dir + "/" + remote_name)).strip().split()
        if not observed or observed[0] != digest:
            raise ValueError("runtime mixture staged hash readback failed")
        transfers.append({"name": remote_name, "sha256": digest})
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json") as handle:
        json.dump(warmup_lifecycle["warmup_receipt"], handle, sort_keys=True)
        handle.flush()
        inner_warmup = Path(handle.name)
        digest = sha256_file(inner_warmup)
        runner((*_scp_prefix(instance), str(inner_warmup), "root@" + str(instance["host"]) + ":" + remote_dir + "/gpu-warmup.json"))
        observed = runner((*_ssh_prefix(instance), "sha256sum " + remote_dir + "/gpu-warmup.json")).strip().split()
        if not observed or observed[0] != digest:
            raise ValueError("runtime mixture staged warm-up receipt readback failed")
        transfers.append({"name": "gpu-warmup.json", "sha256": digest})
    runner((*_ssh_prefix(instance), "set -eu; test -d /prepared/code; test \"$(git -C /prepared/code rev-parse HEAD)\" = " + str(request["code_revision"]) + "; test -d /cache/parent; test ! -L /cache/parent; PYTHONPATH=/prepared/code/trainer/src python -c \"from lehome_train.groot.checkpoint_identity import policy_artifact_sha256; assert policy_artifact_sha256('/cache/parent') == '" + PARENT_CHECKPOINT["artifact_sha256"] + "'\"; mkdir -p /prepared/final-slot; test ! -e /prepared/final-slot/bootstrap-code; mv /prepared/code /prepared/final-slot/bootstrap-code"))
    runner((*_ssh_prefix(instance), "set -eu; test -d /prepared/runtime; mkdir -p /prepared/config /output; mv " + remote_dir + "/launch.json /prepared/config/launch.json; mv " + remote_dir + "/experiment.json /prepared/config/experiment.json; mv " + remote_dir + "/runtime-train.json /prepared/config/runtime-train.json; mv " + remote_dir + "/runtime-hydrate.json /prepared/config/runtime-hydrate.json; mv " + remote_dir + "/runtime-warmup.json /prepared/config/runtime-warmup.json; mv " + remote_dir + "/modality.py /prepared/config/modality.py; mv " + remote_dir + "/runtime.token /prepared/config/runtime.token; mv " + remote_dir + "/gpu-warmup.json /prepared/config/gpu-warmup.json; mv " + remote_dir + "/runtime-warmup-binding.json /prepared/config/runtime-warmup-binding.json; mv " + remote_dir + "/selected-workers.json /prepared/config/selected-workers.json; mv " + remote_dir + "/source-evidence.json /prepared/runtime/source-evidence.json; mv " + remote_dir + "/bc-readback.json /prepared/config/bc-readback.json; mv " + remote_dir + "/rollout-readback.json /prepared/config/rollout-readback.json; mv " + remote_dir + "/deployment-receipt.json /prepared/config/deployment-receipt.json; git clone --quiet --no-checkout " + remote_dir + "/code.bundle /prepared/code; git -C /prepared/code checkout --quiet --detach " + str(request["code_revision"]) + "; test \"$(git -C /prepared/code rev-parse HEAD)\" = " + str(request["code_revision"]) + "; test -z \"$(git -C /prepared/code status --porcelain)\"; chmod 600 /prepared/config/runtime.token; test ! -L /prepared/code; test ! -L /cache/parent"))
    if resume_transfers:
        runner((*_ssh_prefix(instance), "set -eu; mv " + remote_dir + "/runtime-resume.tar /prepared/config/runtime-resume.tar; mv " + remote_dir + "/runtime-resume.json /prepared/config/runtime-resume.json"))
    return {"paid_action": True, "action": "runtime-stage", "instance_id": instance["instance_id"], "code_bundle_sha256": request["code_bundle_sha256"], "bootstrap_receipt_sha256": sha256_file(Path(str(request["bootstrap_receipt"]))), "warmup_lifecycle_receipt_sha256": sha256_file(Path(str(request["gpu_warmup_receipt"]))), "launch_config_sha256": sha256_file(Path(str(request["launch_config"]))), "selected_loader_workers": selected_workers, "selected_workers_sha256": selected_workers_sha256, "runtime_resume": bool(resume_transfers), "transfers": transfers}


def runtime_mixture_hydrate(*, instance: Mapping[str, object], request: Mapping[str, object], runner: Runner) -> dict[str, object]:
    identity = _runtime_hydration_identity(instance, request)
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
    if action == "runtime-bootstrap-stage":
        return _runtime_abort_on_failure(instance=instance, request=request, runner=runner, operation=lambda: runtime_mixture_bootstrap_stage(instance=instance, request=request, runner=runner))
    if action == "runtime-warmup-stage":
        return _runtime_abort_on_failure(instance=instance, request=request, runner=runner, operation=lambda: runtime_mixture_warmup_stage(instance=instance, request=request, runner=runner))
    if action == "runtime-stage":
        return _runtime_abort_on_failure(instance=instance, request=request, runner=runner, operation=lambda: runtime_mixture_stage(instance=instance, request=request, runner=runner))
    if action == "runtime-hydrate":
        return _runtime_abort_on_failure(instance=instance, request=request, runner=runner, operation=lambda: runtime_mixture_hydrate(instance=instance, request=request, runner=runner))
    if action == "runtime-train":
        return _runtime_abort_on_failure(instance=instance, request=request, runner=runner, operation=lambda: runtime_mixture_train(instance=instance, request=request, runner=runner))
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


def main_for_test(
    argv: list[str], *, runner: Runner = _run,
    transport_factory: Callable[..., HubTransport] = HuggingFaceHubTransport,
) -> dict[str, object]:
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("derive-corrective-receipt", "materialize", "prepare", "capture-offers", "capture-runtime-pilot-offer", "bootstrap-canary", "promote", "replacement-resume", "rent", "runtime-pilot-rent", "runtime-gpu-warmup-rent", "runtime-gpu-rent-recover", "stage", "runtime-pilot-plan", "runtime-bootstrap-stage", "runtime-warmup-stage", "runtime-stage", "runtime-hydrate", "runtime-pilot-run", "runtime-gpu-warmup", "runtime-train", "runtime-checkpoint-publish", "runtime-checkpoint-complete", "runtime-checkpoint-interrupted", "runtime-checkpoint-replacement-resume", "runtime-checkpoint-dispose", "tune", "train", "status", "resume", "destroy", "runtime-pilot-destroy")); parser.add_argument("--request", required=True); parser.add_argument("--execute", action="store_true"); parser.add_argument("--token-file")
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
    if args.action == "runtime-gpu-rent-recover": return recover_runtime_gpu_rent(request=request, runner=runner)
    if args.action == "capture-runtime-pilot-offer": return capture_runtime_pilot_offer(runner=runner)
    if args.action == "bootstrap-canary": return bootstrap_canary(evidence=request, runner=runner)
    if args.action == "promote": return {"paid_action": True, "action": "promote", "instance": promote_canary(capability_receipt=request, runner=runner)}
    if args.action == "replacement-resume":
        terminal = request.get("provider_interruption_terminal")
        capability = request.get("capability_receipt")
        descriptor_output = request.get("resume_descriptor_output")
        if not isinstance(terminal, Mapping) or not isinstance(capability, Mapping) or not isinstance(descriptor_output, str):
            raise ValueError("replacement-resume requires interruption terminal, replacement capability receipt, and descriptor output")
        return {"paid_action": True, "action": "replacement-resume", **replacement_resume_descriptor(terminal=terminal, capability_receipt=capability, runner=runner, transport=transport_factory(timeout_seconds=30.0), token=_read_private_token(args.token_file), descriptor_output=Path(descriptor_output))}
    if args.action == "rent": return rent(evidence=request, runner=runner)
    if args.action == "runtime-pilot-rent": return rent_runtime_cpu_pilot(evidence=request, runner=runner)
    if args.action == "runtime-gpu-warmup-rent": return rent_runtime_gpu_warmup(evidence=request, runner=runner)
    if args.action == "destroy":
        return destroy(
            instance_id=request.get("instance_id"),  # type: ignore[arg-type]
            training_receipt=request,
            runner=runner,
            transport=transport_factory(timeout_seconds=30.0),
            token=_read_private_token(args.token_file),
        )
    if args.action == "runtime-pilot-destroy":
        lifecycle_path = request.get("lifecycle_receipt")
        if type(lifecycle_path) is not str or type(request.get("instance_id")) is not int:
            raise ValueError("runtime-pilot-destroy requires instance_id and lifecycle_receipt path")
        return destroy_runtime_cpu_pilot(
            instance_id=request["instance_id"],
            lifecycle_receipt=_load_regular_json(Path(lifecycle_path), "runtime CPU pilot lifecycle receipt"),
            runner=runner,
        )
    instance = request.get("instance")
    if not isinstance(instance, Mapping): raise ValueError("remote action requires an instance receipt")
    if args.action == "runtime-pilot-run":
        return _runtime_abort_on_failure(instance=instance, request=request, runner=runner, operation=lambda: run_runtime_cpu_pilot(instance=instance, request=request, runner=runner))
    if args.action == "runtime-gpu-warmup":
        return _runtime_abort_on_failure(instance=instance, request=request, runner=runner, operation=lambda: run_runtime_gpu_warmup(instance=instance, request=request, runner=runner))
    if args.action == "runtime-checkpoint-complete":
        return _runtime_abort_on_failure(instance=instance, request=request, runner=runner, operation=lambda: {"paid_action": True, "action": args.action, "terminal": _runtime_checkpoint_terminal_output(request, runtime_checkpoint_terminal(instance=instance, request=request)["terminal"])})
    if args.action == "runtime-checkpoint-interrupted":
        token = _read_private_token(args.token_file)
        hub = _RuntimeCheckpointHub(transport=transport_factory(timeout_seconds=30.0), token=token)
        return runtime_anchor_interruption_terminal(instance=instance, request=request, runner=runner, hub=hub)
    token = _read_private_token(args.token_file)
    hub = _RuntimeCheckpointHub(transport=transport_factory(timeout_seconds=30.0), token=token)
    if args.action == "runtime-checkpoint-publish":
        return _runtime_abort_on_failure(instance=instance, request=request, runner=runner, operation=lambda: publish_runtime_checkpoint(instance=instance, request=request, publisher=_runtime_checkpoint_publisher(request=request, token=token), hub=hub))
    if args.action == "runtime-checkpoint-replacement-resume":
        terminal_path, destination = request.get("terminal_receipt"), request.get("resume_destination")
        if type(terminal_path) is not str or type(destination) is not str or not Path(destination).is_absolute():
            raise ValueError("runtime checkpoint replacement resume requires terminal receipt and absolute destination")
        return _runtime_abort_on_failure(instance=instance, request=request, runner=runner, operation=lambda: resume_runtime_checkpoint(replacement=instance, request=request, terminal=_load_regular_json(Path(terminal_path), "runtime checkpoint terminal"), hub=hub, destination=Path(destination)))
    if args.action == "runtime-checkpoint-dispose":
        terminal_path = request.get("terminal_receipt")
        if type(terminal_path) is not str:
            raise ValueError("runtime checkpoint disposal requires terminal receipt")
        return destroy_runtime_checkpoint_completion(
            instance=instance, request=request,
            terminal=_load_regular_json(Path(terminal_path), "runtime checkpoint terminal"),
            hub=hub, runner=runner,
        )
    return remote_action(action=args.action, instance=instance, request=request, runner=runner)


if __name__ == "__main__": print(json.dumps(main_for_test(__import__("sys").argv[1:]), sort_keys=True))
