"""Explicit Vast authority for one corrective four-worker wave.

This module contains no implicit provider action.  Its caller must explicitly
capture offer evidence, call ``launch_wave`` after an instance exists, and pass
an externally captured disposal receipt before a destructive invocation.
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
import sys
import tarfile
import shutil
import tempfile
import time
import shlex
import uuid
from typing import Callable, Mapping, Sequence


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# GHCR is the immutable mirror of the verified Docker Hub image. Both manifests
# have config digest d36b7a84... and the same ordered 35 layer digest/size pairs;
# the registry-specific manifest serialization alone changes the top digest.
APPROVED_IMAGE_REPOSITORY = "ghcr.io/ryanjin333/lehome-rollout"
APPROVED_IMAGE_DIGEST = "sha256:25870f001eb0ab356222dbfd15352c42666f566adb41732bcdfd7a12d104f50d"
APPROVED_GROOT_ROOT = "/opt/isaac-groot"
APPROVED_GROOT_REVISION = "23ace64f17aa5015259b8609d371eb61a357c776"
# The image's historic /opt/gr00t-runtime/bin/python is a broken build-time
# symlink.  Recreate this immutable wrapper at the receipt-bound path instead.
APPROVED_GROOT_PYTHON = "/opt/gr00t-runtime/bin/python-validated-wrapper"
APPROVED_GROOT_NATIVE_PYTHON = "/opt/python/cpython-3.10.18-linux-x86_64-gnu/bin/python3.10"
APPROVED_GROOT_NATIVE_PYTHON_SHA256 = "ee22b22d759f77275c82503976968bc3193f577a4a039d0540bdb95e1b54bf1e"
APPROVED_GROOT_PYTHON_SHA256 = "760c0ad783861ad329821442e0d1385bd915d1bdaef87646c331bbf035d3c389"
APPROVED_ASSET_REVISION = "bea65fd960ad5a1bb3bd3fa77164b28001c08ef9"
APPROVED_HF_ENDPOINTS = frozenset({"https://hf-mirror.com", "https://huggingface.co"})
APPROVED_QWEN_REPOSITORY = "Qwen/Qwen3-VL-2B-Instruct"
APPROVED_QWEN_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
APPROVED_QWEN_ROOT = "/cache/models/nvidia/Cosmos-Reason2-2B"
APPROVED_QWEN_FILES = {
    "model.safetensors": "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
    "config.json": "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "tokenizer.json": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    "tokenizer_config.json": "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    "chat_template.json": "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4",
}
APPROVED_CONTROLLER_WIRE_WHEELS = (
    (
        "msgpack-1.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "https://files.pythonhosted.org/packages/a8/a1/ad7b84b91ab5a324e707f4c9761633e357820b011a01e34ce658c1dda7cc/msgpack-1.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "5e1da8f11a3dd397f0a32c76165cf0c4eb95b31013a94f6ecc0b280c05c91b59",
    ),
    (
        "pyzmq-27.0.1-cp311-cp311-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl",
        "https://files.pythonhosted.org/packages/6c/29/0652a39d4e876e0d61379047ecf7752685414ad2e253434348246f7a2a39/pyzmq-27.0.1-cp311-cp311-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl",
        "c512824360ea7490390566ce00bee880e19b526b312b25cc0bc30a0fe95cb67f",
    ),
)
# A freshly cloned Git repository carries these comments in .git/info/exclude.
# Do not inherit arbitrary local exclusions: they could conceal checkout changes
# from the trial's clean-tree gate.  The one additional pattern is deliberately
# anchored to the compatibility symlink below (not its parent directory).
_GIT_INFO_EXCLUDE_DEFAULT = (
    "# git ls-files --others --exclude-from=.git/info/exclude\n"
    "# Lines that start with '#' are comments.\n"
    "# For a project mostly in C, the following would be a good set of\n"
    "# exclude patterns (uncomment them if you want to use them):\n"
    "# *.[oa]\n"
    "# *~\n"
)
_QWEN_CHECKOUT_LINK = "nvidia/Cosmos-Reason2-2B"
# Vast's offer filter takes memory in GiB, whereas the raw offer/readback
# payload reports ``cpu_ram`` in MiB.  Keep those units explicit at the edge.
OFFER_QUERY = "gpu_name=RTX_3090 num_gpus=4 reliability>=0.95 cpu_cores_effective>=64 cpu_ram>=128 disk_space>=300 duration>=1"
VAST_SSH_IDENTITY = os.environ.get("LEHOME_VAST_SSH_IDENTITY", str(Path.home() / ".ssh" / "vast_quest"))


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _remote_hf_endpoint() -> str:
    """Use the China-reachable mirror by default, never accept an arbitrary host."""
    endpoint = os.environ.get("LEHOME_FLYWHEEL_HF_ENDPOINT", "https://hf-mirror.com")
    if endpoint not in APPROVED_HF_ENDPOINTS:
        raise ValueError("corrective HF endpoint is not approved")
    return endpoint


def _hf_download(command: str) -> str:
    return "HF_ENDPOINT=" + shlex.quote(_remote_hf_endpoint()) + " " + command


def _controller_pythonpath(checkout: str, *, wire_target: str | None = None) -> str:
    """Prefix reviewed controller code without replacing image-native Isaac paths."""
    prefix = (wire_target + ":") if wire_target is not None else ""
    return "PYTHONPATH=" + shlex.quote(prefix + checkout + "/source/lehome:" + checkout) + "${PYTHONPATH:+:$PYTHONPATH}"


def _controller_import_preflight(checkout: str, *, wire_target: str | None = None) -> str:
    # isaaclab_tasks imports Warp before AppLauncher has initialized bundled
    # simulator dependencies, so only inspect module discoverability here.
    return _controller_pythonpath(checkout, wire_target=wire_target) + " /opt/lehome-challenge/.venv/bin/python -c " + shlex.quote("import importlib.util; assert importlib.util.find_spec('isaaclab'); assert importlib.util.find_spec('lehome')")


def _controller_wire_setup(remote_dir: str, checkout: str) -> list[str]:
    """Install the exact cp311 wire clients outside the reviewed checkout/image."""
    target = remote_dir + "/controller-wire"
    wheelhouse = remote_dir + "/controller-wheels"
    wheel_paths = [wheelhouse + "/" + name for name, _, _ in APPROVED_CONTROLLER_WIRE_WHEELS]
    commands = ["mkdir -p " + shlex.quote(target) + " " + shlex.quote(wheelhouse)]
    for name, url, digest in APPROVED_CONTROLLER_WIRE_WHEELS:
        wheel = wheelhouse + "/" + name
        commands.extend((
            "curl --fail --location --proto '=https' --tlsv1.2 " + shlex.quote(url) + " --output " + shlex.quote(wheel),
            "test \"$(sha256sum " + shlex.quote(wheel) + " | cut -d' ' -f1)\" = " + shlex.quote(digest),
        ))
    commands.extend((
        "/opt/lehome-challenge/.venv/bin/python -c " + shlex.quote("from pathlib import Path; import zipfile; target=Path('" + target + "'); [zipfile.ZipFile(wheel).extractall(target) for wheel in map(Path, " + repr(wheel_paths) + ")]"),
        _controller_pythonpath(checkout, wire_target=target) + " /opt/lehome-challenge/.venv/bin/python -c " + shlex.quote('import msgpack, zmq; assert msgpack.__version__ == "1.1.0"; assert zmq.__version__ == "27.0.1"; assert msgpack.__file__.startswith("' + target + '"); assert zmq.__file__.startswith("' + target + '")'),
    ))
    return commands


def _approved_image_identity(baseline: Mapping[str, object], *, context: str) -> str:
    """Return the receipt-bound OCI digest accepted by this rollout image."""
    image_identity = baseline.get("image_identity")
    if image_identity != APPROVED_IMAGE_DIGEST:
        raise ValueError(f"{context} baseline image identity is not the approved digest")
    return str(image_identity)


def _image_identity_preflight(image_identity: str) -> str:
    """Check the value supplied to workers without trusting inherited SSH env."""
    return (
        "LEHOME_FLYWHEEL_IMAGE_IDENTITY=" + shlex.quote(image_identity)
        + " /bin/sh -c "
        + shlex.quote(
            "test \"${LEHOME_FLYWHEEL_IMAGE_IDENTITY-}\" = "
            + shlex.quote(image_identity)
            + " || { echo 'image identity preflight failed' >&2; exit 1; }"
        )
    )


def _qwen_base_setup(checkout: str) -> list[str]:
    """Hydrate and pin the open base expected by step12000's absolute path."""
    download = _hf_download(
        "/opt/lehome-challenge/.venv/bin/hf download " + APPROVED_QWEN_REPOSITORY
        + " --revision " + shlex.quote(APPROVED_QWEN_REVISION)
        + " --include model.safetensors config.json preprocessor_config.json tokenizer.json tokenizer_config.json chat_template.json --local-dir "
        + shlex.quote(APPROVED_QWEN_ROOT)
    )
    checks = [
        "test -f " + shlex.quote(APPROVED_QWEN_ROOT + "/" + name)
        + " && test \"$(sha256sum " + shlex.quote(APPROVED_QWEN_ROOT + "/" + name) + " | cut -d' ' -f1)\" = " + shlex.quote(digest)
        for name, digest in APPROVED_QWEN_FILES.items()
    ]
    checkout_link = checkout + "/" + _QWEN_CHECKOUT_LINK
    # Keep the trial's required `cwd == checkout` clean.  This validates all
    # visible untracked content before installing the only permitted exclusion.
    # A prior interrupted setup may already have created the *exact* link; that
    # is the sole recoverable dirty state, and all other untracked paths fail.
    exclude_program = "\n".join(
        [
            "from pathlib import Path",
            "import subprocess",
            f"checkout = Path({checkout!r})",
            f"target = checkout / {_QWEN_CHECKOUT_LINK!r}",
            "exclude = checkout / '.git' / 'info' / 'exclude'",
            f"default = {_GIT_INFO_EXCLUDE_DEFAULT!r}",
            "expected = default + '/nvidia/Cosmos-Reason2-2B\\n'",
            "if not (checkout / '.git').is_dir(): raise SystemExit('missing git checkout')",
            "if exclude.parent.is_symlink() or exclude.is_symlink() or (exclude.exists() and not exclude.is_file()): raise SystemExit('unsafe git exclude')",
            "if exclude.exists() and exclude.read_text(encoding='utf-8') not in ('', default, expected): raise SystemExit('unexpected git exclude content')",
            "status = subprocess.run(['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=all'], check=True, text=True, capture_output=True).stdout",
            "if status not in ('', '?? nvidia/Cosmos-Reason2-2B\\n'): raise SystemExit('checkout has unexpected changes')",
            "if (target.exists() or target.is_symlink()) and (not target.is_symlink() or target.resolve() != Path('/cache/models/nvidia/Cosmos-Reason2-2B')): raise SystemExit('unexpected compatibility link')",
            "if status and not target.is_symlink(): raise SystemExit('missing compatibility link')",
            "exclude.parent.mkdir(mode=0o700, exist_ok=True)",
            "exclude.write_text(expected, encoding='utf-8')",
        ]
    )
    link_setup = [
        "/opt/lehome-challenge/.venv/bin/python -c " + shlex.quote(exclude_program),
        "mkdir -p " + shlex.quote(checkout + "/nvidia"),
        "if [ -e " + shlex.quote(checkout_link) + " ] || [ -L " + shlex.quote(checkout_link) + " ]; then test -L " + shlex.quote(checkout_link) + "; else ln -s " + shlex.quote(APPROVED_QWEN_ROOT) + " " + shlex.quote(checkout_link) + "; fi",
        "test -L " + shlex.quote(checkout_link),
        "test \"$(readlink -f " + shlex.quote(checkout_link) + ")\" = " + shlex.quote(APPROVED_QWEN_ROOT),
        "test -z \"$(git -C " + shlex.quote(checkout) + " status --porcelain --untracked-files=all)\"",
    ]
    return [download, *checks, *link_setup]


def _groot_wrapper_setup() -> list[str]:
    """Materialize the exact receipt-bound wrapper around the native interpreter."""
    wrapper = "#!/bin/sh\nPYTHONPATH=/opt/gr00t-runtime/lib/python3.10/site-packages:/opt/isaac-groot${PYTHONPATH:+:$PYTHONPATH}\nexport PYTHONPATH\nexec /opt/python/cpython-3.10.18-linux-x86_64-gnu/bin/python3.10 \"$@\"\n"
    return [
        "test -x " + shlex.quote(APPROVED_GROOT_NATIVE_PYTHON),
        "test \"$(sha256sum " + shlex.quote(APPROVED_GROOT_NATIVE_PYTHON) + " | cut -d' ' -f1)\" = " + shlex.quote(APPROVED_GROOT_NATIVE_PYTHON_SHA256),
        "mkdir -p " + shlex.quote(str(Path(APPROVED_GROOT_PYTHON).parent)),
        "printf '%s' " + shlex.quote(wrapper) + " > " + shlex.quote(APPROVED_GROOT_PYTHON),
        "chmod 755 " + shlex.quote(APPROVED_GROOT_PYTHON),
        "test \"$(sha256sum " + shlex.quote(APPROVED_GROOT_PYTHON) + " | cut -d' ' -f1)\" = " + shlex.quote(APPROVED_GROOT_PYTHON_SHA256),
        shlex.quote(APPROVED_GROOT_PYTHON) + " -c " + shlex.quote("import gr00t; assert gr00t.__file__.startswith('/opt/isaac-groot')"),
    ]


def _write_new(path: Path, value: Mapping[str, object]) -> dict[str, object]:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_text(encoding="utf-8") != encoded:
            raise ValueError("refusing to overwrite differing lifecycle receipt")
    else:
        path.write_text(encoded, encoding="utf-8")
    return dict(value)


_PROVIDER_SNAPSHOT_FIELDS = {
    "offer": ("id", "is_bid", "gpu_name", "num_gpus", "cpu_cores_effective", "cpu_ram", "driver_version", "dph_total"),
    "instance": ("id", "actual_status", "is_bid", "gpu_name", "num_gpus", "cpu_cores_effective", "cpu_ram", "driver_version", "dph_total", "ssh_host", "ssh_port"),
    "volume": ("id", "storage_total_cost"),
}


def _provider_snapshot_row(row: Mapping[str, object], kind: str) -> dict[str, object]:
    """Persist only the provider facts used by corrective evidence validation."""
    return {field: row[field] for field in _PROVIDER_SNAPSHOT_FIELDS[kind] if field in row}


def capture_offer_evidence(*, offers: Sequence[Mapping[str, object]], instances: Sequence[Mapping[str, object]], output: Path, now_unix: int, ttl_seconds: int, preferred_offer_id: int | None = None, volumes: Sequence[Mapping[str, object]] = (), retained_instance_id: int | None = None, prior_provider_evidence: Mapping[str, object] | None = None, prior_instance_receipt: Mapping[str, object] | None = None) -> dict[str, object]:
    """Bind a live external provider snapshot, without creating an instance."""
    if type(now_unix) is not int or type(ttl_seconds) is not int or ttl_seconds <= 0:
        raise ValueError("provider evidence time window is invalid")
    def hourly(item: Mapping[str, object]) -> float:
        # Vast's dph_total already includes the instance disk charge.  Only
        # detached retained volumes need to be added separately below.
        return float(item.get("dph_total", math.inf))
    existing_spend = sum(hourly(item) for item in instances) + sum(float(item.get("storage_total_cost", 0)) for item in volumes)
    if retained_instance_id is not None:
        if type(retained_instance_id) is not int or retained_instance_id <= 0 or prior_provider_evidence is None or prior_instance_receipt is None:
            raise ValueError("retained instance evidence requires a prior evidence and instance receipt")
        retained = next((item for item in instances if item.get("id") == retained_instance_id), None)
        if retained is None:
            raise ValueError("retained instance is not found in live provider rows")
        prior_hash = _canonical_hash(prior_provider_evidence)
        offer_id = prior_provider_evidence.get("offer_id")
        prior_price = prior_provider_evidence.get("instance_hourly_cost_usd")
        if prior_provider_evidence.get("kind") != "external_provider_offer_evidence" or type(offer_id) is not int or not isinstance(prior_price, (int, float)) or not math.isfinite(float(prior_price)) or float(prior_price) <= 0:
            raise ValueError("retained instance prior evidence lacks a stable offer identity")
        if prior_instance_receipt.get("kind") != "corrective_vast_instance" or prior_instance_receipt.get("instance_id") != retained_instance_id or prior_instance_receipt.get("provider_evidence_sha256") != prior_hash:
            raise ValueError("retained instance receipt is not bound to prior provider evidence")
        compatible = (retained.get("actual_status") == "running" and retained.get("is_bid") is False and retained.get("gpu_name") == "RTX 3090" and retained.get("num_gpus") == 4 and _effective_cores(retained) >= 64 and _memory_mib(retained) >= 128_000 and _is_approved_r580(retained.get("driver_version")) and hourly(retained) > 0 and math.isclose(hourly(retained), float(prior_price), rel_tol=0.0, abs_tol=1e-9) and retained.get("ssh_host") == prior_instance_receipt.get("host") and retained.get("ssh_port", 22) == prior_instance_receipt.get("port"))
        if not compatible:
            raise ValueError("retained instance live row is incompatible with its prior receipt")
        if not math.isfinite(existing_spend) or existing_spend > 2.0:
            raise ValueError("live provider account spend exceeds shared $2/hr cap")
        offer = {"id": offer_id, "dph_total": float(prior_price)}
        account_total = existing_spend
    else:
        matching = [offer for offer in offers if offer.get("is_bid") is False and offer.get("gpu_name") == "RTX 3090" and offer.get("num_gpus") == 4 and type(offer.get("id")) is int and type(offer.get("dph_total")) in (int, float) and _effective_cores(offer) >= 64 and _memory_mib(offer) >= 128_000 and _is_approved_r580(offer.get("driver_version"))]
        if not matching:
            raise ValueError("live provider offers contain no on-demand 4xRTX3090 choice")
        acceptable = [item for item in matching if math.isfinite(existing_spend + hourly(item)) and hourly(item) > 0 and existing_spend + hourly(item) <= 2.0]
        if not acceptable:
            raise ValueError("live provider account spend exceeds shared $2/hr cap")
        acceptable.sort(key=lambda offer: (float(offer["dph_total"]), int(offer["id"])))
        offer = next((item for item in acceptable if item["id"] == preferred_offer_id), acceptable[0])
        account_total = existing_spend + hourly(offer)
    snapshot = {
        "offers": [_provider_snapshot_row(row, "offer") for row in offers],
        "instances": [_provider_snapshot_row(row, "instance") for row in instances],
        "volumes": [_provider_snapshot_row(row, "volume") for row in volumes],
        "captured_at_unix": now_unix,
    }
    source_path = output.with_name(output.stem + "-source.json")
    _write_new(source_path, snapshot)
    evidence = {"schema_version": 1, "kind": "external_provider_offer_evidence", "evidence_id": _canonical_hash(snapshot), "queried_at_unix": now_unix, "expires_at_unix": now_unix + ttl_seconds, "source_snapshot_path": source_path.name, "source_snapshot_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "source_response_sha256": _canonical_hash(snapshot), "rental_kind": "on-demand", "instance_hourly_cost_usd": hourly(offer), "account_hourly_total_usd": account_total, "offer_id": offer["id"], "gpu_name": "RTX 3090", "num_gpus": 4}
    return _write_new(output, evidence)


def _read_manifest(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file(): raise ValueError("corrective wave manifest is unavailable")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != "corrective_rft_wave" or not isinstance(value.get("attempts"), list) or len(value["attempts"]) != 4:
        raise ValueError("corrective wave manifest is invalid")
    baseline = value.get("baseline")
    if not isinstance(baseline, dict) or not isinstance(baseline.get("rollout_image"), str):
        raise ValueError("corrective wave lacks a baseline-bound rollout image")
    repository, separator, digest = baseline["rollout_image"].partition("@")
    if repository != APPROVED_IMAGE_REPOSITORY or separator != "@" or digest != APPROVED_IMAGE_DIGEST:
        raise ValueError("corrective baseline rollout image must be pinned to the approved repository digest")
    _approved_image_identity(baseline, context="corrective wave")
    provider = value.get("provider")
    if not isinstance(provider, dict) or provider.get("rental_kind") != "on-demand" or provider.get("gpu_name") != "RTX 3090" or provider.get("num_gpus") != 4 or float(provider.get("account_hourly_total_usd", math.inf)) > 2.0:
        raise ValueError("corrective wave provider facts violate 4x3090 shared cap")
    return value


def _is_approved_r580(value: object) -> bool:
    """Only accept a complete R580 build in the audited compatibility window."""
    if not isinstance(value, str):
        return False
    matched = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?", value)
    if matched is None:
        return False
    version = tuple(int(item or 0) for item in matched.groups())
    return (580, 65, 6) <= version < (590, 0, 0)


def _number(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _effective_cores(value: Mapping[str, object]) -> float:
    """Vast raw offer/readback field; missing capacity is inadmissible."""
    return _number(value.get("cpu_cores_effective"))


def _memory_mib(value: Mapping[str, object]) -> float:
    """Vast raw offer/readback memory is MiB; missing capacity is inadmissible."""
    return _number(value.get("cpu_ram"))


def _run_raw(runner: Callable[[tuple[str, ...]], object], command: tuple[str, ...]) -> object:
    """Use raw JSON only; never relay command output into user-facing logs."""
    result = runner(command)
    text = result if isinstance(result, str) else getattr(result, "stdout", "")
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("vastai raw response is invalid") from error


def _wait_for_running_instance(instance_id: int, *, runner: Callable[[tuple[str, ...]], object], polls: int = 360, sleep: Callable[[float], None] = time.sleep) -> dict[str, object]:
    """Allow the approved large image up to 30 minutes to become SSH-ready."""
    for _ in range(polls):
        readback = _run_raw(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
        if isinstance(readback, dict) and readback.get("id") == instance_id and readback.get("actual_status") == "running" and isinstance(readback.get("ssh_host"), str) and readback.get("ssh_host") and type(readback.get("ssh_port", 22)) is int:
            return readback
        sleep(5.0)
    raise ValueError("vastai instance did not reach running SSH-ready state")


def rent_wave(manifest_path: Path, *, lifecycle_root: Path, runner: Callable[[tuple[str, ...]], object], now_unix: int, image: str | None = None, sleep: Callable[[float], None] = time.sleep) -> dict[str, object]:
    """Explicit rent operation with live offer/instance query and identity readback."""
    manifest = _read_manifest(manifest_path)
    baseline_image = manifest["baseline"]["rollout_image"]
    if image is not None and image != baseline_image:
        raise ValueError("rent image must exactly match the baseline-bound rollout image")
    image = str(baseline_image)
    instances = _run_raw(runner, ("vastai", "--raw", "show", "instances"))
    volumes = _run_raw(runner, ("vastai", "--raw", "show", "volumes"))
    offers = _run_raw(runner, ("vastai", "--raw", "search", "offers", OFFER_QUERY, "--on-demand", "--storage", "300"))
    if not isinstance(instances, list) or not isinstance(offers, list) or not isinstance(volumes, list):
        raise ValueError("vastai raw offer or instance response is invalid")
    evidence = capture_offer_evidence(offers=offers, instances=instances, volumes=volumes, output=lifecycle_root / f"wave-{manifest['wave_index']:06d}-provider-renewal-{now_unix}.json", now_unix=now_unix, ttl_seconds=300, preferred_offer_id=manifest["provider"]["offer_id"])
    if evidence["offer_id"] != manifest["provider"]["offer_id"]:
        raise ValueError("live offer no longer matches corrective wave")
    created = _run_raw(runner, ("vastai", "--raw", "create", "instance", str(evidence["offer_id"]), "--image", image, "--env", f"-e LEHOME_FLYWHEEL_IMAGE_IDENTITY={manifest['baseline']['image_identity']}", "--disk", "300", "--ssh", "--direct", "--cancel-unavail"))
    instance_id = created.get("new_contract") if isinstance(created, dict) else None
    if type(instance_id) is not int or instance_id <= 0:
        raise ValueError("vastai rent response lacks instance ID")
    try:
        readback = _wait_for_running_instance(instance_id, runner=runner, sleep=sleep)
    except BaseException:
        _cleanup_new_instance(instance_id, runner)
        raise
    if not isinstance(readback, dict) or readback.get("id") != instance_id or readback.get("is_bid") is not False or readback.get("gpu_name") != "RTX 3090" or readback.get("num_gpus") != 4 or _effective_cores(readback) < 64 or _memory_mib(readback) < 128_000 or not _is_approved_r580(readback.get("driver_version")) or float(readback.get("dph_total", math.inf)) != evidence["instance_hourly_cost_usd"]:
        _cleanup_new_instance(instance_id, runner)
        raise ValueError("vastai instance readback does not match approved offer")
    receipt = {"schema_version": 1, "kind": "corrective_vast_instance", "instance_id": instance_id, "host": readback.get("ssh_host"), "port": readback.get("ssh_port", 22), "provider_response_sha256": _canonical_hash(readback), "provider_evidence_sha256": _canonical_hash(evidence), "wave_index": manifest["wave_index"]}
    if not isinstance(receipt["host"], str) or not receipt["host"] or type(receipt["port"]) is not int:
        raise ValueError("vastai instance readback lacks SSH endpoint")
    return _write_new(lifecycle_root / f"wave-{manifest['wave_index']:06d}-instance.json", receipt)


def renew_retained_lease(manifest_path: Path, prior_receipt_path: Path, *, lifecycle_root: Path, runner: Callable[[tuple[str, ...]], object]) -> dict[str, object]:
    """Bind a later wave to the same still-running collector with fresh evidence."""
    manifest = _read_manifest(manifest_path)
    prior = _read_json_object(prior_receipt_path, "retained lease receipt")
    evidence = manifest.get("provider_evidence")
    instance_id = prior.get("instance_id")
    if prior.get("kind") != "corrective_vast_instance" or type(instance_id) is not int or instance_id <= 0 or not isinstance(evidence, dict):
        raise ValueError("retained lease receipt or provider evidence is invalid")
    readback = _run_raw(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
    if (
        not isinstance(readback, dict) or readback.get("id") != instance_id
        or readback.get("actual_status") != "running" or readback.get("is_bid") is not False
        or readback.get("gpu_name") != "RTX 3090" or readback.get("num_gpus") != 4
        or _effective_cores(readback) < 64 or _memory_mib(readback) < 128_000
        or not _is_approved_r580(readback.get("driver_version"))
        or readback.get("ssh_host") != prior.get("host") or readback.get("ssh_port", 22) != prior.get("port")
    ):
        raise ValueError("retained Vast lease live readback is incompatible")
    receipt = {
        "schema_version": 1, "kind": "corrective_vast_instance", "instance_id": instance_id,
        "host": prior["host"], "port": prior["port"],
        "provider_response_sha256": _canonical_hash(readback),
        "provider_evidence_sha256": _canonical_hash(evidence),
        "wave_index": manifest["wave_index"],
        "lease_wave_index": prior.get("lease_wave_index", prior.get("wave_index")),
        "prior_instance_receipt_sha256": hashlib.sha256(prior_receipt_path.read_bytes()).hexdigest(),
    }
    return _write_new(lifecycle_root / f"wave-{manifest['wave_index']:06d}-instance.json", receipt)


def adopt_retained_lease(manifest_path: Path, prior_receipt_path: Path, prior_provider_evidence_path: Path, fresh_provider_evidence_path: Path, *, lifecycle_root: Path, runner: Callable[[tuple[str, ...]], object]) -> dict[str, object]:
    """Bind a fresh campaign's wave zero to one independently proven lease."""
    manifest = _read_manifest(manifest_path)
    if manifest["wave_index"] != 0:
        raise ValueError("retained lease adoption is only for a fresh wave zero")
    prior = _read_json_object(prior_receipt_path, "prior retained instance receipt")
    prior_evidence = _read_json_object(prior_provider_evidence_path, "prior provider evidence")
    fresh_evidence = _read_json_object(fresh_provider_evidence_path, "fresh retained provider evidence")
    if manifest.get("provider_evidence") != fresh_evidence:
        raise ValueError("fresh campaign manifest does not bind the supplied provider evidence")
    provider = manifest["provider"]
    if any(provider.get(key) != fresh_evidence.get(key) for key in ("rental_kind", "instance_hourly_cost_usd", "account_hourly_total_usd", "offer_id", "gpu_name", "num_gpus")):
        raise ValueError("fresh campaign provider facts do not match retained lease evidence")
    instance_id = prior.get("instance_id")
    if prior.get("kind") != "corrective_vast_instance" or type(instance_id) is not int or instance_id <= 0 or prior.get("provider_evidence_sha256") != _canonical_hash(prior_evidence):
        raise ValueError("prior retained lease is not bound to provider evidence")
    readback = _run_raw(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
    instances = _run_raw(runner, ("vastai", "--raw", "show", "instances"))
    volumes = _run_raw(runner, ("vastai", "--raw", "show", "volumes"))
    if not isinstance(readback, dict) or not isinstance(instances, list) or not isinstance(volumes, list):
        raise ValueError("retained Vast lease readback is invalid")
    captured_at = fresh_evidence.get("queried_at_unix")
    expires_at = fresh_evidence.get("expires_at_unix")
    if type(captured_at) is not int or type(expires_at) is not int or expires_at < time.time_ns() // 1_000_000_000:
        raise ValueError("fresh retained provider evidence is expired")
    def hourly(item: Mapping[str, object]) -> float:
        return float(item.get("dph_total", math.inf))
    live_total = sum(hourly(item) for item in instances) + sum(float(item.get("storage_total_cost", 0)) for item in volumes)
    compatible = (readback.get("actual_status") == "running" and readback.get("is_bid") is False and readback.get("gpu_name") == "RTX 3090" and readback.get("num_gpus") == 4 and _effective_cores(readback) >= 64 and _memory_mib(readback) >= 128_000 and _is_approved_r580(readback.get("driver_version")) and hourly(readback) > 0 and math.isclose(hourly(readback), float(prior_evidence.get("instance_hourly_cost_usd", math.inf)), rel_tol=0.0, abs_tol=1e-9) and readback.get("ssh_host") == prior.get("host") and readback.get("ssh_port", 22) == prior.get("port"))
    if not compatible or not math.isfinite(live_total) or live_total > 2.0 or fresh_evidence.get("account_hourly_total_usd") != live_total:
        raise ValueError("fresh retained provider evidence no longer matches live account facts")
    receipt = {
        "schema_version": 1, "kind": "corrective_vast_instance", "instance_id": instance_id,
        "host": readback.get("ssh_host"), "port": readback.get("ssh_port", 22),
        "provider_response_sha256": _canonical_hash(readback),
        "provider_evidence_sha256": _canonical_hash(fresh_evidence), "wave_index": 0,
        "lease_wave_index": prior.get("lease_wave_index", prior.get("wave_index")),
        "prior_instance_receipt_sha256": hashlib.sha256(prior_receipt_path.read_bytes()).hexdigest(),
        "prior_provider_evidence_sha256": hashlib.sha256(prior_provider_evidence_path.read_bytes()).hexdigest(),
        "fresh_provider_evidence_sha256": hashlib.sha256(fresh_provider_evidence_path.read_bytes()).hexdigest(),
    }
    if not isinstance(receipt["host"], str) or not receipt["host"] or type(receipt["port"]) is not int:
        raise ValueError("retained Vast lease readback lacks SSH endpoint")
    return _write_new(lifecycle_root / "wave-000000-instance.json", receipt)


def _cleanup_new_instance(instance_id: int, runner: Callable[[tuple[str, ...]], object]) -> None:
    runner(("vastai", "destroy", "instance", str(instance_id), "--yes"))
    try:
        absent = _run_raw(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
    except (ValueError, RuntimeError):
        return
    if absent not in (None, {}, []):
        raise RuntimeError("newly-created Vast instance cleanup did not verify absence")


def remote_launch_wave(manifest_path: Path, instance_receipt: Mapping[str, object], *, lifecycle_root: Path, runner: Callable[[tuple[str, ...]], object], code_bundle: Path, token_file: Path) -> dict[str, object]:
    """Launch four workers from the same image-native Git-bundle boundary as canary."""
    lifecycle_root.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(manifest_path)
    lease_wave = instance_receipt.get("lease_wave_index", instance_receipt.get("wave_index"))
    if instance_receipt.get("kind") != "corrective_vast_instance" or type(lease_wave) is not int or lease_wave < 0 or lease_wave > manifest["wave_index"] or not isinstance(instance_receipt.get("host"), str) or type(instance_receipt.get("port")) is not int:
        raise ValueError("instance lifecycle receipt is not bound to corrective wave")
    if code_bundle.is_symlink() or not code_bundle.is_file() or token_file.is_symlink() or not token_file.is_file():
        raise ValueError("full wave requires a code Git bundle and securely provisioned token file")
    baseline = manifest["baseline"]
    if baseline.get("controller_python") != "/opt/lehome-challenge/.venv/bin/python" or baseline.get("groot_root") != APPROVED_GROOT_ROOT or baseline.get("groot_revision") != APPROVED_GROOT_REVISION or baseline.get("groot_python") != APPROVED_GROOT_PYTHON:
        raise ValueError("full wave baseline does not match proven image-native runtime")
    image_identity = _approved_image_identity(baseline, context="full wave")
    remote = f"root@{instance_receipt['host']}"; port = str(instance_receipt["port"])
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    remote_dir = f"/workspace/corrective/wave-{manifest['wave_index']:06d}-{manifest_hash[:12]}"
    checkout = f"{remote_dir}/code"; output_root = f"{remote_dir}/campaign"
    remote_bundle, remote_token = f"{remote_dir}/code.bundle", f"{remote_dir}/hf.token"
    digest = hashlib.sha256(code_bundle.read_bytes()).hexdigest()
    _require_completed(runner(("ssh", "-i", VAST_SSH_IDENTITY, "-o", "IdentitiesOnly=yes", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-p", port, remote, "mkdir", "-p", remote_dir)), "full-wave staging directory")
    _require_completed(runner(("scp", "-i", VAST_SSH_IDENTITY, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", port, str(code_bundle), f"{remote}:{remote_bundle}")), "full-wave code bundle staging")
    _require_completed(runner(("scp", "-i", VAST_SSH_IDENTITY, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", port, str(token_file), f"{remote}:{remote_token}")), "full-wave token staging")
    local_output_root = _manifest_output_root(manifest["attempts"])
    policy_revision, policy_digest = baseline.get("parent_checkpoint_revision"), baseline.get("parent_checkpoint_artifact_sha256")
    if not isinstance(policy_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", policy_revision) or not isinstance(policy_digest, str) or _SHA256.fullmatch(policy_digest) is None:
        raise ValueError("full wave lacks immutable private policy binding")
    policy_root = "/workspace/checkpoints/lehome-groot-n17-models-" + policy_revision
    policy_path = policy_root + "/policies/step-12000"
    mapping = {"policy_path": "image:" + policy_path, "policy_revision_file": "image:" + policy_root + "/revision.txt", "release_assets_root": "Assets/objects/Challenge_Garment/Release", "groot_root": "image:" + APPROVED_GROOT_ROOT, "groot_python": "image:" + APPROVED_GROOT_PYTHON, "controller_python": "image:/opt/lehome-challenge/.venv/bin/python", "output_root": "campaign", "trial_script": "scripts/run_groot_flywheel_trial.py"}
    campaign_root = manifest_path.parent.parent if manifest_path.parent.name == "waves" else manifest_path.parent
    terminal_attempt_ids = {
        str(item["attempt_id"])
        for item in manifest["attempts"]
        if (campaign_root / "raw" / str(item["attempt_id"]) / "SHA256SUMS.json").is_file()
    }
    remote_attempts = [
        {**attempt, "command": _remote_command(attempt["command"], baseline, mapping, checkout, output_root, local_output_root)}
        for attempt in manifest["attempts"] if str(attempt["attempt_id"]) not in terminal_attempt_ids
    ]
    if not remote_attempts:
        raise ValueError("corrective wave has no unverified attempts to launch")
    controller_pythonpath = checkout + "/source/lehome:" + checkout
    revision = str(baseline["code_revision"])
    checkout_setup = "if [ -e " + shlex.quote(checkout) + " ]; then test -d " + shlex.quote(checkout + "/.git") + " && test \"$(git -C " + shlex.quote(checkout) + " rev-parse HEAD)\" = " + shlex.quote(revision) + " && git -C " + shlex.quote(checkout) + " diff --quiet; else git clone --no-checkout " + shlex.quote(remote_bundle) + " " + shlex.quote(checkout) + " && git -C " + shlex.quote(checkout) + " checkout --detach " + shlex.quote(revision) + "; fi"
    wire_target = remote_dir + "/controller-wire"
    setup = ["set -eu", "test -x /opt/lehome-challenge/.venv/bin/python", _image_identity_preflight(image_identity), *_groot_wrapper_setup(), "test \"$(git -C " + shlex.quote(APPROVED_GROOT_ROOT) + " rev-parse HEAD)\" = " + shlex.quote(APPROVED_GROOT_REVISION), "chmod 600 " + shlex.quote(remote_token), "export HF_TOKEN=\"$(cat " + shlex.quote(remote_token) + ")\"", checkout_setup, "test \"$(git -C " + shlex.quote(checkout) + " rev-parse HEAD)\" = " + shlex.quote(revision), "git -C " + shlex.quote(checkout) + " diff --quiet", *_controller_wire_setup(remote_dir, checkout), _controller_import_preflight(checkout, wire_target=wire_target), _hf_download("/opt/lehome-challenge/.venv/bin/hf download ryanjin333/lehome-groot-n17-models --revision " + shlex.quote(policy_revision) + " --include 'policies/step-12000/*' --local-dir " + shlex.quote(policy_root)), "printf '%s\\n' " + shlex.quote(policy_revision) + " > " + shlex.quote(policy_root + "/revision.txt"), _controller_pythonpath(checkout, wire_target=wire_target) + " /opt/lehome-challenge/.venv/bin/python -c " + shlex.quote("from pathlib import Path; from scripts.run_groot_flywheel_trial import policy_artifact_sha256; assert policy_artifact_sha256(Path('" + policy_path + "')) == '" + policy_digest + "'")]
    setup.extend(_asset_checkout_setup(checkout))
    setup.extend(_qwen_base_setup(checkout))
    script_lines = ["set -eu", "trap 'rm -f " + shlex.quote(remote_token) + "; unset HF_TOKEN' EXIT", *setup[1:], f"mkdir -p {shlex.quote(output_root)}", "pids='' "]
    for attempt in remote_attempts:
        command = " ".join(shlex.quote(token) for token in attempt["command"])
        slot = int(attempt["worker_slot"])
        log = shlex.quote(f"{output_root}/worker-{slot}.log")
        status_file = shlex.quote(f"{output_root}/worker-{slot}.returncode")
        script_lines.append(f"( cd {shlex.quote(checkout)} && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 {_controller_pythonpath(checkout, wire_target=wire_target)} LEHOME_FLYWHEEL_WORKER_GPU={slot} LEHOME_FLYWHEEL_IMAGE_IDENTITY={shlex.quote(image_identity)} {command} >{log} 2>&1; rc=$?; printf '%s\\n' \"$rc\" >{status_file}; exit 0 ) & pids=\"$pids $!\"")
    script_lines.extend(("for pid in $pids; do wait \"$pid\" || true; done", f"python3 -c {shlex.quote(_terminal_writer_program(output_root, {'attempts': remote_attempts}))}", "exit 0"))
    result = runner(("ssh", "-i", VAST_SSH_IDENTITY, "-o", "IdentitiesOnly=yes", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-p", port, remote, "sh", "-lc", "\n".join(script_lines)))
    # Always sync the exact campaign output; a failed remote launch is evidence
    # for diagnosis, not a reason to destroy a paid instance automatically.
    sync_root = lifecycle_root / f"synced-wave-{manifest['wave_index']:06d}"
    sync_root.parent.mkdir(parents=True, exist_ok=True)
    sync_result = runner(("scp", "-r", "-i", VAST_SSH_IDENTITY, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", port, f"{remote}:{output_root}/.", str(sync_root)))
    completed = getattr(result, "returncode", 0)
    if getattr(sync_result, "returncode", 0) not in (0, None) or completed not in (0, None):
        _write_new(lifecycle_root / f"wave-{manifest['wave_index']:06d}-remote.json", {"schema_version": 1, "kind": "corrective_vast_remote_launch", "wave_index": manifest["wave_index"], "instance_id": instance_receipt["instance_id"], "status": "remote_transport_failure", "transport_returncode": completed, "manifest_sha256": manifest_hash, "bundle_sha256": digest})
        raise RuntimeError("remote launch transport failed; synchronized terminal receipt retained for diagnosis")
    receipt = _validate_remote_terminal(sync_root, {**manifest, "attempts": remote_attempts})
    for attempt in remote_attempts:
        _validate_canary_policy_receipt(sync_root / f"policy-server-receipt-{attempt['attempt_id']}.json", attempt, baseline, attempt["command"])
    return _write_new(lifecycle_root / f"wave-{manifest['wave_index']:06d}-remote.json", {"schema_version": 1, "kind": "corrective_vast_remote_launch", "wave_index": manifest["wave_index"], "instance_id": instance_receipt["instance_id"], "status": "remote_terminal", "manifest_sha256": manifest_hash, "bundle_sha256": digest, "worker_returncodes": receipt["worker_returncodes"], "remote_terminal_sha256": receipt["remote_terminal_sha256"]})


def remote_launch_canary(canary_manifest: Path, instance_receipt: Mapping[str, object], *, lifecycle_root: Path, runner: Callable[[tuple[str, ...]], object], bundle: Path, token_file: Path | None) -> dict[str, object]:
    """Launch exactly one episode after image-native runtime preflight.

    The token path is validated remotely and never interpolated into commands,
    logs, receipts, or manifests; an unavailable provisioned secret fails closed.
    """
    lifecycle_root.mkdir(parents=True, exist_ok=True)
    value = _read_json_object(canary_manifest, "canary manifest")
    if value.get("kind") != "corrective_rft_canary" or value.get("episode_count") != 1 or not isinstance(value.get("attempt"), dict):
        raise ValueError("canary manifest must bind exactly one episode")
    if (
        instance_receipt.get("kind") != "corrective_vast_instance"
        or instance_receipt.get("wave_index") != value.get("wave_index")
        or type(instance_receipt.get("instance_id")) is not int or int(instance_receipt["instance_id"]) <= 0
        or not isinstance(instance_receipt.get("host"), str) or not instance_receipt["host"]
        or type(instance_receipt.get("port")) is not int or not 1 <= int(instance_receipt["port"]) <= 65535
        or not isinstance(instance_receipt.get("provider_response_sha256"), str) or _SHA256.fullmatch(instance_receipt["provider_response_sha256"]) is None
        or not isinstance(instance_receipt.get("provider_evidence_sha256"), str) or _SHA256.fullmatch(instance_receipt["provider_evidence_sha256"]) is None
    ):
        raise ValueError("canary instance lifecycle receipt is not bound to corrective wave")
    if token_file is None or token_file.is_symlink() or not token_file.is_file():
        raise ValueError("canary requires a local securely provisioned token file")
    attempt = value["attempt"]
    if int(attempt.get("worker_slot", -1)) not in range(4):
        raise ValueError("canary worker slot is invalid")
    if bundle.is_symlink() or not bundle.is_file():
        raise ValueError("canary code git bundle is unavailable")
    baseline = value.get("baseline")
    if not isinstance(baseline, dict) or baseline.get("controller_python") != "/opt/lehome-challenge/.venv/bin/python" or baseline.get("groot_root") != APPROVED_GROOT_ROOT or baseline.get("groot_revision") != APPROVED_GROOT_REVISION or baseline.get("groot_python") != APPROVED_GROOT_PYTHON:
        raise ValueError("canary baseline does not match the proven image-native runtime")
    image_identity = _approved_image_identity(baseline, context="canary")
    policy_revision = baseline.get("parent_checkpoint_revision")
    policy_digest = baseline.get("parent_checkpoint_artifact_sha256")
    if not isinstance(policy_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", policy_revision) or not isinstance(policy_digest, str) or _SHA256.fullmatch(policy_digest) is None:
        raise ValueError("canary baseline lacks the immutable private policy revision and digest")
    remote = f"root@{instance_receipt['host']}"; port = str(instance_receipt["port"])
    manifest_hash = hashlib.sha256(canary_manifest.read_bytes()).hexdigest()
    remote_dir = f"/workspace/corrective/canary-{value['wave_index']:06d}-{manifest_hash[:12]}"
    remote_bundle = remote_dir + "/code.bundle"
    remote_token = remote_dir + "/hf.token"
    remote_campaign = remote_dir + "/campaign"
    bundle_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
    stage_result: object | None = None
    try:
        stage_result = runner(("ssh", "-i", VAST_SSH_IDENTITY, "-o", "IdentitiesOnly=yes", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-p", port, remote, "mkdir", "-p", remote_dir))
        _require_completed(stage_result, "canary remote staging directory")
        stage_result = runner(("scp", "-i", VAST_SSH_IDENTITY, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", port, str(bundle), f"{remote}:{remote_bundle}"))
        _require_completed(stage_result, "canary code-bundle staging")
        stage_result = runner(("scp", "-i", VAST_SSH_IDENTITY, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", port, str(token_file), f"{remote}:{remote_token}"))
        _require_completed(stage_result, "canary token staging")
    except RuntimeError as error:
        return _write_early_abort(lifecycle_root, int(value["wave_index"]), str(attempt["attempt_id"]), int(instance_receipt["instance_id"]), str(error), canary_manifest_sha256=manifest_hash, staged_bundle_sha256=bundle_hash, transport_returncode=getattr(stage_result, "returncode", None))
    local_output = _manifest_output_root([attempt] * 4)
    policy_root = "/workspace/checkpoints/lehome-groot-n17-models-" + policy_revision
    policy_path = policy_root + "/policies/step-12000"
    mapping = {"policy_path": "image:" + policy_path, "policy_revision_file": "image:" + policy_root + "/revision.txt", "release_assets_root": "Assets/objects/Challenge_Garment/Release", "groot_root": "image:" + APPROVED_GROOT_ROOT, "groot_python": "image:" + APPROVED_GROOT_PYTHON, "controller_python": "image:/opt/lehome-challenge/.venv/bin/python", "output_root": "campaign", "trial_script": "scripts/run_groot_flywheel_trial.py"}
    rewritten = _remote_command(attempt["command"], baseline, mapping, remote_dir + "/code", remote_campaign, local_output)
    runtime = [
        _image_identity_preflight(image_identity),
        *_groot_wrapper_setup(),
        "test \"$(git -C " + shlex.quote(APPROVED_GROOT_ROOT) + " rev-parse HEAD)\" = " + shlex.quote(APPROVED_GROOT_REVISION),
        "test \"$(sha256sum " + shlex.quote(remote_bundle) + " | cut -d' ' -f1)\" = " + shlex.quote(bundle_hash),
        "chmod 600 " + shlex.quote(remote_token),
        "export HF_TOKEN=\"$(cat " + shlex.quote(remote_token) + ")\"",
        "test -x /opt/lehome-challenge/.venv/bin/hf",
        "if [ -e " + shlex.quote(remote_dir + "/code") + " ]; then test -d " + shlex.quote(remote_dir + "/code/.git") + " && test \"$(git -C " + shlex.quote(remote_dir + "/code") + " rev-parse HEAD)\" = " + shlex.quote(str(value["baseline"].get("code_revision", ""))) + " && git -C " + shlex.quote(remote_dir + "/code") + " diff --quiet; else git clone --no-checkout " + shlex.quote(remote_bundle) + " " + shlex.quote(remote_dir + "/code") + " && git -C " + shlex.quote(remote_dir + "/code") + " checkout --detach " + shlex.quote(str(value["baseline"].get("code_revision", ""))) + "; fi",
        "test \"$(git -C " + shlex.quote(remote_dir + "/code") + " rev-parse HEAD)\" = " + shlex.quote(str(value["baseline"].get("code_revision", ""))),
        "git -C " + shlex.quote(remote_dir + "/code") + " diff --quiet",
        "mkdir -p " + shlex.quote(policy_path),
        _hf_download("/opt/lehome-challenge/.venv/bin/hf download ryanjin333/lehome-groot-n17-models --revision " + shlex.quote(policy_revision) + " --include 'policies/step-12000/*' --local-dir " + shlex.quote(policy_root)),
        "printf '%s\\n' " + shlex.quote(policy_revision) + " > " + shlex.quote(policy_root + "/revision.txt"),
        *_controller_wire_setup(remote_dir, remote_dir + "/code"),
        _controller_import_preflight(remote_dir + "/code", wire_target=remote_dir + "/controller-wire"),
        _controller_pythonpath(remote_dir + "/code", wire_target=remote_dir + "/controller-wire") + " /opt/lehome-challenge/.venv/bin/python -c " + shlex.quote("from pathlib import Path; from scripts.run_groot_flywheel_trial import policy_artifact_sha256; import sys; sys.exit(0 if policy_artifact_sha256(Path('" + policy_path + "')) == '" + policy_digest + "' else 1)"),
    ]
    runtime.extend(_asset_checkout_setup(remote_dir + "/code"))
    runtime.extend(_qwen_base_setup(remote_dir + "/code"))
    command = " ".join(shlex.quote(item) for item in rewritten)
    controller_pythonpath = remote_dir + "/code/source/lehome:" + remote_dir + "/code"
    remote_log = remote_dir + "/canary.log"
    script = "set +e\n( set -eu\ntrap 'rm -f " + shlex.quote(remote_token) + "; unset HF_TOKEN' EXIT\n" + "\n".join(runtime) + "\nmkdir -p " + shlex.quote(remote_campaign) + "\ncd " + shlex.quote(remote_dir + "/code") + " && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 " + _controller_pythonpath(remote_dir + "/code", wire_target=remote_dir + "/controller-wire") + " LEHOME_FLYWHEEL_WORKER_GPU=0 LEHOME_FLYWHEEL_IMAGE_IDENTITY=" + shlex.quote(image_identity) + " " + command + " ) > " + shlex.quote(remote_log) + " 2>&1\nrc=$?\nprintf '%s\\n' \"$rc\" > " + shlex.quote(remote_dir + "/canary.returncode") + "\nexit $rc"
    result = runner(("ssh", "-i", VAST_SSH_IDENTITY, "-o", "IdentitiesOnly=yes", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-p", port, remote, "sh", "-lc", script))
    sync = lifecycle_root / f"canary-{value['wave_index']:06d}-sync"
    # The corrective controller consumes raw episodes and policy receipts from
    # this campaign root; never copy the code checkout into its evidence tree.
    sync_result = runner(("scp", "-r", "-i", VAST_SSH_IDENTITY, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", port, f"{remote}:{remote_campaign}/.", str(sync)))
    returncode_copy = lifecycle_root / f"canary-{value['wave_index']:06d}-returncode.tmp"
    log_result = runner(("scp", "-i", VAST_SSH_IDENTITY, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", port, f"{remote}:{remote_dir}/canary.returncode", str(returncode_copy)))
    diagnostic_copy = lifecycle_root / f"canary-{value['wave_index']:06d}-diagnostic.tmp"
    diagnostic_result = runner(("scp", "-i", VAST_SSH_IDENTITY, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new", "-P", port, f"{remote}:{remote_log}", str(diagnostic_copy)))
    completed = getattr(result, "returncode", 0)
    base_receipt = {"schema_version": 1, "attempt_id": attempt["attempt_id"], "instance_id": instance_receipt["instance_id"], "transport_returncode": completed, "canary_manifest_sha256": manifest_hash, "staged_bundle_sha256": bundle_hash}
    if getattr(sync_result, "returncode", 0) not in (0, None) or getattr(log_result, "returncode", 0) not in (0, None) or getattr(diagnostic_result, "returncode", 0) not in (0, None):
        return _write_remote_abort(lifecycle_root, int(value["wave_index"]), attempt_id=str(attempt["attempt_id"]), base_receipt=base_receipt, sync_root=sync, returncode_copy=returncode_copy, diagnostic_copy=diagnostic_copy, token_file=token_file, setup="campaign, returncode, or diagnostic synchronization failed", sync_returncode=getattr(sync_result, "returncode", None), returncode_sync_returncode=getattr(log_result, "returncode", None))
    try:
        token_detected = _ingest_canary_diagnostic(diagnostic_copy, sync / "canary.log", token_file)
    except ValueError as error:
        return _write_remote_abort(lifecycle_root, int(value["wave_index"]), attempt_id=str(attempt["attempt_id"]), base_receipt=base_receipt, sync_root=sync, returncode_copy=returncode_copy, diagnostic_copy=None, token_file=token_file, setup=f"diagnostic log rejected: {error}", sync_returncode=getattr(sync_result, "returncode", None), returncode_sync_returncode=getattr(log_result, "returncode", None))
    hashes = {path.relative_to(sync).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sync.rglob("*")) if path.is_file() and not path.is_symlink()} if sync.is_dir() else {}
    base_receipt = {**base_receipt, "synced_evidence_sha256": _canonical_hash(hashes)}
    if token_detected:
        return _write_remote_abort(lifecycle_root, int(value["wave_index"]), attempt_id=str(attempt["attempt_id"]), base_receipt=base_receipt, sync_root=sync, returncode_copy=returncode_copy, diagnostic_copy=None, token_file=token_file, setup="diagnostic log contained staged token and was redacted", sync_returncode=getattr(sync_result, "returncode", None), returncode_sync_returncode=getattr(log_result, "returncode", None))
    if completed not in (0, None):
        return _write_remote_abort(lifecycle_root, int(value["wave_index"]), attempt_id=str(attempt["attempt_id"]), base_receipt=base_receipt, sync_root=sync, returncode_copy=returncode_copy, diagnostic_copy=None, token_file=token_file, setup="remote canary command failed", sync_returncode=getattr(sync_result, "returncode", None), returncode_sync_returncode=getattr(log_result, "returncode", None))
    raw = sync / "raw" / str(attempt["attempt_id"])
    if raw.is_symlink() or not raw.is_dir():
        return _write_remote_abort(lifecycle_root, int(value["wave_index"]), attempt_id=str(attempt["attempt_id"]), base_receipt=base_receipt, sync_root=sync, returncode_copy=returncode_copy, diagnostic_copy=None, token_file=token_file, setup="canonical raw episode is missing after rc0 remote canary", sync_returncode=getattr(sync_result, "returncode", None), returncode_sync_returncode=getattr(log_result, "returncode", None))
    _verify_canonical_episode(raw, str(attempt["attempt_id"]))
    policy_receipt = sync / f"policy-server-receipt-{attempt['attempt_id']}.json"
    _validate_canary_policy_receipt(policy_receipt, attempt, baseline, rewritten)
    episode = _read_json_object(raw / "episode.json", "canary terminal episode")
    if episode.get("outcome") != "success" or episode.get("accepted_success") is not True:
        return _write_new(lifecycle_root / f"canary-{value['wave_index']:06d}-non-training-abort.json", {**base_receipt, "kind": "corrective_canary_non_training_abort", "non_training_admitted": False, "outcome": episode.get("outcome"), "raw_manifest_sha256": hashlib.sha256((raw / "SHA256SUMS.json").read_bytes()).hexdigest(), "policy_receipt_sha256": hashlib.sha256(policy_receipt.read_bytes()).hexdigest()})
    attempt_path = lifecycle_root / f"canary-{value['wave_index']:06d}-attempt-receipt.json"
    attempt_receipt = _materialize_canary_attempt_receipt(canary_manifest, sync, attempt_path)
    terminal_path = lifecycle_root / f"canary-{value['wave_index']:06d}-terminal.json"
    terminal = _write_new(terminal_path, {**base_receipt, "kind": "corrective_canary_terminal", "raw_manifest_sha256": hashlib.sha256((raw / "SHA256SUMS.json").read_bytes()).hexdigest(), "policy_receipt_sha256": hashlib.sha256(policy_receipt.read_bytes()).hexdigest(), "attempt_receipt_sha256": hashlib.sha256(attempt_path.read_bytes()).hexdigest()})
    return {**terminal, "terminal_receipt_path": str(terminal_path), "attempt_receipt_path": str(attempt_path), "attempt_receipt": attempt_receipt}


def _asset_checkout_setup(checkout: str) -> list[str]:
    """Reuse only a fully verified asset checkout; never silently repin it."""
    assets = "/workspace/lehome-release-assets"
    git = "git -C " + shlex.quote(assets)
    lfs_manifest = assets + "/.git/lehome-lfs-manifest"
    return [
        "if [ -e " + shlex.quote(assets) + " ]; then test -d " + shlex.quote(assets + "/.git") + " && test \"$(" + git + " rev-parse HEAD)\" = " + shlex.quote(APPROVED_ASSET_REVISION) + " && " + git + " diff --quiet; else git clone --no-checkout https://huggingface.co/datasets/lehome/asset_challenge " + shlex.quote(assets) + " && " + git + " checkout --detach " + shlex.quote(APPROVED_ASSET_REVISION) + "; fi",
        # Git-LFS does not install its filter in the pinned image, so hydrate
        # the exact dataset revision with HF and then validate every LFS OID.
        _hf_download("/opt/lehome-challenge/.venv/bin/hf download lehome/asset_challenge --repo-type dataset --revision " + shlex.quote(APPROVED_ASSET_REVISION) + " --local-dir " + shlex.quote(assets)),
        git + " lfs install --local",
        git + " lfs ls-files --long > " + shlex.quote(lfs_manifest),
        "while read -r oid marker path; do test \"$(sha256sum " + shlex.quote(assets) + "/\"$path\" | cut -d' ' -f1)\" = \"$oid\"; done < " + shlex.quote(lfs_manifest),
        "test -z \"$(" + git + " lfs ls-files --long | awk '$2 != \"*\" {print}')\"",
        "! grep -RIl '^version https://git-lfs.github.com/spec/v1$' " + shlex.quote(assets + "/objects") + " " + shlex.quote(assets + "/robots") + " " + shlex.quote(assets + "/scenes") + " " + shlex.quote(assets + "/textures"),
        # HF hydration can leave only clean-filter/index metadata marked .M.
        # Normalize the exact four tracked roots after OID verification; the
        # empty staged and worktree diffs prove this did not alter content.
        git + " add --renormalize -- objects robots scenes textures",
        git + " diff --cached --quiet",
        git + " diff --quiet",
        "test -z \"$(" + git + " status --porcelain --untracked-files=all)\"",
        "mkdir -p " + shlex.quote(checkout + "/Assets"),
        "for d in objects robots scenes textures; do ln -sfn " + shlex.quote(assets) + "/$d " + shlex.quote(checkout + "/Assets") + "/$d; done",
    ]


def _write_early_abort(root: Path, wave_index: int, attempt_id: str, instance_id: int, reason: str, *, canary_manifest_sha256: str, staged_bundle_sha256: str, transport_returncode: object) -> dict[str, object]:
    retry = uuid.uuid4().hex
    evidence = root / f"canary-{wave_index:06d}-early-abort-{retry}"
    _write_new(evidence / "setup.json", {"schema_version": 1, "kind": "corrective_canary_setup_abort", "attempt_id": attempt_id, "instance_id": instance_id, "reason": reason, "transport_returncode": transport_returncode})
    _write_new(evidence / "transport.json", {"schema_version": 1, "transport_returncode": transport_returncode, "phase": "staging"})
    _write_new(evidence / "canary.returncode", {"status": "unavailable", "reason": "remote command was not started"})
    evidence_hash = _evidence_root_sha256(evidence)
    receipt_path = root / f"canary-{wave_index:06d}-abort-{retry}.json"
    receipt = _write_new(receipt_path, {"schema_version": 1, "kind": "corrective_canary_abort", "attempt_id": attempt_id, "instance_id": instance_id, "canary_manifest_sha256": canary_manifest_sha256, "staged_bundle_sha256": staged_bundle_sha256, "transport_returncode": transport_returncode, "non_training_admitted": False, "early_setup_failure": True, "retry_id": retry, "abort_evidence_root": str(evidence), "synced_evidence_root": str(evidence), "synced_evidence_sha256": evidence_hash, "abort_evidence_sha256": evidence_hash})
    return {**receipt, "abort_receipt_path": str(receipt_path), "publisher_synced_evidence_root": str(evidence)}


def _copy_regular_tree(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        return
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("synchronized abort evidence may not contain symlinks")
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _evidence_root_sha256(root: Path) -> str:
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*")) if path.is_file() and not path.is_symlink()}
    if not hashes:
        raise ValueError("abort evidence root must be nonempty")
    return _canonical_hash(hashes)


def _ingest_canary_diagnostic(source: Path, destination: Path, token_file: Path) -> bool:
    """Safely retain useful remote diagnostics without ever retaining a token."""
    if source.is_symlink() or not source.is_file():
        raise ValueError("remote diagnostic log is unavailable or non-regular")
    token = token_file.read_bytes()
    if not token:
        raise ValueError("local staged token is empty")
    payload = source.read_bytes()
    detected = token in payload
    if detected:
        payload = payload.replace(token, b"[REDACTED_HF_TOKEN]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("local diagnostic destination is unsafe")
    destination.write_bytes(payload)
    return detected


def _write_remote_abort(root: Path, wave_index: int, *, attempt_id: str, base_receipt: Mapping[str, object], sync_root: Path, returncode_copy: Path, diagnostic_copy: Path | None, token_file: Path, setup: str, sync_returncode: object, returncode_sync_returncode: object) -> dict[str, object]:
    retry = uuid.uuid4().hex
    evidence = root / f"canary-{wave_index:06d}-abort-evidence-{retry}"
    _write_new(evidence / "setup.json", {"schema_version": 1, "kind": "corrective_canary_setup_abort", "attempt_id": attempt_id, "reason": setup})
    _write_new(evidence / "transport.json", {"schema_version": 1, "transport_returncode": base_receipt["transport_returncode"], "campaign_sync_returncode": sync_returncode, "returncode_sync_returncode": returncode_sync_returncode})
    if returncode_copy.is_file() and not returncode_copy.is_symlink():
        shutil.copy2(returncode_copy, evidence / "canary.returncode")
    else:
        _write_new(evidence / "canary.returncode", {"status": "unavailable", "reason": "returncode synchronization failed"})
    _copy_regular_tree(sync_root, evidence / "campaign")
    if diagnostic_copy is not None:
        try:
            _ingest_canary_diagnostic(diagnostic_copy, evidence / "canary.log", token_file)
        except ValueError:
            _write_new(evidence / "canary.log", {"status": "unavailable", "reason": "diagnostic log rejected"})
    elif (sync_root / "canary.log").is_file() and not (sync_root / "canary.log").is_symlink():
        shutil.copy2(sync_root / "canary.log", evidence / "canary.log")
    else:
        _write_new(evidence / "canary.log", {"status": "unavailable", "reason": "diagnostic log synchronization failed"})
    evidence_hash = _evidence_root_sha256(evidence)
    receipt_path = root / f"canary-{wave_index:06d}-abort-{retry}.json"
    receipt = _write_new(receipt_path, {**base_receipt, "kind": "corrective_canary_abort", "non_training_admitted": False, "retry_id": retry, "sync_returncode": sync_returncode, "returncode_sync_returncode": returncode_sync_returncode, "abort_evidence_root": str(evidence), "synced_evidence_root": str(evidence), "synced_evidence_sha256": evidence_hash})
    return {**receipt, "abort_receipt_path": str(receipt_path), "publisher_synced_evidence_root": str(evidence)}


def _materialize_canary_attempt_receipt(canary_manifest: Path, sync_root: Path, output: Path) -> dict[str, object]:
    """Import the campaign's canonical single-attempt verifier without CLI coupling."""
    campaign_path = Path(__file__).with_name("run_groot_corrective_campaign.py")
    spec = __import__("importlib.util").util.spec_from_file_location("corrective_campaign_for_lifecycle", campaign_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("corrective campaign receipt helper is unavailable")
    module = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.materialize_corrective_canary_attempt_receipt(canary_manifest, synced_campaign_root=sync_root, output=output)


def _require_completed(result: object, label: str) -> None:
    if getattr(result, "returncode", 0) not in (0, None):
        raise RuntimeError(f"{label} failed")


def _validate_canary_policy_receipt(path: Path, attempt: Mapping[str, object], baseline: Mapping[str, object], rewritten_command: Sequence[str]) -> None:
    receipt = _read_json_object(path, "canary policy receipt")
    slot = int(attempt.get("worker_slot", -1))
    expected = {"episode_id": attempt["attempt_id"], "backend": "policy_server", "checkpoint_revision": baseline["parent_checkpoint_revision"], "checkpoint_digest": baseline["parent_checkpoint_artifact_sha256"], "code_revision": baseline["code_revision"], "image_identity": baseline["image_identity"], "policy_device": f"cuda:{slot}", "parity_stage": "server_cpu", "simulator_device": "cpu", "groot_revision": baseline["groot_revision"], "python_path": baseline["groot_python"], "policy_seed": attempt.get("seed"), "port": 9100 + slot}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("canary policy receipt does not match immutable runtime")
    command = receipt.get("command")
    policy_path = rewritten_command[rewritten_command.index("--policy-path") + 1]
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command) or "--model-path" not in command or command[command.index("--model-path") + 1] != policy_path:
        raise ValueError("canary policy receipt command does not bind the rewritten policy path")


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is unavailable")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is invalid")
    return value


def _safe_relative(value: object, *, label: str) -> str:
    if label in {"groot_python", "controller_python"} and isinstance(value, str) and value.startswith("image:") and Path(value.removeprefix("image:")).is_absolute():
        return value
    if not isinstance(value, str) or not value or Path(value).is_absolute() or any(part in {"", ".", ".."} for part in Path(value).parts):
        raise ValueError(f"remote bundle {label} path is unsafe")
    return value


def _remote_bundle_mapping(path: Path, digest: str) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema_version", "kind", "bundle_sha256", "paths"}
    if not isinstance(value, dict) or set(value) != expected or value.get("schema_version") != 1 or value.get("kind") != "corrective_remote_bundle" or value.get("bundle_sha256") != digest or not isinstance(value.get("paths"), dict):
        raise ValueError("remote bundle manifest is invalid")
    keys = {"policy_path", "policy_revision_file", "release_assets_root", "groot_root", "groot_python", "controller_python", "output_root", "trial_script"}
    if set(value["paths"]) != keys:
        raise ValueError("remote bundle manifest does not map every runtime path")
    return {key: _safe_relative(item, label=key) for key, item in value["paths"].items()}


def _manifest_output_root(attempts: object) -> str:
    if not isinstance(attempts, list) or len(attempts) != 4:
        raise ValueError("remote manifest does not have four attempts")
    roots: set[str] = set()
    for attempt in attempts:
        command = attempt.get("command") if isinstance(attempt, dict) else None
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError("remote worker command is invalid")
        try:
            root = command[command.index("--output-root") + 1]
        except (ValueError, IndexError) as error:
            raise ValueError("remote worker command lacks output root") from error
        if not Path(root).is_absolute():
            raise ValueError("remote worker output root must be absolute")
        roots.add(root)
    if len(roots) != 1:
        raise ValueError("remote worker output roots must be identical")
    return roots.pop()


def _remote_command(command: object, baseline: object, mapping: Mapping[str, str], checkout: str, remote_output_root: str, local_output_root: str) -> list[str]:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command) or not isinstance(baseline, dict):
        raise ValueError("remote worker command is invalid")
    replacements = {str(baseline[key]): (mapping[key].removeprefix("image:") if mapping[key].startswith("image:") else f"{checkout}/{mapping[key]}") for key in ("policy_path", "policy_revision_file", "release_assets_root", "groot_root", "groot_python", "controller_python")}
    rewritten: list[str] = []
    for token in command:
        if token in replacements:
            rewritten.append(replacements[token])
        elif token == "scripts/run_groot_flywheel_trial.py":
            rewritten.append(f"{checkout}/{mapping['trial_script']}")
        elif token.startswith(local_output_root + "/"):
            rewritten.append(f"{remote_output_root}/{token.removeprefix(local_output_root + '/')}")
        elif token == local_output_root:
            rewritten.append(remote_output_root)
        elif Path(token).is_absolute():
            raise ValueError("remote worker command retains an unmapped local path")
        else:
            rewritten.append(token)
    return rewritten


def build_approved_bundle(manifest_path: Path, *, checkout: Path, output: Path) -> dict[str, object]:
    """Create a free, checksummed tar from the reviewed checkout and pinned paths."""
    manifest = _read_manifest(manifest_path)
    if checkout.is_symlink() or not checkout.is_dir() or output.is_symlink():
        raise ValueError("bundle checkout and output must be safe paths")
    baseline = manifest["baseline"]
    mapping = {
        "policy_path": "policy", "policy_revision_file": "revision.json", "release_assets_root": "assets",
        "groot_root": "groot", "groot_python": "image:" + str(baseline["groot_python"]), "controller_python": "image:" + str(baseline["controller_python"]), "output_root": "campaign",
        "trial_script": "scripts/run_groot_flywheel_trial.py",
    }
    sources = {
        "policy_path": Path(str(baseline["policy_path"])), "policy_revision_file": Path(str(baseline["policy_revision_file"])),
        "release_assets_root": Path(str(baseline["release_assets_root"])), "groot_root": Path(str(baseline["groot_root"])),
        "trial_script": checkout / mapping["trial_script"],
    }
    if any(path.is_symlink() or not path.exists() for path in sources.values()):
        raise ValueError("approved bundle source path is missing or unsafe")
    if output.exists():
        raise ValueError("refusing to overwrite approved bundle")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, mode="x") as archive:
        # Controller code is small and explicitly staged.  Isaac binaries are
        # represented by the pinned interpreter, never recursively archived.
        for relative in ("scripts", "source/lehome"):
            source = checkout / relative
            if source.is_dir() and not source.is_symlink():
                _add_safe_tar(archive, source, relative)
        for key, source in sources.items():
            _add_safe_tar(archive, source, mapping[key])
    _verify_safe_tar(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    receipt = {"schema_version": 1, "kind": "corrective_remote_bundle", "bundle_sha256": digest, "paths": mapping}
    _write_new(output.with_name(output.name + ".manifest.json"), receipt)
    return receipt


def build_code_bundle(*, checkout: Path, revision: str, output: Path, base_bundle: Path) -> dict[str, object]:
    """Create one complete, credential-free bundle from a reviewed compact base."""
    lock = output.with_name(output.name + ".lock")
    if (checkout.is_symlink() or not checkout.is_dir() or output.exists() or output.is_symlink()
            or lock.exists() or lock.is_symlink() or base_bundle.is_symlink() or not base_bundle.is_file()
            or not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise ValueError("code bundle inputs are invalid")
    def run(args: tuple[str, ...]) -> str:
        completed = subprocess.run(args, cwd=checkout, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise ValueError("code checkout is not a clean committed revision")
        return completed.stdout.strip()
    if run(("git", "rev-parse", "HEAD")) != revision or run(("git", "status", "--porcelain")):
        raise ValueError("code checkout is not the requested clean revision")
    created_lock = False
    try:
        with tempfile.TemporaryDirectory(prefix="corrective-code-bundle-") as temporary:
            materialized = Path(temporary) / "checkout"
            subprocess.run(("git", "bundle", "verify", str(base_bundle)), cwd=checkout, check=True, capture_output=True, text=True)
            subprocess.run(("git", "clone", "--no-checkout", str(base_bundle), str(materialized)), check=True, capture_output=True, text=True)
            subprocess.run(("git", "fetch", "--no-tags", "--no-write-fetch-head", str(checkout), f"{revision}:refs/heads/release"), cwd=materialized, check=True, capture_output=True, text=True)
            if subprocess.run(("git", "rev-parse", "refs/heads/release"), cwd=materialized, check=True, capture_output=True, text=True).stdout.strip() != revision:
                raise ValueError("code bundle materialization did not preserve requested revision")
            subprocess.run(("git", "bundle", "create", str(lock), "refs/heads/release"), cwd=materialized, check=True, capture_output=True, text=True)
            created_lock = True
            subprocess.run(("git", "bundle", "verify", str(lock)), cwd=materialized, check=True, capture_output=True, text=True)
            clone = Path(temporary) / "verify"
            subprocess.run(("git", "clone", "--no-checkout", str(lock), str(clone)), check=True, capture_output=True, text=True)
            subprocess.run(("git", "checkout", revision), cwd=clone, check=True, capture_output=True, text=True)
            if subprocess.run(("git", "rev-parse", "HEAD"), cwd=clone, check=True, capture_output=True, text=True).stdout.strip() != revision:
                raise ValueError("code bundle clone did not preserve requested revision")
        os.replace(lock, output)
        created_lock = False
    finally:
        if created_lock and lock.exists():
            lock.unlink()
    return {"schema_version": 1, "kind": "corrective_code_git_bundle", "revision": revision, "base_bundle_sha256": hashlib.sha256(base_bundle.read_bytes()).hexdigest(), "bundle_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def _add_safe_tar(archive: tarfile.TarFile, source: Path, arcname: str) -> None:
    for path in sorted((source.rglob("*") if source.is_dir() else (source,)), key=lambda item: item.as_posix()):
        if path.is_symlink() or not path.is_file():
            if path.is_symlink():
                raise ValueError("approved bundle may not contain symlinks")
            continue
        relative = path.relative_to(source) if source.is_dir() else Path()
        archive.add(path, arcname=(Path(arcname) / relative).as_posix(), recursive=False)


def _verify_safe_tar(path: Path) -> None:
    with tarfile.open(path, mode="r") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            if member.issym() or member.islnk() or member_path.is_absolute() or any(part in {"", ".", ".."} for part in member_path.parts):
                raise ValueError("approved bundle contains an unsafe tar member")


def _terminal_writer_program(output_root: str, manifest: Mapping[str, object]) -> str:
    # JSON is embedded as a quoted Python literal, never interpreted by a shell.
    workers = [{"worker_slot": int(item["worker_slot"]), "attempt_id": str(item["attempt_id"])} for item in manifest["attempts"]]
    return ("import hashlib,json,pathlib;root=pathlib.Path(" + repr(output_root) + ");workers=[];"
            "\nfor item in " + repr(workers) + ":\n slot=item['worker_slot']; p=root/f'worker-{slot}.returncode'; raw=root/'raw'/item['attempt_id']/'SHA256SUMS.json';"
            "\n if not p.is_file(): raise SystemExit(2)\n rc=int(p.read_text().strip()); h=hashlib.sha256(raw.read_bytes()).hexdigest() if raw.is_file() else None; workers.append({'worker_slot':slot,'attempt_id':item['attempt_id'],'returncode':rc,'raw_receipt_path':str(raw.relative_to(root)),'raw_receipt_sha256':h})"
            "\nvalue={'schema_version':1,'kind':'corrective_remote_terminal','workers':workers};(root/'remote-terminal.json').write_text(json.dumps(value,sort_keys=True)+'\\n')")


def _validate_remote_terminal(sync_root: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    terminal = sync_root / "remote-terminal.json"
    if terminal.is_symlink() or not terminal.is_file():
        raise ValueError("remote terminal receipt is missing after synchronization")
    try:
        value = json.loads(terminal.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("remote terminal receipt is invalid") from error
    workers = value.get("workers") if isinstance(value, dict) and value.get("schema_version") == 1 and value.get("kind") == "corrective_remote_terminal" else None
    expected_ids = {int(item["worker_slot"]): str(item["attempt_id"]) for item in manifest["attempts"]}
    if not expected_ids or not set(expected_ids) <= {0, 1, 2, 3} or not isinstance(workers, list) or {item.get("worker_slot") for item in workers if isinstance(item, dict)} != set(expected_ids) or len(workers) != len(expected_ids):
        raise ValueError("remote terminal receipt does not bind scheduled worker slots")
    returncodes: dict[str, int] = {}
    for worker in workers:
        if not isinstance(worker, dict) or type(worker.get("returncode")) is not int:
            raise ValueError("remote terminal worker return code is invalid")
        slot = str(worker["worker_slot"]); returncodes[slot] = worker["returncode"]
    if any(returncode != 0 for returncode in returncodes.values()):
        raise ValueError("remote terminal records a nonzero worker return code")
    for worker in workers:
        if worker.get("attempt_id") != expected_ids[int(worker["worker_slot"])]:
            raise ValueError("remote terminal attempt identity does not match manifest")
        raw_path = worker.get("raw_receipt_path")
        if not isinstance(raw_path, str) or Path(raw_path).is_absolute() or any(part in {"", ".", ".."} for part in Path(raw_path).parts):
            raise ValueError("remote terminal raw receipt path is unsafe")
        raw = sync_root.joinpath(*Path(raw_path).parts)
        expected = worker.get("raw_receipt_sha256")
        if raw.is_symlink() or not raw.is_file() or not isinstance(expected, str) or _SHA256.fullmatch(expected) is None or hashlib.sha256(raw.read_bytes()).hexdigest() != expected:
            raise ValueError("remote terminal raw receipt is missing or mismatched")
        _verify_canonical_episode(raw.parent, expected_ids[int(worker["worker_slot"])])
    return {"worker_returncodes": returncodes, "remote_terminal_sha256": hashlib.sha256(terminal.read_bytes()).hexdigest()}


def _verify_canonical_episode(episode_dir: Path, expected_episode_id: str) -> None:
    try:
        from lehome.flywheel.artifacts import verify_episode_manifest
    except ModuleNotFoundError:
        source = Path(__file__).resolve().parents[1] / "source" / "lehome"
        sys.path.insert(0, str(source))
        from lehome.flywheel.artifacts import verify_episode_manifest
    episode, _manifest = verify_episode_manifest(episode_dir)
    if episode.get("episode_id") != expected_episode_id:
        raise ValueError("remote terminal canonical episode identity is invalid")


def ingest_synced_campaign(sync_root: Path, campaign_root: Path, *, attempt_ids: Sequence[str]) -> dict[str, object]:
    """Atomically admit synchronized raw episodes and policy receipts once."""
    if sync_root.is_symlink() or campaign_root.is_symlink() or len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("synced campaign ingest paths or IDs are unsafe")
    staged: list[tuple[Path, Path]] = []
    for attempt_id in attempt_ids:
        source = sync_root / "raw" / attempt_id
        _verify_canonical_episode(source, attempt_id)
        destination = campaign_root / "raw" / attempt_id
        receipt = sync_root / f"policy-server-receipt-{attempt_id}.json"
        if destination.exists() or destination.is_symlink() or not receipt.is_file() or receipt.is_symlink() or (campaign_root / receipt.name).exists():
            raise ValueError("synced campaign ingest would duplicate or lacks a policy receipt")
        staged.append((source, destination))
    staging = campaign_root.parent / f".{campaign_root.name}.ingest-{_canonical_hash(list(attempt_ids))[:12]}"
    if staging.exists() or staging.is_symlink():
        raise ValueError("synced campaign temporary ingest path already exists")
    renamed: list[tuple[Path, Path]] = []
    try:
        for source, destination in staged:
            relative = destination.relative_to(campaign_root)
            copied = staging / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, copied, copy_function=shutil.copy2, symlinks=False)
            _verify_canonical_episode(copied, destination.name)
            receipt = sync_root / f"policy-server-receipt-{destination.name}.json"
            receipt_destination = staging / receipt.name
            shutil.copy2(receipt, receipt_destination)
        for source, destination in staged:
            target = campaign_root / destination.relative_to(campaign_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            episode_stage = staging / target.relative_to(campaign_root)
            policy_stage = staging / f"policy-server-receipt-{destination.name}.json"
            policy_target = campaign_root / policy_stage.name
            episode_stage.rename(target); renamed.append((target, episode_stage))
            policy_stage.rename(policy_target); renamed.append((policy_target, policy_stage))
    except BaseException:
        for target, original in reversed(renamed):
            if target.exists() and not original.exists():
                original.parent.mkdir(parents=True, exist_ok=True)
                target.rename(original)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return _write_new(campaign_root / "ingest-receipts" / ("-".join(attempt_ids) + ".json"), {"schema_version": 1, "kind": "corrective_synced_campaign_ingest", "attempt_ids": list(attempt_ids)})


def destroy_after_publication(instance_id: int, publication_receipt: Path, lifecycle_receipt: Path, *, runner: Callable[[tuple[str, ...]], object]) -> bool:
    """Destroy only when private immutable HF readback and instance binding exist."""
    publication = json.loads(publication_receipt.read_text(encoding="utf-8")) if publication_receipt.is_file() and not publication_receipt.is_symlink() else None
    lifecycle = json.loads(lifecycle_receipt.read_text(encoding="utf-8")) if lifecycle_receipt.is_file() and not lifecycle_receipt.is_symlink() else None
    if not isinstance(publication, dict) or publication.get("disposable") is not True or not isinstance(publication.get("immutable_revision"), str) or publication.get("fresh_readback_verified") is not True or publication.get("tree_listing_verified") is not True:
        raise ValueError("publisher disposal receipt lacks immutable HF proof")
    if not isinstance(lifecycle, dict) or lifecycle.get("kind") != "corrective_vast_instance" or lifecycle.get("instance_id") != instance_id:
        raise ValueError("lifecycle receipt is not bound to instance")
    runner(("vastai", "destroy", "instance", str(instance_id), "--yes"))
    absent = _run_raw(runner, ("vastai", "--raw", "show", "instance", str(instance_id)))
    if absent not in (None, {}, []):
        raise ValueError("vastai destroy readback still finds instance")
    return True


def destroy_after_canary_publication(instance_id: int, publication_receipt: Path, lifecycle_receipt: Path, *, canary_attempt_id: str, runner: Callable[[tuple[str, ...]], object]) -> bool:
    """Canary disposal is deliberately distinct from the 150-episode release."""
    publication = json.loads(publication_receipt.read_text(encoding="utf-8")) if publication_receipt.is_file() and not publication_receipt.is_symlink() else None
    if not isinstance(publication, dict) or publication.get("kind") != "corrective_rft_private_canary" or publication.get("attempt_id") != canary_attempt_id or publication.get("instance_id") != instance_id or publication.get("repository_private") is not True or publication.get("training_admission") is not False:
        raise ValueError("canary disposal requires an exact private HF canary publication receipt")
    return destroy_after_publication(instance_id, publication_receipt, lifecycle_receipt, runner=runner)


def _subprocess_runner(command: tuple[str, ...]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError("external lifecycle command failed; inspect its local lifecycle log")
    return completed.stdout


def _nonraising_subprocess_runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    evidence = actions.add_parser("capture-evidence")
    evidence.add_argument("--lifecycle-root", type=Path, required=True)
    evidence.add_argument("--wave-index", type=int, required=True)
    evidence.add_argument("--preferred-offer-id", type=int)
    evidence.add_argument("--retained-instance-id", type=int)
    evidence.add_argument("--prior-provider-evidence", type=Path)
    evidence.add_argument("--prior-instance-receipt", type=Path)
    evidence.add_argument("--execute", action="store_true")
    for name in ("rent", "remote-launch"):
        item = actions.add_parser(name)
        item.add_argument("--manifest", type=Path, required=True)
        item.add_argument("--lifecycle-root", type=Path, required=True)
    rent = actions.choices["rent"]
    rent.add_argument("--execute", action="store_true")
    renew = actions.add_parser("renew-lease")
    renew.add_argument("--manifest", type=Path, required=True)
    renew.add_argument("--prior-instance-receipt", type=Path, required=True)
    renew.add_argument("--lifecycle-root", type=Path, required=True)
    renew.add_argument("--execute", action="store_true")
    adopt = actions.add_parser("adopt-retained-lease")
    adopt.add_argument("--manifest", type=Path, required=True)
    adopt.add_argument("--prior-instance-receipt", type=Path, required=True)
    adopt.add_argument("--prior-provider-evidence", type=Path, required=True)
    adopt.add_argument("--fresh-provider-evidence", type=Path, required=True)
    adopt.add_argument("--lifecycle-root", type=Path, required=True)
    adopt.add_argument("--execute", action="store_true")
    remote = actions.choices["remote-launch"]
    remote.add_argument("--instance-receipt", type=Path, required=True)
    remote.add_argument("--code-git-bundle", type=Path, required=True)
    remote.add_argument("--token-file", type=Path, required=True)
    remote.add_argument("--execute", action="store_true")
    bundle = actions.add_parser("build-bundle")
    bundle.add_argument("--manifest", type=Path, required=True)
    bundle.add_argument("--checkout", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    bundle.add_argument("--execute", action="store_true")
    code_bundle = actions.add_parser("build-code-bundle")
    code_bundle.add_argument("--checkout", type=Path, required=True)
    code_bundle.add_argument("--revision", required=True)
    code_bundle.add_argument("--base-bundle", type=Path, required=True)
    code_bundle.add_argument("--output", type=Path, required=True)
    code_bundle.add_argument("--execute", action="store_true")
    canary = actions.add_parser("canary-launch")
    canary.add_argument("--canary-manifest", type=Path, required=True)
    canary.add_argument("--instance-receipt", type=Path, required=True)
    canary.add_argument("--lifecycle-root", type=Path, required=True)
    canary.add_argument("--code-git-bundle", type=Path, required=True)
    canary.add_argument("--token-file", type=Path, help="local secret file staged without reading its value")
    canary.add_argument("--execute", action="store_true")
    destroy = actions.add_parser("destroy")
    destroy.add_argument("--instance-id", type=int, required=True)
    destroy.add_argument("--publication-receipt", type=Path, required=True)
    destroy.add_argument("--instance-receipt", type=Path, required=True)
    destroy.add_argument("--execute", action="store_true")
    canary_destroy = actions.add_parser("canary-destroy")
    canary_destroy.add_argument("--instance-id", type=int, required=True)
    canary_destroy.add_argument("--publication-receipt", type=Path, required=True)
    canary_destroy.add_argument("--instance-receipt", type=Path, required=True)
    canary_destroy.add_argument("--attempt-id", required=True)
    canary_destroy.add_argument("--execute", action="store_true")
    ingest = actions.add_parser("ingest")
    ingest.add_argument("--sync-root", type=Path, required=True)
    ingest.add_argument("--campaign-root", type=Path, required=True)
    ingest.add_argument("--attempt-id", action="append", required=True)
    ingest.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "dry_run", "action": args.action}, sort_keys=True))
        return 0
    if args.action == "capture-evidence":
        instances = _run_raw(_subprocess_runner, ("vastai", "--raw", "show", "instances"))
        volumes = _run_raw(_subprocess_runner, ("vastai", "--raw", "show", "volumes"))
        if args.retained_instance_id is not None:
            if args.preferred_offer_id is not None or args.prior_provider_evidence is None or args.prior_instance_receipt is None:
                raise ValueError("retained evidence requires prior evidence/receipt and no offer selector")
            offers: object = []
            prior_evidence = _read_json_object(args.prior_provider_evidence, "prior provider evidence")
            prior_receipt = _read_json_object(args.prior_instance_receipt, "prior instance receipt")
        else:
            if args.prior_provider_evidence is not None or args.prior_instance_receipt is not None:
                raise ValueError("prior evidence options require a retained instance ID")
            offers = _run_raw(_subprocess_runner, ("vastai", "--raw", "search", "offers", OFFER_QUERY, "--on-demand", "--storage", "300"))
            prior_evidence = prior_receipt = None
        if not isinstance(instances, list) or not isinstance(volumes, list) or not isinstance(offers, list):
            raise ValueError("vastai raw evidence response is invalid")
        result = capture_offer_evidence(offers=offers, instances=instances, volumes=volumes, output=args.lifecycle_root / f"wave-{args.wave_index:06d}-provider.json", now_unix=__import__("time").time_ns() // 1_000_000_000, ttl_seconds=300, preferred_offer_id=args.preferred_offer_id, retained_instance_id=args.retained_instance_id, prior_provider_evidence=prior_evidence, prior_instance_receipt=prior_receipt)
    elif args.action == "build-bundle":
        result = build_approved_bundle(args.manifest, checkout=args.checkout, output=args.output)
    elif args.action == "build-code-bundle":
        result = build_code_bundle(checkout=args.checkout, revision=args.revision, output=args.output, base_bundle=args.base_bundle)
    elif args.action == "rent":
        result = rent_wave(args.manifest, lifecycle_root=args.lifecycle_root, runner=_subprocess_runner, now_unix=__import__("time").time_ns() // 1_000_000_000)
    elif args.action == "renew-lease":
        result = renew_retained_lease(args.manifest, args.prior_instance_receipt, lifecycle_root=args.lifecycle_root, runner=_subprocess_runner)
    elif args.action == "adopt-retained-lease":
        result = adopt_retained_lease(args.manifest, args.prior_instance_receipt, args.prior_provider_evidence, args.fresh_provider_evidence, lifecycle_root=args.lifecycle_root, runner=_subprocess_runner)
    elif args.action == "remote-launch":
        instance = json.loads(args.instance_receipt.read_text(encoding="utf-8"))
        result = remote_launch_wave(args.manifest, instance, lifecycle_root=args.lifecycle_root, runner=_nonraising_subprocess_runner, code_bundle=args.code_git_bundle, token_file=args.token_file)
    elif args.action == "canary-launch":
        instance = json.loads(args.instance_receipt.read_text(encoding="utf-8"))
        result = remote_launch_canary(args.canary_manifest, instance, lifecycle_root=args.lifecycle_root, runner=_nonraising_subprocess_runner, bundle=args.code_git_bundle, token_file=args.token_file)
    elif args.action == "canary-destroy":
        result = {"destroyed": destroy_after_canary_publication(args.instance_id, args.publication_receipt, args.instance_receipt, canary_attempt_id=args.attempt_id, runner=_subprocess_runner)}
    elif args.action == "ingest":
        result = ingest_synced_campaign(args.sync_root, args.campaign_root, attempt_ids=tuple(args.attempt_id))
    else:
        result = {"destroyed": destroy_after_publication(args.instance_id, args.publication_receipt, args.instance_receipt, runner=_subprocess_runner)}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
