"""Frozen contracts for a native public-GR00T-N1.5 seen-garment harvest.

The functions in this module are deliberately offline.  They build and check
immutable evidence, render the upstream evaluator argv, and fail closed at the
first-100, worker-admission, publication, and provider-stop boundaries.  They
do not provision infrastructure or execute policy code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import time
from typing import Callable, Mapping, Sequence

from .n15_reproduction import CONTRACT


class HarvestError(RuntimeError):
    """A public-N1.5 harvest identity or evidence gate failed."""


CATEGORIES: Mapping[str, str] = {
    "top_long": "Top_Long",
    "top_short": "Top_Short",
    "pant_long": "Pant_Long",
    "pant_short": "Pant_Short",
}
_CATEGORY_ORDER = tuple(CATEGORIES)
_CATEGORY_DATASET_ROOTS = {
    category: "Datasets/example/four_types_merged" for category in CATEGORIES
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_REPOSITORY = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")
_ATTEMPT_ID = re.compile(
    r"n15-seen-(?:top-long|top-short|pant-long|pant-short)-g[0-9]{2}-e[0-9]{2}-s[0-9]{6}"
)
_ATTEMPT_FIELDS = frozenset(
    {
        "attempt_id", "ordinal", "category", "garment", "garment_index",
        "episode_index", "process_seed",
    }
)
ROLLOUT_IMAGE_ID = "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7"
RUNTIME_PYTHON = "/opt/lehome-challenge/.venv/bin/python"
_FORBIDDEN_LOG = re.compile(
    r"Traceback \(most recent call last\):|Error during evaluation:|"
    r"non[- ]?finite|CUDA error|policy transport failure|"
    r"(?:^|[^A-Za-z])(?:nan|[+-]?inf)(?:$|[^A-Za-z])",
    re.IGNORECASE | re.MULTILINE,
)
_EPISODE_LOG = re.compile(
    r"Episode\s+(?P<episode>\d+)/25:\s+Return=(?P<return>[-+0-9.eE]+),\s+"
    r"Length=(?P<length>\d+),\s+Success=(?P<success>True|False)"
)
_COMPLETION_LOG = "Evaluation completed successfully"
_EXCLUSIONS = {
    "augmentation": True,
    "curriculum": True,
    "focused_evaluator_episode_identities": True,
    "focused_evaluator_seed_42": True,
    "hard_states": True,
    "historical_episodes": True,
    "perturbation": True,
    "release_unseen_garments": True,
}
_SUMMARY = {
    "attempt_count": 1000,
    "attempts_per_category": 250,
    "attempts_per_garment": 25,
    "category_count": 4,
    "garment_count": 40,
    "garments_per_category": 10,
    "native_process_count": 40,
    "episodes_per_process": 25,
    "first_wave_process_count": 4,
}
_SCOPE = {
    "dataset_partition": "Seen",
    "release_evaluator_assets_included": False,
    # The focused evaluator uses these two category names too.  The excluded
    # identities are its Release garments and fixed seed/episode pairs, not
    # the required Seen_0..Seen_9 collection garments.
    "required_seen_garments_overlap_focused_categories": True,
}
_ASSIGNMENTS = {
    4: {
        "0": ["top_long"],
        "1": ["top_short"],
        "2": ["pant_long"],
        "3": ["pant_short"],
    },
    2: {
        "0": ["top_long", "pant_long"],
        "1": ["top_short", "pant_short"],
    },
}


@dataclass(frozen=True, slots=True)
class HarvestProvenance:
    """Pinned checkpoint and upstream identities carried by every manifest."""

    checkpoint_tree_sha256: str
    checkpoint_receipt_sha256: str
    runtime_receipt_sha256: str
    source_tree_sha256: str
    dataset_snapshot_sha256: str
    rollout_image_sha256: str

    def __post_init__(self) -> None:
        if (
            any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.checkpoint_tree_sha256,
                    self.checkpoint_receipt_sha256,
                    self.runtime_receipt_sha256,
                    self.source_tree_sha256,
                    self.dataset_snapshot_sha256,
                    self.rollout_image_sha256,
                )
            )
        ):
            raise HarvestError("checkpoint provenance is invalid")


def canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError):
        raise HarvestError("value is not canonical strict JSON") from None


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _manifest_provenance(provenance: HarvestProvenance) -> dict[str, object]:
    return {
        "checkpoint_receipt_sha256": provenance.checkpoint_receipt_sha256,
        "checkpoint_tree_sha256": provenance.checkpoint_tree_sha256,
        "dataset_snapshot_sha256": provenance.dataset_snapshot_sha256,
        "dataset_repository": CONTRACT.dataset_repository,
        "dataset_revision": CONTRACT.dataset_revision,
        "source_repository": CONTRACT.source_repository,
        "source_revision": CONTRACT.source_revision,
        "source_tree_sha256": provenance.source_tree_sha256,
        "runtime_receipt_sha256": provenance.runtime_receipt_sha256,
        "rollout_image_sha256": provenance.rollout_image_sha256,
        "upstream_evaluator": "python -P -m scripts.eval",
    }


def build_manifest(*, provenance: HarvestProvenance, base_seed: int = 100_000) -> dict[str, object]:
    """Build the fixed 40-garment, 1,000-attempt uniform manifest."""

    if type(base_seed) is not int or base_seed < 1000 or base_seed + 40 > 999_999:
        raise HarvestError("base seed is outside the six-digit frozen range")
    if base_seed <= 42 < base_seed + 40:
        raise HarvestError("base seed overlaps focused evaluator seed 42")
    attempts: list[dict[str, object]] = []
    ordinal = 0
    # One upstream process evaluates 25 episodes for one garment.  Seen_0 in
    # each category is wave 1 (exactly 100 outcomes), so the breaker can run
    # before the remaining 36 garment processes.  A process seed is not
    # misrepresented as an independently injected per-episode seed.
    for garment_index in range(10):
        for category_index, (category, prefix) in enumerate(CATEGORIES.items()):
            process_seed = base_seed + garment_index * 4 + category_index
            for episode_index in range(1, 26):
                attempts.append(
                    {
                        "attempt_id": (
                            f"n15-seen-{category.replace('_', '-')}-"
                            f"g{garment_index:02d}-e{episode_index:02d}-s{process_seed:06d}"
                        ),
                        "category": category,
                        "episode_index": episode_index,
                        "garment": f"{prefix}_Seen_{garment_index}",
                        "garment_index": garment_index,
                        "ordinal": ordinal,
                        "process_seed": process_seed,
                    }
                )
                ordinal += 1
    value: dict[str, object] = {
        "schema_version": 1,
        "kind": "lehome_public_n15_uniform_seen_harvest_v1",
        "provenance": _manifest_provenance(provenance),
        "scope": dict(_SCOPE),
        "exclusions": dict(_EXCLUSIONS),
        "summary": dict(_SUMMARY),
        "attempts": attempts,
    }
    return validate_manifest(value)


def _require_exact_mapping(value: object, fields: set[str] | frozenset[str], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise HarvestError(f"{label} schema is invalid")
    return dict(value)


def validate_manifest(value: object) -> dict[str, object]:
    """Validate the entire manifest, including deterministic row identity."""

    manifest = _require_exact_mapping(
        value,
        {"schema_version", "kind", "provenance", "scope", "exclusions", "summary", "attempts"},
        "harvest manifest",
    )
    if manifest["schema_version"] != 1 or manifest["kind"] != "lehome_public_n15_uniform_seen_harvest_v1":
        raise HarvestError("harvest manifest identity is invalid")
    provenance = _require_exact_mapping(
        manifest["provenance"],
        {
            "checkpoint_receipt_sha256", "checkpoint_tree_sha256", "dataset_repository",
            "dataset_revision", "dataset_snapshot_sha256", "source_repository", "source_revision",
            "source_tree_sha256", "runtime_receipt_sha256", "rollout_image_sha256",
            "upstream_evaluator",
        },
        "harvest provenance",
    )
    if (
        _SHA256.fullmatch(str(provenance["checkpoint_tree_sha256"])) is None
        or _SHA256.fullmatch(str(provenance["checkpoint_receipt_sha256"])) is None
        or any(
            _SHA256.fullmatch(str(provenance[key])) is None
            for key in (
                "dataset_snapshot_sha256", "source_tree_sha256",
                "runtime_receipt_sha256", "rollout_image_sha256",
            )
        )
        or provenance["dataset_repository"] != CONTRACT.dataset_repository
        or provenance["dataset_revision"] != CONTRACT.dataset_revision
        or provenance["source_repository"] != CONTRACT.source_repository
        or provenance["source_revision"] != CONTRACT.source_revision
        or provenance["upstream_evaluator"] != "python -P -m scripts.eval"
    ):
        raise HarvestError("harvest provenance is invalid")
    if manifest["scope"] != _SCOPE or manifest["exclusions"] != _EXCLUSIONS or manifest["summary"] != _SUMMARY:
        raise HarvestError("harvest scope, exclusions, or summary changed")
    attempts = manifest["attempts"]
    if not isinstance(attempts, list) or len(attempts) != 1000:
        raise HarvestError("harvest must contain exactly 1,000 attempts")
    raw_first_seed = attempts[0].get("process_seed") if isinstance(attempts[0], Mapping) else None
    if type(raw_first_seed) is not int or raw_first_seed < 1000 or raw_first_seed + 40 > 999_999:
        raise HarvestError("harvest base process seed is invalid")
    base_process_seed = raw_first_seed
    ids: set[str] = set()
    process_episode_pairs: set[tuple[int, int]] = set()
    process_seeds: set[int] = set()
    counts_by_category = {category: 0 for category in CATEGORIES}
    counts_by_garment = {
        f"{prefix}_Seen_{index}": 0
        for prefix in CATEGORIES.values()
        for index in range(10)
    }
    for ordinal, raw in enumerate(attempts):
        row = _require_exact_mapping(raw, _ATTEMPT_FIELDS, "harvest attempt")
        attempt_id = row["attempt_id"]
        category = row["category"]
        garment_index = row["garment_index"]
        episode_index = row["episode_index"]
        process_seed = row["process_seed"]
        if (
            not isinstance(attempt_id, str)
            or _ATTEMPT_ID.fullmatch(attempt_id) is None
            or category not in CATEGORIES
            or type(garment_index) is not int
            or not 0 <= garment_index < 10
            or type(episode_index) is not int
            or not 1 <= episode_index <= 25
            or type(process_seed) is not int
            or not 0 <= process_seed <= 999_999
            or row["ordinal"] != ordinal
        ):
            raise HarvestError("harvest attempt identity is invalid")
        expected_garment_index = ordinal // 100
        expected_category = _CATEGORY_ORDER[(ordinal // 25) % 4]
        expected_episode_index = ordinal % 25 + 1
        expected_garment = f"{CATEGORIES[expected_category]}_Seen_{expected_garment_index}"
        expected_process_seed = (
            base_process_seed
            + expected_garment_index * 4
            + _CATEGORY_ORDER.index(expected_category)
        )
        expected_id = (
            f"n15-seen-{expected_category.replace('_', '-')}-"
            f"g{expected_garment_index:02d}-e{expected_episode_index:02d}-s{process_seed:06d}"
        )
        if (
            category != expected_category
            or garment_index != expected_garment_index
            or episode_index != expected_episode_index
            or row["garment"] != expected_garment
            or process_seed != expected_process_seed
            or attempt_id != expected_id
            or "_Unseen_" in str(row["garment"])
            or process_seed == 42
        ):
            raise HarvestError("harvest attempt violates the frozen seen schedule")
        pair = (process_seed, episode_index)
        if attempt_id in ids or pair in process_episode_pairs:
            raise HarvestError("harvest attempt IDs and process/episode pairs must be globally unique")
        ids.add(attempt_id)
        process_episode_pairs.add(pair)
        process_seeds.add(process_seed)
        counts_by_category[category] += 1
        counts_by_garment[expected_garment] += 1
    if (
        set(counts_by_category.values()) != {250}
        or set(counts_by_garment.values()) != {25}
        or len(process_seeds) != 40
    ):
        raise HarvestError("harvest is not exactly uniform")
    if any(sum(row["category"] == category for row in attempts[:100]) != 25 for category in CATEGORIES):
        raise HarvestError("first-100 prefix is not category balanced")
    return manifest


def _safe_new_file(path: Path, payload: bytes, *, label: str) -> None:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise HarvestError(f"{label} path is unsafe")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise HarvestError(f"{label} already exists")
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        created = True
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o444)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        if created and path.exists():
            path.unlink()
        raise HarvestError(f"{label} could not be written atomically") from error


def manifest_receipt(manifest: object) -> dict[str, object]:
    verified = validate_manifest(manifest)
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_harvest_manifest_receipt_v1",
        "manifest_sha256": _digest(verified),
        "checkpoint_tree_sha256": verified["provenance"]["checkpoint_tree_sha256"],
        "attempt_count": 1000,
    }


def write_manifest_bundle(*, manifest: object, manifest_path: Path, receipt_path: Path) -> dict[str, object]:
    verified = validate_manifest(manifest)
    if manifest_path == receipt_path:
        raise HarvestError("manifest and receipt paths must differ")
    receipt = manifest_receipt(verified)
    _safe_new_file(manifest_path, canonical_bytes(verified), label="harvest manifest")
    try:
        _safe_new_file(receipt_path, canonical_bytes(receipt), label="harvest receipt")
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        raise
    return receipt


def verify_manifest_receipt(*, manifest: object, receipt: object) -> dict[str, object]:
    expected = manifest_receipt(manifest)
    if receipt != expected:
        raise HarvestError("harvest manifest receipt mismatch")
    return expected


def evaluate_first_100(*, manifest: object, outcomes: object) -> dict[str, object]:
    verified = validate_manifest(manifest)
    if not isinstance(outcomes, list) or len(outcomes) != 100:
        raise HarvestError("first-100 evidence must contain exactly 100 outcomes")
    expected_ids = [row["attempt_id"] for row in verified["attempts"][:100]]
    success_count = 0
    infrastructure_invalid_count = 0
    for expected_id, raw in zip(expected_ids, outcomes, strict=True):
        row = _require_exact_mapping(raw, {"attempt_id", "official_outcome", "cloth_fidelity"}, "first-100 outcome")
        fidelity = _require_exact_mapping(row["cloth_fidelity"], {"measured", "valid"}, "cloth fidelity")
        if row["attempt_id"] != expected_id:
            raise HarvestError("first-100 attempt identity mismatch")
        if fidelity != {"measured": True, "valid": True}:
            raise HarvestError("first-100 cloth fidelity gate failed")
        if row["official_outcome"] == "success":
            success_count += 1
        elif row["official_outcome"] == "infrastructure_invalid":
            infrastructure_invalid_count += 1
        elif row["official_outcome"] != "policy_failure":
            raise HarvestError("first-100 official outcome is invalid")
    if success_count < 5:
        raise HarvestError("first-100 has fewer than five official successes")
    if infrastructure_invalid_count > 2:
        raise HarvestError("first-100 has more than 2% infrastructure-invalid attempts")
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_first_100_gate_v1",
        "decision": "continue",
        "attempt_count": 100,
        "success_count": success_count,
        "infrastructure_invalid_count": infrastructure_invalid_count,
        "manifest_sha256": _digest(verified),
    }


def _validate_worker_receipt(
    value: object,
    *,
    worker_count: int,
    checkpoint_tree_sha256: str,
) -> bool:
    receipt = _require_exact_mapping(
        value,
        {"schema_version", "kind", "worker_count", "memory_check", "smokes"},
        f"{worker_count}-worker admission",
    )
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "lehome_public_n15_worker_admission_v1"
        or receipt["worker_count"] != worker_count
    ):
        raise HarvestError(f"{worker_count}-worker admission identity is invalid")
    memory = _require_exact_mapping(
        receipt["memory_check"],
        {"passed", "policy_process_count", "gpu_oom", "semantic_overrides"},
        f"{worker_count}-worker memory check",
    )
    passed = memory["passed"] is True
    if (
        type(memory["policy_process_count"]) is not int
        or memory["policy_process_count"] != worker_count
        or type(memory["gpu_oom"]) is not bool
        or memory["semantic_overrides"] is not False
    ):
        raise HarvestError(f"{worker_count}-worker memory check is invalid")
    if passed and memory["gpu_oom"] is not False:
        raise HarvestError(f"{worker_count}-worker memory check failed")
    smokes = receipt["smokes"]
    if not passed:
        if smokes != []:
            raise HarvestError(f"failed {worker_count}-worker admission cannot carry passing smokes")
        return False
    if not isinstance(smokes, list) or len(smokes) != worker_count:
        raise HarvestError(f"{worker_count}-worker smoke evidence is incomplete")
    for worker_id, raw in enumerate(smokes):
        smoke = _require_exact_mapping(
            raw,
            {
                "worker_id", "categories", "official_outcome", "infrastructure_invalid",
                "cloth_fidelity", "checkpoint_tree_sha256",
            },
            f"{worker_count}-worker smoke",
        )
        fidelity = _require_exact_mapping(smoke["cloth_fidelity"], {"measured", "valid"}, "worker smoke cloth fidelity")
        if (
            smoke["worker_id"] != worker_id
            or smoke["categories"] != _ASSIGNMENTS[worker_count][str(worker_id)]
            or smoke["official_outcome"] not in {"success", "policy_failure"}
            or smoke["infrastructure_invalid"] is not False
            or fidelity != {"measured": True, "valid": True}
            or smoke["checkpoint_tree_sha256"] != checkpoint_tree_sha256
        ):
            raise HarvestError(f"{worker_count}-worker smoke gate failed")
    return True


def admit_workers(
    *,
    four_worker_receipt: object,
    two_worker_receipt: object | None,
    checkpoint_tree_sha256: str,
) -> dict[str, object]:
    if _SHA256.fullmatch(checkpoint_tree_sha256) is None:
        raise HarvestError("worker admission checkpoint identity is invalid")
    if _validate_worker_receipt(
        four_worker_receipt,
        worker_count=4,
        checkpoint_tree_sha256=checkpoint_tree_sha256,
    ):
        count = 4
        fallback = False
    else:
        if two_worker_receipt is None:
            raise HarvestError("four-worker admission failed and two-worker fallback is missing")
        if not _validate_worker_receipt(
            two_worker_receipt,
            worker_count=2,
            checkpoint_tree_sha256=checkpoint_tree_sha256,
        ):
            raise HarvestError("two-worker fallback did not pass admission")
        count = 2
        fallback = True
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_worker_selection_v1",
        "worker_count": count,
        "fallback_from_four": fallback,
        "assignments": _ASSIGNMENTS[count],
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
    }


def _safe_runtime_root(path: Path, *, label: str, must_exist: bool = True) -> Path:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts) or path.is_symlink():
        raise HarvestError(f"{label} is unsafe")
    if must_exist and not path.is_dir():
        raise HarvestError(f"{label} is unavailable")
    return path


def regular_tree_identity(
    root: Path,
    *,
    relative_paths: Sequence[str] | None = None,
    ignore_python_cache: bool = True,
) -> tuple[int, str]:
    """Measure a regular-file tree without following symlinks."""

    base = _safe_runtime_root(Path(root), label="measured tree root")
    selected: list[tuple[str, Path]] = []
    if relative_paths is None:
        candidates = sorted(base.rglob("*"))
        for path in candidates:
            relative = path.relative_to(base).as_posix()
            if path.is_dir():
                continue
            selected.append((relative, path))
    else:
        seen: set[str] = set()
        for relative in sorted(relative_paths):
            pure = PurePosixPath(relative)
            if (
                not relative
                or pure.is_absolute()
                or any(part in {"", ".", ".."} for part in pure.parts)
                or relative in seen
            ):
                raise HarvestError("measured tree path is unsafe or duplicated")
            seen.add(relative)
            selected.append((relative, base / relative))
    rows: list[tuple[str, bytes]] = []
    for relative, path in selected:
        if ignore_python_cache and ("__pycache__" in Path(relative).parts or relative.endswith(".pyc")):
            continue
        if path.is_symlink() or not path.is_file():
            raise HarvestError(f"measured tree contains unsafe entry: {relative}")
        rows.append((relative, path.read_bytes()))
    if not rows:
        raise HarvestError("measured tree is empty")
    payload = b"".join(
        relative.encode("utf-8")
        + b"\0"
        + hashlib.sha256(content).hexdigest().encode("ascii")
        + b"\n"
        for relative, content in rows
    )
    return len(rows), hashlib.sha256(payload).hexdigest()


def _git_output(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HarvestError("pinned source Git measurement failed") from error
    if result.returncode:
        raise HarvestError("pinned source Git measurement failed")
    return result.stdout


def _default_image_validator(
    receipt_path: Path,
    inspect_path: Path,
    *,
    expected_image_id: str,
) -> dict[str, object]:
    try:
        receipt_raw = receipt_path.read_bytes()
        receipt = json.loads(receipt_raw)
        inspect_raw = inspect_path.read_bytes()
        inspected = json.loads(inspect_raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HarvestError("rollout image evidence is malformed") from error
    if (
        not isinstance(receipt, Mapping)
        or receipt_raw != canonical_bytes(receipt)
        or receipt.get("kind") != "lehome_official_image_inspection_v1"
        or receipt.get("reference") != expected_image_id
        or receipt.get("image_id") != expected_image_id
        or receipt.get("docker_inspect_sha256") != hashlib.sha256(inspect_raw).hexdigest()
        or not isinstance(inspected, list)
        or len(inspected) != 1
        or not isinstance(inspected[0], Mapping)
        or inspected[0].get("Id") != expected_image_id
    ):
        raise HarvestError("rollout image identity is invalid")
    return dict(receipt)


def measure_runtime_contract(
    *,
    source_root: Path,
    source_revision: str,
    checkpoint_root: Path,
    training_identity_receipt: Path,
    rollout_image_receipt: Path,
    docker_inspect_receipt: Path,
    python_executable: Path,
    python_version: str,
    lerobot_package_root: Path,
    training_validator: Callable[..., Mapping[str, object]] | None = None,
    inputs_validator: Callable[..., object] | None = None,
    expected_lerobot_tree: tuple[int, str] | None = None,
    expected_image_id: str = ROLLOUT_IMAGE_ID,
    image_validator: Callable[..., Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Bind live harvest inputs to Task 1 and the pinned rollout runtime."""

    source = _safe_runtime_root(Path(source_root), label="public source root")
    checkpoint = _safe_runtime_root(Path(checkpoint_root), label="checkpoint root")
    if source_revision != CONTRACT.source_revision and training_validator is None:
        raise HarvestError("public source revision mismatch")
    if _REVISION.fullmatch(source_revision) is None:
        raise HarvestError("public source revision is invalid")
    if _git_output(source, "rev-parse", "HEAD").strip() != source_revision:
        raise HarvestError("public source revision mismatch")
    if _git_output(source, "status", "--porcelain", "--untracked-files=all"):
        raise HarvestError("public source checkout is not fully clean")
    tracked = [item for item in _git_output(source, "ls-files", "-z").split("\0") if item]
    source_count, source_tree = regular_tree_identity(
        source, relative_paths=tracked, ignore_python_cache=False
    )
    checkpoint_count, checkpoint_tree = regular_tree_identity(checkpoint)
    package_count, package_tree = regular_tree_identity(Path(lerobot_package_root))
    expected_package = expected_lerobot_tree or (
        CONTRACT.lerobot_package_file_count,
        CONTRACT.lerobot_package_tree_sha256,
    )
    if (package_count, package_tree) != expected_package:
        raise HarvestError("installed LeRobot package tree mismatch")
    if str(python_executable) != RUNTIME_PYTHON or not re.fullmatch(r"3\.11(?:\.\d+)?", python_version):
        raise HarvestError("runtime interpreter identity mismatch")
    if training_validator is None:
        from rollout_appliance.native_reference_site.training_identity import (
            validate_training_identity_receipt,
        )

        training_validator = validate_training_identity_receipt
    try:
        training = dict(
            training_validator(
                Path(training_identity_receipt),
                expected_contract=CONTRACT,
                expected_pretrained_root=checkpoint,
            )
        )
    except Exception as error:
        raise HarvestError("complete Task 1 training identity is invalid") from error
    training_root_raw = training.get("training_root")
    if inputs_validator is None:
        from .n15_reproduction import verify_inputs

        inputs_validator = verify_inputs
    if isinstance(training_root_raw, str):
        training_root = Path(training_root_raw)
        source_receipt = training_root / "evidence/source-receipt.json"
        snapshots_receipt = training_root / "evidence/resolved-snapshots-receipt.json"
    else:
        # Dependency-injected tests still exercise the call boundary.
        source_receipt = Path(training_identity_receipt)
        snapshots_receipt = Path(training_identity_receipt)
    try:
        verified_inputs = inputs_validator(
            checkout=source,
            source_receipt=source_receipt,
            resolved_snapshots_receipt=snapshots_receipt,
            vm_id=CONTRACT.vm_id,
            disk_id=CONTRACT.disk_id,
            contract=CONTRACT,
        )
    except Exception as error:
        raise HarvestError("live Task 1 source/dataset snapshots are invalid") from error
    image_check = image_validator or _default_image_validator
    try:
        image = dict(
            image_check(
                Path(rollout_image_receipt),
                Path(docker_inspect_receipt),
                expected_image_id=expected_image_id,
            )
        )
    except Exception as error:
        raise HarvestError("rollout image evidence is invalid") from error
    image_id = image.get("image_id")
    image_digest = str(image_id).removeprefix("sha256:")
    if image_id != expected_image_id or _SHA256.fullmatch(image_digest) is None:
        raise HarvestError("rollout image evidence is invalid")
    training_receipt_sha = hashlib.sha256(Path(training_identity_receipt).read_bytes()).hexdigest()
    dataset_snapshot_sha = getattr(verified_inputs, "resolved_snapshots_receipt_sha256", None)
    if not isinstance(dataset_snapshot_sha, str) or _SHA256.fullmatch(dataset_snapshot_sha) is None:
        raise HarvestError("dataset snapshot evidence is invalid")
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_harvest_runtime_v1",
        "source_revision": source_revision,
        "source_file_count": source_count,
        "source_tree_sha256": source_tree,
        "checkpoint_file_count": checkpoint_count,
        "checkpoint_tree_sha256": checkpoint_tree,
        "training_identity_receipt_sha256": training_receipt_sha,
        "dataset_snapshot_sha256": dataset_snapshot_sha,
        "base_model_metadata_sha256": getattr(verified_inputs, "base_model_metadata_sha256"),
        "dataset_metadata_sha256": getattr(verified_inputs, "dataset_metadata_sha256"),
        "rollout_image_id": image_id,
        "rollout_image_sha256": image_digest,
        "docker_inspect_sha256": image["docker_inspect_sha256"],
        "python_executable": str(python_executable),
        "python_version": python_version,
        "lerobot_package_file_count": package_count,
        "lerobot_package_tree_sha256": package_tree,
        "semantic_environment": {
            "policy_overrides": False,
            "checkpoint_compatibility_override": False,
            "cloth_fidelity_monitor": "observational_only",
        },
    }


def _expected_processes(manifest: Mapping[str, object], expected_attempt_count: int) -> list[dict[str, object]]:
    if expected_attempt_count not in {100, 1000}:
        raise HarvestError("collector expected attempt count must be 100 or 1,000")
    rows = manifest["attempts"][:expected_attempt_count]
    result: list[dict[str, object]] = []
    seen: set[tuple[object, object, object]] = set()
    for row in rows:
        identity = (row["category"], row["garment"], row["process_seed"])
        if identity in seen:
            continue
        seen.add(identity)
        result.append(row)
    expected_process_count = 4 if expected_attempt_count == 100 else 40
    if len(result) != expected_process_count:
        raise HarvestError("collector process schedule drift")
    return result


def collect_native_outcomes(
    *,
    manifest: object,
    process_status: object,
    harvest_root: Path,
    expected_attempt_count: int,
) -> dict[str, object]:
    """Authenticate native logs and observational cloth evidence for every ID."""

    verified = validate_manifest(manifest)
    root = _safe_runtime_root(Path(harvest_root), label="harvest root")
    status = _require_exact_mapping(
        process_status,
        {"schema_version", "kind", "processes"},
        "process status receipt",
    )
    if status["schema_version"] != 1 or status["kind"] != "lehome_public_n15_process_status_v1":
        raise HarvestError("process status identity is invalid")
    expected = _expected_processes(verified, expected_attempt_count)
    statuses = status["processes"]
    if not isinstance(statuses, list) or len(statuses) != len(expected):
        raise HarvestError("process status count is incomplete")
    outcomes: list[dict[str, object]] = []
    for expected_row, raw_status in zip(expected, statuses, strict=True):
        process = _require_exact_mapping(
            raw_status,
            {"process_id", "category", "garment", "process_seed", "exit_code"},
            "process status",
        )
        process_id = (
            f"{expected_row['category']}-{expected_row['garment_index']:02d}-"
            f"s{expected_row['process_seed']:06d}"
        )
        if process != {
            "process_id": process_id,
            "category": expected_row["category"],
            "garment": expected_row["garment"],
            "process_seed": expected_row["process_seed"],
            "exit_code": 0,
        }:
            raise HarvestError("native process identity or exit status is invalid")
        process_root = root / "processes" / process_id
        if process_root.is_symlink() or not process_root.is_dir():
            raise HarvestError("native process output is unavailable")
        log = process_root / "evaluator.log"
        if log.is_symlink() or not log.is_file():
            raise HarvestError("native evaluator log is unavailable")
        try:
            text = log.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError) as error:
            raise HarvestError("native evaluator log is malformed") from error
        if _FORBIDDEN_LOG.search(text) or text.count(_COMPLETION_LOG) != 1:
            raise HarvestError("native evaluator log contains an infrastructure failure")
        matches = list(_EPISODE_LOG.finditer(text))
        if len(matches) != 25 or [int(match.group("episode")) for match in matches] != list(range(1, 26)):
            raise HarvestError("native evaluator log does not contain exactly 25 ordered outcomes")
        try:
            from rollout_appliance.native_reference_site.cloth_fidelity import (
                validate_cloth_fidelity_evidence,
            )

            fidelity = validate_cloth_fidelity_evidence(
                process_root / "cloth-fidelity.jsonl",
                expected_episodes=[(str(expected_row["garment"]), index) for index in range(1, 26)],
            )
        except (ImportError, OSError, ValueError) as error:
            raise HarvestError("native cloth fidelity evidence is invalid") from error
        if fidelity["measured_episode_count"] != 25 or fidelity["fidelity_invalid_count"] != 0:
            raise HarvestError("native cloth fidelity gate failed")
        process_attempts = [
            row
            for row in verified["attempts"][:expected_attempt_count]
            if row["category"] == expected_row["category"]
            and row["garment"] == expected_row["garment"]
            and row["process_seed"] == expected_row["process_seed"]
        ]
        if len(process_attempts) != 25:
            raise HarvestError("native process-to-manifest mapping drift")
        for row, match in zip(process_attempts, matches, strict=True):
            episode_return = float(match.group("return"))
            if not (-float("inf") < episode_return < float("inf")):
                raise HarvestError("native evaluator return is non-finite")
            outcomes.append(
                {
                    "attempt_id": row["attempt_id"],
                    "official_outcome": "success" if match.group("success") == "True" else "policy_failure",
                    "return": episode_return,
                    "length": int(match.group("length")),
                    "cloth_fidelity": {"measured": True, "valid": True},
                }
            )
    expected_ids = [row["attempt_id"] for row in verified["attempts"][:expected_attempt_count]]
    if [row["attempt_id"] for row in outcomes] != expected_ids:
        raise HarvestError("collected outcome identity order is incomplete")
    success_count = sum(row["official_outcome"] == "success" for row in outcomes)
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_collected_outcomes_v1",
        "manifest_sha256": _digest(verified),
        "attempt_count": expected_attempt_count,
        "process_count": len(expected),
        "success_count": success_count,
        "policy_failure_count": expected_attempt_count - success_count,
        "infrastructure_invalid_count": 0,
        "fidelity_invalid_count": 0,
        "outcomes": outcomes,
    }


def validate_collected_outcomes(
    value: object, *, manifest: object, expected_attempt_count: int = 1000
) -> dict[str, object]:
    verified = validate_manifest(manifest)
    result = _require_exact_mapping(
        value,
        {
            "schema_version", "kind", "manifest_sha256", "attempt_count", "process_count",
            "success_count", "policy_failure_count", "infrastructure_invalid_count",
            "fidelity_invalid_count", "outcomes",
        },
        "collected outcomes",
    )
    expected_process_count = 4 if expected_attempt_count == 100 else 40
    outcomes = result["outcomes"]
    if (
        result["schema_version"] != 1
        or result["kind"] != "lehome_public_n15_collected_outcomes_v1"
        or result["manifest_sha256"] != _digest(verified)
        or result["attempt_count"] != expected_attempt_count
        or result["process_count"] != expected_process_count
        or result["infrastructure_invalid_count"] != 0
        or result["fidelity_invalid_count"] != 0
        or not isinstance(outcomes, list)
        or len(outcomes) != expected_attempt_count
    ):
        raise HarvestError("collected outcome summary is invalid")
    expected_ids = [row["attempt_id"] for row in verified["attempts"][:expected_attempt_count]]
    successes = 0
    for expected_id, raw in zip(expected_ids, outcomes, strict=True):
        row = _require_exact_mapping(
            raw, {"attempt_id", "official_outcome", "return", "length", "cloth_fidelity"},
            "collected outcome",
        )
        if (
            row["attempt_id"] != expected_id
            or row["official_outcome"] not in {"success", "policy_failure"}
            or not isinstance(row["return"], (int, float))
            or type(row["length"]) is not int
            or row["length"] < 0
            or row["cloth_fidelity"] != {"measured": True, "valid": True}
        ):
            raise HarvestError("collected outcome is invalid")
        successes += row["official_outcome"] == "success"
    if result["success_count"] != successes or result["policy_failure_count"] != expected_attempt_count - successes:
        raise HarvestError("collected outcome counts do not reconcile")
    return result


def _bundle_entries(root: Path) -> list[dict[str, object]]:
    base = _safe_runtime_root(root, label="harvest publication bundle")
    entries: list[dict[str, object]] = []
    for path in sorted(base.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(base).as_posix()
        if path.is_symlink() or not path.is_file():
            raise HarvestError(f"publication bundle contains unsafe entry: {relative}")
        payload = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )
    if not entries:
        raise HarvestError("publication bundle is empty")
    return entries


def _entries_tree(entries: Sequence[Mapping[str, object]]) -> str:
    return _digest(
        [
            {"path": row["path"], "sha256": row["sha256"], "size": row["size"]}
            for row in entries
        ]
    )


def _readback_entries(
    *,
    api: object,
    downloader: Callable[..., str],
    repository: str,
    revision: str,
    remote_prefix: str,
    entries: Sequence[Mapping[str, object]],
    token: object,
) -> str:
    expected = {f"{remote_prefix}/{row['path']}" for row in entries}
    try:
        observed = {
            path
            for path in api.list_repo_files(
                repo_id=repository, repo_type="dataset", revision=revision
            )
            if path.startswith(remote_prefix + "/")
        }
    except Exception as error:
        raise HarvestError("Hugging Face readback listing failed") from error
    if observed != expected:
        raise HarvestError("Hugging Face readback file set mismatch")
    measured: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="lehome-n15-harvest-readback-") as temporary:
        for row in entries:
            try:
                downloaded = Path(
                    downloader(
                        repo_id=repository,
                        repo_type="dataset",
                        filename=f"{remote_prefix}/{row['path']}",
                        revision=revision,
                        token=token,
                        local_dir=temporary,
                    )
                )
            except Exception as error:
                raise HarvestError("Hugging Face byte readback failed") from error
            if (
                downloaded.is_symlink()
                or not downloaded.is_file()
                or downloaded.stat().st_size != row["size"]
                or hashlib.sha256(downloaded.read_bytes()).hexdigest() != row["sha256"]
            ):
                raise HarvestError("Hugging Face byte readback mismatch")
            measured.append(dict(row))
    return _entries_tree(measured)


def publish_harvest_bundle(
    *,
    bundle_root: Path,
    manifest: object,
    manifest_receipt_value: object,
    final_outcomes: object,
    repository: str,
    revision: str,
    token: str,
    authenticated_api: object,
    anonymous_api: object,
    downloader: Callable[..., str],
    operation_factory: Callable[..., object],
) -> dict[str, object]:
    """Upload once, then authenticate and anonymously read back every byte."""

    verified = validate_manifest(manifest)
    receipt = verify_manifest_receipt(manifest=verified, receipt=manifest_receipt_value)
    outcomes = validate_collected_outcomes(final_outcomes, manifest=verified)
    if not token:
        raise HarvestError("Hugging Face token is unavailable")
    if not isinstance(repository, str) or _REPOSITORY.fullmatch(repository) is None:
        raise HarvestError("Hugging Face repository is invalid")
    if revision != "main":
        raise HarvestError("Hugging Face mutable publication ref must be main")
    try:
        authenticated_info = authenticated_api.repo_info(
            repo_id=repository, repo_type="dataset", revision=revision
        )
        anonymous_info = anonymous_api.repo_info(
            repo_id=repository, repo_type="dataset", revision=revision
        )
    except Exception as error:
        raise HarvestError("public Hugging Face repository is unavailable") from error
    if getattr(authenticated_info, "private", None) is not False or getattr(anonymous_info, "private", None) is not False:
        raise HarvestError("Hugging Face repository is not public")
    root = _safe_runtime_root(Path(bundle_root), label="harvest publication bundle")
    required = {
        "manifest.json": canonical_bytes(verified),
        "manifest-receipt.json": canonical_bytes(receipt),
        "final-outcomes.json": canonical_bytes(outcomes),
    }
    for relative, payload in required.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise HarvestError(f"publication bundle required evidence mismatch: {relative}")
    entries = _bundle_entries(root)
    manifest_sha = _digest(verified)
    remote_prefix = f"harvest/public-n15-{manifest_sha[:16]}"
    existing = authenticated_api.list_repo_files(
        repo_id=repository, repo_type="dataset", revision=revision
    )
    prefix_files = [path for path in existing if path.startswith(remote_prefix + "/")]
    if prefix_files:
        immutable_revision = getattr(anonymous_info, "sha", None)
    else:
        operations = [
            operation_factory(
                path_in_repo=f"{remote_prefix}/{row['path']}",
                path_or_fileobj=str(root / str(row["path"])),
            )
            for row in entries
        ]
        try:
            commit = authenticated_api.create_commit(
                repo_id=repository,
                repo_type="dataset",
                revision=revision,
                operations=operations,
                commit_message="Publish public N1.5 native harvest",
            )
        except Exception as error:
            raise HarvestError("Hugging Face upload failed") from error
        immutable_revision = getattr(commit, "oid", None)
    if not isinstance(immutable_revision, str) or _REVISION.fullmatch(immutable_revision) is None:
        raise HarvestError("Hugging Face upload did not return an immutable revision")
    uploaded_tree = _entries_tree(entries)
    authenticated_tree = _readback_entries(
        api=authenticated_api,
        downloader=downloader,
        repository=repository,
        revision=immutable_revision,
        remote_prefix=remote_prefix,
        entries=entries,
        token=token,
    )
    anonymous_tree = _readback_entries(
        api=anonymous_api,
        downloader=downloader,
        repository=repository,
        revision=immutable_revision,
        remote_prefix=remote_prefix,
        entries=entries,
        token=False,
    )
    if len({uploaded_tree, authenticated_tree, anonymous_tree}) != 1:
        raise HarvestError("Hugging Face upload/readback tree mismatch")
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_harvest_hf_readback_v1",
        "repository": repository,
        "repository_type": "dataset",
        "repository_private": False,
        "remote_prefix": remote_prefix,
        "immutable_revision": immutable_revision,
        "manifest_sha256": manifest_sha,
        "manifest_receipt_sha256": _digest(receipt),
        "upload_tree_sha256": uploaded_tree,
        "authenticated_readback_tree_sha256": authenticated_tree,
        "anonymous_readback_tree_sha256": anonymous_tree,
        "readback_verified": True,
        "anonymous_readback_verified": True,
    }


def native_worker_plan(
    *,
    manifest: object,
    admission: object,
    source_root: Path,
    checkpoint_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Render one exact native ``scripts.eval`` command per seen garment."""

    verified = validate_manifest(manifest)
    selection = _require_exact_mapping(
        admission,
        {"schema_version", "kind", "worker_count", "fallback_from_four", "assignments", "checkpoint_tree_sha256"},
        "worker selection",
    )
    count = selection["worker_count"]
    if (
        selection["schema_version"] != 1
        or selection["kind"] != "lehome_public_n15_worker_selection_v1"
        or count not in {2, 4}
        or selection["assignments"] != _ASSIGNMENTS[count]
        or selection["checkpoint_tree_sha256"] != verified["provenance"]["checkpoint_tree_sha256"]
        or selection["fallback_from_four"] is not (count == 2)
    ):
        raise HarvestError("worker selection does not bind the manifest")
    source_root = _safe_runtime_root(source_root, label="public source root")
    checkpoint_root = _safe_runtime_root(checkpoint_root, label="checkpoint root")
    output_root = _safe_runtime_root(output_root, label="harvest output root", must_exist=False)
    eval_file = source_root / "scripts/eval.py"
    if not eval_file.is_file() or eval_file.is_symlink():
        raise HarvestError("public scripts.eval entrypoint is unavailable")
    workers: list[dict[str, object]] = []
    for worker_id, categories in selection["assignments"].items():
        rows = [row for row in verified["attempts"] if row["category"] in categories]
        process_rows: list[dict[str, object]] = []
        seen_processes: set[tuple[str, str, int]] = set()
        for row in rows:
            identity = (row["category"], row["garment"], row["process_seed"])
            if identity not in seen_processes:
                seen_processes.add(identity)
                process_rows.append(row)
        commands = []
        for row in process_rows:
            process_id = f"{row['category']}-{row['garment_index']:02d}-s{row['process_seed']:06d}"
            attempt_root = output_root / "processes" / process_id
            commands.append(
                [
                    "python", "-P", "-m", "scripts.eval",
                    "--policy_type", "lerobot",
                    "--policy_path", str(checkpoint_root),
                    "--garment_type", row["category"],
                    "--garment_filter", row["garment"],
                    "--dataset_root", str(source_root / _CATEGORY_DATASET_ROOTS[row["category"]]),
                    "--num_episodes", "25",
                    "--seed", str(row["process_seed"]),
                    "--enable_cameras",
                    "--headless",
                    "--device", "cpu",
                    "--save_datasets",
                    "--eval_dataset_path", str(attempt_root / "dataset"),
                    "--log_suffix", process_id,
                ]
            )
        workers.append({"worker_id": int(worker_id), "categories": categories, "commands": commands})
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_native_worker_plan_v1",
        "manifest_sha256": _digest(verified),
        "worker_count": count,
        "workers": workers,
    }


def terminal_receipt(
    *,
    manifest: object,
    manifest_receipt: object,
    publication_receipt: object,
    provider_receipt: object,
) -> dict[str, object]:
    """Verify immutable public readback and the exact terminal STOPPED state."""

    verified = validate_manifest(manifest)
    verified_manifest_receipt = verify_manifest_receipt(manifest=verified, receipt=manifest_receipt)
    publication = _require_exact_mapping(
        publication_receipt,
        {
            "schema_version", "kind", "repository", "repository_type", "repository_private", "remote_prefix",
            "immutable_revision", "manifest_sha256", "manifest_receipt_sha256",
            "upload_tree_sha256", "authenticated_readback_tree_sha256",
            "anonymous_readback_tree_sha256", "readback_verified", "anonymous_readback_verified",
        },
        "Hugging Face publication receipt",
    )
    manifest_receipt_sha256 = _digest(verified_manifest_receipt)
    trees = (
        publication["upload_tree_sha256"],
        publication["authenticated_readback_tree_sha256"],
        publication["anonymous_readback_tree_sha256"],
    )
    if (
        publication["schema_version"] != 1
        or publication["kind"] != "lehome_public_n15_harvest_hf_readback_v1"
        or not isinstance(publication["repository"], str)
        or _REPOSITORY.fullmatch(publication["repository"]) is None
        or publication["repository_type"] != "dataset"
        or publication["repository_private"] is not False
    ):
        raise HarvestError("Hugging Face destination is not a public repository")
    if not isinstance(publication["immutable_revision"], str) or _REVISION.fullmatch(publication["immutable_revision"]) is None:
        raise HarvestError("Hugging Face revision is not immutable")
    if (
        not isinstance(publication["remote_prefix"], str)
        or not publication["remote_prefix"].startswith("harvest/public-n15-")
        or ".." in publication["remote_prefix"].split("/")
        or publication["manifest_sha256"] != _digest(verified)
        or publication["manifest_receipt_sha256"] != manifest_receipt_sha256
    ):
        raise HarvestError("Hugging Face publication provenance mismatch")
    if (
        any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in trees)
        or len(set(trees)) != 1
        or publication["readback_verified"] is not True
        or publication["anonymous_readback_verified"] is not True
    ):
        raise HarvestError("Hugging Face authenticated/anonymous byte readback failed")
    provider = validate_provider_stop_receipt(provider_receipt)
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_harvest_terminal_v1",
        "terminal": True,
        "manifest_sha256": _digest(verified),
        "immutable_revision": publication["immutable_revision"],
        "provider_state": "STOPPED",
        "vm_id": CONTRACT.vm_id,
        "disk_id": CONTRACT.disk_id,
    }


def validate_provider_stop_receipt(provider_receipt: object) -> dict[str, object]:
    provider = _require_exact_mapping(
        provider_receipt,
        {
            "schema_version", "kind", "vm_id", "instance_name", "disk_id", "state",
            "protected_disk_preserved", "created_resources", "deleted_resources",
            "provider_response_sha256", "captured_unix_seconds",
        },
        "provider stopped receipt",
    )
    if (
        provider["schema_version"] != 1
        or provider["kind"] != "lehome_public_n15_provider_stopped_v1"
        or provider["vm_id"] != CONTRACT.vm_id
        or provider["instance_name"] != "lehome-rollout"
        or provider["disk_id"] != CONTRACT.disk_id
        or provider["state"] != "STOPPED"
        or provider["protected_disk_preserved"] is not True
        or provider["created_resources"] != []
        or provider["deleted_resources"] != []
        or not isinstance(provider["provider_response_sha256"], str)
        or _SHA256.fullmatch(provider["provider_response_sha256"]) is None
        or type(provider["captured_unix_seconds"]) is not int
        or provider["captured_unix_seconds"] <= 0
    ):
        raise HarvestError("exact provider VM is not STOPPED with protected disk preserved")
    return provider


def provider_stop_receipt_from_response(response: object) -> dict[str, object]:
    """Validate one authoritative nested Nebius STOPPED response."""

    if not isinstance(response, Mapping):
        raise HarvestError("provider response is invalid")
    metadata = response.get("metadata")
    status = response.get("status")
    spec = response.get("spec")
    if not isinstance(metadata, Mapping) or not isinstance(status, Mapping) or not isinstance(spec, Mapping):
        raise HarvestError("provider response is invalid")
    disks = spec.get("secondary_disks")
    found: list[object] = []
    if not isinstance(disks, list):
        raise HarvestError("provider protected disk evidence is missing")
    for disk in disks:
        existing = disk.get("existing_disk") if isinstance(disk, Mapping) else None
        if not isinstance(disk, Mapping) or not isinstance(existing, Mapping) or set(existing) != {"id"}:
            raise HarvestError("provider protected disk attachment is invalid")
        found.append(existing.get("id"))
    if (
        metadata.get("id") != CONTRACT.vm_id
        or metadata.get("name") != "lehome-rollout"
        or status.get("state") != "STOPPED"
        or found != [CONTRACT.disk_id]
    ):
        raise HarvestError("provider response is not the exact stopped rollout VM")
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_provider_stopped_v1",
        "vm_id": CONTRACT.vm_id,
        "instance_name": "lehome-rollout",
        "disk_id": CONTRACT.disk_id,
        "state": "STOPPED",
        "protected_disk_preserved": True,
        "created_resources": [],
        "deleted_resources": [],
        "provider_response_sha256": _digest(response),
        "captured_unix_seconds": int(time.time()),
    }
