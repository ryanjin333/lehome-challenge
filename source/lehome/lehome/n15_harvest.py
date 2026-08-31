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
_FORBIDDEN_IMAGE_ENV = frozenset({
    "LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT",
    "LEHOME_NATIVE_REFERENCE_SANITIZED_CONFIG_ROOT",
    "LEHOME_NATIVE_REFERENCE_CHECKPOINT_COMPATIBILITY_RECEIPT",
    "LEHOME_CPU_ACTION",
})
_FORBIDDEN_LOG = re.compile(
    r"Traceback \(most recent call last\):|Error during evaluation:|"
    r"non[- ]?finite|CUDA error|policy transport failure|"
    r"(?:^|[^A-Za-z])(?:nan|[+-]?inf)(?:$|[^A-Za-z])",
    re.IGNORECASE | re.MULTILINE,
)
_CAPACITY_FAILURE_LOG = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUDNN_STATUS_ALLOC_FAILED|"
    r"(?:^|\n)Killed(?:\n|$)|std::bad_alloc|MemoryError",
    re.IGNORECASE,
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
_ADMISSION_SMOKE_SEEDS = {4: 940_000, 2: 920_000}

_OBSERVATIONAL_SITECUSTOMIZE = '''\
"""Harvest-only observational hooks; no policy or action compatibility path."""
import os
from pathlib import Path
import sys


def fatal(message):
    print(f"native harvest site error: {message}", file=sys.stderr, flush=True)
    os._exit(72)


raw_log_root = os.environ.get("LEHOME_NATIVE_REFERENCE_LOG_PROJECT_ROOT", "")
raw_source_root = os.environ.get("LEHOME_NATIVE_REFERENCE_SOURCE_ROOT", "")
if not raw_log_root or not raw_source_root:
    fatal("log and source roots are required")
try:
    source_root = Path(raw_source_root).resolve(strict=True)
    requested = Path(raw_log_root)
    if not requested.is_absolute() or ".." in requested.parts or requested.is_symlink():
        fatal("log root is unsafe")
    requested.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_root = requested.resolve(strict=True)
except OSError:
    fatal("log root is unavailable")
if log_root == source_root or source_root in log_root.parents:
    fatal("log root is inside pinned source")
try:
    import lehome.utils.logger as lehome_logger
    lehome_logger.get_project_root = lambda: log_root
except Exception as error:
    fatal(f"logger routing failed: {error}")

raw_fidelity = os.environ.get("LEHOME_NATIVE_CLOTH_FIDELITY_EVIDENCE", "")
if raw_fidelity:
    fidelity_path = Path(raw_fidelity)
    if (
        not fidelity_path.is_absolute()
        or ".." in fidelity_path.parts
        or fidelity_path.exists()
        or fidelity_path.is_symlink()
    ):
        fatal("cloth fidelity evidence path is unsafe")
    try:
        import gymnasium
        from cloth_fidelity import install_cloth_fidelity_monitor_on_env

        original_make = gymnasium.make
        installed = False

        def monitored_make(*args, **kwargs):
            global installed
            created = original_make(*args, **kwargs)
            if installed:
                fatal("native evaluator created more than one environment")
            install_cloth_fidelity_monitor_on_env(created.unwrapped, fidelity_path)
            installed = True
            return created

        gymnasium.make = monitored_make
    except Exception as error:
        fatal(f"cloth fidelity monitor installation failed: {error}")
'''


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


def admission_smoke_schedule(worker_count: int) -> list[dict[str, object]]:
    """Return frozen capacity-smoke identities outside the harvest namespace."""

    if worker_count not in _ASSIGNMENTS:
        raise HarvestError("worker admission count must be two or four")
    result: list[dict[str, object]] = []
    for worker_id in range(worker_count):
        category = _ASSIGNMENTS[worker_count][str(worker_id)][0]
        garment = f"{CATEGORIES[category]}_Seen_0"
        seed = _ADMISSION_SMOKE_SEEDS[worker_count] + worker_id
        result.append({
            "worker_id": worker_id,
            "smoke_id": f"n15-admission-w{worker_count}-{category.replace('_', '-')}-s{seed:06d}",
            "category": category,
            "garment": garment,
            "process_seed": seed,
            "episode_index": 1,
        })
    return result


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


def write_observational_site(output_dir: Path) -> dict[str, object]:
    """Materialize the reviewed, observation-only import hook for native eval."""

    root = _safe_runtime_root(
        Path(output_dir), label="observational site root", must_exist=False,
    )
    if root.exists() or root.is_symlink():
        raise HarvestError("observational site root already exists")
    try:
        root.mkdir(mode=0o700)
    except OSError as error:
        raise HarvestError("observational site root could not be created") from error
    payload = _OBSERVATIONAL_SITECUSTOMIZE.encode("utf-8")
    _safe_new_file(root / "sitecustomize.py", payload, label="observational sitecustomize")
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_observational_site_v1",
        "relative_path": "sitecustomize.py",
        "sitecustomize_sha256": hashlib.sha256(payload).hexdigest(),
    }


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


def _canonical_file(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise HarvestError(f"{label} is unavailable")
    try:
        raw = candidate.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HarvestError(f"{label} is malformed") from error
    if not isinstance(value, Mapping) or raw != canonical_bytes(value):
        raise HarvestError(f"{label} is not canonical")
    return dict(value), raw


def _status_tsv(path: Path, *, worker_count: int, label: str) -> tuple[str, list[int]]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise HarvestError(f"{label} is unavailable")
    raw = candidate.read_bytes()
    try:
        lines = raw.decode("ascii").splitlines()
        rows = [tuple(int(field) for field in line.split("\t")) for line in lines]
    except (UnicodeError, ValueError) as error:
        raise HarvestError(f"{label} is malformed") from error
    if (
        len(rows) != worker_count
        or any(
            len(row) != 2
            or row[0] != worker_id
            or not 0 <= row[1] <= 255
            for worker_id, row in enumerate(rows)
        )
    ):
        raise HarvestError(f"{label} is incomplete or malformed")
    return hashlib.sha256(raw).hexdigest(), [row[1] for row in rows]


def _derive_memory_check(root: Path, *, worker_count: int) -> dict[str, object]:
    evidence = _safe_runtime_root(Path(root), label="worker admission evidence")
    status_sha, exit_codes = _status_tsv(
        evidence / "memory-status.tsv", worker_count=worker_count,
        label="zero-episode process status",
    )
    log_rows = []
    capacity_failures: list[dict[str, object]] = []
    for worker_id, exit_code in enumerate(exit_codes):
        log = evidence / "memory" / f"worker-{worker_id}.log"
        if log.is_symlink() or not log.is_file():
            raise HarvestError("zero-episode memory log is unavailable")
        raw = log.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeError as error:
            raise HarvestError("zero-episode memory log is malformed") from error
        if exit_code:
            if _CAPACITY_FAILURE_LOG.search(text) is None:
                raise HarvestError("zero-episode process failed for a non-capacity reason")
            capacity_failures.append({
                "worker_id": worker_id,
                "exit_code": exit_code,
                "log_sha256": hashlib.sha256(raw).hexdigest(),
            })
        elif (
            _FORBIDDEN_LOG.search(text)
            or _EPISODE_LOG.search(text)
            or "Starting evaluation: 0 episodes" not in text
            or text.count(_COMPLETION_LOG) != 1
        ):
            raise HarvestError("zero-episode memory process did not complete natively")
        log_rows.append({"worker_id": worker_id, "log_sha256": hashlib.sha256(raw).hexdigest()})
    measurement = evidence / "memory.tsv"
    if measurement.is_symlink() or not measurement.is_file():
        raise HarvestError("GPU memory measurements are unavailable")
    raw_measurement = measurement.read_bytes()
    try:
        lines = raw_measurement.decode("ascii").splitlines()
        if lines[:1] != ["sample_index\tactive_process_count\tgpu_used_mib\tgpu_total_mib"]:
            raise ValueError("header")
        rows = [tuple(int(field) for field in line.split("\t")) for line in lines[1:]]
    except (UnicodeError, ValueError) as error:
        raise HarvestError("GPU memory measurements are malformed") from error
    if not rows or [row[0] for row in rows] != list(range(len(rows))):
        raise HarvestError("GPU memory measurement sequence is incomplete")
    totals = {row[3] for row in rows}
    if (
        len(totals) != 1
        or next(iter(totals)) <= 0
        or any(not 0 <= row[1] <= worker_count or not 0 <= row[2] <= row[3] for row in rows)
        or rows[0][1] != 0
        or not any(row[1] > 0 for row in rows)
    ):
        raise HarvestError("GPU memory measurements do not cover native processes")
    if not capacity_failures and not any(row[1] == worker_count for row in rows):
        raise HarvestError("GPU memory measurements do not cover all native processes")
    total = next(iter(totals))
    baseline = rows[0][2]
    peak = max(row[2] for row in rows if row[1] > 0)
    if peak <= baseline:
        raise HarvestError("GPU memory measurements did not observe policy allocation")
    headroom = total - peak
    passed = not capacity_failures and headroom * 10 >= total
    capacity_failure = None
    if capacity_failures:
        capacity_failure = {
            "stage": "zero_episode",
            "reason": "process_memory_failure",
            "workers": capacity_failures,
        }
    elif not passed:
        capacity_failure = {
            "stage": "zero_episode",
            "reason": "insufficient_memory_headroom",
            "workers": [],
        }
    return {
        "passed": passed,
        "zero_episode_process_count": worker_count,
        "measurements_sha256": hashlib.sha256(raw_measurement).hexdigest(),
        "process_status_sha256": status_sha,
        "baseline_used_mib": baseline,
        "peak_used_mib": peak,
        "gpu_total_mib": total,
        "headroom_mib": headroom,
        "semantic_overrides": False,
        "capacity_failure": capacity_failure,
        "process_logs": log_rows,
    }


def assess_worker_memory(*, evidence_root: Path, worker_count: int) -> dict[str, object]:
    if worker_count not in {2, 4}:
        raise HarvestError("worker memory admission count must be two or four")
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_worker_memory_measurement_v1",
        "worker_count": worker_count,
        "memory_check": _derive_memory_check(evidence_root, worker_count=worker_count),
    }


def _runtime_admission_identity(
    manifest: Mapping[str, object], runtime_receipt: Path
) -> tuple[dict[str, object], str]:
    runtime, raw = _canonical_file(runtime_receipt, label="harvest runtime receipt")
    if (
        runtime.get("kind") != "lehome_public_n15_harvest_runtime_v1"
        or runtime.get("checkpoint_tree_sha256") != manifest["provenance"]["checkpoint_tree_sha256"]
        or runtime.get("semantic_environment") != {
            "policy_overrides": False,
            "checkpoint_compatibility_override": False,
            "cloth_fidelity_monitor": "observational_only",
        }
    ):
        raise HarvestError("worker admission runtime identity is invalid")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest["provenance"]["runtime_receipt_sha256"]:
        raise HarvestError("worker admission runtime receipt does not bind the manifest")
    return runtime, digest


def _derive_worker_admission(
    *, manifest: Mapping[str, object], runtime_receipt: Path,
    evidence_root: Path, worker_count: int,
) -> dict[str, object]:
    runtime, runtime_sha = _runtime_admission_identity(manifest, runtime_receipt)
    root = _safe_runtime_root(Path(evidence_root), label="worker admission evidence")
    memory = _derive_memory_check(root, worker_count=worker_count)
    smokes: list[dict[str, object]] = []
    failed_smokes: list[dict[str, object]] = []
    capacity_failure = memory["capacity_failure"]
    if memory["passed"]:
        status_sha, smoke_exit_codes = _status_tsv(
            root / "smoke-status.tsv", worker_count=worker_count,
            label="one-episode smoke status",
        )
        harvest_ids = {str(row["attempt_id"]) for row in manifest["attempts"]}
        harvest_seeds = {int(row["process_seed"]) for row in manifest["attempts"]}
        for scheduled, exit_code in zip(
            admission_smoke_schedule(worker_count), smoke_exit_codes, strict=True,
        ):
            worker_id = int(scheduled["worker_id"])
            smoke_root = root / "smokes" / f"worker-{worker_id}"
            smoke_file = smoke_root / "smoke-id.txt"
            try:
                observed_smoke_id = smoke_file.read_text(encoding="ascii")
            except (OSError, UnicodeError) as error:
                raise HarvestError("worker smoke identity evidence is unavailable") from error
            if (
                observed_smoke_id != f"{scheduled['smoke_id']}\n"
                or scheduled["smoke_id"] in harvest_ids
                or scheduled["process_seed"] in harvest_seeds
            ):
                raise HarvestError("worker smoke identity overlaps or mismatches the harvest")
            log = smoke_root / "evaluator.log"
            if log.is_symlink() or not log.is_file():
                raise HarvestError("worker smoke log is unavailable")
            raw_log = log.read_bytes()
            try:
                text = raw_log.decode("utf-8")
            except UnicodeError as error:
                raise HarvestError("worker smoke log is malformed") from error
            if exit_code:
                if _CAPACITY_FAILURE_LOG.search(text) is None:
                    raise HarvestError("worker smoke failed for a non-capacity reason")
                failed_smokes.append({
                    "worker_id": worker_id,
                    "smoke_id": scheduled["smoke_id"],
                    "process_seed": scheduled["process_seed"],
                    "exit_code": exit_code,
                    "evaluator_log_sha256": hashlib.sha256(raw_log).hexdigest(),
                })
                continue
            match = re.search(
                r"Episode\s+1/1:\s+Return=([-+0-9.eE]+),\s+Length=(\d+),\s+Success=(True|False)",
                text,
            )
            if _FORBIDDEN_LOG.search(text) or match is None or text.count(_COMPLETION_LOG) != 1:
                raise HarvestError("worker smoke did not produce one official native outcome")
            fidelity_path = smoke_root / "cloth-fidelity.jsonl"
            try:
                from rollout_appliance.native_reference_site.cloth_fidelity import (
                    validate_cloth_fidelity_evidence,
                )
                fidelity = validate_cloth_fidelity_evidence(
                    fidelity_path, expected_episodes=[(str(scheduled["garment"]), 1)]
                )
            except (ImportError, OSError, ValueError) as error:
                raise HarvestError("worker smoke cloth evidence is invalid") from error
            if fidelity["measured_episode_count"] != 1 or fidelity["fidelity_invalid_count"] != 0:
                raise HarvestError("worker smoke cloth fidelity failed")
            smokes.append({
                "worker_id": worker_id,
                "categories": _ASSIGNMENTS[worker_count][str(worker_id)],
                "smoke_id": scheduled["smoke_id"],
                "purpose": "capacity_admission_only",
                "category": scheduled["category"],
                "garment": scheduled["garment"],
                "process_seed": scheduled["process_seed"],
                "episode_index": 1,
                "official_outcome": "success" if match.group(3) == "True" else "policy_failure",
                "infrastructure_invalid": False,
                "cloth_fidelity": {"measured": True, "valid": True},
                "evaluator_log_sha256": hashlib.sha256(raw_log).hexdigest(),
                "cloth_fidelity_sha256": hashlib.sha256(fidelity_path.read_bytes()).hexdigest(),
                "checkpoint_tree_sha256": runtime["checkpoint_tree_sha256"],
                "runtime_receipt_sha256": runtime_sha,
            })
        memory["smoke_process_status_sha256"] = status_sha
        if failed_smokes:
            capacity_failure = {
                "stage": "one_episode_smoke",
                "reason": "process_memory_failure",
                "workers": failed_smokes,
            }
    passed = memory["passed"] and not failed_smokes
    return {
        "schema_version": 2,
        "kind": "lehome_public_n15_worker_admission_v2",
        "worker_count": worker_count,
        "runtime_receipt_sha256": runtime_sha,
        "checkpoint_tree_sha256": runtime["checkpoint_tree_sha256"],
        "passed": passed,
        "capacity_failure": capacity_failure,
        "memory_check": memory,
        "smokes": smokes,
        "failed_smokes": failed_smokes,
    }


def assess_worker_admission(
    *, manifest: object, runtime_receipt: Path, evidence_root: Path, worker_count: int,
) -> dict[str, object]:
    if worker_count not in {2, 4}:
        raise HarvestError("worker admission count must be two or four")
    return _derive_worker_admission(
        manifest=validate_manifest(manifest), runtime_receipt=runtime_receipt,
        evidence_root=evidence_root, worker_count=worker_count,
    )


def admit_workers(
    *, manifest: object, runtime_receipt: Path,
    four_worker_evidence_root: Path, two_worker_evidence_root: Path | None,
) -> dict[str, object]:
    verified = validate_manifest(manifest)
    four = _derive_worker_admission(
        manifest=verified, runtime_receipt=runtime_receipt,
        evidence_root=four_worker_evidence_root, worker_count=4,
    )
    if four["passed"]:
        count, fallback, admission, rejected = 4, False, four, None
    else:
        if two_worker_evidence_root is None:
            raise HarvestError("four-worker admission failed and two-worker fallback is missing")
        two = _derive_worker_admission(
            manifest=verified, runtime_receipt=runtime_receipt,
            evidence_root=two_worker_evidence_root, worker_count=2,
        )
        if not two["passed"]:
            raise HarvestError("two-worker fallback did not pass admission")
        count, fallback, admission, rejected = 2, True, two, four
    return {
        "schema_version": 2,
        "kind": "lehome_public_n15_worker_selection_v2",
        "worker_count": count,
        "fallback_from_four": fallback,
        "assignments": _ASSIGNMENTS[count],
        "checkpoint_tree_sha256": verified["provenance"]["checkpoint_tree_sha256"],
        "admission": admission,
        "rejected_four_worker_admission": rejected,
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
    config = inspected[0].get("Config") if isinstance(inspected, list) and inspected else None
    baked_environment = config.get("Env") if isinstance(config, Mapping) else None
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
        or not isinstance(baked_environment, list)
        or any(
            not isinstance(item, str)
            or item.split("=", 1)[0] in _FORBIDDEN_IMAGE_ENV
            for item in baked_environment
        )
    ):
        raise HarvestError("rollout image identity or semantic environment is invalid")
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


def _native_process_log(
    process_root: Path, *, episode_count: int,
) -> tuple[bytes, list[re.Match[str]]]:
    log = process_root / "evaluator.log"
    if log.is_symlink() or not log.is_file():
        raise HarvestError("native evaluator log is unavailable")
    try:
        raw = log.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise HarvestError("native evaluator log is malformed") from error
    pattern = re.compile(
        rf"Episode\s+(?P<episode>\d+)/{episode_count}:\s+Return=(?P<return>[-+0-9.eE]+),\s+"
        r"Length=(?P<length>\d+),\s+Success=(?P<success>True|False)"
    )
    matches = list(pattern.finditer(text))
    if (
        _FORBIDDEN_LOG.search(text)
        or text.count(_COMPLETION_LOG) != 1
        or len(matches) != episode_count
        or [int(match.group("episode")) for match in matches]
        != list(range(1, episode_count + 1))
    ):
        raise HarvestError("native evaluator log does not contain exact official outcomes")
    return raw, matches


def _default_parquet_episode_reader(paths: Sequence[Path]) -> list[int]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise HarvestError("PyArrow is required to authenticate success datasets") from error
    episodes: list[int] = []
    for path in paths:
        try:
            table = parquet.read_table(path, columns=["episode_index"])
            episodes.extend(int(value) for value in table.column("episode_index").to_pylist())
        except Exception as error:
            raise HarvestError("success dataset Parquet is corrupt") from error
    return episodes


def inspect_success_datasets(
    *, manifest: object, harvest_root: Path, expected_attempt_count: int,
    parquet_episode_reader: Callable[[Sequence[Path]], Sequence[int]] | None = None,
) -> dict[str, object]:
    """Authenticate every native successful episode dataset and its attempt binding."""

    verified = validate_manifest(manifest)
    root = _safe_runtime_root(Path(harvest_root), label="harvest root")
    reader = parquet_episode_reader or _default_parquet_episode_reader
    process_receipts: list[dict[str, object]] = []
    all_success_ids: list[str] = []
    for expected_row in _expected_processes(verified, expected_attempt_count):
        process_id = (
            f"{expected_row['category']}-{expected_row['garment_index']:02d}-"
            f"s{expected_row['process_seed']:06d}"
        )
        process_root = root / "processes" / process_id
        _, matches = _native_process_log(process_root, episode_count=25)
        attempts = [
            row for row in verified["attempts"][:expected_attempt_count]
            if row["category"] == expected_row["category"]
            and row["garment"] == expected_row["garment"]
            and row["process_seed"] == expected_row["process_seed"]
        ]
        successful = [
            row for row, match in zip(attempts, matches, strict=True)
            if match.group("success") == "True"
        ]
        dataset_parent = process_root / "dataset"
        if dataset_parent.is_symlink() or not dataset_parent.is_dir():
            raise HarvestError("successful native process dataset root is unavailable")
        children = sorted(dataset_parent.iterdir())
        if len(children) != 1 or children[0].name != "001" or not children[0].is_dir() or children[0].is_symlink():
            raise HarvestError("native process has missing or extra success datasets")
        dataset = children[0]
        success_count = len(successful)
        try:
            info = json.loads((dataset / "meta/info.json").read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise HarvestError("success dataset metadata is malformed") from error
        garment_info_path = dataset / "meta/garment_info.json"
        if garment_info_path.is_symlink():
            raise HarvestError("success dataset metadata is malformed")
        if garment_info_path.is_file():
            try:
                garment_info = json.loads(garment_info_path.read_bytes())
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise HarvestError("success dataset metadata is malformed") from error
        else:
            if success_count:
                raise HarvestError("successful dataset garment metadata is missing")
            for artifact_name in ("data", "videos"):
                artifact_root = dataset / artifact_name
                if artifact_root.is_symlink() or (
                    artifact_root.exists()
                    and (
                        not artifact_root.is_dir()
                        or any(artifact_root.rglob("*"))
                    )
                ):
                    raise HarvestError("zero-success dataset contains episode artifacts")
            garment_info = {}
        if not isinstance(garment_info, Mapping):
            raise HarvestError("success dataset metadata does not bind successful attempts")
        expected_garment_info = (
            {
                str(expected_row["garment"]): {
                    str(index): garment_info.get(str(expected_row["garment"]), {}).get(str(index))
                    for index in range(success_count)
                }
            }
            if success_count
            else {}
        )
        if (
            not isinstance(info, Mapping)
            or info.get("total_episodes") != success_count
            or type(info.get("total_frames")) is not int
            or info["total_frames"] < 0
            or (success_count > 0 and info["total_frames"] <= 0)
            or garment_info != expected_garment_info
            or any(
                not isinstance(value, Mapping)
                for value in garment_info.get(str(expected_row["garment"]), {}).values()
            )
        ):
            raise HarvestError("success dataset metadata does not bind successful attempts")
        parquet_paths = sorted((dataset / "data").rglob("*.parquet"))
        if success_count and not parquet_paths:
            raise HarvestError("successful episode Parquet artifacts are missing")
        if any(path.is_symlink() or not path.is_file() for path in parquet_paths):
            raise HarvestError("success dataset contains unsafe Parquet artifacts")
        try:
            episode_rows = [int(value) for value in reader(parquet_paths)]
        except HarvestError:
            raise
        except Exception as error:
            raise HarvestError("success dataset Parquet is corrupt") from error
        if (
            (success_count > 0 and not episode_rows)
            or set(episode_rows) != set(range(success_count))
            or any(episode_rows.count(index) < 1 for index in range(success_count))
        ):
            raise HarvestError("success dataset episode indices do not match official successes")
        entries = _bundle_entries(dataset)
        tree_sha = _entries_tree(entries)
        bindings = [
            {"attempt_id": row["attempt_id"], "dataset_episode_index": index}
            for index, row in enumerate(successful)
        ]
        all_success_ids.extend(row["attempt_id"] for row in successful)
        process_receipts.append({
            "process_id": process_id,
            "garment": expected_row["garment"],
            "process_seed": expected_row["process_seed"],
            "dataset_relative_path": f"processes/{process_id}/dataset/001",
            "dataset_file_count": len(entries),
            "dataset_tree_sha256": tree_sha,
            "successful_attempts": bindings,
        })
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_success_datasets_v1",
        "manifest_sha256": _digest(verified),
        "attempt_count": expected_attempt_count,
        "process_count": len(process_receipts),
        "success_count": len(all_success_ids),
        "successful_attempt_ids": all_success_ids,
        "processes": process_receipts,
    }


def validate_success_dataset_receipt(
    value: object, *, manifest: object, harvest_root: Path, expected_attempt_count: int,
) -> dict[str, object]:
    verified = validate_manifest(manifest)
    expected = _require_exact_mapping(
        value,
        {
            "schema_version", "kind", "manifest_sha256", "attempt_count",
            "process_count", "success_count", "successful_attempt_ids", "processes",
        },
        "success dataset receipt",
    )
    root = _safe_runtime_root(Path(harvest_root), label="harvest root")
    processes = expected["processes"]
    if (
        expected["schema_version"] != 1
        or expected["kind"] != "lehome_public_n15_success_datasets_v1"
        or expected["manifest_sha256"] != _digest(verified)
        or expected["attempt_count"] != expected_attempt_count
        or expected["process_count"] != (4 if expected_attempt_count == 100 else 40)
        or not isinstance(processes, list)
        or len(processes) != expected["process_count"]
    ):
        raise HarvestError("success dataset receipt summary is invalid")
    bound_ids: list[str] = []
    expected_processes = _expected_processes(verified, expected_attempt_count)
    for raw, expected_process in zip(processes, expected_processes, strict=True):
        row = _require_exact_mapping(
            raw,
            {
                "process_id", "garment", "process_seed", "dataset_relative_path",
                "dataset_file_count", "dataset_tree_sha256", "successful_attempts",
            },
            "success dataset process receipt",
        )
        process_id = (
            f"{expected_process['category']}-{expected_process['garment_index']:02d}-"
            f"s{expected_process['process_seed']:06d}"
        )
        expected_relative = f"processes/{process_id}/dataset/001"
        if (
            row["process_id"] != process_id
            or row["garment"] != expected_process["garment"]
            or row["process_seed"] != expected_process["process_seed"]
            or row["dataset_relative_path"] != expected_relative
            or type(row["dataset_file_count"]) is not int
            or row["dataset_file_count"] < 1
            or not isinstance(row["dataset_tree_sha256"], str)
            or _SHA256.fullmatch(row["dataset_tree_sha256"]) is None
        ):
            raise HarvestError("success dataset process identity is invalid")
        relative = PurePosixPath(str(row["dataset_relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise HarvestError("success dataset receipt path is unsafe")
        dataset = root / relative
        entries = _bundle_entries(dataset)
        if row["dataset_file_count"] != len(entries) or row["dataset_tree_sha256"] != _entries_tree(entries):
            raise HarvestError("success dataset byte tree changed after inspection")
        bindings = row["successful_attempts"]
        if not isinstance(bindings, list):
            raise HarvestError("success dataset attempt bindings are invalid")
        _, matches = _native_process_log(root / "processes" / process_id, episode_count=25)
        process_attempts = [
            attempt for attempt in verified["attempts"][:expected_attempt_count]
            if attempt["category"] == expected_process["category"]
            and attempt["garment"] == expected_process["garment"]
            and attempt["process_seed"] == expected_process["process_seed"]
        ]
        expected_success_attempts = [
            attempt
            for attempt, match in zip(process_attempts, matches, strict=True)
            if match.group("success") == "True"
        ]
        expected_bindings = [
            {"attempt_id": attempt["attempt_id"], "dataset_episode_index": dataset_index}
            for dataset_index, attempt in enumerate(expected_success_attempts)
        ]
        if bindings != expected_bindings:
            raise HarvestError("success dataset artifact is not bound to its official process outcome")
        for index, binding in enumerate(bindings):
            mapped = _require_exact_mapping(
                binding, {"attempt_id", "dataset_episode_index"},
                "success dataset attempt binding",
            )
            if mapped["dataset_episode_index"] != index:
                raise HarvestError("success dataset episode binding is not ordered")
            bound_ids.append(str(mapped["attempt_id"]))
    if (
        expected["successful_attempt_ids"] != bound_ids
        or expected["success_count"] != len(bound_ids)
        or len(bound_ids) != len(set(bound_ids))
    ):
        raise HarvestError("success dataset attempts are missing, extra, or duplicated")
    return expected


def collect_native_outcomes(
    *,
    manifest: object,
    process_status: object,
    harvest_root: Path,
    expected_attempt_count: int,
    success_dataset_receipt: object,
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
        _, matches = _native_process_log(process_root, episode_count=25)
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
    datasets = validate_success_dataset_receipt(
        success_dataset_receipt, manifest=verified, harvest_root=root,
        expected_attempt_count=expected_attempt_count,
    )
    official_success_ids = [
        row["attempt_id"] for row in outcomes if row["official_outcome"] == "success"
    ]
    if datasets["successful_attempt_ids"] != official_success_ids:
        raise HarvestError("success datasets do not match official successful attempts")
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
        "success_dataset_receipt_sha256": _digest(datasets),
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
            "fidelity_invalid_count", "success_dataset_receipt_sha256", "outcomes",
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
        or not isinstance(result["success_dataset_receipt_sha256"], str)
        or _SHA256.fullmatch(result["success_dataset_receipt_sha256"]) is None
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
        {
            "schema_version", "kind", "worker_count", "fallback_from_four",
            "assignments", "checkpoint_tree_sha256", "admission",
            "rejected_four_worker_admission",
        },
        "worker selection",
    )
    count = selection["worker_count"]
    if (
        selection["schema_version"] != 2
        or selection["kind"] != "lehome_public_n15_worker_selection_v2"
        or count not in {2, 4}
        or selection["assignments"] != _ASSIGNMENTS[count]
        or selection["checkpoint_tree_sha256"] != verified["provenance"]["checkpoint_tree_sha256"]
        or selection["fallback_from_four"] is not (count == 2)
        or not isinstance(selection["admission"], Mapping)
        or selection["admission"].get("worker_count") != count
        or selection["admission"].get("passed") is not True
        or selection["admission"].get("checkpoint_tree_sha256")
        != verified["provenance"]["checkpoint_tree_sha256"]
        or (
            count == 4
            and selection["rejected_four_worker_admission"] is not None
        )
        or (
            count == 2
            and (
                not isinstance(selection["rejected_four_worker_admission"], Mapping)
                or selection["rejected_four_worker_admission"].get("passed") is not False
                or selection["rejected_four_worker_admission"].get("worker_count") != 4
            )
        )
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
        if (
            not isinstance(disk, Mapping)
            or "existing_disk" not in disk
            or set(disk) - {"existing_disk", "attach_mode", "device_id"}
            or not isinstance(existing, Mapping)
            or set(existing) != {"id"}
            or ("attach_mode" in disk and disk["attach_mode"] != "READ_WRITE")
            or ("device_id" in disk and disk["device_id"] != "lehome")
        ):
            raise HarvestError("provider protected disk attachment is invalid")
        found.append(existing.get("id"))
    if (
        metadata.get("id") != CONTRACT.vm_id
        or metadata.get("name") != "lehome-rollout"
        or status.get("state") != "STOPPED"
        or found != [CONTRACT.disk_id]
    ):
        raise HarvestError("provider response is not the exact stopped rollout VM/disk attachment")
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
