"""Build and validate the immutable BEHAVIOR-1K B100 rollout task manifest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from b1k_rollout.identity import (
    BEHAVIOR_REVISION,
    canonical_json_sha256,
    reject_credential_material,
)


SOURCE_REPOSITORY = "StanfordVL/BEHAVIOR-1K"
TASK_DATA_PATH = "docs/challenge/task_data.json"
TASK_DATA_SHA256 = "9bea32b2860a36ac11f9425c944c16b260a1b71e649389e0097169b117e4e8e7"
GENERATOR_DOCUMENTATION_PATH = "docs/gen_2026_task_data.py"
GENERATOR_DOCUMENTATION_SHA256 = "6fbd9434d6d1a9ecc3f64d5081fa522bc98d6363b5f65e39cd454c59aaad968d"
OFFICIAL_PROTOCOL_PATH = "docs/challenge/evaluation.md"
OFFICIAL_PROTOCOL_SHA256 = "39c92a3c206d950b4580831c32992dd79374245df9298e0da0b9d468278db260"
EVALUATOR_PATH = "OmniGibson/omnigibson/eval/evaluator.py"
EVALUATOR_SHA256 = "eb121fff3214107ec08c9eb0428b7d898ab7972b197b77ea1a873383d844b288"
GENERATOR_VERSION = "3"
MANIFEST_SCHEMA_VERSION = 1
EXPECTED_TASK_COUNT = 100
ROBOT_MODEL = "R1Pro"
_SUPPORTED_MODES = frozenset(("train", "public_test"))
PUBLIC_TEST_INSTANCE_COUNT = 20
OFFICIAL_PUBLIC_TEST_INDICES = tuple(range(10))
CANONICAL_MANIFEST_SHA256 = "7ab5ee6ef1c5e48b421f4dac6ef45537f081b78feba0735ae1c805f529462d92"
_MANIFEST_FIELDS = frozenset(("schema_version", "provenance", "tasks"))
_PROVENANCE_FIELDS = frozenset(
    (
        "source_repository",
        "source_commit",
        "task_data_path",
        "task_data_sha256",
        "robot_model",
        "generator_documentation_path",
        "generator_documentation_sha256",
        "evaluation_protocol_path",
        "evaluation_protocol_sha256",
        "evaluator_path",
        "evaluator_sha256",
        "public_test_instance_count",
        "official_reporting_indices",
        "generator_version",
        "task_count",
        "canonical_sha256",
    )
)
_TASK_FIELDS = frozenset(("task_name", "source_task_id", "requested_instances"))
_REQUEST_FIELDS = frozenset(("mode", "index"))


def build_task_manifest(
    task_data_path: Path,
    *,
    requested_instances: Sequence[Mapping[str, object]] | None = None,
    expected_task_data_sha256: str = TASK_DATA_SHA256,
) -> dict[str, object]:
    """Derive the complete B100 manifest from pinned committed task data.

    Upstream's committed generator documents this pinned JSON as the canonical
    B100 task export and source order. Generator, protocol, and evaluator hashes
    are retained as supporting evidence; no unavailable runtime CSV is claimed
    as task-source provenance.
    """

    requests = _validated_requests(requested_instances)
    source_bytes = _read_task_data(task_data_path)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != expected_task_data_sha256:
        raise ValueError("task data source SHA-256 is stale or does not match the pinned source")
    source_tasks = _load_source_tasks(source_bytes)
    if len(source_tasks) != EXPECTED_TASK_COUNT:
        raise ValueError(
            f"official committed task data must contain exactly {EXPECTED_TASK_COUNT} unique tasks"
        )

    tasks = [
        {
            "task_name": task_name,
            "source_task_id": source_task_id,
            "requested_instances": [dict(request) for request in requests],
        }
        for task_name, source_task_id in source_tasks.items()
    ]
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "provenance": {
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": BEHAVIOR_REVISION,
            "task_data_path": TASK_DATA_PATH,
            "task_data_sha256": source_sha256,
            "robot_model": ROBOT_MODEL,
            "generator_documentation_path": GENERATOR_DOCUMENTATION_PATH,
            "generator_documentation_sha256": GENERATOR_DOCUMENTATION_SHA256,
            "evaluation_protocol_path": OFFICIAL_PROTOCOL_PATH,
            "evaluation_protocol_sha256": OFFICIAL_PROTOCOL_SHA256,
            "evaluator_path": EVALUATOR_PATH,
            "evaluator_sha256": EVALUATOR_SHA256,
            "public_test_instance_count": PUBLIC_TEST_INSTANCE_COUNT,
            "official_reporting_indices": list(OFFICIAL_PUBLIC_TEST_INDICES),
            "generator_version": GENERATOR_VERSION,
            "task_count": len(tasks),
        },
        "tasks": tasks,
    }
    provenance = _mapping(manifest["provenance"], label="provenance")
    manifest["provenance"] = {
        **provenance,
        "canonical_sha256": canonical_manifest_sha256(manifest),
    }
    _validate_task_manifest(manifest, expected_task_data_sha256=expected_task_data_sha256)
    return manifest


def canonical_manifest_sha256(manifest: Mapping[str, object]) -> str:
    """Hash the manifest without its self-referential provenance hash field."""

    payload = _manifest_without_hash(manifest)
    return canonical_json_sha256(payload)


def render_task_manifest(manifest: Mapping[str, object]) -> str:
    """Render a validated manifest deterministically for committing or comparison."""

    validate_task_manifest(manifest)
    return json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def load_task_manifest(path: Path) -> dict[str, object]:
    """Load and fail closed on a malformed or stale committed manifest."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"task manifest is unreadable: {path}") from error
    manifest = _mapping(raw, label="task manifest")
    validate_task_manifest(manifest)
    return manifest


def validate_task_manifest(manifest: Mapping[str, object]) -> None:
    """Check the immutable on-disk contract independent of source availability."""

    _validate_task_manifest(
        manifest,
        expected_task_data_sha256=TASK_DATA_SHA256,
        require_official_campaign=True,
        expected_manifest_sha256=CANONICAL_MANIFEST_SHA256,
    )


def _validate_task_manifest(
    manifest: Mapping[str, object],
    *,
    expected_task_data_sha256: str,
    require_official_campaign: bool = False,
    expected_manifest_sha256: str | None = None,
) -> None:
    """Validate a manifest against one explicit task-data identity."""

    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("task manifest fields are invalid")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("task manifest schema version is invalid")

    provenance = _mapping(manifest.get("provenance"), label="provenance")
    if set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("task manifest provenance fields are invalid")
    if provenance.get("source_repository") != SOURCE_REPOSITORY:
        raise ValueError("task manifest source repository is invalid")
    if provenance.get("source_commit") != BEHAVIOR_REVISION:
        raise ValueError("task manifest source commit is invalid")
    if provenance.get("task_data_path") != TASK_DATA_PATH:
        raise ValueError("task manifest task data path is invalid")
    if provenance.get("task_data_sha256") != expected_task_data_sha256:
        raise ValueError("task data source SHA-256 is invalid")
    if provenance.get("robot_model") != ROBOT_MODEL:
        raise ValueError("task manifest robot model is invalid")
    if provenance.get("generator_documentation_path") != GENERATOR_DOCUMENTATION_PATH:
        raise ValueError("task manifest generator documentation path is invalid")
    if provenance.get("generator_documentation_sha256") != GENERATOR_DOCUMENTATION_SHA256:
        raise ValueError("task manifest generator documentation hash is invalid")
    if provenance.get("evaluation_protocol_path") != OFFICIAL_PROTOCOL_PATH:
        raise ValueError("task manifest evaluation protocol path is invalid")
    if provenance.get("evaluation_protocol_sha256") != OFFICIAL_PROTOCOL_SHA256:
        raise ValueError("task manifest evaluation protocol hash is invalid")
    if provenance.get("evaluator_path") != EVALUATOR_PATH:
        raise ValueError("task manifest evaluator path is invalid")
    if provenance.get("evaluator_sha256") != EVALUATOR_SHA256:
        raise ValueError("task manifest evaluator hash is invalid")
    if provenance.get("public_test_instance_count") != PUBLIC_TEST_INSTANCE_COUNT:
        raise ValueError("task manifest public test instance count is invalid")
    if provenance.get("official_reporting_indices") != list(OFFICIAL_PUBLIC_TEST_INDICES):
        raise ValueError("task manifest official reporting indices are invalid")
    if provenance.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("task manifest generator version is invalid")
    if provenance.get("task_count") != EXPECTED_TASK_COUNT:
        raise ValueError("task manifest provenance must record exactly 100 tasks")

    tasks = _sequence(manifest.get("tasks"), label="tasks")
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise ValueError("task manifest must contain exactly 100 tasks")
    source_task_ids: list[int] = []
    names: set[str] = set()
    for task in tasks:
        item = _mapping(task, label="task")
        if set(item) != _TASK_FIELDS:
            raise ValueError("task manifest task fields are invalid")
        task_name = item.get("task_name")
        if not isinstance(task_name, str) or not task_name:
            raise ValueError("task name is invalid")
        reject_credential_material(task_name)
        if "lehome" in task_name.casefold():
            raise ValueError("task name must not reference LeHome")
        source_task_id = item.get("source_task_id")
        if type(source_task_id) is not int or source_task_id < 0:
            raise ValueError("source task id is invalid")
        source_task_ids.append(source_task_id)
        requests = _validated_requests(
            _sequence(item.get("requested_instances"), label="requested instances")
        )
        if require_official_campaign and list(requests) != _official_public_test_requests():
            raise ValueError("task requests must equal the official public_test indices 0..9")
        names.add(task_name)
    if len(names) != EXPECTED_TASK_COUNT:
        raise ValueError("task names must be unique")
    if source_task_ids != list(range(EXPECTED_TASK_COUNT)):
        raise ValueError("source task ids must be the exact canonical range 0..99 in order")

    expected_hash = canonical_manifest_sha256(manifest)
    if provenance.get("canonical_sha256") != expected_hash:
        raise ValueError("task manifest canonical SHA-256 is stale or invalid")
    if expected_manifest_sha256 is not None and expected_hash != expected_manifest_sha256:
        raise ValueError("task manifest is not the approved canonical campaign")


def _load_source_tasks(source_bytes: bytes) -> dict[str, int]:
    try:
        raw = json.loads(source_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("pinned task data is invalid JSON") from error
    source = _mapping(raw, label="pinned task data")
    if set(source) != {"tasks"}:
        raise ValueError("pinned task data must contain only the tasks list")
    raw_tasks = _sequence(source.get("tasks"), label="pinned task data tasks")
    tasks: dict[str, int] = {}
    for task_id, raw_task in enumerate(raw_tasks):
        task = _mapping(raw_task, label="pinned task data task")
        task_name = task.get("id")
        if not isinstance(task_name, str) or not task_name or task_name != task_name.strip():
            raise ValueError("pinned task data contains an invalid task id")
        if task_name in tasks:
            raise ValueError("pinned task data task names must be unique")
        tasks[task_name] = task_id
    return tasks


def _read_task_data(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"pinned task data is unreadable: {path}") from error


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} SHA-256 is invalid")


def _validated_requests(
    requested_instances: Sequence[Mapping[str, object]] | None,
) -> tuple[dict[str, object], ...]:
    if requested_instances is None:
        requested_instances = tuple(
            {"mode": "public_test", "index": index}
            for index in OFFICIAL_PUBLIC_TEST_INDICES
        )
    if not requested_instances:
        raise ValueError("at least one requested instance is required")
    normalized: list[dict[str, object]] = []
    for request in requested_instances:
        item = _mapping(request, label="requested instance")
        if set(item) != _REQUEST_FIELDS:
            raise ValueError("requested instance fields are invalid")
        mode = item.get("mode")
        index = item.get("index")
        if mode not in _SUPPORTED_MODES:
            raise ValueError("requested instance mode must be train or public_test")
        if type(index) is not int or index < 0:
            raise ValueError("requested instance index must be a non-negative integer")
        if mode == "public_test" and index >= PUBLIC_TEST_INSTANCE_COUNT:
            raise ValueError("public_test requested instance index must be in range 0..19")
        normalized.append({"mode": mode, "index": index})
    if len({(item["mode"], item["index"]) for item in normalized}) != len(normalized):
        raise ValueError("requested instances must be unique")
    return tuple(normalized)


def _official_public_test_requests() -> list[dict[str, object]]:
    return [
        {"mode": "public_test", "index": index}
        for index in OFFICIAL_PUBLIC_TEST_INDICES
    ]


def _manifest_without_hash(manifest: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = dict(manifest)
    provenance = _mapping(payload.get("provenance"), label="provenance")
    payload["provenance"] = {
        key: value for key, value in provenance.items() if key != "canonical_sha256"
    }
    return payload


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _sequence(value: object, *, label: str) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a list")
    return list(value)
