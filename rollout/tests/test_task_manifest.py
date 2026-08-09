from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from b1k_rollout.identity import BEHAVIOR_REVISION
from b1k_rollout.task_manifest import (
    CANONICAL_MANIFEST_SHA256,
    EVALUATOR_PATH,
    EVALUATOR_SHA256,
    GENERATOR_DOCUMENTATION_PATH,
    GENERATOR_DOCUMENTATION_SHA256,
    GENERATOR_VERSION,
    OFFICIAL_PUBLIC_TEST_INDICES,
    OFFICIAL_PROTOCOL_PATH,
    OFFICIAL_PROTOCOL_SHA256,
    PUBLIC_TEST_INSTANCE_COUNT,
    ROBOT_MODEL,
    SOURCE_REPOSITORY,
    TASK_DATA_PATH,
    TASK_DATA_SHA256,
    build_task_manifest,
    canonical_manifest_sha256,
    load_task_manifest,
)


def _task_data(path: Path, task_names: list[str]) -> Path:
    path.write_text(json.dumps({"tasks": [{"id": name} for name in task_names]}), encoding="utf-8")
    return path


def _official_task_names() -> list[str]:
    return [f"official_task_{index:03d}" for index in range(100)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_for_test(path: Path, **kwargs: object) -> dict[str, object]:
    return build_task_manifest(path, expected_task_data_sha256=_sha256(path), **kwargs)


def test_build_task_manifest_preserves_canonical_source_task_id_order(
    tmp_path: Path,
) -> None:
    task_data = _task_data(tmp_path / "task_data.json", list(reversed(_official_task_names())))

    manifest = _build_for_test(task_data)

    assert manifest["provenance"] == {
        "canonical_sha256": canonical_manifest_sha256(manifest),
        "evaluator_path": EVALUATOR_PATH,
        "evaluator_sha256": EVALUATOR_SHA256,
        "evaluation_protocol_path": OFFICIAL_PROTOCOL_PATH,
        "evaluation_protocol_sha256": OFFICIAL_PROTOCOL_SHA256,
        "generator_documentation_path": GENERATOR_DOCUMENTATION_PATH,
        "generator_documentation_sha256": GENERATOR_DOCUMENTATION_SHA256,
        "generator_version": GENERATOR_VERSION,
        "official_reporting_indices": list(OFFICIAL_PUBLIC_TEST_INDICES),
        "public_test_instance_count": PUBLIC_TEST_INSTANCE_COUNT,
        "robot_model": ROBOT_MODEL,
        "source_commit": BEHAVIOR_REVISION,
        "source_repository": SOURCE_REPOSITORY,
        "task_count": 100,
        "task_data_path": TASK_DATA_PATH,
        "task_data_sha256": _sha256(task_data),
    }
    assert [task["task_name"] for task in manifest["tasks"]] == list(
        reversed(_official_task_names())
    )
    assert [task["source_task_id"] for task in manifest["tasks"]] == list(range(100))
    assert all(
        task["requested_instances"]
        == [{"index": index, "mode": "public_test"} for index in OFFICIAL_PUBLIC_TEST_INDICES]
        for task in manifest["tasks"]
    )


@pytest.mark.parametrize(
    ("task_names", "requested_instances", "message"),
    [
        (_official_task_names()[:-1], None, "exactly 100"),
        (_official_task_names()[:-1] + ["official_task_098"], None, "unique"),
        (_official_task_names(), [{"mode": "hidden_test", "index": 0}], "train or public_test"),
        (_official_task_names(), [{"mode": "public_test", "index": 20}], "range 0..19"),
        (_official_task_names(), [{"mode": "train", "index": -1}], "non-negative"),
    ],
)
def test_build_task_manifest_fails_closed_for_invalid_upstream_or_requests(
    tmp_path: Path,
    task_names: list[str],
    requested_instances: list[dict[str, object]] | None,
    message: str,
) -> None:
    task_data = _task_data(tmp_path / "task_data.json", task_names)

    with pytest.raises(ValueError, match=message):
        _build_for_test(task_data, requested_instances=requested_instances)


def test_build_task_manifest_rejects_a_source_hash_mismatch_or_stale_json(tmp_path: Path) -> None:
    task_data = _task_data(tmp_path / "task_data.json", _official_task_names())
    expected_sha256 = _sha256(task_data)

    with pytest.raises(ValueError, match="source SHA-256"):
        build_task_manifest(task_data)
    task_data.write_text(json.dumps({"tasks": [{"id": name} for name in reversed(_official_task_names())]}), encoding="utf-8")
    with pytest.raises(ValueError, match="source SHA-256"):
        build_task_manifest(task_data, expected_task_data_sha256=expected_sha256)


def test_load_task_manifest_rejects_a_stale_canonical_hash(tmp_path: Path) -> None:
    committed = Path(__file__).parents[1] / "task-manifest.json"
    manifest = json.loads(committed.read_text(encoding="utf-8"))
    manifest["provenance"]["canonical_sha256"] = "0" * 64
    path = tmp_path / "task-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical SHA-256"):
        load_task_manifest(path)


@pytest.mark.parametrize("forgery", ["source_hash", "source_id", "reorder"])
def test_load_task_manifest_rejects_rehashed_source_provenance_or_b100_order(
    tmp_path: Path, forgery: str
) -> None:
    committed = Path(__file__).parents[1] / "task-manifest.json"
    manifest = json.loads(committed.read_text(encoding="utf-8"))
    if forgery == "source_hash":
        manifest["provenance"]["task_data_sha256"] = "0" * 64
        message = "task data source"
    elif forgery == "source_id":
        manifest["tasks"][50]["source_task_id"] = 100
        message = "source task ids"
    else:
        manifest["tasks"][0], manifest["tasks"][1] = manifest["tasks"][1], manifest["tasks"][0]
        message = "source task ids"
    manifest["provenance"]["canonical_sha256"] = canonical_manifest_sha256(manifest)
    path = tmp_path / "forged-task-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_task_manifest(path)


@pytest.mark.parametrize(
    "forgery",
    [
        "task_name_mutation",
        "task_name_swap",
        "mode",
        "missing_index",
        "substituted_index",
        "credential_name",
        "lehome_name",
    ],
)
def test_public_load_rejects_every_rehashed_campaign_forgery(
    tmp_path: Path, forgery: str
) -> None:
    committed = Path(__file__).parents[1] / "task-manifest.json"
    manifest = json.loads(committed.read_text(encoding="utf-8"))
    requests = manifest["tasks"][0]["requested_instances"]
    if forgery == "task_name_mutation":
        manifest["tasks"][0]["task_name"] = "forged_task_name"
    elif forgery == "task_name_swap":
        manifest["tasks"][0]["task_name"], manifest["tasks"][1]["task_name"] = (
            manifest["tasks"][1]["task_name"],
            manifest["tasks"][0]["task_name"],
        )
    elif forgery == "mode":
        requests[0]["mode"] = "train"
    elif forgery == "missing_index":
        requests.pop()
    elif forgery == "substituted_index":
        requests[-1]["index"] = 19
    elif forgery == "credential_name":
        manifest["tasks"][0]["task_name"] = "ghp_" + "a" * 36
    else:
        manifest["tasks"][0]["task_name"] = "lehome_task"
    manifest["provenance"]["canonical_sha256"] = canonical_manifest_sha256(manifest)
    path = tmp_path / "forged-public-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        load_task_manifest(path)


def test_committed_manifest_is_an_exact_verified_b100_public_test_campaign() -> None:
    manifest_path = Path(__file__).parents[1] / "task-manifest.json"

    manifest = load_task_manifest(manifest_path)

    assert manifest["provenance"]["source_repository"] == SOURCE_REPOSITORY
    assert manifest["provenance"]["source_commit"] == BEHAVIOR_REVISION
    assert manifest["provenance"]["task_data_path"] == TASK_DATA_PATH
    assert manifest["provenance"]["task_data_sha256"] == TASK_DATA_SHA256
    assert manifest["provenance"]["generator_documentation_path"] == GENERATOR_DOCUMENTATION_PATH
    assert manifest["provenance"]["generator_documentation_sha256"] == GENERATOR_DOCUMENTATION_SHA256
    assert manifest["provenance"]["evaluator_path"] == EVALUATOR_PATH
    assert manifest["provenance"]["evaluator_sha256"] == EVALUATOR_SHA256
    assert manifest["provenance"]["evaluation_protocol_path"] == OFFICIAL_PROTOCOL_PATH
    assert manifest["provenance"]["evaluation_protocol_sha256"] == OFFICIAL_PROTOCOL_SHA256
    assert manifest["provenance"]["robot_model"] == "R1Pro"
    assert manifest["provenance"]["public_test_instance_count"] == 20
    assert manifest["provenance"]["official_reporting_indices"] == list(range(10))
    assert manifest["provenance"]["generator_version"] == GENERATOR_VERSION
    assert manifest["provenance"]["task_count"] == 100
    assert manifest["provenance"]["canonical_sha256"] == CANONICAL_MANIFEST_SHA256
    assert manifest["provenance"]["canonical_sha256"] == canonical_manifest_sha256(manifest)
    assert len(manifest["tasks"]) == 100
    assert len({task["task_name"] for task in manifest["tasks"]}) == 100
    assert [task["source_task_id"] for task in manifest["tasks"]] == list(range(100))
    assert all(
        task["requested_instances"] == [{"index": index, "mode": "public_test"} for index in range(10)]
        for task in manifest["tasks"]
    )


def test_committed_manifest_generator_exposes_no_noncanonical_campaign_options() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "build-task-manifest.py"
    ).read_text(encoding="utf-8")

    assert '"--mode"' not in script
    assert '"--instance-index"' not in script
