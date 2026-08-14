from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "checkpoint_evaluation_publish_under_test",
    REPOSITORY / "scripts" / "publish_groot_checkpoint_evaluation.py",
)
assert SPEC is not None and SPEC.loader is not None
PUBLISH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PUBLISH
SPEC.loader.exec_module(PUBLISH)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _invocation() -> dict[str, object]:
    return {
        "kind": "public_unseen_tops_checkpoint_evaluation", "policy_repo": "models",
        "policy_revision": "a" * 40, "policy_step": 1000,
        "policy_artifact_sha256": "b" * 64, "code_revision": "c" * 40,
    }


def _plan(invocation: dict[str, object]) -> dict[str, object]:
    return {
        "release_id": "d" * 64, "remote_prefix": "evaluations/groot-n17-step-1000/" + "d" * 64,
        "metrics": {"episodes": 40}, "invocation": invocation,
    }


def test_publication_receipt_binds_immutable_private_readback_to_instance_and_invocation(tmp_path: Path) -> None:
    invocation = _invocation()
    instance = _write(tmp_path / "instance.json", {
        "kind": "groot_checkpoint_evaluation_instance", "instance_id": 99,
        "invocation_sha256": PUBLISH._canonical(invocation),
    })
    receipt = PUBLISH.build_publication_receipt(_plan(invocation), immutable_revision="e" * 40, instance_receipt_path=instance)
    assert receipt == {
        "schema_version": 1, "kind": "groot_checkpoint_evaluation_publication",
        "repository": "ryanjin333/lehome-groot-n17-data", "repository_private": True,
        "immutable_revision": "e" * 40, "remote_prefix": _plan(invocation)["remote_prefix"],
        "release_id": "d" * 64, "evaluation_metrics": {"episodes": 40},
        "invocation": invocation, "invocation_sha256": PUBLISH._canonical(invocation),
        "instance_id": 99, "instance_receipt_sha256": hashlib.sha256(instance.read_bytes()).hexdigest(),
        "tree_listing_verified": True, "fresh_readback_verified": True, "disposable": True,
    }


@pytest.mark.parametrize("mutator", ["wrong_instance", "stale_invocation", "bad_commit", "missing_prefix"])
def test_publisher_rejects_mismatched_instance_invocation_or_readback_identity(tmp_path: Path, mutator: str) -> None:
    invocation = _invocation()
    instance_value = {"kind": "groot_checkpoint_evaluation_instance", "instance_id": 99, "invocation_sha256": PUBLISH._canonical(invocation)}
    if mutator == "wrong_instance":
        instance_value["instance_id"] = 0
    if mutator == "stale_invocation":
        instance_value["invocation_sha256"] = "f" * 64
    instance = _write(tmp_path / "instance.json", instance_value)
    plan = _plan(invocation)
    if mutator == "missing_prefix":
        plan["remote_prefix"] = ""
    commit = "not-a-commit" if mutator == "bad_commit" else "e" * 40
    with pytest.raises(ValueError, match="instance|invocation|readback"):
        PUBLISH.build_publication_receipt(plan, immutable_revision=commit, instance_receipt_path=instance)
