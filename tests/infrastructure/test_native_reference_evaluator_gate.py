"""Offline contract tests for the isolated native public-reference gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import hashlib

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "rollout_appliance" / "run_native_reference_evaluator_gate.sh"
RUNBOOK = ROOT / "docs" / "experiments" / "2026-08-28-native-reference-evaluator-gate-runbook.md"


def _identity() -> dict[str, object]:
    return {
        "source_repository": "theo-zhou/lehome-groot-submission-4",
        "source_revision": "d384fe00508acd96ab1c3c5dc265e08261f94b3b",
        "source_tree_sha256": "eada9f80b0dda1428177fe4551efa8059fe85845d4db5b32bb673f88a50c6bb2",
        "checkpoint_tree_sha256": "b" * 64,
        "metadata_tree_sha256": "c" * 64,
        "assets_tree_sha256": "d" * 64,
        "cache_trust_manifest_sha256": "e" * 64,
        "lerobot_version": "0.4.3",
        "policy_class": "scripts.eval_policy.lerobot_policy.LeRobotPolicy",
        "policy_device": "cuda:0",
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_runtime": "12.8",
        "vm_id": "computeinstance-u00t6xfqhadrcmssa2",
        "disk_id": "computedisk-u00pbe55crxy7jr56x",
        "image": "ghcr.io/example/lehome@sha256:" + "f" * 64,
        "simulator_device": "cpu",
        "task_description": "fold the garment on the table",
        "action_horizon": 16,
        "action_dimension": 12,
        "success_checker": "pinned_raw_success_distance_second_mesh_points",
    }


def _attempts() -> list[dict[str, object]]:
    from scripts.verify_native_reference_evaluator_gate import oracle_attempts

    return [
        {
            **row,
            "success": row["expected_success"],
            "videos": [_artifact(f"videos/{row['attempt_id']}.mp4")],
            "log": _artifact(f"logs/{row['attempt_id']}.log"),
            "receipt": _artifact(f"receipts/{row['attempt_id']}.json"),
        }
        for row in oracle_attempts()
    ]


def _bundle(*, attempts: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": "lehome_native_reference_execution_result_v2",
        "identity": _identity(),
        "attempts": _attempts() if attempts is None else attempts,
    }


def _artifact(path: str) -> dict[str, object]:
    return {"path": path, "size": 1, "sha256": "f" * 64}


def _materialize_artifacts(root: Path, bundle: dict[str, object]) -> None:
    for attempt in bundle["attempts"]:  # type: ignore[index]
        for artifact in [attempt["log"], attempt["receipt"], *attempt["videos"]]:  # type: ignore[index]
            path = root / artifact["path"]  # type: ignore[index]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
            artifact.update({"size": 1, "sha256": hashlib.sha256(b"x").hexdigest()})  # type: ignore[index]


def test_native_oracle_accepts_exact_seven_of_eight_with_one_identity() -> None:
    from scripts.verify_native_reference_evaluator_gate import verify_native_reference_result

    receipt = verify_native_reference_result(_bundle())

    assert receipt["status"] == "oracle_matched_pending_finalization"
    assert receipt["successes"] == 7
    assert receipt["attempt_count"] == 8
    assert receipt["oracle_vector"] == [True, True, True, True, True, True, False, True]
    assert receipt["identity"] == _identity()


def test_native_oracle_emits_typed_fail_fast_stop_after_top_long_admission_miss() -> None:
    from scripts.verify_native_reference_evaluator_gate import verify_native_reference_result

    attempts = _attempts()[:2]
    attempts[1]["success"] = False
    receipt = verify_native_reference_result(_bundle(attempts=attempts))

    assert receipt["status"] == "evaluator_compatibility_stop"
    assert receipt["reason"] == "top_long_admission_failed"
    assert receipt["attempt_count"] == 2


def test_native_oracle_rejects_identity_drift_and_defers_fidelity_to_a_bound_review() -> None:
    from scripts.verify_native_reference_evaluator_gate import NativeReferenceGateError, verify_native_reference_result

    drifted = _bundle()
    drifted["identity"] = {**_identity(), "lerobot_version": "0.5.0"}
    with pytest.raises(NativeReferenceGateError, match="LeRobot"):
        verify_native_reference_result(drifted)


def test_native_gate_cli_writes_immutable_receipt(tmp_path: Path) -> None:
    from scripts.verify_native_reference_evaluator_gate import main

    bundle = tmp_path / "result.json"
    output = tmp_path / "receipt.json"
    document = _bundle()
    _materialize_artifacts(tmp_path, document)
    bundle.write_text(json.dumps(document), encoding="utf-8")

    assert main(["verify-execution", "--result", str(bundle), "--bundle-root", str(tmp_path), "--receipt", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "oracle_matched_pending_finalization"
    with pytest.raises(SystemExit, match="already exists"):
        main(["verify-execution", "--result", str(bundle), "--bundle-root", str(tmp_path), "--receipt", str(output)])


def _launcher_environment(tmp_path: Path) -> dict[str, str]:
    output = tmp_path / "native-reference-202608290001"
    source = tmp_path / "source-cache"
    checkpoint = tmp_path / "checkpoint-cache"
    metadata = tmp_path / "metadata-cache"
    source.mkdir()
    checkpoint.mkdir()
    metadata.mkdir()
    return {
        **os.environ,
        "LEHOME_NATIVE_REFERENCE_VALIDATE_ONLY": "1",
        "LEHOME_NATIVE_REFERENCE_VM_ID": "computeinstance-u00t6xfqhadrcmssa2",
        "LEHOME_NATIVE_REFERENCE_DISK_ID": "computedisk-u00pbe55crxy7jr56x",
        "LEHOME_NATIVE_REFERENCE_IMAGE": "ghcr.io/example/lehome@sha256:" + "d" * 64,
        "LEHOME_NATIVE_REFERENCE_SOURCE_ROOT": str(source),
        "LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT": str(checkpoint),
        "LEHOME_NATIVE_REFERENCE_METADATA_ROOT": str(metadata),
        "LEHOME_NATIVE_REFERENCE_ASSETS_ROOT": str(source),
        "LEHOME_NATIVE_REFERENCE_OUTPUT_ROOT": str(output),
        "LEHOME_NATIVE_REFERENCE_SOURCE_TREE_SHA256": "eada9f80b0dda1428177fe4551efa8059fe85845d4db5b32bb673f88a50c6bb2",
        "LEHOME_NATIVE_REFERENCE_CHECKPOINT_TREE_SHA256": "b" * 64,
        "LEHOME_NATIVE_REFERENCE_METADATA_TREE_SHA256": "c" * 64,
        "LEHOME_NATIVE_REFERENCE_ASSETS_TREE_SHA256": "d" * 64,
    }


def test_native_launcher_validate_only_refuses_unprepared_caches_without_cloud_actions(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=_launcher_environment(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "native reference checkpoint cache is incomplete" in result.stderr
    assert not (tmp_path / "native-reference-202608290001").exists()


def test_native_launcher_isolated_contract_never_creates_resources_or_uses_n17_gateway() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "LEHOME_NATIVE_REFERENCE_VM_ID" in text
    assert "LEHOME_NATIVE_REFERENCE_DISK_ID" in text
    assert "LeRobotPolicy" in text
    assert "Top_Long_Seen_0" in text
    assert "Top_Short_Seen_0" in text
    assert "Pant_Long_Seen_0" in text
    assert "Pant_Short_Seen_0" in text
    assert "LEHOME_NATIVE_REFERENCE_VALIDATE_ONLY" in text
    assert "LEHOME_NATIVE_REFERENCE_MODE" in text
    assert "source-stage" in text
    assert "CACHE_TRUST_MANIFEST" in text
    assert "cache_trust_origin" in text
    assert "torch.cuda.is_available" in text
    assert "verify-execution" in text
    assert "compile-stage" in text
    assert "invalid_reason" not in text
    assert "GIT_LFS_SKIP_SMUDGE=1" in text
    assert "pretrained_model" in text
    assert "run_groot_persistent_worker.py" not in text
    assert "eval_groot_n17" not in text
    assert "nebius compute instance create" not in text
    assert "nebius compute instance start" not in text
    assert "docker build" not in text
    assert "filter.lfs.smudge" in text


def test_native_launcher_pins_every_checkpoint_file_and_requires_exact_set() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    for filename in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_2_groot_pack_inputs_v3.safetensors",
        "policy_postprocessor_step_0_groot_action_unpack_unnormalize_v1.safetensors",
        "train_config.json",
    ):
        assert filename in text
    assert "unexpected file set" in text


def test_native_stage_compiler_parses_real_public_episode_lines_and_binds_files(tmp_path: Path) -> None:
    from scripts.verify_native_reference_evaluator_gate import compile_native_stage

    root = tmp_path / "bundle"
    (root / "logs").mkdir(parents=True)
    (root / "videos" / "stage-1" / "success").mkdir(parents=True)
    log = root / "logs" / "stage-1.log"
    log.write_text("Episode 1/2: Return=1.00, Success=True\nEpisode 2/2: Return=2.00, Success=True\n", encoding="utf-8")
    for episode in (0, 1):
        (root / "videos" / "stage-1" / "success" / f"episode{episode}_observation_images_top_rgb.mp4").write_bytes(b"video")

    result = compile_native_stage(root, stage=1, category="top_long", garment="Top_Long_Seen_0", identity=_identity())

    assert [row["success"] for row in result["attempts"]] == [True, True]
    assert [row["log"]["path"] for row in result["attempts"]] == ["logs/stage-1.log", "logs/stage-1.log"]
    assert all(row["videos"][0]["size"] == 5 for row in result["attempts"])
    assert all((root / row["receipt"]["path"]).is_file() for row in result["attempts"])


def test_native_finalization_requires_fidelity_publication_and_stopped_vm_receipts() -> None:
    from scripts.verify_native_reference_evaluator_gate import NativeReferenceGateError, finalize_native_reference_gate, verify_native_reference_result

    execution = verify_native_reference_result(_bundle())
    with pytest.raises(NativeReferenceGateError, match="fidelity review"):
        finalize_native_reference_gate(execution, {}, {}, {})

    execution_sha = hashlib.sha256(json.dumps(execution, sort_keys=True, separators=(",", ":")).encode() + b"\n").hexdigest()
    fidelity = {
        "schema_version": 1, "kind": "lehome_native_reference_fidelity_review_v1", "execution_receipt_sha256": execution_sha,
        "review_method": "manual_video_audit", "attempts": [{"attempt_id": row["attempt_id"], "cloth_present": True, "cloth_flight": False, "nonfinite": False, "safety_failure": False, "evidence_sha256": "a" * 64} for row in _attempts()],
    }
    publication = {"schema_version": 1, "kind": "lehome_native_reference_hf_readback_v1", "execution_receipt_sha256": execution_sha, "immutable_revision": "b" * 40, "bundle_manifest_sha256": "c" * 64, "readback_verified": True}
    stopped = {"schema_version": 1, "kind": "lehome_native_reference_vm_stopped_v1", "execution_receipt_sha256": execution_sha, "vm_id": _identity()["vm_id"], "disk_id": _identity()["disk_id"], "image": _identity()["image"], "state": "STOPPED", "attached_disk_ids": [_identity()["disk_id"]]}

    final = finalize_native_reference_gate(execution, fidelity, publication, stopped)
    assert final["status"] == "passed"


def test_native_gate_runbook_preserves_the_no_collection_admission_boundary() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert "d384fe00508acd96ab1c3c5dc265e08261f94b3b" in text
    assert "Top_Long_Seen_0" in text
    assert "7/8" in text
    assert "Do not start collection or training" in text
    assert "Hugging Face" in text
