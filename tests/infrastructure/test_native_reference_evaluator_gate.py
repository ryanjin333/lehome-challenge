"""Offline contract tests for the isolated native public-reference gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import hashlib
import time

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
        "cache_trust_manifest_sha256": hashlib.sha256(b"cache").hexdigest(),
        "provider_running_receipt_sha256": hashlib.sha256(b"running").hexdigest(),
        "provider_source_image_id": "computeimage-u00zf6w3yf72gakhcy",
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
            "videos": [
                _artifact(f"videos/{row['attempt_id']}_{view}_rgb.mp4")
                for view in ("left", "right", "top")
            ],
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
    (root / "evidence").mkdir(parents=True, exist_ok=True)
    (root / "evidence/cache-trust-manifest.json").write_bytes(b"cache")
    (root / "evidence/provider-running-receipt.json").write_bytes(b"running")
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


def test_native_oracle_cli_returns_stable_nonzero_for_typed_stop(tmp_path: Path) -> None:
    from scripts.verify_native_reference_evaluator_gate import main

    document = _bundle(attempts=_attempts()[:2])
    document["attempts"][1]["success"] = False  # type: ignore[index]
    _materialize_artifacts(tmp_path, document)
    result = tmp_path / "result.json"; receipt = tmp_path / "receipt.json"
    result.write_text(json.dumps(document), encoding="utf-8")

    assert main(["verify-execution", "--result", str(result), "--bundle-root", str(tmp_path), "--receipt", str(receipt)]) == 3
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "evaluator_compatibility_stop"


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
    assert "fetch-cache-manifest" in text
    assert "bind-provider-receipt --state RUNNING" in text
    assert "LEHOME_NATIVE_REFERENCE_PROVIDER_RUNNING_RECEIPT" in text
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


def test_native_launcher_accepts_the_exact_seven_checkpoint_filenames_before_source_gate(tmp_path: Path) -> None:
    environment = _launcher_environment(tmp_path)
    checkpoint = Path(environment["LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT"])
    processor = '{"steps":[{}, {}, {"config":{"action_horizon":16}}]}'
    postprocessor = '{"steps":[{"config":{"env_action_dim":12}}]}'
    for filename in (
        "config.json", "model.safetensors", "policy_preprocessor_step_2_groot_pack_inputs_v3.safetensors",
        "policy_postprocessor_step_0_groot_action_unpack_unnormalize_v1.safetensors", "train_config.json",
    ):
        (checkpoint / filename).write_text("x", encoding="utf-8")
    (checkpoint / "policy_preprocessor.json").write_text(processor, encoding="utf-8")
    (checkpoint / "policy_postprocessor.json").write_text(postprocessor, encoding="utf-8")
    fake_bin = tmp_path / "fake-bin"; fake_bin.mkdir()
    fake_hash = fake_bin / "sha256sum"
    fake_hash.write_text(
        "#!/usr/bin/env bash\ncase \"${@: -1}\" in\n"
            "*train_config.json) echo 81cd0cfe2b2f70dbf55bc7739f9a1f248aebd0e281994f415964d9d0f6e3c118 ;;\n"
            "*config.json) echo b7c385bc57456eae603e929b84defb7e991194aade2aad70785e21991e37614c ;;\n"
        "*model.safetensors) echo d8d91b6c11cb5aa18fe9a48e7da88eae0ec7e5a227a315ba67ee167d645cde76 ;;\n"
        "*policy_preprocessor.json) echo a258dac8fa4e4e138990776e156cae36ae6cf172504a8c9e5f2d5864c9126009 ;;\n"
        "*policy_postprocessor.json) echo f9e18fa7da47e2b6d7ba3459236b140e28f834ce5640ba199be1412d50672fa7 ;;\n"
        "*step_2_groot_pack_inputs_v3.safetensors|*step_0_groot_action_unpack_unnormalize_v1.safetensors) echo 74dcbba5d152b7e07c239d8cd66b19b1fd08aa37ff930aa5f2e94cd772a4a912 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_hash.chmod(0o755)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"

    result = subprocess.run(["bash", str(LAUNCHER)], env=environment, text=True, capture_output=True, check=False)

    assert result.returncode == 2
    assert "checkpoint" not in result.stderr
    assert "source cache is incomplete" in result.stderr


def test_native_stage_compiler_parses_real_public_episode_lines_and_binds_files(tmp_path: Path) -> None:
    from scripts.verify_native_reference_evaluator_gate import compile_native_stage

    root = tmp_path / "bundle"
    (root / "logs").mkdir(parents=True)
    (root / "videos" / "stage-1" / "success").mkdir(parents=True)
    log = root / "logs" / "stage-1.log"
    log.write_text("Episode 1/2: Return=1.00, Success=True\nEpisode 2/2: Return=2.00, Success=True\n", encoding="utf-8")
    for episode in (0, 1):
        for view in ("left", "right", "top"):
            (root / "videos" / "stage-1" / "success" / f"episode{episode}_observation_images_{view}_rgb.mp4").write_bytes(b"video")

    result = compile_native_stage(root, stage=1, category="top_long", garment="Top_Long_Seen_0", identity=_identity())

    assert [row["success"] for row in result["attempts"]] == [True, True]
    assert [row["log"]["path"] for row in result["attempts"]] == ["logs/stage-1.log", "logs/stage-1.log"]
    assert all(len(row["videos"]) == 3 for row in result["attempts"])
    assert all((root / row["receipt"]["path"]).is_file() for row in result["attempts"])


def test_native_stage_compiler_rejects_missing_side_or_wrong_outcome_directory(tmp_path: Path) -> None:
    from scripts.verify_native_reference_evaluator_gate import NativeReferenceGateError, compile_native_stage

    root = tmp_path / "bundle"; (root / "logs").mkdir(parents=True)
    (root / "videos" / "stage-1" / "success").mkdir(parents=True)
    (root / "videos" / "stage-1" / "failure").mkdir(parents=True)
    (root / "logs" / "stage-1.log").write_text("Episode 1/2: Return=1, Success=True\nEpisode 2/2: Return=1, Success=True\n", encoding="utf-8")
    for episode in (0, 1):
        for view in ("left", "right", "top"):
            directory = "failure" if (episode, view) == (0, "left") else "success"
            if (episode, view) != (1, "right"):
                (root / "videos" / "stage-1" / directory / f"episode{episode}_observation_images_{view}_rgb.mp4").write_bytes(b"video")

    with pytest.raises(NativeReferenceGateError, match="videos"):
        compile_native_stage(root, stage=1, category="top_long", garment="Top_Long_Seen_0", identity=_identity())


def test_native_finalization_requires_fidelity_publication_and_stopped_vm_receipts() -> None:
    from scripts.verify_native_reference_evaluator_gate import NativeReferenceGateError, finalize_native_reference_gate, verify_native_reference_result

    execution = verify_native_reference_result(_bundle())
    with pytest.raises(NativeReferenceGateError, match="fidelity review"):
        finalize_native_reference_gate(execution, {}, {}, {})

    execution_sha = hashlib.sha256(json.dumps(execution, sort_keys=True, separators=(",", ":")).encode() + b"\n").hexdigest()
    fidelity = {
        "schema_version": 1, "kind": "lehome_native_reference_fidelity_review_v1", "execution_receipt_sha256": execution_sha,
        "review_method": "manual_video_audit", "attempts": [{"attempt_id": row["attempt_id"], "cloth_present": True, "cloth_flight": False, "nonfinite": False, "safety_failure": False, "evidence_sha256": execution["attempt_evidence_sha256"][row["attempt_id"]]} for row in _attempts()],
    }
    publication = {"schema_version": 1, "kind": "lehome_native_reference_hf_readback_v1", "execution_receipt_sha256": execution_sha, "immutable_revision": "b" * 40, "bundle_manifest_sha256": "c" * 64, "readback_verified": True}
    stopped = {"schema_version": 1, "kind": "lehome_native_reference_provider_observation_v1", "vm_id": _identity()["vm_id"], "vm_name": "lehome-rollout", "disk_id": _identity()["disk_id"], "provider_source_image_id": _identity()["provider_source_image_id"], "state": "STOPPED", "captured_unix_seconds": int(time.time()), "provider_response_sha256": "f" * 64}

    final = finalize_native_reference_gate(execution, fidelity, publication, stopped)
    assert final["status"] == "passed"

    fidelity["attempts"][0]["evidence_sha256"] = "a" * 64
    with pytest.raises(NativeReferenceGateError, match="evidence"):
        finalize_native_reference_gate(execution, fidelity, publication, stopped)


def test_public_cache_manifest_must_be_read_back_from_fixed_hf_repo() -> None:
    from scripts.verify_native_reference_evaluator_gate import fetch_public_cache_manifest

    manifest = {
        "schema_version": 1, "kind": "lehome_native_reference_cache_trust_manifest_v2",
        "source_repository": "ryanjin333/lehome-groot-n17-rollouts", "immutable_revision": "a" * 40,
        "path": "reference-checks/native/cache-trust-manifest.json",
        "checkpoint_tree_sha256": "b" * 64, "metadata_tree_sha256": "c" * 64, "assets_tree_sha256": "d" * 64,
    }
    calls: list[str] = []
    def downloader(url: str) -> bytes:
        calls.append(url); return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"

    observed, raw = fetch_public_cache_manifest("a" * 40, "reference-checks/native/cache-trust-manifest.json", downloader=downloader)
    assert observed == manifest
    assert raw == downloader(calls[0])
    assert "/datasets/ryanjin333/lehome-groot-n17-rollouts/resolve/" in calls[0]


def test_provider_observation_uses_exact_adapter_shape_and_state() -> None:
    from scripts.verify_native_reference_evaluator_gate import capture_provider_observation

    raw = {
        "metadata": {"id": "computeinstance-u00t6xfqhadrcmssa2", "name": "lehome-rollout"},
        "status": {"state": "RUNNING"},
        "spec": {"boot_disk": {"managed_disk": {"spec": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}}}, "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}]},
    }
    class Provider:
        def get(self, instance_id: str) -> dict[str, object]:
            assert instance_id == "computeinstance-u00t6xfqhadrcmssa2"; return raw

    receipt = capture_provider_observation(Provider(), expected_state="RUNNING")
    assert receipt["state"] == "RUNNING"
    assert receipt["provider_response_sha256"] == hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode() + b"\n").hexdigest()


def test_native_gate_runbook_preserves_the_no_collection_admission_boundary() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert "d384fe00508acd96ab1c3c5dc265e08261f94b3b" in text
    assert "Top_Long_Seen_0" in text
    assert "7/8" in text
    assert "Do not start collection or training" in text
    assert "Hugging Face" in text
