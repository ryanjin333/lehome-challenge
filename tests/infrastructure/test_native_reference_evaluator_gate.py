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
CANONICAL_CACHE_MANIFEST = ROOT / "rollout_appliance" / "native_reference_canonical_cache_manifest.json"


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
        "source_root": "/mnt/lehome/reference-native/source",
        "python_executable": "/usr/bin/python3",
        "python_version": "3.12.0",
        "torch_version": "2.7.0",
        "lerobot_origin": "/opt/python/lerobot/__init__.py",
        "scripts_eval_origin": "/mnt/lehome/reference-native/source/scripts/eval.py",
        "lehome_origin": "/mnt/lehome/reference-native/source/source/lehome/lehome/__init__.py",
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
    assert "inventory-cache" in text
    assert "write_cache_inventory_manifest" in text
    assert "CACHE_TRUST_MANIFEST" in text
    assert "fetch-cache-manifest" in text
    assert "bind-provider-receipt --state RUNNING" in text
    assert "LEHOME_NATIVE_REFERENCE_PROVIDER_RUNNING_RECEIPT" in text
    assert "torch.cuda.is_available" in text
    assert 'EXACT_VM_ID="computeinstance-u00t6xfqhadrcmssa2"' in text
    assert 'PROTECTED_DISK_ID="computedisk-u00pbe55crxy7jr56x"' in text
    assert "validate_running_provider_binding" in text
    assert text.count("validate_checkpoint full") >= 2
    assert "probe_host_runtime" in text
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


def test_host_runtime_probe_ignores_shadow_scripts_package_in_caller_cwd(tmp_path: Path) -> None:
    source = tmp_path / "pinned-source"
    package_root = source / "source" / "lehome"
    shadow = tmp_path / "shadow-cwd"
    for package, body in (
        (source / "scripts", ""),
        (package_root / "lehome", ""),
        (package_root / "lerobot", '__version__ = "0.4.3"\n'),
        (package_root / "torch", '__version__ = "test-torch"\n'),
        (shadow / "scripts", ""),
    ):
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(body, encoding="utf-8")
    (source / "scripts" / "eval.py").write_text("PINNED = True\n", encoding="utf-8")
    (shadow / "scripts" / "eval.py").write_text("SHADOW = True\n", encoding="utf-8")
    receipt = tmp_path / "host-runtime.json"
    environment = {**os.environ, "PYTHONPATH": str(shadow)}

    result = subprocess.run(
        [
            os.sys.executable,
            str(ROOT / "scripts" / "verify_native_reference_evaluator_gate.py"),
            "probe-host-runtime",
            "--source-root",
            str(source),
            "--receipt",
            str(receipt),
        ],
        cwd=shadow,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    runtime = json.loads(receipt.read_text(encoding="utf-8"))
    assert Path(runtime["scripts_eval_origin"]) == (source / "scripts" / "eval.py").resolve()
    assert shadow.resolve() not in Path(runtime["scripts_eval_origin"]).parents


def _write_test_canonical_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    import scripts.verify_native_reference_evaluator_gate as gate

    metadata = tmp_path / "metadata"
    assets = tmp_path / "assets"
    metadata_file = metadata / "top_long_merged" / "meta" / "info.json"
    metadata_file.parent.mkdir(parents=True)
    metadata_file.write_bytes(b"metadata")
    asset_paths = [assets / root / "pinned.bin" for root in gate.ASSETS_RUNTIME_ROOTS]
    for index, path in enumerate(asset_paths):
        path.parent.mkdir(parents=True)
        path.write_bytes(f"asset-{index}".encode())

    def row(path: Path, root: Path) -> dict[str, object]:
        raw = path.read_bytes()
        return {
            "path": path.relative_to(root).as_posix(),
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    manifest = {
        "schema_version": 1,
        "kind": "lehome_native_reference_canonical_cache_manifest_v1",
        "metadata": {
            "repository_type": "model",
            "repository": gate.METADATA_REPOSITORY,
            "revision": gate.METADATA_REVISION,
            "root": "dataset_meta",
            "files": [row(metadata_file, metadata)],
        },
        "assets": {
            "repository_type": "dataset",
            "repository": gate.ASSETS_REPOSITORY,
            "revision": gate.ASSETS_REVISION,
            "runtime_roots": list(gate.ASSETS_RUNTIME_ROOTS),
            "files": [row(path, assets) for path in asset_paths],
        },
    }
    manifest_path = tmp_path / "canonical.json"
    raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    manifest_path.write_bytes(raw)
    monkeypatch.setattr(gate, "CANONICAL_CACHE_MANIFEST_SHA256", hashlib.sha256(raw).hexdigest())
    return manifest_path, metadata, assets


def test_canonical_metadata_cache_rejects_mutation_and_extra_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.verify_native_reference_evaluator_gate as gate

    manifest, metadata, _ = _write_test_canonical_manifest(tmp_path, monkeypatch)
    assert gate.validate_canonical_cache_tree(metadata, section="metadata", manifest_path=manifest)
    target = next(path for path in metadata.rglob("*") if path.is_file())
    target.write_bytes(b"mutated!")
    with pytest.raises(gate.NativeReferenceGateError, match="digest mismatch"):
        gate.validate_canonical_cache_tree(metadata, section="metadata", manifest_path=manifest)
    target.write_bytes(b"metadata")
    (metadata / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(gate.NativeReferenceGateError, match="file set is not exact"):
        gate.validate_canonical_cache_tree(metadata, section="metadata", manifest_path=manifest)


def test_canonical_assets_cache_rejects_mutation_and_extra_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.verify_native_reference_evaluator_gate as gate

    manifest, _, assets = _write_test_canonical_manifest(tmp_path, monkeypatch)
    assert gate.validate_canonical_cache_tree(assets, section="assets", manifest_path=manifest)
    target = assets / "objects" / "pinned.bin"
    target.write_bytes(b"mutated")
    with pytest.raises(gate.NativeReferenceGateError, match="digest mismatch"):
        gate.validate_canonical_cache_tree(assets, section="assets", manifest_path=manifest)
    target.write_bytes(b"asset-0")
    (assets / "textures" / "extra.bin").write_bytes(b"extra")
    with pytest.raises(gate.NativeReferenceGateError, match="file set is not exact"):
        gate.validate_canonical_cache_tree(assets, section="assets", manifest_path=manifest)


def test_committed_canonical_manifest_pins_public_provenance_and_runtime_roots() -> None:
    from scripts.verify_native_reference_evaluator_gate import CANONICAL_CACHE_MANIFEST_SHA256

    raw = CANONICAL_CACHE_MANIFEST.read_bytes()
    manifest = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == CANONICAL_CACHE_MANIFEST_SHA256
    assert manifest["metadata"] | {"files": None} == {
        "repository_type": "model",
        "repository": "theo-zhou/lehome-groot-submission-4",
        "revision": "d384fe00508acd96ab1c3c5dc265e08261f94b3b",
        "root": "dataset_meta",
        "files": None,
    }
    assert manifest["assets"] | {"files": None} == {
        "repository_type": "dataset",
        "repository": "lehome/asset_challenge",
        "revision": "bea65fd960ad5a1bb3bd3fa77164b28001c08ef9",
        "runtime_roots": ["objects", "robots", "scenes", "textures"],
        "files": None,
    }
    assert len(manifest["metadata"]["files"]) == 20
    assert len(manifest["assets"]["files"]) == 445


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
    publication = {"schema_version": 2, "kind": "lehome_native_reference_hf_readback_v2", "execution_receipt_sha256": execution_sha, "repository": "ryanjin333/lehome-groot-n17-rollouts", "remote_prefix": "reference-checks/native-" + "a" * 16, "immutable_revision": "b" * 40, "bundle_manifest_sha256": "c" * 64, "published_unix_seconds": int(time.time()), "readback_verified": True}
    stopped = {"schema_version": 1, "kind": "lehome_native_reference_provider_observation_v1", "vm_id": _identity()["vm_id"], "vm_name": "lehome-rollout", "disk_id": _identity()["disk_id"], "provider_source_image_id": _identity()["provider_source_image_id"], "state": "STOPPED", "captured_unix_seconds": int(time.time()), "provider_response_sha256": "f" * 64}

    final = finalize_native_reference_gate(execution, fidelity, publication, stopped)
    assert final["status"] == "passed"

    fidelity["attempts"][0]["evidence_sha256"] = "a" * 64
    with pytest.raises(NativeReferenceGateError, match="evidence"):
        finalize_native_reference_gate(execution, fidelity, publication, stopped)

    stopped["captured_unix_seconds"] = publication["published_unix_seconds"] - 1
    with pytest.raises(NativeReferenceGateError, match="post-publication"):
        finalize_native_reference_gate(execution, fidelity | {"attempts": [{**row, "evidence_sha256": execution["attempt_evidence_sha256"][row["attempt_id"]]} for row in fidelity["attempts"]]}, publication, stopped)


def test_public_cache_manifest_must_be_read_back_from_fixed_hf_repo() -> None:
    from scripts.verify_native_reference_evaluator_gate import fetch_public_cache_manifest

    manifest = {
        "schema_version": 2, "kind": "lehome_native_reference_cache_trust_manifest_v2",
        "source_repository": "ryanjin333/lehome-groot-n17-rollouts",
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


def test_publish_native_bundle_uploads_and_anonymously_readbacks_every_file(tmp_path: Path) -> None:
    from scripts.verify_native_reference_evaluator_gate import publish_native_reference_bundle, verify_native_reference_result

    document = _bundle(); _materialize_artifacts(tmp_path, document)
    (tmp_path / "result.json").write_text(json.dumps(document), encoding="utf-8")
    execution = verify_native_reference_result(document, bundle_root=tmp_path)
    (tmp_path / "execution-receipt.json").write_text(json.dumps(execution), encoding="utf-8")
    uploaded: dict[str, bytes] = {}
    class Transport:
        def resolve_approved_ref(self, **_kwargs): return "a" * 40
        def list_tree(self, **kwargs):
            prefix = kwargs["remote_prefix"]
            return [type("Entry", (), {"relative_path": f"{prefix}/{path}", "entry_type": "file"}) for path in uploaded]
        def upload_files(self, **kwargs):
            root = kwargs["source"]
            for entry in kwargs["entries"]: uploaded[entry.relative_path] = (root / entry.relative_path).read_bytes()
            return "b" * 40
        def download_files(self, **kwargs):
            for path in kwargs["relative_paths"]:
                destination = kwargs["destination"] / path; destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(uploaded[path])
            return kwargs["revision"]

    receipt = publish_native_reference_bundle(tmp_path, execution, token="test-token", transport=Transport(), now=lambda: 1000)
    assert receipt["kind"] == "lehome_native_reference_hf_readback_v2"
    assert receipt["readback_verified"] is True
    assert "bundle-manifest.json" in uploaded


def test_publish_cache_manifest_uploads_one_immutable_file_and_returns_launcher_inputs(tmp_path: Path) -> None:
    from scripts.verify_native_reference_evaluator_gate import publish_cache_manifest

    manifest = {
        "schema_version": 2, "kind": "lehome_native_reference_cache_trust_manifest_v2",
        "source_repository": "ryanjin333/lehome-groot-n17-rollouts",
        "path": "reference-checks/native-cache-20260828/cache-trust-manifest.json",
        "checkpoint_tree_sha256": "b" * 64, "metadata_tree_sha256": "c" * 64, "assets_tree_sha256": "d" * 64,
    }
    path = tmp_path / "cache-trust-manifest.json"; path.write_text(json.dumps(manifest), encoding="utf-8")
    uploaded: dict[str, bytes] = {}
    class Transport:
        def resolve_approved_ref(self, **_kwargs): return "a" * 40
        def list_tree(self, **kwargs):
            prefix = kwargs["remote_prefix"]
            return [type("Entry", (), {"relative_path": f"{prefix}/{name}", "entry_type": "file"}) for name in uploaded]
        def upload_files(self, **kwargs):
            for entry in kwargs["entries"]: uploaded[entry.relative_path] = (kwargs["source"] / entry.relative_path).read_bytes()
            return "b" * 40
        def download_files(self, **kwargs):
            for name in kwargs["relative_paths"]:
                target = kwargs["destination"] / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(uploaded[name])
            return kwargs["revision"]

    receipt = publish_cache_manifest(path, token="test-token", transport=Transport())
    assert receipt == {
        "schema_version": 1,
        "kind": "lehome_native_reference_cache_manifest_readback_v1",
        "repository": "ryanjin333/lehome-groot-n17-rollouts",
        "immutable_revision": "b" * 40,
        "path": manifest["path"],
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "readback_verified": True,
    }


def test_native_identity_requires_host_python_and_pinned_source_origins() -> None:
    from scripts.verify_native_reference_evaluator_gate import NativeReferenceGateError, verify_native_reference_result

    identity = _identity()
    assert verify_native_reference_result({**_bundle(), "identity": identity})["status"] == "oracle_matched_pending_finalization"
    identity["scripts_eval_origin"] = "/tmp/eval.py"
    with pytest.raises(NativeReferenceGateError, match="origin"):
        verify_native_reference_result({**_bundle(), "identity": identity})


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
    assert "scp" in text
    assert "LEHOME_NATIVE_REFERENCE_IMAGE" not in text
    assert "inventory-cache" in text
    assert "publish-cache-manifest" in text
