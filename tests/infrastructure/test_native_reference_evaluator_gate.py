"""Offline contract tests for the isolated native public-reference gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import hashlib
import sys
import time
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "rollout_appliance" / "run_native_reference_evaluator_gate.sh"
CONTAINER_WRAPPER = ROOT / "rollout_appliance" / "run_native_reference_evaluator_container.sh"
NATIVE_SITE_CUSTOMIZE = ROOT / "rollout_appliance" / "native_reference_site" / "sitecustomize.py"
CHECKPOINT_COMPATIBILITY_SHIM = NATIVE_SITE_CUSTOMIZE.parent / "checkpoint_compatibility.py"
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
        "runtime_image_receipt_sha256": hashlib.sha256(b"runtime-image").hexdigest(),
        "checkpoint_compatibility_receipt_sha256": hashlib.sha256(b"compatibility").hexdigest(),
        "peft_wheel_sha256": "0bf06847a3551e3019fc58c440cffc9a6b73e6e2962c95b52e224f77bbdb50f1",
        "peft_overlay_receipt_sha256": hashlib.sha256(b"peft-overlay").hexdigest(),
        "flash_attention_wheel_sha256": "cd1a45ebfc1731a13e55ad68e0c9ad92390ddfffba306f9222be67c6d5a805af",
        "flash_attention_overlay_receipt_sha256": hashlib.sha256(b"flash-overlay").hexdigest(),
        "flash_attention_runtime_receipt_sha256": hashlib.sha256(b"flash-runtime").hexdigest(),
        "pynput_backend": "dummy",
        "pynput_backend_receipt_sha256": hashlib.sha256(b"pynput-backend").hexdigest(),
        "provider_source_image_id": "computeimage-u00zf6w3yf72gakhcy",
        "runtime_image_reference": "lehome-rollout:build",
        "runtime_image_id": "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7",
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
    (root / "evidence/runtime-image-receipt.json").write_bytes(b"runtime-image")
    sanitized = b"{}\n"
    (root / "checkpoint-config-view").mkdir()
    (root / "checkpoint-config-view/config.json").write_bytes(sanitized)
    compatibility = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "lehome_native_reference_checkpoint_compatibility_v1",
                "checkpoint_root": "/mnt/lehome/cache/reference-theo-d384fe0/repo/pretrained_model",
                "raw_config_sha256": "b7c385bc57456eae603e929b84defb7e991194aade2aad70785e21991e37614c",
                "sanitized_config_root": str((root / "checkpoint-config-view").resolve()),
                "sanitized_config_sha256": hashlib.sha256(sanitized).hexdigest(),
                "removed_fields": [
                    {"key": "decay_lr_ratio", "value": 0.1},
                    {"key": "num_decay_steps", "value": 4000},
                ],
                "lerobot_distribution": "lerobot",
                "lerobot_version": "0.4.3",
                "lerobot_wheel_filename": "lerobot-0.4.3-py3-none-any.whl",
                "lerobot_wheel_sha256": "b08c1c15b2356bd4e658122deabfb9dacd2d7447de4a4327720991723d4edf2c",
                "groot_config_origin": "/opt/python/lerobot/policies/groot/configuration_groot.py",
                "groot_config_missing_fields": ["decay_lr_ratio", "num_decay_steps"],
                "rationale": "inference_only_remove_unsupported_training_scheduler_fields",
                "original_checkpoint_unchanged": True,
                "installed_lerobot_package_root": "/opt/python/lerobot",
                "expected_lerobot_package_tree_sha256": "db3b4e18b166d4bb7fb4354cec82a7fbd15bb24230f9d71269a017c774e0852f",
                "expected_lerobot_package_file_count": 289,
                "installed_lerobot_package_tree_sha256": "db3b4e18b166d4bb7fb4354cec82a7fbd15bb24230f9d71269a017c774e0852f",
                "installed_lerobot_package_file_count": 289,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    (root / "evidence/checkpoint-compatibility-receipt.json").write_bytes(compatibility)
    bundle["identity"]["checkpoint_compatibility_receipt_sha256"] = hashlib.sha256(compatibility).hexdigest()  # type: ignore[index]
    peft_overlay = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "lehome_native_reference_peft_overlay_v1",
                "wheel_path": "/mnt/lehome/reference-native/dependencies/peft-0.18.1-py3-none-any.whl",
                "wheel_filename": "peft-0.18.1-py3-none-any.whl",
                "wheel_url": "https://files.pythonhosted.org/packages/b3/14/b4e3f574acf349ae6f61f9c000a77f97a3b315b4bb6ad03791e79ae4a568/peft-0.18.1-py3-none-any.whl",
                "wheel_size": 556960,
                "wheel_sha256": "0bf06847a3551e3019fc58c440cffc9a6b73e6e2962c95b52e224f77bbdb50f1",
                "distribution_name": "peft",
                "peft_version": "0.18.1",
                "peft_origin": "/mnt/lehome/reference-native/dependencies/peft-0.18.1-py3-none-any.whl/peft/__init__.py",
                "required_symbols": ["LoraConfig", "get_peft_model"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    (root / "evidence/peft-overlay-receipt.json").write_bytes(peft_overlay)
    bundle["identity"]["peft_overlay_receipt_sha256"] = hashlib.sha256(peft_overlay).hexdigest()  # type: ignore[index]
    flash_overlay = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "lehome_native_reference_flash_attention_overlay_v1",
                "wheel_path": "/mnt/lehome/reference-native/dependencies/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
                "wheel_filename": "flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
                "wheel_url": "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
                "wheel_size": 256027206,
                "wheel_sha256": "cd1a45ebfc1731a13e55ad68e0c9ad92390ddfffba306f9222be67c6d5a805af",
                "distribution_name": "flash_attn",
                "flash_attn_version": "2.8.3",
                "wheel_tag": "cp311-cp311-linux_x86_64",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    (root / "evidence/flash-attention-overlay-receipt.json").write_bytes(flash_overlay)
    bundle["identity"]["flash_attention_overlay_receipt_sha256"] = hashlib.sha256(flash_overlay).hexdigest()  # type: ignore[index]
    flash_runtime = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "lehome_native_reference_flash_attention_runtime_v1",
                "torch_version": "2.7.0+cu128",
                "torch_cuda_version": "12.8",
                "torch_cxx11_abi": True,
                "cuda_capability": [12, 0],
                "flash_attn_version": "2.8.3",
                "flash_attn_origin": "/opt/lehome-challenge/.venv/lib/python3.11/site-packages/flash_attn/__init__.py",
                "kernel": {"shape": [1, 2, 4, 64], "dtype": "float16", "finite": True},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    (root / "evidence/flash-attention-runtime-receipt.json").write_bytes(flash_runtime)
    bundle["identity"]["flash_attention_runtime_receipt_sha256"] = hashlib.sha256(flash_runtime).hexdigest()  # type: ignore[index]
    pynput_backend = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "lehome_native_reference_pynput_backend_v1",
                "pynput_backend": "dummy",
                "keyboard_listener_module": "pynput.keyboard._base",
                "keyboard_key_module": "pynput.keyboard._base",
                "x11_modules_loaded": False,
                "keyboard_control_started": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    (root / "evidence/pynput-backend-receipt.json").write_bytes(pynput_backend)
    bundle["identity"]["pynput_backend_receipt_sha256"] = hashlib.sha256(pynput_backend).hexdigest()  # type: ignore[index]
    preflight = {
        "schema_version": 2,
        "kind": "lehome_native_reference_preflight_v2",
        "identity": bundle["identity"],
        "cuda_probe_sha256": "a" * 64,
        "host_runtime_sha256": "b" * 64,
        "runtime_image_receipt_sha256": bundle["identity"]["runtime_image_receipt_sha256"],
        "checkpoint_compatibility_receipt_sha256": bundle["identity"]["checkpoint_compatibility_receipt_sha256"],
        "peft_overlay_receipt_sha256": bundle["identity"]["peft_overlay_receipt_sha256"],
        "flash_attention_overlay_receipt_sha256": bundle["identity"]["flash_attention_overlay_receipt_sha256"],
        "flash_attention_runtime_receipt_sha256": bundle["identity"]["flash_attention_runtime_receipt_sha256"],
        "pynput_backend_receipt_sha256": bundle["identity"]["pynput_backend_receipt_sha256"],
    }
    (root / "preflight.json").write_text(
        json.dumps(preflight, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
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
    run_stage = text.split("run_stage() {", 1)[1].split("\n}", 1)[0]

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
    assert "LEHOME_NATIVE_REFERENCE_RUNTIME_IMAGE_RECEIPT" in text
    assert "bind-runtime-image-receipt" in text
    assert "torch.cuda.is_available" in text
    assert 'EXACT_VM_ID="computeinstance-u00t6xfqhadrcmssa2"' in text
    assert 'PROTECTED_DISK_ID="computedisk-u00pbe55crxy7jr56x"' in text
    assert "validate_running_provider_binding" in text
    assert text.count("validate_checkpoint full") >= 2
    assert "probe_host_runtime" in text
    assert 'from isaaclab.app import AppLauncher' in (
        ROOT / "scripts" / "verify_native_reference_evaluator_gate.py"
    ).read_text(encoding="utf-8")
    assert "/opt/lehome-challenge/third_party/IsaacLab/source/isaaclab" in text
    assert "/opt/lehome-challenge/third_party/IsaacLab/source/isaaclab_tasks" in text
    assert text.index("probe_cuda; probe_host_runtime") < text.index("run_stage 1")
    assert "validate_runtime_asset_bindings" in text
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
    assert "b83ecf7af081c6c6a60073a854be3b63b66bbb0dbe021a4683dc5428d0f360d8" in text
    assert 'show "$SOURCE_REVISION:pyproject.toml"' in text
    assert run_stage.count("--enable_cameras") == 1
    assert "--headless --enable_cameras --device cpu" in run_stage


def test_native_launcher_module_invocation_supports_relative_imports_and_pinned_source_wins(
    tmp_path: Path,
) -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    pinned = tmp_path / "pinned"
    shadow = tmp_path / "runtime"
    for root, marker in ((pinned, "PINNED"), (shadow, "SHADOW")):
        package = root / "scripts"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "origin.py").write_text(f'VALUE = "{marker}"\n', encoding="utf-8")
        (package / "eval.py").write_text(
            "from .origin import VALUE\nprint(VALUE)\n",
            encoding="utf-8",
        )

    result = subprocess.run(
        [os.sys.executable, "-P", "-m", "scripts.eval"],
        cwd=shadow,
        env={
            **os.environ,
            "PYTHONSAFEPATH": "1",
            "PYTHONPATH": f"{pinned}:{shadow}",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PINNED"
    assert 'readonly RUNTIME_REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"' in text
    assert '"$PYTHON_BIN" -P -m scripts.eval' in text
    assert 'cd -- "$RUNTIME_REPO_ROOT"' in text
    assert 'PYTHONSAFEPATH=1' in text
    assert 'PYTHONPATH="$PEFT_WHEEL_PATH:$SOURCE_ROOT/source/lehome:$SOURCE_ROOT:$ISAACLAB_ROOT:$ISAACLAB_TASKS_ROOT"' in text
    assert '"$PYTHON_BIN" "$SOURCE_ROOT/scripts/eval.py"' not in text
    assert '--ee_urdf_path "$ASSETS_ROOT/robots/so101_new_calib.urdf"' in text


def test_native_run_stage_binds_arguments_before_expanding_stage_log(tmp_path: Path) -> None:
    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    run_stage = launcher_text.split("run_stage() {", 1)[1].split("\n}", 1)[0]
    fake_python = tmp_path / "fake-python"
    trace = tmp_path / "python-invocations.txt"
    backend_trace = tmp_path / "evaluator-pynput-backend.txt"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$TRACE\"\n"
        "if [[ \" $* \" == *' -m scripts.eval '* ]]; then\n"
        "  printf '%s\\n' \"${PYNPUT_BACKEND-unset}\" >> \"$BACKEND_TRACE\"\n"
        "  printf '%s\\n' 'Episode 1/2: Return=1.00, Success=True'\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    output_root = tmp_path / "output"
    runtime_root = tmp_path / "runtime"
    source_root = tmp_path / "source"
    for path in (output_root / "logs", runtime_root, source_root):
        path.mkdir(parents=True)
    script = tmp_path / "exercise-run-stage.sh"
    script.write_text(
        "set -euo pipefail\n"
        f"OUTPUT_ROOT={str(output_root)!r}\n"
        f"RUNTIME_REPO_ROOT={str(runtime_root)!r}\n"
        f"SOURCE_ROOT={str(source_root)!r}\n"
        f"PYTHON_BIN={str(fake_python)!r}\n"
        f"SCRIPT_DIR={str(LAUNCHER.parent)!r}\n"
        f"NATIVE_SITE_ROOT={str(LAUNCHER.parent / 'native_reference_site')!r}\n"
        "CHECKPOINT_ROOT=/checkpoint\n"
        "METADATA_ROOT=/metadata\n"
        "ASSETS_ROOT=/assets\n"
        "ISAACLAB_ROOT=/isaaclab\n"
        "ISAACLAB_TASKS_ROOT=/isaaclab_tasks\n"
        "SANITIZED_CONFIG_ROOT=/sanitized\n"
        "CHECKPOINT_COMPATIBILITY_RECEIPT=/compatibility.json\n"
        "PEFT_WHEEL_PATH=/peft-wheel\n"
        "PEFT_OVERLAY_RECEIPT=/peft-overlay.json\n"
        "PEFT_OVERLAY_RECEIPT_SHA256=" + "0" * 64 + "\n"
        "validate_stage_integrity() { :; }\n"
        f"run_stage() {{{run_stage}\n}}\n"
        "run_stage 1 top_long Top_Long_Seen_0\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(script)],
        env={**os.environ, "TRACE": str(trace), "BACKEND_TRACE": str(backend_trace)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    invocations = trace.read_text(encoding="utf-8")
    assert "-m scripts.eval" in invocations
    assert invocations.count("--enable_cameras") == 1
    assert "-m scripts.eval --headless --enable_cameras --device cpu" in invocations
    assert "compile-stage" in invocations
    assert backend_trace.read_text(encoding="utf-8") == "dummy\n"


def test_runtime_asset_bindings_require_the_same_device_and_inode(tmp_path: Path) -> None:
    from scripts.verify_native_reference_evaluator_gate import (
        NativeReferenceGateError,
        validate_runtime_asset_bindings,
    )

    canonical = tmp_path / "runtime" / "Assets"
    for root in ("objects", "robots", "scenes", "textures"):
        (canonical / root).mkdir(parents=True)

    receipt = validate_runtime_asset_bindings(canonical, canonical.parent)
    assert receipt["runtime_assets_root"] == str(canonical.resolve())

    other_runtime = tmp_path / "other-runtime"
    for root in ("objects", "robots", "scenes", "textures"):
        (other_runtime / "Assets" / root).mkdir(parents=True)
    with pytest.raises(NativeReferenceGateError, match="device/inode"):
        validate_runtime_asset_bindings(canonical, other_runtime)


def test_host_runtime_probe_ignores_shadow_scripts_package_in_caller_cwd(tmp_path: Path) -> None:
    source = tmp_path / "pinned-source"
    package_root = source / "source" / "lehome"
    trusted_isaaclab = tmp_path / "trusted-isaaclab"
    shadow = tmp_path / "shadow-cwd"
    for package, body in (
        (source / "scripts", ""),
        (package_root / "lehome", ""),
        (package_root / "lerobot", '__version__ = "0.4.3"\n'),
        (package_root / "torch", '__version__ = "test-torch"\n'),
        (trusted_isaaclab / "isaaclab", ""),
        (trusted_isaaclab / "isaaclab" / "app", "class AppLauncher: pass\n"),
        (shadow / "scripts", ""),
        (shadow / "isaaclab", ""),
        (shadow / "isaaclab" / "app", "class AppLauncher: pass\n"),
    ):
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(body, encoding="utf-8")
    (source / "scripts" / "eval.py").write_text("PINNED = True\n", encoding="utf-8")
    (shadow / "scripts" / "eval.py").write_text("SHADOW = True\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "pinned"], check=True)
    tree_before = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(source.rglob("*")) if path.is_file() and ".git" not in path.parts)
    ).hexdigest()
    receipt = tmp_path / "host-runtime.json"
    environment = {
        **os.environ,
        "PYTHONPATH": f"{package_root}:{source}:{trusted_isaaclab}:{shadow}",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    result = subprocess.run(
        [
            os.sys.executable,
            str(ROOT / "scripts" / "verify_native_reference_evaluator_gate.py"),
            "probe-host-runtime",
            "--source-root",
            str(source),
            "--isaaclab-root",
            str(trusted_isaaclab),
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
    assert Path(runtime["isaaclab_app_origin"]) == (
        trusted_isaaclab / "isaaclab" / "app" / "__init__.py"
    ).resolve()
    assert runtime["app_launcher_class"] == "isaaclab.app.AppLauncher"
    tree_after = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(source.rglob("*")) if path.is_file() and ".git" not in path.parts)
    ).hexdigest()
    assert tree_after == tree_before
    assert subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ) == ""
    assert not any(source.rglob("__pycache__"))
    probe_launcher = LAUNCHER.read_text(encoding="utf-8").split(
        "probe_host_runtime() {", 1
    )[1].split("\n}", 1)[0]
    assert "PYTHONDONTWRITEBYTECODE=1" in probe_launcher


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


def test_capture_provider_direct_script_entrypoint_resolves_repository_package(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    raw = {
        "metadata": {"id": "computeinstance-u00t6xfqhadrcmssa2", "name": "lehome-rollout"},
        "status": {"state": "RUNNING"},
        "spec": {
            "boot_disk": {"managed_disk": {"spec": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}}},
            "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}],
        },
    }
    nebius = fake_bin / "nebius"
    nebius.write_text(
        "#!/usr/bin/env python3\nimport json\nprint(json.dumps(" + repr(raw) + "))\n",
        encoding="utf-8",
    )
    nebius.chmod(0o755)
    receipt = tmp_path / "provider.json"
    environment = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "PYTHONPATH": ""}

    result = subprocess.run(
        [
            os.sys.executable,
            str(ROOT / "scripts" / "verify_native_reference_evaluator_gate.py"),
            "capture-provider",
            "--state",
            "RUNNING",
            "--receipt",
            str(receipt),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(receipt.read_text(encoding="utf-8"))["state"] == "RUNNING"


def test_capture_runtime_image_direct_cli_inspects_only_the_approved_local_reference(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "docker-calls.json"
    raw = [
        {
            "Id": "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7",
            "RepoTags": ["lehome-rollout:build"],
        }
    ]
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(calls)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        f"print(json.dumps({raw!r}, sort_keys=True, separators=(',', ':')))\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    receipt = tmp_path / "runtime-image.json"

    result = subprocess.run(
        [
            os.sys.executable,
            str(ROOT / "scripts" / "verify_native_reference_evaluator_gate.py"),
            "capture-runtime-image",
            "--receipt",
            str(receipt),
        ],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "PYTHONPATH": ""},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(calls.read_text(encoding="utf-8")) == [
        "image",
        "inspect",
        "--",
        "lehome-rollout:build",
    ]
    document = json.loads(receipt.read_text(encoding="utf-8"))
    assert document["runtime_image_reference"] == "lehome-rollout:build"
    assert document["runtime_image_id"] == raw[0]["Id"]
    expected_raw = (json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert document["docker_inspect_sha256"] == hashlib.sha256(expected_raw).hexdigest()


def test_capture_runtime_image_rejects_inspect_without_approved_tag(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps([{'Id':'sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7','RepoTags':['other:tag']}]))\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    receipt = tmp_path / "runtime-image.json"

    result = subprocess.run(
        [
            os.sys.executable,
            str(ROOT / "scripts" / "verify_native_reference_evaluator_gate.py"),
            "capture-runtime-image",
            "--receipt",
            str(receipt),
        ],
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "PYTHONPATH": ""},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "image ID/tag is invalid" in result.stderr
    assert not receipt.exists()


def test_runtime_image_receipt_rejects_wrong_id_or_missing_tag() -> None:
    from scripts.verify_native_reference_evaluator_gate import (
        NativeReferenceGateError,
        validate_runtime_image_observation,
    )

    receipt = {
        "schema_version": 1,
        "kind": "lehome_native_reference_runtime_image_observation_v1",
        "runtime_image_reference": "lehome-rollout:build",
        "runtime_image_id": "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7",
        "captured_unix_seconds": int(time.time()),
        "docker_inspect_sha256": "f" * 64,
    }
    assert validate_runtime_image_observation(receipt) == receipt

    with pytest.raises(NativeReferenceGateError, match="runtime image"):
        validate_runtime_image_observation({**receipt, "runtime_image_id": "sha256:" + "0" * 64})
    with pytest.raises(NativeReferenceGateError, match="runtime image"):
        validate_runtime_image_observation({key: value for key, value in receipt.items() if key != "runtime_image_reference"})


def test_bind_runtime_image_receipt_copies_exact_validated_bytes(tmp_path: Path) -> None:
    source = tmp_path / "runtime-source.json"
    target = tmp_path / "runtime-bound.json"
    document = {
        "schema_version": 1,
        "kind": "lehome_native_reference_runtime_image_observation_v1",
        "runtime_image_reference": "lehome-rollout:build",
        "runtime_image_id": "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7",
        "captured_unix_seconds": int(time.time()),
        "docker_inspect_sha256": "f" * 64,
    }
    source.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            os.sys.executable,
            str(ROOT / "scripts" / "verify_native_reference_evaluator_gate.py"),
            "bind-runtime-image-receipt",
            "--input",
            str(source),
            "--receipt",
            str(target),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == source.read_bytes()


def test_runtime_image_receipt_is_bound_before_cuda_and_final_identity_validation() -> None:
    from scripts.verify_native_reference_evaluator_gate import NativeReferenceGateError, verify_native_reference_result

    text = LAUNCHER.read_text(encoding="utf-8")
    execution = text[text.index('mkdir --mode=0700 -- "$OUTPUT_ROOT"') :]
    assert execution.index("bind-runtime-image-receipt") < execution.index("probe_cuda;")
    assert execution.index("validate_runtime_image_binding") < execution.index("probe_cuda;")
    assert execution.index("prepare-checkpoint-compatibility") < execution.index("probe_cuda;")
    assert "CHECKPOINT_COMPATIBILITY_RECEIPT_SHA256" in execution
    assert "checkpoint_compatibility_receipt_sha256" in text

    bundle = _bundle()
    assert verify_native_reference_result(bundle)["identity"]["runtime_image_reference"] == "lehome-rollout:build"
    bundle["identity"] = {**bundle["identity"], "runtime_image_id": "sha256:" + "0" * 64}
    with pytest.raises(NativeReferenceGateError, match="runtime image"):
        verify_native_reference_result(bundle)


def test_execute_mode_requires_runtime_image_receipt_before_cache_or_cuda_work(tmp_path: Path) -> None:
    environment = _launcher_environment(tmp_path)
    environment.pop("LEHOME_NATIVE_REFERENCE_VALIDATE_ONLY")
    environment["LEHOME_NATIVE_REFERENCE_MODE"] = "execute"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stderr.strip() == "error: runtime image receipt is unavailable or unsafe"
    assert "checkpoint" not in result.stderr
    assert not (tmp_path / "native-reference-202608290001").exists()


def _write_fake_huggingface_hub(import_root: Path) -> Path:
    module = import_root / "huggingface_hub.py"
    module.write_text(
        """from pathlib import Path
from types import SimpleNamespace

_FILES = {}

class RepoFile:
    def __init__(self, path):
        self.path = path

class HfApi:
    def __init__(self, token=None):
        self.token = token

    def repo_info(self, **kwargs):
        return SimpleNamespace(oid="a" * 40)

    def list_repo_tree(self, revision, path_in_repo, **kwargs):
        if revision == "a" * 40:
            return []
        return [RepoFile(path) for path in sorted(_FILES) if path.startswith(path_in_repo + "/")]

    def upload_folder(self, folder_path, allow_patterns, path_in_repo, **kwargs):
        root = Path(folder_path)
        for relative in allow_patterns:
            _FILES[f"{path_in_repo}/{relative}"] = (root / relative).read_bytes()
        return SimpleNamespace(oid="b" * 40)

def hf_hub_download(filename, local_dir, **kwargs):
    target = Path(local_dir) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_FILES[filename])
    return str(target)
""",
        encoding="utf-8",
    )
    return module


def _direct_publisher_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    import_root = tmp_path / "fake-imports"
    import_root.mkdir()
    _write_fake_huggingface_hub(import_root)
    token = tmp_path / "hf-token"
    token.write_text("test-token", encoding="utf-8")
    token.chmod(0o600)
    return {**os.environ, "PYTHONPATH": str(import_root)}, token


def test_publish_bundle_direct_script_entrypoint_resolves_repository_package(tmp_path: Path) -> None:
    environment, token = _direct_publisher_environment(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "artifact.txt").write_text("artifact", encoding="utf-8")
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps({"status": "oracle_matched_pending_finalization"}), encoding="utf-8")
    receipt = tmp_path / "publication.json"

    result = subprocess.run(
        [
            os.sys.executable,
            str(ROOT / "scripts" / "verify_native_reference_evaluator_gate.py"),
            "publish-bundle",
            "--bundle-root",
            str(bundle),
            "--execution",
            str(execution),
            "--token-file",
            str(token),
            "--receipt",
            str(receipt),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(receipt.read_text(encoding="utf-8"))["readback_verified"] is True


def test_publish_cache_manifest_direct_script_entrypoint_resolves_repository_package(tmp_path: Path) -> None:
    environment, token = _direct_publisher_environment(tmp_path)
    manifest = tmp_path / "cache-trust-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "lehome_native_reference_cache_trust_manifest_v2",
                "source_repository": "ryanjin333/lehome-groot-n17-rollouts",
                "path": "reference-checks/native-cache-direct-cli/cache-trust-manifest.json",
                "checkpoint_tree_sha256": "b" * 64,
                "metadata_tree_sha256": "c" * 64,
                "assets_tree_sha256": "d" * 64,
            }
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "cache-publication.json"

    result = subprocess.run(
        [
            os.sys.executable,
            str(ROOT / "scripts" / "verify_native_reference_evaluator_gate.py"),
            "publish-cache-manifest",
            "--manifest",
            str(manifest),
            "--token-file",
            str(token),
            "--receipt",
            str(receipt),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(receipt.read_text(encoding="utf-8"))["readback_verified"] is True


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
    assert text.count("bash rollout_appliance/run_native_reference_evaluator_container.sh") == 4
    assert "bash rollout_appliance/run_native_reference_evaluator_gate.sh" not in text
    assert "capture-runtime-image" in text
    assert "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7" in text
    assert "wrapper launches by immutable ID" in text
    assert "dual authenticated-asset mounts" in text
    assert "/mnt/lehome/challenge-assets/Assets" not in text
    assert "LEHOME_NATIVE_REFERENCE_RUNTIME_IMAGE_RECEIPT" in text
    assert "Do not build or pull an image" in text


def test_runbook_uses_only_the_canonical_host_container_wrapper() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    for mode in ("source-stage", "inventory-cache", "validate-only", "execute"):
        assert f"run_native_reference_evaluator_container.sh {mode}" in text
    assert "Do not invoke `run_native_reference_evaluator_gate.sh` on the bare" in text


@pytest.mark.parametrize(
    ("mode", "mode_environment"),
    (
        ("source-stage", {}),
        (
            "inventory-cache",
            {
                "LEHOME_NATIVE_REFERENCE_CACHE_MANIFEST_OUTPUT": "/mnt/lehome/reference-native/cache-trust-manifest.json",
                "LEHOME_NATIVE_REFERENCE_CACHE_MANIFEST_PATH": "reference-checks/native-cache-test/cache-trust-manifest.json",
            },
        ),
        (
            "validate-only",
            {
                "LEHOME_NATIVE_REFERENCE_OUTPUT_ROOT": "/mnt/lehome/reference-native/native-reference-202608290001",
                "LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_REVISION": "a" * 40,
                "LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_PATH": "reference-checks/native-cache-test/cache-trust-manifest.json",
            },
        ),
        (
            "execute",
            {
                "LEHOME_NATIVE_REFERENCE_OUTPUT_ROOT": "/mnt/lehome/reference-native/native-reference-202608290001",
                "LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_REVISION": "a" * 40,
                "LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_PATH": "reference-checks/native-cache-test/cache-trust-manifest.json",
                "LEHOME_NATIVE_REFERENCE_PROVIDER_RUNNING_RECEIPT": "/mnt/lehome/reference-native/provider-running-receipt.json",
                "LEHOME_NATIVE_REFERENCE_RUNTIME_IMAGE_RECEIPT": "/mnt/lehome/reference-native/runtime-image-receipt.json",
            },
        ),
    ),
)
def test_native_container_wrapper_builds_complete_exact_command_per_mode(
    mode: str,
    mode_environment: dict[str, str],
) -> None:
    revision = "1" * 40
    result = subprocess.run(
        ["bash", str(CONTAINER_WRAPPER), mode, "--print-command"],
        env={
            **os.environ,
            "LEHOME_NATIVE_REFERENCE_RUNTIME_REVISION": revision,
            **mode_environment,
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    tokens = shlex.split(result.stdout)
    image = "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7"
    image_index = tokens.index(image)
    assert tokens[:7] == [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--gpus",
        "all",
    ]
    assert tokens[tokens.index("--network") + 1] == "host"
    assert tokens[tokens.index("--entrypoint") + 1] == "bash"
    assert tokens.index("--entrypoint") < image_index
    runtime_root = Path("/mnt/lehome/runtime-code") / revision
    assert tokens[image_index + 1] == str(
        runtime_root
        / "rollout_appliance"
        / "run_native_reference_evaluator_gate.sh"
    )
    assert all(index < image_index for index, token in enumerate(tokens) if token == "--env")

    forwarded: dict[str, str] = {}
    for index, token in enumerate(tokens):
        if token == "--env":
            key, value = tokens[index + 1].split("=", 1)
            forwarded[key] = value
    common = {
        "LEHOME_NATIVE_REFERENCE_MODE": mode,
        "LEHOME_NATIVE_REFERENCE_VALIDATE_ONLY": "1" if mode == "validate-only" else "0",
        "LEHOME_NATIVE_REFERENCE_PYTHON": "/isaac-sim/python.sh",
        "PYTHONEXE": "/opt/lehome-challenge/.venv/bin/python",
        "LEHOME_NATIVE_REFERENCE_SOURCE_ROOT": "/mnt/lehome/reference-native/source",
        "LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT": "/mnt/lehome/cache/reference-theo-d384fe0/repo/pretrained_model",
        "LEHOME_NATIVE_REFERENCE_METADATA_ROOT": "/mnt/lehome/cache/reference-theo-d384fe0/repo/dataset_meta",
        "LEHOME_NATIVE_REFERENCE_ASSETS_ROOT": "/mnt/lehome/reference-native/assets",
        "LEHOME_NATIVE_REFERENCE_VM_ID": "computeinstance-u00t6xfqhadrcmssa2",
        "LEHOME_NATIVE_REFERENCE_DISK_ID": "computedisk-u00pbe55crxy7jr56x",
    }
    assert forwarded == {**common, **mode_environment}

    mounts = [tokens[index + 1] for index, token in enumerate(tokens) if token == "--mount"]
    assert "type=bind,src=/mnt/lehome,dst=/mnt/lehome" in mounts
    assert f"type=bind,src={runtime_root},dst={runtime_root},readonly" in mounts
    for root in ("objects", "robots", "scenes", "textures"):
        assert any(
            mount.startswith(f"type=bind,src=/mnt/lehome/eval/assets/{root},")
            and mount.endswith(f"/Assets/{root},readonly")
            for mount in mounts
        )
        assert (
            f"type=bind,src=/mnt/lehome/eval/assets/{root},"
            f"dst=/mnt/lehome/reference-native/assets/{root},readonly"
        ) in mounts


def test_prepare_runtime_mountpoints_models_a_fresh_revision_staged_tree(tmp_path: Path) -> None:
    from scripts.verify_native_reference_evaluator_gate import prepare_runtime_asset_mountpoints

    seed = tmp_path / "seed"
    (seed / "Assets").mkdir(parents=True)
    (seed / "Assets" / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(seed)], check=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(seed), "add", "Assets/.gitignore"], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-qm", "stage"], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(seed), "rev-parse", "HEAD"], text=True
    ).strip()
    staged = tmp_path / "runtime-code" / revision
    staged.parent.mkdir()
    seed.rename(staged)

    receipt = prepare_runtime_asset_mountpoints(staged)

    assert receipt["runtime_revision"] == revision
    assert receipt["runtime_root"] == str(staged.resolve())
    assert [row["root"] for row in receipt["mountpoints"]] == [
        "objects",
        "robots",
        "scenes",
        "textures",
    ]
    assert all((staged / "Assets" / root).is_dir() for root in ("objects", "robots", "scenes", "textures"))
    assert subprocess.check_output(
        ["git", "-C", str(staged), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ) == ""


def test_two_pinned_module_invocations_redirect_logs_and_leave_source_immutable(tmp_path: Path) -> None:
    source = tmp_path / "pinned-source"
    package_root = source / "source" / "lehome"
    runtime_root = tmp_path / "runtime"
    external_project = tmp_path / "external-project"
    (source / "scripts").mkdir(parents=True)
    (package_root / "lehome" / "utils").mkdir(parents=True)
    runtime_root.mkdir()
    for path in (
        source / "scripts" / "__init__.py",
        package_root / "lehome" / "__init__.py",
        package_root / "lehome" / "utils" / "__init__.py",
    ):
        path.write_text("", encoding="utf-8")
    (package_root / "lehome" / "utils" / "logger.py").write_text(
        "from pathlib import Path\n"
        f"def get_project_root(): return Path({str(source)!r})\n",
        encoding="utf-8",
    )
    (source / "scripts" / "helper.py").write_text("VALUE = 'PINNED'\n", encoding="utf-8")
    (source / "scripts" / "eval.py").write_text(
        "from pathlib import Path\n"
        "from .helper import VALUE\n"
        "from lehome.utils.logger import get_project_root\n"
        "target=get_project_root()/'logs'/'top_long'\n"
        "target.mkdir(parents=True,exist_ok=True)\n"
        "(target/'eval.log').open('a').write(VALUE+'\\n')\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "pinned"], check=True)
    before = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    tree_before = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(source.rglob("*")) if path.is_file() and ".git" not in path.parts)
    ).hexdigest()
    environment = {
        **os.environ,
        "PYTHONPATH": f"{package_root}:{source}:{NATIVE_SITE_CUSTOMIZE.parent}",
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LEHOME_NATIVE_REFERENCE_LOG_PROJECT_ROOT": str(external_project),
        "LEHOME_NATIVE_REFERENCE_SOURCE_ROOT": str(source),
    }

    for _ in range(2):
        result = subprocess.run(
            [os.sys.executable, "-P", "-m", "scripts.eval"],
            cwd=runtime_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    tree_after = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(source.rglob("*")) if path.is_file() and ".git" not in path.parts)
    ).hexdigest()
    assert before == ""
    assert tree_after == tree_before
    assert subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ) == ""
    assert (external_project / "logs" / "top_long" / "eval.log").read_text(encoding="utf-8") == "PINNED\nPINNED\n"
    assert not any(source.rglob("__pycache__"))


def test_checkpoint_compatibility_shim_sanitizes_only_training_fields_and_preserves_load_paths(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    sanitized = tmp_path / "sanitized"
    fake_packages = tmp_path / "fake-packages"
    checkpoint.mkdir()
    sanitized.mkdir()
    raw_config = {
        "type": "groot",
        "num_decay_steps": 4000,
        "decay_lr_ratio": 0.1,
    }
    raw = json.dumps(raw_config, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    (checkpoint / "config.json").write_bytes(raw)
    sanitized_payload = json.dumps(
        {"type": "groot"}, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"
    (sanitized / "config.json").write_bytes(sanitized_payload)
    receipt = tmp_path / "compatibility.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "lehome_native_reference_checkpoint_compatibility_v1",
                "checkpoint_root": str(checkpoint.resolve()),
                "raw_config_sha256": hashlib.sha256(raw).hexdigest(),
                "sanitized_config_root": str(sanitized.resolve()),
                "sanitized_config_sha256": hashlib.sha256(sanitized_payload).hexdigest(),
                "removed_fields": [
                    {"key": "decay_lr_ratio", "value": 0.1},
                    {"key": "num_decay_steps", "value": 4000},
                ],
                "lerobot_distribution": "lerobot",
                "lerobot_version": "0.4.3",
                "lerobot_wheel_filename": "lerobot-0.4.3-py3-none-any.whl",
                "lerobot_wheel_sha256": "b08c1c15b2356bd4e658122deabfb9dacd2d7447de4a4327720991723d4edf2c",
                "groot_config_origin": str(
                    (fake_packages / "lerobot/policies/groot/configuration_groot.py").resolve()
                ),
                "groot_config_missing_fields": ["decay_lr_ratio", "num_decay_steps"],
                "rationale": "inference_only_remove_unsupported_training_scheduler_fields",
                "original_checkpoint_unchanged": True,
                "installed_lerobot_package_root": str((fake_packages / "lerobot").resolve()),
                "expected_lerobot_package_tree_sha256": "",
                "expected_lerobot_package_file_count": 0,
                "installed_lerobot_package_tree_sha256": "",
                "installed_lerobot_package_file_count": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    for package in (
        fake_packages / "lerobot",
        fake_packages / "lerobot/configs",
        fake_packages / "lerobot/policies",
        fake_packages / "lerobot/policies/groot",
    ):
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
    (fake_packages / "lerobot/policies/groot/configuration_groot.py").write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class GrootConfig:\n"
        "    type: str = 'groot'\n"
        "    pretrained_path: str | None = None\n",
        encoding="utf-8",
    )
    (fake_packages / "lerobot/configs/policies.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "from lerobot.policies.groot.configuration_groot import GrootConfig\n"
        "class PreTrainedConfig:\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, path, cli_overrides=None):\n"
        "        values=json.loads((Path(path)/'config.json').read_text())\n"
        "        extra=set(values)-set(GrootConfig.__dataclass_fields__)\n"
        "        if extra: raise TypeError(f'unexpected fields: {sorted(extra)}')\n"
        "        return GrootConfig(**values)\n",
        encoding="utf-8",
    )
    dist_info = fake_packages / "lerobot-0.4.3.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: lerobot\nVersion: 0.4.3\n", encoding="utf-8"
    )
    package_digest = hashlib.sha256()
    package_files = sorted(
        path for path in (fake_packages / "lerobot").rglob("*") if path.is_file()
    )
    for path in package_files:
        relative = path.relative_to(fake_packages / "lerobot").as_posix()
        package_digest.update(
            relative.encode()
            + b"\0"
            + hashlib.sha256(path.read_bytes()).hexdigest().encode()
            + b"\n"
        )
    package_tree_sha256 = package_digest.hexdigest()
    receipt_document = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_document.update(
        {
            "expected_lerobot_package_tree_sha256": package_tree_sha256,
            "expected_lerobot_package_file_count": len(package_files),
            "installed_lerobot_package_tree_sha256": package_tree_sha256,
            "installed_lerobot_package_file_count": len(package_files),
        }
    )
    receipt.write_text(
        json.dumps(receipt_document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import json,sys\n"
        "from pathlib import Path\n"
        "from checkpoint_compatibility import install_checkpoint_config_view\n"
        "checkpoint=Path(sys.argv[1]); sanitized=Path(sys.argv[2]); receipt=Path(sys.argv[3])\n"
        "original=checkpoint.joinpath('config.json').read_bytes()\n"
        "install_checkpoint_config_view("
        "checkpoint, sanitized, receipt, "
        f"expected_config_sha256={hashlib.sha256(raw).hexdigest()!r}, "
        f"expected_package_tree_sha256={package_tree_sha256!r}, "
        f"expected_package_file_count={len(package_files)!r})\n"
        "from lerobot.configs.policies import PreTrainedConfig\n"
        "try: PreTrainedConfig.from_pretrained(checkpoint.parent,cli_overrides={})\n"
        "except RuntimeError as error: wrong_path='unexpected path' in str(error)\n"
        "else: wrong_path=False\n"
        "cfg=PreTrainedConfig.from_pretrained(checkpoint, cli_overrides={})\n"
        "try: PreTrainedConfig.from_pretrained(checkpoint,cli_overrides={})\n"
        "except RuntimeError as error: repeated='more than once' in str(error)\n"
        "else: repeated=False\n"
        "cfg.pretrained_path=str(checkpoint)\n"
        "def make_policy(policy_cfg): return policy_cfg.pretrained_path\n"
        "def make_processors(*, policy_cfg, pretrained_path): return policy_cfg.pretrained_path,pretrained_path\n"
        "weights=make_policy(cfg); processors=make_processors(policy_cfg=cfg,pretrained_path=str(checkpoint))\n"
        "assert checkpoint.joinpath('config.json').read_bytes()==original\n"
        "print(json.dumps({'weights':weights,'processors':processors,'config':cfg.type,'wrong_path':wrong_path,'repeated':repeated}))\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            os.sys.executable,
            "-S",
            "-P",
            str(driver),
            str(checkpoint),
            str(sanitized),
            str(receipt),
        ],
        env={
            **os.environ,
            "PYTHONPATH": f"{fake_packages}:{CHECKPOINT_COMPATIBILITY_SHIM.parent}",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    assert loaded["weights"] == str(checkpoint)
    assert loaded["processors"] == [str(checkpoint), str(checkpoint)]
    assert loaded["wrong_path"] is True
    assert loaded["repeated"] is True
    assert json.loads((sanitized / "config.json").read_text()) == {"type": "groot"}
    assert json.loads((checkpoint / "config.json").read_text()) == raw_config

    cache_directory = fake_packages / "lerobot/__pycache__"
    cache_directory.mkdir(exist_ok=True)
    (cache_directory / "unexpected-package-data.json").write_text("{}\n", encoding="utf-8")
    tampered = subprocess.run(
        [
            os.sys.executable,
            "-S",
            "-P",
            str(driver),
            str(checkpoint),
            str(sanitized),
            str(receipt),
        ],
        env={
            **os.environ,
            "PYTHONPATH": f"{fake_packages}:{CHECKPOINT_COMPATIBILITY_SHIM.parent}",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert tampered.returncode != 0
    assert "compatibility receipt does not bind the exact config view" in tampered.stderr


def test_final_execution_verifier_rejects_tampered_compatibility_receipt(tmp_path: Path) -> None:
    from scripts.verify_native_reference_evaluator_gate import (
        NativeReferenceGateError,
        verify_native_reference_result,
    )

    bundle = _bundle()
    _materialize_artifacts(tmp_path, bundle)
    path = tmp_path / "evidence/checkpoint-compatibility-receipt.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["removed_fields"][0]["value"] = 0.2
    raw = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    bundle["identity"]["checkpoint_compatibility_receipt_sha256"] = hashlib.sha256(raw).hexdigest()  # type: ignore[index]

    with pytest.raises(NativeReferenceGateError, match="compatibility receipt contract"):
        verify_native_reference_result(bundle, bundle_root=tmp_path)


def test_prepare_checkpoint_compatibility_creates_one_exclusive_exact_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.verify_native_reference_evaluator_gate as gate

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    values = {
        "type": "groot",
        "hidden_size": 1024,
        "num_decay_steps": 4000,
        "decay_lr_ratio": 0.1,
    }
    raw = json.dumps(values, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    (checkpoint / "config.json").write_bytes(raw)
    package_root = tmp_path / "packages"
    module_root = package_root / "lerobot/policies/groot"
    module_root.mkdir(parents=True)
    for package in (
        package_root / "lerobot",
        package_root / "lerobot/policies",
        module_root,
    ):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (module_root / "configuration_groot.py").write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class GrootConfig:\n"
        "    type: str = 'groot'\n"
        "    hidden_size: int = 1024\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256()
    package_files = sorted(
        path for path in (package_root / "lerobot").rglob("*") if path.is_file()
    )
    for path in package_files:
        relative = path.relative_to(package_root / "lerobot").as_posix()
        digest.update(
            relative.encode()
            + b"\0"
            + hashlib.sha256(path.read_bytes()).hexdigest().encode()
            + b"\n"
        )

    class FakeDistribution:
        version = "0.4.3"

        @staticmethod
        def locate_file(path: object) -> Path:
            return package_root / str(path)

    monkeypatch.syspath_prepend(str(package_root))
    for name in tuple(sys.modules):
        if name == "lerobot" or name.startswith("lerobot."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(gate, "CHECKPOINT_CONFIG_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(gate, "LEROBOT_PACKAGE_TREE_SHA256", digest.hexdigest())
    monkeypatch.setattr(gate, "LEROBOT_PACKAGE_FILE_COUNT", len(package_files))
    monkeypatch.setattr(gate.importlib.metadata, "distribution", lambda name: FakeDistribution())
    sanitized = tmp_path / "view"
    receipt_path = tmp_path / "compatibility.json"

    receipt = gate.prepare_checkpoint_compatibility(checkpoint, sanitized, receipt_path)

    assert (checkpoint / "config.json").read_bytes() == raw
    assert json.loads((sanitized / "config.json").read_text()) == {
        "type": "groot",
        "hidden_size": 1024,
    }
    assert receipt["removed_fields"] == [
        {"key": "decay_lr_ratio", "value": 0.1},
        {"key": "num_decay_steps", "value": 4000},
    ]
    assert receipt["lerobot_wheel_sha256"] == "b08c1c15b2356bd4e658122deabfb9dacd2d7447de4a4327720991723d4edf2c"
    assert receipt["installed_lerobot_package_tree_sha256"] == digest.hexdigest()
    assert receipt["installed_lerobot_package_file_count"] == len(package_files)
    with pytest.raises(gate.NativeReferenceGateError, match="already exists"):
        gate.prepare_checkpoint_compatibility(checkpoint, sanitized, tmp_path / "second.json")


@pytest.mark.parametrize("mutation", ("modified", "extra", "pycache_extra"))
def test_prepare_checkpoint_compatibility_rejects_same_version_tampered_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    import scripts.verify_native_reference_evaluator_gate as gate

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    values = {
        "type": "groot",
        "num_decay_steps": 4000,
        "decay_lr_ratio": 0.1,
    }
    raw = json.dumps(values, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    (checkpoint / "config.json").write_bytes(raw)
    distribution_root = tmp_path / "site-packages"
    module_root = distribution_root / "lerobot/policies/groot"
    module_root.mkdir(parents=True)
    for package in (
        distribution_root / "lerobot",
        distribution_root / "lerobot/policies",
        module_root,
    ):
        (package / "__init__.py").write_text("", encoding="utf-8")
    config_module = module_root / "configuration_groot.py"
    config_module.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class GrootConfig:\n"
        "    type: str = 'groot'\n",
        encoding="utf-8",
    )

    def manifest() -> tuple[str, int]:
        digest = hashlib.sha256()
        files = sorted(
            path
            for path in (distribution_root / "lerobot").rglob("*")
            if path.is_file()
        )
        for path in files:
            relative = path.relative_to(distribution_root / "lerobot").as_posix()
            digest.update(
                relative.encode()
                + b"\0"
                + hashlib.sha256(path.read_bytes()).hexdigest().encode()
                + b"\n"
            )
        return digest.hexdigest(), len(files)

    expected_digest, expected_count = manifest()
    if mutation == "modified":
        config_module.write_text(config_module.read_text() + "TAMPERED = True\n", encoding="utf-8")
    elif mutation == "extra":
        (distribution_root / "lerobot/unexpected_plugin.py").write_text(
            "IMPORTABLE = True\n", encoding="utf-8"
        )
    else:
        cache_directory = distribution_root / "lerobot/__pycache__"
        cache_directory.mkdir()
        (cache_directory / "unexpected-package-data.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(distribution_root))
    for name in tuple(sys.modules):
        if name == "lerobot" or name.startswith("lerobot."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(gate, "CHECKPOINT_CONFIG_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(gate, "LEROBOT_PACKAGE_TREE_SHA256", expected_digest, raising=False)
    monkeypatch.setattr(gate, "LEROBOT_PACKAGE_FILE_COUNT", expected_count, raising=False)

    class FakeDistribution:
        version = "0.4.3"

        @staticmethod
        def locate_file(path: object) -> Path:
            return distribution_root / str(path)

    monkeypatch.setattr(gate.importlib.metadata, "distribution", lambda name: FakeDistribution())

    with pytest.raises(gate.NativeReferenceGateError, match="installed LeRobot package tree"):
        gate.prepare_checkpoint_compatibility(
            checkpoint, tmp_path / "view", tmp_path / "compatibility.json"
        )


def _write_peft_overlay_wheel(
    path: Path,
    *,
    version: str = "0.18.1",
    include_symbols: bool = True,
    unsafe_member: str | None = None,
    spoof_origin: bool = False,
) -> None:
    init = ""
    if include_symbols:
        init = "class LoraConfig: pass\ndef get_peft_model(*args, **kwargs): return args\n"
    if spoof_origin:
        init += "__file__ = '/untrusted/peft/__init__.py'\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("peft/__init__.py", init)
        archive.writestr(
            "peft-0.18.1.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: peft\nVersion: {version}\n",
        )
        if unsafe_member is not None:
            archive.writestr(unsafe_member, b"unsafe")


def _configure_peft_overlay_constants(
    monkeypatch: pytest.MonkeyPatch, wheel: Path
) -> None:
    import scripts.verify_native_reference_evaluator_gate as gate

    monkeypatch.setattr(gate, "PEFT_WHEEL_PATH", wheel.resolve())
    if wheel.exists():
        monkeypatch.setattr(gate, "PEFT_WHEEL_SIZE", wheel.stat().st_size)
        monkeypatch.setattr(
            gate, "PEFT_WHEEL_SHA256", hashlib.sha256(wheel.read_bytes()).hexdigest()
        )


def test_prepare_peft_overlay_rejects_absent_wrong_and_tampered_wheels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.verify_native_reference_evaluator_gate as gate

    missing = tmp_path / "peft-0.18.1-py3-none-any.whl"
    _configure_peft_overlay_constants(monkeypatch, missing)
    with pytest.raises(gate.NativeReferenceGateError, match="PEFT wheel is unavailable"):
        gate.prepare_peft_overlay(tmp_path / "missing-receipt.json")

    wrong = tmp_path / "wrong-name.whl"
    _write_peft_overlay_wheel(wrong)
    _configure_peft_overlay_constants(monkeypatch, wrong)
    with pytest.raises(gate.NativeReferenceGateError, match="filename"):
        gate.prepare_peft_overlay(tmp_path / "wrong-receipt.json")

    wheel = tmp_path / "peft-0.18.1-py3-none-any.whl"
    _write_peft_overlay_wheel(wheel)
    _configure_peft_overlay_constants(monkeypatch, wheel)
    tampered = bytearray(wheel.read_bytes())
    tampered[20] ^= 1
    wheel.write_bytes(tampered)
    with pytest.raises(gate.NativeReferenceGateError, match="digest"):
        gate.prepare_peft_overlay(tmp_path / "tampered-receipt.json")


def test_prepare_peft_overlay_rejects_unsafe_wheel_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.verify_native_reference_evaluator_gate as gate

    wheel = tmp_path / "peft-0.18.1-py3-none-any.whl"
    _write_peft_overlay_wheel(wheel, unsafe_member="../escape.py")
    _configure_peft_overlay_constants(monkeypatch, wheel)

    with pytest.raises(gate.NativeReferenceGateError, match="unsafe PEFT wheel member"):
        gate.prepare_peft_overlay(tmp_path / "receipt.json")


@pytest.mark.parametrize(
    ("version", "include_symbols", "message"),
    [
        ("0.18.0", True, "metadata"),
        ("0.18.1", False, "symbols"),
    ],
)
def test_prepare_peft_overlay_rejects_wrong_metadata_or_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    include_symbols: bool,
    message: str,
) -> None:
    import scripts.verify_native_reference_evaluator_gate as gate

    wheel = tmp_path / "peft-0.18.1-py3-none-any.whl"
    _write_peft_overlay_wheel(wheel, version=version, include_symbols=include_symbols)
    _configure_peft_overlay_constants(monkeypatch, wheel)

    with pytest.raises(gate.NativeReferenceGateError, match=message):
        gate.prepare_peft_overlay(tmp_path / "receipt.json")


def test_prepare_peft_overlay_rejects_wrong_zipimport_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.verify_native_reference_evaluator_gate as gate

    wheel = tmp_path / "peft-0.18.1-py3-none-any.whl"
    _write_peft_overlay_wheel(wheel, spoof_origin=True)
    _configure_peft_overlay_constants(monkeypatch, wheel)

    with pytest.raises(gate.NativeReferenceGateError, match="origin"):
        gate.prepare_peft_overlay(tmp_path / "receipt.json")


def test_prepare_peft_overlay_records_exact_zipimport_origin_and_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.verify_native_reference_evaluator_gate as gate

    wheel = tmp_path / "peft-0.18.1-py3-none-any.whl"
    receipt_path = tmp_path / "receipt.json"
    _write_peft_overlay_wheel(wheel)
    _configure_peft_overlay_constants(monkeypatch, wheel)

    receipt = gate.prepare_peft_overlay(receipt_path)

    assert receipt["wheel_path"] == str(wheel.resolve())
    assert receipt["peft_origin"] == f"{wheel.resolve()}/peft/__init__.py"
    assert receipt["peft_version"] == "0.18.1"
    assert receipt_path.read_bytes() == (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def test_execution_verifier_rejects_tampered_or_missing_peft_overlay_evidence(
    tmp_path: Path,
) -> None:
    from scripts.verify_native_reference_evaluator_gate import (
        NativeReferenceGateError,
        verify_native_reference_result,
    )

    document = _bundle()
    _materialize_artifacts(tmp_path, document)
    peft_receipt = {
        "schema_version": 1,
        "kind": "lehome_native_reference_peft_overlay_v1",
        "wheel_path": "/mnt/lehome/reference-native/dependencies/peft-0.18.1-py3-none-any.whl",
        "wheel_filename": "peft-0.18.1-py3-none-any.whl",
        "wheel_url": "https://files.pythonhosted.org/packages/b3/14/b4e3f574acf349ae6f61f9c000a77f97a3b315b4bb6ad03791e79ae4a568/peft-0.18.1-py3-none-any.whl",
        "wheel_size": 556960,
        "wheel_sha256": "0bf06847a3551e3019fc58c440cffc9a6b73e6e2962c95b52e224f77bbdb50f1",
        "distribution_name": "peft",
        "peft_version": "0.18.1",
        "peft_origin": "/mnt/lehome/reference-native/dependencies/peft-0.18.1-py3-none-any.whl/peft/__init__.py",
        "required_symbols": ["LoraConfig", "get_peft_model"],
    }
    path = tmp_path / "evidence/peft-overlay-receipt.json"
    raw = (json.dumps(peft_receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    identity = document["identity"]
    identity["peft_wheel_sha256"] = peft_receipt["wheel_sha256"]  # type: ignore[index]
    identity["peft_overlay_receipt_sha256"] = hashlib.sha256(raw).hexdigest()  # type: ignore[index]
    assert verify_native_reference_result(document, bundle_root=tmp_path)["status"] == "oracle_matched_pending_finalization"

    path.write_bytes(raw.replace(b"0.18.1", b"0.18.0", 1))
    with pytest.raises(NativeReferenceGateError, match="PEFT overlay"):
        verify_native_reference_result(document, bundle_root=tmp_path)

    path.unlink()
    with pytest.raises(NativeReferenceGateError, match="missing required artifact"):
        verify_native_reference_result(document, bundle_root=tmp_path)


def _write_flash_attention_overlay_wheel(
    path: Path,
    *,
    tag: str = "cp311-cp311-linux_x86_64",
    version: str = "2.8.3",
    distribution_name: str = "flash_attn",
    unsafe_member: str | None = None,
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("flash_attn/__init__.py", f"__version__ = {version!r}\n")
        archive.writestr(
            "flash_attn-2.8.3.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {distribution_name}\nVersion: {version}\n",
        )
        archive.writestr(
            "flash_attn-2.8.3.dist-info/WHEEL",
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}\n",
        )
        if unsafe_member is not None:
            archive.writestr(unsafe_member, b"unsafe")


def _configure_flash_attention_overlay_constants(
    monkeypatch: pytest.MonkeyPatch, wheel: Path
) -> None:
    import scripts.verify_native_reference_evaluator_gate as gate

    monkeypatch.setattr(gate, "FLASH_ATTENTION_WHEEL_PATH", wheel.resolve())
    if wheel.exists():
        monkeypatch.setattr(gate, "FLASH_ATTENTION_WHEEL_SIZE", wheel.stat().st_size)
        monkeypatch.setattr(
            gate,
            "FLASH_ATTENTION_WHEEL_SHA256",
            hashlib.sha256(wheel.read_bytes()).hexdigest(),
        )


def test_flash_attention_overlay_rejects_bad_wheel_identity_metadata_and_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.verify_native_reference_evaluator_gate as gate

    missing = tmp_path / "flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
    _configure_flash_attention_overlay_constants(monkeypatch, missing)
    with pytest.raises(gate.NativeReferenceGateError, match="FlashAttention wheel is unavailable"):
        gate.inspect_flash_attention_overlay()

    wrong = tmp_path / "wrong.whl"
    _write_flash_attention_overlay_wheel(wrong)
    _configure_flash_attention_overlay_constants(monkeypatch, wrong)
    with pytest.raises(gate.NativeReferenceGateError, match="filename"):
        gate.inspect_flash_attention_overlay()

    wheel = tmp_path / "flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
    _write_flash_attention_overlay_wheel(wheel, tag="cp311-cp311-manylinux_2_28_x86_64")
    _configure_flash_attention_overlay_constants(monkeypatch, wheel)
    with pytest.raises(gate.NativeReferenceGateError, match="platform tag"):
        gate.inspect_flash_attention_overlay()

    _write_flash_attention_overlay_wheel(wheel, unsafe_member="../escape.py")
    _configure_flash_attention_overlay_constants(monkeypatch, wheel)
    with pytest.raises(gate.NativeReferenceGateError, match="unsafe FlashAttention wheel member"):
        gate.inspect_flash_attention_overlay()

    _write_flash_attention_overlay_wheel(wheel, distribution_name="flash-attn")
    _configure_flash_attention_overlay_constants(monkeypatch, wheel)
    with pytest.raises(gate.NativeReferenceGateError, match="metadata"):
        gate.inspect_flash_attention_overlay()


def test_flash_attention_overlay_receipt_and_execution_evidence_are_identity_bound(
    tmp_path: Path,
) -> None:
    from scripts.verify_native_reference_evaluator_gate import (
        NativeReferenceGateError,
        verify_native_reference_result,
    )

    document = _bundle()
    _materialize_artifacts(tmp_path, document)
    assert verify_native_reference_result(document, bundle_root=tmp_path)["status"] == "oracle_matched_pending_finalization"

    path = tmp_path / "evidence" / "flash-attention-runtime-receipt.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(NativeReferenceGateError, match="FlashAttention"):
        verify_native_reference_result(document, bundle_root=tmp_path)


def test_flash_attention_preflight_binding_is_checked_at_final_execution_verification(
    tmp_path: Path,
) -> None:
    from scripts.verify_native_reference_evaluator_gate import (
        NativeReferenceGateError,
        verify_native_reference_result,
    )

    document = _bundle()
    _materialize_artifacts(tmp_path, document)
    preflight = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    preflight["flash_attention_runtime_receipt_sha256"] = "0" * 64
    (tmp_path / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")
    with pytest.raises(NativeReferenceGateError, match="preflight"):
        verify_native_reference_result(document, bundle_root=tmp_path)


def test_pynput_dummy_backend_is_preflight_bound_and_rejects_any_x11_or_control_drift(
    tmp_path: Path,
) -> None:
    from scripts.verify_native_reference_evaluator_gate import (
        NativeReferenceGateError,
        verify_native_reference_result,
    )

    document = _bundle()
    _materialize_artifacts(tmp_path, document)
    assert verify_native_reference_result(document, bundle_root=tmp_path)["status"] == "oracle_matched_pending_finalization"

    receipt = tmp_path / "evidence" / "pynput-backend-receipt.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["keyboard_control_started"] = True
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NativeReferenceGateError, match="pynput"):
        verify_native_reference_result(document, bundle_root=tmp_path)


def test_native_evaluator_uses_only_dummy_pynput_backend_without_xvfb_or_gui_install() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    run_stage = launcher.split("run_stage() {", 1)[1].split("\n}", 1)[0]

    assert "probe_pynput_dummy_backend" in launcher
    assert launcher.index("probe_pynput_dummy_backend") < launcher.index("run_stage 1")
    assert run_stage.count("PYNPUT_BACKEND=dummy") == 1
    assert "xvfb-run" not in launcher
    assert "Xvfb" not in launcher
    assert "apt-get" not in launcher


def test_native_flash_attention_contract_is_read_only_offline_then_installed_before_model_construction() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    wrapper = CONTAINER_WRAPPER.read_text(encoding="utf-8")
    runbook = RUNBOOK.read_text(encoding="utf-8")
    wheel = "flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"

    for text in (launcher, wrapper, runbook):
        assert wheel in text
        assert "cd1a45ebfc1731a13e55ad68e0c9ad92390ddfffba306f9222be67c6d5a805af" in text
    assert "validate-flash-attention-overlay" in launcher
    assert "prepare-flash-attention-overlay" in launcher
    assert "probe_flash_attention_runtime" in launcher
    assert launcher.index("probe_flash_attention_runtime") < launcher.index("run_stage 1")
    assert launcher.count("validate-flash-attention-overlay") >= 2
    assert "uv pip install --offline --no-deps --python /opt/lehome-challenge/.venv/bin/python" in launcher
    assert "--network host" in wrapper
    assert "--network none" not in wrapper
    assert "readonly" in wrapper
    assert "Dao-AILab/flash-attention/releases" in runbook
