from __future__ import annotations

import pytest

import numpy as np

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import types

import scripts.run_groot_flywheel_trial as trial_module
from scripts.run_groot_flywheel_trial import (
    _verify_runtime_asset_mount,
    _live_runtime_identity, _manifest_path, _scene_state_matches, build_parser, read_pinned_revision, run_randomization_acceptance,
    run_snapshot_acceptance, run_trial, validate_args, _validate_live_runtime_identity,
)


def test_evaluation_system_exit_is_not_reported_as_a_success() -> None:
    def exit_during_evaluation(_args, _app) -> None:
        raise SystemExit(0)

    with pytest.raises(RuntimeError, match="Isaac live phase exited before completion") as caught:
        trial_module._run_evaluation_or_raise(exit_during_evaluation, object(), object())

    assert isinstance(caught.value.__cause__, SystemExit)
    assert caught.value.__cause__.code == 0


def test_evaluation_exception_is_reported_before_kit_shutdown(capsys) -> None:
    def fail_during_evaluation(_args, _app) -> None:
        raise RuntimeError("unreadable garment displayColor")

    with pytest.raises(RuntimeError, match="unreadable garment displayColor"):
        trial_module._run_evaluation_or_raise(fail_during_evaluation, object(), object())

    assert "unreadable garment displayColor" in capsys.readouterr().err


def test_pre_evaluation_exception_is_reported_before_kit_shutdown(capsys) -> None:
    with pytest.raises(RuntimeError, match="asset checkout is dirty"):
        with trial_module._visible_kit_exception_boundary():
            raise RuntimeError("asset checkout is dirty")

    assert "asset checkout is dirty" in capsys.readouterr().err


def test_policy_artifact_sha256_hashes_the_index_and_every_referenced_shard(tmp_path) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    shards = {
        "model-00001-of-00002.safetensors": b"first shard",
        "model-00002-of-00002.safetensors": b"second shard",
    }
    index = {
        "metadata": {"total_size": sum(len(payload) for payload in shards.values())},
        "weight_map": {
            "model.layer.0": "model-00001-of-00002.safetensors",
            "model.layer.1": "model-00002-of-00002.safetensors",
        },
    }
    (policy / "model.safetensors.index.json").write_text(
        json.dumps(index, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, payload in shards.items():
        (policy / name).write_bytes(payload)
    files = ("model.safetensors.index.json", *sorted(shards))
    manifest = {
        "files": [
            {
                "path": name,
                "sha256": hashlib.sha256((policy / name).read_bytes()).hexdigest(),
            }
            for name in files
        ],
        "schema_version": 1,
    }
    expected = hashlib.sha256(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()

    assert trial_module.policy_artifact_sha256(policy) == expected
    (policy / "model-00002-of-00002.safetensors").write_bytes(b"changed second shard")
    assert trial_module.policy_artifact_sha256(policy) != expected


@pytest.mark.parametrize(
    "shard_names",
    (
        ("weights.safetensors",),
        ("../model-00001-of-00001.safetensors",),
        (
            "model-00001-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ),
    ),
)
def test_policy_artifact_sha256_rejects_unsafe_or_incomplete_shard_sets(
    tmp_path,
    shard_names,
) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    f"model.layer.{index}": name
                    for index, name in enumerate(shard_names)
                },
            }
        ),
        encoding="utf-8",
    )
    for name in shard_names:
        if Path(name).name == name:
            (policy / name).write_bytes(b"shard")

    with pytest.raises(ValueError, match="shard|index"):
        trial_module.policy_artifact_sha256(policy)


def test_policy_artifact_sha256_rejects_unreferenced_or_ambiguous_weights(tmp_path) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {"model.layer": "model-00001-of-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    (policy / "model-00001-of-00001.safetensors").write_bytes(b"referenced")
    (policy / "model-00002-of-00002.safetensors").write_bytes(b"unreferenced")

    with pytest.raises(ValueError, match="unreferenced"):
        trial_module.policy_artifact_sha256(policy)

    (policy / "model-00002-of-00002.safetensors").unlink()
    (policy / "model.safetensors").write_bytes(b"monolithic")
    with pytest.raises(ValueError, match="ambiguous"):
        trial_module.policy_artifact_sha256(policy)


def test_policy_artifact_sha256_rejects_monolithic_weights_with_loose_canonical_shard(tmp_path) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "model.safetensors").write_bytes(b"monolithic")
    (policy / "model-00001-of-00001.safetensors").write_bytes(b"loose shard")

    with pytest.raises(ValueError, match="ambiguous"):
        trial_module.policy_artifact_sha256(policy)


def test_policy_artifact_sha256_rejects_monolithic_weights_with_dangling_index(tmp_path) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "model.safetensors").write_bytes(b"monolithic")
    (policy / "model.safetensors.index.json").symlink_to("missing-index.json")

    with pytest.raises(ValueError, match="ambiguous|invalid"):
        trial_module.policy_artifact_sha256(policy)


def test_policy_artifact_sha256_rejects_indexed_weights_with_dangling_monolithic(tmp_path) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {"model.layer": "model-00001-of-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    (policy / "model-00001-of-00001.safetensors").write_bytes(b"shard")
    (policy / "model.safetensors").symlink_to("missing-model.safetensors")

    with pytest.raises(ValueError, match="ambiguous|invalid"):
        trial_module.policy_artifact_sha256(policy)


def test_live_execution_identity_rejects_mismatched_code_policy_or_container(tmp_path, monkeypatch) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "model.safetensors").write_bytes(b"policy")
    revision = tmp_path / "revision.txt"
    revision.write_text("a" * 40 + "\n", encoding="utf-8")
    args = build_parser().parse_args([
        "--policy-path", str(policy), "--policy-revision-file", str(revision), "--garment", "Pant_Long_Seen_0",
        "--episode-id", "identity", "--policy-repo", "org/policy", "--policy-step", "1",
        "--code-revision", "b" * 40, "--asset-revision", "c" * 40, "--simulator-version", "isaac",
        "--category", "pant_long", "--release-stage", "seen", "--policy-artifact-sha256", "d" * 64,
        "--image-identity", "sha256:" + "e" * 64,
    ])

    with pytest.raises(ValueError, match="code revision"):
        trial_module._validate_live_execution_identity(
            args,
            code_identity_reader=lambda _args: "f" * 40,
            policy_identity_reader=lambda _args: ("a" * 40, "d" * 64),
            image_identity_reader=lambda: "sha256:" + "e" * 64,
        )
    with pytest.raises(ValueError, match="policy revision"):
        trial_module._validate_live_execution_identity(
            args,
            code_identity_reader=lambda _args: "b" * 40,
            policy_identity_reader=lambda _args: ("f" * 40, "d" * 64),
            image_identity_reader=lambda: "sha256:" + "e" * 64,
        )
    with pytest.raises(ValueError, match="container image"):
        trial_module._validate_live_execution_identity(
            args,
            code_identity_reader=lambda _args: "b" * 40,
            policy_identity_reader=lambda _args: ("a" * 40, "d" * 64),
            image_identity_reader=lambda: "sha256:" + "f" * 64,
        )


def test_runtime_image_identity_requires_a_separately_injected_oci_digest(monkeypatch) -> None:
    monkeypatch.delenv("LEHOME_FLYWHEEL_IMAGE_IDENTITY", raising=False)
    with pytest.raises(ValueError, match="unavailable"):
        trial_module._runtime_container_image_identity()

    expected = "sha256:" + "e" * 64
    monkeypatch.setenv("LEHOME_FLYWHEEL_IMAGE_IDENTITY", expected)
    assert trial_module._runtime_container_image_identity() == expected


def test_app_launcher_argument_registration_isolated_from_wrapper_argv(monkeypatch) -> None:
    original_argv = ["trial.py", "--garment", "Top_Long_Seen_0"]
    monkeypatch.setattr(sys, "argv", original_argv)
    observed: list[list[str]] = []

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(_parser) -> None:
            observed.append(list(sys.argv))

    trial_module._add_app_launcher_args(argparse.ArgumentParser(), AppLauncher)

    assert observed == [["trial.py"]]
    assert sys.argv is original_argv


def test_app_launcher_argument_registration_restores_argv_after_error(monkeypatch) -> None:
    original_argv = ["trial.py", "--garment", "Top_Long_Seen_0"]
    monkeypatch.setattr(sys, "argv", original_argv)

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(_parser) -> None:
            assert sys.argv == ["trial.py"]
            raise RuntimeError("registration failed")

    with pytest.raises(RuntimeError, match="registration failed"):
        trial_module._add_app_launcher_args(argparse.ArgumentParser(), AppLauncher)

    assert sys.argv is original_argv


def test_acceptance_launch_forwards_headless_to_isaac_app(monkeypatch, tmp_path) -> None:
    launched: list[bool] = []
    observed_argv: list[list[str]] = []

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            observed_argv.append(list(sys.argv))
            parser.add_argument("--headless", action="store_true")

    utils = types.ModuleType("scripts.utils"); utils.__path__ = []
    common = types.ModuleType("scripts.utils.common")
    common.launch_app_from_args = lambda args: launched.append(args.headless) or object()
    common.close_app = lambda _app: None
    app_module = types.ModuleType("isaaclab.app"); app_module.AppLauncher = AppLauncher
    isaaclab = types.ModuleType("isaaclab"); isaaclab.__path__ = []
    for name, module in {"scripts.utils": utils, "scripts.utils.common": common, "isaaclab": isaaclab, "isaaclab.app": app_module}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(trial_module, "run_snapshot_acceptance", lambda _args: 0)
    args = build_parser().parse_args(["--snapshot-roundtrip-only", "--garment", "Pant_Long_Seen_0", "--headless", "--output-root", str(tmp_path)])
    original_argv = ["trial.py", "--garment", "Pant_Long_Seen_0"]
    monkeypatch.setattr(sys, "argv", original_argv)

    assert run_trial(args, runtime_preflight=lambda: None) == 0
    assert launched == [True]
    assert observed_argv == [["trial.py"]]
    assert sys.argv is original_argv


def test_trial_cli_requires_pinned_policy_and_existing_matrix(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(["--policy-path", str(tmp_path / "missing"), "--policy-revision", "main"])
    with pytest.raises(ValueError, match="pinned"):
        validate_args(args)


def test_production_trial_rejects_a_non_digest_image_identity(tmp_path) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    args = build_parser().parse_args([
        "--policy-path", str(policy), "--policy-revision", "a" * 40,
        "--episode-id", "identity-check", "--policy-repo", "org/policy", "--policy-step", "1",
        "--code-revision", "b" * 40, "--asset-revision", "c" * 40,
        "--simulator-version", "isaac-sim-5.1", "--garment", "Pant_Long_Seen_0",
        "--category", "pant_long", "--release-stage", "seen", "--policy-artifact-sha256", "d" * 64,
        "--image-identity", "sha256:image",
    ])

    with pytest.raises(ValueError, match="OCI SHA-256"):
        validate_args(args)


def test_live_runtime_identity_reads_installed_simulator_and_clean_assets_checkout(monkeypatch, tmp_path) -> None:
    checkout = tmp_path / "assets-checkout"
    assets = checkout / "Release"
    assets.mkdir(parents=True)
    (assets / "garment.usd").write_bytes(b"materialized")
    args = argparse.Namespace(release_assets_root=assets)
    commands: list[tuple[str, ...]] = []

    def git_run(command, **_kwargs):
        commands.append(command)
        if command[-1] == "--show-toplevel":
            return types.SimpleNamespace(returncode=0, stdout=str(checkout) + "\n")
        if command[-1] == "HEAD":
            return types.SimpleNamespace(returncode=0, stdout=("c" * 40) + "\n")
        if command[3:5] == ("ls-files", "-z"):
            return types.SimpleNamespace(returncode=0, stdout="Release/garment.usd\0")
        if command[3:5] == ("lfs", "ls-files"):
            return types.SimpleNamespace(returncode=0, stdout=("f" * 64) + " * Release/garment.usd\n")
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(trial_module, "package_version", lambda package: "5.1.0.0")
    monkeypatch.setattr(trial_module, "_code_checkout_root", Path.cwd)
    monkeypatch.setattr(trial_module.subprocess, "run", git_run)
    monkeypatch.setattr(trial_module, "_verify_runtime_asset_mount", lambda *_args: None)

    assert _live_runtime_identity(args, object()) == ("5.1.0.0", "c" * 40)
    assert commands == [
        ("git", "-C", str(assets), "rev-parse", "--show-toplevel"),
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        ("git", "-C", str(checkout), "status", "--porcelain=v1", "--untracked-files=all"),
        ("git", "-C", str(checkout), "ls-files", "-z"),
        ("git", "-C", str(checkout), "lfs", "ls-files", "--long"),
    ]


@pytest.mark.parametrize(
    ("git_responses", "message"),
    [
        ({"HEAD": "d" * 40}, "asset revision"),
        ({"status": "?? Release/untracked.usd\n"}, "clean Git checkout"),
        ({"lfs": ("f" * 64) + " - Release/garment.usd\n"}, "Git LFS"),
        ({"lfs_returncode": 1}, "Git LFS"),
    ],
)
def test_live_runtime_identity_rejects_unpinned_dirty_or_unmaterialized_assets(monkeypatch, tmp_path, git_responses, message) -> None:
    checkout = tmp_path / "assets-checkout"
    assets = checkout / "Release"
    assets.mkdir(parents=True)
    (assets / "garment.usd").write_bytes(b"materialized")
    args = argparse.Namespace(
        release_assets_root=assets,
        asset_revision="c" * 40,
        image_identity="sha256:" + "a" * 64,
        simulator_version="5.1.0.0",
    )

    def git_run(command, **_kwargs):
        if command[-1] == "--show-toplevel":
            return types.SimpleNamespace(returncode=0, stdout=str(checkout) + "\n")
        if command[-1] == "HEAD":
            return types.SimpleNamespace(returncode=0, stdout=git_responses.get("HEAD", "c" * 40) + "\n")
        if command[3:5] == ("status", "--porcelain=v1"):
            return types.SimpleNamespace(returncode=0, stdout=git_responses.get("status", ""))
        if command[3:5] == ("ls-files", "-z"):
            return types.SimpleNamespace(returncode=0, stdout="Release/garment.usd\0")
        if command[3:5] == ("lfs", "ls-files"):
            return types.SimpleNamespace(
                returncode=git_responses.get("lfs_returncode", 0),
                stdout=git_responses.get("lfs", ("f" * 64) + " * Release/garment.usd\n"),
            )
        raise AssertionError(command)

    monkeypatch.setattr(trial_module, "package_version", lambda _package: "5.1.0.0")
    monkeypatch.setattr(trial_module, "_code_checkout_root", Path.cwd)
    monkeypatch.setattr(trial_module.subprocess, "run", git_run)
    monkeypatch.setattr(trial_module, "_verify_runtime_asset_mount", lambda *_args: None)

    with pytest.raises(ValueError, match=message):
        _validate_live_runtime_identity(args, object(), runtime_identity_reader=_live_runtime_identity)


def test_live_runtime_identity_rejects_the_parent_code_checkout(monkeypatch, tmp_path) -> None:
    assets = tmp_path / "Assets" / "objects" / "Challenge_Garment" / "Release"
    assets.mkdir(parents=True)
    monkeypatch.setattr(trial_module, "_code_checkout_root", lambda: tmp_path)
    monkeypatch.setattr(trial_module, "package_version", lambda _package: "5.1.0.0")

    with pytest.raises(ValueError, match="dedicated asset checkout"):
        _live_runtime_identity(argparse.Namespace(release_assets_root=assets), object())


def test_runtime_asset_mount_requires_every_legacy_loader_path_to_resolve_to_the_verified_checkout(monkeypatch, tmp_path) -> None:
    checkout = tmp_path / "assets-checkout"
    runtime_assets = tmp_path / "code-checkout" / "Assets"
    for name in ("objects", "robots", "scenes", "textures"):
        (checkout / name).mkdir(parents=True, exist_ok=True)
        runtime_assets.mkdir(parents=True, exist_ok=True)
        (runtime_assets / name).symlink_to(checkout / name, target_is_directory=True)
    release = checkout / "objects" / "Challenge_Garment" / "Release"
    release.mkdir(parents=True)
    monkeypatch.setattr(trial_module, "_runtime_assets_root", lambda: runtime_assets)

    _verify_runtime_asset_mount(checkout, release)
    (runtime_assets / "textures").unlink()
    (runtime_assets / "textures").mkdir()
    with pytest.raises(ValueError, match="symlinked"):
        _verify_runtime_asset_mount(checkout, release)


@pytest.mark.parametrize(
    ("observed_simulator", "observed_assets", "message"),
    [
        ("isaac-sim-5.0", "c" * 40, "simulator version"),
        ("isaac-sim-5.1", "e" * 40, "asset revision"),
    ],
)
def test_production_trial_rejects_live_runtime_identity_mismatch(
    tmp_path,
    observed_simulator: str,
    observed_assets: str,
    message: str,
) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    args = build_parser().parse_args([
        "--policy-path", str(policy), "--policy-revision", "a" * 40,
        "--episode-id", "identity-check", "--policy-repo", "org/policy", "--policy-step", "1",
        "--code-revision", "b" * 40, "--asset-revision", "c" * 40,
        "--simulator-version", "isaac-sim-5.1", "--release-assets-root", str(tmp_path / "asset-checkout" / "Release"), "--garment", "Pant_Long_Seen_0",
        "--category", "pant_long", "--release-stage", "seen", "--policy-artifact-sha256", "d" * 64,
        "--image-identity", "sha256:" + "e" * 64,
    ])

    with pytest.raises(ValueError, match=message):
        _validate_live_runtime_identity(
            args,
            object(),
            runtime_identity_reader=lambda _args, _app: (observed_simulator, observed_assets),
        )


def test_trial_cli_reads_revision_from_a_regular_file(tmp_path) -> None:
    revision = tmp_path / "revision.txt"
    revision.write_text("a" * 40 + "\n", encoding="utf-8")
    assert read_pinned_revision(revision) == "a" * 40


def test_simulator_acceptance_requires_an_explicit_release_garment() -> None:
    args = build_parser().parse_args(["--render-randomization-sheet"])
    with pytest.raises(ValueError, match="--garment"):
        validate_args(args)


class FakeEnv:
    garment_name = "Pant_Long_Seen_0"
    robot_position = np.arange(12, dtype=np.float32)
    robot_velocity = np.zeros(12, dtype=np.float32)
    cloth_position = np.arange(30, dtype=np.float32).reshape(10, 3)
    cloth_velocity = np.ones((10, 3), dtype=np.float32)
    rng_state = {"seed": 42}
    def reset(self): self.robot_position[:] = -1
    def render(self): pass
    def flywheel_capture_state(self): return {name: getattr(self, name) for name in ("robot_position", "robot_velocity", "cloth_position", "cloth_velocity", "rng_state", "garment_name")}
    def flywheel_restore_state(self, snapshot):
        self.robot_position = np.asarray(snapshot.robot_position, dtype=np.float32)
        self.robot_velocity = np.asarray(snapshot.robot_velocity, dtype=np.float32)
        self.cloth_position = np.asarray(snapshot.cloth_position, dtype=np.float32)
        self.cloth_velocity = np.asarray(snapshot.cloth_velocity, dtype=np.float32)
        self.rng_state = snapshot.rng_state
    def _get_observations(self): return {f"observation.images.{name}": np.zeros((4, 4, 3), dtype=np.uint8) for name in ("top_rgb", "left_rgb", "right_rgb")}
    def apply_flywheel_randomization(self, record):
        receipt = dict(record.values)
        if receipt: receipt.update(table_texture_path="/assets/table.png", table_shader_input="file")
        return receipt
    def close(self): pass


class IndependentRandomizationEnv(FakeEnv):
    def __init__(self) -> None:
        self.garment_name = "Pant_Long_Seen_0"
        self.robot_position = np.arange(12, dtype=np.float32)
        self.robot_velocity = np.zeros(12, dtype=np.float32)
        self.cloth_position = np.arange(30, dtype=np.float32).reshape(10, 3)
        self.cloth_velocity = np.ones((10, 3), dtype=np.float32)
        self.rng_state = {"seed": 42}
        self.baseline_light = 100.0
        self.light = self.baseline_light

    def apply_flywheel_randomization(self, record):
        receipt = super().apply_flywheel_randomization(record)
        if receipt:
            # A reused environment would expose compounded light state here.
            self.light *= receipt["light_intensity_scale"]
            receipt["light_intensity_scale"] = self.light / self.baseline_light
        return receipt


class VisualSnapshotEnv(FakeEnv):
    def __init__(self) -> None:
        self.garment_name = "Pant_Long_Seen_0"
        self.robot_position = np.arange(12, dtype=np.float32)
        self.robot_velocity = np.zeros(12, dtype=np.float32)
        self.cloth_position = np.arange(30, dtype=np.float32).reshape(10, 3)
        self.cloth_velocity = np.ones((10, 3), dtype=np.float32)
        self.rng_state = {"seed": 42}
        self.scene_state = {"camera_world_poses": [{"position": [1.0, 2.0, 3.0], "orientation": [1.0, 0.0, 0.0, 0.0]}] * 3, "robot_root_poses": [{"position": [4.0, 5.0, 6.0], "orientation": [0.0, 1.0, 0.0, 0.0]}] * 2, "light_intensity": 120.0, "light_color": [0.75, 0.75, 0.75], "table_texture_path": "/assets/1.png", "table_shader_input": "file", "garment_display_color": [[0.8, 0.7, 0.6]], "garment_reset_pose": [0.0] * 6}

    def reset(self):
        self.robot_position[:] = -1
        self.scene_state = {"light_intensity": 0.0}

    def flywheel_capture_state(self):
        state = super().flywheel_capture_state()
        state["scene_state"] = self.scene_state
        return state

    def flywheel_restore_state(self, snapshot):
        super().flywheel_restore_state(snapshot)
        self.scene_state = snapshot.scene_state

    def _get_observations(self):
        value = int(self.scene_state.get("light_intensity", 0.0))
        return {f"observation.images.{name}": np.full((4, 4, 3), value, dtype=np.uint8) for name in ("top_rgb", "left_rgb", "right_rgb")}


def test_acceptance_modes_write_real_orchestrated_reports(tmp_path) -> None:
    args = build_parser().parse_args(["--snapshot-roundtrip-only", "--garment", "Pant_Long_Seen_0", "--output-root", str(tmp_path)])
    assert run_snapshot_acceptance(args, env_factory=lambda _: FakeEnv()) == 0
    assert (tmp_path / "snapshot-acceptance.json").is_file()
    args = build_parser().parse_args(["--render-randomization-sheet", "--garment", "Pant_Long_Seen_0", "--output-root", str(tmp_path), "--strategies", "canonical", "mild", "strong"])
    writes = []
    assert run_randomization_acceptance(args, env_factory=lambda _: FakeEnv(), image_writer=lambda path, frame: writes.append(path)) == 0
    assert len(writes) == 9
    assert (tmp_path / "randomization-receipts.json").is_file()


def test_snapshot_acceptance_verifies_scene_state_and_render_round_trip(tmp_path) -> None:
    args = build_parser().parse_args(["--snapshot-roundtrip-only", "--garment", "Pant_Long_Seen_0", "--output-root", str(tmp_path)])
    assert run_snapshot_acceptance(args, env_factory=lambda _: VisualSnapshotEnv()) == 0
    report = __import__("json").loads((tmp_path / "snapshot-acceptance.json").read_text())
    assert report["scene_state_match"] is True
    assert {"camera", "robot_root", "light", "material"}.issubset(report["restore_coverage"])


def test_scene_state_comparison_allows_float32_rounding_but_not_asset_identity_changes() -> None:
    expected = {"light_intensity": 1200.0, "table_texture_path": "/assets/1.png"}
    assert _scene_state_matches(expected, {"light_intensity": 1200.00001, "table_texture_path": "/assets/1.png"})
    assert not _scene_state_matches(expected, {"light_intensity": 1200.0, "table_texture_path": "/assets/2.png"})


def test_randomization_acceptance_uses_an_independent_canonical_baseline_per_strategy(tmp_path) -> None:
    args = build_parser().parse_args([
        "--render-randomization-sheet", "--garment", "Pant_Long_Seen_0", "--output-root", str(tmp_path),
        "--strategies", "canonical", "mild", "strong",
    ])
    environments: list[IndependentRandomizationEnv] = []

    def factory(_):
        environment = IndependentRandomizationEnv()
        environments.append(environment)
        return environment

    assert run_randomization_acceptance(args, env_factory=factory, image_writer=lambda *_: None) == 0
    assert len(environments) == 3
    report = __import__("json").loads((tmp_path / "randomization-receipts.json").read_text())
    for entry in report["strategies"]:
        if not entry["sampled"]:
            assert entry["receipt"] == {}
            continue
        assert entry["receipt"].get("light_intensity_scale") == pytest.approx(
            entry["sampled"]["light_intensity_scale"]
        )


def test_manifest_creation_is_atomic_and_immutable(tmp_path) -> None:
    args = argparse.Namespace(output_root=tmp_path, policy_path=tmp_path, episode_id="episode-1", policy_repo="org/policy", policy_step=1, code_revision="b" * 40, asset_revision="c" * 40, simulator_version="isaac", garment="Pant_Long_Seen_0", category="pant_long", release_stage="seen", seed=1, strategy="canonical", policy_artifact_sha256="d" * 64, image_identity="sha256:image")
    path = _manifest_path(args, "a" * 40)
    assert _manifest_path(args, "a" * 40) == path
    args.seed = 2
    with pytest.raises(ValueError, match="overwrite"):
        _manifest_path(args, "a" * 40)
    assert not list(tmp_path.glob(".*.tmp"))


def test_normal_trial_pins_one_assigned_garment_and_rejects_episode_id_reuse(tmp_path, capsys) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    base = [
        "--policy-path", str(policy), "--policy-revision", "a" * 40,
        "--episode-id", "only-once", "--policy-repo", "org/policy", "--policy-step", "1",
        "--code-revision", "b" * 40, "--asset-revision", "c" * 40,
        "--simulator-version", "isaac", "--category", "pant_long", "--release-stage", "seen",
        "--policy-artifact-sha256", "d" * 64, "--image-identity", "sha256:image",
        "--output-root", str(tmp_path / "run"), "--dry-run",
    ]
    first = build_parser().parse_args([*base, "--garment", "Pant_Long_Seen_0"])
    assert run_trial(first) == 0
    launched = __import__("json").loads(capsys.readouterr().out)
    command = launched["command"]
    assert command[command.index("--garment_name") + 1] == "Pant_Long_Seen_0"
    assert command.count("--num_episodes") == 1
    assert command[command.index("--num_episodes") + 1] == "1"
    # Per-episode immutable evidence is activated by the flywheel manifest;
    # requesting generic evaluator video here would make every worker target
    # the shared output_root/videos/{success,failure}/episode0_* paths.
    assert "--save_video" not in command
    assert "--video_dir" not in command

    conflicting = build_parser().parse_args([*base, "--garment", "Pant_Long_Seen_1"])
    with pytest.raises(ValueError, match="overwrite"):
        run_trial(conflicting)


def test_real_evaluation_parser_accepts_trial_garment_name_without_abbreviation() -> None:
    parser_path = Path(__file__).parents[2] / "scripts" / "utils" / "parser.py"
    spec = importlib.util.spec_from_file_location("real_evaluation_parser", parser_path)
    assert spec is not None and spec.loader is not None
    parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_module)

    parsed = parser_module.setup_eval_parser().parse_args(
        ["--garment_name", "Top_Long_Seen_0"]
    )

    assert parsed.garment_name == "Top_Long_Seen_0"


def test_parallel_flywheel_trial_commands_never_target_shared_legacy_videos(tmp_path, capsys) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    output_root = tmp_path / "run"
    commands = []
    manifests = []
    for episode_id in ("parallel-worker-a", "parallel-worker-b"):
        args = build_parser().parse_args([
            "--policy-path", str(policy), "--policy-revision", "a" * 40,
            "--episode-id", episode_id, "--policy-repo", "org/policy", "--policy-step", "1",
            "--code-revision", "b" * 40, "--asset-revision", "c" * 40,
            "--simulator-version", "isaac", "--garment", "Pant_Long_Seen_0",
            "--category", "pant_long", "--release-stage", "seen",
            "--policy-artifact-sha256", "d" * 64, "--image-identity", "sha256:image",
            "--output-root", str(output_root), "--dry-run",
        ])
        assert run_trial(args) == 0
        launched = __import__("json").loads(capsys.readouterr().out)
        commands.append(launched["command"])
        manifests.append(Path(launched["manifest"]))

    # The manifest is what activates AutonomousRecorder's immutable raw/<id>/videos
    # evidence; neither parallel worker may request generic episode0_* videos.
    assert manifests == [
        output_root / "flywheel-manifest-parallel-worker-a.json",
        output_root / "flywheel-manifest-parallel-worker-b.json",
    ]
    assert all("--save_video" not in command for command in commands)
    assert all("--video_dir" not in command for command in commands)
    assert all(str(output_root / "videos") not in command for command in commands)


def test_normal_trial_runs_one_manifest_garment_through_the_evaluation_boundary(monkeypatch, tmp_path) -> None:
    policy = tmp_path / "policy"; policy.mkdir()
    launched: list[tuple[str, str, int]] = []
    app_launcher_argv: list[list[str]] = []

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(_parser):
            app_launcher_argv.append(list(sys.argv))

    def setup_eval_parser():
        parser = argparse.ArgumentParser(add_help=False)
        for name in ("policy_type", "policy_path", "garment_type", "max_steps", "seed", "task", "device", "video_dir", "garment_name"):
            parser.add_argument(f"--{name}")
        parser.add_argument("--num_episodes", type=int)
        parser.add_argument("--save_video", action="store_true")
        parser.add_argument("--headless", action="store_true")
        return parser

    def evaluate(args, _app):
        manifest = __import__("json").loads(Path(args.flywheel_manifest).read_text())
        identity = manifest["identity"]
        assert args.num_episodes == 1
        assert args.garment_name == identity["garment_name"] == manifest["garment"]
        assert args.save_video is False
        assert args.video_dir is None
        launched.append((args.garment_name, identity["episode_id"], args.num_episodes))

    utils = types.ModuleType("scripts.utils"); utils.__path__ = []
    common = types.ModuleType("scripts.utils.common")
    common.launch_app_from_args = lambda _args: object()
    common.close_app = lambda _app: None
    parser_module = types.ModuleType("scripts.utils.parser"); parser_module.setup_eval_parser = setup_eval_parser
    evaluation_module = types.ModuleType("scripts.utils.evaluation"); evaluation_module.eval = evaluate
    app_module = types.ModuleType("isaaclab.app"); app_module.AppLauncher = AppLauncher
    isaaclab = types.ModuleType("isaaclab"); isaaclab.__path__ = []
    lehome_tasks = types.ModuleType("lehome.tasks"); lehome_tasks.__path__ = []
    bedroom = types.ModuleType("lehome.tasks.bedroom")
    for name, module in {
        "scripts.utils": utils, "scripts.utils.common": common, "scripts.utils.parser": parser_module,
        "scripts.utils.evaluation": evaluation_module, "isaaclab": isaaclab, "isaaclab.app": app_module,
        "lehome.tasks": lehome_tasks, "lehome.tasks.bedroom": bedroom,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    args = build_parser().parse_args([
        "--policy-path", str(policy), "--policy-revision", "a" * 40, "--garment", "Pant_Long_Seen_0",
        "--episode-id", "isolated-worker", "--policy-repo", "org/policy", "--policy-step", "1",
        "--code-revision", "b" * 40, "--asset-revision", "c" * 40, "--simulator-version", "isaac",
        "--release-assets-root", str(tmp_path / "asset-checkout" / "Release"),
        "--category", "pant_long", "--release-stage", "seen", "--policy-artifact-sha256", "d" * 64,
        "--image-identity", "sha256:" + "e" * 64, "--output-root", str(tmp_path / "run"),
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "f" * 40,
        "--groot-python", str(tmp_path / "python3.10"), "--policy-server-port", "5511",
        "--policy-server-readiness-timeout", "1", "--policy-server-request-timeout", "1",
        "--policy-server-termination-grace", "1", "--policy-server-log", str(tmp_path / "server.log"),
        "--device", "cuda:0",
    ])
    monkeypatch.setattr(trial_module, "validate_policy_server_runtime", lambda _args: {"groot_revision": "f" * 40, "python_version": "3.10.18", "python_path": "python"})
    monkeypatch.setattr(trial_module, "_require_free_loopback_port", lambda _port: None)
    monkeypatch.setattr(trial_module, "_await_policy_server_ready", lambda *_args, **_kwargs: None)
    class Supervisor:
        def install_signal_handlers(self): pass
        def close(self): pass
        def restore_signal_handlers(self): pass
    monkeypatch.setattr(trial_module, "_spawn_policy_server", lambda *_args, **_kwargs: Supervisor())
    original_argv = ["trial.py", "--garment", "Pant_Long_Seen_0"]
    monkeypatch.setattr(sys, "argv", original_argv)
    with pytest.raises(ValueError, match="simulator version"):
        run_trial(
            args,
            runtime_identity_reader=lambda _args, _app: ("wrong-runtime", "c" * 40),
            execution_identity_validator=lambda _args: None,
            runtime_preflight=lambda: None,
        )
    assert launched == []
    assert not (tmp_path / "run" / "flywheel-manifest-isolated-worker.json").exists()
    assert run_trial(
        args,
        runtime_identity_reader=lambda _args, _app: ("isaac", "c" * 40),
        execution_identity_validator=lambda _args: None,
        runtime_preflight=lambda: None,
    ) == 0
    assert launched == [("Pant_Long_Seen_0", "isolated-worker", 1)]
    assert app_launcher_argv == [["trial.py"], ["trial.py"]]
    assert sys.argv is original_argv


def test_production_trial_checks_host_before_policy_server_app_launcher_or_output(tmp_path) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    output_root = tmp_path / "output"
    args = build_parser().parse_args([
        "--policy-path", str(policy), "--policy-revision", "a" * 40,
        "--garment", "Pant_Long_Seen_0", "--episode-id", "preflight-order",
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "b" * 40,
        "--asset-revision", "c" * 40, "--simulator-version", "5.1.0.0",
        "--category", "pant_long", "--release-stage", "seen", "--policy-artifact-sha256", "d" * 64,
        "--image-identity", "sha256:" + "e" * 64, "--output-root", str(output_root),
        "--release-assets-root", str(tmp_path / "assets" / "Release"),
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "f" * 40,
        "--groot-python", str(tmp_path / "python3.10"), "--policy-server-port", "5511",
        "--policy-server-readiness-timeout", "1", "--policy-server-request-timeout", "1",
        "--policy-server-termination-grace", "1", "--policy-server-log", str(tmp_path / "server.log"),
        "--device", "cuda:0",
    ])
    events: list[str] = []

    def preflight() -> None:
        events.append("preflight")
        raise ValueError("incompatible host")

    with pytest.raises(ValueError, match="incompatible host"):
        run_trial(args, runtime_preflight=preflight)

    assert events == ["preflight"]
    assert not output_root.exists()


def test_real_evaluation_rejects_manifest_environment_garment_mismatch_before_recording(monkeypatch, tmp_path) -> None:
    """Load evaluation with inert Isaac dependencies; its preflight must fail first."""
    manifest = {
        "policy_revision": "a" * 40,
        "episode_id": "worker-1",
        "garment": "Pant_Long_Seen_0",
        "seed": 42,
        "identity": {"episode_id": "worker-1", "policy_repo": "repo", "policy_revision": "a" * 40,
                     "policy_step": 1, "code_revision": "b" * 40, "asset_revision": "c" * 40,
                     "simulator_version": "isaac", "garment_name": "Pant_Long_Seen_0", "category": "pant_long",
                     "release_stage": "seen", "seed": 42, "instruction": "fold the garment on the table", "strategy": "canonical"},
    }
    path = tmp_path / "manifest.json"; path.write_text(__import__("json").dumps(manifest))
    package = types.ModuleType("scripts.utils"); package.__path__ = [str(Path(__file__).parents[2] / "scripts" / "utils")]
    modules = {
        "scripts.utils": package,
        "gymnasium": types.ModuleType("gymnasium"),
        "torch": types.ModuleType("torch"),
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.envs": types.ModuleType("isaaclab.envs"),
        "isaaclab_tasks": types.ModuleType("isaaclab_tasks"),
        "isaaclab_tasks.utils": types.ModuleType("isaaclab_tasks.utils"),
        "scripts.eval_policy": types.ModuleType("scripts.eval_policy"),
        "scripts.eval_policy.base_policy": types.ModuleType("scripts.eval_policy.base_policy"),
        "scripts.utils.eval_utils": types.ModuleType("scripts.utils.eval_utils"),
        "lehome.utils.record": types.ModuleType("lehome.utils.record"),
        "lerobot": types.ModuleType("lerobot"),
        "lerobot.datasets": types.ModuleType("lerobot.datasets"),
        "lerobot.datasets.lerobot_dataset": types.ModuleType("lerobot.datasets.lerobot_dataset"),
        "scripts.utils.common": types.ModuleType("scripts.utils.common"),
        "lehome.utils.logger": types.ModuleType("lehome.utils.logger"),
    }
    modules["isaaclab.envs"].DirectRLEnv = object
    modules["isaaclab_tasks.utils"].parse_env_cfg = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must fail before environment creation"))
    modules["scripts.eval_policy"].PolicyRegistry = object
    modules["scripts.eval_policy.base_policy"].BasePolicy = object
    modules["scripts.utils.eval_utils"].convert_ee_pose_to_joints = lambda *_: None
    modules["scripts.utils.eval_utils"].save_videos_from_observations = lambda *_: None
    modules["scripts.utils.eval_utils"].calculate_and_print_metrics = lambda *_: None
    modules["lehome.utils.record"].RateLimiter = object
    modules["lehome.utils.record"].get_next_experiment_path_with_gap = lambda *_: None
    modules["lehome.utils.record"].append_episode_initial_pose = lambda *_: None
    modules["lerobot.datasets.lerobot_dataset"].LeRobotDataset = object
    modules["scripts.utils.common"].stabilize_garment_after_reset = lambda *_: None
    modules["lehome.utils.logger"].get_logger = lambda *_: types.SimpleNamespace(info=lambda *_: None)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "scripts.utils.evaluation", raising=False)
    evaluation = importlib.import_module("scripts.utils.evaluation")
    args = argparse.Namespace(flywheel_manifest=str(path), num_episodes=1, garment_name="Pant_Long_Seen_1")
    with pytest.raises(ValueError, match="requested garment"):
        evaluation.eval(args, object())
    identity = evaluation._flywheel_identity(manifest)
    wrong_loaded_object = types.SimpleNamespace(
        cfg=types.SimpleNamespace(garment_name="Pant_Long_Seen_0", garment_version="Release"),
        object=types.SimpleNamespace(prim_name="Pant_Long_Seen_1"),
    )
    with pytest.raises(ValueError, match="active environment garment"):
        evaluation._validate_active_flywheel_garment(wrong_loaded_object, identity)


def test_policy_server_command_and_receipt_omit_the_api_token(tmp_path) -> None:
    args = argparse.Namespace(
        groot_python=tmp_path / "python3.10",
        policy_path=tmp_path / "policy",
        policy_server_port=5511,
        policy_server_request_timeout=2.5,
        policy_server_readiness_timeout=30.0,
        code_revision="a" * 40,
        policy_artifact_sha256="b" * 64,
        image_identity="sha256:" + "c" * 64,
        output_root=tmp_path,
        episode_id="server-boundary",
    )
    token = "token-" + "x" * 48
    command = trial_module.build_policy_server_command(args)
    assert token not in " ".join(command)
    assert "--api-token-env" in command
    receipt = trial_module.write_policy_server_receipt(
        args,
        groot_revision="d" * 40,
        python_version="3.10.18",
        checkpoint_revision="e" * 40,
        command=command,
    )
    content = receipt.read_text(encoding="utf-8")
    assert token not in content
    payload = json.loads(content)
    assert payload["host"] == "127.0.0.1"
    assert payload["checkpoint_revision"] == "e" * 40


def test_policy_server_runtime_validation_requires_clean_checkout_and_python310(monkeypatch, tmp_path) -> None:
    root = tmp_path / "Isaac-GR00T"
    package = root / "gr00t"
    package.mkdir(parents=True)
    interpreter = tmp_path / "python3.10"
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)
    args = argparse.Namespace(groot_root=root, groot_revision="a" * 40, groot_python=interpreter)

    def runner(command, **_kwargs):
        if command[-1] == "--show-toplevel":
            return types.SimpleNamespace(returncode=0, stdout=str(root) + "\n")
        if command[-1] == "HEAD":
            return types.SimpleNamespace(returncode=0, stdout="a" * 40 + "\n")
        if command[-1] == "--untracked-files=all":
            return types.SimpleNamespace(returncode=0, stdout="")
        if command[-1] == "--version":
            return types.SimpleNamespace(returncode=0, stdout="Python 3.10.18\n")
        return types.SimpleNamespace(returncode=0, stdout=str(package / "__init__.py") + "\n")

    runtime = trial_module.validate_policy_server_runtime(args, runner=runner)
    assert runtime == {"groot_revision": "a" * 40, "python_version": "3.10.18", "python_path": str(interpreter)}

    def dirty_runner(command, **kwargs):
        result = runner(command, **kwargs)
        if command[-1] == "--untracked-files=all":
            return types.SimpleNamespace(returncode=0, stdout="M gr00t/policy.py\n")
        return result
    with pytest.raises(ValueError, match="clean"):
        trial_module.validate_policy_server_runtime(args, runner=dirty_runner)


def test_policy_server_supervisor_reaps_after_term_then_kill_and_signal(monkeypatch) -> None:
    events = []

    class Process:
        pid = 4242
        def poll(self): return None
        def wait(self, timeout):
            events.append(("wait", timeout))
            if len([event for event in events if event[0] == "wait"]) == 1:
                raise subprocess.TimeoutExpired("server", timeout)
            return 0

    process = Process()
    supervisor = trial_module.PolicyServerSupervisor(
        process, termination_grace=0.1,
        killpg=lambda pid, sig: events.append((pid, sig)),
    )
    supervisor.install_signal_handlers()
    with pytest.raises(SystemExit, match="143"):
        supervisor._signal_handler(signal.SIGTERM, None)
    supervisor.restore_signal_handlers()
    assert events == [
        (4242, signal.SIGTERM), ("wait", 0.1),
        (4242, signal.SIGKILL), ("wait", 0.1),
    ]


def test_policy_server_supervisor_reaps_an_inherited_group_server_without_signalling_the_trial(monkeypatch) -> None:
    events = []

    class Process:
        pid = 4243
        def poll(self): return None
        def terminate(self): events.append("terminate")
        def kill(self): events.append("kill")
        def wait(self, timeout):
            events.append(("wait", timeout))
            if len([event for event in events if isinstance(event, tuple)]) == 1:
                raise subprocess.TimeoutExpired("server", timeout)
            return 0

    supervisor = trial_module.PolicyServerSupervisor(
        Process(),
        termination_grace=0.1,
        owns_process_group=False,
        killpg=lambda *_args: (_ for _ in ()).throw(AssertionError("trial group must not be signalled")),
    )

    supervisor.close()

    assert events == ["terminate", ("wait", 0.1), "kill", ("wait", 0.1)]


def test_policy_server_child_isolated_to_physical_gpu_and_parent_visibility_restores(monkeypatch, tmp_path) -> None:
    captured = {}

    class Process:
        pid = 7
        def poll(self): return 0
        def wait(self, timeout): return 0

    def popen(_command, **kwargs):
        captured.update(kwargs)
        return Process()

    args = argparse.Namespace(
        policy_server_log=tmp_path / "server.log",
        policy_server_termination_grace=1.0,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setattr(trial_module.subprocess, "Popen", popen)
    supervisor = trial_module._spawn_policy_server(
        args, token="x" * 48, command=["python", "server.py"], physical_gpu="2",
    )
    assert captured["start_new_session"] is False
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert captured["env"]["LEHOME_GROOT_POLICY_API_TOKEN"] == "x" * 48
    assert "x" * 48 not in args.policy_server_log.read_text(encoding="utf-8")
    parent = trial_module.ParentCudaVisibility()
    parent.clear()
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    parent.restore()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    supervisor.close()


def test_readiness_rejects_early_child_exit_and_binds_the_generated_token() -> None:
    class Exited:
        def poll(self): return 17
    with pytest.raises(RuntimeError, match="exited before readiness"):
        trial_module._await_policy_server_ready(
            types.SimpleNamespace(process=Exited()), port=5511, token="bound-token",
            readiness_timeout=1, request_timeout=1,
        )

    class Alive:
        def poll(self): return None
    received = []
    class Client:
        def __init__(self, endpoint, token, timeout): received.append((endpoint, token, timeout))
        def ping(self): pass
        def close(self): pass
    trial_module._await_policy_server_ready(
        types.SimpleNamespace(process=Alive()), port=5511, token="bound-token",
        readiness_timeout=1, request_timeout=2.5, client_factory=Client,
    )
    assert received == [("tcp://127.0.0.1:5511", "bound-token", 2.5)]


def test_scoped_parent_token_reaches_the_real_groot_server_evaluation_construction(monkeypatch, tmp_path) -> None:
    token_env = "LEHOME_GROOT_POLICY_API_TOKEN"
    token = "token-" + "z" * 48
    received = {}
    package = types.ModuleType("scripts.utils")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "scripts" / "utils")]

    class Registry:
        @staticmethod
        def is_registered(name): return name == "groot_server"
        @staticmethod
        def list_policies(): return ["groot_server"]
        @staticmethod
        def create(name, **kwargs):
            received.update({"name": name, "token": os.environ[kwargs["policy_server_token_env"]], "kwargs": kwargs})
            return object()

    class Environment:
        def initialize_obs(self): pass
        def close(self): pass

    modules = {
        "scripts.utils": package,
        "gymnasium": types.SimpleNamespace(make=lambda *_args, **_kwargs: types.SimpleNamespace(unwrapped=Environment())),
        "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)),
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.envs": types.ModuleType("isaaclab.envs"),
        "isaaclab_tasks": types.ModuleType("isaaclab_tasks"),
        "isaaclab_tasks.utils": types.ModuleType("isaaclab_tasks.utils"),
        "scripts.eval_policy": types.ModuleType("scripts.eval_policy"),
        "scripts.eval_policy.base_policy": types.ModuleType("scripts.eval_policy.base_policy"),
        "scripts.utils.eval_utils": types.ModuleType("scripts.utils.eval_utils"),
        "lehome.utils.record": types.ModuleType("lehome.utils.record"),
        "lerobot": types.ModuleType("lerobot"),
        "lerobot.datasets": types.ModuleType("lerobot.datasets"),
        "lerobot.datasets.lerobot_dataset": types.ModuleType("lerobot.datasets.lerobot_dataset"),
        "scripts.utils.common": types.ModuleType("scripts.utils.common"),
        "lehome.utils.logger": types.ModuleType("lehome.utils.logger"),
    }
    modules["isaaclab.envs"].DirectRLEnv = object
    modules["isaaclab_tasks.utils"].parse_env_cfg = lambda *_args, **_kwargs: types.SimpleNamespace(sim=types.SimpleNamespace(), garment_cfg_base_path=None, particle_cfg_path=None)
    modules["scripts.eval_policy"].PolicyRegistry = Registry
    modules["scripts.eval_policy.base_policy"].BasePolicy = object
    modules["scripts.utils.eval_utils"].convert_ee_pose_to_joints = lambda *_args: None
    modules["scripts.utils.eval_utils"].save_videos_from_observations = lambda *_args: None
    modules["scripts.utils.eval_utils"].calculate_and_print_metrics = lambda *_args: None
    modules["lehome.utils.record"].RateLimiter = object
    modules["lehome.utils.record"].get_next_experiment_path_with_gap = lambda *_args: None
    modules["lehome.utils.record"].append_episode_initial_pose = lambda *_args: None
    modules["lerobot.datasets.lerobot_dataset"].LeRobotDataset = object
    modules["scripts.utils.common"].stabilize_garment_after_reset = lambda *_args: None
    modules["lehome.utils.logger"].get_logger = lambda *_args: types.SimpleNamespace(info=lambda *_args: None)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "scripts.utils.evaluation", raising=False)
    evaluation = importlib.import_module("scripts.utils.evaluation")
    monkeypatch.setattr(evaluation, "run_evaluation_loop", lambda **_kwargs: [])
    release = tmp_path / "Release"; release.mkdir()
    (release / "Release_test_list.txt").write_text("Top_Long_Seen_0\n", encoding="utf-8")
    args = argparse.Namespace(
        flywheel_manifest=None, task="LeHome-BiSO101", device="cuda:0", use_random_seed=False,
        seed=1, garment_cfg_base_path=str(tmp_path), particle_cfg_path="particles", policy_type="groot_server",
        policy_path="/checkpoint", policy_server_endpoint="tcp://127.0.0.1:5511",
        policy_server_token_env=token_env, policy_server_request_timeout=1.0,
        task_description="fold the garment on the table", use_ee_pose=False, garment_type="custom",
    )
    monkeypatch.setenv(token_env, "prior-token")
    scoped = trial_module.ParentPolicyToken(token_env, token)
    scoped.install()
    try:
        evaluation.eval(args, object())
    finally:
        scoped.restore()
    assert received["name"] == "groot_server"
    assert received["token"] == token
    assert os.environ[token_env] == "prior-token"


def test_policy_server_child_unblocks_sigterm_before_runtime_load(tmp_path) -> None:
    program = (
        "import signal, time; "
        "signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM}); "
        "from scripts.run_groot_policy_server import unblock_termination_signals; "
        "unblock_termination_signals(); print('ready', flush=True); time.sleep(30)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", program], cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=3) == -signal.SIGTERM
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


def _clear_utils_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if name == "scripts.utils" or name.startswith("scripts.utils."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_utils_common_import_does_not_load_dataset_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_utils_modules(monkeypatch)
    common = types.ModuleType("scripts.utils.common")
    common.launch_app = lambda *_args, **_kwargs: None
    common.launch_app_from_args = lambda *_args, **_kwargs: None
    common.close_app = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "scripts.utils.common", common)
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "lerobot", None)

    namespace: dict[str, object] = {}
    exec("from scripts.utils import common", namespace)

    assert namespace["common"].__name__ == "scripts.utils.common"
    assert "scripts.utils.dataset_inspection" not in sys.modules
    assert "scripts.utils.dataset_processing" not in sys.modules
    with pytest.raises(ModuleNotFoundError, match="pyarrow"):
        importlib.import_module("scripts.utils").inspect


def test_evaluation_only_imports_lerobot_when_saving_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_utils_modules(monkeypatch)
    common = types.ModuleType("scripts.utils.common")
    common.launch_app = lambda *_args, **_kwargs: None
    common.launch_app_from_args = lambda *_args, **_kwargs: None
    common.close_app = lambda *_args, **_kwargs: None
    common.stabilize_garment_after_reset = lambda *_args, **_kwargs: None
    modules = {
        "scripts.utils.common": common,
        "gymnasium": types.ModuleType("gymnasium"),
        "torch": types.ModuleType("torch"),
        "numpy": types.ModuleType("numpy"),
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.envs": types.ModuleType("isaaclab.envs"),
        "isaaclab_tasks": types.ModuleType("isaaclab_tasks"),
        "isaaclab_tasks.utils": types.ModuleType("isaaclab_tasks.utils"),
        "scripts.eval_policy": types.ModuleType("scripts.eval_policy"),
        "scripts.eval_policy.base_policy": types.ModuleType("scripts.eval_policy.base_policy"),
        "scripts.utils.eval_utils": types.ModuleType("scripts.utils.eval_utils"),
        "lehome.utils.record": types.ModuleType("lehome.utils.record"),
        "lehome.utils.logger": types.ModuleType("lehome.utils.logger"),
        "lerobot": None,
    }
    modules["isaaclab.envs"].DirectRLEnv = object
    modules["isaaclab_tasks.utils"].parse_env_cfg = lambda *_args, **_kwargs: None
    modules["scripts.eval_policy"].PolicyRegistry = object
    modules["scripts.eval_policy.base_policy"].BasePolicy = object
    modules["scripts.utils.eval_utils"].convert_ee_pose_to_joints = lambda *_args, **_kwargs: None
    modules["scripts.utils.eval_utils"].save_videos_from_observations = lambda *_args, **_kwargs: None
    modules["scripts.utils.eval_utils"].calculate_and_print_metrics = lambda *_args, **_kwargs: None
    modules["lehome.utils.record"].RateLimiter = object
    modules["lehome.utils.record"].get_next_experiment_path_with_gap = lambda *_args, **_kwargs: None
    modules["lehome.utils.record"].append_episode_initial_pose = lambda *_args, **_kwargs: None
    modules["lehome.utils.logger"].get_logger = lambda *_args, **_kwargs: types.SimpleNamespace(info=lambda *_args: None)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    evaluation = importlib.import_module("scripts.utils.evaluation")

    assert "lerobot.datasets.lerobot_dataset" not in sys.modules
    with pytest.raises(ModuleNotFoundError, match="lerobot"):
        evaluation.run_evaluation_loop(None, None, types.SimpleNamespace(save_datasets=True))
