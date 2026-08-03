from __future__ import annotations

import pytest

import numpy as np

import argparse

from scripts.run_groot_flywheel_trial import (
    _manifest_path, build_parser, read_pinned_revision, run_randomization_acceptance,
    run_snapshot_acceptance, validate_args,
)


def test_trial_cli_requires_pinned_policy_and_existing_matrix(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(["--policy-path", str(tmp_path / "missing"), "--policy-revision", "main"])
    with pytest.raises(ValueError, match="pinned"):
        validate_args(args)


def test_trial_cli_reads_revision_from_a_regular_file(tmp_path) -> None:
    revision = tmp_path / "revision.txt"
    revision.write_text("a" * 40 + "\n", encoding="utf-8")
    assert read_pinned_revision(revision) == "a" * 40


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


def test_acceptance_modes_write_real_orchestrated_reports(tmp_path) -> None:
    args = build_parser().parse_args(["--snapshot-roundtrip-only", "--garment", "Pant_Long_Seen_0", "--output-root", str(tmp_path)])
    assert run_snapshot_acceptance(args, env_factory=lambda _: FakeEnv()) == 0
    assert (tmp_path / "snapshot-acceptance.json").is_file()
    args = build_parser().parse_args(["--render-randomization-sheet", "--output-root", str(tmp_path), "--strategies", "canonical", "mild", "strong"])
    writes = []
    assert run_randomization_acceptance(args, env_factory=lambda _: FakeEnv(), image_writer=lambda path, frame: writes.append(path)) == 0
    assert len(writes) == 9
    assert (tmp_path / "randomization-receipts.json").is_file()


def test_manifest_creation_is_atomic_and_immutable(tmp_path) -> None:
    args = argparse.Namespace(output_root=tmp_path, policy_path=tmp_path, episode_id="episode-1", policy_repo="org/policy", policy_step=1, code_revision="b" * 40, asset_revision="c" * 40, simulator_version="isaac", garment="Pant_Long_Seen_0", category="pant_long", release_stage="seen", seed=1, strategy="canonical", policy_artifact_sha256="d" * 64, image_identity="sha256:image")
    path = _manifest_path(args, "a" * 40)
    assert _manifest_path(args, "a" * 40) == path
    args.seed = 2
    with pytest.raises(ValueError, match="overwrite"):
        _manifest_path(args, "a" * 40)
    assert not list(tmp_path.glob(".*.tmp"))
