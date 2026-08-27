from __future__ import annotations

import json

import numpy as np
import pytest

from lehome.flywheel.isaac_recorder import AutonomousRecorder, CANONICAL_VIDEO_FILENAMES, MixedSourceRecorder
from lehome.flywheel.snapshots import Snapshot, canonical_reset_hash
from lehome.flywheel.models import ActionSource, EpisodeIdentity, EpisodeOutcome, QualityGrade, RejectionReason


def observation() -> dict[str, np.ndarray]:
    return {
        "observation.state": np.zeros(12, dtype=np.float32),
        "observation.images.top_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.left_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.right_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
    }


def identity(episode_id: str) -> EpisodeIdentity:
    return EpisodeIdentity(
        episode_id, "repo", "a" * 40, 1, "b" * 40, "c" * 40, "isaac",
        "Pant_Long_Seen_0", "pant_long", "seen", 1, "fold the garment", "canonical",
    )


def snapshot() -> Snapshot:
    return Snapshot(1, (0.0,) * 12, (0.0,) * 12, ((0.0, 0.0, 0.0),), ((0.0, 0.0, 0.0),), {"seed": 1}, "Pant_Long_Seen_0", {"strategy": "canonical"})


def accepted_outcome(grade: QualityGrade = QualityGrade.A) -> EpisodeOutcome:
    return EpisodeOutcome("success", True, grade, (), "accept")


def fake_encoder(monkeypatch, recorder: MixedSourceRecorder) -> None:
    def encode(root, *, fps=30):
        directory = root / "videos"
        directory.mkdir(exist_ok=True)
        for camera in ("top_rgb", "left_rgb", "right_rgb"):
            (directory / f"{camera}.mp4").write_bytes(b"video")
        return ("top_rgb.mp4", "left_rgb.mp4", "right_rgb.mp4")
    monkeypatch.setattr(recorder.video_sink, "encode", encode)


def test_mixed_recorder_selects_only_complete_post_takeover_expert_windows(tmp_path, monkeypatch) -> None:
    recorder = MixedSourceRecorder(tmp_path, identity=identity("mixed"), mode="dagger", horizon=16)
    fake_encoder(monkeypatch, recorder)
    recorder.record_snapshot("reset", snapshot())
    for offset in range(2):
        recorder.record_step(observation(), np.ones(12), reward=0.0, success=False, action_source=ActionSource.POLICY, segment=1, policy_request_id="request", policy_chunk_offset=offset)
    recorder.record_snapshot("takeover", snapshot())
    for offset in range(16):
        recorder.record_step(observation(), np.ones(12), reward=1.0, success=True, action_source=ActionSource.EXPERT, segment=2, expert_sequence=offset, expert_sample_age_ms=1.0)
    recorder.record_snapshot("terminal", snapshot())
    final = recorder.finish(outcome=accepted_outcome(), controls=("space", "a"))

    assert final.episode["trainable"] is True
    assert [window.observation_step for window in final.expert_windows] == [2]
    assert final.selection_report.selected == 1
    assert final.annotations[0]["action_source"] == "policy"
    assert final.annotations[2]["action_source"] == "expert"
    assert final.annotations[0]["category"] == "pant_long"
    assert final.annotations[0]["garment_name"] == "Pant_Long_Seen_0"
    assert final.annotations[0]["seed"] == 1


def test_mixed_recorder_enforces_source_specific_provenance_and_hold_exclusion(tmp_path) -> None:
    recorder = MixedSourceRecorder(tmp_path, identity=identity("sources"), mode="expert")
    with pytest.raises(ValueError, match="policy"):
        recorder.record_step(observation(), np.ones(12), reward=0.0, success=False, action_source=ActionSource.POLICY, segment=0)
    with pytest.raises(ValueError, match="expert"):
        recorder.record_step(observation(), np.ones(12), reward=0.0, success=False, action_source=ActionSource.EXPERT, segment=0, expert_sequence=1, expert_sample_age_ms=1.0, policy_request_id="forbidden")
    with pytest.raises(ValueError, match="hold"):
        recorder.record_step(observation(), np.ones(12), reward=0.0, success=False, action_source=ActionSource.HOLD, segment=0, expert_sequence=1)


def test_mixed_recorder_never_selects_a_window_crossing_a_hold_frame(tmp_path, monkeypatch) -> None:
    recorder = MixedSourceRecorder(tmp_path, identity=identity("hold"), mode="dagger", horizon=2)
    fake_encoder(monkeypatch, recorder)
    recorder.record_snapshot("reset", snapshot())
    recorder.record_snapshot("takeover", snapshot())
    recorder.record_step(observation(), np.ones(12), reward=0.0, success=False, action_source=ActionSource.EXPERT, segment=1, expert_sequence=1, expert_sample_age_ms=1.0)
    recorder.record_step(observation(), np.ones(12), reward=0.0, success=False, action_source=ActionSource.HOLD, segment=1)
    recorder.record_step(observation(), np.ones(12), reward=1.0, success=True, action_source=ActionSource.EXPERT, segment=1, expert_sequence=2, expert_sample_age_ms=1.0)
    recorder.record_snapshot("terminal", snapshot())
    result = recorder.finish(outcome=accepted_outcome(), controls=("space", "a"))
    assert result.episode["trainable"] is True
    assert result.episode["bc_target_count"] == 0
    assert result.selection_report.hold == 1


def test_mixed_recorder_evidence_and_quality_failures_are_diagnostic(tmp_path, monkeypatch) -> None:
    recorder = MixedSourceRecorder(tmp_path, identity=identity("missing-takeover"), mode="dagger", horizon=1)
    fake_encoder(monkeypatch, recorder)
    recorder.record_snapshot("reset", snapshot())
    recorder.record_step(observation(), np.ones(12), reward=0.0, success=False, action_source=ActionSource.POLICY, segment=1, policy_request_id="request", policy_chunk_offset=0)
    recorder.record_snapshot("terminal", snapshot())
    result = recorder.finish(outcome=accepted_outcome(), controls=())
    assert result.episode["trainable"] is False
    assert result.episode["bc_target_count"] == 0
    assert "missing_takeover_snapshot" in result.episode["diagnostic_reasons"]

    failed = MixedSourceRecorder(tmp_path, identity=identity("failed"), mode="practice", horizon=1)
    failed.record_snapshot("reset", snapshot())
    failed.record_step(observation(), np.ones(12), reward=0.0, success=False, action_source=ActionSource.EXPERT, segment=1, expert_sequence=1, expert_sample_age_ms=1.0)
    failed.record_snapshot("terminal", snapshot())
    outcome = EpisodeOutcome("timeout", False, QualityGrade.C, (RejectionReason.FAILED_EPISODE,), "discard")
    result = failed.finish(outcome=outcome, controls=("d",))
    assert result.episode["trainable"] is False
    assert result.episode["bc_target_count"] == 0


def test_mixed_recorder_encoder_failure_forces_diagnostic_zero_targets(tmp_path, monkeypatch) -> None:
    recorder = MixedSourceRecorder(tmp_path, identity=identity("encoder-failure"), mode="expert", horizon=1)
    recorder.record_snapshot("reset", snapshot())
    recorder.record_step(observation(), np.ones(12), reward=1.0, success=True, action_source=ActionSource.EXPERT, segment=1, expert_sequence=1, expert_sample_age_ms=1.0)
    recorder.record_snapshot("terminal", snapshot())
    monkeypatch.setattr(recorder.video_sink, "encode", lambda root, fps=30: (_ for _ in ()).throw(RuntimeError("encoder failed")))
    result = recorder.finish(outcome=accepted_outcome(QualityGrade.B), controls=("a",))
    assert result.episode["trainable"] is False
    assert result.episode["bc_target_count"] == 0
    assert "video_encoding_failed" in result.episode["diagnostic_reasons"]


def test_autonomous_encoder_failure_fails_finalization_without_publishing_a_video_less_artifact(tmp_path, monkeypatch) -> None:
    recorder = AutonomousRecorder.for_test(tmp_path, policy_revision="a" * 40)
    recorder.record_step(observation(), np.ones(12), reward=1.0, success=True, request_id="r1", chunk_offset=0)
    monkeypatch.setattr(recorder.video_sink, "encode", lambda root, fps=30: (_ for _ in ()).throw(RuntimeError("encoder failed")))

    with pytest.raises(RuntimeError, match="encoder failed"):
        recorder.finish(reason="success", accepted_success=True)

    assert recorder.writer.staging.is_dir()
    assert not (tmp_path / "raw" / "episode-test").exists()


def test_autonomous_timeout_publishes_every_canonical_video_for_campaign_completion(tmp_path, monkeypatch) -> None:
    recorder = AutonomousRecorder.for_test(tmp_path, policy_revision="a" * 40)
    recorder.record_step(observation(), np.ones(12), reward=0.0, success=False, request_id="r1", chunk_offset=0)

    def encode(root, *, fps=30):
        videos = root / "videos"
        videos.mkdir()
        for filename in CANONICAL_VIDEO_FILENAMES:
            (videos / filename).write_bytes(b"video")
        return CANONICAL_VIDEO_FILENAMES

    monkeypatch.setattr(recorder.video_sink, "encode", encode)
    result = recorder.finish(reason="horizon", accepted_success=False)

    assert result.episode["outcome"] == "timeout"
    assert all((result.path / "videos" / filename).is_file() for filename in CANONICAL_VIDEO_FILENAMES)


def test_simple_curriculum_recorder_requires_and_persists_complete_fidelity(tmp_path, monkeypatch) -> None:
    recorder = AutonomousRecorder.for_test(
        tmp_path, policy_revision="a" * 40, simple_curriculum_collection=True,
    )
    recorder.record_step(observation(), np.ones(12), reward=0.0, success=False, request_id="r1", chunk_offset=0)

    def encode(root, *, fps=30):
        videos = root / "videos"
        videos.mkdir()
        for filename in CANONICAL_VIDEO_FILENAMES:
            (videos / filename).write_bytes(b"video")
        return CANONICAL_VIDEO_FILENAMES

    monkeypatch.setattr(recorder.video_sink, "encode", encode)
    with pytest.raises(ValueError, match="fidelity"):
        recorder.finish(reason="horizon", accepted_success=False)

    recorder = AutonomousRecorder.for_test(
        tmp_path / "second", policy_revision="b" * 40, simple_curriculum_collection=True,
    )
    recorder.record_step(observation(), np.ones(12), reward=0.0, success=False, request_id="r1", chunk_offset=0)
    monkeypatch.setattr(recorder.video_sink, "encode", encode)
    fidelity = {
        "missing_cloth": False, "cloth_flight": False,
        "nonfinite_cloth_state": False, "safety_failure": True,
        "monitor_active": True, "monitor_observed": True,
    }
    result = recorder.finish(reason="horizon", accepted_success=False, fidelity=fidelity)

    assert result.episode["fidelity"] == fidelity
    assert result.episode["safety_failure"] is True


def test_mixed_recorder_transport_rejection_cannot_be_trainable(tmp_path, monkeypatch) -> None:
    recorder = MixedSourceRecorder(tmp_path, identity=identity("transport"), mode="expert", horizon=1)
    fake_encoder(monkeypatch, recorder)
    recorder.record_snapshot("reset", snapshot())
    recorder.record_step(observation(), np.ones(12), reward=1.0, success=True, action_source=ActionSource.EXPERT, segment=1, expert_sequence=1, expert_sample_age_ms=1.0)
    recorder.record_snapshot("terminal", snapshot())
    result = recorder.finish(
        outcome=EpisodeOutcome("success", True, QualityGrade.A, (RejectionReason.DISCONNECTED,), "accept"),
        controls=("a",),
    )
    assert result.episode["trainable"] is False
    assert result.episode["bc_target_count"] == 0


def test_autonomous_recorder_marks_policy_source_and_terminal_reason(tmp_path) -> None:
    recorder = AutonomousRecorder.for_test(tmp_path, policy_revision="a" * 40)
    recorder.record_step(observation(), np.ones(12), reward=0.2, success=False, request_id="r1", chunk_offset=0)
    final = recorder.finish(reason="horizon", accepted_success=False)

    assert final.episode["terminal_reason"] == "horizon"
    assert final.annotations[0]["action_source"] == "policy"
    assert final.annotations[0]["policy_request_id"] == "r1"

    payload = json.loads((final.path / "episode.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "autonomous"
    assert payload["bc_target_count"] == 0


def test_recorder_checksum_covers_reset_and_terminal_snapshots(tmp_path) -> None:
    recorder = AutonomousRecorder.for_test(tmp_path, policy_revision="a" * 40)
    snapshot = Snapshot(1, (0.0,) * 12, (0.0,) * 12, ((0.0, 0.0, 0.0),), ((0.0, 0.0, 0.0),), {"seed": 1}, "Pant_Long_Seen_0", {"strategy": "canonical"})
    recorder.record_snapshot("reset", snapshot)
    recorder.record_step(observation(), np.ones(12), reward=0.0, success=False, request_id="r", chunk_offset=0)
    recorder.record_snapshot("terminal", snapshot)
    final = recorder.finish(reason="horizon", accepted_success=False)
    manifest = json.loads((final.path / "SHA256SUMS.json").read_text(encoding="utf-8"))
    assert "snapshots/reset.json" in manifest and "snapshots/terminal.json" in manifest
    assert final.episode["reset_hash"] == canonical_reset_hash(snapshot)


def test_autonomous_recorder_authenticates_h16_continuation_snapshots(tmp_path) -> None:
    """A future recovery source must retain physical state at policy boundaries.

    The boundary is recorded after action 15 and before action 16, so the
    snapshot name is the next annotation index rather than the prior action.
    """

    recorder = AutonomousRecorder.for_test(tmp_path, policy_revision="a" * 40)
    boundary = snapshot()
    recorder.record_snapshot("reset", boundary)
    for step in range(16):
        recorder.record_step(observation(), np.ones(12), reward=0.0, success=False, request_id="r0", chunk_offset=step)
    recorder.record_continuation_snapshot(16, boundary)
    recorder.record_snapshot("terminal", boundary)
    final = recorder.finish(reason="horizon", accepted_success=False)

    continuation = final.path / "snapshots" / "continuations" / "000016.json"
    manifest = json.loads((final.path / "SHA256SUMS.json").read_text(encoding="utf-8"))
    assert continuation.is_file()
    assert "snapshots/continuations/000016.json" in manifest
    with pytest.raises(ValueError, match="H16"):
        recorder.record_continuation_snapshot(17, boundary)


def test_autonomous_recorder_excludes_boundary_snapshot_at_first_success(tmp_path, monkeypatch) -> None:
    """A boundary immediately before the successful action is not pre-success evidence."""

    recorder = AutonomousRecorder.for_test(tmp_path, policy_revision="a" * 40)
    boundary = snapshot()
    for step in range(16):
        recorder.record_step(
            observation(), np.ones(12), reward=0.0, success=False,
            request_id="r0", chunk_offset=step,
        )
    recorder.record_continuation_snapshot(16, boundary)
    recorder.record_step(
        observation(), np.ones(12), reward=1.0, success=True,
        request_id="r1", chunk_offset=0,
    )
    monkeypatch.setattr(recorder.video_sink, "encode", lambda root, fps=30: _encode_videos(root))

    final = recorder.finish(reason="success", accepted_success=True)

    relative = "snapshots/continuations/000016.json"
    manifest = json.loads((final.path / "SHA256SUMS.json").read_text(encoding="utf-8"))
    assert relative not in manifest
    assert not (final.path / relative).exists()


def test_autonomous_recorder_persists_simulator_contact_evidence(tmp_path, monkeypatch) -> None:
    recorder = AutonomousRecorder.for_test(tmp_path, policy_revision="a" * 40)
    recorder.record_step(observation(), np.ones(12), reward=0.0, success=False, request_id="r", chunk_offset=0)
    monkeypatch.setattr(recorder.video_sink, "encode", lambda root, fps=30: _encode_videos(root))

    final = recorder.finish(
        reason="horizon",
        accepted_success=False,
        visible_contact={"observed": True, "source": "simulator_particle_to_gripper_distance", "minimum_distance_m": 0.012},
    )

    assert final.episode["visible_contact"]["observed"] is True


def _encode_videos(root) -> tuple[str, ...]:
    videos = root / "videos"
    videos.mkdir(exist_ok=True)
    for filename in CANONICAL_VIDEO_FILENAMES:
        (videos / filename).write_bytes(b"video")
    return CANONICAL_VIDEO_FILENAMES


def test_recorder_rejects_identity_with_a_different_episode_id(tmp_path) -> None:
    identity = EpisodeIdentity("other", "repo", "a" * 40, 1, "b" * 40, "c" * 40, "isaac", "Pant_Long_Seen_0", "pant_long", "seen", 1, "fold the garment on the table", "canonical")
    import pytest
    with pytest.raises(ValueError, match="episode ID"):
        AutonomousRecorder(tmp_path, policy_revision="a" * 40, episode_id="episode", identity=identity)
