from __future__ import annotations

import json

import numpy as np

from lehome.flywheel.isaac_recorder import AutonomousRecorder
from lehome.flywheel.snapshots import Snapshot


def observation() -> dict[str, np.ndarray]:
    return {
        "observation.state": np.zeros(12, dtype=np.float32),
        "observation.images.top_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.left_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.right_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
    }


def test_autonomous_recorder_marks_policy_source_and_terminal_reason(tmp_path) -> None:
    recorder = AutonomousRecorder.for_test(tmp_path, policy_revision="a" * 40)
    recorder.record_step(observation(), np.ones(12), reward=0.2, success=False, request_id="r1", chunk_offset=0)
    final = recorder.finish(reason="horizon", accepted_success=False)

    assert final.episode["terminal_reason"] == "horizon"
    assert final.annotations[0]["action_source"] == "policy"
    assert final.annotations[0]["policy_request_id"] == "r1"

    payload = json.loads((final.path / "episode.json").read_text(encoding="utf-8"))
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
