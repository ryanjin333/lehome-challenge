from __future__ import annotations

import json
from pathlib import Path

from lehome_train.data.convert import LEGACY_VIDEO_PATH
from lehome_train.flywheel.rft import materialize_rft_snapshot
from test_flywheel_materialize import _raw_rft_episode


def test_rft_snapshot_trains_only_seen_successes_and_keeps_episode_holdout(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    episodes = [
        _raw_rft_episode(raw_root, episode_id="seen-success-1"),
        _raw_rft_episode(raw_root, episode_id="seen-success-2"),
        _raw_rft_episode(raw_root, episode_id="unseen-success", holdout=True),
    ]

    result = materialize_rft_snapshot(
        episodes,
        tmp_path / "snapshot",
        source_repository="ryanjin333/lehome-groot-n17-data",
        source_revision="a" * 40,
        release_id="b" * 64,
        split_seed=20260809,
        validation_fraction=0.1,
    )

    assert result["accepted_seen_successes"] == 2
    assert result["excluded_public_unseen"] == 1
    assert result["validation"]["valid"] is True
    manifest = json.loads(
        (tmp_path / "snapshot" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["episode_count"] == 2
    assert manifest["future_actions"]["horizon"] == 16
    assert len(manifest["train_episode_ids"]) == 1
    assert len(manifest["validation_episode_ids"]) == 1
    selection = json.loads(
        (tmp_path / "snapshot" / "meta" / "rft-selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert {item["raw_episode_id"] for item in selection["episodes"]} == {
        "seen-success-1",
        "seen-success-2",
    }
    assert selection["action_horizon"] == 16
    info = json.loads(
        (tmp_path / "snapshot" / "meta" / "info.json").read_text(encoding="utf-8")
    )
    expected_cameras = {
        "observation.images.top_rgb",
        "observation.images.left_rgb",
        "observation.images.right_rgb",
    }
    assert expected_cameras <= set(info["features"])
    for camera in expected_cameras:
        assert info["features"][camera]["dtype"] == "video"
        assert (
            tmp_path
            / "snapshot"
            / LEGACY_VIDEO_PATH.format(
                episode_chunk=0,
                episode_index=0,
                video_key=camera,
            )
        ).is_file()
