from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval_groot_n17_matrix import (
    MatrixValidationError,
    load_matrix,
    parse_episode_metric,
    validate_asset_matrix,
    validate_run_path,
)


def test_load_matrix_accepts_the_fixed_24_trial_contract() -> None:
    trials = load_matrix(Path("configs/eval_groot_n17_seen_dev.json"))

    assert len(trials) == 24
    assert len({trial.trial_id for trial in trials}) == 24
    assert {trial.seed for trial in trials} == {42, 43, 44}
    assert all("_Seen_" in trial.garment_name for trial in trials)
    assert all(trial.category in {"top_long", "top_short", "pant_long", "pant_short"} for trial in trials)


def test_load_matrix_rejects_duplicate_and_unseen_trials(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "categories": ["top_long", "top_short", "pant_long", "pant_short"],
        "seeds": [42, 43, 44],
        "trials": [
            {
                "trial_id": "duplicate",
                "category": "top_long",
                "garment_name": "Top_Long_Seen_0",
                "seed": 42,
            },
            {
                "trial_id": "duplicate",
                "category": "top_long",
                "garment_name": "Top_Long_Seen_1",
                "seed": 42,
            },
        ],
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MatrixValidationError, match="unique"):
        load_matrix(path)

    payload["trials"][1]["garment_name"] = "Top_Long_Unseen_0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MatrixValidationError, match="official seen"):
        load_matrix(path)


def test_parse_episode_metric_requires_a_single_complete_metric() -> None:
    output = "Episode 1/1: Return=12.50, Length=321, Success=True"

    assert parse_episode_metric(output) == {
        "return": 12.5,
        "length": 321,
        "success": True,
    }

    with pytest.raises(MatrixValidationError, match="metric"):
        parse_episode_metric("Evaluation completed successfully")


def test_validate_run_path_rejects_paths_outside_run_root(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()

    assert validate_run_path(root, root / "trial-000") == root / "trial-000"
    with pytest.raises(MatrixValidationError, match="run root"):
        validate_run_path(root, tmp_path / "outside")


def test_validate_asset_matrix_uses_official_first_two_seen_ids(tmp_path: Path) -> None:
    release = tmp_path / "Release"
    for prefix in ("Top_Long", "Top_Short", "Pant_Long", "Pant_Short"):
        category = release / prefix
        category.mkdir(parents=True)
        (category / f"{prefix}.txt").write_text(
            "\n".join(
                [
                    f"{prefix}_Unseen_0",
                    f"{prefix}_Seen_1",
                    f"{prefix}_Seen_0",
                    f"{prefix}_Seen_2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    trials = load_matrix(Path("configs/eval_groot_n17_seen_dev.json"))
    validate_asset_matrix(tmp_path, trials)

    broken = list(trials)
    broken[0] = broken[0].__class__(
        broken[0].trial_id,
        broken[0].category,
        "Top_Long_Seen_2",
        broken[0].seed,
    )
    with pytest.raises(MatrixValidationError, match="first-two"):
        validate_asset_matrix(tmp_path, broken)
