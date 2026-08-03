from pathlib import Path

import pytest

from lehome.flywheel.matrix import (
    PUBLIC_UNSEEN_HOLDOUTS,
    build_public_matrix,
    matrix_sha256,
    validate_release_assets,
)


def test_public_matrix_has_frozen_breadth_and_holdouts() -> None:
    matrix = build_public_matrix()

    assert len(matrix.trials) == 280
    assert sum(trial.release_stage == "seen" for trial in matrix.trials) == 200
    assert sum(trial.release_stage == "public_unseen" for trial in matrix.trials) == 80
    assert sum(trial.category == "pant_long" for trial in matrix.trials) == 70
    assert len({(trial.garment_name, trial.seed) for trial in matrix.trials}) == 280
    assert matrix.training_holdouts == (
        "Top_Long_Unseen_1",
        "Top_Short_Unseen_1",
        "Pant_Long_Unseen_1",
        "Pant_Short_Unseen_1",
    )
    assert matrix.training_holdouts == PUBLIC_UNSEEN_HOLDOUTS
    assert matrix_sha256(matrix) == matrix_sha256(build_public_matrix())


def test_matrix_rejects_missing_generated_release_assets(tmp_path: Path) -> None:
    matrix = build_public_matrix()
    release = tmp_path / "Release"
    release.mkdir()
    with pytest.raises(ValueError, match="missing"):
        validate_release_assets(release, matrix)
