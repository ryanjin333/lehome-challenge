from __future__ import annotations

import json
from pathlib import Path

import pytest

from fixtures.source_dataset import make_source_dataset
from lehome_train.data.convert import convert_dataset
from lehome_train.data.stats import write_train_statistics
from lehome_train.data.validate import validate_prepared_dataset
from lehome_train.groot.modality import (
    ACTION_HORIZON,
    modality_contract,
    write_runtime_modality_config,
)


MAPPING_PATH = Path(__file__).parents[1] / "config" / "lehome_four_types_mapping.json"
CONVERTER_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SOURCE_REVISION = "89abcdef0123456789abcdef0123456789abcdef"
CONTAINER_DIGEST = "sha256:" + ("a" * 64)


def _prepared_dataset(tmp_path: Path) -> Path:
    source = make_source_dataset(tmp_path)
    output = tmp_path / "prepared"
    convert_dataset(
        source,
        output,
        mapping_path=MAPPING_PATH,
        source_repository="ryanjin333/four_types_merged",
        source_revision=SOURCE_REVISION,
        converter_commit=CONVERTER_COMMIT,
        converter_container_digest=CONTAINER_DIGEST,
        split_seed=17,
        validation_fraction=1 / 3,
    )
    write_train_statistics(output)
    write_runtime_modality_config(output / "meta" / "lehome_groot_modality.py")
    return output


def test_modality_contract_is_joint_space_12d_with_relative_arms_and_absolute_grippers() -> None:
    contract = modality_contract()

    assert contract["video"]["delta_indices"] == [0]
    assert contract["video"]["modality_keys"] == ["top_rgb", "left_rgb", "right_rgb"]
    assert contract["state"]["delta_indices"] == [0]
    assert contract["state"]["dimension"] == 12
    assert contract["action"]["delta_indices"] == list(range(ACTION_HORIZON))
    assert contract["action"]["dimension"] == 12
    assert [item["representation"] for item in contract["action"]["groups"]] == [
        "relative",
        "absolute",
        "relative",
        "absolute",
    ]
    assert [item["dimension"] for item in contract["action"]["groups"]] == [5, 1, 5, 1]
    assert contract["language"]["instruction"] == "fold the garment on the table"


def test_validation_writes_hashed_report_and_keeps_split_offline(tmp_path: Path) -> None:
    dataset = _prepared_dataset(tmp_path)

    report = validate_prepared_dataset(dataset)

    assert report["valid"] is True
    assert report["trainer_validation_split"] == "offline_only"
    assert report["train_episode_count"] == 2
    assert report["validation_episode_count"] == 1
    assert report["loader_integration"] == "not_run_no_pinned_runtime"
    hashes = json.loads((dataset / "meta" / "prepared_hashes.json").read_text(encoding="utf-8"))
    assert set(hashes["artifacts"]) == {
        "meta/lehome_groot_modality.py",
        "meta/relative_stats.json",
        "meta/stats.json",
        "meta/validation_report.json",
    }
    persisted = json.loads((dataset / "meta" / "validation_report.json").read_text(encoding="utf-8"))
    assert persisted == report


def test_validation_fails_on_missing_relative_stats_or_split_leakage(tmp_path: Path) -> None:
    dataset = _prepared_dataset(tmp_path)
    (dataset / "meta" / "relative_stats.json").unlink()

    with pytest.raises(ValueError, match="relative_stats"):
        validate_prepared_dataset(dataset)

    write_train_statistics(dataset)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation_episode_ids"] = [manifest["train_episode_ids"][0]]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="disjoint"):
        validate_prepared_dataset(dataset)


def test_validation_requires_the_offline_validation_episode_to_exist(tmp_path: Path) -> None:
    dataset = _prepared_dataset(tmp_path)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation_episode_ids"] = ["999"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly match offline split"):
        validate_prepared_dataset(dataset)
