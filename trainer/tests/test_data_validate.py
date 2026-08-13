from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fixtures.source_dataset import make_source_dataset
from lehome_train.data.convert import convert_dataset
from lehome_train.data.stats import write_train_statistics
from lehome_train.data.validate import (
    _action_horizon,
    _compare_statistics,
    _validate_manifest,
    validate_prepared_dataset,
)
from lehome_train.groot.modality import (
    ACTION_HORIZON,
    modality_contract,
    runtime_modality_config_source,
    write_runtime_modality_config,
)


MAPPING_PATH = Path(__file__).parents[1] / "config" / "lehome_four_types_mapping.json"
CONVERTER_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SOURCE_REVISION = "89abcdef0123456789abcdef0123456789abcdef"
CONTAINER_DIGEST = "sha256:" + ("a" * 64)


def test_statistics_comparison_accepts_pinned_float32_accumulation_drift() -> None:
    _compare_statistics([0.5784710391752136], [0.5782300233840942], "stats")


def test_statistics_comparison_rejects_material_drift() -> None:
    with pytest.raises(ValueError, match="verified training split"):
        _compare_statistics([0.5784710391752136], [0.577], "stats")


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


def test_runtime_modality_contract_supports_step_12000_horizon_40(tmp_path: Path) -> None:
    path = write_runtime_modality_config(
        tmp_path / "lehome_groot_modality.py", action_horizon=40
    )

    assert "delta_indices=list(range(40))" in path.read_text(encoding="utf-8")
    assert modality_contract(action_horizon=40)["action"]["delta_indices"] == list(range(40))


def test_manifest_validator_preserves_legacy_step_12000_horizon_40() -> None:
    manifest = {
        "fixed_language_instruction": "fold the garment on the table",
        "action_schema": {"storage": "absolute", "dimension": 12},
        "state_schema": {"dimension": 12},
        "future_actions": {"horizon": 40, "loader_allow_padding": False},
        "train_episode_ids": ["0"],
        "validation_episode_ids": ["1"],
    }

    assert _validate_manifest(manifest) == (["0"], ["1"])


def test_manifest_validator_rejects_40_step_corrective_rft_target() -> None:
    manifest = {
        "source_format": "verified_flywheel_rft_release",
        "future_actions": {"horizon": 40},
    }

    with pytest.raises(ValueError, match="corrective RFT.*exactly 16"):
        _action_horizon(manifest)


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
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    config_record = next(
        item
        for item in manifest["statistics"]["files"]
        if item["relative_path"] == "meta/lehome_groot_modality.py"
    )
    assert config_record["sha256"]
    assert (dataset / "meta" / "lehome_groot_modality.py").read_text(
        encoding="utf-8"
    ) == runtime_modality_config_source()


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


def test_validation_rejects_comments_that_only_spoof_modality_substrings(
    tmp_path: Path,
) -> None:
    dataset = _prepared_dataset(tmp_path)
    path = dataset / "meta" / "lehome_groot_modality.py"
    path.write_text(
        "# \"top_rgb\", \"left_rgb\", \"right_rgb\"\n"
        "# \"left_arm\", \"left_gripper\", \"right_arm\", \"right_gripper\"\n"
        "# list(range(16))\n"
        "# ActionRepresentation.RELATIVE\n"
        "# ActionRepresentation.ABSOLUTE\n",
        encoding="utf-8",
    )
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_record = next(
        item
        for item in manifest["statistics"]["files"]
        if item["relative_path"] == "meta/lehome_groot_modality.py"
    )
    config_record["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical"):
        validate_prepared_dataset(dataset)


def test_validation_requires_all_three_checked_camera_mappings(tmp_path: Path) -> None:
    dataset = _prepared_dataset(tmp_path)
    metadata_path = dataset / "meta" / "modality.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["video"].pop("top_rgb")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="video"):
        validate_prepared_dataset(dataset)


def test_validation_rejects_mutated_finite_converted_artifact(tmp_path: Path) -> None:
    dataset = _prepared_dataset(tmp_path)
    path = dataset / "data" / "chunk-000" / "episode_000003.parquet"
    table = pq.read_table(path)
    actions = table["action"].to_pylist()
    actions[0][0] += 1.0
    altered = table.set_column(
        table.schema.get_field_index("action"),
        "action",
        pa.array(actions, type=table["action"].type),
    )
    pq.write_table(altered, path, compression="zstd")

    with pytest.raises(ValueError, match="artifact hash"):
        validate_prepared_dataset(dataset)


def test_validation_rejects_mutated_recorded_statistics(tmp_path: Path) -> None:
    dataset = _prepared_dataset(tmp_path)
    path = dataset / "meta" / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    stats["action"]["mean"][0] += 1.0
    path.write_text(json.dumps(stats), encoding="utf-8")

    with pytest.raises(ValueError, match="recorded statistics hash"):
        validate_prepared_dataset(dataset)


def test_validation_rejects_empty_or_reassigned_deterministic_holdout(
    tmp_path: Path,
) -> None:
    dataset = _prepared_dataset(tmp_path)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["train_episode_ids"].append(manifest["validation_episode_ids"][0])
    manifest["validation_episode_ids"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="validation"):
        validate_prepared_dataset(dataset)
