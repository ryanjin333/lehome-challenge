from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fixtures.source_dataset import make_source_dataset
from lehome_train.data.convert import convert_dataset
from lehome_train.data.stats import compute_reference_statistics, write_train_statistics


MAPPING_PATH = Path(__file__).parents[1] / "config" / "lehome_four_types_mapping.json"
CONVERTER_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SOURCE_REVISION = "89abcdef0123456789abcdef0123456789abcdef"
CONTAINER_DIGEST = "sha256:" + ("a" * 64)


def _prepared_dataset(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source = make_source_dataset(tmp_path)
    output = tmp_path / "prepared"
    manifest = convert_dataset(
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
    return output, manifest


def test_statistics_use_only_train_episodes_and_write_finite_12d_values(
    tmp_path: Path,
) -> None:
    dataset, manifest = _prepared_dataset(tmp_path)
    validation_id = int(manifest["validation_episode_ids"][0])
    validation_data = dataset / "data" / "chunk-000" / f"episode_{validation_id:06d}.parquet"
    table = pq.read_table(validation_data)
    validation_actions = [
        [1_000_000.0 + dimension for dimension in range(12)]
        for _ in range(table.num_rows)
    ]
    changed = table.set_column(
        table.schema.get_field_index("action"),
        "action",
        pa.array(validation_actions, type=table["action"].type),
    )
    # The fixture already gives each episode a disjoint numerical range.  The
    # validation episode's values must not affect train-only extrema.
    pq.write_table(changed, validation_data, compression="zstd")

    statistics = compute_reference_statistics(dataset)

    assert set(statistics.stats) == {"observation.state", "action"}
    assert len(statistics.stats["observation.state"]["mean"]) == 12
    assert len(statistics.stats["action"]["q99"]) == 12
    assert statistics.stats["action"]["max"][0] < 1_000_000.0
    assert set(statistics.relative_stats) == {"left_arm", "right_arm"}
    assert len(statistics.relative_stats["left_arm"]["mean"]) == 16
    assert len(statistics.relative_stats["left_arm"]["mean"][0]) == 5

    result = write_train_statistics(dataset)

    assert result["runtime"] == "python_reference"
    for name in ("stats.json", "relative_stats.json"):
        saved = json.loads((dataset / "meta" / name).read_text(encoding="utf-8"))
        assert saved
    assert not (dataset / "meta" / "norm_stats.json").exists()


def test_statistics_refuse_openpi_normalization_artifacts(tmp_path: Path) -> None:
    dataset, _ = _prepared_dataset(tmp_path)
    (dataset / "meta" / "norm_stats.json").write_text(
        '{"state":{"mean":[0]}}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="OpenPI norm_stats.json"):
        compute_reference_statistics(dataset)


def test_statistics_refuse_nonfinite_training_values(tmp_path: Path) -> None:
    dataset, manifest = _prepared_dataset(tmp_path)
    train_id = int(manifest["train_episode_ids"][0])
    path = dataset / "data" / "chunk-000" / f"episode_{train_id:06d}.parquet"
    table = pq.read_table(path)
    rows = table["action"].to_pylist()
    rows[0][0] = float("nan")
    table = table.set_column(
        table.schema.get_field_index("action"),
        "action",
        pa.array(rows, type=table["action"].type),
    )
    pq.write_table(table, path, compression="zstd")

    with pytest.raises(ValueError, match="non-finite"):
        compute_reference_statistics(dataset)
