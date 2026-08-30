"""Linux-image acceptance gate; skips only outside the real pinned runtime."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from fixtures.source_dataset import make_source_dataset
from lehome_train.constants import ISAAC_GROOT_REVISION
from lehome_train.data.convert import convert_dataset
from lehome_train.data.stats import write_train_statistics
from lehome_train.data.validate import validate_prepared_dataset


MAPPING_PATH = Path(__file__).parents[1] / "config" / "lehome_four_types_mapping.json"
CONVERTER_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SOURCE_REVISION = "89abcdef0123456789abcdef0123456789abcdef"
CONTAINER_DIGEST = "sha256:" + ("a" * 64)
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="the pinned Isaac-GR00T acceptance gate runs only in the Linux image",
)
def test_pinned_runtime_computes_train_only_stats_and_consumes_one_loader_batch(
    tmp_path: Path,
) -> None:
    groot_root = os.environ.get("LEHOME_GROOT_ROOT")
    assert groot_root, "Linux trainer-image acceptance requires LEHOME_GROOT_ROOT"
    assert (Path(groot_root) / "gr00t" / "data" / "stats.py").is_file(), (
        "LEHOME_GROOT_ROOT must point at the pinned Isaac-GR00T checkout"
    )
    revision = subprocess.run(
        ["git", "-C", groot_root, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == ISAAC_GROOT_REVISION
    source = make_source_dataset(tmp_path)
    prepared = tmp_path / "prepared"
    convert_dataset(
        source,
        prepared,
        mapping_path=MAPPING_PATH,
        source_repository="ryanjin333/four_types_merged",
        source_revision=SOURCE_REVISION,
        converter_commit=CONVERTER_COMMIT,
        converter_container_digest=CONTAINER_DIGEST,
        split_seed=17,
        validation_fraction=1 / 3,
    )

    statistics = write_train_statistics(prepared, groot_root=groot_root)
    report = validate_prepared_dataset(prepared, groot_root=groot_root)

    assert statistics["runtime"] == "pinned_gr00t"
    assert report["loader_integration"] == "pinned_loader_one_batch"
