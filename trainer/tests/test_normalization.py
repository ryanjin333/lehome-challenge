from __future__ import annotations

import json
from pathlib import Path

import pytest

from lehome_train.data.normalization import normalization_identity
from lehome_train.io import canonical_json_sha256, sha256_file


NORMALIZATION_PATHS = (
    "meta/lehome_groot_modality.py",
    "meta/relative_stats.json",
    "meta/stats.json",
)


def _normalization_dataset(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    dataset = tmp_path / "prepared"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "lehome_groot_modality.py").write_text("# pinned modality\n", encoding="utf-8")
    (meta / "relative_stats.json").write_text('{"relative":true}\n', encoding="utf-8")
    (meta / "stats.json").write_text('{"stats":true}\n', encoding="utf-8")
    (meta / "validation_report.json").write_text('{"valid":true}\n', encoding="utf-8")
    hashes = {
        relative_path: sha256_file(dataset / relative_path)
        for relative_path in (*NORMALIZATION_PATHS, "meta/validation_report.json")
    }
    (meta / "prepared_hashes.json").write_text(
        json.dumps({"schema_version": 1, "artifacts": hashes}),
        encoding="utf-8",
    )
    return dataset, hashes


def test_normalization_identity_is_canonical_hash_of_verified_artifact_hashes(
    tmp_path: Path,
) -> None:
    dataset, hashes = _normalization_dataset(tmp_path)

    assert normalization_identity(dataset) == canonical_json_sha256(
        {
            "schema_version": 1,
            "artifacts": [
                {"relative_path": path, "sha256": hashes[path]}
                for path in NORMALIZATION_PATHS
            ],
        }
    )


def test_normalization_identity_rejects_tampered_statistics(tmp_path: Path) -> None:
    dataset, _hashes = _normalization_dataset(tmp_path)
    (dataset / "meta" / "stats.json").write_text('{"stats":false}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="normalization artifact hash mismatch"):
        normalization_identity(dataset)


def test_normalization_identity_requires_exact_prepared_hash_allowlist(
    tmp_path: Path,
) -> None:
    dataset, hashes = _normalization_dataset(tmp_path)
    hashes["meta/untrusted.json"] = "a" * 64
    (dataset / "meta" / "prepared_hashes.json").write_text(
        json.dumps({"schema_version": 1, "artifacts": hashes}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allowlist is incompatible"):
        normalization_identity(dataset)
