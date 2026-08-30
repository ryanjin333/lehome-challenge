from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import lehome_train.io as trainer_io
from lehome_train.io import (
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
    load_json,
    parse_json,
    sha256_file,
)
from lehome_train.models import CameraMapping


def test_canonical_json_is_utf8_sorted_compact_and_schema_aware() -> None:
    mapping = CameraMapping(source_key="caméra", target_modality="video.front")

    encoded = canonical_json_bytes({"z": 1, "mapping": mapping, "a": "雪"})

    assert encoded == (
        '{"a":"雪","mapping":{"source_key":"caméra",'
        '"target_modality":"video.front"},"z":1}'
    ).encode()
    assert canonical_json_sha256({"z": 1, "mapping": mapping, "a": "雪"}) == (
        hashlib.sha256(encoded).hexdigest()
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes({"loss": value})


def test_parse_and_load_json_are_strict_and_typed(tmp_path: Path) -> None:
    path = tmp_path / "camera.json"
    path.write_text(
        '{"source_key":"observation.images.front",'
        '"target_modality":"video.front"}',
        encoding="utf-8",
    )

    assert load_json(CameraMapping, path) == CameraMapping(
        source_key="observation.images.front",
        target_modality="video.front",
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_json(CameraMapping, b'{"source_key":')
    with pytest.raises(ValueError, match="finite"):
        parse_json(
            CameraMapping,
            '{"source_key":NaN,"target_modality":"video.front"}',
        )


def test_parse_json_rejects_duplicate_object_fields() -> None:
    payload = (
        '{"source_key":"first","source_key":"second",'
        '"target_modality":"video.front"}'
    )

    with pytest.raises(ValueError, match="duplicate field"):
        parse_json(CameraMapping, payload)


def test_sha256_file_streams_exact_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    content = (b"lehome\x00" * 200_000) + b"tail"
    path.write_bytes(content)

    assert sha256_file(path, chunk_size=4_096) == hashlib.sha256(content).hexdigest()


def test_atomic_write_json_replaces_destination_with_canonical_bytes(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "status.json"
    destination.write_text('{"state":"old"}', encoding="utf-8")

    atomic_write_json(destination, {"state": "complete", "step": 2})

    assert destination.read_bytes() == b'{"state":"complete","step":2}'
    assert json.loads(destination.read_text(encoding="utf-8"))["state"] == "complete"
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_atomic_write_failure_preserves_destination_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "status.json"
    destination.write_bytes(b'{"state":"old"}')

    def fail_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr("lehome_train.io.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failure"):
        atomic_write_json(destination, {"state": "complete"})

    assert destination.read_bytes() == b'{"state":"old"}'
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_atomic_write_reports_parent_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "status.json"

    def fail_parent_fsync(parent: Path) -> None:
        raise OSError("synthetic parent fsync failure")

    monkeypatch.setattr("lehome_train.io._fsync_parent", fail_parent_fsync)

    with pytest.raises(OSError, match="parent fsync failure"):
        atomic_write_json(destination, {"state": "complete"})

    assert destination.read_bytes() == b'{"state":"complete"}'


def test_parent_directory_open_failure_is_not_suppressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_open(path: Path, flags: int) -> int:
        raise PermissionError("synthetic directory open failure")

    monkeypatch.setattr(trainer_io.os, "open", fail_open)

    with pytest.raises(PermissionError, match="directory open failure"):
        trainer_io._fsync_parent(tmp_path)


def test_parent_directory_fsync_failure_is_not_suppressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = os.open(tmp_path, os.O_RDONLY)

    monkeypatch.setattr(trainer_io.os, "open", lambda path, flags: descriptor)
    monkeypatch.setattr(
        trainer_io.os,
        "fsync",
        lambda opened_descriptor: (_ for _ in ()).throw(
            OSError("synthetic directory fsync failure")
        ),
    )

    with pytest.raises(OSError, match="directory fsync failure"):
        trainer_io._fsync_parent(tmp_path)
