"""Fail-closed tests for the inference-only reference checkpoint view."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest


REPOSITORY = "theo-zhou/lehome-groot-submission-4"
REVISION = "d384fe00508acd96ab1c3c5dc265e08261f94b3b"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path) -> tuple[Path, str, dict[str, object]]:
    source = tmp_path / "pretrained_model"
    source.mkdir()
    config: dict[str, object] = {
        "type": "groot",
        "device": "cuda",
        "chunk_size": 16,
        "num_decay_steps": 4000,
        "decay_lr_ratio": 0.1,
        "nested": {"preserved": [True, None, 3]},
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (source / "model-00001-of-00002.safetensors").write_bytes(b"first-weight-shard")
    (source / "model-00002-of-00002.safetensors").write_bytes(b"second-weight-shard")
    (source / "model.safetensors.index.json").write_bytes(b'{"weight_map":{}}')
    return source, _sha256(source / "config.json"), config


def _prepare(source: Path, destination: Path, digest: str) -> dict[str, object]:
    from scripts.prepare_reference_checkpoint import prepare_reference_checkpoint

    return prepare_reference_checkpoint(
        source_pretrained_model=source,
        destination_view=destination,
        expected_source_config_sha256=digest,
        source_repository=REPOSITORY,
        source_revision=REVISION,
    )


def _verify(source: Path, destination: Path, digest: str) -> dict[str, object]:
    from scripts.prepare_reference_checkpoint import verify_reference_checkpoint

    return verify_reference_checkpoint(
        source_pretrained_model=source,
        destination_view=destination,
        expected_source_config_sha256=digest,
        source_repository=REPOSITORY,
        source_revision=REVISION,
    )


def test_prepare_creates_canonical_provenance_bound_no_copy_view(tmp_path: Path) -> None:
    from scripts.prepare_reference_checkpoint import MANIFEST_FILENAME

    source, digest, source_config = _source(tmp_path)
    destination = tmp_path / "reference-view"

    manifest = _prepare(source, destination, digest)

    adapted = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    expected_adapted = {
        key: value
        for key, value in source_config.items()
        if key not in {"num_decay_steps", "decay_lr_ratio"}
    }
    assert adapted == expected_adapted
    assert set(adapted) == set(source_config) - {"num_decay_steps", "decay_lr_ratio"}
    assert manifest["removed_fields"] == {
        "decay_lr_ratio": 0.1,
        "num_decay_steps": 4000,
    }
    manifest_path = destination / MANIFEST_FILENAME
    assert manifest_path.read_bytes() == (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")
    assert _verify(source, destination, digest) == manifest

    artifact_names = sorted(path.name for path in source.iterdir() if path.name != "config.json")
    assert [row["relative_name"] for row in manifest["linked_artifacts"]] == artifact_names
    for name in artifact_names:
        link = destination / name
        assert link.is_symlink()
        assert os.path.isabs(os.readlink(link))
        assert Path(os.readlink(link)) == (source / name).resolve()
        assert link.stat().st_ino == (source / name).stat().st_ino
    assert not (destination / "config.json").is_symlink()
    assert not manifest_path.is_symlink()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_decay_steps", 3999),
        ("num_decay_steps", 4000.0),
        ("decay_lr_ratio", 0.2),
        ("decay_lr_ratio", "0.1"),
    ],
)
def test_prepare_rejects_wrong_compatibility_field_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    source, _, config = _source(tmp_path)
    config[field] = value
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        _prepare(source, tmp_path / "view", _sha256(source / "config.json"))
    assert not (tmp_path / "view").exists()


@pytest.mark.parametrize("field", ["num_decay_steps", "decay_lr_ratio"])
def test_prepare_rejects_missing_compatibility_field(tmp_path: Path, field: str) -> None:
    source, _, config = _source(tmp_path)
    del config[field]
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        _prepare(source, tmp_path / "view", _sha256(source / "config.json"))


def test_prepare_rejects_wrong_source_digest_and_type(tmp_path: Path) -> None:
    source, digest, config = _source(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _prepare(source, tmp_path / "wrong-digest", "0" * 64)

    config["type"] = "not-groot"
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="type.*groot"):
        _prepare(source, tmp_path / "wrong-type", _sha256(source / "config.json"))
    assert digest != _sha256(source / "config.json")


@pytest.mark.parametrize("kind", ["file", "empty-directory", "nonempty-directory", "symlink"])
def test_prepare_refuses_every_existing_destination(tmp_path: Path, kind: str) -> None:
    source, digest, _ = _source(tmp_path)
    destination = tmp_path / "view"
    if kind == "file":
        destination.write_text("keep", encoding="utf-8")
    elif kind == "empty-directory":
        destination.mkdir()
    elif kind == "nonempty-directory":
        destination.mkdir()
        (destination / "keep").write_text("keep", encoding="utf-8")
    else:
        destination.symlink_to(tmp_path / "missing-target")

    with pytest.raises(ValueError, match="destination.*exists"):
        _prepare(source, destination, digest)
    if kind == "file":
        assert destination.read_text(encoding="utf-8") == "keep"
    elif kind == "nonempty-directory":
        assert (destination / "keep").read_text(encoding="utf-8") == "keep"
    elif kind == "symlink":
        assert destination.is_symlink()


@pytest.mark.parametrize("symlink_name", ["config.json", "model-00001-of-00002.safetensors"])
def test_prepare_rejects_source_symlinks(tmp_path: Path, symlink_name: str) -> None:
    source, digest, _ = _source(tmp_path)
    original = source / symlink_name
    target = tmp_path / f"real-{symlink_name}"
    target.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(target)

    with pytest.raises(ValueError, match="symlink|regular file"):
        _prepare(source, tmp_path / "view", digest)


def test_prepare_rejects_nested_and_unsafe_source_entries(tmp_path: Path) -> None:
    source, digest, _ = _source(tmp_path)
    (source / "nested-directory").mkdir()
    with pytest.raises(ValueError, match="regular file"):
        _prepare(source, tmp_path / "nested-view", digest)

    (source / "nested-directory").rmdir()
    (source / ".hidden-weight").write_bytes(b"hidden")
    with pytest.raises(ValueError, match="unsafe artifact name"):
        _prepare(source, tmp_path / "unsafe-view", digest)


def test_verify_rejects_source_artifact_mutation(tmp_path: Path) -> None:
    source, digest, _ = _source(tmp_path)
    destination = tmp_path / "view"
    _prepare(source, destination, digest)
    artifact = source / "model-00001-of-00002.safetensors"
    artifact.write_bytes(b"other-weight-shard")

    with pytest.raises(ValueError, match="SHA-256"):
        _verify(source, destination, digest)


def test_verify_rejects_swapped_symlink(tmp_path: Path) -> None:
    source, digest, _ = _source(tmp_path)
    destination = tmp_path / "view"
    _prepare(source, destination, digest)
    link = destination / "model-00001-of-00002.safetensors"
    link.unlink()
    link.symlink_to((source / "model-00002-of-00002.safetensors").resolve())

    with pytest.raises(ValueError, match="target"):
        _verify(source, destination, digest)


def test_verify_rejects_adapted_config_tamper(tmp_path: Path) -> None:
    source, digest, _ = _source(tmp_path)
    destination = tmp_path / "view"
    _prepare(source, destination, digest)
    adapted_path = destination / "config.json"
    adapted = json.loads(adapted_path.read_text(encoding="utf-8"))
    adapted["chunk_size"] = 99
    adapted_path.chmod(0o644)
    adapted_path.write_text(json.dumps(adapted), encoding="utf-8")

    with pytest.raises(ValueError, match="adapted config"):
        _verify(source, destination, digest)


def test_verify_rejects_manifest_tamper_even_when_json_remains_valid(tmp_path: Path) -> None:
    from scripts.prepare_reference_checkpoint import MANIFEST_FILENAME

    source, digest, _ = _source(tmp_path)
    destination = tmp_path / "view"
    manifest = _prepare(source, destination, digest)
    manifest["source_revision"] = "0" * 40
    manifest_path = destination / MANIFEST_FILENAME
    manifest_path.chmod(0o644)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="revision"):
        _verify(source, destination, digest)


def test_verify_rejects_manifest_removed_field_type_tamper(tmp_path: Path) -> None:
    from scripts.prepare_reference_checkpoint import MANIFEST_FILENAME

    source, digest, _ = _source(tmp_path)
    destination = tmp_path / "view"
    manifest = _prepare(source, destination, digest)
    manifest["removed_fields"]["num_decay_steps"] = 4000.0
    manifest_path = destination / MANIFEST_FILENAME
    manifest_path.chmod(0o644)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="removed fields"):
        _verify(source, destination, digest)


def test_verify_rejects_self_consistent_noncanonical_adapted_config_tamper(tmp_path: Path) -> None:
    from scripts.prepare_reference_checkpoint import MANIFEST_FILENAME

    source, digest, _ = _source(tmp_path)
    destination = tmp_path / "view"
    manifest = _prepare(source, destination, digest)
    adapted_path = destination / "config.json"
    adapted = json.loads(adapted_path.read_text(encoding="utf-8"))
    adapted_path.chmod(0o644)
    adapted_path.write_text(json.dumps(adapted, indent=2), encoding="utf-8")
    manifest["adapted_config_sha256"] = _sha256(adapted_path)
    manifest_path = destination / MANIFEST_FILENAME
    manifest_path.chmod(0o644)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="adapted config"):
        _verify(source, destination, digest)


def test_verify_rejects_noncanonical_manifest_and_extra_view_entry(tmp_path: Path) -> None:
    from scripts.prepare_reference_checkpoint import MANIFEST_FILENAME

    source, digest, _ = _source(tmp_path)
    destination = tmp_path / "view"
    manifest = _prepare(source, destination, digest)
    manifest_path = destination / MANIFEST_FILENAME
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    with pytest.raises(ValueError, match="canonical"):
        _verify(source, destination, digest)

    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    (destination / "extra").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="entries"):
        _verify(source, destination, digest)


def test_verify_rejects_source_config_mutation(tmp_path: Path) -> None:
    source, digest, config = _source(tmp_path)
    destination = tmp_path / "view"
    _prepare(source, destination, digest)
    config["chunk_size"] = 17
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _verify(source, destination, digest)


def test_cli_create_and_verify_modes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.prepare_reference_checkpoint import main

    source, digest, _ = _source(tmp_path)
    destination = tmp_path / "view"
    shared = [
        "--source-pretrained-model", str(source),
        "--destination-view", str(destination),
        "--expected-source-config-sha256", digest,
        "--source-repository", REPOSITORY,
        "--source-revision", REVISION,
    ]
    assert main(["create", *shared]) == 0
    create_output = json.loads(capsys.readouterr().out)
    assert create_output["source_revision"] == REVISION
    assert main(["verify", *shared]) == 0
    verify_output = json.loads(capsys.readouterr().out)
    assert verify_output == create_output
