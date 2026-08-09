"""Immutable derived GR00T configuration without duplicating upstream weights."""

from __future__ import annotations

import json
import os
from pathlib import Path
import errno
from typing import Any, Mapping

from lehome_train.constants import COSMOS_REPOSITORY, COSMOS_REVISION
from lehome_train.io import atomic_write_json, sha256_file
from lehome_train.b1k.snapshot_integrity import ValidatedArtifact, read_snapshot_json, verify_artifact_stat_invariants
from lehome_train.b1k.snapshot_state import bound_destination, destination_lock, fsync_directory, open_staged_destination, validate_destination_binding


_DERIVED_RECEIPT = ".b1k-derived-receipt.json"
_DERIVED_INTENT = ".b1k-derived-intent.json"


def _regular_files(root: Path, *, exclude: set[str] = frozenset()) -> list[dict[str, object]]:
    """Return an exact, symlink-free recursive file inventory."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("model directory is unsafe")
    artifacts: list[dict[str, object]] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            if (current_path / name).is_symlink():
                raise ValueError("model directory contains unsafe symlinks")
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("model directory contains unsafe files")
            relative = path.relative_to(root).as_posix()
            if relative not in exclude:
                artifacts.append({"path": relative, "byte_size": path.stat().st_size, "sha256": sha256_file(path)})
    return sorted(artifacts, key=lambda item: str(item["path"]))


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        value = read_snapshot_json(path, "derived model receipt")
    except ValueError as error:
        raise ValueError("derived model receipt is invalid") from error
    return value


def _validated_cosmos_identity(value: Mapping[str, object]) -> dict[str, str]:
    expected = {"repository", "revision", "receipt_sha256", "artifacts_sha256"}
    if set(value) != expected or value.get("repository") != COSMOS_REPOSITORY or value.get("revision") != COSMOS_REVISION:
        raise ValueError("derived model Cosmos snapshot identity is invalid")
    normalized = {key: value[key] for key in expected}
    if any(type(item) is not str or len(item) != 64 or any(character not in "0123456789abcdef" for character in item) for key, item in normalized.items() if key.endswith("_sha256")):
        raise ValueError("derived model Cosmos snapshot identity is invalid")
    return {key: str(item) for key, item in normalized.items()}


def _derivation_identity(cosmos_path: str, artifacts: list[dict[str, object]], upstream_config_sha256: str, cosmos_identity: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "cosmos_path": cosmos_path,
        "cosmos_snapshot": _validated_cosmos_identity(cosmos_identity),
        "upstream_config_sha256": upstream_config_sha256,
        "artifacts": artifacts,
    }


def _stage_file_paths(root: Path, artifacts: list[dict[str, object]]) -> None:
    """Reject links and every stage entry outside the deterministic artifact set."""

    expected_files = {str(item["path"]) for item in artifacts} | {"config.json", _DERIVED_INTENT, _DERIVED_RECEIPT}
    expected_directories = {""}
    for relative in expected_files:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or relative not in expected_directories:
                raise ValueError("derived model incomplete staging directory is unsafe")
        for name in files:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.is_file() or relative not in expected_files:
                raise ValueError("derived model incomplete staging directory is unsafe")


def _fsync_stage_directories(root: Path) -> None:
    """Durably record every hardlink and control-file directory bottom-up."""

    directories = [Path(current) for current, _, _ in os.walk(root, followlinks=False)]
    for directory in reversed(directories):
        fsync_directory(directory)


def _write_or_replace_expected_file(source: Path, target: Path, *, verified: bool = False) -> None:
    """Only replace the exact expected target inside a validated sibling stage."""

    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise ValueError("derived model incomplete staging directory is unsafe")
        if target.stat().st_size == source.stat().st_size and (not verified or (target.stat().st_dev, target.stat().st_ino) == (source.stat().st_dev, source.stat().st_ino)):
            return
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno == errno.EXDEV:
            raise ValueError("derived model requires same-filesystem hardlinks") from error
        raise


def _validation_identity(artifacts: tuple[ValidatedArtifact, ...]) -> tuple[list[dict[str, object]], str]:
    values = {artifact.path: artifact for artifact in artifacts}
    config = values.get("config.json")
    if config is None:
        raise ValueError("upstream validation is missing config.json")
    return ([artifact.to_dict() for artifact in artifacts if artifact.path != "config.json"], config.sha256)


def _hardlinks_match(upstream: Path, derived: Path, artifacts: list[dict[str, object]]) -> bool:
    expected = {str(item["path"]) for item in artifacts}
    observed: set[str] = set()
    for current, directories, files in os.walk(derived, followlinks=False):
        current_path = Path(current)
        if any((current_path / name).is_symlink() for name in directories):
            return False
        for name in files:
            target = current_path / name
            relative = target.relative_to(derived).as_posix()
            if relative in {"config.json", _DERIVED_RECEIPT, _DERIVED_INTENT}:
                continue
            if target.is_symlink() or not target.is_file() or relative not in expected:
                return False
            source = upstream / relative
            if source.is_symlink() or not source.is_file():
                return False
            source_stat, target_stat = source.stat(), target.stat()
            if (source_stat.st_dev, source_stat.st_ino, source_stat.st_size) != (target_stat.st_dev, target_stat.st_ino, target_stat.st_size):
                return False
            observed.add(relative)
    return observed == expected


def _valid_derived_receipt(upstream: Path, derived: Path, cosmos_path: str, cosmos_identity: Mapping[str, object], *, allow_staging_intent: bool = False, upstream_validation: tuple[ValidatedArtifact, ...] | None = None) -> dict[str, str] | None:
    """Validate a completed derived directory before it is ever reused."""

    receipt_path = derived / _DERIVED_RECEIPT
    intent_path = derived / _DERIVED_INTENT
    if derived.is_symlink() or not derived.is_dir() or receipt_path.is_symlink() or not receipt_path.is_file() or (intent_path.exists() and not allow_staging_intent):
        return None
    try:
        receipt = _read_receipt(receipt_path)
        source_config = upstream / "config.json"
        derived_config = derived / "config.json"
        if source_config.is_symlink() or derived_config.is_symlink() or not source_config.is_file() or not derived_config.is_file():
            return None
        source_artifacts, upstream_config_sha256 = _validation_identity(upstream_validation) if upstream_validation is not None else (_regular_files(upstream, exclude={"config.json"}), sha256_file(source_config))
        derived_config_value = json.loads(derived_config.read_text(encoding="utf-8"))
        expected_keys = {"schema_version", "cosmos_path", "cosmos_snapshot", "upstream_config_sha256", "derived_config_sha256", "artifacts"}
        if set(receipt) != expected_keys or receipt["schema_version"] != 1 or receipt["cosmos_path"] != cosmos_path or receipt["cosmos_snapshot"] != _validated_cosmos_identity(cosmos_identity):
            return None
        if not isinstance(derived_config_value, dict) or derived_config_value.get("model_name") != cosmos_path:
            return None
        if receipt["upstream_config_sha256"] != upstream_config_sha256 or receipt["derived_config_sha256"] != sha256_file(derived_config):
            return None
        if receipt["artifacts"] != source_artifacts:
            return None
        if upstream_validation is not None:
            verify_artifact_stat_invariants(upstream, upstream_validation)
        if upstream_validation is not None and not _hardlinks_match(upstream, derived, source_artifacts):
            return None
        if upstream_validation is None and _regular_files(derived, exclude={"config.json", _DERIVED_RECEIPT, _DERIVED_INTENT}) != source_artifacts:
            return None
        return {"upstream_config_sha256": str(receipt["upstream_config_sha256"]), "derived_config_sha256": str(receipt["derived_config_sha256"])}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def derive_groot_config(
    upstream: str | Path,
    derived: str | Path,
    *,
    cosmos_path: str = "/workspace/models/cosmos",
    cosmos_identity: Mapping[str, object],
    upstream_validation: tuple[ValidatedArtifact, ...] | None = None,
) -> dict[str, str]:
    """Serialize one derived destination lifecycle with a sibling flock."""

    with destination_lock(derived):
        return _derive_groot_config_locked(
            upstream,
            derived,
            cosmos_path=cosmos_path,
            cosmos_identity=cosmos_identity,
            upstream_validation=upstream_validation,
        )


def _derive_groot_config_locked(
    upstream: str | Path,
    derived: str | Path,
    *,
    cosmos_path: str = "/workspace/models/cosmos",
    cosmos_identity: Mapping[str, object],
    upstream_validation: tuple[ValidatedArtifact, ...] | None = None,
) -> dict[str, str]:
    """Atomically promote a receipted derived model, or reuse one after rehashing it.

    The only mutable work location is a deterministic sibling ``.incomplete``
    directory.  Completed trees are never repaired in place: a bad receipt is a
    fail-closed condition so a caller can inspect or replace it deliberately.
    """

    upstream, derived = Path(upstream), Path(derived)
    source_config = upstream / "config.json"
    if cosmos_path != "/workspace/models/cosmos" or source_config.is_symlink() or not source_config.is_file():
        raise ValueError("derived model paths are invalid")
    try:
        derived.resolve().relative_to(upstream.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("derived model must not be nested in upstream")
    if upstream.is_symlink() or not upstream.is_dir():
        raise ValueError("upstream model contains unsafe files")
    if upstream_validation is not None:
        verify_artifact_stat_invariants(upstream, upstream_validation)
    source_artifacts, upstream_config_sha256 = _validation_identity(upstream_validation) if upstream_validation is not None else (_regular_files(upstream, exclude={"config.json"}), sha256_file(source_config))
    identity = _derivation_identity(cosmos_path, source_artifacts, upstream_config_sha256, cosmos_identity)
    if derived.exists() or derived.is_symlink():
        reused = _valid_derived_receipt(upstream, derived, cosmos_path, cosmos_identity, upstream_validation=upstream_validation)
        if reused is None:
            raise ValueError("derived model receipt is missing or does not validate")
        return reused

    staging, _ = open_staged_destination(
        derived,
        intent_name=_DERIVED_INTENT,
        identity=identity,
        read_intent=_read_receipt,
        label="derived model",
    )
    staging = bound_destination(derived, staging)
    promoted = bound_destination(derived)
    _stage_file_paths(staging, source_artifacts)
    intent_path = staging / _DERIVED_INTENT
    receipt_path = staging / _DERIVED_RECEIPT
    if receipt_path.exists():
        if intent_path.exists() and _read_receipt(intent_path) != identity:
            raise ValueError("derived model incomplete staging receipt does not match")
        if _valid_derived_receipt(upstream, staging, cosmos_path, cosmos_identity, allow_staging_intent=True, upstream_validation=upstream_validation) is None:
            raise ValueError("derived model incomplete staged receipt does not validate")
        validate_destination_binding(derived)
        intent_path.unlink(missing_ok=True)
        validate_destination_binding(derived)
        fsync_directory(staging)
        validate_destination_binding(derived)
        _fsync_stage_directories(staging)
        os.replace(staging, promoted)
        validate_destination_binding(derived)
        fsync_directory(promoted.parent)
        validate_destination_binding(derived)
        result = _valid_derived_receipt(upstream, promoted, cosmos_path, cosmos_identity, upstream_validation=upstream_validation)
        if result is None:
            raise ValueError("derived model receipt did not validate after promotion")
        return result
    if intent_path.is_symlink() or not intent_path.is_file() or _read_receipt(intent_path) != identity:
        raise ValueError("derived model incomplete staging receipt does not match")
    try:
        for artifact in source_artifacts:
            validate_destination_binding(derived)
            relative = Path(str(artifact["path"]))
            source, target = upstream / relative, staging / relative
            _write_or_replace_expected_file(source, target, verified=upstream_validation is not None)
        config = json.loads(source_config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("upstream config is invalid")
        config["model_name"] = cosmos_path
        target_config = staging / "config.json"
        validate_destination_binding(derived)
        atomic_write_json(target_config, config)
        validate_destination_binding(derived)
        receipt = {
            **identity,
            "derived_config_sha256": sha256_file(target_config),
        }
        validate_destination_binding(derived)
        atomic_write_json(staging / _DERIVED_RECEIPT, receipt)
        validate_destination_binding(derived)
        if upstream_validation is not None:
            verify_artifact_stat_invariants(upstream, upstream_validation)
        if _valid_derived_receipt(upstream, staging, cosmos_path, cosmos_identity, allow_staging_intent=True, upstream_validation=upstream_validation) is None:
            raise ValueError("derived model receipt did not validate before promotion")
        validate_destination_binding(derived)
        intent_path.unlink()
        validate_destination_binding(derived)
        fsync_directory(staging)
        validate_destination_binding(derived)
        _fsync_stage_directories(staging)
        os.replace(staging, promoted)
        validate_destination_binding(derived)
        fsync_directory(promoted.parent)
        validate_destination_binding(derived)
    except Exception:
        # Keep the exact sibling staging tree for inspection; never delete a
        # completed cache or a path outside this deterministic staging target.
        raise
    result = _valid_derived_receipt(upstream, promoted, cosmos_path, cosmos_identity, upstream_validation=upstream_validation)
    if result is None:
        raise ValueError("derived model receipt did not validate after promotion")
    return result
