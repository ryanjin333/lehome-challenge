"""Atomic, checksum-verified storage for immutable raw episodes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Mapping
from uuid import uuid4


MANIFEST_NAME = "SHA256SUMS.json"


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("artifact path must be relative and path-safe")
    return path


def _directory_fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    """Write a canonical JSON file durably without exposing a partial result."""
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite existing artifact file: {path.name}")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _directory_fsync(path.parent)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _iter_regular_files(root: Path) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("episode artifact root must be a real directory")
    files: list[Path] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory_name in directory_names:
            if (current_path / directory_name).is_symlink():
                raise ValueError("episode artifacts must not contain symlinks")
        for file_name in file_names:
            path = current_path / file_name
            if path.is_symlink():
                raise ValueError("episode artifacts must not contain symlinks")
            if not path.is_file():
                raise ValueError("episode artifacts may contain only regular files")
            files.append(path)
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_sha256_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    """Return a sorted complete manifest; a terminal manifest cannot list itself."""
    manifest: dict[str, dict[str, int | str]] = {}
    for path in _iter_regular_files(root):
        relative = path.relative_to(root).as_posix()
        if relative == MANIFEST_NAME:
            raise ValueError("manifest must not include itself")
        manifest[relative] = {"sha256": _sha256(path), "size": path.stat().st_size}
    return manifest


def _load_manifest(path: Path) -> dict[str, dict[str, object]]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("manifest contains duplicate paths")
            result[key] = value
        return result

    if path.is_symlink() or not path.is_file():
        raise ValueError("episode manifest is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("episode manifest is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("episode manifest must be an object")
    result: dict[str, dict[str, object]] = {}
    for relative, entry in payload.items():
        if not isinstance(relative, str) or not isinstance(entry, dict):
            raise ValueError("episode manifest entry is invalid")
        _safe_relative_path(relative)
        if relative == MANIFEST_NAME:
            raise ValueError("manifest must not include itself")
        size, digest = entry.get("size"), entry.get("sha256")
        if not isinstance(size, int) or size < 0 or not isinstance(digest, str):
            raise ValueError("episode manifest entry is invalid")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("episode manifest hash is invalid")
        result[relative] = entry
    return result


class EpisodeArtifactWriter:
    """Write an episode under ``.pending`` and atomically publish it to ``raw``."""

    def __init__(self, run_root: Path, episode_id: str) -> None:
        _safe_relative_path(episode_id)
        self.run_root = run_root.resolve()
        self.episode_id = episode_id
        self.staging = self.run_root / ".pending" / episode_id
        destination = self.run_root / "raw" / episode_id
        if destination.exists() or destination.is_symlink() or self.staging.exists() or self.staging.is_symlink():
            raise ValueError(f"episode artifact already exists: {episode_id}")
        pending = self.staging.parent
        pending.mkdir(parents=True, exist_ok=True)
        if pending.is_symlink():
            raise ValueError("episode staging directory must not be a symlink")
        self.staging.mkdir()
        self._finalized = False

    def append_annotation(self, value: Mapping[str, object]) -> None:
        if self._finalized:
            raise ValueError("episode has already been finalized")
        path = self.staging / "annotations.jsonl"
        if path.is_symlink():
            raise ValueError("episode artifacts must not contain symlinks")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def finalize(
        self,
        episode: Mapping[str, object],
        *,
        required_videos: tuple[str, ...] = (),
    ) -> Path:
        if self._finalized:
            raise ValueError("episode has already been finalized")
        if not (self.staging / "annotations.jsonl").is_file():
            raise ValueError("episode annotations are missing")
        if len(set(required_videos)) != len(required_videos):
            raise ValueError("required videos must not contain duplicates")
        for name in required_videos:
            relative = _safe_relative_path(name)
            path = self.staging / "videos" / Path(*relative.parts)
            if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"required video is missing or empty: {name}")
        if (self.staging / MANIFEST_NAME).exists() or (self.staging / MANIFEST_NAME).is_symlink():
            raise ValueError("manifest must not include itself")
        _iter_regular_files(self.staging)
        payload = dict(episode)
        recorded_id = payload.setdefault("episode_id", self.episode_id)
        if recorded_id != self.episode_id:
            raise ValueError("episode metadata ID does not match artifact ID")
        atomic_write_json(self.staging / "episode.json", payload)
        manifest = build_sha256_manifest(self.staging)
        atomic_write_json(self.staging / MANIFEST_NAME, manifest)
        destination = self.run_root / "raw" / self.episode_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.parent.is_symlink() or destination.exists() or destination.is_symlink():
            raise ValueError(f"episode artifact already exists: {self.episode_id}")
        self.staging.rename(destination)
        _directory_fsync(destination.parent)
        self._finalized = True
        return destination


def verify_episode_manifest(episode_dir: Path) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Verify an episode and return its metadata with the verified manifest."""
    if episode_dir.is_symlink() or not episode_dir.is_dir():
        raise ValueError("episode artifact root must be a real directory")
    manifest = _load_manifest(episode_dir / MANIFEST_NAME)
    actual = _iter_regular_files(episode_dir)
    actual_names = {
        relative
        for path in actual
        if (relative := path.relative_to(episode_dir).as_posix()) != MANIFEST_NAME
    }
    listed_names = set(manifest)
    unlisted = actual_names - listed_names
    missing = listed_names - actual_names
    if unlisted:
        raise ValueError(f"episode contains unlisted files: {sorted(unlisted)}")
    if missing:
        raise ValueError(f"episode manifest lists missing files: {sorted(missing)}")
    for relative in sorted(listed_names):
        path = episode_dir / Path(*PurePosixPath(relative).parts)
        entry = manifest[relative]
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"episode manifest size mismatch: {relative}")
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"episode manifest hash mismatch: {relative}")
    if "episode.json" not in manifest:
        raise ValueError("episode manifest is missing episode.json")
    try:
        episode = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("episode metadata is invalid") from error
    if not isinstance(episode, dict) or episode.get("episode_id") != episode_dir.name:
        raise ValueError("episode metadata ID does not match artifact ID")
    return episode, manifest


def verify_episode(episode_dir: Path) -> dict[str, object]:
    """Fail closed unless every regular file is manifest-listed and checksummed."""
    return verify_episode_manifest(episode_dir)[0]
