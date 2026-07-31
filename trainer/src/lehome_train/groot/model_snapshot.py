"""Pinned, complete base-model hydration without implicit credentials."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Callable, Mapping, Protocol

from lehome_train.constants import MODEL_REVISION
from lehome_train.io import atomic_write_json, sha256_file


BASE_MODEL_REPOSITORY = "nvidia/GR00T-N1.7-3B"
MODEL_SNAPSHOT_MANIFEST = "lehome_model_snapshot.json"


@dataclass(frozen=True, slots=True)
class ModelSnapshotFile:
    relative_path: str
    byte_size: int

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in self.relative_path
            or self.relative_path == MODEL_SNAPSHOT_MANIFEST
        ):
            raise ValueError("model snapshot contains an unsafe path")
        if type(self.byte_size) is not int or self.byte_size < 0:
            raise ValueError("model snapshot file size is invalid")


class ModelSnapshotTransport(Protocol):
    def list_files(
        self, *, repository: str, revision: str, token: str | None
    ) -> tuple[str, tuple[ModelSnapshotFile, ...]]: ...

    def download_snapshot(
        self,
        *,
        repository: str,
        revision: str,
        destination: Path,
        token: str | None,
    ) -> str: ...


class HuggingFaceModelSnapshotTransport:
    """Lazy public/private Hub adapter with implicit authentication disabled."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        if type(timeout_seconds) not in (int, float) or timeout_seconds <= 0:
            raise ValueError("model Hub timeout must be positive")
        self.timeout_seconds = float(timeout_seconds)

    def _library(self) -> Any:
        try:
            return importlib.import_module("huggingface_hub")
        except ImportError:
            raise RuntimeError("huggingface_hub is unavailable") from None

    @staticmethod
    def _token(token: str | None) -> str | bool:
        return False if token is None else token

    def _revision(self, repository: str, revision: str, token: str | None) -> str:
        info = self._library().HfApi(token=self._token(token)).model_info(
            repo_id=repository,
            revision=revision,
            token=self._token(token),
            timeout=self.timeout_seconds,
        )
        observed = getattr(info, "sha", None)
        if observed != revision:
            raise ValueError("model Hub request resolved a different revision")
        return observed

    def list_files(
        self, *, repository: str, revision: str, token: str | None
    ) -> tuple[str, tuple[ModelSnapshotFile, ...]]:
        observed = self._revision(repository, revision, token)
        api = self._library().HfApi(token=self._token(token))
        tree = tuple(
            api.list_repo_tree(
                repo_id=repository,
                repo_type="model",
                revision=revision,
                recursive=True,
                expand=True,
                token=self._token(token),
            )
        )
        files: list[ModelSnapshotFile] = []
        for entry in tree:
            path = getattr(entry, "path", None)
            size = getattr(entry, "size", None)
            entry_type = getattr(entry, "type", None)
            if entry_type in {"directory", "tree"} or hasattr(entry, "tree_id"):
                continue
            if not isinstance(path, str) or type(size) is not int:
                raise ValueError("model Hub tree contains an unsupported entry")
            files.append(ModelSnapshotFile(path, size))
        if not files or len({item.relative_path for item in files}) != len(files):
            raise ValueError("model Hub tree is empty or duplicated")
        if self._revision(repository, revision, token) != observed:
            raise ValueError("model Hub revision changed during listing")
        return observed, tuple(sorted(files, key=lambda item: item.relative_path))

    def download_snapshot(
        self,
        *,
        repository: str,
        revision: str,
        destination: Path,
        token: str | None,
    ) -> str:
        cache = destination.parent / f".{destination.name}.hf-cache"
        try:
            self._library().snapshot_download(
                repo_id=repository,
                repo_type="model",
                revision=revision,
                local_dir=destination,
                cache_dir=cache,
                token=self._token(token),
            )
            local_metadata = destination / ".cache"
            if local_metadata.is_dir() and not local_metadata.is_symlink():
                shutil.rmtree(local_metadata)
        finally:
            shutil.rmtree(cache, ignore_errors=True)
        return self._revision(repository, revision, token)


def _process_token(environ: Mapping[str, str] | None) -> str | None:
    environment = os.environ if environ is None else environ
    value = environment.get("HF_TOKEN")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("HF_TOKEN must be non-empty when provided")
    return value


def _complete_files(root: Path) -> dict[str, Path]:
    observed: dict[str, Path] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        for entry in os.scandir(directory):
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if entry.is_symlink():
                raise ValueError("model snapshot contains a symlink")
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                observed[relative] = path
            else:
                raise ValueError("model snapshot contains a special path")
    return observed


def download_base_model(
    destination_path: str | os.PathLike[str],
    *,
    repository: str,
    revision: str,
    transport: ModelSnapshotTransport,
    staging_root: str | os.PathLike[str],
    environ: Mapping[str, str] | None = None,
    free_space_probe: Callable[[Path], int] | None = None,
) -> Path:
    """Atomically expose the exact pinned complete model snapshot."""

    if repository != BASE_MODEL_REPOSITORY or revision != MODEL_REVISION:
        raise ValueError("base model repository and revision must equal the pinned identity")
    destination = Path(destination_path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite a base model destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(staging_root)
    if not staging.is_dir() or staging.is_symlink():
        raise ValueError("model staging root must be an existing regular directory")
    if os.stat(staging).st_dev != os.stat(destination.parent).st_dev:
        raise ValueError("model staging and destination must share one filesystem")
    token = _process_token(environ)
    observed_revision, entries = transport.list_files(
        repository=repository,
        revision=revision,
        token=token,
    )
    if observed_revision != revision:
        raise ValueError("model listing did not preserve the pinned revision")
    payload_bytes = sum(entry.byte_size for entry in entries)
    reserve = max(1024**3, payload_bytes // 20)
    available = (
        shutil.disk_usage(staging).free
        if free_space_probe is None
        else free_space_probe(staging)
    )
    if type(available) is not int or available < 0:
        raise ValueError("model staging free-space probe is invalid")
    if available < payload_bytes + reserve:
        raise ValueError("model staging filesystem has insufficient capacity")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".incomplete", dir=staging
        )
    )
    try:
        downloaded_revision = transport.download_snapshot(
            repository=repository,
            revision=revision,
            destination=temporary,
            token=token,
        )
        if downloaded_revision != revision:
            raise ValueError("model download did not preserve the pinned revision")
        observed = _complete_files(temporary)
        expected = {entry.relative_path: entry.byte_size for entry in entries}
        if set(observed) != set(expected) or any(
            observed[path].stat().st_size != size for path, size in expected.items()
        ):
            raise ValueError("model snapshot is not complete")
        manifest_entries = [
            {
                "relative_path": path,
                "byte_size": expected[path],
                "sha256": sha256_file(observed[path]),
            }
            for path in sorted(expected)
        ]
        atomic_write_json(
            temporary / MODEL_SNAPSHOT_MANIFEST,
            {
                "revision": revision,
                "artifacts": manifest_entries,
            },
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
