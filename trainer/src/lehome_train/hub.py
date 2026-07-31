"""Secret-safe Hub operations behind an injected transport."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from time import sleep
from typing import Any, Callable, Literal, Mapping, Protocol

from lehome_train.constants import DEFAULT_DATA_REPO, DEFAULT_MODEL_REPO
from lehome_train.models import SyncEntry, validate_artifact_relative_path


_COMMIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_INITIAL_RETRY_DELAY_SECONDS = 0.25
_MAX_RETRY_DELAY_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class HubAccess:
    """Read and write permissions reported by a Hub transport."""

    can_read: bool
    can_write: bool
    private_repository: bool = True


@dataclass(frozen=True, slots=True)
class HubTreeEntry:
    """One path and type observed in a complete immutable repository tree."""

    relative_path: str
    entry_type: Literal["file", "directory", "symlink", "special"]

    def __post_init__(self) -> None:
        validate_artifact_relative_path(self.relative_path)
        if self.entry_type not in {"file", "directory", "symlink", "special"}:
            raise ValueError("Hub tree entry has an unsupported path type")


class HubTransientError(RuntimeError):
    """A retryable transport failure whose details must not escape."""


class HubTransport(Protocol):
    """The only remote boundary used by dataset publication and retrieval."""

    def check_access(self, *, repository: str, token: str) -> HubAccess: ...

    def upload_files(
        self,
        *,
        repository: str,
        revision: str,
        source: Path,
        entries: tuple[SyncEntry, ...],
        token: str,
        remote_prefix: str | None = None,
    ) -> str: ...

    def download_files(
        self,
        *,
        repository: str,
        revision: str,
        destination: Path,
        relative_paths: tuple[str, ...],
        token: str,
        remote_prefix: str | None = None,
    ) -> str: ...

    def list_tree(
        self,
        *,
        repository: str,
        revision: str,
        token: str,
    ) -> tuple[HubTreeEntry, ...]: ...


class HubRepositoryTransport(Protocol):
    """Explicit private-repository creation and verification boundary."""

    def ensure_private_repository(
        self,
        *,
        repository: str,
        repo_type: str,
        token: str,
        create: bool,
        timeout_seconds: float,
    ) -> HubAccess: ...


_APPROVED_REPOSITORIES = {
    DEFAULT_DATA_REPO: "dataset",
    DEFAULT_MODEL_REPO: "model",
}


class HuggingFaceHubTransport:
    """Lazy real Hub adapter with explicit tokens and bounded network calls."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        if type(timeout_seconds) not in (int, float) or not math.isfinite(
            float(timeout_seconds)
        ) or timeout_seconds <= 0:
            raise ValueError("Hub timeout must be a finite positive number")
        self.timeout_seconds = float(timeout_seconds)

    def _library(self) -> Any:
        try:
            library = importlib.import_module("huggingface_hub")
            requests = importlib.import_module("requests")
        except ImportError:
            raise RuntimeError("huggingface_hub transport dependency is unavailable") from None

        timeout_seconds = self.timeout_seconds

        class FiniteTimeoutSession(requests.Session):
            def request(self, method: str, url: str, **kwargs: object) -> Any:
                if kwargs.get("timeout") is None:
                    kwargs["timeout"] = timeout_seconds
                return super().request(method, url, **kwargs)

        library.configure_http_backend(
            backend_factory=FiniteTimeoutSession,
        )
        return library

    @staticmethod
    def _repo_type(repository: str) -> str:
        try:
            return _APPROVED_REPOSITORIES[repository]
        except KeyError:
            raise ValueError("Hub transport repository is not approved") from None

    @staticmethod
    def _revision(value: object) -> str:
        revision = getattr(value, "oid", None) or getattr(value, "sha", None)
        if isinstance(revision, str) and _COMMIT_REVISION.fullmatch(revision):
            return revision
        url = getattr(value, "commit_url", None)
        if isinstance(url, str):
            candidate = url.rstrip("/").rsplit("/", 1)[-1]
            if _COMMIT_REVISION.fullmatch(candidate):
                return candidate
        raise ValueError("Hub response did not identify an immutable revision")

    def _api(self, token: str) -> Any:
        return self._library().HfApi(token=token)

    def _repo_info(self, *, repository: str, revision: str | None, token: str) -> Any:
        api = self._api(token)
        return api.repo_info(
            repo_id=repository,
            repo_type=self._repo_type(repository),
            revision=revision,
            token=token,
            timeout=self.timeout_seconds,
        )

    def check_access(self, *, repository: str, token: str) -> HubAccess:
        try:
            info = self._repo_info(repository=repository, revision=None, token=token)
        except Exception:
            raise PermissionError("private Hub repository access check failed") from None
        private = getattr(info, "private", None) is True
        permissions = getattr(info, "permissions", None)
        can_write = False
        if isinstance(permissions, Mapping):
            can_write = permissions.get("write") is True
        elif isinstance(permissions, str):
            can_write = permissions.casefold() in {"write", "admin"}
        else:
            # Some Hub server versions omit permissions for authorized owners.
            # Actual writes remain fail-closed at upload time.
            can_write = private
        return HubAccess(
            can_read=private,
            can_write=can_write,
            private_repository=private,
        )

    def ensure_private_repository(
        self,
        *,
        repository: str,
        repo_type: str,
        token: str,
        create: bool,
        timeout_seconds: float,
    ) -> HubAccess:
        expected_type = self._repo_type(repository)
        if repo_type != expected_type:
            raise ValueError("approved Hub repository type is incompatible")
        if timeout_seconds != self.timeout_seconds:
            raise ValueError("repository operation timeout differs from transport timeout")
        api = self._api(token)
        if create:
            try:
                api.create_repo(
                    repo_id=repository,
                    repo_type=repo_type,
                    private=True,
                    exist_ok=True,
                    token=token,
                )
            except Exception:
                raise PermissionError("private Hub repository creation failed") from None
        access = self.check_access(repository=repository, token=token)
        if not access.private_repository:
            raise PermissionError("approved Hub repository must be private")
        return access

    def upload_files(
        self,
        *,
        repository: str,
        revision: str,
        source: Path,
        entries: tuple[SyncEntry, ...],
        token: str,
        remote_prefix: str | None = None,
    ) -> str:
        api = self._api(token)
        if remote_prefix is not None:
            validate_artifact_relative_path(remote_prefix, "remote_prefix")
        upload_arguments: dict[str, object] = {
            "repo_id": repository,
            "repo_type": self._repo_type(repository),
            "revision": revision,
            "folder_path": str(source),
            "allow_patterns": [entry.relative_path for entry in entries],
            "token": token,
        }
        if remote_prefix is not None:
            upload_arguments["path_in_repo"] = remote_prefix
        try:
            result = api.upload_folder(**upload_arguments)
        except (ConnectionError, TimeoutError):
            raise HubTransientError("Hub upload timed out") from None
        return self._revision(result)

    def download_files(
        self,
        *,
        repository: str,
        revision: str,
        destination: Path,
        relative_paths: tuple[str, ...],
        token: str,
        remote_prefix: str | None = None,
    ) -> str:
        if remote_prefix is not None:
            validate_artifact_relative_path(remote_prefix, "remote_prefix")
        info = self._repo_info(repository=repository, revision=revision, token=token)
        observed = self._revision(info)
        if observed != revision:
            raise ValueError("Hub readback resolved a different immutable revision")
        library = self._library()
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="lehome-hf-cache-") as cache:
            for relative_path in relative_paths:
                remote_path = (
                    relative_path
                    if remote_prefix is None
                    else f"{remote_prefix}/{relative_path}"
                )
                try:
                    downloaded = library.hf_hub_download(
                        repo_id=repository,
                        repo_type=self._repo_type(repository),
                        revision=revision,
                        filename=remote_path,
                        token=token,
                        cache_dir=cache,
                        etag_timeout=self.timeout_seconds,
                    )
                except (ConnectionError, TimeoutError):
                    raise HubTransientError("Hub download timed out") from None
                target = destination / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(downloaded, target)
        final = self._repo_info(repository=repository, revision=revision, token=token)
        if self._revision(final) != revision:
            raise ValueError("Hub readback revision changed during download")
        return revision

    def list_tree(
        self,
        *,
        repository: str,
        revision: str,
        token: str,
    ) -> tuple[HubTreeEntry, ...]:
        """List every path at one immutable commit using bounded HTTP calls."""

        info = self._repo_info(repository=repository, revision=revision, token=token)
        if self._revision(info) != revision:
            raise ValueError("Hub tree listing resolved a different immutable revision")
        api = self._api(token)
        common = {
            "repo_id": repository,
            "repo_type": self._repo_type(repository),
            "revision": revision,
            "token": token,
        }
        try:
            raw_tree = tuple(
                api.list_repo_tree(
                    **common,
                    recursive=True,
                    expand=True,
                )
            )
            listed_files = tuple(api.list_repo_files(**common))
        except (ConnectionError, TimeoutError):
            raise HubTransientError("Hub tree listing timed out") from None

        file_paths = set(listed_files)
        if len(file_paths) != len(listed_files):
            raise ValueError("Hub tree listing returned duplicate files")
        entries: list[HubTreeEntry] = []
        tree_file_paths: set[str] = set()
        for raw_entry in raw_tree:
            path = getattr(raw_entry, "path", None)
            if not isinstance(path, str):
                raise ValueError("Hub tree listing returned an invalid path")
            raw_type = getattr(raw_entry, "type", None) or getattr(
                raw_entry,
                "entry_type",
                None,
            )
            if raw_type in {"symlink", "link"}:
                entry_type: Literal["file", "directory", "symlink", "special"] = (
                    "symlink"
                )
            elif path in file_paths:
                entry_type = "file"
                tree_file_paths.add(path)
            elif hasattr(raw_entry, "tree_id"):
                entry_type = "directory"
            else:
                entry_type = "special"
            entries.append(HubTreeEntry(path, entry_type))
        if tree_file_paths != file_paths:
            raise ValueError("Hub tree and file listings disagree")
        final = self._repo_info(repository=repository, revision=revision, token=token)
        if self._revision(final) != revision:
            raise ValueError("Hub tree listing revision changed during listing")
        return tuple(entries)

def _process_token(environ: Mapping[str, str] | None) -> str:
    token = (os.environ if environ is None else environ).get("HF_TOKEN")
    if not isinstance(token, str) or not token or any(character.isspace() for character in token):
        raise ValueError("HF_TOKEN must be present in the process environment")
    return token


def require_access(
    *,
    transport: HubTransport,
    repository: str,
    read: bool,
    write: bool,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Require the requested private-repository permissions."""

    token = _process_token(environ)
    try:
        access = transport.check_access(repository=repository, token=token)
    except Exception:
        raise PermissionError("Hub permission check failed") from None
    if not isinstance(access, HubAccess):
        raise PermissionError("Hub permission check returned an invalid response")
    if read and not access.can_read:
        raise PermissionError("Hub read permission is required")
    if write and not access.can_write:
        raise PermissionError("Hub write permission is required")
    if not access.private_repository:
        raise PermissionError("Hub repository must be private")


def ensure_approved_private_repository(
    *,
    transport: HubRepositoryTransport,
    repository: str,
    create: bool,
    environ: Mapping[str, str] | None = None,
    timeout_seconds: float = 30.0,
) -> HubAccess:
    """Explicitly create or verify one approved repository as private."""

    if repository not in _APPROVED_REPOSITORIES:
        raise ValueError("Hub repository is not in the approved private allowlist")
    if type(create) is not bool:
        raise ValueError("repository creation flag must be boolean")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(
        float(timeout_seconds)
    ) or timeout_seconds <= 0:
        raise ValueError("Hub timeout must be a finite positive number")
    token = _process_token(environ)
    try:
        access = transport.ensure_private_repository(
            repository=repository,
            repo_type=_APPROVED_REPOSITORIES[repository],
            token=token,
            create=create,
            timeout_seconds=float(timeout_seconds),
        )
    except (PermissionError, ValueError):
        raise
    except Exception:
        raise RuntimeError("private Hub repository operation failed") from None
    if not isinstance(access, HubAccess) or not access.private_repository:
        raise PermissionError("approved Hub repository must be private")
    return access


def upload_files(
    *,
    transport: HubTransport,
    repository: str,
    revision: str,
    source: str | Path,
    entries: tuple[SyncEntry, ...],
    remote_prefix: str | None = None,
    environ: Mapping[str, str] | None = None,
    max_attempts: int = 3,
    sleeper: Callable[[float], None] = sleep,
) -> str:
    """Upload an explicit allowlist while passing the process token in memory."""

    token = _process_token(environ)
    if remote_prefix is not None:
        validate_artifact_relative_path(remote_prefix, "remote_prefix")
    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    for attempt in range(1, max_attempts + 1):
        try:
            arguments: dict[str, object] = {
                "repository": repository,
                "revision": revision,
                "source": Path(source),
                "entries": entries,
                "token": token,
            }
            if remote_prefix is not None:
                arguments["remote_prefix"] = remote_prefix
            resolved = transport.upload_files(**arguments)
            break
        except HubTransientError:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Hub upload failed after {max_attempts} attempts"
                ) from None
            sleeper(
                min(
                    _INITIAL_RETRY_DELAY_SECONDS * (2 ** (attempt - 1)),
                    _MAX_RETRY_DELAY_SECONDS,
                )
            )
        except Exception:
            raise RuntimeError("Hub upload failed") from None
    if not isinstance(resolved, str) or not _COMMIT_REVISION.fullmatch(resolved):
        raise ValueError("Hub upload did not resolve to an immutable revision")
    return resolved


def download_files(
    *,
    transport: HubTransport,
    repository: str,
    revision: str,
    destination: str | Path,
    relative_paths: tuple[str, ...],
    remote_prefix: str | None = None,
    environ: Mapping[str, str] | None = None,
    max_attempts: int = 3,
    sleeper: Callable[[float], None] = sleep,
) -> str:
    """Download explicit paths from one full immutable commit revision."""

    if not isinstance(revision, str) or not _COMMIT_REVISION.fullmatch(revision):
        raise ValueError("Hub download revision must be an immutable 40-character commit")
    if not relative_paths:
        raise ValueError("Hub download requires an explicit non-empty path allowlist")
    for relative_path in relative_paths:
        validate_artifact_relative_path(relative_path)
    if remote_prefix is not None:
        validate_artifact_relative_path(remote_prefix, "remote_prefix")
    token = _process_token(environ)
    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    for attempt in range(1, max_attempts + 1):
        try:
            arguments: dict[str, object] = {
                "repository": repository,
                "revision": revision,
                "destination": Path(destination),
                "relative_paths": relative_paths,
                "token": token,
            }
            if remote_prefix is not None:
                arguments["remote_prefix"] = remote_prefix
            observed = transport.download_files(**arguments)
            break
        except HubTransientError:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Hub download failed after {max_attempts} attempts"
                ) from None
            sleeper(
                min(
                    _INITIAL_RETRY_DELAY_SECONDS * (2 ** (attempt - 1)),
                    _MAX_RETRY_DELAY_SECONDS,
                )
            )
        except Exception:
            raise RuntimeError("Hub download failed") from None
    if observed != revision:
        raise ValueError("Hub download did not preserve the immutable revision")
    return observed


def list_repository_tree(
    *,
    transport: HubTransport,
    repository: str,
    revision: str,
    environ: Mapping[str, str] | None = None,
    max_attempts: int = 3,
    sleeper: Callable[[float], None] = sleep,
) -> tuple[HubTreeEntry, ...]:
    """List a complete repository tree at one full immutable commit revision."""

    if not isinstance(revision, str) or not _COMMIT_REVISION.fullmatch(revision):
        raise ValueError("Hub tree revision must be an immutable 40-character commit")
    token = _process_token(environ)
    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    for attempt in range(1, max_attempts + 1):
        try:
            observed = transport.list_tree(
                repository=repository,
                revision=revision,
                token=token,
            )
            break
        except HubTransientError:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Hub tree listing failed after {max_attempts} attempts"
                ) from None
            sleeper(
                min(
                    _INITIAL_RETRY_DELAY_SECONDS * (2 ** (attempt - 1)),
                    _MAX_RETRY_DELAY_SECONDS,
                )
            )
        except Exception:
            raise RuntimeError("Hub tree listing failed") from None
    if not isinstance(observed, tuple) or not all(
        isinstance(entry, HubTreeEntry) for entry in observed
    ):
        raise ValueError("Hub tree listing returned an invalid response")
    paths = tuple(entry.relative_path for entry in observed)
    if len(set(paths)) != len(paths):
        raise ValueError("Hub tree listing returned duplicate paths")
    return observed
