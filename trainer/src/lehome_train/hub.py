"""Secret-safe Hub operations behind an injected transport."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
_RATE_LIMIT_RETRY_DELAY_SECONDS = 300.0
_MAX_PARALLEL_DOWNLOADS = 16


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
        # Repository listings legitimately contain control files such as
        # ``.gitattributes``.  They are not publication artifacts, so retain
        # the traversal/alias protections without applying the stricter rule
        # that rejects every dot-prefixed component.
        if (
            not self.relative_path
            or self.relative_path.startswith("/")
            or "\\" in self.relative_path
            or "\x00" in self.relative_path
            or any(component in {"", ".", ".."} for component in self.relative_path.split("/"))
        ):
            raise ValueError("Hub tree entry path is not canonical and relative")
        if self.entry_type not in {"file", "directory", "symlink", "special"}:
            raise ValueError("Hub tree entry has an unsupported path type")


class HubTransientError(RuntimeError):
    """A retryable transport failure whose details must not escape."""


class HubRateLimitError(HubTransientError):
    """A retryable rate limit that needs the provider's full window to clear."""


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
    def _is_rate_limited(error: Exception) -> bool:
        response = getattr(error, "response", None)
        if getattr(response, "status_code", None) == 429 or getattr(error, "status_code", None) == 429:
            return True
        return "rate limit" in str(error).casefold() or "too many requests" in str(error).casefold()

    @staticmethod
    def _preserve_partial_prefixed_download(
        *, destination: Path, remote_prefix: str | None, relative_paths: tuple[str, ...],
    ) -> None:
        """Keep any complete materialized files if snapshot_download aborts mid-batch."""

        if remote_prefix is None:
            return
        source_root = destination / remote_prefix
        if not source_root.is_dir() or source_root.is_symlink():
            return
        for relative_path in relative_paths:
            source, target = source_root / relative_path, destination / relative_path
            if not source.is_file() or source.is_symlink() or target.exists() or target.is_symlink():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
        for directory in sorted(
            (path for path in source_root.rglob("*") if path.is_dir() and not path.is_symlink()),
            key=lambda path: len(path.parts), reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            source_root.rmdir()
        except OSError:
            pass
        try:
            (destination / remote_prefix.split("/", 1)[0]).rmdir()
        except OSError:
            pass

    @staticmethod
    def _safe_download_path(destination: Path, relative_path: str) -> Path:
        """Create a non-symlinked parent path beneath one caller-owned destination."""

        validate_artifact_relative_path(relative_path)
        if destination.exists() and destination.is_symlink():
            raise ValueError("Hub download destination must not be a symlink")
        destination.mkdir(parents=True, exist_ok=True)
        current = destination
        for component in relative_path.split("/")[:-1]:
            current /= component
            if current.exists():
                if current.is_symlink() or not current.is_dir():
                    raise ValueError("Hub download path contains an unsafe component")
            else:
                current.mkdir()
        return destination / relative_path

    @classmethod
    def _materialize_prefixed_download(
        cls,
        *,
        destination: Path,
        remote_prefix: str,
        relative_path: str,
        downloaded: str | Path,
    ) -> None:
        """Move one direct exact-file result into the readback's stable layout."""

        remote_path = f"{remote_prefix}/{relative_path}"
        expected_source = cls._safe_download_path(destination, remote_path)
        source = Path(downloaded)
        if source != expected_source or not source.is_file() or source.is_symlink():
            raise ValueError("Hub direct download returned an unsafe file path")
        target = cls._safe_download_path(destination, relative_path)
        if target.exists() and target.is_symlink():
            raise ValueError("Hub download target must not be a symlink")
        os.replace(source, target)

    @staticmethod
    def _repo_type(repository: str) -> str:
        try:
            return _APPROVED_REPOSITORIES[repository]
        except KeyError:
            raise ValueError("Hub transport repository is not approved") from None

    @staticmethod
    def _fine_grained_repository_write(
        access_token: Mapping[str, object],
        repository: str,
    ) -> bool:
        fine_grained = access_token.get("fineGrained")
        if not isinstance(fine_grained, Mapping):
            return False
        scopes = fine_grained.get("scoped")
        if not isinstance(scopes, list):
            return False
        owner = repository.partition("/")[0].casefold()
        repository_name = repository.casefold()
        for scope in scopes:
            if not isinstance(scope, Mapping):
                continue
            permissions = scope.get("permissions")
            entity = scope.get("entity")
            if (
                not isinstance(permissions, list)
                or "repo.write" not in permissions
                or not isinstance(entity, Mapping)
            ):
                continue
            entity_type = entity.get("type")
            entity_name = entity.get("name")
            if not isinstance(entity_type, str) or not isinstance(entity_name, str):
                continue
            normalized_type = entity_type.casefold()
            normalized_name = entity_name.casefold()
            if normalized_type in {"user", "organization", "org"}:
                if normalized_name == owner:
                    return True
            elif normalized_type in {"dataset", "model"}:
                if normalized_name in {repository_name, repository_name.partition("/")[2]}:
                    return True
        return False

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
        repository_write: bool | None = None
        if isinstance(permissions, Mapping) and "write" in permissions:
            repository_write = permissions.get("write") is True
        elif isinstance(permissions, str):
            repository_write = permissions.casefold() in {"write", "admin"}
        if repository_write is None:
            try:
                identity = self._api(token).whoami(token=token)
            except Exception:
                raise PermissionError("private Hub repository access check failed") from None
            token_write = False
            if isinstance(identity, Mapping):
                account_name = identity.get("name")
                auth = identity.get("auth")
                if isinstance(auth, Mapping):
                    access_token = auth.get("accessToken")
                    if isinstance(access_token, Mapping):
                        role = access_token.get("role")
                        repository_owner = repository.partition("/")[0]
                        token_write = (
                            isinstance(account_name, str)
                            and account_name.casefold() == repository_owner.casefold()
                            and isinstance(role, str)
                            and role.casefold() == "write"
                        )
                        if isinstance(role, str) and role.casefold() == "finegrained":
                            token_write = self._fine_grained_repository_write(
                                access_token,
                                repository,
                            )
            repository_write = token_write
        return HubAccess(
            can_read=private,
            can_write=private and repository_write,
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
            return self._revision(result)
        except Exception as error:
            # A large-folder commit can become durable before the client loses
            # the final response. Resolve the branch head and let the caller's
            # immutable full readback decide whether the upload is complete.
            try:
                recovered = self._repo_info(
                    repository=repository,
                    revision=revision,
                    token=token,
                )
                return self._revision(recovered)
            except Exception:
                if isinstance(error, (ConnectionError, TimeoutError)):
                    raise HubTransientError("Hub upload timed out") from None
                raise error

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
        for relative_path in relative_paths:
            validate_artifact_relative_path(relative_path)
        info = self._repo_info(repository=repository, revision=revision, token=token)
        observed = self._revision(info)
        if observed != revision:
            raise ValueError("Hub readback resolved a different immutable revision")
        library = self._library()
        destination.mkdir(parents=True, exist_ok=True)
        try:
            if remote_prefix is None:
                library.snapshot_download(
                    repo_id=repository,
                    repo_type=self._repo_type(repository),
                    revision=revision,
                    allow_patterns=list(relative_paths),
                    token=token,
                    local_dir=destination,
                    etag_timeout=self.timeout_seconds,
                )
            else:
                for relative_path in relative_paths:
                    remote_path = f"{remote_prefix}/{relative_path}"
                    target = self._safe_download_path(destination, relative_path)
                    if target.is_symlink() or (target.exists() and not target.is_file()):
                        raise ValueError("Hub download target must be a regular file")
                    if target.is_file():
                        # Prefixed readback callers verify the completed artifact
                        # content before receipt, so retain exact files materialized
                        # before a rate limit instead of requesting them again.
                        continue
                    self._safe_download_path(destination, remote_path)
                    downloaded = library.hf_hub_download(
                        repo_id=repository,
                        repo_type=self._repo_type(repository),
                        revision=revision,
                        filename=remote_path,
                        token=token,
                        local_dir=destination,
                        local_dir_use_symlinks=False,
                        etag_timeout=self.timeout_seconds,
                    )
                    self._materialize_prefixed_download(
                        destination=destination,
                        remote_prefix=remote_prefix,
                        relative_path=relative_path,
                        downloaded=downloaded,
                    )
        except (ConnectionError, TimeoutError):
            self._preserve_partial_prefixed_download(
                destination=destination, remote_prefix=remote_prefix, relative_paths=relative_paths,
            )
            shutil.rmtree(destination / ".cache", ignore_errors=True)
            raise HubTransientError("Hub download timed out") from None
        except Exception as error:
            self._preserve_partial_prefixed_download(
                destination=destination, remote_prefix=remote_prefix, relative_paths=relative_paths,
            )
            shutil.rmtree(destination / ".cache", ignore_errors=True)
            if self._is_rate_limited(error):
                raise HubRateLimitError("Hub download rate limited") from None
            raise
        if remote_prefix is not None:
            shutil.rmtree(destination / remote_prefix.split("/", 1)[0])
        shutil.rmtree(destination / ".cache", ignore_errors=True)
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
        except HubRateLimitError:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Hub download rate limited after {max_attempts} attempts"
                ) from None
            sleeper(_RATE_LIMIT_RETRY_DELAY_SECONDS)
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
