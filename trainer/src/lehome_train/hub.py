"""Secret-safe Hub operations behind an injected transport."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from time import sleep
from typing import Callable, Mapping, Protocol

from lehome_train.models import SyncEntry, validate_artifact_relative_path


_COMMIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_INITIAL_RETRY_DELAY_SECONDS = 0.25
_MAX_RETRY_DELAY_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class HubAccess:
    """Read and write permissions reported by a Hub transport."""

    can_read: bool
    can_write: bool


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
    ) -> str: ...

    def download_files(
        self,
        *,
        repository: str,
        revision: str,
        destination: Path,
        relative_paths: tuple[str, ...],
        token: str,
    ) -> str: ...


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


def upload_files(
    *,
    transport: HubTransport,
    repository: str,
    revision: str,
    source: str | Path,
    entries: tuple[SyncEntry, ...],
    environ: Mapping[str, str] | None = None,
    max_attempts: int = 3,
    sleeper: Callable[[float], None] = sleep,
) -> str:
    """Upload an explicit allowlist while passing the process token in memory."""

    token = _process_token(environ)
    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    for attempt in range(1, max_attempts + 1):
        try:
            resolved = transport.upload_files(
                repository=repository,
                revision=revision,
                source=Path(source),
                entries=entries,
                token=token,
            )
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
    token = _process_token(environ)
    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    for attempt in range(1, max_attempts + 1):
        try:
            observed = transport.download_files(
                repository=repository,
                revision=revision,
                destination=Path(destination),
                relative_paths=relative_paths,
                token=token,
            )
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
