"""Central secret policy and fail-closed upload allowlist generation.

The policy is intentionally narrow and auditable.  It denies known credential
store names, cache/environment path components, and well-formed provider token
prefixes.  It does not attempt generalized entropy scanning.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Final, Iterable, Pattern

from lehome_train.models import SyncEntry, validate_artifact_relative_path


# Exact credential-store basenames, compared case-insensitively.  Keep this
# centralized so future providers extend one documented boundary.
CREDENTIAL_FILENAMES: Final = frozenset(
    {
        ".git-credentials",
        ".netrc",
        "_netrc",
        "auth.json",
        "credentials",
        "credentials.json",
        "github_token",
        "huggingface_token",
        "hf_token",
        "openai_api_key",
        "runpod_api_key",
        "stored_tokens",
        "token",
    }
)

# Entire components denied regardless of depth.  Dot-components are rejected
# separately, including dot-caches not enumerated here.
DENIED_PATH_COMPONENTS: Final = frozenset(
    {
        "__pycache__",
        "cache",
        "caches",
        "node_modules",
    }
)

ENVIRONMENT_FILENAMES: Final = frozenset(
    {
        "environment.yml",
        "environment.yaml",
    }
)

# Supported provider access-token formats.  Prefixes and conservative minimum
# lengths minimize ordinary-text false positives while catching real tokens and
# synthetic test tokens.  No generic high-entropy heuristic is used.
ACCESS_TOKEN_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{34,128}(?![A-Za-z0-9_])"),
    re.compile(
        r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{36,255}"
        r"(?![A-Za-z0-9_])"
    ),
    re.compile(
        r"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{50,255}"
        r"(?![A-Za-z0-9_])"
    ),
    re.compile(
        r"(?<![A-Za-z0-9_])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,255}"
        r"(?![A-Za-z0-9_])"
    ),
    re.compile(r"(?<![A-Za-z0-9_])rpa_[A-Za-z0-9]{20,255}(?![A-Za-z0-9_])"),
)

_SCAN_CHUNK_SIZE: Final = 1024 * 1024
_SCAN_OVERLAP: Final = 512
_OPEN_SUPPORTS_DIR_FD: Final = os.open in os.supports_dir_fd


class ArtifactRejected(ValueError):
    """An artifact failed the upload policy.

    Messages contain only policy categories.  They never include file content,
    matched tokens, or other rejected secret text.
    """


def _rejected(category: str) -> ArtifactRejected:
    return ArtifactRejected(f"artifact rejected by upload policy: {category}")


def _validated_artifact_path(value: object) -> str | None:
    try:
        raw_path = os.fspath(value)
        if not isinstance(raw_path, str):
            return None
        return validate_artifact_relative_path(raw_path)
    except (TypeError, ValueError):
        return None


def _validate_relative_path(value: object) -> tuple[str, ...]:
    canonical = _validated_artifact_path(value)
    if canonical is None:
        raise _rejected("noncanonical relative path")
    parts = tuple(canonical.split("/"))

    for part in parts:
        folded = part.casefold()
        if folded in DENIED_PATH_COMPONENTS:
            raise _rejected("cache path")
    basename = parts[-1].casefold()
    if basename in CREDENTIAL_FILENAMES:
        raise _rejected("credential-store filename")
    if basename in ENVIRONMENT_FILENAMES or basename.startswith(".env"):
        raise _rejected("environment filename")
    return parts


def _close_descriptor(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        # Read-only descriptor close errors do not affect the artifact decision.
        pass


def _secure_descriptor_support_available() -> bool:
    return (
        hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_CLOEXEC")
        and hasattr(os, "O_NOFOLLOW")
        and _OPEN_SUPPORTS_DIR_FD
    )


def _open_regular_file_descriptor(
    root: Path,
    parts: tuple[str, ...],
) -> tuple[int | None, os.stat_result | None, str | None]:
    """Open a descendant without ever resolving a component by pathname twice."""

    if not _secure_descriptor_support_available():
        return None, None, "secure descriptor traversal unavailable"

    common_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_flags = common_flags | os.O_DIRECTORY
    directory_descriptors: list[int] = []
    file_descriptor: int | None = None
    try:
        root_descriptor = os.open(root, directory_flags)
        directory_descriptors.append(root_descriptor)
        parent_descriptor = root_descriptor
        for component in parts[:-1]:
            parent_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            directory_descriptors.append(parent_descriptor)
        file_descriptor = os.open(
            parts[-1],
            common_flags,
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            _close_descriptor(file_descriptor)
            return None, None, "non-regular file"
        return file_descriptor, metadata, None
    except (OSError, TypeError):
        if file_descriptor is not None:
            _close_descriptor(file_descriptor)
        return None, None, "unreadable path"
    finally:
        for descriptor in reversed(directory_descriptors):
            _close_descriptor(descriptor)


def _inspect_content(
    descriptor: int,
    initial_metadata: os.stat_result,
) -> tuple[tuple[str, int, bool] | None, str | None]:
    import hashlib

    digest = hashlib.sha256()
    byte_size = 0
    tail = ""
    try:
        while chunk := os.read(descriptor, _SCAN_CHUNK_SIZE):
            byte_size += len(chunk)
            digest.update(chunk)
            text = tail + chunk.decode("utf-8", errors="ignore")
            if any(pattern.search(text) for pattern in ACCESS_TOKEN_PATTERNS):
                return ("", 0, True), None
            tail = text[-_SCAN_OVERLAP:]
        final_metadata = os.fstat(descriptor)
    except OSError:
        return None, "unreadable content"

    unchanged = (
        initial_metadata.st_dev == final_metadata.st_dev
        and initial_metadata.st_ino == final_metadata.st_ino
        and initial_metadata.st_size == final_metadata.st_size
        and initial_metadata.st_mtime_ns == final_metadata.st_mtime_ns
        and initial_metadata.st_ctime_ns == final_metadata.st_ctime_ns
        and byte_size == final_metadata.st_size
    )
    if not unchanged:
        return None, "content changed during inspection"
    return (digest.hexdigest(), byte_size, False), None


def generate_upload_allowlist(
    experiment_root: str | os.PathLike[str],
    relative_paths: Iterable[str | os.PathLike[str]],
) -> tuple[SyncEntry, ...]:
    """Validate and identify explicit experiment-relative upload candidates."""

    root = Path(experiment_root)

    entries: list[SyncEntry] = []
    seen: set[str] = set()
    for supplied in relative_paths:
        parts = _validate_relative_path(supplied)
        canonical = "/".join(parts)
        if canonical in seen:
            raise _rejected("duplicate path")
        seen.add(canonical)
        descriptor, metadata, open_error = _open_regular_file_descriptor(root, parts)
        if open_error is not None or descriptor is None or metadata is None:
            raise _rejected(open_error or "unreadable path")
        try:
            inspection, inspection_error = _inspect_content(descriptor, metadata)
        finally:
            _close_descriptor(descriptor)
        if inspection_error is not None or inspection is None:
            raise _rejected(inspection_error or "unreadable content")
        sha256, byte_size, contains_token = inspection
        if contains_token:
            raise _rejected("supported access token")
        entries.append(
            SyncEntry(
                relative_path=canonical,
                sha256=sha256,
                byte_size=byte_size,
            )
        )

    return tuple(sorted(entries, key=lambda entry: entry.relative_path))
