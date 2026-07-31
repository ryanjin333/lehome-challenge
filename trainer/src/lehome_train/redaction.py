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

from lehome_train.models import SyncEntry


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


class ArtifactRejected(ValueError):
    """An artifact failed the upload policy.

    Messages contain only policy categories.  They never include file content,
    matched tokens, or other rejected secret text.
    """


def _rejected(category: str) -> ArtifactRejected:
    return ArtifactRejected(f"artifact rejected by upload policy: {category}")


def _validate_relative_path(relative: Path) -> tuple[str, ...]:
    if relative.is_absolute():
        raise _rejected("absolute path")
    parts = relative.parts
    if not parts or parts == (".",):
        raise _rejected("empty path")
    if ".." in parts:
        raise _rejected("path traversal")

    for part in parts:
        folded = part.casefold()
        if part.startswith("."):
            raise _rejected("dot path")
        if folded in DENIED_PATH_COMPONENTS:
            raise _rejected("cache path")
    basename = parts[-1].casefold()
    if basename in CREDENTIAL_FILENAMES:
        raise _rejected("credential-store filename")
    if basename in ENVIRONMENT_FILENAMES or basename.startswith(".env"):
        raise _rejected("environment filename")
    return parts


def _reject_symlink_components(root: Path, parts: tuple[str, ...]) -> Path:
    if root.is_symlink():
        raise _rejected("symlink")

    candidate = root
    for part in parts:
        candidate = candidate / part
        try:
            metadata = candidate.lstat()
        except OSError as error:
            raise _rejected("unreadable path") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise _rejected("symlink")
    return candidate


def _inspect_content(path: Path) -> tuple[str, int, bool]:
    import hashlib

    digest = hashlib.sha256()
    byte_size = 0
    tail = ""
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_SCAN_CHUNK_SIZE):
                byte_size += len(chunk)
                digest.update(chunk)
                text = tail + chunk.decode("utf-8", errors="ignore")
                if any(pattern.search(text) for pattern in ACCESS_TOKEN_PATTERNS):
                    return "", 0, True
                tail = text[-_SCAN_OVERLAP:]
    except OSError as error:
        raise _rejected("unreadable content") from error
    return digest.hexdigest(), byte_size, False


def generate_upload_allowlist(
    experiment_root: str | os.PathLike[str],
    relative_paths: Iterable[str | os.PathLike[str]],
) -> tuple[SyncEntry, ...]:
    """Validate and identify explicit experiment-relative upload candidates."""

    root = Path(experiment_root)
    if not root.is_dir() or root.is_symlink():
        raise _rejected("invalid experiment root")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise _rejected("invalid experiment root") from error

    entries: list[SyncEntry] = []
    seen: set[str] = set()
    for supplied in relative_paths:
        relative = Path(supplied)
        parts = _validate_relative_path(relative)
        candidate = _reject_symlink_components(root, parts)
        try:
            resolved_candidate = candidate.resolve(strict=True)
            resolved_candidate.relative_to(resolved_root)
            metadata = candidate.stat()
        except (OSError, ValueError) as error:
            raise _rejected("path outside experiment root") from error
        if not stat.S_ISREG(metadata.st_mode):
            raise _rejected("non-regular file")

        canonical = Path(*parts).as_posix()
        if canonical in seen:
            raise _rejected("duplicate path")
        seen.add(canonical)
        sha256, byte_size, contains_token = _inspect_content(candidate)
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
