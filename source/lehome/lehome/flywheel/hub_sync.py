"""Idempotent, readback-verified publication of accepted rollout episodes.

An episode counts toward a round's 150 only after local validation; its
Hugging Face publication is tracked separately through these receipts and
must be complete before the round is sealed.  Every remote path is
immutable (``rollout-rounds/<round>/<attempt>/...``), uploads are retried
on transient transport failure, and a receipt is written only after a fresh
readback proves byte-for-byte equality with the local accepted artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Callable, Mapping, Protocol, Sequence
from uuid import uuid4


_ROUND_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_FRESH_ROUND_ID_PATTERN = re.compile(r"^fresh-12k-[a-z0-9-]{1,112}$")
_FRESH_RUN_ID_PATTERN = re.compile(r"^fresh-run-[a-z0-9-]{1,112}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_PUBLICATION_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
MANIFEST_NAME = "SHA256SUMS.json"
REMOTE_ROOT = "rollout-rounds"


class HubSyncError(RuntimeError):
    """Publication or readback failed; no receipt may be claimed."""


def _is_transient_transport_error(error: BaseException) -> bool:
    """Keep credential/validation failures fail-closed and single-shot.

    ``hub_sync`` intentionally does not import the trainer runtime: it is
    also used by the rollout appliance's lightweight finalizer.  The shared
    transport exposes its only retryable failures under these two canonical
    class names.
    """

    return type(error).__name__ in {"HubTransientError", "HubRateLimitError"}


@dataclass(frozen=True, slots=True)
class SyncEntry:
    """One accepted-episode file selected for immutable publication."""

    relative_path: str
    sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class SyncReceipt:
    """Durable proof that one episode is byte-identical on the Hub."""

    attempt_id: str
    repository: str
    round_id: str
    run_id: str | None
    remote_prefix: str
    immutable_revision: str
    entry_count: int
    episode_sha256: str
    readback_verified: bool
    receipt_path: Path


class _HubFileEntry(Protocol):
    relative_path: str
    entry_type: str


class HubTransportLike(Protocol):
    """Structural subset of the trainer HubTransport used for episode sync."""

    def check_access(self, *, repository: str, token: str) -> object: ...

    def upload_files(
        self,
        *,
        repository: str,
        revision: str,
        source: Path,
        entries: Sequence[object],
        token: str,
        remote_prefix: str | None = None,
    ) -> str: ...

    def download_files(
        self,
        *,
        repository: str,
        revision: str,
        destination: Path,
        relative_paths: Sequence[str],
        token: str,
        remote_prefix: str | None = None,
    ) -> str: ...

    def list_tree(
        self,
        *,
        repository: str,
        revision: str,
        token: str,
        remote_prefix: str | None = None,
    ) -> Sequence[_HubFileEntry]: ...


def _directory_fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _directory_fsync(path.parent)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _collect_entries(episode_dir: Path) -> tuple[SyncEntry, ...]:
    """Hash every publishable file; the local manifest index stays local."""
    if episode_dir.is_symlink() or not episode_dir.is_dir():
        raise HubSyncError("accepted episode directory missing or unsafe")
    entries: list[SyncEntry] = []
    for current, directory_names, file_names in os.walk(episode_dir, followlinks=False):
        current_path = Path(current)
        for directory_name in directory_names:
            if (current_path / directory_name).is_symlink():
                raise HubSyncError("accepted episode must not contain symlinked directories")
        for file_name in file_names:
            path = current_path / file_name
            if path.is_symlink():
                raise HubSyncError("accepted episode must not contain symlinked files")
            relative = path.relative_to(episode_dir)
            if PurePosixPath(relative.as_posix()).parts[0] == "..":  # unreachable; defensive
                raise HubSyncError("unsafe path inside accepted episode")
            if relative.as_posix() == MANIFEST_NAME:
                continue
            sha256, byte_size = _sha256_file(path)
            entries.append(SyncEntry(relative.as_posix(), sha256, byte_size))
    if not entries:
        raise HubSyncError("accepted episode has no publishable files")
    entries.sort(key=lambda item: item.relative_path)
    return tuple(entries)


def _episode_digest(entries: Sequence[SyncEntry]) -> str:
    # This is deliberately byte-for-byte the same canonical tree digest used
    # by fresh replay admission (`build_success_replay_matrix`).  A receipt
    # that calls the same artifact by a different digest is not usable as
    # immutable source evidence.
    canonical = json.dumps(
        [{"relative_path": e.relative_path, "sha256": e.sha256, "byte_size": e.byte_size} for e in entries],
        sort_keys=True, separators=(",", ":"),
    ) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class HubSyncDaemon:
    """Background publisher for accepted episodes with idempotent receipts."""

    def __init__(
        self,
        *,
        repository: str,
        round_id: str,
        run_id: str | None = None,
        token: str,
        transport: HubTransportLike,
        accepted_root: Path,
        receipts_root: Path,
        readback_root: Path,
        revision: str,
        max_attempts: int = 3,
    ) -> None:
        if not isinstance(repository, str) or not repository:
            raise HubSyncError("repository must be a non-empty string")
        if not _ROUND_ID_PATTERN.match(round_id):
            raise HubSyncError(f"round_id must be path-safe lowercase, got {round_id!r}")
        if run_id is not None and (
            not isinstance(run_id, str)
            or not run_id
            or "/" in run_id
            or "\\" in run_id
            or run_id in {".", ".."}
        ):
            raise HubSyncError("run_id must be a non-empty path-safe string when supplied")
        if _FRESH_ROUND_ID_PATTERN.fullmatch(round_id) is not None and (
            not isinstance(run_id, str) or _FRESH_RUN_ID_PATTERN.fullmatch(run_id) is None
        ):
            raise HubSyncError("fresh simple collection sync requires its exact fresh run_id")
        if not isinstance(token, str) or not token:
            raise HubSyncError("token must be a non-empty string")
        if (
            not isinstance(revision, str)
            or not _PUBLICATION_REF_PATTERN.fullmatch(revision)
            or _REVISION_PATTERN.fullmatch(revision)
            or any(component in {"", ".", ".."} for component in revision.split("/"))
        ):
            raise HubSyncError(f"revision must be a canonical mutable branch ref, got {revision!r}")
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
            raise HubSyncError("max_attempts must be a positive integer")
        self.repository = repository
        self.round_id = round_id
        self.run_id = run_id
        self._token = token
        self._transport = transport
        self._accepted_root = Path(accepted_root).resolve()
        self.receipts_root = Path(receipts_root).resolve()
        self._readback_root = Path(readback_root).resolve()
        self._publication_ref = revision
        self._max_attempts = max_attempts
        if self._accepted_root.is_symlink() or not self._accepted_root.is_dir():
            raise HubSyncError("accepted root must be a real directory")
        self.receipts_root.mkdir(parents=True, exist_ok=True)
        self._readback_root.mkdir(parents=True, exist_ok=True)

    def _receipt_path(self, attempt_id: str) -> Path:
        if not attempt_id or "/" in attempt_id or attempt_id in {".", ".."}:
            raise HubSyncError(f"attempt_id is not path-safe: {attempt_id!r}")
        return self.receipts_root / f"{attempt_id}.sync.json"

    def _call_with_retry(self, operation: Callable[[], object], label: str) -> object:
        last_error: BaseException | None = None
        for _ in range(self._max_attempts):
            try:
                return operation()
            except Exception as error:  # noqa: BLE001 - transport errors are heterogeneous
                last_error = error
                if not _is_transient_transport_error(error):
                    break
        raise HubSyncError(f"{label} failed after {self._max_attempts} attempts: {last_error}") from last_error

    def sync_episode(self, attempt_id: str, episode_dir: Path) -> SyncReceipt:
        """Publish one accepted episode immutably and prove it by readback."""
        if not attempt_id:
            raise HubSyncError("attempt_id must be non-empty")
        receipt_path = self._receipt_path(attempt_id)
        resolved = Path(episode_dir).resolve()
        if not resolved.is_relative_to(self._accepted_root):
            raise HubSyncError("episode must live inside the accepted root")
        entries = _collect_entries(resolved)
        digest = _episode_digest(entries)

        if receipt_path.is_file() and not receipt_path.is_symlink():
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                existing.get("episode_sha256") != digest
                or existing.get("readback_verified") is not True
                or existing.get("round_id") != self.round_id
                or (self.run_id is not None and existing.get("run_id") != self.run_id)
            ):
                raise HubSyncError("existing sync receipt does not match the current accepted episode")
            return self._receipt_from_payload(existing, receipt_path)
        if receipt_path.is_symlink():
            raise HubSyncError("sync receipt path is unsafe")

        remote_prefix = f"{REMOTE_ROOT}/{self.round_id}/{attempt_id}"
        immutable_revision = self._call_with_retry(
            lambda: self._transport.upload_files(
                repository=self.repository,
                revision=self._publication_ref,
                source=resolved,
                entries=tuple(entries),
                token=self._token,
                remote_prefix=remote_prefix,
            ),
            f"{attempt_id} upload",
        )
        if not isinstance(immutable_revision, str) or not _REVISION_PATTERN.fullmatch(immutable_revision):
            raise HubSyncError(
                f"{attempt_id} upload did not return an immutable commit: {immutable_revision!r}"
            )

        def _readback() -> None:
            tree = self._transport.list_tree(
                repository=self.repository, revision=immutable_revision, token=self._token, remote_prefix=remote_prefix,
            )
            observed = {
                entry.relative_path.removeprefix(remote_prefix + "/")
                for entry in tree
                if entry.entry_type == "file" and entry.relative_path.startswith(remote_prefix + "/")
            }
            expected = {entry.relative_path for entry in entries}
            if observed != expected:
                raise HubSyncError(
                    f"{attempt_id} remote tree does not match: missing {sorted(expected - observed)}, extra {sorted(observed - expected)}"
                )
            destination = self._readback_root / f"{attempt_id}-{uuid4().hex}"
            try:
                self._transport.download_files(
                    repository=self.repository, revision=immutable_revision, destination=destination,
                    relative_paths=tuple(entry.relative_path for entry in entries),
                    token=self._token, remote_prefix=remote_prefix,
                )
                for entry in entries:
                    path = destination / entry.relative_path
                    if path.is_symlink() or not path.is_file():
                        raise HubSyncError(f"{attempt_id} readback missing file: {entry.relative_path}")
                    sha256, byte_size = _sha256_file(path)
                    if sha256 != entry.sha256 or byte_size != entry.byte_size:
                        raise HubSyncError(f"{attempt_id} readback hash mismatch: {entry.relative_path}")
            finally:
                shutil.rmtree(destination, ignore_errors=True)

        self._call_with_retry(_readback, f"{attempt_id} readback")

        payload = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "repository": self.repository,
            "round_id": self.round_id,
            "remote_prefix": remote_prefix,
            "publication_ref": self._publication_ref,
            "immutable_revision": immutable_revision,
            "entry_count": len(entries),
            "episode_sha256": digest,
            "readback_verified": True,
        }
        if self.run_id is not None:
            payload["run_id"] = self.run_id
        _write_json_atomic(receipt_path, payload)
        return self._receipt_from_payload(payload, receipt_path)

    def pending_for_round(self, attempt_ids: Sequence[str]) -> tuple[str, ...]:
        """Attempts in the round that still lack a verified publication receipt."""
        pending: list[str] = []
        for attempt_id in attempt_ids:
            receipt_path = self._receipt_path(attempt_id)
            if receipt_path.is_symlink() or not receipt_path.is_file():
                pending.append(attempt_id)
                continue
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                payload.get("readback_verified") is not True
                or payload.get("round_id") != self.round_id
                or (self.run_id is not None and payload.get("run_id") != self.run_id)
            ):
                pending.append(attempt_id)
        return tuple(pending)

    def round_sealable(self, attempt_ids: Sequence[str]) -> bool:
        """A round may be sealed only when every accepted episode is durable."""
        return bool(attempt_ids) and not self.pending_for_round(attempt_ids)

    def _receipt_from_payload(self, payload: Mapping[str, object], receipt_path: Path) -> SyncReceipt:
        return SyncReceipt(
            attempt_id=str(payload["attempt_id"]),
            repository=str(payload["repository"]),
            round_id=str(payload["round_id"]),
            run_id=str(payload["run_id"]) if isinstance(payload.get("run_id"), str) else None,
            remote_prefix=str(payload["remote_prefix"]),
            immutable_revision=str(payload["immutable_revision"]),
            entry_count=int(payload["entry_count"]),  # type: ignore[arg-type]
            episode_sha256=str(payload["episode_sha256"]),
            readback_verified=bool(payload["readback_verified"]),
            receipt_path=receipt_path,
        )


__all__ = ["HubSyncDaemon", "HubSyncError", "SyncEntry", "SyncReceipt"]
