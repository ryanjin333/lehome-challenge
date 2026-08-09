"""Private Hugging Face release verification, immutable proof, and retry-safe probes."""

from __future__ import annotations

import io
import json
import os
import re
import stat
import subprocess
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .dockerhub import TokenSource


_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_OPERATION_RE = re.compile(r"^[a-f0-9]{32}$")
_BUCKET_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_LOCAL_PROBE_NAME_RE = re.compile(r"^[a-f0-9]{32}\.(?:upload|readback)\.json$")
_REQUIRED_REPOSITORIES = {
    "model": ("ryanjin333/behavior1k-groot-n17-models", "model"),
    "dataset": ("ryanjin333/behavior1k-groot-n17-rollouts", "dataset"),
}
_CHECKPOINT_BUCKET = "ryanjin333/behavior1k-groot-n17-checkpoints"
_PROBE_BYTES = b'{"purpose":"b1k-private-release-bootstrap"}\n'


class HubProbeError(ValueError):
    """Raised when private Hub verification or an exact bootstrap probe is unsafe."""


class HuggingFaceClient(Protocol):
    def repo_info(self, repo_id: str, repo_type: str, token: str) -> Mapping[str, Any]: ...
    def upload_bytes(self, repo_id: str, repo_type: str, path: str, content: bytes, token: str) -> str: ...
    def read_file(self, repo_id: str, repo_type: str, path: str, revision: str, token: str) -> bytes: ...
    def delete_file(self, repo_id: str, repo_type: str, path: str, expected_commit: str, token: str) -> str: ...
    def absence_proof(self, repo_id: str, repo_type: str, path: str, revision: str, token: str) -> "HubAbsenceProof": ...
    def resolve_exact_file_head(self, repo_id: str, repo_type: str, path: str, content: bytes, token: str) -> str | None: ...
    def current_revision(self, repo_id: str, repo_type: str, token: str) -> str | None: ...


@dataclass(frozen=True)
class HubRepository:
    repo_id: str
    repo_type: str


@dataclass(frozen=True)
class HubAbsenceProof:
    repo_id: str
    repo_type: str
    key: str
    revision: str
    repository_private: bool
    revision_exists: bool
    key_absent: bool


@dataclass(frozen=True)
class HubProbeOperation:
    operation_id: str
    role: str
    repository: HubRepository
    prefix: str
    key: str


@dataclass(frozen=True)
class CheckpointBucket:
    """Exact private Hub Storage Bucket destination, never a Hub model repository."""

    bucket_id: str = _CHECKPOINT_BUCKET


@dataclass(frozen=True)
class ReleaseDestinations:
    model: HubRepository
    checkpoint_bucket: CheckpointBucket
    dataset: HubRepository


class CheckpointBucketHelperClient:
    """Version-1 client for the existing isolated Hub 1.24 bucket-helper contract."""

    def __init__(self, executable: str, token_file: str, *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run):
        if not isinstance(executable, str) or not executable.startswith("/") or not isinstance(token_file, str) or not token_file.startswith("/"):
            raise HubProbeError("checkpoint bucket helper configuration is invalid")
        self._executable, self._token_file, self._runner = executable, token_file, runner

    def info(self, bucket: CheckpointBucket) -> Mapping[str, Any]:
        return self._request("info", {"bucket_id": self._bucket(bucket)}, 30)

    def list(self, bucket: CheckpointBucket, prefix: str) -> tuple[str, ...]:
        result = self._request("list", {"bucket_id": self._bucket(bucket), "prefix": self._prefix(prefix)}, 30)
        files = result["files"]
        assert isinstance(files, list)
        return tuple(item["path"] for item in files)

    def upload(self, bucket: CheckpointBucket, local_path: str, remote_path: str) -> None:
        self._request("upload", {"bucket_id": self._bucket(bucket), "local_path": self._local(local_path), "remote_path": self._key(remote_path)}, 21_600)

    def download(self, bucket: CheckpointBucket, remote_path: str, local_path: str) -> None:
        self._request("download", {"bucket_id": self._bucket(bucket), "remote_path": self._key(remote_path), "local_path": self._local(local_path)}, 21_600)

    def delete(self, bucket: CheckpointBucket, remote_path: str) -> None:
        self._request("delete", {"bucket_id": self._bucket(bucket), "paths": [self._key(remote_path)]}, 300)

    def _request(self, operation: str, payload: Mapping[str, object], timeout: float) -> Mapping[str, Any]:
        request = json.dumps({"version": 1, "operation": operation, "payload": dict(payload)}, sort_keys=True, separators=(",", ":")) + "\n"
        environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE") if key in os.environ}
        environment["B1K_HF_TOKEN_FILE"] = self._token_file
        try:
            completed = self._runner((self._executable,), input=request, text=True, capture_output=True, env=environment, timeout=timeout, check=False)
            response = json.loads(completed.stdout)
        except Exception:
            raise HubProbeError("checkpoint bucket helper operation failed") from None
        expected = {"info": {"private"}, "list": {"files"}, "upload": set(), "download": set(), "delete": set()}[operation]
        if completed.returncode != 0 or not isinstance(response, dict) or response.get("ok") is not True or not isinstance(response.get("result"), dict) or set(response["result"]) != expected or (operation == "info" and type(response["result"].get("private")) is not bool):
            raise HubProbeError("checkpoint bucket helper operation failed")
        if operation == "list":
            files = response["result"]["files"]
            if not isinstance(files, list) or any(
                not isinstance(item, dict)
                or set(item) != {"path", "size", "xet_hash", "type"}
                or not self._is_key(item.get("path"))
                or type(item.get("size")) is not int
                or item["size"] < 0
                or item.get("xet_hash") is not None and not isinstance(item.get("xet_hash"), str)
                or item.get("type") != "file"
                for item in files
            ):
                raise HubProbeError("checkpoint bucket helper operation failed")
        return response["result"]

    @staticmethod
    def _bucket(bucket: CheckpointBucket) -> str:
        if not isinstance(bucket, CheckpointBucket) or bucket.bucket_id != _CHECKPOINT_BUCKET:
            raise HubProbeError("checkpoint bucket must match the exact pinned B1K bucket")
        return bucket.bucket_id

    @staticmethod
    def _key(value: str) -> str:
        if not CheckpointBucketHelperClient._is_key(value):
            raise HubProbeError("checkpoint bucket key is invalid")
        return value

    @staticmethod
    def _prefix(value: str) -> str:
        if value != "" and not CheckpointBucketHelperClient._is_key(value):
            raise HubProbeError("checkpoint bucket prefix is invalid")
        return value

    @staticmethod
    def _is_key(value: object) -> bool:
        return isinstance(value, str) and bool(_BUCKET_KEY_RE.fullmatch(value)) and ".." not in value.split("/")

    @staticmethod
    def _local(value: str) -> str:
        if not isinstance(value, str) or not value.startswith("/workspace/checkpoints/"):
            raise HubProbeError("checkpoint bucket local path is invalid")
        return value


class CheckpointProbeFiles(Protocol):
    """Local staging boundary; all helper-compatible paths remain below /workspace/checkpoints/."""

    def write_bytes(self, path: str, content: bytes) -> None: ...
    def read_bytes(self, path: str) -> bytes: ...
    def remove(self, path: str) -> None: ...


class WorkspaceCheckpointProbeFiles:
    """Filesystem implementation for the helper's /workspace/checkpoints restriction."""

    _root = Path("/workspace/checkpoints")
    _logical_root = "/workspace/checkpoints"
    _staging_directory = ".b1k-release-probes"

    def write_bytes(self, path: str, content: bytes) -> None:
        if not isinstance(content, bytes):
            raise HubProbeError("checkpoint probe local staging failed")
        parent = self._staging_descriptor(create=True)
        descriptor: int | None = None
        try:
            name = self._name(path)
            try:
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=parent,
                )
            except FileExistsError:
                descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), dir_fd=parent)
                self._validate_regular_descriptor(descriptor, require_private=True)
                if self._read_descriptor(descriptor) != content:
                    raise OSError("existing probe content differs")
                return
            self._validate_regular_descriptor(descriptor, require_private=False)
            self._write_descriptor(descriptor, content)
            os.fchmod(descriptor, 0o600)
            self._validate_regular_descriptor(descriptor, require_private=True)
        except (OSError, ValueError):
            raise HubProbeError("checkpoint probe local staging failed") from None
        finally:
            self._close(descriptor)
            self._close(parent)

    def read_bytes(self, path: str) -> bytes:
        parent = self._staging_descriptor(create=False)
        if parent is None:
            raise HubProbeError("checkpoint probe local readback failed")
        descriptor: int | None = None
        try:
            descriptor = os.open(self._name(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), dir_fd=parent)
            self._validate_regular_descriptor(descriptor, require_private=False)
            return self._read_descriptor(descriptor)
        except (OSError, ValueError):
            raise HubProbeError("checkpoint probe local readback failed") from None
        finally:
            self._close(descriptor)
            self._close(parent)

    def remove(self, path: str) -> None:
        parent = self._staging_descriptor(create=False)
        if parent is None:
            return
        descriptor: int | None = None
        try:
            name = self._name(path)
            try:
                descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), dir_fd=parent)
            except FileNotFoundError:
                return
            self._validate_regular_descriptor(descriptor, require_private=False)
            os.unlink(name, dir_fd=parent)
        except (OSError, ValueError):
            raise HubProbeError("checkpoint probe local cleanup failed") from None
        finally:
            self._close(descriptor)
            self._close(parent)

    @classmethod
    def _target(cls, value: str) -> Path:
        return cls._root / cls._staging_directory / cls._name(value)

    @classmethod
    def _name(cls, value: str) -> str:
        prefix = f"{cls._logical_root}/{cls._staging_directory}/"
        if not isinstance(value, str) or not value.startswith(prefix):
            raise HubProbeError("checkpoint probe local path is invalid")
        name = value.removeprefix(prefix)
        if not _LOCAL_PROBE_NAME_RE.fullmatch(name):
            raise HubProbeError("checkpoint probe local path is invalid")
        return name

    @classmethod
    def _staging_descriptor(cls, *, create: bool) -> int | None:
        root = cls._root_descriptor()
        try:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            try:
                staging = os.open(cls._staging_directory, flags, dir_fd=root)
            except FileNotFoundError:
                if not create:
                    return None
                try:
                    os.mkdir(cls._staging_directory, 0o700, dir_fd=root)
                except FileExistsError:
                    pass
                staging = os.open(cls._staging_directory, flags, dir_fd=root)
            metadata = os.fstat(staging)
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
                cls._close(staging)
                raise OSError("invalid probe staging directory")
            return staging
        except OSError:
            raise HubProbeError("checkpoint probe local staging failed") from None
        finally:
            cls._close(root)

    @classmethod
    def _root_descriptor(cls) -> int:
        if not cls._root.is_absolute() or any(part in {"", ".", ".."} for part in cls._root.parts[1:]):
            raise HubProbeError("checkpoint probe local staging failed")
        descriptor: int | None = None
        try:
            descriptor = os.open(os.sep, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
            for component in cls._root.parts[1:]:
                child = os.open(
                    component,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                cls._close(descriptor)
                descriptor = child
                if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                    raise OSError("checkpoint root is not a directory")
            return descriptor
        except OSError:
            cls._close(descriptor)
            raise HubProbeError("checkpoint probe local staging failed") from None

    @staticmethod
    def _validate_regular_descriptor(descriptor: int, *, require_private: bool) -> None:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or require_private and stat.S_IMODE(metadata.st_mode) != 0o600:
            raise OSError("invalid probe file")

    @staticmethod
    def _read_descriptor(descriptor: int) -> bytes:
        chunks = []
        while True:
            chunk = os.read(descriptor, 8192)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    @staticmethod
    def _write_descriptor(descriptor: int, content: bytes) -> None:
        offset = 0
        while offset < len(content):
            offset += os.write(descriptor, content[offset:])
        os.fsync(descriptor)

    @staticmethod
    def _close(descriptor: int | None) -> None:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


@dataclass(frozen=True)
class CheckpointBucketProbeOperation:
    operation_id: str
    bucket: CheckpointBucket
    prefix: str
    key: str
    upload_path: str
    download_path: str


@dataclass(frozen=True)
class CheckpointBucketProbeReceipt:
    bucket_id: str
    prefix: str
    key: str


@dataclass
class _CheckpointBucketProbeState:
    operation: CheckpointBucketProbeOperation
    uploaded: bool = False
    readback_verified: bool = False
    deleted: bool = False
    receipt: CheckpointBucketProbeReceipt | None = None


@dataclass(frozen=True)
class HubProbeReceipt:
    role: str
    repo_id: str
    repo_type: str
    prefix: str
    key: str
    upload_commit: str
    delete_commit: str


@dataclass
class _ProbeState:
    operation: HubProbeOperation
    upload_commit: str | None = None
    delete_commit: str | None = None
    readback_verified: bool = False
    receipt: HubProbeReceipt | None = None


class HuggingFaceHubClient:
    """Concrete adapter pinned to the ``huggingface_hub==0.36.2`` API surface."""

    def __init__(
        self,
        *,
        api_factory: Callable[[str], Any] | None = None,
        download: Callable[..., str] | None = None,
        missing_file_errors: tuple[type[BaseException], ...] | None = None,
    ):
        if api_factory is None or download is None or missing_file_errors is None:
            try:
                from huggingface_hub import HfApi, hf_hub_download
                from huggingface_hub.errors import EntryNotFoundError
            except Exception:
                raise HubProbeError("huggingface_hub dependency is unavailable") from None
            api_factory = api_factory or (lambda token: HfApi(token=token))
            download = download or hf_hub_download
            missing_file_errors = missing_file_errors or (EntryNotFoundError,)
        self._api_factory = api_factory
        self._download = download
        self._missing_file_errors = missing_file_errors

    def repo_info(self, repo_id: str, repo_type: str, token: str) -> Mapping[str, Any]:
        try:
            info = self._api(token).repo_info(repo_id=repo_id, repo_type=repo_type, token=token, timeout=30)
        except Exception:
            raise HubProbeError("Hub repository lookup failed") from None
        return {"private": getattr(info, "private", None)}

    def upload_bytes(self, repo_id: str, repo_type: str, path: str, content: bytes, token: str) -> str:
        try:
            result = self._api(token).upload_file(
                path_or_fileobj=io.BytesIO(content), path_in_repo=path, repo_id=repo_id, repo_type=repo_type, token=token
            )
        except Exception:
            raise HubProbeError("Hub upload failed") from None
        return self._commit_id(result, "Hub upload")

    def read_file(self, repo_id: str, repo_type: str, path: str, revision: str, token: str) -> bytes:
        try:
            downloaded = self._download(
                repo_id=repo_id, filename=path, repo_type=repo_type, revision=revision, token=token,
                force_download=True, local_files_only=False, etag_timeout=30,
            )
            return Path(downloaded).read_bytes()
        except Exception:
            raise HubProbeError("Hub immutable readback failed") from None

    def delete_file(self, repo_id: str, repo_type: str, path: str, expected_commit: str, token: str) -> str:
        try:
            result = self._api(token).delete_file(
                path_in_repo=path, repo_id=repo_id, repo_type=repo_type, parent_commit=expected_commit, token=token
            )
        except Exception:
            raise HubProbeError("Hub exact deletion failed") from None
        return self._commit_id(result, "Hub exact deletion")

    def absence_proof(self, repo_id: str, repo_type: str, path: str, revision: str, token: str) -> HubAbsenceProof:
        private = self.repo_info(repo_id, repo_type, token).get("private") is True
        try:
            self._api(token).repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision, token=token, timeout=30)
        except Exception:
            raise HubProbeError("Hub immutable revision lookup failed") from None
        try:
            self._download(
                repo_id=repo_id, filename=path, repo_type=repo_type, revision=revision, token=token,
                force_download=True, local_files_only=False, etag_timeout=30,
            )
        except self._missing_file_errors:
            absent = True
        except Exception:
            raise HubProbeError("Hub absence lookup failed") from None
        else:
            absent = False
        return HubAbsenceProof(repo_id, repo_type, path, revision, private, True, absent)

    def resolve_exact_file_head(self, repo_id: str, repo_type: str, path: str, content: bytes, token: str) -> str | None:
        try:
            commits = self._api(token).list_repo_commits(repo_id=repo_id, repo_type=repo_type, token=token)
            head = next(iter(commits), None)
            revision = getattr(head, "commit_id", getattr(head, "oid", None))
            if not isinstance(revision, str) or not _COMMIT_RE.fullmatch(revision):
                return None
            actual = self.read_file(repo_id, repo_type, path, revision, token)
        except HubProbeError:
            return None
        except Exception:
            return None
        return revision if actual == content else None

    def current_revision(self, repo_id: str, repo_type: str, token: str) -> str | None:
        try:
            commits = self._api(token).list_repo_commits(repo_id=repo_id, repo_type=repo_type, token=token)
            head = next(iter(commits), None)
            revision = getattr(head, "commit_id", getattr(head, "oid", None))
        except Exception:
            return None
        return revision if isinstance(revision, str) and _COMMIT_RE.fullmatch(revision) else None

    def _api(self, token: str) -> Any:
        return self._api_factory(token)

    @staticmethod
    def _commit_id(result: Any, operation: str) -> str:
        commit = getattr(result, "oid", getattr(result, "commit_id", None))
        if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
            raise HubProbeError(f"{operation} failed")
        return commit


class HuggingFaceReleaseVerifier:
    """Verify exact private destinations and reconcile one idempotent probe operation."""

    def __init__(self, hub: HuggingFaceClient, credentials: TokenSource):
        self._hub = hub
        self._credentials = credentials
        self._operations: dict[str, _ProbeState] = {}
        self._checkpoint_operations: dict[str, _CheckpointBucketProbeState] = {}

    def verify_private_repositories(self, repositories: Mapping[str, HubRepository]) -> dict[str, HubRepository]:
        self._validate_repository_set(repositories)
        token = self._credentials.resolve()
        for role, repository in repositories.items():
            self._verify_private(role, repository, token)
        return dict(repositories)

    def verify_private_destinations(self, destinations: ReleaseDestinations, checkpoint_client: CheckpointBucketHelperClient) -> ReleaseDestinations:
        if not isinstance(destinations, ReleaseDestinations):
            raise HubProbeError("release destinations are invalid")
        self.verify_private_repositories({"model": destinations.model, "dataset": destinations.dataset})
        self._verify_private_checkpoint_bucket(checkpoint_client, destinations.checkpoint_bucket)
        return destinations

    def begin_checkpoint_bucket_probe(
        self, bucket: CheckpointBucket, *, operation_id: str | None = None
    ) -> CheckpointBucketProbeOperation:
        CheckpointBucketHelperClient._bucket(bucket)
        identifier = operation_id or uuid.uuid4().hex
        if not isinstance(identifier, str) or not _OPERATION_RE.fullmatch(identifier):
            raise HubProbeError("probe operation ID must be a unique 32-character lowercase hex value")
        existing = self._checkpoint_operations.get(identifier)
        if existing is not None:
            if existing.operation.bucket != bucket:
                raise HubProbeError("probe operation ID is already bound to a different exact target")
            return existing.operation
        prefix = f"b1k-bootstrap-{identifier}"
        operation = CheckpointBucketProbeOperation(
            identifier,
            bucket,
            prefix,
            f"{prefix}/probe.json",
            f"/workspace/checkpoints/.b1k-release-probes/{identifier}.upload.json",
            f"/workspace/checkpoints/.b1k-release-probes/{identifier}.readback.json",
        )
        self._checkpoint_operations[identifier] = _CheckpointBucketProbeState(operation)
        return operation

    def bootstrap_checkpoint_bucket_probe(
        self,
        checkpoint_client: CheckpointBucketHelperClient,
        bucket: CheckpointBucket,
        *,
        operation_id: str | None = None,
        files: CheckpointProbeFiles | None = None,
    ) -> CheckpointBucketProbeReceipt:
        """Run the bounded reversible Storage Bucket proof through the pinned helper.

        The local staging paths are deliberately below ``/workspace/checkpoints`` so
        the helper's own path and symlink checks remain in force on the rent host.
        """

        operation = self.begin_checkpoint_bucket_probe(bucket, operation_id=operation_id)
        state = self._checkpoint_operations[operation.operation_id]
        local_files = files or WorkspaceCheckpointProbeFiles()
        self._verify_private_checkpoint_bucket(checkpoint_client, bucket)
        if state.receipt is not None:
            return state.receipt
        if state.deleted:
            if not state.readback_verified:
                raise HubProbeError("checkpoint probe operation lacks byte-for-byte readback proof")
            self._require_checkpoint_absence(checkpoint_client, operation)
            self._cleanup_checkpoint_local_files(local_files, operation)
            state.receipt = CheckpointBucketProbeReceipt(bucket.bucket_id, operation.prefix, operation.key)
            return state.receipt

        present = self._checkpoint_key_present(checkpoint_client, operation)
        if not state.uploaded:
            if present:
                state.uploaded = True
            else:
                self._checkpoint_call("checkpoint probe local staging", local_files.write_bytes, operation.upload_path, _PROBE_BYTES)
                try:
                    self._checkpoint_call("checkpoint bucket exact upload", checkpoint_client.upload, bucket, operation.upload_path, operation.key)
                except HubProbeError as upload_failure:
                    if not self._checkpoint_key_present(checkpoint_client, operation):
                        raise upload_failure
                if not self._checkpoint_key_present(checkpoint_client, operation):
                    raise HubProbeError("checkpoint bucket exact upload is not observable")
                state.uploaded = True
        elif not present:
            raise HubProbeError("checkpoint probe object unexpectedly disappeared")

        primary_failure: HubProbeError | None = None
        try:
            self._checkpoint_call("checkpoint probe local readback cleanup", local_files.remove, operation.download_path)
            self._checkpoint_call("checkpoint bucket download", checkpoint_client.download, bucket, operation.key, operation.download_path)
            actual = self._checkpoint_call("checkpoint probe local readback", local_files.read_bytes, operation.download_path)
            if not isinstance(actual, bytes) or actual != _PROBE_BYTES:
                raise HubProbeError("checkpoint bucket byte-for-byte readback did not match the exact uploaded bytes")
            state.readback_verified = True
        except HubProbeError as error:
            primary_failure = error

        try:
            self._delete_checkpoint_with_reconciliation(checkpoint_client, operation)
            state.deleted = True
        except HubProbeError as cleanup_failure:
            if primary_failure is not None:
                raise HubProbeError("checkpoint bucket readback failed; exact probe cleanup failed") from cleanup_failure
            raise
        self._cleanup_checkpoint_local_files(local_files, operation)
        if primary_failure is not None:
            raise primary_failure
        state.receipt = CheckpointBucketProbeReceipt(bucket.bucket_id, operation.prefix, operation.key)
        return state.receipt

    def begin_probe_operation(
        self,
        role: str,
        repository: HubRepository,
        *,
        operation_id: str | None = None,
        prefix: str | None = None,
    ) -> HubProbeOperation:
        self._validate_role_repository(role, repository)
        identifier = operation_id or uuid.uuid4().hex
        if not isinstance(identifier, str) or not _OPERATION_RE.fullmatch(identifier):
            raise HubProbeError("probe operation ID must be a unique 32-character lowercase hex value")
        existing = self._operations.get(identifier)
        if existing is not None:
            if existing.operation.role != role or existing.operation.repository != repository or (prefix is not None and existing.operation.prefix != prefix):
                raise HubProbeError("probe operation ID is already bound to a different exact target")
            return existing.operation
        resolved_prefix = f"b1k-bootstrap-{identifier}" if prefix is None else prefix
        if not isinstance(resolved_prefix, str) or not re.fullmatch(r"b1k-bootstrap-[a-z0-9-]{1,120}", resolved_prefix):
            raise HubProbeError("probe prefix is invalid")
        operation = HubProbeOperation(identifier, role, repository, resolved_prefix, f"{resolved_prefix}/probe.json")
        self._operations[identifier] = _ProbeState(operation)
        return operation

    def bootstrap_probe(
        self,
        role: str,
        repository: HubRepository,
        *,
        operation_id: str | None = None,
        prefix: str | None = None,
    ) -> HubProbeReceipt:
        operation = self.begin_probe_operation(role, repository, operation_id=operation_id, prefix=prefix)
        state = self._operations[operation.operation_id]
        if state.receipt is not None:
            return state.receipt
        token = self._credentials.resolve()
        self._verify_private(role, repository, token)
        if state.delete_commit is not None:
            if not state.readback_verified:
                raise HubProbeError("probe operation lacks immutable readback proof")
            self._verify_absence(repository, operation.key, state.delete_commit, token)
            receipt = HubProbeReceipt(role, repository.repo_id, repository.repo_type, operation.prefix, operation.key, state.upload_commit or "", state.delete_commit)
            state.receipt = receipt
            return receipt
        if state.upload_commit is None:
            try:
                state.upload_commit = self._upload(repository, operation.key, _PROBE_BYTES, token)
            except HubProbeError as upload_failure:
                state.upload_commit = self._resolve_exact_head(repository, operation.key, _PROBE_BYTES, token)
                if state.upload_commit is None:
                    raise upload_failure
        primary_failure: HubProbeError | None = None
        try:
            self._readback_immutable(repository, operation.key, state.upload_commit, _PROBE_BYTES, token)
            state.readback_verified = True
        except HubProbeError as error:
            primary_failure = error
        try:
            state.delete_commit = self._delete_with_reconciliation(repository, operation.key, _PROBE_BYTES, state.upload_commit, token)
            self._verify_absence(repository, operation.key, state.delete_commit, token)
        except HubProbeError as cleanup_failure:
            if primary_failure is not None:
                raise HubProbeError("immutable readback failed; exact probe cleanup failed") from cleanup_failure
            raise
        if primary_failure is not None:
            raise primary_failure
        receipt = HubProbeReceipt(role, repository.repo_id, repository.repo_type, operation.prefix, operation.key, state.upload_commit, state.delete_commit)
        state.receipt = receipt
        return receipt

    def verify_remote_probe(
        self,
        role: str,
        repository: HubRepository,
        *,
        prefix: str,
        upload_commit: str,
    ) -> HubProbeReceipt:
        """Read, delete, and prove absence of an image-created exact probe.

        The rented image performs the upload with its inherited token file; the
        controller only performs independent immutable readback and cleanup.
        """
        self._validate_role_repository(role, repository)
        self._validate_remote_probe_prefix(role, prefix)
        self._validate_commit(upload_commit)
        key = f"{prefix}/probe.json"
        token = self._credentials.resolve()
        self._verify_private(role, repository, token)
        primary_failure: HubProbeError | None = None
        try:
            self._readback_immutable(repository, key, upload_commit, _PROBE_BYTES, token)
        except HubProbeError as error:
            # A bad immutable readback is still a remote mutation made by the
            # rented image.  Reconcile only this exact key before surfacing it.
            primary_failure = error
        try:
            delete_commit = self._delete_with_reconciliation(repository, key, _PROBE_BYTES, upload_commit, token)
            self._verify_absence(repository, key, delete_commit, token)
        except HubProbeError as cleanup_failure:
            if primary_failure is not None:
                raise HubProbeError("immutable readback failed; exact probe cleanup failed") from cleanup_failure
            raise
        if primary_failure is not None:
            raise primary_failure
        return HubProbeReceipt(role, repository.repo_id, repository.repo_type, prefix, key, upload_commit, delete_commit)

    def reconcile_remote_probe(self, role: str, repository: HubRepository, *, prefix: str) -> None:
        """Reconcile one deterministic image-created probe with lost upload evidence.

        This never writes a probe.  It only recovers an exact byte-matching key
        for deletion, or proves that the exact key is absent at a current,
        immutable repository revision.
        """
        self._validate_role_repository(role, repository)
        self._validate_remote_probe_prefix(role, prefix)
        key = f"{prefix}/probe.json"
        token = self._credentials.resolve()
        self._verify_private(role, repository, token)
        upload_commit = self._resolve_exact_head(repository, key, _PROBE_BYTES, token)
        if upload_commit is not None:
            self.verify_remote_probe(role, repository, prefix=prefix, upload_commit=upload_commit)
            return
        revision = self._call("remote probe current revision", token, self._hub.current_revision, repository.repo_id, repository.repo_type, token)
        if revision is None:
            raise HubProbeError("remote probe current revision is unavailable")
        self._validate_commit(revision)
        self._verify_absence(repository, key, revision, token)

    def readback_immutable(self, repository: HubRepository, key: str, commit: str, expected: bytes) -> None:
        self._validate_repository(repository)
        token = self._credentials.resolve()
        self._verify_private("readback target", repository, token)
        self._readback_immutable(repository, key, commit, expected, token)

    def verify_absence(self, repository: HubRepository, key: str, revision: str) -> HubAbsenceProof:
        self._validate_repository(repository)
        self._validate_key(key)
        self._validate_commit(revision)
        return self._verify_absence(repository, key, revision, self._credentials.resolve())

    def _delete_with_reconciliation(self, repository: HubRepository, key: str, content: bytes, parent_commit: str, token: str) -> str:
        try:
            return self._delete_exact(repository, key, parent_commit, token)
        except HubProbeError as first_failure:
            current_revision = self._call("probe deletion reconciliation", token, self._hub.current_revision, repository.repo_id, repository.repo_type, token)
            if current_revision is not None:
                self._validate_commit(current_revision)
                try:
                    self._verify_absence(repository, key, current_revision, token)
                    return current_revision
                except HubProbeError:
                    pass
            current_head = self._resolve_exact_head(repository, key, content, token)
            if current_head is None or current_head == parent_commit:
                raise first_failure
            return self._delete_exact(repository, key, current_head, token)

    @staticmethod
    def _checkpoint_call(operation: str, callback: Callable[..., Any], *args: object) -> Any:
        try:
            return callback(*args)
        except Exception:
            raise HubProbeError(f"{operation} failed") from None

    @staticmethod
    def _verify_private_checkpoint_bucket(checkpoint_client: CheckpointBucketHelperClient, bucket: CheckpointBucket) -> None:
        info = HuggingFaceReleaseVerifier._checkpoint_call("checkpoint bucket privacy lookup", checkpoint_client.info, bucket)
        if not isinstance(info, Mapping) or dict(info) != {"private": True}:
            raise HubProbeError("checkpoint bucket must be explicitly private")

    @staticmethod
    def _checkpoint_key_present(checkpoint_client: CheckpointBucketHelperClient, operation: CheckpointBucketProbeOperation) -> bool:
        listed = HuggingFaceReleaseVerifier._checkpoint_call("checkpoint bucket exact listing", checkpoint_client.list, operation.bucket, operation.prefix)
        if not isinstance(listed, tuple) or any(not isinstance(path, str) for path in listed) or len(set(listed)) != len(listed):
            raise HubProbeError("checkpoint bucket exact listing is ambiguous")
        if listed == ():
            return False
        if listed == (operation.key,):
            return True
        raise HubProbeError("checkpoint bucket exact listing is ambiguous")

    @classmethod
    def _require_checkpoint_absence(cls, checkpoint_client: CheckpointBucketHelperClient, operation: CheckpointBucketProbeOperation) -> None:
        if cls._checkpoint_key_present(checkpoint_client, operation):
            raise HubProbeError("checkpoint bucket exact deletion is not observable")

    @classmethod
    def _delete_checkpoint_with_reconciliation(cls, checkpoint_client: CheckpointBucketHelperClient, operation: CheckpointBucketProbeOperation) -> None:
        try:
            cls._checkpoint_call("checkpoint bucket exact deletion", checkpoint_client.delete, operation.bucket, operation.key)
        except HubProbeError as delete_failure:
            if not cls._checkpoint_key_present(checkpoint_client, operation):
                return
            try:
                cls._checkpoint_call("checkpoint bucket exact deletion", checkpoint_client.delete, operation.bucket, operation.key)
            except HubProbeError:
                raise delete_failure
            cls._require_checkpoint_absence(checkpoint_client, operation)
            return
        if not cls._checkpoint_key_present(checkpoint_client, operation):
            return
        cls._checkpoint_call("checkpoint bucket exact deletion", checkpoint_client.delete, operation.bucket, operation.key)
        cls._require_checkpoint_absence(checkpoint_client, operation)

    @staticmethod
    def _cleanup_checkpoint_local_files(files: CheckpointProbeFiles, operation: CheckpointBucketProbeOperation) -> None:
        for path in (operation.upload_path, operation.download_path):
            HuggingFaceReleaseVerifier._checkpoint_call("checkpoint probe local cleanup", files.remove, path)

    def _resolve_exact_head(self, repository: HubRepository, key: str, content: bytes, token: str) -> str | None:
        resolved = self._call("exact probe reconciliation", token, self._hub.resolve_exact_file_head, repository.repo_id, repository.repo_type, key, content, token)
        if resolved is None:
            return None
        self._validate_commit(resolved)
        return resolved

    def _verify_absence(self, repository: HubRepository, key: str, revision: str, token: str) -> HubAbsenceProof:
        proof = self._call("probe absence verification", token, self._hub.absence_proof, repository.repo_id, repository.repo_type, key, revision, token)
        if not isinstance(proof, HubAbsenceProof) or (proof.repo_id, proof.repo_type, proof.key, proof.revision) != (repository.repo_id, repository.repo_type, key, revision) or not (proof.repository_private and proof.revision_exists and proof.key_absent):
            raise HubProbeError("probe absence verification is ambiguous")
        return proof

    def _readback_immutable(self, repository: HubRepository, key: str, commit: str, expected: bytes, token: str) -> None:
        self._validate_repository(repository)
        self._validate_key(key)
        self._validate_commit(commit)
        if not isinstance(expected, bytes):
            raise HubProbeError("immutable readback expectation must be bytes")
        actual = self._call("immutable readback", token, self._hub.read_file, repository.repo_id, repository.repo_type, key, commit, token)
        if not isinstance(actual, bytes) or actual != expected:
            raise HubProbeError("immutable readback did not match the exact uploaded bytes")

    def _verify_private(self, role: str, repository: HubRepository, token: str) -> None:
        info = self._call("repository privacy lookup", token, self._hub.repo_info, repository.repo_id, repository.repo_type, token)
        if not isinstance(info, Mapping) or info.get("private") is not True:
            raise HubProbeError(f"{role} Hugging Face repository must be explicitly private")

    def _upload(self, repository: HubRepository, key: str, content: bytes, token: str) -> str:
        commit = self._call("probe upload", token, self._hub.upload_bytes, repository.repo_id, repository.repo_type, key, content, token)
        self._validate_commit(commit)
        return commit

    def _delete_exact(self, repository: HubRepository, key: str, commit: str, token: str) -> str:
        delete_commit = self._call("exact probe deletion", token, self._hub.delete_file, repository.repo_id, repository.repo_type, key, commit, token)
        self._validate_commit(delete_commit)
        return delete_commit

    @staticmethod
    def _validate_repository_set(repositories: Mapping[str, HubRepository]) -> None:
        if not isinstance(repositories, Mapping) or set(repositories) != set(_REQUIRED_REPOSITORIES):
            raise HubProbeError("exactly model and dataset private repositories are required")
        for role, repository in repositories.items():
            HuggingFaceReleaseVerifier._validate_role_repository(role, repository)

    @staticmethod
    def _validate_role_repository(role: str, repository: HubRepository) -> None:
        expected = _REQUIRED_REPOSITORIES.get(role)
        if expected is None:
            raise HubProbeError("unrecognized Hugging Face repository role")
        HuggingFaceReleaseVerifier._validate_repository(repository)
        if (repository.repo_id, repository.repo_type) != expected:
            raise HubProbeError(f"{role} repository must match its exact pinned Hugging Face target")

    @staticmethod
    def _validate_remote_probe_prefix(role: str, prefix: object) -> None:
        suffixes = {
            "model": "smoke-model",
            "dataset": "(?:success|failure)-fixture",
        }
        suffix = suffixes[role]
        if not isinstance(prefix, str) or not re.fullmatch(rf"b1k-bootstrap-[a-f0-9]{{32}}-{suffix}", prefix):
            raise HubProbeError("remote probe prefix is not an exact role-specific runtime target")

    @staticmethod
    def _validate_repository(repository: HubRepository) -> None:
        if not isinstance(repository, HubRepository) or (repository.repo_id, repository.repo_type) not in set(_REQUIRED_REPOSITORIES.values()):
            raise HubProbeError("repository must be one of the exact pinned B1K Hugging Face targets")

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str) or not key or key.startswith("/") or ".." in key.split("/"):
            raise HubProbeError("probe key must be an exact relative path")

    @staticmethod
    def _validate_commit(commit: object) -> None:
        if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
            raise HubProbeError("an immutable Hugging Face commit is required")

    @staticmethod
    def _call(operation: str, token: str, callback: Callable[..., Any], *args: object) -> Any:
        try:
            return callback(*args)
        except Exception:
            raise HubProbeError(f"{operation} failed") from None
