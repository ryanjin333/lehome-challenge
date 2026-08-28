#!/usr/bin/env python3
"""Publish one simple-curriculum collection as an immutable public bundle.

This program deliberately owns no collection or provider lifecycle work.  It
receives already-authenticated local evidence from the one-VM controller,
uploads a fixed tree under a run-specific prefix, and only returns a receipt
after both authenticated and anonymous fresh downloads match every byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Mapping, Protocol, Sequence
from types import SimpleNamespace
from uuid import uuid4

from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout as RequestsTimeout


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "source" / "lehome", REPO_ROOT / "trainer" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lehome_train.redaction import ArtifactRejected, generate_upload_allowlist


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(r"^fresh-run-[a-z0-9-]{1,112}$")
_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")
_PUBLICATION_ROOT = "collection-rounds"
_SENSITIVE_PATH = re.compile(r"(?:token|secret|credential|password|api[_-]?key)", re.I)
_SENSITIVE_CONTENT = re.compile(
    rb"(?i)(?:bearer\s+[a-z0-9._-]{12,}|(?:token|secret|credential|password|(?:hf|sk|api|access|auth)[_-]?(?:token|key))\s*['\"]?\s*[:=]\s*['\"]?[a-z0-9._-]{8,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


class CollectionPublicationError(RuntimeError):
    """The final collection artifact is not durably public and verifiable."""


class HubTransientError(RuntimeError):
    """A bounded-retry transport interruption."""


class HubRateLimitError(HubTransientError):
    """A bounded-retry public Hub rate limit."""


@dataclass(frozen=True, slots=True)
class PublicationEntry:
    relative_path: str
    sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class CollectionPublicationBundle:
    """One immutable, exact collection tree prepared by the controller."""

    root: Path
    run_id: str
    repository: str
    revision: str
    files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CollectionPublicationResult:
    repository: str
    remote_prefix: str
    immutable_revision: str
    entries: tuple[PublicationEntry, ...]
    readback_verified: bool
    public_readback_verified: bool
    bundle_sha256: str


@dataclass(frozen=True, slots=True)
class _RemoteEntry:
    relative_path: str
    entry_type: str


class PublicCollectionTransport(Protocol):
    """Small deterministic seam for public collection upload/readback."""

    def resolve_approved_ref(self, *, repository: str, ref: str, token: str) -> str: ...

    def list_tree(
        self, *, repository: str, revision: str, token: str, remote_prefix: str | None = None,
    ) -> Sequence[object]: ...

    def upload_files(
        self, *, repository: str, revision: str, source: Path, entries: Sequence[PublicationEntry],
        token: str, remote_prefix: str | None = None, parent_commit: str | None = None,
    ) -> str: ...

    def download_files(
        self, *, repository: str, revision: str, destination: Path, relative_paths: Sequence[str],
        token: str | None, remote_prefix: str | None = None,
    ) -> str: ...


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise CollectionPublicationError("publication file is missing or unsafe")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _lexical_absolute(path: Path) -> Path:
    """Make an absolute pathname without resolving any symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path) -> Path:
    """Reject every existing lexical component rather than resolving it.

    ``Path.resolve`` is deliberately unsuitable at this boundary: it turns a
    symlinked publication root into an apparently ordinary directory.  The
    caller-provided raw tree is instead checked component-by-component before
    any walk, copy, hash, or content scan touches it.
    """

    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise CollectionPublicationError("publication source path is unavailable") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise CollectionPublicationError("publication source path contains a symlink")
    return absolute


def _safe_bundle_root(path: Path) -> Path:
    root = _reject_symlink_components(Path(path))
    try:
        metadata = os.lstat(root)
    except OSError as error:
        raise CollectionPublicationError("publication root is missing or unsafe") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise CollectionPublicationError("publication root is missing or unsafe")
    return root


def _open_bundle_entry(root: Path, relative: str) -> tuple[int, os.stat_result]:
    """Open one allowed entry through no-follow descriptors only.

    The directory descriptor pins the already-checked root.  Each component
    is opened with ``O_NOFOLLOW`` and then type-checked, preventing a path
    swap from escaping the bundle between validation and copy/hash/scan.
    """

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(root, flags)
    except OSError as error:
        raise CollectionPublicationError("publication root is missing or unsafe") from error
    current_fd = root_fd
    try:
        root_stat = os.fstat(current_fd)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise CollectionPublicationError("publication root is missing or unsafe")
        parts = PurePosixPath(relative).parts
        for index, part in enumerate(parts):
            final = index + 1 == len(parts)
            entry_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            if not final:
                entry_flags |= getattr(os, "O_DIRECTORY", 0)
            try:
                next_fd = os.open(part, entry_flags, dir_fd=current_fd)
            except OSError as error:
                raise CollectionPublicationError("publication source contains an unsafe path") from error
            os.close(current_fd)
            current_fd = next_fd
            metadata = os.fstat(current_fd)
            if (not final and not stat.S_ISDIR(metadata.st_mode)) or (final and not stat.S_ISREG(metadata.st_mode)):
                raise CollectionPublicationError("publication source contains an unsafe path")
        return current_fd, os.fstat(current_fd)
    except BaseException:
        os.close(current_fd)
        raise


def _sha256_descriptor(descriptor: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise CollectionPublicationError("publication source is unreadable") from error
    return digest.hexdigest(), size


def _scan_publication_descriptor_and_content_fd(descriptor: int, relative: str) -> None:
    """Scan a descriptor already opened below the pinned bundle root."""

    if any(_SENSITIVE_PATH.search(part) for part in PurePosixPath(relative).parts):
        raise CollectionPublicationError("publication source includes a credential-like descriptor")
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        tail = b""
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
            while chunk := handle.read(1 << 20):
                if _SENSITIVE_CONTENT.search(tail + chunk):
                    raise CollectionPublicationError("publication source includes credential-like content")
                tail = (tail + chunk)[-256:]
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as error:
        raise CollectionPublicationError("publication source is unreadable") from error


def _scan_publication_descriptor_and_content(path: Path) -> None:
    """Reject credential-bearing descriptors/content before public staging.

    Task-6 recorder/finalizer artifacts intentionally contain no credentials,
    but the collection root also hosts controller caches and debugging output.
    Check both the descriptor (path/name) and the bytes for *every* file that
    can enter the public bundle.  This is kept dependency-free because it is
    also the last local guard before anonymous Hub readback.
    """

    # Use the repository-wide, descriptor-safe uploader policy first.  It
    # opens the file beneath its parent with O_NOFOLLOW, checks canonical
    # relative paths and provider-token content, and re-stats the descriptor
    # after reading.  The small local policy below remains intentionally more
    # conservative for this anonymous public collection: controller debug
    # blobs that spell out a generic token/secret also never leave the host.
    try:
        generate_upload_allowlist(path.parent, (path.name,))
    except ArtifactRejected as error:
        raise CollectionPublicationError("publication source violates the canonical upload policy") from error
    if path.is_symlink() or not path.is_file() or _SENSITIVE_PATH.search(path.name):
        raise CollectionPublicationError("publication source includes a credential-like descriptor")
    try:
        tail = b""
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                if _SENSITIVE_CONTENT.search(tail + chunk):
                    raise CollectionPublicationError("publication source includes credential-like content")
                tail = (tail + chunk)[-256:]
    except OSError as error:
        raise CollectionPublicationError("publication source is unreadable") from error


def _entry_digest(entries: Sequence[PublicationEntry]) -> str:
    return hashlib.sha256(_canonical([
        {"relative_path": item.relative_path, "sha256": item.sha256, "byte_size": item.byte_size}
        for item in entries
    ])).hexdigest()


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not _SAFE_RELATIVE.fullmatch(value):
        raise CollectionPublicationError("publication path is not canonical")
    parts = PurePosixPath(value).parts
    if not parts or any(part in {"", ".", ".."} or part.startswith(".") for part in parts):
        raise CollectionPublicationError("publication path is not canonical")
    return value


def _collect_entries(bundle: CollectionPublicationBundle) -> tuple[PublicationEntry, ...]:
    root = _safe_bundle_root(Path(bundle.root))
    if not _RUN_ID.fullmatch(bundle.run_id):
        raise CollectionPublicationError("publication requires a fresh run ID")
    if not isinstance(bundle.repository, str) or "/" not in bundle.repository or not bundle.repository.strip():
        raise CollectionPublicationError("public repository is invalid")
    if not isinstance(bundle.revision, str) or not bundle.revision or _COMMIT.fullmatch(bundle.revision):
        raise CollectionPublicationError("publication requires a mutable approved ref")
    if not bundle.files or len(set(bundle.files)) != len(bundle.files):
        raise CollectionPublicationError("publication file allowlist is empty or duplicated")
    entries: list[PublicationEntry] = []
    for raw in bundle.files:
        relative = _safe_relative(raw)
        if relative.split("/", 1)[0] not in {"manifests", "fresh", "replay", "reports", "seals"}:
            raise CollectionPublicationError("publication path is outside the canonical collection layout")
        descriptor, before = _open_bundle_entry(root, relative)
        try:
            _scan_publication_descriptor_and_content_fd(descriptor, relative)
            sha256, byte_size = _sha256_descriptor(descriptor)
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
                raise CollectionPublicationError("publication source changed while being inspected")
        finally:
            os.close(descriptor)
        entries.append(PublicationEntry(relative, sha256, byte_size))
    return tuple(sorted(entries, key=lambda item: item.relative_path))


@contextmanager
def _stage_descriptor_safe_bundle(
    bundle: CollectionPublicationBundle, entries: tuple[PublicationEntry, ...],
):
    """Copy verified raw bytes to a private staging tree before transport.

    A transport such as ``upload_folder`` walks a pathname later.  Publishing
    from a descriptor-copied staging tree closes the practical local TOCTOU
    gap between raw-tree validation and that later library traversal.
    """

    root = _safe_bundle_root(Path(bundle.root))
    staging = Path(tempfile.mkdtemp(prefix="lehome-curriculum-raw-stage-", dir=root.parent))
    try:
        for entry in entries:
            descriptor, before = _open_bundle_entry(root, entry.relative_path)
            target = staging / entry.relative_path
            try:
                _scan_publication_descriptor_and_content_fd(descriptor, entry.relative_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with os.fdopen(os.dup(descriptor), "rb", closefd=True) as source, target.open("xb") as destination:
                    while chunk := source.read(1 << 20):
                        digest.update(chunk)
                        size += len(chunk)
                        destination.write(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())
                after = os.fstat(descriptor)
                if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
                    raise CollectionPublicationError("publication source changed while being staged")
                if (digest.hexdigest(), size) != (entry.sha256, entry.byte_size):
                    raise CollectionPublicationError("publication source changed after validation")
                _scan_publication_descriptor_and_content(target)
            finally:
                os.close(descriptor)
        yield CollectionPublicationBundle(
            root=staging, run_id=bundle.run_id, repository=bundle.repository,
            revision=bundle.revision, files=tuple(entry.relative_path for entry in entries),
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _is_transient(error: BaseException) -> bool:
    """Only the canonical Hub retry categories are retryable."""

    return isinstance(error, (RequestsConnectionError, RequestsTimeout)) or type(error).__name__ in {
        "HubTransientError", "HubRateLimitError",
    }


def _retry(operation, *, label: str, max_attempts: int):
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise CollectionPublicationError("publication retry limit is invalid")
    for attempt in range(max_attempts):
        try:
            return operation()
        except Exception as error:  # noqa: BLE001 - transport boundary is intentionally narrow below
            if not _is_transient(error) or attempt + 1 == max_attempts:
                raise CollectionPublicationError(f"{label} failed") from None
    raise AssertionError("retry loop is exhausted")


def _tree_files(tree: Sequence[object], *, prefix: str) -> set[str]:
    observed: set[str] = set()
    base = prefix + "/"
    for item in tree:
        path = getattr(item, "relative_path", None)
        entry_type = getattr(item, "entry_type", None)
        if not isinstance(path, str) or not isinstance(entry_type, str):
            raise CollectionPublicationError("remote tree is malformed")
        if not path.startswith(base):
            raise CollectionPublicationError("remote tree escaped collection prefix")
        if entry_type == "file":
            observed.add(path.removeprefix(base))
        elif entry_type not in {"directory"}:
            raise CollectionPublicationError("remote collection contains an unsafe entry")
    return observed


def _verify_download(
    *, transport: PublicCollectionTransport, bundle: CollectionPublicationBundle,
    revision: str, prefix: str, entries: tuple[PublicationEntry, ...], token: str | None,
) -> None:
    destination = Path(tempfile.mkdtemp(prefix="lehome-curriculum-readback-", dir=bundle.root.parent))
    try:
        observed = _retry(
            lambda: transport.download_files(
                repository=bundle.repository, revision=revision, destination=destination,
                relative_paths=tuple(item.relative_path for item in entries), token=token,
                remote_prefix=prefix,
            ),
            label="public readback" if token is None else "readback", max_attempts=3,
        )
        if observed != revision:
            raise CollectionPublicationError("readback did not bind the immutable revision")
        for entry in entries:
            sha256, byte_size = _sha256(destination / entry.relative_path)
            if sha256 != entry.sha256 or byte_size != entry.byte_size:
                raise CollectionPublicationError("readback bytes do not match the immutable bundle")
    finally:
        shutil.rmtree(destination, ignore_errors=True)


def publish_collection_bundle(
    bundle: CollectionPublicationBundle, *, token: str, transport: PublicCollectionTransport,
    max_attempts: int = 3,
) -> CollectionPublicationResult:
    """Upload one collection tree then prove authenticated and public readback.

    A pre-existing tree can only be resumed if it is exactly the same tree.
    That resume still performs both independent fresh readbacks.
    """

    if not isinstance(token, str) or not token or any(character.isspace() for character in token):
        raise CollectionPublicationError("Hub token is unavailable")
    entries = _collect_entries(bundle)
    with _stage_descriptor_safe_bundle(bundle, entries) as staged:
        return _publish_staged_collection_bundle(
            staged, entries=entries, token=token, transport=transport, max_attempts=max_attempts,
        )


def _publish_staged_collection_bundle(
    bundle: CollectionPublicationBundle, *, entries: tuple[PublicationEntry], token: str,
    transport: PublicCollectionTransport, max_attempts: int,
) -> CollectionPublicationResult:
    """Publish bytes copied from descriptor-verified staging only."""

    prefix = f"{_PUBLICATION_ROOT}/{bundle.run_id}"
    expected = {entry.relative_path for entry in entries}
    head = _retry(
        lambda: transport.resolve_approved_ref(repository=bundle.repository, ref=bundle.revision, token=token),
        label="publication ref resolution", max_attempts=max_attempts,
    )
    if not isinstance(head, str) or not _COMMIT.fullmatch(head):
        raise CollectionPublicationError("publication ref did not resolve to an immutable commit")
    existing = _tree_files(
        _retry(
            lambda: transport.list_tree(repository=bundle.repository, revision=head, token=token, remote_prefix=prefix),
            label="collection collision check", max_attempts=max_attempts,
        ),
        prefix=prefix,
    )
    if existing and existing != expected:
        raise CollectionPublicationError("immutable collection collision")
    if existing:
        revision = head
        try:
            # Matching names are insufficient: immutable resume is allowed
            # only when the already-published bytes are identical.
            _verify_download(
                transport=transport, bundle=bundle, revision=revision, prefix=prefix,
                entries=entries, token=token,
            )
        except CollectionPublicationError as error:
            raise CollectionPublicationError("immutable collection collision") from error
    else:
        try:
            revision = _retry(
                lambda: transport.upload_files(
                    repository=bundle.repository, revision=bundle.revision, source=bundle.root,
                    entries=entries, token=token, remote_prefix=prefix, parent_commit=head,
                ),
                label="collection upload", max_attempts=max_attempts,
            )
        except CollectionPublicationError as upload_error:
            # A lost response or a concurrent branch update is ambiguous.  Do
            # not retry a mutable upload blind: re-open the current immutable
            # tree and allow only the exact byte-identical prefix to resume.
            current_head = _retry(
                lambda: transport.resolve_approved_ref(
                    repository=bundle.repository, ref=bundle.revision, token=token,
                ),
                label="publication ref reconciliation", max_attempts=max_attempts,
            )
            if not isinstance(current_head, str) or _COMMIT.fullmatch(current_head) is None:
                raise CollectionPublicationError("publication ref reconciliation did not resolve an immutable commit") from None
            current = _tree_files(
                _retry(
                    lambda: transport.list_tree(
                        repository=bundle.repository, revision=current_head, token=token, remote_prefix=prefix,
                    ),
                    label="collection collision reconciliation", max_attempts=max_attempts,
                ),
                prefix=prefix,
            )
            if current != expected:
                if current:
                    raise CollectionPublicationError("immutable collection collision") from None
                raise upload_error
            try:
                _verify_download(
                    transport=transport, bundle=bundle, revision=current_head, prefix=prefix,
                    entries=entries, token=token,
                )
            except CollectionPublicationError as error:
                raise CollectionPublicationError("immutable collection collision") from error
            revision = current_head
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise CollectionPublicationError("collection upload did not return an immutable commit")
        observed = _tree_files(
            _retry(
                lambda: transport.list_tree(repository=bundle.repository, revision=revision, token=token, remote_prefix=prefix),
                label="collection tree verification", max_attempts=max_attempts,
            ),
            prefix=prefix,
        )
        if observed != expected:
            raise CollectionPublicationError("remote collection tree does not match the immutable bundle")
        _verify_download(
            transport=transport, bundle=bundle, revision=revision, prefix=prefix,
            entries=entries, token=token,
        )
    _verify_download(transport=transport, bundle=bundle, revision=revision, prefix=prefix, entries=entries, token=None)
    return CollectionPublicationResult(
        repository=bundle.repository, remote_prefix=prefix, immutable_revision=revision,
        entries=entries, readback_verified=True, public_readback_verified=True,
        bundle_sha256=_entry_digest(entries),
    )


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CollectionPublicationError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise CollectionPublicationError(f"{label} is malformed") from None
    if not isinstance(value, dict):
        raise CollectionPublicationError(f"{label} is malformed")
    return value


def _verified_gpu_stop(
    root: Path, *, terminal_outcome: str, rollout_instance_id: str,
) -> dict[str, object]:
    observation_path = root / "stage-receipts" / "gpu-stop-observation.json"
    observation = _json_object(observation_path, label="GPU stop observation")
    required = {
        "schema_version", "kind", "provider", "instance_id", "state", "verified",
        "observed_at_utc", "provider_response_sha256",
    }
    if (
        set(observation) != required
        or observation.get("schema_version") != 1
        or observation.get("kind") != "lehome_simple_curriculum_verified_gpu_stop_v1"
        or observation.get("provider") != "nebius_compute_api"
        or observation.get("instance_id") != rollout_instance_id
        or observation.get("state") != "STOPPED"
        or observation.get("verified") is not True
        or not isinstance(observation.get("observed_at_utc"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(observation.get("provider_response_sha256")))
    ):
        raise CollectionPublicationError("GPU stop is not authoritatively verified for the exact rollout VM")
    stop = _json_object(root / "stage-receipts" / "gpu-stop.json", label="GPU stop receipt")
    output = stop.get("output")
    if not isinstance(output, Mapping) or (
        output.get("terminal_outcome") != terminal_outcome
        or output.get("stop_status") != "succeeded"
        or output.get("rollout_instance_id") != rollout_instance_id
        or output.get("verified_stopped") is not True
        or output.get("stop_observation_sha256") != hashlib.sha256(observation_path.read_bytes()).hexdigest()
    ):
        raise CollectionPublicationError("GPU stop receipt does not bind the verified stopped observation")
    return observation


def _task6_controller_module():
    """Load the one canonical producer verifier without duplicating its rules."""

    name = "_lehome_simple_curriculum_task6_verifier"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    script = REPO_ROOT / "scripts" / "run_simple_curriculum_collection.py"
    if not script.is_file():
        raise CollectionPublicationError("Task 6 canonical evidence verifier is unavailable")
    scripts_directory = str(script.parent)
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise CollectionPublicationError("Task 6 canonical evidence verifier is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _authenticate_complete_task6_evidence(
    root: Path, *, run_id: str, round_id: str,
) -> tuple[int, int, int, int, dict[str, int]]:
    """Re-open the exact Task 6 evidence chain before sealing completion.

    Completion never trusts summary IDs.  The Task 6 validator replays the
    report/matrix joins, all 1,000 terminal artifact and receipt bindings, the
    physical partition ledgers, and the exact 400-row visual-only replay
    ledger.  This publisher intentionally consumes that same authority rather
    than carrying a weaker copy of its schema.
    """

    controller = _task6_controller_module()
    # These three attributes are the only collection configuration fields used
    # by the canonical fresh/replay evidence validators.  They are explicit so
    # a publisher cannot manufacture a new policy/runtime identity.
    config = SimpleNamespace(campaign_root=root, run_id=run_id, round_id=round_id)
    report_path = root / "reports" / "fresh-source-report.json"
    matrix_path = root / "reports" / "fresh-source-matrix.json"
    manifest_path = root / "reports" / "fresh-terminal-artifacts.json"
    try:
        authenticated = controller._validate_fresh_source_outputs(
            config, report=report_path, matrix=matrix_path,
        )
        controller._validate_fresh_terminal_artifact_manifest(
            config, authenticated=authenticated, manifest=manifest_path,
        )
        # The source aggregate is not sufficient alone: every physical
        # partition matrix/manifest/ledger must still be exact at publication.
        for partition, start, end, target, lease_budget in (
            ("calibration-head", 0, 100, 100, 150),
            ("calibration-tail", 100, 400, 300, 400),
            ("curriculum-a", 0, 300, 300, 400),
            ("curriculum-b", 300, 600, 300, 400),
        ):
            matrix = root / "partitions" / f"{partition}.json"
            manifest = root / "partitions" / f"{partition}.manifest.json"
            controller._validate_partition_manifest(
                manifest, matrix=matrix, partition_id=partition,
                inputs={
                    "partition_id": partition, "row_start": start, "row_end": end,
                    "target": target, "lease_budget": lease_budget,
                },
            )
            controller._validate_partition_ledger(
                root / "fresh" / partition / "ledger.sqlite3", matrix=matrix,
                max_attempts=lease_budget, target=target,
            )
        replay = controller._discover_success_replay(
            config, matrix=root / "replay" / "replay.json", ledger=root / "replay" / "ledger.sqlite3",
        )
        if replay.get("result") != "complete":
            raise ValueError("success replay did not meet the exact complete contract")
    except (OSError, ValueError, RuntimeError) as error:
        raise CollectionPublicationError("collection complete requires authoritative Task 6 evidence") from error
    fresh_successes = sum(
        1 for context in authenticated.values()
        if isinstance(context.get("trial"), Mapping) and context["trial"].get("accepted_success") is True
    )
    return 1000, fresh_successes, 400, 200, {
        "pant_long": 50, "pant_short": 50, "top_long": 50, "top_short": 50,
    }


def _complete_stage_chain_is_clean(root: Path) -> None:
    """A complete seal cannot coexist with a prior fidelity/cost/infra abort."""
    receipts = root / "stage-receipts"
    for path in _iter_regular_files(receipts):
        if path.name == "gpu-stop-observation.json":
            continue
        payload = _json_object(path, label="stage receipt")
        output = payload.get("output")
        if not isinstance(output, Mapping):
            continue
        if output.get("decision") in {"fidelity_stop", "infrastructure_stop", "insufficient_source_stop"}:
            raise CollectionPublicationError("collection complete conflicts with a terminal gate abort")
        if output.get("terminal_outcome") not in {None, "complete"}:
            raise CollectionPublicationError("collection complete conflicts with a terminal infrastructure abort")
        if output.get("stop_status") in {"failed", "pending"}:
            raise CollectionPublicationError("collection complete conflicts with an unverified GPU stop")


def _seal_kind(terminal_outcome: str) -> str:
    if terminal_outcome == "complete":
        return "collection_complete"
    if terminal_outcome in {"fidelity_stop", "infrastructure_stop", "infrastructure_stop_failure"}:
        return "fidelity_infrastructure_stop"
    if terminal_outcome in {"insufficient_source_stop", "replay_shortage"}:
        return "insufficient_fresh_source"
    raise CollectionPublicationError("terminal outcome has no honest final seal")


def build_final_seal(
    campaign_root: Path, *, run_id: str, round_id: str, terminal_outcome: str,
    rollout_instance_id: str,
) -> dict[str, object]:
    """Build an honest terminal seal; complete is deliberately hardest to claim."""

    root = _safe_bundle_root(Path(campaign_root))
    if not _RUN_ID.fullmatch(run_id):
        raise CollectionPublicationError("final seal collection identity is invalid")
    kind = _seal_kind(terminal_outcome)
    observation = _verified_gpu_stop(
        root, terminal_outcome=terminal_outcome, rollout_instance_id=rollout_instance_id,
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "kind": f"lehome_simple_curriculum_{kind}_seal_v1",
        "run_id": run_id,
        "round_id": round_id,
        "terminal_outcome": terminal_outcome,
        "rollout_instance_id": rollout_instance_id,
        "gpu_stop_verified": True,
        "gpu_stop_observed_at_utc": observation["observed_at_utc"],
    }
    if kind == "collection_complete":
        _complete_stage_chain_is_clean(root)
        (
            fresh_total, fresh_successes, replay_attempts, replay_successes, replay_categories,
        ) = _authenticate_complete_task6_evidence(root, run_id=run_id, round_id=round_id)
        body.update(
            fresh_valid_outcomes=fresh_total,
            fresh_official_successes=fresh_successes,
            replay_attempts=replay_attempts,
            replay_accepted_successes=replay_successes,
            replay_accepted_by_category=replay_categories,
            all_hub_readbacks_verified=True,
        )
    else:
        body.update(all_hub_readbacks_verified=False)
    body["seal_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = _canonical(payload)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise CollectionPublicationError("immutable local publication collision")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _iter_regular_files(root: Path) -> tuple[Path, ...]:
    root = _safe_bundle_root(root)
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        if any((current_path / name).is_symlink() for name in directories):
            raise CollectionPublicationError("publication source contains a symlinked directory")
        for name in names:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise CollectionPublicationError("publication source contains an unsafe file")
            files.append(path)
    return tuple(sorted(files))


def _copy_to_staging(*, source: Path, root: Path, staging: Path, remote: str) -> None:
    _safe_relative(remote)
    root = _safe_bundle_root(root)
    if not source.is_absolute() or not source.is_relative_to(root):
        raise CollectionPublicationError("publication source file is unsafe")
    relative = source.relative_to(root).as_posix()
    descriptor, before = _open_bundle_entry(root, relative)
    target = staging / remote
    try:
        _scan_publication_descriptor_and_content_fd(descriptor, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise CollectionPublicationError("duplicate staged collection path")
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as input_handle, target.open("xb") as output_handle:
            while chunk := input_handle.read(1 << 20):
                digest.update(chunk)
                size += len(chunk)
                output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
            raise CollectionPublicationError("publication source changed while being staged")
        if _sha256(target) != (digest.hexdigest(), size):
            raise CollectionPublicationError("staged collection bytes differ from the source")
        _scan_publication_descriptor_and_content(target)
    finally:
        os.close(descriptor)


def _existing_regular(root: Path, relative: str) -> Path | None:
    """Return one reviewed optional file, rejecting unsafe lookalikes."""

    root = _safe_bundle_root(root)
    relative = _safe_relative(relative)
    path = root / relative
    if not path.exists() and not path.is_symlink():
        return None
    descriptor, _metadata = _open_bundle_entry(root, relative)
    os.close(descriptor)
    return path


def _safe_nonpublic_runtime_directory(path: Path) -> set[Path]:
    """Recognize an explicitly private runtime cache without staging it."""

    if not path.exists() and not path.is_symlink():
        return set()
    if path.is_symlink() or not path.is_dir():
        raise CollectionPublicationError("nonpublic runtime state is unsafe")
    # Traverse solely for path safety.  These cache bytes are intentionally
    # neither required as evidence nor copied to public storage.
    return set(_iter_regular_files(path))


def _complete_nonpublic_paths(root: Path) -> set[Path]:
    """Approved controller/runtime state that deliberately stays private."""

    ignored: set[Path] = set()
    for relative in (
        "reports/final-publication.json", "reports/final-publication-readback.json",
        "replay/ledger.sqlite3-wal", "replay/ledger.sqlite3-shm",
    ):
        path = _existing_regular(root, relative)
        if path is not None:
            ignored.add(path.relative_to(root))
    for partition in ("calibration-head", "calibration-tail", "curriculum-a", "curriculum-b"):
        partition_root = root / "fresh" / partition
        for name in ("rollout-preemption.json", "ledger.sqlite3-wal", "ledger.sqlite3-shm"):
            path = _existing_regular(root, f"fresh/{partition}/{name}")
            if path is not None:
                ignored.add(path.relative_to(root))
        cache = partition_root / "hf-readback"
        ignored.update(path.relative_to(root) for path in _safe_nonpublic_runtime_directory(cache))
    # The appliance deliberately keeps a private HF/runtime cache at the
    # campaign root.  It is not input evidence and must never be a condition
    # of collection completion or public staging.
    ignored.update(path.relative_to(root) for path in _safe_nonpublic_runtime_directory(root / "hf-cache"))
    return ignored


def _complete_reviewed_paths(root: Path, *, seal_path: Path) -> tuple[Path, ...]:
    """Return the exact reviewed source files for a complete collection.

    A complete public bundle is deliberately not a recursive campaign-root
    upload.  Every source file must either be a fixed Task 6 receipt/input or
    be reachable from the already-revalidated fresh/replay artifact chains.
    """

    static = {
        "inputs/policy-identity.json", "inputs/seen-catalog.json",
        "matrices/calibration.json", "matrices/calibration.receipt.json",
        "matrices/curriculum.json", "matrices/curriculum.receipt.json",
        "reports/calibration-head.json", "reports/first-100-gate.json",
        "reports/calibration-tail.json", "reports/calibration.json",
        "reports/fresh-source-report.json", "reports/fresh-source-matrix.json",
        "reports/fresh-terminal-artifacts.json", "replay/replay.json",
        "replay/replay.json.sha256", "replay/ledger.sqlite3",
        "replay/success-replay-readback-seal.json",
        "stage-receipts/budget-state.json", "stage-receipts/gpu-stop-state.json",
        "stage-receipts/gpu-stop-observation.json",
    }
    for partition in ("calibration-head", "calibration-tail", "curriculum-a", "curriculum-b"):
        static.update({
            f"partitions/{partition}.json", f"partitions/{partition}.manifest.json",
            f"fresh/{partition}/ledger.sqlite3", f"reports/partitions/{partition}.json",
        })
    for stage in (
        "calibration-matrix", "calibration-head", "first-100-gate", "calibration-tail",
        "calibration-report", "curriculum-matrix", "curriculum-a", "curriculum-b",
        "fresh-report", "replay-matrix", "success-replay", "gpu-stop",
    ):
        static.add(f"stage-receipts/{stage}.json")
    manifest = _json_object(root / "reports" / "fresh-terminal-artifacts.json", label="fresh terminal artifact manifest")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise CollectionPublicationError("collection complete source manifest is malformed")
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("attempt_id"), str) or not isinstance(entry.get("finalized_artifact_root"), str):
            raise CollectionPublicationError("collection complete source manifest is malformed")
        artifact = Path(entry["finalized_artifact_root"])
        try:
            relative_artifact = artifact.relative_to(root)
        except ValueError as error:
            raise CollectionPublicationError("collection complete artifact escapes root") from error
        static.update(path.relative_to(root).as_posix() for path in _iter_regular_files(artifact))
        receipt = artifact.parent.parent / "hf-sync-receipts" / f"{entry['attempt_id']}.sync.json"
        static.add(receipt.relative_to(root).as_posix())
    replay_seal = _json_object(root / "replay" / "success-replay-readback-seal.json", label="success replay readback seal")
    accepted = replay_seal.get("accepted_attempt_ids")
    if not isinstance(accepted, list) or len(accepted) != 200:
        raise CollectionPublicationError("collection complete replay source manifest is malformed")
    for attempt in accepted:
        if not isinstance(attempt, str):
            raise CollectionPublicationError("collection complete replay source manifest is malformed")
        artifact = root / "replay" / "accepted" / attempt
        static.update(path.relative_to(root).as_posix() for path in _iter_regular_files(artifact))
        static.add(f"replay/hf-sync-receipts/{attempt}.sync.json")
    static.add(seal_path.relative_to(root).as_posix())
    allowed = {Path(relative) for relative in static}
    # Controller receipts/inputs are useful when present but are not producer
    # completion proof.  Do not reject a real Task-6 output merely because a
    # cache was absent or a post-crash stage receipt was never written.
    allowed = {path for path in allowed if _existing_regular(root, path.as_posix()) is not None}
    if seal_path.is_symlink() or not seal_path.is_file() or not seal_path.is_relative_to(root / "seals"):
        raise CollectionPublicationError("final seal is outside the canonical seals directory")
    allowed.add(seal_path.relative_to(root))
    source_areas = ("inputs", "matrices", "partitions", "stage-receipts", "reports", "seals", "fresh", "replay")
    observed: set[Path] = set()
    for area in source_areas:
        candidate = root / area
        if candidate.exists() or candidate.is_symlink():
            observed.update(path.relative_to(root) for path in _iter_regular_files(candidate))
    # Explicitly exclude only known controller/runtime state.  Any other
    # byte in a reviewed source area is an unknown artifact and makes a
    # complete seal fail closed rather than silently publishing it.
    observed -= _complete_nonpublic_paths(root)
    if observed != allowed:
        raise CollectionPublicationError("collection complete has missing or unreviewed source files")
    return tuple(root / path for path in sorted(allowed))


_FAILURE_EVIDENCE: Mapping[str, tuple[str, ...]] = {
    "lehome_simple_curriculum_fidelity_infrastructure_stop_seal_v1": (
        "stage-receipts/budget-state.json", "stage-receipts/gpu-stop-state.json",
        "stage-receipts/gpu-stop.json", "stage-receipts/gpu-stop-observation.json",
        "stage-receipts/calibration-matrix.json", "stage-receipts/calibration-head.json",
        "stage-receipts/first-100-gate.json", "reports/calibration-head.json",
        "reports/first-100-gate.json",
    ),
    "lehome_simple_curriculum_insufficient_fresh_source_seal_v1": (
        "stage-receipts/budget-state.json", "stage-receipts/gpu-stop-state.json",
        "stage-receipts/gpu-stop.json", "stage-receipts/gpu-stop-observation.json",
        "stage-receipts/fresh-report.json", "stage-receipts/replay-matrix.json",
        "stage-receipts/success-replay.json", "reports/fresh-source-report.json",
        "reports/fresh-source-matrix.json", "replay/replay.json",
        "replay/replay.json.sha256", "replay/success-replay-readback-seal.json",
    ),
}


def _failure_reviewed_paths(root: Path, *, seal_path: Path, kind: str) -> tuple[Path, ...]:
    """Stage only concise, fixed diagnostic evidence for non-complete seals."""

    try:
        candidates = _FAILURE_EVIDENCE[kind]
    except KeyError as error:
        raise CollectionPublicationError("final seal kind has no fixed evidence allowlist") from error
    if seal_path.is_symlink() or not seal_path.is_file() or not seal_path.is_relative_to(root / "seals"):
        raise CollectionPublicationError("final seal is outside the canonical seals directory")
    paths = [seal_path]
    for relative in candidates:
        source = _existing_regular(root, relative)
        if source is not None:
            paths.append(source)
    return tuple(sorted(set(paths)))


def _stage_collection_bundle(root: Path, *, seal_path: Path, run_id: str, repository: str, revision: str) -> CollectionPublicationBundle:
    """Map only reviewed campaign evidence into the five canonical sections."""

    staging = Path(tempfile.mkdtemp(prefix="lehome-curriculum-publication-", dir=root))
    try:
        seal = _json_object(seal_path, label="final seal")
        if seal.get("kind") == "lehome_simple_curriculum_collection_complete_seal_v1":
            for source in _complete_reviewed_paths(root, seal_path=seal_path):
                relative = source.relative_to(root).as_posix()
                if relative.startswith(("inputs/", "matrices/", "partitions/", "stage-receipts/")):
                    remote = f"manifests/{relative}"
                else:
                    remote = relative
                _copy_to_staging(source=source, root=root, staging=staging, remote=remote)
            files = tuple(path.relative_to(staging).as_posix() for path in _iter_regular_files(staging))
            return CollectionPublicationBundle(staging, run_id, repository, revision, files)
        for source in _failure_reviewed_paths(root, seal_path=seal_path, kind=str(seal.get("kind"))):
            relative = source.relative_to(root).as_posix()
            remote = f"manifests/{relative}" if relative.startswith(("inputs/", "matrices/", "partitions/", "stage-receipts/")) else relative
            _copy_to_staging(source=source, root=root, staging=staging, remote=remote)
        files = tuple(path.relative_to(staging).as_posix() for path in _iter_regular_files(staging))
        return CollectionPublicationBundle(staging, run_id, repository, revision, files)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


class HuggingFacePublicDatasetTransport:
    """Lazy public-dataset transport; anonymous readback is a separate path."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        if type(timeout_seconds) not in (int, float) or timeout_seconds <= 0:
            raise ValueError("Hub timeout is invalid")
        self.timeout_seconds = float(timeout_seconds)

    def _library(self):
        try:
            import huggingface_hub  # type: ignore[import-not-found]
        except ImportError:
            raise RuntimeError("huggingface_hub transport dependency is unavailable") from None
        return huggingface_hub

    @staticmethod
    def _revision(value: object) -> str:
        revision = getattr(value, "oid", None) or getattr(value, "sha", None)
        if not isinstance(revision, str) or _COMMIT.fullmatch(revision) is None:
            raise CollectionPublicationError("Hub response did not identify an immutable revision")
        return revision

    @staticmethod
    def _raise_transport(error: Exception) -> None:
        status = getattr(getattr(error, "response", None), "status_code", None) or getattr(error, "status_code", None)
        if status == 429:
            raise HubRateLimitError("public Hub rate limit") from None
        if isinstance(error, (ConnectionError, TimeoutError, RequestsConnectionError, RequestsTimeout)):
            raise HubTransientError("public Hub transport interruption") from None
        raise error

    @staticmethod
    def _is_missing_tree_prefix(error: Exception) -> bool:
        """Recognize only huggingface_hub's 404 for this requested folder.

        ``list_repo_tree(path_in_repo=...)`` in huggingface_hub 0.36 raises
        EntryNotFoundError for an absent folder instead of returning an empty
        iterator.  Authentication and access errors can reuse related error
        classes, so both the exact class name and the HTTP 404 are required.
        """

        status = getattr(getattr(error, "response", None), "status_code", None) or getattr(error, "status_code", None)
        return type(error).__name__ == "EntryNotFoundError" and status == 404

    def _api(self, token: str | bool):
        return self._library().HfApi(token=token)

    def resolve_approved_ref(self, *, repository: str, ref: str, token: str) -> str:
        try:
            return self._revision(self._api(token).repo_info(
                repo_id=repository, repo_type="dataset", revision=ref, token=token, timeout=self.timeout_seconds,
            ))
        except Exception as error:  # noqa: BLE001 - normalized at boundary
            self._raise_transport(error)
            raise AssertionError("unreachable")

    def list_tree(self, *, repository: str, revision: str, token: str, remote_prefix: str | None = None) -> tuple[_RemoteEntry, ...]:
        if _COMMIT.fullmatch(revision) is None or not remote_prefix:
            raise CollectionPublicationError("public tree request is not immutable and bounded")
        try:
            entries = self._api(token).list_repo_tree(
                repo_id=repository, repo_type="dataset", revision=revision, token=token,
                path_in_repo=remote_prefix, recursive=True, expand=False,
            )
            result: list[_RemoteEntry] = []
            for raw in entries:
                path = getattr(raw, "path", None)
                if not isinstance(path, str):
                    raise CollectionPublicationError("public Hub tree path is invalid")
                if type(raw).__name__ == "RepoFile":
                    entry_type = "file"
                elif type(raw).__name__ == "RepoFolder":
                    entry_type = "directory"
                else:
                    raise CollectionPublicationError("public Hub tree entry is invalid")
                result.append(_RemoteEntry(path, entry_type))
            return tuple(result)
        except CollectionPublicationError:
            raise
        except Exception as error:  # noqa: BLE001
            if self._is_missing_tree_prefix(error):
                return ()
            self._raise_transport(error)
            raise AssertionError("unreachable")

    def upload_files(self, *, repository: str, revision: str, source: Path, entries: Sequence[PublicationEntry], token: str, remote_prefix: str | None = None, parent_commit: str | None = None) -> str:
        if not remote_prefix or _COMMIT.fullmatch(parent_commit or "") is None:
            raise CollectionPublicationError("public upload prefix is missing")
        try:
            result = self._api(token).upload_folder(
                repo_id=repository, repo_type="dataset", revision=revision, folder_path=str(source),
                allow_patterns=[entry.relative_path for entry in entries], path_in_repo=remote_prefix, token=token,
                parent_commit=parent_commit,
            )
            return self._revision(result)
        except Exception as error:  # noqa: BLE001
            self._raise_transport(error)
            raise AssertionError("unreachable")

    def download_files(self, *, repository: str, revision: str, destination: Path, relative_paths: Sequence[str], token: str | None, remote_prefix: str | None = None) -> str:
        if _COMMIT.fullmatch(revision) is None or not remote_prefix:
            raise CollectionPublicationError("public download request is not immutable and bounded")
        library = self._library()
        hub_token: str | bool = token if token is not None else False
        try:
            for relative in relative_paths:
                remote = f"{remote_prefix}/{relative}"
                downloaded = Path(library.hf_hub_download(
                    repo_id=repository, repo_type="dataset", revision=revision, filename=remote,
                    token=hub_token, local_dir=destination, local_dir_use_symlinks=False,
                    etag_timeout=self.timeout_seconds,
                ))
                source = destination / remote
                target = destination / relative
                if downloaded != source or source.is_symlink() or not source.is_file():
                    raise CollectionPublicationError("public Hub returned an unsafe download path")
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
            shutil.rmtree(destination / remote_prefix.split("/", 1)[0], ignore_errors=True)
            return revision
        except CollectionPublicationError:
            raise
        except Exception as error:  # noqa: BLE001
            self._raise_transport(error)
            raise AssertionError("unreachable")


def _load_token(token_file: Path) -> str:
    try:
        metadata = token_file.lstat()
    except OSError:
        raise CollectionPublicationError("HF token file is unavailable") from None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077 or metadata.st_uid != os.geteuid():
        raise CollectionPublicationError("HF token file must be owner-only and regular")
    try:
        token = token_file.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        raise CollectionPublicationError("HF token file is unreadable") from None
    if not token or any(character.isspace() for character in token):
        raise CollectionPublicationError("HF token is unavailable")
    return token


def _publication_receipt_payload(
    *, result: CollectionPublicationResult, run_id: str, round_id: str,
    terminal_outcome: str, final_seal_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1, "kind": "lehome_simple_curriculum_publication_receipt_v1",
        "run_id": run_id, "round_id": round_id, "terminal_outcome": terminal_outcome,
        "repository": result.repository, "remote_prefix": result.remote_prefix,
        "immutable_revision": result.immutable_revision, "entry_count": len(result.entries),
        "entries": [
            {"relative_path": entry.relative_path, "sha256": entry.sha256, "byte_size": entry.byte_size}
            for entry in result.entries
        ],
        "bundle_sha256": result.bundle_sha256, "final_seal_sha256": final_seal_sha256,
        "readback_verified": result.readback_verified, "public_readback_verified": result.public_readback_verified,
    }


def _parse_publication_receipt(
    root: Path, *, run_id: str, round_id: str, terminal_outcome: str,
) -> tuple[CollectionPublicationResult, Path]:
    """Authenticate the immutable receipt and its exact remote manifest."""

    safe_root = _safe_bundle_root(root)
    receipt_path = safe_root / "reports" / "final-publication.json"
    published = _json_object(receipt_path, label="final publication receipt")
    required = {
        "schema_version", "kind", "run_id", "round_id", "terminal_outcome", "repository", "remote_prefix",
        "immutable_revision", "entry_count", "entries", "bundle_sha256", "final_seal_sha256",
        "readback_verified", "public_readback_verified",
    }
    if (
        set(published) != required or published.get("schema_version") != 1
        or published.get("kind") != "lehome_simple_curriculum_publication_receipt_v1"
        or published.get("run_id") != run_id or published.get("round_id") != round_id
        or published.get("terminal_outcome") != terminal_outcome
        or not isinstance(published.get("repository"), str) or not published["repository"].strip()
        or published.get("remote_prefix") != f"{_PUBLICATION_ROOT}/{run_id}"
        or not isinstance(published.get("immutable_revision"), str)
        or _COMMIT.fullmatch(str(published.get("immutable_revision"))) is None
        or type(published.get("entry_count")) is not int or int(published["entry_count"]) < 1
        or not isinstance(published.get("entries"), list)
        or any(
            not isinstance(published.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", str(published.get(key))) is None
            for key in ("bundle_sha256", "final_seal_sha256")
        )
        or published.get("readback_verified") is not True or published.get("public_readback_verified") is not True
    ):
        raise CollectionPublicationError("final publication receipt is malformed or incomplete")
    entries: list[PublicationEntry] = []
    for raw in published["entries"]:
        if not isinstance(raw, Mapping) or set(raw) != {"relative_path", "sha256", "byte_size"}:
            raise CollectionPublicationError("final publication receipt manifest is malformed")
        relative = _safe_relative(raw.get("relative_path"))
        if relative.split("/", 1)[0] not in {"manifests", "fresh", "replay", "reports", "seals"}:
            raise CollectionPublicationError("final publication receipt manifest is malformed")
        sha256, byte_size = raw.get("sha256"), raw.get("byte_size")
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None or type(byte_size) is not int or byte_size < 0:
            raise CollectionPublicationError("final publication receipt manifest is malformed")
        entries.append(PublicationEntry(relative, sha256, byte_size))
    ordered = tuple(sorted(entries, key=lambda entry: entry.relative_path))
    if tuple(entries) != ordered or len({entry.relative_path for entry in ordered}) != len(ordered):
        raise CollectionPublicationError("final publication receipt manifest is malformed")
    if int(published["entry_count"]) != len(ordered) or _entry_digest(ordered) != published["bundle_sha256"]:
        raise CollectionPublicationError("final publication receipt manifest does not match its immutable digest")
    return CollectionPublicationResult(
        repository=str(published["repository"]), remote_prefix=str(published["remote_prefix"]),
        immutable_revision=str(published["immutable_revision"]), entries=ordered,
        readback_verified=True, public_readback_verified=True, bundle_sha256=str(published["bundle_sha256"]),
    ), receipt_path


def _readback_receipt_payload(*, receipt_path: Path, result: CollectionPublicationResult) -> dict[str, object]:
    return {
        "schema_version": 1, "kind": "lehome_simple_curriculum_public_readback_receipt_v1",
        "publication_receipt_sha256": _sha256(receipt_path)[0], "repository": result.repository,
        "immutable_revision": result.immutable_revision, "remote_prefix": result.remote_prefix,
        "bundle_sha256": result.bundle_sha256, "authenticated_readback_verified": True,
        "anonymous_readback_verified": True,
    }


def _validate_readback_receipt(path: Path, *, receipt_path: Path, result: CollectionPublicationResult) -> None:
    if _json_object(path, label="final publication public readback receipt") != _readback_receipt_payload(
        receipt_path=receipt_path, result=result,
    ):
        raise CollectionPublicationError("final publication public readback receipt is malformed")


def reconcile_collection_publication(
    campaign_root: Path, *, run_id: str, round_id: str, terminal_outcome: str,
    token: str, transport: PublicCollectionTransport,
) -> tuple[CollectionPublicationResult, Path]:
    """Recover only a missing local public-readback receipt after a crash.

    This deliberately never resolves a mutable ref or uploads.  It uses the
    publication receipt's immutable commit and exact manifest to perform new
    authenticated and anonymous downloads, then atomically fills the one
    absent receipt.  A malformed existing readback receipt is immutable
    evidence of corruption and fails closed rather than being overwritten.
    """

    if not isinstance(token, str) or not token or any(character.isspace() for character in token):
        raise CollectionPublicationError("Hub token is unavailable")
    root = _safe_bundle_root(Path(campaign_root))
    result, receipt_path = _parse_publication_receipt(
        root, run_id=run_id, round_id=round_id, terminal_outcome=terminal_outcome,
    )
    readback_path = root / "reports" / "final-publication-readback.json"
    if readback_path.exists() or readback_path.is_symlink():
        _validate_readback_receipt(readback_path, receipt_path=receipt_path, result=result)
        return result, readback_path
    observed = _tree_files(
        _retry(
            lambda: transport.list_tree(
                repository=result.repository, revision=result.immutable_revision, token=token,
                remote_prefix=result.remote_prefix,
            ),
            label="publication resume manifest readback", max_attempts=3,
        ),
        prefix=result.remote_prefix,
    )
    if observed != {entry.relative_path for entry in result.entries}:
        raise CollectionPublicationError("publication resume immutable manifest differs from durable receipt")
    pinned = CollectionPublicationBundle(
        root=root, run_id=run_id, repository=result.repository,
        revision=result.immutable_revision, files=tuple(entry.relative_path for entry in result.entries),
    )
    _verify_download(
        transport=transport, bundle=pinned, revision=result.immutable_revision,
        prefix=result.remote_prefix, entries=result.entries, token=token,
    )
    _verify_download(
        transport=transport, bundle=pinned, revision=result.immutable_revision,
        prefix=result.remote_prefix, entries=result.entries, token=None,
    )
    _write_immutable_json(readback_path, _readback_receipt_payload(receipt_path=receipt_path, result=result))
    return result, readback_path


def publish_collection(
    campaign_root: Path, *, run_id: str, round_id: str, terminal_outcome: str,
    rollout_instance_id: str, repository: str, revision: str, token: str,
    transport: PublicCollectionTransport,
) -> tuple[CollectionPublicationResult, Path, Path]:
    """Build the content-addressed seal, publish it, and persist local receipts."""

    root = _safe_bundle_root(Path(campaign_root))
    seal = build_final_seal(
        root, run_id=run_id, round_id=round_id, terminal_outcome=terminal_outcome,
        rollout_instance_id=rollout_instance_id,
    )
    seal_path = root / "seals" / f"{seal['kind']}-{seal['seal_sha256']}.json"
    _write_immutable_json(seal_path, seal)
    bundle = _stage_collection_bundle(
        root, seal_path=seal_path, run_id=run_id, repository=repository, revision=revision,
    )
    try:
        result = publish_collection_bundle(bundle, token=token, transport=transport)
    finally:
        shutil.rmtree(bundle.root, ignore_errors=True)
    receipt_path = root / "reports" / "final-publication.json"
    readback_path = root / "reports" / "final-publication-readback.json"
    receipt = _publication_receipt_payload(
        result=result, run_id=run_id, round_id=round_id,
        terminal_outcome=terminal_outcome, final_seal_sha256=str(seal["seal_sha256"]),
    )
    _write_immutable_json(receipt_path, receipt)
    _write_immutable_json(readback_path, _readback_receipt_payload(receipt_path=receipt_path, result=result))
    return result, receipt_path, readback_path


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not isinstance(value, str) or not value:
        raise CollectionPublicationError(f"{name} is required")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run only inside the controller's sealed final-publication stage."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments not in ([], ["--reconcile"]):
        raise CollectionPublicationError("publisher accepts only the controller --reconcile mode")
    campaign_root = Path(_required_environment("LEHOME_CAMPAIGN_ROOT"))
    token = _load_token(Path(_required_environment("LEHOME_HF_TOKEN_FILE")))
    run_id = _required_environment("LEHOME_RUN_ID")
    round_id = _required_environment("LEHOME_ROUND_ID")
    outcome = _required_environment("LEHOME_TERMINAL_OUTCOME")
    if arguments == ["--reconcile"]:
        result, readback = reconcile_collection_publication(
            campaign_root, run_id=run_id, round_id=round_id, terminal_outcome=outcome,
            token=token, transport=HuggingFacePublicDatasetTransport(),
        )
        receipt = _safe_bundle_root(campaign_root) / "reports" / "final-publication.json"
    else:
        result, receipt, readback = publish_collection(
            campaign_root,
            run_id=run_id, round_id=round_id, terminal_outcome=outcome,
            rollout_instance_id=_required_environment("LEHOME_ROLLOUT_INSTANCE_ID"),
            repository=_required_environment("LEHOME_ROLLOUT_REPOSITORY"),
            revision=_required_environment("LEHOME_HF_REVISION"),
            token=token,
            transport=HuggingFacePublicDatasetTransport(),
        )
    # All user-visible output is non-sensitive evidence; token values never
    # reach arguments, files, JSON, or stdout.
    print(json.dumps({
        "immutable_revision": result.immutable_revision,
        "remote_prefix": result.remote_prefix,
        "publication_receipt": str(receipt),
        "publication_readback": str(readback),
    }, sort_keys=True))
    return 0


__all__ = (
    "CollectionPublicationBundle", "CollectionPublicationError", "CollectionPublicationResult",
    "PublicationEntry", "PublicCollectionTransport", "build_final_seal", "publish_collection",
    "publish_collection_bundle", "reconcile_collection_publication",
)


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except CollectionPublicationError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
