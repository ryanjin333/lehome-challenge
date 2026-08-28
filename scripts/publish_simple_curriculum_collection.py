#!/usr/bin/env python3
"""Publish one simple-curriculum collection as an immutable public bundle.

This program deliberately owns no collection or provider lifecycle work.  It
receives already-authenticated local evidence from the one-VM controller,
uploads a fixed tree under a run-specific prefix, and only returns a receipt
after both authenticated and anonymous fresh downloads match every byte.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Mapping, Protocol, Sequence
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "source" / "lehome", REPO_ROOT / "trainer" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(r"^fresh-run-[a-z0-9-]{1,112}$")
_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")
_PUBLICATION_ROOT = "collection-rounds"


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
        token: str, remote_prefix: str | None = None,
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
    root = Path(bundle.root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise CollectionPublicationError("publication root is missing or unsafe")
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
        path = (root / relative).resolve(strict=False)
        if not path.is_relative_to(root):
            raise CollectionPublicationError("publication path escapes bundle root")
        sha256, byte_size = _sha256(path)
        entries.append(PublicationEntry(relative, sha256, byte_size))
    return tuple(sorted(entries, key=lambda item: item.relative_path))


def _is_transient(error: BaseException) -> bool:
    """Only the canonical Hub retry categories are retryable."""

    return type(error).__name__ in {"HubTransientError", "HubRateLimitError"}


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
        revision = _retry(
            lambda: transport.upload_files(
                repository=bundle.repository, revision=bundle.revision, source=bundle.root,
                entries=entries, token=token, remote_prefix=prefix,
            ),
            label="collection upload", max_attempts=max_attempts,
        )
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


def _complete_fresh_counts(root: Path, *, run_id: str, round_id: str) -> tuple[int, int]:
    report = _json_object(root / "reports" / "fresh-source-report.json", label="fresh source report")
    matrix = _json_object_or_list(root / "reports" / "fresh-source-matrix.json", label="fresh source matrix")
    terminal = _json_object(root / "reports" / "fresh-terminal-artifacts.json", label="fresh terminal manifest")
    trials = report.get("trials")
    entries = terminal.get("entries")
    if (
        report.get("run_id") != run_id or report.get("round_id") != round_id
        or not isinstance(matrix, list) or not isinstance(trials, list) or not isinstance(entries, list)
    ):
        raise CollectionPublicationError("fresh source evidence is malformed")
    source_ids = {row.get("attempt_id") for row in matrix if isinstance(row, Mapping)}
    trial_ids = {row.get("attempt_id") for row in trials if isinstance(row, Mapping)}
    terminal_ids = {row.get("attempt_id") for row in entries if isinstance(row, Mapping)}
    if (
        len(matrix) != 1000 or len(trials) != 1000 or len(entries) != 1000
        or len(source_ids) != 1000 or source_ids != trial_ids or source_ids != terminal_ids
        or any(not isinstance(item, str) or not item for item in source_ids)
    ):
        raise CollectionPublicationError("collection complete requires exactly 1,000 fresh terminal outcomes")
    successes = sum(1 for trial in trials if isinstance(trial, Mapping) and trial.get("accepted_success") is True)
    return 1000, successes


def _json_object_or_list(path: Path, *, label: str) -> object:
    if path.is_symlink() or not path.is_file():
        raise CollectionPublicationError(f"{label} is missing or unsafe")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise CollectionPublicationError(f"{label} is malformed") from None


def _complete_replay_counts(root: Path) -> tuple[int, int, dict[str, int]]:
    seal = _json_object(root / "replay" / "success-replay-readback-seal.json", label="success replay readback seal")
    categories = {"pant_long", "pant_short", "top_long", "top_short"}
    accepted = seal.get("accepted_attempt_ids")
    by_category = seal.get("accepted_by_category")
    receipts = seal.get("readback_receipts")
    if (
        seal.get("kind") != "lehome_success_replay_readback_seal_v1"
        or seal.get("outcome") != "complete" or seal.get("readback_verified") is not True
        or not isinstance(accepted, list) or not isinstance(by_category, Mapping) or not isinstance(receipts, Mapping)
        or len(accepted) != 200 or len(set(accepted)) != 200
        or set(by_category) != categories or any(by_category.get(category) != 50 for category in categories)
        or set(receipts) != set(accepted)
        or any(
            not isinstance(value, Mapping)
            or set(value) != {"receipt_sha256", "episode_sha256", "immutable_revision"}
            or any(
                not isinstance(value.get(field), str)
                or re.fullmatch(r"[0-9a-f]{64}", str(value.get(field))) is None
                for field in ("receipt_sha256", "episode_sha256")
            )
            or not isinstance(value.get("immutable_revision"), str)
            or _COMMIT.fullmatch(str(value.get("immutable_revision"))) is None
            for value in receipts.values()
        )
    ):
        raise CollectionPublicationError("collection complete requires the exact accepted replay set and readbacks")
    for attempt in accepted:
        if not isinstance(attempt, str):
            raise CollectionPublicationError("collection complete replay attempt identity is invalid")
        artifact = root / "replay" / "accepted" / attempt
        receipt_path = root / "replay" / "hf-sync-receipts" / f"{attempt}.sync.json"
        if artifact.is_symlink() or not artifact.is_dir() or not _iter_regular_files(artifact):
            raise CollectionPublicationError("collection complete replay artifact is missing")
        receipt = _json_object(receipt_path, label="success replay Hub readback receipt")
        binding = receipts[attempt]
        assert isinstance(binding, Mapping)
        if (
            receipt.get("attempt_id") != attempt or receipt.get("readback_verified") is not True
            or _sha256(receipt_path)[0] != binding.get("receipt_sha256")
        ):
            raise CollectionPublicationError("collection complete replay receipt is not readback verified")
    return 400, 200, {category: 50 for category in sorted(categories)}


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

    root = Path(campaign_root).resolve()
    if root.is_symlink() or not root.is_dir() or not _RUN_ID.fullmatch(run_id):
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
        fresh_total, fresh_successes = _complete_fresh_counts(root, run_id=run_id, round_id=round_id)
        replay_attempts, replay_successes, replay_categories = _complete_replay_counts(root)
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
    if root.is_symlink() or not root.is_dir():
        raise CollectionPublicationError("publication source directory is missing or unsafe")
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        if any((current_path / name).is_symlink() for name in directories):
            raise CollectionPublicationError("publication source contains a symlinked directory")
        for name in names:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise CollectionPublicationError("publication source contains an unsafe file")
            if re.search(r"(?:token|secret|credential|password|api[_-]?key)", path.name, re.I):
                raise CollectionPublicationError("publication source includes a credential-like path")
            files.append(path)
    return tuple(sorted(files))


def _copy_to_staging(*, source: Path, root: Path, staging: Path, remote: str) -> None:
    _safe_relative(remote)
    if source.is_symlink() or not source.is_file() or not source.is_relative_to(root):
        raise CollectionPublicationError("publication source file is unsafe")
    target = staging / remote
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise CollectionPublicationError("duplicate staged collection path")
    shutil.copyfile(source, target, follow_symlinks=False)
    if _sha256(source) != _sha256(target):
        raise CollectionPublicationError("staged collection bytes differ from the source")


def _stage_collection_bundle(root: Path, *, seal_path: Path, run_id: str, repository: str, revision: str) -> CollectionPublicationBundle:
    """Map only reviewed campaign evidence into the five canonical sections."""

    staging = Path(tempfile.mkdtemp(prefix="lehome-curriculum-publication-", dir=root))
    try:
        for directory in ("inputs", "matrices", "partitions", "stage-receipts"):
            source_root = root / directory
            if not source_root.exists() and not source_root.is_symlink():
                continue
            for source in _iter_regular_files(source_root):
                _copy_to_staging(
                    source=source, root=root, staging=staging,
                    remote=f"manifests/{directory}/{source.relative_to(source_root).as_posix()}",
                )
        for directory in ("reports", "seals"):
            source_root = root / directory
            if not source_root.exists() and not source_root.is_symlink():
                continue
            for source in _iter_regular_files(source_root):
                if directory == "reports" and source.name in {
                    "final-publication.json", "final-publication-readback.json",
                }:
                    # These are local post-readback receipts.  Including them
                    # would change the immutable upload on a safe resume.
                    continue
                _copy_to_staging(
                    source=source, root=root, staging=staging,
                    remote=f"{directory}/{source.relative_to(source_root).as_posix()}",
                )
        if seal_path.is_relative_to(root / "seals") is False:
            raise CollectionPublicationError("final seal is outside the canonical seals directory")
        for partition in ("calibration-head", "calibration-tail", "curriculum-a", "curriculum-b"):
            partition_root = root / "fresh" / partition
            for name in ("accepted", "evaluation-terminal", "hf-sync-receipts"):
                candidate = partition_root / name
                if candidate.exists() or candidate.is_symlink():
                    for source in _iter_regular_files(candidate):
                        _copy_to_staging(
                            source=source, root=root, staging=staging,
                            remote=f"fresh/{partition}/{name}/{source.relative_to(candidate).as_posix()}",
                        )
        replay = root / "replay"
        if replay.exists() or replay.is_symlink():
            for source in _iter_regular_files(replay):
                # SQLite journals are local controller implementation state;
                # terminal artifacts and immutable replay evidence are public.
                if source.name in {"ledger.sqlite3", "ledger.sqlite3-shm", "ledger.sqlite3-wal"}:
                    continue
                _copy_to_staging(
                    source=source, root=root, staging=staging,
                    remote=f"replay/{source.relative_to(replay).as_posix()}",
                )
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
        if isinstance(error, (ConnectionError, TimeoutError)):
            raise HubTransientError("public Hub transport interruption") from None
        raise error

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
            self._raise_transport(error)
            raise AssertionError("unreachable")

    def upload_files(self, *, repository: str, revision: str, source: Path, entries: Sequence[PublicationEntry], token: str, remote_prefix: str | None = None) -> str:
        if not remote_prefix:
            raise CollectionPublicationError("public upload prefix is missing")
        try:
            result = self._api(token).upload_folder(
                repo_id=repository, repo_type="dataset", revision=revision, folder_path=str(source),
                allow_patterns=[entry.relative_path for entry in entries], path_in_repo=remote_prefix, token=token,
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


def publish_collection(
    campaign_root: Path, *, run_id: str, round_id: str, terminal_outcome: str,
    rollout_instance_id: str, repository: str, revision: str, token: str,
    transport: PublicCollectionTransport,
) -> tuple[CollectionPublicationResult, Path, Path]:
    """Build the content-addressed seal, publish it, and persist local receipts."""

    root = Path(campaign_root).resolve()
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
    receipt = {
        "schema_version": 1, "kind": "lehome_simple_curriculum_publication_receipt_v1",
        "run_id": run_id, "round_id": round_id, "terminal_outcome": terminal_outcome,
        "repository": result.repository, "remote_prefix": result.remote_prefix,
        "immutable_revision": result.immutable_revision, "entry_count": len(result.entries),
        "bundle_sha256": result.bundle_sha256, "final_seal_sha256": seal["seal_sha256"],
        "readback_verified": result.readback_verified, "public_readback_verified": result.public_readback_verified,
    }
    _write_immutable_json(receipt_path, receipt)
    _write_immutable_json(readback_path, {
        "schema_version": 1, "kind": "lehome_simple_curriculum_public_readback_receipt_v1",
        "publication_receipt_sha256": _sha256(receipt_path)[0], "repository": result.repository,
        "immutable_revision": result.immutable_revision, "remote_prefix": result.remote_prefix,
        "bundle_sha256": result.bundle_sha256, "authenticated_readback_verified": True,
        "anonymous_readback_verified": True,
    })
    return result, receipt_path, readback_path


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not isinstance(value, str) or not value:
        raise CollectionPublicationError(f"{name} is required")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run only inside the controller's sealed final-publication stage."""

    if argv:
        raise CollectionPublicationError("publisher accepts controller environment only")
    campaign_root = Path(_required_environment("LEHOME_CAMPAIGN_ROOT"))
    token = _load_token(Path(_required_environment("LEHOME_HF_TOKEN_FILE")))
    result, receipt, readback = publish_collection(
        campaign_root,
        run_id=_required_environment("LEHOME_RUN_ID"),
        round_id=_required_environment("LEHOME_ROUND_ID"),
        terminal_outcome=_required_environment("LEHOME_TERMINAL_OUTCOME"),
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
    "publish_collection_bundle",
)


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except CollectionPublicationError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
