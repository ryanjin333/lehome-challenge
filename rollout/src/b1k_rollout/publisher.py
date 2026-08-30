"""Bounded-memory immutable publication of classified B1K rollout evidence."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from b1k_rollout.contracts import RolloutContract
from b1k_rollout.episodes import (
    EpisodeEnvelope,
    EpisodeIntegrityError,
    copy_verified_artifact,
    load_episode_envelopes,
)
from b1k_rollout.identity import (
    DATASET_REPO,
    canonical_json_bytes,
    canonical_json_sha256,
    reject_credential_material,
    require_immutable_commit,
    require_sha256,
)
from b1k_rollout.outcomes import Outcome, raw_evidence_sha256, revalidate_classification
from b1k_rollout.task_manifest import load_task_manifest
from b1k_rollout.provenance import ProvenanceAuthenticator


_CHUNK_BYTES = 1024 * 1024
_MAX_METADATA_BYTES = 1024 * 1024
_MAX_RELEASE_EPISODES = 1000


class PublicationError(ValueError):
    """A release could not be safely published or reconciled."""


class HubAdapter(Protocol):
    """A path/stream boundary; adapters must not materialize release files."""

    def get_dataset_info(self, repo_id: str) -> object: ...

    def list_tree(self, repo_id: str, *, revision: str, prefix: str) -> Mapping[str, str]: ...

    def upload_tree(
        self, repo_id: str, *, local_dir: Path, remote_prefix: str, commit_message: str
    ) -> str: ...

    def promote_prefix(
        self, repo_id: str, *, staging_prefix: str, release_prefix: str, commit_message: str
    ) -> str: ...

    def delete_prefix(self, repo_id: str, *, prefix: str) -> str: ...

    def download_file_to_path(
        self, repo_id: str, *, revision: str, path: str, destination: Path
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class PublishResult:
    release_id: str
    commit_sha: str
    release_prefix: str
    release_manifest: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _File:
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _PreparedRelease:
    release_id: str
    campaign_id: str
    manifest: Mapping[str, object]
    files: Mapping[str, _File]


def publish_release(
    *,
    hub: HubAdapter,
    episodes: Sequence[EpisodeEnvelope],
    artifact_roots: Mapping[str, Path],
    contract: RolloutContract | None = None,
    task_manifest: Mapping[str, object] | None = None,
    authenticator: ProvenanceAuthenticator | None = None,
) -> PublishResult:
    """Publish a release without retaining artifacts or remote readbacks in memory."""

    if not isinstance(contract, RolloutContract):
        raise PublicationError("an explicit frozen release contract is required")
    if not isinstance(authenticator, ProvenanceAuthenticator):
        raise PublicationError("an explicit provenance authenticator is required")
    try:
        verified = _revalidate_envelopes(episodes, task_manifest=task_manifest, contract=contract, authenticator=authenticator)
    except PublicationError:
        raise
    except Exception as error:
        raise PublicationError("immutable rollout release preparation failed") from error

    with tempfile.TemporaryDirectory(prefix="b1k-release-") as temporary:
        staging_root = Path(temporary).resolve(strict=True)
        try:
            prepared = _prepare_release(
                episodes=verified,
                artifact_roots=artifact_roots,
                contract=contract,
                root=staging_root,
            )
            return _publish_prepared(hub=hub, prepared=prepared, local_root=staging_root)
        except PublicationError:
            raise
        except Exception as error:
            raise PublicationError("immutable rollout release publication failed") from error


def _publish_prepared(*, hub: HubAdapter, prepared: _PreparedRelease, local_root: Path) -> PublishResult:
    release_prefix = f"campaigns/{prepared.campaign_id}/releases/{prepared.release_id}"
    staging_prefix = f"{release_prefix}.incomplete"
    _require_private_dataset(hub)
    staging_may_exist = False
    try:
        head = _head_commit(hub)
        if _tree_at(hub, revision=head, prefix=release_prefix):
            _verify_remote_release(hub, revision=head, prefix=release_prefix, prepared=prepared)
            receipt = _require_staging_absent(hub, staging_prefix)
            _verify_remote_release(hub, revision=receipt, prefix=release_prefix, prepared=prepared)
            return _result(prepared, receipt, release_prefix)

        staged = _tree_at(hub, revision=head, prefix=staging_prefix)
        staging_may_exist = bool(staged)
        if staged:
            _verify_remote_release(hub, revision=head, prefix=staging_prefix, prepared=prepared)
        else:
            staging_may_exist = True
            commit = _immutable_commit(
                hub.upload_tree(
                    DATASET_REPO,
                    local_dir=local_root,
                    remote_prefix=staging_prefix,
                    commit_message=f"stage immutable B1K rollout release {prepared.release_id}",
                )
            )
            _verify_remote_release(hub, revision=commit, prefix=staging_prefix, prepared=prepared)
        try:
            commit = _immutable_commit(
                hub.promote_prefix(
                    DATASET_REPO,
                    staging_prefix=staging_prefix,
                    release_prefix=release_prefix,
                    commit_message=f"publish immutable B1K rollout release {prepared.release_id}",
                )
            )
        except Exception:
            reconciled = _reconcile_final(hub, release_prefix, prepared)
            if reconciled is None:
                raise
            receipt = _require_staging_absent(hub, staging_prefix)
            _verify_remote_release(hub, revision=receipt, prefix=release_prefix, prepared=prepared)
            return _result(prepared, receipt, release_prefix)
        _verify_remote_release(hub, revision=commit, prefix=release_prefix, prepared=prepared)
        receipt = _require_staging_absent(hub, staging_prefix)
        _verify_remote_release(hub, revision=receipt, prefix=release_prefix, prepared=prepared)
        return _result(prepared, receipt, release_prefix)
    except PublicationError:
        if staging_may_exist:
            _cleanup_staging(hub, staging_prefix)
        raise
    except Exception as error:
        if staging_may_exist:
            _cleanup_staging(hub, staging_prefix)
        raise PublicationError("immutable rollout release publication failed") from error


def _revalidate_envelopes(
    episodes: Sequence[EpisodeEnvelope], *, task_manifest: Mapping[str, object] | None, contract: RolloutContract | None, authenticator: ProvenanceAuthenticator | None
) -> tuple[EpisodeEnvelope, ...]:
    if not isinstance(episodes, Sequence) or not episodes:
        raise PublicationError("a release requires at least one classified episode")
    if len(episodes) > _MAX_RELEASE_EPISODES:
        raise PublicationError("a canonical B100 release cannot contain more than 1000 episodes")
    manifest = task_manifest or _default_task_manifest()
    try:
        with tempfile.TemporaryDirectory(prefix="b1k-envelope-validation-") as temporary:
            root = Path(temporary).resolve(strict=True)
            seen: set[str] = set()
            for episode in episodes:
                if not isinstance(episode, EpisodeEnvelope) or not _safe_component(episode.episode_key):
                    raise PublicationError("episode envelope failed canonical integrity validation")
                if episode.episode_key in seen:
                    raise PublicationError("episode envelope failed canonical integrity validation")
                seen.add(episode.episode_key)
                _write_small_json(root / f"{episode.episode_key}.json", _serialized_envelope(episode))
            verified = load_episode_envelopes(root, contract=contract, authenticator=authenticator)
    except PublicationError:
        raise
    except (EpisodeIntegrityError, TypeError, ValueError, OSError) as error:
        raise PublicationError("episode envelope failed canonical integrity validation") from error
    for envelope in verified:
        classified = revalidate_classification(
            envelope.raw_evidence, envelope.provenance, task_manifest=manifest
        )
        if not _classification_matches_envelope(classified, envelope):
            raise PublicationError("episode envelope claimed outcome does not match canonical classification")
    return verified


def _classification_matches_envelope(classified: object, envelope: EpisodeEnvelope) -> bool:
    # Avoid importing a second dataclass contract: equality of every published
    # classification field is the only accepted bridge from raw evidence.
    return (
        getattr(classified, "outcome", None) is envelope.outcome
        and getattr(classified, "reason", None) == envelope.reason
        and getattr(classified, "episode_id", None) == envelope.episode_id
        and getattr(classified, "rollout_id", None) == envelope.rollout_id
        and getattr(classified, "final_q_scores", None) == envelope.final_q_scores
        and getattr(classified, "evaluator_metrics", None) == envelope.evaluator_metrics
        and getattr(classified, "evaluator_identity", None) == envelope.evaluator_identity
        and getattr(classified, "raw_evidence_sha256", None) == envelope.raw_evidence_sha256
        and getattr(classified, "provenance", None) == envelope.provenance
    )


def _default_task_manifest() -> Mapping[str, object]:
    return load_task_manifest(Path(__file__).parents[2] / "task-manifest.json")


def _serialized_envelope(episode: EpisodeEnvelope) -> dict[str, object]:
    raw, raw_encoding = _envelope_value(episode.raw_evidence)
    metrics, metrics_encoding = _envelope_value(episode.evaluator_metrics)
    payload: dict[str, object] = {
        "schema_version": 2, "episode_key": episode.episode_key, "episode_id": episode.episode_id,
        "rollout_id": episode.rollout_id, "evaluator_identity": episode.evaluator_identity,
        "outcome": episode.outcome.value if isinstance(episode.outcome, Outcome) else episode.outcome,
        "reason": episode.reason, "raw_evidence": raw, "raw_evidence_encoding": raw_encoding,
        "raw_evidence_sha256": episode.raw_evidence_sha256, "final_q_scores": episode.final_q_scores,
        "evaluator_metrics": metrics, "evaluator_metrics_encoding": metrics_encoding,
        "provenance": episode.provenance,
        "provenance_attestation": episode.provenance_attestation,
    }
    if episode.canonical_sha256 != canonical_json_sha256(payload):
        raise PublicationError("episode envelope failed canonical integrity validation")
    payload["canonical_sha256"] = episode.canonical_sha256
    return payload


def _envelope_value(value: object) -> tuple[object, str]:
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii"), "bytes"
    if isinstance(value, str):
        return value, "text"
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return json.dumps(value, allow_nan=True, ensure_ascii=False, separators=(",", ":"), sort_keys=True), "nonfinite_json"
    return value, "json"


def _prepare_release(*, episodes: Sequence[EpisodeEnvelope], artifact_roots: Mapping[str, Path], contract: RolloutContract, root: Path) -> _PreparedRelease:
    if not isinstance(artifact_roots, Mapping):
        raise PublicationError("artifact roots must be keyed by episode key")
    rollout_contract = contract
    if rollout_contract.dataset_repository != DATASET_REPO:
        raise PublicationError("release contract is invalid")
    files: dict[str, _File] = {}
    episode_index_writer = _IndexShardWriter(root=root, kind="episodes", files=files)
    counts = {item.value: 0 for item in Outcome}
    bytes_by_partition = {item.value: 0 for item in Outcome}
    ids: set[str] = set()
    keys: set[str] = set()
    for episode in sorted(episodes, key=lambda value: value.episode_key):
        _require_episode_contract(episode, rollout_contract)
        if episode.episode_key in keys:
            raise PublicationError("every classified episode must appear exactly once")
        keys.add(episode.episode_key)
        published_id = episode.episode_id or episode.episode_key
        if not _safe_component(published_id) or published_id in ids:
            raise PublicationError("published episode ids must be unique safe identifiers")
        ids.add(published_id)
        artifact_hashes = _artifact_hashes(episode.raw_evidence)
        artifact_root = artifact_roots.get(episode.episode_key)
        if artifact_hashes and (not isinstance(artifact_root, Path) or not artifact_root.is_absolute()):
            raise PublicationError("an absolute artifact root is required for declared artifacts")
        if not artifact_hashes and artifact_root is not None:
            raise PublicationError("artifact root was supplied without retained artifact hashes")
        partition = episode.outcome.value
        prefix = f"{partition}/{published_id}"
        evidence_name = _raw_evidence_name(episode.raw_evidence)
        evidence = _write_raw_evidence(root / prefix / evidence_name, episode.raw_evidence)
        _add_file(files, f"{prefix}/{evidence_name}", evidence)
        bytes_by_partition[partition] += evidence.size
        envelope_file = _write_small_json(
            root / prefix / "episode-envelope.json", _serialized_envelope(episode)
        )
        _add_file(files, f"{prefix}/episode-envelope.json", envelope_file)
        bytes_by_partition[partition] += envelope_file.size
        for relative, expected in sorted(artifact_hashes.items()):
            if _reserved_artifact_name(relative):
                raise PublicationError("artifact path collides with a reserved release filename")
            assert artifact_root is not None
            target = root / prefix / relative
            try:
                copied = copy_verified_artifact(artifact_root, relative, target, expected)
            except EpisodeIntegrityError as error:
                raise PublicationError("episode artifacts do not match retained evidence") from error
            _add_file(files, f"{prefix}/{relative}", copied)
            bytes_by_partition[partition] += copied.size
        entry = {
            "episode_key": episode.episode_key, "episode_id": episode.episode_id,
            "published_episode_id": published_id, "outcome": partition, "reason": episode.reason,
            "rollout_id": episode.rollout_id, "evaluator_identity": episode.evaluator_identity,
            "episode_envelope_sha256": episode.canonical_sha256,
            "raw_evidence_sha256": episode.raw_evidence_sha256,
            "artifact_hashes": dict(sorted(artifact_hashes.items())),
            "envelope_path": f"{prefix}/episode-envelope.json",
            "envelope_sha256": envelope_file.sha256,
            "envelope_bytes": envelope_file.size,
        }
        reject_credential_material(entry)
        episode_index_writer.append(entry); counts[partition] += 1
    if set(artifact_roots) - keys:
        raise PublicationError("artifact roots include an unclassified episode")

    episode_index = episode_index_writer.finish()
    release_id = canonical_json_sha256(
        {
            "schema_version": 2,
            "contract": rollout_contract.to_dict(),
            "contract_identity": rollout_contract.identity,
            "episode_index": episode_index,
        }
    )
    _add_file(files, "campaign-manifest.json", _write_small_json(root / "campaign-manifest.json", {
        "schema_version": 1, "campaign_id": rollout_contract.campaign_id,
        "contract": rollout_contract.to_dict(), "contract_identity": rollout_contract.identity,
    }))
    payload_index = _write_index_shards(
        root=root,
        kind="payloads",
        records=(
            {"path": path, "sha256": item.sha256, "bytes": item.size}
            for path, item in sorted(files.items())
        ),
        files=files,
    )
    manifest: dict[str, object] = {
        "schema_version": 2, "release_id": release_id, "contract": rollout_contract.to_dict(),
        "contract_identity": rollout_contract.identity, "counts": counts,
        "partitions": {name: {"episodes": counts[name], "bytes": bytes_by_partition[name]} for name in sorted(counts)},
        "episode_index": episode_index,
        "payload_index": payload_index,
        "payload_tree_sha256": canonical_json_sha256(payload_index),
    }
    reject_credential_material(manifest)
    _add_file(files, "release-manifest.json", _write_small_json(root / "release-manifest.json", manifest))
    return _PreparedRelease(release_id=release_id, campaign_id=rollout_contract.campaign_id, manifest=manifest, files=files)


def _write_index_shards(
    *,
    root: Path,
    kind: str,
    records: Iterable[Mapping[str, object]],
    files: dict[str, _File],
) -> list[dict[str, object]]:
    """Write a bounded Merkle-like index without materializing the full release list.

    The compact root manifest authenticates each shard by path, byte size, and
    hash.  Each shard then authenticates its own records, so readers can stream
    only the range they need while the release preparation path retains at most
    one metadata shard in addition to the existing file descriptors.
    """

    writer = _IndexShardWriter(root=root, kind=kind, files=files)
    for record in records:
        writer.append(record)
    return writer.finish()


class _IndexShardWriter:
    """One bounded in-memory shard plus compact descriptors for a release index."""

    def __init__(self, *, root: Path, kind: str, files: dict[str, _File]) -> None:
        if kind not in ("episodes", "payloads"):
            raise PublicationError("release index kind is invalid")
        self._root = root
        self._kind = kind
        self._files = files
        self._descriptors: list[dict[str, object]] = []
        self._pending: list[dict[str, object]] = []
        self._records_bytes = 0
        self._empty_payload_bytes = len(
            canonical_json_bytes({"schema_version": 1, "kind": kind, "records": []})
        )

    def append(self, record: Mapping[str, object]) -> None:
        if not isinstance(record, Mapping):
            raise PublicationError("release index record is invalid")
        candidate = dict(record)
        candidate_bytes = self._record_size(candidate)
        if self._empty_payload_bytes - 2 + candidate_bytes + 1 > _MAX_METADATA_BYTES:
            raise PublicationError("release index record exceeds the bounded size limit")
        if self._encoded_size(candidate_bytes) > _MAX_METADATA_BYTES:
            self._flush()
        self._pending.append(candidate)
        self._records_bytes += candidate_bytes + (1 if len(self._pending) > 1 else 0)

    def finish(self) -> list[dict[str, object]]:
        self._flush()
        if not self._descriptors:
            raise PublicationError("release index cannot be empty")
        return list(self._descriptors)

    def _record_size(self, record: Mapping[str, object]) -> int:
        try:
            return len(canonical_json_bytes(record))
        except (TypeError, ValueError) as error:
            raise PublicationError("release index record cannot be serialized") from error

    def _encoded_size(self, next_record_bytes: int) -> int:
        records_bytes = self._records_bytes + next_record_bytes + (1 if self._pending else 0)
        # ``_empty_payload_bytes`` already includes the empty-list brackets.
        # A non-empty list retains those brackets and adds the comma-separated
        # canonical record bytes; the metadata writer also appends one newline.
        return self._empty_payload_bytes + records_bytes + 1

    def _flush(self) -> None:
        if not self._pending:
            return
        path = f"indexes/{self._kind}/{len(self._descriptors):04d}.json"
        payload = {"schema_version": 1, "kind": self._kind, "records": self._pending}
        reject_credential_material(payload)
        item = _write_small_json(self._root / path, payload)
        _add_file(self._files, path, item)
        self._descriptors.append(
            {
                "path": path,
                "sha256": item.sha256,
                "bytes": item.size,
                "record_count": len(self._pending),
            }
        )
        self._pending.clear()
        self._records_bytes = 0


def _require_episode_contract(episode: EpisodeEnvelope, contract: RolloutContract) -> None:
    """Reject raw official evidence replayed under a different release contract."""

    evidence = _raw_mapping(episode.raw_evidence)
    if evidence is None or not isinstance(evidence.get("contract"), Mapping):
        return
    try:
        evidence_contract = RolloutContract.from_mapping(evidence["contract"])
    except (TypeError, ValueError) as error:
        raise PublicationError("episode evidence does not contain a valid rollout contract") from error
    if evidence_contract.identity != contract.identity:
        raise PublicationError("episode evidence contract does not match the explicit release contract")


def _artifact_hashes(raw_evidence: object) -> dict[str, str]:
    evidence = _raw_mapping(raw_evidence)
    value = evidence.get("artifact_hashes") if evidence is not None else None
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PublicationError("retained artifact hashes are invalid")
    hashes: dict[str, str] = {}
    for path, digest in value.items():
        if not isinstance(path, str) or not _safe_relative_path(path):
            raise PublicationError("retained artifact path is unsafe")
        try:
            hashes[path] = require_sha256(digest, label="retained artifact hash")
        except ValueError as error:
            raise PublicationError("retained artifact hash is invalid") from error
    return hashes


def _raw_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, bytes):
        try: value = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError): return None
    elif isinstance(value, str):
        try: value = json.loads(value)
        except json.JSONDecodeError: return None
    return value if isinstance(value, Mapping) else None


def _raw_evidence_name(value: object) -> str:
    return "raw-evidence.bin" if isinstance(value, (bytes, bytearray)) else "raw-evidence.txt" if isinstance(value, str) else "raw-evidence.json"


def _write_raw_evidence(path: Path, value: object) -> _File:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (bytes, bytearray)):
        return _write_chunks(path, (bytes(value[index:index + _CHUNK_BYTES]) for index in range(0, len(value), _CHUNK_BYTES)))
    if isinstance(value, str):
        return _write_chunks(path, _utf8_chunks(value))
    try:
        encoder = json.JSONEncoder(allow_nan=True, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as error:
        raise PublicationError("retained raw evidence cannot be published") from error
    return _write_chunks(path, _json_chunks(encoder.iterencode(value)))


def _utf8_chunks(value: str):
    for index in range(0, len(value), _CHUNK_BYTES // 4):
        yield value[index:index + _CHUNK_BYTES // 4].encode("utf-8")


def _json_chunks(parts: Iterable[str]) -> Iterable[bytes]:
    for part in parts:
        encoded = part.encode("utf-8")
        for index in range(0, len(encoded), _CHUNK_BYTES):
            yield encoded[index:index + _CHUNK_BYTES]


def _write_small_json(path: Path, value: object) -> _File:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(path.name + ".incomplete")
    digest = hashlib.sha256(); size = 0
    encoder = json.JSONEncoder(allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    try:
        with stage.open("xb") as writer:
            for text in encoder.iterencode(value):
                data = text.encode("utf-8")
                if size + len(data) > _MAX_METADATA_BYTES:
                    raise PublicationError("release metadata exceeds the bounded size limit")
                writer.write(data); digest.update(data); size += len(data)
            if size + 1 > _MAX_METADATA_BYTES:
                raise PublicationError("release metadata exceeds the bounded size limit")
            writer.write(b"\n"); digest.update(b"\n"); size += 1
        os.replace(stage, path)
    except Exception:
        stage.unlink(missing_ok=True)
        raise
    return _File(path, size, digest.hexdigest())


def _write_chunks(path: Path, chunks: Iterable[bytes]) -> _File:
    digest = hashlib.sha256(); size = 0
    with path.open("xb") as writer:
        for chunk in chunks:
            if not isinstance(chunk, bytes): raise PublicationError("release content chunk is invalid")
            digest.update(chunk); size += len(chunk); writer.write(chunk)
    return _File(path, size, digest.hexdigest())


def _require_private_dataset(hub: HubAdapter) -> None:
    try: info = hub.get_dataset_info(DATASET_REPO)
    except Exception as error: raise PublicationError("cannot inspect target dataset privacy") from error
    private = info.get("private") if isinstance(info, Mapping) else getattr(info, "private", None)
    if private is not True: raise PublicationError("the exact rollout dataset repository must be private")


def _head_commit(hub: HubAdapter) -> str:
    try: info = hub.get_dataset_info(DATASET_REPO)
    except Exception as error: raise PublicationError("cannot inspect target dataset revision") from error
    return _immutable_commit(info.get("sha") if isinstance(info, Mapping) else getattr(info, "sha", None))


def _immutable_commit(value: object) -> str:
    try: return require_immutable_commit(value, label="Hub revision")
    except ValueError as error: raise PublicationError("Hub operation did not return an immutable commit") from error


def _tree_at(hub: HubAdapter, *, revision: str, prefix: str) -> Mapping[str, str]:
    try: tree = hub.list_tree(DATASET_REPO, revision=revision, prefix=prefix)
    except Exception as error: raise PublicationError("cannot read immutable release tree") from error
    if not isinstance(tree, Mapping) or not all(isinstance(path, str) and isinstance(digest, str) for path, digest in tree.items()):
        raise PublicationError("Hub release tree is invalid")
    return dict(tree)


def _verify_remote_release(hub: HubAdapter, *, revision: str, prefix: str, prepared: _PreparedRelease) -> None:
    expected = {f"{prefix}/{name}": item.sha256 for name, item in prepared.files.items()}
    if dict(sorted(_tree_at(hub, revision=revision, prefix=prefix).items())) != dict(sorted(expected.items())):
        raise PublicationError("immutable release tree does not match the expected content")
    with tempfile.TemporaryDirectory(prefix="b1k-readback-") as temporary:
        directory = Path(temporary).resolve(strict=True)
        for remote, digest in expected.items():
            destination = directory / hashlib.sha256(remote.encode()).hexdigest()
            try: hub.download_file_to_path(DATASET_REPO, revision=revision, path=remote, destination=destination)
            except Exception as error: raise PublicationError("cannot stream immutable release content") from error
            actual = _hash_file(destination)
            if actual.sha256 != digest or actual.size != prepared.files[remote.removeprefix(prefix + "/")].size:
                raise PublicationError("immutable release content hash verification failed")


def _hash_file(path: Path) -> _File:
    digest = hashlib.sha256(); size = 0
    with path.open("rb") as reader:
        while chunk := reader.read(_CHUNK_BYTES): digest.update(chunk); size += len(chunk)
    return _File(path, size, digest.hexdigest())


def _reconcile_final(hub: HubAdapter, prefix: str, prepared: _PreparedRelease) -> str | None:
    try:
        head = _head_commit(hub)
        if not _tree_at(hub, revision=head, prefix=prefix): return None
        _verify_remote_release(hub, revision=head, prefix=prefix, prepared=prepared)
        return head
    except PublicationError: return None


def _require_staging_absent(hub: HubAdapter, prefix: str) -> str:
    head = _head_commit(hub)
    if _tree_at(hub, revision=head, prefix=prefix):
        _cleanup_staging(hub, prefix); head = _head_commit(hub)
        if _tree_at(hub, revision=head, prefix=prefix): raise PublicationError("staging cleanup left an incomplete release tree")
    return head


def _cleanup_staging(hub: HubAdapter, prefix: str) -> None:
    if not prefix.startswith("campaigns/") or "/releases/" not in prefix or not prefix.endswith(".incomplete"):
        raise PublicationError("staging prefix is invalid")
    try: hub.delete_prefix(DATASET_REPO, prefix=prefix)
    except Exception as error: raise PublicationError("staging cleanup failed") from error
    if _tree_at(hub, revision=_head_commit(hub), prefix=prefix):
        raise PublicationError("staging cleanup left an incomplete release tree")


_RESERVED = frozenset({"raw-evidence.json", "raw-evidence.txt", "raw-evidence.bin", "episode-envelope.json", "campaign-manifest.json", "release-manifest.json"})


def _reserved_artifact_name(path: str) -> bool:
    name = PurePosixPath(path).name.casefold()
    return name in {item.casefold() for item in _RESERVED} or "envelope" in name or "manifest" in name


def _add_file(files: dict[str, _File], path: str, item: _File) -> None:
    if not _safe_relative_path(path) or path in files: raise PublicationError("release file path is invalid or collides with existing content")
    files[path] = item


def _safe_component(value: object) -> bool:
    return isinstance(value, str) and bool(value) and "/" not in value and "\\" not in value and value not in (".", "..")


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return not path.is_absolute() and bool(path.parts) and all(part not in (".", "..") and _safe_component(part) for part in path.parts) and not value.endswith(".incomplete")


def _result(prepared: _PreparedRelease, commit: str, prefix: str) -> PublishResult:
    return PublishResult(prepared.release_id, commit, prefix, dict(prepared.manifest))
