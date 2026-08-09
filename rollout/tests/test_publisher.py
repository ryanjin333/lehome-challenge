from __future__ import annotations

import base64
import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

import b1k_rollout.publisher as publisher_module
from b1k_rollout.episodes import EpisodeEnvelope, load_episode_envelopes, write_episode_envelope
from b1k_rollout.identity import canonical_json_bytes, canonical_json_sha256
from b1k_rollout.outcomes import ClassifiedOutcome, Outcome, classify_outcome, classify_outcome_file
from b1k_rollout.publisher import (
    PublicationError,
    _MAX_METADATA_BYTES,
    _write_small_json,
    publish_release as _publish_release,
)
from b1k_rollout.contracts import RolloutContract
from b1k_rollout.provenance import ProvenanceAuthenticator, canonical_attestation_payload
from b1k_rollout.task_manifest import load_task_manifest


FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = load_task_manifest(Path(__file__).parents[1] / "task-manifest.json")
REPOSITORY = "ryanjin333/behavior1k-groot-n17-rollouts"
AUTH_KEY = b"p" * 32
AUTH = ProvenanceAuthenticator(AUTH_KEY, issuer="test-producer")


def publish_release(**kwargs: object):
    """Keep legacy fixture calls authenticated through the public publisher API."""

    authenticator = kwargs.pop("authenticator", AUTH)
    return _publish_release(authenticator=authenticator, **kwargs)  # type: ignore[arg-type]


class FakeHub:
    """An in-memory boundary with immutable, revision-addressed trees."""

    def __init__(self, *, private: bool = True) -> None:
        self.private = private
        self.current: dict[str, bytes] = {}
        self.revisions: dict[str, dict[str, bytes]] = {}
        self.uploads = 0
        self.info_calls = 0
        self.deleted: list[str] = []
        self.fail_upload = False
        self.fail_upload_on: int | None = None
        self.raise_after_partial_upload = False
        self.fail_delete = False
        self.raise_after_promote = False
        self.leave_staging_on_promote = False
        self.max_chunk = 0
        self._serial = 0
        self.head = self._commit()

    def get_dataset_info(self, repo_id: str) -> Mapping[str, object]:
        assert repo_id == REPOSITORY
        self.info_calls += 1
        return {"private": self.private, "sha": self.head}

    def list_tree(self, repo_id: str, *, revision: str, prefix: str) -> Mapping[str, str]:
        assert repo_id == REPOSITORY
        tree = self.revisions[revision]
        return {
            path: hashlib.sha256(data).hexdigest()
            for path, data in tree.items()
            if path.startswith(prefix + "/")
        }

    def download_file(self, repo_id: str, *, revision: str, path: str) -> bytes:
        assert repo_id == REPOSITORY
        return self.revisions[revision][path]

    def download_file_to_path(
        self, repo_id: str, *, revision: str, path: str, destination: Path
    ) -> None:
        assert repo_id == REPOSITORY
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = self.revisions[revision][path]
        with destination.open("xb") as writer:
            for start in range(0, len(data), 1024 * 1024):
                chunk = data[start : start + 1024 * 1024]
                self.max_chunk = max(self.max_chunk, len(chunk))
                writer.write(chunk)

    def upload_tree(
        self, repo_id: str, *, local_dir: Path, remote_prefix: str, commit_message: str
    ) -> str:
        assert repo_id == REPOSITORY
        assert commit_message
        self.uploads += 1
        if self.fail_upload or self.fail_upload_on == self.uploads:
            raise RuntimeError(("hf_" + "example_secret_token_should_not_escape"))
        for path in sorted(local_dir.rglob("*")):
            if path.is_file():
                relative = path.relative_to(local_dir).as_posix()
                chunks: list[bytes] = []
                with path.open("rb") as reader:
                    while chunk := reader.read(1024 * 1024):
                        self.max_chunk = max(self.max_chunk, len(chunk))
                        chunks.append(chunk)
                self.current[f"{remote_prefix}/{relative}"] = b"".join(chunks)
                if self.raise_after_partial_upload:
                    self._commit()
                    raise RuntimeError("committed partial upload then transport failure")
        return self._commit()

    def promote_prefix(
        self, repo_id: str, *, staging_prefix: str, release_prefix: str, commit_message: str
    ) -> str:
        assert repo_id == REPOSITORY
        assert commit_message
        for path, data in list(self.current.items()):
            if path.startswith(staging_prefix + "/"):
                relative = path.removeprefix(staging_prefix + "/")
                self.current[f"{release_prefix}/{relative}"] = data
                if not self.leave_staging_on_promote:
                    del self.current[path]
        revision = self._commit()
        if self.raise_after_promote:
            raise RuntimeError(("ambiguous transport failure with hf_" + "example_secret_token_should_not_escape"))
        return revision

    def delete_prefix(self, repo_id: str, *, prefix: str) -> str:
        assert repo_id == REPOSITORY
        self.deleted.append(prefix)
        if self.fail_delete:
            raise RuntimeError("cleanup transport failure")
        for path in list(self.current):
            if path.startswith(prefix + "/"):
                del self.current[path]
        return self._commit()

    def _commit(self) -> str:
        self._serial += 1
        revision = f"{self._serial:040x}"
        self.revisions[revision] = dict(self.current)
        self.head = revision
        return revision


def _episodes_and_artifacts(tmp_path: Path):
    envelope_root = tmp_path / "envelopes"
    contract = _fixture_contract()
    artifact_roots: dict[str, Path] = {}
    for filename, key, incomplete in (
        ("closed-success.json", "success-key", False),
        ("closed-failure.json", "failure-key", False),
        ("closed-success.json", "quarantine-key", True),
    ):
        evidence = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
        evidence["episode_id"] = f"episode-{key}"
        if incomplete:
            evidence["completed"] = False
        root = tmp_path / "artifacts" / key
        root.mkdir(parents=True)
        payload = f"artifact for {key}".encode("utf-8")
        (root / "trajectory.jsonl").write_bytes(payload)
        evidence["artifact_hashes"] = {"trajectory.jsonl": hashlib.sha256(payload).hexdigest()}
        write_episode_envelope(
            envelope_root,
            key,
            classify_outcome(evidence, task_manifest=MANIFEST),
            contract=contract,
            authenticator=AUTH,
        )
        artifact_roots[key] = root
    return load_episode_envelopes(envelope_root, contract=contract, authenticator=AUTH), artifact_roots


def _fixture_contract() -> RolloutContract:
    return RolloutContract.from_mapping(
        json.loads((FIXTURES / "closed-success.json").read_text(encoding="utf-8"))["contract"]
    )


def test_publisher_creates_content_addressed_partitioned_release_and_verifies_it(
    tmp_path: Path,
) -> None:
    episodes, artifact_roots = _episodes_and_artifacts(tmp_path)
    hub = FakeHub()

    result = publish_release(
        hub=hub,
        episodes=episodes,
        artifact_roots=artifact_roots,
        contract=_fixture_contract(),
        authenticator=AUTH,
    )

    campaign = "campaign-001"
    campaign_prefix = f"campaigns/{campaign}"
    prefix = f"{campaign_prefix}/releases/{result.release_id}"
    paths = set(hub.list_tree(REPOSITORY, revision=result.commit_sha, prefix=prefix))
    assert result.release_id == result.release_manifest["release_id"]
    assert result.commit_sha in hub.revisions
    assert f"{prefix}/success/episode-success-key/trajectory.jsonl" in paths
    envelope_path = f"{prefix}/success/episode-success-key/episode-envelope.json"
    assert envelope_path in paths
    assert f"{prefix}/failure/episode-failure-key/trajectory.jsonl" in paths
    assert f"{prefix}/quarantine/episode-quarantine-key/trajectory.jsonl" in paths
    assert f"{prefix}/campaign-manifest.json" in paths
    assert f"{campaign_prefix}/campaign-manifest.json" not in hub.revisions[result.commit_sha]
    assert f"{prefix}/release-manifest.json" in paths
    assert not any(path.endswith("SHA256SUMS.json") for path in paths)
    assert result.release_manifest["counts"] == {"success": 1, "failure": 1, "quarantine": 1}
    assert {
        name: result.release_manifest["partitions"][name]["episodes"] for name in Outcome
    } == {"success": 1, "failure": 1, "quarantine": 1}
    assert all(result.release_manifest["partitions"][name]["bytes"] > 0 for name in Outcome)
    assert not any(path.startswith(f"{prefix}.incomplete/") for path in hub.current)
    episode_records = [
        entry
        for shard in result.release_manifest["episode_index"]
        for entry in json.loads(hub.revisions[result.commit_sha][f"{prefix}/{shard['path']}"])["records"]
    ]
    success_entry = next(entry for entry in episode_records if entry["episode_key"] == "success-key")
    assert success_entry["envelope_path"] == "success/episode-success-key/episode-envelope.json"
    assert success_entry["envelope_sha256"] == hashlib.sha256(hub.revisions[result.commit_sha][envelope_path]).hexdigest()
    assert success_entry["envelope_bytes"] == len(hub.revisions[result.commit_sha][envelope_path])
    assert json.loads(hub.revisions[result.commit_sha][envelope_path])["episode_key"] == "success-key"


def test_publisher_bounds_and_readbacks_indexed_metadata_for_a_real_1000_episode_campaign(
    tmp_path: Path,
) -> None:
    base = json.loads((FIXTURES / "closed-success.json").read_text(encoding="utf-8"))
    episodes: list[EpisodeEnvelope] = []
    artifact_roots: dict[str, Path] = {}
    for ordinal in range(1000):
        evidence = copy.deepcopy(base)
        episode_key = f"many-{ordinal:04d}"
        evidence["episode_id"] = f"episode-many-{ordinal:04d}"
        evidence["rollout_id"] = ordinal
        evidence["completed"] = False
        artifact_root = tmp_path / "artifacts" / episode_key
        artifact_root.mkdir(parents=True)
        artifact_hashes: dict[str, str] = {}
        for artifact_ordinal in range(8):
            name = f"artifact-{artifact_ordinal}.bin"
            content = f"{episode_key}:{artifact_ordinal}".encode("utf-8")
            (artifact_root / name).write_bytes(content)
            artifact_hashes[name] = hashlib.sha256(content).hexdigest()
        evidence["artifact_hashes"] = artifact_hashes
        artifact_roots[episode_key] = artifact_root
        episodes.append(
            _envelope_from_classification(
                episode_key,
                classify_outcome(evidence, task_manifest=MANIFEST),
            )
        )
    hub = FakeHub()

    result = publish_release(
        hub=hub,
        episodes=episodes,
        artifact_roots=artifact_roots,
        contract=_fixture_contract(),
        authenticator=AUTH,
    )

    prefix = f"campaigns/campaign-001/releases/{result.release_id}"
    release_manifest = hub.revisions[result.commit_sha][f"{prefix}/release-manifest.json"]
    assert len(release_manifest) <= _MAX_METADATA_BYTES
    assert result.release_manifest["counts"] == {"success": 0, "failure": 0, "quarantine": 1000}
    assert sum(shard["record_count"] for shard in result.release_manifest["episode_index"]) == 1000
    assert result.release_manifest["payload_index"]
    assert not any(path.endswith("SHA256SUMS.json") for path in hub.revisions[result.commit_sha])
    for shard in (*result.release_manifest["episode_index"], *result.release_manifest["payload_index"]):
        remote = f"{prefix}/{shard['path']}"
        content = hub.revisions[result.commit_sha][remote]
        assert len(content) <= _MAX_METADATA_BYTES
        assert len(content) == shard["bytes"]
        assert hashlib.sha256(content).hexdigest() == shard["sha256"]


def test_index_shard_splitter_matches_the_exact_metadata_writer_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = {"episode_key": "first"}
    second = {"episode_key": "other"}
    exact_cap = len(
        canonical_json_bytes({"schema_version": 1, "kind": "episodes", "records": [first]})
    ) + 1
    monkeypatch.setattr(publisher_module, "_MAX_METADATA_BYTES", exact_cap)

    single_files: dict[str, object] = {}
    single = publisher_module._IndexShardWriter(
        root=tmp_path / "single", kind="episodes", files=single_files  # type: ignore[arg-type]
    )
    single.append(first)
    single_shards = single.finish()

    split_files: dict[str, object] = {}
    split = publisher_module._IndexShardWriter(
        root=tmp_path / "split", kind="episodes", files=split_files  # type: ignore[arg-type]
    )
    split.append(first)
    split.append(second)
    split_shards = split.finish()

    assert single_shards[0]["bytes"] == exact_cap
    assert len(split_shards) == 2
    assert all(shard["bytes"] <= exact_cap for shard in split_shards)


def test_publisher_requires_an_explicit_frozen_release_contract(tmp_path: Path) -> None:
    episodes, artifact_roots = _episodes_and_artifacts(tmp_path)

    with pytest.raises(PublicationError, match="explicit.*contract"):
        publish_release(hub=FakeHub(), episodes=episodes, artifact_roots=artifact_roots)


def test_publisher_rejects_any_unverified_mapping_envelope_before_hub_access(tmp_path: Path) -> None:
    episodes, artifact_roots = _episodes_and_artifacts(tmp_path)
    hub = FakeHub()

    with pytest.raises(PublicationError, match="provenance authenticator"):
        _publish_release(
            hub=hub,
            episodes=episodes,
            artifact_roots=artifact_roots,
            contract=_fixture_contract(),
        )

    assert hub.info_calls == 0


def test_publisher_rejects_evidence_from_another_frozen_campaign_contract(tmp_path: Path) -> None:
    episodes, artifact_roots = _episodes_and_artifacts(tmp_path)
    wrong_contract = replace(_fixture_contract(), campaign_id="campaign-002")

    with pytest.raises(PublicationError, match="canonical integrity"):
        publish_release(
            hub=FakeHub(),
            episodes=episodes,
            artifact_roots=artifact_roots,
            contract=wrong_contract,
            authenticator=AUTH,
        )


def test_publisher_is_idempotent_when_the_verified_content_addressed_release_exists(
    tmp_path: Path,
) -> None:
    episodes, artifact_roots = _episodes_and_artifacts(tmp_path)
    hub = FakeHub()

    first = publish_release(hub=hub, episodes=episodes, artifact_roots=artifact_roots, contract=_fixture_contract(), authenticator=AUTH)
    uploads = hub.uploads
    second = publish_release(hub=hub, episodes=episodes, artifact_roots=artifact_roots, contract=_fixture_contract(), authenticator=AUTH)

    assert second == first
    assert hub.uploads == uploads


def test_publisher_requires_the_exact_dataset_repository_to_be_private(tmp_path: Path) -> None:
    episodes, artifact_roots = _episodes_and_artifacts(tmp_path)
    hub = FakeHub(private=False)

    with pytest.raises(PublicationError, match="private"):
        publish_release(hub=hub, episodes=episodes, artifact_roots=artifact_roots, contract=_fixture_contract(), authenticator=AUTH)

    assert hub.uploads == 0


def test_publisher_cleans_only_its_exact_incomplete_prefix_and_does_not_leak_adapter_errors(
    tmp_path: Path,
) -> None:
    episodes, artifact_roots = _episodes_and_artifacts(tmp_path)
    release_id = publish_release(
        hub=FakeHub(), episodes=episodes, artifact_roots=artifact_roots, contract=_fixture_contract()
    ).release_id
    hub = FakeHub()
    hub.fail_upload_on = 1

    with pytest.raises(PublicationError) as raised:
        publish_release(hub=hub, episodes=episodes, artifact_roots=artifact_roots, contract=_fixture_contract())

    assert ("hf_" + "example_secret_token_should_not_escape") not in str(raised.value)
    assert hub.deleted == [f"campaigns/campaign-001/releases/{release_id}.incomplete"]
    assert not any(path.startswith("campaigns/campaign-001/releases/") for path in hub.current)


def test_publisher_reconciles_an_ambiguous_promote_when_fresh_readback_is_complete(
    tmp_path: Path,
) -> None:
    episodes, artifact_roots = _episodes_and_artifacts(tmp_path)
    hub = FakeHub()
    hub.raise_after_promote = True

    result = publish_release(hub=hub, episodes=episodes, artifact_roots=artifact_roots, contract=_fixture_contract())

    assert result.commit_sha in hub.revisions
    assert hub.deleted == []
    assert (
        f"campaigns/campaign-001/releases/{result.release_id}/campaign-manifest.json"
        in hub.revisions[result.commit_sha]
    )
    assert "campaigns/campaign-001/campaign-manifest.json" not in hub.revisions[result.commit_sha]
    assert not any(path.endswith(".incomplete") or ".incomplete/" in path for path in hub.current)


def test_partial_staging_upload_cannot_strand_a_campaign_manifest_outside_the_release(
    tmp_path: Path,
) -> None:
    episodes, artifact_roots = _episodes_and_artifacts(tmp_path)
    hub = FakeHub()
    hub.raise_after_partial_upload = True

    with pytest.raises(PublicationError, match="publication failed"):
        publish_release(hub=hub, episodes=episodes, artifact_roots=artifact_roots, contract=_fixture_contract())

    assert not any(
        path == "campaigns/campaign-001/campaign-manifest.json" for path in hub.current
    )
    assert not any(".incomplete/" in path for path in hub.current)


def test_publisher_rejects_existing_content_or_contract_that_is_not_this_release(
    tmp_path: Path,
) -> None:
    episodes, artifact_roots = _episodes_and_artifacts(tmp_path)
    hub = FakeHub()
    first = publish_release(hub=hub, episodes=episodes, artifact_roots=artifact_roots, contract=_fixture_contract())
    path = (
        f"campaigns/campaign-001/releases/{first.release_id}/"
        "success/episode-success-key/trajectory.jsonl"
    )
    hub.current[path] = b"tampered"
    hub._commit()

    with pytest.raises(PublicationError, match="tree"):
        publish_release(hub=hub, episodes=episodes, artifact_roots=artifact_roots, contract=_fixture_contract())



@pytest.mark.parametrize(
    "changes",
    [
        {"outcome": Outcome.FAILURE},
        {"episode_id": "forged-episode"},
        {"final_q_scores": {"final": 0.0}},
        {"rollout_id": 999},
    ],
)
def test_publisher_revalidates_dataclass_envelopes_before_publishing(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    episodes, artifact_roots = _episodes_and_artifacts(tmp_path)
    forged = replace(episodes[-1], **changes)

    with pytest.raises(PublicationError, match="canonical integrity"):
        publish_release(
            hub=FakeHub(), episodes=(*episodes[:-1], forged), artifact_roots=artifact_roots, contract=_fixture_contract()
        )


def test_publisher_reclassifies_a_correctly_rehashed_terminal_as_not_quarantine(
    tmp_path: Path,
) -> None:
    episodes, artifact_roots = _episodes_and_artifacts(tmp_path)
    forged = _rehash_envelope(
        replace(episodes[-1], outcome=Outcome.QUARANTINE, reason="operator relabel")
    )
    hub = FakeHub()

    with pytest.raises(PublicationError, match="canonical classification"):
        publish_release(
            hub=hub, episodes=(*episodes[:-1], forged), artifact_roots=artifact_roots, contract=_fixture_contract()
        )

    assert hub.info_calls == 0


def test_publisher_preserves_a_parseable_success_file_marked_incomplete(tmp_path: Path) -> None:
    evidence = json.loads((FIXTURES / "closed-success.json").read_text(encoding="utf-8"))
    artifact_root = tmp_path / "incomplete-artifacts"
    artifact_root.mkdir()
    artifact = b"incomplete file artifact"
    (artifact_root / "trajectory.jsonl").write_bytes(artifact)
    evidence["artifact_hashes"] = {"trajectory.jsonl": hashlib.sha256(artifact).hexdigest()}
    source = tmp_path / "success.json.incomplete"
    source.write_text(json.dumps(evidence), encoding="utf-8")
    envelope_root = tmp_path / "incomplete-envelopes"
    contract = _fixture_contract()
    write_episode_envelope(
        envelope_root,
        "incomplete-file",
        classify_outcome_file(
            source,
            task_manifest=MANIFEST,
            episode_key="incomplete-file",
            contract=contract,
            authenticator=AUTH,
        ),
        contract=contract,
        authenticator=AUTH,
    )
    episodes = load_episode_envelopes(envelope_root, contract=contract, authenticator=AUTH)

    hub = FakeHub()
    result = publish_release(
        hub=hub,
        episodes=episodes,
        artifact_roots={"incomplete-file": artifact_root},
        contract=contract,
        authenticator=AUTH,
    )

    assert result.release_manifest["counts"]["quarantine"] == 1
    assert all(AUTH_KEY not in payload for payload in hub.revisions[result.commit_sha].values())


def test_file_origin_positive_infinity_attests_through_write_load_and_publish(tmp_path: Path) -> None:
    evidence = json.loads((FIXTURES / "closed-success.json").read_text(encoding="utf-8"))
    evidence["time"]["normalized_time"] = float("inf")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = b"infinity metric artifact"
    (artifact_root / "trajectory.jsonl").write_bytes(artifact)
    evidence["artifact_hashes"] = {"trajectory.jsonl": hashlib.sha256(artifact).hexdigest()}
    source = tmp_path / "evidence.json"
    source.write_text(json.dumps(evidence), encoding="utf-8")
    contract = _fixture_contract()
    envelope_root = tmp_path / "envelopes"
    write_episode_envelope(
        envelope_root,
        "infinity-file",
        classify_outcome_file(
            source,
            task_manifest=MANIFEST,
            episode_key="infinity-file",
            contract=contract,
            authenticator=AUTH,
        ),
        contract=contract,
        authenticator=AUTH,
    )
    episodes = load_episode_envelopes(envelope_root, contract=contract, authenticator=AUTH)

    result = publish_release(
        hub=FakeHub(),
        episodes=episodes,
        artifact_roots={"infinity-file": artifact_root},
        contract=contract,
        authenticator=AUTH,
    )

    assert result.release_manifest["counts"]["success"] == 1


def test_publisher_preserves_symlink_file_provenance_as_quarantine(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes((FIXTURES / "closed-success.json").read_bytes())
    source = tmp_path / "evidence.json"
    source.symlink_to(target)
    contract = _fixture_contract()
    classified = classify_outcome_file(
        source,
        task_manifest=MANIFEST,
        episode_key="symlink-file",
        contract=contract,
        authenticator=AUTH,
    )
    envelope_root = tmp_path / "symlink-envelopes"
    write_episode_envelope(
        envelope_root, "symlink-file", classified, contract=contract, authenticator=AUTH
    )

    result = publish_release(
        hub=FakeHub(),
        episodes=load_episode_envelopes(envelope_root, contract=contract, authenticator=AUTH),
        artifact_roots={},
        contract=contract,
        authenticator=AUTH,
    )

    assert result.release_manifest["counts"]["quarantine"] == 1


def test_small_metadata_limit_aborts_during_incremental_encoding(tmp_path: Path) -> None:
    destination = tmp_path / "release-manifest.json"

    with pytest.raises(PublicationError, match="bounded size"):
        _write_small_json(destination, {"large": "x" * (2 * 1024 * 1024)})

    assert not destination.exists()
    assert not destination.with_name(destination.name + ".incomplete").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"reason": ("Bearer hf_" + "abcdefghijklmnopqrstuvwxyzABCDEFGH")},
        {"raw_evidence": {"source": "LeHome", "artifact_hashes": {}}},
    ],
)
def test_publisher_rejects_forged_credential_or_lehome_envelopes_before_hub_access(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    episodes, artifact_roots = _episodes_and_artifacts(tmp_path)
    hub = FakeHub()

    with pytest.raises(PublicationError, match="canonical integrity") as raised:
        publish_release(
            hub=hub,
            episodes=(*episodes[:-1], replace(episodes[-1], **changes)),
            artifact_roots=artifact_roots,
            contract=_fixture_contract(),
        )

    assert ("hf_" + "abcdefghijklmnopqrstuvwxyzABCDEFGH") not in str(raised.value)
    assert hub.uploads == 0


@pytest.mark.parametrize(
    "raw_evidence",
    [
        (b"hf_" + b"abcdefghijklmnopqrstuvwxyzABCDEFGH"),
        b"retained LeHome quarantine bytes",
    ],
)
def test_publisher_rejects_correctly_rehashed_byte_backed_unsafe_evidence_before_hub_access(
    raw_evidence: bytes,
) -> None:
    hub = FakeHub()

    with pytest.raises(PublicationError, match="canonical integrity"):
        publish_release(
            hub=hub,
            episodes=(_byte_quarantine_envelope(raw_evidence),),
            artifact_roots={},
            contract=_fixture_contract(),
        )

    assert hub.info_calls == 0
    assert hub.uploads == 0


@pytest.mark.parametrize("metric", [(b"hf_" + b"abcdefghijklmnopqrstuvwxyzABCDEFGH"), b"LeHome metric"])
def test_publisher_rejects_decoded_byte_backed_metrics_before_hub_access(metric: bytes) -> None:
    hub = FakeHub()

    with pytest.raises(PublicationError, match="canonical integrity"):
        publish_release(
            hub=hub,
            episodes=(_byte_quarantine_envelope(b"incomplete", evaluator_metrics=metric),),
            artifact_roots={},
            contract=_fixture_contract(),
        )

    assert hub.info_calls == 0


@pytest.mark.parametrize("artifact_name", ["raw-evidence.json", "release-manifest.json"])
def test_publisher_rejects_reserved_artifact_collisions(tmp_path: Path, artifact_name: str) -> None:
    evidence = json.loads((FIXTURES / "closed-success.json").read_text(encoding="utf-8"))
    root = tmp_path / "collision-artifacts"
    root.mkdir()
    payload = b"collision"
    (root / artifact_name).write_bytes(payload)
    evidence["artifact_hashes"] = {artifact_name: hashlib.sha256(payload).hexdigest()}
    envelope_root = tmp_path / "collision-envelopes"
    write_episode_envelope(
        envelope_root,
        "collision",
        classify_outcome(evidence, task_manifest=MANIFEST),
        contract=_fixture_contract(),
        authenticator=AUTH,
    )
    episodes = load_episode_envelopes(
        envelope_root, contract=_fixture_contract(), authenticator=AUTH
    )

    with pytest.raises(PublicationError, match="reserved"):
        publish_release(hub=FakeHub(), episodes=episodes, artifact_roots={"collision": root}, contract=_fixture_contract())


def test_publisher_requires_cleanup_readback_and_can_retry_after_residual_staging(
    tmp_path: Path,
) -> None:
    episodes, artifact_roots = _episodes_and_artifacts(tmp_path)
    hub = FakeHub()
    hub.leave_staging_on_promote = True
    hub.fail_delete = True

    with pytest.raises(PublicationError, match="staging cleanup"):
        publish_release(hub=hub, episodes=episodes, artifact_roots=artifact_roots, contract=_fixture_contract())

    hub.fail_delete = False
    result = publish_release(hub=hub, episodes=episodes, artifact_roots=artifact_roots, contract=_fixture_contract())
    assert result.commit_sha == hub.head
    assert not any(".incomplete/" in path for path in hub.current)


def test_publisher_streams_multi_megabyte_artifacts_without_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = json.loads((FIXTURES / "closed-success.json").read_text(encoding="utf-8"))
    root = tmp_path / "large-artifacts"
    root.mkdir()
    payload = b"x" * (3 * 1024 * 1024 + 17)
    (root / "trajectory.jsonl").write_bytes(payload)
    evidence["artifact_hashes"] = {"trajectory.jsonl": hashlib.sha256(payload).hexdigest()}
    envelope_root = tmp_path / "large-envelopes"
    write_episode_envelope(
        envelope_root,
        "large",
        classify_outcome(evidence, task_manifest=MANIFEST),
        contract=_fixture_contract(),
        authenticator=AUTH,
    )
    episodes = load_episode_envelopes(
        envelope_root, contract=_fixture_contract(), authenticator=AUTH
    )
    monkeypatch.setattr(Path, "read_bytes", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("read_bytes")))
    hub = FakeHub()

    result = publish_release(hub=hub, episodes=episodes, artifact_roots={"large": root}, contract=_fixture_contract())

    assert result.commit_sha == hub.head
    assert hub.max_chunk <= 1024 * 1024


def _byte_quarantine_envelope(
    raw_evidence: bytes, *, evaluator_metrics: bytes | None = None
) -> EpisodeEnvelope:
    metrics_encoded = (
        base64.b64encode(evaluator_metrics).decode("ascii")
        if evaluator_metrics is not None
        else None
    )
    payload = {
        "schema_version": 2,
        "episode_key": "byte-quarantine",
        "episode_id": None,
        "rollout_id": None,
        "evaluator_identity": None,
        "outcome": "quarantine",
        "reason": "retained byte evidence",
        "raw_evidence": base64.b64encode(raw_evidence).decode("ascii"),
        "raw_evidence_encoding": "bytes",
        "raw_evidence_sha256": hashlib.sha256(raw_evidence).hexdigest(),
        "final_q_scores": None,
        "evaluator_metrics": metrics_encoded,
        "evaluator_metrics_encoding": "bytes" if evaluator_metrics is not None else "json",
        "provenance": {
            "origin": "mapping",
            "disposition": "mapping",
            "basename": None,
            "reason_code": "invalid",
            "diagnostic": "invalid",
            "raw_evidence_sha256": hashlib.sha256(raw_evidence).hexdigest(),
        },
        "provenance_attestation": None,
    }
    return EpisodeEnvelope(
        episode_key="byte-quarantine",
        episode_id=None,
        rollout_id=None,
        evaluator_identity=None,
        outcome=Outcome.QUARANTINE,
        reason="retained byte evidence",
        raw_evidence=raw_evidence,
        raw_evidence_sha256=payload["raw_evidence_sha256"],
        final_q_scores=None,
        evaluator_metrics=evaluator_metrics,
        provenance=payload["provenance"],
        provenance_attestation=None,
        canonical_sha256=canonical_json_sha256(payload),
    )


def _rehash_envelope(envelope: EpisodeEnvelope) -> EpisodeEnvelope:
    raw, raw_encoding = _encoded_envelope_value(envelope.raw_evidence)
    metrics, metrics_encoding = _encoded_envelope_value(envelope.evaluator_metrics)
    payload = {
        "schema_version": 2,
        "episode_key": envelope.episode_key,
        "episode_id": envelope.episode_id,
        "rollout_id": envelope.rollout_id,
        "evaluator_identity": envelope.evaluator_identity,
        "outcome": envelope.outcome.value,
        "reason": envelope.reason,
        "raw_evidence": raw,
        "raw_evidence_encoding": raw_encoding,
        "raw_evidence_sha256": envelope.raw_evidence_sha256,
        "final_q_scores": envelope.final_q_scores,
        "evaluator_metrics": metrics,
        "evaluator_metrics_encoding": metrics_encoding,
        "provenance": envelope.provenance,
        "provenance_attestation": envelope.provenance_attestation,
    }
    payload["provenance_attestation"] = AUTH.sign(
        canonical_attestation_payload(
            _fixture_contract(),
            envelope.episode_key,
            {
                key: payload[key]
                for key in (
                    "episode_id", "rollout_id", "evaluator_identity", "outcome", "reason",
                    "raw_evidence_sha256", "final_q_scores", "evaluator_metrics", "provenance",
                )
            },
        )
    )
    return replace(
        envelope,
        provenance_attestation=payload["provenance_attestation"],
        canonical_sha256=canonical_json_sha256(payload),
    )


def _envelope_from_classification(
    episode_key: str, classified: ClassifiedOutcome
) -> EpisodeEnvelope:
    raw, raw_encoding = _encoded_envelope_value(classified.raw_evidence)
    metrics, metrics_encoding = _encoded_envelope_value(classified.evaluator_metrics)
    payload = {
        "schema_version": 2,
        "episode_key": episode_key,
        "episode_id": classified.episode_id,
        "rollout_id": classified.rollout_id,
        "evaluator_identity": classified.evaluator_identity,
        "outcome": classified.outcome.value,
        "reason": classified.reason,
        "raw_evidence": raw,
        "raw_evidence_encoding": raw_encoding,
        "raw_evidence_sha256": classified.raw_evidence_sha256,
        "final_q_scores": classified.final_q_scores,
        "evaluator_metrics": metrics,
        "evaluator_metrics_encoding": metrics_encoding,
        "provenance": classified.provenance,
        "provenance_attestation": classified.provenance_attestation,
    }
    if payload["provenance_attestation"] is None:
        payload["provenance_attestation"] = AUTH.sign(
            canonical_attestation_payload(
                _fixture_contract(),
                episode_key,
                {
                    key: payload[key]
                    for key in (
                        "episode_id", "rollout_id", "evaluator_identity", "outcome", "reason",
                        "raw_evidence_sha256", "final_q_scores", "evaluator_metrics", "provenance",
                    )
                },
            )
        )
    return EpisodeEnvelope(
        episode_key=episode_key,
        episode_id=classified.episode_id,
        rollout_id=classified.rollout_id,
        evaluator_identity=classified.evaluator_identity,
        outcome=classified.outcome,
        reason=classified.reason,
        raw_evidence=classified.raw_evidence,
        raw_evidence_sha256=classified.raw_evidence_sha256,
        final_q_scores=classified.final_q_scores,
        evaluator_metrics=classified.evaluator_metrics,
        provenance=classified.provenance,
        provenance_attestation=payload["provenance_attestation"],
        canonical_sha256=canonical_json_sha256(payload),
    )


def _encoded_envelope_value(value: object) -> tuple[object, str]:
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii"), "bytes"
    if isinstance(value, str):
        return value, "text"
    return value, "json"
