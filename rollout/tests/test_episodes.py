from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from b1k_rollout.episodes import (
    EpisodeIntegrityError,
    copy_verified_artifact,
    load_episode_envelopes,
    verify_artifact_hashes,
    write_episode_envelope,
)
from b1k_rollout.contracts import RolloutContract
from b1k_rollout.identity import canonical_json_sha256
from b1k_rollout.outcomes import Outcome, classify_outcome, raw_evidence_sha256
from b1k_rollout.provenance import ProvenanceAuthenticator
from b1k_rollout.task_manifest import load_task_manifest
import b1k_rollout.episodes as episodes


FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = load_task_manifest(Path(__file__).parents[1] / "task-manifest.json")
AUTH_KEY = b"e" * 32
AUTH = ProvenanceAuthenticator(AUTH_KEY, issuer="episodes-test")


def _classified(name: str):
    evidence = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return classify_outcome(evidence, task_manifest=MANIFEST)


def _contract() -> RolloutContract:
    return RolloutContract.from_mapping(
        json.loads((FIXTURES / "closed-success.json").read_text(encoding="utf-8"))["contract"]
    )


def _file_classified(tmp_path: Path, *, episode_key: str, filename: str = "evidence.json"):
    path = tmp_path / filename
    path.write_bytes((FIXTURES / "closed-success.json").read_bytes())
    from b1k_rollout.outcomes import classify_outcome_file

    return classify_outcome_file(
        path,
        task_manifest=MANIFEST,
        episode_key=episode_key,
        contract=_contract(),
        authenticator=AUTH,
    )


def test_file_provenance_requires_the_original_authenticator_at_write_and_load(
    tmp_path: Path,
) -> None:
    episode_key = "signed-file"
    contract = _contract()
    classified = _file_classified(tmp_path, episode_key=episode_key)
    wrong_auth = ProvenanceAuthenticator(b"x" * 32, issuer="episodes-test")

    with pytest.raises(EpisodeIntegrityError, match="authenticator and contract"):
        write_episode_envelope(tmp_path / "missing", episode_key, classified)
    with pytest.raises(EpisodeIntegrityError, match="attestation") as raised:
        write_episode_envelope(
            tmp_path / "wrong", episode_key, classified, contract=contract, authenticator=wrong_auth
        )
    assert AUTH_KEY.decode("ascii") not in str(raised.value)

    root = tmp_path / "valid"
    envelope_path = write_episode_envelope(
        root, episode_key, classified, contract=contract, authenticator=AUTH
    )
    assert AUTH_KEY not in envelope_path.read_bytes()
    with pytest.raises(EpisodeIntegrityError, match="authenticator and contract"):
        load_episode_envelopes(root)
    with pytest.raises(EpisodeIntegrityError, match="attestation"):
        load_episode_envelopes(root, contract=contract, authenticator=wrong_auth)
    assert load_episode_envelopes(root, contract=contract, authenticator=AUTH)[0].episode_key == episode_key


@pytest.mark.parametrize("field", ["outcome", "provenance"])
def test_writer_rejects_file_provenance_or_outcome_forged_after_attestation(
    tmp_path: Path, field: str
) -> None:
    episode_key = "forged-file"
    classified = _file_classified(tmp_path, episode_key=episode_key)
    forged = (
        replace(classified, outcome=Outcome.FAILURE)
        if field == "outcome"
        else replace(classified, provenance={**classified.provenance, "basename": "forged.json"})
    )

    with pytest.raises(EpisodeIntegrityError, match="attestation"):
        write_episode_envelope(
            tmp_path / "envelopes",
            episode_key,
            forged,
            contract=_contract(),
            authenticator=AUTH,
        )


def test_file_attestation_cannot_be_replayed_under_another_campaign_contract(tmp_path: Path) -> None:
    episode_key = "replayed-file"
    contract = _contract()
    classified = _file_classified(tmp_path, episode_key=episode_key)
    root = tmp_path / "envelopes"
    write_episode_envelope(root, episode_key, classified, contract=contract, authenticator=AUTH)
    other_contract = replace(contract, campaign_id="campaign-002")

    with pytest.raises(EpisodeIntegrityError, match="attestation"):
        load_episode_envelopes(root, contract=other_contract, authenticator=AUTH)


def test_mapping_provenance_must_not_carry_an_attestation(tmp_path: Path) -> None:
    path = write_episode_envelope(tmp_path, "mapping-file", _classified("closed-success.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance_attestation"] = {"issuer": "episodes-test", "key_id": "0" * 64, "mac": "0" * 64}
    payload["canonical_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "canonical_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EpisodeIntegrityError, match="mapping provenance"):
        load_episode_envelopes(tmp_path)


def test_unreadable_file_provenance_is_signed_and_round_trips_as_quarantine(tmp_path: Path) -> None:
    from b1k_rollout.outcomes import classify_outcome_file

    episode_key = "unreadable-file"
    contract = _contract()
    classified = classify_outcome_file(
        tmp_path / "missing.json",
        task_manifest=MANIFEST,
        episode_key=episode_key,
        contract=contract,
        authenticator=AUTH,
    )
    root = tmp_path / "envelopes"
    write_episode_envelope(root, episode_key, classified, contract=contract, authenticator=AUTH)

    loaded = load_episode_envelopes(root, contract=contract, authenticator=AUTH)
    assert loaded[0].outcome is Outcome.QUARANTINE
    assert loaded[0].provenance["disposition"] == "unreadable"


def test_writer_uses_complete_file_after_staging_and_preserves_raw_quarantine(
    tmp_path: Path,
) -> None:
    classified = _classified("incomplete.json")

    path = write_episode_envelope(tmp_path, "episode-003", classified)

    assert path == tmp_path / "episode-003.json"
    assert not (tmp_path / "episode-003.json.incomplete").exists()
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["schema_version"] == 2
    assert envelope["outcome"] == "quarantine"
    assert envelope["raw_evidence"] == classified.raw_evidence
    assert len(envelope["canonical_sha256"]) == 64


def test_writer_and_loader_preserve_exact_valid_byte_evidence_and_verify_its_hash(
    tmp_path: Path,
) -> None:
    raw = (FIXTURES / "closed-success.json").read_bytes()
    classified = classify_outcome(raw, task_manifest=MANIFEST)
    path = write_episode_envelope(tmp_path, "episode-bytes", classified)

    loaded = load_episode_envelopes(tmp_path)

    assert loaded[0].raw_evidence == raw
    assert loaded[0].raw_evidence_sha256 == hashlib.sha256(raw).hexdigest()
    assert json.loads(path.read_text(encoding="utf-8"))["raw_evidence_encoding"] == "bytes"


def test_loader_enforces_the_canonical_evaluator_identity_tuple(tmp_path: Path) -> None:
    first = write_episode_envelope(tmp_path, "episode-identity-1", _classified("closed-success.json"))
    evidence = json.loads((FIXTURES / "closed-success.json").read_text(encoding="utf-8"))
    evidence["episode_id"] = "episode-identity-other"
    second = write_episode_envelope(
        tmp_path,
        "episode-identity-2",
        classify_outcome(evidence, task_manifest=MANIFEST),
    )

    with pytest.raises(EpisodeIntegrityError, match="evaluator identity"):
        load_episode_envelopes(tmp_path)
    assert first.exists() and second.exists()


def test_loader_rejects_an_evaluator_identity_tuple_forged_away_from_raw_evidence(
    tmp_path: Path,
) -> None:
    path = write_episode_envelope(tmp_path, "episode-identity-forged", _classified("closed-success.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evaluator_identity"]["instance_id"] = 302
    payload["canonical_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "canonical_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EpisodeIntegrityError, match="evaluator identity"):
        load_episode_envelopes(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("outcome", "failure"),
        ("final_q_scores", {"final": 0.5}),
        ("episode_id", "episode-forged"),
        ("rollout_id", 1),
    ],
)
def test_loader_rejects_terminal_fields_forged_away_from_retained_raw_evidence(
    tmp_path: Path, field: str, value: object
) -> None:
    path = write_episode_envelope(tmp_path, "episode-terminal-forged", _classified("closed-success.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    payload["canonical_sha256"] = canonical_json_sha256(
        {key: nested for key, nested in payload.items() if key != "canonical_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EpisodeIntegrityError, match="terminal envelope"):
        load_episode_envelopes(tmp_path)


def test_mapping_evidence_with_upstream_positive_infinity_quarantines_and_round_trips(
    tmp_path: Path,
) -> None:
    evidence = json.loads((FIXTURES / "closed-success.json").read_text(encoding="utf-8"))
    evidence["time"]["normalized_time"] = float("inf")
    classified = classify_outcome(evidence, task_manifest=MANIFEST)

    path = write_episode_envelope(tmp_path, "episode-mapping-infinity", classified)
    loaded = load_episode_envelopes(tmp_path)

    assert classified.outcome.value == "quarantine"
    assert path.exists()
    assert loaded[0].raw_evidence == evidence


def test_mapping_infinity_can_be_authenticated_without_strict_json_canonicalization(
    tmp_path: Path,
) -> None:
    evidence = json.loads((FIXTURES / "closed-success.json").read_text(encoding="utf-8"))
    evidence["time"]["normalized_time"] = float("inf")
    contract = _contract()

    write_episode_envelope(
        tmp_path,
        "episode-mapping-infinity-authenticated",
        classify_outcome(evidence, task_manifest=MANIFEST),
        contract=contract,
        authenticator=AUTH,
    )
    loaded = load_episode_envelopes(tmp_path, contract=contract, authenticator=AUTH)

    assert loaded[0].provenance_attestation is not None
    assert loaded[0].raw_evidence["time"]["normalized_time"] == float("inf")


def test_quarantine_with_invalid_extracted_ids_is_durably_writable(tmp_path: Path) -> None:
    evidence = json.loads((FIXTURES / "closed-success.json").read_text(encoding="utf-8"))
    evidence["episode_id"] = "../invalid"
    evidence["rollout_id"] = -1

    path = write_episode_envelope(
        tmp_path, "episode-quarantine-ids", classify_outcome(evidence, task_manifest=MANIFEST)
    )

    loaded = load_episode_envelopes(tmp_path)
    assert path.exists()
    assert loaded[0].episode_id is None
    assert loaded[0].rollout_id is None


def test_loader_rejects_a_forged_raw_evidence_hash_even_when_the_envelope_hash_is_recomputed(
    tmp_path: Path,
) -> None:
    path = write_episode_envelope(tmp_path, "episode-forged", _classified("closed-success.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["raw_evidence_sha256"] = "0" * 64
    payload["canonical_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "canonical_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EpisodeIntegrityError, match="raw evidence hash"):
        load_episode_envelopes(tmp_path)


def test_loader_accepts_reused_official_rollout_id_for_different_task_instances(
    tmp_path: Path,
) -> None:
    first = write_episode_envelope(tmp_path, "episode-rollout-1", _classified("closed-success.json"))
    evidence = json.loads((FIXTURES / "closed-failure.json").read_text(encoding="utf-8"))
    evidence["task"] = MANIFEST["tasks"][1]["task_name"]
    evidence["instance_id"] = 301
    evidence["instance_index"] = 0
    evidence["rollout_id"] = 0
    second = write_episode_envelope(
        tmp_path,
        "episode-rollout-2",
        classify_outcome(evidence, task_manifest=MANIFEST),
    )

    loaded = load_episode_envelopes(tmp_path)

    assert first.exists() and second.exists()
    assert [envelope.rollout_id for envelope in loaded] == [0, 0]


def test_loader_rejects_incomplete_files_duplicate_keys_and_conflicting_ids(
    tmp_path: Path,
) -> None:
    success = _classified("closed-success.json")
    failure = _classified("closed-failure.json")
    first = write_episode_envelope(tmp_path, "episode-key-1", success)
    second = write_episode_envelope(tmp_path, "episode-key-2", failure)

    second_payload = json.loads(second.read_text(encoding="utf-8"))
    second_payload["episode_id"] = "episode-001"
    second_payload["raw_evidence"]["episode_id"] = "episode-001"
    second_payload["raw_evidence_sha256"] = raw_evidence_sha256(second_payload["raw_evidence"])
    second_payload["provenance"]["raw_evidence_sha256"] = second_payload["raw_evidence_sha256"]
    second_payload["canonical_sha256"] = canonical_json_sha256(
        {key: value for key, value in second_payload.items() if key != "canonical_sha256"}
    )
    second.write_text(json.dumps(second_payload), encoding="utf-8")
    with pytest.raises(EpisodeIntegrityError, match="duplicate episode id"):
        load_episode_envelopes(tmp_path)

    second_payload["episode_id"] = "episode-002"
    second.write_text(json.dumps(second_payload), encoding="utf-8")
    (tmp_path / "episode-key-1.json.incomplete").write_text("{}", encoding="utf-8")
    with pytest.raises(EpisodeIntegrityError, match="incomplete"):
        load_episode_envelopes(tmp_path)

    with pytest.raises(EpisodeIntegrityError, match="already exists"):
        write_episode_envelope(tmp_path, "episode-key-1", success)
    assert first.exists()


@pytest.mark.parametrize("episode_key", ["../escape", "nested/key", ".", "lehome-episode"])
def test_writer_fails_closed_on_unsafe_episode_keys(tmp_path: Path, episode_key: str) -> None:
    with pytest.raises(EpisodeIntegrityError):
        write_episode_envelope(tmp_path, episode_key, _classified("closed-success.json"))


def test_writer_fails_closed_on_a_symlinked_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "episodes"
    root.symlink_to(target, target_is_directory=True)

    with pytest.raises(EpisodeIntegrityError, match="symlink"):
        write_episode_envelope(root, "episode-001", _classified("closed-success.json"))


def test_writer_rejects_a_root_beneath_a_symlinked_parent(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(EpisodeIntegrityError, match="symlink"):
        write_episode_envelope(alias / "episodes", "episode-001", _classified("closed-success.json"))


def test_writer_rejects_an_explicit_root_path_with_parent_traversal(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()

    with pytest.raises(EpisodeIntegrityError, match="traversal"):
        write_episode_envelope(nested / "..", "episode-001", _classified("closed-success.json"))


def test_artifact_verification_rejects_nested_symlinks_even_when_content_hash_matches(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    payload = b"raw rollout artifact"
    (outside / "video.mp4").write_bytes(payload)
    (tmp_path / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(EpisodeIntegrityError, match="symlink"):
        verify_artifact_hashes(
            tmp_path,
            {"nested/video.mp4": hashlib.sha256(payload).hexdigest()},
        )


def test_copy_verified_artifact_streams_large_content_through_one_descriptor(
    tmp_path: Path,
) -> None:
    payload = b"artifact" * (512 * 1024)
    source = tmp_path / "artifacts"
    source.mkdir()
    (source / "trajectory.bin").write_bytes(payload)

    copied = copy_verified_artifact(
        source,
        "trajectory.bin",
        tmp_path / "release" / "trajectory.bin",
        hashlib.sha256(payload).hexdigest(),
    )

    assert copied.size == len(payload)
    assert copied.sha256 == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / "release" / "trajectory.bin").stat().st_size == len(payload)


def test_copy_verified_artifact_keeps_the_opened_parent_descriptor_after_a_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    nested = root / "nested"
    nested.mkdir(parents=True)
    expected = b"verified before parent swap"
    (nested / "trajectory.bin").write_bytes(expected)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "trajectory.bin").write_bytes(b"attacker-controlled")
    original_open = episodes.os.open
    swapped = False

    def swap_after_open(path: str, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        nonlocal swapped
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "nested" and not swapped:
            swapped = True
            nested.rename(root / "nested-original")
            nested.symlink_to(outside, target_is_directory=True)
        return fd

    monkeypatch.setattr(episodes.os, "open", swap_after_open)
    copied = copy_verified_artifact(
        root,
        "nested/trajectory.bin",
        tmp_path / "release" / "trajectory.bin",
        hashlib.sha256(expected).hexdigest(),
    )

    assert copied.sha256 == hashlib.sha256(expected).hexdigest()
    assert (tmp_path / "release" / "trajectory.bin").read_bytes() == expected


def test_writer_durably_creates_a_missing_final_root_and_fsyncs_payload_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "new-episode-root"
    fsync_calls: list[int] = []
    original_fsync = episodes.os.fsync

    def record_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        original_fsync(fd)

    monkeypatch.setattr(episodes.os, "fsync", record_fsync)
    path = write_episode_envelope(root, "episode-durable", _classified("closed-success.json"))

    assert path.exists()
    assert root.is_dir()
    assert len(fsync_calls) >= 4  # parent creation, staged payload, pre/post-rename directory
