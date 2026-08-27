"""Production finalizer wiring for persistent worker output paths."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest

# This suite only exercises ledger/sealing wiring.  The local CPU-only
# checkout lacks the optional Arrow encoder, so provide the import seam that
# ``lehome_train.flywheel.__init__`` eagerly reaches; no fake is used by a
# tested code path here.
try:
    import pyarrow  # noqa: F401
except ModuleNotFoundError:
    pyarrow = types.ModuleType("pyarrow")
    parquet = types.ModuleType("pyarrow.parquet")
    pyarrow.parquet = parquet
    sys.modules["pyarrow"] = pyarrow
    sys.modules["pyarrow.parquet"] = parquet

from lehome.flywheel.task_ledger import TaskLedger
from lehome.flywheel.hub_sync import HubSyncError
import scripts.run_groot_artifact_sync as artifact_sync
from scripts.run_groot_artifact_sync import (
    _load_runtime_token,
    run_evaluation_batch_uploader_once,
    run_evaluation_uploader_once,
    run_finalizer_once,
    run_sealer_once,
)


def test_finalizer_discovers_nested_worker_artifact_from_the_ledger(tmp_path: Path) -> None:
    matrix = [{"garment": "Top_Short_Seen_9", "seed": 313, "category": "top_short"}]
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    database = tmp_path / "ledger.sqlite3"
    run_root = tmp_path / "campaign"

    ledger = TaskLedger(database, attempt_matrix=matrix, max_attempts=1, target_accepted=1)
    lease = ledger.lease_next("worker-1", lease_duration_ns=10**18)
    assert lease is not None
    output_dir = (
        run_root
        / "worker-1"
        / "session-1"
        / lease.attempt.attempt_id
        / lease.lease_id
        / "generation-1"
    )
    (output_dir / "videos").mkdir(parents=True)
    (output_dir / "videos" / "top.mp4").write_bytes(b"video")
    (output_dir / "worker-receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attempt_id": lease.attempt.attempt_id,
                "lease_id": lease.lease_id,
                "worker_id": "worker-1",
                "outcome": {"success": True, "metrics": [{"success": True}]},
            }
        ),
        encoding="utf-8",
    )
    ledger.record_terminal("worker-1", lease.attempt.attempt_id, lease.lease_id, str(output_dir))
    ledger.close()

    finalized = run_finalizer_once(
        database=database,
        attempt_matrix=matrix_path,
        run_root=run_root,
        max_pending_items=16,
        max_pending_bytes=2**30,
        max_attempts=1,
        target_accepted=1,
    )

    assert finalized == 1
    accepted = run_root / "accepted" / lease.attempt.attempt_id
    assert (accepted / "worker-receipt.json").is_file()
    assert (accepted / "SHA256SUMS.json").is_file()
    reopened = TaskLedger(database, attempt_matrix=matrix, max_attempts=1, target_accepted=1)
    assert reopened.status(lease.attempt.attempt_id) == "accepted"
    reopened.close()


def test_artifact_sync_cannot_admit_an_arbitrary_v2_list(tmp_path: Path) -> None:
    matrix = tmp_path / "arbitrary-v2.json"
    matrix.write_text(json.dumps([{"recovery_kind": "controlled_success_recovery_snapshot_v3"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="controlled smoke"):
        artifact_sync._load_attempt_matrix(matrix)


def test_uploader_reads_token_only_from_private_runtime_file(tmp_path: Path) -> None:
    token_file = tmp_path / "hf-token"
    token_file.write_text("hf_runtime_secret\n", encoding="utf-8")
    token_file.chmod(0o600)

    assert _load_runtime_token(token_file=token_file, environ={}) == "hf_runtime_secret"

    token_file.chmod(0o644)
    try:
        _load_runtime_token(token_file=token_file, environ={})
    except RuntimeError as error:
        assert "private regular file" in str(error)
    else:
        raise AssertionError("world-readable Hub token file was accepted")


def test_uploader_pass_publishes_at_most_one_pending_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted = tmp_path / "accepted"
    (accepted / "attempt-a").mkdir(parents=True)
    (accepted / "attempt-b").mkdir()
    token_file = tmp_path / "token"
    token_file.write_text("hf_runtime_secret", encoding="utf-8")
    token_file.chmod(0o600)
    calls: list[str] = []

    class FakeDaemon:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def sync_episode(self, attempt_id: str, _path: Path) -> None:
            calls.append(attempt_id)

    monkeypatch.setattr(artifact_sync, "HubSyncDaemon", FakeDaemon)
    monkeypatch.setattr(artifact_sync, "HuggingFaceHubTransport", lambda: object())

    synced = artifact_sync.run_uploader_once(
        accepted_root=accepted,
        receipts_root=tmp_path / "receipts",
        readback_root=tmp_path / "readback",
        repository="ryanjin333/lehome-groot-n17-rollouts",
        round_id="round-1",
        revision="main",
        token_file=token_file,
    )

    assert synced == 1
    assert calls == ["attempt-a"]


def test_uploader_pass_stops_on_first_hub_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted = tmp_path / "accepted"
    (accepted / "attempt-a").mkdir(parents=True)
    (accepted / "attempt-b").mkdir()
    token_file = tmp_path / "token"
    token_file.write_text("hf_runtime_secret", encoding="utf-8")
    token_file.chmod(0o600)
    calls: list[str] = []

    class FailingDaemon:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def sync_episode(self, attempt_id: str, _path: Path) -> None:
            calls.append(attempt_id)
            raise HubSyncError("rate limited")

    monkeypatch.setattr(artifact_sync, "HubSyncDaemon", FailingDaemon)
    monkeypatch.setattr(artifact_sync, "HuggingFaceHubTransport", lambda: object())

    with pytest.raises(HubSyncError, match="rate limited"):
        artifact_sync.run_uploader_once(
            accepted_root=accepted,
            receipts_root=tmp_path / "receipts",
            readback_root=tmp_path / "readback",
            repository="ryanjin333/lehome-groot-n17-rollouts",
            round_id="round-1",
            revision="main",
            token_file=token_file,
        )

    assert calls == ["attempt-a"]


def test_evaluation_uploader_publishes_success_and_failure_without_accepted_bundle_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_root = tmp_path / "evaluation-terminal"
    receipts_root = tmp_path / "hf-sync-receipts"
    for attempt_id, success in (("attempt-success", True), ("attempt-failure", False)):
        episode = terminal_root / attempt_id
        (episode / "videos").mkdir(parents=True)
        (episode / "videos" / "top.mp4").write_bytes(b"video")
        (episode / "worker-receipt.json").write_text(json.dumps({
            "schema_version": 1,
            "attempt_id": attempt_id,
            "outcome": {"success": success},
        }), encoding="utf-8")
        (episode / "SHA256SUMS.json").write_text("{}\n", encoding="utf-8")
    token_file = tmp_path / "token"
    token_file.write_text("hf_runtime_secret", encoding="utf-8")
    token_file.chmod(0o600)
    calls: list[tuple[str, bool]] = []

    class FakeDaemon:
        def __init__(self, **kwargs: object) -> None:
            self.receipts_root = Path(kwargs["receipts_root"])
            self.receipts_root.mkdir(parents=True, exist_ok=True)

        def sync_episode(self, attempt_id: str, path: Path) -> None:
            receipt = json.loads((path / "worker-receipt.json").read_text(encoding="utf-8"))
            calls.append((attempt_id, receipt["outcome"]["success"]))
            (self.receipts_root / f"{attempt_id}.sync.json").write_text(
                json.dumps({"attempt_id": attempt_id, "readback_verified": True}),
                encoding="utf-8",
            )

    monkeypatch.setattr(artifact_sync, "HubSyncDaemon", FakeDaemon)
    monkeypatch.setattr(artifact_sync, "HuggingFaceHubTransport", lambda: object())
    common = dict(
        terminal_root=terminal_root,
        receipts_root=receipts_root,
        readback_root=tmp_path / "readback",
        repository="owner/evaluation-evidence",
        round_id="final-unseen80",
        revision="main",
        token_file=token_file,
    )
    assert run_evaluation_uploader_once(**common) == 1
    assert run_evaluation_uploader_once(**common) == 1
    assert calls == [("attempt-failure", False), ("attempt-success", True)]
    assert len(tuple(receipts_root.glob("*.sync.json"))) == 2
    assert not (tmp_path / "accepted").exists()

    accepted_root = tmp_path / "accepted"
    accepted_root.mkdir()
    with pytest.raises(ValueError, match="misclassified"):
        run_evaluation_uploader_once(**{**common, "terminal_root": accepted_root})


def test_evaluation_batch_uploader_uses_one_commit_and_one_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_root = tmp_path / "evaluation-terminal"
    receipts_root = tmp_path / "hf-sync-receipts"
    for attempt_id in ("attempt-a", "attempt-b"):
        episode = terminal_root / attempt_id
        (episode / "videos").mkdir(parents=True)
        (episode / "videos" / "top.mp4").write_bytes(attempt_id.encode("ascii"))
        (episode / "worker-receipt.json").write_text(
            json.dumps({"attempt_id": attempt_id}), encoding="utf-8",
        )
        (episode / "SHA256SUMS.json").write_text("{}\n", encoding="utf-8")
    token_file = tmp_path / "token"
    token_file.write_text("hf_runtime_secret", encoding="utf-8")
    token_file.chmod(0o600)
    calls: list[str] = []

    class Entry:
        def __init__(self, relative_path: str) -> None:
            self.relative_path = relative_path
            self.entry_type = "file"

    class FakeTransport:
        def upload_files(self, **kwargs: object) -> str:
            calls.append("upload")
            assert kwargs["source"] == terminal_root
            assert kwargs["remote_prefix"] == "rollout-rounds/evaluation-round"
            self.entries = tuple(kwargs["entries"])
            return "c" * 40

        def list_tree(self, **kwargs: object) -> tuple[Entry, ...]:
            calls.append("list")
            prefix = str(kwargs["remote_prefix"])
            return tuple(Entry(f"{prefix}/{entry.relative_path}") for entry in self.entries)

        def download_files(self, **kwargs: object) -> str:
            calls.append("download")
            destination = Path(kwargs["destination"])
            for relative in kwargs["relative_paths"]:
                source = terminal_root / Path(str(relative)).relative_to(
                    "rollout-rounds/evaluation-round"
                )
                target = destination / str(relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            return "c" * 40

    transport = FakeTransport()
    monkeypatch.setattr(artifact_sync, "HuggingFaceHubTransport", lambda: transport)

    synced = run_evaluation_batch_uploader_once(
        terminal_root=terminal_root,
        receipts_root=receipts_root,
        readback_root=tmp_path / "hf-readback",
        repository="owner/evaluation-evidence",
        round_id="evaluation-round",
        revision="main",
        token_file=token_file,
    )

    assert synced == 2
    assert calls == ["upload", "list", "download"]
    receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(receipts_root.glob("*.sync.json"))
    ]
    assert {receipt["attempt_id"] for receipt in receipts} == {"attempt-a", "attempt-b"}
    assert {receipt["immutable_revision"] for receipt in receipts} == {"c" * 40}
    assert all(receipt["readback_verified"] is True for receipt in receipts)


def test_sealer_closes_the_exact_accepted_hub_durable_round(tmp_path: Path) -> None:
    matrix = [{"garment": "Top_Short_Seen_9", "seed": 313, "category": "top_short"}]
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    database = tmp_path / "ledger.sqlite3"
    ledger = TaskLedger(database, attempt_matrix=matrix, max_attempts=1, target_accepted=1)
    lease = ledger.lease_next("worker-1", lease_duration_ns=10**18)
    assert lease is not None
    ledger.record_terminal("worker-1", lease.attempt.attempt_id, lease.lease_id, "/durable/raw")
    ledger.validate_terminal(lease.attempt.attempt_id, "accepted", artifact_id="accepted-artifact")
    ledger.close()

    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / f"{lease.attempt.attempt_id}.sync.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "round_id": "success-replay-round-1",
                "attempt_id": lease.attempt.attempt_id,
                "repository": "ryanjin333/lehome-groot-n17-rollouts",
                "remote_prefix": f"rollout-rounds/success-replay-round-1/{lease.attempt.attempt_id}",
                "episode_sha256": "a" * 64,
                "immutable_revision": "b" * 40,
                "readback_verified": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    seal = tmp_path / "success-replay-round-1.strict.seal.json"

    result = run_sealer_once(
        database=database,
        attempt_matrix=matrix_path,
        receipts_root=receipts,
        round_id="success-replay-round-1",
        seal_receipt_path=seal,
        max_attempts=1,
        target_accepted=1,
    )

    assert result.episode_count == 1
    document = json.loads(seal.read_text(encoding="utf-8"))
    assert document["episode_sha256s"] == {lease.attempt.attempt_id: "a" * 64}
    assert document["readback_verified"] is True


def test_controlled_recovery_shortfall_never_publishes_a_seal(tmp_path: Path) -> None:
    caps = {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}
    reset, annotations, continuation = tmp_path / "reset.json", tmp_path / "annotations.jsonl", tmp_path / "continuation.json"
    reset.write_text("{}", encoding="utf-8"); annotations.write_text("", encoding="utf-8"); continuation.write_text("{}", encoding="utf-8")
    categories = ["pant_long"] * 4 + ["top_long"] + ["top_short"] * 3
    rows = [
        {
            "attempt_id": f"controlled-{index}", "trial_id": f"controlled-{index}",
            "category": category, "category_acceptance_cap": caps[category],
            "strategy": "canonical", "recovery_kind": "controlled_success_recovery_snapshot_v3",
            "controlled_matrix_sha256": "a" * 64, "perturbation_seed": 71_000 + index,
            "perturbation_fingerprint": f"{index + 100:064x}",
            "source_state_perturbation_fingerprint": f"{index + 200:064x}",
            "source_state_fingerprint": f"{index + 300:064x}", "source_round_id": "round",
            "source_episode_id": f"episode-{index}", "source_episode_digest": f"{index + 400:064x}",
                "source_seed": 50110, "source_continuation_state": [float(index)] * 12,
                "source_snapshot_schema_version": 2, "source_snapshot_authority": "physx_cloth_view_world_v1",
                "source_only_envelope": False,
                "source_immutable_revision": "a" * 40, "source_reset_sha256": "a" * 64,
            "source_annotations_sha256": "b" * 64, "source_continuation_snapshot_sha256": "c" * 64,
            "prefix_stop": 16, "source_first_success_step": 19,
            "perturbation_profile": {"cloth_displacement_m": 0.002, "cloth_velocity_mps": 0.01, "gripper_offset_rad": 0.02},
            "source_reset": str(reset), "source_annotations": str(annotations), "source_continuation_snapshot": str(continuation),
        }
        for index, category in enumerate(categories)
    ]
    matrix_path = tmp_path / "materialization.json"
    matrix_path.write_text(json.dumps({"schema_version": 3, "kind": "controlled_success_recovery_materialization_v3", "matrix_sha256": "a" * 64, "target_accepted": 8, "category_acceptance_caps": caps, "rows": rows}), encoding="utf-8")
    database = tmp_path / "ledger.sqlite3"
    ledger = TaskLedger(database, attempt_matrix=rows, max_attempts=8, target_accepted=8)
    for worker_index in range(7):
        lease = ledger.lease_next(f"worker-{worker_index}", lease_duration_ns=10**18)
        assert lease is not None
        ledger.record_terminal(f"worker-{worker_index}", lease.attempt.attempt_id, lease.lease_id, f"/raw/{worker_index}")
        ledger.validate_terminal(lease.attempt.attempt_id, "accepted", artifact_id=f"accepted-{worker_index}")
    ledger.close()
    seal = tmp_path / "must-not-exist.json"
    with pytest.raises(RuntimeError, match="short of its immutable acceptance caps"):
        run_sealer_once(database=database, attempt_matrix=matrix_path, receipts_root=tmp_path / "receipts", round_id="controlled-recovery-v1-round-1", seal_receipt_path=seal, max_attempts=8, target_accepted=8)
    assert not seal.exists()
