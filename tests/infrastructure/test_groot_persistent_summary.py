from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lehome.flywheel.simple_curriculum import build_calibration_rows
from lehome.flywheel.hub_sync import HubSyncDaemon


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "summarize_groot_persistent_evaluation.py"


def _module():
    spec = importlib.util.spec_from_file_location("persistent_summary_first100", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _orchestrator_module():
    path = ROOT / "scripts" / "run_simple_curriculum_collection.py"
    spec = importlib.util.spec_from_file_location("simple_curriculum_orchestrator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_report_preserves_legacy_fields_and_adds_first_hundred_gate_metrics(tmp_path: Path) -> None:
    module = _module()
    report = module._augment_first_hundred_metrics({
        "episodes": 100,
        "official_successes": 5,
        "gate_trials": [
            {
                "assignment_id": f"attempt-{index}",
                "terminal_event": "accepted" if index < 5 else "rejected",
                "identity": {"code_revision": "c" * 40, "asset_revision": "a" * 40, "simulator_version": "5.1.0.0"},
                "provenance": {
                    "policy_repo": "owner/policy", "policy_revision": "e" * 40,
                    "policy_step": 12000, "policy_artifact_sha256": "b" * 64,
                    "image_identity": "sha256:" + "d" * 64,
                    "simulator_device": "cpu", "cloth_device": "cpu",
                    "renderer_device": "cuda:0", "camera_device": "cuda:0", "policy_device": "cuda:0",
                },
                "fidelity": {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False, "monitor_active": True, "monitor_observed": True},
            }
            for index in range(100)
        ],
        "infrastructure_invalid_executions": 2,
    })

    assert report["episodes"] == 100
    assert report["official_successes"] == 5
    assert report["valid_outcomes"] == 100
    assert report["infrastructure_invalid_executions"] == 2
    assert report["execution_count"] == 102
    assert report["fresh_assignment_ids"] == sorted(f"attempt-{index}" for index in range(100))
    assert len(report["runtime_identities"]) == 1
    assert len(report["runtime_identities"][0]) == 64


def test_simple_summary_rejects_a_symlinked_campaign_root_ancestor(tmp_path: Path) -> None:
    summary = _module()
    materialized_parent = tmp_path / "materialized"
    materialized_parent.mkdir()
    root, matrix, _rows, _ledger_ids = _simple_campaign(materialized_parent)
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(materialized_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="ancestor"):
        summary.build_report(
            campaign_root=alias_parent / root.name,
            matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
            candidate_key="original_baseline", **POLICY,
        )


POLICY = {
    "policy_repo": "owner/policy", "policy_revision": "e" * 40,
    "policy_step": 12000, "policy_artifact_sha256": "b" * 64,
}


def _catalog() -> dict[str, list[str]]:
    return {
        "top_long": [f"Top_Long_Seen_{index}" for index in range(10)],
        "top_short": [f"Top_Short_Seen_{index}" for index in range(10)],
        "pant_long": [f"Pant_Long_Seen_{index}" for index in range(10)],
        "pant_short": [f"Pant_Short_Seen_{index}" for index in range(10)],
    }


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


class _ReceiptTransport:
    """Minimal in-memory transport used to exercise the real receipt producer."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, bytes]] = {}

    def upload_files(self, *, source, entries, remote_prefix, **_kwargs):
        bucket = self.store.setdefault(str(remote_prefix), {})
        for entry in entries:
            bucket[entry.relative_path] = (Path(source) / entry.relative_path).read_bytes()
        return "c" * 40

    def download_files(self, *, destination, relative_paths, remote_prefix, **_kwargs):
        bucket = self.store[str(remote_prefix)]
        for relative in relative_paths:
            target = Path(destination) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bucket[relative])
        return "c" * 40

    def list_tree(self, *, remote_prefix, **_kwargs):
        return tuple(
            SimpleNamespace(relative_path=f"{remote_prefix}/{relative}", entry_type="file")
            for relative in self.store[str(remote_prefix)]
        )


def _simple_campaign(tmp_path: Path, *, rows: list[dict[str, object]] | None = None, campaign_root: Path | None = None, matrix_path: Path | None = None, policy: dict[str, object] | None = None, campaign_round_id: str | None = None, campaign_run_id: str | None = None, missing_receipt: int | None = None, malformed_receipt: int | None = None, retry_then_valid: int | None = None, contradictory_safety: int | None = None, all_success: bool = False, evaluation_terminal: bool = True, episode_mutator=None, receipt_mutator=None, session_for_index=None) -> tuple[Path, Path, list[dict[str, object]], dict[str, str]]:
    from lehome.flywheel.artifact_queue import ArtifactFinalizationQueue
    from lehome.flywheel.isaac_recorder import _identity_payload
    from lehome.flywheel.models import EpisodeIdentity
    from lehome.flywheel.simple_curriculum import build_calibration_rows
    from lehome.flywheel.task_ledger import TaskLedger

    rows = rows or build_calibration_rows(_catalog(), seed_base=900)[:100]
    root = campaign_root or tmp_path / "campaign"; root.mkdir(parents=True, exist_ok=False)
    matrix = matrix_path or tmp_path / "matrix.json"; matrix.parent.mkdir(parents=True, exist_ok=True); matrix.write_bytes(_canonical(rows))
    policy = policy or POLICY
    if (campaign_round_id is None) != (campaign_run_id is None):
        raise ValueError("test campaign provenance requires both run and round ids")
    target = len(rows)
    ledger = TaskLedger(
        root / "ledger.sqlite3", attempt_matrix=rows,
        max_attempts=target + int(retry_then_valid is not None), target_accepted=target,
        completion_metric="terminal_outcomes",
    )
    finalizer = ArtifactFinalizationQueue(
        run_root=root, ledger=ledger, max_pending_items=1, max_pending_bytes=1 << 30,
        evaluation_only=evaluation_terminal,
    )
    ledger_ids: dict[str, str] = {}
    for index, row in enumerate(rows):
        lease = ledger.lease_next("worker", lease_duration_ns=1_000_000_000)
        assert lease is not None
        if index == retry_then_valid:
            ledger.record_interrupted("worker", lease.attempt.attempt_id, lease.lease_id, "test_retry")
            lease = ledger.lease_next("worker", lease_duration_ns=1_000_000_000)
            assert lease is not None
        ledger_id = lease.attempt.attempt_id
        # A retried lease can be a different logical schedule row than this
        # fixture loop index.  Production workers must bind their episode to
        # the durable leased assignment, never to a local loop counter.
        assignment = dict(lease.attempt.assignment)
        session_id = session_for_index(index) if session_for_index is not None else "session"
        output = root / "worker" / session_id / ledger_id / lease.lease_id / f"generation-{index + 1}"
        raw = output / "raw" / ledger_id; raw.mkdir(parents=True)
        videos = output / "videos"; videos.mkdir(); (videos / "top.mp4").write_bytes(b"video")
        ledger_ids[str(assignment["attempt_id"])] = ledger_id
        identity = EpisodeIdentity(
            episode_id=ledger_id,
            policy_repo=str(policy["policy_repo"]),
            policy_revision=str(policy["policy_revision"]),
            policy_step=int(policy["policy_step"]),
            code_revision="c" * 40,
            asset_revision="a" * 40,
            simulator_version="5.1.0.0",
            garment_name=str(assignment["garment_name"]),
            category=str(assignment["category"]),
            release_stage=str(assignment["release_stage"]),
            seed=int(assignment["seed"]),
            instruction="fold the garment",
            strategy="canonical",
            campaign_round_id=campaign_round_id,
            campaign_run_id=campaign_run_id,
        )
        # The regular summary fixtures only need the canonical identity
        # producer.  The 1,000-terminal producer-path integration below also
        # supplies complete recorder-authored episode documents.
        recorded_identity = _identity_payload(identity)
        episode = {
            "episode_id": ledger_id,
            "identity": recorded_identity,
            "provenance": {"policy_artifact_sha256": policy["policy_artifact_sha256"], "simulator_device": "cpu", "policy_device": "cuda:0", "image_identity": "sha256:" + "d" * 64},
            "outcome": "success" if all_success or index < 5 else "timeout", "accepted_success": all_success or index < 5,
            "safety_failure": index == contradictory_safety,
            "fidelity": {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False, "monitor_active": True, "monitor_observed": True},
        }
        if episode_mutator is not None:
            episode_mutator(episode, index)
        (raw / "episode.json").write_bytes(_canonical(episode))
        receipt = {
            "schema_version": 1, "attempt_id": ledger_id, "lease_id": lease.lease_id,
            "worker_id": "worker", "session_id": session_id, "seed": assignment["seed"], "garment": assignment["garment_name"],
            "episode_generation": index + 1, "output_dir": str(output), "action_horizon": 250,
            "outcome": {"success": all_success or index < 5, "metrics": []}, "simulation_device": "cpu", "cloth_device": "cpu",
            "renderer_device": "cuda:0", "camera_device": "cuda:0", "policy_device": "cuda:0",
            "runtime": {"simulation_device": "cpu", "cloth_device": "cpu", "renderer_device": "cuda:0", "camera_device": "cuda:0", "policy_device": "cuda:0"},
        }
        if receipt_mutator is not None:
            receipt_mutator(receipt, index)
        receipt_path = output / "worker-receipt.json"
        if index != missing_receipt:
            receipt_path.write_bytes(b"{broken" if index == malformed_receipt else _canonical(receipt))
        ledger.record_terminal("worker", ledger_id, lease.lease_id, str(output))
        finalizer.enqueue("worker", ledger_id, lease.lease_id, output)
        result = finalizer.finalize_next()
        assert result is not None
        if index == 0:
            assert result.outcome == "accepted"
        if index == 5 and not all_success:
            assert result.outcome == "rejected"
    ledger.close()
    return root, matrix, rows, ledger_ids


def _real_persistent_campaign(
    tmp_path: Path,
    *,
    rows: list[dict[str, object]],
    campaign_root: Path,
    matrix_path: Path,
    policy: dict[str, object],
    campaign_round_id: str,
    campaign_run_id: str,
) -> tuple[Path, Path, list[dict[str, object]]]:
    """Drive terminal evidence through the real worker/manifest/recorder seam.

    Isaac itself is intentionally absent on the controller host.  This local
    session is the same narrow seam used by the persistent worker: it authors
    the real attempt manifest, reloads its immutable identity, records the
    episode with ``AutonomousRecorder``, then lets ``PersistentRolloutWorker``
    author the worker receipt and the real finalizer settle the ledger.  It
    deliberately does not handwrite any reviewed terminal evidence.
    """

    from lehome.flywheel.artifact_queue import ArtifactFinalizationQueue
    from lehome.flywheel.fidelity import fidelity_receipt
    from lehome.flywheel.isaac_recorder import AutonomousRecorder, CANONICAL_VIDEO_FILENAMES
    from lehome.flywheel.models import EpisodeIdentity
    from lehome.flywheel.persistent_manifest import write_persistent_flywheel_manifest
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker
    from lehome.flywheel.snapshots import Snapshot
    from lehome.flywheel.task_ledger import TaskLedger

    campaign_root.mkdir(parents=True, exist_ok=False)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_bytes(_canonical(rows))
    ledger = TaskLedger(
        campaign_root / "ledger.sqlite3", attempt_matrix=rows,
        max_attempts=len(rows), target_accepted=len(rows),
        completion_metric="terminal_outcomes",
    )
    finalizer = ArtifactFinalizationQueue(
        run_root=campaign_root, ledger=ledger, max_pending_items=1,
        max_pending_bytes=1 << 30, evaluation_only=True,
    )

    class _FinalizingController:
        def lease_next(self, worker_id: str):
            return ledger.lease_next(worker_id, lease_duration_ns=1_000_000_000)

        def record_terminal(self, worker_id: str, attempt_id: str, lease_id: str, raw_artifact_id: str):
            ledger.record_terminal(worker_id, attempt_id, lease_id, raw_artifact_id)
            finalizer.enqueue(worker_id, attempt_id, lease_id, Path(raw_artifact_id))
            result = finalizer.finalize_next()
            assert result is not None
            if result.outcome not in {"accepted", "rejected"}:
                raise AssertionError(f"real terminal finalization failed: {result.reason}")
            return result

        def status(self, attempt_id: str) -> str:
            return ledger.status(attempt_id)

        def heartbeat(self, worker_id: str, attempt_id: str, lease_id: str):
            return ledger.heartbeat(worker_id, attempt_id, lease_id, lease_duration_ns=1_000_000_000)

    class _Policy:
        action_horizon = 16

        def reset(self) -> None:
            pass

    successful_seeds = {int(row["seed"]) for row in rows[:5]}
    manifest_args = SimpleNamespace(
        policy_repo=policy["policy_repo"], policy_revision=policy["policy_revision"],
        policy_step=policy["policy_step"], policy_artifact_sha256=policy["policy_artifact_sha256"],
        campaign_round_id=campaign_round_id, campaign_run_id=campaign_run_id,
        device="cpu", policy_device="cuda:0",
    )

    class _ManifestRecorderSession:
        runtime_receipt = {
            "simulation_device": "cpu", "cloth_device": "cpu",
            "cloth_backend": "usd_local_points_v1", "cloth_readback": {"observed": True},
            "visible_contact_canary": {"observed": True},
            "renderer_device": "cuda:0", "camera_device": "cuda:0", "policy_device": "cuda:0",
        }

        def prepare_episode(self, **_kwargs) -> None:
            pass

        def run_episode(self, *, assignment, attempt_output_dir: Path, **_kwargs):
            manifest_path = write_persistent_flywheel_manifest(
                attempt_output_dir, assignment, manifest_args,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            identity = EpisodeIdentity(**manifest["identity"])
            recorder = AutonomousRecorder(
                attempt_output_dir, policy_revision=str(manifest["policy_revision"]),
                episode_id=str(manifest["episode_id"]), identity=identity,
                provenance={
                    "policy_artifact_sha256": manifest["policy_artifact_sha256"],
                    "image_identity": manifest["image_identity"],
                    "simulator_device": manifest["simulator_device"],
                    "policy_device": manifest["policy_device"],
                },
                simple_curriculum_collection=True,
            )

            def encode(root: Path, *, fps: int = 30) -> tuple[str, ...]:
                del fps
                videos = root / "videos"
                videos.mkdir(exist_ok=True)
                for filename in CANONICAL_VIDEO_FILENAMES:
                    (videos / filename).write_bytes(b"recorded-video")
                return CANONICAL_VIDEO_FILENAMES

            recorder.video_sink.encode = encode
            # ``EvaluationSession`` writes its rollout gallery videos at the
            # attempt root after the recorder seals its immutable raw
            # episode.  The queue validates that production handoff layout,
            # while ``encode`` above proves the actual recorder path too.
            encode(attempt_output_dir)
            snapshot = Snapshot(
                3, (0.0,) * 12, (0.0,) * 12,
                ((0.0, 0.0, 0.0),), ((0.0, 0.0, 0.0),),
                {"seed": identity.seed}, identity.garment_name, {"strategy": "canonical"},
                cloth_state_authority="usd_local_points_v1",
            )
            observation = {
                "observation.state": np.zeros(12, dtype=np.float32),
                "observation.images.top_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                "observation.images.left_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                "observation.images.right_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            }
            succeeded = identity.seed in successful_seeds
            recorder.record_snapshot("reset", snapshot)
            recorder.record_step(
                observation, np.zeros(12, dtype=np.float32), reward=1.0 if succeeded else 0.0,
                success=succeeded, request_id=f"request-{identity.episode_id}", chunk_offset=0,
            )
            recorder.record_snapshot("terminal", snapshot)
            recorded = recorder.finish(
                reason="success" if succeeded else "horizon", accepted_success=succeeded,
                visible_contact={
                    "observed": True, "source": "simulator_particle_to_gripper_distance",
                    "minimum_distance_m": 0.001,
                },
                fidelity=fidelity_receipt(
                    missing_cloth=False, cloth_flight=False, nonfinite_cloth_state=False,
                    safety_failure=False, monitor_active=True, monitor_observed=True,
                ),
            )
            recorded_episode = json.loads((recorded.path / "episode.json").read_text(encoding="utf-8"))
            assert recorded_episode["identity"] == manifest["identity"]
            assert recorded_episode["identity"]["episode_id"] == assignment["attempt_id"]
            assert recorded_episode["provenance"]["policy_artifact_sha256"] == manifest["policy_artifact_sha256"]
            assert recorded_episode["fidelity"]["monitor_observed"] is True
            return {"success": succeeded, "metrics": []}

        def close(self) -> None:
            pass

    try:
        worker = PersistentRolloutWorker(
            worker_id="worker", session_id="session", controller=_FinalizingController(),
            simulator_factory=_ManifestRecorderSession, policy=_Policy(),
            output_root=campaign_root, renderer_device="cuda:0", policy_device="cuda:0",
            simulator_device="cpu", heartbeat_interval_seconds=100.0,
            simple_curriculum_collection=True,
        )
        worker.run()
    finally:
        ledger.close()
    return campaign_root, matrix_path, rows


def test_real_persistent_launcher_admits_only_the_exact_200_success_replay_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The executable worker must carry the reviewed 4x100/200 tuple intact.

    This calls the production ``run`` boundary rather than settling a ledger
    fixture by hand.  Descriptor authentication is a separately exhaustive
    recovery-collection contract; the spy proves this launcher invokes it
    before constructing the exact 400/200 TaskLedger tuple.
    """

    import scripts.run_groot_persistent_worker as worker_module
    import lehome.flywheel.recovery_collection as recovery_collection

    matrix_path = tmp_path / "fresh-replay.json"
    matrix_path.write_text("[]", encoding="utf-8")
    matrix = [{"attempt_id": f"fresh-{index}"} for index in range(400)]
    descriptor_calls: list[Path] = []
    captured: dict[str, object] = {}

    def validate_descriptor(path: Path) -> list[dict[str, object]]:
        descriptor_calls.append(Path(path))
        return matrix

    class _Ledger:
        def close(self) -> None:
            pass

    def ledger_factory(_database, **kwargs):
        captured["ledger"] = kwargs
        return _Ledger()

    class _Worker:
        def __init__(self, **kwargs) -> None:
            captured["worker"] = kwargs

        def run(self) -> list[dict[str, object]]:
            return []

    monkeypatch.setattr(recovery_collection, "validate_success_replay_descriptor", validate_descriptor)
    monkeypatch.setattr(worker_module, "_load_matrix", lambda _path: matrix)
    monkeypatch.setattr(worker_module, "PersistentRolloutWorker", _Worker)
    monkeypatch.setenv("LEHOME_SUCCESS_REPLAY_CAMPAIGN", "1")
    args = SimpleNamespace(
        lease_seconds=30.0, preparation_timeout_seconds=30.0,
        source_finalization_timeout_seconds=30.0, device="cpu",
        renderer_device="cuda:0", policy_device="cuda:0", database=tmp_path / "ledger.sqlite3",
        attempt_matrix=matrix_path, max_attempts=400, target_accepted=200,
        completion_metric="accepted_successes", simple_curriculum_collection=False,
        fidelity_diagnostic=False, worker_id="worker", session_id="session",
        output_root=tmp_path / "output", initial_garment="Top_Long_Seen_0",
    )

    assert worker_module.run(args, ledger_factory=ledger_factory) == []
    assert descriptor_calls == [matrix_path]
    assert captured["ledger"] == {
        "attempt_matrix": matrix, "max_attempts": 400,
        "target_accepted": 200, "completion_metric": "accepted_successes",
    }

    args.target_accepted = 199
    with pytest.raises(ValueError, match="CPU success replay campaign is invalid"):
        worker_module.run(args, ledger_factory=ledger_factory)


def test_simple_summary_uses_external_matrix_assignment_ids_and_passes_gate_directly(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert set(report["trials"][0]) == {
        "schedule_index", "trial_id", "attempt_id", "category", "garment", "seed", "official_success",
        "terminal_event", "episode_sha256", "worker_receipt_sha256",
    }
    assert report["gate_trials"][0]["assignment_id"] == rows[0]["attempt_id"]
    assert report["gate_trials"][0]["assignment_id"] != ledger_ids[str(rows[0]["attempt_id"])]
    assert type(report["gate_trials"][0]["official_success"]) is bool
    assert report["gate_trials"][0]["identity"]["policy_artifact_sha256"] == POLICY["policy_artifact_sha256"]

    gate_spec = importlib.util.spec_from_file_location("gate_for_producer_test", ROOT / "scripts" / "check_simple_curriculum_gate.py")
    assert gate_spec and gate_spec.loader
    gate = importlib.util.module_from_spec(gate_spec); sys.modules[gate_spec.name] = gate; gate_spec.loader.exec_module(gate)
    receipt = gate.build_gate_receipt(
        report, report_bytes=_canonical(report), matrix=rows, matrix_bytes=matrix.read_bytes(),
        trusted_policy=POLICY, policy_bytes=_canonical(POLICY), catalog=_catalog(), catalog_bytes=_canonical(_catalog()),
    )
    assert receipt["decision"] == "continue"


def test_simple_partition_report_authenticates_an_unequal_weighted_split(tmp_path: Path) -> None:
    """The 300-row curriculum physical split is deliberately not balanced."""
    summary = _module()
    base = build_calibration_rows(_catalog(), seed_base=900)
    quotas = {"top_long": 78, "top_short": 77, "pant_long": 77, "pant_short": 68}
    rows = [row for category, quota in quotas.items() for row in [entry for entry in base if entry["category"] == category][:quota]]
    # Give every selected row a new unique logical identity while preserving
    # the canonical physical fields required by the terminal evidence path.
    rows = [{**row, "attempt_id": f"curriculum-{index:03d}", "trial_id": f"curriculum-{index:03d}", "logical_stage": "curriculum"} for index, row in enumerate(rows)]
    root, matrix, expected, ledger_ids = _simple_campaign(tmp_path, rows=rows)

    report = summary.build_simple_partition_report(
        campaign_root=root, matrix_path=matrix,
        matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(), **POLICY,
    )

    assert report["episodes"] == 300
    assert report["kind"] == "lehome_simple_curriculum_partition_report_v1"
    assert report["logical_stage"] == "curriculum"
    assert report["trials"][0]["attempt_id"] == ledger_ids[str(expected[0]["attempt_id"])]
    assert report["trials"][0]["assignment_id"] == expected[0]["attempt_id"]
    assert report["trials"][0]["finalized_artifact_root"].endswith(report["trials"][0]["attempt_id"])
    assert report["trials"][0]["runtime"] == {
        "simulation_device": "cpu", "cloth_device": "cpu", "renderer_device": "cuda:0",
        "camera_device": "cuda:0", "policy_device": "cuda:0",
    }


def test_fresh_source_adoption_rehashes_actual_terminal_artifacts_for_all_four_partitions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The 1,000-row fresh-source receipt is rooted in real terminal artifacts."""
    from lehome.flywheel.fresh_replay_evidence import (
        PARENT_ARTIFACT_SHA256,
        PARENT_POLICY_REPO,
        PARENT_REVISION,
        authenticate_fresh_source_contract,
    )

    summary, controller = _module(), _orchestrator_module()
    campaign = tmp_path / "collection"
    round_id, run_id = "fresh-12k-integration", "fresh-run-integration"
    policy = {
        "policy_repo": PARENT_POLICY_REPO,
        "policy_revision": PARENT_REVISION,
        "policy_step": 12_000,
        "policy_artifact_sha256": PARENT_ARTIFACT_SHA256,
    }
    runtime = {
        **policy,
        "rollout_image": "repo/rollout@sha256:" + "a" * 64,
        "trainer_image": "repo/trainer@sha256:" + "b" * 64,
        "simulator_device": "cpu", "cloth_device": "cpu", "policy_device": "cuda:0", "worker_count": 4,
    }
    observer = tmp_path / "observer.json"
    observer.write_text(json.dumps({
        "schema_version": 1, "kind": "lehome_spend_observation_v1", "observer": "fixture",
        "observed_at_utc": "2026-08-28T00:00:00Z", "spent_usd": 0.0,
    }), encoding="utf-8")
    config = controller.CollectionConfig(
        campaign_root=campaign, host_code_root=ROOT, run_id=run_id, round_id=round_id,
        max_wall_seconds=3600.0, max_spend_usd=99.0, paid=False, gpu_stop_command=None,
        runtime_identity=runtime, spend_observer=observer,
    )

    calibration = build_calibration_rows(_catalog(), seed_base=910)
    curriculum = [
        {
            **calibration[index % len(calibration)],
            "attempt_id": f"curriculum-{index:04d}", "trial_id": f"curriculum-{index:04d}",
            "seed": 2_000_000 + index, "source_seed": 2_000_000 + index, "logical_stage": "curriculum",
        }
        for index in range(600)
    ]
    partitions = {
        "calibration-head": calibration[:100],
        "calibration-tail": calibration[100:],
        "curriculum-a": curriculum[:300],
        "curriculum-b": curriculum[300:],
    }

    for partition, rows in partitions.items():
        partition_root = campaign / "fresh" / partition
        matrix = campaign / "partitions" / f"{partition}.json"
        root, physical_matrix, _rows = _real_persistent_campaign(
            tmp_path, rows=rows, campaign_root=partition_root, matrix_path=matrix,
            policy=policy, campaign_round_id=round_id, campaign_run_id=run_id,
        )
        capsys.readouterr()
        report = summary.build_simple_partition_report(
            campaign_root=root, matrix_path=physical_matrix,
            matrix_sha256=hashlib.sha256(physical_matrix.read_bytes()).hexdigest(), **policy,
        )
        assert len(report["trials"]) == len(rows)
        assert {trial["terminal_event"] for trial in report["trials"]} == {"accepted", "rejected"}
        assert sum(trial["terminal_event"] == "accepted" for trial in report["trials"]) == 5
        report_path = campaign / "reports" / "partitions" / f"{partition}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(_canonical(report))
        terminal_root = partition_root / "evaluation-terminal"
        sync = HubSyncDaemon(
            repository="ryanjin333/lehome-groot-n17-rollouts",
            round_id=round_id,
            run_id=run_id,
            token="fixture-token",
            transport=_ReceiptTransport(),
            accepted_root=terminal_root,
            receipts_root=partition_root / "hf-sync-receipts",
            readback_root=partition_root / "hf-readback",
            revision="main",
        )
        for trial in report["trials"]:
            attempt = str(trial["attempt_id"])
            artifact = Path(str(trial["finalized_artifact_root"]))
            assert artifact.parent == terminal_root
            sync.sync_episode(attempt, artifact)

    # Rejected outcomes carry the same recorder-authored campaign identity as
    # accepted ones.  The controller must not fill it in from config.
    rejected_partition = campaign / "fresh" / "calibration-head" / "evaluation-terminal"
    rejected_episode = next(
        path / "raw" / path.name / "episode.json"
        for path in rejected_partition.iterdir()
        if json.loads((path / "raw" / path.name / "episode.json").read_text(encoding="utf-8"))["accepted_success"] is False
    )
    rejected_payload = json.loads(rejected_episode.read_text(encoding="utf-8"))
    rejected_payload["identity"]["campaign_run_id"] = "fresh-run-wrong"
    rejected_episode.write_bytes(_canonical(rejected_payload))
    with pytest.raises(controller.ReceiptMismatchError, match="campaign-bound episode"):
        controller._build_fresh_source_report(config)
    rejected_payload["identity"]["campaign_run_id"] = run_id
    rejected_episode.write_bytes(_canonical(rejected_payload))

    controller._build_fresh_source_report(config)
    report_path = campaign / "reports" / "fresh-source-report.json"
    matrix_path = campaign / "reports" / "fresh-source-matrix.json"
    assert len(authenticate_fresh_source_contract((report_path,), (matrix_path,))) == 1_000
    terminal_manifest = json.loads((campaign / "reports" / "fresh-terminal-artifacts.json").read_text(encoding="utf-8"))
    assert len(terminal_manifest["entries"]) == 1_000

    # Complete the actual journal stage from the actual producer output.
    # Resume must re-open both rejected terminal evidence and accepted Hub
    # receipts rather than trusting the top-level stage hashes.
    journal = controller.StageJournal(config)
    fresh_output = controller.CommandRunner(config)._discover("fresh-report", {})
    journal.complete("fresh-report", None, fresh_output, inputs={})

    rejected = next(entry for entry in terminal_manifest["entries"] if entry["terminal_event"] == "rejected")
    rejected_receipt = Path(str(rejected["finalized_artifact_root"])) / "worker-receipt.json"
    rejected_bytes = rejected_receipt.read_bytes()
    rejected_receipt.write_bytes(b'{"tampered":true}\n')
    with pytest.raises(controller.ReceiptMismatchError, match="fresh terminal"):
        journal._read("fresh-report", None, {})
    rejected_receipt.write_bytes(rejected_bytes)

    accepted_entry = next(entry for entry in terminal_manifest["entries"] if entry["terminal_event"] == "accepted")
    accepted_receipt = (
        Path(str(accepted_entry["finalized_artifact_root"])).parent.parent
        / "hf-sync-receipts" / f"{accepted_entry['attempt_id']}.sync.json"
    )
    accepted_receipt.unlink()
    with pytest.raises(controller.ReceiptMismatchError, match="fresh Hub readback receipt|fresh terminal"):
        journal._read("fresh-report", None, {})

    accepted = next(entry for entry in terminal_manifest["entries"] if entry["terminal_event"] == "accepted")
    episode_path = Path(str(accepted["finalized_artifact_root"])) / "raw" / str(accepted["attempt_id"]) / "episode.json"
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    payload["post_receipt_mutation"] = True
    episode_path.write_bytes(_canonical(payload))

    with pytest.raises(controller.ReceiptMismatchError, match="fresh terminal"):
        controller.CommandRunner(config)._discover("fresh-report", {})


def test_simple_head_retains_mixed_policy_failures_as_authenticated_terminal_evidence(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    rejected_assignment = rows[5]
    rejected_id = ledger_ids[str(rejected_assignment["attempt_id"])]
    terminal = root / "evaluation-terminal" / rejected_id
    assert (terminal / "raw" / rejected_id / "episode.json").is_file()
    assert (terminal / "worker-receipt.json").is_file()
    assert (terminal / "SHA256SUMS.json").is_file()
    rejected_trial = next(trial for trial in report["trials"] if trial["attempt_id"] == rejected_id)
    assert rejected_trial["terminal_event"] == "rejected"
    assert rejected_trial["official_success"] == 0
    assert report["valid_outcomes"] == 100


def test_simple_summary_authenticates_accepted_finalizer_destination(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, _rows, _ledger_ids = _simple_campaign(
        tmp_path, all_success=True, evaluation_terminal=False,
    )

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 100
    assert report["infrastructure_invalid_executions"] == 0


def test_simple_summary_admits_worker_restart_with_a_fresh_session(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, _rows, _ledger_ids = _simple_campaign(
        tmp_path, session_for_index=lambda index: "session-after-restart" if index >= 50 else "session-before-restart",
    )

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 100
    assert report["infrastructure_invalid_executions"] == 0


@pytest.mark.parametrize("fidelity", [
    {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False, "monitor_active": False, "monitor_observed": True},
    {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False, "monitor_active": True, "monitor_observed": False},
    {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False, "monitor_active": True},
])
def test_simple_summary_rejects_incomplete_or_unobserved_terminal_fidelity(tmp_path: Path, fidelity: dict[str, bool]) -> None:
    summary = _module()

    def mutate(episode, index):
        if index == 3:
            episode["fidelity"] = fidelity
    root, matrix, _rows, _ledger_ids = _simple_campaign(tmp_path, episode_mutator=mutate)

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1


def test_simple_summary_rejects_forged_finalized_artifact_destination(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    destination = root / "evaluation-terminal" / ledger_id
    destination.rename(root / "evaluation-terminal" / f"forged-{ledger_id}")

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1


def test_simple_summary_rejects_receipt_raw_output_path_not_bound_to_worker_lease(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    receipt_path = root / "evaluation-terminal" / ledger_id / "worker-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["output_dir"] = str(root / "worker" / "session" / ledger_id / "forged-lease" / "generation-4")
    receipt_path.write_bytes(_canonical(receipt))

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1


def test_simple_summary_rejects_stale_receipt_copied_into_another_finalized_artifact(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    target = ledger_ids[str(rows[3]["attempt_id"])]
    source = ledger_ids[str(rows[4]["attempt_id"])]
    target_receipt = root / "evaluation-terminal" / target / "worker-receipt.json"
    target_receipt.write_bytes((root / "evaluation-terminal" / source / "worker-receipt.json").read_bytes())

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] < 100
    assert report["infrastructure_invalid_executions"] >= 1


def test_simple_summary_rejects_semantically_valid_finalized_episode_with_stale_manifest(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    episode_path = root / "evaluation-terminal" / ledger_id / "raw" / ledger_id / "episode.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["audit_note"] = "tampered-after-finalization"
    episode_path.write_bytes(_canonical(episode))

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1


@pytest.mark.parametrize("case", [
    "missing", "malformed", "duplicate", "nonfinite", "noncanonical_hash", "traversal",
    "missing_entry", "extra_file", "symlink",
])
def test_simple_summary_rejects_invalid_finalized_checksum_manifest(tmp_path: Path, case: str) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    destination = root / "evaluation-terminal" / ledger_id
    manifest_path = destination / "SHA256SUMS.json"
    if case == "missing":
        manifest_path.unlink()
    elif case == "malformed":
        manifest_path.write_text("{broken", encoding="utf-8")
    elif case == "duplicate":
        manifest_path.write_text('{"worker-receipt.json":{},"worker-receipt.json":{}}', encoding="utf-8")
    elif case == "nonfinite":
        manifest_path.write_text('{"worker-receipt.json":{"sha256":NaN,"size":1}}', encoding="utf-8")
    elif case == "noncanonical_hash":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["worker-receipt.json"]["sha256"] = "A" * 64
        manifest_path.write_bytes(_canonical(manifest))
    elif case == "traversal":
        manifest_path.write_text('{"../worker-receipt.json":{"sha256":"' + "a" * 64 + '","size":1}}', encoding="utf-8")
    elif case == "missing_entry":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("worker-receipt.json")
        manifest_path.write_bytes(_canonical(manifest))
    elif case == "extra_file":
        (destination / "extra-evidence.bin").write_bytes(b"unexpected")
    else:
        (destination / "unsafe-evidence").symlink_to(destination / "worker-receipt.json")

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1


def test_simple_summary_rejects_symlinked_finalized_artifact_ancestor(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    destination = root / "evaluation-terminal" / ledger_id
    outside = tmp_path / "outside" / ledger_id; outside.parent.mkdir()
    destination.rename(outside)
    destination.symlink_to(outside, target_is_directory=True)

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1


@pytest.mark.parametrize("fault", ["duplicate", "nonfinite"])
def test_simple_summary_rejects_non_strict_finalized_episode_json(tmp_path: Path, fault: str) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    path = root / "evaluation-terminal" / ledger_id / "raw" / ledger_id / "episode.json"
    episode = json.loads(path.read_text())
    if fault == "duplicate":
        episode.pop("accepted_success")
        path.write_text(json.dumps(episode, sort_keys=True)[:-1] + ',"accepted_success":false,"accepted_success":true}')
    else:
        path.write_text(json.dumps(episode, sort_keys=True)[:-1] + ',"probe":NaN}')

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1
    assert report["execution_count"] == 100


def test_simple_summary_deduplicates_malformed_finalized_episode_execution(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    path = root / "evaluation-terminal" / ledger_id / "raw" / ledger_id / "episode.json"
    path.write_bytes(b"{broken")

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1
    assert report["execution_count"] == 100


@pytest.mark.parametrize("kind", ["missing", "malformed"])
def test_simple_summary_counts_incomplete_receipts_as_invalid_executions(tmp_path: Path, kind: str) -> None:
    summary = _module()
    root, matrix, _rows, _ledger_ids = _simple_campaign(
        tmp_path, missing_receipt=3 if kind == "missing" else None, malformed_receipt=3 if kind == "malformed" else None,
    )

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1
    assert report["execution_count"] == 100


def test_simple_summary_counts_retry_then_valid_as_one_invalid_execution(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, _rows, _ledger_ids = _simple_campaign(tmp_path, retry_then_valid=3)

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 100
    assert report["infrastructure_invalid_executions"] == 1
    assert report["execution_count"] == 101


def test_simple_summary_counts_retry_and_later_invalid_terminal_as_distinct_executions(tmp_path: Path) -> None:
    summary = _module()

    def mutate(receipt, index):
        if index == 3:
            receipt["runtime"] = {**receipt["runtime"], "camera_device": "cuda:1"}
    root, matrix, _rows, _ledger_ids = _simple_campaign(
        tmp_path, retry_then_valid=3, receipt_mutator=mutate,
    )

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 2
    assert report["execution_count"] == 101


def test_simple_summary_counts_stray_evidence_with_retried_ledger_id_separately(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path, retry_then_valid=3)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    stray = root / f"stray-{ledger_id}"; stray.mkdir()
    (stray / "episode.json").write_bytes(b"{broken")

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 100
    assert report["infrastructure_invalid_executions"] == 2
    assert report["execution_count"] == 102


def test_simple_summary_counts_untraversed_symlink_directory_as_unsafe_evidence(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, _rows, _ledger_ids = _simple_campaign(tmp_path)
    outside = tmp_path / "outside"; outside.mkdir()
    (outside / "episode.json").write_text("{}")
    (root / "stray-evidence").symlink_to(outside, target_is_directory=True)

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 100
    assert report["infrastructure_invalid_executions"] == 1
    assert report["execution_count"] == 101


def test_simple_summary_does_not_allow_aggregate_safety_to_disagree_with_gate_fidelity(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, _rows, _ledger_ids = _simple_campaign(tmp_path, contradictory_safety=3)

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["safety_failure"] is False


@pytest.mark.parametrize("evidence_name,identity_key", [("worker-receipt.json", "attempt_id"), ("episode.json", "episode_id")])
@pytest.mark.parametrize("kind", ["malformed", "unsafe", "duplicate", "unbound"])
def test_simple_summary_counts_each_stray_evidence_file_once(tmp_path: Path, evidence_name: str, identity_key: str, kind: str) -> None:
    summary = _module()
    root, matrix, _rows, ledger_ids = _simple_campaign(tmp_path)
    stray = root / "stray"; stray.mkdir()
    if kind == "malformed":
        (stray / evidence_name).write_bytes(b"{broken")
    elif kind == "unsafe":
        (stray / evidence_name).symlink_to(next(root.rglob(evidence_name)))
    elif kind == "duplicate":
        original = next(root.rglob(evidence_name)); duplicate = json.loads(original.read_text())
        assert duplicate[identity_key] in ledger_ids.values()
        (stray / evidence_name).write_bytes(_canonical(duplicate))
    else:
        evidence = json.loads(next(root.rglob(evidence_name)).read_text()); evidence[identity_key] = "unbound-ledger-id"
        (stray / evidence_name).write_bytes(_canonical(evidence))

    report = summary.build_report(campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(), candidate_key="original_baseline", **POLICY)

    assert report["valid_outcomes"] == 100
    assert report["infrastructure_invalid_executions"] == 1
    assert report["execution_count"] == 101


@pytest.mark.parametrize("field", ["lease_id", "worker_id", "session_id", "episode_generation", "output_dir", "runtime"])
def test_simple_summary_rejects_receipt_not_bound_to_ledger_terminal_artifact_or_runtime(tmp_path: Path, field: str) -> None:
    summary = _module()

    def mutate(receipt, index):
        if index == 3:
            receipt[field] = {**receipt["runtime"], "camera_device": "cuda:1"} if field == "runtime" else "mismatch"
    root, matrix, _rows, _ledger_ids = _simple_campaign(tmp_path, receipt_mutator=mutate)

    report = summary.build_report(campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(), candidate_key="original_baseline", **POLICY)

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1


@pytest.mark.parametrize("field", ["code_revision", "asset_revision", "simulator_version", "image_identity"])
@pytest.mark.parametrize("value", ["", " ", True, 1, "not-a-digest"])
def test_simple_summary_rejects_noncanonical_runtime_identity_values(tmp_path: Path, field: str, value: object) -> None:
    summary = _module()

    def mutate(episode, index):
        if index == 3:
            target = episode["provenance"] if field == "image_identity" else episode["identity"]
            target[field] = value
    root, matrix, _rows, _ledger_ids = _simple_campaign(tmp_path, episode_mutator=mutate)

    report = summary.build_report(campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(), candidate_key="original_baseline", **POLICY)

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1
