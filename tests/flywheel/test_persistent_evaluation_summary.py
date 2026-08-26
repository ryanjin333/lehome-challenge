from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from copy import deepcopy

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "summarize_groot_persistent_evaluation.py"
REVISION = "e" * 40
ARTIFACT = "7" * 64


class _FakeFinalReportHub:
    def __init__(self) -> None:
        self.payloads: dict[tuple[str, str], bytes] = {}

    def upload_bytes(self, repository: str, path: str, payload: bytes) -> None:
        self.payloads[(repository, path)] = payload

    def read_bytes(self, repository: str, path: str) -> bytes:
        return self.payloads[(repository, path)]


def _module():
    spec = importlib.util.spec_from_file_location("persistent_evaluation_summary", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _campaign(tmp_path: Path, *, successes: dict[str, int] | None = None) -> tuple[Path, Path, str]:
    root = tmp_path / "campaign"
    root.mkdir()
    categories = ("top_long", "top_short", "pant_long", "pant_short")
    matrix = []
    database = sqlite3.connect(root / "ledger.sqlite3")
    database.executescript(
        "CREATE TABLE attempts (attempt_id TEXT PRIMARY KEY, schedule_index INTEGER UNIQUE, assignment_json TEXT);"
        "CREATE TABLE events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, at_ns INTEGER, event_type TEXT, "
        "attempt_id TEXT, lease_id TEXT, worker_id TEXT, payload_json TEXT);"
    )
    for category_index, category in enumerate(categories):
        for index in range(20):
            schedule_index = category_index * 20 + index
            trial_id = f"{category}-public-unseen-{index}"
            attempt_id = hashlib.sha256(trial_id.encode()).hexdigest()
            assignment = {
                "attempt_id": trial_id,
                "trial_id": trial_id,
                "garment": f"{category}_unseen_{index // 10}",
                "garment_name": f"{category}_unseen_{index // 10}",
                "category": category,
                "release_stage": "public_unseen",
                "seed": 600 + index,
            }
            matrix.append(assignment)
            database.execute(
                "INSERT INTO attempts VALUES (?, ?, ?)",
                (attempt_id, schedule_index, json.dumps(assignment, sort_keys=True, separators=(",", ":"))),
            )
            success = index < ((successes or {}).get(category, 13 if category == "top_long" else 12))
            database.execute(
                "INSERT INTO events(at_ns,event_type,attempt_id,lease_id,worker_id,payload_json) VALUES (?,?,?,?,?,?)",
                (1_000 + schedule_index, "accepted" if success else "rejected", attempt_id, "lease", "worker-1", "{}"),
            )
            episode_root = root / "worker-1" / attempt_id / "generation-1"
            raw = episode_root / "raw" / attempt_id
            raw.mkdir(parents=True)
            episode = {
                "episode_id": attempt_id,
                "identity": {
                    "episode_id": attempt_id,
                    "policy_repo": "ryanjin333/lehome-groot-n17-models",
                    "policy_revision": REVISION,
                    "policy_step": 2000,
                    "code_revision": "c" * 40,
                    "asset_revision": "a" * 40,
                    "simulator_version": "5.1.0.0",
                    "garment_name": assignment["garment_name"],
                    "category": category,
                    "release_stage": "public_unseen",
                    "seed": assignment["seed"],
                    "instruction": "fold the garment on the table",
                    "strategy": "canonical",
                },
                "provenance": {
                    "policy_artifact_sha256": ARTIFACT,
                    "simulator_device": "cuda:0",
                    "policy_device": "cuda:0",
                    "image_identity": "sha256:" + "b" * 64,
                },
                "outcome": "success" if success else "timeout",
                "accepted_success": success,
            }
            (raw / "episode.json").write_text(json.dumps(episode), encoding="utf-8")
            receipt = {
                "schema_version": 1,
                "attempt_id": attempt_id,
                "worker_id": "worker-1",
                "outcome": {"success": success, "metrics": []},
            }
            (episode_root / "worker-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            sync_root = root / "hf-sync-receipts"
            sync_root.mkdir(exist_ok=True)
            sync = {
                "schema_version": 1,
                "attempt_id": attempt_id,
                "repository": "owner/evaluation-artifacts",
                "round_id": "unseen80-final",
                "remote_prefix": f"evaluation/unseen80/{attempt_id}",
                "publication_ref": "main",
                "immutable_revision": "d" * 40,
                "entry_count": 2,
                "episode_sha256": hashlib.sha256(("remote:" + attempt_id).encode()).hexdigest(),
                "readback_verified": True,
            }
            (sync_root / f"{attempt_id}.sync.json").write_text(json.dumps(sync, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    database.commit()
    database.close()
    matrix_path = tmp_path / "unseen-80.json"
    encoded = (json.dumps(matrix, sort_keys=True, separators=(",", ":")) + "\n").encode()
    matrix_path.write_bytes(encoded)
    return root, matrix_path, hashlib.sha256(encoded).hexdigest()


def _job_and_publication(*, digest: str) -> tuple[object, dict[str, object]]:
    source = type("Source", (), {"kind": "bc", "repository": "owner/data", "revision": "a" * 40, "prefix": "bc/full", "manifest_sha256": "d" * 64, "tree_sha256": "e" * 64})()
    job = type("Job", (), {"experiment_id": "a" * 64, "training": type("Training", (), {"target_step": 2000})(), "evaluation": type("Evaluation", (), {"matrix_sha256": digest, "policy_digest": ARTIFACT})(), "trainer": {"image_id": "image", "oci_digest": "sha256:" + "f" * 64, "code_revision": "c" * 40}, "data_sources": (source,)})()
    publication = {
        "schema_version": 2, "experiment_id": "a" * 64, "job_digest": "a" * 64,
        "target_step": 2000, "repository": "ryanjin333/lehome-groot-n17-models",
        "immutable_revision": REVISION, "remote_prefix": "experiments/a/step-2000",
        "artifact_sha256": ARTIFACT, "receipt_sha256": "b" * 64,
        "readback_verified": True, "relative_path": "checkpoint.tar",
        "artifact_byte_size": 1, "descriptor_relative_path": "checkpoint.json",
        "descriptor_sha256": "c" * 64, "descriptor_byte_size": 1,
    }
    return job, publication


def _baseline_evidence(module, legacy: dict[str, object]) -> dict[str, object]:
    evidence: dict[str, object] = {
        "schema_version": 1,
        "kind": "lehome_experiment_paired_unseen20_baseline",
        "matrix_sha256": legacy["matrix_sha256"],
        "policy_digest": ARTIFACT,
        "episode_artifacts": [
            {
                "trial_id": item["trial_id"],
                "official_success": item["official_success"],
                "episode_sha256": item["episode_sha256"],
                "worker_receipt_sha256": item["worker_receipt_sha256"],
            }
            for item in legacy["trials"]
        ],
        "promotion_metrics": {"progress": legacy["progress"], "recovery": legacy["recovery"]},
        "readback_verified": True,
        "sealed": True,
    }
    evidence["report_sha256"] = module._canonical_sha256(evidence)
    return evidence


def _seen_evidence(module, receipt: str) -> dict[str, object]:
    evidence: dict[str, object] = {
        "schema_version": 1,
        "kind": "lehome_experiment_seen_regression_evidence",
        "candidate_checkpoint_receipt_sha256": receipt,
        "major_seen_regression": False,
        "readback_verified": True,
        "sealed": True,
    }
    evidence["report_sha256"] = module._canonical_sha256(evidence)
    return evidence


def test_build_report_binds_all_80_terminal_artifacts_and_policy_identity(tmp_path: Path) -> None:
    module = _module()
    root, matrix, digest = _campaign(tmp_path)

    report = module.build_report(
        campaign_root=root,
        matrix_path=matrix,
        matrix_sha256=digest,
        candidate_key="new_step_2k",
        policy_repo="ryanjin333/lehome-groot-n17-models",
        policy_revision=REVISION,
        policy_step=2000,
        policy_artifact_sha256=ARTIFACT,
    )

    assert report["episodes"] == 80
    assert report["official_successes"] == 49
    assert report["per_category"]["top_long"]["official_successes"] == 13
    assert report["per_category"]["pant_short"]["official_successes"] == 12
    assert report["gates"] == {"overall_ge_70": False, "each_category_ge_60": True}
    assert len(report["trials"]) == 80
    assert report["report_sha256"] == module.report_sha256(report)


def test_build_report_accepts_cpu_cloth_with_a_canonical_cuda_policy_device(tmp_path: Path) -> None:
    module = _module()
    root, matrix, digest = _campaign(tmp_path)
    for episode_path in root.rglob("episode.json"):
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        episode["provenance"]["simulator_device"] = "cpu"
        episode["provenance"]["policy_device"] = "cuda:0"
        episode_path.write_text(json.dumps(episode), encoding="utf-8")

    report = module.build_report(
        campaign_root=root,
        matrix_path=matrix,
        matrix_sha256=digest,
        candidate_key="new_step_2k",
        policy_repo="ryanjin333/lehome-groot-n17-models",
        policy_revision=REVISION,
        policy_step=2000,
        policy_artifact_sha256=ARTIFACT,
    )

    assert report["episodes"] == 80


@pytest.mark.parametrize("policy_device", [None, "cpu", "cuda"])
def test_build_report_rejects_cpu_cloth_without_a_canonical_cuda_policy_device(
    tmp_path: Path,
    policy_device: object,
) -> None:
    module = _module()
    root, matrix, digest = _campaign(tmp_path)
    episode_path = next(root.rglob("episode.json"))
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["provenance"]["simulator_device"] = "cpu"
    episode["provenance"]["policy_device"] = policy_device
    episode_path.write_text(json.dumps(episode), encoding="utf-8")

    with pytest.raises(ValueError, match="device provenance"):
        module.build_report(
            campaign_root=root,
            matrix_path=matrix,
            matrix_sha256=digest,
            candidate_key="new_step_2k",
            policy_repo="ryanjin333/lehome-groot-n17-models",
            policy_revision=REVISION,
            policy_step=2000,
            policy_artifact_sha256=ARTIFACT,
        )


def test_build_report_rejects_mismatched_cuda_simulator_and_policy_devices(tmp_path: Path) -> None:
    module = _module()
    root, matrix, digest = _campaign(tmp_path)
    episode_path = next(root.rglob("episode.json"))
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["provenance"]["simulator_device"] = "cuda:0"
    episode["provenance"]["policy_device"] = "cuda:1"
    episode_path.write_text(json.dumps(episode), encoding="utf-8")

    with pytest.raises(ValueError, match="device provenance"):
        module.build_report(
            campaign_root=root,
            matrix_path=matrix,
            matrix_sha256=digest,
            candidate_key="new_step_2k",
            policy_repo="ryanjin333/lehome-groot-n17-models",
            policy_revision=REVISION,
            policy_step=2000,
            policy_artifact_sha256=ARTIFACT,
        )


def test_build_report_accepts_a_consistent_nonzero_cuda_device(tmp_path: Path) -> None:
    module = _module()
    root, matrix, digest = _campaign(tmp_path)
    for episode_path in root.rglob("episode.json"):
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        episode["provenance"]["simulator_device"] = "cuda:2"
        episode["provenance"]["policy_device"] = "cuda:2"
        episode_path.write_text(json.dumps(episode), encoding="utf-8")

    report = module.build_report(
        campaign_root=root,
        matrix_path=matrix,
        matrix_sha256=digest,
        candidate_key="new_step_2k",
        policy_repo="ryanjin333/lehome-groot-n17-models",
        policy_revision=REVISION,
        policy_step=2000,
        policy_artifact_sha256=ARTIFACT,
    )

    assert report["episodes"] == 80


def test_build_report_rejects_misattributed_checkpoint_and_incomplete_ledger(tmp_path: Path) -> None:
    module = _module()
    root, matrix, digest = _campaign(tmp_path)
    episode = next(root.rglob("episode.json"))
    payload = json.loads(episode.read_text())
    payload["identity"]["policy_step"] = 12000
    episode.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="policy identity"):
        module.build_report(
            campaign_root=root, matrix_path=matrix, matrix_sha256=digest,
            candidate_key="new_step_2k", policy_repo="ryanjin333/lehome-groot-n17-models",
            policy_revision=REVISION, policy_step=2000, policy_artifact_sha256=ARTIFACT,
        )
    payload["identity"]["policy_step"] = 2000
    episode.write_text(json.dumps(payload), encoding="utf-8")
    database = sqlite3.connect(root / "ledger.sqlite3")
    database.execute("DELETE FROM events WHERE attempt_id=(SELECT attempt_id FROM attempts ORDER BY schedule_index LIMIT 1)")
    database.commit()
    database.close()
    with pytest.raises(ValueError, match="terminal"):
        module.build_report(
            campaign_root=root, matrix_path=matrix, matrix_sha256=digest,
            candidate_key="new_step_2k", policy_repo="ryanjin333/lehome-groot-n17-models",
            policy_revision=REVISION, policy_step=2000, policy_artifact_sha256=ARTIFACT,
        )


def test_experiment_report_binds_job_publication_matrix_and_all_categories(tmp_path: Path) -> None:
    module = _module()
    root, matrix, digest = _campaign(tmp_path)
    source = type("Source", (), {"kind": "bc", "repository": "owner/data", "revision": "a" * 40, "prefix": "bc/full", "manifest_sha256": "d" * 64, "tree_sha256": "e" * 64})()
    job = type("Job", (), {"experiment_id": "a" * 64, "training": type("Training", (), {"target_step": 2000})(), "evaluation": type("Evaluation", (), {"matrix_sha256": digest, "policy_digest": ARTIFACT})(), "trainer": {"image_id": "image", "oci_digest": "sha256:" + "f" * 64, "code_revision": "c" * 40}, "data_sources": (source,)})()
    publication = {"schema_version": 1, "experiment_id": "a" * 64, "job_digest": "a" * 64, "target_step": 2000, "repository": "ryanjin333/lehome-groot-n17-models", "immutable_revision": REVISION, "remote_prefix": "experiments/a/step-2000", "artifact_sha256": ARTIFACT, "receipt_sha256": "b" * 64, "readback_verified": True}
    report = module.build_experiment_report(experiment_job=job, checkpoint_publication=publication, campaign_root=root, matrix_path=matrix, matrix_sha256=digest)
    assert report["checkpoint_receipt_sha256"] == "b" * 64
    assert report["policy_digest"] == ARTIFACT
    assert set(report["categories"]) == {"top_long", "top_short", "pant_long", "pant_short"}
    assert set(report["promotion_metrics"]) == {
        "overall_successes", "overall_episodes", "overall_success_rate", "safety_failure",
        "paired_improvement", "gpu_seconds", "infrastructure_retry_count", "progress", "recovery", "pairing",
    }
    assert set(report["provenance"]) == {"trainer", "runtime", "data_sources"}


def test_strict_experiment_report_parser_preserves_promotion_evidence_and_sidecar(tmp_path: Path) -> None:
    module = _module()
    root, matrix, digest = _campaign(tmp_path)
    source = type("Source", (), {"kind": "bc", "repository": "owner/data", "revision": "a" * 40, "prefix": "bc/full", "manifest_sha256": "d" * 64, "tree_sha256": "e" * 64})()
    job = type("Job", (), {"experiment_id": "a" * 64, "training": type("Training", (), {"target_step": 2000})(), "evaluation": type("Evaluation", (), {"matrix_sha256": digest, "policy_digest": ARTIFACT})(), "trainer": {"image_id": "image", "oci_digest": "sha256:" + "f" * 64, "code_revision": "c" * 40}, "data_sources": (source,)})()
    publication = {"schema_version": 1, "experiment_id": "a" * 64, "job_digest": "a" * 64, "target_step": 2000, "repository": "ryanjin333/lehome-groot-n17-models", "immutable_revision": REVISION, "remote_prefix": "experiments/a/step-2000", "artifact_sha256": ARTIFACT, "receipt_sha256": "b" * 64, "readback_verified": True}
    report = module.build_experiment_report(experiment_job=job, checkpoint_publication=publication, campaign_root=root, matrix_path=matrix, matrix_sha256=digest)
    output = module.write_experiment_report(tmp_path / "strict.json", report)
    from lehome_train.groot.experiment_evaluation import load_experiment_evaluation, to_evaluation_score
    parsed = load_experiment_evaluation(output)
    assert parsed.infrastructure_retry_count == 0
    assert parsed.evidence_report_sha256 == report["evidence_report_sha256"]
    with pytest.raises(ValueError, match="baseline_evaluation_required"):
        to_evaluation_score(parsed)

    sidecar = output.with_suffix(".json.sha256")
    sidecar.chmod(0o600)
    sidecar.write_text("0" * 64 + "\n")
    with pytest.raises(ValueError, match="sidecar"):
        load_experiment_evaluation(output)


def test_strict_experiment_report_parser_rejects_metric_tampering(tmp_path: Path) -> None:
    module = _module()
    root, matrix, digest = _campaign(tmp_path)
    source = type("Source", (), {"kind": "bc", "repository": "owner/data", "revision": "a" * 40, "prefix": "bc/full", "manifest_sha256": "d" * 64, "tree_sha256": "e" * 64})()
    job = type("Job", (), {"experiment_id": "a" * 64, "training": type("Training", (), {"target_step": 2000})(), "evaluation": type("Evaluation", (), {"matrix_sha256": digest, "policy_digest": ARTIFACT})(), "trainer": {"image_id": "image", "oci_digest": "sha256:" + "f" * 64, "code_revision": "c" * 40}, "data_sources": (source,)})()
    publication = {"schema_version": 1, "experiment_id": "a" * 64, "job_digest": "a" * 64, "target_step": 2000, "repository": "ryanjin333/lehome-groot-n17-models", "immutable_revision": REVISION, "remote_prefix": "experiments/a/step-2000", "artifact_sha256": ARTIFACT, "receipt_sha256": "b" * 64, "readback_verified": True}
    report = module.build_experiment_report(experiment_job=job, checkpoint_publication=publication, campaign_root=root, matrix_path=matrix, matrix_sha256=digest)
    report["promotion_metrics"]["overall_successes"] += 1
    from lehome_train.groot.experiment_evaluation import parse_experiment_evaluation
    with pytest.raises(ValueError, match="report SHA"):
        parse_experiment_evaluation(report)


def test_strict_experiment_report_parser_rejects_rehashed_artifact_tampering(tmp_path: Path) -> None:
    module = _module()
    root, matrix, digest = _campaign(tmp_path)
    source = type("Source", (), {"kind": "bc", "repository": "owner/data", "revision": "a" * 40, "prefix": "bc/full", "manifest_sha256": "d" * 64, "tree_sha256": "e" * 64})()
    job = type("Job", (), {"experiment_id": "a" * 64, "training": type("Training", (), {"target_step": 2000})(), "evaluation": type("Evaluation", (), {"matrix_sha256": digest, "policy_digest": ARTIFACT})(), "trainer": {"image_id": "image", "oci_digest": "sha256:" + "f" * 64, "code_revision": "c" * 40}, "data_sources": (source,)})()
    publication = {"schema_version": 1, "experiment_id": "a" * 64, "job_digest": "a" * 64, "target_step": 2000, "repository": "ryanjin333/lehome-groot-n17-models", "immutable_revision": REVISION, "remote_prefix": "experiments/a/step-2000", "artifact_sha256": ARTIFACT, "receipt_sha256": "b" * 64, "readback_verified": True}
    report = module.build_experiment_report(experiment_job=job, checkpoint_publication=publication, campaign_root=root, matrix_path=matrix, matrix_sha256=digest)
    report["episode_artifacts"][0]["episode_sha256"] = "not-a-digest"
    report["report_sha256"] = module.report_sha256(report)
    from lehome_train.groot.experiment_evaluation import parse_experiment_evaluation
    with pytest.raises(ValueError, match="artifact"):
        parse_experiment_evaluation(report)


def test_report_counts_infrastructure_retries_without_scoring_them_as_policy_failures(tmp_path: Path) -> None:
    module = _module()
    root, matrix, digest = _campaign(tmp_path)
    database = sqlite3.connect(root / "ledger.sqlite3")
    database.execute(
        "INSERT INTO events(at_ns,event_type,attempt_id,lease_id,worker_id,payload_json) VALUES (?,?,?,?,?,?)",
        (999, "retryable", None, "lease", "worker-1", '{"gpu_seconds":2.5}'),
    )
    database.commit(); database.close()
    report = module.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=digest,
        candidate_key="new_step_2k", policy_repo="ryanjin333/lehome-groot-n17-models",
        policy_revision=REVISION, policy_step=2000, policy_artifact_sha256=ARTIFACT,
    )
    assert report["infrastructure_retry_count"] == 1
    assert report["official_successes"] == 49


def test_load_matrix_accepts_exact_unseen20_cardinality(tmp_path: Path) -> None:
    module = _module()
    rows = [
        {"trial_id": f"{category}-{index}", "category": category, "release_stage": "public_unseen"}
        for category in ("top_long", "top_short", "pant_long", "pant_short")
        for index in range(5)
    ]
    path = tmp_path / "unseen20.json"
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(payload)
    assert len(module._load_matrix(path, hashlib.sha256(payload).hexdigest())) == 20


def test_promotion_pairing_is_available_only_from_exact_sealed_baseline_evidence(tmp_path: Path) -> None:
    module = _module()
    root, matrix, digest = _campaign(tmp_path)
    job, publication = _job_and_publication(digest=digest)
    unavailable = module.build_experiment_report(
        experiment_job=job, checkpoint_publication=publication, campaign_root=root,
        matrix_path=matrix, matrix_sha256=digest,
    )
    assert unavailable["promotion_metrics"]["pairing"] == {"status": "baseline_evaluation_required"}
    from lehome_train.groot.experiment_evaluation import parse_experiment_evaluation, to_evaluation_score
    with pytest.raises(ValueError, match="baseline_evaluation_required"):
        to_evaluation_score(parse_experiment_evaluation(unavailable))

    legacy = module.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=digest,
        candidate_key="new_step_2k", policy_repo=publication["repository"],
        policy_revision=REVISION, policy_step=2000, policy_artifact_sha256=ARTIFACT,
    )
    baseline = _baseline_evidence(module, legacy)
    available = module.build_experiment_report(
        experiment_job=job, checkpoint_publication=publication, campaign_root=root,
        matrix_path=matrix, matrix_sha256=digest, baseline_evidence=baseline,
    )
    pairing = available["promotion_metrics"]["pairing"]
    assert pairing["status"] == "available"
    assert pairing["paired_trials"] == 80
    assert pairing["candidate_wins"] == 0 and pairing["baseline_wins"] == 0
    assert to_evaluation_score(parse_experiment_evaluation(available)).paired_improvement == 0.0

    tampered = deepcopy(baseline)
    tampered["episode_artifacts"][0]["trial_id"] = "wrong"  # type: ignore[index]
    tampered["report_sha256"] = module._canonical_sha256(tampered)
    with pytest.raises(ValueError, match="trial identities"):
        module.build_experiment_report(
            experiment_job=job, checkpoint_publication=publication, campaign_root=root,
            matrix_path=matrix, matrix_sha256=digest, baseline_evidence=tampered,
        )

    wrong_policy = deepcopy(baseline)
    wrong_policy["policy_digest"] = "0" * 64
    wrong_policy["report_sha256"] = module._canonical_sha256(wrong_policy)
    with pytest.raises(ValueError, match="pinned original parent"):
        module.build_experiment_report(
            experiment_job=job, checkpoint_publication=publication, campaign_root=root,
            matrix_path=matrix, matrix_sha256=digest, baseline_evidence=wrong_policy,
        )


def test_final_unseen80_e2e_seals_terminal_readback_artifacts_and_reaches_winner_gate(tmp_path: Path) -> None:
    module = _module()
    root, matrix, digest = _campaign(
        tmp_path, successes={"top_long": 14, "top_short": 14, "pant_long": 14, "pant_short": 14},
    )
    job, publication = _job_and_publication(digest=digest)
    report = module.build_final_unseen80_report(
        experiment_job=job, checkpoint_publication=publication, campaign_root=root,
        matrix_path=matrix, matrix_sha256=digest, candidate_id="dynamic-final",
        seen_regression_evidence=_seen_evidence(module, publication["receipt_sha256"]),
    )
    from lehome_train.groot.experiment_winner import (
        seal_final_unseen80_report,
        select_async_final_winner,
        validate_final_unseen80_report,
    )
    transport = _FakeFinalReportHub()
    output = module.write_final_unseen80_report(
        tmp_path / "final.json",
        report,
        transport=transport,
        repository="owner/final-reports",
        remote_path="finals/dynamic-final.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    parsed = validate_final_unseen80_report(report)
    assert parsed["overall_successes"] == 56
    assert len(report["episode_artifacts"]) == 80
    assert sum(item["official_success"] == 0 for item in report["episode_artifacts"]) == 24
    assert len(tuple((root / "hf-sync-receipts").glob("*.sync.json"))) == 80
    assert all(item["readback_verified"] is True for item in report["episode_artifacts"])
    assert {item["category"] for item in report["episode_artifacts"]} == {"top_long", "top_short", "pant_long", "pant_short"}
    assert output.with_suffix(".json.sha256").is_file()

    baseline = module.build_final_unseen80_report(
        experiment_job=job, checkpoint_publication=publication, campaign_root=root,
        matrix_path=matrix, matrix_sha256=digest, candidate_id="original-12k",
        seen_regression_evidence=_seen_evidence(module, publication["receipt_sha256"]),
    )
    from lehome_train.groot.experiment_winner import publish_final_unseen80_report
    baseline = publish_final_unseen80_report(
        baseline,
        transport=transport,
        repository="owner/final-reports",
        path="finals/original-12k.json",
    )
    result = select_async_final_winner(
        {"dynamic-final": report}, baseline_report=baseline,
        original_12k_checkpoint_digest=ARTIFACT, final_matrix_sha256=digest,
    )
    assert result["decision"] == "winner" and result["candidate_id"] == "dynamic-final"

    missing_receipt = next((root / "hf-sync-receipts").glob("*.sync.json"))
    missing_receipt.unlink()
    with pytest.raises(ValueError, match="Hub receipt"):
        module.build_final_unseen80_report(
            experiment_job=job, checkpoint_publication=publication, campaign_root=root,
            matrix_path=matrix, matrix_sha256=digest, candidate_id="dynamic-final",
            seen_regression_evidence=_seen_evidence(module, publication["receipt_sha256"]),
        )


def test_final_adapter_mode_emits_the_winner_receipt_instead_of_an_unseen20_promotion_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    module = _module()
    for name in (
        "LEHOME_SKIP_ROUND_SEAL",
        "LEHOME_CONTROLLED_RECOVERY_SMOKE",
        "LEHOME_SNAPSHOT_SOURCE_BOOTSTRAP",
        "LEHOME_RESUME_PREEMPTED_ROLLOUT",
    ):
        monkeypatch.setenv(name, "1")
    root, matrix, digest = _campaign(
        tmp_path, successes={"top_long": 14, "top_short": 14, "pant_long": 14, "pant_short": 14},
    )
    experiment_id = "a" * 64
    campaign_root = tmp_path / "adapter-roots"
    campaign_root.mkdir()
    root.rename(campaign_root / experiment_id)
    root = campaign_root / experiment_id
    job, publication = _job_and_publication(digest=digest)
    seen = _seen_evidence(module, publication["receipt_sha256"])
    seen_source = tmp_path / "seen-regression.json"
    seen_source.write_text(json.dumps(seen, sort_keys=True, separators=(",", ":")), encoding="ascii")
    seen_source.chmod(0o444)
    handoff_root = tmp_path / "seen-regression-handoffs"
    from scripts.materialize_finalist_seen_regression_handoff import materialize_handoff
    materialize_handoff(
        root=handoff_root, experiment_id=experiment_id,
        checkpoint_receipt_sha256=publication["receipt_sha256"], evidence_path=seen_source,
    )
    handoff_root.chmod(0o555)
    evaluator_script = ROOT / "scripts" / "run_lehome_experiment_evaluator.py"
    spec = importlib.util.spec_from_file_location("final_evaluator_cli", evaluator_script)
    assert spec and spec.loader
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    calls: list[dict[str, object]] = []

    def summarize(**kwargs: object) -> Path:
        report = module.build_final_unseen80_report(
            experiment_job=kwargs["experiment_job"], checkpoint_publication=kwargs["publication"],
            campaign_root=kwargs["campaign_root"], matrix_path=kwargs["matrix"],
            matrix_sha256=kwargs["matrix_sha256"], candidate_id=kwargs["candidate_id"],
            seen_regression_evidence=kwargs["seen_regression_evidence"],
        )
        return module.write_final_unseen80_report(
            kwargs["campaign_root"] / "final.json",
            report,
            transport=kwargs["final_report_transport"],
            repository=kwargs["final_report_repository"],
            remote_path=kwargs["final_report_path"],
        )

    transport = _FakeFinalReportHub()
    adapter = evaluator.PersistentFourWorkerAdapter(
        campaign_script="fake-campaign", campaign_root=campaign_root,
        runner=lambda _command, **kwargs: calls.append(kwargs), summarizer=summarize,
        mode="final-unseen80", final_report_transport=transport,
        final_report_repository="owner/final-reports", final_report_prefix="finals/",
        seen_regression_handoff_root=handoff_root,
    )
    lease = type("Lease", (), {"experiment_id": experiment_id, "publication": publication, "job": job})()
    returned = adapter.run(lease, str(matrix), digest, 4)
    assert returned["kind"] == "lehome_experiment_final_unseen80"
    assert calls[0]["env"]["LEHOME_ENABLE_HF_UPLOAD"] == "1"
    assert calls[0]["env"]["LEHOME_SIMULATOR_DEVICE"] == "cpu"
    assert calls[0]["env"]["LEHOME_SKIP_ROUND_SEAL"] == "0"
    assert calls[0]["env"]["LEHOME_CONTROLLED_RECOVERY_SMOKE"] == "0"
    assert calls[0]["env"]["LEHOME_SNAPSHOT_SOURCE_BOOTSTRAP"] == "0"
    assert calls[0]["env"]["LEHOME_RESUME_PREEMPTED_ROLLOUT"] == "0"
