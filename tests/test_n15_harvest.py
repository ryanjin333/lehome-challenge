"""Contracts for the fresh, uniform public-N1.5 harvest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CHECKPOINT_TREE = "a" * 64
CHECKPOINT_RECEIPT = "b" * 64


def _manifest(*, runtime_receipt_sha256: str = "c" * 64):
    from lehome.n15_harvest import HarvestProvenance, build_manifest

    return build_manifest(
        provenance=HarvestProvenance(
            checkpoint_tree_sha256=CHECKPOINT_TREE,
            checkpoint_receipt_sha256=CHECKPOINT_RECEIPT,
            runtime_receipt_sha256=runtime_receipt_sha256,
            source_tree_sha256="d" * 64,
            dataset_snapshot_sha256="e" * 64,
            rollout_image_sha256="f" * 64,
        )
    )


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def test_uniform_manifest_is_deterministic_balanced_and_globally_unique() -> None:
    from lehome.n15_harvest import CATEGORIES, canonical_bytes, validate_manifest

    first = _manifest()
    second = _manifest()
    assert canonical_bytes(first) == canonical_bytes(second)
    assert validate_manifest(first) == first
    assert first["summary"] == {
        "attempt_count": 1000,
        "attempts_per_category": 250,
        "attempts_per_garment": 25,
        "category_count": 4,
        "garment_count": 40,
        "garments_per_category": 10,
        "native_process_count": 40,
        "episodes_per_process": 25,
        "first_wave_process_count": 4,
    }
    rows = first["attempts"]
    assert len(rows) == 1000
    assert len({row["attempt_id"] for row in rows}) == 1000
    assert len({(row["process_seed"], row["episode_index"]) for row in rows}) == 1000
    assert len({row["process_seed"] for row in rows}) == 40
    assert 42 not in {row["process_seed"] for row in rows}
    for category, prefix in CATEGORIES.items():
        category_rows = [row for row in rows if row["category"] == category]
        assert len(category_rows) == 250
        assert {row["garment"] for row in category_rows} == {
            f"{prefix}_Seen_{index}" for index in range(10)
        }
        assert {
            garment: sum(row["garment"] == garment for row in category_rows)
            for garment in {row["garment"] for row in category_rows}
        } == {f"{prefix}_Seen_{index}": 25 for index in range(10)}
    assert {row["category"] for row in rows[:100]} == set(CATEGORIES)
    assert {category: sum(row["category"] == category for row in rows[:100]) for category in CATEGORIES} == {
        category: 25 for category in CATEGORIES
    }
    assert {row["garment"] for row in rows[:100]} == {
        f"{prefix}_Seen_0" for prefix in CATEGORIES.values()
    }


def test_manifest_explicitly_keeps_seen_names_but_excludes_release_identities_and_custom_data() -> None:
    from lehome.n15_harvest import HarvestError, validate_manifest

    manifest = _manifest()
    assert all("_Seen_" in row["garment"] for row in manifest["attempts"])
    assert manifest["exclusions"] == {
        "augmentation": True,
        "curriculum": True,
        "focused_evaluator_episode_identities": True,
        "focused_evaluator_seed_42": True,
        "hard_states": True,
        "historical_episodes": True,
        "perturbation": True,
        "release_unseen_garments": True,
    }
    assert manifest["scope"]["required_seen_garments_overlap_focused_categories"] is True
    assert manifest["scope"]["release_evaluator_assets_included"] is False

    forbidden_changes = (
        lambda value: value["attempts"][0].update(garment="Top_Short_Unseen_0"),
        lambda value: value["attempts"][0].update(process_seed=42),
        lambda value: value["attempts"][0].update(historical_episode_id="old"),
        lambda value: value["attempts"][0].update(hard_state={"step": 9}),
        lambda value: value["attempts"][0].update(perturbation={"cloth": True}),
        lambda value: value["attempts"][0].update(curriculum_weight=2.0),
        lambda value: value["attempts"][0].update(augmentation="flip"),
        lambda value: value["attempts"][1].update(attempt_id=value["attempts"][0]["attempt_id"]),
        lambda value: value["attempts"][1].update(
            process_seed=value["attempts"][0]["process_seed"],
            episode_index=value["attempts"][0]["episode_index"],
        ),
    )
    for mutate in forbidden_changes:
        changed = json.loads(json.dumps(manifest))
        mutate(changed)
        with pytest.raises(HarvestError):
            validate_manifest(changed)


def test_manifest_and_receipt_are_created_atomically_and_never_overwritten(tmp_path: Path) -> None:
    from lehome.n15_harvest import HarvestError, write_manifest_bundle

    manifest_path = tmp_path / "harvest.json"
    receipt_path = tmp_path / "harvest.receipt.json"
    result = write_manifest_bundle(
        manifest=_manifest(),
        manifest_path=manifest_path,
        receipt_path=receipt_path,
    )
    assert result["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert json.loads(receipt_path.read_text())["manifest_sha256"] == result["manifest_sha256"]
    assert oct(manifest_path.stat().st_mode & 0o777) == "0o444"
    with pytest.raises(HarvestError, match="already exists"):
        write_manifest_bundle(
            manifest=_manifest(), manifest_path=manifest_path, receipt_path=receipt_path
        )


def _outcomes(manifest: dict[str, object], *, successes: int = 5, invalid: int = 2):
    rows = []
    for index, attempt in enumerate(manifest["attempts"][:100]):
        if index < successes:
            outcome = "success"
        elif index < successes + invalid:
            outcome = "infrastructure_invalid"
        else:
            outcome = "policy_failure"
        rows.append(
            {
                "attempt_id": attempt["attempt_id"],
                "official_outcome": outcome,
                "cloth_fidelity": {"measured": True, "valid": True},
            }
        )
    return rows


def _process_rows(manifest: dict[str, object]) -> list[dict[str, object]]:
    seen: set[tuple[object, ...]] = set()
    rows = []
    for attempt in manifest["attempts"]:
        identity = (attempt["category"], attempt["garment"], attempt["process_seed"])
        if identity in seen:
            continue
        seen.add(identity)
        rows.append(attempt)
    return rows


def _healthy_fidelity(path: Path, garment: str, *, episode_count: int = 25) -> None:
    from rollout_appliance.native_reference_site.cloth_fidelity import _EvidenceWriter

    health = {
        "healthy": True,
        "sample_count": 4,
        "max_position_m": 1.0,
        "max_extent_m": 0.5,
        "max_velocity_mps": 0.2,
        "max_position_limit_m": 25.0,
        "max_extent_limit_m": 10.0,
        "max_velocity_limit_mps": 100.0,
        "missing_cloth": False,
        "cloth_flight": False,
        "nonfinite_cloth_state": False,
    }
    writer = _EvidenceWriter(path)
    for episode in range(1, episode_count + 1):
        writer.append(
            garment=garment, reset_sequence=episode, stage="post_step",
            step_index=1, health=health,
        )
        writer.append(
            garment=garment, reset_sequence=episode, stage="pre_score",
            step_index=2, health=health,
        )
        writer.append_terminal(
            garment=garment, reset_sequence=episode, step_index=2, status="healthy"
        )
    writer.flush(reason="test")


def _materialize_process_evidence(
    root: Path, manifest: dict[str, object], *, process_count: int = 40
) -> dict[str, object]:
    processes = []
    for row in _process_rows(manifest)[:process_count]:
        process_id = f"{row['category']}-{row['garment_index']:02d}-s{row['process_seed']:06d}"
        process_root = root / "processes" / process_id
        process_root.mkdir(parents=True)
        (process_root / "evaluator.log").write_text(
            "\n".join(
                [f"Episode {index}/25: Return=1.0, Length=42, Success={'True' if index == 1 else 'False'}" for index in range(1, 26)]
                + ["Evaluation completed successfully"]
            )
            + "\n",
            encoding="utf-8",
        )
        _healthy_fidelity(process_root / "cloth-fidelity.jsonl", str(row["garment"]))
        dataset = process_root / "dataset/001"
        (dataset / "meta").mkdir(parents=True)
        (dataset / "data/chunk-000").mkdir(parents=True)
        (dataset / "meta/info.json").write_text(
            json.dumps({"total_episodes": 1, "total_frames": 2}) + "\n", encoding="utf-8"
        )
        (dataset / "meta/garment_info.json").write_text(
            json.dumps({str(row["garment"]): {"0": {"object_initial_pose": [0.0] * 6}}}) + "\n",
            encoding="utf-8",
        )
        (dataset / "data/chunk-000/file-000.parquet").write_bytes(b"test-parquet-episode-0")
        processes.append(
            {
                "process_id": process_id,
                "category": row["category"],
                "garment": row["garment"],
                "process_seed": row["process_seed"],
                "exit_code": 0,
            }
        )
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_process_status_v1",
        "processes": processes,
    }


def _success_dataset_receipt(
    root: Path, manifest: dict[str, object], *, attempt_count: int = 1000
) -> dict[str, object]:
    from lehome.n15_harvest import inspect_success_datasets

    return inspect_success_datasets(
        manifest=manifest, harvest_root=root, expected_attempt_count=attempt_count,
        parquet_episode_reader=lambda paths: [0, 0],
    )


def test_first_100_gate_enforces_success_fidelity_and_infrastructure_bounds() -> None:
    from lehome.n15_harvest import HarvestError, evaluate_first_100

    manifest = _manifest()
    passed = evaluate_first_100(manifest=manifest, outcomes=_outcomes(manifest))
    assert passed["decision"] == "continue"
    assert passed["success_count"] == 5
    assert passed["infrastructure_invalid_count"] == 2

    with pytest.raises(HarvestError, match="fewer than five"):
        evaluate_first_100(manifest=manifest, outcomes=_outcomes(manifest, successes=4))
    with pytest.raises(HarvestError, match="more than 2%"):
        evaluate_first_100(manifest=manifest, outcomes=_outcomes(manifest, invalid=3))
    fidelity = _outcomes(manifest)
    fidelity[37]["cloth_fidelity"]["valid"] = False
    with pytest.raises(HarvestError, match="cloth fidelity"):
        evaluate_first_100(manifest=manifest, outcomes=fidelity)
    missing = _outcomes(manifest)[:-1]
    with pytest.raises(HarvestError, match="exactly 100"):
        evaluate_first_100(manifest=manifest, outcomes=missing)


def _runtime_receipt(path: Path) -> dict[str, object]:
    value = {
        "schema_version": 1,
        "kind": "lehome_public_n15_harvest_runtime_v1",
        "source_revision": "d" * 40,
        "source_file_count": 1,
        "source_tree_sha256": "d" * 64,
        "checkpoint_file_count": 1,
        "checkpoint_tree_sha256": CHECKPOINT_TREE,
        "training_identity_receipt_sha256": CHECKPOINT_RECEIPT,
        "dataset_snapshot_sha256": "e" * 64,
        "base_model_metadata_sha256": "1" * 64,
        "dataset_metadata_sha256": "2" * 64,
        "rollout_image_id": "sha256:" + "f" * 64,
        "rollout_image_sha256": "f" * 64,
        "docker_inspect_sha256": "3" * 64,
        "python_executable": "/opt/lehome-challenge/.venv/bin/python",
        "python_version": "3.11.13",
        "lerobot_package_file_count": 1,
        "lerobot_package_tree_sha256": "4" * 64,
        "semantic_environment": {
            "policy_overrides": False,
            "checkpoint_compatibility_override": False,
            "cloth_fidelity_monitor": "observational_only",
        },
    }
    path.write_bytes(_canonical(value))
    return value


def _admission_evidence(
    root: Path,
    manifest: dict[str, object],
    *,
    worker_count: int,
    memory_passed: bool = True,
) -> None:
    from lehome.n15_harvest import admission_smoke_schedule

    root.mkdir(parents=True)
    (root / "memory").mkdir()
    (root / "smokes").mkdir()
    (root / "memory-status.tsv").write_text(
        "".join(f"{worker}\t0\n" for worker in range(worker_count)), encoding="ascii"
    )
    peak = 7_000 if memory_passed else 9_500
    (root / "memory.tsv").write_text(
        "sample_index\tactive_process_count\tgpu_used_mib\tgpu_total_mib\n"
        f"0\t0\t1000\t10000\n1\t{worker_count}\t{peak}\t10000\n2\t0\t1100\t10000\n",
        encoding="ascii",
    )
    for worker in range(worker_count):
        (root / "memory" / f"worker-{worker}.log").write_text(
            "Starting evaluation: 0 episodes\nEvaluation completed successfully\n",
            encoding="utf-8",
        )
    if not memory_passed:
        return
    schedule = admission_smoke_schedule(worker_count)
    (root / "smoke-status.tsv").write_text(
        "".join(f"{worker}\t0\n" for worker in range(worker_count)), encoding="ascii"
    )
    for row in schedule:
        worker = int(row["worker_id"])
        smoke = root / "smokes" / f"worker-{worker}"
        smoke.mkdir()
        (smoke / "smoke-id.txt").write_text(str(row["smoke_id"]) + "\n", encoding="ascii")
        (smoke / "evaluator.log").write_text(
            "Episode 1/1: Return=0.0, Length=42, Success=False\n"
            "Evaluation completed successfully\n",
            encoding="utf-8",
        )
        _healthy_fidelity(smoke / "cloth-fidelity.jsonl", str(row["garment"]), episode_count=1)


def test_admission_smokes_have_frozen_non_harvest_identities_and_seeds() -> None:
    from lehome.n15_harvest import admission_smoke_schedule

    manifest = _manifest()
    harvest_ids = {row["attempt_id"] for row in manifest["attempts"]}
    harvest_seeds = {row["process_seed"] for row in manifest["attempts"]}
    all_smoke_ids: set[str] = set()
    all_smoke_seeds: set[int] = set()
    for worker_count in (4, 2):
        schedule = admission_smoke_schedule(worker_count)
        assert len(schedule) == worker_count
        assert [row["worker_id"] for row in schedule] == list(range(worker_count))
        all_smoke_ids.update(row["smoke_id"] for row in schedule)
        all_smoke_seeds.update(row["process_seed"] for row in schedule)
    assert len(all_smoke_ids) == 6
    assert len(all_smoke_seeds) == 6
    assert all_smoke_ids.isdisjoint(harvest_ids)
    assert all_smoke_seeds.isdisjoint(harvest_seeds)


def test_worker_admission_is_derived_from_measurements_and_native_smokes(
    tmp_path: Path,
) -> None:
    from lehome.n15_harvest import HarvestError, admit_workers

    runtime_path = tmp_path / "runtime.json"
    _runtime_receipt(runtime_path)
    manifest = _manifest(runtime_receipt_sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest())
    four_root = tmp_path / "four"
    _admission_evidence(four_root, manifest, worker_count=4)
    four = admit_workers(
        manifest=manifest, runtime_receipt=runtime_path,
        four_worker_evidence_root=four_root, two_worker_evidence_root=None,
    )
    assert four["worker_count"] == 4
    assert four["admission"]["memory_check"]["measurements_sha256"] == hashlib.sha256(
        (four_root / "memory.tsv").read_bytes()
    ).hexdigest()
    assert all(smoke["cloth_fidelity_sha256"] for smoke in four["admission"]["smokes"])
    assert four["assignments"] == {
        "0": ["top_long"],
        "1": ["top_short"],
        "2": ["pant_long"],
        "3": ["pant_short"],
    }
    weak_four = tmp_path / "weak-four"
    two_root = tmp_path / "two"
    _admission_evidence(weak_four, manifest, worker_count=4, memory_passed=False)
    _admission_evidence(two_root, manifest, worker_count=2)
    two = admit_workers(
        manifest=manifest, runtime_receipt=runtime_path,
        four_worker_evidence_root=weak_four, two_worker_evidence_root=two_root,
    )
    assert two["worker_count"] == 2
    assert two["assignments"] == {
        "0": ["top_long", "pant_long"],
        "1": ["top_short", "pant_short"],
    }
    with pytest.raises(HarvestError, match="two-worker fallback"):
        admit_workers(
            manifest=manifest, runtime_receipt=runtime_path,
            four_worker_evidence_root=weak_four, two_worker_evidence_root=None,
        )
    (four_root / "smokes/worker-0/smoke-id.txt").write_text("wrong\n", encoding="ascii")
    with pytest.raises(HarvestError, match="smoke identity"):
        admit_workers(
            manifest=manifest, runtime_receipt=runtime_path,
            four_worker_evidence_root=four_root, two_worker_evidence_root=None,
        )


@pytest.mark.parametrize("stage", ("zero_episode", "one_episode_smoke"))
def test_measured_capacity_failure_deterministically_falls_back_to_two_workers(
    tmp_path: Path, stage: str,
) -> None:
    from lehome.n15_harvest import admit_workers

    runtime_path = tmp_path / "runtime.json"
    _runtime_receipt(runtime_path)
    manifest = _manifest(runtime_receipt_sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest())
    four_root = tmp_path / "four"
    two_root = tmp_path / "two"
    _admission_evidence(four_root, manifest, worker_count=4)
    _admission_evidence(two_root, manifest, worker_count=2)
    if stage == "zero_episode":
        (four_root / "memory-status.tsv").write_text(
            "0\t137\n1\t0\n2\t0\n3\t0\n", encoding="ascii"
        )
        (four_root / "memory/worker-0.log").write_text(
            "RuntimeError: CUDA out of memory while loading policy\n", encoding="utf-8"
        )
        (four_root / "memory.tsv").write_text(
            "sample_index\tactive_process_count\tgpu_used_mib\tgpu_total_mib\n"
            "0\t0\t1000\t10000\n1\t4\t9950\t10000\n2\t0\t1100\t10000\n",
            encoding="ascii",
        )
        (four_root / "smoke-status.tsv").unlink()
        shutil.rmtree(four_root / "smokes")
    else:
        (four_root / "smoke-status.tsv").write_text(
            "0\t137\n1\t0\n2\t0\n3\t0\n", encoding="ascii"
        )
        (four_root / "smokes/worker-0/evaluator.log").write_text(
            "RuntimeError: CUDA out of memory during one-episode smoke\n", encoding="utf-8"
        )
        (four_root / "smokes/worker-0/cloth-fidelity.jsonl").unlink()

    selected = admit_workers(
        manifest=manifest, runtime_receipt=runtime_path,
        four_worker_evidence_root=four_root, two_worker_evidence_root=two_root,
    )
    assert selected["worker_count"] == 2
    assert selected["fallback_from_four"] is True
    rejected = selected["rejected_four_worker_admission"]
    assert rejected["passed"] is False
    assert rejected["capacity_failure"]["stage"] == stage
    assert selected["admission"]["passed"] is True


def test_non_capacity_or_fidelity_admission_failure_never_falls_back(tmp_path: Path) -> None:
    from lehome.n15_harvest import HarvestError, admit_workers

    runtime_path = tmp_path / "runtime.json"
    _runtime_receipt(runtime_path)
    manifest = _manifest(runtime_receipt_sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest())
    two_root = tmp_path / "two"
    _admission_evidence(two_root, manifest, worker_count=2)
    for fault in ("semantic", "fidelity"):
        four_root = tmp_path / fault
        _admission_evidence(four_root, manifest, worker_count=4)
        if fault == "semantic":
            (four_root / "smoke-status.tsv").write_text(
                "0\t1\n1\t0\n2\t0\n3\t0\n", encoding="ascii"
            )
            (four_root / "smokes/worker-0/evaluator.log").write_text(
                "Traceback (most recent call last): policy contract changed\n", encoding="utf-8"
            )
        else:
            evidence = four_root / "smokes/worker-0/cloth-fidelity.jsonl"
            lines = evidence.read_text(encoding="utf-8").splitlines()
            changed = json.loads(lines[-1])
            changed["status"] = "invalid"
            lines[-1] = json.dumps(changed, sort_keys=True, separators=(",", ":"))
            evidence.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(HarvestError):
            admit_workers(
                manifest=manifest, runtime_receipt=runtime_path,
                four_worker_evidence_root=four_root, two_worker_evidence_root=two_root,
            )


def test_native_worker_plan_uses_public_scripts_eval_for_every_frozen_attempt(tmp_path: Path) -> None:
    from lehome.n15_harvest import admit_workers, native_worker_plan

    source = tmp_path / "public-source"
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    source.mkdir()
    checkpoint.mkdir()
    (source / "scripts").mkdir()
    (source / "scripts/eval.py").write_text("# public eval\n")
    runtime_path = tmp_path / "runtime.json"
    _runtime_receipt(runtime_path)
    manifest = _manifest(runtime_receipt_sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest())
    evidence = tmp_path / "admission"
    _admission_evidence(evidence, manifest, worker_count=4)
    admission = admit_workers(
        manifest=manifest, runtime_receipt=runtime_path,
        four_worker_evidence_root=evidence, two_worker_evidence_root=None,
    )
    plan = native_worker_plan(
        manifest=manifest,
        admission=admission,
        source_root=source,
        checkpoint_root=checkpoint,
        output_root=output,
    )
    assert admission["worker_count"] == 4
    assert len(plan["workers"]) == 4
    commands = [command for worker in plan["workers"] for command in worker["commands"]]
    assert len(commands) == 40
    assert all(command[:5] == ["python", "-P", "-m", "scripts.eval", "--policy_type"] for command in commands)
    assert all("--save_datasets" in command for command in commands)
    assert all(command[command.index("--num_episodes") + 1] == "25" for command in commands)
    assert all(command[command.index("--device") + 1] == "cpu" for command in commands)
    assert all("--use_random_seed" not in command for command in commands)
    assert len({command[command.index("--seed") + 1] for command in commands}) == 40
    assert {command[command.index("--garment_type") + 1] for command in commands} == {
        "top_long", "top_short", "pant_long", "pant_short"
    }
    assert all("_Seen_" in command[command.index("--garment_filter") + 1] for command in commands)
    assert {
        command[command.index("--dataset_root") + 1] for command in commands
    } == {str(source / "Datasets/example/four_types_merged")}


def test_full_collector_authenticates_all_1000_native_outcomes_and_fidelity(tmp_path: Path) -> None:
    from lehome.n15_harvest import collect_native_outcomes

    manifest = _manifest()
    status = _materialize_process_evidence(tmp_path, manifest)
    result = collect_native_outcomes(
        manifest=manifest,
        process_status=status,
        harvest_root=tmp_path,
        expected_attempt_count=1000,
        success_dataset_receipt=_success_dataset_receipt(tmp_path, manifest),
    )
    assert result["attempt_count"] == 1000
    assert result["success_count"] == 40
    assert result["infrastructure_invalid_count"] == 0
    assert result["fidelity_invalid_count"] == 0
    assert len(result["outcomes"]) == 1000
    assert [row["attempt_id"] for row in result["outcomes"]] == [
        row["attempt_id"] for row in manifest["attempts"]
    ]


@pytest.mark.parametrize("fault", ("nonzero", "missing-process", "error-log", "missing-fidelity", "extra-status"))
def test_full_collector_rejects_incomplete_or_invalid_native_evidence(tmp_path: Path, fault: str) -> None:
    from lehome.n15_harvest import HarvestError, collect_native_outcomes

    manifest = _manifest()
    status = _materialize_process_evidence(tmp_path, manifest)
    if fault == "nonzero":
        status["processes"][7]["exit_code"] = 1
    elif fault == "missing-process":
        status["processes"].pop()
    elif fault == "error-log":
        process_id = status["processes"][3]["process_id"]
        (tmp_path / "processes" / process_id / "evaluator.log").write_text(
            "Traceback (most recent call last):\n", encoding="utf-8"
        )
    elif fault == "missing-fidelity":
        process_id = status["processes"][4]["process_id"]
        (tmp_path / "processes" / process_id / "cloth-fidelity.jsonl").unlink()
    else:
        status["processes"].append(dict(status["processes"][0]))
    with pytest.raises(HarvestError):
        collect_native_outcomes(
            manifest=manifest,
            process_status=status,
            harvest_root=tmp_path,
            expected_attempt_count=1000,
            success_dataset_receipt=_success_dataset_receipt(tmp_path, manifest),
        )


@pytest.mark.parametrize("fault", ("missing", "corrupt", "extra", "mismatched"))
def test_success_dataset_inspector_rejects_invalid_or_unbound_artifacts(
    tmp_path: Path, fault: str,
) -> None:
    from lehome.n15_harvest import HarvestError, inspect_success_datasets

    manifest = _manifest()
    _materialize_process_evidence(tmp_path, manifest)
    first = _process_rows(manifest)[0]
    process_id = f"{first['category']}-{first['garment_index']:02d}-s{first['process_seed']:06d}"
    dataset = tmp_path / "processes" / process_id / "dataset/001"
    if fault == "missing":
        (dataset / "data/chunk-000/file-000.parquet").unlink()
    elif fault == "extra":
        (dataset.parent / "002").mkdir()
    elif fault == "mismatched":
        (dataset / "meta/garment_info.json").write_text(
            json.dumps({"Wrong_Garment": {"0": {"object_initial_pose": [0.0] * 6}}}) + "\n"
        )
    reader = (
        (lambda paths: (_ for _ in ()).throw(ValueError("corrupt parquet")))
        if fault == "corrupt"
        else (lambda paths: [0, 0])
    )
    with pytest.raises(HarvestError):
        inspect_success_datasets(
            manifest=manifest, harvest_root=tmp_path, expected_attempt_count=1000,
            parquet_episode_reader=reader,
        )


def test_success_dataset_inspector_accepts_upstream_zero_success_without_garment_metadata(
    tmp_path: Path,
) -> None:
    from lehome.n15_harvest import inspect_success_datasets

    manifest = _manifest()
    _materialize_process_evidence(tmp_path, manifest)
    first = _process_rows(manifest)[0]
    process_id = f"{first['category']}-{first['garment_index']:02d}-s{first['process_seed']:06d}"
    process_root = tmp_path / "processes" / process_id
    (process_root / "evaluator.log").write_text(
        "\n".join(
            [
                f"Episode {index}/25: Return=0.0, Length=42, Success=False"
                for index in range(1, 26)
            ]
            + ["Evaluation completed successfully"]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = process_root / "dataset/001"
    (dataset / "meta/info.json").write_text(
        json.dumps({"total_episodes": 0, "total_frames": 0}) + "\n", encoding="utf-8"
    )
    (dataset / "meta/garment_info.json").unlink()
    shutil.rmtree(dataset / "data")

    receipt = inspect_success_datasets(
        manifest=manifest, harvest_root=tmp_path, expected_attempt_count=1000,
        parquet_episode_reader=lambda paths: [0, 0] if paths else [],
    )
    assert receipt["success_count"] == 39
    assert receipt["processes"][0]["successful_attempts"] == []


@pytest.mark.parametrize("fault", ("official_success", "episode_artifact"))
def test_missing_garment_metadata_is_rejected_when_outcomes_or_artifacts_exist(
    tmp_path: Path, fault: str,
) -> None:
    from lehome.n15_harvest import HarvestError, inspect_success_datasets

    manifest = _manifest()
    _materialize_process_evidence(tmp_path, manifest)
    first = _process_rows(manifest)[0]
    process_id = f"{first['category']}-{first['garment_index']:02d}-s{first['process_seed']:06d}"
    process_root = tmp_path / "processes" / process_id
    dataset = process_root / "dataset/001"
    (dataset / "meta/garment_info.json").unlink()
    if fault == "episode_artifact":
        (process_root / "evaluator.log").write_text(
            "\n".join(
                [
                    f"Episode {index}/25: Return=0.0, Length=42, Success=False"
                    for index in range(1, 26)
                ]
                + ["Evaluation completed successfully"]
            )
            + "\n",
            encoding="utf-8",
        )
        (dataset / "meta/info.json").write_text(
            json.dumps({"total_episodes": 0, "total_frames": 0}) + "\n", encoding="utf-8"
        )
    with pytest.raises(HarvestError):
        inspect_success_datasets(
            manifest=manifest, harvest_root=tmp_path, expected_attempt_count=1000,
            parquet_episode_reader=lambda paths: [0, 0] if paths else [],
        )


@pytest.mark.parametrize("fault", ("process", "path", "binding"))
def test_success_dataset_receipt_revalidation_rejects_rebound_artifacts(
    tmp_path: Path, fault: str,
) -> None:
    from lehome.n15_harvest import HarvestError, validate_success_dataset_receipt

    manifest = _manifest()
    _materialize_process_evidence(tmp_path, manifest)
    receipt = _success_dataset_receipt(tmp_path, manifest)
    if fault == "process":
        receipt["processes"][0]["garment"] = "Top_Long_Seen_9"
    elif fault == "path":
        receipt["processes"][0]["dataset_relative_path"] = receipt["processes"][1][
            "dataset_relative_path"
        ]
    else:
        receipt["processes"][0]["successful_attempts"][0]["attempt_id"] = receipt[
            "successful_attempt_ids"
        ][1]
    with pytest.raises(HarvestError):
        validate_success_dataset_receipt(
            receipt, manifest=manifest, harvest_root=tmp_path, expected_attempt_count=1000,
        )


def test_observational_site_is_checked_in_policy_neutral_and_write_once(tmp_path: Path) -> None:
    from lehome.n15_harvest import HarvestError, write_observational_site

    output = tmp_path / "site"
    receipt = write_observational_site(output)
    payload = (output / "sitecustomize.py").read_text(encoding="utf-8")
    assert receipt["kind"] == "lehome_public_n15_observational_site_v1"
    assert "install_cloth_fidelity_monitor_on_env" in payload
    for forbidden in (
        "LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT",
        "LEHOME_NATIVE_REFERENCE_SANITIZED_CONFIG_ROOT",
        "LEHOME_NATIVE_REFERENCE_CHECKPOINT_COMPATIBILITY_RECEIPT",
        "LEHOME_CPU_ACTION",
    ):
        assert forbidden not in payload
    with pytest.raises(HarvestError, match="already exists"):
        write_observational_site(output)


def test_manifest_binds_complete_measured_runtime_not_checkpoint_labels() -> None:
    from lehome.n15_harvest import HarvestError, HarvestProvenance, build_manifest

    provenance = HarvestProvenance(
        checkpoint_tree_sha256="a" * 64,
        checkpoint_receipt_sha256="b" * 64,
        runtime_receipt_sha256="c" * 64,
        source_tree_sha256="d" * 64,
        dataset_snapshot_sha256="e" * 64,
        rollout_image_sha256="f" * 64,
    )
    value = build_manifest(provenance=provenance)
    assert value["provenance"]["runtime_receipt_sha256"] == "c" * 64
    assert value["provenance"]["source_tree_sha256"] == "d" * 64
    assert value["provenance"]["dataset_snapshot_sha256"] == "e" * 64
    assert value["provenance"]["rollout_image_sha256"] == "f" * 64
    changed = json.loads(json.dumps(value))
    changed["provenance"]["runtime_receipt_sha256"] = "mutable-label"
    with pytest.raises(HarvestError):
        # Receipt identities are frozen into the deterministic manifest, not mutable labels.
        __import__("lehome.n15_harvest", fromlist=["validate_manifest"]).validate_manifest(changed)


def test_runtime_measurement_uses_task1_validators_and_rejects_dirty_source(
    tmp_path: Path,
) -> None:
    from lehome.n15_harvest import HarvestError, measure_runtime_contract

    source = tmp_path / "source"
    source.mkdir()
    (source / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "source"], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    training_receipt = tmp_path / "training.json"
    training_receipt.write_text("{}\n", encoding="ascii")
    image_receipt = tmp_path / "image.json"
    image_receipt.write_text("{}\n", encoding="ascii")
    inspect_receipt = tmp_path / "inspect.json"
    inspect_receipt.write_text("[]\n", encoding="ascii")
    package = tmp_path / "lerobot"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = '0.4.3'\n")
    verified_inputs = SimpleNamespace(
        source_tree="tree", base_model_metadata_sha256="1" * 64,
        dataset_metadata_sha256="2" * 64,
        resolved_snapshots_receipt_sha256="3" * 64,
    )
    calls: list[str] = []
    result = measure_runtime_contract(
        source_root=source,
        source_revision=revision,
        checkpoint_root=checkpoint,
        training_identity_receipt=training_receipt,
        rollout_image_receipt=image_receipt,
        docker_inspect_receipt=inspect_receipt,
        python_executable=Path("/opt/lehome-challenge/.venv/bin/python"),
        python_version="3.11.13",
        lerobot_package_root=package,
        training_validator=lambda *args, **kwargs: calls.append("training") or {
            "checkpoint_root": str(checkpoint.parent),
            "checkpoint_files": {"checkpoints/012000/pretrained_model/model.safetensors": hashlib.sha256(b"weights").hexdigest()},
            "source_receipt_sha256": "4" * 64,
            "resolved_snapshots_receipt_sha256": "3" * 64,
        },
        inputs_validator=lambda *args, **kwargs: calls.append("inputs") or verified_inputs,
        expected_lerobot_tree=(1, __import__("lehome.n15_harvest", fromlist=["regular_tree_identity"]).regular_tree_identity(package)[1]),
        expected_image_id="sha256:" + "9" * 64,
        image_validator=lambda *args, **kwargs: calls.append("image") or {"image_id": "sha256:" + "9" * 64, "docker_inspect_sha256": "8" * 64},
    )
    assert calls == ["training", "inputs", "image"]
    assert result["source_file_count"] == 1
    assert result["checkpoint_file_count"] == 1
    assert result["dataset_snapshot_sha256"] == "3" * 64
    (source / "untracked.txt").write_text("dirty\n")
    with pytest.raises(HarvestError, match="clean"):
        measure_runtime_contract(
            source_root=source, source_revision=revision, checkpoint_root=checkpoint,
            training_identity_receipt=training_receipt, rollout_image_receipt=image_receipt,
            docker_inspect_receipt=inspect_receipt,
            python_executable=Path("/opt/lehome-challenge/.venv/bin/python"), python_version="3.11.13",
            lerobot_package_root=package,
            training_validator=lambda *args, **kwargs: {}, inputs_validator=lambda *args, **kwargs: verified_inputs,
            expected_lerobot_tree=(1, "0" * 64), expected_image_id="sha256:" + "9" * 64,
            image_validator=lambda *args, **kwargs: {},
        )


def test_rollout_image_rejects_baked_compatibility_or_override_environment(tmp_path: Path) -> None:
    from lehome.n15_harvest import HarvestError, _default_image_validator

    image_id = "sha256:" + "9" * 64
    inspect_path = tmp_path / "inspect.json"
    receipt_path = tmp_path / "receipt.json"
    inspect_path.write_text(
        json.dumps([{"Id": image_id, "Config": {"Env": ["PATH=/usr/bin"]}}]) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "kind": "lehome_official_image_inspection_v1", "reference": image_id,
        "image_id": image_id, "repo_digests": [],
        "docker_inspect_sha256": hashlib.sha256(inspect_path.read_bytes()).hexdigest(),
    }
    receipt_path.write_bytes(_canonical(receipt))
    assert _default_image_validator(receipt_path, inspect_path, expected_image_id=image_id)
    inspect_path.write_text(
        json.dumps([{"Id": image_id, "Config": {
            "Env": ["LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT=/bad"]
        }}]) + "\n",
        encoding="utf-8",
    )
    receipt["docker_inspect_sha256"] = hashlib.sha256(inspect_path.read_bytes()).hexdigest()
    receipt_path.write_bytes(_canonical(receipt))
    with pytest.raises(HarvestError, match="environment"):
        _default_image_validator(receipt_path, inspect_path, expected_image_id=image_id)


def test_publication_authenticates_upload_and_anonymous_byte_readback(tmp_path: Path) -> None:
    from lehome.n15_harvest import publish_harvest_bundle

    manifest = _manifest()
    status = _materialize_process_evidence(tmp_path / "run", manifest)
    outcomes = __import__("lehome.n15_harvest", fromlist=["collect_native_outcomes"]).collect_native_outcomes(
        manifest=manifest, process_status=status, harvest_root=tmp_path / "run",
        expected_attempt_count=1000,
        success_dataset_receipt=_success_dataset_receipt(tmp_path / "run", manifest),
    )
    receipt = __import__("lehome.n15_harvest", fromlist=["manifest_receipt"]).manifest_receipt(manifest)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_bytes(_canonical(manifest))
    (bundle / "manifest-receipt.json").write_bytes(_canonical(receipt))
    (bundle / "final-outcomes.json").write_bytes(_canonical(outcomes))
    remote: dict[str, bytes] = {}

    class Api:
        def __init__(self, *, anonymous: bool = False): self.anonymous = anonymous
        def repo_info(self, **_kwargs): return SimpleNamespace(private=False, sha="d" * 40)
        def list_repo_files(self, **_kwargs): return sorted(remote)
        def create_commit(self, *, operations, **_kwargs):
            for operation in operations:
                remote[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
            return SimpleNamespace(oid="d" * 40)

    def download(*, filename: str, local_dir: str | Path, token: object, **_kwargs):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(remote[filename])
        return str(target)

    publication = publish_harvest_bundle(
        bundle_root=bundle,
        manifest=manifest,
        manifest_receipt_value=receipt,
        final_outcomes=outcomes,
        repository="ryanjin333/lehome-groot-n15-rollouts",
        revision="main",
        token="secret",
        authenticated_api=Api(),
        anonymous_api=Api(anonymous=True),
        downloader=download,
        operation_factory=lambda path_in_repo, path_or_fileobj: SimpleNamespace(
            path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj
        ),
    )
    assert publication["immutable_revision"] == "d" * 40
    assert publication["readback_verified"] is True
    assert publication["anonymous_readback_verified"] is True
    assert publication["upload_tree_sha256"] == publication["authenticated_readback_tree_sha256"]
    assert publication["upload_tree_sha256"] == publication["anonymous_readback_tree_sha256"]


def test_publication_rejects_private_destination_or_changed_anonymous_bytes(tmp_path: Path) -> None:
    from lehome.n15_harvest import HarvestError, publish_harvest_bundle

    manifest = _manifest()
    receipt = __import__("lehome.n15_harvest", fromlist=["manifest_receipt"]).manifest_receipt(manifest)
    final = {
        "schema_version": 1, "kind": "lehome_public_n15_collected_outcomes_v1",
        "manifest_sha256": hashlib.sha256(_canonical(manifest)).hexdigest(),
        "attempt_count": 1000, "success_count": 0, "policy_failure_count": 1000,
        "infrastructure_invalid_count": 0, "fidelity_invalid_count": 0,
        "success_dataset_receipt_sha256": "9" * 64,
        "process_count": 40,
        "outcomes": [
            {"attempt_id": row["attempt_id"], "official_outcome": "policy_failure", "return": 0.0,
             "length": 1, "cloth_fidelity": {"measured": True, "valid": True}}
            for row in manifest["attempts"]
        ],
    }
    bundle = tmp_path / "bundle"; bundle.mkdir()
    (bundle / "manifest.json").write_bytes(_canonical(manifest))
    (bundle / "manifest-receipt.json").write_bytes(_canonical(receipt))
    (bundle / "final-outcomes.json").write_bytes(_canonical(final))

    class PrivateApi:
        def repo_info(self, **_kwargs): return SimpleNamespace(private=True, sha="d" * 40)

    with pytest.raises(HarvestError, match="public"):
        publish_harvest_bundle(
            bundle_root=bundle, manifest=manifest, manifest_receipt_value=receipt,
            final_outcomes=final, repository="ryanjin333/lehome-groot-n15-rollouts",
            revision="main", token="secret", authenticated_api=PrivateApi(),
            anonymous_api=PrivateApi(), downloader=lambda **kwargs: "",
            operation_factory=lambda **kwargs: kwargs,
        )

    remote: dict[str, bytes] = {}

    class PublicApi:
        def repo_info(self, **_kwargs): return SimpleNamespace(private=False, sha="d" * 40)
        def list_repo_files(self, **_kwargs): return sorted(remote)
        def create_commit(self, *, operations, **_kwargs):
            for operation in operations:
                remote[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
            return SimpleNamespace(oid="d" * 40)

    def changed_download(*, filename: str, local_dir: str | Path, token: object, **_kwargs):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = remote[filename] + (b"changed" if token is False else b"")
        target.write_bytes(payload)
        return str(target)

    with pytest.raises(HarvestError, match="byte readback mismatch"):
        publish_harvest_bundle(
            bundle_root=bundle, manifest=manifest, manifest_receipt_value=receipt,
            final_outcomes=final, repository="ryanjin333/lehome-groot-n15-rollouts",
            revision="main", token="secret", authenticated_api=PublicApi(),
            anonymous_api=PublicApi(), downloader=changed_download,
            operation_factory=lambda path_in_repo, path_or_fileobj: SimpleNamespace(
                path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj
            ),
        )


def test_terminal_contract_requires_public_immutable_byte_readback_and_exact_stopped_vm() -> None:
    from lehome.n15_harvest import (
        HarvestError, provider_stop_receipt_from_response, terminal_receipt,
    )

    manifest = _manifest()
    manifest_bytes = _canonical(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_receipt = {
        "schema_version": 1,
        "kind": "lehome_public_n15_harvest_manifest_receipt_v1",
        "manifest_sha256": manifest_sha,
        "checkpoint_tree_sha256": CHECKPOINT_TREE,
        "attempt_count": 1000,
    }
    manifest_receipt_sha = hashlib.sha256(_canonical(manifest_receipt)).hexdigest()
    publication = {
        "schema_version": 1,
        "kind": "lehome_public_n15_harvest_hf_readback_v1",
        "repository": "ryanjin333/lehome-groot-n15-rollouts",
        "repository_type": "dataset",
        "repository_private": False,
        "remote_prefix": "harvest/public-n15-" + "c" * 16,
        "immutable_revision": "d" * 40,
        "manifest_sha256": manifest_sha,
        "manifest_receipt_sha256": manifest_receipt_sha,
        "upload_tree_sha256": "e" * 64,
        "authenticated_readback_tree_sha256": "e" * 64,
        "anonymous_readback_tree_sha256": "e" * 64,
        "readback_verified": True,
        "anonymous_readback_verified": True,
    }
    provider = {
        "schema_version": 1,
        "kind": "lehome_public_n15_provider_stopped_v1",
        "vm_id": "computeinstance-u00t6xfqhadrcmssa2",
        "instance_name": "lehome-rollout",
        "disk_id": "computedisk-u00pbe55crxy7jr56x",
        "state": "STOPPED",
        "protected_disk_preserved": True,
        "created_resources": [],
        "deleted_resources": [],
        "provider_response_sha256": "f" * 64,
        "captured_unix_seconds": 1_788_150_400,
    }
    result = terminal_receipt(
        manifest=manifest,
        manifest_receipt=manifest_receipt,
        publication_receipt=publication,
        provider_receipt=provider,
    )
    assert result["terminal"] is True
    assert result["provider_state"] == "STOPPED"
    assert result["immutable_revision"] == "d" * 40

    for mutate, match in (
        (lambda value: value.update(repository_private=True), "public"),
        (lambda value: value.update(immutable_revision="main"), "immutable"),
        (lambda value: value.update(anonymous_readback_tree_sha256="f" * 64), "readback"),
        (lambda value: value.update(anonymous_readback_verified=False), "readback"),
    ):
        changed = dict(publication)
        mutate(changed)
        with pytest.raises(HarvestError, match=match):
            terminal_receipt(
                manifest=manifest,
                manifest_receipt=manifest_receipt,
                publication_receipt=changed,
                provider_receipt=provider,
            )
    running = dict(provider, state="RUNNING")
    with pytest.raises(HarvestError, match="STOPPED"):
        terminal_receipt(
            manifest=manifest,
            manifest_receipt=manifest_receipt,
            publication_receipt=publication,
            provider_receipt=running,
        )
    raw_provider = {
        "metadata": {"id": provider["vm_id"], "name": "lehome-rollout"},
        "status": {"state": "STOPPED"},
        "spec": {"secondary_disks": [{"existing_disk": {"id": provider["disk_id"]}}]},
    }
    assert provider_stop_receipt_from_response(raw_provider)["protected_disk_preserved"] is True
    for established_fields in (
        {"attach_mode": "READ_WRITE"},
        {"device_id": "lehome"},
        {"attach_mode": "READ_WRITE", "device_id": "lehome"},
    ):
        real_shape = json.loads(json.dumps(raw_provider))
        real_shape["spec"]["secondary_disks"][0].update(established_fields)
        assert provider_stop_receipt_from_response(real_shape)["protected_disk_preserved"] is True
    for disks in (
        [{"existing_disk": {"id": provider["disk_id"]}, "mode": "READ_WRITE"}],
        [{"existing_disk": {"id": provider["disk_id"]}, "attach_mode": "READ_ONLY"}],
        [{"existing_disk": {"id": provider["disk_id"]}, "device_id": "unexpected"}],
        [{"managed_disk": {"id": provider["disk_id"]}}],
        [
            {"existing_disk": {"id": provider["disk_id"]}},
            {"existing_disk": {"id": "computedisk-extra"}},
        ],
    ):
        changed = json.loads(json.dumps(raw_provider))
        changed["spec"]["secondary_disks"] = disks
        with pytest.raises(HarvestError, match="disk"):
            provider_stop_receipt_from_response(changed)
