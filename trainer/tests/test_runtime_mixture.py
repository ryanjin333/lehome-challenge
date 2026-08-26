from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from collections import Counter
from dataclasses import replace
from itertools import islice
from pathlib import Path

import pytest


def _sha_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def test_generated_mixture_uses_the_private_rollout_repository() -> None:
    from lehome_train.groot.runtime_mixture import APPROVED_MIXTURE_REPOSITORY

    assert APPROVED_MIXTURE_REPOSITORY == "ryanjin333/lehome-groot-n17-rollouts"


def _round(root: Path, attempt_id: str, *, category: str = "top_long") -> tuple[str, str]:
    attempt = root / "attempts" / attempt_id
    _write(attempt / "episode.json", {
        "episode_id": attempt_id, "accepted_success": True, "outcome": "success", "terminal_reason": "success",
        "mode": "autonomous",
        "identity": {
            "category": category,
            "release_stage": "seen",
            "instruction": "fold the garment on the table",
        },
    })
    (attempt / "annotations.jsonl").write_text(
        "".join(json.dumps({"step": step, "action_source": "policy", "state": [float(step)] * 12, "action": [float(step + 1)] * 12}) + "\n" for step in range(16)), encoding="utf-8"
    )
    for camera in ("top", "left", "right"):
        (attempt / "videos").mkdir(exist_ok=True)
        (attempt / "videos" / f"{camera}_rgb.mp4").write_bytes(b"fixture-video")
    checksums: dict[str, dict[str, object]] = {}
    for file in sorted(path for path in attempt.rglob("*") if path.is_file()):
        checksums[file.relative_to(attempt).as_posix()] = {"sha256": _sha_path(file), "size": file.stat().st_size}
    _write(attempt / "SHA256SUMS.json", checksums)
    return attempt.relative_to(root).as_posix(), _sha_path(attempt / "episode.json")


def _contract(
    tmp_path: Path,
    *,
    rollout_categories: tuple[str, ...] = ("top_long", "top_long", "top_long"),
) -> tuple[Path, Path, Path]:
    from lehome_train.groot.runtime_mixture import canonical_json_sha256, sha256_file, source_tree_sha256

    bc, round_root = tmp_path / "bc", tmp_path / "round-1"
    for root in (bc, round_root):
        _write(root / "receipt.json", {"accepted": True})
        _write(root / "acceptance.json", {"accepted_success": True})
    # Publication receipts live outside the mounted source trees.  Adding a
    # receipt after publishing must not mutate the tree that its readback
    # proves, otherwise source-tree hashing would form a cycle.
    bc_publication = tmp_path / "source-publication" / "bc-readback.json"
    rollout_publication = tmp_path / "source-publication" / "rollout-readback.json"
    _write(bc_publication, {"repository": "ryanjin333/lehome-groot-n17-data", "immutable_revision": "b" * 40, "remote_prefix": "bc/full", "fresh_readback_verified": True, "tree_listing_verified": True})
    _write(rollout_publication, {"repository": "ryanjin333/lehome-groot-n17-rollouts", "immutable_revision": "c" * 40, "remote_prefix": "rollouts/round-1", "fresh_readback_verified": True, "tree_listing_verified": True})
    _write(bc / "manifest.json", {"fps": 30, "fixed_language_instruction": "fold the garment on the table", "future_actions": {"horizon": 16}, "train_episode_ids": [str(index) for index in range(7)], "validation_episode_ids": []})
    # The loader test injects decoding, so opaque video bytes are enough for contract coverage.
    for episode in range(7):
        _write(bc / f"episodes/{episode}.json", {"episode_id": str(episode)})
    attempts = [
        _round(round_root, f"attempt-{number}", category=category)
        for number, category in enumerate(rollout_categories)
    ]
    normalization = tmp_path / "mixture-normalization.json"
    def grouped(dimensions: int) -> dict[str, list[float]]:
        return {name: [float(index) for index in range(dimensions)] for name in ("min", "max", "mean", "std", "q01", "q99")}
    statistics = {"new_embodiment": {
        "state": {
            "left_arm": grouped(5), "left_gripper": grouped(1),
            "right_arm": grouped(5), "right_gripper": grouped(1),
        },
        "action": {
            "left_arm": grouped(5), "left_gripper": grouped(1),
            "right_arm": grouped(5), "right_gripper": grouped(1),
        },
        "relative_action": {
            "left_arm": {name: [float(index) for index in range(5)] for name in ("min", "max", "mean", "std", "q01", "q99")},
            "left_gripper": {name: [0.0] for name in ("min", "max", "mean", "std", "q01", "q99")},
            "right_arm": {name: [float(index) for index in range(5)] for name in ("min", "max", "mean", "std", "q01", "q99")},
            "right_gripper": {name: [0.0] for name in ("min", "max", "mean", "std", "q01", "q99")},
        },
    }}
    _write(normalization, {"schema_version": 3, "train_only": True, "derivation": {"train_window_ids": [f"bc-{index}" for index in range(7)] + [f"rollout-{index}" for index in range(len(attempts))], "sample_count": 16 * (7 + len(attempts))}, "statistics": statistics})
    sources = [
        {"source_id": "bc", "source_type": "bc", "quota": 7, "release_stage": "seen", "source_tree_sha256": source_tree_sha256(bc), "artifact_receipt_path": "receipt.json", "artifact_receipt_sha256": _sha_path(bc / "receipt.json"), "acceptance_receipt_path": "acceptance.json", "acceptance_receipt_sha256": _sha_path(bc / "acceptance.json"), "publication": {"repository": "ryanjin333/lehome-groot-n17-data", "revision": "b" * 40, "prefix": "bc/full", "readback_receipt_path": str(bc_publication), "readback_receipt_sha256": _sha_path(bc_publication)}, "source_identity": {"prepared_manifest_path": "manifest.json", "prepared_manifest_sha256": _sha_path(bc / "manifest.json"), "action_source": "organizer_expert"}},
        {"source_id": "round-1", "source_type": "rollout", "quota": 3, "release_stage": "seen", "source_tree_sha256": source_tree_sha256(round_root), "artifact_receipt_path": "receipt.json", "artifact_receipt_sha256": _sha_path(round_root / "receipt.json"), "acceptance_receipt_path": "acceptance.json", "acceptance_receipt_sha256": _sha_path(round_root / "acceptance.json"), "publication": {"repository": "ryanjin333/lehome-groot-n17-rollouts", "revision": "c" * 40, "prefix": "rollouts/round-1", "readback_receipt_path": str(rollout_publication), "readback_receipt_sha256": _sha_path(rollout_publication)}, "source_identity": {"round_manifest_path": "round.json", "round_manifest_sha256": "a" * 64, "action_source": "policy"}},
    ]
    _write(round_root / "round.json", {"round_id": "round-1", "accepted_attempt_ids": [f"attempt-{index}" for index in range(len(attempts))]})
    sources[1]["source_identity"]["round_manifest_sha256"] = _sha_path(round_root / "round.json")
    # The round identity file arrived after the source tree was first measured.
    sources[1]["source_tree_sha256"] = source_tree_sha256(round_root)
    windows = []
    for episode in range(7):
        windows.append({"window_id": f"bc-{episode}", "source_id": "bc", "source_type": "bc", "source_episode_id": str(episode), "start": 0, "stop": 16, "frame_ids": list(range(16)), "lineage_id": f"bc-{episode}", "split": "train", "source_locator": {"episode_id": str(episode), "prepared_manifest_path": "manifest.json", "prepared_manifest_sha256": _sha_path(bc / "manifest.json")}})
    for index, (attempt_root, attempt_hash) in enumerate(attempts):
        windows.append({"window_id": f"rollout-{index}", "source_id": "round-1", "source_type": "rollout", "source_episode_id": f"attempt-{index}", "start": 0, "stop": 16, "frame_ids": list(range(16)), "lineage_id": f"rollout-{index}", "split": "train", "source_locator": {"attempt_root": attempt_root, "attempt_manifest_path": f"{attempt_root}/episode.json", "attempt_manifest_sha256": attempt_hash}})
    mixture_id = "d" * 64
    manifest = {"schema_version": 2, "kind": "lehome_runtime_mixture", "repository": "ryanjin333/lehome-groot-n17-rollouts", "safe_prefix": f"mixtures/{mixture_id}", "mixture_id": mixture_id, "sources": sources, "camera_schema": ["observation.images.top_rgb", "observation.images.left_rgb", "observation.images.right_rgb"], "image_shape": [480, 640, 3], "state_schema": {"dimension": 12, "storage": "absolute"}, "action_schema": {"dimension": 12, "storage": "absolute"}, "fps": 30, "action_horizon": 16, "instruction": "fold the garment on the table", "schedule_seed": 17, "cycle_size": 10, "mixture_normalization": {"path": "mixture-normalization.json", "sha256": sha256_file(normalization), "byte_size": normalization.stat().st_size}, "window_index": {"path": "windows.json", "sha256": "", "byte_size": 0}}
    index = {"schema_version": 2, "manifest_sha256": canonical_json_sha256(manifest), "windows": windows}
    index_path = tmp_path / "windows.json"
    _write(index_path, index)
    manifest["window_index"] = {"path": "windows.json", "sha256": sha256_file(index_path), "byte_size": index_path.stat().st_size}
    manifest_path = tmp_path / "mixture.json"
    _write(manifest_path, manifest)
    release_receipt = tmp_path / "release-receipt.json"
    _write(release_receipt, {"repository": "ryanjin333/lehome-groot-n17-rollouts", "immutable_revision": "a" * 40, "remote_prefix": f"mixtures/{mixture_id}", "mixture_id": mixture_id, "pending_receipt_sha256": "e" * 64, "artifact_entries": [{"relative_path": relative, "sha256": _sha_path(tmp_path / relative), "byte_size": (tmp_path / relative).stat().st_size} for relative in ("mixture.json", "windows.json", "mixture-normalization.json")], "fresh_readback_verified": True, "tree_listing_verified": True})
    mounts = tmp_path / "mounts.json"
    _write(mounts, {"schema_version": 2, "repository": "ryanjin333/lehome-groot-n17-rollouts", "safe_prefix": f"mixtures/{mixture_id}", "deployment_receipt_path": str(release_receipt), "deployment_receipt_sha256": _sha_path(release_receipt), "mounts": [{"source_id": source["source_id"], "root": str(root), "source_tree_sha256": source["source_tree_sha256"], "artifact_receipt_sha256": source["artifact_receipt_sha256"]} for source, root in zip(sources, (bc, round_root), strict=True)]})
    return manifest_path, index_path, mounts


def test_round_source_identity_is_root_level_and_windows_authenticate_attempts(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import load_runtime_contract

    manifest, _index, mounts = _contract(tmp_path)
    contract = load_runtime_contract(manifest, mounts)
    assert contract.manifest.sources[1].source_identity["round_manifest_path"] == "round.json"
    assert contract.windows[-1].source_locator["attempt_root"] == "attempts/attempt-2"


def test_schema_v3_runtime_manifest_admits_manifest_bound_80_20_batch_cycle(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import RuntimeMixtureDataset, _manifest_digest_binding, load_runtime_contract, sha256_file

    manifest, index, mounts = _contract(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value.update({
        "schema_version": 3,
        "experiment_manifest_sha256": "f" * 64,
        "mixture_weights": {"bc": 80, "rollout": 20, "dagger": 0},
        "source_quotas": {"bc": 51, "rollout": 13, "dagger": 0},
        "cycle_size": 64,
    })
    value["sources"][0]["quota"] = 51
    value["sources"][1]["quota"] = 13
    index_value = json.loads(index.read_text(encoding="utf-8"))
    index_value["manifest_sha256"] = _manifest_digest_binding(value)
    _write(index, index_value)
    value["window_index"].update({"sha256": sha256_file(index), "byte_size": index.stat().st_size})
    _write(manifest, value)
    deployment = tmp_path / "release-receipt.json"
    receipt = json.loads(deployment.read_text(encoding="utf-8"))
    receipt.update({
        "experiment_manifest_sha256": "f" * 64,
        "mixture_weights": {"bc": 80, "rollout": 20, "dagger": 0},
        "source_quotas": {"bc": 51, "rollout": 13, "dagger": 0},
    })
    for entry in receipt["artifact_entries"]:
        target = tmp_path / entry["relative_path"]
        entry.update({"sha256": _sha_path(target), "byte_size": target.stat().st_size})
    _write(deployment, receipt)
    mounts_value = json.loads(mounts.read_text(encoding="utf-8"))
    mounts_value["deployment_receipt_sha256"] = _sha_path(deployment)
    _write(mounts, mounts_value)

    contract = load_runtime_contract(manifest, mounts)

    assert contract.manifest.experiment_manifest_sha256 == "f" * 64
    assert contract.manifest.quotas == {"bc": 51, "rollout": 13, "dagger": 0}
    samples = list(RuntimeMixtureDataset(contract, limit=128))
    assert Counter(sample.source_type for sample in samples[:64]) == {"bc": 51, "rollout": 13}
    assert Counter(sample.source_type for sample in samples[64:]) == {"bc": 51, "rollout": 13}
    resumed = RuntimeMixtureDataset(
        contract, expected_global_step=1, global_batch_size=64, limit=128,
    )
    assert Counter(sample.source_type for sample in resumed) == {"bc": 51, "rollout": 13}
    with pytest.raises(ValueError, match="batch-aligned"):
        RuntimeMixtureDataset(contract, expected_global_step=1, global_batch_size=10)

    original_receipt = json.loads(deployment.read_text(encoding="utf-8"))
    for key, changed in {
        "experiment_manifest_sha256": "e" * 64,
        "mixture_weights": {"bc": 70, "rollout": 30, "dagger": 0},
        "source_quotas": {"bc": 45, "rollout": 19, "dagger": 0},
    }.items():
        _write(deployment, {**original_receipt, key: changed})
        mounts_value["deployment_receipt_sha256"] = _sha_path(deployment)
        _write(mounts, mounts_value)
        with pytest.raises(ValueError, match="experiment binding"):
            load_runtime_contract(manifest, mounts)


def test_schema_v3_runtime_manifest_admits_the_pure_bc_batch_cycle(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import (
        RuntimeMixtureDataset,
        _manifest_digest_binding,
        load_runtime_contract,
        sha256_file,
    )

    manifest, index, mounts = _contract(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value.update({
        "schema_version": 3,
        "experiment_manifest_sha256": "f" * 64,
        "mixture_weights": {"bc": 100, "rollout": 0, "dagger": 0},
        "source_quotas": {"bc": 64, "rollout": 0, "dagger": 0},
        "cycle_size": 64,
    })
    value["sources"][0]["quota"] = 64
    value["sources"][1]["quota"] = 0
    index_value = json.loads(index.read_text(encoding="utf-8"))
    index_value["manifest_sha256"] = _manifest_digest_binding(value)
    _write(index, index_value)
    value["window_index"].update({"sha256": sha256_file(index), "byte_size": index.stat().st_size})
    _write(manifest, value)
    deployment = tmp_path / "release-receipt.json"
    receipt = json.loads(deployment.read_text(encoding="utf-8"))
    receipt.update({
        "experiment_manifest_sha256": "f" * 64,
        "mixture_weights": {"bc": 100, "rollout": 0, "dagger": 0},
        "source_quotas": {"bc": 64, "rollout": 0, "dagger": 0},
    })
    for entry in receipt["artifact_entries"]:
        target = tmp_path / entry["relative_path"]
        entry.update({"sha256": _sha_path(target), "byte_size": target.stat().st_size})
    _write(deployment, receipt)
    mounts_value = json.loads(mounts.read_text(encoding="utf-8"))
    mounts_value["deployment_receipt_sha256"] = _sha_path(deployment)
    _write(mounts, mounts_value)

    contract = load_runtime_contract(manifest, mounts)

    assert contract.manifest.quotas == {"bc": 64, "rollout": 0, "dagger": 0}
    assert Counter(sample.source_type for sample in RuntimeMixtureDataset(contract, limit=64)) == {"bc": 64}


def test_local_mount_can_override_only_the_external_source_readback_receipt_paths(tmp_path: Path) -> None:
    """Hydration must not rewrite immutable mixture bytes just to relocate receipts."""
    from lehome_train.groot.runtime_mixture import _manifest_digest_binding, load_runtime_contract, sha256_file

    manifest, _index, mounts = _contract(tmp_path)
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    mounts_value = json.loads(mounts.read_text(encoding="utf-8"))
    for source, entry in zip(manifest_value["sources"], mounts_value["mounts"], strict=True):
        publication = source["publication"]
        external = Path(publication["readback_receipt_path"])
        publication["readback_receipt_path"] = "/unavailable/authoring-host/receipt.json"
        entry["source_readback_receipt_path"] = str(external)
        entry["source_readback_receipt_sha256"] = _sha_path(external)
    index = tmp_path / "windows.json"
    index_value = json.loads(index.read_text(encoding="utf-8"))
    index_value["manifest_sha256"] = _manifest_digest_binding(manifest_value)
    _write(index, index_value)
    manifest_value["window_index"]["sha256"] = sha256_file(index)
    manifest_value["window_index"]["byte_size"] = index.stat().st_size
    _write(manifest, manifest_value)
    deployment = tmp_path / "release-receipt.json"
    deployment_value = json.loads(deployment.read_text(encoding="utf-8"))
    for artifact in deployment_value["artifact_entries"]:
        target = tmp_path / artifact["relative_path"]
        artifact.update({"sha256": _sha_path(target), "byte_size": target.stat().st_size})
    _write(deployment, deployment_value)
    mounts_value["deployment_receipt_sha256"] = _sha_path(deployment)
    _write(mounts, mounts_value)

    assert load_runtime_contract(manifest, mounts).mounts["bc"] == tmp_path / "bc"


def test_contract_rejects_sources_without_individual_immutable_publication_bindings(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import _manifest_digest_binding, load_runtime_contract, sha256_file

    manifest, index, mounts = _contract(tmp_path)
    manifest_payload = json.loads(manifest.read_text())
    del manifest_payload["sources"][0]["publication"]
    index_payload = json.loads(index.read_text())
    index_payload["manifest_sha256"] = _manifest_digest_binding(manifest_payload)
    _write(index, index_payload)
    manifest_payload["window_index"]["sha256"] = sha256_file(index)
    manifest_payload["window_index"]["byte_size"] = index.stat().st_size
    _write(manifest, manifest_payload)

    with pytest.raises(ValueError, match="publication"):
        load_runtime_contract(manifest, mounts)


def test_dataset_exposes_exact_pinned_statistics_payload(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import RuntimeMixtureDataset, load_runtime_contract

    contract = load_runtime_contract(*(_contract(tmp_path)[::2]))

    assert set(RuntimeMixtureDataset(contract).get_dataset_statistics()) == {
        "new_embodiment",
    }


def test_statistics_integrate_with_the_pinned_state_action_processor(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import RuntimeMixtureDataset, load_runtime_contract

    pinned = "/private/tmp/lehome-corrective-live/final-run-c105615/persistent-training/Isaac-GR00T"
    if not Path(pinned).is_dir():
        pytest.skip("pinned Isaac-GR00T checkout is unavailable")
    # The checked-in processor only needs scipy for pose operations, which this
    # statistics-only integration does not execute. Keep the actual pinned
    # StateActionProcessor while stubbing that unavailable optional dependency.
    scipy = types.ModuleType("scipy")
    spatial = types.ModuleType("scipy.spatial")
    transform = types.ModuleType("scipy.spatial.transform")
    interpolate = types.ModuleType("scipy.interpolate")
    transform.Rotation = type("Rotation", (), {})
    transform.Slerp = type("Slerp", (), {})
    scipy.spatial = spatial
    spatial.transform = transform
    scipy.interpolate = interpolate
    sys.modules.setdefault("scipy", scipy)
    sys.modules.setdefault("scipy.spatial", spatial)
    sys.modules.setdefault("scipy.spatial.transform", transform)
    sys.modules.setdefault("scipy.interpolate", interpolate)
    sys.path.insert(0, pinned)
    try:
        from gr00t.data.state_action.state_action_processor import StateActionProcessor
    finally:
        sys.path.remove(pinned)
    contract = load_runtime_contract(*(_contract(tmp_path)[::2]))
    processor = StateActionProcessor(modality_configs={"new_embodiment": {
        "state": {"delta_indices": [0], "modality_keys": ["left_arm", "left_gripper", "right_arm", "right_gripper"]},
        "action": {"delta_indices": list(range(16)), "modality_keys": ["left_arm", "left_gripper", "right_arm", "right_gripper"]},
    }})

    processor.set_statistics(RuntimeMixtureDataset(contract).get_dataset_statistics(), override=True)

    assert processor.norm_params["new_embodiment"]["action"]["right_arm"]["min"].shape == (5,)


def test_contract_rejects_normalization_not_derived_from_exact_train_windows(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import _manifest_digest_binding, load_runtime_contract, sha256_file

    manifest, index, mounts = _contract(tmp_path)
    normalization = manifest.parent / "mixture-normalization.json"
    payload = json.loads(normalization.read_text())
    payload["derivation"]["sample_count"] = 159
    _write(normalization, payload)
    manifest_payload = json.loads(manifest.read_text())
    manifest_payload["mixture_normalization"]["sha256"] = sha256_file(normalization)
    manifest_payload["mixture_normalization"]["byte_size"] = normalization.stat().st_size
    index_payload = json.loads(index.read_text())
    index_payload["manifest_sha256"] = _manifest_digest_binding(manifest_payload)
    _write(index, index_payload)
    manifest_payload["window_index"]["sha256"] = sha256_file(index)
    manifest_payload["window_index"]["byte_size"] = index.stat().st_size
    _write(manifest, manifest_payload)

    with pytest.raises(ValueError, match="normalization derivation"):
        load_runtime_contract(manifest, mounts)


def test_video_probe_cache_is_by_video_identity_and_checks_each_window_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lehome_train.groot.runtime_mixture import RangeSourceLoader, load_runtime_contract

    contract = load_runtime_contract(*(_contract(tmp_path)[::2]))
    loader = RangeSourceLoader(contract, decoder=lambda *_: None)
    path = tmp_path / "shared.mp4"
    calls: list[Path] = []
    monkeypatch.setattr(
        "lehome_train.groot.runtime_mixture._video_probe",
        lambda candidate: calls.append(candidate) or {"fps": 30, "frame_count": 17},
    )

    first = loader._video_metadata(path)
    second = loader._video_metadata(path)
    loader._validate_video_stop(first, 17)

    assert first == second == {"fps": 30, "frame_count": 17}
    assert calls == [path]
    with pytest.raises(ValueError, match="short"):
        loader._validate_video_stop(second, 18)


def test_schedule_resume_and_worker_partitions_keep_exact_seven_three_ratio(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import RuntimeMixtureDataset, load_runtime_contract

    contract = load_runtime_contract(*(_contract(tmp_path)[::2]))
    uninterrupted = list(RuntimeMixtureDataset(contract, limit=40))
    resumed = list(RuntimeMixtureDataset(contract, global_sample_offset=13, limit=40))
    assert [sample.sample_id for sample in resumed] == [sample.sample_id for sample in uninterrupted[13:]]
    assert Counter(sample.source_type for sample in uninterrupted) == {"bc": 28, "rollout": 12}
    aggregate = [sample for worker in range(4) for sample in RuntimeMixtureDataset(contract, worker_id=worker, worker_count=4, limit=40)]
    assert sorted(sample.global_position for sample in aggregate) == list(range(40))


@pytest.mark.parametrize("workers", (1, 4))
def test_authenticated_resume_seed_resets_to_exact_global_batch_offset(tmp_path: Path, workers: int) -> None:
    from lehome_train.groot.runtime_mixture import RuntimeMixtureDataset, load_runtime_contract

    contract = load_runtime_contract(*(_contract(tmp_path)[::2]))
    uninterrupted = list(RuntimeMixtureDataset(contract, limit=160))
    resumed = RuntimeMixtureDataset(
        contract, expected_global_step=10, global_batch_size=10, limit=160,
        worker_id=0, worker_count=workers,
    )
    assert resumed.seed == contract.manifest.schedule_seed
    resumed.reset_seed(contract.manifest.schedule_seed + 10)
    assert resumed.global_sample_offset == 100
    assert [item.sample_id for item in resumed] == [
        item.sample_id for item in uninterrupted[100::workers]
    ]
    with pytest.raises(ValueError, match="authenticated checkpoint"):
        resumed.reset_seed(contract.manifest.schedule_seed + 9)


def test_pinned_gr00t_trainer_get_train_dataloader_resets_the_authenticated_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lehome_train.groot.runtime_mixture import RuntimeMixtureDataset, load_runtime_contract

    pinned = Path("/private/tmp/lehome-corrective-live/final-run-c105615/persistent-training/Isaac-GR00T/gr00t/experiment/trainer.py")
    if not pinned.is_file():
        pytest.skip("pinned Isaac-GR00T trainer is unavailable")
    captured: dict[str, object] = {}
    torch = types.ModuleType("torch")
    torch.utils = types.SimpleNamespace(data=types.SimpleNamespace(DataLoader=lambda dataset, **kwargs: captured.update(dataset=dataset, kwargs=kwargs) or "pinned-loader"))
    trainer_module = types.ModuleType("transformers.trainer")
    trainer_module.TRAINER_STATE_NAME = "trainer_state.json"
    trainer_module.Trainer = type("Trainer", (), {})
    trainer_module.TrainerState = type("TrainerState", (), {})
    trainer_module.get_last_checkpoint = lambda _path: None
    callback_module = types.ModuleType("transformers.trainer_callback")
    callback_module.TrainerCallback = object
    utils_module = types.ModuleType("transformers.trainer_utils")
    utils_module.EvalPrediction = object
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    monkeypatch.setitem(sys.modules, "transformers.trainer", trainer_module)
    monkeypatch.setitem(sys.modules, "transformers.trainer_callback", callback_module)
    monkeypatch.setitem(sys.modules, "transformers.trainer_utils", utils_module)
    spec = importlib.util.spec_from_file_location("pinned_gr00t_trainer", pinned)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = load_runtime_contract(*(_contract(tmp_path)[::2]))
    dataset = RuntimeMixtureDataset(contract, expected_global_step=10, global_batch_size=10)
    trainer = object.__new__(module.Gr00tTrainer)
    trainer.args = types.SimpleNamespace(ignore_data_skip=False, dataloader_num_workers=0, dataloader_pin_memory=False)
    trainer.state = types.SimpleNamespace(global_step=10)
    trainer.train_dataset = dataset
    trainer.data_collator = lambda value: value
    trainer._get_collator_with_removed_columns = lambda value, **_kwargs: value
    trainer._train_batch_size = 1

    assert trainer.get_train_dataloader() == "pinned-loader"
    assert dataset.global_sample_offset == 100
    assert captured["dataset"] is dataset


@pytest.mark.parametrize("workers", (0, 1, 4))
def test_torch_dataloader_workers_and_prefetch_preserve_global_positions(tmp_path: Path, workers: int) -> None:
    torch = pytest.importorskip("torch")
    from lehome_train.groot.runtime_mixture import RuntimeMixtureDataset, load_runtime_contract

    contract = load_runtime_contract(*(_contract(tmp_path)[::2]))
    options: dict[str, object] = {"batch_size": None, "num_workers": workers}
    if workers:
        options["prefetch_factor"] = 2
    output = list(torch.utils.data.DataLoader(RuntimeMixtureDataset(contract, limit=40), **options))
    assert [sample.global_position for sample in output] == list(range(40))
    assert Counter(sample.source_type for sample in output) == {"bc": 28, "rollout": 12}


def test_rollout_loader_rejects_checksum_or_annotation_drift(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import RangeSourceLoader, load_runtime_contract

    manifest, _index, mounts = _contract(tmp_path)
    contract = load_runtime_contract(manifest, mounts)
    window = next(item for item in contract.windows if item.source_type == "rollout")
    attempt = tmp_path / "round-1" / window.source_locator["attempt_root"]
    (attempt / "annotations.jsonl").write_text('{"step":0,"action_source":"expert"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash|checksum|annotation|short"):
        RangeSourceLoader(contract, decoder=lambda *_: None).load(window)


def test_rollout_loader_parses_canonical_jsonl_annotations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lehome_train.groot.runtime_mixture import RangeSourceLoader, load_runtime_contract

    manifest, _index, mounts = _contract(tmp_path)
    contract = load_runtime_contract(manifest, mounts)
    monkeypatch.setattr("lehome_train.groot.runtime_mixture._video_probe", lambda *_args, **_kwargs: {"fps": 30, "frame_count": 16})
    window = next(item for item in contract.windows if item.source_type == "rollout")

    payload = RangeSourceLoader(contract, decoder=lambda *_: None).load(window)

    assert len(payload["actions"]) == 16


def test_manifest_accepts_only_the_campaign_private_repository(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import load_runtime_contract

    manifest, _index, mounts = _contract(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["repository"] = "private/lehome-training"
    _write(manifest, payload)
    with pytest.raises(ValueError, match="approved private repository"):
        load_runtime_contract(manifest, mounts)


def test_mount_descriptor_requires_a_fresh_readback_of_the_exact_runtime_release(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import load_runtime_contract

    manifest, _index, mounts = _contract(tmp_path)
    descriptor = json.loads(mounts.read_text())
    descriptor["safe_prefix"] = "mixtures/" + "b" * 64
    _write(mounts, descriptor)
    with pytest.raises(ValueError, match="mount release identity"):
        load_runtime_contract(manifest, mounts)


def test_provenance_rejects_bc_validation_or_unaccepted_rollout_windows(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import canonical_json_sha256, load_runtime_contract, source_tree_sha256

    manifest, index, mounts = _contract(tmp_path)
    prepared = tmp_path / "bc" / "manifest.json"
    document = json.loads(prepared.read_text())
    document["train_episode_ids"] = [str(index) for index in range(6)]
    document["validation_episode_ids"] = ["6"]
    _write(prepared, document)
    payload = json.loads(manifest.read_text())
    payload["sources"][0]["source_identity"]["prepared_manifest_sha256"] = _sha_path(prepared)
    payload["sources"][0]["source_tree_sha256"] = source_tree_sha256(tmp_path / "bc")
    binding = dict(payload)
    binding["window_index"] = {"path": "windows.json", "sha256": "", "byte_size": 0}
    index_payload = json.loads(index.read_text())
    index_payload["manifest_sha256"] = canonical_json_sha256(binding)
    _write(index, index_payload)
    payload["window_index"]["sha256"] = _sha_path(index)
    payload["window_index"]["byte_size"] = index.stat().st_size
    _write(manifest, payload)
    mounts_payload = json.loads(mounts.read_text())
    mounts_payload["mounts"][0]["source_tree_sha256"] = payload["sources"][0]["source_tree_sha256"]
    deployment = Path(mounts_payload["deployment_receipt_path"])
    deployment_payload = json.loads(deployment.read_text())
    deployment_payload["artifact_entries"] = [
        {"relative_path": relative, "sha256": _sha_path(tmp_path / relative), "byte_size": (tmp_path / relative).stat().st_size}
        for relative in ("mixture.json", "windows.json", "mixture-normalization.json")
    ]
    _write(deployment, deployment_payload)
    mounts_payload["deployment_receipt_sha256"] = _sha_path(deployment)
    _write(mounts, mounts_payload)
    with pytest.raises(ValueError, match="BC window split"):
        load_runtime_contract(manifest, mounts)


def test_real_accepted_attempt_checksum_receipt_when_available() -> None:
    from lehome_train.groot.runtime_mixture import RangeSourceLoader

    raw = Path("/private/tmp/lehome-corrective-live/final-run-c105615/campaign/raw")
    receipts = sorted(raw.glob("*/SHA256SUMS.json"))
    if not receipts:
        pytest.skip("accepted local corrective artifacts are unavailable")
    RangeSourceLoader._verify_checksums(object.__new__(RangeSourceLoader), receipts[0].parent)


def test_pinned_message_is_vla_step_wrapped_in_one_episode_step_message() -> None:
    import numpy as np
    from lehome_train.groot.runtime_mixture import pinned_processor_messages

    class FakeVLA:
        def __init__(self, **kwargs: object) -> None: self.kwargs = kwargs
    backend = type("Backend", (), {"VLAStepData": FakeVLA, "MessageType": type("M", (), {"EPISODE_STEP": type("E", (), {"value": "episode_step"})}), "EmbodimentTag": type("T", (), {"NEW_EMBODIMENT": "new"})})
    messages = pinned_processor_messages({"images": {"top_rgb": np.zeros((1, 2, 2, 3), dtype=np.uint8), "left_rgb": np.zeros((1, 2, 2, 3), dtype=np.uint8), "right_rgb": np.zeros((1, 2, 2, 3), dtype=np.uint8)}, "state": [0.0] * 12, "actions": [[0.0] * 12 for _ in range(16)]}, backend=backend)
    assert messages[0]["type"] == "episode_step"
    assert messages[0]["content"].kwargs["actions"]["left_arm"].dtype == np.float32
    assert messages[0]["content"].kwargs["states"]["left_arm"].shape == (1, 5)


def test_window_indexing_and_loader_caches_are_bounded(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import RangeSourceLoader, RuntimeMixtureDataset, load_runtime_contract

    contract = load_runtime_contract(*(_contract(tmp_path)[::2]))
    dataset = RuntimeMixtureDataset(contract, limit=40)
    assert len({sample.window.window_id for sample in islice(dataset, 40)}) <= 10
    loader = RangeSourceLoader(contract, decoder=lambda *_: None)
    for index in range(20):
        loader._cache(loader._attempt_cache, Path(f"/attempt/{index}"), None)
    assert len(loader._attempt_cache) == loader.cache_cap
    assert RuntimeMixtureDataset(contract).get_initial_actions() == []


def test_runtime_dataset_factory_reads_official_training_global_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lehome_train.groot import runtime_mixture as module

    class OfficialTraining:
        global_batch_size = 64

    class OfficialConfig:
        training = OfficialTraining()

    class Processor:
        def set_statistics(self, *_args: object, **_kwargs: object) -> None:
            return None

    monkeypatch.setattr(
        module,
        "make_dataset_factory",
        lambda **_kwargs: (lambda *args, processor=None, **kwargs: type("Dataset", (), {"get_dataset_statistics": staticmethod(lambda: {})})()),
    )
    factory_cls = module.runtime_dataset_factory_class(
        mixture_manifest="/prepared/runtime/mixture.json",
        window_index="/prepared/runtime/windows.json",
        mounts_descriptor="/prepared/runtime/mounts.json",
        global_sample_offset=0,
        expected_global_step=0,
        global_batch_size=64,
    )
    dataset, unused = factory_cls(OfficialConfig()).build(Processor())
    assert unused is None
    assert dataset is not None

    mismatch = factory_cls(type("Mismatch", (), {"training": type("T", (), {"global_batch_size": 8})()})())
    with pytest.raises(ValueError, match="pinned trainer global batch"):
        mismatch.build(processor=Processor())


def test_runtime_dataset_uses_bound_awr_evidence_as_deterministic_rollout_replay(
    tmp_path: Path,
) -> None:
    from lehome_train.groot.awr_weighting import (
        AwrReplayConfig,
        canonical_evidence_sha256,
        load_progress_evidence,
    )
    from lehome_train.groot.runtime_mixture import (
        RuntimeMixtureDataset,
        canonical_json_sha256,
        load_runtime_contract,
        make_dataset_factory,
    )

    manifest, _index, mounts = _contract(tmp_path)
    contract = load_runtime_contract(manifest, mounts)
    document = {
        "schema_version": 1,
        "kind": "lehome_awr_progress_evidence",
        "mixture_id": contract.manifest.mixture_id,
        "mixture_manifest_sha256": canonical_json_sha256(contract.manifest.raw),
        "episodes": [
            {
                "episode_id": "attempt-0", "lineage_id": "rollout-0", "split": "train",
                "score_kind": "progress", "score": -1.0,
                "provenance_path": "receipts/attempt-0.json", "provenance_sha256": "a" * 64,
            },
            {
                "episode_id": "attempt-1", "lineage_id": "rollout-1", "split": "train",
                "score_kind": "progress", "score": 0.0,
                "provenance_path": "receipts/attempt-1.json", "provenance_sha256": "b" * 64,
            },
            {
                "episode_id": "attempt-2", "lineage_id": "rollout-2", "split": "train",
                "score_kind": "advantage", "score": 2.0,
                "provenance_path": "receipts/attempt-2.json", "provenance_sha256": "c" * 64,
            },
        ],
    }
    evidence_path = tmp_path / "awr-progress.json"
    _write(evidence_path, document)
    evidence = load_progress_evidence(
        evidence_path,
        expected_sha256=canonical_evidence_sha256(document),
        mixture_id=contract.manifest.mixture_id,
        mixture_manifest_sha256=canonical_json_sha256(contract.manifest.raw),
    )

    disabled = list(RuntimeMixtureDataset(contract, limit=200))
    weighted_factory = make_dataset_factory(
        mixture_manifest=manifest,
        mounts_descriptor=mounts,
        awr_evidence=evidence,
        awr_config=AwrReplayConfig(temperature=1.0, minimum=0.5, maximum=2.0),
    )
    weighted = list(islice(weighted_factory(), 20_000))
    unweighted = list(islice(RuntimeMixtureDataset(contract), 20_000))

    assert [sample.sample_id for sample in disabled] == [
        sample.sample_id for sample in RuntimeMixtureDataset(contract, limit=200)
    ]
    rollout_counts = Counter(
        sample.window.source_episode_id for sample in weighted if sample.source_type == "rollout"
    )
    assert rollout_counts["attempt-2"] > rollout_counts["attempt-1"] > rollout_counts["attempt-0"]
    assert [sample.window.window_id for sample in weighted if sample.source_type == "bc"] == [
        sample.window.window_id for sample in unweighted if sample.source_type == "bc"
    ]


def test_awr_replay_balances_categories_before_weighting_within_category(tmp_path: Path) -> None:
    from lehome_train.groot.awr_weighting import (
        AwrReplayConfig,
        canonical_evidence_sha256,
        load_progress_evidence,
    )
    from lehome_train.groot.runtime_mixture import (
        RuntimeMixtureDataset,
        canonical_json_sha256,
        load_runtime_contract,
    )

    categories = (
        "top_long",
        "top_short",
        "pant_long",
        "pant_short",
        "pant_short",
        "pant_short",
    )
    manifest, _index, mounts = _contract(tmp_path, rollout_categories=categories)
    contract = load_runtime_contract(manifest, mounts)
    document = {
        "schema_version": 1,
        "kind": "lehome_awr_progress_evidence",
        "mixture_id": contract.manifest.mixture_id,
        "mixture_manifest_sha256": canonical_json_sha256(contract.manifest.raw),
        "episodes": [
            {
                "episode_id": f"attempt-{index}",
                "lineage_id": f"rollout-{index}",
                "split": "train",
                "score_kind": "progress",
                "score": 0.0,
                "provenance_path": f"receipts/attempt-{index}.json",
                "provenance_sha256": f"{index + 1:x}" * 64,
            }
            for index in range(len(categories))
        ],
    }
    path = tmp_path / "awr-balanced-progress.json"
    _write(path, document)
    evidence = load_progress_evidence(
        path,
        expected_sha256=canonical_evidence_sha256(document),
        mixture_id=contract.manifest.mixture_id,
        mixture_manifest_sha256=canonical_json_sha256(contract.manifest.raw),
    )

    samples = islice(
        RuntimeMixtureDataset(
            contract,
            awr_evidence=evidence,
            awr_config=AwrReplayConfig(temperature=1.0, minimum=0.5, maximum=2.0),
        ),
        100_000,
    )
    by_category = Counter(
        categories[int(sample.window.source_episode_id.removeprefix("attempt-"))]
        for sample in samples
        if sample.source_type == "rollout"
    )

    assert set(by_category) == {"top_long", "top_short", "pant_long", "pant_short"}
    assert max(by_category.values()) / min(by_category.values()) < 1.05


def test_runtime_dataset_fails_closed_for_missing_awr_rollout_evidence(tmp_path: Path) -> None:
    from lehome_train.groot.awr_weighting import (
        AwrReplayConfig,
        canonical_evidence_sha256,
        load_progress_evidence,
    )
    from lehome_train.groot.runtime_mixture import RuntimeMixtureDataset, canonical_json_sha256, load_runtime_contract

    manifest, _index, mounts = _contract(tmp_path)
    contract = load_runtime_contract(manifest, mounts)
    document = {
        "schema_version": 1, "kind": "lehome_awr_progress_evidence",
        "mixture_id": contract.manifest.mixture_id,
        "mixture_manifest_sha256": canonical_json_sha256(contract.manifest.raw),
        "episodes": [{
            "episode_id": "attempt-0", "lineage_id": "rollout-0", "split": "train",
            "score_kind": "progress", "score": 0.0,
            "provenance_path": "receipts/attempt-0.json", "provenance_sha256": "a" * 64,
        }],
    }
    path = tmp_path / "awr-progress.json"
    _write(path, document)
    evidence = load_progress_evidence(
        path, expected_sha256=canonical_evidence_sha256(document),
        mixture_id=contract.manifest.mixture_id,
        mixture_manifest_sha256=canonical_json_sha256(contract.manifest.raw),
    )

    with pytest.raises(ValueError, match="missing AWR evidence"):
        RuntimeMixtureDataset(
            contract,
            awr_evidence=evidence,
            awr_config=AwrReplayConfig(temperature=1.0, minimum=0.5, maximum=2.0),
        )
