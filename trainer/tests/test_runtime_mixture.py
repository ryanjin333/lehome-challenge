from __future__ import annotations

import json
from collections import Counter
from itertools import islice
from pathlib import Path

import pytest


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _source(root: Path, source_id: str, source_type: str, *, episode: str = "0") -> dict[str, object]:
    root.mkdir()
    (root / "receipt.json").write_text('{"accepted":true}', encoding="utf-8")
    (root / "acceptance.json").write_text('{"accepted_success":true}', encoding="utf-8")
    (root / "normalization.json").write_text('{"train_only":true}', encoding="utf-8")
    return {
        "source_id": source_id,
        "source_type": source_type,
        "quota": 7 if source_type == "bc" else 3,
        "release_stage": "seen",
        "source_tree_sha256": "",  # filled after all source files are written
        "artifact_receipt_path": "receipt.json",
        "artifact_receipt_sha256": _sha('{"accepted":true}'),
        "acceptance_receipt_path": "acceptance.json",
        "acceptance_receipt_sha256": _sha('{"accepted_success":true}'),
        "normalization_artifact_path": "normalization.json",
        "normalization_artifact_sha256": _sha('{"train_only":true}'),
        "source_identity": (
            {"prepared_manifest_path": "manifest.json", "prepared_episode_id": episode, "action_source": "organizer_expert"}
            if source_type == "bc"
            else {"attempt_manifest_path": "episode.json", "attempt_id": f"attempt-{episode}", "action_source": "policy"}
        ),
    }


def _contract(tmp_path: Path, *, rollout_windows: int = 3, validation: bool = False) -> tuple[Path, Path, Path]:
    from lehome_train.groot.runtime_mixture import canonical_json_sha256, source_tree_sha256

    bc_root, rollout_root = tmp_path / "bc", tmp_path / "rollout"
    sources = [_source(bc_root, "bc", "bc"), _source(rollout_root, "rollout", "rollout")]
    for source, root in zip(sources, (bc_root, rollout_root), strict=True):
        source["source_tree_sha256"] = source_tree_sha256(root)
    windows: list[dict[str, object]] = []
    for source, count in ((sources[0], 7), (sources[1], rollout_windows)):
        for number in range(count):
            kind = source["source_type"]
            windows.append({
                "window_id": f"{source['source_id']}-{number}", "source_id": source["source_id"], "source_type": kind,
                "source_episode_id": str(number), "start": 0, "stop": 16, "frame_ids": list(range(16)),
                "lineage_id": f"{source['source_id']}-lineage-{number}", "split": "validation" if validation and number == 0 else "train",
                "source_identity": (
                    {"prepared_episode_id": str(number), "prepared_range_start": 0, "prepared_range_stop": 16}
                    if kind == "bc" else {"attempt_id": f"attempt-{number}", "attempt_manifest_sha256": "a" * 64, "accepted_success": True}
                ),
            })
    index = {"schema_version": 1, "manifest_sha256": "", "windows": windows}
    index_path = tmp_path / "windows.json"
    manifest = {
        "schema_version": 1, "kind": "lehome_runtime_mixture", "repository": "private/lehome-training", "revision": "a" * 40,
        "safe_prefix": "runtime-mixtures/phase-1", "sources": sources,
        "camera_schema": ["observation.images.top_rgb", "observation.images.left_rgb", "observation.images.right_rgb"],
        "image_shape": [480, 640, 3], "state_schema": {"dimension": 12, "storage": "absolute"},
        "action_schema": {"dimension": 12, "storage": "absolute"}, "fps": 30, "action_horizon": 16,
        "instruction": "fold the garment on the table", "schedule_seed": 17, "cycle_size": 10,
        "window_index": {"path": "windows.json", "sha256": "", "byte_size": 0},
    }
    manifest_path = tmp_path / "mixture.json"
    # The index binds the canonical manifest excluding the mutable artifact binding.
    index["manifest_sha256"] = canonical_json_sha256(manifest)
    index_path.write_text(json.dumps(index, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    from lehome_train.groot.runtime_mixture import sha256_file
    manifest["window_index"] = {"path": "windows.json", "sha256": sha256_file(index_path), "byte_size": index_path.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    mounts_path = tmp_path / "mounts.json"
    mounts_path.write_text(json.dumps({"schema_version": 1, "mounts": [
        {"source_id": "bc", "root": str(bc_root), "source_tree_sha256": sources[0]["source_tree_sha256"], "artifact_receipt_sha256": sources[0]["artifact_receipt_sha256"]},
        {"source_id": "rollout", "root": str(rollout_root), "source_tree_sha256": sources[1]["source_tree_sha256"], "artifact_receipt_sha256": sources[1]["artifact_receipt_sha256"]},
    ]}), encoding="utf-8")
    return manifest_path, index_path, mounts_path


def test_contract_rejects_tamper_unknown_paths_and_lineage_overlap(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import load_runtime_contract

    manifest, index, mounts = _contract(tmp_path)
    contract = load_runtime_contract(manifest, mounts)
    assert contract.manifest.action_horizon == 16
    payload = json.loads(index.read_text())
    payload["windows"][0]["frame_ids"][1] = 9
    index.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="window index hash"):
        load_runtime_contract(manifest, mounts)


def test_schedule_is_exact_resumeable_and_worker_partition_safe(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import RuntimeMixtureDataset, load_runtime_contract

    manifest, _index, mounts = _contract(tmp_path)
    contract = load_runtime_contract(manifest, mounts)
    expected = [sample.sample_id for sample in RuntimeMixtureDataset(contract, decoder=lambda *_: {}, limit=100)]
    resumed = [sample.sample_id for sample in RuntimeMixtureDataset(contract, decoder=lambda *_: {}, global_sample_offset=13, limit=100)]
    assert resumed == expected[13:]
    first_cycle = [sample.source_type for sample in RuntimeMixtureDataset(contract, decoder=lambda *_: {}, limit=10)]
    assert Counter(first_cycle) == {"bc": 7, "rollout": 3}
    partitions = [
        [sample.sample_id for sample in RuntimeMixtureDataset(contract, decoder=lambda *_: {}, limit=40, worker_id=worker, worker_count=4)]
        for worker in range(4)
    ]
    combined = [item for partition in partitions for item in partition]
    assert [item for _, item in sorted((int(item.split(":", 1)[0]), item) for item in combined)] == expected[:40]


def test_contract_rejects_validation_and_cross_episode_tail_windows(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import load_runtime_contract

    manifest, index, mounts = _contract(tmp_path)
    document = json.loads(index.read_text())
    document["windows"][0]["stop"] = 17
    document["windows"][0]["frame_ids"] = list(range(17))
    # Refresh the binding so this tests the content validation, not only hashing.
    index.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    from lehome_train.groot.runtime_mixture import sha256_file
    root = json.loads(manifest.read_text())
    root["window_index"]["sha256"] = sha256_file(index)
    root["window_index"]["byte_size"] = index.stat().st_size
    manifest.write_text(json.dumps(root), encoding="utf-8")
    with pytest.raises(ValueError, match="horizon|tail"):
        load_runtime_contract(manifest, mounts)


@pytest.mark.parametrize("workers", (0, 1, 4))
def test_actual_dataloader_workers_preserve_global_schedule_and_resume(tmp_path: Path, workers: int) -> None:
    torch = pytest.importorskip("torch")
    from lehome_train.groot.runtime_mixture import RuntimeMixtureDataset, load_runtime_contract

    manifest, _index, mounts = _contract(tmp_path)
    contract = load_runtime_contract(manifest, mounts)
    dataset = RuntimeMixtureDataset(contract, limit=40)
    options: dict[str, object] = {"batch_size": None, "num_workers": workers}
    if workers:
        options["prefetch_factor"] = 2
    output = list(torch.utils.data.DataLoader(dataset, **options))
    assert [sample.global_position for sample in output] == list(range(40))
    assert Counter(sample.source_type for sample in output) == {"bc": 28, "rollout": 12}
    resumed = RuntimeMixtureDataset(contract, global_sample_offset=13, limit=40)
    suffix = list(torch.utils.data.DataLoader(resumed, **options))
    assert [sample.sample_id for sample in suffix] == [sample.sample_id for sample in output[13:]]
