from __future__ import annotations

from pathlib import Path
import io
import json
import tarfile

import pytest

from lehome_train.b1k.rolling_checkpoints import CheckpointCompatibility, CheckpointDescriptor, LocalCheckpointPublisher, RollingCheckpointStore, ResumePolicy, package_checkpoint, safe_restore_tar
from lehome_train.b1k.training import approved_launch_plans


class InMemoryBucket:
    def __init__(self) -> None: self.objects: dict[str, bytes] = {}; self.fail_at: str | None = None; self.delete_calls = 0; self.fail_delete_call: int | None = None
    def write_bytes(self, path: str, value: bytes) -> None:
        if self.fail_at in {"stage", "latest"}: raise OSError("failure")
        self.objects[path] = value
    def read_bytes(self, path: str) -> bytes:
        if self.fail_at == "readback": return b"corrupt"
        return self.objects[path]
    def write_json(self, path: str, value: dict[str, object]) -> None: self.write_bytes(path, json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    def read_json(self, path: str) -> dict[str, object]: return json.loads(self.read_bytes(path))
    def copy(self, source: str, destination: str) -> None:
        if self.fail_at == "promote": raise OSError("failure")
        self.objects[destination] = self.read_bytes(source)
    def list(self, prefix: str) -> tuple[str, ...]: return tuple(sorted(path for path in self.objects if path.startswith(prefix)))
    def delete(self, paths: tuple[str, ...]) -> None:
        self.delete_calls += 1
        if self.fail_at == "delete" or self.fail_delete_call == self.delete_calls: raise OSError("failure")
        for path in paths: self.objects.pop(path, None)
    def upload_file(self, source: Path, destination: str) -> None:
        self.objects[destination] = source.open("rb").read()
    def download_file(self, source: str, destination: Path) -> None:
        destination.write_bytes(self.objects[source])


def _compatibility(world_size: int = 1, cycle_id: str = "cycle-001") -> CheckpointCompatibility:
    plan = approved_launch_plans(num_gpus=world_size)[0]
    return CheckpointCompatibility(materialized_dataset_fingerprint="a" * 64, modality_sha256="b" * 64, stats_sha256="c" * 64, groot_revision="d" * 40, base_model_revision="e" * 40, cosmos_revision="f" * 40, container_digest="sha256:" + "1" * 64, cycle_id=cycle_id, world_size=world_size, plan_identity=plan.identity, physical_batch_size=plan.physical_batch_size, global_batch_size=plan.global_batch_size, gradient_accumulation_steps=plan.gradient_accumulation_steps, effective_global_batch_size=plan.effective_global_batch_size, learning_rate=plan.learning_rate, weight_decay=plan.weight_decay, warmup_ratio=plan.warmup_ratio, launch_argv_sha256="2" * 64)


def test_checkpoint_compatibility_is_immutable_and_validates_approved_batch_arithmetic() -> None:
    compatibility = _compatibility()
    assert compatibility.to_dict()["effective_global_batch_size"] == 256
    with pytest.raises((AttributeError, ValueError)): compatibility.world_size = 2  # type: ignore[misc]


def test_checkpoint_compatibility_requires_cycle_identity_for_migration_safety() -> None:
    prior_schema = _compatibility().to_dict()
    prior_schema.pop("cycle_id")

    with pytest.raises(ValueError, match="schema"):
        CheckpointCompatibility.from_dict(prior_schema)


def test_resume_rejects_verified_checkpoint_from_a_different_cycle(tmp_path: Path) -> None:
    backend, old_store, old_compatibility = _store()
    artifact = _artifact(tmp_path, b"old-cycle", 1_000)
    old_store.publish(artifact, _descriptor(artifact, 1_000, old_compatibility))
    current_store = RollingCheckpointStore(
        backend=backend,
        run_id="b1k-run-001",
        compatibility=_compatibility(cycle_id="cycle-002"),
    )

    with pytest.raises(ValueError, match="incompatible"):
        current_store.resume(ResumePolicy.AUTO, tmp_path / "restore")


def _artifact(tmp_path: Path, content: bytes = b"checkpoint", step: int = 1000, world_size: int = 1) -> Path:
    source = tmp_path / f"checkpoint-{step}"; source.mkdir(exist_ok=True)
    (source / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    (source / "config.json").write_text("{}")
    for name in ("optimizer.pt", "scheduler.pt", "model.safetensors"): (source / name).write_bytes(content)
    for rank in range(world_size): (source / (f"rng_state_{rank}.pth" if world_size > 1 else "rng_state.pth")).write_bytes(content)
    path = tmp_path / f"checkpoint-{step}-{content.hex()}-{world_size}.tar"; package_checkpoint(source, path, step=step, world_size=world_size); return path


def _descriptor(artifact: Path, step: int, compatibility: CheckpointCompatibility) -> CheckpointDescriptor:
    return CheckpointDescriptor.create(run_id="b1k-run-001", step=step, artifact=artifact, compatibility=compatibility)


def _store(world_size: int = 1) -> tuple[InMemoryBucket, RollingCheckpointStore, CheckpointCompatibility]:
    backend = InMemoryBucket(); compatibility = _compatibility(world_size)
    return backend, RollingCheckpointStore(backend=backend, run_id="b1k-run-001", compatibility=compatibility), compatibility


def test_publish_retains_two_verified_descriptors_and_cleans_staging(tmp_path: Path) -> None:
    backend, store, compatibility = _store()
    for step in (1000, 2000, 3000):
        artifact = _artifact(tmp_path, str(step).encode(), step); store.publish(artifact, _descriptor(artifact, step, compatibility))
    assert store.verified_steps() == (2000, 3000)
    assert backend.list("staging/b1k-run-001/") == ()
    latest = backend.read_json("runs/b1k-run-001/latest.json")
    assert latest["step"] == 3000
    assert latest["cycle_id"] == compatibility.cycle_id


def test_local_checkpoint_publisher_packages_uploads_and_records_a_resume_receipt(tmp_path: Path) -> None:
    backend, store, _compatibility = _store()
    _artifact(tmp_path, b"checkpoint", 1000)
    publisher = LocalCheckpointPublisher(
        store=store,
        checkpoint_root=tmp_path,
        receipts_root=tmp_path / "checkpoint-receipts",
    )

    receipt = publisher.publish(1000)

    assert backend.read_json("runs/b1k-run-001/latest.json")["step"] == 1000
    assert receipt == tmp_path / "checkpoint-receipts" / "step-1000.json"
    assert json.loads(receipt.read_text())["step"] == 1000
    assert not list((tmp_path / "checkpoint-receipts").glob("*.tar"))


@pytest.mark.parametrize("failure", ["stage", "readback", "promote", "latest"])
def test_publish_fails_closed_at_each_boundary(tmp_path: Path, failure: str) -> None:
    backend, store, compatibility = _store(); artifact = _artifact(tmp_path); backend.fail_at = failure
    with pytest.raises(ValueError, match="checkpoint"): store.publish(artifact, _descriptor(artifact, 1000, compatibility))


def test_publish_rejects_descriptor_that_does_not_match_artifact(tmp_path: Path) -> None:
    _backend, store, compatibility = _store(); artifact = _artifact(tmp_path)
    descriptor = _descriptor(artifact, 1000, compatibility)
    bad = CheckpointDescriptor(schema_version=descriptor.schema_version, run_id=descriptor.run_id, step=descriptor.step, artifact_sha256="0" * 64, artifact_byte_size=descriptor.artifact_byte_size, compatibility=compatibility, descriptor_sha256=descriptor.descriptor_sha256)
    with pytest.raises(ValueError): store.publish(artifact, bad)


def test_resume_rejects_stale_or_missing_latest_when_verified_state_exists(tmp_path: Path) -> None:
    backend, store, compatibility = _store()
    for step, content in ((1000, b"one"), (2000, b"two")):
        artifact = _artifact(tmp_path, content, step); store.publish(artifact, _descriptor(artifact, step, compatibility))
    older = next(item for item in backend.objects if "/step-1000/" in item and item.endswith("descriptor.json"))
    old_data = CheckpointDescriptor.from_bytes(backend.objects[older]); backend.write_json("runs/b1k-run-001/latest.json", {"step": 1000, "prefix": older.rsplit("/", 1)[0], "descriptor_sha256": old_data.descriptor_sha256})
    with pytest.raises(ValueError, match="latest"):
        store.resume(ResumePolicy.AUTO, tmp_path / "restore")
    backend.objects.pop("runs/b1k-run-001/latest.json")
    with pytest.raises(ValueError, match="latest"):
        store.resume(ResumePolicy.AUTO, tmp_path / "missing")


def test_resume_fails_closed_for_partial_staging_or_malformed_latest_or_never_state(tmp_path: Path) -> None:
    backend, store, compatibility = _store(); artifact = _artifact(tmp_path); store.publish(artifact, _descriptor(artifact, 1000, compatibility))
    backend.write_bytes("staging/b1k-run-001/1000/x/artifact.tar", b"partial")
    with pytest.raises(ValueError, match="partial"): store.resume(ResumePolicy.AUTO, tmp_path / "restore")
    with pytest.raises(ValueError, match="never"): store.resume(ResumePolicy.NEVER, tmp_path / "other")
    backend.objects.pop("staging/b1k-run-001/1000/x/artifact.tar")
    backend.write_json("runs/b1k-run-001/latest.json", {"bad": True})
    with pytest.raises(ValueError, match="latest"): store.resume(ResumePolicy.AUTO, tmp_path / "third")


def test_resume_restores_checkpoint_directly_into_an_existing_output_directory(tmp_path: Path) -> None:
    backend, store, compatibility = _store()
    artifact = _artifact(tmp_path, b"resume", 15_000)
    store.publish(artifact, _descriptor(artifact, 15_000, compatibility))
    output = tmp_path / "outputs" / "b1k-run-001"
    output.mkdir(parents=True)

    restored = store.resume(ResumePolicy.AUTO, output)

    assert restored == output / "checkpoint-15000"
    assert (output / "checkpoint-15000" / "trainer_state.json").is_file()


def test_resume_reuses_an_exact_local_checkpoint_after_validating_the_remote_artifact(tmp_path: Path) -> None:
    backend, store, compatibility = _store()
    artifact = _artifact(tmp_path, b"remote", 1_000)
    store.publish(artifact, _descriptor(artifact, 1_000, compatibility))
    output = tmp_path / "output"; output.mkdir()
    local = safe_restore_tar(artifact, output, step=1_000)
    inode = local.stat().st_ino

    restored = store.resume(ResumePolicy.AUTO, output)

    assert restored == local
    assert restored.stat().st_ino == inode


def test_resume_quarantines_every_unselected_checkpoint_before_returning_the_remote_selection(tmp_path: Path) -> None:
    backend, store, compatibility = _store()
    artifact = _artifact(tmp_path, b"remote", 1_000)
    store.publish(artifact, _descriptor(artifact, 1_000, compatibility))
    output = tmp_path / "output"; output.mkdir()
    selected = safe_restore_tar(artifact, output, step=1_000)
    for name in ("checkpoint-500", "checkpoint-2000", "checkpoint-malformed"):
        (output / name).mkdir()

    restored = store.resume(ResumePolicy.AUTO, output)

    assert restored == selected
    assert {path.name for path in output.iterdir() if path.name.startswith("checkpoint-")} == {"checkpoint-1000"}
    quarantined = output / ".unverified-checkpoints"
    assert {path.name.split("-", 2)[1] for path in quarantined.iterdir()} == {"500", "2000", "malformed"}


def test_resume_reuses_a_private_quarantine_root_after_an_interruption_leaves_it_present(tmp_path: Path) -> None:
    backend, store, compatibility = _store()
    artifact = _artifact(tmp_path, b"remote", 1_000)
    store.publish(artifact, _descriptor(artifact, 1_000, compatibility))
    output = tmp_path / "output"; output.mkdir()
    safe_restore_tar(artifact, output, step=1_000)
    (output / "checkpoint-2000").mkdir()
    store.resume(ResumePolicy.AUTO, output)
    (output / "checkpoint-3000").mkdir()

    restored = store.resume(ResumePolicy.AUTO, output)

    assert restored == output / "checkpoint-1000"
    assert {path.name for path in output.iterdir() if path.name.startswith("checkpoint-")} == {"checkpoint-1000"}
    assert len(tuple((output / ".unverified-checkpoints").iterdir())) == 2


def test_resume_rejects_a_nonprivate_existing_quarantine_root(tmp_path: Path) -> None:
    backend, store, compatibility = _store()
    artifact = _artifact(tmp_path, b"remote", 1_000)
    store.publish(artifact, _descriptor(artifact, 1_000, compatibility))
    output = tmp_path / "output"; output.mkdir()
    (output / "checkpoint-2000").mkdir()
    quarantine = output / ".unverified-checkpoints"; quarantine.mkdir(mode=0o755); quarantine.chmod(0o755)

    with pytest.raises(ValueError, match="quarantine directory"):
        store.resume(ResumePolicy.AUTO, output)


def test_resume_quarantines_and_replaces_a_mismatched_local_checkpoint(tmp_path: Path) -> None:
    backend, store, compatibility = _store()
    artifact = _artifact(tmp_path, b"remote", 1_000)
    store.publish(artifact, _descriptor(artifact, 1_000, compatibility))
    output = tmp_path / "output"; output.mkdir()
    local = safe_restore_tar(artifact, output, step=1_000)
    local_inode = local.stat().st_ino
    (local / "model.safetensors").write_bytes(b"tampered")

    restored = store.resume(ResumePolicy.AUTO, output)

    assert restored == local
    assert restored.stat().st_ino != local_inode
    assert (restored / "model.safetensors").read_bytes() == b"remote"
    assert not list(output.glob(".checkpoint-1000.unverified-*"))


def test_resume_fails_closed_when_newest_two_retention_repair_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _backend, store, compatibility = _store()
    artifact = _artifact(tmp_path, b"remote", 1_000)
    store.publish(artifact, _descriptor(artifact, 1_000, compatibility))
    monkeypatch.setattr(store, "ensure_newest_two", lambda: (_ for _ in ()).throw(ValueError("retention repair failed")))

    with pytest.raises(ValueError, match="retention repair"):
        store.resume(ResumePolicy.AUTO, tmp_path / "output")


def test_corrupt_higher_checkpoint_prevents_retention_deletion(tmp_path: Path) -> None:
    backend, store, compatibility = _store()
    for step in (1000, 2000):
        artifact = _artifact(tmp_path, str(step).encode(), step); store.publish(artifact, _descriptor(artifact, step, compatibility))
    backend.write_bytes("verified/b1k-run-001/step-3000/" + "0" * 64 + "/artifact.tar", b"bad")
    backend.write_bytes("verified/b1k-run-001/step-3000/" + "0" * 64 + "/descriptor.json", b"{}")
    with pytest.raises(ValueError): store._retain_two()
    assert store.backend.list("verified/b1k-run-001/step-1000/")


def test_retention_delete_failure_leaves_newest_verified_and_latest(tmp_path: Path) -> None:
    backend, store, compatibility = _store()
    for step in (1000, 2000):
        artifact = _artifact(tmp_path, str(step).encode(), step); store.publish(artifact, _descriptor(artifact, step, compatibility))
    backend.fail_delete_call = backend.delete_calls + 2; artifact = _artifact(tmp_path, b"three", 3000)
    with pytest.raises(OSError): store.publish(artifact, _descriptor(artifact, 3000, compatibility))
    assert 3000 in store.verified_steps()
    assert backend.read_json("runs/b1k-run-001/latest.json")["step"] == 3000


@pytest.mark.parametrize("mode", ["noop", "partial"])
def test_retention_rejects_silent_or_partial_delete(tmp_path: Path, mode: str) -> None:
    backend, store, compatibility = _store()
    for step in (1000, 2000):
        artifact = _artifact(tmp_path, str(step).encode(), step); store.publish(artifact, _descriptor(artifact, step, compatibility))
    original_delete = backend.delete
    calls = 0
    def bad_delete(paths: tuple[str, ...]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1: original_delete(paths)  # successful publish must clear staging first
        elif mode == "partial": original_delete(paths[:1])
    backend.delete = bad_delete  # type: ignore[method-assign]
    artifact = _artifact(tmp_path, b"three", 3000)
    with pytest.raises(ValueError, match="delete"):
        store.publish(artifact, _descriptor(artifact, 3000, compatibility))
    assert backend.read_json("runs/b1k-run-001/latest.json")["step"] == 3000
    assert any("/step-3000/" in path for path in backend.objects)


@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_native_checkpoint_roundtrip_for_each_world_size(tmp_path: Path, world_size: int) -> None:
    artifact = _artifact(tmp_path, b"payload", 1000, world_size)
    restored = safe_restore_tar(artifact, tmp_path / "restore", step=1000, world_size=world_size)
    assert (restored / "scheduler.pt").read_bytes() == b"payload"


@pytest.mark.parametrize("missing", ["optimizer.pt", "scheduler.pt", "config.json", "model.safetensors", "rng_state.pth"])
def test_package_rejects_every_required_native_payload(tmp_path: Path, missing: str) -> None:
    source = tmp_path / "checkpoint-1000"; source.mkdir(); (source / "trainer_state.json").write_text('{"global_step":1000}'); (source / "config.json").write_text("{}")
    for name in ("optimizer.pt", "scheduler.pt", "model.safetensors", "rng_state.pth"): (source / name).write_bytes(b"x")
    (source / missing).unlink()
    with pytest.raises(ValueError): package_checkpoint(source, tmp_path / "bad.tar", step=1000)


@pytest.mark.parametrize("mutate", ["empty-model", "invalid-config", "bad-index", "missing-rank-rng"])
def test_package_rejects_empty_or_invalid_native_payloads(tmp_path: Path, mutate: str) -> None:
    source = tmp_path / "checkpoint-1000"; source.mkdir(); _artifact(tmp_path, b"x", 1000, 2)
    generated = tmp_path / "checkpoint-1000"
    if mutate == "empty-model": (generated / "model.safetensors").write_bytes(b"")
    elif mutate == "invalid-config": (generated / "config.json").write_text("{")
    elif mutate == "bad-index":
        (generated / "model.safetensors").unlink(); (generated / "model.safetensors.index.json").write_text('{"weight_map":{"x":"missing.safetensors"}}')
    else: (generated / "rng_state_1.pth").unlink()
    with pytest.raises(ValueError): package_checkpoint(generated, tmp_path / f"{mutate}.tar", step=1000, world_size=2)


def test_restore_rejects_traversal_duplicate_links_and_existing_target(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path); output = tmp_path / "restore"; safe_restore_tar(artifact, output, step=1000)
    with pytest.raises(ValueError, match="existing"): safe_restore_tar(artifact, output, step=1000)
    for name, kind in (("checkpoint-1000/../escape", tarfile.REGTYPE), ("checkpoint-1000/link", tarfile.SYMTYPE)):
        malicious = tmp_path / f"{kind}.tar"
        with tarfile.open(malicious, "w") as archive:
            info = tarfile.TarInfo(name); info.type = kind; info.size = 1 if kind == tarfile.REGTYPE else 0
            archive.addfile(info, io.BytesIO(b"x") if kind == tarfile.REGTYPE else None)
        with pytest.raises(ValueError): safe_restore_tar(malicious, tmp_path / str(kind), step=1000)


def test_package_failure_is_atomic_and_leaves_no_destination(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1000"; source.mkdir(); _artifact(tmp_path)
    (source / "zz-unsafe").symlink_to("optimizer.pt")
    destination = tmp_path / "result.tar"
    with pytest.raises(ValueError): package_checkpoint(source, destination, step=1000)
    assert not destination.exists()
    assert not (tmp_path / ".result.tar.incomplete").exists()


def test_package_rejects_unexpected_rng_state_name(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1000"; source.mkdir(); _artifact(tmp_path)
    (source / "rng_state_99.pth").write_bytes(b"x")
    with pytest.raises(ValueError, match="RNG"): package_checkpoint(source, tmp_path / "bad-rng.tar", step=1000)


@pytest.mark.parametrize("member_type", [tarfile.LNKTYPE, tarfile.BLKTYPE])
def test_restore_rejects_hardlinks_and_special_members(tmp_path: Path, member_type: bytes) -> None:
    artifact = tmp_path / f"unsafe-{member_type}.tar"
    with tarfile.open(artifact, "w") as archive:
        info = tarfile.TarInfo("checkpoint-1000/unsafe"); info.type = member_type
        if member_type == tarfile.LNKTYPE: info.linkname = "checkpoint-1000/model.safetensors"
        archive.addfile(info)
    with pytest.raises(ValueError, match="unsafe"): safe_restore_tar(artifact, tmp_path / "restore", step=1000)


def test_restore_rejects_duplicate_and_truncated_tar_and_symlink_parent(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.tar"
    with tarfile.open(duplicate, "w") as archive:
        for _ in range(2):
            info = tarfile.TarInfo("checkpoint-1000/optimizer.pt"); info.size = 1; archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError): safe_restore_tar(duplicate, tmp_path / "duplicate", step=1000)
    truncated = tmp_path / "truncated.tar"; truncated.write_bytes(_artifact(tmp_path).read_bytes()[:20])
    with pytest.raises((ValueError, tarfile.TarError)): safe_restore_tar(truncated, tmp_path / "truncated", step=1000)
    target = tmp_path / "real-output"; target.mkdir(); parent = tmp_path / "symlink-output"; parent.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"): safe_restore_tar(_artifact(tmp_path, b"new"), parent, step=1000)
    symlink_target = tmp_path / "target-output"; symlink_target.mkdir(); (symlink_target / "checkpoint-1000").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="existing"): safe_restore_tar(_artifact(tmp_path, b"other"), symlink_target, step=1000)


def test_resume_rechecks_artifact_after_validated_scan(tmp_path: Path) -> None:
    backend, store, compatibility = _store(); artifact = _artifact(tmp_path, b"original")
    store.publish(artifact, _descriptor(artifact, 1000, compatibility))
    replacement = _artifact(tmp_path, b"replacement")
    artifact_path = next(path for path in backend.objects if path.endswith("artifact.tar"))
    original_download = backend.download_file; reads = 0
    def swapped(path: str, destination: Path) -> None:
        nonlocal reads
        if path == artifact_path:
            reads += 1
            if reads == 2:
                destination.write_bytes(replacement.read_bytes()); return
        original_download(path, destination)
    backend.download_file = swapped  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="artifact"):
        store.resume(ResumePolicy.AUTO, tmp_path / "restore")


def test_descriptor_identity_hashes_artifact_incrementally_without_path_read_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = _artifact(tmp_path, b"x")
    with artifact.open("ab") as handle: handle.write(b"x" * 1024 * 1024)
    compatibility = _compatibility()
    monkeypatch.setattr(Path, "read_bytes", lambda _self: pytest.fail("whole artifact read"))
    descriptor = CheckpointDescriptor.create(run_id="b1k-run-001", step=1000, artifact=artifact, compatibility=compatibility)
    assert descriptor.artifact_byte_size == artifact.stat().st_size
