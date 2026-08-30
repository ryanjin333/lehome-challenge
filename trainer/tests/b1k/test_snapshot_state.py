from __future__ import annotations

import json
import hashlib
import multiprocessing
import os
from pathlib import Path
import shutil
import select
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from lehome_train.io import atomic_write_json
from lehome_train.b1k.snapshot_integrity import build_remote_manifest

try:
    from lehome_train.b1k import snapshot_state
except ImportError:
    snapshot_state = None  # type: ignore[assignment]


def _read_intent(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hold_process_lock(destination: str, entered: object, release: object) -> None:
    from lehome_train.b1k.snapshot_state import destination_lock

    with destination_lock(destination):
        entered.put("locked")
        release.wait(5)


def test_destination_lock_uses_a_private_regular_current_owner_sibling(tmp_path: Path) -> None:
    assert snapshot_state is not None
    destination = tmp_path / "snapshot"

    with snapshot_state.destination_lock(destination):
        lock = tmp_path / ".snapshot.lock"
        stat = lock.lstat()
        assert lock.is_file() and not lock.is_symlink()
        assert stat.st_uid == os.getuid()
        assert stat.st_mode & 0o7777 == 0o600


@pytest.mark.parametrize("mutation", ["symlink", "mode", "owner"])
def test_destination_lock_rejects_unsafe_existing_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    assert snapshot_state is not None
    destination = tmp_path / "snapshot"
    lock = tmp_path / ".snapshot.lock"
    if mutation == "symlink":
        target = tmp_path / "target"; target.write_text("target")
        lock.symlink_to(target)
    else:
        lock.write_text("lock"); lock.chmod(0o600 if mutation == "owner" else 0o644)
        if mutation == "owner":
            monkeypatch.setattr(snapshot_state.os, "getuid", lambda: lock.stat().st_uid + 1)
    with pytest.raises(ValueError, match="lock"):
        with snapshot_state.destination_lock(destination):
            pass


def test_destination_lock_rejects_a_lock_path_replaced_while_acquiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert snapshot_state is not None
    destination = tmp_path / "snapshot"
    lock = tmp_path / ".snapshot.lock"
    original_flock = snapshot_state.fcntl.flock

    def replace_then_lock(descriptor: int, operation: int) -> None:
        if operation == snapshot_state.fcntl.LOCK_EX:
            lock.unlink()
            lock.write_text("replacement")
            lock.chmod(0o600)
        original_flock(descriptor, operation)

    monkeypatch.setattr(snapshot_state.fcntl, "flock", replace_then_lock)
    with pytest.raises(ValueError, match="lock"):
        with snapshot_state.destination_lock(destination):
            pass


@pytest.mark.parametrize("parent_kind", ["missing", "symlink"])
def test_destination_state_rejects_unvalidated_parent_directories(tmp_path: Path, parent_kind: str) -> None:
    assert snapshot_state is not None
    parent = tmp_path / "parent"
    if parent_kind == "symlink":
        target = tmp_path / "target"; target.mkdir(); parent.symlink_to(target)
    destination = parent / "snapshot"
    with pytest.raises(ValueError, match="parent"):
        with snapshot_state.destination_lock(destination):
            pass
    with pytest.raises(ValueError, match="parent"):
        snapshot_state.open_staged_destination(destination, intent_name="intent.json", identity={"ok": True}, read_intent=_read_intent, label="snapshot")


def test_staging_rejects_parent_replacement_after_lock_acquisition(tmp_path: Path) -> None:
    assert snapshot_state is not None
    parent = tmp_path / "parent"; parent.mkdir()
    destination = parent / "snapshot"
    with snapshot_state.destination_lock(destination):
        parent.rename(tmp_path / "moved-parent")
        parent.mkdir()
        with pytest.raises(ValueError, match="parent"):
            snapshot_state.open_staged_destination(destination, intent_name="intent.json", identity={"ok": True}, read_intent=_read_intent, label="snapshot")


def test_destination_lock_resets_inherited_thread_local_reentrancy_state(tmp_path: Path) -> None:
    assert snapshot_state is not None
    destination = tmp_path / "snapshot"
    lock_path = str((tmp_path / ".snapshot.lock").absolute())
    snapshot_state._held_locks.pid = os.getpid() - 1
    snapshot_state._held_locks.paths = {lock_path}
    with snapshot_state.destination_lock(destination):
        assert lock_path in snapshot_state._held_locks.paths


def test_unique_intent_stage_orphan_is_ignorable_after_crash_before_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert snapshot_state is not None
    destination = tmp_path / "snapshot"
    identity = {"repository": "repo", "revision": "pin"}
    created: list[Path] = []

    def crash(point: str, path: Path) -> None:
        if point == "after-intent-fsync":
            created.append(path)
            raise RuntimeError("crash before rename")

    monkeypatch.setattr(snapshot_state, "_INITIALIZE_HOOK", crash)
    with pytest.raises(RuntimeError, match="crash"):
        snapshot_state.open_staged_destination(destination, intent_name="intent.json", identity=identity, read_intent=_read_intent, label="snapshot")
    assert not (tmp_path / ".snapshot.incomplete").exists()
    assert len(created) == 1 and (created[0] / "intent.json").is_file()

    monkeypatch.setattr(snapshot_state, "_INITIALIZE_HOOK", None)
    stage, completed = snapshot_state.open_staged_destination(destination, intent_name="intent.json", identity=identity, read_intent=_read_intent, label="snapshot")
    assert not completed and stage == tmp_path / ".snapshot.incomplete"
    assert _read_intent(stage / "intent.json") == identity


def test_existing_deterministic_stage_wins_and_only_own_unique_stage_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert snapshot_state is not None
    destination = tmp_path / "snapshot"
    identity = {"repository": "repo", "revision": "pin"}
    created: list[Path] = []

    def create_competing_stage(point: str, unique: Path) -> None:
        if point == "after-intent-fsync":
            created.append(unique)
            stage = tmp_path / ".snapshot.incomplete"; stage.mkdir()
            atomic_write_json(stage / "intent.json", identity)

    monkeypatch.setattr(snapshot_state, "_INITIALIZE_HOOK", create_competing_stage)
    stage, completed = snapshot_state.open_staged_destination(destination, intent_name="intent.json", identity=identity, read_intent=_read_intent, label="snapshot")
    assert not completed and stage == tmp_path / ".snapshot.incomplete"
    assert created and (created[0] / "intent.json").is_file()


def test_collision_cleanup_never_removes_a_recreated_unique_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert snapshot_state is not None
    destination = tmp_path / "snapshot"
    identity = {"repository": "repo", "revision": "pin"}
    recreated: list[Path] = []

    def replace_unique_before_collision_cleanup(point: str, unique: Path) -> None:
        if point == "after-intent-fsync":
            stage = tmp_path / ".snapshot.incomplete"; stage.mkdir()
            atomic_write_json(stage / "intent.json", identity)
            shutil.rmtree(unique)
            unique.mkdir(); (unique / "must-survive").write_text("replacement")
            recreated.append(unique)

    monkeypatch.setattr(snapshot_state, "_INITIALIZE_HOOK", replace_unique_before_collision_cleanup)
    stage, completed = snapshot_state.open_staged_destination(destination, intent_name="intent.json", identity=identity, read_intent=_read_intent, label="snapshot")
    assert not completed and stage == tmp_path / ".snapshot.incomplete"
    assert recreated and (recreated[0] / "must-survive").read_text() == "replacement"


def test_collision_cleanup_rejects_parent_replacement_before_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert snapshot_state is not None
    parent = tmp_path / "parent"; parent.mkdir()
    destination = parent / "snapshot"
    identity = {"repository": "repo", "revision": "pin"}

    def replace_parent_before_collision_cleanup(point: str, _unique: Path) -> None:
        if point == "after-intent-fsync":
            parent.rename(tmp_path / "moved-parent")
            parent.mkdir()
            stage = parent / ".snapshot.incomplete"; stage.mkdir()
            atomic_write_json(stage / "intent.json", identity)

    monkeypatch.setattr(snapshot_state, "_INITIALIZE_HOOK", replace_parent_before_collision_cleanup)
    with snapshot_state.destination_lock(destination):
        with pytest.raises(ValueError, match="parent"):
            snapshot_state.open_staged_destination(destination, intent_name="intent.json", identity=identity, read_intent=_read_intent, label="snapshot")


def test_deterministic_stage_is_never_created_without_a_valid_intent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert snapshot_state is not None
    destination = tmp_path / "snapshot"

    def crash(point: str, _path: Path) -> None:
        if point == "after-unique-create":
            raise RuntimeError("crash after unique create")

    monkeypatch.setattr(snapshot_state, "_INITIALIZE_HOOK", crash)
    with pytest.raises(RuntimeError, match="unique create"):
        snapshot_state.open_staged_destination(destination, intent_name="intent.json", identity={"ok": True}, read_intent=_read_intent, label="snapshot")
    assert not (tmp_path / ".snapshot.incomplete").exists()
    orphan = next(tmp_path.glob(".snapshot.initializing-*"))
    assert not (orphan / "intent.json").exists()


def test_destination_lock_serializes_independent_processes(tmp_path: Path) -> None:
    assert snapshot_state is not None
    context = multiprocessing.get_context("fork")
    entered = context.Queue(); release = context.Event()
    first = context.Process(target=_hold_process_lock, args=(str(tmp_path / "snapshot"), entered, release))
    second = context.Process(target=_hold_process_lock, args=(str(tmp_path / "snapshot"), entered, release))
    first.start(); assert entered.get(timeout=5) == "locked"
    second.start()
    with pytest.raises(Exception):
        entered.get(timeout=0.2)
    release.set()
    assert entered.get(timeout=5) == "locked"
    first.join(5); second.join(5)
    assert first.exitcode == 0 and second.exitcode == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_forked_child_closes_inherited_parent_lock_descriptor(tmp_path: Path) -> None:
    assert snapshot_state is not None
    read_fd, write_fd = os.pipe()
    destination = tmp_path / "snapshot"
    with snapshot_state.destination_lock(destination):
        parent_fd = next(iter(snapshot_state._held_locks.parent_fds.values()))
        child = os.fork()
        if child == 0:
            os.close(read_fd)
            try:
                os.fstat(parent_fd)
            except OSError:
                os.write(write_fd, b"closed")
            finally:
                os.close(write_fd)
                os._exit(0)
        os.close(write_fd)
        readable, _, _ = select.select([read_fd], [], [], 2)
        assert readable and os.read(read_fd, 16) == b"closed"
        os.close(read_fd)
        assert os.waitpid(child, 0)[1] == 0


def test_model_snapshot_serializes_download_and_holds_lock_through_post_promotion_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lehome_train.b1k import bootstrap

    assert snapshot_state is not None
    payload = b"complete"
    downloads = 0
    calls_lock = Lock()
    destination = tmp_path / "model"

    class Hub:
        def snapshot_download(self, _repository: str, *, revision: str, local_dir: Path, allow_patterns: tuple[str, ...] | None, token: str) -> Path:
            nonlocal downloads
            with calls_lock:
                downloads += 1
            local_dir.mkdir(parents=True, exist_ok=True); (local_dir / "weights.bin").write_bytes(payload)
            return local_dir

        def remote_manifest(self, repository: str, *, revision: str, allow_patterns: tuple[str, ...] | None, token: str):
            return build_remote_manifest(repository=repository, revision=revision, resolved_revision=revision, allow_patterns=allow_patterns, entries=({"path": "weights.bin", "size": len(payload), "blob_id": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(), "lfs": None},))

    original_validate = bootstrap._validate_snapshot_receipt
    post_promotion_seen = False

    def check_lock(path: Path, **kwargs: object):
        nonlocal post_promotion_seen
        value = original_validate(path, **kwargs)
        if path == destination and destination.exists():
            assert str((tmp_path / ".model.lock").absolute()) in getattr(snapshot_state._held_locks, "paths", set())
            post_promotion_seen = True
        return value

    monkeypatch.setattr(bootstrap, "_validate_snapshot_receipt", check_lock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: bootstrap._ensure_model_snapshot(hub=Hub(), repository="repo", revision="pin", destination=destination, token="memory"), range(2)))
    assert results == [destination, destination]
    assert downloads == 1 and post_promotion_seen
    validation_cache = {}
    assert bootstrap._ensure_model_snapshot(hub=Hub(), repository="repo", revision="pin", destination=destination, token="memory", validation_cache=validation_cache) == destination
    assert validation_cache[destination].hash_passes == 1


def test_model_snapshot_parent_swap_during_download_fails_before_promotion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lehome_train.b1k import bootstrap

    parent = tmp_path / "parent"; parent.mkdir()
    destination = parent / "model"

    class SwappingHub:
        def snapshot_download(self, _repository: str, *, revision: str, local_dir: Path, allow_patterns: tuple[str, ...] | None, token: str) -> None:
            parent.rename(tmp_path / "moved-parent")
            parent.mkdir()
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "weights.bin").write_bytes(b"payload")

    monkeypatch.setattr(bootstrap, "_promote_snapshot", lambda *_args, **_kwargs: pytest.fail("must not promote after parent replacement"))
    with pytest.raises(ValueError, match="parent"):
        bootstrap._ensure_model_snapshot(hub=SwappingHub(), repository="repo", revision="pin", destination=destination, token="memory")
    assert not destination.exists()


def test_model_snapshot_parent_swap_immediately_before_promotion_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lehome_train.b1k import bootstrap

    parent = tmp_path / "parent"; parent.mkdir()
    destination = parent / "model"; payload = b"payload"

    class Hub:
        def snapshot_download(self, _repository: str, *, revision: str, local_dir: Path, allow_patterns: tuple[str, ...] | None, token: str) -> None:
            local_dir.mkdir(parents=True, exist_ok=True); (local_dir / "weights.bin").write_bytes(payload)
        def remote_manifest(self, repository: str, *, revision: str, allow_patterns: tuple[str, ...] | None, token: str):
            return build_remote_manifest(repository=repository, revision=revision, resolved_revision=revision, allow_patterns=allow_patterns, entries=({"path": "weights.bin", "size": len(payload), "blob_id": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(), "lfs": None},))

    original_promote = bootstrap._promote_snapshot
    def swap_then_promote(*args: object, **kwargs: object):
        parent.rename(tmp_path / "moved-parent"); parent.mkdir()
        return original_promote(*args, **kwargs)

    monkeypatch.setattr(bootstrap, "_promote_snapshot", swap_then_promote)
    with pytest.raises(ValueError, match="parent"):
        bootstrap._ensure_model_snapshot(hub=Hub(), repository="repo", revision="pin", destination=destination, token="memory")
    assert not destination.exists()
