"""Process-safe state transitions for immutable sibling snapshots."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
import tempfile
from threading import RLock, local
from typing import Callable, Iterator, Mapping

from lehome_train.io import atomic_write_json


_INITIALIZE_HOOK: Callable[[str, Path], None] | None = None
_held_locks = local()
_active_lock_fds: set[int] = set()
_fork_guard = RLock()


def _before_fork() -> None:
    _fork_guard.acquire()


def _after_fork_parent() -> None:
    _fork_guard.release()


def _after_fork_child() -> None:
    """Do not retain a parent's flock description in a forked child."""

    for descriptor in tuple(_active_lock_fds):
        try:
            os.close(descriptor)
        except OSError:
            pass
    _active_lock_fds.clear()
    _held_locks.pid = os.getpid()
    _held_locks.paths = set()
    _held_locks.parents = {}
    _held_locks.parent_fds = {}
    _fork_guard.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_before_fork, after_in_parent=_after_fork_parent, after_in_child=_after_fork_child)


def _close_registered_descriptor(descriptor: int) -> None:
    """Keep descriptor deregistration and close indivisible with fork."""

    with _fork_guard:
        _active_lock_fds.discard(descriptor)
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lstat(path: Path, label: str) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError(f"{label} is unsafe") from error


def _safe_directory(path: Path, label: str) -> bool:
    observed = _lstat(path, label)
    if observed is None:
        return False
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise ValueError(f"{label} is unsafe")
    return True


def _validated_parent(parent: Path) -> tuple[int, int]:
    observed = _lstat(parent, "snapshot destination parent")
    if observed is None or stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise ValueError("snapshot destination parent is unsafe")
    return observed.st_dev, observed.st_ino


def _held_state() -> tuple[set[str], dict[str, tuple[int, int]], dict[str, int]]:
    """Discard copied thread-local ownership after ``fork()`` changes PID."""

    pid = os.getpid()
    if getattr(_held_locks, "pid", None) != pid:
        _held_locks.pid = pid
        _held_locks.paths = set()
        _held_locks.parents = {}
        _held_locks.parent_fds = {}
    return _held_locks.paths, _held_locks.parents, _held_locks.parent_fds


def _validate_held_parent(lock_path: Path, parent_identity: tuple[int, int]) -> None:
    _, parents, _ = _held_state()
    expected = parents.get(os.fspath(lock_path.absolute()))
    if expected is not None and expected != parent_identity:
        raise ValueError("snapshot destination parent changed while locked")


def _validate_lock_descriptor(descriptor: int, lock_path: Path) -> None:
    observed = os.fstat(descriptor)
    path_stat = _lstat(lock_path, "snapshot destination lock")
    if path_stat is None or stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(observed.st_mode) or observed.st_uid != os.getuid() or observed.st_mode & 0o7777 != 0o600 or (path_stat.st_dev, path_stat.st_ino) != (observed.st_dev, observed.st_ino):
        raise ValueError("snapshot destination lock is unsafe")


def _validate_intent(stage: Path, *, intent_name: str, identity: Mapping[str, object], read_intent: Callable[[Path], Mapping[str, object]], label: str) -> None:
    intent_path = stage / intent_name
    observed = _lstat(intent_path, f"{label} intent")
    if observed is None or stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"{label} intent does not match")
    try:
        value = read_intent(intent_path)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"{label} intent does not match") from error
    if dict(value) != dict(identity):
        raise ValueError(f"{label} intent does not match")


def _remove_own_unique_stage(
    unique: Path,
    *,
    parent: Path,
    parent_identity: tuple[int, int],
    lock_path: Path,
    unique_identity: tuple[int, int],
    prefix: str,
) -> bool:
    """Delete only the mkdtemp directory created by this invocation."""

    current_parent = _validated_parent(parent)
    _validate_held_parent(lock_path, current_parent)
    if current_parent != parent_identity:
        return False
    if unique.parent.absolute() != parent.absolute() or not unique.name.startswith(prefix):
        raise ValueError("unique snapshot staging directory is unsafe")
    observed = _lstat(unique, "unique snapshot staging directory")
    if observed is None:
        return False
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode) or (observed.st_dev, observed.st_ino) != unique_identity:
        return False
    # Collision orphans are intentionally retained.  Recursive removal has a
    # check/delete race and an intent-bearing unique directory cannot block a
    # future deterministic stage initialization.
    return False


def bound_destination(destination: str | Path, path: str | Path | None = None) -> Path:
    """Return a parent-fd anchored path while its destination lock is held."""

    destination = Path(destination)
    path = destination if path is None else Path(path)
    parent = destination.parent
    lock_path = destination.with_name(f".{destination.name}.lock")
    _, parents, parent_fds = _held_state()
    key = os.fspath(lock_path.absolute())
    expected = parents.get(key)
    descriptor = parent_fds.get(key)
    if expected is None or descriptor is None or _validated_parent(parent) != expected:
        raise ValueError("snapshot destination parent changed while locked")
    try:
        relative = path.absolute().relative_to(parent.absolute())
    except ValueError as error:
        raise ValueError("snapshot destination path is unsafe") from error
    # Linux exposes directory descriptors as stable path roots.  macOS does
    # not support pathname traversal through /dev/fd directory descriptors,
    # so callers retain the normal path there and validate immediately after
    # each external-write boundary before any promotion writes can occur.
    anchor = Path(f"/proc/self/fd/{descriptor}")
    return anchor / relative if anchor.is_dir() else path


def validate_destination_binding(path: str | Path) -> None:
    """Fail before further path writes if a locked parent was replaced."""

    path = Path(path)
    bound_destination(path)


@contextmanager
def destination_lock(destination: str | Path) -> Iterator[None]:
    """Take one validated advisory flock for a destination lifecycle."""

    destination = Path(destination)
    owner_pid = os.getpid()
    parent = destination.parent
    parent_identity = _validated_parent(parent)
    lock_path = destination.with_name(f".{destination.name}.lock")
    held, parents, parent_fds = _held_state()
    key = os.fspath(lock_path.absolute())
    if key in held:
        raise RuntimeError("snapshot destination lock is not reentrant")

    existing = _lstat(lock_path, "snapshot destination lock")
    if existing is not None and (stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode)):
        raise ValueError("snapshot destination lock is unsafe")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        parent_flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        parent_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        parent_flags |= os.O_NOFOLLOW
    try:
        with _fork_guard:
            parent_descriptor = os.open(parent, parent_flags)
            _active_lock_fds.add(parent_descriptor)
    except OSError as error:
        raise ValueError("snapshot destination parent is unsafe") from error
    try:
        parent_stat = os.fstat(parent_descriptor)
        if not stat.S_ISDIR(parent_stat.st_mode) or (parent_stat.st_dev, parent_stat.st_ino) != parent_identity:
            raise ValueError("snapshot destination parent is unsafe")
    except BaseException:
        _close_registered_descriptor(parent_descriptor)
        raise
    try:
        with _fork_guard:
            descriptor = os.open(lock_path, flags, 0o600)
            _active_lock_fds.add(descriptor)
    except OSError as error:
        _close_registered_descriptor(parent_descriptor)
        raise ValueError("snapshot destination lock is unsafe") from error
    try:
        _validate_lock_descriptor(descriptor, lock_path)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            _validate_lock_descriptor(descriptor, lock_path)
            if _validated_parent(parent) != parent_identity:
                raise ValueError("snapshot destination parent changed while acquiring lock")
        except BaseException:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            raise
        held.add(key)
        parents[key] = parent_identity
        parent_fds[key] = parent_descriptor
        try:
            yield
        finally:
            if os.getpid() == owner_pid:
                held.discard(key)
                parents.pop(key, None)
                parent_fds.pop(key, None)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        if os.getpid() == owner_pid:
            _close_registered_descriptor(descriptor)
            _close_registered_descriptor(parent_descriptor)


def open_staged_destination(
    destination: str | Path,
    *,
    intent_name: str,
    identity: Mapping[str, object],
    read_intent: Callable[[Path], Mapping[str, object]],
    label: str,
) -> tuple[Path, bool]:
    """Return a completed destination or an intent-bearing deterministic stage.

    Callers hold :func:`destination_lock` throughout the complete lifecycle.
    A temporary unique stage is written and synced before it becomes resumable.
    """

    destination = Path(destination)
    parent = destination.parent
    parent_identity = _validated_parent(parent)
    lock_path = destination.with_name(f".{destination.name}.lock")
    _validate_held_parent(lock_path, parent_identity)
    if _safe_directory(destination, f"completed {label} directory"):
        return destination, True
    staging = destination.with_name(f".{destination.name}.incomplete")
    if _safe_directory(staging, f"{label} incomplete staging directory"):
        _validate_intent(staging, intent_name=intent_name, identity=identity, read_intent=read_intent, label=f"{label} incomplete staging")
        return staging, False

    prefix = f".{destination.name}.initializing-"
    current_parent = _validated_parent(parent)
    _validate_held_parent(lock_path, current_parent)
    unique = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    unique_stat = _lstat(unique, "unique snapshot staging directory")
    if unique_stat is None or stat.S_ISLNK(unique_stat.st_mode) or not stat.S_ISDIR(unique_stat.st_mode):
        raise ValueError("unique snapshot staging directory is unsafe")
    unique_identity = unique_stat.st_dev, unique_stat.st_ino
    if _INITIALIZE_HOOK is not None:
        _INITIALIZE_HOOK("after-unique-create", unique)
    atomic_write_json(unique / intent_name, dict(identity))
    fsync_directory(unique)
    fsync_directory(parent)
    if _INITIALIZE_HOOK is not None:
        _INITIALIZE_HOOK("after-intent-fsync", unique)

    # A cooperating process cannot create this while the caller holds the
    # destination lock, but this branch keeps recovery conservative if a
    # deterministic stage appears between the two probes.
    if _safe_directory(staging, f"{label} incomplete staging directory"):
        _validate_intent(staging, intent_name=intent_name, identity=identity, read_intent=read_intent, label=f"{label} incomplete staging")
        _remove_own_unique_stage(unique, parent=parent, parent_identity=current_parent, lock_path=lock_path, unique_identity=unique_identity, prefix=prefix)
        return staging, False
    try:
        current_parent = _validated_parent(parent)
        _validate_held_parent(lock_path, current_parent)
        os.rename(unique, staging)
    except FileExistsError:
        if not _safe_directory(staging, f"{label} incomplete staging directory"):
            raise
        _validate_intent(staging, intent_name=intent_name, identity=identity, read_intent=read_intent, label=f"{label} incomplete staging")
        _remove_own_unique_stage(unique, parent=parent, parent_identity=current_parent, lock_path=lock_path, unique_identity=unique_identity, prefix=prefix)
        return staging, False
    fsync_directory(parent)
    _validate_intent(staging, intent_name=intent_name, identity=identity, read_intent=read_intent, label=f"{label} incomplete staging")
    return staging, False
