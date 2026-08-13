"""Completion-gated immutable local snapshots for continuous GR00T training."""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import shutil
from pathlib import Path
from time import time
import threading
from typing import Callable

from lehome_train.groot.production_adapters import _verified_checkpoint_state_at
from lehome_train.io import sha256_file


@dataclass(frozen=True, slots=True)
class CompletedCheckpoint:
    optimizer_step: int
    source_sha256: str
    snapshot_root: Path
    observed_at_unix: int


def _tree_sha256(root: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            if path.is_symlink():
                raise ValueError("complete checkpoint contains a symlink")
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def snapshot_checkpoint(checkpoint: str | Path, *, optimizer_step: int) -> CompletedCheckpoint:
    """Verify an official completion marker then take a no-hardlink copy."""
    source = Path(checkpoint)
    try:
        _verified_checkpoint_state_at(source, optimizer_step)
    except ValueError as error:
        raise ValueError("complete checkpoint is required before packaging") from error
    destination = source.parent / f".{source.name}.snapshot-{optimizer_step}"
    if destination.exists():
        raise ValueError("checkpoint snapshot destination already exists")
    shutil.copytree(source, destination, symlinks=False, copy_function=shutil.copy2)
    digest = _tree_sha256(destination)
    return CompletedCheckpoint(optimizer_step, digest, destination, int(time()))


def run_continuous_supervisor(
    *,
    run_root: Path,
    launch: Callable[[], object],
    package: Callable[[CompletedCheckpoint], object],
    publish: Callable[[object], bool],
    wait: Callable[[], None] | None = None,
) -> tuple[int, ...]:
    """Launch once, observe official save completion, publish in one worker.

    The launch thread never receives publisher credentials.  Publication uses
    an immutable snapshot and a bounded one-worker executor; callers may use a
    blocking wait hook in production or a deterministic no-op hook in tests.
    """
    launch_error: list[BaseException] = []
    finished = threading.Event()

    def train() -> None:
        try:
            launch()
        except BaseException as error:
            launch_error.append(error)
        finally:
            finished.set()

    threading.Thread(target=train, daemon=True).start()
    submitted: dict[int, object] = {}
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-publisher") as executor:
        while not finished.is_set() or len(submitted) < 2:
            for step in (1000, 2000):
                if step in submitted:
                    continue
                checkpoint = run_root / f"checkpoint-{step}"
                if checkpoint.is_dir():
                    try:
                        snapshot = snapshot_checkpoint(checkpoint, optimizer_step=step)
                    except ValueError:
                        # The upstream save directory becomes visible before
                        # trainer_state.json is atomically complete.  Observe
                        # again; never package an incomplete boundary.
                        continue
                    submitted[step] = executor.submit(lambda item=snapshot: publish(package(item)))
            if launch_error:
                raise launch_error[0]
            if wait is None:
                if finished.is_set():
                    break
                threading.Event().wait(0.1)
            else:
                wait()
        immutable = tuple(step for step in (1000, 2000) if step in submitted and submitted[step].result() is True)
    if launch_error:
        raise launch_error[0]
    return immutable
