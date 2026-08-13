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
        # A resumed observer may encounter its prior immutable copy.  Accept it
        # only when it remains a completed checkpoint of the same boundary.
        _verified_checkpoint_state_at(destination, optimizer_step)
        return CompletedCheckpoint(optimizer_step, _tree_sha256(destination), destination, int(time()))
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
) -> tuple[dict[str, object], ...]:
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

    training_thread = threading.Thread(target=train, daemon=False)
    training_thread.start()
    submitted: dict[int, object] = {}
    finished_polls = 0
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
            # Preserve already published checkpoints on interruption; the
            # terminal caller converts this into an authenticated resume state.
            if launch_error and finished.is_set():
                break
            if wait is None:
                if finished.is_set():
                    finished_polls += 1
                    if len(submitted) == 2 or finished_polls >= 20:
                        break
                threading.Event().wait(0.1)
            else:
                wait()
        immutable: list[dict[str, object]] = []
        for step in (1000, 2000):
            if step not in submitted:
                continue
            receipt = submitted[step].result()
            if receipt is True:
                immutable.append({"optimizer_step": step, "readback_verified": True})
            elif isinstance(receipt, dict) and receipt.get("readback_verified") is True and receipt.get("optimizer_step") == step:
                immutable.append(receipt)
    if launch_error:
        # An interrupt does not discard verified work.  The caller receives the
        # last immutable publication and can resume only with its identities.
        training_thread.join()
        return tuple(immutable)
    training_thread.join()
    return tuple(immutable)
