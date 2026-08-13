"""Completion-gated immutable local snapshots for continuous GR00T training."""
from __future__ import annotations

from dataclasses import dataclass
import shutil
from pathlib import Path
from time import time

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
