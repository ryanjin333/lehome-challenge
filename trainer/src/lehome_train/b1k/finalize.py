"""Path-based private finalization with immutable, streaming readback."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable

from lehome_train.b1k.rolling_checkpoints import validate_native_checkpoint
from lehome_train.b1k.training import SUPPORTED_GPU_COUNTS


def _identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256(); size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block); digest.update(block)
    return size, digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FinalEvidence:
    run_contract: Path
    selection_manifest: Path
    materialized_manifest: Path
    modality: Path
    stats: Path
    model_derivation: Path
    revisions_image: Path
    argv: Path
    logs: tuple[Path, ...]
    rolling_receipts: tuple[Path, ...]
    world_size: int

    def files(self) -> tuple[Path, ...]:
        named = (self.run_contract, self.selection_manifest, self.materialized_manifest, self.modality, self.stats, self.model_derivation, self.revisions_image, self.argv, *self.logs, *self.rolling_receipts)
        if self.world_size not in SUPPORTED_GPU_COUNTS or not self.logs or not self.rolling_receipts or any(path.is_symlink() or not path.is_file() for path in named): raise ValueError("final evidence schema is invalid")
        return named


class Finalizer:
    def __init__(
        self,
        *,
        upload_file: Callable[[str, str, Path], str],
        download_file: Callable[[str, str, str, Path], None],
        ensure_branch: Callable[[str], None] | None = None,
    ) -> None:
        self.upload_file, self.download_file, self.ensure_branch = upload_file, download_file, ensure_branch

    def finalize(self, *, run_id: str, checkpoint: Path, evidence: FinalEvidence, final_dir: Path) -> dict[str, str]:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", run_id) or checkpoint.name != "checkpoint-15000": raise ValueError("final checkpoint identity is invalid")
        validate_native_checkpoint(checkpoint, step=15_000, world_size=evidence.world_size)
        branch = f"runs/{run_id}"
        if self.ensure_branch is not None:
            self.ensure_branch(branch)
        sources = {f"checkpoint/{path.relative_to(checkpoint).as_posix()}": path for path in sorted(checkpoint.rglob("*")) if path.is_file() and not path.is_symlink()}
        sources.update({f"evidence/{index:02d}-{path.name}": path for index, path in enumerate(evidence.files())})
        identities = {remote: _identity(path) for remote, path in sources.items()}
        manifest = {"schema_version": 1, "run_id": run_id, "files": {name: {"byte_size": size, "sha256": digest} for name, (size, digest) in identities.items()}}
        staging = Path(tempfile.mkdtemp(prefix="b1k-final-manifest-")); manifest_path = staging / "final-manifest.json"; probe = staging / "readback-probe"
        try:
            manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
            for remote, source in sources.items(): self.upload_file(branch, remote, source)
            commit = self.upload_file(branch, "final-manifest.json", manifest_path)
            if not re.fullmatch(r"[0-9a-f]{40}", commit): raise ValueError("final transport did not return an immutable commit")
            expected = {**identities, "final-manifest.json": _identity(manifest_path)}
            for remote, identity in expected.items():
                probe.unlink(missing_ok=True); self.download_file(branch, remote, commit, probe)
                if _identity(probe) != identity: raise ValueError("final immutable readback failed")
                probe.unlink(missing_ok=True)
            final_dir.mkdir(parents=True, exist_ok=True); receipt = {"branch": branch, "immutable_commit": commit, "manifest_sha256": expected["final-manifest.json"][1]}
            temporary = final_dir / ".final-receipt.incomplete"; temporary.write_text(json.dumps(receipt, sort_keys=True)); temporary.replace(final_dir / "final-receipt.json")
            return receipt
        finally: shutil.rmtree(staging, ignore_errors=True)
