"""Local-first immutable recovery receipts for official trainer checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Callable, Mapping

from lehome_train.groot.production_adapters import _verified_checkpoint_state_at
from lehome_train.io import canonical_json_bytes, canonical_json_sha256, sha256_file

LOCAL_CHECKPOINT_STEPS = (500, 1000, 1500, 2000)
HF_RECOVERY_STEPS = (1000, 2000)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_IDENTITY_KEYS = {"experiment_manifest_sha256", "parent_checkpoint_artifact_sha256", "runtime_mixture_id", "trainer_code_sha256", "trainer_code_revision"}
_AWR_IDENTITY_KEYS = {"awr_evidence_sha256", "awr_config_sha256"}


class AttestationCancelled(RuntimeError):
    """A preemption requested while walking a checkpoint tree."""


@dataclass(frozen=True, slots=True)
class LocalRecoveryCheckpoint:
    optimizer_step: int
    global_sample_offset: int
    checkpoint_path: Path
    receipt_path: Path
    receipt_sha256: str
    last_immutable_publication: dict[str, object] | None
    last_immutable_anchor: dict[str, str] | None
    terminal_immutable_publication: dict[str, object] | None = None
    terminal_immutable_anchor: dict[str, str] | None = None


def _identity(value: Mapping[str, object]) -> dict[str, str]:
    selected = set(value)
    if selected not in (_IDENTITY_KEYS, _IDENTITY_KEYS | _AWR_IDENTITY_KEYS):
        raise ValueError("local checkpoint identity is incompatible")
    for key in selected - {"trainer_code_revision"}:
        if type(value[key]) is not str or _SHA.fullmatch(str(value[key])) is None:
            raise ValueError("local checkpoint identity is incompatible")
    if type(value["trainer_code_revision"]) is not str or _REVISION.fullmatch(str(value["trainer_code_revision"])) is None:
        raise ValueError("local checkpoint identity is incompatible")
    return {key: str(value[key]) for key in sorted(value)}


def validate_local_recovery_admission(
    *, metadata_root: Path, identity: Mapping[str, object],
) -> dict[str, str]:
    """Validate recovery identity/root without creating the fresh metadata leaf."""

    checked = _identity(identity)
    if not metadata_root.is_absolute():
        raise ValueError("local recovery metadata root must be an approved absolute directory")
    current = metadata_root
    while True:
        if current.is_symlink():
            raise ValueError("local recovery metadata root has a symlinked ancestor")
        if current.exists() and not current.is_dir():
            raise ValueError("local recovery metadata root is unsafe")
        parent = current.parent
        if parent == current:
            break
        current = parent
    return checked


def validate_immutable_publication_admission(
    *, publication: Mapping[str, object], anchor: Mapping[str, object], optimizer_step: int,
) -> tuple[dict[str, object], dict[str, str]]:
    """Validate the readback-proven immutable predecessor admission binding."""

    checked_publication = _publication(publication, optimizer_step=optimizer_step)
    checked_anchor = _anchor(anchor)
    if (
        checked_publication is None or checked_anchor is None
        or checked_publication.get("readback_verified") is not True
        or type(checked_publication.get("immutable_revision")) is not str
        or _REVISION.fullmatch(str(checked_publication["immutable_revision"])) is None
    ):
        raise ValueError("local immutable publication admission is incompatible")
    return checked_publication, checked_anchor


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError("checkpoint directory is unsafe")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_regular_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("checkpoint file is unsafe")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_validated_checkpoint_tree(
    root: Path, tree: list[dict[str, object]], *,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    """Durably flush the exact validated tree before publishing its receipt."""

    directories = {root}
    for entry in tree:
        _raise_if_cancelled(cancel_requested)
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("official checkpoint tree has an unsafe path")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("official checkpoint tree changed before receipt fsync")
        _fsync_regular_file(path)
        parent = path.parent
        while True:
            directories.add(parent)
            if parent == root:
                break
            parent = parent.parent
    for directory in sorted(directories, key=lambda item: (-len(item.parts), item.as_posix())):
        _raise_if_cancelled(cancel_requested)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("official checkpoint tree changed before receipt fsync")
        _fsync_dir(directory)


def _verify_or_repair_marker(
    path: Path, digest: str, *, label: str,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    """Finish a crash-interrupted JSON sidecar only after full validation.

    The JSON file is immutable evidence; a missing marker can only be the
    final crash window.  Recreate it with an atomic hard-link after callers
    have checked every schema, identity, tree, and publication binding field.
    """
    marker = path.with_suffix(".COMPLETE")
    expected = (digest + "\n").encode("ascii")
    _raise_if_cancelled(cancel_requested)
    if marker.is_symlink():
        raise ValueError(f"{label} completion marker is unsafe")
    if marker.exists():
        if not marker.is_file() or marker.read_bytes() != expected:
            raise ValueError(f"{label} completion marker drift")
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".COMPLETE", dir=path.parent)
    temporary = Path(name)
    try:
        _raise_if_cancelled(cancel_requested)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            _raise_if_cancelled(cancel_requested)
            os.link(temporary, marker)
        except FileExistsError:
            if marker.is_symlink() or not marker.is_file() or marker.read_bytes() != expected:
                raise ValueError(f"{label} completion marker drift") from None
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_root(root: Path) -> Path:
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("local recovery metadata root must be an approved absolute directory")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("local recovery metadata root is unsafe")
    return root


def _strict_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is partial or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is malformed") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is malformed")
    return value


def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise AttestationCancelled("local checkpoint attestation cancelled")


def _sha256_file_cancelable(path: Path, cancel_requested: Callable[[], bool] | None) -> str:
    """Incrementally hash a checkpoint file while permitting bounded stop."""
    if cancel_requested is None:
        return sha256_file(path)
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            _raise_if_cancelled(cancel_requested)
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_tree(
    root: Path, *, cancel_requested: Callable[[], bool] | None = None,
) -> tuple[list[dict[str, object]], str]:
    """Hash a stable regular-file tree without loading checkpoint bytes into RAM."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("official checkpoint is partial or unsafe")

    def files() -> list[Path]:
        result: list[Path] = []
        for current, dirs, names in os.walk(root, followlinks=False):
            _raise_if_cancelled(cancel_requested)
            directory = Path(current)
            if directory.is_symlink():
                raise ValueError("official checkpoint tree contains a symlink")
            for name in [*dirs, *names]:
                _raise_if_cancelled(cancel_requested)
                path = directory / name
                if path.is_symlink() or not path.is_file() and not path.is_dir():
                    raise ValueError("official checkpoint tree contains a non-regular entry")
                if path.is_file():
                    result.append(path)
        return sorted(result, key=lambda item: item.relative_to(root).as_posix())

    _raise_if_cancelled(cancel_requested)
    before = files()
    tree: list[dict[str, object]] = []
    for path in before:
        _raise_if_cancelled(cancel_requested)
        first = path.stat()
        digest = _sha256_file_cancelable(path, cancel_requested)
        second = path.stat()
        if (first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns) != (second.st_dev, second.st_ino, second.st_size, second.st_mtime_ns):
            raise ValueError("official checkpoint tree changed while being attested")
        tree.append({"path": path.relative_to(root).as_posix(), "size": second.st_size, "sha256": digest})
    _raise_if_cancelled(cancel_requested)
    after = files()
    if [path.relative_to(root).as_posix() for path in before] != [path.relative_to(root).as_posix() for path in after]:
        raise ValueError("official checkpoint tree changed while being attested")
    for path, entry in zip(after, tree, strict=True):
        _raise_if_cancelled(cancel_requested)
        if path.stat().st_size != entry["size"] or _sha256_file_cancelable(path, cancel_requested) != entry["sha256"]:
            raise ValueError("official checkpoint tree changed while being attested")
    return tree, canonical_json_sha256(tree)


def _anchor(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"immutable_anchor_revision", "anchor_sha256"}:
        raise ValueError("local checkpoint immutable anchor is incompatible")
    revision, digest = value.get("immutable_anchor_revision"), value.get("anchor_sha256")
    if type(revision) is not str or _REVISION.fullmatch(revision) is None or type(digest) is not str or _SHA.fullmatch(digest) is None:
        raise ValueError("local checkpoint immutable anchor is incompatible")
    return {"immutable_anchor_revision": revision, "anchor_sha256": digest}


def _publication(value: object, *, optimizer_step: int = 1000) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or type(value.get("optimizer_step")) is not int or value.get("optimizer_step") != optimizer_step:
        raise ValueError("local checkpoint immutable publication is incompatible")
    return json.loads(canonical_json_bytes(dict(value)))


def _receipt_path(root: Path, step: int) -> Path:
    return root / f"checkpoint-{step}.json"


def _publication_path(root: Path, step: int) -> Path:
    return root / f"publication-{step}.json"


def attest_local_checkpoint(
    *, checkpoint: Path, metadata_root: Path, optimizer_step: int,
    identity: Mapping[str, object],
    last_immutable_publication: Mapping[str, object] | None = None,
    last_immutable_anchor: Mapping[str, object] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> LocalRecoveryCheckpoint:
    """Attest an already-complete official directory in an external atomic sidecar."""
    if optimizer_step not in LOCAL_CHECKPOINT_STEPS:
        raise ValueError("local checkpoints are required at exact 500-step boundaries")
    checked_identity = _identity(identity)
    root = _safe_root(metadata_root)
    if checkpoint.is_symlink() or not checkpoint.is_absolute() or checkpoint.name != f"checkpoint-{optimizer_step}":
        raise ValueError("official checkpoint path is incompatible")
    if last_immutable_publication is not None or last_immutable_anchor is not None:
        raise ValueError("local checkpoint HF lineage must be recorded in its immutable publication journal")
    _raise_if_cancelled(cancel_requested)
    _verified_checkpoint_state_at(checkpoint, optimizer_step)
    tree, tree_sha = _canonical_tree(checkpoint, cancel_requested=cancel_requested)
    _raise_if_cancelled(cancel_requested)
    _fsync_validated_checkpoint_tree(
        checkpoint, tree, cancel_requested=cancel_requested,
    )
    _raise_if_cancelled(cancel_requested)
    # v3 intentionally excludes HF publication state.  Trainer completion is
    # durable independently of a delayed Hub readback; journals bind the
    # immutable lineage afterwards without rewriting this sidecar.
    receipt = {"schema_version": 3, "kind": "lehome_local_checkpoint_recovery", "optimizer_step": optimizer_step, "global_sample_offset": optimizer_step * 64, "physical_batch_size": 64, "action_horizon": 16, "official_checkpoint_path": str(checkpoint), "checkpoint_tree": tree, "checkpoint_tree_sha256": tree_sha, "identity": checked_identity}
    path = _receipt_path(root, optimizer_step)
    marker = path.with_suffix(".COMPLETE")
    if path.exists() or marker.exists() or path.is_symlink() or marker.is_symlink():
        existing = _read_receipt(
            path, identity=checked_identity, cancel_requested=cancel_requested,
        )
        if existing.optimizer_step != optimizer_step or existing.checkpoint_path != checkpoint:
            raise FileExistsError("local checkpoint receipt is immutable")
        return existing
    descriptor, name = tempfile.mkstemp(prefix=f".checkpoint-{optimizer_step}.", suffix=".json", dir=root)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(receipt)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor, marker_name = tempfile.mkstemp(prefix=f".checkpoint-{optimizer_step}.", suffix=".COMPLETE", dir=root)
        marker_temporary = Path(marker_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write((canonical_json_sha256(receipt) + "\n").encode("ascii")); stream.flush(); os.fsync(stream.fileno())
            os.replace(marker_temporary, marker)
        except BaseException:
            marker_temporary.unlink(missing_ok=True)
            raise
        _fsync_dir(root)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return _read_receipt(path, identity=checked_identity, cancel_requested=cancel_requested)


def _read_receipt(
    path: Path, *, identity: Mapping[str, object],
    cancel_requested: Callable[[], bool] | None = None,
) -> LocalRecoveryCheckpoint:
    match = re.fullmatch(r"checkpoint-(500|1000|1500|2000)\.json", path.name)
    if match is None:
        raise ValueError("local recovery receipt boundary is invalid")
    marker = path.with_suffix(".COMPLETE")
    receipt = _strict_json(path, "local checkpoint receipt")
    if marker.is_symlink():
        raise ValueError("local checkpoint receipt is partial or unsafe")
    step = int(match.group(1))
    base_fields = {"schema_version", "kind", "optimizer_step", "global_sample_offset", "physical_batch_size", "action_horizon", "official_checkpoint_path", "checkpoint_tree", "checkpoint_tree_sha256", "identity"}
    legacy_fields = base_fields | {"last_immutable_publication", "last_immutable_anchor"}
    schema_version = receipt.get("schema_version")
    if (
        type(schema_version) is not int or schema_version not in (2, 3)
        or set(receipt) != (legacy_fields if schema_version == 2 else base_fields)
        or receipt.get("kind") != "lehome_local_checkpoint_recovery"
        or type(receipt.get("optimizer_step")) is not int or receipt.get("optimizer_step") != step
        or type(receipt.get("global_sample_offset")) is not int or receipt.get("global_sample_offset") != step * 64
        or type(receipt.get("physical_batch_size")) is not int or receipt.get("physical_batch_size") != 64
        or type(receipt.get("action_horizon")) is not int or receipt.get("action_horizon") != 16
        or receipt.get("identity") != _identity(identity) or type(receipt.get("official_checkpoint_path")) is not str
    ):
        raise ValueError("local checkpoint identity or cursor is incompatible")
    checkpoint = Path(str(receipt["official_checkpoint_path"]))
    if not checkpoint.is_absolute() or checkpoint.name != f"checkpoint-{step}":
        raise ValueError("local checkpoint path is incompatible")
    _verified_checkpoint_state_at(checkpoint, step)
    tree = receipt.get("checkpoint_tree")
    if not isinstance(tree, list) or any(not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"} or type(item.get("path")) is not str or not item["path"] or type(item.get("size")) is not int or int(item["size"]) < 0 or type(item.get("sha256")) is not str or _SHA.fullmatch(str(item["sha256"])) is None for item in tree) or receipt.get("checkpoint_tree_sha256") != canonical_json_sha256(tree):
        raise ValueError("local checkpoint tree receipt is incompatible")
    _raise_if_cancelled(cancel_requested)
    observed_tree, observed_sha = _canonical_tree(checkpoint, cancel_requested=cancel_requested)
    if observed_tree != tree or observed_sha != receipt["checkpoint_tree_sha256"]:
        raise ValueError("local checkpoint tree drift")
    publication = anchor = None
    if schema_version == 2:
        publication, anchor = _publication(receipt["last_immutable_publication"]), _anchor(receipt["last_immutable_anchor"])
        if step >= 1500 and (publication is None or anchor is None):
            raise ValueError("local checkpoint immutable anchor is missing")
        if (publication is None) != (anchor is None):
            raise ValueError("local checkpoint immutable anchor is incompatible")
    digest = canonical_json_sha256(receipt)
    if not marker.exists():
        _fsync_validated_checkpoint_tree(
            checkpoint, observed_tree, cancel_requested=cancel_requested,
        )
        _raise_if_cancelled(cancel_requested)
    _verify_or_repair_marker(
        path, digest, label="local checkpoint receipt", cancel_requested=cancel_requested,
    )
    terminal_publication, terminal_anchor = (None, None)
    if step == 1000:
        journal_publication, journal_anchor = _read_publication_journal(
            path.parent, optimizer_step=step, checkpoint_receipt_sha256=digest,
            cancel_requested=cancel_requested,
        )
        if journal_publication is not None:
            publication, anchor = journal_publication, journal_anchor
    elif step >= 1500:
        predecessor_path = _receipt_path(path.parent, 1000)
        predecessor_marker = predecessor_path.with_suffix(".COMPLETE")
        if predecessor_path.exists() or predecessor_marker.exists():
            if not predecessor_path.exists() or not predecessor_marker.exists():
                raise ValueError("local checkpoint predecessor receipt is partial")
            predecessor = _read_receipt(
                predecessor_path, identity=identity, cancel_requested=cancel_requested,
            )
            if predecessor.last_immutable_publication is not None:
                if publication is not None and (
                    publication != predecessor.last_immutable_publication
                    or anchor != predecessor.last_immutable_anchor
                ):
                    raise ValueError("local checkpoint immutable anchor is incompatible")
                publication, anchor = predecessor.last_immutable_publication, predecessor.last_immutable_anchor
        if step == 2000:
            terminal_publication, terminal_anchor = _read_publication_journal(
                path.parent, optimizer_step=step, checkpoint_receipt_sha256=digest,
                cancel_requested=cancel_requested,
            )
    return LocalRecoveryCheckpoint(
        step, step * 64, checkpoint, path, digest, publication, anchor,
        terminal_publication, terminal_anchor,
    )


def _read_publication_journal(
    root: Path, *, optimizer_step: int, checkpoint_receipt_sha256: str,
    cancel_requested: Callable[[], bool] | None = None,
    trust_markerless: bool = False,
) -> tuple[dict[str, object] | None, dict[str, str] | None]:
    """Read an immutable HF publication binding without mutating its receipt."""
    if optimizer_step not in HF_RECOVERY_STEPS:
        raise ValueError("local immutable publication journal has an invalid boundary")
    path = _publication_path(root, optimizer_step)
    marker = path.with_suffix(".COMPLETE")
    if not path.exists() and not marker.exists():
        return None, None
    if path.is_symlink() or marker.is_symlink() or not path.is_file():
        raise ValueError("local immutable publication journal is partial")
    # JSON without COMPLETE is a crash artifact, not independent immutable
    # authority.  Only record_immutable_publication may opt in after it has
    # matched current authenticated publication evidence exactly.
    if not marker.exists() and not trust_markerless:
        return None, None
    value = _strict_json(path, "local immutable publication journal")
    fields = {"schema_version", "kind", "optimizer_step", "checkpoint_receipt_sha256", "publication", "anchor"}
    if (
        set(value) != fields or type(value.get("schema_version")) is not int or value.get("schema_version") != 1
        or value.get("kind") != "lehome_local_checkpoint_immutable_publication"
        or type(value.get("optimizer_step")) is not int or value.get("optimizer_step") != optimizer_step
        or value.get("checkpoint_receipt_sha256") != checkpoint_receipt_sha256
    ):
        raise ValueError("local immutable publication journal is incompatible")
    publication = _publication(value.get("publication"), optimizer_step=optimizer_step)
    anchor = _anchor(value.get("anchor"))
    if publication is None or anchor is None or publication.get("readback_verified") is not True:
        raise ValueError("local immutable publication journal lacks verified publication/anchor")
    digest = canonical_json_sha256(value)
    if marker.exists():
        _verify_or_repair_marker(
            path, digest, label="local immutable publication journal",
            cancel_requested=cancel_requested,
        )
    return publication, anchor


def record_immutable_publication(*, metadata_root: Path, checkpoint: LocalRecoveryCheckpoint, publication: Mapping[str, object], anchor: Mapping[str, object]) -> None:
    """Atomically bind an HF publication to an already-immutable local receipt."""
    if checkpoint.optimizer_step not in HF_RECOVERY_STEPS:
        raise ValueError("local immutable publication requires an HF boundary")
    root = _safe_root(metadata_root)
    checked_publication = _publication(publication, optimizer_step=checkpoint.optimizer_step)
    checked_anchor = _anchor(anchor)
    if checked_publication is None or checked_anchor is None or checked_publication.get("readback_verified") is not True:
        raise ValueError("local immutable publication requires readback-verified publication/anchor")
    payload = {
        "schema_version": 1, "kind": "lehome_local_checkpoint_immutable_publication",
        "optimizer_step": checkpoint.optimizer_step, "checkpoint_receipt_sha256": checkpoint.receipt_sha256,
        "publication": checked_publication, "anchor": checked_anchor,
    }
    path = _publication_path(root, checkpoint.optimizer_step)
    marker = path.with_suffix(".COMPLETE")
    if path.exists() or marker.exists() or path.is_symlink() or marker.is_symlink():
        markerless = not marker.exists()
        existing, existing_anchor = _read_publication_journal(
            root, optimizer_step=checkpoint.optimizer_step,
            checkpoint_receipt_sha256=checkpoint.receipt_sha256,
            trust_markerless=True,
        )
        if existing != checked_publication or existing_anchor != checked_anchor:
            raise FileExistsError("local immutable publication journal is immutable")
        if markerless:
            # The caller supplied the current readback-verified publication
            # and anchor; repair only this exact, fully validated orphan.
            _verify_or_repair_marker(
                path, canonical_json_sha256(_strict_json(path, "local immutable publication journal")),
                label="local immutable publication journal",
            )
        return
    descriptor, name = tempfile.mkstemp(prefix=f".publication-{checkpoint.optimizer_step}.", suffix=".json", dir=root)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(payload)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor, marker_name = tempfile.mkstemp(prefix=f".publication-{checkpoint.optimizer_step}.", suffix=".COMPLETE", dir=root)
        marker_temporary = Path(marker_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write((canonical_json_sha256(payload) + "\n").encode("ascii")); stream.flush(); os.fsync(stream.fileno())
            os.replace(marker_temporary, marker)
        except BaseException:
            marker_temporary.unlink(missing_ok=True)
            raise
        _fsync_dir(root)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def discover_local_recovery(*, metadata_root: Path, identity: Mapping[str, object]) -> LocalRecoveryCheckpoint | None:
    """Return newest verified local boundary; all visible partial state fails closed."""
    checked = _identity(identity)
    if not metadata_root.exists():
        return None
    if metadata_root.is_symlink() or not metadata_root.is_absolute() or not metadata_root.is_dir():
        raise ValueError("local recovery metadata root is unsafe")
    for entry in metadata_root.iterdir():
        if entry.name.startswith(("checkpoint-", "publication-")) and re.fullmatch(r"(?:checkpoint-(500|1000|1500|2000)|publication-(1000|2000))\.(json|COMPLETE)", entry.name) is None:
            raise ValueError("local recovery has an invalid or partial boundary")
    for step in HF_RECOVERY_STEPS:
        publication, publication_marker = (
            _publication_path(metadata_root, step),
            _publication_path(metadata_root, step).with_suffix(".COMPLETE"),
        )
        if publication.exists() or publication_marker.exists():
            receipt, receipt_marker = _receipt_path(metadata_root, step), _receipt_path(metadata_root, step).with_suffix(".COMPLETE")
            # A markerless receipt may be the final JSON-to-COMPLETE crash
            # window and is repaired only after its strict reader validates
            # the tree and identity below.  A missing JSON remains unsafe.
            if not receipt.exists():
                raise ValueError("local immutable publication journal has no completed checkpoint receipt")
    receipts: list[LocalRecoveryCheckpoint] = []
    for step in LOCAL_CHECKPOINT_STEPS:
        path, marker = _receipt_path(metadata_root, step), _receipt_path(metadata_root, step).with_suffix(".COMPLETE")
        if path.exists() or marker.exists() or path.is_symlink() or marker.is_symlink():
            if not path.exists():
                raise ValueError("local checkpoint receipt is partial")
            receipts.append(_read_receipt(path, identity=checked))
    return max(receipts, key=lambda item: item.optimizer_step) if receipts else None
