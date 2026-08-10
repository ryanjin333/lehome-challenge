"""Closed, content-addressed release trees for completed flywheel campaigns."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Mapping
import re

from lehome.flywheel.artifacts import (
    MANIFEST_NAME,
    atomic_write_json,
    build_sha256_manifest,
    verify_episode_manifest,
)
from lehome.flywheel.matrix import PublicMatrix, Trial


_REQUIRED_EPISODE_FILES = frozenset(
    {
        "annotations.jsonl",
        "episode.json",
        "snapshots/reset.json",
        "snapshots/terminal.json",
        "videos/left_rgb.mp4",
        "videos/right_rgb.mp4",
        "videos/top_rgb.mp4",
    }
)
_REPORTS = ("capacity-report.json", "rollout-report.json")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OCI_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a materialized regular file")


def _annotation_count(path: Path) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    raise ValueError("episode annotations contain a blank record")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("episode annotation must be an object")
                count += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("episode annotations are invalid") from error
    return count


@dataclass(frozen=True, slots=True)
class CampaignProvenance:
    policy_repo: str
    policy_revision: str
    policy_step: int
    code_revision: str
    asset_revision: str
    simulator_version: str
    policy_artifact_sha256: str
    image_identity: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.policy_repo, str)
            or not self.policy_repo
            or any(character.isspace() for character in self.policy_repo)
        ):
            raise ValueError("campaign policy repository is invalid")
        for value, label in (
            (self.policy_revision, "policy revision"),
            (self.code_revision, "code revision"),
            (self.asset_revision, "asset revision"),
        ):
            if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
                raise ValueError(f"campaign {label} is not an immutable commit")
        if type(self.policy_step) is not int or self.policy_step <= 0:
            raise ValueError("campaign policy step is invalid")
        if not isinstance(self.simulator_version, str) or not self.simulator_version:
            raise ValueError("campaign simulator version is invalid")
        if (
            not isinstance(self.policy_artifact_sha256, str)
            or _SHA256.fullmatch(self.policy_artifact_sha256) is None
        ):
            raise ValueError("campaign policy artifact digest is invalid")
        if not isinstance(self.image_identity, str) or _OCI_DIGEST.fullmatch(self.image_identity) is None:
            raise ValueError("campaign image identity is not an immutable OCI digest")


@dataclass(frozen=True, slots=True)
class PlannedEpisode:
    trial: Trial
    source: Path

    @property
    def relative_root(self) -> str:
        return (
            f"raw/{self.trial.category}/{self.trial.release_stage}/"
            f"{self.trial.trial_id}"
        )


@dataclass(frozen=True, slots=True)
class ReleasePlan:
    run_root: Path
    matrix: PublicMatrix
    episodes: tuple[PlannedEpisode, ...]
    provenance: CampaignProvenance
    # Kept for publisher API compatibility; this is the maximum permitted
    # annotation count, not an exact per-episode requirement.
    expected_steps: int

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def episode_paths(self) -> tuple[str, ...]:
        return tuple(episode.relative_root for episode in self.episodes)

    @property
    def category_counts(self) -> dict[str, int]:
        return _counts(episode.trial.category for episode in self.episodes)

    @property
    def release_stage_counts(self) -> dict[str, int]:
        return _counts(episode.trial.release_stage for episode in self.episodes)


@dataclass(frozen=True, slots=True)
class ReleaseEntry:
    relative_path: str
    sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class MaterializedRelease:
    root: Path
    release_id: str
    remote_prefix: str
    entries: tuple[ReleaseEntry, ...]

    @property
    def entry_count(self) -> int:
        return len(self.entries)


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_release_plan(
    run_root: Path,
    matrix: PublicMatrix,
    *,
    expected_steps: int = 600,
) -> ReleasePlan:
    """Require one terminal, checksum-valid episode for every matrix trial.

    ``expected_steps`` is the campaign's maximum step budget.  Episodes at
    the budget are valid regardless of terminal outcome; early episodes must
    be accepted successes with the canonical success terminal metadata.
    """
    if type(expected_steps) is not int or expected_steps <= 0:
        raise ValueError("maximum annotation count must be positive")
    root = Path(run_root)
    raw = root / "raw"
    if root.is_symlink() or not root.is_dir() or raw.is_symlink() or not raw.is_dir():
        raise ValueError("campaign raw root must be a materialized directory")
    pending = root / ".pending"
    if pending.exists() or pending.is_symlink():
        if pending.is_symlink() or not pending.is_dir() or any(pending.iterdir()):
            raise ValueError("campaign has incomplete pending episodes")
    for report in _REPORTS:
        _safe_regular(root / report, report)

    expected = {trial.trial_id for trial in matrix.trials}
    if len(expected) != len(matrix.trials):
        raise ValueError("matrix contains duplicate trial IDs")
    actual: set[str] = set()
    for path in raw.iterdir():
        if path.is_symlink() or not path.is_dir():
            raise ValueError("campaign raw root may contain only episode directories")
        actual.add(path.name)
    if actual != expected:
        raise ValueError("campaign raw episodes do not match the exact matrix")

    episodes: list[PlannedEpisode] = []
    campaign_provenance: CampaignProvenance | None = None
    for trial in matrix.trials:
        source = raw / trial.trial_id
        candidate = _validate_episode_release_contract(
            source,
            trial,
            expected_steps=expected_steps,
        )
        if campaign_provenance is None:
            campaign_provenance = candidate
        elif candidate != campaign_provenance:
            raise ValueError("episode provenance is inconsistent across the campaign")
        episodes.append(PlannedEpisode(trial, source))

    ordered = tuple(
        sorted(
            episodes,
            key=lambda episode: (
                episode.trial.category,
                episode.trial.release_stage,
                episode.trial.trial_id,
            ),
        )
    )
    if campaign_provenance is None:
        raise ValueError("campaign has no verified provenance")
    return ReleasePlan(root, matrix, ordered, campaign_provenance, expected_steps)


def _validate_episode_release_contract(
    source: Path,
    trial: Trial,
    *,
    expected_steps: int,
) -> CampaignProvenance:
    metadata, manifest = verify_episode_manifest(source)
    if set(manifest) != _REQUIRED_EPISODE_FILES:
        raise ValueError("episode file set does not match the exact release allowlist")
    identity = metadata.get("identity")
    expected_identity = {
        "category": trial.category,
        "garment_name": trial.garment_name,
        "release_stage": trial.release_stage,
        "seed": trial.seed,
    }
    if not isinstance(identity, dict) or any(
        identity.get(key) != value for key, value in expected_identity.items()
    ):
        raise ValueError("episode identity does not match matrix trial")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("episode provenance is missing")
    candidate = CampaignProvenance(
        policy_repo=identity.get("policy_repo"),
        policy_revision=identity.get("policy_revision"),
        policy_step=identity.get("policy_step"),
        code_revision=identity.get("code_revision"),
        asset_revision=identity.get("asset_revision"),
        simulator_version=identity.get("simulator_version"),
        policy_artifact_sha256=provenance.get("policy_artifact_sha256"),
        image_identity=provenance.get("image_identity"),
    )
    annotation_count = _annotation_count(source / "annotations.jsonl")
    if annotation_count == 0 or annotation_count > expected_steps:
        raise ValueError("episode annotation count violates the maximum-step release contract")
    if annotation_count < expected_steps and not (
        metadata.get("accepted_success") is True
        and metadata.get("outcome") == "success"
        and metadata.get("terminal_reason") == "success"
    ):
        raise ValueError("early episode annotation count requires an accepted success terminal")
    return candidate


def _payload_entries(root: Path) -> tuple[ReleaseEntry, ...]:
    manifest = build_sha256_manifest(root)
    return tuple(
        ReleaseEntry(relative, str(entry["sha256"]), int(entry["size"]))
        for relative, entry in manifest.items()
    )


def materialize_release(
    plan: ReleasePlan,
    destination: Path,
    *,
    matrix_json: str,
    policy_revision: str,
    code_revision: str,
    asset_revision: str,
    image_identity: str,
    policy_step: int = 12000,
) -> MaterializedRelease:
    """Copy the verified campaign into one immutable, sorted upload snapshot."""
    root = Path(destination)
    if root.exists() or root.is_symlink():
        raise ValueError("release destination must not already exist")
    if not matrix_json.endswith("\n"):
        raise ValueError("matrix JSON must have its canonical terminal newline")
    declared = (
        policy_revision,
        code_revision,
        asset_revision,
        image_identity,
        policy_step,
    )
    verified = (
        plan.provenance.policy_revision,
        plan.provenance.code_revision,
        plan.provenance.asset_revision,
        plan.provenance.image_identity,
        plan.provenance.policy_step,
    )
    if declared != verified:
        raise ValueError("declared release provenance does not match verified episodes")
    root.mkdir(parents=True)
    try:
        for episode in plan.episodes:
            target = root / Path(*PurePosixPath(episode.relative_root).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(episode.source, target, copy_function=shutil.copy2)
            copied_provenance = _validate_episode_release_contract(
                target,
                episode.trial,
                expected_steps=plan.expected_steps,
            )
            if copied_provenance != plan.provenance:
                raise ValueError("copied episode provenance changed during release staging")
        reports = root / "reports"
        reports.mkdir()
        for name in _REPORTS:
            shutil.copy2(plan.run_root / name, reports / name)
        (root / "matrix.json").write_text(matrix_json, encoding="utf-8")

        payload_entries = _payload_entries(root)
        release_identity: dict[str, object] = {
            "schema_version": 1,
            "policy_repo": plan.provenance.policy_repo,
            "policy_revision": plan.provenance.policy_revision,
            "policy_step": plan.provenance.policy_step,
            "code_revision": plan.provenance.code_revision,
            "asset_revision": plan.provenance.asset_revision,
            "simulator_version": plan.provenance.simulator_version,
            "policy_artifact_sha256": plan.provenance.policy_artifact_sha256,
            "image_identity": plan.provenance.image_identity,
            "episode_count": plan.episode_count,
            "category_counts": plan.category_counts,
            "release_stage_counts": plan.release_stage_counts,
            "entries": [
                {
                    "path": entry.relative_path,
                    "sha256": entry.sha256,
                    "size": entry.byte_size,
                }
                for entry in payload_entries
            ],
        }
        release_id = _canonical_sha256(release_identity)
        release_manifest = {
            **release_identity,
            "release_id": release_id,
            "remote_prefix": f"rollouts/groot-n17-step-{policy_step}/{release_id}",
        }
        atomic_write_json(root / "release-manifest.json", release_manifest)
        atomic_write_json(root / MANIFEST_NAME, build_sha256_manifest(root))
        entries = verify_release_tree(root)
        return MaterializedRelease(
            root,
            release_id,
            str(release_manifest["remote_prefix"]),
            entries,
        )
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


def verify_release_tree(root: Path) -> tuple[ReleaseEntry, ...]:
    """Recompute the closed top-level allowlist, including its own manifest."""
    release_root = Path(root)
    manifest_path = release_root / MANIFEST_NAME
    _safe_regular(manifest_path, MANIFEST_NAME)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("release manifest is invalid") from error
    if not isinstance(manifest, dict):
        raise ValueError("release manifest must be an object")
    actual: set[str] = set()
    for current, directories, files in os.walk(release_root, followlinks=False):
        current_path = Path(current)
        if any((current_path / name).is_symlink() for name in directories):
            raise ValueError("release tree must not contain symlinks")
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("release tree may contain only regular files")
            actual.add(path.relative_to(release_root).as_posix())
    expected = set(manifest) | {MANIFEST_NAME}
    if actual != expected:
        raise ValueError("release tree does not match its closed manifest")
    entries: list[ReleaseEntry] = []
    for relative in sorted(manifest):
        entry = manifest[relative]
        path = release_root / Path(*PurePosixPath(relative).parts)
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("sha256"), str)
            or type(entry.get("size")) is not int
            or path.stat().st_size != entry["size"]
        ):
            raise ValueError(f"release size mismatch: {relative}")
        digest = _sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(f"release hash mismatch: {relative}")
        entries.append(ReleaseEntry(relative, digest, path.stat().st_size))
    entries.append(
        ReleaseEntry(MANIFEST_NAME, _sha256(manifest_path), manifest_path.stat().st_size)
    )
    return tuple(sorted(entries, key=lambda entry: entry.relative_path))


def validate_remote_file_tree(
    observed: Mapping[str, int],
    *,
    remote_prefix: str,
    expected: tuple[ReleaseEntry, ...],
) -> None:
    """Require an exact immutable Hub file listing beneath one release prefix."""
    prefix = PurePosixPath(remote_prefix)
    if (
        prefix.is_absolute()
        or ".." in prefix.parts
        or "." in prefix.parts
        or len(prefix.parts) < 2
    ):
        raise ValueError("remote release prefix is unsafe")
    prefix_text = prefix.as_posix() + "/"
    scoped: dict[str, int] = {}
    for remote_path, byte_size in observed.items():
        if not isinstance(remote_path, str) or type(byte_size) is not int or byte_size < 0:
            raise ValueError("remote file listing is invalid")
        if remote_path.startswith(prefix_text):
            relative = remote_path.removeprefix(prefix_text)
            if not relative or relative in scoped:
                raise ValueError("remote release tree contains duplicate or unsafe paths")
            scoped[relative] = byte_size
    expected_sizes = {entry.relative_path: entry.byte_size for entry in expected}
    if scoped != expected_sizes:
        raise ValueError("remote release tree does not match the closed local allowlist")


__all__ = [
    "CampaignProvenance",
    "MaterializedRelease",
    "PlannedEpisode",
    "ReleaseEntry",
    "ReleasePlan",
    "build_release_plan",
    "materialize_release",
    "validate_remote_file_tree",
    "verify_release_tree",
]
