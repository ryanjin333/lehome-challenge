"""Run deterministic GR00T N1.7 rollout trials from a frozen matrix.

The matrix runner deliberately starts one Isaac process per trial.  That keeps
the simulator seed, garment, policy state, video directory, and log boundary
unambiguous while the first accepted campaign is being established.  Parallel
workers can be added later without changing the matrix or artifact contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")
_CATEGORY_PREFIX = {
    "top_long": "Top_Long",
    "top_short": "Top_Short",
    "pant_long": "Pant_Long",
    "pant_short": "Pant_Short",
}
_SEEDS = (42, 43, 44)
_METRIC_RE = re.compile(
    r"Episode\s+1/1:\s+Return=(?P<return>[-+]?\d+(?:\.\d+)?),\s+"
    r"Length=(?P<length>\d+),\s+Success=(?P<success>True|False)"
)


class MatrixValidationError(ValueError):
    """Raised when a rollout matrix or artifact path violates the contract."""


@dataclass(frozen=True)
class Trial:
    """One immutable garment/seed rollout assignment."""

    trial_id: str
    category: str
    garment_name: str
    seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "trial_id": self.trial_id,
            "category": self.category,
            "garment_name": self.garment_name,
            "seed": self.seed,
        }


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MatrixValidationError(f"matrix is unavailable or invalid: {path}") from error
    if not isinstance(value, Mapping):
        raise MatrixValidationError("matrix root must be an object")
    return value


def _validate_trial(raw: object, index: int) -> Trial:
    if not isinstance(raw, Mapping):
        raise MatrixValidationError(f"trial {index} must be an object")
    required = {"trial_id", "category", "garment_name", "seed"}
    if set(raw) != required:
        raise MatrixValidationError(f"trial {index} must contain exactly {sorted(required)}")
    trial_id = raw["trial_id"]
    category = raw["category"]
    garment_name = raw["garment_name"]
    seed = raw["seed"]
    if not isinstance(trial_id, str) or not trial_id:
        raise MatrixValidationError(f"trial {index} has an invalid trial_id")
    if category not in _CATEGORIES:
        raise MatrixValidationError(f"trial {index} has an invalid category")
    prefix = _CATEGORY_PREFIX[category]
    if (
        not isinstance(garment_name, str)
        or not re.fullmatch(rf"{re.escape(prefix)}_Seen_[0-9]+", garment_name)
    ):
        raise MatrixValidationError(
            f"trial {index} garment must be an official seen {category} ID"
        )
    if type(seed) is not int or seed not in _SEEDS:
        raise MatrixValidationError(f"trial {index} seed must be one of {_SEEDS}")
    return Trial(trial_id, category, garment_name, seed)


def validate_matrix(payload: Mapping[str, Any]) -> tuple[Trial, ...]:
    """Validate the fixed 24-trial seen-development contract."""

    if payload.get("schema_version") != 1:
        raise MatrixValidationError("matrix schema_version must be 1")
    raw_categories = payload.get("categories")
    raw_seeds = payload.get("seeds")
    raw_trials = payload.get("trials")
    if raw_categories != list(_CATEGORIES):
        raise MatrixValidationError("matrix categories are not the frozen four categories")
    if raw_seeds != list(_SEEDS):
        raise MatrixValidationError("matrix seeds must be [42, 43, 44]")
    if not isinstance(raw_trials, list):
        raise MatrixValidationError("matrix trials must be a list")
    trials = tuple(_validate_trial(raw, index) for index, raw in enumerate(raw_trials))
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise MatrixValidationError("trial IDs must be unique")
    assignments = {(trial.garment_name, trial.seed) for trial in trials}
    if len(assignments) != len(trials):
        raise MatrixValidationError("garment/seed assignments must be unique")
    if len(trials) != 24:
        raise MatrixValidationError("matrix must contain exactly 24 trials")
    garments_by_category: dict[str, set[str]] = {category: set() for category in _CATEGORIES}
    for trial in trials:
        garments_by_category[trial.category].add(trial.garment_name)
    if any(len(garments) != 2 for garments in garments_by_category.values()):
        raise MatrixValidationError("each category must contain exactly two seen garments")
    for category, garments in garments_by_category.items():
        for garment in garments:
            if sum(
                1
                for trial in trials
                if trial.category == category and trial.garment_name == garment
            ) != len(_SEEDS):
                raise MatrixValidationError(
                    f"{garment} must have exactly one trial for each frozen seed"
                )
    return trials


def load_matrix(path: str | os.PathLike[str]) -> tuple[Trial, ...]:
    """Load and validate a matrix JSON file."""

    return validate_matrix(_read_json(Path(path)))


def parse_episode_metric(output: str) -> dict[str, object]:
    """Extract the single episode metric emitted by ``scripts.eval``."""

    matches = list(_METRIC_RE.finditer(output))
    if len(matches) != 1:
        raise MatrixValidationError("evaluation output did not contain one complete episode metric")
    match = matches[0]
    return {
        "return": float(match.group("return")),
        "length": int(match.group("length")),
        "success": match.group("success") == "True",
    }


def validate_run_path(run_root: Path, candidate: Path) -> Path:
    """Return a resolved path only when it remains below the run root."""

    root = run_root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise MatrixValidationError("artifact path escapes the rollout run root") from error
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_sha256(path: Path) -> str:
    return sha256_file(path)


def _official_seen_ids(asset_base: Path, category: str) -> tuple[str, ...]:
    prefix = _CATEGORY_PREFIX[category]
    list_path = asset_base / "Release" / prefix / f"{prefix}.txt"
    try:
        names = tuple(
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, UnicodeError) as error:
        raise MatrixValidationError(
            f"official garment list is unavailable: {list_path}"
        ) from error
    if len(set(names)) != len(names):
        raise MatrixValidationError(f"official garment list contains duplicates: {list_path}")
    seen = tuple(
        sorted(
            name
            for name in names
            if re.fullmatch(rf"{re.escape(prefix)}_Seen_[0-9]+", name)
        )
    )
    if len(seen) < 2:
        raise MatrixValidationError(
            f"official garment list has fewer than two seen IDs: {list_path}"
        )
    return seen


def validate_asset_matrix(asset_base: Path, trials: Sequence[Trial]) -> None:
    """Require the matrix to use the first two official seen IDs per category."""

    for category in _CATEGORIES:
        expected = set(_official_seen_ids(asset_base, category)[:2])
        observed = {trial.garment_name for trial in trials if trial.category == category}
        if observed != expected:
            raise MatrixValidationError(
                f"matrix garments for {category} do not match official first-two seen IDs: "
                f"expected {sorted(expected)}, observed {sorted(observed)}"
            )


def _make_config_overlay(source_base: Path, trial_root: Path, garment_name: str) -> Path:
    source_release = (source_base / "Release").resolve()
    if not source_release.is_dir():
        raise MatrixValidationError(f"Isaac garment Release directory is missing: {source_release}")
    overlay = trial_root / "garment-config"
    release = overlay / "Release"
    release.mkdir(parents=True, exist_ok=False)
    for category in _CATEGORY_PREFIX.values():
        source_category = source_release / category
        if not source_category.is_dir():
            raise MatrixValidationError(f"Isaac garment category is missing: {source_category}")
        (release / category).symlink_to(source_category, target_is_directory=True)
    (release / "Release_test_list.txt").write_text(garment_name + "\n", encoding="utf-8")
    return overlay


def _video_files(root: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.mp4")):
        if path.is_symlink() or not path.is_file():
            raise MatrixValidationError(f"video path is not a regular file: {path}")
        files.append({"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--policy-path", required=True, type=Path)
    parser.add_argument("--asset-base-path", type=Path, default=Path("Assets/objects/Challenge_Garment"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--task", default="LeHome-BiSO101-Direct-Garment-v2")
    parser.add_argument("--task-description", default="fold the garment on the table")
    parser.add_argument("--max-steps", default=600, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_matrix(args: argparse.Namespace) -> dict[str, object]:
    trials = load_matrix(args.matrix)
    matrix_path = args.matrix.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    matrix_digest = matrix_sha256(matrix_path)
    validate_asset_matrix(args.asset_base_path, trials)
    if not args.policy_path.exists():
        raise MatrixValidationError(f"policy path is missing: {args.policy_path}")
    if args.max_steps <= 0:
        raise MatrixValidationError("max_steps must be positive")

    records: list[dict[str, object]] = []
    for trial in trials:
        trial_root = validate_run_path(output_root, output_root / trial.trial_id)
        trial_root.mkdir(parents=True, exist_ok=True)
        log_path = validate_run_path(trial_root, trial_root / "eval.log")
        video_root = validate_run_path(trial_root, trial_root / "videos")
        video_root.mkdir(parents=True, exist_ok=True)
        overlay = _make_config_overlay(args.asset_base_path, trial_root, trial.garment_name)
        command = [
            sys.executable,
            "-m",
            "scripts.eval",
            "--policy_type",
            "groot",
            "--policy_path",
            str(args.policy_path),
            "--garment_type",
            "custom",
            "--garment_cfg_base_path",
            str(overlay),
            "--task",
            args.task,
            "--task_description",
            args.task_description,
            "--num_episodes",
            "1",
            "--max_steps",
            str(args.max_steps),
            "--seed",
            str(trial.seed),
            "--device",
            args.device,
            "--video_dir",
            str(video_root),
        ]
        if args.headless:
            command.append("--headless")
        if args.save_video:
            command.append("--save_video")
        if args.dry_run:
            record = {
                "trial": trial.to_dict(),
                "matrix_sha256": matrix_digest,
                "command": command,
                "status": "dry_run",
            }
            (trial_root / "trial.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            records.append(record)
            continue
        completed = subprocess.run(
            command,
            cwd=args.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            env=os.environ.copy(),
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        record: dict[str, object] = {
            "trial": trial.to_dict(),
            "matrix_sha256": matrix_digest,
            "command": command,
            "returncode": completed.returncode,
            "log": {"relative_path": log_path.relative_to(output_root).as_posix(), "sha256": sha256_file(log_path)},
            "videos": _video_files(video_root),
        }
        try:
            record["metric"] = parse_episode_metric(completed.stdout)
            record["status"] = "passed" if completed.returncode == 0 else "failed"
        except MatrixValidationError as error:
            record["status"] = "failed"
            record["metric_error"] = str(error)
        (trial_root / "trial.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        records.append(record)

    report = {
        "schema_version": 1,
        "matrix": {"path": str(matrix_path), "sha256": matrix_digest, "trial_count": len(trials)},
        "policy_path": str(args.policy_path.resolve()),
        "trials": records,
        "completed_trials": sum(1 for record in records if record.get("status") == "passed"),
        "successes": sum(1 for record in records if isinstance(record.get("metric"), Mapping) and record["metric"].get("success") is True),
    }
    report_path = validate_run_path(output_root, output_root / "rollout-report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run_matrix(args)
    except MatrixValidationError as error:
        print(f"matrix validation error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["completed_trials"] == report["matrix"]["trial_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MatrixValidationError",
    "Trial",
    "load_matrix",
    "matrix_sha256",
    "parse_episode_metric",
    "run_matrix",
    "validate_asset_matrix",
    "validate_matrix",
    "validate_run_path",
]
