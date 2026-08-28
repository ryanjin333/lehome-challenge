"""Shared authenticated evidence contract for fresh visual-only replay."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys
from collections import Counter
from collections.abc import Mapping, Sequence


_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from build_success_replay_matrix import (  # noqa: E402
    CATEGORIES,
    FRESH_SOURCE_CAMPAIGN_KIND,
    FRESH_SOURCE_LOGICAL_STAGE,
    FRESH_SOURCE_REPORT_KIND,
    PARENT_ARTIFACT_SHA256,
    PARENT_POLICY_REPO,
    PARENT_REVISION,
    _canonical_bytes,
    _episode_artifact_sha256,
    _sha256,
    _state_fingerprint,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CUDA_DEVICE = re.compile(r"^cuda:[0-9]+$")
_ROUND_ID = re.compile(r"^fresh-12k-[a-z0-9-]{1,112}$")
_RUN_ID = re.compile(r"^fresh-run-[a-z0-9-]{1,112}$")
_FRESH_FIELDS = frozenset({
    "source_episode_sha256", "source_episode_root", "source_episode_path",
    "source_reset_sha256", "source_annotations_sha256",
    "source_continuation_snapshot_sha256", "source_state_fingerprint",
    "source_report_sha256", "source_matrix_sha256", "source_receipt_sha256",
    "source_receipt_path", "source_remote_prefix", "source_immutable_revision",
    "source_round_id", "source_run_id", "source_report_path", "source_matrix_path",
})
_FRESH_PATH_FIELDS = frozenset({
    "source_episode_root", "source_episode_path", "source_report_path",
    "source_matrix_path", "source_receipt_path",
})
_FRESH_STRING_FIELDS = frozenset({
    "source_remote_prefix", "source_immutable_revision", "source_round_id", "source_run_id",
})
_SNAPSHOT_FIELDS = frozenset({
    "schema_version", "robot_position", "robot_velocity", "cloth_position",
    "cloth_velocity", "rng_state", "garment_name", "randomization", "scene_state",
    "cloth_state_authority",
})


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> None:
    raise ValueError(f"non-finite JSON value {token}")


def _load_json(path: Path, *, label: str, object_only: bool = True) -> object:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is malformed") from error
    if object_only and not isinstance(value, dict):
        raise ValueError(f"{label} is malformed")
    return value


def _regular_absolute(path_value: object, *, label: str) -> Path:
    if not isinstance(path_value, str):
        raise ValueError(f"{label} is missing or unsafe")
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError(f"{label} is missing or unsafe")
    current = path
    while True:
        try:
            metadata = current.lstat()
        except OSError as error:
            raise ValueError(f"{label} is missing or unsafe") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} is missing or unsafe")
        if current.parent == current:
            break
        current = current.parent
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"{label} is missing or unsafe")
    return path


def _absolute_directory(path_value: object, *, label: str) -> Path:
    if not isinstance(path_value, str):
        raise ValueError(f"{label} is missing or unsafe")
    path = Path(path_value)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is missing or unsafe")
    return path


def canonical_source_evidence_paths(
    paths: Sequence[Path], *, label: str,
) -> tuple[Path, ...]:
    """Require builder source evidence to be stable absolute regular files."""

    canonical: list[Path] = []
    for value in paths:
        path = _regular_absolute(str(value), label=label)
        resolved = path.resolve(strict=True)
        if resolved in canonical:
            raise ValueError(f"{label} is duplicated")
        canonical.append(resolved)
    return tuple(canonical)


def _hash_text_field(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _evidence(raw: str, *, label: str) -> dict[str, tuple[str, Path]]:
    try:
        values = json.loads(raw, object_pairs_hook=_strict_pairs, parse_constant=_reject_nonfinite_json)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"fresh {label} evidence is malformed") from error
    if not isinstance(values, list) or not values:
        raise ValueError(f"fresh {label} evidence is required")
    found: dict[str, tuple[str, Path]] = {}
    for item in values:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ValueError(f"fresh {label} evidence is malformed")
        path = _regular_absolute(item.get("path"), label=f"fresh {label} evidence")
        canonical = path.resolve(strict=True)
        digest = _hash_text_field(item.get("sha256"), label=f"fresh {label} evidence digest")
        if _sha256(canonical) != digest or str(canonical) in found:
            raise ValueError(f"fresh {label} evidence is missing or tampered")
        found[str(canonical)] = (digest, canonical)
    return found


def _finite(value: object, *, label: str) -> None:
    if type(value) in (int, float):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} contains a non-finite numeric value")
    elif isinstance(value, list):
        for item in value:
            _finite(item, label=label)
    elif isinstance(value, Mapping):
        for item in value.values():
            _finite(item, label=label)


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _canonical_snapshot(
    payload: object, *, garment: str, continuation: bool,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != _SNAPSHOT_FIELDS:
        raise ValueError("fresh source snapshot does not use the canonical CPU schema")
    randomization = payload.get("randomization")
    expected_randomization = (
        {"strategy": "canonical", "continuation_step": 16}
        if continuation else {"strategy": "canonical"}
    )
    pose = payload.get("scene_state", {}).get("garment_reset_pose") if isinstance(payload.get("scene_state"), Mapping) else None
    if (
        payload.get("schema_version") != 3
        or payload.get("cloth_state_authority") != "usd_local_points_v1"
        or payload.get("garment_name") != garment
        or randomization != expected_randomization
        or not isinstance(payload.get("rng_state"), Mapping)
        or not isinstance(pose, list) or len(pose) != 6
        or any(not _finite_number(value) for value in pose)
    ):
        raise ValueError("fresh source snapshot is not canonical")
    for field, length in (("robot_position", 12), ("robot_velocity", 12)):
        values = payload.get(field)
        if not isinstance(values, list) or len(values) != length:
            raise ValueError("fresh source snapshot physical state is invalid")
        if any(not _finite_number(value) for value in values):
            raise ValueError("fresh source snapshot physical state is invalid")
    for field in ("cloth_position", "cloth_velocity"):
        values = payload.get(field)
        if (
            not isinstance(values, list) or not values
            or any(
                not isinstance(point, list) or len(point) != 3
                or any(not _finite_number(value) for value in point)
                for point in values
            )
        ):
            raise ValueError("fresh source snapshot physical state is invalid")
    _finite(payload, label="fresh source snapshot")
    return payload


def _canonical_episode(payload: object, *, attempt_id: str, category: str, garment: str, round_id: str, run_id: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("fresh source episode is malformed")
    identity, provenance = payload.get("identity"), payload.get("provenance")
    if not isinstance(identity, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("fresh source episode identity is malformed")
    devices = (provenance.get("renderer_device"), provenance.get("camera_device"), provenance.get("policy_device"))
    if (
        payload.get("episode_id") != attempt_id
        or payload.get("accepted_success") is not True or payload.get("outcome") != "success"
        or payload.get("randomization") != {"strategy": "canonical"}
        or identity.get("episode_id") != attempt_id or identity.get("category") != category
        or identity.get("garment_name") != garment or identity.get("release_stage") != "seen"
        or identity.get("campaign_round_id") != round_id or identity.get("campaign_run_id") != run_id
        or identity.get("policy_repo") != PARENT_POLICY_REPO
        or identity.get("policy_revision") != PARENT_REVISION or identity.get("policy_step") != 12_000
        or provenance.get("policy_artifact_sha256") != PARENT_ARTIFACT_SHA256
        or provenance.get("simulator_device") != "cpu" or provenance.get("cloth_device") != "cpu"
        or any(not isinstance(value, str) or _CUDA_DEVICE.fullmatch(value) is None for value in devices)
        or len(set(devices)) != 1
    ):
        raise ValueError("fresh source episode is not an exact canonical 12K CPU success")


def _verify_manifest_file(root: Path, path: Path, manifest: Mapping[str, object], *, label: str) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"fresh source {label} is not canonical") from error
    record = manifest.get(relative)
    if (
        not path.is_file() or path.is_symlink() or not isinstance(record, Mapping)
        or set(record) != {"sha256", "size"}
        or _hash_text_field(record.get("sha256"), label=f"fresh source {label} digest") != _sha256(path)
        or type(record.get("size")) is not int or record["size"] != path.stat().st_size
    ):
        raise ValueError("fresh source checksum manifest does not bind required evidence")
    return str(record["sha256"])


def _validate_report(
    report: Mapping[str, object], *, matrices: Mapping[str, tuple[str, Mapping[str, Mapping[str, object]]]],
) -> tuple[str, str, Mapping[str, Mapping[str, object]], Mapping[str, Mapping[str, object]]]:
    body = dict(report)
    declared = body.pop("report_sha256", None)
    expected_fields = {
        "schema_version", "kind", "campaign_kind", "logical_stage", "round_id", "matrix_sha256",
        "identity", "trials", "safety_failure", "report_sha256", "run_id",
    }
    identity = report.get("identity")
    round_id, run_id, matrix_digest = report.get("round_id"), report.get("run_id"), report.get("matrix_sha256")
    if (
        set(report) != expected_fields or declared != hashlib.sha256(_canonical_bytes(body)).hexdigest()
        or report.get("schema_version") != 1 or report.get("kind") != FRESH_SOURCE_REPORT_KIND
        or report.get("campaign_kind") != FRESH_SOURCE_CAMPAIGN_KIND
        or report.get("logical_stage") != FRESH_SOURCE_LOGICAL_STAGE
        or not isinstance(round_id, str) or _ROUND_ID.fullmatch(round_id) is None
        or not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None
        or report.get("safety_failure") is not False or not isinstance(identity, Mapping)
        or identity != {
            "policy_repo": PARENT_POLICY_REPO, "policy_revision": PARENT_REVISION,
            "policy_step": 12_000, "policy_artifact_sha256": PARENT_ARTIFACT_SHA256,
        }
        or not isinstance(matrix_digest, str) or matrix_digest not in matrices
        or not isinstance(report.get("trials"), list)
    ):
        raise ValueError("fresh source report is not authenticated for the exact 12K policy")
    matrix_path, matrix_rows = matrices[matrix_digest]
    trials: dict[str, Mapping[str, object]] = {}
    for trial in report["trials"]:
        if not isinstance(trial, Mapping) or not isinstance(trial.get("attempt_id"), str):
            raise ValueError("fresh source report trial is malformed")
        attempt = str(trial["attempt_id"])
        matrix_row = matrix_rows.get(attempt)
        devices = (trial.get("renderer_device"), trial.get("camera_device"), trial.get("policy_device"))
        is_success = (
            trial.get("accepted_success") is True
            and trial.get("official_success") is True
            and trial.get("outcome") == "success"
        )
        is_policy_failure = (
            trial.get("accepted_success") is False
            and trial.get("official_success") is False
            and trial.get("outcome") == "failure"
        )
        if (
            attempt in trials or matrix_row is None
            or trial.get("category") != matrix_row.get("category")
            or trial.get("garment_name") != matrix_row.get("garment_name")
            or type(trial.get("accepted_success")) is not bool or type(trial.get("official_success")) is not bool
            or not (is_success or is_policy_failure) or trial.get("simulator_device") != "cpu"
            or trial.get("cloth_device") != "cpu"
            or any(not isinstance(value, str) or _CUDA_DEVICE.fullmatch(value) is None for value in devices)
            or len(set(devices)) != 1
            or any(trial.get(field) is not False for field in ("safety_failure", "numerical_failure", "cloth_failure"))
            or (
                is_success and (
                    _hash_text_field(trial.get("artifact_sha256"), label="fresh source artifact digest") != trial.get("artifact_sha256")
                    or _hash_text_field(trial.get("hub_sync_receipt_sha256"), label="fresh source receipt digest") != trial.get("hub_sync_receipt_sha256")
                )
            )
            or trial.get("remote_prefix") != f"rollout-rounds/{round_id}/{attempt}"
            or trial.get("campaign_round_id") != round_id or trial.get("campaign_run_id") != run_id
        ):
            raise ValueError("fresh source report trial is not an eligible official CPU source")
        trials[attempt] = trial
    return str(matrix_path), str(matrix_digest), matrix_rows, trials


def authenticate_fresh_source_contract(
    source_reports: Sequence[Path], source_matrices: Sequence[Path],
) -> dict[str, dict[str, object]]:
    """Authenticate every source report/matrix trial and return canonical contexts."""

    reports = canonical_source_evidence_paths(source_reports, label="fresh source report")
    matrix_files = canonical_source_evidence_paths(source_matrices, label="fresh source matrix")
    if not reports or not matrix_files:
        raise ValueError("fresh visual-only replay requires source reports and source matrices")
    matrices: dict[str, tuple[str, Mapping[str, Mapping[str, object]]]] = {}
    matrix_attempts: set[str] = set()
    for path in matrix_files:
        rows = _load_json(path, label="fresh source matrix", object_only=False)
        if not isinstance(rows, list) or not rows:
            raise ValueError("fresh source matrix is malformed")
        digest = _sha256(path)
        by_attempt: dict[str, Mapping[str, object]] = {}
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("attempt_id"), str):
                raise ValueError("fresh source matrix row is malformed")
            attempt = str(row["attempt_id"])
            if (
                attempt in matrix_attempts or row.get("trial_id") != attempt
                or row.get("category") not in CATEGORIES
                or not isinstance(row.get("garment_name"), str) or not row["garment_name"]
                or row.get("release_stage") != "seen" or row.get("strategy") != "canonical"
                or row.get("campaign_kind") != FRESH_SOURCE_CAMPAIGN_KIND
                or row.get("logical_stage") != FRESH_SOURCE_LOGICAL_STAGE
                or not isinstance(row.get("campaign_round_id"), str)
                or not isinstance(row.get("campaign_run_id"), str)
            ):
                raise ValueError("fresh source matrix identity is invalid")
            matrix_attempts.add(attempt)
            by_attempt[attempt] = row
        if digest in matrices:
            raise ValueError("fresh source matrix is duplicated")
        matrices[digest] = (str(path), by_attempt)
    contexts: dict[str, dict[str, object]] = {}
    for path in reports:
        report = _load_json(path, label="fresh source report")
        if not isinstance(report, Mapping):
            raise ValueError("fresh source report is malformed")
        matrix_path, matrix_digest, matrix_rows, trials = _validate_report(report, matrices=matrices)
        report_digest = _sha256(path)
        for attempt, trial in trials.items():
            if attempt in contexts:
                raise ValueError("every fresh source must appear in exactly one report and source matrix")
            contexts[attempt] = {
                "trial": trial, "report": report, "source_report_path": str(path),
                "source_report_sha256": report_digest, "source_matrix_path": matrix_path,
                "source_matrix_sha256": matrix_digest, "source_matrix_rows": matrix_rows,
                "source_round_id": report["round_id"], "source_run_id": report["run_id"],
            }
    if set(contexts) != matrix_attempts:
        raise ValueError("every fresh source must appear in exactly one report and source matrix")
    return contexts


def authenticate_selected_fresh_source_artifacts(
    *, episode_root: str | Path, receipt_path: str | Path, attempt_id: str,
    category: str, garment: str, trial: Mapping[str, object], report: Mapping[str, object],
) -> dict[str, str | int]:
    """Authenticate the local artifact tree for one selected fresh replay parent."""

    round_id, run_id = report.get("round_id"), report.get("run_id")
    if (
        category not in CATEGORIES or not isinstance(round_id, str) or not isinstance(run_id, str)
        or trial.get("accepted_success") is not True
        or trial.get("official_success") is not True
        or trial.get("outcome") != "success"
        or trial.get("category") != category or trial.get("garment_name") != garment
    ):
        raise ValueError("fresh source parent is not an official accepted success")

    root = _absolute_directory(str(episode_root), label="fresh source artifact root")
    receipt = _regular_absolute(str(receipt_path), label="fresh source receipt")
    episode = root / "raw" / attempt_id / "episode.json"
    reset = root / "raw" / attempt_id / "snapshots" / "reset.json"
    annotations = root / "raw" / attempt_id / "annotations.jsonl"
    snapshot = root / "raw" / attempt_id / "snapshots" / "continuations" / "000016.json"
    for path, label in ((episode, "episode"), (reset, "reset"), (annotations, "annotations"), (snapshot, "continuation snapshot")):
        _regular_absolute(str(path), label=f"fresh source {label}")
    manifest = _load_json(root / "SHA256SUMS.json", label="fresh source checksum manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("fresh source checksum manifest is malformed")
    checked = {
        "episode": _verify_manifest_file(root, episode, manifest, label="episode"),
        "reset": _verify_manifest_file(root, reset, manifest, label="reset"),
        "annotations": _verify_manifest_file(root, annotations, manifest, label="annotations"),
        "h16": _verify_manifest_file(root, snapshot, manifest, label="continuation snapshot"),
    }
    artifact_sha256 = _episode_artifact_sha256(root)
    receipt_sha256 = _sha256(receipt)
    if (
        artifact_sha256 != trial.get("artifact_sha256")
        or receipt_sha256 != trial.get("hub_sync_receipt_sha256")
    ):
        raise ValueError("fresh source artifact hashes are not authenticated")
    episode_json = _load_json(episode, label="fresh source episode")
    reset_json = _load_json(reset, label="fresh source reset snapshot")
    snapshot_json = _load_json(snapshot, label="fresh source continuation snapshot")
    _canonical_episode(
        episode_json, attempt_id=attempt_id, category=category, garment=garment,
        round_id=round_id, run_id=run_id,
    )
    _canonical_snapshot(reset_json, garment=garment, continuation=False)
    continuation = _canonical_snapshot(snapshot_json, garment=garment, continuation=True)
    receipt_json = _load_json(receipt, label="fresh source receipt")
    expected_prefix = f"rollout-rounds/{round_id}/{attempt_id}"
    immutable_revision = receipt_json.get("immutable_revision") if isinstance(receipt_json, Mapping) else None
    if (
        not isinstance(receipt_json, Mapping) or receipt_json.get("readback_verified") is not True
        or receipt_json.get("attempt_id") != attempt_id or receipt_json.get("round_id") != round_id
        or receipt_json.get("run_id") != run_id or receipt_json.get("episode_sha256") != artifact_sha256
        or receipt_json.get("remote_prefix") != expected_prefix or trial.get("remote_prefix") != expected_prefix
        or not isinstance(immutable_revision, str) or _REVISION.fullmatch(immutable_revision) is None
    ):
        raise ValueError("fresh source receipt does not bind the immutable artifact")
    return {
        "source_episode_sha256": artifact_sha256,
        "source_episode_root": str(root.resolve(strict=True)),
        "source_episode_path": str(episode.resolve(strict=True)),
        "source_reset_sha256": checked["reset"],
        "source_annotations_sha256": checked["annotations"],
        "source_continuation_snapshot_sha256": checked["h16"],
        "source_state_fingerprint": _state_fingerprint(
            category=category, garment=garment, continuation=continuation,
        ),
        "source_receipt_sha256": receipt_sha256,
        "source_receipt_path": str(receipt.resolve(strict=True)),
        "source_remote_prefix": expected_prefix,
        "source_immutable_revision": immutable_revision,
        "restore_snapshot": str(snapshot.resolve(strict=True)),
        "restore_snapshot_sha256": checked["h16"],
        "restore_snapshot_cloth_frame": "usd_local_points_v1",
        "restore_snapshot_step": 16,
    }


def validate_exact_fresh_visual_only(
    *, matrix_path: str | Path, max_attempts: int, source_reports_json: str, source_matrices_json: str,
) -> None:
    """Authenticate the only tuple allowed to raise success replay to 200."""

    if max_attempts != 400:
        raise ValueError("fresh visual-only replay requires exactly 400 attempts")
    matrix = _regular_absolute(str(matrix_path), label="fresh visual-only replay matrix")
    rows = _load_json(matrix, label="fresh visual-only replay matrix", object_only=False)
    if not isinstance(rows, list) or len(rows) != 400:
        raise ValueError("fresh visual-only replay requires exactly 400 rows")
    reports = {
        str(path.resolve(strict=True)): (digest, path.resolve(strict=True))
        for _, (digest, path) in _evidence(source_reports_json, label="source report").items()
    }
    raw_matrices = {
        str(path.resolve(strict=True)): (digest, path.resolve(strict=True))
        for _, (digest, path) in _evidence(source_matrices_json, label="source matrix").items()
    }
    report_context = authenticate_fresh_source_contract(
        tuple(path for _, path in reports.values()),
        tuple(path for _, path in raw_matrices.values()),
    )
    report_trials = {
        attempt: (
            context["trial"], context["report"], context["source_report_sha256"],
            context["source_matrix_path"], context["source_matrix_rows"],
        )
        for attempt, context in report_context.items()
    }

    counts: Counter[str] = Counter()
    seen_attempts: set[str] = set()
    seen_seeds: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("fresh visual-only replay row is malformed")
        category, attempt_id, seed, parent = row.get("category"), row.get("attempt_id"), row.get("seed"), row.get("parent_episode_id")
        if (
            category not in CATEGORIES or not isinstance(attempt_id, str) or not attempt_id
            or row.get("trial_id") != attempt_id or attempt_id in seen_attempts
            or type(seed) is not int or seed in seen_seeds
            or row.get("strategy") != "visual_only" or row.get("category_acceptance_cap") != 50
            or row.get("replay_kind") != "verified_success_early_snapshot_v1" or row.get("restore_snapshot_step") != 16
            or not isinstance(parent, str) or not parent or row.get("lineage_id") != parent
            or not _FRESH_FIELDS <= set(row)
            or any(_hash_text_field(row.get(field), label=f"fresh row {field}") != row.get(field) for field in _FRESH_FIELDS - _FRESH_PATH_FIELDS - _FRESH_STRING_FIELDS)
            or any(not isinstance(row.get(field), str) for field in _FRESH_PATH_FIELDS | _FRESH_STRING_FIELDS)
            or not isinstance(row.get("source_round_id"), str) or _ROUND_ID.fullmatch(str(row["source_round_id"])) is None
            or not isinstance(row.get("source_run_id"), str) or _RUN_ID.fullmatch(str(row["source_run_id"])) is None
            or _REVISION.fullmatch(str(row.get("source_immutable_revision"))) is None
            or row.get("source_remote_prefix") != f"rollout-rounds/{row['source_round_id']}/{parent}"
        ):
            raise ValueError("fresh visual-only replay row is not fully bound")
        report_path = str(row["source_report_path"])
        matrix_path_text = str(row["source_matrix_path"])
        if (
            report_path not in reports or matrix_path_text not in raw_matrices
            or row.get("source_report_sha256") != reports[report_path][0]
            or row.get("source_matrix_sha256") != raw_matrices[matrix_path_text][0]
            or parent not in report_trials
        ):
            raise ValueError("fresh source report or matrix binding is invalid")
        trial, report, _, authenticated_matrix_path, matrix_rows = report_trials[parent]
        source_row = matrix_rows.get(parent)
        round_id, run_id = report.get("round_id"), report.get("run_id")
        if (
            matrix_path_text != authenticated_matrix_path or report.get("matrix_sha256") != row.get("source_matrix_sha256")
            or round_id != row.get("source_round_id") or run_id != row.get("source_run_id")
            or not isinstance(source_row, Mapping) or source_row.get("category") != category
            or source_row.get("garment_name") != row.get("garment")
            or source_row.get("campaign_round_id") != round_id or source_row.get("campaign_run_id") != run_id
            or trial.get("accepted_success") is not True or trial.get("official_success") is not True
            or trial.get("outcome") != "success"
            or trial.get("artifact_sha256") != row.get("source_episode_sha256")
            or trial.get("hub_sync_receipt_sha256") != row.get("source_receipt_sha256")
        ):
            raise ValueError("fresh source report/matrix row is not authenticated")

        authenticated = authenticate_selected_fresh_source_artifacts(
            episode_root=str(row["source_episode_root"]), receipt_path=str(row["source_receipt_path"]),
            attempt_id=parent, category=str(category), garment=str(row.get("garment")),
            trial=trial, report=report,
        )
        if any(row.get(field) != value for field, value in authenticated.items()):
            raise ValueError("fresh source artifact hashes are not authenticated")
        seen_attempts.add(attempt_id)
        seen_seeds.add(seed)
        counts[str(category)] += 1
    if counts != Counter({category: 100 for category in CATEGORIES}):
        raise ValueError("fresh visual-only replay requires exactly 100 rows per category")
