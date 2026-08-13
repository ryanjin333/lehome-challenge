"""Deterministic LeRobot v3-to-v2 repackaging for pinned GR00T N1.7."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import threading
import uuid
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from lehome_train.constants import ISAAC_GROOT_REVISION
from lehome_train.data.inspect import (
    artifact_identities,
    format_v3_path,
    inspect_dataset,
    load_v3_episode_records,
    read_json_object,
)
from lehome_train.data.mapping import (
    ACTION_HORIZON,
    FIXED_INSTRUCTION,
    JOINT_NAMES,
    load_checked_mapping,
)
from lehome_train.data.split import split_episode_ids
from lehome_train.io import (
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
)


LEGACY_DATA_PATH = (
    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
)
LEGACY_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/"
    "episode_{episode_index:06d}.mp4"
)
_JOURNAL_NAME = "conversion-journal.json"
_DERIVED_STATISTICS_PATHS = (
    "meta/stats.json", "meta/relative_stats.json", "meta/lehome_groot_modality.py",
)
# The journal is a local crash-recovery record, not adversarially authenticated
# storage.  We fail closed on mismatched known receipts; a coordinated local
# rewrite of both content and receipt requires a separate signed trust root.


def _same_or_nested(left: Path, right: Path) -> bool:
    left_resolved, right_resolved = left.resolve(strict=False), right.resolve(strict=False)
    return left_resolved == right_resolved or left_resolved in right_resolved.parents or right_resolved in left_resolved.parents


def _exclusive_sibling_lock(
    root: Path, *, suffix: str, ownership: str,
) -> tuple[Path, tuple[int, int, bytes]]:
    """Claim one conversion resource with an inode- and token-bound lock."""
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.parent / f".{root.name}.{suffix}.lock"
    if lock.is_symlink():
        raise ValueError(f"persistent conversion {ownership} lock is unsafe")
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError(f"persistent conversion {ownership} is already owned") from None
    token = uuid.uuid4().hex.encode()
    try:
        os.write(descriptor, token)
        identity = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    return lock, (identity.st_dev, identity.st_ino, token)


def _persistent_lock(staging: Path) -> tuple[Path, tuple[int, int, bytes]]:
    return _exclusive_sibling_lock(staging, suffix="conversion", ownership="staging")


def _release_persistent_lock(lock: Path, identity: tuple[int, int, bytes]) -> None:
    try:
        current = lock.stat()
        if (
            (current.st_dev, current.st_ino) == identity[:2]
            and lock.is_file()
            and not lock.is_symlink()
            and lock.read_bytes() == identity[2]
        ):
            lock.unlink()
    except FileNotFoundError:
        pass


@contextmanager
def persistent_destination_operation_lock(destination: str | Path) -> Iterable[None]:
    """Serialize persistent CLI conversion and statistics recovery for one output."""
    lock, identity = _exclusive_sibling_lock(
        Path(destination), suffix="data-convert-operation", ownership="destination",
    )
    try:
        yield
    finally:
        _release_persistent_lock(lock, identity)


def _sha256_size(path: Path) -> dict[str, object]:
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    return {"sha256": digest, "byte_size": path.stat().st_size}


def _matches_receipt(path: Path, receipt: object) -> bool:
    return (
        isinstance(receipt, Mapping)
        and not path.is_symlink()
        and path.is_file()
        and receipt == _sha256_size(path)
    )


def _valid_receipt(receipt: object) -> bool:
    return (
        isinstance(receipt, Mapping)
        and set(receipt) == {"sha256", "byte_size"}
        and isinstance(receipt.get("sha256"), str)
        and __import__("re").fullmatch(r"[0-9a-f]{64}", receipt["sha256"]) is not None
        and type(receipt.get("byte_size")) is int
        and receipt["byte_size"] >= 0
    )


def _expected_episode_table(table: pa.Table, episode_id: int) -> pa.Table:
    task_index = table.schema.get_field_index("task_index")
    if task_index < 0:
        raise ValueError(f"episode {episode_id} has no task_index column")
    return table.set_column(
        task_index, "task_index", pa.array([0] * table.num_rows, type=pa.int64())
    )


def _parquet_matches_expected(path: Path, expected: pa.Table) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        actual = pq.read_table(path)
    except (OSError, pa.ArrowInvalid):
        return False
    return actual.schema == expected.schema and actual.equals(expected)


def _journal_identity(
    *, inspection: Mapping[str, Any], mapping: Mapping[str, Any], source_repository: str,
    source_revision: str, converter_commit: str, converter_container_digest: str,
    split_seed: int, validation_fraction: float, records: list[dict[str, Any]],
    camera_keys: list[str], info: Mapping[str, Any], source: Path,
) -> dict[str, object]:
    chunks_size = int(info["chunks_size"])
    jobs: dict[str, object] = {}
    for record in sorted(records, key=lambda item: int(item["episode_index"])):
        episode_id = int(record["episode_index"])
        jobs[_legacy_path(Path("."), LEGACY_DATA_PATH, episode_id=episode_id, chunks_size=chunks_size).as_posix()] = {
            "kind": "data", "episode_id": episode_id,
            "source_range": [int(record["dataset_from_index"]), int(record["dataset_to_index"])],
        }
        for camera_key in camera_keys:
            relative_source = format_v3_path(source, str(info["video_path"]), chunk_index=int(record[f"videos/{camera_key}/chunk_index"]), file_index=int(record[f"videos/{camera_key}/file_index"]), video_key=camera_key).relative_to(source).as_posix()
            relative = _legacy_path(Path("."), LEGACY_VIDEO_PATH, episode_id=episode_id, chunks_size=chunks_size, video_key=camera_key).as_posix()
            jobs[relative] = {"kind": "video", "episode_id": episode_id, "camera_key": camera_key, "source_path": relative_source, "start": float(record[f"videos/{camera_key}/from_timestamp"]), "frame_count": int(record["dataset_to_index"]) - int(record["dataset_from_index"]), "fps": float(info["fps"])}
    return {
        "schema_version": 1, "kind": "lerobot_v3_conversion_journal",
        "source_repository": source_repository, "source_revision": source_revision,
        "source_manifest_sha256": inspection["source_manifest_sha256"],
        "source_artifacts": inspection["source_artifacts"],
        "mapping_sha256": canonical_json_sha256(mapping), "converter_commit": converter_commit,
        "converter_container_digest": converter_container_digest,
        "pinned_groot_revision": ISAAC_GROOT_REVISION, "split_seed": split_seed,
        "validation_fraction": validation_fraction, "jobs": dict(sorted(jobs.items())),
    }


def _load_or_create_journal(staging: Path, identity: Mapping[str, object]) -> tuple[dict[str, object], threading.Lock]:
    journal_path = staging / _JOURNAL_NAME
    if journal_path.exists():
        if journal_path.is_symlink() or not journal_path.is_file():
            raise ValueError("persistent conversion journal is unsafe")
        current = read_json_object(journal_path)
        receipts = current.pop("receipts", None)
        if current != identity or not isinstance(receipts, Mapping):
            raise ValueError("persistent conversion journal identity is incompatible")
        journal = dict(identity) | {"receipts": dict(receipts)}
    else:
        if any(staging.iterdir()):
            raise ValueError("persistent staging root must be empty before its first journal")
        journal = dict(identity) | {"receipts": {}}
        atomic_write_json(journal_path, journal)
    return journal, threading.Lock()


def _validate_staging_tree(staging: Path, journal: Mapping[str, object]) -> None:
    jobs = journal.get("jobs")
    receipts = journal.get("receipts")
    if not isinstance(jobs, Mapping) or not isinstance(receipts, Mapping):
        raise ValueError("persistent conversion journal is malformed")
    allowed = set(jobs) | {_JOURNAL_NAME} | {
        "meta/info.json", "meta/episodes.jsonl", "meta/episodes_stats.jsonl",
        "meta/tasks.jsonl", "meta/modality.json", "meta/lehome_mapping.json",
        "manifest.json",
    }
    for path in staging.rglob("*"):
        if path.is_symlink() or (path.exists() and not path.is_dir() and not path.is_file()):
            raise ValueError("persistent staging contains unsafe entry")
        if path.is_file() and path.relative_to(staging).as_posix() not in allowed:
            raise ValueError("persistent staging contains unexpected file")
    if not set(receipts).issubset(set(jobs)):
        raise ValueError("persistent conversion journal has unexpected receipts")
    if not all(_valid_receipt(receipt) for receipt in receipts.values()):
        raise ValueError("persistent conversion journal has malformed receipts")
    for relative, receipt in receipts.items():
        path = staging / str(relative)
        if path.exists() and not _matches_receipt(path, receipt):
            raise ValueError("persistent conversion receipted file changed")


def _record_receipt(staging: Path, journal: dict[str, object], lock: threading.Lock, relative: str) -> None:
    with lock:
        receipts = journal["receipts"]
        assert isinstance(receipts, dict)
        receipts[relative] = _sha256_size(staging / relative)
        atomic_write_json(staging / _JOURNAL_NAME, journal)


def _claim_and_adopt_orphan_data(
    *, orphan: Path, staging: Path, info: Mapping[str, Any], records: list[dict[str, Any]],
    tables: Mapping[int, pa.Table], camera_keys: list[str], journal: dict[str, object], lock: threading.Lock,
) -> None:
    """Claim once, then copy only exact source-semantic Parquets (never video)."""
    claimed = orphan.with_name(orphan.name + ".conversion-quarantine")
    if orphan.exists():
        if orphan.is_symlink() or not orphan.is_dir():
            raise ValueError("unbound staging adoption root is unsafe")
        if claimed.exists():
            raise ValueError("unbound staging quarantine already exists")
        orphan.replace(claimed)
    if claimed.is_symlink() or not claimed.is_dir():
        raise ValueError("unbound staging quarantine is unavailable")
    expected: dict[str, pa.Table] = {}
    chunks = int(info["chunks_size"])
    for record in records:
        episode = int(record["episode_index"])
        relative = _legacy_path(Path("."), LEGACY_DATA_PATH, episode_id=episode, chunks_size=chunks).as_posix()
        expected[relative] = _expected_episode_table(tables[episode], episode)
    actual: set[str] = set()
    expected_videos: set[str] = set()
    for record in records:
        episode = int(record["episode_index"])
        for camera in camera_keys:
            expected_videos.add(_legacy_path(Path("."), LEGACY_VIDEO_PATH, episode_id=episode, chunks_size=chunks, video_key=camera).as_posix())
    for path in claimed.rglob("*"):
        if path.is_symlink() or (path.exists() and not path.is_dir() and not path.is_file()):
            raise ValueError("unbound staging quarantine contains unsafe entry")
        if path.is_file():
            relative = path.relative_to(claimed).as_posix()
            if relative.startswith("meta/"):
                raise ValueError("unbound staging quarantine metadata is not adoptable")
            if relative.startswith("data/"):
                actual.add(relative)
            elif relative.startswith("videos/"):
                if relative not in expected_videos:
                    raise ValueError("unbound staging quarantine has missing or extra video jobs")
            else:
                raise ValueError("unbound staging quarantine has unexpected path")
    if actual != set(expected):
        raise ValueError("unbound staging quarantine has missing or extra episode Parquets")
    for relative, table in expected.items():
        source = claimed / relative
        if not _parquet_matches_expected(source, table):
            raise ValueError("unbound staging episode Parquet is not source-semantically exact")
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if not _parquet_matches_expected(destination, table):
            raise ValueError("adopted episode Parquet failed destination semantic verification")
        _record_receipt(staging, journal, lock, relative)


def _persistent_destination_manifest(
    destination: Path,
    *, inspection: Mapping[str, Any], mapping: Mapping[str, Any],
    source_repository: str, source_revision: str, converter_commit: str,
    converter_container_digest: str, split_seed: int, validation_fraction: float,
    records: list[dict[str, Any]], source_dataset: str,
) -> dict[str, object]:
    """Accept a promoted conversion only as a byte-authenticated resume point."""
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("persistent converted destination is unsafe")
    _validate_destination_tree_nodes(destination)
    manifest = read_json_object(destination / "manifest.json")
    split = split_episode_ids(
        tuple(str(item) for item in inspection["episode_ids"]),
        seed=split_seed, validation_fraction=validation_fraction,
    )
    # These are the complete deterministic conversion fields that downstream
    # loader/statistics code trusts, beyond source provenance alone.
    required = {
        "source_format": "lerobot_v3_sharded", "output_format": "groot_lerobot_v2.1_per_episode",
        "schema_version": 1, "source_dataset": source_dataset,
        "source_repository": source_repository, "source_revision": source_revision,
        "source_manifest_sha256": inspection["source_manifest_sha256"],
        "source_artifacts": inspection["source_artifacts"],
        "mapping_sha256": canonical_json_sha256(mapping), "converter_commit": converter_commit,
        "converter_container_digest": converter_container_digest,
        "pinned_groot_revision": ISAAC_GROOT_REVISION, "split_seed": split_seed,
        "validation_fraction": validation_fraction,
        "fps": inspection["fps"], "frame_count": inspection["frame_count"],
        "episode_count": inspection["episode_count"],
        "train_episode_ids": list(split.train), "validation_episode_ids": list(split.validation),
        "camera_schema": [
            {**camera, "dtype": "video", "shape": [480, 640, 3]}
            for camera in mapping["cameras"]
        ],
        "state_schema": {"source_key": "observation.state", "dimension": 12, "names": list(JOINT_NAMES)},
        "action_schema": mapping["action"],
        "fixed_language_instruction": FIXED_INSTRUCTION,
        "future_actions": {
            "horizon": ACTION_HORIZON, "loader_allow_padding": False,
            "materialized_windows": False, "tail_convention": "drop_incomplete_windows",
            "valid_action_mask": "implicit_all_true_for_emitted_windows",
            "valid_window_counts": {
                str(record["episode_index"]): max(0, int(record["dataset_to_index"]) - int(record["dataset_from_index"]) - ACTION_HORIZON + 1)
                for record in sorted(records, key=lambda item: int(item["episode_index"]))
            },
        },
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("persistent converted destination contract is incompatible")
    artifacts = manifest.get("output_artifacts")
    if not isinstance(artifacts, list) or manifest.get("output_manifest_sha256") != canonical_json_sha256(artifacts):
        raise ValueError("persistent converted destination artifact manifest is invalid")
    actual = artifact_identities(destination, exclude={"manifest.json"})
    # Statistics runs after atomic conversion promotion.  It records its three
    # generated files in the manifest, so allow only those exact dynamic paths.
    statistics = manifest.get("statistics")
    dynamic: dict[str, str] = {}
    if statistics == {"status": "pending_task_4_train_only", "files": []}:
        pass
    elif isinstance(statistics, Mapping) and statistics.get("status") == "computed_task_4_train_only":
        files = statistics.get("files")
        if not isinstance(files, list):
            raise ValueError("persistent converted destination statistics are invalid")
        for item in files:
            if not isinstance(item, Mapping) or not isinstance(item.get("relative_path"), str) or not isinstance(item.get("sha256"), str):
                raise ValueError("persistent converted destination statistics are invalid")
            dynamic[item["relative_path"]] = item["sha256"]
        if set(dynamic) != set(_DERIVED_STATISTICS_PATHS):
            raise ValueError("persistent converted destination statistics are incomplete")
    else:
        raise ValueError("persistent converted destination statistics are invalid")
    observed = {item["relative_path"]: item for item in actual}
    partial_derived = set(_DERIVED_STATISTICS_PATHS).intersection(observed).difference(dynamic)
    if partial_derived:
        if not partial_derived.issubset(set(_DERIVED_STATISTICS_PATHS)):
            raise ValueError("persistent converted destination has unexpected derived outputs")
        for relative in partial_derived:
            path = destination / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError("persistent converted destination derived output is unsafe")
            path.unlink()
        actual = artifact_identities(destination, exclude={"manifest.json"})
        observed = {item["relative_path"]: item for item in actual}
    if any(path not in observed or observed[path]["sha256"] != digest for path, digest in dynamic.items()):
        raise ValueError("persistent converted destination statistics changed")
    actual_base = {path: item for path, item in observed.items() if path not in dynamic}
    listed = {item.get("relative_path"): item for item in artifacts if isinstance(item, Mapping)}
    if listed != actual_base:
        raise ValueError("persistent converted destination artifact tree changed")
    return manifest


def _validate_destination_tree_nodes(destination: Path) -> None:
    """Reject nodes that artifact hashing intentionally does not traverse."""
    for path in destination.rglob("*"):
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise ValueError("persistent converted destination is unsafe") from error
        if stat.S_ISDIR(mode) or stat.S_ISREG(mode):
            continue
        raise ValueError("persistent converted destination contains unsafe entry")


def _write_json_lines(path: Path, values: Iterable[Mapping[str, object]]) -> None:
    payload = b"\n".join(canonical_json_bytes(value) for value in values) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _validate_hex_revision(value: str, field_name: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a full lowercase immutable commit")


def _validate_provenance(
    source_repository: str,
    source_revision: str,
    converter_commit: str,
    converter_container_digest: str,
) -> None:
    if not source_repository or source_repository.strip() != source_repository:
        raise ValueError("source_repository must be a nonempty immutable repository ID")
    _validate_hex_revision(source_revision, "source_revision")
    _validate_hex_revision(converter_commit, "converter_commit")
    if (
        len(converter_container_digest) != 71
        or not converter_container_digest.startswith("sha256:")
        or any(
            character not in "0123456789abcdef"
            for character in converter_container_digest[7:]
        )
    ):
        raise ValueError("converter container digest must be sha256:<64 lowercase hex>")


def _group_data_records(
    records: Iterable[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                int(record["data/chunk_index"]),
                int(record["data/file_index"]),
            )
        ].append(record)
    return grouped


def _load_episode_tables(
    source: Path,
    info: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> dict[int, pa.Table]:
    """Apply pinned v3 global-index offsets to reconstruct episode tables."""

    pattern = info["data_path"]
    if not isinstance(pattern, str):
        raise ValueError("info data_path must be a string")
    tables: dict[int, pa.Table] = {}
    for (chunk_index, file_index), file_records in sorted(
        _group_data_records(records).items()
    ):
        source_path = format_v3_path(
            source,
            pattern,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        table = pq.read_table(source_path)
        ordered = sorted(
            file_records,
            key=lambda record: int(record["dataset_from_index"]),
        )
        file_offset = int(ordered[0]["dataset_from_index"])
        for record in ordered:
            episode_id = int(record["episode_index"])
            start = int(record["dataset_from_index"]) - file_offset
            stop = int(record["dataset_to_index"]) - file_offset
            tables[episode_id] = table.slice(start, stop - start)
    return tables


def _legacy_path(
    root: Path,
    pattern: str,
    *,
    episode_id: int,
    chunks_size: int,
    video_key: str | None = None,
) -> Path:
    path = root / pattern.format(
        episode_chunk=episode_id // chunks_size,
        episode_index=episode_id,
        video_key=video_key,
    )
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("legacy output path escapes conversion root") from error
    return path


def _write_episode_data(
    output: Path,
    info: Mapping[str, Any],
    records: list[dict[str, Any]],
    tables: Mapping[int, pa.Table],
    *,
    journal: dict[str, object] | None = None,
    journal_lock: threading.Lock | None = None,
) -> None:
    chunks_size = int(info["chunks_size"])
    for record in sorted(records, key=lambda item: int(item["episode_index"])):
        episode_id = int(record["episode_index"])
        table = _expected_episode_table(tables[episode_id], episode_id)
        destination = _legacy_path(
            output,
            LEGACY_DATA_PATH,
            episode_id=episode_id,
            chunks_size=chunks_size,
        )
        relative = destination.relative_to(output).as_posix()
        receipt = None if journal is None else journal["receipts"].get(relative)  # type: ignore[index]
        if _matches_receipt(destination, receipt) and _parquet_matches_expected(destination, table):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination, compression="zstd")
        if journal is not None and journal_lock is not None:
            _record_receipt(output, journal, journal_lock, relative)


def _extract_video_segment(
    source: Path,
    destination: Path,
    *,
    start: float,
    expected_frame_count: int,
    expected_fps: float,
) -> None:
    if not source.is_file() or source.suffix.lower() != ".mp4":
        raise ValueError(f"invalid source MP4 shard: {source}")
    if (
        not math.isfinite(start)
        or start < 0
        or expected_frame_count <= 0
        or not math.isfinite(expected_fps)
        or expected_fps <= 0
    ):
        raise ValueError("invalid v3 video timestamp slice")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ss",
        f"{start:.9f}",
        "-frames:v",
        str(expected_frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{expected_fps:.12g}",
        "-g",
        "30",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-y",
        str(destination),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            timeout=300,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is required for LeRobot v3 video slicing") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"ffmpeg timed out slicing {source}") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if error.stderr else "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg failed slicing {source}: {detail}") from error
    _validate_output_video(
        destination,
        expected_frame_count=expected_frame_count,
        expected_fps=expected_fps,
    )


def _validate_output_video(
    path: Path,
    *,
    expected_frame_count: int,
    expected_fps: float,
) -> None:
    """Fail closed unless ffprobe observes the exact intended temporal schema."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            timeout=60,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        streams = payload.get("streams")
        if not isinstance(streams, list) or len(streams) != 1:
            raise ValueError("expected exactly one output video stream")
        stream = streams[0]
        frame_count = int(stream["nb_read_frames"])
        numerator, denominator = stream["avg_frame_rate"].split("/", maxsplit=1)
        fps = int(numerator) / int(denominator)
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ) as error:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"could not validate converted video: {path}") from error
    if frame_count != expected_frame_count or abs(fps - expected_fps) > 1e-9:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            "converted video temporal schema mismatch: "
            f"expected {expected_frame_count} frames at {expected_fps} FPS, "
            f"observed {frame_count} frames at {fps} FPS"
        )


def _write_episode_videos(
    source: Path,
    output: Path,
    info: Mapping[str, Any],
    records: list[dict[str, Any]],
    camera_keys: list[str],
    *,
    journal: dict[str, object] | None = None,
    journal_lock: threading.Lock | None = None,
) -> None:
    source_pattern = info["video_path"]
    if not isinstance(source_pattern, str):
        raise ValueError("info video_path must be a string")
    chunks_size = int(info["chunks_size"])
    fps = float(info["fps"])
    jobs: list[tuple[Path, Path, float, int, float]] = []
    for camera_key in camera_keys:
        for record in sorted(records, key=lambda item: int(item["episode_index"])):
            episode_id = int(record["episode_index"])
            source_video = format_v3_path(
                source,
                source_pattern,
                chunk_index=int(record[f"videos/{camera_key}/chunk_index"]),
                file_index=int(record[f"videos/{camera_key}/file_index"]),
                video_key=camera_key,
            )
            destination = _legacy_path(
                output,
                LEGACY_VIDEO_PATH,
                episode_id=episode_id,
                chunks_size=chunks_size,
                video_key=camera_key,
            )
            jobs.append(
                (
                    source_video,
                    destination,
                    float(record[f"videos/{camera_key}/from_timestamp"]),
                    int(record["dataset_to_index"])
                    - int(record["dataset_from_index"]),
                    fps,
                ),
            )

    def run(job: tuple[Path, Path, float, int, float]) -> None:
        source_video, destination, start, frame_count, expected_fps = job
        relative = destination.relative_to(output).as_posix()
        receipt = None if journal is None else journal["receipts"].get(relative)  # type: ignore[index]
        if _matches_receipt(destination, receipt):
            try:
                _validate_output_video(destination, expected_frame_count=frame_count, expected_fps=expected_fps)
                return
            except RuntimeError:
                pass
        _extract_video_segment(
            source_video,
            destination,
            start=start,
            expected_frame_count=frame_count,
            expected_fps=expected_fps,
        )
        if journal is not None and journal_lock is not None:
            _record_receipt(output, journal, journal_lock, relative)

    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as executor:
        tuple(executor.map(run, jobs))


def _v2_info(
    source_info: Mapping[str, Any],
    *,
    episode_count: int,
    camera_count: int,
) -> dict[str, Any]:
    info = dict(source_info)
    info["codebase_version"] = "v2.1"
    info["data_path"] = LEGACY_DATA_PATH
    info["video_path"] = LEGACY_VIDEO_PATH
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    features = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in info["features"].items()
    }
    for feature in features.values():
        if isinstance(feature, dict) and feature.get("dtype") != "video":
            feature.pop("fps", None)
    info["features"] = features
    chunks_size = int(info["chunks_size"])
    info["total_chunks"] = math.ceil(episode_count / chunks_size)
    info["total_videos"] = episode_count * camera_count
    info["total_tasks"] = 1
    return info


def _v2_episode_metadata(record: Mapping[str, Any]) -> dict[str, object]:
    episode = {
        key: value
        for key, value in record.items()
        if not key.startswith(("data/", "videos/", "stats/", "meta/"))
        and key not in {"dataset_from_index", "dataset_to_index"}
    }
    episode["episode_index"] = int(record["episode_index"])
    episode["length"] = int(record["dataset_to_index"]) - int(
        record["dataset_from_index"]
    )
    episode["tasks"] = [FIXED_INSTRUCTION]
    return episode


def _modality_metadata() -> dict[str, object]:
    groups = {
        "left_arm": {"start": 0, "end": 5},
        "left_gripper": {"start": 5, "end": 6},
        "right_arm": {"start": 6, "end": 11},
        "right_gripper": {"start": 11, "end": 12},
    }
    return {
        "video": {
            camera: {"original_key": f"observation.images.{camera}"}
            for camera in ("top_rgb", "left_rgb", "right_rgb")
        },
        "state": {
            key: {**bounds, "original_key": "observation.state"}
            for key, bounds in groups.items()
        },
        "action": {
            key: {**bounds, "original_key": "action"}
            for key, bounds in groups.items()
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        },
    }


def convert_dataset(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    mapping_path: str | Path | None,
    source_repository: str,
    source_revision: str,
    converter_commit: str,
    converter_container_digest: str,
    split_seed: int = 42,
    validation_fraction: float = 0.1,
    persistent_staging_root: str | Path | None = None,
    unbound_staging_data_adoption_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate and convert a complete local v3 snapshot without remote writes."""

    mapping = load_checked_mapping(mapping_path)
    _validate_provenance(
        source_repository,
        source_revision,
        converter_commit,
        converter_container_digest,
    )
    source = Path(source_path)
    destination = Path(destination_path)
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("conversion destination must not be inside the source dataset")
    inspection = inspect_dataset(source)
    if not inspection["valid"]:
        errors = inspection["validation_errors"]
        raise ValueError(f"source schema validation failed: {errors[0]}")
    if inspection["proposed_mapping"] != mapping:
        raise ValueError("checked mapping does not match observed source schema")
    if destination.exists():
        if persistent_staging_root is None:
            raise FileExistsError(
                f"refusing to overwrite existing conversion destination: {destination}"
            )
        return _persistent_destination_manifest(
            destination, inspection=inspection, mapping=mapping,
            source_repository=source_repository, source_revision=source_revision,
            converter_commit=converter_commit,
            converter_container_digest=converter_container_digest, split_seed=split_seed,
            validation_fraction=validation_fraction, records=load_v3_episode_records(source), source_dataset=source.name,
        )

    split = split_episode_ids(
        inspection["episode_ids"],
        seed=split_seed,
        validation_fraction=validation_fraction,
    )
    info = read_json_object(source / "meta" / "info.json")
    records = load_v3_episode_records(source)
    camera_keys = [camera["source_key"] for camera in mapping["cameras"]]
    episode_tables = _load_episode_tables(source, info, records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    persistent = persistent_staging_root is not None
    lock_path: Path | None = None
    lock_identity: tuple[int, int, bytes] | None = None
    if persistent:
        temporary = Path(persistent_staging_root)  # type: ignore[arg-type]
        adoption = None if unbound_staging_data_adoption_root is None else Path(unbound_staging_data_adoption_root)
        quarantine = None if adoption is None else adoption.with_name(adoption.name + ".conversion-quarantine")
        if _same_or_nested(temporary, source) or _same_or_nested(temporary, destination) or (adoption is not None and _same_or_nested(temporary, adoption)) or (quarantine is not None and _same_or_nested(temporary, quarantine)):
            raise ValueError("persistent staging root must not overlap source, destination, or adoption roots")
        if destination.parent.exists() and temporary.parent.exists() and destination.parent.stat().st_dev != temporary.parent.stat().st_dev:
            raise ValueError("persistent staging and destination must share a filesystem for atomic promotion")
        if temporary.is_symlink():
            raise ValueError("persistent staging root must not be a symlink")
        if temporary.exists() and not temporary.is_dir():
            raise ValueError("persistent staging root must be a directory")
        temporary.mkdir(parents=True, exist_ok=True)
        lock_path, lock_identity = _persistent_lock(temporary)
    else:
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
        )
    try:
        journal: dict[str, object] | None = None
        journal_lock: threading.Lock | None = None
        if persistent:
            journal, journal_lock = _load_or_create_journal(
                temporary,
                _journal_identity(
                    inspection=inspection, mapping=mapping, source_repository=source_repository,
                    source_revision=source_revision, converter_commit=converter_commit,
                    converter_container_digest=converter_container_digest, split_seed=split_seed,
                    validation_fraction=validation_fraction, records=records, camera_keys=camera_keys,
                    info=info, source=source,
                ),
            )
            _validate_staging_tree(temporary, journal)
        adopted_episode_data = False
        if unbound_staging_data_adoption_root is not None:
            assert journal is not None and journal_lock is not None
            _claim_and_adopt_orphan_data(
                orphan=Path(unbound_staging_data_adoption_root), staging=temporary,
                info=info, records=records, tables=episode_tables, camera_keys=camera_keys, journal=journal,
                lock=journal_lock,
            )
            adopted_episode_data = True
        _write_episode_data(temporary, info, records, episode_tables, journal=journal, journal_lock=journal_lock)
        _write_episode_videos(source, temporary, info, records, camera_keys, journal=journal, journal_lock=journal_lock)
        (temporary / "meta").mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            temporary / "meta" / "info.json",
            _v2_info(
                info,
                episode_count=len(records),
                camera_count=len(camera_keys),
            ),
        )
        sorted_records = sorted(records, key=lambda item: int(item["episode_index"]))
        _write_json_lines(
            temporary / "meta" / "episodes.jsonl",
            (_v2_episode_metadata(record) for record in sorted_records),
        )
        _write_json_lines(
            temporary / "meta" / "episodes_stats.jsonl",
            (
                {"episode_index": int(record["episode_index"]), "stats": {}}
                for record in sorted_records
            ),
        )
        _write_json_lines(
            temporary / "meta" / "tasks.jsonl",
            [{"task_index": 0, "task": FIXED_INSTRUCTION}],
        )
        atomic_write_json(temporary / "meta" / "modality.json", _modality_metadata())
        atomic_write_json(temporary / "meta" / "lehome_mapping.json", mapping)
        valid_window_counts = {
            str(record["episode_index"]): max(
                0,
                int(record["dataset_to_index"])
                - int(record["dataset_from_index"])
                - ACTION_HORIZON
                + 1,
            )
            for record in sorted_records
        }
        # A crash after the prior attempt's manifest write leaves an allowed
        # stale manifest in persistent staging.  It must never enter the new
        # artifact set (nor make that set self-referential).
        output_artifacts = artifact_identities(temporary, exclude={"manifest.json"})
        manifest: dict[str, object] = {
            "schema_version": 1,
            "source_dataset": source.name,
            "source_format": "lerobot_v3_sharded",
            "output_format": "groot_lerobot_v2.1_per_episode",
            "source_repository": source_repository,
            "source_revision": source_revision,
            "source_manifest_sha256": inspection["source_manifest_sha256"],
            "source_artifacts": inspection["source_artifacts"],
            "output_artifacts": output_artifacts,
            "output_manifest_sha256": canonical_json_sha256(output_artifacts),
            "mapping_sha256": canonical_json_sha256(mapping),
            "converter_commit": converter_commit,
            "converter_container_digest": converter_container_digest,
            "pinned_groot_revision": ISAAC_GROOT_REVISION,
            "fps": inspection["fps"],
            "frame_count": inspection["frame_count"],
            "episode_count": inspection["episode_count"],
            "split_seed": split_seed,
            "validation_fraction": validation_fraction,
            "train_episode_ids": list(split.train),
            "validation_episode_ids": list(split.validation),
            "camera_schema": [
                {
                    **camera,
                    "dtype": "video",
                    "shape": [480, 640, 3],
                }
                for camera in mapping["cameras"]
            ],
            "state_schema": {
                "source_key": "observation.state",
                "dimension": 12,
                "names": list(JOINT_NAMES),
            },
            "action_schema": mapping["action"],
            "fixed_language_instruction": FIXED_INSTRUCTION,
            "statistics": {
                "status": "pending_task_4_train_only",
                "files": [],
            },
            "future_actions": {
                "horizon": ACTION_HORIZON,
                "loader_allow_padding": False,
                "materialized_windows": False,
                "tail_convention": "drop_incomplete_windows",
                "valid_action_mask": "implicit_all_true_for_emitted_windows",
                "valid_window_counts": valid_window_counts,
            },
        }
        if persistent:
            manifest["conversion_journal_sha256"] = _sha256_size(temporary / _JOURNAL_NAME)["sha256"]
            manifest["adopted_episode_data"] = adopted_episode_data
        atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        if lock_path is not None and lock_identity is not None:
            _release_persistent_lock(lock_path, lock_identity)
        return manifest
    except BaseException:
        if not persistent:
            shutil.rmtree(temporary, ignore_errors=True)
        if lock_path is not None and lock_identity is not None:
            _release_persistent_lock(lock_path, lock_identity)
        raise
