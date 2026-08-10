#!/usr/bin/env python3
"""Self-staging, bounded real Isaac-GR00T B1K episode-zero canary.

It snapshots only the documented episode-zero metadata/data/RGB artifacts and
the immutable GR00T base model, then executes pinned ``train_b1k.py`` through
``torchrun`` for exactly one optimizer step.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import hashlib
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any


_PROBE_BYTES = b'{"purpose":"b1k-private-release-bootstrap"}\n'
_UPSTREAM = Path("/opt/isaac-groot/scripts/b1k/train_b1k.py")
_MODALITY = Path("/opt/isaac-groot/examples/b1k/r1pro.py")
_MODALITY_JSON = Path("/opt/isaac-groot/examples/b1k/r1pro.json")
_DATASET_REPOSITORY = "behavior-1k/2026-challenge-demos"
_DATASET_REVISION = "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"
_MODEL_REPOSITORY = "nvidia/GR00T-N1.7-3B"
_MODEL_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
_CANARY_ROOT = Path("/workspace/smoke-canary")
_EPISODE_FILE = "meta/episodes/chunk-000/file-000.parquet"
_DATA_FILE = "data/chunk-000/file-000.parquet"
_RGB_VIDEO_FILES = (
    "videos/observation.rgb.left_realsense_link_camera_0/chunk-000/file-000.mp4",
    "videos/observation.rgb.right_realsense_link_camera_0/chunk-000/file-000.mp4",
    "videos/observation.rgb.zed_link_camera_0/chunk-000/file-000.mp4",
)
_DATASET_ALLOWLIST = (
    "meta/info.json", "meta/stats.json", "meta/tasks.parquet", "meta/tasks.jsonl",
    "meta/episodes.jsonl", _EPISODE_FILE, _DATA_FILE, *_RGB_VIDEO_FILES,
)
_RGB_KEYS = tuple(path.split("/")[1] for path in _RGB_VIDEO_FILES)
_MODEL_REQUIRED = (
    "config.json", "model.safetensors.index.json",
    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args(argv)
    if not args.prefix.startswith("b1k-bootstrap-") or not _UPSTREAM.is_file() or not _MODALITY.is_file():
        raise RuntimeError("pinned B1K training canary is unavailable")
    token_file = Path(os.environ["B1K_HF_TOKEN_FILE"])
    metadata = token_file.stat()
    if token_file.is_symlink() or stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_uid != os.getuid():
        raise RuntimeError("runtime token file is invalid")
    dataset, model = _stage_canary(token_file)
    output = Path("/workspace/outputs/b1k-smoke-canary")
    command = (
        "torchrun", "--nproc_per_node=1", "--master_port=29501", str(_UPSTREAM),
        "--base-model-path", str(model), "--dataset-path", str(dataset), "--output-dir", str(output),
        "--experiment-name", "b1k-smoke-canary", "--embodiment-tag", "NEW_EMBODIMENT",
        "--modality-config-path", str(_MODALITY), "--num-gpus", "1", "--global-batch-size", "1",
        "--gradient-accumulation-steps", "1", "--max-steps", "1", "--save-steps", "1",
        "--save-total-limit", "1", "--learning-rate", "0.0001", "--weight-decay", "0.00001",
        "--warmup-ratio", "0", "--decode-only-used-frames",
    )
    environment = _training_environment(token_file)
    subprocess.run(command, check=True, env=environment, timeout=1500)
    _checkpoint_bucket_probe(token_file)
    from huggingface_hub import HfApi

    token = token_file.read_text(encoding="utf-8").strip()
    upload = HfApi(token=token).upload_file(path_or_fileobj=io.BytesIO(_PROBE_BYTES), path_in_repo=f"{args.prefix}/probe.json", repo_id="ryanjin333/behavior1k-groot-n17-models", repo_type="model", token=token)
    commit = getattr(upload, "oid", getattr(upload, "commit_id", None))
    if not isinstance(commit, str):
        raise RuntimeError("model smoke probe upload has no immutable commit")
    print(json.dumps({"runtime_uid": os.getuid(), "token_file_uid": metadata.st_uid, "token_file_mode": stat.S_IMODE(metadata.st_mode), "gpu_count": _gpu_count(), "optimizer_steps": 1, "lifecycle_preflight": "passed", "checkpoint_bucket_probe": "passed", "remote_probe_upload_commit": commit, "container_digest": os.environ.get("CONTAINER_DIGEST", "")}, sort_keys=True, separators=(",", ":")))
    return 0


def _gpu_count() -> int:
    import torch
    return int(torch.cuda.device_count())


def _stage_canary(token_file: Path) -> tuple[Path, Path]:
    """Materialize only episode zero plus the exact pinned GR00T snapshot."""
    from huggingface_hub import HfApi, snapshot_download

    token = token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("runtime token file is empty")
    root = _CANARY_ROOT
    if root.is_symlink():
        raise RuntimeError("smoke canary root is unsafe")
    root.mkdir(parents=True, exist_ok=True)
    dataset, model = root / "dataset", root / "model"
    api = HfApi(token=token)
    _require_exact_revision(api.repo_info(_DATASET_REPOSITORY, repo_type="dataset", revision=_DATASET_REVISION, token=token), _DATASET_REVISION, "dataset")
    _require_exact_revision(api.repo_info(_MODEL_REPOSITORY, repo_type="model", revision=_MODEL_REVISION, token=token), _MODEL_REVISION, "model")
    _safe_reset(dataset); _safe_reset(model)
    snapshot_download(repo_id=_DATASET_REPOSITORY, repo_type="dataset", revision=_DATASET_REVISION, token=token, allow_patterns=list(_DATASET_ALLOWLIST), local_dir=dataset, local_dir_use_symlinks=False)
    _filter_episode_zero(dataset / _EPISODE_FILE, dataset / _EPISODE_FILE)
    _filter_episode_zero(dataset / _DATA_FILE, dataset / _DATA_FILE)
    _install_rgb_only_info(dataset)
    _MODALITY_JSON.exists() or (_raise("pinned R1Pro modality JSON is unavailable"))
    shutil.copyfile(_MODALITY_JSON, dataset / "meta" / "modality.json")
    snapshot_download(repo_id=_MODEL_REPOSITORY, repo_type="model", revision=_MODEL_REVISION, token=token, local_dir=model, local_dir_use_symlinks=False)
    _validate_canary(dataset, model)
    _write_stage_receipt(root, dataset, model)
    return dataset, model


def _filter_episode_zero(source: Path, destination: Path) -> None:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if source.is_symlink() or not source.is_file() or destination.is_symlink():
        raise RuntimeError("canary parquet path is unsafe")
    table = pq.read_table(source)
    if "episode_index" not in table.column_names:
        raise RuntimeError("canary parquet lacks episode_index")
    filtered = table.filter(pc.equal(table["episode_index"], 0))
    if filtered.num_rows < 1:
        raise RuntimeError("canary parquet does not contain episode 0")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".smoke.tmp")
    pq.write_table(filtered, temporary)
    temporary.replace(destination)


def _install_rgb_only_info(dataset: Path) -> None:
    info = dataset / "meta" / "info.json"
    try:
        value = json.loads(info.read_text(encoding="utf-8"))
        features = value["features"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("canary info metadata is invalid") from error
    if not isinstance(features, dict) or not all(key in features for key in _RGB_KEYS):
        raise RuntimeError("canary metadata is missing required RGB features")
    value["features"] = {key: feature for key, feature in features.items() if not key.startswith("observation.depth")}
    info.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _validate_canary(dataset: Path, model: Path) -> None:
    required = ("meta/info.json", "meta/stats.json", "meta/tasks.parquet", "meta/modality.json", _EPISODE_FILE, _DATA_FILE, *_RGB_VIDEO_FILES)
    if dataset.is_symlink() or model.is_symlink() or not dataset.is_dir() or not model.is_dir() or any((dataset / item).is_symlink() or not (dataset / item).is_file() or (dataset / item).stat().st_size <= 0 for item in required):
        raise RuntimeError("canary dataset artifacts are incomplete")
    try:
        features = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))["features"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("canary info metadata is invalid") from error
    if set(key for key in features if key.startswith("observation.rgb")) != set(_RGB_KEYS) or any("depth" in key for key in features):
        raise RuntimeError("canary must retain exactly the RGB input features")
    if any((model / item).is_symlink() or not (model / item).is_file() or (model / item).stat().st_size <= 0 for item in _MODEL_REQUIRED):
        raise RuntimeError("canary model snapshot is incomplete")


def _training_environment(token_file: Path) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key in {"PATH", "HOME", "B1K_HF_TOKEN_FILE", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "WANDB_MODE", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"}}
    environment["B1K_HF_TOKEN_FILE"] = str(token_file)
    environment["PYTHONPATH"] = "/opt/isaac-groot"
    environment.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    environment.setdefault("HF_HUB_CACHE", "/workspace/.cache/huggingface/hub")
    environment["WANDB_MODE"] = "offline"
    environment["WANDB_DIR"] = "/workspace/logs/wandb"
    return environment


def _require_exact_revision(info: Any, revision: str, label: str) -> None:
    if getattr(info, "sha", None) != revision and not (isinstance(info, dict) and info.get("sha") == revision):
        raise RuntimeError(f"pinned {label} revision did not resolve exactly")


def _safe_reset(path: Path) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError("canary staging path is unsafe")
        shutil.rmtree(path)
    path.mkdir(mode=0o700)


def _write_stage_receipt(root: Path, dataset: Path, model: Path) -> None:
    files = []
    for source, label, revision in ((dataset, _DATASET_REPOSITORY, _DATASET_REVISION), (model, _MODEL_REPOSITORY, _MODEL_REVISION)):
        for path in sorted(source.rglob("*")):
            if path.is_file() and not path.is_symlink():
                files.append({"source": label, "revision": revision, "path": path.relative_to(source).as_posix(), "sha256": _sha256(path)})
    if not files:
        raise RuntimeError("canary snapshot inventory is empty")
    temporary = root / ".canary-receipt.tmp"
    temporary.write_text(json.dumps({"dataset_revision": _DATASET_REVISION, "model_revision": _MODEL_REVISION, "files": files}, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.chmod(temporary, 0o600); temporary.replace(root / "canary-receipt.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _raise(message: str) -> None:
    raise RuntimeError(message)


def _checkpoint_bucket_probe(token_file: Path) -> None:
    """Exercise the exact private rolling-checkpoint bucket through its helper.

    This is deliberately reversible: stage, upload, byte-readback, delete, and
    prove absence for one namespaced object before returning runtime evidence.
    """
    helper = "/opt/b1k-bucket-helper/bin/b1k-bucket-helper"
    bucket = "ryanjin333/behavior1k-groot-n17-checkpoints"
    prefix = f"smoke/{os.getpid()}-{os.urandom(8).hex()}"
    remote = f"{prefix}/checkpoint.bin"
    with tempfile.TemporaryDirectory(dir="/workspace/checkpoints", prefix="b1k-smoke-") as directory:
        source, readback = Path(directory) / "source", Path(directory) / "readback"
        source.write_bytes(_PROBE_BYTES)
        def call(operation: str, payload: dict[str, object]) -> dict[str, object]:
            completed = subprocess.run((helper,), input=json.dumps({"version": 1, "operation": operation, "payload": {"bucket_id": bucket, **payload}}) + "\n", text=True, capture_output=True, check=False, env={"PATH": os.environ.get("PATH", ""), "B1K_HF_TOKEN_FILE": str(token_file)})
            try:
                response = json.loads(completed.stdout)
            except ValueError as error:
                raise RuntimeError(f"checkpoint bucket {operation} helper returned invalid evidence") from error
            if completed.returncode != 0 or response.get("ok") is not True:
                raise RuntimeError(f"checkpoint bucket {operation} helper operation failed")
            return response["result"]
        if call("info", {}).get("private") is not True:
            raise RuntimeError("checkpoint bucket is not private")
        primary_failure: Exception | None = None
        try:
            call("upload", {"local_path": str(source), "remote_path": remote})
            call("download", {"remote_path": remote, "local_path": str(readback)})
            if readback.read_bytes() != _PROBE_BYTES:
                raise RuntimeError("checkpoint bucket readback did not match")
        except Exception as error:
            primary_failure = error

        cleanup_failure: Exception | None = None
        try:
            call("delete", {"paths": [remote]})
        except Exception as error:
            cleanup_failure = error
        try:
            files = call("list", {"prefix": prefix + "/"}).get("files")
            if files != []:
                raise RuntimeError("checkpoint bucket smoke cleanup did not prove absence")
        except Exception as error:
            cleanup_failure = cleanup_failure or error

        if primary_failure is not None:
            if cleanup_failure is not None:
                raise primary_failure from cleanup_failure
            raise primary_failure
        if cleanup_failure is not None:
            raise cleanup_failure


if __name__ == "__main__":
    raise SystemExit(main())
