from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _module():
    path = Path(__file__).parents[2] / "b1k_launchkit" / "training_smoke.py"
    spec = importlib.util.spec_from_file_location("b1k_training_smoke", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canary_allowlist_is_episode_zero_rgb_only_and_uses_pinned_sources() -> None:
    module = _module()

    assert module._DATASET_REPOSITORY == "behavior-1k/2026-challenge-demos"
    assert module._DATASET_REVISION == "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"
    assert module._MODEL_REPOSITORY == "nvidia/GR00T-N1.7-3B"
    assert module._MODEL_REVISION == "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
    assert module._EPISODE_FILE in module._DATASET_ALLOWLIST
    assert module._DATA_FILE in module._DATASET_ALLOWLIST
    assert all("depth" not in item for item in module._DATASET_ALLOWLIST)
    assert all("chunk-000/file-000" in item or item.startswith("meta/") for item in module._DATASET_ALLOWLIST)


def test_filter_episode_zero_rejects_missing_episode_and_preserves_only_zero(tmp_path: Path) -> None:
    module = _module()
    source, destination = tmp_path / "source.parquet", tmp_path / "destination.parquet"
    pq.write_table(pa.table({"episode_index": [0, 1, 0], "value": [3, 4, 5]}), source)

    module._filter_episode_zero(source, destination)

    assert pq.read_table(destination).column("episode_index").to_pylist() == [0, 0]
    pq.write_table(pa.table({"episode_index": [1], "value": [3]}), source)
    with pytest.raises(RuntimeError, match="episode 0"):
        module._filter_episode_zero(source, destination)


def test_validate_canary_rejects_missing_or_wrong_revision_artifacts(tmp_path: Path) -> None:
    module = _module()
    dataset, model = tmp_path / "dataset", tmp_path / "model"
    (dataset / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (dataset / "data" / "chunk-000").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text('{"features":{"observation.rgb.left_realsense_link_camera_0":{},"observation.rgb.right_realsense_link_camera_0":{},"observation.rgb.zed_link_camera_0":{}}}', encoding="utf-8")
    (dataset / "meta" / "modality.json").write_text("{}", encoding="utf-8")
    (dataset / "meta" / "stats.json").write_text("{}", encoding="utf-8")
    (dataset / "meta" / "tasks.parquet").write_bytes(b"tasks")
    pq.write_table(pa.table({"episode_index": [0]}), dataset / module._EPISODE_FILE)
    pq.write_table(pa.table({"episode_index": [0]}), dataset / module._DATA_FILE)
    for path in module._RGB_VIDEO_FILES:
        target = dataset / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b"video")
    model.mkdir()
    for name in module._MODEL_REQUIRED:
        (model / name).write_bytes(b"weights")

    module._validate_canary(dataset, model)
    (model / module._MODEL_REQUIRED[-1]).unlink()
    with pytest.raises(RuntimeError, match="model"):
        module._validate_canary(dataset, model)


def test_training_subprocess_environment_keeps_token_file_and_adds_upstream_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")
    environment = module._training_environment(Path("/workspace/.cache/huggingface/token"))

    assert environment["B1K_HF_TOKEN_FILE"] == "/workspace/.cache/huggingface/token"
    assert environment["PYTHONPATH"] == "/opt/isaac-groot"
    assert "HF_TOKEN" not in environment


@pytest.mark.parametrize("failure", ["upload", "download", "mismatch", "delete"])
def test_checkpoint_bucket_probe_reconciles_the_exact_remote_object_after_every_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    module = _module()
    calls: list[str] = []
    remote_objects: set[str] = set()

    class TemporaryDirectory:
        def __init__(self, **_kwargs: object) -> None:
            self.path = tmp_path / "probe"

        def __enter__(self) -> str:
            self.path.mkdir()
            return str(self.path)

        def __exit__(self, *_args: object) -> None:
            return None

    def run(_command: tuple[str, ...], *, input: str, **_kwargs: object) -> object:
        request = json.loads(input)
        operation = request["operation"]
        payload = request["payload"]
        calls.append(operation)
        remote = payload.get("remote_path")
        if operation == "info":
            result: object = {"private": True}
        elif operation == "upload":
            assert isinstance(remote, str)
            remote_objects.add(remote)
            if failure == "upload":
                return type("Completed", (), {"returncode": 1, "stdout": '{"ok":false,"result":{}}'})()
            result = {}
        elif operation == "download":
            if failure == "download":
                return type("Completed", (), {"returncode": 1, "stdout": '{"ok":false,"result":{}}'})()
            Path(payload["local_path"]).write_bytes(b"wrong" if failure == "mismatch" else module._PROBE_BYTES)
            result = {}
        elif operation == "delete":
            assert isinstance(payload["paths"], list)
            remote_objects.discard(payload["paths"][0])
            if failure == "delete":
                return type("Completed", (), {"returncode": 1, "stdout": '{"ok":false,"result":{}}'})()
            result = {}
        elif operation == "list":
            result = {"files": [] if not remote_objects else [{"path": path} for path in remote_objects]}
        else:
            raise AssertionError(operation)
        return type("Completed", (), {"returncode": 0, "stdout": json.dumps({"ok": True, "result": result})})()

    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", TemporaryDirectory)
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.os, "urandom", lambda _size: b"x" * 8)

    expected = {
        "upload": "checkpoint bucket upload helper operation failed",
        "download": "checkpoint bucket download helper operation failed",
        "mismatch": "checkpoint bucket readback did not match",
        "delete": "checkpoint bucket delete helper operation failed",
    }[failure]
    with pytest.raises(RuntimeError, match=expected):
        module._checkpoint_bucket_probe(tmp_path / "token")

    assert calls[0] == "info"
    assert "delete" in calls
    assert calls[-1] == "list"
    assert remote_objects == set()


def test_checkpoint_bucket_probe_preserves_upload_failure_when_cleanup_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    calls: list[str] = []

    class TemporaryDirectory:
        def __init__(self, **_kwargs: object) -> None:
            self.path = tmp_path / "probe"

        def __enter__(self) -> str:
            self.path.mkdir()
            return str(self.path)

        def __exit__(self, *_args: object) -> None:
            return None

    def run(_command: tuple[str, ...], *, input: str, **_kwargs: object) -> object:
        operation = json.loads(input)["operation"]
        calls.append(operation)
        result = {"private": True} if operation == "info" else {"files": []} if operation == "list" else {}
        returncode = 1 if operation in {"upload", "delete"} else 0
        return type("Completed", (), {"returncode": returncode, "stdout": json.dumps({"ok": returncode == 0, "result": result})})()

    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", TemporaryDirectory)
    monkeypatch.setattr(module.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="checkpoint bucket upload helper operation failed") as error:
        module._checkpoint_bucket_probe(tmp_path / "token")

    assert calls == ["info", "upload", "delete", "list"]
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "checkpoint bucket delete helper operation failed"
