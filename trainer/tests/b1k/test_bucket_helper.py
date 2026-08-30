from __future__ import annotations

import importlib.machinery
import importlib.util
from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


HELPER = Path(__file__).parents[2] / "b1k_launchkit" / "bucket_helper" / "b1k-bucket-helper"


def test_helper_source_is_executable_and_has_the_pinned_install_contract() -> None:
    project = HELPER.parent / "pyproject.toml"

    assert HELPER.is_file()
    assert HELPER.stat().st_mode & 0o111
    assert 'huggingface-hub==1.24.0' in project.read_text(encoding="utf-8")
    assert "hatchling==1.27.0" in project.read_text(encoding="utf-8")


class FakeHub:
    __version__ = "1.24.0"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _call(self, name: str, *args: object, **kwargs: object) -> None:
        self.calls.append((name, args, kwargs))

    def create_bucket(self, *args: object, **kwargs: object) -> None: self._call("create_bucket", *args, **kwargs)
    def bucket_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
        self._call("bucket_info", *args, **kwargs); return SimpleNamespace(private=True)
    def list_bucket_tree(self, *args: object, **kwargs: object) -> list[SimpleNamespace]:
        self._call("list_bucket_tree", *args, **kwargs)
        return [SimpleNamespace(path="verified/a", size=3, xet_hash=None, type="file")]
    def batch_bucket_files(self, *args: object, **kwargs: object) -> None: self._call("batch_bucket_files", *args, **kwargs)
    def download_bucket_files(self, *args: object, **kwargs: object) -> None: self._call("download_bucket_files", *args, **kwargs)
    def copy_files(self, *args: object, **kwargs: object) -> None: self._call("copy_files", *args, **kwargs)


def _module() -> object:
    loader = importlib.machinery.SourceFileLoader("b1k_bucket_helper_test", str(HELPER))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "operation,payload,expected_call",
    [
        ("ensure", {"bucket_id": "owner/checkpoints", "create": True}, "bucket_info"),
        ("info", {"bucket_id": "owner/checkpoints"}, "bucket_info"),
        ("list", {"bucket_id": "owner/checkpoints", "prefix": "verified"}, "list_bucket_tree"),
        ("upload", {"bucket_id": "owner/checkpoints", "local_path": "/workspace/checkpoints/source", "remote_path": "verified/a"}, "batch_bucket_files"),
        ("download", {"bucket_id": "owner/checkpoints", "remote_path": "verified/a", "local_path": "/workspace/checkpoints/download"}, "download_bucket_files"),
        ("copy", {"bucket_id": "owner/checkpoints", "source": "verified/a", "destination": "verified/b"}, "copy_files"),
        ("delete", {"bucket_id": "owner/checkpoints", "paths": ["verified/a"]}, "batch_bucket_files"),
    ],
)
def test_helper_invokes_only_the_expected_mocked_hub_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, payload: dict[str, object], expected_call: str
) -> None:
    module = _module()
    fake = FakeHub()
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    (checkpoint_root / "source").write_bytes(b"source")
    real_path = Path
    module.Path = lambda value: checkpoint_root / str(value).removeprefix("/workspace/checkpoints").lstrip("/") if str(value).startswith("/workspace/checkpoints") else real_path(value)
    module.hub = fake
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.delenv("B1K_HF_TOKEN_FILE", raising=False)
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps({"version": 1, "operation": operation, "payload": payload})))
    output = StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    module.main()
    response = json.loads(output.getvalue())
    assert response["ok"] is True
    assert any(call[0] == expected_call for call in fake.calls)
    assert all(call[2].get("token") == "test-token" for call in fake.calls)


@pytest.mark.parametrize(
    "operation,payload",
    [
        ("list", {"bucket_id": "owner/checkpoints", "prefix": "../escape"}),
        ("delete", {"bucket_id": "owner/checkpoints", "paths": ["verified/*"]}),
        ("copy", {"bucket_id": "owner/checkpoints", "source": "a", "destination": "b", "unexpected": True}),
    ],
)
def test_helper_rejects_bad_payload_before_hub_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, payload: dict[str, object]) -> None:
    module = _module(); fake = FakeHub(); module.hub = fake
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.delenv("B1K_HF_TOKEN_FILE", raising=False)
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps({"version": 1, "operation": operation, "payload": payload})))
    output = StringIO(); monkeypatch.setattr(sys, "stdout", output)
    with pytest.raises(SystemExit): module.main()
    assert json.loads(output.getvalue()) == {"ok": False, "error": "operation_failed"}
    assert fake.calls == []


def test_helper_list_filters_bucket_folders_before_reading_file_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); fake = FakeHub()
    fake.list_bucket_tree = lambda *_args, **_kwargs: [SimpleNamespace(type="folder", path="verified"), SimpleNamespace(type="file", path="verified/a", size=3, xet_hash=None)]
    module.hub = fake
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.delenv("B1K_HF_TOKEN_FILE", raising=False)
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps({"version": 1, "operation": "list", "payload": {"bucket_id": "owner/checkpoints", "prefix": "verified"}})))
    output = StringIO(); monkeypatch.setattr(sys, "stdout", output)
    module.main()
    assert json.loads(output.getvalue())["result"]["files"] == [{"path": "verified/a", "size": 3, "xet_hash": None, "type": "file"}]
