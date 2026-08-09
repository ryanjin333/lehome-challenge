from __future__ import annotations

import importlib.machinery
import importlib.util
from io import StringIO
import json
import os
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


def test_helper_accepts_the_orchestrator_explicit_private_token_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    fake = FakeHub()
    module.hub = fake
    token_file = tmp_path / "hf.token"
    token_file.write_text("local-private-token", encoding="utf-8")
    token_file.chmod(0o600)
    monkeypatch.setenv("B1K_HF_TOKEN_FILE", str(token_file))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        sys,
        "stdin",
        StringIO(json.dumps({"version": 1, "operation": "info", "payload": {"bucket_id": "owner/checkpoints"}})),
    )
    output = StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    module.main()

    assert json.loads(output.getvalue()) == {"ok": True, "result": {"private": True}}
    assert fake.calls[0][2]["token"] == "local-private-token"


@pytest.mark.parametrize(
    "unsafe_kind",
    ["relative", "permissive", "symlink", "ancestor_symlink", "nonregular", "oversized"],
)
def test_helper_rejects_unsafe_explicit_token_files_before_hub_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe_kind: str
) -> None:
    module = _module()
    fake = FakeHub()
    module.hub = fake
    private_file = tmp_path / "private.token"
    private_file.write_text("private-token", encoding="utf-8")
    private_file.chmod(0o600)
    token_file = private_file
    if unsafe_kind == "relative":
        token_value = "relative.token"
    elif unsafe_kind == "permissive":
        private_file.chmod(0o640)
        token_value = str(private_file)
    elif unsafe_kind == "symlink":
        token_file = tmp_path / "linked.token"
        token_file.symlink_to(private_file)
        token_value = str(token_file)
    elif unsafe_kind == "ancestor_symlink":
        real_directory = tmp_path / "real"
        real_directory.mkdir()
        nested_file = real_directory / "nested.token"
        nested_file.write_text("private-token", encoding="utf-8")
        nested_file.chmod(0o600)
        linked_directory = tmp_path / "linked-directory"
        linked_directory.symlink_to(real_directory, target_is_directory=True)
        token_value = str(linked_directory / nested_file.name)
    elif unsafe_kind == "nonregular":
        token_file = tmp_path / "token.fifo"
        os.mkfifo(token_file, mode=0o600)
        token_value = str(token_file)
    else:
        private_file.write_bytes(b"x" * (module.MAX_TOKEN_BYTES + 1))
        token_value = str(private_file)
    monkeypatch.setenv("B1K_HF_TOKEN_FILE", token_value)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        sys,
        "stdin",
        StringIO(json.dumps({"version": 1, "operation": "info", "payload": {"bucket_id": "owner/checkpoints"}})),
    )
    output = StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    with pytest.raises(SystemExit):
        module.main()

    assert json.loads(output.getvalue()) == {"ok": False, "error": "operation_failed"}
    assert fake.calls == []


def test_private_token_reader_rejects_oversized_file_across_short_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    token_file = tmp_path / "oversized.token"
    token_file.write_bytes(b"x" * (module.MAX_TOKEN_BYTES + 1))
    token_file.chmod(0o600)
    real_read = os.read
    monkeypatch.setattr(module.os, "read", lambda descriptor, count: real_read(descriptor, min(count, 17)))

    with pytest.raises(ValueError, match="invalid token file"):
        module.read_private_token_file(str(token_file))
