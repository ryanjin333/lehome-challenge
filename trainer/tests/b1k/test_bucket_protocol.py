from __future__ import annotations

import json
import subprocess

import pytest

from lehome_train.b1k.bucket_protocol import BucketHelperClient, BucketNotFound, ProtocolError


def test_client_uses_one_redacted_json_request_and_strict_response() -> None:
    seen: dict[str, object] = {}

    def runner(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["command"] = command
        seen["input"] = kwargs["input"]
        seen["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout='{"ok":true,"result":{"private":true}}\n', stderr="")

    client = BucketHelperClient(executable="/opt/b1k-bucket-helper/bin/b1k-bucket-helper", runner=runner)
    assert client.request("info", {"bucket_id": "owner/checkpoints"}) == {"private": True}
    assert seen["command"] == ("/opt/b1k-bucket-helper/bin/b1k-bucket-helper",)
    assert "HF_TOKEN" not in str(seen["input"])
    assert "HF_TOKEN" not in seen["env"]


@pytest.mark.parametrize("stdout", ["{}\n", '{"ok":true,"result":{},"extra":1}\n', '{"ok":true,"ok":false}\n'])
def test_client_rejects_malformed_or_duplicate_protocol_responses(stdout: str) -> None:
    client = BucketHelperClient(
        executable="/fixed/helper",
        runner=lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="token=secret"),
    )
    with pytest.raises(ProtocolError, match="bucket helper"):
        client.request("info", {"bucket_id": "owner/checkpoints"})


def test_client_rejects_unsafe_bucket_and_paths_before_subprocess() -> None:
    client = BucketHelperClient(executable="/fixed/helper", runner=lambda *_args, **_kwargs: pytest.fail("must not run"))
    with pytest.raises(ProtocolError):
        client.request("list", {"bucket_id": "../bad", "prefix": ""})
    with pytest.raises(ProtocolError):
        client.request("download", {"bucket_id": "owner/checkpoints", "paths": ["../bad"]})


@pytest.mark.parametrize(
    ("operation", "result"),
    [("ensure", {"private": True}), ("upload", {}), ("download", {}), ("copy", {}), ("delete", {})],
)
def test_client_accepts_only_exact_known_operation_schemas(operation: str, result: dict[str, object]) -> None:
    client = BucketHelperClient(executable="/fixed/helper", runner=lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout=json.dumps({"ok": True, "result": result}), stderr=""))
    payload = {"bucket_id": "owner/checkpoints"}
    if operation in {"upload", "download"}: payload.update({"local_path": "/workspace/checkpoints/a", "remote_path": "verified/a"})
    if operation == "copy": payload.update({"source": "verified/a", "destination": "verified/b"})
    if operation == "delete": payload["paths"] = ["verified/a"]
    assert client.request(operation, payload) == result


def test_client_rejects_wildcard_delete() -> None:
    client = BucketHelperClient(executable="/fixed/helper", runner=lambda *_args, **_kwargs: pytest.fail("must not run"))
    with pytest.raises(ProtocolError): client.request("delete", {"bucket_id": "owner/checkpoints", "paths": ["verified/*"]})


@pytest.mark.parametrize(
    "operation,payload",
    [
        ("unknown", {"bucket_id": "owner/checkpoints"}),
        ("list", {"bucket_id": "owner/checkpoints"}),
        ("upload", {"bucket_id": "owner/checkpoints", "local_path": "/tmp/file", "remote_path": "a"}),
        ("copy", {"bucket_id": "owner/checkpoints", "source": "a", "destination": "b", "extra": True}),
        ("ensure", {"bucket_id": "owner/checkpoints", "create": "yes"}),
    ],
)
def test_client_rejects_missing_extra_or_unsafe_operation_fields(operation: str, payload: dict[str, object]) -> None:
    client = BucketHelperClient(executable="/fixed/helper", runner=lambda *_args, **_kwargs: pytest.fail("must not run"))
    with pytest.raises(ProtocolError): client.request(operation, payload)


@pytest.mark.parametrize(
    "operation,payload,timeout",
    [
        ("info", {"bucket_id": "owner/checkpoints"}, 30),
        ("list", {"bucket_id": "owner/checkpoints", "prefix": "verified"}, 30),
        ("ensure", {"bucket_id": "owner/checkpoints"}, 30),
        ("copy", {"bucket_id": "owner/checkpoints", "source": "a", "destination": "b"}, 300),
        ("delete", {"bucket_id": "owner/checkpoints", "paths": ["a"]}, 300),
        ("upload", {"bucket_id": "owner/checkpoints", "local_path": "/workspace/checkpoints/a", "remote_path": "a"}, 21600),
        ("download", {"bucket_id": "owner/checkpoints", "local_path": "/workspace/checkpoints/a", "remote_path": "a"}, 21600),
    ],
)
def test_client_uses_bounded_operation_specific_timeouts(operation: str, payload: dict[str, object], timeout: int) -> None:
    observed: list[object] = []
    result = {"private": True} if operation in {"info", "ensure"} else {"files": []} if operation == "list" else {}
    def runner(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.append(kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"ok": True, "result": result}), stderr="")
    BucketHelperClient(executable="/fixed/helper", runner=runner).request(operation, payload)
    assert observed == [timeout]


def test_client_rejects_non_finite_or_non_positive_timeout_overrides() -> None:
    with pytest.raises(ValueError): BucketHelperClient(executable="/fixed/helper", timeout_seconds=0)
    with pytest.raises(ValueError): BucketHelperClient(executable="/fixed/helper", timeout_seconds=float("inf"))


def test_client_maps_only_strict_not_found_error_to_typed_exception() -> None:
    client = BucketHelperClient(executable="/fixed/helper", runner=lambda command, **_kwargs: subprocess.CompletedProcess(command, 1, stdout='{"ok":false,"error":"not_found"}', stderr=""))
    with pytest.raises(BucketNotFound): client.request("info", {"bucket_id": "owner/checkpoints"})
    client = BucketHelperClient(executable="/fixed/helper", runner=lambda command, **_kwargs: subprocess.CompletedProcess(command, 1, stdout='{"ok":false,"error":"operation_failed"}', stderr=""))
    with pytest.raises(ProtocolError): client.request("info", {"bucket_id": "owner/checkpoints"})
