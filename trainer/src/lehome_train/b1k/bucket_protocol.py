"""Secret-free strict subprocess protocol for the isolated bucket helper."""

from __future__ import annotations

import json
import math
import os
from pathlib import PurePosixPath
import re
import subprocess
from typing import Callable, Mapping


_BUCKET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_WORKSPACE_CHECKPOINTS = "/workspace/checkpoints/"
_OPERATION_TIMEOUTS = {
    "info": 30, "ensure": 30, "list": 30,
    "copy": 300, "delete": 300,
    "upload": 21_600, "download": 21_600,
}


class ProtocolError(ValueError):
    pass


class BucketNotFound(ProtocolError):
    pass


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("bucket helper returned duplicate fields")
        result[key] = value
    return result


def _safe_path(value: object) -> bool:
    return type(value) is str and bool(value) and "*" not in value and "?" not in value and not PurePosixPath(value).is_absolute() and ".." not in PurePosixPath(value).parts


def _safe_local_path(value: object) -> bool:
    return type(value) is str and value.startswith(_WORKSPACE_CHECKPOINTS) and "\x00" not in value


class BucketHelperClient:
    def __init__(self, *, executable: str, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run, timeout_seconds: float | None = None) -> None:
        if not executable.startswith("/") or timeout_seconds is not None and (type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("bucket helper requires a fixed absolute executable and bounded timeout")
        self.executable, self.runner, self.timeout_seconds = executable, runner, timeout_seconds

    def request(self, operation: str, payload: Mapping[str, object]) -> dict[str, object]:
        payload = dict(payload)
        bucket_id = payload.get("bucket_id")
        if type(operation) is not str or not _BUCKET.fullmatch(str(bucket_id)):
            raise ProtocolError("bucket helper request is unsafe")
        allowed = {
            "info": {"bucket_id"},
            "ensure": {"bucket_id", "create"},
            "list": {"bucket_id", "prefix"},
            "upload": {"bucket_id", "local_path", "remote_path"},
            "download": {"bucket_id", "remote_path", "local_path"},
            "copy": {"bucket_id", "source", "destination"},
            "delete": {"bucket_id", "paths"},
        }.get(operation)
        if allowed is None or not set(payload) <= allowed or set(payload) == {"bucket_id"} and operation not in {"info", "ensure"}:
            raise ProtocolError("bucket helper request is invalid")
        required = {
            "info": {"bucket_id"}, "ensure": {"bucket_id"}, "list": {"bucket_id", "prefix"},
            "upload": {"bucket_id", "local_path", "remote_path"}, "download": {"bucket_id", "remote_path", "local_path"},
            "copy": {"bucket_id", "source", "destination"}, "delete": {"bucket_id", "paths"},
        }[operation]
        if not required <= set(payload) or ("create" in payload and type(payload["create"]) is not bool):
            raise ProtocolError("bucket helper request is invalid")
        for key in ("path", "prefix", "source", "destination", "remote_path"):
            if key in payload and payload[key] not in ("", None) and not _safe_path(payload[key]):
                raise ProtocolError("bucket helper request is unsafe")
        if "paths" in payload and (not isinstance(payload["paths"], list) or not all(_safe_path(value) for value in payload["paths"])):
            raise ProtocolError("bucket helper request is unsafe")
        if "local_path" in payload and not _safe_local_path(payload["local_path"]):
            raise ProtocolError("bucket helper request is unsafe")
        request = json.dumps({"version": 1, "operation": operation, "payload": payload}, sort_keys=True, separators=(",", ":")) + "\n"
        environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "B1K_HF_TOKEN_FILE") if key in os.environ}
        try:
            completed = self.runner((self.executable,), input=request, text=True, capture_output=True, env=environment, timeout=self.timeout_seconds if self.timeout_seconds is not None else _OPERATION_TIMEOUTS[operation], check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ProtocolError("bucket helper invocation failed") from error
        try:
            response = json.loads(completed.stdout, object_pairs_hook=_strict_pairs)
        except (json.JSONDecodeError, ProtocolError) as error:
            raise ProtocolError("bucket helper returned invalid response") from error
        if completed.returncode != 0:
            if isinstance(response, dict) and response == {"ok": False, "error": "not_found"}: raise BucketNotFound("bucket helper object was not found")
            raise ProtocolError("bucket helper operation failed")
        if not isinstance(response, dict) or set(response) != {"ok", "result"} or response["ok"] is not True or not isinstance(response["result"], dict):
            raise ProtocolError("bucket helper returned invalid response")
        result = response["result"]
        expected = {"info": {"private"}, "ensure": {"private"}, "list": {"files"}, "upload": set(), "download": set(), "copy": set(), "delete": set()}.get(operation)
        if expected is None or set(result) != expected:
            raise ProtocolError("bucket helper returned invalid response")
        if operation in {"info", "ensure"} and type(result["private"]) is not bool:
            raise ProtocolError("bucket helper returned invalid response")
        if operation == "list" and (not isinstance(result["files"], list) or not all(isinstance(item, dict) and set(item) == {"path", "size", "xet_hash", "type"} and _safe_path(item["path"]) and type(item["size"]) is int and item["size"] >= 0 and (item["xet_hash"] is None or type(item["xet_hash"]) is str) and type(item["type"]) is str for item in result["files"])):
            raise ProtocolError("bucket helper returned invalid response")
        return result
