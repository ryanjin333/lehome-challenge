#!/usr/bin/env python3
"""HTTP adapter from the official LeHome DockerPolicy to the N1.7 ZMQ server."""

from __future__ import annotations

import argparse
import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from typing import Any, Callable, Mapping

import numpy as np

_REQUIRED_KEYS = {
    "observation.state",
    "observation.images.top_rgb",
    "observation.images.left_rgb",
    "observation.images.right_rgb",
}
_OPTIONAL_KEYS = {"action", "observation.top_depth"}
_MAX_BODY_BYTES = 64 * 1024 * 1024


class BridgeProtocolError(ValueError):
    """The official HTTP or internal action contract was violated."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BridgeProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_envelope(value: object, *, key: str) -> np.ndarray:
    if type(value) is not dict or set(value) != {"base64", "shape", "dtype"}:
        raise BridgeProtocolError(f"{key} ndarray envelope is malformed")
    encoded, shape, dtype_name = value["base64"], value["shape"], value["dtype"]
    if type(encoded) is not str or type(shape) is not list or type(dtype_name) is not str:
        raise BridgeProtocolError(f"{key} ndarray envelope types are invalid")
    if not shape or len(shape) > 4 or any(type(item) is not int or item <= 0 for item in shape):
        raise BridgeProtocolError(f"{key} ndarray shape is invalid")
    try:
        dtype = np.dtype(dtype_name)
    except TypeError:
        raise BridgeProtocolError(f"{key} ndarray dtype is invalid") from None
    if dtype.hasobject or dtype.kind not in "buif":
        raise BridgeProtocolError(f"{key} ndarray dtype is unsafe")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        raise BridgeProtocolError(f"{key} ndarray base64 is invalid") from None
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if expected != len(raw):
        raise BridgeProtocolError(f"{key} ndarray byte length does not match its shape")
    return np.frombuffer(raw, dtype=dtype).reshape(tuple(shape)).copy()


def decode_official_observation(payload: object) -> dict[str, np.ndarray]:
    """Decode only the ndarray fields emitted by the official DockerPolicy."""

    if type(payload) is not dict:
        raise BridgeProtocolError("inference payload must be a JSON object")
    keys = set(payload)
    if not _REQUIRED_KEYS.issubset(keys) or not keys.issubset(_REQUIRED_KEYS | _OPTIONAL_KEYS):
        raise BridgeProtocolError("inference payload contains missing or forbidden fields")
    decoded: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key in {"action", "observation.state"}:
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (12,) or not np.isfinite(array).all():
                raise BridgeProtocolError(f"{key} must be finite 12-D data")
            decoded[key] = array
        else:
            decoded[key] = _decode_envelope(value, key=key)
    for key in _REQUIRED_KEYS - {"observation.state"}:
        frame = decoded[key]
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise BridgeProtocolError(f"{key} must be an HWC uint8 RGB frame")
    return decoded


class OfficialDockerPolicyBridge:
    """Translate official JSON requests without exposing garment metadata."""

    def __init__(
        self,
        client: Any,
        *,
        observation_builder: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        action_validator: Callable[[Mapping[str, Any]], np.ndarray] | None = None,
    ) -> None:
        self._client = client
        if observation_builder is None or action_validator is None:
            from scripts.eval_policy.groot_policy import (
                build_groot_observation,
                validate_policy_server_action_chunk,
            )

            observation_builder = observation_builder or build_groot_observation
            action_validator = action_validator or validate_policy_server_action_chunk
        self._observation_builder = observation_builder
        self._action_validator = action_validator

    @staticmethod
    def decode_json(raw: bytes) -> dict[str, Any]:
        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_strict_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    BridgeProtocolError(f"non-finite JSON token: {token}")
                ),
            )
        except (UnicodeError, json.JSONDecodeError):
            raise BridgeProtocolError("request body is not strict JSON") from None
        if type(value) is not dict:
            raise BridgeProtocolError("request body must be a JSON object")
        return value

    def reset(self, payload: object) -> dict[str, str]:
        if payload != {}:
            raise BridgeProtocolError("reset body must be an empty object")
        self._client.reset()
        return {"status": "ok"}

    def infer(self, payload: object) -> dict[str, list[list[float]]]:
        decoded = decode_official_observation(payload)
        policy_observation = {key: decoded[key] for key in _REQUIRED_KEYS}
        observation = self._observation_builder(policy_observation)
        try:
            action, _info = self._client.get_action(observation)
            chunk = self._action_validator(action)
        except Exception as error:
            raise BridgeProtocolError("policy response is not a finite 16x12 action chunk") from error
        if chunk.shape != (16, 12) or chunk.dtype != np.float32 or not np.isfinite(chunk).all():
            raise BridgeProtocolError("policy response is not a finite 16x12 action chunk")
        return {"actions": chunk.tolist()}


class _Handler(BaseHTTPRequestHandler):
    server_version = "LeHomeOfficialBridge/1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _reply(self, status: HTTPStatus, value: Mapping[str, object]) -> None:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        length_text = self.headers.get("Content-Length", "")
        try:
            length = int(length_text)
        except ValueError:
            self._reply(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
            return
        if length < 0 or length > _MAX_BODY_BYTES:
            self._reply(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request body too large"})
            return
        try:
            payload = self.server.bridge.decode_json(self.rfile.read(length))  # type: ignore[attr-defined]
            if self.path == "/reset":
                response = self.server.bridge.reset(payload)  # type: ignore[attr-defined]
            elif self.path == "/infer":
                response = self.server.bridge.infer(payload)  # type: ignore[attr-defined]
            else:
                self._reply(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
        except BridgeProtocolError as error:
            self._reply(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except Exception:
            self._reply(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "policy transport failure"})
            return
        self._reply(HTTPStatus.OK, response)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="127.0.0.1", choices=["127.0.0.1"])
    parser.add_argument("--listen-port", type=int, default=8080)
    parser.add_argument("--policy-server-endpoint", required=True)
    parser.add_argument("--policy-server-token-env", required=True)
    parser.add_argument("--policy-server-request-timeout", type=float, default=600.0)
    return parser


def main() -> int:
    from scripts.eval_policy.groot_policy import PolicyServerClient

    args = build_parser().parse_args()
    token = os.environ.get(args.policy_server_token_env, "")
    client = PolicyServerClient(
        args.policy_server_endpoint,
        token,
        args.policy_server_request_timeout,
    )
    client.ping()
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), _Handler)
    server.bridge = OfficialDockerPolicyBridge(client)  # type: ignore[attr-defined]
    try:
        server.serve_forever()
    finally:
        server.server_close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
