"""GR00T N1.7 policy server speaking the LeHome Challenge docker-policy protocol.

Serves a trained GR00T checkpoint over the exact HTTP contract expected by
``scripts/eval --policy_type docker`` (see ``dummy_docker_policy/``):

    POST /reset  -> {"status": "ok"}
    POST /infer  -> {"actions": [[12 floats], ...]}

Observation/action conversion is the checked contract from
``scripts/eval_policy/groot_policy.py`` in the lehome-groot-trainer worktree
(ryanjin333 GR00T N1.7 pipeline): fixed instruction, three HWC uint8 cameras,
one flat 12-D joint state split into four GR00T modality groups, and a
16-step action chunk flattened back to (16, 12) joint order.

Runtime: the pinned Isaac-GR00T venv at /opt/gr00t-runtime (Python 3.10).
This file intentionally depends only on numpy + the standard library + gr00t.
"""

import argparse
import base64
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List

import numpy as np

_INSTRUCTION = "fold the garment on the table"
_CAMERAS = ("top_rgb", "left_rgb", "right_rgb")
_STATE_GROUPS = (
    ("left_arm", slice(0, 5)),
    ("left_gripper", slice(5, 6)),
    ("right_arm", slice(6, 11)),
    ("right_gripper", slice(11, 12)),
)
_ACTION_GROUPS = tuple(name for name, _ in _STATE_GROUPS)
_ACTION_KEYS = tuple(f"action.{name}" for name in _ACTION_GROUPS)
_ACTION_DIMENSION = 12
# policies/step-12000 (initial BC run) trains with action_horizon=40 per its
# config.json; override via env if a future run changes it.
_ACTION_HORIZON = int(os.environ.get("GROOT_ACTION_HORIZON", "40"))


def _as_frame(value: Any, *, key: str) -> np.ndarray:
    frame = np.asarray(value)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"{key} must be an HWC RGB frame")
    if frame.dtype != np.uint8:
        raise ValueError(f"{key} must have dtype uint8, got {frame.dtype}")
    return frame


def _as_state(value: Any) -> np.ndarray:
    state = np.asarray(value)
    if state.shape != (_ACTION_DIMENSION,):
        raise ValueError(f"observation.state must shape ({_ACTION_DIMENSION},), got {state.shape}")
    state = state.astype(np.float32, copy=False)
    if not np.isfinite(state).all():
        raise ValueError("observation.state contains a non-finite value")
    return state


def build_groot_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one LeHome observation to the strict GR00T policy format."""
    try:
        state = _as_state(observation["observation.state"])
        frames = {
            camera: _as_frame(observation[f"observation.images.{camera}"], key=camera)
            for camera in _CAMERAS
        }
    except KeyError as error:
        raise ValueError(f"missing LeHome observation key: {error.args[0]}") from None

    return {
        "video": {camera: frame[None, None, ...] for camera, frame in frames.items()},
        "state": {name: state[indices][None, None, ...] for name, indices in _STATE_GROUPS},
        "language": {"annotation.human.task_description": [[_INSTRUCTION]]},
    }


def flatten_groot_action_chunk(action: Dict[str, Any]) -> np.ndarray:
    """Flatten a GR00T action chunk to ``(horizon, 12)`` joint order."""
    parts: List[np.ndarray] = []
    horizon = None
    for group, key in zip(_ACTION_GROUPS, _ACTION_KEYS):
        values = np.asarray(action[key], dtype=np.float32)
        if values.ndim == 3:
            if values.shape[0] != 1:
                raise ValueError(f"GR00T action {key} must have batch size 1, got {values.shape}")
            values = values[0]
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError(f"GR00T action {key} must have shape (1,T,D) or (T,D), got {values.shape}")
        if horizon is None:
            horizon = values.shape[0]
        elif values.shape[0] != horizon:
            raise ValueError("GR00T action groups have different horizons")
        expected = 5 if group.endswith("_arm") else 1
        if values.shape[1] != expected:
            raise ValueError(f"GR00T action {key} must have dimension {expected}, got {values.shape[1]}")
        parts.append(values)
    if horizon != _ACTION_HORIZON:
        raise ValueError(f"GR00T action horizon must be {_ACTION_HORIZON}, got {horizon}")
    result = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if result.shape[1] != _ACTION_DIMENSION or not np.isfinite(result).all():
        raise ValueError("GR00T action chunk is not finite 12-D joint data")
    return result


def deserialize_observation(raw: dict) -> Dict[str, np.ndarray]:
    observation = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "base64" in value:
            buf = base64.b64decode(value["base64"])
            observation[key] = np.frombuffer(buf, dtype=value["dtype"]).reshape(value["shape"])
        elif isinstance(value, list):
            observation[key] = np.array(value, dtype=np.float32)
    return observation


class GrootPolicyServer:
    def __init__(self, model_path: str, device: str) -> None:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy import Gr00tPolicy

        if not os.path.isdir(model_path):
            raise ValueError(f"model path is not a directory: {model_path}")
        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=model_path,
            device=device,
            strict=True,
        )

    def reset(self) -> None:
        self.policy.reset()

    def infer(self, observation: Dict[str, np.ndarray]) -> List[np.ndarray]:
        groot_observation = build_groot_observation(observation)
        action, _info = self.policy.get_action(groot_observation)
        chunk = flatten_groot_action_chunk(action)
        return [chunk[i] for i in range(chunk.shape[0])]

    def run(self, host: str, port: int) -> None:
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                try:
                    request = json.loads(body)
                    if self.path == "/reset":
                        server_ref.reset()
                        response = {"status": "ok"}
                    elif self.path == "/infer":
                        observation = deserialize_observation(request)
                        actions = server_ref.infer(observation)
                        response = {"actions": [a.tolist() for a in actions]}
                    else:
                        self.send_error(404, f"Unknown endpoint: {self.path}")
                        return
                except Exception as error:  # noqa: BLE001 - report to simulator, keep serving
                    payload = json.dumps({"error": str(error)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    print(f"[{self.command}] {self.path} failed: {error}", file=sys.stderr, flush=True)
                    return

                body_bytes = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                self.wfile.write(body_bytes)

            def log_message(self, format, *args):
                print(f"[{self.command}] {self.path}", flush=True)

        httpd = HTTPServer((host, port), Handler)
        print(f"GR00T policy server listening on {host}:{port}", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Materialized GR00T checkpoint directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default=os.environ.get("GROOT_POLICY_DEVICE", "cuda:0"))
    args = parser.parse_args()

    server = GrootPolicyServer(args.model_path, args.device)
    server.run(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
