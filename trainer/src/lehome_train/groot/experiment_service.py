"""Small authenticated JSON HTTP facade for the single-writer controller."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import stat
import time
from pathlib import Path
from typing import Any, Mapping


_MAX_BODY = 65_536


def validate_bind_address(address: str, *, allow_tls_proxy: bool = False) -> str:
    if address in {"0.0.0.0", "::"} and not allow_tls_proxy:
        raise ValueError("wildcard bind requires an explicit TLS proxy")
    if not address:
        raise ValueError("bind address is required")
    return address


def load_bearer_token(path: str | Path) -> str:
    value = Path(path)
    if value.is_symlink() or not value.is_file() or stat.S_IMODE(value.stat().st_mode) != 0o600:
        raise ValueError("controller token file must be regular mode 0600")
    token = value.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("controller token is empty")
    return token


def _lease_json(lease: Any) -> dict[str, object]:
    if lease is None:
        return {"lease": None}
    if lease.job is None:
        raise RuntimeError("controller lease lacks immutable job")
    value: dict[str, object] = {
        "lease": True,
        "lease_id": lease.lease_id,
        "experiment_id": lease.experiment_id,
        "worker_id": lease.worker_id,
        "capability": lease.capability,
        "expires_ns": lease.expires_ns,
        "job": dict(lease.job.raw),
    }
    if lease.publication is not None:
        value["publication"] = dict(lease.publication)
    if lease.parent_publication is not None:
        value["parent_publication"] = dict(lease.parent_publication)
    if lease.evaluation_matrix_sha256 is not None:
        value["evaluation_matrix_sha256"] = lease.evaluation_matrix_sha256
    return value


def _required(body: Mapping[str, object], fields: set[str], *, optional: set[str] = set()) -> None:
    if set(body) - optional != fields:
        raise ValueError("request has unknown or missing field")


class ExperimentService(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], controller: Any, token_file: str | Path, *, allow_tls_proxy: bool = False) -> None:
        validate_bind_address(address[0], allow_tls_proxy=allow_tls_proxy)
        self.controller, self.token = controller, load_bearer_token(token_file)
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    server: ExperimentService

    def log_message(self, *_: object) -> None:
        return

    def _json(self, status: int, body: dict[str, object]) -> None:
        payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/capacity":
            if self.headers.get("Authorization") != "Bearer " + self.server.token:
                self._json(401, {"error": "unauthorized"})
                return
            try:
                self._json(200, self.server.controller.capacity_snapshot(now_ns=time.time_ns()))
            except (TypeError, ValueError) as error:
                self._json(400, {"error": str(error)[:256] or "invalid request"})
            except RuntimeError as error:
                self._json(409, {"error": str(error)[:256] or "conflict"})
        else:
            self._json(404, {"error": "not found"})

    def _body(self) -> dict[str, object]:
        length_raw = self.headers.get("Content-Length")
        if length_raw is None or not length_raw.isdecimal() or int(length_raw) > _MAX_BODY:
            raise ValueError("request body is invalid or too large")
        decoded = json.loads(self.rfile.read(int(length_raw)))
        if not isinstance(decoded, dict):
            raise ValueError("request must be an object")
        return decoded

    @staticmethod
    def _lease(controller: Any, body: Mapping[str, object]) -> Any:
        _required(body, {"lease_id", "experiment_id", "worker_id", "now_ns"}, optional={"reason", "receipt_sha256", "report"})
        return controller.lease_for(body["lease_id"], body["experiment_id"], body["worker_id"], now_ns=body["now_ns"])

    def do_POST(self) -> None:
        if self.headers.get("Authorization") != "Bearer " + self.server.token:
            self._json(401, {"error": "unauthorized"})
            return
        try:
            body = self._body()
            controller = self.server.controller
            if self.path == "/lease":
                _required(body, {"worker_id", "capability", "now_ns", "lease_ns"}, optional={"manifest_set_sha256"})
                manifest_set = body.get("manifest_set_sha256")
                if manifest_set is not None and type(manifest_set) is not str:
                    raise ValueError("manifest set digest is invalid")
                result = _lease_json(controller.lease_next(body["worker_id"], body["capability"], body["now_ns"], body["lease_ns"], manifest_set))
            elif self.path == "/heartbeat":
                _required(body, {"worker_id", "lease_id", "now_ns", "lease_ns"})
                result = _lease_json(controller.heartbeat(body["worker_id"], body["lease_id"], body["now_ns"], body["lease_ns"]))
            elif self.path == "/complete":
                _required(body, {"lease_id", "experiment_id", "worker_id", "receipt_sha256", "now_ns"})
                state = controller.reconcile_terminal_receipt_by_identity(
                    body["lease_id"], body["experiment_id"], body["worker_id"],
                    body["receipt_sha256"], body["now_ns"],
                )
                result = {"status": state.lower()}
            elif self.path == "/publication":
                _required(body, {"experiment_id", "publication", "now_ns"})
                if not isinstance(body["publication"], Mapping):
                    raise ValueError("publication must be an object")
                state = controller.publication_verified(body["experiment_id"], body["publication"], body["now_ns"])
                result = {"status": state.lower()}
            elif self.path == "/dependency":
                _required(body, {"receipt", "now_ns"})
                if not isinstance(body["receipt"], Mapping):
                    raise ValueError("dependency receipt must be an object")
                result = {"unblocked": controller.satisfy_dependency(body["receipt"], body["now_ns"])}
            elif self.path == "/awr-admission":
                _required(body, {"experiment_id", "receipt", "now_ns"})
                if not isinstance(body["receipt"], Mapping):
                    raise ValueError("AWR-style admission receipt must be an object")
                result = {
                    "status": "ready",
                    "receipt_sha256": controller.satisfy_awr_style_admission(
                        body["experiment_id"], body["receipt"], body["now_ns"],
                    ),
                }
            elif self.path == "/finalists":
                _required(body, {"experiment_ids", "matrix_sha256", "now_ns"})
                if not isinstance(body["experiment_ids"], list):
                    raise ValueError("finalist IDs must be a list")
                result = {"enqueued": controller.enqueue_finalists(body["experiment_ids"], matrix_sha256=body["matrix_sha256"], now_ns=body["now_ns"])}
            elif self.path in {"/retryable", "/block"}:
                _required(body, {"lease_id", "experiment_id", "worker_id", "reason", "now_ns"})
                lease = controller.lease_for(body["lease_id"], body["experiment_id"], body["worker_id"], now_ns=body["now_ns"])
                (controller.retryable if self.path == "/retryable" else controller.block_infrastructure)(lease, body["reason"], body["now_ns"])
                result = {"status": "retryable" if self.path == "/retryable" else "blocked"}
            elif self.path == "/evaluation":
                _required(body, {"lease_id", "experiment_id", "worker_id", "report", "now_ns"})
                if not isinstance(body["report"], Mapping):
                    raise ValueError("evaluation report must be an object")
                lease = controller.lease_for(body["lease_id"], body["experiment_id"], body["worker_id"], now_ns=body["now_ns"])
                controller.submit_evaluation(lease, body["report"], body["now_ns"])
                result = {"status": "completed"}
            elif self.path == "/final-evaluation":
                _required(body, {"lease_id", "experiment_id", "worker_id", "report", "now_ns"})
                if not isinstance(body["report"], Mapping):
                    raise ValueError("final evaluation report must be an object")
                lease = controller.lease_for(body["lease_id"], body["experiment_id"], body["worker_id"], now_ns=body["now_ns"])
                controller.submit_final_evaluation(lease, body["report"], body["now_ns"])
                result = {"status": "completed"}
            elif self.path == "/final-winner":
                _required(body, {"baseline_report", "matrix_sha256", "now_ns"})
                baseline = body["baseline_report"]
                if baseline is not None and not isinstance(baseline, Mapping):
                    raise ValueError("baseline report must be an object or null")
                result = controller.final_winner_decision(
                    baseline_report=baseline,
                    matrix_sha256=body["matrix_sha256"],
                    now_ns=body["now_ns"],
                )
            else:
                self._json(404, {"error": "not found"})
                return
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self._json(400, {"error": str(error)[:256] or "invalid request"})
            return
        except RuntimeError as error:
            self._json(409, {"error": str(error)[:256] or "conflict"})
            return
        self._json(200, result)
