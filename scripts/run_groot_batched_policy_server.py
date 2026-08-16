"""Run one four-session, ROUTER-based batched GR00T policy gateway."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal
from stat import S_ISREG
import sys
import tempfile
from time import time_ns
from typing import Any

from lehome.flywheel.policy_batcher import PolicyBatcher
from lehome.flywheel.policy_protocol import (
    DuplicateRequestError,
    ExpiredRequestError,
    PolicyDigestError,
    PolicyRequest,
    PolicyResponse,
    SessionRequestGuard,
    SessionStateError,
    pack_envelope,
    unpack_envelope,
)


def unblock_termination_signals() -> None:
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})


def seed_policy_runtime(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("policy seed must be in 0..2^32-1")
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BatchedPolicyGateway:
    """Socket-independent session admission, batching, and safe response routing."""

    def __init__(
        self, model: Any, *, policy_sha256: str, batch_window_ns: int, now_ns: Any = time_ns,
        seed_identity: int = 0,
    ) -> None:
        self._now_ns = now_ns
        self._policy_sha256 = policy_sha256
        self._guard = SessionRequestGuard(policy_sha256=policy_sha256)
        self._batcher = PolicyBatcher(
            model,
            policy_sha256=policy_sha256,
            batch_window_ns=batch_window_ns,
            seed_identity=seed_identity,
            clock_ns=now_ns,
        )
        self._peers: dict[tuple[str, int, str], bytes] = {}
        self._sessions: set[str] = set()
        self._metrics = {
            "accepted": 0,
            "rejected": 0,
            "responses": 0,
            "dropped_cancelled": 0,
            "dropped_expired": 0,
            "dropped_stale": 0,
        }

    def readiness(self) -> dict[str, object]:
        return {"ready": True, "policy_sha256": self._policy_sha256, "max_sessions": 4}

    def metrics(self) -> dict[str, int]:
        return {**self._metrics, "pending": self._batcher.pending_count, "model_calls": self._batcher.model_calls}

    def drain_receipts(self) -> list[dict[str, object]]:
        return self._batcher.drain_receipts()

    def record_response_sent(self) -> None:
        """Count a response only after its action has reached the ROUTER socket."""

        self._metrics["responses"] += 1

    @staticmethod
    def _identity(request: PolicyRequest) -> tuple[str, int, str]:
        return request.session_id, request.episode_generation, request.request_id

    def _error(self, request: PolicyRequest, code: str) -> bytes:
        self._metrics["rejected"] += 1
        return pack_envelope(PolicyResponse.error(request, error_code=code))

    def receive(self, peer: bytes, payload: bytes) -> bytes | None:
        """Accept one decoded ROUTER payload; inference replies arrive on ``flush``."""

        try:
            request = unpack_envelope(payload)
            if not isinstance(request, PolicyRequest):
                return None
        except ValueError:
            return None
        now_ns = self._now_ns()
        # Do these checks before capacity admission: a rejected reset must not
        # consume one of the gateway's four durable session slots.
        if request.policy_sha256 != self._policy_sha256:
            return self._error(request, "policy_digest_mismatch")
        if now_ns >= request.deadline_ns:
            return self._error(request, "expired")
        if request.operation == "reset" and request.session_id not in self._sessions and len(self._sessions) >= 4:
            return self._error(request, "session_limit")
        try:
            self._guard.accept(request, now_ns=now_ns)
        except PolicyDigestError:
            return self._error(request, "policy_digest_mismatch")
        except ExpiredRequestError:
            return self._error(request, "expired")
        except DuplicateRequestError:
            return self._error(request, "duplicate_request")
        except SessionStateError:
            return self._error(request, "unknown_session")
        self._metrics["accepted"] += 1
        if request.operation == "reset":
            self._sessions.add(request.session_id)
            for identity in tuple(self._peers):
                if identity[0] == request.session_id and identity[1] != request.episode_generation:
                    del self._peers[identity]
                    self._metrics["dropped_stale"] += 1
            return pack_envelope(PolicyResponse.ok(request))
        if request.operation == "cancel":
            cancelled = (request.session_id, request.episode_generation, request.cancelled_request_id or "")
            peer_was_pending = self._peers.pop(cancelled, None) is not None
            batch_was_pending = self._batcher.cancel(PolicyRequest.infer(
                session_id=cancelled[0], episode_generation=cancelled[1], request_id=cancelled[2],
                policy_sha256=request.policy_sha256, deadline_ns=request.deadline_ns, observation={}
            ))
            if peer_was_pending or batch_was_pending:
                self._metrics["dropped_cancelled"] += 1
            return pack_envelope(PolicyResponse.ok(request))
        self._peers[self._identity(request)] = peer
        self._batcher.enqueue(request, received_ns=now_ns)
        return None

    def flush(self) -> list[tuple[bytes, bytes]]:
        now_ns = self._now_ns()
        routed: list[tuple[bytes, bytes]] = []
        flushed = self._batcher.flush(
            now_ns=now_ns,
            is_live=lambda request, now: self._guard.is_request_live(request, now_ns=now),
        )
        for discarded in flushed.discarded:
            peer = self._peers.pop(self._identity(discarded.request), None)
            if peer is None:
                continue
            self._metrics[f"dropped_{discarded.reason}"] += 1
        for result in flushed:
            identity = self._identity(result.request)
            peer = self._peers.pop(identity, None)
            if peer is None:
                self._metrics["dropped_stale"] += 1
                continue
            # Synchronous model execution can cross a deadline.  Re-read the
            # clock at the final routing boundary rather than trusting the
            # admission timestamp from before inference.
            route_now_ns = self._now_ns()
            if not self._guard.is_request_live(result.request, now_ns=route_now_ns):
                if route_now_ns >= result.request.deadline_ns:
                    self._metrics["dropped_expired"] += 1
                else:
                    self._metrics["dropped_cancelled"] += 1
                continue
            routed.append((peer, pack_envelope(result.response)))
        return routed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-window-ms", type=float, default=5.0)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument("--receipt-file", type=Path, required=True)
    return parser


def _safe_output_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.name in {"", ".", ".."} or not candidate.parent.is_dir():
        raise ValueError("policy output parent directory is unavailable")
    if candidate.is_symlink():
        raise ValueError("policy output path must not be a symlink")
    canonical = candidate.parent.resolve(strict=True) / candidate.name
    if canonical.exists() and (canonical.is_symlink() or not canonical.is_file()):
        raise ValueError("policy output path must be a regular file")
    return canonical


def validate_output_paths(ready_file: Path, metrics_file: Path, receipt_file: Path) -> tuple[Path, Path, Path]:
    """Reject unsafe output targets and aliases before a model is loaded."""

    paths = tuple(_safe_output_path(path) for path in (ready_file, metrics_file, receipt_file))
    if len(set(paths)) != len(paths):
        raise ValueError("policy output paths must be distinct canonical files")
    return paths


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path: Path, value: dict[str, object]) -> None:
    canonical = _safe_output_path(path)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{canonical.name}.", suffix=".tmp", dir=canonical.parent)
    try:
        try:
            os.fchmod(fd, 0o600)
            if os.write(fd, payload) != len(payload):
                raise RuntimeError("policy JSON output was not fully written")
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, canonical)
        _fsync_directory(canonical.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_readiness(path: Path, *, ready: bool, policy_sha256: str) -> None:
    _write_json(path, {"ready": ready, "policy_sha256": policy_sha256})


def append_receipt(path: Path, receipt: dict[str, object]) -> None:
    """Durably append one immutable receipt without following a symlink."""

    canonical = _safe_output_path(path)
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(canonical, flags, 0o600)
    try:
        if not S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError("policy receipt path is not a regular file")
        if os.write(fd, payload) != len(payload):
            raise RuntimeError("policy receipt was not fully written")
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(canonical.parent)


def flush_and_route(
    gateway: BatchedPolicyGateway,
    *,
    receipt_path: Path,
    send: Any,
    receipt_writer: Any = append_receipt,
) -> None:
    """Persist all model receipts before emitting the matching action responses."""

    routed = gateway.flush()
    for receipt in gateway.drain_receipts():
        receipt_writer(receipt_path, receipt)
    for peer, response in routed:
        send(peer, response)
        gateway.record_response_sent()


def _load_policy(args: argparse.Namespace) -> Any:
    try:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy import Gr00tPolicy
    except ImportError as error:
        raise RuntimeError("pinned NVIDIA GR00T runtime is unavailable") from error
    return Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, model_path=str(args.model_path), device=args.device, strict=True)


def run(args: argparse.Namespace, *, socket: Any | None = None, model: Any | None = None) -> int:
    unblock_termination_signals()
    if args.host != "127.0.0.1" or not 1 <= args.port <= 65535:
        raise ValueError("batched policy gateway must bind a valid loopback TCP port")
    if args.model_path.is_symlink() or not args.model_path.is_dir():
        raise ValueError("GR00T policy model path must be a materialized directory")
    if args.batch_window_ms <= 0:
        raise ValueError("batch window must be positive")
    ready_file, metrics_file, receipt_file = validate_output_paths(
        args.ready_file, args.metrics_file, args.receipt_file
    )
    seed_policy_runtime(args.seed)
    gateway = BatchedPolicyGateway(
        model if model is not None else _load_policy(args),
        policy_sha256=args.policy_sha256,
        batch_window_ns=round(args.batch_window_ms * 1_000_000),
        seed_identity=args.seed,
    )
    try:
        import zmq
    except ImportError as error:
        raise RuntimeError("batched policy gateway requires pinned pyzmq") from error
    router = socket or zmq.Context.instance().socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    router.bind(f"tcp://{args.host}:{args.port}")
    try:
        write_readiness(ready_file, ready=True, policy_sha256=args.policy_sha256)
        poller = zmq.Poller()
        poller.register(router, zmq.POLLIN)
        while True:
            for _socket, _event in poller.poll(timeout=max(1, round(args.batch_window_ms))):
                message = router.recv_multipart()
                if len(message) != 2:
                    continue
                response = gateway.receive(message[0], message[1])
                if response is not None:
                    router.send_multipart([message[0], response])
            flush_and_route(
                gateway,
                receipt_path=receipt_file,
                send=lambda peer, response: router.send_multipart([peer, response]),
            )
            _write_json(metrics_file, gateway.metrics())
    finally:
        write_readiness(ready_file, ready=False, policy_sha256=args.policy_sha256)
        _write_json(metrics_file, gateway.metrics())
        router.close(linger=0)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"batched policy gateway error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
