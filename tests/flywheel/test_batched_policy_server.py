from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from time import time_ns
import types

import numpy as np
import pytest


_REPOSITORY_ROOT = Path(__file__).parents[2]


def _server_module():
    module_name = "run_groot_batched_policy_server_under_test"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name, _REPOSITORY_ROOT / "scripts" / "run_groot_batched_policy_server.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def _gateway_class():
    return _server_module().BatchedPolicyGateway


class _Model:
    def get_action(self, observation):
        batch = observation["state"]["left_arm"].shape[0]
        return {
            "left_arm": np.zeros((batch, 16, 5), dtype=np.float32),
            "left_gripper": np.zeros((batch, 16, 1), dtype=np.float32),
            "right_arm": np.zeros((batch, 16, 5), dtype=np.float32),
            "right_gripper": np.zeros((batch, 16, 1), dtype=np.float32),
        }, {}


def _request(protocol, *, session_id: str, request_id: str, operation: str = "infer", digest: str = "a" * 64, generation: int = 1):
    common = dict(session_id=session_id, episode_generation=generation, request_id=request_id, policy_sha256=digest, deadline_ns=10_000)
    if operation == "reset":
        return protocol.PolicyRequest.reset(**common)
    if operation == "cancel":
        return protocol.PolicyRequest.cancel(**common, cancelled_request_id="infer")
    return protocol.PolicyRequest.infer(
        **common,
        observation={
            "video": {camera: np.zeros((1, 1, 2, 3, 3), dtype=np.uint8) for camera in ("top_rgb", "left_rgb", "right_rgb")},
            "state": {
                "left_arm": np.zeros((1, 1, 5), dtype=np.float32), "left_gripper": np.zeros((1, 1, 1), dtype=np.float32),
                "right_arm": np.zeros((1, 1, 5), dtype=np.float32), "right_gripper": np.zeros((1, 1, 1), dtype=np.float32),
            },
            "language": {"annotation.human.task_description": [["fold the garment on the table"]]},
        },
    )


def test_gateway_routes_only_live_guarded_requests_and_reports_readiness_metrics() -> None:
    from lehome.flywheel import policy_protocol as protocol
    BatchedPolicyGateway = _gateway_class()

    now = [1_000]
    gateway = BatchedPolicyGateway(_Model(), policy_sha256="a" * 64, batch_window_ns=100, now_ns=lambda: now[0])
    address = b"worker-0"
    reset = _request(protocol, session_id="worker-0", request_id="reset", operation="reset")
    assert protocol.unpack_envelope(gateway.receive(address, protocol.pack_envelope(reset))) .status == "ok"
    infer = _request(protocol, session_id="worker-0", request_id="infer")
    assert gateway.receive(address, protocol.pack_envelope(infer)) is None
    cancel = _request(protocol, session_id="worker-0", request_id="cancel", operation="cancel")
    assert protocol.unpack_envelope(gateway.receive(address, protocol.pack_envelope(cancel))).status == "ok"

    assert gateway.flush() == []
    assert gateway.readiness() == {"ready": True, "policy_sha256": "a" * 64, "max_sessions": 4}
    assert gateway.metrics()["dropped_cancelled"] == 1


def test_gateway_rejects_policy_digest_mismatch_and_fifth_session() -> None:
    from lehome.flywheel import policy_protocol as protocol
    BatchedPolicyGateway = _gateway_class()

    gateway = BatchedPolicyGateway(_Model(), policy_sha256="a" * 64, batch_window_ns=100, now_ns=lambda: 1_000)
    wrong = _request(protocol, session_id="worker-x", request_id="wrong", operation="reset", digest="b" * 64)
    response = protocol.unpack_envelope(gateway.receive(b"x", protocol.pack_envelope(wrong)))
    assert response.error_code == "policy_digest_mismatch"
    for index in range(4):
        request = _request(protocol, session_id=f"worker-{index}", request_id=f"reset-{index}", operation="reset")
        assert protocol.unpack_envelope(gateway.receive(str(index).encode(), protocol.pack_envelope(request))).status == "ok"
    excess = _request(protocol, session_id="worker-4", request_id="reset-4", operation="reset")
    response = protocol.unpack_envelope(gateway.receive(b"4", protocol.pack_envelope(excess)))
    assert response.error_code == "session_limit"


def test_gateway_does_not_let_one_slow_peer_block_another_session_response() -> None:
    from lehome.flywheel import policy_protocol as protocol
    BatchedPolicyGateway = _gateway_class()

    now = [1_000]
    gateway = BatchedPolicyGateway(_Model(), policy_sha256="a" * 64, batch_window_ns=100, now_ns=lambda: now[0])
    for index in range(2):
        reset = _request(protocol, session_id=f"worker-{index}", request_id=f"reset-{index}", operation="reset")
        assert gateway.receive(f"peer-{index}".encode(), protocol.pack_envelope(reset)) is not None
    slow = _request(protocol, session_id="worker-0", request_id="slow")
    fast = _request(protocol, session_id="worker-1", request_id="fast")
    assert gateway.receive(b"slow-peer", protocol.pack_envelope(slow)) is None
    assert gateway.receive(b"fast-peer", protocol.pack_envelope(fast)) is None

    now[0] = 1_100
    routed = gateway.flush()

    assert [(peer, protocol.unpack_envelope(payload).request_id) for peer, payload in routed] == [
        (b"slow-peer", "slow"), (b"fast-peer", "fast"),
    ]


def test_gateway_discards_queued_old_generation_reply_before_routing() -> None:
    from lehome.flywheel import policy_protocol as protocol
    BatchedPolicyGateway = _gateway_class()

    now = [1_000]
    gateway = BatchedPolicyGateway(_Model(), policy_sha256="a" * 64, batch_window_ns=100, now_ns=lambda: now[0])
    reset_one = _request(protocol, session_id="worker-0", request_id="reset-1", operation="reset")
    assert gateway.receive(b"peer", protocol.pack_envelope(reset_one)) is not None
    pending = _request(protocol, session_id="worker-0", request_id="infer-1")
    assert gateway.receive(b"peer", protocol.pack_envelope(pending)) is None
    reset_two = _request(protocol, session_id="worker-0", request_id="reset-2", operation="reset", generation=2)
    assert gateway.receive(b"peer", protocol.pack_envelope(reset_two)) is not None

    now[0] = 1_100
    assert gateway.flush() == []
    assert gateway.metrics()["dropped_stale"] == 1


def test_gateway_rechecks_deadline_after_model_work_and_emits_auditable_batch_receipt() -> None:
    from lehome.flywheel import policy_protocol as protocol

    BatchedPolicyGateway = _gateway_class()
    now = [1_000]

    class DeadlineCrossingModel(_Model):
        def get_action(self, observation):
            result = super().get_action(observation)
            now[0] = 2_000
            return result

    gateway = BatchedPolicyGateway(
        DeadlineCrossingModel(), policy_sha256="a" * 64, batch_window_ns=100,
        now_ns=lambda: now[0], seed_identity=17,
    )
    reset = _request(protocol, session_id="worker-0", request_id="reset", operation="reset")
    reset_response = gateway.receive(b"peer", protocol.pack_envelope(reset))
    assert reset_response is not None
    decoded_reset = protocol.unpack_envelope(reset_response)
    assert decoded_reset.status == "ok", decoded_reset.error_code
    infer = _request(protocol, session_id="worker-0", request_id="infer")
    infer = protocol.PolicyRequest.infer(
        session_id=infer.session_id, episode_generation=infer.episode_generation,
        request_id=infer.request_id, policy_sha256=infer.policy_sha256,
        deadline_ns=1_050, observation=infer.observation,
    )
    assert gateway.receive(b"peer", protocol.pack_envelope(infer)) is None

    assert gateway.flush() == []
    receipt = gateway.drain_receipts()[0]
    assert receipt["identities"] == [{"session_id": "worker-0", "episode_generation": 1, "request_id": "infer"}]
    assert receipt["batch_occupancy"] == 1
    assert receipt["seed_identity"] == 17
    assert receipt["returned_action_chunk_sha256"]
    assert gateway.metrics()["dropped_expired"] == 1


def test_append_receipt_is_jsonl_durable_and_refuses_symlink_target(tmp_path) -> None:
    module = _server_module()
    receipt_path = tmp_path / "inference-receipts.jsonl"
    receipt = {"schema_version": 1, "request_id": "request-1"}

    module.append_receipt(receipt_path, receipt)

    assert [json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines()] == [receipt]
    symlink = tmp_path / "receipt-link.jsonl"
    symlink.symlink_to(receipt_path)
    with pytest.raises(ValueError, match="symlink"):
        module.append_receipt(symlink, receipt)


def test_receipt_failure_sends_no_action_before_the_durable_append(tmp_path) -> None:
    from lehome.flywheel import policy_protocol as protocol

    module = _server_module()
    now = [1_000]
    gateway = module.BatchedPolicyGateway(
        _Model(), policy_sha256="a" * 64, batch_window_ns=100, now_ns=lambda: now[0]
    )
    reset = _request(protocol, session_id="worker-0", request_id="reset", operation="reset")
    assert gateway.receive(b"peer", protocol.pack_envelope(reset)) is not None
    infer = _request(protocol, session_id="worker-0", request_id="infer")
    assert gateway.receive(b"peer", protocol.pack_envelope(infer)) is None
    now[0] = 1_100
    sent = []

    with pytest.raises(OSError, match="disk full"):
        module.flush_and_route(
            gateway,
            receipt_path=tmp_path / "receipts.jsonl",
            send=lambda peer, payload: sent.append((peer, payload)),
            receipt_writer=lambda _path, _receipt: (_ for _ in ()).throw(OSError("disk full")),
        )

    assert sent == []
    assert gateway.metrics()["responses"] == 0
    assert gateway._peers == {}


def test_gateway_releases_peer_for_pre_inference_expiry_and_counts_the_drop() -> None:
    from lehome.flywheel import policy_protocol as protocol

    gateway_class = _gateway_class()
    now = [1_000]
    gateway = gateway_class(_Model(), policy_sha256="a" * 64, batch_window_ns=100, now_ns=lambda: now[0])
    reset = _request(protocol, session_id="worker-0", request_id="reset", operation="reset")
    assert gateway.receive(b"peer", protocol.pack_envelope(reset)) is not None
    request = _request(protocol, session_id="worker-0", request_id="expired")
    request = protocol.PolicyRequest.infer(
        session_id=request.session_id, episode_generation=request.episode_generation,
        request_id=request.request_id, policy_sha256=request.policy_sha256,
        deadline_ns=1_050, observation=request.observation,
    )
    assert gateway.receive(b"peer", protocol.pack_envelope(request)) is None

    now[0] = 1_050
    assert gateway.flush() == []
    assert gateway.metrics()["dropped_expired"] == 1
    assert gateway._peers == {}


def test_gateway_rejects_request_expired_on_the_session_client_wall_clock() -> None:
    from lehome.flywheel import policy_protocol as protocol

    gateway = _gateway_class()(_Model(), policy_sha256="a" * 64, batch_window_ns=100)
    reset = _request(protocol, session_id="worker-0", request_id="reset", operation="reset")
    reset = protocol.PolicyRequest.reset(
        session_id=reset.session_id, episode_generation=reset.episode_generation,
        request_id=reset.request_id, policy_sha256=reset.policy_sha256,
        deadline_ns=time_ns() + 1_000_000_000,
    )
    assert gateway.receive(b"peer", protocol.pack_envelope(reset)) is not None
    expired = _request(protocol, session_id="worker-0", request_id="expired")
    expired = protocol.PolicyRequest.infer(
        session_id=expired.session_id, episode_generation=expired.episode_generation,
        request_id=expired.request_id, policy_sha256=expired.policy_sha256,
        deadline_ns=time_ns() - 1, observation=expired.observation,
    )

    response = protocol.unpack_envelope(gateway.receive(b"peer", protocol.pack_envelope(expired)))

    assert response.status == "error"
    assert response.error_code == "expired"


def test_output_paths_reject_symlinks_and_canonical_aliases(tmp_path) -> None:
    module = _server_module()
    ready = tmp_path / "ready.json"
    metrics = tmp_path / "metrics.json"
    receipt = tmp_path / "receipts.jsonl"
    receipt.symlink_to(ready)

    with pytest.raises(ValueError, match="symlink"):
        module.validate_output_paths(ready, metrics, receipt)
    with pytest.raises(ValueError, match="distinct"):
        module.validate_output_paths(ready, metrics, tmp_path / "." / "ready.json")


def test_first_receipt_fsyncs_file_and_parent_before_action_send(monkeypatch, tmp_path) -> None:
    module = _server_module()
    receipt_path = tmp_path / "receipts.jsonl"
    fsync_calls = []
    real_fsync = module.os.fsync
    monkeypatch.setattr(module.os, "fsync", lambda fd: fsync_calls.append(fd) or real_fsync(fd))

    module.append_receipt(receipt_path, {"schema_version": 1})

    assert len(fsync_calls) == 2


def test_readiness_is_durably_marked_not_ready_on_shutdown(tmp_path) -> None:
    module = _server_module()
    path = tmp_path / "ready.json"

    module.write_readiness(path, ready=True, policy_sha256="a" * 64)
    module.write_readiness(path, ready=False, policy_sha256="a" * 64)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "policy_sha256": "a" * 64,
        "ready": False,
    }


def test_run_marks_ready_file_not_ready_when_polling_stops(monkeypatch, tmp_path) -> None:
    module = _server_module()
    monkeypatch.setattr(module, "seed_policy_runtime", lambda _seed: None)

    class Socket:
        def setsockopt(self, *_args):
            pass

        def bind(self, _endpoint):
            pass

        def close(self, **_kwargs):
            pass

    class Poller:
        def register(self, *_args):
            pass

        def poll(self, **_kwargs):
            raise KeyboardInterrupt

    fake_zmq = types.SimpleNamespace(ROUTER=1, LINGER=2, POLLIN=3, Poller=Poller)
    monkeypatch.setitem(sys.modules, "zmq", fake_zmq)
    model_path = tmp_path / "model"
    model_path.mkdir()
    ready = tmp_path / "ready.json"
    args = types.SimpleNamespace(
        model_path=model_path, policy_sha256="a" * 64, host="127.0.0.1", port=5500,
        device="cpu", seed=1, batch_window_ms=1.0, ready_file=ready,
        metrics_file=tmp_path / "metrics.json", receipt_file=tmp_path / "receipts.jsonl",
    )

    with pytest.raises(KeyboardInterrupt):
        module.run(args, socket=Socket(), model=_Model())

    assert json.loads(ready.read_text(encoding="utf-8"))["ready"] is False


def test_write_json_unlinks_its_temporary_file_when_fsync_fails(monkeypatch, tmp_path) -> None:
    module = _server_module()
    destination = tmp_path / "metrics.json"
    monkeypatch.setattr(module.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("fsync failed")))

    with pytest.raises(OSError, match="fsync failed"):
        module._write_json(destination, {"metric": 1})

    assert not destination.exists()
    assert list(tmp_path.glob(".metrics.json.*.tmp")) == []


def test_run_marks_ready_not_ready_when_poller_startup_fails(monkeypatch, tmp_path) -> None:
    module = _server_module()
    monkeypatch.setattr(module, "seed_policy_runtime", lambda _seed: None)

    class Socket:
        def setsockopt(self, *_args):
            pass

        def bind(self, _endpoint):
            pass

        def close(self, **_kwargs):
            pass

    class FailingPoller:
        def register(self, *_args):
            raise RuntimeError("poller register failed")

    fake_zmq = types.SimpleNamespace(ROUTER=1, LINGER=2, POLLIN=3, Poller=FailingPoller)
    monkeypatch.setitem(sys.modules, "zmq", fake_zmq)
    model_path = tmp_path / "model"
    model_path.mkdir()
    ready = tmp_path / "ready.json"
    args = types.SimpleNamespace(
        model_path=model_path, policy_sha256="a" * 64, host="127.0.0.1", port=5500,
        device="cpu", seed=1, batch_window_ms=1.0, ready_file=ready,
        metrics_file=tmp_path / "metrics.json", receipt_file=tmp_path / "receipts.jsonl",
    )

    with pytest.raises(RuntimeError, match="poller register failed"):
        module.run(args, socket=Socket(), model=_Model())

    assert json.loads(ready.read_text(encoding="utf-8"))["ready"] is False
