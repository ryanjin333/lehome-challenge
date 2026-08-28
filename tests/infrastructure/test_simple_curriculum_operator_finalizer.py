"""Offline contracts for the local, exact-VM terminal finalizer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location("operator_finalizer", ROOT / "scripts" / "finalize_simple_curriculum_collection.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def _handoff(module):
    body = {
        "schema_version": 1, "kind": "lehome_simple_curriculum_operator_stop_handoff_v1",
        "run_id": "fresh-run-20260828-finalizer", "round_id": "fresh-12k-20260828-finalizer",
        "instance_id": "computeinstance-u00t6xfqhadrcmssa2", "terminal_outcome": "complete",
        "predecessor_receipt_sha256": None, "code_revision": "a" * 40, "code_tree_sha256": "b" * 64,
        "runtime_identity": {"mode": "test"}, "runtime_identity_sha256": module._digest({"mode": "test"}),
        "first_100_receipt_sha256": "c" * 64,
        "evidence": [
            {"stage": stage, "receipt_sha256": "c" * 64, "file_sha256": "d" * 64}
            for stage in ("calibration-matrix", "calibration-head", "first-100-gate", "fresh-report", "replay-matrix", "success-replay")
        ],
    }
    return {**body, "handoff_sha256": module._digest(body)}


def test_finalizer_stops_exact_vm_before_publishing_and_is_idempotent(tmp_path: Path) -> None:
    finalizer = _module(); calls: list[str] = []
    class Provider:
        state = "RUNNING"
        def get(self, instance_id):
            calls.append("get")
            return {"id": instance_id, "name": "lehome-rollout", "state": self.state,
                    "disks": ["computedisk-u00pbe55crxy7jr56x"]}
        def stop(self, instance_id):
            calls.append("stop"); self.state = "STOPPED"
    class Publisher:
        def publish(self, root, *, handoff, stop_observation, seal):
            assert calls[-1] == "get" and stop_observation["state"] == "STOPPED"
            calls.append("publish")
            return {"immutable_revision": "c" * 40, "readback_verified": True, "public_readback_verified": True}
    result = finalizer.finalize_operator_handoff(_handoff(finalizer), provider=Provider(), publisher=Publisher(), staging_parent=tmp_path, stop_timeout_seconds=2)
    assert result["result"] == "finalized" and calls == ["get", "stop", "get", "publish"]


def test_finalizer_stops_even_when_handoff_is_invalid_and_never_publishes(tmp_path: Path) -> None:
    finalizer = _module(); calls: list[str] = []
    class Provider:
        state = "RUNNING"
        def get(self, instance_id):
            calls.append("get")
            return {"id": instance_id, "name": "lehome-rollout", "state": self.state,
                    "disks": ["computedisk-u00pbe55crxy7jr56x"]}
        def stop(self, instance_id): calls.append("stop"); self.state = "STOPPED"
    class Publisher:
        def publish(self, *args, **kwargs): raise AssertionError("invalid handoff must not publish")
    bad = _handoff(finalizer); bad["instance_id"] = "wrong"
    try:
        finalizer.finalize_operator_handoff(bad, provider=Provider(), publisher=Publisher(), staging_parent=tmp_path, stop_timeout_seconds=2)
    except finalizer.FinalizationError as error:
        assert "infrastructure_stop_failure" in str(error)
    else: raise AssertionError("expected finalizer failure")
    assert calls == ["get", "stop", "get"]


def test_restricted_adapters_fetch_only_handoff_and_never_offer_lifecycle_create(monkeypatch, tmp_path: Path) -> None:
    finalizer = _module(); calls: list[tuple[str, ...]] = []
    class Result:
        returncode = 0; stderr = b""
        stdout = json.dumps(_handoff(finalizer)).encode()
    def run(command, **_kwargs): calls.append(tuple(command)); return Result()
    monkeypatch.setattr(finalizer.subprocess, "run", run)
    payload = finalizer.fetch_remote_handoff(ssh_target="operator@host", port=22, campaign_root="/mnt/lehome/campaign", destination=tmp_path / "handoff.json")
    assert payload["instance_id"] == "computeinstance-u00t6xfqhadrcmssa2"
    assert "ClearAllForwardings=yes" in calls[0] and "BatchMode=yes" in calls[0]
    assert calls[0][-3:] == ("cat", "--", "/mnt/lehome/campaign/reports/operator-stop-handoff.json")
    assert not any(word in {"create", "start", "delete", "list"} for word in calls[0])


def test_cli_stops_exact_vm_when_ssh_handoff_fetch_fails(monkeypatch, tmp_path: Path) -> None:
    finalizer = _module(); stopped: list[object] = []
    monkeypatch.setattr(finalizer, "fetch_remote_handoff", lambda **_kwargs: (_ for _ in ()).throw(finalizer.FinalizationError("fetch failed")))
    monkeypatch.setattr(finalizer, "stop_exact_instance", lambda provider, **_kwargs: stopped.append(provider) or {})
    assert finalizer.main(["--ssh-target", "operator@host", "--ssh-port", "22", "--remote-campaign-root", "/mnt/lehome/campaign", "--run-id", "fresh-run-20260828-finalizer", "--round-id", "fresh-12k-20260828-finalizer", "--hf-token-file", str(tmp_path / "token"), "--stop-timeout-seconds", "2"]) == 2
    assert len(stopped) == 1


def test_finalizer_rejects_unknown_or_unproven_complete_handoff() -> None:
    finalizer = _module()
    unknown = _handoff(finalizer); unknown["terminal_outcome"] = "anything"; body = dict(unknown); body.pop("handoff_sha256"); unknown["handoff_sha256"] = finalizer._digest(body)
    with __import__("pytest").raises(finalizer.FinalizationError, match="outcome"):
        finalizer.validate_handoff(unknown)
    incomplete = _handoff(finalizer); incomplete["evidence"] = []; body = dict(incomplete); body.pop("handoff_sha256"); incomplete["handoff_sha256"] = finalizer._digest(body)
    with __import__("pytest").raises(finalizer.FinalizationError, match="evidence"):
        finalizer.validate_handoff(incomplete)


def test_stop_times_out_after_exact_id_validation() -> None:
    finalizer = _module()
    class Provider:
        def get(self, instance_id): return {"id": instance_id, "name": "lehome-rollout", "state": "RUNNING", "disks": ["computedisk-u00pbe55crxy7jr56x"]}
        def stop(self, instance_id): return None
    with __import__("pytest").raises(finalizer.FinalizationError, match="STOPPED"):
        finalizer.stop_exact_instance(Provider(), timeout_seconds=0.0001)


def test_finalizer_rejects_missing_protected_disk_before_stop_dispatch() -> None:
    finalizer = _module(); calls: list[str] = []
    class Provider:
        def get(self, instance_id): calls.append("get"); return {"id": instance_id, "name": "lehome-rollout", "state": "RUNNING", "disks": []}
        def stop(self, instance_id): calls.append("stop")
    with __import__("pytest").raises(finalizer.FinalizationError):
        finalizer.stop_exact_instance(Provider(), timeout_seconds=1)
    assert calls == ["get"]
