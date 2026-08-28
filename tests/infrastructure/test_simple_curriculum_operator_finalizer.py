"""Offline contracts for the local, exact-VM terminal finalizer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


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


def test_cli_stops_exact_vm_when_fetch_raises_raw_oserror(monkeypatch, tmp_path: Path) -> None:
    finalizer = _module(); stopped: list[object] = []
    monkeypatch.setattr(finalizer, "fetch_remote_handoff", lambda **_kwargs: (_ for _ in ()).throw(OSError("temporary disk error")))
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


def test_provider_validator_accepts_only_real_nested_nebius_identity_shape() -> None:
    finalizer = _module()
    raw = {"metadata": {"id": "computeinstance-u00t6xfqhadrcmssa2", "name": "lehome-rollout"}, "status": {"state": "STOPPED"}, "spec": {"boot_disk": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}, "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}]}}
    assert finalizer._validate_instance(raw)["state"] == "STOPPED"
    raw["spec"]["secondary_disks"].append({"existing_disk": {"id": "computedisk-other"}})
    with __import__("pytest").raises(finalizer.FinalizationError): finalizer._validate_instance(raw)


def test_hf_finalizer_reconciles_lost_receipt_response_without_second_upload(tmp_path: Path) -> None:
    finalizer = _module(); calls: list[object] = []
    class Entry:
        relative_path = "reports/final-publication.json"
    class Transport:
        def resolve_approved_ref(self, **_kwargs): return "a" * 40
        def list_tree(self, **_kwargs):
            return ("reports/operator-stop-handoff.json", "reports/stopped-observation.json", "seals/final-seal.json", "reports/final-publication.json")
        def upload_files(self, **_kwargs): raise AssertionError("lost response retry must not upload again")
        def download_files(self, *, destination, relative_paths, **_kwargs):
            for relative in relative_paths:
                path = destination / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("{}\n", encoding="utf-8")
            return "a" * 40
    module = SimpleNamespace(
        _load_token=lambda _path: "token",
        CollectionPublicationBundle=lambda **kwargs: SimpleNamespace(**kwargs),
        _collect_entries=lambda bundle: (Entry(),) if bundle.files == ("reports/final-publication.json",) else (Entry(), Entry(), Entry()),
        _tree_files=lambda entries, **_kwargs: set(entries),
        _verify_download=lambda **kwargs: calls.append(kwargs["token"]),
        HuggingFacePublicDatasetTransport=lambda: Transport(),
    )
    handoff = _handoff(finalizer); stopped = {"observation_sha256": "e" * 64}; seal = {"seal_sha256": "f" * 64}
    result = finalizer.HfFinalizerPublisher(tmp_path / "token", module=module, transport=Transport()).publish(tmp_path, handoff=handoff, stop_observation=stopped, seal=seal)
    assert result["immutable_revision"] == "a" * 40
    assert calls == ["token", None]


def test_hf_lost_response_reconcile_stages_real_receipt_for_actual_collector(tmp_path: Path) -> None:
    finalizer = _module(); source = ROOT / "scripts/publish_simple_curriculum_collection.py"
    spec = importlib.util.spec_from_file_location("real_publisher_for_finalizer", source); assert spec and spec.loader
    publisher_module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = publisher_module; spec.loader.exec_module(publisher_module)
    token = tmp_path / "token"; token.write_text("hf_fake_token", encoding="utf-8"); token.chmod(0o600)
    handoff = _handoff(finalizer); stopped = {"observation_sha256": "e" * 64}; seal = {"seal_sha256": "f" * 64}; calls: list[object] = []; remote_bytes: dict[str, bytes] = {}
    class Transport:
        def resolve_approved_ref(self, **_kwargs): return "a" * 40
        def list_tree(self, *, remote_prefix, **_kwargs):
            if not remote_bytes:
                for path in ("reports/operator-stop-handoff.json", "reports/stopped-observation.json", "seals/final-seal.json"):
                    remote_bytes[path] = (tmp_path / path).read_bytes()
                remote_bytes["reports/final-publication.json"] = b"{}\n"
            return tuple(SimpleNamespace(relative_path=f"{remote_prefix}/{path}", entry_type="file") for path in ("reports/operator-stop-handoff.json", "reports/stopped-observation.json", "seals/final-seal.json", "reports/final-publication.json"))
        def upload_files(self, **_kwargs): raise AssertionError("exact receipt prefix must reconcile without upload")
        def download_files(self, *, destination, relative_paths, token, **_kwargs):
            calls.append(token)
            for relative in relative_paths:
                source_path = tmp_path / relative; source_path.parent.mkdir(parents=True, exist_ok=True)
                if not source_path.exists(): source_path.write_bytes(b"{}\n")
            if len(calls) == 2:
                remote_bytes.update({path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file() and path.name != "token"})
            for relative in relative_paths:
                target = destination / relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(remote_bytes[relative])
            return "a" * 40
    # The first authenticated fetch materializes the remote receipt at the
    # temporary root before the repository's descriptor-safe collector opens
    # it. This is the crash-recovery staging invariant.
    finalizer._atomic_json(tmp_path / "reports/operator-stop-handoff.json", handoff)
    finalizer._atomic_json(tmp_path / "reports/stopped-observation.json", stopped)
    finalizer._atomic_json(tmp_path / "seals/final-seal.json", seal)
    transport = Transport(); transport.list_tree(remote_prefix=f"collection-rounds/{handoff['run_id']}")
    transport.download_files(destination=tmp_path, relative_paths=("reports/final-publication.json",), token="hf_fake_token")
    entries = publisher_module._collect_entries(publisher_module.CollectionPublicationBundle(root=tmp_path, run_id=handoff["run_id"], repository="ryanjin333/lehome-groot-n17-rollouts", revision="main", files=("reports/final-publication.json",)))
    assert entries[0].relative_path == "reports/final-publication.json" and calls == ["hf_fake_token"]
