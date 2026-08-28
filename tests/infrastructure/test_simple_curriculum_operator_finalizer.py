"""Offline contracts for the local, exact-VM terminal finalizer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location("operator_finalizer", ROOT / "scripts" / "finalize_simple_curriculum_collection.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def _instance(module, state: str = "RUNNING", *, disks: list[object] | None = None) -> dict[str, object]:
    return {
        "metadata": {"id": module.EXACT_INSTANCE_ID, "name": module.EXACT_INSTANCE_NAME},
        "status": {"state": state},
        "spec": {
            "boot_disk": {"managed_disk": {"spec": {"source_image_id": module.EXACT_IMAGE_ID}}},
            "secondary_disks": disks if disks is not None else [
                {"existing_disk": {"id": module.PROTECTED_DISK_ID}},
            ],
        },
    }


def _handoff(module):
    body = {
        "schema_version": 1, "kind": "lehome_simple_curriculum_operator_stop_handoff_v1",
        "run_id": "fresh-run-20260828-finalizer", "round_id": "fresh-12k-20260828-finalizer",
        "instance_id": "computeinstance-u00t6xfqhadrcmssa2", "terminal_outcome": "complete",
        "predecessor_receipt_sha256": "c" * 64, "code_revision": "a" * 40, "code_tree_sha256": "b" * 64,
        "runtime_identity": {"mode": "test"}, "runtime_identity_sha256": module._digest({"mode": "test"}),
        "first_100_receipt_sha256": "c" * 64,
        "evidence": [
            {"stage": stage, "receipt_sha256": "c" * 64, "file_sha256": "d" * 64}
            for stage in (
                "calibration-matrix", "calibration-head", "first-100-gate",
                "calibration-tail", "calibration-report", "curriculum-matrix",
                "curriculum-a", "curriculum-b", "fresh-report", "replay-matrix",
                "success-replay",
            )
        ],
    }
    return {**body, "handoff_sha256": module._digest(body)}


def _final_receipt(module, handoff, seal, *, revision="a" * 40, bundle="b" * 64):
    body = {
        "schema_version": 1,
        "kind": "lehome_simple_curriculum_operator_finalization_receipt_v1",
        "run_id": handoff["run_id"],
        "round_id": handoff["round_id"],
        "evidence_revision": revision,
        "evidence_bundle_sha256": bundle,
        "final_seal_sha256": seal["seal_sha256"],
        "readback_verified": True,
        "public_readback_verified": True,
    }
    return {**body, "receipt_sha256": module._digest(body)}


def test_finalizer_stops_exact_vm_before_publishing_and_is_idempotent(tmp_path: Path) -> None:
    finalizer = _module(); calls: list[str] = []
    class Provider:
        state = "RUNNING"
        def get(self, instance_id):
            calls.append("get")
            return _instance(finalizer, self.state)
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
            return _instance(finalizer, self.state)
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


def test_restricted_ssh_adapter_rejects_option_like_target(tmp_path: Path) -> None:
    finalizer = _module()
    with __import__("pytest").raises(finalizer.FinalizationError, match="SSH target"):
        finalizer.fetch_remote_handoff(
            ssh_target="-Ffile@host", port=22, campaign_root="/mnt/lehome/campaign", destination=tmp_path / "handoff.json"
        )


def test_nebius_adapter_turns_command_timeout_into_finalization_failure(monkeypatch) -> None:
    finalizer = _module()
    monkeypatch.setattr(finalizer.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("nebius", 1)))
    with __import__("pytest").raises(finalizer.FinalizationError, match="timed out"):
        finalizer.SubprocessNebiusProvider().get("computeinstance-u00t6xfqhadrcmssa2")


def test_nebius_adapter_uses_explicit_noninteractive_deadline_bounded_flags(monkeypatch) -> None:
    finalizer = _module(); calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    fixture = json.loads((ROOT / "infrastructure/nebius/.tools/cpu-only-v7-vm-before-start.json").read_text())

    class Result:
        returncode = 0
        stderr = ""
        stdout = json.dumps(fixture)

    def run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return Result()

    monkeypatch.setattr(finalizer.subprocess, "run", run)
    provider = finalizer.SubprocessNebiusProvider()
    provider.set_stop_deadline(time.monotonic() + 3.0)
    provider.get(finalizer.EXACT_INSTANCE_ID)
    provider.stop(finalizer.EXACT_INSTANCE_ID)

    assert len(calls) == 2
    for command, kwargs in calls:
        assert "--no-browser" in command
        assert "--no-progress" in command
        assert "--no-check-update" in command
        assert "--auth-timeout" in command
        assert "--per-retry-timeout" in command
        assert "--timeout" in command
        assert "--retries" in command
        assert kwargs["timeout"] <= 3.0


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


def test_handoff_requires_exact_stage_chain_and_authenticated_links() -> None:
    finalizer = _module()

    def resign(payload):
        body = dict(payload); body.pop("handoff_sha256", None)
        payload["handoff_sha256"] = finalizer._digest(body)

    for mutate in (
        lambda value: value["evidence"].pop(),
        lambda value: value["evidence"].append(dict(value["evidence"][-1])),
        lambda value: value["evidence"].__setitem__(0, {**value["evidence"][0], "stage": "unknown"}),
        lambda value: value.__setitem__("first_100_receipt_sha256", "e" * 64),
        lambda value: value.__setitem__("predecessor_receipt_sha256", "not-a-digest"),
    ):
        handoff = _handoff(finalizer)
        mutate(handoff); resign(handoff)
        with __import__("pytest").raises(finalizer.FinalizationError):
            finalizer.validate_handoff(handoff)


def test_handoff_allows_only_real_reachable_stop_prefixes() -> None:
    finalizer = _module()

    def resign(payload):
        body = dict(payload); body.pop("handoff_sha256", None)
        payload["handoff_sha256"] = finalizer._digest(body)

    valid = _handoff(finalizer)
    valid["terminal_outcome"] = "infrastructure_stop_failure"
    valid["evidence"] = []
    valid["predecessor_receipt_sha256"] = None
    valid["first_100_receipt_sha256"] = None
    resign(valid)
    finalizer.validate_handoff(valid)

    invalid = _handoff(finalizer)
    invalid["terminal_outcome"] = "fidelity_stop"
    invalid["evidence"] = invalid["evidence"][:2]
    invalid["predecessor_receipt_sha256"] = invalid["evidence"][-1]["receipt_sha256"]
    invalid["first_100_receipt_sha256"] = None
    resign(invalid)
    with __import__("pytest").raises(finalizer.FinalizationError):
        finalizer.validate_handoff(invalid)


def test_handoff_outcome_stage_sets_and_digests_are_fail_closed() -> None:
    finalizer = _module()

    def resign(payload):
        body = dict(payload); body.pop("handoff_sha256", None)
        payload["handoff_sha256"] = finalizer._digest(body)

    expected_lengths = {
        "complete": 11, "replay_shortage": 11,
        "fidelity_stop": 3, "insufficient_source_stop": 3,
        "infrastructure_stop": 0, "infrastructure_stop_failure": 0,
    }
    for outcome, length in expected_lengths.items():
        handoff = _handoff(finalizer)
        handoff["terminal_outcome"] = outcome
        handoff["evidence"] = handoff["evidence"][:length]
        handoff["predecessor_receipt_sha256"] = handoff["evidence"][-1]["receipt_sha256"] if length else None
        handoff["first_100_receipt_sha256"] = "c" * 64 if length >= 3 else None
        resign(handoff)
        finalizer.validate_handoff(handoff)

    for field in ("receipt_sha256", "file_sha256"):
        handoff = _handoff(finalizer)
        handoff["evidence"][0][field] = "bad"
        resign(handoff)
        with __import__("pytest").raises(finalizer.FinalizationError):
            finalizer.validate_handoff(handoff)


def test_v2_provisional_readbacks_are_independent_and_pinned(tmp_path: Path) -> None:
    """A public read cannot overwrite/authenticate the private readback tree."""
    finalizer = _module()
    spec = importlib.util.spec_from_file_location("publisher_for_v2_readback", ROOT / "scripts/publish_simple_curriculum_collection.py")
    assert spec and spec.loader
    publisher = importlib.util.module_from_spec(spec); sys.modules[spec.name] = publisher; spec.loader.exec_module(publisher)
    run_id = "fresh-run-20260828-v2readback"; round_id = "fresh-12k-20260828-v2readback"
    files = (
        "manifests/provisional/evidence-manifest.json", "manifests/provisional/task6-validation.json",
        "manifests/provisional/hub-artifact-references.json",
    )
    task = {"schema_version": 1, "kind": "lehome_simple_curriculum_task6_validation_v1", "terminal_outcome": "infrastructure_stop", "result": "not_complete", "fresh_reference_count": 0, "replay_accepted_reference_count": 0}
    refs = {"schema_version": 1, "kind": "lehome_simple_curriculum_hub_artifact_references_v1", "fresh": [], "success_replay": []}
    manifest = {"schema_version": 1, "kind": "lehome_simple_curriculum_provisional_evidence_manifest_v1", "run_id": run_id, "round_id": round_id, "instance_id": finalizer.EXACT_INSTANCE_ID, "campaign_root": f"/mnt/lehome/eval/{run_id}", "terminal_outcome": "infrastructure_stop", "reachable_stages": [], "stage_receipts": [], "terminal_chain_head": None, "first_100_receipt_sha256": None, "task6_validation_sha256": publisher.hashlib.sha256(publisher._canonical(task)).hexdigest(), "hub_artifact_references_sha256": publisher.hashlib.sha256(publisher._canonical(refs)).hexdigest(), "completion_claim": "none", "gpu_stop_verified": False}
    payloads = {files[0]: publisher._canonical(manifest), files[1]: publisher._canonical(task), files[2]: publisher._canonical(refs)}
    for relative, payload in payloads.items():
        target = tmp_path / relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(payload)
    entries = publisher._collect_entries(publisher.CollectionPublicationBundle(tmp_path, run_id, "ryanjin333/lehome-groot-n17-rollouts", "main", files))
    provisional = {"immutable_revision": "a" * 40, "bundle_sha256": publisher._entry_digest(entries), "manifest_sha256": publisher.hashlib.sha256(payloads[files[0]]).hexdigest(), "receipt_sha256": "b" * 64, "repository": "ryanjin333/lehome-groot-n17-rollouts", "remote_prefix": f"collection-rounds/{run_id}/manifests/provisional"}
    handoff = {"schema_version": 2, "kind": "lehome_simple_curriculum_operator_stop_handoff_v2", "run_id": run_id, "round_id": round_id, "instance_id": finalizer.EXACT_INSTANCE_ID, "terminal_outcome": "infrastructure_stop", "predecessor_receipt_sha256": None, "code_revision": "c" * 40, "code_tree_sha256": "d" * 64, "runtime_identity": {"mode": "test"}, "runtime_identity_sha256": finalizer._digest({"mode": "test"}), "first_100_receipt_sha256": None, "evidence": [], "provisional_publication": provisional}
    handoff["handoff_sha256"] = finalizer._digest(handoff)
    class Transport:
        destinations = []
        def list_tree(self, *, remote_prefix, **_kwargs): return tuple(SimpleNamespace(relative_path=f"{remote_prefix}/{name}", entry_type="file") for name in files)
        def download_files(self, *, destination, relative_paths, token, **_kwargs):
            self.destinations.append(destination)
            for relative in relative_paths:
                target = destination / relative; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payloads[relative] if token is not None else payloads[relative])
            return "a" * 40
    transport = Transport()
    finalizer.HfFinalizerPublisher(tmp_path / "token", module=publisher, transport=transport)._verify_provisional(module=publisher, transport=transport, root=tmp_path, token="token", handoff=handoff)
    assert len({str(path) for path in transport.destinations}) == 2


def test_stop_times_out_after_exact_id_validation() -> None:
    finalizer = _module()
    class Provider:
        calls = 0
        def get(self, instance_id):
            self.calls += 1
            return _instance(finalizer, "RUNNING" if self.calls == 1 else "STOPPING")
        def stop(self, instance_id): return None
    with __import__("pytest").raises(finalizer.FinalizationError, match="STOPPED"):
        finalizer.stop_exact_instance(Provider(), timeout_seconds=0.0001)


def test_finalizer_rejects_missing_protected_disk_before_stop_dispatch() -> None:
    finalizer = _module(); calls: list[str] = []
    class Provider:
        def get(self, instance_id): calls.append("get"); return _instance(finalizer, "RUNNING", disks=[])
        def stop(self, instance_id): calls.append("stop")
    with __import__("pytest").raises(finalizer.FinalizationError):
        finalizer.stop_exact_instance(Provider(), timeout_seconds=1)
    assert calls == ["get"]


def test_provider_validator_accepts_only_real_nested_nebius_identity_shape() -> None:
    finalizer = _module()
    raw = json.loads((ROOT / "infrastructure/nebius/.tools/cpu-only-v7-vm-before-start.json").read_text())
    assert finalizer._validate_instance(raw)["state"] == "STOPPED"
    raw["spec"]["secondary_disks"].append({"existing_disk": {"id": "computedisk-other"}})
    with __import__("pytest").raises(finalizer.FinalizationError): finalizer._validate_instance(raw)
    raw["spec"]["secondary_disks"] = [{"managed_disk": {"name": "extra"}}]
    with __import__("pytest").raises(finalizer.FinalizationError): finalizer._validate_instance(raw)


def test_provider_validator_rejects_legacy_flat_shape_and_stop_polls_stopping() -> None:
    finalizer = _module()
    with __import__("pytest").raises(finalizer.FinalizationError):
        finalizer._validate_instance({"id": finalizer.EXACT_INSTANCE_ID, "name": finalizer.EXACT_INSTANCE_NAME, "state": "STOPPED", "disks": [finalizer.PROTECTED_DISK_ID]})

    class Provider:
        states = iter(("RUNNING", "STOPPING", "STOPPED"))
        def get(self, instance_id): return _instance(finalizer, next(self.states))
        def stop(self, instance_id): return None

    assert finalizer.stop_exact_instance(Provider(), timeout_seconds=1)["state"] == "STOPPED"
    for unsafe in ("STARTING", "RESTARTING"):
        with __import__("pytest").raises(finalizer.FinalizationError):
            finalizer._validate_instance(_instance(finalizer, unsafe))


def test_emergency_cli_stops_exact_vm_without_any_operator_metadata(monkeypatch) -> None:
    finalizer = _module(); stopped: list[object] = []
    monkeypatch.setattr(finalizer, "stop_exact_instance", lambda provider, **_kwargs: stopped.append(provider) or {})
    assert finalizer.main(["--emergency-stop-only"]) == 0
    assert len(stopped) == 1


def test_hf_finalizer_reconciles_lost_receipt_response_without_second_upload(tmp_path: Path) -> None:
    finalizer = _module(); calls: list[object] = []
    class Entry:
        def __init__(self, relative_path: str) -> None:
            self.relative_path = relative_path
            self.sha256 = "a" * 64
            self.byte_size = 1
    handoff = _handoff(finalizer); stopped = {"observation_sha256": "e" * 64}; seal = {"seal_sha256": "f" * 64}
    evidence_files = ("reports/operator-stop-handoff.json", "reports/stopped-observation.json", "seals/final-seal.json")
    receipt = _final_receipt(
        finalizer, handoff, seal, revision="b" * 40,
        bundle=finalizer._entry_digest(tuple(Entry(name) for name in evidence_files)),
    )
    token = tmp_path / "token"; token.write_text("token", encoding="utf-8"); token.chmod(0o600)
    class Transport:
        def resolve_approved_ref(self, **_kwargs): return "a" * 40
        def list_tree(self, *, revision, **_kwargs):
            return evidence_files if revision == "b" * 40 else evidence_files + ("reports/final-publication.json",)
        def upload_files(self, **_kwargs): raise AssertionError("lost response retry must not upload again")
        def download_files(self, *, destination, relative_paths, **_kwargs):
            for relative in relative_paths:
                path = destination / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
            return "a" * 40
    module = SimpleNamespace(
        _load_token=lambda _path: "token",
        CollectionPublicationBundle=lambda **kwargs: SimpleNamespace(**kwargs),
        _collect_entries=lambda bundle: tuple(Entry(name) for name in bundle.files),
        _tree_files=lambda entries, **_kwargs: set(entries),
        _verify_download=lambda **kwargs: calls.append(kwargs["token"]),
        HuggingFacePublicDatasetTransport=lambda: Transport(),
    )
    result = finalizer.HfFinalizerPublisher(token, module=module, transport=Transport()).publish(tmp_path, handoff=handoff, stop_observation=stopped, seal=seal)
    assert result["immutable_revision"] == "a" * 40
    assert calls == ["token", None, "token", None]


def test_finalization_receipt_requires_exact_hash_and_binds_evidence(tmp_path: Path) -> None:
    finalizer = _module(); handoff = _handoff(finalizer); seal = {"seal_sha256": "f" * 64}
    good = _final_receipt(finalizer, handoff, seal)
    finalizer.validate_finalization_receipt(
        good, handoff=handoff, evidence_revision="a" * 40,
        evidence_bundle_sha256="b" * 64, seal=seal,
    )
    stale = dict(good); stale["round_id"] = "fresh-12k-stale"
    with __import__("pytest").raises(finalizer.FinalizationError, match="receipt"):
        finalizer.validate_finalization_receipt(
            stale, handoff=handoff, evidence_revision="a" * 40,
            evidence_bundle_sha256="b" * 64, seal=seal,
        )


def test_hf_lost_response_reconcile_stages_real_receipt_for_actual_collector(tmp_path: Path) -> None:
    finalizer = _module(); source = ROOT / "scripts/publish_simple_curriculum_collection.py"
    spec = importlib.util.spec_from_file_location("real_publisher_for_finalizer", source); assert spec and spec.loader
    publisher_module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = publisher_module; spec.loader.exec_module(publisher_module)
    token = tmp_path / "token"; token.write_text("hf_fake_token", encoding="utf-8"); token.chmod(0o600)
    handoff = _handoff(finalizer); stopped = {"observation_sha256": "e" * 64}; seal = {"seal_sha256": "f" * 64}; calls: list[object] = []; remote_bytes: dict[str, bytes] = {}; uploads: list[object] = []
    receipt: dict[str, object] = {}
    class Transport:
        def resolve_approved_ref(self, **_kwargs): return "a" * 40
        def list_tree(self, *, remote_prefix, revision, **_kwargs):
            nonlocal receipt
            if not remote_bytes:
                for path in ("reports/operator-stop-handoff.json", "reports/stopped-observation.json", "seals/final-seal.json"):
                    remote_bytes[path] = (tmp_path / path).read_bytes()
                evidence_entries = publisher_module._collect_entries(publisher_module.CollectionPublicationBundle(
                    root=tmp_path, run_id=handoff["run_id"], repository="ryanjin333/lehome-groot-n17-rollouts",
                    revision="main", files=("reports/operator-stop-handoff.json", "reports/stopped-observation.json", "seals/final-seal.json"),
                ))
                receipt = _final_receipt(
                    finalizer, handoff, seal, revision="b" * 40,
                    bundle=publisher_module._entry_digest(evidence_entries),
                )
                remote_bytes["reports/final-publication.json"] = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
            names = ("reports/operator-stop-handoff.json", "reports/stopped-observation.json", "seals/final-seal.json")
            if revision != "b" * 40:
                names += ("reports/final-publication.json",)
            return tuple(SimpleNamespace(relative_path=f"{remote_prefix}/{path}", entry_type="file") for path in names)
        def upload_files(self, **kwargs): uploads.append(kwargs); raise AssertionError("exact receipt prefix must reconcile without upload")
        def download_files(self, *, destination, relative_paths, token, revision, **_kwargs):
            calls.append(token)
            for relative in relative_paths:
                target = destination / relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(remote_bytes[relative])
            return revision
    result = finalizer.HfFinalizerPublisher(token, module=publisher_module, transport=Transport()).publish(
        tmp_path, handoff=handoff, stop_observation=stopped, seal=seal
    )
    assert result["immutable_revision"] == "a" * 40
    assert uploads == []
    # One receipt staging download, then authenticated + anonymous readback of
    # the actual descriptor-collected four-file bundle.
    assert calls == ["hf_fake_token", "hf_fake_token", None, "hf_fake_token", None]


def test_hf_finalizer_rejects_malformed_or_mismatched_existing_receipt(tmp_path: Path) -> None:
    finalizer = _module(); handoff = _handoff(finalizer); stopped = {"observation_sha256": "e" * 64}; seal = {"seal_sha256": "f" * 64}
    token = tmp_path / "token"; token.write_text("token", encoding="utf-8"); token.chmod(0o600)
    bad_receipts = ({}, {**_final_receipt(finalizer, handoff, seal), "run_id": "fresh-run-stale"})
    for bad in bad_receipts:
        class Transport:
            def resolve_approved_ref(self, **_kwargs): return "a" * 40
            def list_tree(self, **_kwargs): return ("reports/operator-stop-handoff.json", "reports/stopped-observation.json", "seals/final-seal.json", "reports/final-publication.json")
            def upload_files(self, **_kwargs): raise AssertionError("malformed existing receipt must fail closed")
            def download_files(self, *, destination, relative_paths, **_kwargs):
                for relative in relative_paths:
                    path = destination / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(bad), encoding="utf-8")
        module = SimpleNamespace(
            _load_token=lambda _path: "token", CollectionPublicationBundle=lambda **kwargs: SimpleNamespace(**kwargs),
            _tree_files=lambda entries, **_kwargs: set(entries), _collect_entries=lambda _bundle: (_ for _ in ()).throw(AssertionError("receipt must validate before collection")),
            _verify_download=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("receipt must validate before readback")),
        )
        with __import__("pytest").raises(finalizer.FinalizationError, match="receipt"):
            finalizer.HfFinalizerPublisher(token, module=module, transport=Transport()).publish(tmp_path / str(len(str(bad))), handoff=handoff, stop_observation=stopped, seal=seal)
