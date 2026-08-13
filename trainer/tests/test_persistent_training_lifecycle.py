from pathlib import Path
import hashlib
import json

import pytest

import importlib.util
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("persistent_training_lifecycle_under_test", REPOSITORY / "scripts" / "run_groot_persistent_training.py")
assert SPEC is not None and SPEC.loader is not None
LIFECYCLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LIFECYCLE
SPEC.loader.exec_module(LIFECYCLE)


class FailRunner:
    def __call__(self, _command: tuple[str, ...]) -> str:
        raise AssertionError("provider must not be called")


class FakeHub:
    def list_tree(self, *, repository, revision, token):
        from lehome_train.hub import HubTreeEntry
        return (HubTreeEntry("prefix/checkpoints/step-1000.tar", "file"), HubTreeEntry("prefix/checkpoints/step-2000.tar", "file"))
    def download_files(self, *, repository, revision, destination, relative_paths, token, remote_prefix=None):
        for relative in relative_paths:
            target = destination / relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b"artifact")
        return revision


def test_dry_run_never_calls_provider(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text('{"generation_sha256":"' + "a" * 64 + '","config_sha256":"' + "b" * 64 + '"}')
    report = LIFECYCLE.main_for_test(["status", "--request", str(request)], runner=FailRunner())
    assert report["paid_action"] is False


def test_destroy_requires_two_immutable_checkpoints_bound_to_instance(tmp_path: Path) -> None:
    receipt = {"kind": "continuous_corrective_training_terminal", "instance_id": 7, "immutable_checkpoint_steps": [1000]}
    with pytest.raises(ValueError, match="instance-bound disposal"):
        LIFECYCLE.destroy(instance_id=7, training_receipt=receipt)


def test_capture_rent_and_destroy_use_injected_cli_and_fresh_readback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[tuple[str, ...]] = []
    def runner(command: tuple[str, ...]) -> str:
        commands.append(command)
        if command[:4] == ("vastai", "--raw", "search", "offers"):
            return '[{"id":7,"gpu_name":"RTX PRO 6000 WS","num_gpus":1,"gpu_ram":96000,"dph_total":0.7,"min_bid":0.5,"is_bid":false,"driver_version":"595.71.05","untrusted_token":"never-persist"}]'
        if command[:4] == ("vastai", "--raw", "show", "instances"): return "[]"
        if command[:4] == ("vastai", "--raw", "show", "volumes"): return "[]"
        if command[:4] == ("vastai", "--raw", "create", "instance"): return '{"new_contract":9}'
        if command[:4] == ("vastai", "--raw", "show", "instance"): return '{"id":9,"gpu_name":"RTX PRO 6000 WS","num_gpus":1,"gpu_ram":96000,"dph_total":0.7,"ssh_host":"host","ssh_port":22,"driver_version":"595.71.05"}'
        if command[:3] == ("vastai", "destroy", "instance"): return ""
        raise AssertionError(command)
    monkeypatch.setattr(LIFECYCLE.time, "time", lambda: 100)
    image = "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:" + "a" * 64
    evidence = LIFECYCLE.capture_offers(runner=runner, now_unix=100) | {"trainer_image": image, "training_capability": {"hardware": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "driver_version": "595.71.05", "image_digest": image.rpartition("@")[2], "cuda_runtime": "12.8", "torch_cuda": "12.8", "compute_capability": "12.0", "optimizer_step": {"passed": True, "loss": .2}, "nvml": {"utilization_percent": 80}}}
    assert evidence["offer"]["gpu_name"] == "RTX PRO 6000 WS"
    assert "untrusted_token" not in evidence["offer"]
    instance = LIFECYCLE.rent(evidence=evidence, runner=runner)
    assert instance["instance_id"] == 9
    with pytest.raises(ValueError, match="destroy absence"):
        LIFECYCLE.destroy(instance_id=9, training_receipt={"kind": "continuous_corrective_training_terminal", "instance_id": 9, "immutable_checkpoint_steps": [1000, 2000], "immutable_checkpoint_publications": [{"optimizer_step": 1000, "repository": "ryanjin333/lehome-groot-n17-models", "immutable_revision": "a" * 40, "remote_prefix": "prefix", "relative_path": "checkpoints/step-1000.tar", "artifact_sha256": hashlib.sha256(b"artifact").hexdigest(), "artifact_byte_size": 8, "readback_verified": True}, {"optimizer_step": 2000, "repository": "ryanjin333/lehome-groot-n17-models", "immutable_revision": "b" * 40, "remote_prefix": "prefix", "relative_path": "checkpoints/step-2000.tar", "artifact_sha256": hashlib.sha256(b"artifact").hexdigest(), "artifact_byte_size": 8, "readback_verified": True}]}, runner=runner, transport=FakeHub(), token="test-token")


def test_rent_requires_capability_receipt_for_exact_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LIFECYCLE.time, "time", lambda: 100)
    evidence = {"offer": {"id": 7, "min_bid": .5}, "search_mode": "interruptible", "expires_at_unix": 101, "trainer_image": "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:" + "a" * 64}
    with pytest.raises(ValueError, match="capability"):
        LIFECYCLE.rent(evidence=evidence, runner=lambda _: "{}")


def test_status_parses_authenticated_terminal() -> None:
    instance = {"instance_id": 7, "host": "host", "port": 22}
    terminal = {"kind": "continuous_corrective_training_terminal", "instance_id": 7}
    result = LIFECYCLE.remote_action(action="status", instance=instance, request={}, runner=lambda _command: json.dumps(terminal))
    assert result["terminal"] == terminal


def test_materialize_builds_a_verified_sealed_generation(tmp_path: Path) -> None:
    # The lifecycle's free preparation action must exercise the same canonical
    # mix/materialization implementation as production, rather than accepting
    # a hand-written receipt.
    from test_flywheel_mix import _prepared_source

    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=2)
    corrective_a = _prepared_source(tmp_path / "corrective-a", kind="flywheel", grade="A", episodes=1)
    corrective_b = _prepared_source(tmp_path / "corrective-b", kind="flywheel", grade="B", episodes=1)
    destination = tmp_path / "generation"
    request = tmp_path / "materialize.json"
    request.write_text(json.dumps({
        "organizer_root": str(organizer),
        "corrective_roots": [str(corrective_a), str(corrective_b)],
        "destination": str(destination),
        "seed": 20260812,
    }))

    report = LIFECYCLE.main_for_test(["materialize", "--request", str(request)])

    assert report["paid_action"] is False
    assert report["generation_root"] == str(destination)
    assert (destination.with_name(destination.name + ".generation.json")).is_file()
