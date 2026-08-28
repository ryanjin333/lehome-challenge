"""Regression contract for the remote terminal handoff boundary."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location("terminal_handoff_controller", ROOT / "scripts" / "run_simple_curriculum_collection.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def test_paid_terminal_writes_handoff_without_self_stop_or_publication(tmp_path: Path) -> None:
    controller = _module()
    root = tmp_path / "campaign"; (root / "inputs").mkdir(parents=True)
    (root / "inputs" / "seen-catalog.json").write_text(json.dumps({
        "top_long": [f"Top_Long_Seen_{i}" for i in range(10)], "top_short": [f"Top_Short_Seen_{i}" for i in range(10)],
        "pant_long": [f"Pant_Long_Seen_{i}" for i in range(10)], "pant_short": [f"Pant_Short_Seen_{i}" for i in range(10)],
    }), encoding="utf-8")
    config = controller.CollectionConfig(
        root, ROOT, "fresh-run-20260828-handoff", "fresh-12k-20260828-handoff", 3600.0, 99.0, True, None,
        {**controller._ORIGINAL_12K, "rollout_image": "repo/r@sha256:" + "a" * 64, "trainer_image": "repo/t@sha256:" + "b" * 64},
        tmp_path / "spend.json",
    )
    config.spend_observer.write_text(json.dumps({"schema_version": 1, "kind": "lehome_spend_observation_v1", "observer": "test", "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"), "spent_usd": 0.0}), encoding="utf-8")

    class Runner:
        calls: list[str] = []
        def run(self, stage: str, **_kwargs):
            self.calls.append(stage)
            if stage == "calibration-matrix":
                raise RuntimeError("simulated infrastructure failure")
            raise AssertionError(stage)
        def stop_gpu(self, _command: str): raise AssertionError("remote controller must not self-stop")

    runner = Runner()
    assert controller.run_collection(config, runner=runner) == "operator_stop_required"
    handoff = json.loads((root / "reports" / "operator-stop-handoff.json").read_text(encoding="utf-8"))
    assert handoff["terminal_outcome"] == "infrastructure_stop_failure"
    assert handoff["instance_id"] == "computeinstance-u00t6xfqhadrcmssa2"
    assert runner.calls == ["calibration-matrix"]
    assert controller.run_collection(config, runner=runner) == "operator_stop_required"
    assert runner.calls == ["calibration-matrix"]


def test_operator_wrapper_traps_even_an_unpersistable_handoff_for_local_finalization(monkeypatch, tmp_path: Path) -> None:
    controller = _module()
    root = tmp_path / "campaign"; (root / "inputs").mkdir(parents=True)
    (root / "inputs" / "seen-catalog.json").write_text(json.dumps({name: [f"{name}-{i}" for i in range(10)] for name in ("top_long", "top_short", "pant_long", "pant_short")}), encoding="utf-8")
    config = controller.CollectionConfig(root, ROOT, "fresh-run-20260828-writefail", "fresh-12k-20260828-writefail", 3600.0, 99.0, True, None, {**controller._ORIGINAL_12K, "rollout_image": "repo/r@sha256:" + "a" * 64, "trainer_image": "repo/t@sha256:" + "b" * 64}, tmp_path / "spend.json")
    from datetime import UTC, datetime
    config.spend_observer.write_text(json.dumps({"schema_version": 1, "kind": "lehome_spend_observation_v1", "observer": "test", "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"), "spent_usd": 0.0}), encoding="utf-8")
    monkeypatch.setattr(controller, "_write_immutable_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk failure")))
    class Runner:
        calls = []
        def run(self, stage, **kwargs): self.calls.append(stage); raise RuntimeError("prep failure")
        def stop_gpu(self, command): raise AssertionError("remote self-stop")
    with __import__("pytest").raises(OSError, match="disk failure"):
        controller.run_collection(config, runner=Runner())
    wrapper = (ROOT / "scripts/run_simple_curriculum_with_finalizer.sh").read_text(encoding="utf-8")
    assert "trap finalize EXIT" in wrapper and "finalize_simple_curriculum_collection.py" in wrapper
    assert "LEHOME_REMOTE_CONTROLLER_COMMAND" not in wrapper
    assert "/mnt/lehome/operator/simple-curriculum-invocation.env" in wrapper
    assert "LEHOME_OPERATOR_REVIEWED_REVISION" in wrapper
    assert "LEHOME_PAID_COLLECTION=1" in wrapper


def test_command_runner_preparation_failure_writes_handoff_and_restart_is_inert(monkeypatch, tmp_path: Path) -> None:
    controller = _module()
    root = tmp_path / "campaign"; root.mkdir()
    config = controller.CollectionConfig(root, ROOT, "fresh-run-20260828-prepfail", "fresh-12k-20260828-prepfail", 3600.0, 99.0, True, None, {**controller._ORIGINAL_12K, "rollout_image": "repo/r@sha256:" + "a" * 64, "trainer_image": "repo/t@sha256:" + "b" * 64}, tmp_path / "spend.json")
    from datetime import UTC, datetime
    config.spend_observer.write_text(json.dumps({"schema_version": 1, "kind": "lehome_spend_observation_v1", "observer": "test", "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"), "spent_usd": 0.0}), encoding="utf-8")
    monkeypatch.setattr(controller, "_prepare_controller_inputs", lambda _config: (_ for _ in ()).throw(RuntimeError("preparation failed")))
    runner = controller.CommandRunner(config)
    assert controller.run_collection(config, runner=runner) == "operator_stop_required"
    handoff = json.loads((root / "reports/operator-stop-handoff.json").read_text())
    assert handoff["terminal_outcome"] == "infrastructure_stop_failure"
    assert controller.run_collection(config, runner=runner) == "operator_stop_required"


def test_operator_wrapper_executes_fixed_remote_argv_over_stdin_with_clean_environment(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir(); remote = tmp_path / "remote"
    revision = "a" * 40; run_id = "fresh-run-20260828123456-01"; round_id = "fresh-12k-20260828123456-01"; campaign = f"/mnt/lehome/eval/{run_id}"
    (remote / "operator").mkdir(parents=True); (remote / "runtime-code" / revision / "rollout_appliance").mkdir(parents=True)
    (remote / "operator/simple-curriculum-invocation.env").write_text(f"LEHOME_REVIEWED_REVISION={revision}\nLEHOME_RUN_ID={run_id}\nLEHOME_ROUND_ID={round_id}\nLEHOME_CAMPAIGN_ROOT={campaign}\n", encoding="utf-8")
    log = tmp_path / "log"
    def executable(name: str, text: str) -> None:
        path = fake_bin / name; path.write_text("#!/bin/sh\n" + text, encoding="utf-8"); path.chmod(0o755)
    executable("stat", "echo '0 600'")
    executable("uv", f"echo finalizer >> {log}; exit 0")
    executable("sudo", "exec \"$@\"")
    executable("ssh", "target=\"$REMOTE_ROOT\"; while [ \"$1\" != sh ]; do shift; done; shift; shift; shift; sed \"s|/mnt/lehome|$target|g\" | /bin/sh -s -- \"$@\"")
    (remote / "runtime-code" / revision / "rollout_appliance/run_simple_curriculum_collection.sh").write_text(f"#!/bin/sh\necho controller >> {log}\n", encoding="utf-8")
    (remote / "runtime-code" / revision / "rollout_appliance/run_simple_curriculum_collection.sh").chmod(0o755)
    token = tmp_path / "token"; token.write_text("not-printed", encoding="utf-8")
    result = subprocess.run(("env", "-i", f"PATH={fake_bin}:/usr/bin:/bin", "REMOTE_ROOT=" + str(remote), "LOG=" + str(log), "LEHOME_OPERATOR_SSH_TARGET=fake", "LEHOME_OPERATOR_CAMPAIGN_ROOT=" + campaign, "LEHOME_OPERATOR_RUN_ID=" + run_id, "LEHOME_OPERATOR_ROUND_ID=" + round_id, "LEHOME_OPERATOR_REVIEWED_REVISION=" + revision, "LEHOME_OPERATOR_HF_TOKEN_FILE=" + str(token), "/bin/bash", str(ROOT / "scripts/run_simple_curriculum_with_finalizer.sh")), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").splitlines() == ["controller", "finalizer"]
    log.unlink()
    (remote / "operator/simple-curriculum-invocation.env").write_text(f"LEHOME_REVIEWED_REVISION={'b' * 40}\nLEHOME_RUN_ID={run_id}\nLEHOME_ROUND_ID={round_id}\nLEHOME_CAMPAIGN_ROOT={campaign}\n", encoding="utf-8")
    mismatch = subprocess.run(("env", "-i", f"PATH={fake_bin}:/usr/bin:/bin", "REMOTE_ROOT=" + str(remote), "LOG=" + str(log), "LEHOME_OPERATOR_SSH_TARGET=fake", "LEHOME_OPERATOR_CAMPAIGN_ROOT=" + campaign, "LEHOME_OPERATOR_RUN_ID=" + run_id, "LEHOME_OPERATOR_ROUND_ID=" + round_id, "LEHOME_OPERATOR_REVIEWED_REVISION=" + revision, "LEHOME_OPERATOR_HF_TOKEN_FILE=" + str(token), "/bin/bash", str(ROOT / "scripts/run_simple_curriculum_with_finalizer.sh")), text=True, capture_output=True)
    assert "controller" not in (log.read_text(encoding="utf-8") if log.exists() else "")
    assert "finalizer" in (log.read_text(encoding="utf-8") if log.exists() else "")
