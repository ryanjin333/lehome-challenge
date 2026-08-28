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


def test_paid_terminal_handoff_requires_a_pinned_provisional_bundle_before_stop(tmp_path: Path) -> None:
    """The controller may hand off only evidence it has durably staged while running."""
    module = _module()
    run_id = "fresh-run-20260828-provisional"
    root = tmp_path / run_id; root.mkdir()
    config = _config(module, root, run_id=run_id)
    with __import__("pytest").raises(module.ReceiptMismatchError, match="provisional"):
        module._provisional_receipt(config)


def test_paid_handoff_rejects_an_unsafe_campaign_root_before_any_staging(tmp_path: Path) -> None:
    module = _module()
    config = _config(module, tmp_path / "not-the-run-id", run_id="fresh-run-20260828-root")
    with __import__("pytest").raises(module.ReceiptMismatchError, match="campaign root"):
        module.require_operator_campaign_root(config)


def _module():
    spec = importlib.util.spec_from_file_location("terminal_handoff_controller", ROOT / "scripts" / "run_simple_curriculum_collection.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def _config(module, root: Path, *, run_id: str):
    return module.CollectionConfig(
        root, ROOT, run_id, "fresh-12k-20260828-provisional", 3600.0, 99.0, True, None,
        {**module._ORIGINAL_12K, "rollout_image": "repo/r@sha256:" + "a" * 64,
         "trainer_image": "repo/t@sha256:" + "b" * 64}, root / "spend.json",
    )


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
    assert "trap finish EXIT" in wrapper and "finalize_simple_curriculum_collection.py" in wrapper
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
    record = f"LEHOME_REVIEWED_REVISION={revision}\nLEHOME_CAMPAIGN_ROOT={campaign}\nLEHOME_RUN_ID={run_id}\nLEHOME_ROUND_ID={round_id}\nLEHOME_SPEND_BASELINE_USD=20.25\nLEHOME_SPEND_BASELINE_AT_UTC=2026-08-28T14:25:00Z\nLEHOME_MAX_HOURLY_BURN_USD=1.50\nLEHOME_SPEND_OBSERVER_COMMAND=/mnt/lehome/runtime-code/{revision}/scripts/run_conservative_spend_observer.py\n"
    (remote / "operator/simple-curriculum-invocation.env").write_text(record, encoding="utf-8")
    log = tmp_path / "log"
    def executable(name: str, text: str) -> None:
        path = fake_bin / name; path.write_text("#!/bin/sh\n" + text, encoding="utf-8"); path.chmod(0o755)
    executable("stat", "case \"$1:$2\" in -c:%u) /usr/bin/id -u;; -c:*) echo 600;; -f:%u) /usr/bin/id -u;; -f:%Lp) echo 600;; *) echo 600;; esac")
    executable("uv", f"echo finalizer >> {log}; exit 0")
    executable("sudo", "exec \"$@\"")
    executable("ssh", "target=\"$REMOTE_ROOT\"; while [ \"$1\" != sh ]; do shift; done; shift; shift; shift; sed \"s|/mnt/lehome|$target|g\" | /bin/sh -s -- \"$@\"")
    (remote / "runtime-code" / revision / "rollout_appliance/run_simple_curriculum_collection.sh").write_text(f"#!/bin/sh\necho controller >> {log}\n", encoding="utf-8")
    (remote / "runtime-code" / revision / "rollout_appliance/run_simple_curriculum_collection.sh").chmod(0o755)
    token = tmp_path / "token"; token.write_text("not-printed", encoding="utf-8")
    result = subprocess.run(("env", "-i", f"PATH={fake_bin}:/usr/bin:/bin", "REMOTE_ROOT=" + str(remote), "LOG=" + str(log), "LEHOME_OPERATOR_SSH_TARGET=operator@host", "LEHOME_OPERATOR_CAMPAIGN_ROOT=" + campaign, "LEHOME_OPERATOR_RUN_ID=" + run_id, "LEHOME_OPERATOR_ROUND_ID=" + round_id, "LEHOME_OPERATOR_REVIEWED_REVISION=" + revision, "LEHOME_OPERATOR_HF_TOKEN_FILE=" + str(token), "/bin/bash", str(ROOT / "scripts/run_simple_curriculum_with_finalizer.sh")), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").splitlines() == ["controller", "finalizer"]
    log.unlink()
    (remote / "operator/simple-curriculum-invocation.env").write_text(record.replace(revision, "b" * 40), encoding="utf-8")
    mismatch = subprocess.run(("env", "-i", f"PATH={fake_bin}:/usr/bin:/bin", "REMOTE_ROOT=" + str(remote), "LOG=" + str(log), "LEHOME_OPERATOR_SSH_TARGET=operator@host", "LEHOME_OPERATOR_CAMPAIGN_ROOT=" + campaign, "LEHOME_OPERATOR_RUN_ID=" + run_id, "LEHOME_OPERATOR_ROUND_ID=" + round_id, "LEHOME_OPERATOR_REVIEWED_REVISION=" + revision, "LEHOME_OPERATOR_HF_TOKEN_FILE=" + str(token), "/bin/bash", str(ROOT / "scripts/run_simple_curriculum_with_finalizer.sh")), text=True, capture_output=True)
    assert "controller" not in (log.read_text(encoding="utf-8") if log.exists() else "")
    assert "finalizer" in (log.read_text(encoding="utf-8") if log.exists() else "")


def test_operator_wrapper_runs_one_finalizer_and_aggregates_all_exit_statuses(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir(); log = tmp_path / "log"
    def executable(name: str, text: str) -> None:
        path = fake_bin / name; path.write_text("#!/bin/sh\n" + text, encoding="utf-8"); path.chmod(0o755)
    executable("stat", "case \"$1:$2\" in -f:%u) /usr/bin/id -u;; -f:%Lp) echo 600;; *) exit 9;; esac")
    executable("uv", f"echo finalizer >> {log}; exit \"${{FINALIZER_STATUS:-0}}\"")
    executable("ssh", f"cat >/dev/null; echo controller >> {log}; exit \"${{CONTROLLER_STATUS:-0}}\"")
    token = tmp_path / "token"; token.write_text("not-printed", encoding="utf-8")
    revision = "a" * 40; run_id = "fresh-run-20260828123456-01"; round_id = "fresh-12k-20260828123456-01"
    campaign = f"/mnt/lehome/eval/{run_id}"
    base = (
        "env", "-i", f"PATH={fake_bin}:/usr/bin:/bin", "LEHOME_OPERATOR_SSH_TARGET=operator@host",
        "LEHOME_OPERATOR_CAMPAIGN_ROOT=" + campaign, "LEHOME_OPERATOR_RUN_ID=" + run_id,
        "LEHOME_OPERATOR_ROUND_ID=" + round_id, "LEHOME_OPERATOR_REVIEWED_REVISION=" + revision,
        "LEHOME_OPERATOR_HF_TOKEN_FILE=" + str(token),
    )
    for controller_status, finalizer_status, expected in ((0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)):
        if log.exists(): log.unlink()
        result = subprocess.run(
            base + (f"CONTROLLER_STATUS={controller_status}", f"FINALIZER_STATUS={finalizer_status}", "/bin/bash", str(ROOT / "scripts/run_simple_curriculum_with_finalizer.sh")),
            text=True, capture_output=True,
        )
        assert result.returncode == expected, result.stderr
        assert log.read_text(encoding="utf-8").splitlines() == ["controller", "finalizer"]


def test_operator_wrapper_emergency_finalizes_every_precontroller_rejection(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir(); log = tmp_path / "log"
    def executable(name: str, text: str) -> None:
        path = fake_bin / name; path.write_text("#!/bin/sh\n" + text, encoding="utf-8"); path.chmod(0o755)
    executable("stat", "case \"$1:$2\" in -f:%u) if [ \"${TOKEN_OWNER:-self}\" = self ]; then /usr/bin/id -u; else echo 99999; fi;; -f:%Lp) echo \"${TOKEN_MODE:-600}\";; *) exit 9;; esac")
    executable("uv", f"echo \"$*\" >> {log}; exit 0")
    executable("ssh", "echo controller-must-not-run >&2; exit 97")
    token = tmp_path / "token"; token.write_text("token", encoding="utf-8")
    link = tmp_path / "token-link"; link.symlink_to(token)
    revision = "a" * 40; run_id = "fresh-run-20260828123456-01"; round_id = "fresh-12k-20260828123456-01"
    campaign = f"/mnt/lehome/eval/{run_id}"
    base = {
        "LEHOME_OPERATOR_SSH_TARGET": "operator@host", "LEHOME_OPERATOR_SSH_PORT": "22",
        "LEHOME_OPERATOR_CAMPAIGN_ROOT": campaign, "LEHOME_OPERATOR_RUN_ID": run_id,
        "LEHOME_OPERATOR_ROUND_ID": round_id, "LEHOME_OPERATOR_REVIEWED_REVISION": revision,
        "LEHOME_OPERATOR_HF_TOKEN_FILE": str(token),
    }
    invalid_cases = (
        {"LEHOME_OPERATOR_SSH_TARGET": ""},
        {"LEHOME_OPERATOR_RUN_ID": "bad"},
        {"LEHOME_OPERATOR_ROUND_ID": "bad"},
        {"LEHOME_OPERATOR_REVIEWED_REVISION": "bad"},
        {"LEHOME_OPERATOR_CAMPAIGN_ROOT": "/tmp/bad"},
        {"LEHOME_OPERATOR_SSH_TARGET": "-Ffile@host"},
        {"LEHOME_OPERATOR_SSH_PORT": "70000"},
        {"LEHOME_OPERATOR_HF_TOKEN_FILE": str(tmp_path / "missing")},
        {"LEHOME_OPERATOR_HF_TOKEN_FILE": str(link)},
        {"TOKEN_OWNER": "other"},
        {"TOKEN_MODE": "644"},
        {"LEHOME_OPERATOR_HF_TOKEN_FILE": str(tmp_path / "empty")},
        {"LEHOME_OPERATOR_STOP_TIMEOUT_SECONDS": "zero"},
    )
    (tmp_path / "empty").write_text("", encoding="utf-8")
    for invalid in invalid_cases:
        if log.exists(): log.unlink()
        environment = {**base, **invalid}
        result = subprocess.run(
            ("env", "-i", f"PATH={fake_bin}:/usr/bin:/bin", *(f"{key}={value}" for key, value in environment.items()), "/bin/bash", str(ROOT / "scripts/run_simple_curriculum_with_finalizer.sh")),
            text=True, capture_output=True,
        )
        assert result.returncode != 0, (invalid, result.stderr)
        calls = log.read_text(encoding="utf-8").splitlines()
        assert len(calls) == 1 and "--emergency-stop-only" in calls[0], (invalid, calls)
