"""Regression contract for the remote terminal handoff boundary."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
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
