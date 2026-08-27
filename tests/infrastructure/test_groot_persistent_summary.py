from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "summarize_groot_persistent_evaluation.py"


def _module():
    spec = importlib.util.spec_from_file_location("persistent_summary_first100", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_report_preserves_legacy_fields_and_adds_first_hundred_gate_metrics(tmp_path: Path) -> None:
    module = _module()
    report = module._augment_first_hundred_metrics({
        "episodes": 100,
        "official_successes": 5,
        "trials": [
            {
                "attempt_id": f"attempt-{index}",
                "terminal_event": "accepted" if index < 5 else "rejected",
                "identity": {"code_revision": "c" * 40, "asset_revision": "a" * 40, "simulator_version": "5.1.0.0"},
                "provenance": {
                    "policy_repo": "owner/policy", "policy_revision": "e" * 40,
                    "policy_step": 12000, "policy_artifact_sha256": "b" * 64,
                    "image_identity": "sha256:" + "d" * 64,
                    "simulator_device": "cpu", "cloth_device": "cpu",
                    "renderer_device": "cuda:0", "camera_device": "cuda:0", "policy_device": "cuda:0",
                },
                "fidelity": {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False},
            }
            for index in range(100)
        ],
        "infrastructure_invalid_executions": 2,
    })

    assert report["episodes"] == 100
    assert report["official_successes"] == 5
    assert report["valid_outcomes"] == 100
    assert report["infrastructure_invalid_executions"] == 2
    assert report["execution_count"] == 102
    assert report["fresh_assignment_ids"] == sorted(f"attempt-{index}" for index in range(100))
    assert len(report["runtime_identities"]) == 1
    assert len(report["runtime_identities"][0]) == 64
