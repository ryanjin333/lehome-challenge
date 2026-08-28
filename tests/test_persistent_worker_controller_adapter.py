from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "diagnostic",
    [
        None,
        {
            "stage": "post_stabilization",
            "physical_health": {
                "max_position_m": 1.7,
                "max_extent_m": 1.2,
                "max_velocity_mps": 4.9,
                "max_position_limit_m": 1.57,
                "max_extent_limit_m": 1.8,
                "max_velocity_limit_mps": 4.75,
                "exceeded_metrics": ["max_position_m", "max_velocity_mps"],
            },
        },
    ],
    ids=("legacy-call", "bounded-diagnostic"),
)
def test_ledger_worker_controller_forwards_optional_fidelity_diagnostic(
    tmp_path: Path, diagnostic: dict[str, object] | None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from lehome.flywheel.task_ledger import TaskLedger
    from scripts.run_groot_persistent_worker import LedgerWorkerController

    ledger = TaskLedger(
        tmp_path / "fidelity-controller.sqlite3",
        attempt_matrix=[{
            "attempt_id": "trial-a",
            "garment": "Top_Short_Seen_2",
            "seed": 2026082789,
        }],
        max_attempts=1,
        target_accepted=1,
        completion_metric="terminal_outcomes",
    )
    try:
        controller = LedgerWorkerController(ledger, lease_duration_ns=10**18)
        lease = controller.lease_next("worker-1")
        assert lease is not None
        abort_kwargs = {
            "session_id": "session-1",
            "generation": 3,
            "fidelity_code": "cloth_flight",
            "fidelity": {
                "missing_cloth": False,
                "cloth_flight": True,
                "nonfinite_cloth_state": False,
                "safety_failure": False,
                "monitor_active": True,
                "monitor_observed": True,
            },
            "runtime": {
                "simulation_device": "cpu",
                "cloth_device": "cpu",
                "renderer_device": "cuda:0",
                "camera_device": "cuda:0",
                "policy_device": "cuda:0",
            },
        }
        if diagnostic is not None:
            abort_kwargs["diagnostic"] = diagnostic

        assert controller.record_fidelity_abort(
            "worker-1", lease.attempt.attempt_id, lease.lease_id, **abort_kwargs,
        ) == "infrastructure_abort"

        abort = next(event for event in ledger.events() if event.event_type == "infrastructure_abort")
        campaign_end = next(event for event in ledger.events() if event.event_type == "campaign_ended")
        if diagnostic is None:
            assert "diagnostic" not in abort.payload
            assert "diagnostic" not in campaign_end.payload
        else:
            assert abort.payload["diagnostic"] == diagnostic
            assert campaign_end.payload["diagnostic"] == diagnostic
    finally:
        ledger.close()
