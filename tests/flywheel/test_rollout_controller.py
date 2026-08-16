from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_REPOSITORY_ROOT = Path(__file__).parents[2]


def _controller_module():
    module_name = "run_groot_rollout_controller_under_test"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name, _REPOSITORY_ROOT / "scripts" / "run_groot_rollout_controller.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def test_local_transport_exposes_ledger_backed_lease_heartbeat_terminal_and_stop(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    module = _controller_module()
    clock = [1_000]
    ledger = TaskLedger(
        tmp_path / "rollouts.sqlite3",
        attempt_matrix=[{"garment": "Top_Long_Seen_0", "seed": 101}],
        target_accepted=1,
        clock_ns=lambda: clock[0],
    )
    try:
        controller = module.LocalRolloutController(ledger, lease_duration_ns=100)
        leased = controller.handle({"operation": "lease", "worker_id": "worker-a"})
        assert leased["status"] == "leased"
        assert leased["attempt"]["schedule_index"] == 0

        clock[0] = 1_050
        heartbeat = controller.handle({
            "operation": "heartbeat", "worker_id": "worker-a", "attempt_id": leased["attempt"]["attempt_id"],
            "lease_id": leased["lease_id"],
        })
        assert heartbeat["status"] == "leased"
        terminal = controller.handle({
            "operation": "terminal", "worker_id": "worker-a", "attempt_id": leased["attempt"]["attempt_id"],
            "lease_id": leased["lease_id"], "raw_artifact_id": "raw-episode",
        })
        assert terminal == {"status": "terminal_pending_validation"}
        stopped = controller.handle({"operation": "stop", "reason": "preempted"})
        assert stopped == {"status": "stopped"}
        assert controller.handle({"operation": "lease", "worker_id": "worker-b"}) == {"status": "unavailable"}
        resumed = controller.handle({"operation": "resume", "reason": "replacement-vm-ready"})
        assert resumed == {"status": "resumed"}
        # This attempt was already terminal before preemption; resuming does
        # not re-execute canonicalized terminal work.
        assert controller.handle({"operation": "lease", "worker_id": "worker-b"}) == {"status": "unavailable"}
    finally:
        ledger.close()


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf", "1e30", "1e-30"])
def test_cli_rejects_non_finite_or_unrepresentable_lease_duration_before_creating_a_database(tmp_path, duration) -> None:
    module = _controller_module()
    matrix = tmp_path / "matrix.json"
    database = tmp_path / "rollouts.sqlite3"
    matrix.write_text('[{"garment":"Top_Long_Seen_0","seed":101}]', encoding="utf-8")

    with pytest.raises(SystemExit) as exit_status:
        module.main(["--database", str(database), "--attempt-matrix", str(matrix), "--lease-seconds", duration])

    assert exit_status.value.code == 2
    assert not database.exists()
