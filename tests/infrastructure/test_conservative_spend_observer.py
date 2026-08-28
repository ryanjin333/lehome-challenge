"""Offline contract tests for the local conservative spend observer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _observer_module():
    spec = importlib.util.spec_from_file_location(
        "conservative_spend_observer", ROOT / "scripts" / "run_conservative_spend_observer.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _controller_module():
    spec = importlib.util.spec_from_file_location(
        "simple_curriculum_observer_integration", ROOT / "scripts" / "run_simple_curriculum_collection.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(controller, root: Path, spend_observer: Path):
    return controller.CollectionConfig(
        campaign_root=root / "campaign", host_code_root=ROOT,
        run_id="fresh-run-20260828-observer", round_id="fresh-12k-20260828-observer",
        max_wall_seconds=3600.0, max_spend_usd=99.0, paid=True,
        gpu_stop_command="/usr/local/libexec/lehome-stop-gpu",
        runtime_identity={
            "rollout_image": "repo/rollout@sha256:" + "a" * 64,
            "trainer_image": "repo/trainer@sha256:" + "b" * 64,
            "policy_repo": "ryanjin333/lehome-groot-n17-models",
            "policy_revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
            "policy_step": 12000,
            "policy_artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
            "simulator_device": "cpu", "cloth_device": "cpu", "policy_device": "cuda:0", "worker_count": 4,
        }, spend_observer=spend_observer,
    )


def test_receipts_remain_fresh_for_controller_and_restart_never_regresses(tmp_path: Path) -> None:
    observer = _observer_module()
    output = tmp_path / "spend-observation.json"
    baseline = datetime(2026, 8, 28, tzinfo=UTC)
    first_now = baseline + timedelta(minutes=6)

    first = observer.write_observation(
        output=output, baseline_usd="20.25", baseline_observed_at=baseline,
        max_hourly_burn_usd="1.50", now=first_now, observer_name="offline-test",
    )
    assert set(first) == {"schema_version", "kind", "observer", "observed_at_utc", "spent_usd"}
    assert first["observed_at_utc"] == "2026-08-28T00:06:00Z"
    assert first["spent_usd"] > 20.25

    later = observer.write_observation(
        output=output, baseline_usd="20.25", baseline_observed_at=baseline,
        max_hourly_burn_usd="1.50", now=first_now + timedelta(minutes=6), observer_name="offline-test",
    )
    assert later["spent_usd"] >= first["spent_usd"]
    assert json.loads(output.read_text(encoding="utf-8")) == later

    with pytest.raises(observer.ObserverError, match="regress|baseline"):
        observer.write_observation(
            output=output, baseline_usd="20.25", baseline_observed_at=baseline,
            max_hourly_burn_usd="0.01", now=first_now + timedelta(minutes=7), observer_name="offline-test",
        )


def test_observer_rejects_future_baseline_unsafe_output_and_invalid_interval(tmp_path: Path) -> None:
    observer = _observer_module()
    now = datetime(2026, 8, 28, tzinfo=UTC)
    output = tmp_path / "spend-observation.json"

    with pytest.raises(observer.ObserverError, match="future"):
        observer.write_observation(
            output=output, baseline_usd="20.25", baseline_observed_at=now + timedelta(seconds=1),
            max_hourly_burn_usd="1.50", now=now, observer_name="offline-test",
        )
    output.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(observer.ObserverError, match="unsafe"):
        observer.write_observation(
            output=output, baseline_usd="20.25", baseline_observed_at=now,
            max_hourly_burn_usd="1.50", now=now, observer_name="offline-test",
        )
    with pytest.raises(observer.ObserverError, match="interval"):
        observer.validate_interval(31)


def test_one_shot_cli_writes_exact_receipt_without_waiting(tmp_path: Path) -> None:
    observer = _observer_module()
    output = tmp_path / "spend-observation.json"
    baseline = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    assert observer.main((
        "--output", str(output), "--baseline-usd", "20.25",
        "--baseline-observed-at-utc", baseline, "--max-hourly-burn-usd", "1.50", "--once",
    )) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["kind"] == "lehome_spend_observation_v1"


def test_controller_accepts_fresh_conservative_receipt_after_six_minutes_and_stops_at_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer, controller = _observer_module(), _controller_module()
    baseline = datetime(2026, 8, 28, tzinfo=UTC)
    output = tmp_path / "spend-observation.json"

    class FrozenDatetime(datetime):
        current = baseline + timedelta(minutes=6)

        @classmethod
        def now(cls, _timezone=None):
            return cls.current

    monkeypatch.setattr(controller, "datetime", FrozenDatetime)
    observer.write_observation(
        output=output, baseline_usd="20.25", baseline_observed_at=baseline,
        max_hourly_burn_usd="1.50", now=FrozenDatetime.current, observer_name="controller-test",
    )
    journal = controller.StageJournal(_config(controller, tmp_path, output))
    journal.check_budget()  # Baseline is six minutes old, but receipt is fresh.

    FrozenDatetime.current += timedelta(minutes=6)
    observer.write_observation(
        output=output, baseline_usd="20.25", baseline_observed_at=baseline,
        max_hourly_burn_usd="1.50", now=FrozenDatetime.current, observer_name="controller-test",
    )
    journal.check_budget()  # Later restart/update remains monotonic and fresh.

    capped_output = tmp_path / "capped-spend-observation.json"
    observer.write_observation(
        output=capped_output, baseline_usd="98.99", baseline_observed_at=baseline,
        max_hourly_burn_usd="1.50", now=baseline + timedelta(seconds=24), observer_name="cap-test",
    )
    FrozenDatetime.current = baseline + timedelta(seconds=24)
    with pytest.raises(controller.BudgetLimitError, match="budget"):
        controller.StageJournal(_config(controller, tmp_path / "cap", capped_output)).check_budget()
