from __future__ import annotations

import pytest


def _fidelity(**overrides: bool) -> dict[str, bool]:
    return {
        "missing_cloth": False,
        "cloth_flight": False,
        "nonfinite_cloth_state": False,
        "safety_failure": False,
        "monitor_active": True,
        "monitor_observed": True,
        **overrides,
    }


def test_validate_fidelity_requires_the_exact_observed_six_field_receipt() -> None:
    from lehome.flywheel.fidelity import validate_fidelity

    assert validate_fidelity(_fidelity()) == _fidelity()
    with pytest.raises(ValueError, match="fidelity"):
        validate_fidelity({key: value for key, value in _fidelity().items() if key != "monitor_observed"})
    with pytest.raises(ValueError, match="monitor"):
        validate_fidelity(_fidelity(monitor_active=False))


def test_fidelity_receipt_factory_returns_only_the_validated_six_fields() -> None:
    from lehome.flywheel.fidelity import fidelity_receipt

    assert fidelity_receipt(
        missing_cloth=False,
        cloth_flight=True,
        nonfinite_cloth_state=False,
        safety_failure=False,
        monitor_active=True,
        monitor_observed=True,
    ) == _fidelity(cloth_flight=True)


def test_fidelity_receipt_factory_rejects_inactive_monitors() -> None:
    from lehome.flywheel.fidelity import fidelity_receipt

    with pytest.raises(ValueError, match="monitor"):
        fidelity_receipt(
            missing_cloth=True,
            cloth_flight=False,
            nonfinite_cloth_state=False,
            safety_failure=False,
            monitor_active=False,
            monitor_observed=True,
        )


def test_cloth_fidelity_error_carries_only_a_validated_cloth_failure() -> None:
    from lehome.flywheel.fidelity import ClothFidelityError

    error = ClothFidelityError("missing_cloth", _fidelity(missing_cloth=True))

    assert error.code == "missing_cloth"
    assert error.fidelity == _fidelity(missing_cloth=True)
    with pytest.raises(ValueError, match="code"):
        ClothFidelityError("monitor_observed", _fidelity(monitor_observed=True))


def test_fidelity_diagnostic_accepts_bounded_physical_and_policy_step_evidence() -> None:
    from lehome.flywheel.fidelity import validate_fidelity_diagnostic

    diagnostic = {
        "stage": "policy_step",
        "step_index": 203,
        "physical_health": {
            "max_position_m": 1.7,
            "max_extent_m": 1.2,
            "max_velocity_mps": 4.9,
            "max_position_limit_m": 1.57,
            "max_extent_limit_m": 1.8,
            "max_velocity_limit_mps": 4.75,
            "exceeded_metrics": ["max_position_m", "max_velocity_mps"],
        },
        "policy_action": {
            "policy_action_limits_available": True,
            "policy_action_dimension": 12,
            "policy_action_nonfinite_count": 0,
            "policy_action_outside_live_joint_limit_count": 2,
            "policy_action_total_steps": 204,
            "policy_action_joint_diagnostics": {
                "left_elbow_flex": {
                    "target_finite": True,
                    "outside_live_joint_limit": True,
                    "limit_violation_rad": 0.25,
                    "target_to_live_joint_position_delta_rad": 1.4,
                },
            },
        },
    }

    assert validate_fidelity_diagnostic(diagnostic) == diagnostic


@pytest.mark.parametrize(
    "diagnostic",
    [
        {"stage": "unknown"},
        {"stage": "policy_step"},
        {"stage": "post_stabilization", "step_index": 1},
        {"stage": "post_stabilization", "unexpected": "unbounded"},
        {
            "stage": "policy_step", "step_index": 0,
            "policy_action": {"policy_action_total_steps": 1, "free_text": "x" * 10_000},
        },
        {
            "stage": "post_stabilization",
            "physical_health": {
                "max_position_m": float("inf"), "max_extent_m": 1.0,
                "max_velocity_mps": 1.0, "max_position_limit_m": 1.0,
                "max_extent_limit_m": 1.0, "max_velocity_limit_mps": 1.0,
                "exceeded_metrics": ["max_position_m"],
            },
        },
        {
            "stage": "policy_step", "step_index": 0,
            "policy_action": {
                "policy_action_joint_diagnostics": {
                    f"joint-{index}": {
                        "target_finite": True, "outside_live_joint_limit": False,
                        "limit_violation_rad": 0.0,
                        "target_to_live_joint_position_delta_rad": 0.0,
                    }
                    for index in range(13)
                },
            },
        },
    ],
)
def test_fidelity_diagnostic_rejects_malformed_or_unbounded_evidence(diagnostic) -> None:
    from lehome.flywheel.fidelity import validate_fidelity_diagnostic

    with pytest.raises(ValueError, match="diagnostic"):
        validate_fidelity_diagnostic(diagnostic)


def test_cloth_fidelity_error_keeps_optional_diagnostic_and_legacy_constructor() -> None:
    from lehome.flywheel.fidelity import ClothFidelityError

    legacy = ClothFidelityError("cloth_flight", _fidelity(cloth_flight=True))
    measured = ClothFidelityError(
        "cloth_flight", _fidelity(cloth_flight=True),
        diagnostic={
            "stage": "reset_write_readback",
            "write_readback": {
                "max_position_delta_m": 0.0002,
                "max_velocity_delta_mps": 0.003,
            },
        },
    )

    assert legacy.diagnostic is None
    assert measured.diagnostic == {
        "stage": "reset_write_readback",
        "write_readback": {
            "max_position_delta_m": 0.0002,
            "max_velocity_delta_mps": 0.003,
        },
    }
