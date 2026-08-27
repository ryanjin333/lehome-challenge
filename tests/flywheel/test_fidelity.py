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


def test_cloth_fidelity_error_carries_only_a_validated_cloth_failure() -> None:
    from lehome.flywheel.fidelity import ClothFidelityError

    error = ClothFidelityError("missing_cloth", _fidelity(missing_cloth=True))

    assert error.code == "missing_cloth"
    assert error.fidelity == _fidelity(missing_cloth=True)
    with pytest.raises(ValueError, match="code"):
        ClothFidelityError("monitor_observed", _fidelity(monitor_observed=True))
