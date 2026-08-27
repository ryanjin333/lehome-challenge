"""Immutable first-100 fidelity receipt contract."""

from __future__ import annotations

from typing import Mapping


CLOTH_FIDELITY_CODES = (
    "missing_cloth", "cloth_flight", "nonfinite_cloth_state",
)
FIDELITY_CODES = frozenset(CLOTH_FIDELITY_CODES) | {"safety_failure"}
FIDELITY_FIELDS = FIDELITY_CODES | {"monitor_active", "monitor_observed"}


def validate_fidelity(
    fidelity: object,
    *,
    code: str | None = None,
    require_monitors: bool = True,
) -> dict[str, bool]:
    """Return the exact six-field receipt or reject unauthenticated evidence."""

    if not isinstance(fidelity, Mapping) or set(fidelity) != FIDELITY_FIELDS:
        raise ValueError("fidelity receipt is incomplete")
    if any(type(fidelity[field]) is not bool for field in FIDELITY_FIELDS):
        raise ValueError("fidelity receipt must contain booleans")
    if code is not None and (code not in FIDELITY_CODES or fidelity[code] is not True):
        raise ValueError("fidelity code is invalid")
    if require_monitors and (
        fidelity["monitor_active"] is not True
        or fidelity["monitor_observed"] is not True
    ):
        raise ValueError("fidelity monitors must be active and observed")
    return {field: fidelity[field] for field in sorted(FIDELITY_FIELDS)}


class ClothFidelityError(RuntimeError):
    """A measured physical-cloth failure with its immutable receipt."""

    def __init__(
        self, code: str, fidelity: Mapping[str, object], *, detail: str | None = None,
    ) -> None:
        if code not in CLOTH_FIDELITY_CODES:
            raise ValueError("cloth fidelity code is invalid")
        self.code = code
        self.fidelity = validate_fidelity(fidelity, code=code)
        super().__init__(detail or code)
