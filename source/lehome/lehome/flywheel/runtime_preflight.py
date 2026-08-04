"""Fail-closed host compatibility checks for Isaac Sim rollout processes.

Isaac Sim 5.1 x86_64 is deliberately constrained to the NVIDIA R580 driver
line here.  A newer-looking driver is not automatically safe: R590 (including
595.71.05) has reproduced an RTX scene database crash in the rollout runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import platform
import re
import subprocess
from typing import Callable, Protocol, Sequence


_DRIVER_COMMAND = (
    "nvidia-smi",
    "--query-gpu=driver_version",
    "--format=csv,noheader,nounits",
)
_VERSION = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")
_MINIMUM_R580 = (580, 65, 6)
_NEXT_UNREVIEWED_BRANCH = (590, 0, 0)
_PROBE_TIMEOUT_SECONDS = 5.0


class _CompletedProcess(Protocol):
    returncode: int
    stdout: str


Runner = Callable[..., _CompletedProcess]


@dataclass(frozen=True)
class IsaacRuntimeReceipt:
    """A deterministic, secret-free result for normalized image startup."""

    compatible: bool
    error_code: str | None
    driver_versions: tuple[str, ...]
    machine: str
    policy: str = "isaac_sim_5.1_linux_x86_64_r580"
    system: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "compatible": self.compatible,
            "driver_versions": list(self.driver_versions),
            "error_code": self.error_code,
            "machine": self.machine,
            "policy": self.policy,
            "schema_version": 1,
            "system": self.system,
        }


class IsaacRuntimePreflightError(ValueError):
    """Compatibility failure carrying the safe, machine-readable receipt."""

    def __init__(self, receipt: IsaacRuntimeReceipt, message: str) -> None:
        super().__init__(message)
        self.receipt = receipt


def _receipt(
    *,
    compatible: bool,
    error_code: str | None,
    driver_versions: Sequence[str] = (),
    machine: str,
    system: str,
) -> IsaacRuntimeReceipt:
    return IsaacRuntimeReceipt(
        compatible=compatible,
        error_code=error_code,
        driver_versions=tuple(driver_versions),
        machine=machine,
        system=system,
    )


def inspect_isaac_sim_5_1_runtime(
    *,
    system: Callable[[], str] = platform.system,
    machine: Callable[[], str] = platform.machine,
    runner: Runner = subprocess.run,
) -> IsaacRuntimeReceipt:
    """Probe the host using injectable dependencies; never expose probe output.

    The receipt intentionally records only normalized driver versions and a
    stable failure code.  In particular, it excludes command stderr and the
    ambient environment, which can contain unrelated credentials.
    """

    observed_system = system()
    observed_machine = machine()
    if observed_system != "Linux":
        return _receipt(
            compatible=False,
            error_code="unsupported_system",
            machine=observed_machine,
            system=observed_system,
        )
    if observed_machine != "x86_64":
        return _receipt(
            compatible=False,
            error_code="unsupported_machine",
            machine=observed_machine,
            system=observed_system,
        )
    try:
        completed = runner(
            _DRIVER_COMMAND,
            check=False,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _receipt(
            compatible=False,
            error_code="nvidia_smi_unavailable",
            machine=observed_machine,
            system=observed_system,
        )
    if completed.returncode != 0:
        return _receipt(
            compatible=False,
            error_code="nvidia_smi_failed",
            machine=observed_machine,
            system=observed_system,
        )
    versions = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    parsed = tuple(_VERSION.fullmatch(version) for version in versions)
    if not versions or any(version is None for version in parsed):
        return _receipt(
            compatible=False,
            error_code="malformed_driver_version",
            machine=observed_machine,
            system=observed_system,
        )
    if len(set(versions)) != 1:
        return _receipt(
            compatible=False,
            error_code="mixed_driver_versions",
            driver_versions=versions,
            machine=observed_machine,
            system=observed_system,
        )
    assert parsed[0] is not None
    numeric = tuple(int(value) for value in parsed[0].groups())
    if not (
        numeric[0] == 580
        and _MINIMUM_R580 <= numeric < _NEXT_UNREVIEWED_BRANCH
    ):
        return _receipt(
            compatible=False,
            error_code="unreviewed_driver_version",
            driver_versions=versions,
            machine=observed_machine,
            system=observed_system,
        )
    return _receipt(
        compatible=True,
        error_code=None,
        driver_versions=versions,
        machine=observed_machine,
        system=observed_system,
    )


def require_isaac_sim_5_1_runtime(**kwargs: object) -> IsaacRuntimeReceipt:
    """Require a reviewed host before starting any rollout-side runtime work."""

    receipt = inspect_isaac_sim_5_1_runtime(**kwargs)
    if receipt.compatible:
        return receipt
    if receipt.error_code == "unreviewed_driver_version":
        detail = ", ".join(receipt.driver_versions) or "unknown"
        raise IsaacRuntimePreflightError(
            receipt,
            "Isaac Sim 5.1 requires NVIDIA R580 >= 580.65.06 and < 590.0.0; "
            f"host reported {detail}. Switch to a reviewed R580-driver host before retrying.",
        )
    raise IsaacRuntimePreflightError(
        receipt,
        "Isaac Sim 5.1 runtime preflight failed "
        f"({receipt.error_code}); use a Linux x86_64 host with a reviewed R580 NVIDIA driver.",
    )


__all__ = [
    "IsaacRuntimePreflightError",
    "IsaacRuntimeReceipt",
    "inspect_isaac_sim_5_1_runtime",
    "require_isaac_sim_5_1_runtime",
]
