"""Hardware-injected dual SO101 leader sampling for the Mac bridge."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Protocol


JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
_CALIBRATION_FIELDS = frozenset({"id", "drive_mode", "homing_offset", "range_min", "range_max"})


class LeaderBus(Protocol):
    serial_identity: str

    def sync_read(self, register: str) -> dict[str, float]: ...


@dataclass(frozen=True, slots=True)
class CalibrationIdentity:
    path: Path
    sha256: str
    motor_limits: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class LeaderSample:
    monotonic_ns: int
    positions: tuple[float, ...]
    left_serial: str
    right_serial: str


def _load_calibration(path: Path) -> CalibrationIdentity:
    if not path.is_file():
        raise ValueError("leader calibration must be a regular JSON file")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("leader calibration is invalid JSON") from error
    if not isinstance(parsed, dict) or set(parsed) != set(JOINTS):
        raise ValueError("leader calibration must contain the six SO101 joints")
    for joint in JOINTS:
        value = parsed[joint]
        if not isinstance(value, dict) or set(value) != _CALIBRATION_FIELDS:
            raise ValueError("leader calibration does not match the SO101 calibration format")
        if not all(isinstance(value[field], int) for field in _CALIBRATION_FIELDS):
            raise ValueError("leader calibration fields must be integers")
        if value["range_min"] >= value["range_max"]:
            raise ValueError("leader calibration range is invalid")
    with path.open("rb") as calibration_file:
        digest = hashlib.file_digest(calibration_file, "sha256").hexdigest()
    motor_limits = tuple((float(parsed[joint]["range_min"]), float(parsed[joint]["range_max"])) for joint in JOINTS)
    return CalibrationIdentity(path=path, sha256=digest, motor_limits=motor_limits)


def _finite_positions(values: tuple[object, ...]) -> tuple[float, ...]:
    if len(values) != 12 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("leader read must contain 12 finite positions")
    return tuple(float(value) for value in values)


class DualLeaderReader:
    """Reads injected buses in the immutable organizer left-then-right order."""

    def __init__(
        self,
        left_bus: LeaderBus,
        right_bus: LeaderBus,
        *,
        left_calibration: Path,
        right_calibration: Path,
    ) -> None:
        if left_bus is right_bus or left_bus.serial_identity == right_bus.serial_identity:
            raise ValueError("leader buses must have distinct serial identities")
        self.left_bus = left_bus
        self.right_bus = right_bus
        self.left_calibration = _load_calibration(Path(left_calibration))
        self.right_calibration = _load_calibration(Path(right_calibration))
        self.left_motor_limits = self.left_calibration.motor_limits
        self.right_motor_limits = self.right_calibration.motor_limits

    def read(self) -> LeaderSample:
        left = self.left_bus.sync_read("Present_Position")
        right = self.right_bus.sync_read("Present_Position")
        try:
            values = tuple(left[name] for name in JOINTS) + tuple(right[name] for name in JOINTS)
        except (KeyError, TypeError) as error:
            raise ValueError("leader bus did not return the required SO101 joints") from error
        return LeaderSample(
            monotonic_ns=time.monotonic_ns(),
            positions=_finite_positions(values),
            left_serial=self.left_bus.serial_identity,
            right_serial=self.right_bus.serial_identity,
        )


def open_feetech_bus(*, port: str, calibration_path: Path) -> LeaderBus:
    """Open one real bus only when an operator explicitly invokes the bridge CLI.

    The canonical Feetech implementation is reused from ``source/lehome``; it is
    deliberately imported here rather than at module import time so test runs do
    not import serial transport, Isaac, or keyboard-listener code.
    """
    calibration = _load_calibration(calibration_path)
    try:
        from lehome.devices.lerobot.common.motors import (
            FeetechMotorsBus,
            Motor,
            MotorCalibration,
            MotorNormMode,
        )
    except ImportError as error:  # pragma: no cover - requires the extracted transport at runtime
        raise RuntimeError("canonical LeHome Feetech transport is unavailable") from error
    parsed: dict[str, Any] = json.loads(calibration.path.read_text(encoding="utf-8"))
    motors = {
        name: Motor(index + 1, "sts3215", MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.RANGE_M100_100)
        for index, name in enumerate(JOINTS)
    }
    bus = FeetechMotorsBus(
        port=port,
        motors=motors,
        calibration={name: MotorCalibration(**values) for name, values in parsed.items()},
    )
    bus.serial_identity = port
    bus.connect()
    bus.disable_torque()
    return bus
