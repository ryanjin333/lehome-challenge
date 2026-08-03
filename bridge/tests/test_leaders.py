from __future__ import annotations

import json
from pathlib import Path

import pytest

from lehome_bridge.client import read_secret_file
from lehome_bridge.leaders import JOINTS, DualLeaderReader


class FakeBus:
    def __init__(self, *, serial: str, positions: dict[str, float]) -> None:
        self.serial_identity = serial
        self.positions = positions
        self.reads = 0

    def sync_read(self, register: str) -> dict[str, float]:
        assert register == "Present_Position"
        self.reads += 1
        return self.positions


def calibration(root: Path, name: str) -> Path:
    path = root / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                joint: {
                    "id": index + 1,
                    "drive_mode": 0,
                    "homing_offset": 0,
                    "range_min": -1000,
                    "range_max": 1000,
                }
                for index, joint in enumerate(JOINTS)
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_dual_reader_returns_left_then_right_joint_order(tmp_path: Path) -> None:
    left = FakeBus(serial="L1", positions={name: index for index, name in enumerate(JOINTS)})
    right = FakeBus(serial="R1", positions={name: index + 10 for index, name in enumerate(JOINTS)})
    reader = DualLeaderReader(
        left,
        right,
        left_calibration=calibration(tmp_path, "left"),
        right_calibration=calibration(tmp_path, "right"),
    )
    sample = reader.read()
    assert sample.positions == (0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15)
    assert sample.left_serial == "L1"
    assert sample.right_serial == "R1"
    assert left.reads == right.reads == 1


def test_reader_rejects_same_bus_or_invalid_calibration(tmp_path: Path) -> None:
    bus = FakeBus(serial="same", positions={name: 0 for name in JOINTS})
    with pytest.raises(ValueError, match="distinct"):
        DualLeaderReader(
            bus,
            bus,
            left_calibration=calibration(tmp_path, "left"),
            right_calibration=calibration(tmp_path, "right"),
        )
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="calibration"):
        DualLeaderReader(
            FakeBus(serial="left", positions={name: 0 for name in JOINTS}),
            FakeBus(serial="right", positions={name: 0 for name in JOINTS}),
            left_calibration=invalid,
            right_calibration=calibration(tmp_path, "right-2"),
        )


def test_secret_file_must_be_private_and_at_least_32_bytes(tmp_path: Path) -> None:
    secret = tmp_path / "secret"
    secret.write_bytes(b"x" * 32)
    secret.chmod(0o600)
    assert read_secret_file(secret) == b"x" * 32
    secret.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        read_secret_file(secret)
