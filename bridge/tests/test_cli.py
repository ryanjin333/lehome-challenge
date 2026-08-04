from __future__ import annotations

from pathlib import Path

import pytest

from lehome_bridge.cli import build_parser, default_secret_path, main


def test_bridge_defaults_to_loopback_and_30hz_without_opening_hardware() -> None:
    args = build_parser().parse_args(
        [
            "--left-port", "/dev/left",
            "--right-port", "/dev/right",
            "--left-calibration", "left.json",
            "--right-calibration", "right.json",
        ]
    )
    assert args.host == "127.0.0.1"
    assert args.port == 18080
    assert args.hz == 30
    assert args.session_nonce is None


def test_bridge_secret_path_is_not_a_command_line_option() -> None:
    parser = build_parser()
    assert "--secret-file" not in parser.format_help()
    assert default_secret_path().name == "bridge-session.secret"


def test_mac_bridge_removes_its_one_session_secret_on_success_and_error(tmp_path: Path, monkeypatch) -> None:
    secret_path = tmp_path / "bridge-session.secret"

    class Bus:
        def disconnect(self) -> None:
            pass

    class Reader:
        pass

    class Connection:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def connect(self) -> None:
            pass

        def close(self) -> None:
            pass

        def request_stop(self) -> None:
            pass

    monkeypatch.setattr("lehome_bridge.cli.default_secret_path", lambda: secret_path)
    monkeypatch.setattr("lehome_bridge.cli.open_feetech_bus", lambda **_kwargs: Bus())
    monkeypatch.setattr("lehome_bridge.cli.DualLeaderReader", lambda *_args, **_kwargs: Reader())
    monkeypatch.setattr("lehome_bridge.cli.BridgeConnection", Connection)

    arguments = [
        "--left-port", "/dev/left", "--right-port", "/dev/right",
        "--left-calibration", "left.json", "--right-calibration", "right.json", "--session-nonce", "nonce",
    ]
    secret_path.write_bytes(b"x" * 32)
    secret_path.chmod(0o600)
    monkeypatch.setattr("lehome_bridge.cli.stream", lambda *_args, **_kwargs: None)
    assert main(arguments) == 0
    assert not secret_path.exists()

    secret_path.write_bytes(b"x" * 32)
    secret_path.chmod(0o600)

    def fail(*_args, **_kwargs) -> None:
        raise RuntimeError("stream failed")

    monkeypatch.setattr("lehome_bridge.cli.stream", fail)
    with pytest.raises(RuntimeError, match="stream failed"):
        main(arguments)
    assert not secret_path.exists()


def test_mac_bridge_refuses_to_remove_a_replaced_secret_path(tmp_path: Path, monkeypatch) -> None:
    secret_path = tmp_path / "bridge-session.secret"

    class Bus:
        def disconnect(self) -> None:
            pass

    class Connection:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def connect(self) -> None:
            pass

        def close(self) -> None:
            pass

        def request_stop(self) -> None:
            pass

    monkeypatch.setattr("lehome_bridge.cli.default_secret_path", lambda: secret_path)
    monkeypatch.setattr("lehome_bridge.cli.open_feetech_bus", lambda **_kwargs: Bus())
    monkeypatch.setattr("lehome_bridge.cli.DualLeaderReader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("lehome_bridge.cli.BridgeConnection", Connection)
    secret_path.write_bytes(b"x" * 32)
    secret_path.chmod(0o600)

    def replace(*_args, **_kwargs) -> None:
        secret_path.unlink()
        secret_path.write_bytes(b"unrelated")
        secret_path.chmod(0o600)

    monkeypatch.setattr("lehome_bridge.cli.stream", replace)
    with pytest.raises(RuntimeError, match="not opened by this session"):
        main([
            "--left-port", "/dev/left", "--right-port", "/dev/right",
            "--left-calibration", "left.json", "--right-calibration", "right.json", "--session-nonce", "nonce",
        ])
    assert secret_path.read_bytes() == b"unrelated"


def test_mac_bridge_scrubs_a_private_but_invalid_short_session_secret_before_setup(tmp_path: Path, monkeypatch) -> None:
    secret_path = tmp_path / "bridge-session.secret"
    secret_path.write_bytes(b"short")
    secret_path.chmod(0o600)
    monkeypatch.setattr("lehome_bridge.cli.default_secret_path", lambda: secret_path)

    with pytest.raises(ValueError, match="at least 32 bytes"):
        main([
            "--left-port", "/dev/left", "--right-port", "/dev/right",
            "--left-calibration", "left.json", "--right-calibration", "right.json", "--session-nonce", "nonce",
        ])

    assert not secret_path.exists()
