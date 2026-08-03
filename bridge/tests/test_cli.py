from __future__ import annotations

from lehome_bridge.cli import build_parser, default_secret_path


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
