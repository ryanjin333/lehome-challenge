"""Safe operator entry point for the macOS-only physical leader bridge."""

from __future__ import annotations

import argparse
from pathlib import Path

from .client import BridgeConnection, DEFAULT_HOST, DEFAULT_PORT, read_secret_file, stream
from .leaders import DualLeaderReader, open_feetech_bus


def default_secret_path() -> Path:
    """Fixed private location; a secret path is never accepted in process argv."""
    return Path.home() / ".local" / "state" / "lehome-bridge" / "bridge-session.secret"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-port", required=True, help="macOS device path for the left SO101 bus")
    parser.add_argument("--right-port", required=True, help="macOS device path for the right SO101 bus")
    parser.add_argument("--left-calibration", type=Path, required=True)
    parser.add_argument("--right-calibration", type=Path, required=True)
    parser.add_argument("--session-nonce", help="short-lived nonce shown by the remote collector")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--hz", type=int, default=30)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.host != DEFAULT_HOST:
        raise ValueError("the bridge may connect only to the loopback SSH-forward endpoint")
    if not args.session_nonce:
        raise ValueError("a short-lived remote session nonce is required")
    if not isinstance(args.port, int) or not 1 <= args.port <= 65535:
        raise ValueError("bridge port must be in the TCP port range")
    if not isinstance(args.hz, int) or args.hz <= 0:
        raise ValueError("bridge hz must be positive")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    # No serial transport is imported or opened until argument validation and
    # private-secret validation both succeed.
    secret = read_secret_file(default_secret_path())
    left = open_feetech_bus(port=args.left_port, calibration_path=args.left_calibration)
    right = open_feetech_bus(port=args.right_port, calibration_path=args.right_calibration)
    reader = DualLeaderReader(
        left,
        right,
        left_calibration=args.left_calibration,
        right_calibration=args.right_calibration,
    )
    connection = BridgeConnection(
        reader,
        secret=secret,
        session_nonce=args.session_nonce,
        host=args.host,
        port=args.port,
        hz=args.hz,
    )
    try:
        connection.connect()
        stream(reader, connection, hz=args.hz)
    except KeyboardInterrupt:
        connection.request_stop()
    finally:
        connection.close()
        for bus in (left, right):
            disconnect = getattr(bus, "disconnect", None)
            if callable(disconnect):
                disconnect()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
