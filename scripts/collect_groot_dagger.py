"""Prepare and operate a loopback-only physical SO101 DAgger collection session."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import secrets
import stat
import re
import threading
from typing import Any

from lehome.flywheel.bridge_receiver import BridgeReceiver, LoopbackBridgeServer
from lehome.flywheel.intervention import InterventionController
from lehome.flywheel.quality import QualityThresholds, load_quality_thresholds


CONTROLS = {
    "space": "activate_or_request_takeover",
    "a": "accept_after_official_success",
    "d": "discard",
    "r": "reset",
    "escape": "safe_exit",
}
LOOPBACK_HOST = "127.0.0.1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def default_secret_path() -> Path:
    return Path.home() / ".local" / "state" / "lehome-groot" / "bridge-session.secret"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("practice", "expert", "dagger"), default="practice")
    parser.add_argument("--listen-host", default=LOOPBACK_HOST)
    parser.add_argument("--listen-port", type=int, default=18080)
    parser.add_argument("--run-root", type=Path, default=Path("runs/groot-dagger"))
    parser.add_argument("--enable-training-output", action="store_true")
    parser.add_argument("--quality-thresholds", type=Path)
    parser.add_argument("--organizer-dataset-revision")
    parser.add_argument("--organizer-dataset-sha256")
    parser.add_argument("--left-calibration-sha256")
    parser.add_argument("--right-calibration-sha256")
    parser.add_argument("--interactive", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> QualityThresholds | None:
    if args.listen_host != LOOPBACK_HOST:
        raise ValueError("DAgger receiver must be loopback-only")
    if not isinstance(args.listen_port, int) or not 1 <= args.listen_port <= 65535:
        raise ValueError("collector listen port must be in the TCP port range")
    calibration_hashes = (args.left_calibration_sha256, args.right_calibration_sha256)
    if any(value is not None for value in calibration_hashes) and not all(
        isinstance(value, str) and _SHA256.fullmatch(value) for value in calibration_hashes
    ):
        raise ValueError("collector calibration hashes must be paired lowercase SHA-256 values")
    if args.interactive and not all(calibration_hashes):
        raise ValueError("interactive collection requires both expected calibration hashes")
    if args.enable_training_output and args.mode not in {"expert", "dagger"}:
        raise ValueError("training output requires expert or dagger mode")
    if not args.enable_training_output:
        return None
    if args.quality_thresholds is None:
        raise ValueError("quality thresholds manifest is required for training output")
    if not args.organizer_dataset_revision or not args.organizer_dataset_sha256:
        raise ValueError("training output requires pinned organizer dataset revision and SHA-256")
    return load_quality_thresholds(
        args.quality_thresholds,
        expected_dataset_revision=args.organizer_dataset_revision,
        expected_dataset_sha256=args.organizer_dataset_sha256,
    )


def create_session_secret(path: Path) -> bytes:
    """Create a one-session secret atomically with owner-only permissions."""
    secret_path = Path(path)
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("refusing to reuse an existing bridge session secret") from error
    secret = secrets.token_bytes(32)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(secret)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        secret_path.unlink(missing_ok=True)
        raise
    if stat.S_IMODE(secret_path.stat().st_mode) != 0o600:
        secret_path.unlink(missing_ok=True)
        raise RuntimeError("failed to create a mode-0600 bridge session secret")
    return secret


@dataclass(slots=True)
class CollectorSession:
    controller: InterventionController
    session_nonce: str
    quality_thresholds: QualityThresholds | None
    bridge_server: LoopbackBridgeServer
    controls: list[str] = field(default_factory=list)

    def record_control(self, control: str) -> None:
        if control not in CONTROLS:
            raise ValueError("unsupported DAgger control")
        self.controls.append(control)

    def status(self) -> dict[str, object]:
        bridge = self.bridge_server.receiver.current()
        return {
            "mode": self.controller.mode,
            "state": self.controller.state,
            "action_source": self.controller.action_source,
            "bridge_age_ms": bridge.sample_age_ms,
            "bridge_jitter_ms": self.bridge_server.receiver.jitter_ms,
            "bridge_state": bridge.reason or "eligible",
        }

    def start_listener(self) -> threading.Thread:
        self.bridge_server.start()

        def serve() -> None:
            try:
                self.bridge_server.serve_one_client()
            except (ConnectionError, OSError, RuntimeError, ValueError):
                # The receiver retains the fail-closed disconnect state. Do not
                # print transport details that could reveal operator context.
                return

        thread = threading.Thread(target=serve, name="lehome-bridge-receiver", daemon=True)
        thread.start()
        return thread

    def close_listener(self) -> None:
        self.bridge_server.close()


def _write_session_manifest(root: Path, session: CollectorSession, args: argparse.Namespace) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "session-manifest.json"
    if manifest.exists():
        raise ValueError("refusing to overwrite an existing DAgger session manifest")
    document: dict[str, Any] = {
        "mode": args.mode,
        "listen_host": args.listen_host,
        "listen_port": args.listen_port,
        "enable_training_output": args.enable_training_output,
        "controls": CONTROLS,
        "session_nonce": session.session_nonce,
    }
    if session.quality_thresholds is not None:
        document["quality_threshold_dataset"] = {
            "revision": session.quality_thresholds.dataset_revision,
            "sha256": session.quality_thresholds.dataset_sha256,
        }
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, manifest)


def prepare_session(args: argparse.Namespace, *, secret_path: Path | None = None) -> CollectorSession:
    thresholds = validate_args(args)
    secret = create_session_secret(default_secret_path() if secret_path is None else secret_path)
    expected_calibrations = None
    if args.left_calibration_sha256 is not None:
        expected_calibrations = (args.left_calibration_sha256, args.right_calibration_sha256)
    receiver = BridgeReceiver(expected_calibrations=expected_calibrations)
    session_nonce = secrets.token_urlsafe(24)
    session = CollectorSession(
        controller=InterventionController(mode=args.mode),
        session_nonce=session_nonce,
        quality_thresholds=thresholds,
        bridge_server=LoopbackBridgeServer(
            secret=secret,
            session_nonce=session_nonce,
            port=args.listen_port,
            receiver=receiver,
        ),
    )
    _write_session_manifest(args.run_root, session, args)
    return session


def ssh_forward_command(port: int) -> str:
    """A copyable command template that exposes no secret or secret path."""
    return f"ssh -N -L {port}:127.0.0.1:{port} USER@APPROVED_NORTH_AMERICAN_HOST"


def run_interactive(session: CollectorSession) -> None:  # pragma: no cover - operator path
    print("Controls:", json.dumps(CONTROLS, sort_keys=True))
    while True:
        print(json.dumps(session.status(), sort_keys=True))
        control = input("control> ").strip().lower()
        if control == "escape":
            session.record_control(control)
            return
        if control not in CONTROLS:
            print("unrecognized control")
            continue
        session.record_control(control)
        # Robot state, official success, bridge health, and actual actions are
        # supplied by the Isaac integration loop; this control loop never emits
        # an unvalidated command by itself.


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session = prepare_session(args)
    print(json.dumps({"ssh_forward": ssh_forward_command(args.listen_port), **session.status()}, sort_keys=True))
    if args.interactive:
        session.start_listener()
        try:
            run_interactive(session)
        finally:
            session.close_listener()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
