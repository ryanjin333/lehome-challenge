from __future__ import annotations

import stat

import pytest

from scripts.collect_groot_dagger import build_parser, create_session_secret, prepare_session, validate_args


def arguments(*values: str):
    return build_parser().parse_args(list(values))


def test_collector_defaults_to_practice_and_loopback() -> None:
    args = arguments()
    assert args.mode == "practice"
    assert args.listen_host == "127.0.0.1"
    assert args.enable_training_output is False


def test_training_output_requires_mode_and_pinned_threshold_manifest(tmp_path) -> None:
    args = arguments(
        "--mode", "dagger", "--enable-training-output", "--quality-thresholds", str(tmp_path / "missing.json"),
        "--organizer-dataset-revision", "a" * 40, "--organizer-dataset-sha256", "b" * 64,
    )
    with pytest.raises(ValueError, match="quality thresholds"):
        validate_args(args)
    with pytest.raises(ValueError, match="loopback"):
        validate_args(arguments("--listen-host", "0.0.0.0"))


def test_interactive_collection_requires_known_calibration_hashes() -> None:
    with pytest.raises(ValueError, match="calibration"):
        validate_args(arguments("--interactive"))


def test_practice_session_has_no_export_and_secret_is_mode_0600(tmp_path) -> None:
    args = arguments("--run-root", str(tmp_path / "practice"))
    session = prepare_session(args, secret_path=tmp_path / "bridge-session.secret")
    assert session.controller.mode == "practice"
    assert not (args.run_root / "exports").exists()
    assert stat.S_IMODE((tmp_path / "bridge-session.secret").stat().st_mode) == 0o600
