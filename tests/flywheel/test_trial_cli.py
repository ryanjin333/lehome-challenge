from __future__ import annotations

import pytest

from scripts.run_groot_flywheel_trial import build_parser, read_pinned_revision, validate_args


def test_trial_cli_requires_pinned_policy_and_existing_matrix(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(["--policy-path", str(tmp_path / "missing"), "--policy-revision", "main"])
    with pytest.raises(ValueError, match="pinned"):
        validate_args(args)


def test_trial_cli_reads_revision_from_a_regular_file(tmp_path) -> None:
    revision = tmp_path / "revision.txt"
    revision.write_text("a" * 40 + "\n", encoding="utf-8")
    assert read_pinned_revision(revision) == "a" * 40
