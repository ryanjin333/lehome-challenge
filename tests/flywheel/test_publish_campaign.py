from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.publish_groot_flywheel_campaign import (
    _invalidate_disposal_receipt,
    _remote_files,
    _revision,
    _token_from_process,
    main,
)


def test_token_is_read_from_launch_process_without_becoming_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _self: b"OTHER=value\0HF_TOKEN=hf_process_only\0",
    )

    assert _token_from_process(1) == "hf_process_only"

    monkeypatch.setattr(Path, "read_bytes", lambda _self: b"OTHER=value\0")
    with pytest.raises(ValueError, match="unavailable"):
        _token_from_process(1)


def test_remote_listing_accepts_only_files_with_bounded_sizes() -> None:
    class Api:
        def list_repo_tree(self, **_kwargs):
            return (
                SimpleNamespace(path="prefix", size=None),
                SimpleNamespace(path="prefix/a.bin", size=3),
            )

    assert _remote_files(
        Api(), repository="ryanjin333/lehome-groot-n17-data", revision="a" * 40, token="secret"
    ) == {"prefix/a.bin": 3}


def test_hub_revision_requires_a_full_immutable_commit() -> None:
    assert _revision(SimpleNamespace(oid="a" * 40)) == "a" * 40
    assert _revision(SimpleNamespace(commit_url="https://huggingface.co/x/commit/" + "b" * 40)) == "b" * 40
    with pytest.raises(ValueError, match="immutable commit"):
        _revision(SimpleNamespace(oid="main"))


def test_new_attempt_invalidates_only_a_regular_prior_disposal_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "remote-verification.json"
    receipt.write_text('{"disposable":true}\n', encoding="utf-8")

    assert _invalidate_disposal_receipt(tmp_path) == receipt
    assert not receipt.exists()

    receipt.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="unsafe"):
        _invalidate_disposal_receipt(tmp_path)


def test_invalid_repository_still_invalidates_prior_disposal_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "remote-verification.json"
    receipt.write_text('{"disposable":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="approved private dataset"):
        main(
            [
                "--matrix", str(tmp_path / "matrix.json"),
                "--run-root", str(tmp_path),
                "--staging-root", str(tmp_path / "staging"),
                "--readback-root", str(tmp_path / "readback"),
                "--repository", "attacker/wrong-repo",
                "--token-environ-pid", "1",
                "--policy-revision", "a" * 40,
                "--code-revision", "b" * 40,
                "--asset-revision", "c" * 40,
                "--image-identity", "sha256:" + "d" * 64,
                "--policy-artifact-sha256", "e" * 64,
            ]
        )

    assert not receipt.exists()
