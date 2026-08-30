from __future__ import annotations

import os
from pathlib import Path

import pytest

from b1k_rollout.token_bootstrap import TokenBootstrapError, install_token


def test_atomic_token_install_replaces_a_restart_symlink_without_touching_its_target(tmp_path: Path) -> None:
    token_path = tmp_path / "workspace" / ".cache" / "huggingface" / "token"
    token_path.parent.mkdir(parents=True, mode=0o700)
    outside = tmp_path / "outside-token"
    outside.write_text("keep-me", encoding="utf-8")
    token_path.symlink_to(outside)

    install_token("new-secret", token_path=token_path, uid=os.getuid(), gid=os.getgid())

    assert outside.read_text(encoding="utf-8") == "keep-me"
    assert not token_path.is_symlink()
    assert token_path.read_text(encoding="utf-8") == "new-secret"
    assert token_path.stat().st_mode & 0o777 == 0o600


def test_atomic_token_install_rejects_a_symlinked_parent_before_writing(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    token_path = tmp_path / "workspace" / ".cache" / "huggingface" / "token"
    token_path.parent.parent.mkdir(parents=True, mode=0o700)
    token_path.parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(TokenBootstrapError, match="unsafe token parent"):
        install_token("new-secret", token_path=token_path, uid=os.getuid(), gid=os.getgid())

    assert not (outside / "token").exists()


def test_atomic_token_install_creates_missing_private_parent_components(tmp_path: Path) -> None:
    token_path = tmp_path / "fresh-workspace" / ".cache" / "huggingface" / "token"

    install_token("new-secret", token_path=token_path, uid=os.getuid(), gid=os.getgid())

    assert token_path.read_text(encoding="utf-8") == "new-secret"
    assert token_path.parent.stat().st_mode & 0o777 == 0o700
