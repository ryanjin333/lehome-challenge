from __future__ import annotations

from pathlib import Path

import pytest

from lehome_train.redaction import (
    ACCESS_TOKEN_PATTERNS,
    CREDENTIAL_FILENAMES,
    DENIED_PATH_COMPONENTS,
    ArtifactRejected,
    generate_upload_allowlist,
)


def test_upload_allowlist_is_sorted_canonical_and_hashed(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    (root / "reports").mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "reports" / "result.json").write_text('{"passed":true}', encoding="utf-8")
    (root / "checkpoints" / "state.bin").write_bytes(b"checkpoint")

    entries = generate_upload_allowlist(
        root,
        ["reports/result.json", Path("checkpoints/state.bin")],
    )

    assert [entry.relative_path for entry in entries] == [
        "checkpoints/state.bin",
        "reports/result.json",
    ]
    assert [entry.byte_size for entry in entries] == [10, 15]
    assert all(len(entry.sha256) == 64 for entry in entries)


@pytest.mark.parametrize(
    "candidate",
    [
        "../outside.txt",
        "reports/../../outside.txt",
        "/etc/passwd",
    ],
)
def test_upload_allowlist_rejects_absolute_and_traversal_paths(
    tmp_path: Path,
    candidate: str,
) -> None:
    root = tmp_path / "experiment"
    root.mkdir()

    with pytest.raises(ArtifactRejected):
        generate_upload_allowlist(root, [candidate])


def test_upload_allowlist_rejects_symlinks_at_any_component(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("not a token", encoding="utf-8")
    (root / "linked-directory").symlink_to(outside, target_is_directory=True)
    (root / "linked-file").symlink_to(outside / "secret.txt")

    with pytest.raises(ArtifactRejected):
        generate_upload_allowlist(root, ["linked-directory/secret.txt"])
    with pytest.raises(ArtifactRejected):
        generate_upload_allowlist(root, ["linked-file"])


@pytest.mark.parametrize(
    "candidate",
    [
        ".hidden",
        "reports/.private/result.json",
        "cache/model.bin",
        "__pycache__/module.pyc",
        "environment.yml",
        ".env.production",
    ],
)
def test_upload_allowlist_rejects_dotfiles_caches_and_env_files(
    tmp_path: Path,
    candidate: str,
) -> None:
    root = tmp_path / "experiment"
    path = root / candidate
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("safe", encoding="utf-8")

    with pytest.raises(ArtifactRejected):
        generate_upload_allowlist(root, [candidate])


@pytest.mark.parametrize(
    "filename",
    [
        "token",
        "stored_tokens",
        "credentials.json",
        "huggingface_token",
        "github_token",
        "openai_api_key",
        "runpod_api_key",
    ],
)
def test_upload_allowlist_rejects_credential_store_filenames(
    tmp_path: Path,
    filename: str,
) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    (root / filename).write_text("safe", encoding="utf-8")

    assert filename.casefold() in CREDENTIAL_FILENAMES
    with pytest.raises(ArtifactRejected):
        generate_upload_allowlist(root, [filename])


@pytest.mark.parametrize(
    "secret",
    [
        "hf_" + ("A1" * 20),
        "ghp_" + ("a1" * 20),
        "github_pat_" + ("Ab1_" * 16),
        "sk-proj-" + ("Ab1_" * 10),
        "rpa_" + ("Ab1" * 12),
    ],
)
def test_upload_allowlist_rejects_supported_token_shapes_without_echoing_secret(
    tmp_path: Path,
    secret: str,
) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    path = root / "log.txt"
    path.write_text(f"prefix {secret} suffix", encoding="utf-8")

    with pytest.raises(ArtifactRejected) as caught:
        generate_upload_allowlist(root, ["log.txt"])

    assert secret not in str(caught.value)
    assert any(pattern.search(secret) for pattern in ACCESS_TOKEN_PATTERNS)


def test_deny_lists_are_immutable_and_documented() -> None:
    assert isinstance(CREDENTIAL_FILENAMES, frozenset)
    assert isinstance(DENIED_PATH_COMPONENTS, frozenset)
    assert "cache" in DENIED_PATH_COMPONENTS
