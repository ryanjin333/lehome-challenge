from __future__ import annotations

import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

from b1k_rollout import cli
from b1k_rollout.cli import CliError, preflight
from b1k_rollout.controller import CheckpointReceipt
from b1k_rollout.identity import MODEL_REPO


_OFFICIAL_CAMPAIGN_CONTENT_ROOT = cli._EXPECTED_CAMPAIGN_CONTENT_ROOT


@pytest.fixture(autouse=True)
def _restore_official_campaign_content_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_EXPECTED_CAMPAIGN_CONTENT_ROOT", _OFFICIAL_CAMPAIGN_CONTENT_ROOT)


def _complete_assets(data_path: Path, *, pin_fixture_root: bool = True) -> None:
    (data_path / "omnigibson-robot-assets").mkdir(parents=True)
    (data_path / "omnigibson-robot-assets" / "VERSION").write_text("3.8.2\n", encoding="utf-8")
    (data_path / "behavior-1k-assets").mkdir()
    (data_path / "behavior-1k-assets" / "VERSION").write_text("3.9.0\n", encoding="utf-8")
    tasks = data_path / "2026-challenge-task-instances"
    (tasks / "metadata").mkdir(parents=True)
    manifest = json.loads((Path(__file__).parents[1] / "task-manifest.json").read_text(encoding="utf-8"))
    task_rows = [(task["task_name"], task["source_task_id"]) for task in manifest["tasks"]]
    (tasks / "metadata" / "B100_task_misc.csv").write_text(
        "Task ID,Task,Rooms to inlcude\n" + "".join(f"{task_id},{name},kitchen\n" for name, task_id in task_rows),
        encoding="utf-8",
    )
    for task_name, task_id in task_rows:
        states = tasks / "scene_test" / "public" / f"scene-{task_id}" / "json" / f"scene-{task_id}_task_{task_name}_instances"
        states.mkdir(parents=True)
        for state_id in range(301, 311):
            (states / f"scene-{task_id}_task_{task_name}_0_{state_id}_template-tro_state.json").write_text(
                json.dumps({"task": task_name, "instance": state_id}), encoding="utf-8"
            )
    files = [(tasks / "metadata" / "B100_task_misc.csv"), *sorted((tasks / "scene_test" / "public").rglob("*-tro_state.json"))]
    if pin_fixture_root:
        cli._EXPECTED_CAMPAIGN_CONTENT_ROOT = cli._campaign_content_root(tasks, files)


def _environment(token_file: Path, data_path: Path) -> dict[str, str]:
    return {
        "AUTO_DESTROY": "0",
        "B1K_HF_TOKEN_FILE": str(token_file),
        "CONTAINER_DIGEST": "sha256:" + "a" * 64,
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "OMNIGIBSON_DATA_PATH": str(data_path),
        "TASK_MANIFEST_PATH": str(Path(__file__).parents[1] / "task-manifest.json"),
    }


def test_preflight_requires_private_token_file_and_simulator_assets(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("read-scope-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    data_path = tmp_path / "assets"
    data_path.mkdir()
    _complete_assets(data_path)
    environment = _environment(token_file, data_path)

    result = preflight(environment=environment)

    assert result.token_file == token_file
    assert result.data_path == data_path
    assert result.image_digest == environment["CONTAINER_DIGEST"]


def test_preflight_rejects_auto_destroy_or_world_readable_token_file(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("read-scope-token\n", encoding="utf-8")
    token_file.chmod(0o644)
    data_path = tmp_path / "assets"
    data_path.mkdir()
    environment = _environment(token_file, data_path)
    environment["AUTO_DESTROY"] = "1"

    with pytest.raises(CliError, match="AUTO_DESTROY"):
        preflight(environment=environment)

    environment["AUTO_DESTROY"] = "0"
    with pytest.raises(CliError, match="group or other"):
        preflight(environment=environment)


def test_preflight_does_not_require_or_return_an_environment_secret(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("read-scope-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    data_path = tmp_path / "assets"
    data_path.mkdir()
    _complete_assets(data_path)
    environment = _environment(token_file, data_path)
    environment["HF_TOKEN"] = "must-not-be-consumed"

    result = preflight(environment=environment)

    assert "HF_TOKEN" not in result.to_dict()
    assert os.fspath(token_file) == result.to_dict()["token_file"]


def test_preflight_rejects_an_empty_simulator_asset_directory(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("read-scope-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    data_path = tmp_path / "assets"
    data_path.mkdir()

    with pytest.raises(CliError, match="required VERSION marker"):
        preflight(environment=_environment(token_file, data_path))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda root: (root / "omnigibson-robot-assets" / "VERSION").write_text("3.8.1\n"), "robot assets"),
        (lambda root: (root / "behavior-1k-assets" / "VERSION").write_text("3.9.1\n"), "BEHAVIOR assets"),
        (lambda root: (root / "2026-challenge-task-instances" / "metadata" / "B100_task_misc.csv").unlink(), "B100_task_misc.csv"),
        (lambda root: next((root / "2026-challenge-task-instances" / "scene_test" / "public").rglob("*_310_template-tro_state.json")).unlink(), "public campaign"),
    ],
)
def test_preflight_rejects_stale_or_partial_official_asset_layout(
    tmp_path: Path, mutate: object, message: str
) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("read-scope-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    data_path = tmp_path / "assets"
    _complete_assets(data_path)
    mutate(data_path)  # type: ignore[operator]

    with pytest.raises(CliError, match=message):
        preflight(environment=_environment(token_file, data_path))


def test_preflight_rejects_a_symlinked_public_task_state(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("read-scope-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    data_path = tmp_path / "assets"
    _complete_assets(data_path)
    state = next(
        (data_path / "2026-challenge-task-instances" / "scene_test" / "public").rglob(
            "*_301_template-tro_state.json"
        )
    )
    state.unlink()
    state.symlink_to(data_path / "behavior-1k-assets" / "VERSION")

    with pytest.raises(CliError, match="public campaign"):
        preflight(environment=_environment(token_file, data_path))


def test_preflight_rejects_duplicate_public_task_states(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("read-scope-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    data_path = tmp_path / "assets"
    _complete_assets(data_path)
    state = next(
        (data_path / "2026-challenge-task-instances" / "scene_test" / "public").rglob(
            "*_301_template-tro_state.json"
        )
    )
    duplicate = state.with_name(f"duplicate_{state.name}")
    duplicate.write_text("{}", encoding="utf-8")

    with pytest.raises(CliError, match="public campaign"):
        preflight(environment=_environment(token_file, data_path))


def test_preflight_rejects_tampered_public_state_content(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("read-scope-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    data_path = tmp_path / "assets"
    _complete_assets(data_path)
    state = next((data_path / "2026-challenge-task-instances" / "scene_test" / "public").rglob("*-tro_state.json"))
    state.write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(CliError, match="content root"):
        preflight(environment=_environment(token_file, data_path))


def test_preflight_rejects_all_empty_public_state_fixtures_against_the_production_root(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("read-scope-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    data_path = tmp_path / "assets"
    _complete_assets(data_path, pin_fixture_root=False)
    for state in (data_path / "2026-challenge-task-instances" / "scene_test" / "public").rglob("*-tro_state.json"):
        state.write_text("{}", encoding="utf-8")

    with pytest.raises(CliError, match="content root"):
        preflight(environment=_environment(token_file, data_path))


def test_safe_archive_extraction_rejects_tampering_and_unsafe_members(tmp_path: Path) -> None:
    archive = tmp_path / "asset.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("asset/VERSION", "3.8.2\n")
    expected = cli._file_sha256(archive)
    assert cli._verify_archive_sha256(archive, expected=expected) == archive.resolve()
    snapshot = tmp_path / "snapshot.zip"
    snapshot.symlink_to(archive)
    assert cli._verify_archive_sha256(snapshot, expected=expected) == archive.resolve()
    cli._safe_extract_zip(archive, destination=tmp_path / "extracted")
    archive.write_bytes(b"tampered")
    with pytest.raises(CliError, match="SHA-256"):
        cli._verify_archive_sha256(snapshot, expected=expected)
    snapshot.unlink()
    snapshot.symlink_to(tmp_path / "missing.zip")
    with pytest.raises(CliError, match="SHA-256"):
        cli._verify_archive_sha256(snapshot, expected=expected)

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as writer:
        writer.writestr("../escape", "no")
    with pytest.raises(CliError, match="unsafe member"):
        cli._safe_extract_zip(unsafe, destination=tmp_path / "unsafe-extracted")

    symlink = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink, "w") as writer:
        member = zipfile.ZipInfo("asset/link")
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        writer.writestr(member, "target")
    with pytest.raises(CliError, match="unsafe member"):
        cli._safe_extract_zip(symlink, destination=tmp_path / "symlink-extracted")


def test_official_asset_archives_and_campaign_content_root_are_pinned() -> None:
    assert cli._ASSET_DATASET_REPOSITORY == "behavior-1k/zipped-datasets"
    assert cli._ASSET_DATASET_REVISION == "9f0d57d465726976ed98138d3f8b8ca3e2186775"
    assert [(item.filename, item.sha256) for item in cli._PINNED_ASSET_ARCHIVES] == [
        ("omnigibson-robot-assets-3.8.2.zip", "3d813b2181e0581cf2300a40892de70f8475fe59346ceeea4fb9bf7ff21ce126"),
        ("behavior-1k-assets-3.9.0.zip", "09e9fce600f841dc611aa96c0b1b9f9074f56f0e67898b37b39bd00c38a0095e"),
        ("2026-challenge-task-instances.zip", "d4ac3d72dd585178e85d542f28d1b48233b886c80e196e2f9d3aa5993ba21f81"),
    ]
    assert cli._EXPECTED_CAMPAIGN_CONTENT_ROOT == "14a87fdfeefa04e9d1cc2035bc6eff424ea5f916b84d67b04d141a6366b20aec"


def test_campaign_command_composes_and_runs_the_production_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, str]] = []

    monkeypatch.setattr(cli, "run_campaign", lambda environment=None: calls.append(dict(environment or {})) or None)

    assert cli.main(["campaign"]) == 0
    assert calls == [{}]


def test_gpu_ids_are_never_silently_defaulted() -> None:
    with pytest.raises(CliError, match="GPU_IDS"):
        cli._gpu_ids(None)
    assert cli._gpu_ids("0,2") == (0, 2)


def test_checkpoint_bootstrap_uses_the_exact_final_manifest_materializer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("read-scope-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    data_path = tmp_path / "assets"
    data_path.mkdir()
    _complete_assets(data_path)
    destination = tmp_path / "checkpoint"
    environment = _environment(token_file, data_path) | {
        "HF_MODEL_REPO": MODEL_REPO,
        "MODEL_COMMIT": "a" * 40,
        "CHECKPOINT_ARTIFACT_SHA256": "b" * 64,
        "CHECKPOINT_DIR": str(destination),
    }
    calls: list[tuple[str, str, Path]] = []

    class FakeMaterializer:
        def __init__(self, *, hub: object, expected_manifest_sha256: str) -> None:
            assert expected_manifest_sha256 == environment["CHECKPOINT_ARTIFACT_SHA256"]
            assert hub == "hub"

        def download(self, *, repository: str, revision: str, destination: Path) -> CheckpointReceipt:
            calls.append((repository, revision, destination))
            destination.mkdir()
            return CheckpointReceipt(revision, environment["CHECKPOINT_ARTIFACT_SHA256"], destination.resolve())

    monkeypatch.setattr(cli, "HuggingFaceModelAdapter", lambda token_file: "hub", raising=False)
    monkeypatch.setattr(cli, "FinalPolicyMaterializer", FakeMaterializer, raising=False)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: pytest.fail("broad snapshot download"))

    assert cli.bootstrap_checkpoint(environment=environment) == destination.resolve()
    assert calls == [(MODEL_REPO, environment["MODEL_COMMIT"], destination)]
