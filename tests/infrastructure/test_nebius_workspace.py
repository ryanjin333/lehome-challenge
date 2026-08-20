"""Guest shared-disk contract: mount, UUID admission, and role leases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

GUEST_DIR = Path(__file__).resolve().parents[2] / "infrastructure" / "nebius" / "guest"
sys.path.insert(0, str(GUEST_DIR))

from lehome_workspace import (  # noqa: E402
    MANIFEST_NAME,
    MANIFEST_SCHEMA_VERSION,
    WorkspaceConfig,
    WorkspaceError,
    CommandResult,
    acquire_role,
    blkid_uuid,
    ensure_layout,
    format_blank_disk,
    initialize_manifest,
    load_manifest,
    manifest_path,
    mount_workspace,
    read_mount_table,
    release_role,
    write_manifest_atomic,
)


UUID = "123e4567-e89b-12d3-a456-426614174000"


class FakeRunner:
    def __init__(self, blkid_exit: int = 0, blkid_stdout: str = UUID):
        self.blkid_exit = blkid_exit
        self.blkid_stdout = blkid_stdout
        self.commands: list[tuple[str, ...]] = []
        self.mount_exit = 0

    def __call__(self, command):
        self.commands.append(tuple(command))
        if command[0] == "blkid":
            return CommandResult(self.blkid_exit, self.blkid_stdout, "")
        if command[0] == "mount":
            return CommandResult(self.mount_exit, "", "" if self.mount_exit == 0 else "mount failed")
        return CommandResult(0, "", "")


@pytest.fixture()
def config(tmp_path):
    return WorkspaceConfig(device="/dev/vdb", role="training", run_id="run-1", mount_point=str(tmp_path / "lehome"))


def test_config_rejects_bad_role_or_device():
    with pytest.raises(WorkspaceError):
        WorkspaceConfig(device="/dev/vdb", role="eval", run_id="r")
    with pytest.raises(WorkspaceError):
        WorkspaceConfig(device="vdb", role="training", run_id="r")
    with pytest.raises(WorkspaceError):
        WorkspaceConfig(device="/dev/vdb", role="training", run_id="", )
    with pytest.raises(WorkspaceError):
        WorkspaceConfig(device="/dev/vdb", role="training", run_id="r", expected_uuid="not-a-uuid")


def test_blkid_parses_uuid_and_blank_disk():
    assert blkid_uuid("/dev/vdb", FakeRunner()) == UUID
    assert blkid_uuid("/dev/vdb", FakeRunner(blkid_exit=2, blkid_stdout="")) is None


def test_blkid_failure_fails_closed():
    runner = FakeRunner(blkid_exit=1)
    runner.blkid_stdout = ""
    with pytest.raises(WorkspaceError, match="blkid failed"):
        blkid_uuid("/dev/vdb", runner)


def test_format_refuses_nonblank_disk():
    runner = FakeRunner()  # blkid reports an existing filesystem
    with pytest.raises(WorkspaceError, match="already carries a filesystem"):
        format_blank_disk("/dev/vdb", UUID, runner)


def test_format_blank_disk_pins_uuid():
    runner = FakeRunner(blkid_exit=2, blkid_stdout="")
    format_blank_disk("/dev/vdb", UUID.upper(), runner)
    mkfs = [c for c in runner.commands if c[0] == "mkfs.ext4"]
    assert mkfs == [("mkfs.ext4", "-q", "-U", UUID, "/dev/vdb")]


def test_mount_at_lehome_with_conflict_refusal(tmp_path, config):
    runner = FakeRunner()
    mount_workspace(config, runner, "proc /proc proc rw\n")
    mount = [c for c in runner.commands if c[0] == "mount"]
    assert mount == [("mount", "-o", "noatime", "/dev/vdb", config.mount_point)]

    with pytest.raises(WorkspaceError, match="occupied"):
        mount_workspace(config, runner, f"/dev/vdc {config.mount_point} ext4 rw\n")

    other = WorkspaceConfig(device="/dev/vdb", role="training", run_id="run-1", mount_point=str(tmp_path / "elsewhere"))
    with pytest.raises(WorkspaceError, match="already mounted"):
        mount_workspace(other, runner, f"/dev/vdb {config.mount_point} ext4 rw\n")


def test_idempotent_remount_of_same_target(config):
    runner = FakeRunner()
    mount_workspace(config, runner, f"/dev/vdb {config.mount_point} ext4 rw\n")
    assert not [c for c in runner.commands if c[0] == "mount"]


def test_read_mount_table_skips_virtual_filesystems():
    table = read_mount_table("proc /proc proc rw\ntmpfs /tmp tmpfs rw\n/dev/vda1 / ext4 rw\n")
    assert table == {"/dev/vda1": "/"}


def test_initialize_manifest_atomic_and_single_lease(config, tmp_path):
    (tmp_path / "lehome").mkdir()
    clock = iter(range(100, 200))
    manifest = initialize_manifest(config, UUID, lambda: next(clock))
    path = manifest_path(config)
    assert path.name == MANIFEST_NAME
    on_disk = json.loads(path.read_text())
    assert on_disk == manifest
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["disk_uuid"] == UUID
    assert manifest["active_role"] == "training"
    assert manifest["run_id"] == "run-1"
    assert not list(path.parent.glob(".*.tmp")), "no temporary file may survive an atomic write"

    with pytest.raises(WorkspaceError, match="reinitialize"):
        initialize_manifest(config, UUID, lambda: 1)


def test_write_manifest_refuses_nan(config, tmp_path):
    (tmp_path / "lehome").mkdir()
    with pytest.raises(ValueError):
        write_manifest_atomic(manifest_path(config), {"bad": float("nan")})


def test_acquire_role_rejects_conflicting_role_and_wrong_disk(config, tmp_path):
    (tmp_path / "lehome").mkdir()
    initialize_manifest(config, UUID, lambda: 1)

    rollout = WorkspaceConfig(device="/dev/vdb", role="rollout", run_id="run-1", mount_point=config.mount_point)
    with pytest.raises(WorkspaceError, match="role lease conflict"):
        acquire_role(rollout, UUID, lambda: 2)

    wrong_disk = WorkspaceConfig(device="/dev/vdb", role="training", run_id="run-1", mount_point=config.mount_point)
    with pytest.raises(WorkspaceError, match="disk_uuid"):
        acquire_role(wrong_disk, "999e4567-e89b-12d3-a456-426614174999", lambda: 3)

    wrong_run = WorkspaceConfig(device="/dev/vdb", role="training", run_id="run-2", mount_point=config.mount_point)
    with pytest.raises(WorkspaceError, match="run identity mismatch"):
        acquire_role(wrong_run, UUID, lambda: 4)


def test_reacquire_same_role_and_clean_handoff(config, tmp_path):
    (tmp_path / "lehome").mkdir()
    initialize_manifest(config, UUID, lambda: 1)
    reacquired = acquire_role(config, UUID, lambda: 5)
    assert reacquired["active_role"] == "training"
    assert reacquired["role_lease_at_ns"] == 5

    released = release_role(config, lambda: 9)
    assert released["active_role"] is None
    assert released["last_clean_handoff_ns"] == 9

    rollout = WorkspaceConfig(device="/dev/vdb", role="rollout", run_id="run-1", mount_point=config.mount_point)
    acquired = acquire_role(rollout, UUID, lambda: 10)
    assert acquired["active_role"] == "rollout"

    training_again = WorkspaceConfig(device="/dev/vdb", role="training", run_id="run-1", mount_point=config.mount_point)
    with pytest.raises(WorkspaceError, match="role lease conflict"):
        acquire_role(training_again, UUID, lambda: 11)


def test_release_role_requires_current_holder(config, tmp_path):
    (tmp_path / "lehome").mkdir()
    initialize_manifest(config, UUID, lambda: 1)
    rollout = WorkspaceConfig(device="/dev/vdb", role="rollout", run_id="run-1", mount_point=config.mount_point)
    with pytest.raises(WorkspaceError, match="cannot release"):
        release_role(rollout, lambda: 2)


def test_release_role_requires_the_original_run_identity(config, tmp_path):
    (tmp_path / "lehome").mkdir()
    initialize_manifest(config, UUID, lambda: 1)
    replacement = WorkspaceConfig(
        device="/dev/vdb", role="training", run_id="replacement-run", mount_point=config.mount_point,
    )
    with pytest.raises(WorkspaceError, match="run identity mismatch"):
        release_role(replacement, lambda: 2)


def test_release_workspace_never_mounts_or_formats(config, tmp_path):
    from lehome_workspace import release_workspace

    (tmp_path / "lehome").mkdir()
    initialize_manifest(config, UUID, lambda: 1)
    runner = FakeRunner()
    released = release_workspace(config, runner, lambda: 2)
    assert released["active_role"] is None
    assert [command[0] for command in runner.commands] == ["blkid"]


def test_load_manifest_rejects_symlink(config, tmp_path):
    (tmp_path / "lehome").mkdir()
    target = tmp_path / "elsewhere.json"
    target.write_text("{}")
    link = Path(config.mount_point) / MANIFEST_NAME
    link.symlink_to(target)
    with pytest.raises(WorkspaceError, match="missing or unsafe"):
        load_manifest(link)


def test_ensure_layout_creates_durable_tree(config, tmp_path):
    (tmp_path / "lehome").mkdir()
    created = ensure_layout(config)
    assert (Path(config.mount_point) / "datasets" / "bc" / "full").is_dir()
    assert (Path(config.mount_point) / "rollouts" / "accepted").is_dir()
    assert (Path(config.mount_point) / "ledgers").is_dir()
    assert len(created) == 15



class BlankThenFormattedRunner(FakeRunner):
    def __init__(self):
        super().__init__(blkid_exit=2, blkid_stdout="")
        self._formatted = False

    def __call__(self, command):
        self.commands.append(tuple(command))
        if command[0] == "mkfs.ext4":
            self._formatted = True
            return CommandResult(0, "", "")
        if command[0] == "blkid":
            if self._formatted:
                return CommandResult(0, UUID, "")
            return CommandResult(2, "", "")
        return CommandResult(0, "", "")


def test_admit_workspace_formats_blank_then_leases(config, tmp_path):
    from lehome_workspace import admit_workspace

    runner = BlankThenFormattedRunner()
    seen = {"n": 0}

    def clock():
        seen["n"] += 1
        return seen["n"]

    Path(config.mount_point).mkdir()
    manifest = admit_workspace(config, runner, clock, mount_table_text="")
    assert manifest["active_role"] == "training"
    assert manifest["disk_uuid"] == UUID
    assert any(command[0] == "mkfs.ext4" for command in runner.commands)
    assert (Path(config.mount_point) / "datasets" / "bc" / "full").is_dir()


def test_workspace_cli_requires_role_and_run_id():
    from lehome_workspace import main
    import pytest

    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    with pytest.raises(SystemExit) as missing_exit:
        main([])
    assert missing_exit.value.code == 2


def test_workspace_unit_waits_for_cloud_init_runtime_env():
    unit = (Path(__file__).resolve().parents[2] / "infrastructure" / "nebius" / "guest" / "systemd" / "lehome-workspace.service").read_text(encoding="utf-8")
    assert "cloud-init.service" in unit
    assert "EnvironmentFile=-/etc/lehome/runtime.env" in unit
    assert "missing /etc/lehome/runtime.env" in unit
