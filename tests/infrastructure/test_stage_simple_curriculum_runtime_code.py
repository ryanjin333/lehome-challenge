"""Executable transport contracts for immutable one-VM runtime staging."""

import os
from pathlib import Path
import subprocess
import sys
import importlib.util
import hashlib
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts" / "stage_simple_curriculum_runtime_code.sh"
REMOTE_HELPER = ROOT / "scripts" / "stage_runtime_code_remote.py"


def _remote_module():
    spec = importlib.util.spec_from_file_location("stage_runtime_code_remote", REMOTE_HELPER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    for directory in ("source/lehome", "trainer/src", "scripts", "rollout_appliance", "configs"):
        (repo / directory).mkdir(parents=True, exist_ok=True)
    for directory in ("source/lehome", "trainer/src", "scripts", "rollout_appliance"):
        (repo / directory / ".fixture").write_text("fixture\n", encoding="utf-8")
    shutil.copy2(REMOTE_HELPER, repo / "scripts" / "stage_runtime_code_remote.py")
    (repo / "configs/eval_groot_n17_public_280.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q", str(repo)), check=True)
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
    subprocess.run(("git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"), check=True)
    return repo


def _fakes(tmp_path: Path) -> tuple[Path, Path]:
    fake = tmp_path / "fake"; fake.mkdir(); log = tmp_path / "transport.log"
    (fake / "ssh").write_text(
        "#!/usr/bin/env bash\nset -eu\nprintf 'ssh %s\\n' \"$*\" >> \"$LEHOME_TEST_LOG\"\n"
        "if [ \"${LEHOME_TEST_MOUNT_FAIL:-0}\" = 1 ] && [[ \"$*\" = *mountpoint* ]]; then exit 72; fi\ncase \"$*\" in *mktemp*) printf '%s\\n' /mnt/lehome/runtime-code/.runtime-code-stage.Abcdef12 ;; *stage_runtime_code_remote.py*) printf '{\"schema_version\":1,\"status\":\"%s\"}\\n' \"${LEHOME_TEST_MODE:-staged}\" ;; esac\n",
        encoding="utf-8",
    )
    (fake / "scp").write_text(
        "#!/usr/bin/env bash\nset -eu\nprintf 'scp %s\\n' \"$*\" >> \"$LEHOME_TEST_LOG\"\nif [ \"${LEHOME_TEST_SCP_FAIL:-0}\" = 1 ]; then exit 71; fi\n",
        encoding="utf-8",
    )
    (fake / "git").write_text(
        "#!/usr/bin/env bash\nif [ \"${1:-}\" = rev-parse ] && [ \"${2:-}\" = HEAD ] && [ -n \"${LEHOME_TEST_REVISION:-}\" ]; then printf '%s\\n' \"$LEHOME_TEST_REVISION\"; exit 0; fi\nexec /usr/bin/git \"$@\"\n",
        encoding="utf-8",
    )
    for path in fake.iterdir(): path.chmod(0o755)
    return fake, log


def _run(repo: Path, fake: Path, log: Path, **extra: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **extra, "PATH": f"{fake}:{os.environ['PATH']}", "LEHOME_TEST_LOG": str(log)}
    return subprocess.run(("bash", str(HELPER), "--ssh-target", "operator@example"), cwd=repo, env=env, text=True, capture_output=True, check=False)


def test_stager_rejects_invalid_transport_and_revision_lengths(tmp_path: Path) -> None:
    assert subprocess.run(("bash", str(HELPER), "--ssh-target", "bad target"), text=True, capture_output=True).returncode == 2
    repo = _repo(tmp_path); fake, log = _fakes(tmp_path)
    for revision in ("a" * 39, "a" * 41, "g" * 40):
        result = _run(repo, fake, log, LEHOME_TEST_REVISION=revision)
        assert result.returncode == 2
        assert "exact HEAD" in result.stderr


@pytest.mark.parametrize("mode", ("happy", "existing", "collision"))
def test_stager_uses_agent_transport_and_carries_no_clobber_remote_contract(tmp_path: Path, mode: str) -> None:
    repo = _repo(tmp_path); fake, log = _fakes(tmp_path)
    result = _run(repo, fake, log, LEHOME_TEST_MODE=mode)

    assert result.returncode == 0, result.stderr + "\n" + (log.read_text(encoding="utf-8") if log.exists() else "no transport log")
    assert '"schema_version":1' in result.stdout
    assert f'"status":"{mode}"' in result.stdout
    transport = log.read_text(encoding="utf-8")
    assert "ClearAllForwardings=yes" in transport
    assert "-i " not in transport
    assert "stage_runtime_code_remote.py" in transport
    assert "LEHOME_REVIEWED_REVISION=" in transport


def test_scp_failure_cleans_only_the_allocated_remote_stage(tmp_path: Path) -> None:
    repo = _repo(tmp_path); fake, log = _fakes(tmp_path)
    result = _run(repo, fake, log, LEHOME_TEST_SCP_FAIL="1")

    assert result.returncode == 71
    transport = log.read_text(encoding="utf-8")
    assert ".runtime-code-stage.Abcdef12" in transport
    assert "rm -rf -- '/mnt/lehome/runtime-code/.runtime-code-stage.Abcdef12'" in transport


def test_mount_missing_stops_before_remote_stage_allocation_or_copy(tmp_path: Path) -> None:
    repo = _repo(tmp_path); fake, log = _fakes(tmp_path)
    result = _run(repo, fake, log, LEHOME_TEST_MOUNT_FAIL="1")

    assert result.returncode == 72
    transport = log.read_text(encoding="utf-8")
    assert transport.count("ssh ") == 1
    assert "scp " not in transport and "mktemp" not in transport


def test_source_contract_uses_exact_types_and_no_explicit_credential_file() -> None:
    text = HELPER.read_text(encoding="utf-8")
    assert "--identity-file" not in text
    assert "test -d \"$required\" && test ! -L \"$required\"" in text
    assert "test -f configs/eval_groot_n17_public_280.json && test ! -L" in text
    assert "[[ \"$revision\" =~ ^[0-9a-f]{40}$ ]]" in text
    assert "--bundle-sha256 '$bundle_sha256'" in text
    assert 'bundle="$bundle_dir/code.bundle"' in text
    assert "_SHA256.fullmatch(bundle_sha256)" in REMOTE_HELPER.read_text(encoding="utf-8")
    assert "mountpoint -q /mnt/lehome && mkdir -p" in text


@pytest.mark.skipif(sys.platform != "linux" or os.uname().machine != "x86_64", reason="renameat2 test requires Linux x86_64")
def test_remote_promotion_is_no_replace_for_absent_empty_and_nonempty_finals(tmp_path: Path) -> None:
    module = _remote_module(); source = _repo(tmp_path / "repo")
    revision = subprocess.run(("git", "-C", str(source), "rev-parse", "HEAD"), text=True, capture_output=True, check=True).stdout.strip()
    bundle = tmp_path / "source.bundle"; subprocess.run(("git", "-C", str(source), "bundle", "create", str(bundle), "HEAD"), check=True)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest(); base = tmp_path / "runtime-code"; base.mkdir()

    def stage(name: str) -> Path:
        path = base / name; path.mkdir(); (path / "code.bundle").write_bytes(bundle.read_bytes()); return path

    first = module.stage_runtime_code(revision=revision, bundle_sha256=digest, stage=stage(".runtime-code-stage.Abcdef12"), base=base, check_mount=False)
    final = base / revision
    assert first["status"] == "staged" and final.is_dir() and not (base / ".runtime-code-stage.Abcdef12").exists()
    original = (final / "configs/eval_groot_n17_public_280.json").read_bytes()

    empty = base / ("b" * 40); empty.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        module.stage_runtime_code(revision="b" * 40, bundle_sha256=digest, stage=stage(".runtime-code-stage.Bbcdef12"), base=base, check_mount=False)
    assert empty.is_dir() and not any(empty.iterdir())
    assert (final / "configs/eval_groot_n17_public_280.json").read_bytes() == original
    nonempty = base / ("c" * 40); nonempty.mkdir(); (nonempty / "keep").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        module.stage_runtime_code(revision="c" * 40, bundle_sha256=digest, stage=stage(".runtime-code-stage.Cbcdef12"), base=base, check_mount=False)
    assert (nonempty / "keep").read_text(encoding="utf-8") == "keep"


def test_remote_mount_failure_occurs_before_any_git_or_stage_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _remote_module(); base = tmp_path / "runtime-code"; base.mkdir(); stage = base / ".runtime-code-stage.Abcdef12"; stage.mkdir()
    calls: list[tuple[str, ...]] = []
    def fail_mount(*argv: str) -> None:
        calls.append(argv); raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(module, "_run", fail_mount)

    with pytest.raises(subprocess.CalledProcessError):
        module.stage_runtime_code(revision="a" * 40, bundle_sha256="b" * 64, stage=stage, base=base, check_mount=True)
    assert calls == [("mountpoint", "-q", "/mnt/lehome")]
    assert stage.is_dir() and not list(stage.iterdir())


def test_runbook_carries_the_current_staged_revision_without_a_pinned_literal() -> None:
    text = (ROOT / "docs/experiments/2026-08-27-simple-curriculum-runbook.md").read_text(encoding="utf-8")
    assert 'LEHOME_REVIEWED_REVISION="$(git rev-parse HEAD)"' in text
    assert '"${LEHOME_REVIEWED_REVISION:?carry the staged revision from the operator command}"' in text
    assert "export LEHOME_REVIEWED_REVISION='$LEHOME_REVIEWED_REVISION'; exec bash -l" in text
    assert "remaining Section 3 block and the paid command in this same shell" in text
    assert text.index("mountpoint -q /mnt/lehome") < text.index("stage_simple_curriculum_runtime_code.sh")
    assert "551a85e2105003365a7c33a26af1d5f5924181d7" not in text
