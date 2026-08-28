"""Executable transport contracts for immutable one-VM runtime staging."""

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts" / "stage_simple_curriculum_runtime_code.sh"


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    for directory in ("source/lehome", "trainer/src", "scripts", "rollout_appliance", "configs"):
        (repo / directory).mkdir(parents=True, exist_ok=True)
    (repo / "configs/eval_groot_n17_public_280.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q", str(repo)), check=True)
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
    subprocess.run(("git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"), check=True)
    return repo


def _fakes(tmp_path: Path) -> tuple[Path, Path]:
    fake = tmp_path / "fake"; fake.mkdir(); log = tmp_path / "transport.log"
    (fake / "ssh").write_text(
        "#!/usr/bin/env bash\nset -eu\nprintf 'ssh %s\\n' \"$*\" >> \"$LEHOME_TEST_LOG\"\n"
        "case \"$*\" in *mktemp*) printf '%s\\n' /mnt/lehome/runtime-code/.runtime-code-stage.Abcdef12 ;; *'bash -s'*) cat > \"$LEHOME_TEST_REMOTE_SCRIPT\"; printf '{\"schema_version\":1,\"status\":\"%s\"}\\n' \"${LEHOME_TEST_MODE:-staged}\" ;; esac\n",
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
    env = {**os.environ, **extra, "PATH": f"{fake}:{os.environ['PATH']}", "LEHOME_TEST_LOG": str(log), "LEHOME_TEST_REMOTE_SCRIPT": str(log.with_suffix('.remote'))}
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
    remote = log.with_suffix('.remote').read_text(encoding="utf-8")
    assert 'mv -T "$stage/checkout" "$final"' in remote
    assert 'runtime code final collision is not exact' in remote
    assert 'if mv -T "$stage/checkout" "$final"; then status=staged; else verify_final "$final"' in remote
    assert 'configs/eval_groot_n17_public_280.json' in remote


def test_scp_failure_cleans_only_the_allocated_remote_stage(tmp_path: Path) -> None:
    repo = _repo(tmp_path); fake, log = _fakes(tmp_path)
    result = _run(repo, fake, log, LEHOME_TEST_SCP_FAIL="1")

    assert result.returncode == 71
    transport = log.read_text(encoding="utf-8")
    assert ".runtime-code-stage.Abcdef12" in transport
    assert "rm -rf -- '/mnt/lehome/runtime-code/.runtime-code-stage.Abcdef12'" in transport


def test_source_contract_uses_exact_types_and_no_explicit_credential_file() -> None:
    text = HELPER.read_text(encoding="utf-8")
    assert "--identity-file" not in text
    assert "test -d \"$required\" && test ! -L \"$required\"" in text
    assert "test -f configs/eval_groot_n17_public_280.json && test ! -L" in text
    assert "[[ \"$revision\" =~ ^[0-9a-f]{40}$ ]]" in text
    assert "[[ \"$bundle_sha256\" =~ ^[0-9a-f]{64}$ ]]" in text


def test_runbook_carries_the_current_staged_revision_without_a_pinned_literal() -> None:
    text = (ROOT / "docs/experiments/2026-08-27-simple-curriculum-runbook.md").read_text(encoding="utf-8")
    assert 'LEHOME_REVIEWED_REVISION="$(git rev-parse HEAD)"' in text
    assert '"${LEHOME_REVIEWED_REVISION:?carry the staged revision from the operator command}"' in text
    assert "551a85e2105003365a7c33a26af1d5f5924181d7" not in text
