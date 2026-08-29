"""Contract checks for the approved one-VM collection operator handoff."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = ROOT / "docs" / "experiments" / "2026-08-27-simple-curriculum-runbook.md"


def _paid_command(text: str) -> str:
    match = re.search(r"## 5\. One paid command.*?```bash\n(.*?)```", text, re.S)
    assert match, "the runbook must have an executable paid-command fence"
    return match.group(1)


def test_simple_curriculum_runbook_covers_the_exact_paid_boundary_without_lifecycle_commands() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    required = (
        "computeinstance-u00t6xfqhadrcmssa2",
        "30ac1a84da67b099e115ad147bcd61e9d60046d3",
        "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
        "40 seen garments", "10 per category", "400", "600", "400 replay", "200 accepted",
        "four persistent workers", "CPU cloth", "CUDA policy", "$100", "first-100",
        "collection-rounds/<run-id>", "authenticated", "anonymous", "OPERATOR_HF_TOKEN_FILE",
        "fidelity_infrastructure_stop", "insufficient_fresh_source", "collection_complete",
            "operator-stop-handoff.json", "run_simple_curriculum_with_finalizer.sh", "operator-owned local 0600 token path",
    )
    assert all(value in text for value in required)
    assert "A-500" in text and "hard-state" in text and "old rollout" in text


def test_runbook_has_no_removed_remote_stop_hook_or_controller_publication_language() -> None:
    text = RUNBOOK.read_text(encoding="utf-8").lower()
    for stale in ("trusted hook", "trusted terminal gpu stop", "stop-hook failure", "controller polls the typed spend receipt before and after every paid stage\nand while a child runs. it writes durable budget state. it does not pass the\nbudget check to final public publication"):
        assert stale not in text
    assert "run_simple_curriculum_with_finalizer.sh" in text
    assert "never create an image or vm" in text
    assert "no secret value belongs in this runbook" in text
    assert not re.search(r"(?:hf_|api_|access_)?token\s*=\s*['\"][^<'\"\s]{8,}", text, re.I)
    assert not re.search(r"(?:nebius|terraform).*\b(?:create|delete|destroy|start)\b", text, re.I)


def test_simple_curriculum_runbook_has_clean_environment_and_staged_catalog_contract() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    paid = _paid_command(text)

    assert "LEHOME_OPERATOR_SSH_TARGET=" in paid
    assert "LEHOME_OPERATOR_REVIEWED_REVISION=" in paid
    assert "<" not in paid and ">" not in paid
    assert 'LEHOME_OPERATOR_RUN_ID="$LEHOME_RUN_ID"' in paid
    assert 'LEHOME_OPERATOR_ROUND_ID="$LEHOME_ROUND_ID"' in paid
    assert 'test ! -e "$LEHOME_CAMPAIGN_ROOT"' in text
    assert "configs/eval_groot_n17_public_280.json" in text
    assert "catalog-source.sha256" in text and "seen-catalog.sha256" in text
    assert "LEHOME_REVIEWED_REVISION=%q" in text and "LEHOME_CAMPAIGN_ROOT=%q" in text
    assert 'test "${LEHOME_CAMPAIGN_ROOT:-}" = "/mnt/lehome/eval/$LEHOME_RUN_ID"' in text
    assert "40 unique seen garments, 10 per category" in text
    assert "do not copy an old rollout input" in text
    assert "same IDs and exact command are reused rather than regenerated" in text
    assert "run_conservative_spend_observer.py" in text
    assert "LEHOME_SPEND_BASELINE_USD=20.25" in text
    assert "LEHOME_SPEND_BASELINE_AT_UTC=2026-08-28T14:25:00Z" in text
    assert "LEHOME_MAX_HOURLY_BURN_USD=1.50" in text
    assert "LEHOME_SPEND_OBSERVER_COMMAND" in text
    assert "--interval-seconds 30" in text
    assert "trap cleanup_spend_observer EXIT INT TERM" in text
    assert "LEHOME_OPERATOR_HF_TOKEN_FILE=" in paid


def test_runbook_executes_the_paid_controller_only_through_the_operator_wrapper() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    paid = _paid_command(text)
    section_seven = text.split("## 7. Fresh terminal evidence, replay, and publication", 1)[1]

    assert "./scripts/run_simple_curriculum_with_finalizer.sh" in paid
    assert "sudo env -i" not in paid
    assert "run_simple_curriculum_collection.sh" not in paid
    assert "finalize_simple_curriculum_collection.py" not in section_seven
    assert "trap 'finalize" not in section_seven
    assert ". \"$LEHOME_INVOCATION_FILE\"" not in text


def test_simple_curriculum_runbook_separates_remote_root_token_from_local_operator_token() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert "per-episode sync" in text
    assert "local operator finalizer" in text
    assert "test -f /mnt/lehome/secrets/hf_token && test ! -L /mnt/lehome/secrets/hf_token" in text
    assert "test \"$(stat -c '%u' /mnt/lehome/secrets/hf_token)\" = 0" in text
    assert "test \"$(stat -c '%a' /mnt/lehome/secrets/hf_token)\" = 600" in text
    assert "test -s /mnt/lehome/secrets/hf_token" in text


def test_simple_curriculum_runbook_orders_provider_start_and_post_start_checkpoints() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert text.index("## 1. Provider preflight while stopped") < text.index("## 2. Start exactly the approved VM")
    assert text.index("## 2. Start exactly the approved VM") < text.index("## 3. Post-start checkpoint and immutable input staging")
    assert "cloud-init status --wait" in text
    assert "mountpoint -q /mnt/lehome" in text
    assert "nvidia-smi --query-gpu" in text
    assert "Do not attempt the post-start checks on a stopped VM." in text


def test_controller_help_imports_under_clean_environment() -> None:
    source_paths = os.pathsep.join((str(ROOT), str(ROOT / "source" / "lehome"), str(ROOT / "trainer" / "src")))
    result = subprocess.run(
        (sys.executable, str(ROOT / "scripts" / "run_simple_curriculum_collection.py"), "--help"),
        cwd=ROOT,
        env={"PATH": os.environ["PATH"], "PYTHONPATH": source_paths},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--campaign-root" in result.stdout


def test_operator_wrapper_bounds_established_ssh_session_before_exit_finalizer() -> None:
    wrapper = (ROOT / "scripts" / "run_simple_curriculum_with_finalizer.sh").read_text(encoding="utf-8")

    assert "ServerAliveInterval=" in wrapper
    assert "ServerAliveCountMax=" in wrapper
    assert "LEHOME_OPERATOR_SSH_SESSION_TIMEOUT_SECONDS" in wrapper
    assert "subprocess.run" in wrapper and "timeout=" in wrapper


def test_operator_wrapper_session_deadline_reaches_exit_finalizer(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir(); log = tmp_path / "log"
    for name, body in {
        "stat": "case \"$1:$2\" in -f:%u) /usr/bin/id -u;; -f:%Lp) echo 600;; *) exit 9;; esac",
        "uv": f"echo finalizer >> {log}; exit 0",
        "ssh": "sleep 10",
    }.items():
        executable = fake_bin / name
        executable.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
        executable.chmod(0o755)
    token = tmp_path / "token"; token.write_text("token", encoding="utf-8")
    run_id = "fresh-run-20260828123456-timeout"; round_id = "fresh-12k-20260828123456-timeout"
    started = time.monotonic()
    result = subprocess.run(
        (
            "env", "-i", f"PATH={fake_bin}:/usr/bin:/bin",
            "LEHOME_OPERATOR_SSH_TARGET=operator@host",
            f"LEHOME_OPERATOR_CAMPAIGN_ROOT=/mnt/lehome/eval/{run_id}",
            f"LEHOME_OPERATOR_RUN_ID={run_id}", f"LEHOME_OPERATOR_ROUND_ID={round_id}",
            f"LEHOME_OPERATOR_REVIEWED_REVISION={'a' * 40}",
            f"LEHOME_OPERATOR_HF_TOKEN_FILE={token}",
            "LEHOME_OPERATOR_SSH_SESSION_TIMEOUT_SECONDS=0.1",
            "/bin/bash", str(ROOT / "scripts/run_simple_curriculum_with_finalizer.sh"),
        ),
        text=True, capture_output=True, timeout=5,
    )
    assert result.returncode == 1
    assert time.monotonic() - started < 3
    assert log.read_text(encoding="utf-8").splitlines() == ["finalizer"]
