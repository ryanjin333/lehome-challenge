"""Contract checks for the approved one-VM collection operator handoff."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


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
        "collection-rounds/<run-id>", "authenticated", "anonymous", "--reconcile",
        "fidelity_infrastructure_stop", "insufficient_fresh_source", "collection_complete",
        "/usr/local/libexec/lehome-stop-gpu", "LEHOME_PAID_COLLECTION=1",
    )
    assert all(value in text for value in required)
    assert "A-500" in text and "hard-state" in text and "old rollout" in text
    assert "never create an image or VM" in text
    assert "No secret value belongs in this runbook" in text
    assert not re.search(r"(?:hf_|api_|access_)?token\s*=\s*['\"][^<'\"\s]{8,}", text, re.I)
    assert not re.search(r"(?:nebius|terraform).*\b(?:create|delete|destroy|start)\b", text, re.I)


def test_simple_curriculum_runbook_has_clean_environment_and_staged_catalog_contract() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    paid = _paid_command(text)

    assert "PYTHONPATH=\"$LEHOME_HOST_CODE_ROOT:$LEHOME_HOST_CODE_ROOT/source/lehome:$LEHOME_HOST_CODE_ROOT/trainer/src\"" in paid
    assert "<" not in paid and ">" not in paid
    assert 'LEHOME_RUN_ID="$LEHOME_RUN_ID"' in paid
    assert 'LEHOME_ROUND_ID="$LEHOME_ROUND_ID"' in paid
    assert 'test ! -e "$LEHOME_CAMPAIGN_ROOT"' in text
    assert "configs/eval_groot_n17_public_280.json" in text
    assert "catalog-source.sha256" in text and "seen-catalog.sha256" in text
    assert "40 unique seen garments, 10 per category" in text
    assert "do not copy an old rollout input" in text
    assert "same IDs and exact command are reused rather than regenerated" in text


def test_simple_curriculum_runbook_requires_root_owned_token_for_root_paid_process() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    paid = _paid_command(text)

    assert "sudo env -i" in paid
    assert "publisher runs as root through `sudo env -i`" in text
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
