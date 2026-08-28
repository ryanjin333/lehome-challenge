"""Contract checks for the approved one-VM collection operator handoff."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = ROOT / "docs" / "experiments" / "2026-08-27-simple-curriculum-runbook.md"


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
