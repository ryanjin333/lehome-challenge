"""Static contract tests for immutable one-VM runtime-code staging."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts" / "stage_simple_curriculum_runtime_code.sh"


def test_runtime_code_stager_uses_a_bundle_detached_checkout_and_atomic_final_path() -> None:
    text = HELPER.read_text(encoding="utf-8")

    assert "git diff --quiet" in text
    assert "git status --porcelain" in text
    assert "git bundle create" in text
    assert "git -C \"$tmp/repository\" bundle verify" in text
    assert "git -C \"$tmp/checkout\" checkout --detach" in text
    assert "base=/mnt/lehome/runtime-code; final=\"$base/$revision\"" in text
    assert "mv \"$tmp/checkout\" \"$final\"" in text
    assert "ClearAllForwardings=yes" in text
    assert "configs/eval_groot_n17_public_280.json" in text
    assert "rm -rf \"$tmp\"" in text
    assert "nebius" not in text.lower()
    assert "terraform" not in text.lower()
    assert "hf_token" not in text.lower()


def test_runtime_code_stager_requires_nonsecret_transport_inputs() -> None:
    text = HELPER.read_text(encoding="utf-8")

    assert "--ssh-target" in text
    assert "--ssh-port" in text
    assert "--identity-file" in text
    assert "printf '{\"schema_version\":1" in text
    assert "remote bundle digest mismatch" in text
    assert "runtime code final path already exists but is not exact" in text
