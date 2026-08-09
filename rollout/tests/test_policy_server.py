from __future__ import annotations

from pathlib import Path

from b1k_rollout.policy_server import build_command


def test_policy_launcher_execs_only_the_pinned_upstream_b1k_r1pro_server(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.safetensors").write_bytes(b"fixture")

    assert build_command(checkpoint=checkpoint, host="127.0.0.1", port=8000) == (
        "/opt/isaac-groot/.venv/bin/python", "/opt/isaac-groot/scripts/b1k/serve_b1k.py",
        "--model-path", str(checkpoint),
        "--modality-config-path", "/opt/isaac-groot/examples/b1k/r1pro.py",
        "--embodiment-tag", "NEW_EMBODIMENT", "--host", "127.0.0.1", "--port", "8000",
    )
