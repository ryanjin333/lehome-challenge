from pathlib import Path
import json

import pytest

from lehome_train.groot.continuous_training import run_continuous_supervisor, snapshot_checkpoint


def test_observer_never_packages_checkpoint_without_completion_marker(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="complete checkpoint"):
        snapshot_checkpoint(checkpoint, optimizer_step=1000)


def test_snapshot_is_independent_byte_copy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"original")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1000, "log_history": [{"step": 1000, "loss": 0.25}]}))
    snapshot = snapshot_checkpoint(checkpoint, optimizer_step=1000)
    (checkpoint / "weights.bin").write_bytes(b"changed")
    assert (snapshot.snapshot_root / "weights.bin").read_bytes() == b"original"


def test_supervisor_reads_completed_checkpoints_not_caller_steps(tmp_path: Path) -> None:
    seen: list[int] = []
    def launch() -> None:
        for step in (1000, 2000):
            checkpoint = tmp_path / f"checkpoint-{step}"; checkpoint.mkdir()
            (checkpoint / "weights.bin").write_bytes(b"weights")
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step, "log_history": [{"step": step, "loss": 0.2}]}))
    assert [item["optimizer_step"] for item in run_continuous_supervisor(run_root=tmp_path, launch=launch, package=lambda item: item, publish=lambda item: seen.append(item.optimizer_step) or True)] == [1000, 2000]
    assert seen == [1000, 2000]


def test_supervisor_returns_published_checkpoint_after_interrupt(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1000"; checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1000, "log_history": [{"step": 1000, "loss": 0.2}]}))
    assert [item["optimizer_step"] for item in run_continuous_supervisor(run_root=tmp_path, launch=lambda: (_ for _ in ()).throw(KeyboardInterrupt()), package=lambda item: item, publish=lambda item: {"optimizer_step": item.optimizer_step, "readback_verified": True})] == [1000]
