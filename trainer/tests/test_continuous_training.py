from pathlib import Path
import json

import pytest

from lehome_train.groot.continuous_training import snapshot_checkpoint


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
