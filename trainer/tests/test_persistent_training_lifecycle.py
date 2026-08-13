from pathlib import Path

import pytest

from run_groot_persistent_training import destroy, main_for_test


class FailProvider:
    def __getattr__(self, _name: str):
        raise AssertionError("provider must not be called")


def test_dry_run_never_calls_provider(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text('{"generation_sha256":"' + "a" * 64 + '","config_sha256":"' + "b" * 64 + '"}')
    report = main_for_test(["prepare", "--request", str(request)], provider=FailProvider())
    assert report["paid_action"] is False


def test_destroy_requires_two_immutable_checkpoints_bound_to_instance(tmp_path: Path) -> None:
    receipt = {"instance_id": 7, "immutable_checkpoint_steps": [1000]}
    with pytest.raises(ValueError, match="instance-bound disposal"):
        destroy(instance_id=7, training_receipt=receipt)
