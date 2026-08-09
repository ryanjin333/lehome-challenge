from __future__ import annotations

import pytest

from lehome_train.b1k.training import approved_launch_plans, is_recognized_cuda_oom


@pytest.mark.parametrize(
    ("num_gpus", "global_batches"),
    [(1, (256, 128, 64)), (2, (512, 256, 128)), (3, (768, 384, 192)), (4, (1024, 512, 256))],
)
def test_initial_run_candidates_are_explicit_launch_identities(
    num_gpus: int, global_batches: tuple[int, ...]
) -> None:
    plans = approved_launch_plans(num_gpus=num_gpus)

    assert tuple(plan.effective_global_batch_size for plan in plans) == global_batches
    assert tuple(plan.physical_batch_size for plan in plans) == (64, 32, 16)
    assert tuple(plan.global_batch_size for plan in plans) == tuple(batch * num_gpus for batch in (64, 32, 16))
    assert all(plan.max_steps == 15_000 and plan.save_steps == 1_000 for plan in plans)
    assert all(plan.learning_rate == 1e-4 and plan.weight_decay == 1e-5 and plan.warmup_ratio == 0.05 for plan in plans)
    assert all(plan.num_gpus == num_gpus for plan in plans)
    assert len({plan.identity for plan in plans}) == 3
    assert all(
        plan.effective_global_batch_size
        == plan.physical_batch_size * plan.gradient_accumulation_steps * plan.num_gpus
        for plan in plans
    )
    assert all(plan.gradient_accumulation_steps == 4 for plan in plans)
    assert all(left.physical_batch_size > right.physical_batch_size for left, right in zip(plans, plans[1:]))


@pytest.mark.parametrize("num_gpus", [False, 0, 5])
def test_initial_run_candidates_reject_unsupported_gpu_counts(num_gpus: object) -> None:
    with pytest.raises(ValueError, match="one to four GPUs"):
        approved_launch_plans(num_gpus=num_gpus)


def test_only_recognized_cuda_oom_failures_are_retryable() -> None:
    assert is_recognized_cuda_oom("torch.OutOfMemoryError: CUDA out of memory")
    assert is_recognized_cuda_oom("RuntimeError: CUDA out of memory. Tried to allocate")
    assert not is_recognized_cuda_oom("KeyError: observation.rgb.left")
    assert not is_recognized_cuda_oom("NCCL communicator failed")
    assert not is_recognized_cuda_oom(Exception("CUDA error"))
