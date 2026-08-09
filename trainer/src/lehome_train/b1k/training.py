"""Narrow, explicit initial-run batch identities and OOM classification."""

from __future__ import annotations

from dataclasses import dataclass
import re


OOM_PATTERNS = (
    re.compile(r"(?:torch\.)?OutOfMemoryError:.*CUDA", re.IGNORECASE | re.DOTALL),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"CUDNN_STATUS_ALLOC_FAILED", re.IGNORECASE),
)
SUPPORTED_GPU_COUNTS = frozenset({1, 2, 3, 4})
_EFFECTIVE_BATCH_CANDIDATES = {
    1: (256, 128, 64),
    2: (512, 256, 128),
    3: (768, 384, 192),
    4: (1024, 512, 256),
}
_PER_DEVICE_BATCHES = (64, 32, 16)


@dataclass(frozen=True, slots=True)
class LaunchPlan:
    """One retry-safe optimizer-update batch configuration."""

    identity: str
    num_gpus: int
    physical_batch_size: int
    global_batch_size: int
    gradient_accumulation_steps: int
    effective_global_batch_size: int
    max_steps: int = 15_000
    save_steps: int = 1_000
    checkpoint_keep: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05


def approved_launch_plans(*, num_gpus: int) -> tuple[LaunchPlan, ...]:
    """Return only approved initial-run identities, ordered high to low batch."""

    if type(num_gpus) is not int or num_gpus not in _EFFECTIVE_BATCH_CANDIDATES:
        raise ValueError("this initial run supports one to four GPUs")
    plans: list[LaunchPlan] = []
    for effective_global, physical_batch in zip(_EFFECTIVE_BATCH_CANDIDATES[num_gpus], _PER_DEVICE_BATCHES):
        upstream_global = physical_batch * num_gpus
        accumulation, remainder = divmod(effective_global, upstream_global)
        if remainder or accumulation <= 0:
            raise RuntimeError("approved effective batch cannot be represented by pinned upstream semantics")
        plans.append(
            LaunchPlan(
                identity=f"b1k-gpu{num_gpus}-effective-batch{effective_global}",
                num_gpus=num_gpus,
                physical_batch_size=physical_batch,
                global_batch_size=upstream_global,
                gradient_accumulation_steps=accumulation,
                effective_global_batch_size=effective_global,
            )
        )
    return tuple(plans)


def is_recognized_cuda_oom(failure: object) -> bool:
    """Classify only explicit CUDA allocation failures; never relabel exceptions."""

    return type(failure) is str and any(pattern.search(failure) for pattern in OOM_PATTERNS)
