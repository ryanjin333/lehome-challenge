#!/usr/bin/env python3
"""Parse canonical B1K argv against the pinned upstream tyro schema."""

from __future__ import annotations

from pathlib import Path

import tyro

from gr00t.configs.finetune_config import FinetuneConfig
from lehome_train.b1k.launch import build_b1k_command
from lehome_train.b1k.training import approved_launch_plans


def main() -> None:
    root = Path("/opt/isaac-groot")
    for world_size in (1, 2, 3, 4):
        plan = approved_launch_plans(num_gpus=world_size)[0]
        command = build_b1k_command(
            plan, checkout=str(root), dataset_path="/workspace/data/b1k",
            base_model_path="/workspace/models/groot", output_dir="/workspace/outputs/b1k-cli-check",
            experiment_name="b1k-cli-check", resume_from_checkpoint=False,
        )
        arguments = list(command[command.index(str(root / "scripts/b1k/train_b1k.py")) + 1 :])
        parsed = tyro.cli(FinetuneConfig, args=arguments)
        assert parsed.base_model_path == "/workspace/models/groot"
        assert parsed.dataset_path == "/workspace/data/b1k"
        assert parsed.embodiment_tag == "NEW_EMBODIMENT"
        assert parsed.modality_config_path == str(root / "examples/b1k/r1pro.py")
        assert parsed.num_gpus == world_size
        assert parsed.global_batch_size == plan.global_batch_size
        assert parsed.gradient_accumulation_steps == plan.gradient_accumulation_steps
        assert parsed.max_steps == 15_000 and parsed.save_steps == 1_000 and parsed.save_total_limit == 2
        assert parsed.decode_only_used_frames is True
        assert parsed.learning_rate == plan.learning_rate and parsed.weight_decay == plan.weight_decay and parsed.warmup_ratio == plan.warmup_ratio
        assert parsed.experiment_name == "b1k-cli-check" and parsed.output_dir == "/workspace/outputs/b1k-cli-check"
        assert parsed.resume_from_checkpoint is False


if __name__ == "__main__":
    main()
