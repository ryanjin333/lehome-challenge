"""Local composition contract for the train-ready GR00T flywheel path.

This test deliberately exercises the real artifact, materialization, mixing,
normalization, augmentation, and launch-contract modules without claiming that
the offline sample-sheet selection has rendered images or that a GPU trainer
has run.  Those remain accepted-Vast-image gates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from lehome_train.constants import ISAAC_GROOT_REVISION, MODEL_REVISION
from lehome_train.data.normalization import normalization_identity
from lehome_train.flywheel.augmentation import (
    augmentation_profile,
    build_sample_sheet_report,
)
from lehome_train.flywheel.materialize import materialize_episode
from lehome_train.flywheel.mix import (
    ACTION_HORIZON,
    build_mix_plan,
    materialize_mixed_snapshot,
)
from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.groot.launch import build_launch, launch_finetune_to_step
from lehome_train.io import sha256_file
from test_flywheel_materialize import _raw_episode
from test_flywheel_mix import _prepared_source


def _pinned_clean_official_checkout(
    root: Path,
    monkeypatch,
) -> Path:
    """Create the minimum clean checkout fixture for the launch boundary.

    A synthetic Git object cannot have NVIDIA's real commit hash, so the
    checkout-head probe is pinned to the production constant while the real
    cleanliness probe remains active.  This keeps the test local without
    masquerading as an official source checkout.
    """

    checkout = root / "Isaac-GR00T"
    entrypoint = checkout / "gr00t" / "experiment" / "launch_finetune.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# minimal official launcher fixture\n", encoding="utf-8")
    for command in (
        ("git", "init", str(checkout)),
        ("git", "-C", str(checkout), "config", "user.email", "test@example.invalid"),
        ("git", "-C", str(checkout), "config", "user.name", "Test"),
        ("git", "-C", str(checkout), "add", "."),
        ("git", "-C", str(checkout), "commit", "-m", "minimal fixture"),
    ):
        subprocess.run(command, check=True, capture_output=True, text=True)
    monkeypatch.setattr(
        "lehome_train.groot.launch._checkout_head",
        lambda _checkout, _environment: ISAAC_GROOT_REVISION,
    )
    return checkout


def _source_frames(plan, *, split: str, raw: bool) -> set[tuple[object, ...]]:
    """Return the provenance identities covered by one plan split."""

    if raw:
        return {
            (
                item.raw_manifest_sha256,
                item.raw_episode_id,
                frame_id,
            )
            for item in plan.selections
            if item.split == split
            for frame_id in item.raw_frame_ids
        }
    return {
        (
            item.source_manifest_sha256,
            item.source_episode_id,
            frame_id,
        )
        for item in plan.selections
        if item.split == split
        for frame_id in item.source_frame_ids
    }


def test_local_flywheel_chain_reaches_pinned_groot_launch_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Compose accepted raw episodes through local train-ready launch inputs."""

    # The raw fixtures use the recorder's real terminal artifact writer and
    # checksum verifier.  Grade A and B both remain eligible, while the mix
    # applies the published 1.0/0.5 weighting itself.
    raw_a = _raw_episode(tmp_path / "raw-a", grade="A")
    raw_b = _raw_episode(tmp_path / "raw-b", grade="B")
    grade_a = tmp_path / "materialized-a"
    grade_b = tmp_path / "materialized-b"
    assert materialize_episode(raw_a, grade_a).selected_observations == 1
    assert materialize_episode(raw_b, grade_b).selected_observations == 1
    assert json.loads(
        (grade_a / "meta" / "materialization-provenance.json").read_text(
            encoding="utf-8"
        )
    )["quality_grade"] == "A"
    assert json.loads(
        (grade_b / "meta" / "materialization-provenance.json").read_text(
            encoding="utf-8"
        )
    )["quality_grade"] == "B"

    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=2)
    plan = build_mix_plan(organizer, [grade_a, grade_b], seed=20260804)
    assert plan.organizer_training_frames * 3 == plan.flywheel_training_frames * 7
    assert plan.organizer_training_frames == 7 * ACTION_HORIZON
    assert plan.flywheel_training_frames == 3 * ACTION_HORIZON
    assert _source_frames(plan, split="train", raw=False).isdisjoint(
        _source_frames(plan, split="validation", raw=False)
    )
    assert _source_frames(plan, split="train", raw=True).isdisjoint(
        _source_frames(plan, split="validation", raw=True)
    )

    mixed = tmp_path / "mixed"
    result = materialize_mixed_snapshot(plan, organizer, [grade_a, grade_b], mixed)
    assert result["statistics"]["runtime"] == "python_reference"
    assert result["validation"]["valid"] is True
    assert result["validation"]["trainer_validation_split"] == "offline_only"
    manifest = json.loads((mixed / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["train_episode_ids"]).isdisjoint(
        manifest["validation_episode_ids"]
    )
    assert manifest["statistics"]["status"] == "computed_task_4_train_only"
    assert normalization_identity(mixed) == normalization_identity(mixed)

    mild = augmentation_profile("mild")
    sample_frames = tuple(
        {"episode_id": str(index // ACTION_HORIZON), "frame_index": index % ACTION_HORIZON}
        for index in range(32)
    )
    sheet = build_sample_sheet_report("mild", seed=20260804, frames=sample_frames)
    assert sheet["profile_sha256"] == mild.sha256
    assert sheet["render_status"] == "pending_accepted_trainer_image"
    assert sheet["camera_keys"] == ["top_rgb", "left_rgb", "right_rgb"]

    # The local manifest content hash is a pinned snapshot identity for this
    # launch-contract test, not a claimed Hub dataset revision or publication.
    config = FineTuneLaunchConfig(
        base_model_path="/models/GR00T-N1.7-3B",
        base_model_revision=MODEL_REVISION,
        dataset_path=str(mixed),
        dataset_revision=sha256_file(mixed / "manifest.json")[:40],
        modality_config_path=str(mixed / "meta" / "lehome_groot_modality.py"),
        output_dir=str(tmp_path / "output"),
        experiment_name="local-flywheel-contract",
        physical_batch_size=1,
        global_batch_size=4,
        gradient_accumulation_steps=1,
        num_gpus=4,
        max_steps=100,
        save_steps=100,
        warmup_ratio=0.05,
        augmentation_profile="mild",
    )
    checkout = _pinned_clean_official_checkout(tmp_path, monkeypatch)
    launch = build_launch(
        config,
        visible_devices="0,1,2,3",
        environment={"PATH": os.environ["PATH"]},
        official_checkout=checkout,
    )

    assert launch.command[:2] == (
        sys.executable,
        str(checkout / "gr00t" / "experiment" / "launch_finetune.py"),
    )
    assert launch.command[launch.command.index("--dataset-path") + 1] == str(mixed)
    assert launch.command[launch.command.index("--modality-config-path") + 1] == str(
        mixed / "meta" / "lehome_groot_modality.py"
    )
    assert launch.command[launch.command.index("--num-gpus") + 1] == "4"
    assert launch.command[launch.command.index("--global-batch-size") + 1] == "4"
    assert launch.environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    jitter = launch.command.index("--color-jitter-params")
    assert launch.command[jitter + 1 : jitter + 9] == (
        "brightness",
        "0.2",
        "contrast",
        "0.2",
        "saturation",
        "0.2",
        "hue",
        "0.05",
    )

    calls: list[tuple[tuple[str, ...], dict[str, str], bool]] = []

    def runner(
        command: tuple[str, ...], *, env: dict[str, str], check: bool
    ) -> subprocess.CompletedProcess[object]:
        calls.append((command, env, check))
        return subprocess.CompletedProcess(command, 0)

    launch_finetune_to_step(
        config,
        stop_after_optimizer_step=100,
        visible_devices="0,1,2,3",
        environment={"PATH": os.environ["PATH"]},
        official_checkout=checkout,
        runner=runner,
    )
    distributed, environment, checked = calls[0]
    assert distributed[0] == sys.executable
    assert distributed[1:3] == ("-m", "torch.distributed.run")
    assert "--nproc_per_node=4" in distributed
    wrapper = distributed.index("lehome_train.groot.chunk_launch")
    assert distributed[wrapper - 1] == "-m"
    assert distributed[wrapper + 1 : wrapper + 3] == ("--stop-after-step", "100")
    upstream = distributed.index("--") + 1
    assert distributed[upstream] == str(
        checkout / "gr00t" / "experiment" / "launch_finetune.py"
    )
    assert distributed[distributed.index("--num-gpus") + 1] == "4"
    assert distributed[distributed.index("--global-batch-size") + 1] == "4"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert checked is True
