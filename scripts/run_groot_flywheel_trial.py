"""One-process boundary for a provenance-complete GR00T flywheel trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Sequence

import numpy as np


_PINNED = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def read_pinned_revision(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("policy revision file must be a regular file")
    revision = path.read_text(encoding="utf-8").strip()
    if not _PINNED.fullmatch(revision):
        raise ValueError("policy revision must be a pinned 40-character SHA")
    return revision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--policy-revision")
    parser.add_argument("--policy-revision-file", type=Path)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--garment")
    parser.add_argument("--episode-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=("canonical", "mild", "strong"), default="canonical")
    parser.add_argument("--policy-repo")
    parser.add_argument("--policy-step", type=int)
    parser.add_argument("--code-revision")
    parser.add_argument("--asset-revision")
    parser.add_argument("--simulator-version")
    parser.add_argument("--category", choices=("top_long", "top_short", "pant_long", "pant_short"))
    parser.add_argument("--release-stage", choices=("seen", "public_unseen"))
    parser.add_argument("--policy-artifact-sha256")
    parser.add_argument("--image-identity")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/flywheel"))
    parser.add_argument("--task", default="LeHome-BiSO101-Direct-Garment-v2")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--snapshot-roundtrip-only", action="store_true")
    parser.add_argument("--render-randomization-sheet", action="store_true")
    parser.add_argument("--strategies", nargs="+", default=["canonical"])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> str:
    acceptance_mode = args.snapshot_roundtrip_only or args.render_randomization_sheet
    revision = args.policy_revision
    if args.policy_revision_file is not None:
        if revision is not None:
            raise ValueError("provide one policy revision source")
        revision = read_pinned_revision(args.policy_revision_file)
    if not acceptance_mode and (not isinstance(revision, str) or not _PINNED.fullmatch(revision)):
        raise ValueError("policy revision must be pinned to a 40-character SHA")
    if not acceptance_mode and (args.policy_path is None or not args.policy_path.exists() or not args.policy_path.is_dir()):
        raise ValueError("policy path must be an existing directory")
    if args.matrix is not None and (args.matrix.is_symlink() or not args.matrix.is_file()):
        raise ValueError("matrix must be an existing regular file")
    if not args.snapshot_roundtrip_only and not args.render_randomization_sheet and not args.garment:
        raise ValueError("--garment is required for a one-trial invocation")
    if args.seed < 0 or args.max_steps <= 0:
        raise ValueError("seed must be non-negative and max-steps must be positive")
    if any(strategy not in {"canonical", "mild", "strong"} for strategy in args.strategies):
        raise ValueError("unsupported randomization strategy")
    if not acceptance_mode:
        build_identity(args, revision)
    return revision or ""


def build_identity(args: argparse.Namespace, revision: str):
    from lehome.flywheel.models import EpisodeIdentity
    values = (args.episode_id, args.policy_repo, args.policy_step, args.code_revision, args.asset_revision, args.simulator_version, args.garment, args.category, args.release_stage, args.policy_artifact_sha256, args.image_identity)
    if any(value is None or value == "" for value in values):
        raise ValueError("normal flywheel trials require complete immutable provenance")
    if not _SHA256.fullmatch(args.policy_artifact_sha256):
        raise ValueError("policy artifact SHA-256 must be a 64-character lowercase digest")
    return EpisodeIdentity(args.episode_id, args.policy_repo, revision, args.policy_step, args.code_revision, args.asset_revision, args.simulator_version, args.garment, args.category, args.release_stage, args.seed, "fold the garment on the table", args.strategy)


def _production_env(args: argparse.Namespace):
    """Construct the normal registered Isaac task lazily, only after AppLauncher."""
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    import lehome.tasks.bedroom  # noqa: F401
    cfg = parse_env_cfg(args.task, device=args.device)
    cfg.sim.use_fabric = False
    cfg.use_random_seed = False
    cfg.seed = args.seed
    cfg.garment_name = args.garment or cfg.garment_name
    cfg.garment_version = "Release"
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.initialize_obs()
    env.reset()
    return env


def _images(env) -> dict[str, np.ndarray]:
    observations = env._get_observations()
    return {key.rsplit(".", 1)[-1]: np.asarray(observations[f"observation.images.{key.rsplit('.', 1)[-1]}"]) for key in ("top_rgb", "left_rgb", "right_rgb")}


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_snapshot_acceptance(args: argparse.Namespace, *, env_factory=_production_env) -> int:
    from lehome.flywheel.snapshots import capture_snapshot, restore_snapshot
    report: dict[str, object] = {"garment": args.garment, "seed": args.seed, "tolerance": 1e-5, "passed": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = None
    try:
        env = env_factory(args)
        before = _images(env)
        snapshot = capture_snapshot(env, randomization={"strategy": "canonical"})
        env.reset()
        restore_snapshot(env, snapshot)
        env.render()
        after = _images(env)
        difference = max(float(np.max(np.abs(before[name].astype(float) - after[name].astype(float)))) for name in before)
        report.update({"camera_difference": difference, "restore_coverage": ["robot", "cloth", "rng", "garment", "randomization"], "passed": difference <= report["tolerance"]})
    except Exception as error:
        report["simulation_error"] = str(error)
    finally:
        if env is not None and hasattr(env, "close"): env.close()
        _write_report(args.output_root / "snapshot-acceptance.json", report)
    return 0 if report["passed"] else 1


def run_randomization_acceptance(args: argparse.Namespace, *, env_factory=_production_env, image_writer=None) -> int:
    from lehome.flywheel.randomization import sample_randomization
    if image_writer is None:
        def image_writer(path, frame):
            import cv2
            if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)): raise RuntimeError(f"failed to write {path}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"seed": args.seed, "strategies": [], "passed": False}
    env = None
    try:
        env = env_factory(args)
        for index, strategy in enumerate(args.strategies):
            env.reset()
            record = sample_randomization(strategy, seed=args.seed + index)
            receipt = env.apply_flywheel_randomization(record)
            if dict(receipt) != dict(record.values): raise RuntimeError(f"{strategy} randomization readback mismatch")
            env.render()
            images = _images(env)
            paths = []
            for camera, frame in images.items():
                path = args.output_root / f"randomization-{strategy}-{camera}.png"
                image_writer(path, frame); paths.append(path.name)
            report["strategies"].append({"strategy": strategy, "sampled": dict(record.values), "receipt": dict(receipt), "images": paths})
        report["passed"] = len(report["strategies"]) * 3 == 9
    except Exception as error:
        report["simulation_error"] = str(error)
    finally:
        if env is not None and hasattr(env, "close"): env.close()
        _write_report(args.output_root / "randomization-receipts.json", report)
    return 0 if report["passed"] else 1


def _manifest_path(args: argparse.Namespace, revision: str) -> Path:
    args.output_root.mkdir(parents=True, exist_ok=True)
    suffix = f"-{args.episode_id}" if args.episode_id else ""
    path = args.output_root / f"flywheel-manifest{suffix}.json"
    identity = build_identity(args, revision)
    payload = {
        "schema_version": 1,
        "policy_revision": revision,
        "policy_path": str(args.policy_path.resolve()),
        "seed": args.seed,
        "garment": args.garment,
        "strategy": args.strategy,
        "identity": {"episode_id": identity.episode_id, "policy_repo": identity.policy_repo, "policy_revision": identity.policy_revision, "policy_step": identity.policy_step, "code_revision": identity.code_revision, "asset_revision": identity.asset_revision, "simulator_version": identity.simulator_version, "garment_name": identity.garment_name, "category": identity.category, "release_stage": identity.release_stage, "seed": identity.seed, "instruction": identity.instruction, "strategy": identity.strategy},
        "policy_artifact_sha256": args.policy_artifact_sha256,
        "image_identity": args.image_identity,
    }
    if args.episode_id:
        payload["episode_id"] = args.episode_id
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_trial(args: argparse.Namespace) -> int:
    revision = validate_args(args)
    if args.snapshot_roundtrip_only or args.render_randomization_sheet:
        # These modes use the same launched Isaac process as normal evaluation.
        from isaaclab.app import AppLauncher
        from scripts.utils import common
        parser = argparse.ArgumentParser(add_help=False); AppLauncher.add_app_launcher_args(parser)
        launch_args, _ = parser.parse_known_args([])
        app = common.launch_app_from_args(launch_args)
        try:
            return run_snapshot_acceptance(args) if args.snapshot_roundtrip_only else run_randomization_acceptance(args)
        finally:
            common.close_app(app)
    manifest = _manifest_path(args, revision)
    command = [
        "--policy_type", "groot", "--policy_path", str(args.policy_path),
        "--garment_type", "custom", "--num_episodes", "1", "--max_steps", str(args.max_steps),
        "--seed", str(args.seed), "--task", args.task, "--device", args.device,
        "--save_video", "--video_dir", str(args.output_root / "videos"),
    ]
    if args.headless:
        command.append("--headless")
    if args.dry_run:
        print(json.dumps({"command": command, "manifest": str(manifest)}, sort_keys=True))
        return 0
    # Keep the legacy parser untouched: inject this opt-in attribute only in
    # this dedicated process, after it has parsed the normal evaluation args.
    from isaaclab.app import AppLauncher
    from scripts.utils import common
    from scripts.utils.parser import setup_eval_parser

    parser = setup_eval_parser()
    AppLauncher.add_app_launcher_args(parser)
    evaluation_args = parser.parse_args(command)
    evaluation_args.flywheel_manifest = str(manifest)
    simulation_app = common.launch_app_from_args(evaluation_args)
    try:
        import lehome.tasks.bedroom  # noqa: F401
        from scripts.utils.evaluation import eval as evaluate

        evaluate(evaluation_args, simulation_app)
    finally:
        common.close_app(simulation_app)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_trial(args)
    except ValueError as error:
        print(f"trial validation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "read_pinned_revision", "run_trial", "validate_args"]
