"""One-process boundary for a provenance-complete GR00T flywheel trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Sequence


_PINNED = re.compile(r"^[0-9a-f]{40}$")


def read_pinned_revision(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("policy revision file must be a regular file")
    revision = path.read_text(encoding="utf-8").strip()
    if not _PINNED.fullmatch(revision):
        raise ValueError("policy revision must be a pinned 40-character SHA")
    return revision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--policy-revision")
    parser.add_argument("--policy-revision-file", type=Path)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--garment")
    parser.add_argument("--episode-id")
    parser.add_argument("--seed", type=int, default=42)
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
    revision = args.policy_revision
    if args.policy_revision_file is not None:
        if revision is not None:
            raise ValueError("provide one policy revision source")
        revision = read_pinned_revision(args.policy_revision_file)
    if not isinstance(revision, str) or not _PINNED.fullmatch(revision):
        raise ValueError("policy revision must be pinned to a 40-character SHA")
    if not args.policy_path.exists() or not args.policy_path.is_dir():
        raise ValueError("policy path must be an existing directory")
    if args.matrix is not None and (args.matrix.is_symlink() or not args.matrix.is_file()):
        raise ValueError("matrix must be an existing regular file")
    if not args.snapshot_roundtrip_only and not args.render_randomization_sheet and not args.garment:
        raise ValueError("--garment is required for a one-trial invocation")
    if args.seed < 0 or args.max_steps <= 0:
        raise ValueError("seed must be non-negative and max-steps must be positive")
    if any(strategy not in {"canonical", "mild", "strong"} for strategy in args.strategies):
        raise ValueError("unsupported randomization strategy")
    return revision


def _manifest_path(args: argparse.Namespace, revision: str) -> Path:
    args.output_root.mkdir(parents=True, exist_ok=True)
    suffix = f"-{args.episode_id}" if args.episode_id else ""
    path = args.output_root / f"flywheel-manifest{suffix}.json"
    payload = {
        "schema_version": 1,
        "policy_revision": revision,
        "policy_path": str(args.policy_path.resolve()),
        "seed": args.seed,
        "garment": args.garment,
    }
    if args.episode_id:
        payload["episode_id"] = args.episode_id
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_trial(args: argparse.Namespace) -> int:
    revision = validate_args(args)
    manifest = _manifest_path(args, revision)
    if args.snapshot_roundtrip_only or args.render_randomization_sheet:
        raise RuntimeError("Isaac acceptance modes require the accepted Linux Isaac runtime")
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
