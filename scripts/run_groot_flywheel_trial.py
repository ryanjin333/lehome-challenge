"""One-process boundary for a provenance-complete GR00T flywheel trial."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable, Sequence

import numpy as np
from lehome.flywheel.artifacts import atomic_write_json


_PINNED = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OCI_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHARD_NAME = re.compile(r"^model-([0-9]{5})-of-([0-9]{5})\.safetensors$")
_RUNTIME_ASSET_DIRECTORIES = ("objects", "robots", "scenes", "textures")


def _code_checkout_root() -> Path:
    """Return the source checkout containing this production entry point."""
    return Path(__file__).resolve().parents[1]


def _runtime_assets_root() -> Path:
    """Return the conventional relative asset root used by the Isaac task."""
    return Path.cwd() / "Assets"


def _verify_runtime_asset_mount(checkout_root: Path, release_assets_root: Path) -> None:
    """Require Isaac's legacy relative paths to resolve into the verified bundle."""
    runtime_assets = _runtime_assets_root()
    for name in _RUNTIME_ASSET_DIRECTORIES:
        mounted = runtime_assets / name
        expected = (checkout_root / name).resolve()
        if not mounted.is_symlink() or not expected.is_dir() or mounted.resolve() != expected:
            raise ValueError("Isaac runtime assets must be symlinked from the dedicated asset checkout")
    if (runtime_assets / "objects" / "Challenge_Garment" / "Release").resolve() != release_assets_root:
        raise ValueError("Isaac runtime garment root does not match the verified Release assets")


def _validate_declared_production_provenance(args: argparse.Namespace) -> None:
    """Reject mutable identities before an Isaac process can consume them."""
    image_identity = getattr(args, "image_identity", "")
    if not isinstance(image_identity, str) or not _OCI_DIGEST.fullmatch(image_identity):
        raise ValueError("production flywheel trials require an OCI SHA-256 image digest")
    if getattr(args, "release_assets_root", None) is None:
        raise ValueError("production flywheel trials require --release-assets-root from a dedicated asset checkout")


def _scene_state_matches(expected: object, observed: object) -> bool:
    """Compare JSON scene receipts while accepting float32 Isaac readback noise."""
    if isinstance(expected, dict) and isinstance(observed, dict):
        return set(expected) == set(observed) and all(
            _scene_state_matches(expected[key], observed[key]) for key in expected
        )
    if isinstance(expected, list) and isinstance(observed, list):
        return len(expected) == len(observed) and all(
            _scene_state_matches(left, right) for left, right in zip(expected, observed, strict=True)
        )
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return bool(np.isclose(expected, observed, atol=1e-5))
    return expected == observed


def read_pinned_revision(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("policy revision file must be a regular file")
    revision = path.read_text(encoding="utf-8").strip()
    if not _PINNED.fullmatch(revision):
        raise ValueError("policy revision must be a pinned 40-character SHA")
    return revision


def _sha256_regular_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("policy artifact must be a materialized regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_artifact_sha256(policy_path: Path) -> str:
    """Hash the exact monolithic or indexed safetensors weights a policy loads."""
    root = Path(policy_path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("policy checkpoint root must be a materialized directory")
    monolithic = root / "model.safetensors"
    index = root / "model.safetensors.index.json"
    monolithic_present = monolithic.exists() or monolithic.is_symlink()
    index_present = index.exists() or index.is_symlink()
    if monolithic_present:
        if index_present:
            raise ValueError("policy checkpoint has ambiguous monolithic and indexed weights")
        if any(_SHARD_NAME.fullmatch(path.name) for path in root.iterdir()):
            raise ValueError("policy checkpoint has ambiguous monolithic and sharded weights")
        return _sha256_regular_file(monolithic)
    if index.is_symlink() or not index.is_file():
        raise ValueError("policy checkpoint index is invalid")
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise TypeError
        shard_names = sorted(set(weight_map.values()))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("policy checkpoint index is invalid") from error
    parsed_names: list[tuple[int, int]] = []
    for name in shard_names:
        if not isinstance(name, str):
            raise ValueError("policy checkpoint index has an invalid shard name")
        match = _SHARD_NAME.fullmatch(name)
        if match is None or Path(name).name != name:
            raise ValueError("policy checkpoint index has an unsafe shard name")
        parsed_names.append((int(match.group(1)), int(match.group(2))))
    totals = {total for _, total in parsed_names}
    if len(totals) != 1:
        raise ValueError("policy checkpoint index has inconsistent shard totals")
    total = totals.pop()
    if total <= 0 or len(shard_names) != total or {number for number, _ in parsed_names} != set(range(1, total + 1)):
        raise ValueError("policy checkpoint index has an incomplete shard set")
    discovered = {
        path.name
        for path in root.iterdir()
        if _SHARD_NAME.fullmatch(path.name)
    }
    referenced = set(shard_names)
    if discovered - referenced:
        raise ValueError("policy checkpoint contains unreferenced weight shards")
    if referenced - discovered:
        raise ValueError("policy checkpoint index references missing weight shards")
    files = (index.name, *shard_names)
    manifest = {
        "schema_version": 1,
        "files": [
            {"path": name, "sha256": _sha256_regular_file(root / name)}
            for name in files
        ],
    }
    return hashlib.sha256(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _live_code_identity(args: argparse.Namespace) -> str:
    """Require the executed checkout, not a caller-supplied label, to be immutable."""
    root = _code_checkout_root().resolve()
    try:
        revision = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"), check=False, capture_output=True, text=True,
        )
        status = subprocess.run(
            ("git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"),
            check=False, capture_output=True, text=True,
        )
    except OSError as error:
        raise ValueError("executed code checkout is not readable Git evidence") from error
    identity = revision.stdout.strip()
    if revision.returncode != 0 or not _PINNED.fullmatch(identity) or status.returncode != 0 or status.stdout:
        raise ValueError("executed code checkout must be a clean pinned Git revision")
    return identity


def _live_policy_identity(args: argparse.Namespace) -> tuple[str, str]:
    """Hash the exact checkpoint weights named by the documented rollout command."""
    if args.policy_revision_file is None:
        raise ValueError("live policy identity requires a pinned revision file")
    revision = read_pinned_revision(args.policy_revision_file)
    return revision, policy_artifact_sha256(args.policy_path)


def _runtime_container_image_identity() -> str:
    """Read the orchestrator-injected immutable OCI digest from this container."""
    identity = os.environ.get("LEHOME_FLYWHEEL_IMAGE_IDENTITY", "")
    if not _OCI_DIGEST.fullmatch(identity):
        raise ValueError("launched container image identity is unavailable or not an OCI SHA-256 digest")
    return identity


def _validate_live_execution_identity(
    args: argparse.Namespace,
    *,
    code_identity_reader: Callable[[argparse.Namespace], str] = _live_code_identity,
    policy_identity_reader: Callable[[argparse.Namespace], tuple[str, str]] = _live_policy_identity,
    image_identity_reader: Callable[[], str] = _runtime_container_image_identity,
) -> None:
    """Fail before Isaac starts unless every recorded execution identity is live."""
    code_revision = code_identity_reader(args)
    policy_revision, policy_digest = policy_identity_reader(args)
    image_identity = image_identity_reader()
    if code_revision != args.code_revision:
        raise ValueError("declared code revision does not match the executed checkout")
    expected_revision = read_pinned_revision(args.policy_revision_file) if args.policy_revision_file is not None else args.policy_revision
    if policy_revision != expected_revision:
        raise ValueError("declared policy revision does not match the mounted policy")
    if policy_digest != args.policy_artifact_sha256:
        raise ValueError("declared policy artifact SHA-256 does not match the mounted policy")
    if image_identity != args.image_identity:
        raise ValueError("declared container image identity does not match the launched container")


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
    parser.add_argument(
        "--release-assets-root",
        type=Path,
    )
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
    if args.snapshot_roundtrip_only and args.render_randomization_sheet:
        raise ValueError("acceptance modes are mutually exclusive")
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
    if not args.garment:
        raise ValueError("--garment is required for trial and simulator acceptance invocations")
    if args.seed < 0 or args.max_steps <= 0:
        raise ValueError("seed must be non-negative and max-steps must be positive")
    if any(strategy not in {"canonical", "mild", "strong"} for strategy in args.strategies):
        raise ValueError("unsupported randomization strategy")
    if not acceptance_mode:
        build_identity(args, revision)
        if not args.dry_run:
            _validate_declared_production_provenance(args)
    return revision or ""


def build_identity(args: argparse.Namespace, revision: str):
    from lehome.flywheel.models import EpisodeIdentity
    values = (args.episode_id, args.policy_repo, args.policy_step, args.code_revision, args.asset_revision, args.simulator_version, args.garment, args.category, args.release_stage, args.policy_artifact_sha256, args.image_identity)
    if any(value is None or value == "" for value in values):
        raise ValueError("normal flywheel trials require complete immutable provenance")
    if not _SHA256.fullmatch(args.policy_artifact_sha256):
        raise ValueError("policy artifact SHA-256 must be a 64-character lowercase digest")
    return EpisodeIdentity(args.episode_id, args.policy_repo, revision, args.policy_step, args.code_revision, args.asset_revision, args.simulator_version, args.garment, args.category, args.release_stage, args.seed, "fold the garment on the table", args.strategy)


def _live_runtime_identity(args: argparse.Namespace, _simulation_app: object) -> tuple[str, str]:
    """Read simulator and Release-assets identity from the launched runtime."""

    try:
        simulator_version = package_version("isaacsim")
    except PackageNotFoundError as error:
        raise ValueError("launched runtime does not expose Isaac Sim version evidence") from error
    requested_assets_root = Path(args.release_assets_root)
    if requested_assets_root.is_symlink():
        raise ValueError("launched runtime Release assets are not a real directory")
    assets_root = requested_assets_root.resolve()
    if assets_root.name != "Release" or not assets_root.is_dir():
        raise ValueError("launched runtime Release assets are not a real directory")
    code_checkout = _code_checkout_root().resolve()
    if assets_root.is_relative_to(code_checkout):
        raise ValueError("launched runtime Release assets must come from a dedicated asset checkout, not the parent code checkout")
    if Path.cwd().resolve() != code_checkout:
        raise ValueError("production flywheel trials must launch from the code checkout root so Isaac uses verified asset links")
    try:
        checkout = subprocess.run(
            ("git", "-C", str(assets_root), "rev-parse", "--show-toplevel"),
            check=False,
            capture_output=True,
            text=True,
        )
        if checkout.returncode != 0 or not checkout.stdout.strip():
            raise ValueError("launched runtime Release assets are not in a readable clean Git checkout")
        checkout_root = Path(checkout.stdout.strip()).resolve()
        if checkout_root == code_checkout or checkout_root.is_relative_to(code_checkout):
            raise ValueError("launched runtime Release assets must come from a dedicated asset checkout, not the parent code checkout")
        try:
            release_path = assets_root.relative_to(checkout_root)
        except ValueError as error:
            raise ValueError("launched runtime Release assets are outside their discovered Git checkout") from error
        revision = subprocess.run(
            ("git", "-C", str(checkout_root), "rev-parse", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
        )
        _verify_runtime_asset_mount(checkout_root, assets_root)
        status = subprocess.run(
            ("git", "-C", str(checkout_root), "status", "--porcelain=v1", "--untracked-files=all"),
            check=False,
            capture_output=True,
            text=True,
        )
        tracked = subprocess.run(
            ("git", "-C", str(checkout_root), "ls-files", "-z"),
            check=False,
            capture_output=True,
            text=True,
        )
        lfs = subprocess.run(
            ("git", "-C", str(checkout_root), "lfs", "ls-files", "--long"),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ValueError("launched runtime Release assets are not in a readable clean Git checkout") from error
    asset_revision = revision.stdout.strip()
    tracked_paths = tuple(path for path in tracked.stdout.split("\0") if path)
    if not any(path == str(release_path) or path.startswith(str(release_path) + "/") for path in tracked_paths):
        raise ValueError("launched runtime Release assets are not tracked in their dedicated checkout")
    if (
        revision.returncode != 0
        or not _PINNED.fullmatch(asset_revision)
        or status.returncode != 0
        or status.stdout
        or tracked.returncode != 0
        or not tracked_paths
    ):
        raise ValueError("launched runtime Release assets are not in a readable clean Git checkout")
    for tracked_path in tracked_paths:
        candidate = checkout_root / tracked_path
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("launched runtime Release assets are not fully materialized")
        with candidate.open("rb") as handle:
            if handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError("launched runtime Release assets contain an unmaterialized Git LFS pointer")
    if lfs.returncode != 0:
        raise ValueError("launched runtime Release assets cannot establish Git LFS integrity")
    for line in lfs.stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) != 3 or not _SHA256.fullmatch(fields[0]) or fields[1] != "*":
            raise ValueError("launched runtime Release assets contain unmaterialized Git LFS content")
    return simulator_version, asset_revision


def _validate_live_runtime_identity(
    args: argparse.Namespace,
    simulation_app: object,
    *,
    runtime_identity_reader: Callable[[argparse.Namespace, object], tuple[str, str]],
) -> None:
    _validate_declared_production_provenance(args)
    simulator_version, asset_revision = runtime_identity_reader(args, simulation_app)
    if simulator_version != args.simulator_version:
        raise ValueError("declared simulator version does not match the launched runtime")
    if asset_revision != args.asset_revision:
        raise ValueError("declared asset revision does not match the launched Release assets")


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
        restored = capture_snapshot(env, randomization={"strategy": "canonical"})
        scene_state_match = _scene_state_matches(snapshot.scene_state, restored.scene_state)
        report.update({"camera_difference": difference, "scene_state_match": scene_state_match, "restore_coverage": ["robot", "cloth", "rng", "garment", "randomization", "camera", "robot_root", "light", "material"], "passed": difference <= report["tolerance"] and scene_state_match})
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
    try:
        for index, strategy in enumerate(args.strategies):
            # A new environment per strategy prevents perturbations from one
            # acceptance row becoming the baseline of the next row.
            env = env_factory(args)
            try:
                env.reset()
                record = sample_randomization(strategy, seed=args.seed + index)
                receipt = env.apply_flywheel_randomization(record)
                from lehome.flywheel.randomization import validate_randomization_receipt
                validate_randomization_receipt(dict(record.values), dict(receipt))
                env.render()
                images = _images(env)
                paths = []
                for camera, frame in images.items():
                    path = args.output_root / f"randomization-{strategy}-{camera}.png"
                    image_writer(path, frame); paths.append(path.name)
                report["strategies"].append({"strategy": strategy, "sampled": dict(record.values), "receipt": dict(receipt), "images": paths})
            finally:
                if hasattr(env, "close"):
                    env.close()
        report["passed"] = len(report["strategies"]) * 3 == len(args.strategies) * 3
    except Exception as error:
        report["simulation_error"] = str(error)
    finally:
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
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise ValueError("refusing to overwrite differing flywheel manifest")
        return path
    atomic_write_json(path, payload)
    return path


def run_trial(
    args: argparse.Namespace,
    *,
    runtime_identity_reader: Callable[[argparse.Namespace, object], tuple[str, str]] = _live_runtime_identity,
    execution_identity_validator: Callable[[argparse.Namespace], None] = _validate_live_execution_identity,
) -> int:
    revision = validate_args(args)
    if args.snapshot_roundtrip_only or args.render_randomization_sheet:
        # These modes use the same launched Isaac process as normal evaluation.
        from isaaclab.app import AppLauncher
        from scripts.utils import common
        parser = argparse.ArgumentParser(add_help=False); AppLauncher.add_app_launcher_args(parser)
        launch_args, _ = parser.parse_known_args(["--headless"] if args.headless else [])
        app = common.launch_app_from_args(launch_args)
        try:
            return run_snapshot_acceptance(args) if args.snapshot_roundtrip_only else run_randomization_acceptance(args)
        finally:
            common.close_app(app)
    command = [
        "--policy_type", "groot", "--policy_path", str(args.policy_path),
        "--garment_type", "custom", "--num_episodes", "1", "--max_steps", str(args.max_steps),
        "--garment_name", args.garment,
        "--seed", str(args.seed), "--task", args.task, "--device", args.device,
    ]
    if args.headless:
        command.append("--headless")
    if args.dry_run:
        manifest = _manifest_path(args, revision)
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
    # Code, checkpoint, and OCI identity are available before Isaac allocates
    # GPU resources, so reject a mislabeled invocation before simulator startup.
    execution_identity_validator(args)
    simulation_app = common.launch_app_from_args(evaluation_args)
    try:
        _validate_live_runtime_identity(
            args,
            simulation_app,
            runtime_identity_reader=runtime_identity_reader,
        )
        manifest = _manifest_path(args, revision)
        evaluation_args.flywheel_manifest = str(manifest)
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
