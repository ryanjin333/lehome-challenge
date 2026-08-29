"""Fail-closed public 96-episode evaluator contract for the original GR00T N1.7 12K policy.

This is intentionally separate from the generic experiment evaluator.  It runs
one sequential two-episode Isaac stage per public Release garment through the
pinned GR00T PolicyServer boundary, with CPU cloth simulation and CUDA policy
inference.  ``--dry-run`` validates and emits only a plan; it never starts a
policy server, CUDA, Isaac, a provider action, or a publication action.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from scripts.groot_n17_public96_raw_checker import RAW_CHECKER_OVERLAY_ID, overlay_sha256


CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")
CATEGORY_PREFIX = {
    "top_long": "Top_Long", "top_short": "Top_Short",
    "pant_long": "Pant_Long", "pant_short": "Pant_Short",
}
TASK = "LeHome-BiSO101-Direct-Garment-v2"
TASK_TEXT = "fold the garment on the table"
CHECKPOINT = {
    "repository": "ryanjin333/lehome-groot-n17-models",
    "revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
    "subpath": "policies/step-12000",
    "step": 12000,
    "artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
    "runtime_policy_sha256": "e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa",
}
_HEX = re.compile(r"^[0-9a-f]{64}$")
_EPISODE = re.compile(r"Episode\s+(?P<index>[12])/2:\s+Return=(?P<return>[-+]?\d+(?:\.\d+)?),\s+Length=(?P<length>\d+),\s+Success=(?P<success>True|False)")


class Public96ContractError(ValueError):
    """The public96 result is incomplete, non-public, or unsafe."""


class CheckpointIdentityError(Public96ContractError):
    """The checked N1.7 cache is not the immutable 12K policy."""


@dataclass(frozen=True)
class Stage:
    stage_id: str
    category: str
    garment_name: str
    release_stage: str
    seed: int
    episode_indices: tuple[int, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    if root.is_symlink():
        raise CheckpointIdentityError("policy cache must be a non-symlink directory")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise CheckpointIdentityError("policy cache must be a non-symlink directory")
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CheckpointIdentityError("policy cache must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise CheckpointIdentityError("policy cache contains a non-regular entry")
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
        count += 1
    if not count:
        raise CheckpointIdentityError("policy cache must not be empty")
    return digest.hexdigest()


def canonical_policy_artifact_sha256(policy_root: Path) -> str:
    """Use the established rollout verifier, never a caller-provided digest."""
    try:
        from scripts.run_groot_flywheel_trial import policy_artifact_sha256
        return policy_artifact_sha256(policy_root)
    except (ImportError, ValueError) as error:
        raise CheckpointIdentityError("checkpoint artifact cannot be proven by the canonical verifier") from error


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Public96ContractError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Public96ContractError(f"{label} is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise Public96ContractError(f"{label} must be an object")
    return value


def _matrix_digest(path: Path, checksum_path: Path) -> str:
    if path.is_symlink() or checksum_path.is_symlink() or not path.is_file() or not checksum_path.is_file():
        raise Public96ContractError("frozen matrix or digest is missing or unsafe")
    checksum = checksum_path.read_text(encoding="ascii").strip()
    expected = checksum.split()[0] if checksum else ""
    actual = sha256_file(path)
    if not _HEX.fullmatch(expected) or actual != expected:
        raise Public96ContractError("frozen matrix byte digest mismatch")
    return actual


def load_frozen_matrix(path: Path, checksum_path: Path) -> tuple[Stage, ...]:
    _matrix_digest(path, checksum_path)
    payload = _read_json(path, "frozen matrix")
    required = {"schema_version", "kind", "categories", "seed", "episodes_per_stage", "stages"}
    if set(payload) != required or payload.get("schema_version") != 1 or payload.get("kind") != "lehome_groot_n17_public96_reference_v1":
        raise Public96ContractError("frozen matrix schema is invalid")
    if payload.get("categories") != list(CATEGORIES) or payload.get("seed") != 42 or payload.get("episodes_per_stage") != 2 or not isinstance(payload.get("stages"), list):
        raise Public96ContractError("frozen matrix contract metadata is invalid")
    stages: list[Stage] = []
    for index, raw in enumerate(payload["stages"]):
        required_stage = {"stage_id", "category", "garment_name", "release_stage", "seed", "episode_indices"}
        if not isinstance(raw, Mapping) or set(raw) != required_stage:
            raise Public96ContractError(f"stage {index} schema is invalid")
        try:
            stage = Stage(
                stage_id=raw["stage_id"], category=raw["category"], garment_name=raw["garment_name"],
                release_stage=raw["release_stage"], seed=raw["seed"], episode_indices=tuple(raw["episode_indices"]),
            )
        except TypeError as error:
            raise Public96ContractError(f"stage {index} fields are invalid") from error
        if (not isinstance(stage.stage_id, str) or not stage.stage_id or stage.category not in CATEGORIES
                or stage.release_stage not in {"seen", "unseen"} or type(stage.seed) is not int or stage.seed != 42
                or stage.episode_indices != (1, 2)):
            raise Public96ContractError(f"stage {index} violates the sequential two-episode contract")
        expected_garment = f"{CATEGORY_PREFIX[stage.category]}_{'Seen' if stage.release_stage == 'seen' else 'Unseen'}_"
        if not isinstance(stage.garment_name, str) or not stage.garment_name.startswith(expected_garment):
            raise Public96ContractError(f"stage {index} garment does not match category/release stage")
        stages.append(stage)
    if len(stages) != 48:
        raise Public96ContractError("frozen matrix must contain exactly 48 stages / 96 episodes")
    if len({stage.stage_id for stage in stages}) != 48 or len({stage.garment_name for stage in stages}) != 48:
        raise Public96ContractError("frozen matrix stages and garments must be unique")
    expected = [
        (f"{category.replace('_', '-')}-{release}-{garment}", category, f"{CATEGORY_PREFIX[category]}_{release.title()}_{garment}", release)
        for category in CATEGORIES for release, garment_range in (("seen", range(10)), ("unseen", range(2)))
        for garment in garment_range
    ]
    observed = [(stage.stage_id, stage.category, stage.garment_name, stage.release_stage) for stage in stages]
    if observed != expected:
        raise Public96ContractError("frozen matrix category/garment order is not the public Release order")
    return tuple(stages)


def validate_checkpoint_identity(receipt: Mapping[str, object], policy_root: Path) -> dict[str, object]:
    required = {"kind", *CHECKPOINT, "cache_path", "cache_tree_sha256"}
    if set(receipt) != required or receipt.get("kind") != "lehome_groot_n17_checkpoint_identity_v1":
        raise CheckpointIdentityError("checkpoint identity receipt schema is invalid")
    for name, expected in CHECKPOINT.items():
        if receipt.get(name) != expected:
            label = "runtime policy" if name == "runtime_policy_sha256" else name.replace("_", " ")
            raise CheckpointIdentityError(f"checkpoint {label} identity mismatch")
    root = policy_root.resolve(strict=True)
    artifact = canonical_policy_artifact_sha256(root)
    if artifact != CHECKPOINT["artifact_sha256"]:
        raise CheckpointIdentityError("checkpoint artifact SHA-256 mismatch")
    if receipt.get("cache_path") != str(root) or receipt.get("cache_tree_sha256") != tree_sha256(root):
        raise CheckpointIdentityError("checkpoint immutable cache identity mismatch")
    return dict(receipt)


def validate_output_path(output_root: Path, candidate: Path) -> Path:
    if output_root.is_symlink():
        raise Public96ContractError("output root is unsafe")
    root = output_root.resolve(strict=True)
    if not root.is_dir():
        raise Public96ContractError("output root is unsafe")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise Public96ContractError("output artifact path escapes run root") from error
    return resolved


def _stage_dir(output_root: Path, stage: Stage) -> Path:
    # Command construction is also used by validation-only before the fresh
    # output root exists.  Runtime creation validates the resolved path once.
    return output_root.resolve() / stage.stage_id


def build_stage_command(stage: Stage, *, repo_root: Path, policy_path: Path, output_root: Path, policy_server_port: int, token_env: str) -> list[str]:
    stage_root = _stage_dir(output_root, stage)
    return [
        sys.executable, "-m", "scripts.eval_groot_n17_public96_stage",
        "--public96_raw_checker_overlay", RAW_CHECKER_OVERLAY_ID,
        "--policy_type", "groot_server", "--policy_path", str(policy_path.resolve()),
        "--policy_server_endpoint", f"tcp://127.0.0.1:{policy_server_port}",
        "--policy_server_token_env", token_env, "--policy_server_request_timeout", "600",
        "--garment_type", "custom", "--garment_cfg_base_path", str(stage_root / "garment-config"),
        "--task", TASK, "--task_description", TASK_TEXT, "--num_episodes", "2", "--max_steps", "600",
        "--seed", "42", "--device", "cpu", "--video_dir", str(stage_root / "videos"),
        "--save_video", "--headless",
    ]


def build_policy_server_command(*, policy_path: Path, port: int, token_env: str) -> list[str]:
    return [sys.executable, "-m", "scripts.run_groot_policy_server", "--model-path", str(policy_path.resolve()),
            "--host", "127.0.0.1", "--port", str(port), "--api-token-env", token_env, "--device", "cuda:0", "--seed", "42"]


def _make_overlay(asset_root: Path, stage_root: Path, garment_name: str) -> None:
    release = asset_root.resolve(strict=True) / "Release"
    if not release.is_dir() or release.is_symlink():
        raise Public96ContractError("public Release asset root is unavailable or unsafe")
    overlay = stage_root / "garment-config" / "Release"
    overlay.mkdir(parents=True, exist_ok=False)
    for prefix in CATEGORY_PREFIX.values():
        source = release / prefix
        if source.is_symlink() or not source.is_dir():
            raise Public96ContractError("public Release category is unavailable or unsafe")
        (overlay / prefix).symlink_to(source, target_is_directory=True)
    (overlay / "Release_test_list.txt").write_text(garment_name + "\n", encoding="utf-8")


def _parse_stage_metrics(log: str) -> list[dict[str, object]]:
    records = []
    for match in _EPISODE.finditer(log):
        records.append({"episode_index": int(match.group("index")), "return": float(match.group("return")), "length": int(match.group("length")), "success": match.group("success") == "True"})
    if len(records) != 2 or [record["episode_index"] for record in records] != [1, 2]:
        raise Public96ContractError("stage did not emit exactly two sequential episode metrics")
    return records


def _artifact(path: Path, root: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise Public96ContractError("required public96 artifact is missing or unsafe")
    return {"relative_path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise Public96ContractError("public96 output path already exists or is unsafe")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _verified_artifact(value: object, *, root: Path, expected_path: str | None = None) -> None:
    if not isinstance(value, Mapping) or set(value) != {"relative_path", "sha256"}:
        raise Public96ContractError("artifact descriptor is invalid")
    relative, digest = value.get("relative_path"), value.get("sha256")
    if not isinstance(relative, str) or not _HEX.fullmatch(digest if isinstance(digest, str) else ""):
        raise Public96ContractError("artifact descriptor is invalid")
    if expected_path is not None and relative != expected_path:
        raise Public96ContractError("artifact relative path does not bind its episode")
    path = validate_output_path(root, root / relative)
    if path != root / relative or sha256_file(path) != digest:
        raise Public96ContractError("artifact file digest mismatch")


def verify_result(result: Mapping[str, object], *, stages: Sequence[Stage], matrix_sha256: str, output_root: Path) -> dict[str, object]:
    if result.get("kind") != "lehome_groot_n17_public96_result_v1" or result.get("matrix_sha256") != matrix_sha256:
        raise Public96ContractError("public96 result identity is invalid")
    checkpoint = result.get("checkpoint")
    episodes = result.get("episodes")
    overlay = result.get("raw_checker_overlay")
    if not isinstance(checkpoint, Mapping) or dict(checkpoint) != CHECKPOINT or not isinstance(overlay, Mapping) or dict(overlay) != {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()} or not isinstance(episodes, list) or len(episodes) != 96:
        raise Public96ContractError("public96 result must contain exactly 96 episodes and the pinned runtime policy")
    expected = {(stage.stage_id, stage.episode_indices[index]): stage for stage in stages for index in range(2)}
    seen: set[tuple[str, int]] = set(); categories: dict[str, Counter[str]] = {category: Counter() for category in CATEGORIES}
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise Public96ContractError("public96 episode is invalid")
        identity = (episode.get("stage_id"), episode.get("episode_index"))
        if identity not in expected or identity in seen:
            raise Public96ContractError("public96 episode identity is duplicate, missing, or invalid")
        stage = expected[identity]
        if any(episode.get(name) != getattr(stage, name) for name in ("category", "garment_name", "release_stage", "seed")):
            raise Public96ContractError("public96 episode does not bind its frozen stage")
        outcome, success = episode.get("outcome"), episode.get("success")
        if outcome not in {"success", "policy_failure"} or success != (outcome == "success"):
            raise Public96ContractError("infrastructure/fidelity invalid episode cannot enter the public96 denominator")
        artifacts = episode.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {"log", "videos", "receipt"}:
            raise Public96ContractError("public96 episode lacks required result/video/log/receipt artifacts")
        _verified_artifact(artifacts["log"], root=output_root, expected_path=f"{stage.stage_id}/stage.log")
        _verified_artifact(artifacts["receipt"], root=output_root, expected_path=f"{stage.stage_id}/stage-receipt.json")
        if not isinstance(artifacts["videos"], Mapping) or set(artifacts["videos"]) != {"top_rgb", "left_rgb", "right_rgb"}:
            raise Public96ContractError("episode lacks its exact three camera videos")
        source_index = int(episode["episode_index"]) - 1
        folder = "success" if success else "failure"
        for camera, descriptor in artifacts["videos"].items():
            _verified_artifact(descriptor, root=output_root, expected_path=f"{stage.stage_id}/videos/{folder}/episode{source_index}_observation_{camera}.mp4")
        receipt_path = output_root / str(artifacts["receipt"]["relative_path"])
        receipt_payload = _read_json(receipt_path, "stage receipt")
        receipt_episodes = receipt_payload.get("episodes")
        if (receipt_payload.get("kind") != "lehome_groot_n17_public96_stage_receipt_v1" or receipt_payload.get("stage_id") != stage.stage_id
                or receipt_payload.get("raw_checker_overlay") != {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()}
                or not isinstance(receipt_episodes, list) or not any(item.get("episode_index") == episode.get("episode_index") and item.get("videos") == artifacts.get("videos") for item in receipt_episodes if isinstance(item, Mapping))):
            raise Public96ContractError("stage receipt does not bind the expected stage")
        seen.add(identity); categories[stage.category]["episodes"] += 1; categories[stage.category]["successes"] += int(success); categories[stage.category]["policy_failures"] += int(not success)
    if len(seen) != 96:
        raise Public96ContractError("public96 result has missing episodes")
    summarized = {category: dict(categories[category]) for category in CATEGORIES}
    if any(summary.get("episodes") != 24 for summary in summarized.values()):
        raise Public96ContractError("public96 result is not category complete")
    overall = {key: sum(summary.get(key, 0) for summary in summarized.values()) for key in ("episodes", "successes", "policy_failures")}
    return {"overall": overall, "categories": summarized}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True); parser.add_argument("--matrix-sha256", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True); parser.add_argument("--checkpoint-identity-receipt", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy-server-port", type=int, default=9117); parser.add_argument("--policy-server-token-env", default="LEHOME_GROOT_N17_PUBLIC96_POLICY_TOKEN")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    stages = load_frozen_matrix(args.matrix, args.matrix_sha256); matrix_digest = _matrix_digest(args.matrix, args.matrix_sha256)
    identity = validate_checkpoint_identity(_read_json(args.checkpoint_identity_receipt, "checkpoint identity receipt"), args.policy_path)
    if not args.output_root.is_absolute() or args.output_root.exists() or args.output_root.is_symlink() or not args.output_root.parent.is_dir():
        raise Public96ContractError("output root must be a new absolute path beneath an existing safe parent")
    args.output_root.mkdir(); output_root = args.output_root.resolve(strict=True)
    policy_command = build_policy_server_command(policy_path=args.policy_path, port=args.policy_server_port, token_env=args.policy_server_token_env)
    stage_commands = [build_stage_command(stage, repo_root=Path.cwd(), policy_path=args.policy_path, output_root=output_root, policy_server_port=args.policy_server_port, token_env=args.policy_server_token_env) for stage in stages]
    base = {"kind": "lehome_groot_n17_public96_validation_v1", "matrix_sha256": matrix_digest, "checkpoint": identity, "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()}, "policy_server_command": policy_command, "stage_commands": stage_commands, "publication": {"status": "not_attempted", "vm_stop": "not_attempted"}}
    if args.dry_run:
        _write_new_json(output_root / "validation-only-receipt.json", base)
        return base
    token = os.environ.get(args.policy_server_token_env, "")
    if len(token) < 32:
        raise Public96ContractError("policy-server token is missing; evaluation was not started")
    server_log = output_root / "policy-server.log"; server_handle = server_log.open("x", encoding="utf-8")
    server = subprocess.Popen(policy_command, stdout=server_handle, stderr=subprocess.STDOUT, text=True)
    episodes: list[dict[str, object]] = []
    invalids: list[dict[str, str]] = []
    try:
        time.sleep(2.0)
        if server.poll() is not None:
            raise Public96ContractError("N1.7 policy server exited before public96 evaluation")
        for stage, command in zip(stages, stage_commands, strict=True):
            stage_root = _stage_dir(output_root, stage)
            try:
                stage_root.mkdir(); _make_overlay(args.asset_root, stage_root, stage.garment_name)
                log = stage_root / "stage.log"; completed = subprocess.run(command, cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
                log.write_text(completed.stdout, encoding="utf-8")
                metrics = _parse_stage_metrics(completed.stdout)
                if completed.returncode != 0 or "PUBLIC96_STAGE_COMPLETE" not in completed.stdout or "Traceback" in completed.stdout:
                    raise Public96ContractError("stage process exited nonzero after metrics")
                episode_videos = []
                for metric in metrics:
                    folder = "success" if metric["success"] else "failure"
                    source_index = int(metric["episode_index"]) - 1
                    cameras = {camera: stage_root / "videos" / folder / f"episode{source_index}_observation_{camera}.mp4" for camera in ("top_rgb", "left_rgb", "right_rgb")}
                    episode_videos.append(cameras)
                    for video in cameras.values(): _artifact(video, output_root)
                stage_receipt = {"kind": "lehome_groot_n17_public96_stage_receipt_v1", "stage_id": stage.stage_id, "command": command, "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()}, "log": _artifact(log, output_root), "episodes": [{"episode_index": metric["episode_index"], "videos": {camera: _artifact(video, output_root) for camera, video in cameras.items()}} for metric, cameras in zip(metrics, episode_videos, strict=True)]}
                receipt_path = stage_root / "stage-receipt.json"; _write_new_json(receipt_path, stage_receipt)
                for metric, cameras in zip(metrics, episode_videos, strict=True):
                    episodes.append({"stage_id": stage.stage_id, "category": stage.category, "garment_name": stage.garment_name, "release_stage": stage.release_stage, "seed": stage.seed, "episode_index": metric["episode_index"], "outcome": "success" if metric["success"] else "policy_failure", "success": metric["success"], "return": metric["return"], "length": metric["length"], "artifacts": {"log": _artifact(log, output_root), "videos": {camera: _artifact(video, output_root) for camera, video in cameras.items()}, "receipt": _artifact(receipt_path, output_root)}})
            except (OSError, subprocess.SubprocessError, Public96ContractError) as error:
                invalids.append({"stage_id": stage.stage_id, "reason": str(error)})
    finally:
        server.terminate()
        try: server.wait(timeout=20)
        except subprocess.TimeoutExpired: server.kill(); server.wait()
        server_handle.close()
    result = {"kind": "lehome_groot_n17_public96_result_v1", "matrix_sha256": matrix_digest, "checkpoint": identity, "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()}, "episodes": episodes, "invalid_stages": invalids, "publication": {"status": "not_attempted", "vm_stop": "not_attempted"}}
    _write_new_json(output_root / "result.json", result)
    if invalids:
        _write_new_json(output_root / "verifier-receipt.json", {"kind": "lehome_groot_n17_public96_verifier_receipt_v1", "status": "invalid", "invalid_stages": invalids, "result": _artifact(output_root / "result.json", output_root), "policy_server_log": _artifact(server_log, output_root), "matrix_sha256": matrix_digest, "checkpoint": identity, "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()}, "publication": {"status": "not_attempted", "vm_stop": "not_attempted"}})
        raise Public96ContractError("public96 run contains infrastructure/fidelity invalid stages")
    summary = verify_result(result, stages=stages, matrix_sha256=matrix_digest)
    receipt = {"kind": "lehome_groot_n17_public96_verifier_receipt_v1", "result": _artifact(output_root / "result.json", output_root), "policy_server_log": _artifact(server_log, output_root), "summary": summary, "matrix_sha256": matrix_digest, "checkpoint": identity, "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()}, "publication": {"status": "not_attempted", "vm_stop": "not_attempted"}}
    _write_new_json(output_root / "verifier-receipt.json", receipt)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(_parser().parse_args(argv))
    except Public96ContractError as error:
        print(f"public96 validation error: {error}", file=sys.stderr); return 2
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
