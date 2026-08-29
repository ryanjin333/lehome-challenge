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
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
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
_POLICY_SERVER_STARTUP_TIMEOUT_DEFAULT_SECONDS = 180.0
_POLICY_SERVER_STARTUP_TIMEOUT_MIN_SECONDS = 30.0
_POLICY_SERVER_STARTUP_TIMEOUT_MAX_SECONDS = 600.0
_MAX_EXTERNAL_POLICY_SERVER_READINESS_RECEIPT_BYTES = 64 * 1024
_CANONICAL_DECIMAL_SECONDS = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")


class Public96ContractError(ValueError):
    """The public96 result is incomplete, non-public, or unsafe."""


class CheckpointIdentityError(Public96ContractError):
    """The checked N1.7 cache is not the immutable 12K policy."""


def validate_policy_server_startup_timeout(value: object) -> float:
    """Keep GPU model initialization bounded without treating it as a 2s probe."""
    if type(value) is str:
        if not _CANONICAL_DECIMAL_SECONDS.fullmatch(value):
            raise Public96ContractError("policy server startup timeout must be a finite number of seconds")
    elif type(value) not in {int, float}:
        raise Public96ContractError("policy server startup timeout must be a number of seconds")
    try:
        timeout = float(value)
    except (OverflowError, ValueError) as error:
        raise Public96ContractError("policy server startup timeout must be a finite number of seconds") from error
    if not math.isfinite(timeout) or not _POLICY_SERVER_STARTUP_TIMEOUT_MIN_SECONDS <= timeout <= _POLICY_SERVER_STARTUP_TIMEOUT_MAX_SECONDS:
        raise Public96ContractError("policy server startup timeout must be between 30 and 600 seconds")
    return timeout


def video_filename_for_key(source_index: int, key: str) -> str:
    """Return the filename emitted by the video saver for an observation key."""
    if type(source_index) is not int or source_index < 0:
        raise Public96ContractError("video source index is invalid")
    if key not in {"observation.images.top_rgb", "observation.images.left_rgb", "observation.images.right_rgb"}:
        raise Public96ContractError("video observation key is invalid")
    return f"episode{source_index}_{key.replace('.', '_')}.mp4"


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


def validate_checkpoint_identity(
    receipt: Mapping[str, object], policy_root: Path, *, policy_artifact_verifier=None,
) -> dict[str, object]:
    required = {"kind", *CHECKPOINT, "cache_path", "cache_tree_sha256"}
    if set(receipt) != required or receipt.get("kind") != "lehome_groot_n17_checkpoint_identity_v1":
        raise CheckpointIdentityError("checkpoint identity receipt schema is invalid")
    for name, expected in CHECKPOINT.items():
        if receipt.get(name) != expected:
            label = "runtime policy" if name == "runtime_policy_sha256" else name.replace("_", " ")
            raise CheckpointIdentityError(f"checkpoint {label} identity mismatch")
    if policy_root.is_symlink() or not policy_root.is_dir():
        raise CheckpointIdentityError("policy cache must be an existing non-symlink directory")
    try:
        root = policy_root.resolve(strict=True)
    except OSError as error:
        raise CheckpointIdentityError("policy cache must be an existing non-symlink directory") from error
    artifact = (canonical_policy_artifact_sha256 if policy_artifact_verifier is None else policy_artifact_verifier)(root)
    if artifact != CHECKPOINT["artifact_sha256"]:
        raise CheckpointIdentityError("checkpoint artifact SHA-256 mismatch")
    if receipt.get("cache_path") != str(root) or receipt.get("cache_tree_sha256") != tree_sha256(root):
        raise CheckpointIdentityError("checkpoint immutable cache identity mismatch")
    return dict(receipt)


def validate_output_path(output_root: Path, candidate: Path) -> Path:
    if output_root.is_symlink():
        raise Public96ContractError("output root is unsafe")
    try:
        root = output_root.resolve(strict=True)
    except OSError as error:
        raise Public96ContractError("output root is unsafe") from error
    if not root.is_dir():
        raise Public96ContractError("output root is unsafe")
    if candidate.is_symlink() or not candidate.is_file():
        raise Public96ContractError("output artifact is missing, non-regular, or a symlink")
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise Public96ContractError("output artifact path escapes run root") from error
    for part in relative.parts:
        if (root / part).is_symlink():
            raise Public96ContractError("output artifact path contains a symlink")
    resolved = candidate.resolve(strict=True)
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
        "--public96_runtime_policy_sha256", CHECKPOINT["runtime_policy_sha256"],
        "--policy_type", "groot_server", "--policy_path", str(policy_path.resolve()),
        "--policy_server_endpoint", f"tcp://127.0.0.1:{policy_server_port}",
        "--policy_server_token_env", token_env, "--policy_server_request_timeout", "600",
        "--garment_type", "custom", "--garment_cfg_base_path", str(stage_root / "garment-config"),
        "--task", TASK, "--task_description", TASK_TEXT, "--num_episodes", "2", "--max_steps", "600",
        "--seed", "42", "--device", "cpu", "--video_dir", str(stage_root / "videos"),
        "--save_video", "--headless",
    ]


def build_policy_server_command(*, policy_path: Path, port: int, token_env: str, readiness_receipt: Path = Path("policy-server-readiness.json")) -> list[str]:
    return [sys.executable, "-m", "scripts.run_groot_n17_public96_policy_server", "--model-path", str(policy_path.resolve()),
            "--host", "127.0.0.1", "--port", str(port), "--api-token-env", token_env, "--device", "cuda:0", "--seed", "42", "--readiness-receipt", str(readiness_receipt)]


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


def validate_release_assets(asset_root: Path, stages: Sequence[Stage]) -> None:
    try:
        release = asset_root.resolve(strict=True) / "Release"
    except OSError as error:
        raise Public96ContractError("public Release asset root is unavailable or unsafe") from error
    if asset_root.is_symlink() or release.is_symlink() or not release.is_dir():
        raise Public96ContractError("public Release asset root is unavailable or unsafe")
    expected = {stage.garment_name for stage in stages}
    found: set[str] = set()
    for prefix in CATEGORY_PREFIX.values():
        directory, listing = release / prefix, release / prefix / f"{prefix}.txt"
        if directory.is_symlink() or listing.is_symlink() or not directory.is_dir() or not listing.is_file():
            raise Public96ContractError("public Release list/directory is unavailable or unsafe")
        found.update(line.strip() for line in listing.read_text(encoding="utf-8").splitlines() if line.strip())
    if not expected <= found:
        raise Public96ContractError("public Release lists do not contain the frozen public96 garments")


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


def _video_artifact(path: Path, root: Path) -> dict[str, str]:
    """Describe a rendered video only after proving it is a material file."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise Public96ContractError("required public96 video is missing, unsafe, or empty")
    return _artifact(path, root)


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    _write_new_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def _write_new_bytes(path: Path, contents: bytes) -> None:
    """Write evaluator evidence once, without a check-then-overwrite race."""
    if path.exists() or path.is_symlink():
        raise Public96ContractError("public96 output path already exists or is unsafe")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(contents)
    except FileExistsError as error:
        raise Public96ContractError("public96 output path already exists or is unsafe") from error


def _infrastructure_invalid_episode(stage: Stage, episode_index: int, reason: str) -> dict[str, object]:
    return {
        "stage_id": stage.stage_id,
        "category": stage.category,
        "garment_name": stage.garment_name,
        "release_stage": stage.release_stage,
        "seed": stage.seed,
        "episode_index": episode_index,
        "outcome": "infrastructure_invalid",
        "success": False,
        "invalid_reason": reason,
        "artifacts": {},
    }


def _write_invalid_evidence(*, output_root: Path, stages: Sequence[Stage], matrix_digest: str, identity: Mapping[str, object], server_log: Path | None, invalids: Sequence[Mapping[str, str]], episodes: Sequence[Mapping[str, object]], failure_reason: str | None = None) -> None:
    """Persist an unscored, fully assigned result without inventing stage evidence."""
    summary = _summary_from_episodes(episodes, assigned_episodes=sum(len(stage.episode_indices) for stage in stages), status="invalid")
    result: dict[str, object] = {
        "kind": "lehome_groot_n17_public96_result_v1",
        "matrix_sha256": matrix_digest,
        "checkpoint": dict(identity),
        "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()},
        "episodes": list(episodes),
        "invalid_stages": [dict(item) for item in invalids],
        "status": "invalid",
        "summary": summary,
        "publication": {"status": "not_attempted", "vm_stop": "not_attempted"},
    }
    if failure_reason is not None:
        result["failure_reason"] = failure_reason
    _write_new_json(output_root / "result.json", result)

    receipt: dict[str, object] = {
        "kind": "lehome_groot_n17_public96_verifier_receipt_v1",
        "status": "invalid",
        "invalid_stages": [dict(item) for item in invalids],
        "summary": summary,
        "result": _artifact(output_root / "result.json", output_root),
        "matrix_sha256": matrix_digest,
        "checkpoint": dict(identity),
        "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()},
        "publication": {"status": "not_attempted", "vm_stop": "not_attempted"},
    }
    if server_log is not None and server_log.is_file():
        receipt["policy_server_log"] = _artifact(server_log, output_root)
    if failure_reason is not None:
        receipt["failure_reason"] = failure_reason
    _write_new_json(output_root / "verifier-receipt.json", receipt)


def _verified_artifact(
    value: object, *, root: Path, expected_path: str | None = None, require_nonempty: bool = False,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"relative_path", "sha256"}:
        raise Public96ContractError("artifact descriptor is invalid")
    relative, digest = value.get("relative_path"), value.get("sha256")
    if not isinstance(relative, str) or not _HEX.fullmatch(digest if isinstance(digest, str) else ""):
        raise Public96ContractError("artifact descriptor is invalid")
    relative_path = PurePosixPath(relative)
    if (relative_path.is_absolute() or not relative_path.parts or any(part in {".", ".."} for part in relative_path.parts)
            or relative_path.as_posix() != relative):
        raise Public96ContractError("artifact path must be a safe relative path")
    if expected_path is not None and relative != expected_path:
        raise Public96ContractError("artifact relative path does not bind its episode")
    candidate = root.joinpath(*relative_path.parts)
    path = validate_output_path(root, candidate)
    if require_nonempty and path.stat().st_size <= 0:
        raise Public96ContractError("artifact video is empty")
    if path != candidate or sha256_file(path) != digest:
        raise Public96ContractError("artifact file digest mismatch")


def _stage_identity(stage: Stage) -> dict[str, object]:
    return {
        "stage_id": stage.stage_id,
        "category": stage.category,
        "garment_name": stage.garment_name,
        "release_stage": stage.release_stage,
        "seed": stage.seed,
        "episode_indices": list(stage.episode_indices),
    }


def _validate_readiness_payload(readiness: object, *, policy_root: Path) -> dict[str, object]:
    required = {"kind", "artifact_sha256", "runtime_policy_sha256", "model_path", "device", "adapter", "raw_checker_overlay"}
    expected = {
        "kind": "lehome_groot_n17_public96_policy_server_readiness_v1",
        "artifact_sha256": CHECKPOINT["artifact_sha256"],
        "runtime_policy_sha256": CHECKPOINT["runtime_policy_sha256"],
        "model_path": str(policy_root.resolve(strict=True)),
        "device": "cuda:0",
        "adapter": "nvidia_gr00t_policy_server_public96_v1",
        "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()},
    }
    if not isinstance(readiness, Mapping) or set(readiness) != required or dict(readiness) != expected:
        raise Public96ContractError("policy server readiness does not bind the pinned N1.7 policy")
    return expected


def _load_external_readiness_receipt(path: Path, *, policy_root: Path) -> tuple[dict[str, object], bytes, str]:
    """Read a sidecar-owned readiness receipt once before copying its attestation."""
    if not path.is_absolute():
        raise Public96ContractError("external policy server readiness receipt is missing or unsafe")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if type(no_follow) is not int or not no_follow:
        raise Public96ContractError("external policy server readiness receipt is missing or unsafe")
    non_block = getattr(os, "O_NONBLOCK", None)
    if type(non_block) is not int or not non_block:
        raise Public96ContractError("external policy server readiness receipt is missing or unsafe")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow | non_block
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Public96ContractError("external policy server readiness receipt is missing or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= _MAX_EXTERNAL_POLICY_SERVER_READINESS_RECEIPT_BYTES:
            raise Public96ContractError("external policy server readiness receipt is missing or unsafe")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(64 * 1024, (_MAX_EXTERNAL_POLICY_SERVER_READINESS_RECEIPT_BYTES + 1) - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > _MAX_EXTERNAL_POLICY_SERVER_READINESS_RECEIPT_BYTES:
                raise Public96ContractError("external policy server readiness receipt is missing or unsafe")
        contents = b"".join(chunks)
    except OSError as error:
        raise Public96ContractError("external policy server readiness receipt is invalid JSON") from error
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(contents.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Public96ContractError("external policy server readiness receipt is invalid JSON") from error
    return _validate_readiness_payload(payload, policy_root=policy_root), contents, hashlib.sha256(contents).hexdigest()


def _append_external_policy_server_event(path: Path, event: str) -> None:
    """Record evaluator-owned external-server admission evidence without process output."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"event={event}\n")


def _validate_stage_receipt(
    receipt: object, *, stage: Stage, command: Sequence[str], episode_artifacts: Mapping[int, Mapping[str, object]],
    log: object, readiness: Mapping[str, object], policy_root: Path, output_root: Path,
) -> None:
    required = {"kind", "schema_version", "stage", "command", "child_completion_sentinel", "log", "episodes", "policy_server_readiness"}
    sentinel = {"raw_checker_overlay": {"overlay_id": RAW_CHECKER_OVERLAY_ID, "overlay_sha256": overlay_sha256()}, "runtime_policy_sha256": CHECKPOINT["runtime_policy_sha256"]}
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise Public96ContractError("stage receipt schema is invalid")
    if (receipt.get("kind") != "lehome_groot_n17_public96_stage_receipt_v1" or receipt.get("schema_version") != 1
            or receipt.get("stage") != _stage_identity(stage) or receipt.get("command") != list(command)
            or receipt.get("child_completion_sentinel") != sentinel or receipt.get("log") != log):
        raise Public96ContractError("stage receipt does not bind the frozen stage, command, log, and sentinel")
    readiness_binding = receipt.get("policy_server_readiness")
    if not isinstance(readiness_binding, Mapping) or set(readiness_binding) != {"artifact", "binding"}:
        raise Public96ContractError("stage receipt policy readiness binding is invalid")
    _verified_artifact(readiness_binding["artifact"], root=output_root, expected_path="policy-server-readiness.json")
    readiness_path = output_root / "policy-server-readiness.json"
    if _read_json(readiness_path, "policy server readiness") != readiness_binding["binding"] or readiness_binding["binding"] != readiness:
        raise Public96ContractError("stage receipt policy readiness binding is invalid")
    _validate_readiness_payload(readiness_binding["binding"], policy_root=policy_root)
    episodes = receipt.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 2:
        raise Public96ContractError("stage receipt must contain exactly two episode video records")
    expected_episodes = [{"episode_index": index, "videos": dict(episode_artifacts[index])} for index in stage.episode_indices]
    if episodes != expected_episodes:
        raise Public96ContractError("stage receipt video artifacts do not exactly bind both episodes")


def _summary_from_episodes(episodes: Sequence[Mapping[str, object]], *, assigned_episodes: int | None = None, status: str | None = None) -> dict[str, object]:
    categories: dict[str, Counter[str]] = {category: Counter() for category in CATEGORIES}
    for episode in episodes:
        category = episode.get("category")
        if category in categories:
            categories[category]["episodes"] += 1
            outcome = episode.get("outcome")
            categories[category]["successes"] += int(outcome == "success")
            categories[category]["policy_failures"] += int(outcome == "policy_failure")
            categories[category]["infrastructure_invalid"] += int(outcome == "infrastructure_invalid")
            categories[category]["fidelity_invalid"] += int(isinstance(outcome, str) and outcome.endswith("_invalid") and outcome != "infrastructure_invalid")
    def finalize(counts: Counter[str]) -> dict[str, object]:
        scored = counts["successes"] + counts["policy_failures"]
        invalid = counts["infrastructure_invalid"] + counts["fidelity_invalid"]
        return {"episodes": counts["episodes"], "successes": counts["successes"], "policy_failures": counts["policy_failures"], "infrastructure_invalid": counts["infrastructure_invalid"], "fidelity_invalid": counts["fidelity_invalid"], "invalid_episodes": invalid, "scored_episodes": scored, "success_rate": (None if status == "invalid" else (counts["successes"] / scored if scored else None))}
    summarized = {category: finalize(categories[category]) for category in CATEGORIES}
    overall_counts = Counter()
    for counts in categories.values(): overall_counts.update(counts)
    overall = finalize(overall_counts)
    summary: dict[str, object] = {"overall": overall, "categories": summarized}
    if assigned_episodes is not None: summary["assigned_episodes"] = assigned_episodes
    if status is not None: summary["status"] = status
    return summary


def verify_result(
    result: Mapping[str, object], *, stages: Sequence[Stage], matrix_sha256: str, output_root: Path,
    policy_artifact_verifier=None, policy_server_port: int = 9117,
    policy_server_token_env: str = "LEHOME_GROOT_N17_PUBLIC96_POLICY_TOKEN",
) -> dict[str, object]:
    result_keys = {"kind", "matrix_sha256", "checkpoint", "raw_checker_overlay", "episodes", "invalid_stages", "publication", "status", "summary"}
    if set(result) != result_keys or result.get("kind") != "lehome_groot_n17_public96_result_v1" or result.get("matrix_sha256") != matrix_sha256:
        raise Public96ContractError("public96 result identity is invalid")
    checkpoint = result.get("checkpoint")
    episodes = result.get("episodes")
    overlay = result.get("raw_checker_overlay")
    checkpoint_keys = {"kind", *CHECKPOINT, "cache_path", "cache_tree_sha256"}
    if (not isinstance(checkpoint, Mapping) or set(checkpoint) != checkpoint_keys or any(checkpoint.get(name) != expected for name, expected in CHECKPOINT.items())
            or checkpoint.get("kind") != "lehome_groot_n17_checkpoint_identity_v1" or not isinstance(checkpoint.get("cache_path"), str) or not _HEX.fullmatch(str(checkpoint.get("cache_tree_sha256")))
            or not isinstance(overlay, Mapping) or dict(overlay) != {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()} or not isinstance(episodes, list) or len(episodes) != 96):
        raise Public96ContractError("public96 result must contain exactly 96 episodes and the pinned runtime policy")
    if result.get("status") != "valid" or result.get("invalid_stages") != [] or result.get("publication") != {"status": "not_attempted", "vm_stop": "not_attempted"}:
        raise Public96ContractError("public96 result is not a complete valid unscored publication result")
    try:
        policy_root = Path(checkpoint["cache_path"])
        validate_checkpoint_identity(checkpoint, policy_root, policy_artifact_verifier=policy_artifact_verifier)
    except (OSError, TypeError, ValueError, CheckpointIdentityError) as error:
        raise Public96ContractError("public96 result policy cache identity is invalid") from error
    readiness_path = output_root / "policy-server-readiness.json"
    readiness = _validate_readiness_payload(_read_json(readiness_path, "policy server readiness"), policy_root=policy_root)
    expected = {(stage.stage_id, stage.episode_indices[index]): stage for stage in stages for index in range(2)}
    seen: set[tuple[str, int]] = set(); categories: dict[str, Counter[str]] = {category: Counter() for category in CATEGORIES}
    stage_evidence: dict[str, dict[str, object]] = {}
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
            key = f"observation.images.{camera}"
            _verified_artifact(descriptor, root=output_root, expected_path=f"{stage.stage_id}/videos/{folder}/{video_filename_for_key(source_index, key)}", require_nonempty=True)
        evidence = stage_evidence.setdefault(stage.stage_id, {"log": artifacts["log"], "receipt": artifacts["receipt"], "episodes": {}})
        if evidence["log"] != artifacts["log"] or evidence["receipt"] != artifacts["receipt"]:
            raise Public96ContractError("stage result artifacts disagree between sequential episodes")
        evidence["episodes"][int(episode["episode_index"])] = artifacts["videos"]
        seen.add(identity); categories[stage.category]["episodes"] += 1; categories[stage.category]["successes"] += int(success); categories[stage.category]["policy_failures"] += int(not success)
    if len(seen) != 96:
        raise Public96ContractError("public96 result has missing episodes")
    for stage in stages:
        evidence = stage_evidence.get(stage.stage_id)
        if not isinstance(evidence, Mapping) or set(evidence.get("episodes", {})) != {1, 2}:
            raise Public96ContractError("stage receipt is missing sequential episode evidence")
        receipt_path = output_root / str(evidence["receipt"]["relative_path"])
        receipt_payload = _read_json(receipt_path, "stage receipt")
        expected_command = build_stage_command(
            stage, repo_root=Path.cwd(), policy_path=policy_root, output_root=output_root,
            policy_server_port=policy_server_port, token_env=policy_server_token_env,
        )
        _validate_stage_receipt(
            receipt_payload, stage=stage, command=expected_command, episode_artifacts=evidence["episodes"],
            log=evidence["log"], readiness=readiness, policy_root=policy_root, output_root=output_root,
        )
    summarized = {category: dict(categories[category]) for category in CATEGORIES}
    if any(summary.get("episodes") != 24 for summary in summarized.values()):
        raise Public96ContractError("public96 result is not category complete")
    summary = _summary_from_episodes(episodes)
    if result.get("summary") != summary:
        raise Public96ContractError("public96 result summary does not exactly match verified episodes")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True); parser.add_argument("--matrix-sha256", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True); parser.add_argument("--checkpoint-identity-receipt", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy-server-port", type=int, default=9117); parser.add_argument("--policy-server-token-env", default="LEHOME_GROOT_N17_PUBLIC96_POLICY_TOKEN")
    parser.add_argument("--policy-server-startup-timeout", type=validate_policy_server_startup_timeout, default=_POLICY_SERVER_STARTUP_TIMEOUT_DEFAULT_SECONDS)
    parser.add_argument("--external-policy-server-readiness-receipt", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def await_authenticated_policy_server_ready(
    *, port: int, token: str, readiness_timeout: float, request_timeout: float,
    process: subprocess.Popen[str] | None = None,
) -> None:
    """Require a token-bound policy ping before an Isaac stage may start."""
    if not 1 <= port <= 65535 or len(token) < 32 or readiness_timeout <= 0 or request_timeout <= 0:
        raise Public96ContractError("policy server readiness arguments are invalid")
    from scripts.eval_policy.groot_policy import PolicyServerClient

    deadline = time.monotonic() + readiness_timeout
    delay = 0.05
    while True:
        if process is not None and process.poll() is not None:
            raise Public96ContractError("N1.7 policy server exited before public96 evaluation")
        client = PolicyServerClient(f"tcp://127.0.0.1:{port}", token, request_timeout)
        try:
            client.ping()
            if process is None or process.poll() is None:
                return
        except Exception:
            pass
        finally:
            client.close()
        now = time.monotonic()
        if now >= deadline:
            raise Public96ContractError("N1.7 policy server did not pass token-bound readiness")
        time.sleep(min(delay, max(0.0, deadline - now)))
        delay = min(delay * 2, 0.5)


def run(args: argparse.Namespace) -> dict[str, object]:
    stages = load_frozen_matrix(args.matrix, args.matrix_sha256); matrix_digest = _matrix_digest(args.matrix, args.matrix_sha256)
    identity = validate_checkpoint_identity(_read_json(args.checkpoint_identity_receipt, "checkpoint identity receipt"), args.policy_path)
    validate_release_assets(args.asset_root, stages)
    startup_timeout = validate_policy_server_startup_timeout(getattr(args, "policy_server_startup_timeout", _POLICY_SERVER_STARTUP_TIMEOUT_DEFAULT_SECONDS))
    external_receipt_path = getattr(args, "external_policy_server_readiness_receipt", None)
    external_readiness: dict[str, object] | None = None
    external_readiness_contents: bytes | None = None
    external_readiness_sha256: str | None = None
    if external_receipt_path is not None:
        external_readiness, external_readiness_contents, external_readiness_sha256 = _load_external_readiness_receipt(
            Path(external_receipt_path), policy_root=args.policy_path,
        )
    if not args.output_root.is_absolute() or args.output_root.exists() or args.output_root.is_symlink() or not args.output_root.parent.is_dir():
        raise Public96ContractError("output root must be a new absolute path beneath an existing safe parent")
    args.output_root.mkdir(); output_root = args.output_root.resolve(strict=True)
    readiness_receipt = output_root / "policy-server-readiness.json"
    policy_command = build_policy_server_command(policy_path=args.policy_path, port=args.policy_server_port, token_env=args.policy_server_token_env, readiness_receipt=readiness_receipt)
    if external_readiness_contents is not None:
        _write_new_bytes(readiness_receipt, external_readiness_contents)
    stage_commands = [build_stage_command(stage, repo_root=Path.cwd(), policy_path=args.policy_path, output_root=output_root, policy_server_port=args.policy_server_port, token_env=args.policy_server_token_env) for stage in stages]
    assignments = [{"stage_id": stage.stage_id, "category": stage.category, "garment_name": stage.garment_name, "release_stage": stage.release_stage, "seed": stage.seed, "episode_indices": list(stage.episode_indices), "overlay_path": str(_stage_dir(output_root, stage) / "garment-config"), "command": command} for stage, command in zip(stages, stage_commands, strict=True)]
    base = {"kind": "lehome_groot_n17_public96_validation_v1", "matrix_sha256": matrix_digest, "checkpoint": identity, "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()}, "policy_server_command": policy_command, "policy_server_startup_timeout_seconds": startup_timeout, "stage_commands": stage_commands, "assignments": assignments, "publication": {"status": "not_attempted", "vm_stop": "not_attempted"}}
    if external_readiness_sha256 is not None:
        base["policy_server_mode"] = "external"
        base["policy_server_command"] = None
        base["external_policy_server_readiness_receipt_sha256"] = external_readiness_sha256
    if args.dry_run:
        _write_new_json(output_root / "validation-only-receipt.json", base)
        return base
    token = os.environ.get(args.policy_server_token_env, "")
    if len(token) < 32:
        raise Public96ContractError("policy-server token is missing; evaluation was not started")
    server_log = output_root / "policy-server.log"
    server_handle = None
    if external_readiness is not None:
        _write_new_bytes(
            server_log,
            f"source_mode=external\nexternal_readiness_receipt_sha256={external_readiness_sha256}\nevent=readiness_receipt_validated\n".encode("ascii"),
        )
    else:
        server_handle = server_log.open("x", encoding="utf-8")
    episodes: list[dict[str, object]] = []
    invalids: list[dict[str, str]] = []
    server: subprocess.Popen[str] | None = None
    startup_failure: Public96ContractError | None = None
    readiness: dict[str, object] = external_readiness if external_readiness is not None else {}
    try:
        if external_readiness is not None:
            try:
                await_authenticated_policy_server_ready(
                    port=args.policy_server_port, token=token, readiness_timeout=10.0,
                    request_timeout=1.0, process=None,
                )
                _append_external_policy_server_event(server_log, "authenticated_admission_passed")
            except Public96ContractError as error:
                startup_failure = error
                _append_external_policy_server_event(server_log, "authenticated_admission_failed")
        else:
            try:
                server = subprocess.Popen(policy_command, stdout=server_handle, stderr=subprocess.STDOUT, text=True)
            except (OSError, subprocess.SubprocessError) as error:
                startup_failure = Public96ContractError(f"policy server startup failed: {error}")
            if startup_failure is None:
                try:
                    deadline = time.monotonic() + startup_timeout
                    delay = 0.1
                    while not readiness_receipt.is_file() and server.poll() is None:
                        now = time.monotonic()
                        if now >= deadline:
                            raise Public96ContractError("N1.7 policy server did not produce readiness before the startup timeout")
                        time.sleep(min(delay, max(0.0, deadline - now)))
                        delay = min(delay * 2, 1.0)
                    if server.poll() is not None or not readiness_receipt.is_file():
                        raise Public96ContractError("N1.7 policy server exited before public96 evaluation")
                    readiness = _validate_readiness_payload(_read_json(readiness_receipt, "policy server readiness"), policy_root=args.policy_path)
                    await_authenticated_policy_server_ready(
                        port=args.policy_server_port, token=token, readiness_timeout=10.0,
                        request_timeout=1.0, process=server,
                    )
                except Public96ContractError as error:
                    startup_failure = error
        if startup_failure is None:
            for stage_index, (stage, command) in enumerate(zip(stages, stage_commands, strict=True)):
                if external_readiness is not None:
                    try:
                        await_authenticated_policy_server_ready(
                            port=args.policy_server_port, token=token, readiness_timeout=10.0,
                            request_timeout=1.0, process=None,
                        )
                        _append_external_policy_server_event(server_log, f"authenticated_pre_stage_passed stage_id={stage.stage_id}")
                    except Public96ContractError as error:
                        reason = str(error)
                        _append_external_policy_server_event(server_log, f"authenticated_pre_stage_failed stage_id={stage.stage_id}")
                        for remaining in stages[stage_index:]:
                            invalids.append({"stage_id": remaining.stage_id, "reason": reason})
                            for episode_index in remaining.episode_indices:
                                episodes.append(_infrastructure_invalid_episode(remaining, episode_index, reason))
                        break
                stage_root = _stage_dir(output_root, stage)
                try:
                    stage_root.mkdir(); _make_overlay(args.asset_root, stage_root, stage.garment_name)
                    log = stage_root / "stage.log"; completed = subprocess.run(command, cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
                    log.write_text(completed.stdout, encoding="utf-8")
                    metrics = _parse_stage_metrics(completed.stdout)
                    marker = [line for line in completed.stdout.splitlines() if line.startswith("PUBLIC96_STAGE_COMPLETE ")]
                    if completed.returncode != 0 or len(marker) != 1 or "Traceback" in completed.stdout:
                        raise Public96ContractError("stage process exited nonzero after metrics")
                    try:
                        child = json.loads(marker[0].removeprefix("PUBLIC96_STAGE_COMPLETE "))
                    except json.JSONDecodeError as error:
                        raise Public96ContractError("stage completion sentinel is invalid") from error
                    if child != {"raw_checker_overlay": {"overlay_id": RAW_CHECKER_OVERLAY_ID, "overlay_sha256": overlay_sha256()}, "runtime_policy_sha256": CHECKPOINT["runtime_policy_sha256"]}:
                        raise Public96ContractError("stage completion sentinel does not attest the pinned runtime and overlay")
                    episode_videos = []
                    for metric in metrics:
                        folder = "success" if metric["success"] else "failure"
                        source_index = int(metric["episode_index"]) - 1
                        cameras = {camera: stage_root / "videos" / folder / video_filename_for_key(source_index, f"observation.images.{camera}") for camera in ("top_rgb", "left_rgb", "right_rgb")}
                        episode_videos.append(cameras)
                        for video in cameras.values(): _video_artifact(video, output_root)
                    stage_receipt = {
                        "kind": "lehome_groot_n17_public96_stage_receipt_v1", "schema_version": 1,
                        "stage": _stage_identity(stage), "command": command,
                        "child_completion_sentinel": child, "log": _artifact(log, output_root),
                        "episodes": [{"episode_index": metric["episode_index"], "videos": {camera: _video_artifact(video, output_root) for camera, video in cameras.items()}} for metric, cameras in zip(metrics, episode_videos, strict=True)],
                        "policy_server_readiness": {"artifact": _artifact(readiness_receipt, output_root), "binding": readiness},
                    }
                    receipt_path = stage_root / "stage-receipt.json"; _write_new_json(receipt_path, stage_receipt)
                    for metric, cameras in zip(metrics, episode_videos, strict=True):
                        episodes.append({"stage_id": stage.stage_id, "category": stage.category, "garment_name": stage.garment_name, "release_stage": stage.release_stage, "seed": stage.seed, "episode_index": metric["episode_index"], "outcome": "success" if metric["success"] else "policy_failure", "success": metric["success"], "return": metric["return"], "length": metric["length"], "artifacts": {"log": _artifact(log, output_root), "videos": {camera: _video_artifact(video, output_root) for camera, video in cameras.items()}, "receipt": _artifact(receipt_path, output_root)}})
                except (OSError, subprocess.SubprocessError, Public96ContractError) as error:
                    invalids.append({"stage_id": stage.stage_id, "reason": str(error)})
                    for episode_index in stage.episode_indices:
                        episodes.append(_infrastructure_invalid_episode(stage, episode_index, str(error)))
    finally:
        if server is not None:
            server.terminate()
            try: server.wait(timeout=20)
            except subprocess.TimeoutExpired: server.kill(); server.wait()
        if server_handle is not None:
            server_handle.close()
    if startup_failure is not None:
        reason = str(startup_failure)
        invalids = [{"stage_id": stage.stage_id, "reason": reason} for stage in stages]
        episodes = [_infrastructure_invalid_episode(stage, episode_index, reason) for stage in stages for episode_index in stage.episode_indices]
        _write_invalid_evidence(output_root=output_root, stages=stages, matrix_digest=matrix_digest, identity=identity, server_log=server_log, invalids=invalids, episodes=episodes, failure_reason=reason)
        raise startup_failure
    if invalids:
        _write_invalid_evidence(output_root=output_root, stages=stages, matrix_digest=matrix_digest, identity=identity, server_log=server_log, invalids=invalids, episodes=episodes)
        raise Public96ContractError("public96 run contains infrastructure/fidelity invalid stages")
    result = {"kind": "lehome_groot_n17_public96_result_v1", "matrix_sha256": matrix_digest, "checkpoint": identity, "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()}, "episodes": episodes, "invalid_stages": invalids, "publication": {"status": "not_attempted", "vm_stop": "not_attempted"}, "status": "valid", "summary": _summary_from_episodes(episodes)}
    summary = verify_result(result, stages=stages, matrix_sha256=matrix_digest, output_root=output_root, policy_server_port=args.policy_server_port, policy_server_token_env=args.policy_server_token_env)
    _write_new_json(output_root / "result.json", result)
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
