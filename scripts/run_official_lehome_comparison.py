#!/usr/bin/env python3
"""Run and seal a fail-closed comparison with the pinned official LeHome evaluator."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Callable, Iterable, Mapping, Sequence


SOURCE_REVISION = "a805ad2f7ab52a4583066fc4ee5180459a7f9d15"
ASSET_REVISION = "bea65fd960ad5a1bb3bd3fa77164b28001c08ef9"
SOURCE_REPOSITORY = "https://github.com/lehome-official/lehome-challenge.git"
ASSET_REPOSITORY = "lehome/asset_challenge"
N17_IDENTITY = {
    "repository": "ryanjin333/lehome-groot-n17-models",
    "revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
    "subpath": "policies/step-12000",
    "step": 12000,
    "artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
    "runtime_policy_sha256": "e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa",
}
COMPETITOR_FILES = {
    "config.json": "b7c385bc57456eae603e929b84defb7e991194aade2aad70785e21991e37614c",
    "model.safetensors": "d8d91b6c11cb5aa18fe9a48e7da88eae0ec7e5a227a315ba67ee167d645cde76",
    "policy_preprocessor.json": "a258dac8fa4e4e138990776e156cae36ae6cf172504a8c9e5f2d5864c9126009",
    "policy_postprocessor.json": "f9e18fa7da47e2b6d7ba3459236b140e28f834ce5640ba199be1412d50672fa7",
    "policy_preprocessor_step_2_groot_pack_inputs_v3.safetensors": "74dcbba5d152b7e07c239d8cd66b19b1fd08aa37ff930aa5f2e94cd772a4a912",
    "policy_postprocessor_step_0_groot_action_unpack_unnormalize_v1.safetensors": "74dcbba5d152b7e07c239d8cd66b19b1fd08aa37ff930aa5f2e94cd772a4a912",
    "train_config.json": "81cd0cfe2b2f70dbf55bc7739f9a1f248aebd0e281994f415964d9d0f6e3c118",
}
COMPETITOR_TREE_SHA256 = "fd0b4e91491e1001272ec199f971cb6bce4c966e4d6d0191b6947a3adfddd74a"
ROLLOUT_IMAGE_ID = "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7"
POLICY_IMAGE_REFERENCE = "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746"
TASK = "LeHome-BiSO101-Direct-Garment-v2"
TASK_DESCRIPTION = "fold the garment on the table"
SEED = 42
MAX_STEPS = 600
EPISODES_PER_GARMENT = 2
DEFAULT_PROFILE = "default"
N15_FOCUSED_PROFILE = "n15-focused"
N15_FOCUSED_CATEGORIES = ("top_short", "pant_long")
N15_FOCUSED_FLOORS = {"top_short": 18, "pant_long": 13}
N15_FOCUSED_MAXIMUM_DEFICIT = 2
OFFICIAL_SCORER_SHA256 = "cf17ffb9e015160e9fe9b1ed273870f1cabf0222a4864fc0cd56e642ed792862"
FROZEN_REFERENCE_MATRIX_SHA256 = "bb3c11ddb10eb53ba3cd2b189850d74bc8f2bfa45d15153812b806060b4b80b5"
CAMERAS = ("top_rgb", "left_rgb", "right_rgb")
CATEGORY_DIRECTORIES = (
    ("top_long", "Top_Long"),
    ("top_short", "Top_Short"),
    ("pant_long", "Pant_Long"),
    ("pant_short", "Pant_Short"),
)
_HEX40 = re.compile(r"[0-9a-f]{40}")
_EVALUATING = re.compile(
    r"Evaluating:\s+(?P<garment>\S+)\s+\(Release\)\s+\((?P<index>\d+)/(?P<total>\d+)\)"
)
_EPISODE = re.compile(
    r"Episode\s+(?P<episode>[12])/2:\s+Return=(?P<return>[-+0-9.eE]+),\s+"
    r"Length=(?P<length>\d+),\s+Success=(?P<success>True|False)"
)
_COMPLETION = "Evaluation completed successfully"
_FORBIDDEN_LOG = re.compile(
    r"Traceback \(most recent call last\):|Error during evaluation:|"
    r"non[- ]?finite|CUDA error|policy transport failure|"
    r"(?:^|[^A-Za-z])(?:nan|[+-]?inf)(?:$|[^A-Za-z])",
    re.IGNORECASE | re.MULTILINE,
)


class ComparisonError(RuntimeError):
    """A fidelity, process, or artifact gate made the comparison invalid."""


@dataclass(frozen=True)
class MatrixRow:
    category: str
    garment: str
    episode_index: int
    seed: int = SEED


@dataclass(frozen=True)
class PolicyDefinition:
    policy_id: str
    policy_type: str
    docker_url: str | None = None
    checkpoint_root: Path | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,63}", self.policy_id):
            raise ValueError("policy_id is unsafe")
        if self.policy_type == "docker":
            if self.docker_url is None or self.checkpoint_root is not None:
                raise ValueError("docker policy requires only docker_url")
        elif self.policy_type == "lerobot":
            if self.checkpoint_root is None or self.docker_url is not None:
                raise ValueError("lerobot policy requires only checkpoint_root")
        else:
            raise ValueError("only official docker and lerobot policies are allowed")


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError):
        raise ComparisonError("receipt value is not canonical strict JSON") from None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ComparisonError(f"{label} is unavailable or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise ComparisonError(f"{label} is invalid JSON") from None
    if not isinstance(value, dict):
        raise ComparisonError(f"{label} must be a JSON object")
    return value


def _tree_sha256(root: Path, *, exclude_assets_mount: bool = False) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if exclude_assets_mount and relative.parts and relative.parts[0] == "Assets":
            continue
        if ".git" in relative.parts or path.is_dir():
            continue
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ComparisonError(f"unsafe entry in immutable tree: {relative}")
        digest.update(relative.as_posix().encode("utf-8") + b"\0")
        digest.update(_sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def _git_output(root: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise ComparisonError(f"git identity check failed for {root}: {result.stderr.strip()}")
    return result.stdout.strip()


def checkout_identity(
    root: Path,
    expected_revision: str,
    *,
    label: str,
    exclude_assets_mount: bool = False,
) -> dict[str, str]:
    candidate = Path(root)
    if candidate.is_symlink() or not candidate.is_dir() or _HEX40.fullmatch(expected_revision) is None:
        raise ComparisonError(f"{label} checkout path or revision is invalid")
    root = candidate.resolve(strict=True)
    revision = _git_output(root, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise ComparisonError(f"{label} revision drift: {revision}")
    status_arguments = ["status", "--porcelain", "--untracked-files=all"]
    if exclude_assets_mount:
        status_arguments.extend(["--", ".", ":(exclude)Assets"])
    if _git_output(root, *status_arguments):
        raise ComparisonError(f"{label} checkout is modified")
    return {
        "revision": revision,
        "tree_sha256": _tree_sha256(root, exclude_assets_mount=exclude_assets_mount),
    }


def validate_evaluation_assets(canonical_root: Path, evaluation_root: Path, *, mode: str) -> None:
    canonical_candidate = Path(canonical_root)
    evaluation_candidate = Path(evaluation_root)
    if canonical_candidate.is_symlink() or not canonical_candidate.is_dir():
        raise ComparisonError("canonical asset root is unavailable or unsafe")
    if evaluation_candidate.is_symlink() or not evaluation_candidate.is_dir():
        raise ComparisonError(f"{mode} evaluation assets are unavailable or unsafe")
    canonical = canonical_candidate.resolve(strict=True)
    evaluation = evaluation_candidate.resolve(strict=True)
    if mode == "full":
        if evaluation != canonical:
            raise ComparisonError("full evaluation assets must be the canonical asset root")
        return
    if mode != "smoke":
        raise ComparisonError("evaluation asset mode is invalid")
    overlay_relative = Path("objects/Challenge_Garment/Release/Release_test_list.txt")
    overlay = evaluation / overlay_relative
    if overlay.is_symlink() or not overlay.is_file():
        raise ComparisonError("smoke evaluation assets lack the external list overlay")
    if overlay.read_text(encoding="utf-8") != "Top_Long_Seen_0\n":
        raise ComparisonError("smoke evaluation assets contain the wrong garment list")
    if overlay.stat().st_mode & 0o222:
        raise ComparisonError("smoke evaluation asset list must be read-only")
    for top in ("objects", "robots", "scenes", "textures"):
        view_top = evaluation / top
        canonical_top = canonical / top
        if not view_top.is_dir() or not canonical_top.is_dir() or not os.path.samefile(view_top, canonical_top):
            raise ComparisonError(f"smoke evaluation assets drifted from canonical {top}")
        for view_path in view_top.rglob("*"):
            relative = view_path.relative_to(evaluation)
            if relative == overlay_relative:
                continue
            canonical_path = canonical / relative
            if view_path.is_symlink() or canonical_path.is_symlink():
                raise ComparisonError("smoke evaluation assets contain a symlink")
            if not canonical_path.exists() or not os.path.samefile(view_path, canonical_path):
                raise ComparisonError(f"smoke evaluation asset drift: {relative}")


def validate_reference_matrix(matrix_path: Path, checksum_path: Path, rows: Sequence[MatrixRow]) -> str:
    if matrix_path.is_symlink() or checksum_path.is_symlink() or not matrix_path.is_file() or not checksum_path.is_file():
        raise ComparisonError("frozen reference matrix or checksum is unavailable")
    digest = _sha256_file(matrix_path)
    checksum_parts = checksum_path.read_text(encoding="ascii").strip().split()
    if not checksum_parts or checksum_parts[0] != digest:
        raise ComparisonError("frozen reference matrix checksum mismatch")
    try:
        payload = json.loads(matrix_path.read_text(encoding="utf-8"))
        stages = payload["stages"]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        raise ComparisonError("frozen reference matrix is invalid") from None
    if (
        set(payload) != {"schema_version", "kind", "categories", "seed", "episodes_per_stage", "stages"}
        or payload["schema_version"] != 1
        or payload["kind"] != "lehome_groot_n17_public96_reference_v1"
        or payload["categories"] != [category for category, _ in CATEGORY_DIRECTORIES]
        or payload["seed"] != SEED
        or payload["episodes_per_stage"] != EPISODES_PER_GARMENT
        or not isinstance(stages, list)
    ):
        raise ComparisonError("frozen reference matrix schema drift")
    expected: list[tuple[str, str, int, int]] = []
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != {
            "stage_id", "category", "garment_name", "release_stage", "seed", "episode_indices"
        } or stage["seed"] != SEED or stage["episode_indices"] != [1, 2]:
            raise ComparisonError("frozen reference matrix stage drift")
        expected.extend(
            (stage["category"], stage["garment_name"], episode, stage["seed"])
            for episode in stage["episode_indices"]
        )
    observed = [(row.category, row.garment, row.episode_index, row.seed) for row in rows]
    if expected != observed or len(observed) != 96:
        raise ComparisonError("native Release lists do not match the frozen reference matrix")
    return digest


def validate_n17_checkpoint(
    checkpoint_root: Path,
    identity_receipt: Path,
    *,
    validator: Callable[[Mapping[str, object], Path], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if identity_receipt.is_symlink() or not identity_receipt.is_file():
        raise ComparisonError("N1.7 checkpoint identity receipt is unavailable")
    try:
        payload = json.loads(identity_receipt.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise ComparisonError("N1.7 checkpoint identity receipt is invalid") from None
    if not isinstance(payload, dict):
        raise ComparisonError("N1.7 checkpoint identity receipt must be an object")
    if validator is None:
        from scripts.eval_groot_n17_public96 import validate_checkpoint_identity

        validator = validate_checkpoint_identity
    try:
        validated = validator(payload, checkpoint_root)
    except Exception as error:
        raise ComparisonError(f"N1.7 checkpoint identity failed: {error}") from error
    if any(validated.get(key) != value for key, value in N17_IDENTITY.items()):
        raise ComparisonError("N1.7 checkpoint identity drift")
    return dict(validated)


def validate_competitor_checkpoint(checkpoint_root: Path) -> dict[str, object]:
    root = Path(checkpoint_root)
    if root.is_symlink() or not root.is_dir():
        raise ComparisonError("competitor checkpoint is unavailable or unsafe")
    entries = list(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ComparisonError("competitor checkpoint has an unsafe entry")
    observed = {path.name for path in entries}
    if len(entries) != len(COMPETITOR_FILES) or observed != set(COMPETITOR_FILES):
        raise ComparisonError("competitor checkpoint has an unexpected file set")
    digests = {name: _sha256_file(root / name) for name in sorted(COMPETITOR_FILES)}
    if digests != dict(sorted(COMPETITOR_FILES.items())):
        raise ComparisonError("competitor checkpoint digest mismatch")
    tree_digest = _tree_sha256(root)
    if tree_digest != COMPETITOR_TREE_SHA256:
        raise ComparisonError("competitor checkpoint tree digest mismatch")
    return {
        "repository": "theo-zhou/lehome-groot-submission-4",
        "revision": "d384fe00508acd96ab1c3c5dc265e08261f94b3b",
        "file_sha256": digests,
        "tree_sha256": tree_digest,
    }


def validate_candidate_n15_checkpoint(
    checkpoint_root: Path, identity_receipt: Path
) -> dict[str, object]:
    """Bind the native policy directory to Task 1's verified training receipt."""
    root = Path(checkpoint_root)
    if root.is_symlink() or not root.is_dir():
        raise ComparisonError("candidate N1.5 checkpoint is unavailable or unsafe")
    receipt = _load_json_object(identity_receipt, "candidate N1.5 identity receipt")
    receipt_checkpoint = receipt.get("checkpoint_root")
    files = receipt.get("checkpoint_files")
    if (
        receipt.get("kind") != "lehome_public_n15_verified_training_output_v1"
        or receipt.get("step") != 12000
        or not isinstance(receipt_checkpoint, str)
        or Path(receipt_checkpoint).resolve(strict=True) / "pretrained_model" != root.resolve(strict=True)
        or not isinstance(files, Mapping)
        or not files
    ):
        raise ComparisonError("candidate N1.5 identity receipt is invalid")
    prefix = "checkpoints/012000/pretrained_model/"
    expected = {
        str(relative)[len(prefix) :]: digest
        for relative, digest in files.items()
        if isinstance(relative, str) and relative.startswith(prefix)
    }
    observed_paths = [path for path in sorted(root.rglob("*")) if not path.is_dir()]
    if (
        not expected
        or any(path.is_symlink() or not path.is_file() for path in observed_paths)
        or {path.relative_to(root).as_posix() for path in observed_paths} != set(expected)
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
            or _sha256_file(root / relative) != digest
            for relative, digest in expected.items()
        )
    ):
        raise ComparisonError("candidate N1.5 checkpoint file identity drift")
    return {
        "kind": receipt["kind"],
        "step": 12000,
        "identity_receipt_sha256": _sha256_file(identity_receipt),
        "tree_sha256": _tree_sha256(root),
        "file_count": len(expected),
    }


def validate_runtime_evidence(
    *,
    runtime_identity: Mapping[str, object],
    rollout_image: Mapping[str, object],
    policy_image: Mapping[str, object],
    cuda_receipt: Mapping[str, object],
    readiness_receipt: Mapping[str, object],
    n17_checkpoint: Path,
) -> dict[str, object]:
    if set(runtime_identity) != {"revision", "tree_sha256"} or any(
        type(runtime_identity.get(key)) is not str
        for key in ("revision", "tree_sha256")
    ):
        raise ComparisonError("reviewed runtime identity is invalid")
    if _HEX40.fullmatch(str(runtime_identity["revision"])) is None or re.fullmatch(
        r"[0-9a-f]{64}", str(runtime_identity["tree_sha256"])
    ) is None:
        raise ComparisonError("reviewed runtime identity is invalid")
    image_keys = {"kind", "reference", "image_id", "repo_digests", "docker_inspect_sha256"}
    for value in (rollout_image, policy_image):
        if set(value) != image_keys or value.get("kind") != "lehome_official_image_inspection_v1":
            raise ComparisonError("runtime image inspection receipt is invalid")
        if (
            type(value.get("reference")) is not str
            or type(value.get("image_id")) is not str
            or not str(value["image_id"]).startswith("sha256:")
            or type(value.get("repo_digests")) is not list
            or any(type(item) is not str for item in value["repo_digests"])
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("docker_inspect_sha256"))) is None
        ):
            raise ComparisonError("runtime image inspection receipt is invalid")
    if rollout_image["reference"] != ROLLOUT_IMAGE_ID or rollout_image["image_id"] != ROLLOUT_IMAGE_ID:
        raise ComparisonError("rollout image identity drift")
    if (
        policy_image["reference"] != POLICY_IMAGE_REFERENCE
        or POLICY_IMAGE_REFERENCE not in policy_image["repo_digests"]
    ):
        raise ComparisonError("policy image identity drift")
    cuda_keys = {"cuda_available", "cuda_device_count", "cuda_runtime", "cuda_device_name"}
    if (
        set(cuda_receipt) != cuda_keys
        or cuda_receipt.get("cuda_available") is not True
        or type(cuda_receipt.get("cuda_device_count")) is not int
        or cuda_receipt["cuda_device_count"] < 1
        or not isinstance(cuda_receipt.get("cuda_runtime"), str)
        or not cuda_receipt["cuda_runtime"]
        or not isinstance(cuda_receipt.get("cuda_device_name"), str)
        or not cuda_receipt["cuda_device_name"]
    ):
        raise ComparisonError("CUDA runtime receipt is invalid")
    readiness_keys = {
        "kind", "artifact_sha256", "runtime_policy_sha256", "model_path", "device", "adapter", "raw_checker_overlay"
    }
    overlay = readiness_receipt.get("raw_checker_overlay")
    if (
        set(readiness_receipt) != readiness_keys
        or readiness_receipt.get("kind") != "lehome_groot_n17_public96_policy_server_readiness_v1"
        or readiness_receipt.get("artifact_sha256") != N17_IDENTITY["artifact_sha256"]
        or readiness_receipt.get("runtime_policy_sha256") != N17_IDENTITY["runtime_policy_sha256"]
        or readiness_receipt.get("model_path") != str(n17_checkpoint.resolve())
        or readiness_receipt.get("device") != "cuda:0"
        or readiness_receipt.get("adapter") != "nvidia_gr00t_policy_server_public96_v1"
        or not isinstance(overlay, Mapping)
        or set(overlay) != {"id", "sha256"}
        or not isinstance(overlay.get("id"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(overlay.get("sha256"))) is None
    ):
        raise ComparisonError("policy server readiness receipt is invalid")
    return {
        "runtime": dict(runtime_identity),
        "rollout_image": dict(rollout_image),
        "policy_image": dict(policy_image),
        "cuda": dict(cuda_receipt),
        "policy_server_readiness": dict(readiness_receipt),
        "policy_device": "cuda:0",
    }


def validate_competitor_runtime_evidence(root: Path) -> dict[str, object]:
    from scripts import verify_native_reference_evaluator_gate as native_gate

    evidence_root = Path(root).resolve(strict=True)
    validators = {
        "peft_overlay": ("peft-overlay-receipt.json", native_gate._validate_peft_overlay_receipt),
        "flash_attention_overlay": (
            "flash-attention-overlay-receipt.json",
            native_gate._validate_flash_attention_overlay_receipt,
        ),
        "flash_attention_runtime": (
            "flash-attention-runtime-receipt.json",
            native_gate._validate_flash_attention_runtime_receipt,
        ),
        "public_dependencies_overlay": (
            "public-pyproject-dependencies-overlay-receipt.json",
            native_gate._validate_public_pyproject_dependencies_overlay_receipt,
        ),
        "public_dependencies_runtime": (
            "public-pyproject-dependencies-runtime-receipt.json",
            native_gate._validate_public_pyproject_dependencies_runtime_receipt,
        ),
        "pynput_backend": (
            "pynput-backend-receipt.json",
            native_gate._validate_pynput_backend_receipt,
        ),
    }
    validated: dict[str, object] = {}
    for label, (filename, validator) in validators.items():
        path = evidence_root / filename
        document = _load_json_object(path, f"competitor {label} receipt")
        try:
            receipt = validator(document)
        except Exception as error:
            raise ComparisonError(f"competitor runtime evidence failed: {label}: {error}") from error
        validated[label] = {"receipt": receipt, "sha256": _sha256_file(path)}
    return {
        "python_executable": "/opt/lehome-challenge/.venv/bin/python",
        "pythonexe": "/opt/lehome-challenge/.venv/bin/python",
        "pythonpath_peft_overlay": str(native_gate.PEFT_WHEEL_PATH),
        "evidence": validated,
    }


def runtime_adapter_identities(runtime_root: Path) -> dict[str, str]:
    relatives = (
        "scripts/run_official_lehome_comparison.py",
        "scripts/serve_official_docker_policy_bridge.py",
        "scripts/run_groot_n17_public96_policy_server.py",
        "rollout_appliance/run_official_lehome_comparison_container.sh",
        "rollout_appliance/native_reference_site/sitecustomize.py",
        "rollout_appliance/native_reference_site/checkpoint_compatibility.py",
    )
    identities: dict[str, str] = {}
    for relative in relatives:
        path = runtime_root / relative
        if path.is_symlink() or not path.is_file():
            raise ComparisonError(f"reviewed runtime adapter is unavailable: {relative}")
        identities[relative] = _sha256_file(path)
    return identities


def focused_runtime_adapter_identities(runtime_root: Path) -> dict[str, str]:
    relatives = (
        "scripts/run_official_lehome_comparison.py",
        "rollout_appliance/run_public_n15_focused_gate.sh",
        "rollout_appliance/native_reference_site/sitecustomize.py",
        "rollout_appliance/native_reference_site/checkpoint_compatibility.py",
        "rollout_appliance/native_reference_site/cloth_fidelity.py",
    )
    identities: dict[str, str] = {}
    for relative in relatives:
        path = runtime_root / relative
        if path.is_symlink() or not path.is_file():
            raise ComparisonError(f"reviewed focused runtime adapter is unavailable: {relative}")
        identities[relative] = _sha256_file(path)
    return identities


def metadata_identities(metadata_root: Path) -> dict[str, object]:
    root = Path(metadata_root).resolve(strict=True)
    category_digests: dict[str, str] = {}
    for category, _directory in CATEGORY_DIRECTORIES:
        category_root = root / f"{category}_merged"
        if category_root.is_symlink() or not category_root.is_dir():
            raise ComparisonError(f"authenticated category metadata is unavailable: {category}")
        category_digests[category] = _tree_sha256(category_root)
    return {
        "tree_sha256": _tree_sha256(root),
        "category_tree_sha256": category_digests,
        "policy_visibility": "LeRobot construction only; never included in DockerPolicy observations",
    }


def validate_smoke_prerequisite(
    receipt: Mapping[str, object], *, expected: Mapping[str, object]
) -> dict[str, object]:
    if (
        receipt.get("kind") != "lehome_official_policy_comparison_v1"
        or receipt.get("status") != "valid"
        or receipt.get("mode") != "smoke"
    ):
        raise ComparisonError("full comparison requires a valid smoke receipt")
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value:
            raise ComparisonError(f"smoke prerequisite identity/config drift: {key}")
    results = receipt.get("results")
    if not isinstance(results, list) or [row.get("policy_id") for row in results if isinstance(row, Mapping)] != [
        "ours-12k", "competitor-n15"
    ]:
        raise ComparisonError("smoke prerequisite policy order is invalid")
    for result in results:
        outcomes = result.get("outcomes") if isinstance(result, Mapping) else None
        if (
            not isinstance(result, Mapping)
            or result.get("status") != "valid"
            or result.get("episode_count") != 2
            or not isinstance(outcomes, list)
            or len(outcomes) != 2
        ):
            raise ComparisonError("smoke prerequisite must contain exactly two valid outcomes per policy")
        observed = [
            (row.get("category"), row.get("garment"), row.get("episode_index"), row.get("seed"))
            for row in outcomes
            if isinstance(row, Mapping)
        ]
        if observed != [
            ("custom", "Top_Long_Seen_0", 1, SEED),
            ("custom", "Top_Long_Seen_0", 2, SEED),
        ]:
            raise ComparisonError("smoke prerequisite outcome order/config is invalid")
    return dict(receipt)


_EXECUTION_SEAL_FILES = {
    "comparison-receipt.sha256.json",
    "execution-manifest.json",
    "status.json",
}


def _execution_artifact_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in _EXECUTION_SEAL_FILES or relative.startswith("publication-"):
            continue
        if path.is_symlink() or not path.is_file():
            raise ComparisonError(f"execution bundle contains an unsafe artifact: {relative}")
        paths.append(path)
    return paths


def seal_execution_bundle(bundle_root: Path) -> dict[str, object]:
    root = Path(bundle_root).resolve(strict=True)
    receipt_path = root / "comparison-receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ComparisonError("comparison receipt is unavailable for execution sealing")
    for name in _EXECUTION_SEAL_FILES:
        if (root / name).exists() or (root / name).is_symlink():
            raise ComparisonError("execution bundle is already sealed")
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in _execution_artifact_paths(root)
    ]
    manifest = {
        "schema_version": 1,
        "kind": "lehome_official_execution_manifest_v1",
        "entries": entries,
    }
    manifest_sha = _write_immutable(root / "execution-manifest.json", manifest)
    receipt_sha = _sha256_file(receipt_path)
    companion = {
        "receipt_sha256": receipt_sha,
        "execution_manifest_sha256": manifest_sha,
    }
    _write_immutable(root / "comparison-receipt.sha256.json", companion)
    _write_immutable(
        root / "status.json",
        {
            "status": "valid",
            "receipt_sha256": receipt_sha,
            "execution_manifest_sha256": manifest_sha,
        },
    )
    return {"receipt_sha256": receipt_sha, "manifest_sha256": manifest_sha, "entries": entries}


def validate_sealed_execution(receipt_path: Path) -> dict[str, object]:
    receipt = Path(receipt_path).resolve(strict=True)
    root = receipt.parent
    if receipt.name != "comparison-receipt.json":
        raise ComparisonError("sealed comparison receipt has the wrong filename")
    comparison = _load_json_object(receipt, "comparison receipt")
    companion = _load_json_object(root / "comparison-receipt.sha256.json", "comparison receipt companion")
    status = _load_json_object(root / "status.json", "comparison status")
    manifest_path = root / "execution-manifest.json"
    manifest = _load_json_object(manifest_path, "execution manifest")
    receipt_sha = _sha256_file(receipt)
    manifest_sha = _sha256_file(manifest_path)
    if companion != {
        "receipt_sha256": receipt_sha,
        "execution_manifest_sha256": manifest_sha,
    } or status != {
        "status": "valid",
        "receipt_sha256": receipt_sha,
        "execution_manifest_sha256": manifest_sha,
    }:
        raise ComparisonError("execution seal companion/status mismatch")
    if (
        set(manifest) != {"schema_version", "kind", "entries"}
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != "lehome_official_execution_manifest_v1"
        or not isinstance(manifest.get("entries"), list)
    ):
        raise ComparisonError("execution manifest schema is invalid")
    expected_entries: list[dict[str, object]] = []
    observed_paths: set[str] = set()
    for entry in manifest["entries"]:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "size", "sha256"}:
            raise ComparisonError("execution manifest entry is invalid")
        relative = entry.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in observed_paths
        ):
            raise ComparisonError("execution manifest path is unsafe or duplicated")
        observed_paths.add(relative)
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or type(entry.get("size")) is not int
            or entry["size"] != path.stat().st_size
            or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256"))) is None
            or entry["sha256"] != _sha256_file(path)
        ):
            raise ComparisonError(f"execution manifest artifact mismatch: {relative}")
        expected_entries.append(dict(entry))
    actual_paths = {path.relative_to(root).as_posix() for path in _execution_artifact_paths(root)}
    if actual_paths != observed_paths or "comparison-receipt.json" not in observed_paths:
        raise ComparisonError("execution manifest file set mismatch")
    return {
        "comparison": comparison,
        "receipt_sha256": receipt_sha,
        "manifest_sha256": manifest_sha,
        "entries": expected_entries,
    }


def deterministic_remote_prefix(receipt_sha256: str, manifest_sha256: str) -> str:
    for value in (receipt_sha256, manifest_sha256):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ComparisonError("publication identity digest is invalid")
    return f"official-comparisons/{receipt_sha256[:16]}-{manifest_sha256[:16]}"


def _publication_entries(bundle_root: Path, sealed: Mapping[str, object]) -> list[dict[str, object]]:
    entries = [dict(entry) for entry in sealed["entries"]]
    for relative in sorted(_EXECUTION_SEAL_FILES):
        path = bundle_root / relative
        if path.is_symlink() or not path.is_file():
            raise ComparisonError(f"publication seal artifact is unavailable: {relative}")
        entries.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256_file(path)})
    entries.sort(key=lambda entry: str(entry["path"]))
    return entries


def _immutable_repo_revision(api: object, *, repository: str, revision: str) -> str:
    try:
        info = api.repo_info(repo_id=repository, repo_type="dataset", revision=revision)
    except Exception:
        info = None
    candidate = getattr(info, "sha", None)
    if type(candidate) is str and _HEX40.fullmatch(candidate):
        return candidate
    if _HEX40.fullmatch(revision):
        return revision
    raise ComparisonError("exact remote prefix exists but its immutable revision is unavailable")


def _anonymous_verify_publication(
    *,
    api: object,
    downloader: Callable[..., str],
    repository: str,
    revision: str,
    remote_prefix: str,
    entries: Sequence[Mapping[str, object]],
) -> None:
    prefix_marker = remote_prefix + "/"
    remote_files = {
        path
        for path in api.list_repo_files(
            repo_id=repository,
            repo_type="dataset",
            revision=revision,
        )
        if path.startswith(prefix_marker)
    }
    expected_remote_files = {f"{remote_prefix}/{entry['path']}" for entry in entries}
    if remote_files != expected_remote_files:
        raise ComparisonError("existing deterministic remote prefix is missing, extra, or drifted")
    with tempfile.TemporaryDirectory(prefix="lehome-official-readback-") as temporary:
        root = Path(temporary)
        for entry in entries:
            downloaded = Path(
                downloader(
                    repo_id=repository,
                    repo_type="dataset",
                    filename=f"{remote_prefix}/{entry['path']}",
                    revision=revision,
                    token=False,
                    local_dir=root,
                )
            )
            if (
                downloaded.is_symlink()
                or not downloaded.is_file()
                or downloaded.stat().st_size != entry["size"]
                or _sha256_file(downloaded) != entry["sha256"]
            ):
                raise ComparisonError("existing deterministic remote prefix is missing, extra, or drifted")


def load_release_matrix(assets_root: Path, *, episodes_per_garment: int) -> list[MatrixRow]:
    if episodes_per_garment != EPISODES_PER_GARMENT:
        raise ComparisonError("official comparison requires exactly two episodes per garment")
    base = Path(assets_root) / "objects" / "Challenge_Garment" / "Release"
    rows: list[MatrixRow] = []
    seen: set[str] = set()
    for category, directory in CATEGORY_DIRECTORIES:
        path = base / directory / f"{directory}.txt"
        if path.is_symlink() or not path.is_file():
            raise ComparisonError(f"native Release list is unavailable: {path}")
        garments = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(garments) != 12 or len(set(garments)) != 12:
            raise ComparisonError(f"{category} native Release list must contain 12 unique garments")
        if any("/" in garment or "\\" in garment or garment in seen for garment in garments):
            raise ComparisonError("native Release lists contain unsafe or duplicate garment names")
        seen.update(garments)
        for garment in garments:
            for episode_index in range(1, episodes_per_garment + 1):
                rows.append(MatrixRow(category, garment, episode_index))
    if len(rows) != 96:
        raise ComparisonError("native Release matrix must contain exactly 96 episodes")
    return rows


def load_profile_matrix(assets_root: Path, *, profile: str = DEFAULT_PROFILE) -> list[MatrixRow]:
    """Load a predeclared evaluator matrix without changing native list order."""
    rows = load_release_matrix(assets_root, episodes_per_garment=EPISODES_PER_GARMENT)
    if profile == DEFAULT_PROFILE:
        return rows
    if profile == N15_FOCUSED_PROFILE:
        focused = [row for row in rows if row.category in N15_FOCUSED_CATEGORIES]
        if (
            len(focused) != 48
            or [row.category for row in focused[:24]] != ["top_short"] * 24
            or [row.category for row in focused[24:]] != ["pant_long"] * 24
        ):
            raise ComparisonError("N1.5 focused matrix category/order drift")
        return focused
    raise ComparisonError("unknown comparison profile")


def smoke_matrix() -> list[MatrixRow]:
    return [MatrixRow("custom", "Top_Long_Seen_0", episode) for episode in (1, 2)]


def build_eval_command(
    policy: PolicyDefinition,
    *,
    source_root: Path,
    assets_root: Path,
    dataset_root: Path,
    video_dir: Path,
    garment_type: str,
    python_bin: str,
) -> list[str]:
    if garment_type not in {category for category, _ in CATEGORY_DIRECTORIES} | {"custom"}:
        raise ComparisonError("garment_type is outside the official evaluator contract")
    command = [
        python_bin,
        "-P",
        "-m",
        "scripts.eval",
        "--headless",
        "--enable_cameras",
        "--device",
        "cpu",
        "--task",
        TASK,
        "--task_description",
        TASK_DESCRIPTION,
        "--garment_type",
        garment_type,
        "--num_episodes",
        str(EPISODES_PER_GARMENT),
        "--max_steps",
        str(MAX_STEPS),
        "--seed",
        str(SEED),
        "--save_video",
        "--video_dir",
        str(video_dir),
        "--garment_cfg_base_path",
        str(Path(assets_root) / "objects" / "Challenge_Garment"),
        "--particle_cfg_path",
        str(Path(source_root) / "source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml"),
        "--ee_urdf_path",
        str(Path(assets_root) / "robots/so101_new_calib.urdf"),
        "--dataset_root",
        str(Path(dataset_root) / f"{'top_long' if garment_type == 'custom' else garment_type}_merged"),
        "--policy_type",
        policy.policy_type,
    ]
    if policy.policy_type == "docker":
        command.extend(["--docker_url", str(policy.docker_url)])
    else:
        command.extend(["--policy_path", str(policy.checkpoint_root)])
    return command


def _parse_category_log(log: Path, expected: Sequence[MatrixRow]) -> list[dict[str, object]]:
    if log.is_symlink() or not log.is_file():
        raise ComparisonError(f"infrastructure_invalid: missing evaluator log {log}")
    text = log.read_text(encoding="utf-8", errors="strict")
    if _FORBIDDEN_LOG.search(text):
        raise ComparisonError(f"infrastructure_invalid: evaluator error marker in {log.name}")
    if text.count(_COMPLETION) != 1:
        raise ComparisonError(f"infrastructure_invalid: missing or duplicate completion sentinel in {log.name}")
    expected_garments = list(dict.fromkeys(row.garment for row in expected))
    observed_garments = [match.group("garment") for match in _EVALUATING.finditer(text)]
    if observed_garments != expected_garments:
        raise ComparisonError(f"infrastructure_invalid: garment order mismatch in {log.name}")
    episode_matches = list(_EPISODE.finditer(text))
    if len(episode_matches) != len(expected):
        raise ComparisonError(f"infrastructure_invalid: episode count mismatch in {log.name}")
    outcomes: list[dict[str, object]] = []
    for row, match in zip(expected, episode_matches, strict=True):
        episode = int(match.group("episode"))
        if episode != row.episode_index:
            raise ComparisonError(f"infrastructure_invalid: episode order mismatch in {log.name}")
        try:
            episode_return = float(match.group("return"))
        except ValueError:
            raise ComparisonError(f"infrastructure_invalid: invalid return in {log.name}") from None
        if not (-float("inf") < episode_return < float("inf")):
            raise ComparisonError(f"infrastructure_invalid: nonfinite return in {log.name}")
        outcomes.append(
            {
                "category": row.category,
                "garment": row.garment,
                "episode_index": row.episode_index,
                "seed": row.seed,
                "success": match.group("success") == "True",
                "return": episode_return,
                "length": int(match.group("length")),
                "log": log.name,
            }
        )
    return outcomes


def _nonempty_video(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and path.stat().st_size > 0


def _ffprobe_video(path: Path) -> bool:
    if not _nonempty_video(path):
        return False
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "video"


def _retained_videos(
    command_root: Path,
    final_outcomes: Sequence[Mapping[str, object]],
    *,
    probe: Callable[[Path], bool],
) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    for outcome in final_outcomes:
        episode_zero = int(outcome["episode_index"]) - 1
        expected_folder = "success" if outcome["success"] else "failure"
        opposite_folder = "failure" if outcome["success"] else "success"
        for camera in CAMERAS:
            name = f"episode{episode_zero}_observation_images_{camera}.mp4"
            expected = command_root / expected_folder / name
            opposite = command_root / opposite_folder / name
            if not probe(expected):
                raise ComparisonError(
                    f"infrastructure_invalid: missing or undecodable retained official video {name}"
                )
            descriptor = {
                    "path": expected.relative_to(command_root.parent.parent).as_posix(),
                    "sha256": _sha256_file(expected),
                    "size": expected.stat().st_size,
                    "represents_garment": outcome["garment"],
                    "episode_index": outcome["episode_index"],
                    "camera": camera,
                }
            if opposite.exists():
                if opposite.is_symlink() or not opposite.is_file():
                    raise ComparisonError(f"infrastructure_invalid: unsafe stale official video {name}")
                descriptor["stale_opposite_path"] = opposite.relative_to(command_root.parent.parent).as_posix()
                descriptor["stale_opposite_sha256"] = _sha256_file(opposite)
            artifacts.append(descriptor)
    return artifacts


def compile_policy_result(
    *,
    policy_id: str,
    matrix: Sequence[MatrixRow],
    logs_root: Path,
    videos_root: Path,
    fidelity_root: Path | None = None,
    video_probe: Callable[[Path], bool] = _nonempty_video,
) -> dict[str, object]:
    if len(matrix) == 96:
        categories = [category for category, _ in CATEGORY_DIRECTORIES]
        expected_per_category = 24
    elif len(matrix) == 48 and tuple(dict.fromkeys(row.category for row in matrix)) == N15_FOCUSED_CATEGORIES:
        categories = list(N15_FOCUSED_CATEGORIES)
        expected_per_category = 24
    elif list(matrix) == smoke_matrix():
        categories = ["custom"]
        expected_per_category = 2
    else:
        raise ComparisonError("infrastructure_invalid: comparison matrix is neither official full nor smoke")
    focused = len(matrix) == 48 and tuple(
        dict.fromkeys(row.category for row in matrix)
    ) == N15_FOCUSED_CATEGORIES
    if focused and fidelity_root is None:
        raise ComparisonError("fidelity_invalid: focused comparison requires measured cloth evidence")
    outcomes: list[dict[str, object]] = []
    videos: list[dict[str, object]] = []
    cloth_fidelity: dict[str, object] = {
        "measured_episode_count": 0,
        "fidelity_invalid_count": 0,
        "event_count": 0,
        "categories": {},
    }
    for category in categories:
        expected = [row for row in matrix if row.category == category]
        if len(expected) != expected_per_category:
            raise ComparisonError("infrastructure_invalid: category matrix count drift")
        log = Path(logs_root) / f"{policy_id}-{category}.log"
        category_outcomes = _parse_category_log(log, expected)
        outcomes.extend(category_outcomes)
        if focused:
            from rollout_appliance.native_reference_site.cloth_fidelity import (
                validate_cloth_fidelity_evidence,
            )

            try:
                measured = validate_cloth_fidelity_evidence(
                    Path(fidelity_root) / f"{policy_id}-{category}" / "cloth-fidelity.jsonl",
                    expected_episodes=[(row.garment, row.episode_index) for row in expected],
                )
            except ValueError as error:
                raise ComparisonError(f"fidelity_invalid: {category}: {error}") from None
            cloth_fidelity["categories"][category] = measured
            for key in ("measured_episode_count", "fidelity_invalid_count", "event_count"):
                cloth_fidelity[key] += int(measured[key])
        final_garment = expected[-1].garment
        final_outcomes = [row for row in category_outcomes if row["garment"] == final_garment]
        videos.extend(
            _retained_videos(
                Path(videos_root) / f"{policy_id}-{category}",
                final_outcomes,
                probe=video_probe,
            )
        )
    if [
        (row["category"], row["garment"], row["episode_index"], row["seed"])
        for row in outcomes
    ] != [(row.category, row.garment, row.episode_index, row.seed) for row in matrix]:
        raise ComparisonError("infrastructure_invalid: compiled outcome order drift")
    fidelity_invalid_count = int(cloth_fidelity["fidelity_invalid_count"])
    if focused and fidelity_invalid_count:
        raise ComparisonError(
            f"fidelity_invalid: measured {fidelity_invalid_count} invalid cloth episodes"
        )
    return {
        "policy_id": policy_id,
        "status": "valid",
        "episode_count": len(outcomes),
        "success_count": sum(bool(row["success"]) for row in outcomes),
        "fidelity_invalid_count": fidelity_invalid_count,
        "infrastructure_invalid_count": 0,
        "cloth_fidelity": cloth_fidelity if focused else None,
        "outcomes": outcomes,
        "retained_official_videos": videos,
        "video_scope": "official filenames overwrite per garment; retained files represent only each category's final garment",
    }


def _write_immutable(path: Path, value: object) -> str:
    raw = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256_bytes(raw)


def _copy_immutable_file(source: Path, destination: Path) -> dict[str, object]:
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise ComparisonError(f"evidence source is unavailable or unsafe: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
        for block in iter(lambda: input_stream.read(1024 * 1024), b""):
            output_stream.write(block)
            digest.update(block)
            size += len(block)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    return {"path": destination.as_posix(), "size": size, "sha256": digest.hexdigest()}


def _command_parity(commands: Mapping[str, Sequence[str]]) -> dict[str, object]:
    normalized: dict[str, list[str]] = {}
    policy_specific = {"--policy_type", "--docker_url", "--policy_path"}
    for key, command in commands.items():
        common: list[str] = []
        index = 0
        while index < len(command):
            token = command[index]
            if token in policy_specific:
                index += 2
                continue
            if token == "--video_dir":
                common.extend([token, "<fresh-policy-category-video-dir>"])
                index += 2
                continue
            common.append(token)
            index += 1
        normalized[key] = common
    category_digests: dict[str, str] = {}
    known_categories = [category for category, _ in CATEGORY_DIRECTORIES] + ["custom"]
    categories = [
        category
        for category in known_categories
        if any(key.endswith(f"-{category}") for key in normalized)
    ]
    if not categories:
        raise ComparisonError("infrastructure_invalid: no comparison categories found")
    for category in categories:
        selected = [value for key, value in normalized.items() if key.endswith(f"-{category}")]
        if len(selected) != 2 or selected[0] != selected[1]:
            raise ComparisonError(f"infrastructure_invalid: command parity drift for {category}")
        category_digests[category] = _sha256_bytes(_canonical_bytes(selected[0]))
    return {"verified": True, "category_common_command_sha256": category_digests}


def assess_n15_focused_promotion(
    receipt: Mapping[str, object],
    *,
    publication: Mapping[str, object],
    receipt_sha256: str,
) -> dict[str, object]:
    """Return pass only after the paired gate and anonymous publication readback hold."""
    if (
        receipt.get("kind") != "lehome_official_policy_comparison_v1"
        or receipt.get("status") != "valid"
        or receipt.get("mode") != "full"
        or receipt.get("profile") != N15_FOCUSED_PROFILE
    ):
        raise ComparisonError("N1.5 focused receipt identity/status is invalid")
    required_provenance = {
        "official_source",
        "canonical_assets",
        "reviewed_runtime",
        "candidate_checkpoint",
        "reference_checkpoint",
        "metadata",
        "scorer_sha256",
        "frozen_reference_matrix_sha256",
        "matrix_sha256",
        "command_parity",
        "simulator_device",
        "policy_device",
        "seed",
        "max_steps",
        "episodes_per_garment",
    }
    if any(key not in receipt or receipt[key] in (None, "", {}, []) for key in required_provenance):
        raise ComparisonError("N1.5 focused provenance is incomplete")
    sha256 = lambda value: re.fullmatch(r"[0-9a-f]{64}", str(value)) is not None
    source = receipt.get("official_source")
    assets = receipt.get("canonical_assets")
    runtime = receipt.get("reviewed_runtime")
    candidate_checkpoint = receipt.get("candidate_checkpoint")
    reference_checkpoint = receipt.get("reference_checkpoint")
    metadata = receipt.get("metadata")
    archive = receipt.get("evidence_archive")
    expected_adapters = {
        "scripts/run_official_lehome_comparison.py",
        "rollout_appliance/run_public_n15_focused_gate.sh",
        "rollout_appliance/native_reference_site/sitecustomize.py",
        "rollout_appliance/native_reference_site/checkpoint_compatibility.py",
        "rollout_appliance/native_reference_site/cloth_fidelity.py",
    }
    if (
        not isinstance(source, Mapping)
        or set(source) != {"repository", "revision", "tree_sha256"}
        or source.get("repository") != SOURCE_REPOSITORY
        or source.get("revision") != SOURCE_REVISION
        or not sha256(source.get("tree_sha256"))
        or not isinstance(assets, Mapping)
        or set(assets) != {"repository", "revision", "tree_sha256"}
        or assets.get("repository") != ASSET_REPOSITORY
        or assets.get("revision") != ASSET_REVISION
        or not sha256(assets.get("tree_sha256"))
        or not isinstance(runtime, Mapping)
        or set(runtime) != {"revision", "tree_sha256", "adapter_sha256"}
        or _HEX40.fullmatch(str(runtime.get("revision"))) is None
        or not sha256(runtime.get("tree_sha256"))
        or not isinstance(runtime.get("adapter_sha256"), Mapping)
        or set(runtime["adapter_sha256"]) != expected_adapters
        or any(not sha256(value) for value in runtime["adapter_sha256"].values())
        or not isinstance(candidate_checkpoint, Mapping)
        or set(candidate_checkpoint) != {"kind", "step", "identity_receipt_sha256", "tree_sha256", "file_count"}
        or candidate_checkpoint.get("kind") != "lehome_public_n15_verified_training_output_v1"
        or candidate_checkpoint.get("step") != 12000
        or not sha256(candidate_checkpoint.get("identity_receipt_sha256"))
        or not sha256(candidate_checkpoint.get("tree_sha256"))
        or type(candidate_checkpoint.get("file_count")) is not int
        or candidate_checkpoint["file_count"] < 1
        or reference_checkpoint != {
            "repository": "theo-zhou/lehome-groot-submission-4",
            "revision": "d384fe00508acd96ab1c3c5dc265e08261f94b3b",
            "file_sha256": dict(sorted(COMPETITOR_FILES.items())),
            "tree_sha256": COMPETITOR_TREE_SHA256,
        }
        or not isinstance(metadata, Mapping)
        or set(metadata) != {"tree_sha256", "category_tree_sha256", "policy_visibility"}
        or not sha256(metadata.get("tree_sha256"))
        or not isinstance(metadata.get("category_tree_sha256"), Mapping)
        or set(metadata["category_tree_sha256"]) != {category for category, _ in CATEGORY_DIRECTORIES}
        or any(not sha256(value) for value in metadata["category_tree_sha256"].values())
        or metadata.get("policy_visibility") != "LeRobot construction only; never included in DockerPolicy observations"
        or receipt.get("scorer_sha256") != OFFICIAL_SCORER_SHA256
        or receipt.get("frozen_reference_matrix_sha256") != FROZEN_REFERENCE_MATRIX_SHA256
        or not isinstance(archive, Mapping)
    ):
        raise ComparisonError("N1.5 focused exact provenance drift")
    archive_links = {
        "runtime-identity.json": _sha256_bytes(
            _canonical_bytes({"revision": runtime["revision"], "tree_sha256": runtime["tree_sha256"]})
        ),
        "candidate-checkpoint-identity.json": candidate_checkpoint["identity_receipt_sha256"],
        "candidate-checkpoint-compatibility.json": receipt.get("candidate_compatibility_receipt_sha256"),
        "reference-checkpoint-compatibility.json": receipt.get("reference_compatibility_receipt_sha256"),
    }
    for name, expected_sha in archive_links.items():
        descriptor = archive.get(name)
        if (
            not sha256(expected_sha)
            or not isinstance(descriptor, Mapping)
            or set(descriptor) != {"path", "size", "sha256"}
            or descriptor.get("path") != f"evidence/{name}"
            or type(descriptor.get("size")) is not int
            or descriptor["size"] < 1
            or descriptor.get("sha256") != expected_sha
        ):
            raise ComparisonError("N1.5 focused exact provenance archive drift")
    if (
        receipt.get("simulator_device") != "cpu"
        or receipt.get("policy_device") != "cuda:0"
        or receipt.get("seed") != SEED
        or receipt.get("max_steps") != MAX_STEPS
        or receipt.get("episodes_per_garment") != EPISODES_PER_GARMENT
        or not isinstance(receipt.get("command_parity"), Mapping)
        or receipt["command_parity"].get("verified") is not True
        or set(receipt["command_parity"]) != {"verified", "category_common_command_sha256"}
        or set(receipt["command_parity"]["category_common_command_sha256"]) != set(N15_FOCUSED_CATEGORIES)
        or any(not sha256(value) for value in receipt["command_parity"]["category_common_command_sha256"].values())
    ):
        raise ComparisonError("N1.5 focused provenance contract drift")
    matrix_payload = receipt.get("matrix")
    if not isinstance(matrix_payload, list) or len(matrix_payload) != 48:
        raise ComparisonError("N1.5 focused matrix provenance is incomplete")
    try:
        matrix = [
            MatrixRow(
                category=row["category"],
                garment=row["garment"],
                episode_index=row["episode_index"],
                seed=row["seed"],
            )
            for row in matrix_payload
            if isinstance(row, Mapping)
        ]
    except (KeyError, TypeError, ValueError):
        raise ComparisonError("N1.5 focused matrix provenance is invalid") from None
    if (
        len(matrix) != 48
        or tuple(dict.fromkeys(row.category for row in matrix)) != N15_FOCUSED_CATEGORIES
        or any(row.seed != SEED for row in matrix)
        or receipt.get("matrix_sha256") != _sha256_bytes(_canonical_bytes(matrix_payload))
    ):
        raise ComparisonError("N1.5 focused matrix provenance drift")
    for category in N15_FOCUSED_CATEGORIES:
        category_rows = [row for row in matrix if row.category == category]
        garments = list(dict.fromkeys(row.garment for row in category_rows))
        if (
            len(category_rows) != 24
            or len(garments) != 12
            or any(
                [row.episode_index for row in category_rows if row.garment == garment] != [1, 2]
                for garment in garments
            )
        ):
            raise ComparisonError("N1.5 focused matrix provenance is not twelve paired garments per category")
    expected_provenance = [
        (row.category, row.garment, row.episode_index, row.seed) for row in matrix
    ]
    results = receipt.get("results")
    if (
        not isinstance(results, list)
        or len(results) != 2
        or [row.get("policy_id") for row in results if isinstance(row, Mapping)]
        != ["candidate-n15", "reference-n15"]
    ):
        raise ComparisonError("N1.5 focused paired policy provenance is invalid")
    by_policy: dict[str, Mapping[str, object]] = {}
    for result in results:
        if not isinstance(result, Mapping):
            raise ComparisonError("N1.5 focused result provenance is invalid")
        outcomes = result.get("outcomes")
        if (
            result.get("status") != "valid"
            or result.get("episode_count") != 48
            or result.get("fidelity_invalid_count") != 0
            or result.get("infrastructure_invalid_count") != 0
            or not isinstance(outcomes, list)
            or len(outcomes) != 48
            or not isinstance(result.get("cloth_fidelity"), Mapping)
            or result["cloth_fidelity"].get("measured_episode_count") != 48
            or result["cloth_fidelity"].get("fidelity_invalid_count") != 0
            or set(result["cloth_fidelity"].get("categories", {})) != set(N15_FOCUSED_CATEGORIES)
            or any(
                not isinstance(summary, Mapping)
                or summary.get("measured_episode_count") != 24
                or summary.get("fidelity_invalid_count") != 0
                or any(not sha256(summary.get(key)) for key in ("first_event_sha256", "last_event_sha256", "evidence_sha256"))
                for summary in result["cloth_fidelity"].get("categories", {}).values()
            )
        ):
            raise ComparisonError("N1.5 focused fidelity or infrastructure gate failed")
        observed = [
            (row.get("category"), row.get("garment"), row.get("episode_index"), row.get("seed"))
            for row in outcomes
            if isinstance(row, Mapping)
        ]
        if observed != expected_provenance or any(
            type(row.get("success")) is not bool for row in outcomes if isinstance(row, Mapping)
        ):
            raise ComparisonError("N1.5 focused episode provenance drift")
        by_policy[str(result["policy_id"])] = result

    category_scores: dict[str, dict[str, int]] = {}
    for category in N15_FOCUSED_CATEGORIES:
        candidate = sum(
            bool(row["success"])
            for row in by_policy["candidate-n15"]["outcomes"]
            if row["category"] == category
        )
        reference = sum(
            bool(row["success"])
            for row in by_policy["reference-n15"]["outcomes"]
            if row["category"] == category
        )
        floor = N15_FOCUSED_FLOORS[category]
        if candidate < floor:
            raise ComparisonError(f"{category} floor failed: {candidate}/24 < {floor}/24")
        if reference - candidate > N15_FOCUSED_MAXIMUM_DEFICIT:
            raise ComparisonError(
                f"{category} deficit failed: candidate is {reference - candidate} behind reference"
            )
        category_scores[category] = {
            "candidate": candidate,
            "reference": reference,
            "floor": floor,
            "maximum_deficit": N15_FOCUSED_MAXIMUM_DEFICIT,
        }

    if (
        re.fullmatch(r"[0-9a-f]{64}", receipt_sha256) is None
        or publication.get("kind") != "lehome_official_policy_comparison_publication_v1"
        or publication.get("comparison_receipt_sha256") != receipt_sha256
        or _HEX40.fullmatch(str(publication.get("immutable_revision"))) is None
        or publication.get("anonymous_file_set_verified") is not True
        or publication.get("anonymous_byte_readback_verified") is not True
    ):
        raise ComparisonError("N1.5 focused publication readback is missing or invalid")
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_focused_promotion_v1",
        "status": "pass",
        "profile": N15_FOCUSED_PROFILE,
        "comparison_receipt_sha256": receipt_sha256,
        "publication_immutable_revision": publication["immutable_revision"],
        "publication_readback_verified": True,
        "category_scores": category_scores,
    }


def _execution_env(
    *,
    source_root: Path,
    log_root: Path,
    isaaclab_root: Path,
    isaaclab_tasks_root: Path,
    native_site_root: Path,
    policy: PolicyDefinition,
    sanitized_config_root: Path | None,
    compatibility_receipt: Path | None,
    cloth_fidelity_monitor: bool = False,
) -> dict[str, str]:
    env = os.environ.copy()
    python_path = [
        str(source_root / "source/lehome"),
        str(source_root),
        str(isaaclab_root),
        str(isaaclab_tasks_root),
        str(native_site_root),
    ]
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONSAFEPATH": "1",
            "PYNPUT_BACKEND": "dummy",
            "PYTHONPATH": os.pathsep.join(python_path),
            "LEHOME_NATIVE_REFERENCE_LOG_PROJECT_ROOT": str(log_root),
            "LEHOME_NATIVE_REFERENCE_SOURCE_ROOT": str(source_root),
        }
    )
    if cloth_fidelity_monitor:
        env["LEHOME_NATIVE_CLOTH_FIDELITY_EVIDENCE"] = str(
            log_root / "cloth-fidelity.jsonl"
        )
    if policy.policy_type == "lerobot":
        if sanitized_config_root is None or compatibility_receipt is None:
            raise ComparisonError("competitor compatibility view inputs are required")
        env.update(
            {
                "PYTHONEXE": "/opt/lehome-challenge/.venv/bin/python",
                "LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT": str(policy.checkpoint_root),
                "LEHOME_NATIVE_REFERENCE_SANITIZED_CONFIG_ROOT": str(sanitized_config_root),
                "LEHOME_NATIVE_REFERENCE_CHECKPOINT_COMPATIBILITY_RECEIPT": str(compatibility_receipt),
            }
        )
        from scripts.verify_native_reference_evaluator_gate import PEFT_WHEEL_PATH

        python_path.insert(0, str(PEFT_WHEEL_PATH))
        env["PYTHONPATH"] = os.pathsep.join(python_path)
    return env


def _validate_focused_cuda_receipt(receipt: Mapping[str, object]) -> dict[str, object]:
    if (
        set(receipt) != {"cuda_available", "cuda_device_count", "cuda_runtime", "cuda_device_name"}
        or receipt.get("cuda_available") is not True
        or type(receipt.get("cuda_device_count")) is not int
        or receipt["cuda_device_count"] < 1
        or not isinstance(receipt.get("cuda_runtime"), str)
        or not receipt["cuda_runtime"]
        or not isinstance(receipt.get("cuda_device_name"), str)
        or not receipt["cuda_device_name"]
    ):
        raise ComparisonError("focused N1.5 CUDA receipt is invalid")
    return dict(receipt)


def _validate_focused_rollout_image(receipt: Mapping[str, object]) -> dict[str, object]:
    if (
        receipt.get("kind") != "lehome_official_image_inspection_v1"
        or receipt.get("reference") != ROLLOUT_IMAGE_ID
        or receipt.get("image_id") != ROLLOUT_IMAGE_ID
        or re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("docker_inspect_sha256"))) is None
    ):
        raise ComparisonError("focused N1.5 rollout image receipt is invalid")
    return dict(receipt)


def execute_n15_focused_comparison(args: argparse.Namespace) -> Path:
    """Execute two native N1.5 policies sequentially on the focused matrix."""
    source_root = args.source_root.resolve(strict=True)
    canonical_assets_root = args.canonical_assets_root.resolve(strict=True)
    output_root = args.output_root
    if output_root.exists() or output_root.is_symlink():
        raise ComparisonError("output root must be a new path")
    output_root.mkdir(parents=True, mode=0o700)
    args._official_output_created = True
    status_path = output_root / "status.json"
    source_before = checkout_identity(
        source_root, SOURCE_REVISION, label="official source", exclude_assets_mount=True
    )
    assets_before = checkout_identity(
        canonical_assets_root, ASSET_REVISION, label="canonical assets"
    )
    validate_evaluation_assets(canonical_assets_root, canonical_assets_root, mode="full")
    runtime_root = args.runtime_root.resolve(strict=True)
    runtime_before = {"revision": args.runtime_revision, "tree_sha256": _tree_sha256(runtime_root)}
    if _load_json_object(args.runtime_identity_receipt, "runtime identity receipt") != runtime_before:
        raise ComparisonError("reviewed runtime identity receipt mismatch")
    adapters_before = focused_runtime_adapter_identities(runtime_root)
    candidate_before = validate_candidate_n15_checkpoint(
        args.candidate_checkpoint, args.candidate_identity_receipt
    )
    reference_before = validate_competitor_checkpoint(args.reference_checkpoint)
    metadata_before = metadata_identities(args.metadata_root)
    native_runtime_before = validate_competitor_runtime_evidence(args.native_runtime_evidence_root)
    rollout_image_before = _validate_focused_rollout_image(
        _load_json_object(args.rollout_image_receipt, "rollout image receipt")
    )
    cuda_before = _validate_focused_cuda_receipt(
        _load_json_object(args.cuda_receipt, "CUDA receipt")
    )
    scorer = source_root / "source/lehome/lehome/utils/success_checker_chanllege.py"
    if scorer.is_symlink() or not scorer.is_file():
        raise ComparisonError("official scorer is unavailable")
    full_matrix = load_profile_matrix(canonical_assets_root, profile=DEFAULT_PROFILE)
    frozen_reference_matrix_sha = validate_reference_matrix(
        args.reference_matrix, args.reference_matrix_sha256, full_matrix
    )
    matrix = load_profile_matrix(canonical_assets_root, profile=N15_FOCUSED_PROFILE)
    matrix_payload = [asdict(row) for row in matrix]
    matrix_sha = _sha256_bytes(_canonical_bytes(matrix_payload))
    policies = (
        PolicyDefinition("candidate-n15", "lerobot", checkpoint_root=args.candidate_checkpoint),
        PolicyDefinition("reference-n15", "lerobot", checkpoint_root=args.reference_checkpoint),
    )
    policy_views = {
        "candidate-n15": (args.candidate_sanitized_config_root, args.candidate_compatibility_receipt),
        "reference-n15": (args.reference_sanitized_config_root, args.reference_compatibility_receipt),
    }
    evidence_sources: dict[str, Path] = {
        "runtime-identity.json": args.runtime_identity_receipt,
        "rollout-image.json": args.rollout_image_receipt,
        "cuda-runtime.json": args.cuda_receipt,
        "candidate-checkpoint-identity.json": args.candidate_identity_receipt,
        "candidate-checkpoint-compatibility.json": args.candidate_compatibility_receipt,
        "reference-checkpoint-compatibility.json": args.reference_compatibility_receipt,
        "reference-matrix.json": args.reference_matrix,
        "reference-matrix.sha256": args.reference_matrix_sha256,
    }
    native_evidence_root = args.native_runtime_evidence_root.resolve(strict=True)
    for path in sorted(native_evidence_root.iterdir()):
        evidence_sources[f"native-runtime/{path.name}"] = path
    evidence_archive: dict[str, dict[str, object]] = {}
    for relative, source in evidence_sources.items():
        descriptor = _copy_immutable_file(source, output_root / "evidence" / relative)
        descriptor["path"] = (output_root / "evidence" / relative).relative_to(output_root).as_posix()
        evidence_archive[relative] = descriptor
    commands: dict[str, list[str]] = {}
    try:
        # Tuple order is contractual: candidate finishes before the reference starts.
        for policy in policies:
            sanitized_root, compatibility_receipt = policy_views[policy.policy_id]
            for category in N15_FOCUSED_CATEGORIES:
                command_id = f"{policy.policy_id}-{category}"
                video_dir = output_root / "videos" / command_id
                video_dir.mkdir(parents=True)
                command = build_eval_command(
                    policy,
                    source_root=source_root,
                    assets_root=canonical_assets_root,
                    dataset_root=args.metadata_root,
                    video_dir=video_dir,
                    garment_type=category,
                    python_bin=args.python_bin,
                )
                commands[command_id] = command
                log = output_root / "logs" / f"{command_id}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                runtime_log = output_root / "official-runtime" / command_id
                runtime_log.mkdir(parents=True)
                env = _execution_env(
                    source_root=source_root,
                    log_root=runtime_log,
                    isaaclab_root=args.isaaclab_root,
                    isaaclab_tasks_root=args.isaaclab_tasks_root,
                    native_site_root=args.native_site_root,
                    policy=policy,
                    sanitized_config_root=sanitized_root,
                    compatibility_receipt=compatibility_receipt,
                    cloth_fidelity_monitor=True,
                )
                with log.open("xb") as stream:
                    result = subprocess.run(
                        command,
                        cwd=source_root,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                if result.returncode != 0:
                    raise ComparisonError(
                        f"infrastructure_invalid: {command_id} exited {result.returncode}"
                    )
        parity = _command_parity(commands)
        results = [
            compile_policy_result(
                policy_id=policy.policy_id,
                matrix=matrix,
                logs_root=output_root / "logs",
                videos_root=output_root / "videos",
                fidelity_root=output_root / "official-runtime",
                video_probe=_ffprobe_video,
            )
            for policy in policies
        ]
        source_after = checkout_identity(
            source_root, SOURCE_REVISION, label="official source", exclude_assets_mount=True
        )
        assets_after = checkout_identity(
            canonical_assets_root, ASSET_REVISION, label="canonical assets"
        )
        runtime_after = {"revision": args.runtime_revision, "tree_sha256": _tree_sha256(runtime_root)}
        if (
            source_after != source_before
            or assets_after != assets_before
            or runtime_after != runtime_before
            or focused_runtime_adapter_identities(runtime_root) != adapters_before
            or validate_candidate_n15_checkpoint(
                args.candidate_checkpoint, args.candidate_identity_receipt
            )
            != candidate_before
            or validate_competitor_checkpoint(args.reference_checkpoint) != reference_before
            or metadata_identities(args.metadata_root) != metadata_before
            or validate_competitor_runtime_evidence(args.native_runtime_evidence_root)
            != native_runtime_before
            or _validate_focused_rollout_image(
                _load_json_object(args.rollout_image_receipt, "rollout image receipt")
            )
            != rollout_image_before
            or _validate_focused_cuda_receipt(
                _load_json_object(args.cuda_receipt, "CUDA receipt")
            )
            != cuda_before
        ):
            raise ComparisonError("infrastructure_invalid: focused immutable identity changed")
        receipt = {
            "schema_version": 1,
            "kind": "lehome_official_policy_comparison_v1",
            "status": "valid",
            "mode": "full",
            "profile": N15_FOCUSED_PROFILE,
            "created_at_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "official_source": {
                "repository": SOURCE_REPOSITORY,
                "revision": SOURCE_REVISION,
                **source_before,
            },
            "canonical_assets": {
                "repository": ASSET_REPOSITORY,
                "revision": ASSET_REVISION,
                **assets_before,
            },
            "reviewed_runtime": {**runtime_before, "adapter_sha256": adapters_before},
            "rollout_image": rollout_image_before,
            "cuda": cuda_before,
            "native_runtime_evidence": native_runtime_before,
            "candidate_checkpoint": candidate_before,
            "reference_checkpoint": reference_before,
            "candidate_compatibility_receipt_sha256": _sha256_file(
                args.candidate_compatibility_receipt
            ),
            "reference_compatibility_receipt_sha256": _sha256_file(
                args.reference_compatibility_receipt
            ),
            "metadata": metadata_before,
            "scorer_sha256": _sha256_file(scorer),
            "frozen_reference_matrix_sha256": frozen_reference_matrix_sha,
            "evidence_archive": evidence_archive,
            "simulator_device": "cpu",
            "policy_device": "cuda:0",
            "seed": SEED,
            "max_steps": MAX_STEPS,
            "episodes_per_garment": EPISODES_PER_GARMENT,
            "matrix_sha256": matrix_sha,
            "matrix": matrix_payload,
            "commands": commands,
            "command_parity": parity,
            "results": results,
            "publication": "not_performed; publication/readback is required before promotion",
        }
        receipt_path = output_root / "comparison-receipt.json"
        _write_immutable(receipt_path, receipt)
        seal_execution_bundle(output_root)
        return receipt_path
    except Exception as error:
        _write_immutable(
            status_path,
            {"status": "infrastructure_invalid", "error_type": type(error).__name__, "error": str(error)},
        )
        raise


def execute_comparison(args: argparse.Namespace) -> Path:
    source_root = args.source_root.resolve(strict=True)
    canonical_assets_root = args.canonical_assets_root.resolve(strict=True)
    evaluation_assets_root = args.evaluation_assets_root.resolve(strict=True)
    output_root = args.output_root
    if output_root.exists() or output_root.is_symlink():
        raise ComparisonError("output root must be a new path")
    output_root.mkdir(parents=True, mode=0o700)
    args._official_output_created = True
    status_path = output_root / "status.json"
    source_before = checkout_identity(
        source_root,
        SOURCE_REVISION,
        label="official source",
        exclude_assets_mount=True,
    )
    assets_before = checkout_identity(canonical_assets_root, ASSET_REVISION, label="canonical assets")
    runtime_root = args.runtime_root.resolve(strict=True)
    runtime_before = {"revision": args.runtime_revision, "tree_sha256": _tree_sha256(runtime_root)}
    if _load_json_object(args.runtime_identity_receipt, "runtime identity receipt") != runtime_before:
        raise ComparisonError("reviewed runtime identity receipt mismatch")
    adapters_before = runtime_adapter_identities(runtime_root)
    validate_evaluation_assets(canonical_assets_root, evaluation_assets_root, mode=args.mode)
    scorer = source_root / "source/lehome/lehome/utils/success_checker_chanllege.py"
    if scorer.is_symlink() or not scorer.is_file():
        raise ComparisonError("official scorer is unavailable")
    if args.mode == "full":
        matrix = load_release_matrix(canonical_assets_root, episodes_per_garment=2)
        categories = [category for category, _ in CATEGORY_DIRECTORIES]
    else:
        smoke_list = (
            evaluation_assets_root
            / "objects/Challenge_Garment/Release/Release_test_list.txt"
        )
        if smoke_list.is_symlink() or not smoke_list.is_file() or smoke_list.read_text(encoding="utf-8") != "Top_Long_Seen_0\n":
            raise ComparisonError("external smoke asset-list view must contain only Top_Long_Seen_0")
        matrix = smoke_matrix()
        categories = ["custom"]
    reference_matrix_sha = validate_reference_matrix(
        args.reference_matrix,
        args.reference_matrix_sha256,
        load_release_matrix(canonical_assets_root, episodes_per_garment=2),
    )
    n17_before = validate_n17_checkpoint(args.n17_checkpoint, args.n17_identity_receipt)
    competitor_before = validate_competitor_checkpoint(args.competitor_checkpoint)
    metadata_before = metadata_identities(args.metadata_root)
    runtime_evidence_before = validate_runtime_evidence(
        runtime_identity=runtime_before,
        rollout_image=_load_json_object(args.rollout_image_receipt, "rollout image receipt"),
        policy_image=_load_json_object(args.policy_image_receipt, "policy image receipt"),
        cuda_receipt=_load_json_object(args.cuda_receipt, "CUDA receipt"),
        readiness_receipt=_load_json_object(args.policy_server_readiness_receipt, "policy server readiness receipt"),
        n17_checkpoint=args.n17_checkpoint,
    )
    competitor_runtime_before = validate_competitor_runtime_evidence(args.competitor_runtime_evidence_root)
    if args.policy_server_startup_log.is_symlink() or not args.policy_server_startup_log.is_file():
        raise ComparisonError("policy server startup log is unavailable or unsafe")
    policy_server_log = {
        "scope": "startup_through_authenticated_readiness",
        "path": str(args.policy_server_startup_log),
        "sha256": _sha256_file(args.policy_server_startup_log),
        "size": args.policy_server_startup_log.stat().st_size,
    }
    matrix_payload = [asdict(row) for row in matrix]
    matrix_sha = _sha256_bytes(_canonical_bytes(matrix_payload))
    policies = (
        PolicyDefinition("ours-12k", "docker", docker_url=args.docker_url),
        PolicyDefinition("competitor-n15", "lerobot", checkpoint_root=args.competitor_checkpoint),
    )
    current_identity = {
        "official_source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, **source_before},
        "canonical_assets": {"repository": ASSET_REPOSITORY, "revision": ASSET_REVISION, **assets_before},
        "reviewed_runtime": {**runtime_before, "adapter_sha256": adapters_before},
        "runtime_evidence": runtime_evidence_before,
        "competitor_runtime_evidence": competitor_runtime_before,
        "n17_checkpoint": n17_before,
        "competitor_checkpoint": competitor_before,
        "metadata": metadata_before,
        "scorer_sha256": _sha256_file(scorer),
        "frozen_reference_matrix_sha256": reference_matrix_sha,
        "simulator_device": "cpu",
        "policy_device": runtime_evidence_before["policy_device"],
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "episodes_per_garment": EPISODES_PER_GARMENT,
    }
    if args.mode == "full":
        if args.smoke_receipt is None:
            raise ComparisonError("full comparison requires a valid smoke receipt")
        smoke_seal = validate_sealed_execution(args.smoke_receipt)
        validate_smoke_prerequisite(smoke_seal["comparison"], expected=current_identity)
    elif args.smoke_receipt is not None:
        raise ComparisonError("smoke mode does not accept a smoke receipt")
    evidence_sources: dict[str, Path] = {
        "runtime-identity.json": args.runtime_identity_receipt,
        "rollout-image.json": args.rollout_image_receipt,
        "policy-image.json": args.policy_image_receipt,
        "cuda-runtime.json": args.cuda_receipt,
        "policy-server-readiness.json": args.policy_server_readiness_receipt,
        "policy-server-startup.log": args.policy_server_startup_log,
        "n17-checkpoint-identity.json": args.n17_identity_receipt,
        "competitor-checkpoint-compatibility.json": args.compatibility_receipt,
        "reference-matrix.json": args.reference_matrix,
        "reference-matrix.sha256": args.reference_matrix_sha256,
    }
    competitor_evidence_root = args.competitor_runtime_evidence_root.resolve(strict=True)
    for path in sorted(competitor_evidence_root.iterdir()):
        evidence_sources[f"competitor-runtime/{path.name}"] = path
    evidence_archive: dict[str, dict[str, object]] = {}
    for relative, source in evidence_sources.items():
        descriptor = _copy_immutable_file(source, output_root / "evidence" / relative)
        descriptor["path"] = (output_root / "evidence" / relative).relative_to(output_root).as_posix()
        evidence_archive[relative] = descriptor
    smoke_prerequisite = None
    if args.mode == "full":
        smoke_prerequisite = {
            "receipt_sha256": smoke_seal["receipt_sha256"],
            "execution_manifest_sha256": smoke_seal["manifest_sha256"],
        }
    commands: dict[str, list[str]] = {}
    try:
        for policy in policies:
            for category in categories:
                command_id = f"{policy.policy_id}-{category}"
                video_dir = output_root / "videos" / command_id
                video_dir.mkdir(parents=True)
                command = build_eval_command(
                    policy,
                    source_root=source_root,
                    assets_root=evaluation_assets_root,
                    dataset_root=args.metadata_root,
                    video_dir=video_dir,
                    garment_type=category,
                    python_bin=args.python_bin,
                )
                commands[command_id] = command
                log = output_root / "logs" / f"{command_id}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                runtime_log = output_root / "official-runtime" / command_id
                runtime_log.mkdir(parents=True)
                env = _execution_env(
                    source_root=source_root,
                    log_root=runtime_log,
                    isaaclab_root=args.isaaclab_root,
                    isaaclab_tasks_root=args.isaaclab_tasks_root,
                    native_site_root=args.native_site_root,
                    policy=policy,
                    sanitized_config_root=args.sanitized_config_root,
                    compatibility_receipt=args.compatibility_receipt,
                )
                with log.open("xb") as stream:
                    result = subprocess.run(
                        command,
                        cwd=source_root,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                if result.returncode != 0:
                    raise ComparisonError(f"infrastructure_invalid: {command_id} exited {result.returncode}")
        parity = _command_parity(commands)
        results = [
            compile_policy_result(
                policy_id=policy.policy_id,
                matrix=matrix,
                logs_root=output_root / "logs",
                videos_root=output_root / "videos",
                video_probe=_ffprobe_video,
            )
            for policy in policies
        ]
        source_after = checkout_identity(
            source_root,
            SOURCE_REVISION,
            label="official source",
            exclude_assets_mount=True,
        )
        assets_after = checkout_identity(canonical_assets_root, ASSET_REVISION, label="canonical assets")
        runtime_after = {"revision": args.runtime_revision, "tree_sha256": _tree_sha256(runtime_root)}
        adapters_after = runtime_adapter_identities(runtime_root)
        n17_after = validate_n17_checkpoint(args.n17_checkpoint, args.n17_identity_receipt)
        competitor_after = validate_competitor_checkpoint(args.competitor_checkpoint)
        metadata_after = metadata_identities(args.metadata_root)
        runtime_evidence_after = validate_runtime_evidence(
            runtime_identity=runtime_after,
            rollout_image=_load_json_object(args.rollout_image_receipt, "rollout image receipt"),
            policy_image=_load_json_object(args.policy_image_receipt, "policy image receipt"),
            cuda_receipt=_load_json_object(args.cuda_receipt, "CUDA receipt"),
            readiness_receipt=_load_json_object(args.policy_server_readiness_receipt, "policy server readiness receipt"),
            n17_checkpoint=args.n17_checkpoint,
        )
        competitor_runtime_after = validate_competitor_runtime_evidence(args.competitor_runtime_evidence_root)
        policy_log_after = {
            "scope": "startup_through_authenticated_readiness",
            "path": str(args.policy_server_startup_log),
            "sha256": _sha256_file(args.policy_server_startup_log),
            "size": args.policy_server_startup_log.stat().st_size,
        }
        if (
            source_after != source_before
            or assets_after != assets_before
            or runtime_after != runtime_before
            or adapters_after != adapters_before
            or n17_after != n17_before
            or competitor_after != competitor_before
            or metadata_after != metadata_before
            or runtime_evidence_after != runtime_evidence_before
            or competitor_runtime_after != competitor_runtime_before
            or policy_log_after != policy_server_log
        ):
            raise ComparisonError("infrastructure_invalid: immutable source, asset, policy, or metadata identity changed")
        receipt = {
            "schema_version": 1,
            "kind": "lehome_official_policy_comparison_v1",
            "status": "valid",
            "mode": args.mode,
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            **current_identity,
            "policy_server_startup_log": {
                **policy_server_log,
                "path": evidence_archive["policy-server-startup.log"]["path"],
            },
            "evidence_archive": evidence_archive,
            "smoke_prerequisite": smoke_prerequisite,
            "matrix_sha256": matrix_sha,
            "matrix": matrix_payload,
            "commands": commands,
            "command_parity": parity,
            "results": results,
            "publication": "not_performed; use the explicit publish command after reviewing this valid receipt",
        }
        receipt_path = output_root / "comparison-receipt.json"
        _write_immutable(receipt_path, receipt)
        seal_execution_bundle(output_root)
        return receipt_path
    except Exception as error:
        _write_immutable(
            status_path,
            {
                "status": "infrastructure_invalid",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def publish_comparison(args: argparse.Namespace) -> Path:
    sealed = validate_sealed_execution(args.receipt)
    receipt_path = args.receipt.resolve(strict=True)
    receipt = sealed["comparison"]
    if (
        receipt.get("kind") != "lehome_official_policy_comparison_v1"
        or receipt.get("status") != "valid"
        or receipt.get("mode") != "full"
    ):
        raise ComparisonError("only a valid full official comparison receipt may be published")
    bundle_root = receipt_path.parent
    entries = _publication_entries(bundle_root, sealed)
    remote_prefix = deterministic_remote_prefix(sealed["receipt_sha256"], sealed["manifest_sha256"])
    if args.remote_prefix is not None and args.remote_prefix != remote_prefix:
        raise ComparisonError("remote prefix must be the deterministic execution identity prefix")
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    except ImportError as error:
        raise ComparisonError("huggingface_hub is required for explicit publication") from error
    token = os.environ.get(args.token_env, "")
    if not token:
        raise ComparisonError("Hugging Face publication token is unavailable")
    api = HfApi(token=token)
    anonymous_api = HfApi(token=False)
    existing_files = api.list_repo_files(
        repo_id=args.repository,
        repo_type="dataset",
        revision=args.revision,
    )
    prefix_marker = remote_prefix + "/"
    recovered_existing = any(path == remote_prefix or path.startswith(prefix_marker) for path in existing_files)
    if recovered_existing:
        _anonymous_verify_publication(
            api=anonymous_api,
            downloader=hf_hub_download,
            repository=args.repository,
            revision=args.revision,
            remote_prefix=remote_prefix,
            entries=entries,
        )
        immutable_revision = _immutable_repo_revision(
            anonymous_api, repository=args.repository, revision=args.revision
        )
    else:
        operations = [
            CommitOperationAdd(
                path_in_repo=f"{remote_prefix}/{entry['path']}",
                path_or_fileobj=str(bundle_root / str(entry["path"])),
            )
            for entry in entries
        ]
        parent_revision = _immutable_repo_revision(api, repository=args.repository, revision=args.revision)
        try:
            commit = api.create_commit(
                repo_id=args.repository,
                repo_type="dataset",
                revision=args.revision,
                parent_commit=parent_revision,
                operations=operations,
                commit_message="Publish official LeHome policy comparison",
            )
        except Exception:
            _anonymous_verify_publication(
                api=anonymous_api,
                downloader=hf_hub_download,
                repository=args.repository,
                revision=args.revision,
                remote_prefix=remote_prefix,
                entries=entries,
            )
            immutable_revision = _immutable_repo_revision(
                anonymous_api, repository=args.repository, revision=args.revision
            )
            recovered_existing = True
        else:
            immutable_revision = getattr(commit, "oid", None)
            if type(immutable_revision) is not str or _HEX40.fullmatch(immutable_revision) is None:
                raise ComparisonError("publication did not return an immutable revision")
    _anonymous_verify_publication(
        api=anonymous_api,
        downloader=hf_hub_download,
        repository=args.repository,
        revision=immutable_revision,
        remote_prefix=remote_prefix,
        entries=entries,
    )
    publication = {
        "schema_version": 1,
        "kind": "lehome_official_policy_comparison_publication_v1",
        "comparison_receipt_sha256": _sha256_file(receipt_path),
        "execution_manifest_sha256": sealed["manifest_sha256"],
        "repository": args.repository,
        "remote_prefix": remote_prefix,
        "immutable_revision": immutable_revision,
        "remote_file_count": len(entries),
        "recovered_existing_prefix": recovered_existing,
        "anonymous_file_set_verified": True,
        "anonymous_byte_readback_verified": True,
    }
    output = args.publication_receipt
    _write_immutable(output, publication)
    return output


def verify_n15_focused_promotion(args: argparse.Namespace) -> Path:
    receipt = _load_json_object(args.receipt, "focused comparison receipt")
    publication = _load_json_object(args.publication_receipt, "focused publication receipt")
    decision = assess_n15_focused_promotion(
        receipt,
        publication=publication,
        receipt_sha256=_sha256_file(args.receipt),
    )
    _write_immutable(args.promotion_receipt, decision)
    return args.promotion_receipt


def prepare_n15_candidate_compatibility(args: argparse.Namespace) -> Path:
    """Produce the candidate-only inference config view from its training receipt."""
    from rollout_appliance.native_reference_site.checkpoint_compatibility import (
        prepare_candidate_checkpoint_config_view,
    )

    prepare_candidate_checkpoint_config_view(
        args.candidate_checkpoint,
        args.training_identity_receipt,
        args.sanitized_config_root,
        args.compatibility_receipt,
    )
    return args.compatibility_receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=("smoke", "full"), required=True)
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--canonical-assets-root", type=Path, required=True)
    run.add_argument("--evaluation-assets-root", type=Path, required=True)
    run.add_argument("--metadata-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--competitor-checkpoint", type=Path, required=True)
    run.add_argument("--n17-checkpoint", type=Path, required=True)
    run.add_argument("--n17-identity-receipt", type=Path, required=True)
    run.add_argument("--reference-matrix", type=Path, required=True)
    run.add_argument("--reference-matrix-sha256", type=Path, required=True)
    run.add_argument("--sanitized-config-root", type=Path, required=True)
    run.add_argument("--compatibility-receipt", type=Path, required=True)
    run.add_argument("--native-site-root", type=Path, required=True)
    run.add_argument("--runtime-root", type=Path, required=True)
    run.add_argument("--runtime-revision", required=True)
    run.add_argument("--runtime-identity-receipt", type=Path, required=True)
    run.add_argument("--rollout-image-receipt", type=Path, required=True)
    run.add_argument("--policy-image-receipt", type=Path, required=True)
    run.add_argument("--cuda-receipt", type=Path, required=True)
    run.add_argument("--policy-server-readiness-receipt", type=Path, required=True)
    run.add_argument("--policy-server-startup-log", type=Path, required=True)
    run.add_argument("--competitor-runtime-evidence-root", type=Path, required=True)
    run.add_argument("--smoke-receipt", type=Path)
    run.add_argument("--isaaclab-root", type=Path, required=True)
    run.add_argument("--isaaclab-tasks-root", type=Path, required=True)
    run.add_argument("--docker-url", default="http://127.0.0.1:8080")
    run.add_argument("--python-bin", default=sys.executable)
    focused = commands.add_parser("run-n15-focused")
    focused.add_argument("--profile", choices=(N15_FOCUSED_PROFILE,), required=True)
    focused.add_argument("--source-root", type=Path, required=True)
    focused.add_argument("--canonical-assets-root", type=Path, required=True)
    focused.add_argument("--metadata-root", type=Path, required=True)
    focused.add_argument("--output-root", type=Path, required=True)
    focused.add_argument("--candidate-checkpoint", type=Path, required=True)
    focused.add_argument("--candidate-identity-receipt", type=Path, required=True)
    focused.add_argument("--candidate-sanitized-config-root", type=Path, required=True)
    focused.add_argument("--candidate-compatibility-receipt", type=Path, required=True)
    focused.add_argument("--reference-checkpoint", type=Path, required=True)
    focused.add_argument("--reference-sanitized-config-root", type=Path, required=True)
    focused.add_argument("--reference-compatibility-receipt", type=Path, required=True)
    focused.add_argument("--reference-matrix", type=Path, required=True)
    focused.add_argument("--reference-matrix-sha256", type=Path, required=True)
    focused.add_argument("--native-site-root", type=Path, required=True)
    focused.add_argument("--native-runtime-evidence-root", type=Path, required=True)
    focused.add_argument("--runtime-root", type=Path, required=True)
    focused.add_argument("--runtime-revision", required=True)
    focused.add_argument("--runtime-identity-receipt", type=Path, required=True)
    focused.add_argument("--rollout-image-receipt", type=Path, required=True)
    focused.add_argument("--cuda-receipt", type=Path, required=True)
    focused.add_argument("--isaaclab-root", type=Path, required=True)
    focused.add_argument("--isaaclab-tasks-root", type=Path, required=True)
    focused.add_argument("--python-bin", default=sys.executable)
    prepare = commands.add_parser("prepare-n15-candidate-compatibility")
    prepare.add_argument("--candidate-checkpoint", type=Path, required=True)
    prepare.add_argument("--training-identity-receipt", type=Path, required=True)
    prepare.add_argument("--sanitized-config-root", type=Path, required=True)
    prepare.add_argument("--compatibility-receipt", type=Path, required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--receipt", type=Path, required=True)
    publish.add_argument("--repository", required=True)
    publish.add_argument("--remote-prefix")
    publish.add_argument("--revision", default="main")
    publish.add_argument("--token-env", default="HF_TOKEN")
    publish.add_argument("--publication-receipt", type=Path, required=True)
    verify = commands.add_parser("verify-n15-focused")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--publication-receipt", type=Path, required=True)
    verify.add_argument("--promotion-receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "run":
            output = execute_comparison(args)
        elif args.command == "run-n15-focused":
            output = execute_n15_focused_comparison(args)
        elif args.command == "prepare-n15-candidate-compatibility":
            output = prepare_n15_candidate_compatibility(args)
        elif args.command == "publish":
            output = publish_comparison(args)
        else:
            output = verify_n15_focused_promotion(args)
    except (ComparisonError, OSError, subprocess.SubprocessError) as error:
        if args.command in {"run", "run-n15-focused"} and getattr(args, "_official_output_created", False):
            status_path = args.output_root / "status.json"
            if not status_path.exists():
                _write_immutable(
                    status_path,
                    {
                        "status": "infrastructure_invalid",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
