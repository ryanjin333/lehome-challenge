"""Publish one exact public-280 campaign and verify a fresh immutable readback."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Mapping

from lehome.flywheel.artifacts import atomic_write_json
from lehome.flywheel.matrix import (
    canonical_matrix_json,
    load_public_matrix,
    matrix_sha256,
)
from lehome.flywheel.release import (
    ReleaseEntry,
    build_release_plan,
    materialize_release,
    validate_remote_file_tree,
    verify_release_tree,
)


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DATASET_REPOSITORY = "ryanjin333/lehome-groot-n17-data"
_MODEL_REPOSITORY = "ryanjin333/lehome-groot-n17-models"
_CANONICAL_MATRIX_SHA256 = "a3b15b5e4df2c68be6f3ea06eae4d8c2418714c45cf6af2a7f42e982225464b9"
_POLICY_STEP = 12000
_SIMULATOR_VERSION = "5.1.0.0"


def _invalidate_disposal_receipt(run_root: Path) -> Path:
    """Remove only the prior generated authorization marker before an attempt."""
    root = Path(run_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("campaign run root must be a materialized directory")
    receipt = root / "remote-verification.json"
    if receipt.is_symlink() or (receipt.exists() and not receipt.is_file()):
        raise ValueError("prior disposal receipt path is unsafe")
    if receipt.exists():
        receipt.unlink()
    return receipt


def _token_from_process(pid: int) -> str:
    if type(pid) is not int or pid <= 0:
        raise ValueError("token environment PID must be positive")
    try:
        values = dict(
            item.split(b"=", 1)
            for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            if b"=" in item
        )
        token = values.get(b"HF_TOKEN", b"").decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        raise ValueError("launch-process HF token is unavailable") from None
    if not token or any(character.isspace() for character in token):
        raise ValueError("launch-process HF token is unavailable")
    return token


def _revision(value: object) -> str:
    revision = getattr(value, "oid", None) or getattr(value, "sha", None)
    if isinstance(revision, str) and _COMMIT.fullmatch(revision):
        return revision
    url = getattr(value, "commit_url", None)
    if isinstance(url, str):
        candidate = url.rstrip("/").rsplit("/", 1)[-1]
        if _COMMIT.fullmatch(candidate):
            return candidate
    raise ValueError("Hub response did not resolve to an immutable commit")


def _remote_files(api: object, *, repository: str, revision: str, token: str) -> dict[str, int]:
    try:
        tree = getattr(api, "list_repo_tree")(
            repo_id=repository,
            repo_type="dataset",
            revision=revision,
            recursive=True,
            expand=True,
            token=token,
        )
        observed: dict[str, int] = {}
        for item in tree:
            path, size = getattr(item, "path", None), getattr(item, "size", None)
            if isinstance(path, str) and type(size) is int:
                if path in observed:
                    raise ValueError("Hub tree returned a duplicate file")
                observed[path] = size
        return observed
    except ValueError:
        raise
    except Exception:
        raise RuntimeError("immutable Hub tree listing failed") from None


def _upload(
    api: object,
    *,
    release_root: Path,
    entries: tuple[ReleaseEntry, ...],
    repository: str,
    branch: str,
    remote_prefix: str,
    token: str,
) -> str:
    arguments: dict[str, object] = {
        "repo_id": repository,
        "repo_type": "dataset",
        "revision": branch,
        "folder_path": str(release_root),
        "path_in_repo": remote_prefix,
        "allow_patterns": [entry.relative_path for entry in entries],
        "commit_message": f"Publish verified GR00T rollout release {remote_prefix.rsplit('/', 1)[-1]}",
        "token": token,
    }
    try:
        return _revision(getattr(api, "upload_folder")(**arguments))
    except Exception:
        # The commit may be durable even when the final HTTP response is lost.
        try:
            return _revision(
                getattr(api, "repo_info")(
                    repo_id=repository,
                    repo_type="dataset",
                    revision=branch,
                    token=token,
                )
            )
        except Exception:
            raise RuntimeError("Hub upload failed") from None


def _verify_expected_public_counts(plan) -> None:
    if plan.episode_count != 280:
        raise ValueError("public rollout release must contain exactly 280 episodes")
    if plan.category_counts != {
        "pant_long": 70,
        "pant_short": 70,
        "top_long": 70,
        "top_short": 70,
    }:
        raise ValueError("public rollout category counts are invalid")
    if plan.release_stage_counts != {"public_unseen": 80, "seen": 200}:
        raise ValueError("public rollout release-stage counts are invalid")


def _verify_declared_provenance(args: argparse.Namespace, plan) -> None:
    verified = plan.provenance
    if (
        verified.policy_repo != _MODEL_REPOSITORY
        or verified.policy_step != _POLICY_STEP
        or verified.simulator_version != _SIMULATOR_VERSION
        or args.policy_revision != verified.policy_revision
        or args.code_revision != verified.code_revision
        or args.asset_revision != verified.asset_revision
        or args.policy_artifact_sha256 != verified.policy_artifact_sha256
        or args.image_identity != verified.image_identity
        or args.policy_step != verified.policy_step
        or args.simulator_version != verified.simulator_version
    ):
        raise ValueError("declared publisher provenance does not match all verified episodes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--readback-root", type=Path, required=True)
    parser.add_argument("--repository", default=_DATASET_REPOSITORY)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--token-environ-pid", type=int, required=True)
    parser.add_argument("--policy-revision", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--asset-revision", required=True)
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--policy-artifact-sha256", required=True)
    parser.add_argument("--policy-step", type=int, default=_POLICY_STEP)
    parser.add_argument("--simulator-version", default=_SIMULATOR_VERSION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt_path = _invalidate_disposal_receipt(args.run_root)
    if args.repository != _DATASET_REPOSITORY:
        raise ValueError("rollout publishing is restricted to the approved private dataset")
    if args.readback_root.exists() or args.readback_root.is_symlink():
        raise ValueError("readback destination must not already exist")

    matrix = load_public_matrix(args.matrix)
    if matrix_sha256(matrix) != _CANONICAL_MATRIX_SHA256:
        raise ValueError("publisher matrix does not match the pinned public-280 identity")
    plan = build_release_plan(args.run_root, matrix, expected_steps=600)
    _verify_expected_public_counts(plan)
    _verify_declared_provenance(args, plan)
    release = materialize_release(
        plan,
        args.staging_root,
        matrix_json=canonical_matrix_json(matrix),
        policy_revision=args.policy_revision,
        code_revision=args.code_revision,
        asset_revision=args.asset_revision,
        image_identity=args.image_identity,
        policy_step=args.policy_step,
    )
    token = _token_from_process(args.token_environ_pid)

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        raise RuntimeError("huggingface_hub is unavailable") from None
    api = HfApi(token=token)
    try:
        repository = api.repo_info(
            repo_id=args.repository,
            repo_type="dataset",
            token=token,
        )
    except Exception:
        raise PermissionError("private dataset access check failed") from None
    if getattr(repository, "private", None) is not True:
        raise PermissionError("approved dataset repository must remain private")

    commit = _upload(
        api,
        release_root=release.root,
        entries=release.entries,
        repository=args.repository,
        branch=args.branch,
        remote_prefix=release.remote_prefix,
        token=token,
    )
    observed = _remote_files(
        api,
        repository=args.repository,
        revision=commit,
        token=token,
    )
    validate_remote_file_tree(
        observed,
        remote_prefix=release.remote_prefix,
        expected=release.entries,
    )

    try:
        snapshot_download(
            repo_id=args.repository,
            repo_type="dataset",
            revision=commit,
            allow_patterns=[f"{release.remote_prefix}/**"],
            local_dir=args.readback_root,
            token=token,
            max_workers=16,
        )
    except Exception:
        raise RuntimeError("fresh immutable Hub readback failed") from None
    readback = args.readback_root / Path(*release.remote_prefix.split("/"))
    readback_entries = verify_release_tree(readback)
    if readback_entries != release.entries:
        raise ValueError("fresh Hub readback does not match the local release")

    receipt: dict[str, object] = {
        "schema_version": 1,
        "repository": args.repository,
        "repository_private": True,
        "immutable_revision": commit,
        "remote_prefix": release.remote_prefix,
        "release_id": release.release_id,
        "episode_count": plan.episode_count,
        "category_counts": plan.category_counts,
        "release_stage_counts": plan.release_stage_counts,
        "entry_count": release.entry_count,
        "byte_size": sum(entry.byte_size for entry in release.entries),
        "policy_repo": plan.provenance.policy_repo,
        "policy_revision": plan.provenance.policy_revision,
        "policy_step": plan.provenance.policy_step,
        "policy_artifact_sha256": plan.provenance.policy_artifact_sha256,
        "code_revision": plan.provenance.code_revision,
        "asset_revision": plan.provenance.asset_revision,
        "simulator_version": plan.provenance.simulator_version,
        "image_identity": plan.provenance.image_identity,
        "tree_listing_verified": True,
        "fresh_readback_verified": True,
        "disposable": True,
    }
    atomic_write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
