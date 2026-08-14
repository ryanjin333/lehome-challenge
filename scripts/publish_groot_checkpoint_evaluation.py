"""Publish one private, readback-verified top-40 checkpoint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import math
from typing import Mapping, Sequence

from lehome.flywheel.artifacts import atomic_write_json, build_sha256_manifest, verify_episode_manifest
from lehome.flywheel.matrix import build_public_matrix, matrix_sha256


_REPOSITORY = "ryanjin333/lehome-groot-n17-data"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DISPOSAL_RECEIPT = "checkpoint-evaluation-disposal.json"
_MANIFEST = "SHA256SUMS.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _canonical_trial_ids() -> list[str]:
    return [
        trial.trial_id for trial in build_public_matrix().trials
        if trial.release_stage == "public_unseen" and trial.category in {"top_long", "top_short"}
    ]


def _regular_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _snapshot_entries(root: Path) -> dict[str, dict[str, object]]:
    manifest = _regular_json(Path(root) / _MANIFEST, "checkpoint evaluation snapshot manifest")
    for relative, entry in manifest.items():
        path = Path(root) / Path(*PurePosixPath(relative).parts)
        if not isinstance(entry, dict) or path.is_symlink() or not path.is_file() or entry.get("sha256") != _sha256(path) or entry.get("size") != path.stat().st_size:
            raise ValueError("checkpoint evaluation snapshot manifest does not match staged files")
    manifest_path = Path(root) / _MANIFEST
    manifest[_MANIFEST] = {"sha256": _sha256(manifest_path), "size": manifest_path.stat().st_size}
    return manifest


def _validated_report(report_path: Path) -> tuple[dict[str, object], list[str], dict[str, object]]:
    report = _regular_json(report_path, "checkpoint evaluation report")
    selection = report.get("selection")
    expected_ids = _canonical_trial_ids()
    if (
        not isinstance(selection, dict)
        or selection.get("kind") != "public_unseen_tops_evaluation"
        or selection.get("rft_data_eligible") is not False
    ):
        raise ValueError("checkpoint evaluation report must be top-40 diagnostic data and must never be RFT eligible")
    if selection.get("trial_count") != 40 or selection.get("trial_ids") != expected_ids or selection.get("category_counts") != {"top_long": 20, "top_short": 20}:
        raise ValueError("checkpoint evaluation report does not bind the exact canonical public-unseen top 40")
    evaluation = report.get("checkpoint_evaluation")
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("invocation"), dict):
        raise ValueError("checkpoint evaluation report lacks immutable final invocation evidence")
    invocation = evaluation["invocation"]
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("episodes") != 40 or not isinstance(metrics.get("per_category"), dict):
        raise ValueError("checkpoint evaluation report lacks immutable top-40 metrics")
    if invocation.get("matrix_sha256") != matrix_sha256(build_public_matrix()) or invocation.get("selected_trial_ids") != expected_ids:
        raise ValueError("checkpoint evaluation invocation does not bind the canonical top-40 matrix selection")
    return report, expected_ids, evaluation


def build_publication_plan(report_path: Path, staging_root: Path) -> dict[str, object]:
    """Verify and materialize the closed diagnostic snapshot without network access."""
    report_path = Path(report_path)
    report, trial_ids, evaluation = _validated_report(report_path)
    run_root = report_path.parent
    manifests = evaluation.get("episode_manifests")
    receipts = evaluation.get("policy_server_receipts")
    if not isinstance(manifests, list) or not isinstance(receipts, list) or len(manifests) != 40 or len(receipts) != 40:
        raise ValueError("checkpoint evaluation report requires exactly 40 episode-manifest and receipt bindings")
    by_manifest = {item.get("trial_id"): item for item in manifests if isinstance(item, dict)}
    by_receipt = {item.get("trial_id"): item for item in receipts if isinstance(item, dict)}
    if set(by_manifest) != set(trial_ids) or set(by_receipt) != set(trial_ids):
        raise ValueError("checkpoint evaluation final bindings must cover every canonical top-40 trial exactly once")
    root = Path(staging_root)
    if root.exists() or root.is_symlink():
        raise ValueError("checkpoint evaluation staging root must not already exist")
    root.mkdir(parents=True)
    category_metrics = {category: {"episodes": 0, "official_successes": 0, "visible_contact_count": 0} for category in ("top_long", "top_short")}
    try:
        for trial_id in trial_ids:
            episode = run_root / "raw" / trial_id
            manifest = episode / "SHA256SUMS.json"
            receipt = run_root / f"policy-server-receipt-{trial_id}.json"
            manifest_binding, receipt_binding = by_manifest[trial_id], by_receipt[trial_id]
            if (
                manifest_binding.get("path") != f"raw/{trial_id}/SHA256SUMS.json"
                or receipt_binding.get("path") != receipt.name
                or manifest_binding.get("sha256") != _sha256(manifest)
                or receipt_binding.get("sha256") != _sha256(receipt)
            ):
                raise ValueError("checkpoint evaluation report binding hash does not match local artifact")
            metadata, episode_manifest = verify_episode_manifest(episode)
            identity, provenance = metadata.get("identity"), metadata.get("provenance")
            required_files = {
                "annotations.jsonl", "episode.json", "snapshots/reset.json", "snapshots/terminal.json",
                "videos/left_rgb.mp4", "videos/right_rgb.mp4", "videos/top_rgb.mp4",
            }
            if not required_files.issubset(episode_manifest):
                raise ValueError("checkpoint evaluation episode lacks required allowlisted raw files")
            category = identity.get("category") if isinstance(identity, dict) else None
            if category not in category_metrics:
                raise ValueError("checkpoint evaluation episode category is outside the canonical top-40")
            category_metrics[category]["episodes"] += 1
            if metadata.get("accepted_success") is True and metadata.get("outcome") == "success":
                category_metrics[category]["official_successes"] += 1
            contact = metadata.get("visible_contact")
            if isinstance(contact, dict) and contact.get("observed") is True and contact.get("source") == "simulator_particle_to_gripper_distance" and isinstance(contact.get("minimum_distance_m"), (int, float)) and math.isfinite(float(contact["minimum_distance_m"])) and float(contact["minimum_distance_m"]) >= 0:
                category_metrics[category]["visible_contact_count"] += 1
            invocation = evaluation["invocation"]
            if (
                not isinstance(identity, dict) or not isinstance(provenance, dict)
                or identity.get("episode_id") != trial_id
                or any(identity.get(key) != invocation.get(key) for key in ("policy_repo", "policy_revision", "policy_step", "code_revision", "asset_revision", "simulator_version", "strategy"))
                or provenance.get("policy_artifact_sha256") != invocation.get("policy_artifact_sha256")
                or provenance.get("image_identity") != invocation.get("image_identity")
                or provenance.get("simulator_device") != invocation.get("simulator_device")
            ):
                raise ValueError("checkpoint evaluation artifact provenance does not match declared invocation")
            receipt_payload = _regular_json(receipt, "checkpoint evaluation policy-server receipt")
            if (
                receipt_payload.get("episode_id") != trial_id
                or receipt_payload.get("checkpoint_revision") != invocation.get("policy_revision")
                or receipt_payload.get("checkpoint_digest") != invocation.get("policy_artifact_sha256")
                or receipt_payload.get("code_revision") != invocation.get("code_revision")
                or receipt_payload.get("image_identity") != invocation.get("image_identity")
                or receipt_payload.get("groot_revision") != invocation.get("groot_revision")
                or receipt_payload.get("python_version") != invocation.get("groot_python_version")
            ):
                raise ValueError("checkpoint evaluation policy-server receipt does not match declared invocation")
            shutil.copytree(episode, root / "raw" / trial_id, copy_function=shutil.copy2)
            (root / "receipts").mkdir(exist_ok=True)
            shutil.copy2(receipt, root / "receipts" / receipt.name)
        reports = root / "reports"
        reports.mkdir()
        shutil.copy2(report_path, reports / "capacity-report.json")
        per_category = {category: {**values, "success_rate": values["official_successes"] / values["episodes"]} for category, values in category_metrics.items()}
        derived_metrics = {"episodes": 40, "official_successes": sum(item["official_successes"] for item in category_metrics.values()), "visible_contact_count": sum(item["visible_contact_count"] for item in category_metrics.values()), "per_category": per_category}
        derived_metrics["success_rate"] = derived_metrics["official_successes"] / 40
        if evaluation["metrics"] != derived_metrics:
            raise ValueError("checkpoint evaluation report metrics do not match verified episode artifacts")
        identity = {"invocation": evaluation["invocation"], "trial_ids": trial_ids, "kind": "diagnostic_evaluation_not_rft"}
        release_id = _canonical(identity)
        manifest = {"schema_version": 1, "release_id": release_id, "remote_prefix": f"evaluations/groot-n17-step-{identity['invocation']['policy_step']}/{release_id}", "identity": identity, "metrics": derived_metrics}
        atomic_write_json(root / "evaluation-manifest.json", manifest)
        atomic_write_json(root / "SHA256SUMS.json", build_sha256_manifest(root))
        entries = _snapshot_entries(root)
        return {"root": str(root), "release_id": release_id, "remote_prefix": manifest["remote_prefix"], "entries": entries, "report": report, "metrics": derived_metrics, "invocation": evaluation["invocation"]}
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


def _invalidate_disposal_receipt(run_root: Path) -> Path:
    receipt = Path(run_root) / _DISPOSAL_RECEIPT
    if receipt.is_symlink() or (receipt.exists() and not receipt.is_file()):
        raise ValueError("checkpoint evaluation disposal receipt path is unsafe")
    if receipt.exists():
        receipt.unlink()
    return receipt


def _token_from_process(pid: int) -> str:
    if type(pid) is not int or pid <= 0:
        raise ValueError("token environment PID must be positive")
    try:
        values = dict(item.split(b"=", 1) for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if b"=" in item)
        token = values.get(b"HF_TOKEN", b"").decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        raise ValueError("launch-process HF token is unavailable") from None
    if not token or any(character.isspace() for character in token):
        raise ValueError("launch-process HF token is unavailable")
    return token


def _revision(value: object) -> str:
    candidate = getattr(value, "oid", None) or getattr(value, "sha", None)
    if isinstance(candidate, str) and _COMMIT.fullmatch(candidate):
        return candidate
    raise ValueError("Hub response did not resolve to an immutable commit")


def _publish_and_readback(plan: dict[str, object], *, repository: str, branch: str, token: str, readback_root: Path, api: object | None = None, snapshot_download_fn: object | None = None) -> str:
    if readback_root.exists() or readback_root.is_symlink():
        raise ValueError("checkpoint evaluation readback root must not already exist")
    if api is None or snapshot_download_fn is None:
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError:
            raise RuntimeError("huggingface_hub is unavailable") from None
        api = HfApi(token=token)
        snapshot_download_fn = snapshot_download
    try:
        info = api.repo_info(repo_id=repository, repo_type="dataset", token=token)
    except Exception:
        raise PermissionError("private dataset access check failed") from None
    if getattr(info, "private", None) is not True:
        raise PermissionError("approved dataset repository must remain private")
    entries = plan["entries"]
    if not isinstance(entries, dict):
        raise ValueError("checkpoint evaluation plan entries are invalid")
    try:
        commit = _revision(api.upload_folder(
            repo_id=repository, repo_type="dataset", revision=branch, folder_path=str(plan["root"]),
            path_in_repo=str(plan["remote_prefix"]), allow_patterns=sorted(entries),
            commit_message=f"Publish verified GR00T checkpoint evaluation {plan['release_id']}", token=token,
        ))
        tree = api.list_repo_tree(repo_id=repository, repo_type="dataset", revision=commit, recursive=True, expand=True, token=token)
    except Exception:
        raise RuntimeError("checkpoint evaluation Hub upload or immutable tree listing failed") from None
    prefix = str(plan["remote_prefix"]).rstrip("/") + "/"
    observed = {item.path.removeprefix(prefix): item.size for item in tree if isinstance(getattr(item, "path", None), str) and item.path.startswith(prefix) and type(getattr(item, "size", None)) is int}
    if observed != {path: entry["size"] for path, entry in entries.items() if isinstance(entry, dict)}:
        raise ValueError("immutable Hub tree does not match the closed checkpoint evaluation snapshot")
    try:
        snapshot_download_fn(repo_id=repository, repo_type="dataset", revision=commit, allow_patterns=[f"{plan['remote_prefix']}/**"], local_dir=readback_root, token=token, max_workers=16)
    except Exception:
        raise RuntimeError("fresh immutable checkpoint evaluation readback failed") from None
    readback = readback_root / Path(*str(plan["remote_prefix"]).split("/"))
    if _snapshot_entries(readback) != entries:
        raise ValueError("fresh checkpoint evaluation readback hashes do not match the staged snapshot")
    return commit


def build_publication_receipt(plan: Mapping[str, object], *, immutable_revision: str, instance_receipt_path: Path) -> dict[str, object]:
    """Bind publication disposal authority to one exact launched evaluation.

    Upload and fresh readback occur before this is called.  The receipt remains
    useless for disposal unless the immutable Hub revision, closed prefix, local
    instance receipt bytes, and exact invocation all agree.
    """
    if not isinstance(immutable_revision, str) or _COMMIT.fullmatch(immutable_revision) is None:
        raise ValueError("immutable readback revision is invalid")
    invocation = plan.get("invocation")
    remote_prefix = plan.get("remote_prefix")
    release_id = plan.get("release_id")
    metrics = plan.get("metrics")
    if (not isinstance(invocation, dict) or not isinstance(remote_prefix, str) or not remote_prefix
            or not isinstance(release_id, str) or not re.fullmatch(r"[0-9a-f]{64}", release_id)
            or not isinstance(metrics, dict)):
        raise ValueError("publication readback plan lacks an immutable invocation or prefix")
    instance = _regular_json(instance_receipt_path, "evaluation instance receipt")
    instance_id = instance.get("instance_id")
    if (instance.get("kind") != "groot_checkpoint_evaluation_instance" or type(instance_id) is not int or instance_id <= 0
            or instance.get("invocation_sha256") != _canonical(invocation)):
        raise ValueError("publication instance receipt does not match the exact invocation")
    return {
        "schema_version": 1, "kind": "groot_checkpoint_evaluation_publication",
        "repository": _REPOSITORY, "repository_private": True,
        "immutable_revision": immutable_revision, "remote_prefix": remote_prefix, "release_id": release_id,
        "evaluation_metrics": metrics, "invocation": invocation, "invocation_sha256": _canonical(invocation),
        "instance_id": instance_id, "instance_receipt_sha256": _sha256(instance_receipt_path),
        "tree_listing_verified": True, "fresh_readback_verified": True, "disposable": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--readback-root", type=Path)
    parser.add_argument("--repository", default=_REPOSITORY)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--token-environ-pid", type=int)
    parser.add_argument("--instance-receipt", type=Path)
    parser.add_argument("--dry-run-plan", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repository != _REPOSITORY:
        raise ValueError("checkpoint evaluation publishing is restricted to the approved private dataset")
    receipt_path = _invalidate_disposal_receipt(args.report.parent)
    plan = build_publication_plan(args.report, args.staging_root)
    if args.dry_run_plan:
        print(json.dumps({key: value for key, value in plan.items() if key != "report"}, sort_keys=True))
        return 0
    if args.readback_root is None or args.token_environ_pid is None or args.instance_receipt is None:
        raise ValueError("publication requires --readback-root, --token-environ-pid, and --instance-receipt")
    commit = _publish_and_readback(plan, repository=args.repository, branch=args.branch, token=_token_from_process(args.token_environ_pid), readback_root=args.readback_root)
    receipt = build_publication_receipt(plan, immutable_revision=commit, instance_receipt_path=args.instance_receipt)
    atomic_write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
