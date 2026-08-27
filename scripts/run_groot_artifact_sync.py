"""Background finalizer/uploader loops for the rollout appliance.

``--role finalizer`` validates raw terminal episodes and settles ledger
outcomes; ``--role uploader`` publishes accepted episodes to the private
Hub with readback-verified receipts.  Both are pure CPU/disk services and
never touch the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import time
from collections import Counter
from typing import Mapping
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "source" / "lehome"))
sys.path.insert(0, str(REPO_ROOT / "trainer" / "src"))

from lehome.flywheel.artifact_queue import ArtifactFinalizationQueue, QueueFullError  # noqa: E402
from lehome.flywheel.hub_sync import (  # noqa: E402
    REMOTE_ROOT,
    HubSyncDaemon,
    HubSyncError,
    SyncEntry,
    _collect_entries,
    _episode_digest,
    _sha256_file,
    _write_json_atomic,
)
from lehome.flywheel.task_ledger import TaskLedger  # noqa: E402
from lehome_train.flywheel.publish import RolloutRoundSeal, seal_rollout_round  # noqa: E402
from lehome_train.hub import HuggingFaceHubTransport  # noqa: E402


def _load_runtime_token(
    *, token_file: Path | None, environ: Mapping[str, str] = os.environ,
) -> str:
    """Load one Hub token without exposing it through container metadata."""
    if token_file is not None:
        path = Path(token_file)
        try:
            metadata = path.lstat()
        except OSError:
            raise RuntimeError("HF token file is unavailable") from None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & 0o077
            or metadata.st_uid != os.geteuid()
        ):
            raise RuntimeError("HF token file must be an owner-only private regular file")
        try:
            token = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            raise RuntimeError("HF token file is unreadable") from None
    else:
        token = environ.get("HF_TOKEN", "").strip()
    if not token or any(character.isspace() for character in token):
        raise RuntimeError("HF token must be injected at runtime; never bake it into images")
    return token


def _load_attempt_matrix(
    path: Path, *, controlled_recovery_smoke: bool = False,
) -> tuple[dict[str, object], ...]:
    from lehome.flywheel.recovery_collection import load_attempt_matrix

    matrix = load_attempt_matrix(path)
    controlled = [item for item in matrix if item.get("recovery_kind") == "controlled_success_recovery_snapshot_v3"]
    if controlled:
        if len(controlled) != len(matrix):
            raise ValueError("controlled recovery matrix must not mix legacy assignments")
        if controlled_recovery_smoke:
            if len(matrix) != 1:
                raise ValueError("controlled recovery smoke must contain exactly one row")
            row = matrix[0]
            if row.get("controlled_smoke") is not True:
                raise ValueError("controlled recovery smoke requires an explicit smoke descriptor")
            smoke_hashes = {
                row.get("controlled_smoke_matrix_sha256"),
                row.get("controlled_smoke_materialization_sha256"),
            }
            if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in smoke_hashes):
                raise ValueError("controlled recovery smoke lineage hashes are invalid")
            if row.get("category") not in {"pant_long", "top_long", "top_short"}:
                raise ValueError("controlled recovery smoke category is invalid")
            caps = {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}
            if row.get("category_acceptance_cap") != caps[row["category"]]:
                raise ValueError("controlled recovery smoke category cap is invalid")
            for field in (
                "source_episode_digest", "source_state_fingerprint",
                "source_reset_sha256", "source_annotations_sha256",
                "source_continuation_snapshot_sha256", "perturbation_fingerprint",
                "source_state_perturbation_fingerprint",
            ):
                if not isinstance(row.get(field), str) or re.fullmatch(r"[0-9a-f]{64}", row[field]) is None:
                    raise ValueError("controlled recovery smoke source lineage is invalid")
            if (not isinstance(row.get("controlled_smoke_run_id"), str)
                    or re.fullmatch(r"[0-9a-f]{32}", row["controlled_smoke_run_id"]) is None
                    or type(row.get("controlled_smoke_row_index")) is not int
                    or row["controlled_smoke_row_index"] < 0
                    or not isinstance(row.get("controlled_smoke_identity"), str)
                    or re.fullmatch(r"[0-9a-f]{20}", row["controlled_smoke_identity"]) is None):
                raise ValueError("controlled recovery smoke run identity is invalid")
            if (not isinstance(row.get("source_round_id"), str) or not row["source_round_id"]
                    or not isinstance(row.get("source_episode_id"), str) or not row["source_episode_id"]
                    or not isinstance(row.get("source_immutable_revision"), str)
                    or re.fullmatch(r"[0-9a-f]{40}", row["source_immutable_revision"]) is None):
                raise ValueError("controlled recovery smoke source lineage is invalid")
            # This repeats the worker's pre-mutation validation at finalizer
            # admission: source paths, digests, replay prefix and bounded
            # perturbation must all still authenticate before the only smoke
            # terminal can count as accepted.
            from lehome.flywheel.recovery_collection import load_controlled_recovery
            load_controlled_recovery(row)
            return tuple(matrix)
        if len(matrix) == 1 and matrix[0].get("controlled_smoke") is True:
            raise ValueError("controlled recovery smoke requires explicit smoke mode")
        caps = {"pant_long": 4, "top_long": 1, "top_short": 3}
        matrix_hashes = {item.get("controlled_matrix_sha256") for item in matrix}
        if not 8 <= len(matrix) <= 96 or len(matrix_hashes) != 1 or not isinstance(next(iter(matrix_hashes)), str) or re.fullmatch(r"[0-9a-f]{64}", next(iter(matrix_hashes))) is None:
            raise ValueError("controlled recovery materialization is incomplete")
        counts = Counter(item.get("category") for item in matrix)
        if any(counts.get(category, 0) < cap for category, cap in caps.items()):
            raise ValueError("controlled recovery matrix has an invalid bounded retry schedule")
        for item in matrix:
            if item.get("category_acceptance_cap") != caps.get(item.get("category")):
                raise ValueError("controlled recovery matrix has an invalid category acceptance cap")
    return tuple(matrix)


def run_finalizer_once(
    *,
    database: Path,
    attempt_matrix: Path,
    run_root: Path,
    max_pending_items: int,
    max_pending_bytes: int,
    max_attempts: int,
    target_accepted: int,
    completion_metric: str = "accepted_successes",
    controlled_recovery_smoke: bool = False,
    evaluation_terminal: bool = False,
) -> int:
    """Finalize every raw terminal episode currently on disk; return count."""
    if controlled_recovery_smoke and (max_attempts != 1 or target_accepted != 1):
        raise ValueError("controlled recovery smoke finalization requires exactly one attempt and acceptance")
    if controlled_recovery_smoke and evaluation_terminal:
        raise ValueError("controlled recovery smoke cannot use evaluation terminal publication")
    ledger = TaskLedger(
        database, attempt_matrix=_load_attempt_matrix(
            attempt_matrix, controlled_recovery_smoke=controlled_recovery_smoke,
        ),
        max_attempts=max_attempts, target_accepted=target_accepted, completion_metric=completion_metric,
    )
    queue = ArtifactFinalizationQueue(
        run_root=run_root, ledger=ledger,
        max_pending_items=max_pending_items, max_pending_bytes=max_pending_bytes,
        evaluation_only=evaluation_terminal,
        evaluation_terminal_root=(run_root / "evaluation-terminal") if evaluation_terminal else None,
    )
    finalized = 0
    try:
        # Persistent workers intentionally isolate artifacts under
        # worker/session/attempt/lease/generation. The append-only ledger is
        # the authoritative handoff and records that exact closed directory;
        # never assume the obsolete flat ``attempts/<id>`` layout.
        for attempt in ledger.attempts():
            if ledger.status(attempt.attempt_id) != "terminal_pending_validation":
                continue
            terminal_events = [
                event
                for event in ledger.events(attempt.attempt_id)
                if event.event_type == "terminal_pending_validation"
            ]
            if not terminal_events:
                raise RuntimeError("pending terminal attempt has no handoff event")
            event = terminal_events[-1]
            raw_artifact_id = event.payload.get("raw_artifact_id")
            if not isinstance(raw_artifact_id, str) or not raw_artifact_id:
                raise RuntimeError("pending terminal handoff has no raw artifact path")
            if not isinstance(event.worker_id, str) or not isinstance(event.lease_id, str):
                raise RuntimeError("pending terminal handoff has no worker/lease identity")
            try:
                queue.enqueue(
                    event.worker_id,
                    attempt.attempt_id,
                    event.lease_id,
                    Path(raw_artifact_id),
                )
            except QueueFullError:
                # Drain the bounded in-memory queue now; the next poll resumes
                # from the still-pending append-only ledger entry.
                break
            while queue.pending_count:
                queue.finalize_next()
                finalized += 1
    finally:
        ledger.close()
    return finalized


def _run_episode_uploader_once(
    *,
    episode_root: Path,
    receipts_root: Path,
    readback_root: Path,
    repository: str,
    round_id: str,
    revision: str,
    token_file: Path | None = None,
) -> int:
    """Publish one episode lacking a verified receipt from a bounded root."""
    token = _load_runtime_token(token_file=token_file)
    daemon = HubSyncDaemon(
        repository=repository, round_id=round_id, token=token,
        transport=HuggingFaceHubTransport(), accepted_root=episode_root,
        receipts_root=receipts_root, readback_root=readback_root, revision=revision,
    )
    synced = 0
    if episode_root.is_dir():
        for child in sorted(episode_root.iterdir()):
            if not child.is_dir() or child.is_symlink():
                continue
            receipt_path = receipts_root / f"{child.name}.sync.json"
            if receipt_path.exists():
                continue
            daemon.sync_episode(child.name, child)
            synced += 1
            # One episode per pass bounds Hub API pressure. The outer loop
            # supplies the deliberate inter-episode throttle.
            break
    return synced


def run_uploader_once(
    *,
    accepted_root: Path,
    receipts_root: Path,
    readback_root: Path,
    repository: str,
    round_id: str,
    revision: str,
    token_file: Path | None = None,
) -> int:
    """Publish one trainable accepted episode; return count."""
    return _run_episode_uploader_once(
        episode_root=accepted_root,
        receipts_root=receipts_root,
        readback_root=readback_root,
        repository=repository,
        round_id=round_id,
        revision=revision,
        token_file=token_file,
    )


def run_evaluation_uploader_once(
    *,
    terminal_root: Path,
    receipts_root: Path,
    readback_root: Path,
    repository: str,
    round_id: str,
    revision: str,
    token_file: Path | None = None,
) -> int:
    """Publish one valid evaluation terminal, success or failure.

    This source root is separate from ``accepted`` so a failed evaluation can
    have immutable read-back evidence without becoming trainable rollout data.
    """
    if (
        not terminal_root.is_absolute()
        or terminal_root.name != "evaluation-terminal"
        or terminal_root.is_symlink()
        or not terminal_root.is_dir()
    ):
        raise ValueError("evaluation terminal root is unsafe or misclassified")
    return _run_episode_uploader_once(
        episode_root=terminal_root,
        receipts_root=receipts_root,
        readback_root=readback_root,
        repository=repository,
        round_id=round_id,
        revision=revision,
        token_file=token_file,
    )


def run_evaluation_batch_uploader_once(
    *,
    terminal_root: Path,
    receipts_root: Path,
    readback_root: Path,
    repository: str,
    round_id: str,
    revision: str,
    token_file: Path | None = None,
    immutable_revision: str | None = None,
) -> int:
    """Publish a complete evaluation round with one commit and one readback.

    Evaluation campaigns contain many small files per episode. Publishing each
    episode through a separate Hub commit can exhaust the API request window
    even though the bytes are modest. This path preserves the normal
    per-episode receipt schema while binding every receipt to one immutable
    batch commit and verifying the complete round in one readback.
    """
    root = Path(terminal_root)
    if (
        not root.is_absolute()
        or root.name != "evaluation-terminal"
        or root.is_symlink()
        or not root.is_dir()
    ):
        raise ValueError("evaluation terminal root is unsafe or misclassified")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", round_id):
        raise ValueError("evaluation round ID is unsafe")
    if (
        not isinstance(revision, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", revision)
        or re.fullmatch(r"[0-9a-f]{40}", revision)
        or any(component in {"", ".", ".."} for component in revision.split("/"))
    ):
        raise ValueError("evaluation publication ref is unsafe")

    token = _load_runtime_token(token_file=token_file)
    episodes: list[tuple[str, Path, tuple[SyncEntry, ...], str]] = []
    flattened: list[SyncEntry] = []
    for child in sorted(root.iterdir()):
        if child.is_symlink() or not child.is_dir():
            continue
        entries = _collect_entries(child)
        episodes.append((child.name, child, entries, _episode_digest(entries)))
        flattened.extend(
            SyncEntry(
                relative_path=f"{child.name}/{entry.relative_path}",
                sha256=entry.sha256,
                byte_size=entry.byte_size,
            )
            for entry in entries
        )
    if not episodes:
        raise HubSyncError("evaluation terminal root has no episodes")

    remote_prefix = f"{REMOTE_ROOT}/{round_id}"
    transport = HuggingFaceHubTransport()
    if immutable_revision is None:
        immutable_revision = transport.upload_files(
            repository=repository,
            revision=revision,
            source=root,
            entries=tuple(flattened),
            token=token,
            remote_prefix=remote_prefix,
        )
    if not isinstance(immutable_revision, str) or not re.fullmatch(
        r"[0-9a-f]{40}", immutable_revision,
    ):
        raise HubSyncError("evaluation batch upload did not return an immutable commit")

    expected = {entry.relative_path for entry in flattened}
    observed: set[str] = set()
    for tree_attempt in range(4):
        tree = transport.list_tree(
            repository=repository,
            revision=immutable_revision,
            token=token,
            remote_prefix=remote_prefix,
        )
        observed = {
            entry.relative_path.removeprefix(remote_prefix + "/")
            for entry in tree
            if entry.entry_type == "file"
            and entry.relative_path.startswith(remote_prefix + "/")
        }
        if observed == expected:
            break
        if tree_attempt < 3:
            time.sleep(5)
    if observed != expected:
        raise HubSyncError(
            "evaluation batch remote tree does not match: "
            f"missing {sorted(expected - observed)}, extra {sorted(observed - expected)}"
        )

    readback = Path(readback_root)
    readback.mkdir(parents=True, exist_ok=True)
    destination = readback / f"batch-{uuid4().hex}"
    try:
        transport.download_files(
            repository=repository,
            revision=immutable_revision,
            destination=destination,
            relative_paths=tuple(
                f"{remote_prefix}/{entry.relative_path}" for entry in flattened
            ),
            token=token,
        )
        for entry in flattened:
            path = destination / remote_prefix / entry.relative_path
            if path.is_symlink() or not path.is_file():
                raise HubSyncError(
                    f"evaluation batch readback missing file: {entry.relative_path}"
                )
            sha256, byte_size = _sha256_file(path)
            if sha256 != entry.sha256 or byte_size != entry.byte_size:
                raise HubSyncError(
                    f"evaluation batch readback hash mismatch: {entry.relative_path}"
                )
    finally:
        shutil.rmtree(destination, ignore_errors=True)

    receipts = Path(receipts_root)
    receipts.mkdir(parents=True, exist_ok=True)
    created = 0
    for attempt_id, _path, entries, digest in episodes:
        receipt_path = receipts / f"{attempt_id}.sync.json"
        if receipt_path.is_symlink():
            raise HubSyncError("evaluation sync receipt path is unsafe")
        if receipt_path.is_file():
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                existing.get("episode_sha256") != digest
                or existing.get("round_id") != round_id
                or existing.get("readback_verified") is not True
            ):
                raise HubSyncError(
                    "existing evaluation receipt does not match the batch episode"
                )
            continue
        _write_json_atomic(receipt_path, {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "repository": repository,
            "round_id": round_id,
            "remote_prefix": f"{remote_prefix}/{attempt_id}",
            "publication_ref": revision,
            "immutable_revision": immutable_revision,
            "entry_count": len(entries),
            "episode_sha256": digest,
            "readback_verified": True,
        })
        created += 1
    return created


def run_sealer_once(
    *,
    database: Path,
    attempt_matrix: Path,
    receipts_root: Path,
    round_id: str,
    seal_receipt_path: Path,
    max_attempts: int,
    target_accepted: int,
    completion_metric: str = "accepted_successes",
) -> RolloutRoundSeal:
    """Seal the exact accepted ledger set after every Hub readback is durable."""

    matrix = _load_attempt_matrix(attempt_matrix)
    controlled = bool(matrix and matrix[0].get("recovery_kind") == "controlled_success_recovery_snapshot_v3")
    ledger = TaskLedger(
        database,
        attempt_matrix=matrix,
        max_attempts=max_attempts,
        target_accepted=target_accepted, completion_metric=completion_metric,
    )
    try:
        accepted_attempts = tuple(
            attempt.attempt_id
            for attempt in ledger.attempts()
            if ledger.status(attempt.attempt_id) == "accepted"
        )
        if controlled:
            caps = {"pant_long": 4, "top_long": 1, "top_short": 3}
            accepted_categories = Counter(
                attempt.assignment.get("category")
                for attempt in ledger.attempts()
                if ledger.status(attempt.attempt_id) == "accepted"
            )
            if target_accepted != 8 or len(accepted_attempts) != 8 or accepted_categories != Counter(caps):
                raise RuntimeError("controlled recovery campaign is short of its immutable acceptance caps")
    finally:
        ledger.close()
    return seal_rollout_round(
        receipts_root=receipts_root,
        round_id=round_id,
        attempt_ids=accepted_attempts,
        seal_receipt_path=seal_receipt_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=("finalizer", "uploader", "evaluation-uploader", "sealer"),
        required=True,
    )
    parser.add_argument("--once", action="store_true", help="run a single pass and exit (used by tests)")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--failure-backoff-seconds", type=float, default=300.0)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--attempt-matrix", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--max-pending-items", type=int, default=16)
    parser.add_argument("--max-pending-bytes", type=int, default=16 * 2**30)
    parser.add_argument("--max-attempts", type=int, default=400)
    parser.add_argument("--target-accepted", type=int, default=150)
    parser.add_argument("--completion-metric", choices=("accepted_successes", "terminal_outcomes"), default="accepted_successes")
    parser.add_argument("--accepted-root", type=Path)
    parser.add_argument("--terminal-root", type=Path)
    parser.add_argument("--receipts-root", type=Path)
    parser.add_argument("--readback-root", type=Path)
    parser.add_argument("--repository")
    parser.add_argument("--round-id")
    parser.add_argument("--revision")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument(
        "--batch", action="store_true",
        help="publish one complete evaluation round with one commit/readback",
    )
    parser.add_argument(
        "--batch-revision",
        help="resume batch verification from an existing immutable commit",
    )
    parser.add_argument("--seal-receipt", type=Path)
    parser.add_argument(
        "--controlled-recovery-smoke", action="store_true",
        help="allow the single immutable controlled-recovery smoke descriptor only",
    )
    parser.add_argument(
        "--evaluation-terminal", action="store_true",
        help="retain every valid terminal outcome outside the trainable accepted root",
    )
    args = parser.parse_args(argv)
    if args.batch and (args.role != "evaluation-uploader" or not args.once):
        parser.error("--batch requires --role evaluation-uploader --once")
    if args.batch_revision and not args.batch:
        parser.error("--batch-revision requires --batch")

    while True:
        if args.role == "finalizer":
            if not (args.database and args.attempt_matrix and args.run_root):
                parser.error("finalizer requires --database, --attempt-matrix, --run-root")
            run_finalizer_once(
                database=args.database, attempt_matrix=args.attempt_matrix, run_root=args.run_root,
                max_pending_items=args.max_pending_items, max_pending_bytes=args.max_pending_bytes,
                max_attempts=args.max_attempts, target_accepted=args.target_accepted,
                completion_metric=args.completion_metric,
                controlled_recovery_smoke=args.controlled_recovery_smoke,
                evaluation_terminal=args.evaluation_terminal,
            )
        elif args.role in {"uploader", "evaluation-uploader"}:
            episode_root = args.accepted_root if args.role == "uploader" else args.terminal_root
            if not (episode_root and args.receipts_root and args.readback_root
                    and args.repository and args.round_id and args.revision):
                parser.error("uploader requires its episode/receipts/readback roots and repository/round/revision")
            try:
                common = {
                    "receipts_root": args.receipts_root,
                    "readback_root": args.readback_root,
                    "repository": args.repository,
                    "round_id": args.round_id,
                    "revision": args.revision,
                    "token_file": args.token_file,
                }
                if args.role == "uploader":
                    run_uploader_once(accepted_root=episode_root, **common)
                elif args.batch:
                    run_evaluation_batch_uploader_once(
                        terminal_root=episode_root,
                        immutable_revision=args.batch_revision,
                        **common,
                    )
                else:
                    run_evaluation_uploader_once(terminal_root=episode_root, **common)
            except HubSyncError:
                if args.once:
                    raise
                time.sleep(args.failure_backoff_seconds)
                continue
        else:
            if not (
                args.database and args.attempt_matrix and args.receipts_root
                and args.round_id and args.seal_receipt
            ):
                parser.error(
                    "sealer requires --database, --attempt-matrix, --receipts-root, "
                    "--round-id, and --seal-receipt"
                )
            run_sealer_once(
                database=args.database,
                attempt_matrix=args.attempt_matrix,
                receipts_root=args.receipts_root,
                round_id=args.round_id,
                seal_receipt_path=args.seal_receipt,
                max_attempts=args.max_attempts,
                target_accepted=args.target_accepted,
                completion_metric=args.completion_metric,
            )
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
