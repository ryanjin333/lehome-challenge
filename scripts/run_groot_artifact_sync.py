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
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "source" / "lehome"))
sys.path.insert(0, str(REPO_ROOT / "trainer" / "src"))

from lehome.flywheel.artifact_queue import ArtifactFinalizationQueue, QueueFullError  # noqa: E402
from lehome.flywheel.hub_sync import HubSyncDaemon, HubSyncError  # noqa: E402
from lehome.flywheel.task_ledger import TaskLedger  # noqa: E402
from lehome_train.hub import HuggingFaceHubTransport  # noqa: E402


def _load_attempt_matrix(path: Path) -> tuple[dict[str, object], ...]:
    matrix = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(matrix, list) or not all(isinstance(item, dict) for item in matrix):
        raise ValueError("attempt matrix must be a JSON array of objects")
    return tuple(matrix)


def run_finalizer_once(
    *,
    database: Path,
    attempt_matrix: Path,
    run_root: Path,
    max_pending_items: int,
    max_pending_bytes: int,
) -> int:
    """Finalize every raw terminal episode currently on disk; return count."""
    ledger = TaskLedger(
        database, attempt_matrix=_load_attempt_matrix(attempt_matrix),
        max_attempts=400, target_accepted=150,
    )
    queue = ArtifactFinalizationQueue(
        run_root=run_root, ledger=ledger,
        max_pending_items=max_pending_items, max_pending_bytes=max_pending_bytes,
    )
    attempts_root = run_root / "attempts"
    finalized = 0
    try:
        if attempts_root.is_dir():
            for child in sorted(attempts_root.iterdir()):
                if not child.is_dir() or child.is_symlink():
                    continue
                receipt_path = child / "worker-receipt.json"
                if not receipt_path.is_file():
                    continue
                if ledger.status(child.name) != "terminal_pending_validation":
                    continue
                try:
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    queue.enqueue(str(receipt["worker_id"]), child.name, str(receipt["lease_id"]), child)
                except (QueueFullError, KeyError, ValueError):
                    # Backpressure or an unreadable receipt: drain what is
                    # queued now and retry this episode on the next pass.
                    break
            while queue.pending_count:
                queue.finalize_next()
                finalized += 1
    finally:
        ledger.close()
    return finalized


def run_uploader_once(
    *,
    accepted_root: Path,
    receipts_root: Path,
    readback_root: Path,
    repository: str,
    round_id: str,
    revision: str,
) -> int:
    """Publish every accepted episode lacking a verified receipt; return count."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN must be injected at runtime; never bake it into images")
    daemon = HubSyncDaemon(
        repository=repository, round_id=round_id, token=token,
        transport=HuggingFaceHubTransport(), accepted_root=accepted_root,
        receipts_root=receipts_root, readback_root=readback_root, revision=revision,
    )
    synced = 0
    if accepted_root.is_dir():
        for child in sorted(accepted_root.iterdir()):
            if not child.is_dir() or child.is_symlink():
                continue
            receipt_path = receipts_root / f"{child.name}.sync.json"
            if receipt_path.exists():
                continue
            try:
                daemon.sync_episode(child.name, child)
                synced += 1
            except HubSyncError:
                continue  # retry on the next pass; receipts remain the source of truth
    return synced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("finalizer", "uploader"), required=True)
    parser.add_argument("--once", action="store_true", help="run a single pass and exit (used by tests)")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--attempt-matrix", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--max-pending-items", type=int, default=16)
    parser.add_argument("--max-pending-bytes", type=int, default=16 * 2**30)
    parser.add_argument("--accepted-root", type=Path)
    parser.add_argument("--receipts-root", type=Path)
    parser.add_argument("--readback-root", type=Path)
    parser.add_argument("--repository")
    parser.add_argument("--round-id")
    parser.add_argument("--revision")
    args = parser.parse_args(argv)

    while True:
        if args.role == "finalizer":
            if not (args.database and args.attempt_matrix and args.run_root):
                parser.error("finalizer requires --database, --attempt-matrix, --run-root")
            run_finalizer_once(
                database=args.database, attempt_matrix=args.attempt_matrix, run_root=args.run_root,
                max_pending_items=args.max_pending_items, max_pending_bytes=args.max_pending_bytes,
            )
        else:
            if not (args.accepted_root and args.receipts_root and args.readback_root
                    and args.repository and args.round_id and args.revision):
                parser.error("uploader requires accepted/receipts/readback roots and repository/round/revision")
            run_uploader_once(
                accepted_root=args.accepted_root, receipts_root=args.receipts_root,
                readback_root=args.readback_root, repository=args.repository,
                round_id=args.round_id, revision=args.revision,
            )
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
