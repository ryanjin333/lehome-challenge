"""SQLite-backed single writer for asynchronous experiment leases."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import uuid
from typing import Mapping, Sequence

from lehome_train.groot.experiment_job import ExperimentJob, _parse, experiment_identity
from lehome_train.groot.experiment_evaluation import parse_experiment_evaluation, to_evaluation_score
from lehome_train.groot.experiment_manifest import APPROVED_ORIGINAL_12K_CHECKPOINT
from lehome_train.groot.experiment_publication import bind_checkpoint_publication, parse_checkpoint_publication
from lehome_train.groot.experiment_promotion import EvaluationScore, rank_key, select_2k_finalists
from lehome_train.io import canonical_json_sha256


_PRODUCTION_BUDGET_LIMITS = {
    "gpu_seconds_ceiling": 604_800.0,
    "spend_ceiling": 10_000.0,
    "estimated_gpu_seconds_per_step": 60.0,
    "gpu_price_per_second": 1.0,
}
_PRODUCTION_TRAINING_LEASE_SECONDS = 60.0
# A training result is written durably by the worker before it asks the
# controller to transition the job.  The request itself can be lost exactly as
# its 60-second lease expires.  Keep that original attempt non-leaseable long
# enough to reconcile the receipt, rather than accidentally paying for a
# duplicate run.  This is deliberately controller-owned, not a worker retry
# timeout, so it survives service/process restarts.
TERMINAL_RECEIPT_GRACE_NS = 60_000_000_000


def validate_production_budget(value: object) -> dict[str, float]:
    """Return one finite, positive, conservatively bounded paid-run budget."""
    if not isinstance(value, Mapping) or set(value) != set(_PRODUCTION_BUDGET_LIMITS):
        raise ValueError("production budget must contain exact positive bounded fields")
    result: dict[str, float] = {}
    for key, limit in _PRODUCTION_BUDGET_LIMITS.items():
        raw = value[key]
        if type(raw) not in (int, float):
            raise ValueError("production budget values must be positive bounded numbers")
        number = float(raw)
        if not math.isfinite(number) or not 0.0 < number <= limit:
            raise ValueError("production budget values must be positive bounded numbers")
        result[key] = number
    return result


@dataclass(frozen=True, slots=True)
class JobLease:
    lease_id: str
    experiment_id: str
    worker_id: str
    capability: str
    expires_ns: int
    job: ExperimentJob | None = None
    publication: Mapping[str, object] | None = None
    parent_publication: Mapping[str, object] | None = None
    evaluation_matrix_sha256: str | None = None


class ExperimentController:
    """The only SQLite writer; workers can only lease or report transitions."""

    def __init__(self, database: str | Path, *, max_gpu_leases: int = 3, gradient_step_ceiling: int = 7000, tied_runner_gradient_step_ceiling: int = 8000, gpu_seconds_ceiling: float | None = None, spend_ceiling: float | None = None, estimated_gpu_seconds_per_step: float = 0.0, gpu_price_per_second: float = 0.0) -> None:
        if max_gpu_leases != 3:
            raise ValueError("topology is exactly two trainers plus one evaluator")
        if type(gradient_step_ceiling) is not int or type(tied_runner_gradient_step_ceiling) is not int or not 0 < gradient_step_ceiling <= tied_runner_gradient_step_ceiling:
            raise ValueError("gradient budget is invalid")
        if (gpu_seconds_ceiling is None) != (spend_ceiling is None):
            raise ValueError("GPU budget is invalid")
        if gpu_seconds_ceiling is not None:
            paid_budget = validate_production_budget({
                "gpu_seconds_ceiling": gpu_seconds_ceiling,
                "spend_ceiling": spend_ceiling,
                "estimated_gpu_seconds_per_step": estimated_gpu_seconds_per_step,
                "gpu_price_per_second": gpu_price_per_second,
            })
            gpu_seconds_ceiling = paid_budget["gpu_seconds_ceiling"]
            spend_ceiling = paid_budget["spend_ceiling"]
            estimated_gpu_seconds_per_step = paid_budget["estimated_gpu_seconds_per_step"]
            gpu_price_per_second = paid_budget["gpu_price_per_second"]
        elif any(type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0 for value in (estimated_gpu_seconds_per_step, gpu_price_per_second)):
            raise ValueError("GPU budget is invalid")
        self.path = Path(database)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.max_gpu_leases = max_gpu_leases
        self.gradient_step_ceiling = gradient_step_ceiling
        self.tied_runner_gradient_step_ceiling = tied_runner_gradient_step_ceiling
        self.gpu_seconds_ceiling = None if gpu_seconds_ceiling is None else float(gpu_seconds_ceiling)
        self.spend_ceiling = None if spend_ceiling is None else float(spend_ceiling)
        self.estimated_gpu_seconds_per_step = float(estimated_gpu_seconds_per_step)
        self.gpu_price_per_second = float(gpu_price_per_second)
        self.original_12k_checkpoint_digest = APPROVED_ORIGINAL_12K_CHECKPOINT["artifact_sha256"]
        self._jobs: dict[str, ExperimentJob] = {}
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (experiment_id TEXT PRIMARY KEY, canonical BLOB NOT NULL, state TEXT NOT NULL, capability TEXT NOT NULL, created_order INTEGER NOT NULL, receipt_sha256 TEXT);
            CREATE TABLE IF NOT EXISTS dependencies (experiment_id TEXT NOT NULL, dependency TEXT NOT NULL, PRIMARY KEY(experiment_id, dependency));
            CREATE TABLE IF NOT EXISTS leases (lease_id TEXT PRIMARY KEY, experiment_id TEXT UNIQUE NOT NULL, worker_id TEXT NOT NULL, capability TEXT NOT NULL, expires_ns INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL, job_digest TEXT NOT NULL DEFAULT '', state TEXT NOT NULL, worker_id TEXT, attempt INTEGER NOT NULL, timestamp_ns INTEGER NOT NULL, detail TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS artifacts (experiment_id TEXT PRIMARY KEY, receipt_sha256 TEXT NOT NULL, completion_lease_id TEXT, completion_worker_id TEXT, publication TEXT, verified INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS evaluations (experiment_id TEXT PRIMARY KEY, report TEXT NOT NULL, received_ns INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS final_evaluations (experiment_id TEXT PRIMARY KEY, matrix_sha256 TEXT NOT NULL, state TEXT NOT NULL, report TEXT, received_ns INTEGER);
            CREATE TABLE IF NOT EXISTS dependency_receipts (sha256 TEXT PRIMARY KEY, receipt TEXT NOT NULL, received_ns INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS awr_admissions (experiment_id TEXT PRIMARY KEY, receipt_sha256 TEXT NOT NULL, receipt TEXT NOT NULL, received_ns INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS campaign (singleton INTEGER PRIMARY KEY CHECK(singleton=1), manifest_set_sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS promotion_candidates (parent_experiment_id TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL, created_ns INTEGER NOT NULL, tied_runner INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(parent_experiment_id,kind));
            CREATE TABLE IF NOT EXISTS promotion_children (experiment_id TEXT PRIMARY KEY, parent_experiment_id TEXT NOT NULL, tied_runner INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS budget_reservations (lease_id TEXT PRIMARY KEY, experiment_id TEXT UNIQUE NOT NULL, gradient_steps INTEGER NOT NULL, gpu_seconds REAL NOT NULL, spend REAL NOT NULL, started_ns INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS budget_usage (experiment_id TEXT PRIMARY KEY, gradient_steps INTEGER NOT NULL, gpu_seconds REAL NOT NULL, spend REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS terminal_handoffs (experiment_id TEXT PRIMARY KEY, lease_id TEXT UNIQUE NOT NULL, worker_id TEXT NOT NULL, attempt INTEGER NOT NULL, grace_deadline_ns INTEGER NOT NULL);
            CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
        """)
        columns = {row[1] for row in self._connection.execute("PRAGMA table_info(events)")}
        if "job_digest" not in columns:
            self._connection.execute("ALTER TABLE events ADD COLUMN job_digest TEXT NOT NULL DEFAULT ''")
        promotion_columns = {row[1] for row in self._connection.execute("PRAGMA table_info(promotion_candidates)")}
        if "tied_runner" not in promotion_columns:
            self._connection.execute("ALTER TABLE promotion_candidates ADD COLUMN tied_runner INTEGER NOT NULL DEFAULT 0")
        child_columns = {row[1] for row in self._connection.execute("PRAGMA table_info(promotion_children)")}
        if "parent_experiment_id" not in child_columns:
            self._connection.execute("ALTER TABLE promotion_children ADD COLUMN parent_experiment_id TEXT")
        reservation_columns = {row[1] for row in self._connection.execute("PRAGMA table_info(budget_reservations)")}
        if "started_ns" not in reservation_columns:
            self._connection.execute("ALTER TABLE budget_reservations ADD COLUMN started_ns INTEGER NOT NULL DEFAULT 0")
        artifact_columns = {row[1] for row in self._connection.execute("PRAGMA table_info(artifacts)")}
        if "completion_lease_id" not in artifact_columns:
            self._connection.execute("ALTER TABLE artifacts ADD COLUMN completion_lease_id TEXT")
        if "completion_worker_id" not in artifact_columns:
            self._connection.execute("ALTER TABLE artifacts ADD COLUMN completion_worker_id TEXT")
        self._connection.commit()

    @contextmanager
    def _transaction(self):
        """Use an immediate writer lock: controller transitions are serializable."""
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    @contextmanager
    def _maybe_transaction(self):
        if self._connection.in_transaction:
            yield
        else:
            with self._transaction():
                yield

    def _event(self, experiment_id: str, state: str, worker_id: str | None, now_ns: int, detail: str = "") -> None:
        attempt = self._connection.execute("SELECT COUNT(*) FROM events WHERE experiment_id=?", (experiment_id,)).fetchone()[0]
        self._connection.execute("INSERT INTO events(experiment_id,job_digest,state,worker_id,attempt,timestamp_ns,detail) VALUES(?,?,?,?,?,?,?)", (experiment_id, experiment_id, state, worker_id, attempt, now_ns, detail[:256]))

    def _job(self, experiment_id: str) -> ExperimentJob:
        if experiment_id in self._jobs:
            return self._jobs[experiment_id]
        row = self._connection.execute("SELECT canonical FROM jobs WHERE experiment_id=?", (experiment_id,)).fetchone()
        if row is None:
            raise ValueError("unknown experiment")
        job = _parse(json.loads(row[0]))
        self._jobs[experiment_id] = job
        return job

    def _publication(self, experiment_id: str) -> Mapping[str, object] | None:
        row = self._connection.execute("SELECT publication,verified FROM artifacts WHERE experiment_id=?", (experiment_id,)).fetchone()
        if row is None or row[1] != 1 or row[0] is None:
            return None
        return parse_checkpoint_publication(json.loads(row[0])).canonical

    def _parent_publication(self, experiment_id: str) -> Mapping[str, object] | None:
        """Read recorded promotion lineage only; never infer a parent by path."""
        job = self._job(experiment_id)
        if job.admission.get("kind") != "continuation":
            return None
        row = self._connection.execute("SELECT parent_experiment_id FROM promotion_children WHERE experiment_id=?", (experiment_id,)).fetchone()
        if row is None or type(row[0]) is not str:
            raise RuntimeError("continuation lease lacks recorded parent publication")
        publication = self._publication(str(row[0]))
        if publication is None:
            raise RuntimeError("continuation parent publication is unavailable")
        expected = {"repository": publication["repository"], "revision": publication["immutable_revision"], "subpath": publication["remote_prefix"], "artifact_sha256": publication["artifact_sha256"], "receipt_sha256": publication["receipt_sha256"]}
        if dict(job.parent_checkpoint) != expected:
            raise RuntimeError("continuation parent publication no longer matches immutable job")
        return publication

    def _evaluation_matrix(self, experiment_id: str, capability: str) -> str | None:
        """Return the controller-owned matrix identity carried by a lease."""
        if capability == "training":
            return None
        if capability == "evaluation":
            return self._job(experiment_id).evaluation.matrix_sha256
        if capability == "final_evaluation":
            row = self._connection.execute(
                "SELECT matrix_sha256 FROM final_evaluations WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if row is None or re.fullmatch(r"[0-9a-f]{64}", str(row[0])) is None:
                raise RuntimeError("final evaluation lease lacks an immutable matrix")
            return str(row[0])
        raise ValueError("lease capability is invalid")

    @staticmethod
    def manifest_set_identity(jobs: Sequence[ExperimentJob]) -> str:
        """The initial immutable set is the controller's restart identity."""
        documents = [(job.experiment_id, dict(job.raw)) for job in sorted(jobs, key=lambda item: item.experiment_id)]
        return canonical_json_sha256({"schema_version": 1, "jobs": documents})

    def manifest_set_sha256(self) -> str | None:
        row = self._connection.execute("SELECT manifest_set_sha256 FROM campaign WHERE singleton=1").fetchone()
        return None if row is None else str(row[0])

    def capacity_snapshot(self, *, now_ns: int) -> dict[str, object]:
        """Return the exact authenticated lifecycle view under one writer lock.

        The external capacity daemon consumes this instead of guessing from
        worker process state.  Lease expiry is part of the same transaction so
        it cannot start/stop from a stale queue view.
        """
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("capacity snapshot clock is invalid")
        with self._lock, self._transaction():
            self._reconcile_expired_leases(now_ns)
            ready_training = int(self._connection.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('READY','RETRYABLE')").fetchone()[0])
            leaseable_training = self._leaseable_training_count()
            eval_ready = int(self._connection.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('EVAL_READY','EVAL_RETRYABLE')").fetchone()[0])
            eval_ready += int(self._connection.execute("SELECT COUNT(*) FROM final_evaluations WHERE state IN ('READY','RETRYABLE')").fetchone()[0])
            rows = self._connection.execute("SELECT lease_id,experiment_id,worker_id,capability,expires_ns FROM leases ORDER BY capability,worker_id,lease_id").fetchall()
            active_leases = [
                {"lease_id": str(row[0]), "experiment_id": str(row[1]), "worker_id": str(row[2]), "capability": str(row[3]), "expires_ns": int(row[4])}
                for row in rows
            ]
            return {
                "schema_version": 1,
                "ready_training_count": ready_training,
                "leaseable_training_count": leaseable_training,
                "eval_ready_count": eval_ready,
                "active_leases": active_leases,
                "idle_stop_recommended": leaseable_training == 0 and not any(item["capability"] == "training" for item in active_leases),
            }

    def _recovery_dependency_digests(self, job: ExperimentJob) -> tuple[str, ...]:
        """Return the exact persisted read-back receipt(s) for one recovery source.

        A promotion inherits a recovery source only when its original receipt
        remains in the controller's durable dependency table.  Parent IDs are
        also SHA-shaped, so looking only at ``job.dependencies`` would let a
        continuation confuse lineage with source-readback authority.
        """
        sources = [item for item in job.data_sources if item.kind == "recovery"]
        if not sources:
            return ()
        if len(sources) != 1:
            return ()
        item = sources[0]
        expected_source = {
            "repository": item.repository,
            "revision": item.revision,
            "prefix": item.prefix,
            "manifest_sha256": item.manifest_sha256,
            "tree_sha256": item.tree_sha256,
        }
        threshold = 15 if job.arm == "g" else 5
        verified: list[str] = []
        for dependency in job.dependencies:
            row = self._connection.execute(
                "SELECT receipt FROM dependency_receipts WHERE sha256=?", (dependency,)
            ).fetchone()
            if row is None:
                continue
            try:
                receipt = json.loads(str(row[0]))
            except (TypeError, ValueError):
                continue
            if (
                not isinstance(receipt, dict)
                or canonical_json_sha256(receipt) != dependency
                or receipt.get("schema_version") != 1
                or receipt.get("kind") != "verified_recovery_dependency"
                or receipt.get("readback_verified") is not True
                or receipt.get("source") != expected_source
            ):
                continue
            trajectories = receipt.get("trajectories")
            if not isinstance(trajectories, dict) or set(trajectories) != {
                "top_long", "top_short", "pant_long", "pant_short"
            }:
                continue
            if all(
                isinstance(identifiers, list)
                and all(type(identifier) is str and identifier for identifier in identifiers)
                and len(identifiers) == len(set(identifiers))
                and len(identifiers) >= threshold
                for identifiers in trajectories.values()
            ):
                verified.append(dependency)
        return tuple(sorted(verified))

    def _initial_state(self, job: ExperimentJob) -> str:
        if job.admission.get("kind") == "awr_style_weighted_replay":
            return "PENDING_MATERIALIZATION"
        if any(item.kind == "recovery" for item in job.data_sources):
            return "READY" if self._recovery_dependency_digests(job) else "BLOCKED_DATA"
        return "READY"

    def _gradient_steps(self, job: ExperimentJob) -> int:
        """Charge a continuation from its authenticated controller lineage.

        Hub prefixes are content locations, not training cursors.  A promoted
        child's delta therefore comes from the immutable parent job and its
        verified publication, both recorded by this controller transaction.
        Independent roots and seed repeats always pay their full target.
        """
        if job.admission.get("kind") != "continuation":
            return job.training.target_step

        source_experiment_id = job.admission.get("source_experiment_id")
        if type(source_experiment_id) is not str or not source_experiment_id:
            raise RuntimeError("continuation budget lacks a parent experiment")
        row = self._connection.execute(
            "SELECT parent_experiment_id FROM promotion_children WHERE experiment_id=?",
            (job.experiment_id,),
        ).fetchone()
        if row is None or str(row[0]) != source_experiment_id:
            raise RuntimeError("continuation budget parent lineage is invalid")

        parent = self._job(source_experiment_id)
        publication = self._publication(source_experiment_id)
        if publication is None:
            raise RuntimeError("continuation budget parent publication is unavailable")
        parent_step = publication.get("target_step")
        if type(parent_step) is not int or parent_step != parent.training.target_step:
            raise RuntimeError("continuation parent publication step does not match its immutable job")
        if not 0 < parent_step < job.training.target_step:
            raise RuntimeError("continuation parent step is not before the child target")

        expected_checkpoint = {
            "repository": publication["repository"],
            "revision": publication["immutable_revision"],
            "subpath": publication["remote_prefix"],
            "artifact_sha256": publication["artifact_sha256"],
            "receipt_sha256": publication["receipt_sha256"],
        }
        if dict(job.parent_checkpoint) != expected_checkpoint:
            raise RuntimeError("continuation budget parent publication no longer matches the immutable child")
        return job.training.target_step - parent_step

    def _budget_totals(self) -> tuple[int, float, float]:
        used = self._connection.execute("SELECT COALESCE(SUM(gradient_steps),0),COALESCE(SUM(gpu_seconds),0),COALESCE(SUM(spend),0) FROM budget_usage").fetchone()
        reserved = self._connection.execute("SELECT COALESCE(SUM(gradient_steps),0),COALESCE(SUM(gpu_seconds),0),COALESCE(SUM(spend),0) FROM budget_reservations").fetchone()
        return int(used[0]) + int(reserved[0]), float(used[1]) + float(reserved[1]), float(used[2]) + float(reserved[2])

    def _leaseable_training_count(self) -> int:
        """Count jobs a newly available trainer could lease at this instant.

        This is deliberately stricter than raw ``READY`` state: it applies the
        same evaluator backpressure, topology slots, continuation validation,
        and planned gradient/GPU budget accounting as training admission, but
        never mutates reservations or queue state.  The capacity daemon uses
        it to stop idle paid VMs without erasing the raw ready-work signal.
        """
        active = int(self._connection.execute("SELECT COUNT(*) FROM leases").fetchone()[0])
        active_training = int(self._connection.execute("SELECT COUNT(*) FROM leases WHERE capability='training'").fetchone()[0])
        available_slots = min(3 - active, 2 - active_training)
        if available_slots <= 0:
            return 0
        eval_backlog = int(self._connection.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('EVAL_READY','EVAL_RETRYABLE')").fetchone()[0])
        independent_ready = int(self._connection.execute("SELECT COUNT(*) FROM jobs j WHERE j.state IN ('READY','RETRYABLE') AND NOT EXISTS(SELECT 1 FROM dependencies d WHERE d.experiment_id=j.experiment_id)").fetchone()[0])
        if eval_backlog > 2 and independent_ready == 0:
            return 0
        rows = self._connection.execute("SELECT experiment_id,created_order FROM jobs WHERE state IN ('READY','RETRYABLE')").fetchall()
        rows = sorted(
            rows,
            key=lambda row: (
                -self._job(str(row[0])).training.target_step,
                -int(self._job(str(row[0])).admission.get("kind") == "seed_repeat"),
                int(row[1]),
            ),
        )
        total_steps, total_gpu_seconds, total_spend = self._budget_totals()
        tied_campaign = self._connection.execute("SELECT 1 FROM promotion_children WHERE tied_runner=1 LIMIT 1").fetchone()
        ceiling = self.tied_runner_gradient_step_ceiling if tied_campaign is not None else self.gradient_step_ceiling
        count = 0
        for experiment_id, _ in rows:
            if count >= available_slots:
                break
            try:
                steps = self._gradient_steps(self._job(str(experiment_id)))
            except (KeyError, TypeError, ValueError, RuntimeError):
                continue
            # Capacity is the paid-VM start authority, so it must reserve the
            # exact minimum lease requested by the production worker rather
            # than a smaller offline step-time estimate.
            gpu_seconds = max(
                steps * self.estimated_gpu_seconds_per_step,
                _PRODUCTION_TRAINING_LEASE_SECONDS,
            )
            spend = gpu_seconds * self.gpu_price_per_second
            if (
                total_steps + steps > ceiling
                or (self.gpu_seconds_ceiling is not None and total_gpu_seconds + gpu_seconds > self.gpu_seconds_ceiling)
                or (self.spend_ceiling is not None and total_spend + spend > self.spend_ceiling)
            ):
                continue
            total_steps += steps
            total_gpu_seconds += gpu_seconds
            total_spend += spend
            count += 1
        return count

    def _reserve_budget(self, lease_id: str, job: ExperimentJob, now_ns: int, lease_ns: int) -> bool:
        """Reserve both the planned gradients and the initial live GPU lease.

        A step-time estimate is only an estimate.  The live lease duration is
        an independently billable upper bound, so it must be reserved before a
        training worker is allowed to start.
        """
        steps = self._gradient_steps(job)
        gpu_seconds = max(steps * self.estimated_gpu_seconds_per_step, lease_ns / 1_000_000_000)
        spend = gpu_seconds * self.gpu_price_per_second
        total_steps, total_gpu_seconds, total_spend = self._budget_totals()
        # Once the controller admits the explicit tied finalist, 8K is the
        # campaign ceiling for both finalists.  Applying that ceiling only to
        # the tied lease makes admission depend on which finalist leases first.
        tied_campaign = self._connection.execute(
            "SELECT 1 FROM promotion_children WHERE tied_runner=1 LIMIT 1"
        ).fetchone()
        ceiling = self.tied_runner_gradient_step_ceiling if tied_campaign is not None else self.gradient_step_ceiling
        if total_steps + steps > ceiling or (self.gpu_seconds_ceiling is not None and total_gpu_seconds + gpu_seconds > self.gpu_seconds_ceiling) or (self.spend_ceiling is not None and total_spend + spend > self.spend_ceiling):
            return False
        self._connection.execute("INSERT INTO budget_reservations VALUES(?,?,?,?,?,?)", (lease_id, job.experiment_id, steps, gpu_seconds, spend, now_ns))
        return True

    def _reserve_evaluation_budget(self, lease_id: str, experiment_id: str, now_ns: int, lease_ns: int) -> bool:
        """Reserve the rollout GPU for the full evaluation lease duration."""
        gpu_seconds = lease_ns / 1_000_000_000
        spend = gpu_seconds * self.gpu_price_per_second
        _, total_gpu_seconds, total_spend = self._budget_totals()
        if (self.gpu_seconds_ceiling is not None and total_gpu_seconds + gpu_seconds > self.gpu_seconds_ceiling) or (self.spend_ceiling is not None and total_spend + spend > self.spend_ceiling):
            return False
        self._connection.execute("INSERT INTO budget_reservations VALUES(?,?,?,?,?,?)", (lease_id, experiment_id, 0, gpu_seconds, spend, now_ns))
        return True

    def _extend_evaluation_budget(self, lease_id: str, expires_ns: int) -> bool:
        """Grow an evaluation reservation before extending its heartbeat."""
        return self._extend_wall_clock_budget(lease_id, expires_ns, capability="evaluation")

    def _extend_training_budget(self, lease_id: str, expires_ns: int) -> bool:
        """Grow a training reservation before extending its heartbeat."""
        return self._extend_wall_clock_budget(lease_id, expires_ns, capability="training")

    def _extend_wall_clock_budget(self, lease_id: str, expires_ns: int, *, capability: str) -> bool:
        """Reserve every live GPU second covered by one extended lease."""
        row = self._connection.execute("SELECT gpu_seconds,spend,started_ns FROM budget_reservations WHERE lease_id=?", (lease_id,)).fetchone()
        if row is None:
            raise RuntimeError(capability + " lease has no GPU budget reservation")
        reserved_gpu_seconds, reserved_spend, started_ns = float(row[0]), float(row[1]), int(row[2])
        required_gpu_seconds = max(reserved_gpu_seconds, max(0.0, (expires_ns - started_ns) / 1_000_000_000))
        required_spend = required_gpu_seconds * self.gpu_price_per_second
        added_gpu_seconds = required_gpu_seconds - reserved_gpu_seconds
        added_spend = required_spend - reserved_spend
        _, total_gpu_seconds, total_spend = self._budget_totals()
        if (self.gpu_seconds_ceiling is not None and total_gpu_seconds + added_gpu_seconds > self.gpu_seconds_ceiling) or (self.spend_ceiling is not None and total_spend + added_spend > self.spend_ceiling):
            return False
        self._connection.execute("UPDATE budget_reservations SET gpu_seconds=?,spend=? WHERE lease_id=?", (required_gpu_seconds, required_spend, lease_id))
        return True

    def _release_budget(
        self, lease_id: str, now_ns: int, *, consume_steps: bool = False,
        settle_elapsed: bool = True,
    ) -> None:
        row = self._connection.execute("SELECT experiment_id,gradient_steps,started_ns FROM budget_reservations WHERE lease_id=?", (lease_id,)).fetchone()
        if row is None:
            return
        self._connection.execute("DELETE FROM budget_reservations WHERE lease_id=?", (lease_id,))
        elapsed_seconds = (
            max(0.0, (now_ns - int(row[2])) / 1_000_000_000)
            if settle_elapsed else 0.0
        )
        self._connection.execute(
            "INSERT INTO budget_usage(experiment_id,gradient_steps,gpu_seconds,spend) VALUES(?,?,?,?) "
            "ON CONFLICT(experiment_id) DO UPDATE SET gradient_steps=MAX(budget_usage.gradient_steps,excluded.gradient_steps),gpu_seconds=budget_usage.gpu_seconds+excluded.gpu_seconds,spend=budget_usage.spend+excluded.spend",
            (row[0], int(row[1]) if consume_steps else 0, elapsed_seconds, elapsed_seconds * self.gpu_price_per_second),
        )

    def _settle_budget_for_completion_grace(self, lease_id: str, now_ns: int) -> None:
        """Settle elapsed GPU time while retaining this run's gradient hold.

        An expired training worker can still deliver its exact terminal
        receipt during the controller-owned handoff.  Releasing the entire
        reservation here would make a later success free in the gradient
        ledger (or let a different job spend the same budget).  Retain only
        the authenticated gradient delta; the original attempt's wall-clock
        GPU/spend is settled now and must never be charged again at receipt
        reconciliation or grace timeout.
        """
        row = self._connection.execute(
            "SELECT experiment_id,started_ns FROM budget_reservations WHERE lease_id=?",
            (lease_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("training completion grace has no budget reservation")
        elapsed_seconds = max(0.0, (now_ns - int(row[1])) / 1_000_000_000)
        self._connection.execute(
            "UPDATE budget_reservations SET gpu_seconds=0.0,spend=0.0,started_ns=? WHERE lease_id=?",
            (now_ns, lease_id),
        )
        self._connection.execute(
            "INSERT INTO budget_usage(experiment_id,gradient_steps,gpu_seconds,spend) VALUES(?,?,?,?) "
            "ON CONFLICT(experiment_id) DO UPDATE SET gradient_steps=MAX(budget_usage.gradient_steps,excluded.gradient_steps),gpu_seconds=budget_usage.gpu_seconds+excluded.gpu_seconds,spend=budget_usage.spend+excluded.spend",
            (str(row[0]), 0, elapsed_seconds, elapsed_seconds * self.gpu_price_per_second),
        )

    def _consume_completion_grace_gradient(
        self, lease_id: str, experiment_id: str, now_ns: int,
    ) -> None:
        """Convert one retained grace reservation into exactly-once usage."""
        job = self._job(experiment_id)
        expected_steps = self._gradient_steps(job)
        row = self._connection.execute(
            "SELECT experiment_id,gradient_steps,gpu_seconds,spend FROM budget_reservations WHERE lease_id=?",
            (lease_id,),
        ).fetchone()
        if (
            row is None
            or str(row[0]) != experiment_id
            or int(row[1]) != expected_steps
            or float(row[2]) != 0.0
            or float(row[3]) != 0.0
        ):
            raise RuntimeError("training completion grace budget reservation is invalid")
        total_steps, _total_gpu_seconds, _total_spend = self._budget_totals()
        tied_campaign = self._connection.execute(
            "SELECT 1 FROM promotion_children WHERE tied_runner=1 LIMIT 1"
        ).fetchone()
        ceiling = (
            self.tied_runner_gradient_step_ceiling
            if tied_campaign is not None else self.gradient_step_ceiling
        )
        # The retained reservation is already included in ``total_steps``.
        # A breach therefore indicates an inconsistent/legacy ledger rather
        # than permission to publish an unaccounted late training result.
        if total_steps > ceiling:
            raise RuntimeError("training completion grace exceeds the campaign gradient budget")
        self._release_budget(
            lease_id, now_ns, consume_steps=True, settle_elapsed=False,
        )

    def budget_usage(self) -> tuple[int, float, float]:
        row = self._connection.execute("SELECT COALESCE(SUM(gradient_steps),0),COALESCE(SUM(gpu_seconds),0),COALESCE(SUM(spend),0) FROM budget_usage").fetchone()
        return int(row[0]), float(row[1]), float(row[2])

    def pending_promotions(self, parent_experiment_id: str) -> tuple[str, ...]:
        rows = self._connection.execute("SELECT kind FROM promotion_candidates WHERE parent_experiment_id=? AND state='PENDING' ORDER BY CASE kind WHEN 'step_1000' THEN 0 WHEN 'seed_repeat' THEN 1 ELSE 2 END", (parent_experiment_id,)).fetchall()
        return tuple(str(row[0]) for row in rows)

    def add_jobs(self, jobs: Sequence[ExperimentJob], *, manifest_set_sha256: str | None = None) -> None:
        """Bootstrap once; a restart can only replay the exact immutable set."""
        expected = self.manifest_set_identity(jobs)
        if manifest_set_sha256 is not None and manifest_set_sha256 != expected:
            raise ValueError("manifest set digest does not match canonical jobs")
        with self._lock, self._transaction():
            campaign = self.manifest_set_sha256()
            if campaign is not None:
                if campaign != expected:
                    raise ValueError("controller is already bootstrapped with a different manifest set")
                # Crash/restart retry: each identity must be the exact persisted job.
                for job in jobs:
                    row = self._connection.execute("SELECT canonical FROM jobs WHERE experiment_id=?", (job.experiment_id,)).fetchone()
                    if row is None or json.loads(row[0]) != dict(job.raw):
                        raise ValueError("controller bootstrap replay is not identical")
                return
            self._connection.execute("INSERT INTO campaign VALUES(1,?)", (expected,))
            for job in jobs:
                # ExperimentJob admits only 500/1K/2K terminal rungs; no zero-step pseudo jobs.
                canonical = json.dumps(dict(job.raw), sort_keys=True, separators=(",", ":"))
                state = self._initial_state(job)
                try:
                    order = self._connection.execute("SELECT COALESCE(MAX(created_order), -1) + 1 FROM jobs").fetchone()[0]
                    self._connection.execute("INSERT INTO jobs VALUES(?,?,?,?,?,NULL)", (job.experiment_id, canonical, state, "training", order))
                except sqlite3.IntegrityError as error:
                    raise ValueError("duplicate experiment job") from error
                self._jobs[job.experiment_id] = job
                for dependency in job.dependencies:
                    self._connection.execute("INSERT INTO dependencies VALUES(?,?)", (job.experiment_id, dependency))
                self._event(job.experiment_id, state, None, 0)

    def satisfy_dependency(self, receipt: Mapping[str, object], now_ns: int) -> int:
        """Unblock recovery arms only after a read-back verified source receipt."""
        value = dict(receipt)
        categories = {"top_long", "top_short", "pant_long", "pant_short"}
        source, trajectories = value.get("source"), value.get("trajectories")
        if set(value) != {"schema_version", "kind", "source", "readback_verified", "trajectories"} or value.get("schema_version") != 1 or value.get("kind") != "verified_recovery_dependency" or value.get("readback_verified") is not True or not isinstance(source, Mapping) or set(source) != {"repository", "revision", "prefix", "manifest_sha256", "tree_sha256"} or not isinstance(trajectories, Mapping) or set(trajectories) != categories:
            raise ValueError("recovery dependency receipt is invalid")
        counts: dict[str, int] = {}
        for category in categories:
            ids = trajectories[category]
            if not isinstance(ids, list) or not all(type(item) is str and item for item in ids) or len(ids) != len(set(ids)):
                raise ValueError("recovery trajectories are not distinct")
            counts[category] = len(ids)
        digest = canonical_json_sha256(value)
        with self._lock, self._transaction():
            # A source match is not authority to unlock a job.  The immutable
            # job set declares the exact canonical receipt identity it accepts.
            # Keep unrelated-but-plausible receipts out of the durable proof
            # table too, so a later state change cannot accidentally reuse one.
            if self._connection.execute("SELECT 1 FROM dependencies WHERE dependency=? LIMIT 1", (digest,)).fetchone() is None:
                return 0
            self._connection.execute("INSERT OR IGNORE INTO dependency_receipts VALUES(?,?,?)", (digest, json.dumps(value, sort_keys=True, separators=(",", ":")), now_ns))
            unblocked = 0
            for (experiment_id,) in self._connection.execute("SELECT experiment_id FROM jobs WHERE state='BLOCKED_DATA' ORDER BY created_order").fetchall():
                job = self._job(str(experiment_id))
                sources = [item for item in job.data_sources if item.kind == "recovery"]
                if len(sources) != 1:
                    continue
                item = sources[0]
                expected = {"repository": item.repository, "revision": item.revision, "prefix": item.prefix, "manifest_sha256": item.manifest_sha256, "tree_sha256": item.tree_sha256}
                threshold = 15 if job.arm == "g" else 5
                if digest in job.dependencies and dict(source) == expected and all(counts[name] >= threshold for name in categories):
                    self._connection.execute("UPDATE jobs SET state='READY' WHERE experiment_id=?", (experiment_id,))
                    self._event(str(experiment_id), "READY", None, now_ns, "verified_recovery:" + digest)
                    unblocked += 1
            return unblocked

    def satisfy_awr_style_admission(self, experiment_id: str, receipt: Mapping[str, object], now_ns: int) -> str:
        """Admit one weighted replay job only from its explicit read-back receipt.

        This intentionally does not reuse generic recovery ``dependencies``:
        evidence, replay configuration, request-set source, child profile, and
        winner receipt are all bound by the dedicated admission document.
        """
        if type(experiment_id) is not str or not experiment_id or type(now_ns) is not int:
            raise ValueError("AWR-style admission request is invalid")
        from lehome_train.groot.experiment_ablation import validate_awr_style_materialization_receipt

        with self._lock, self._transaction():
            job = self._job(experiment_id)
            receipt_sha = validate_awr_style_materialization_receipt(job, receipt)
            existing = self._connection.execute(
                "SELECT receipt_sha256 FROM awr_admissions WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if existing is not None:
                if str(existing[0]) != receipt_sha:
                    raise ValueError("AWR-style job already has a different admission receipt")
                if self.state(experiment_id) != "READY":
                    raise RuntimeError("AWR-style admission state is inconsistent")
                return receipt_sha
            if self.state(experiment_id) != "PENDING_MATERIALIZATION":
                raise RuntimeError("AWR-style job is not pending materialization")
            encoded = json.dumps(dict(receipt), sort_keys=True, separators=(",", ":"))
            self._connection.execute(
                "INSERT INTO awr_admissions VALUES(?,?,?,?)",
                (experiment_id, receipt_sha, encoded, now_ns),
            )
            self._connection.execute("UPDATE jobs SET state='READY' WHERE experiment_id=?", (experiment_id,))
            self._event(experiment_id, "READY", None, now_ns, "awr_style_readback:" + receipt_sha)
            return receipt_sha

    def _reconcile_expired_leases(self, now_ns: int) -> None:
        """Atomically revoke and settle every lease which has already expired.

        A worker owns a transition only through its live row in ``leases``.
        Reconciliation therefore happens before every operation that can
        renew, rehydrate, or settle that ownership.  Deleting the row before
        settling its reservation makes repeated reconciliation idempotent and
        leaves a late worker unable to mutate a replacement attempt.
        """
        # An expired training worker may already have durably written its
        # result while its first /complete request was lost.  Keep that exact
        # identity non-leaseable for a bounded controller-owned handoff.
        # Processing old handoffs first makes the retry release idempotent.
        for experiment_id, lease_id in self._connection.execute(
            "SELECT experiment_id,lease_id FROM terminal_handoffs WHERE grace_deadline_ns < ?",
            (now_ns,),
        ).fetchall():
            experiment_id = str(experiment_id)
            row = self._connection.execute(
                "SELECT state FROM jobs WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
            self._connection.execute(
                "DELETE FROM terminal_handoffs WHERE experiment_id=?", (experiment_id,)
            )
            if row is None or str(row[0]) != "COMPLETION_GRACE":
                continue
            # The attempt never supplied a durable terminal receipt.  Its
            # GPU/spend was already settled at lease expiry; discard only the
            # retained gradient hold without charging any training steps.
            self._release_budget(str(lease_id), now_ns, settle_elapsed=False)
            self._connection.execute(
                "UPDATE jobs SET state='RETRYABLE' WHERE experiment_id=?", (experiment_id,)
            )
            self._event(experiment_id, "RETRYABLE", None, now_ns, "terminal receipt grace expired")

        for lease_id, experiment_id, worker_id, capability in self._connection.execute("SELECT lease_id,experiment_id,worker_id,capability FROM leases WHERE expires_ns < ?", (now_ns,)).fetchall():
            self._connection.execute("DELETE FROM leases WHERE lease_id=?", (lease_id,))
            if capability == "training":
                artifact = self._connection.execute(
                    "SELECT 1 FROM artifacts WHERE experiment_id=?", (experiment_id,)
                ).fetchone()
                if artifact is None:
                    self._settle_budget_for_completion_grace(str(lease_id), now_ns)
                    attempt = int(self._connection.execute(
                        "SELECT COUNT(*) FROM events WHERE experiment_id=?", (experiment_id,)
                    ).fetchone()[0])
                    self._connection.execute(
                        "INSERT INTO terminal_handoffs(experiment_id,lease_id,worker_id,attempt,grace_deadline_ns) VALUES(?,?,?,?,?)",
                        (experiment_id, lease_id, worker_id, attempt, now_ns + TERMINAL_RECEIPT_GRACE_NS),
                    )
                    self._connection.execute(
                        "UPDATE jobs SET state='COMPLETION_GRACE' WHERE experiment_id=?", (experiment_id,)
                    )
                    self._event(experiment_id, "COMPLETION_GRACE", str(worker_id), now_ns, "terminal receipt handoff")
                    continue
            self._release_budget(str(lease_id), now_ns)
            if capability == "final_evaluation":
                state = "RETRYABLE"
                self._connection.execute("UPDATE final_evaluations SET state=? WHERE experiment_id=?", (state, experiment_id))
            else:
                state = "EVAL_RETRYABLE" if capability == "evaluation" else "RETRYABLE"
                self._connection.execute("UPDATE jobs SET state=? WHERE experiment_id=?", (state, experiment_id))
            self._event(experiment_id, state, None, now_ns, "lease expired")

    def _require_live_lease(self, lease: JobLease, now_ns: int) -> None:
        """Commit expiry reconciliation before admitting one lease transition.

        This helper is called while ``_lock`` is held, but deliberately opens
        and closes its own transaction.  A stale request must raise only after
        the lease deletion, retry state, and budget settlement have committed;
        raising from inside that transaction would roll the reconciliation
        back and revive the expired authority.
        """
        with self._transaction():
            self._reconcile_expired_leases(now_ns)
            row = self._connection.execute(
                "SELECT worker_id,capability FROM leases WHERE lease_id=? AND experiment_id=?",
                (lease.lease_id, lease.experiment_id),
            ).fetchone()
        if row is None or str(row[0]) != lease.worker_id or str(row[1]) != lease.capability:
            raise ValueError("invalid lease")

    def lease_next(self, worker_id: str, capability: str, now_ns: int, lease_ns: int, manifest_set_sha256: str | None = None) -> JobLease | None:
        if capability not in {"training", "evaluation", "final_evaluation"} or type(worker_id) is not str or not worker_id or type(now_ns) is not int or type(lease_ns) is not int or lease_ns <= 0:
            raise ValueError("lease request is invalid")
        with self._lock, self._transaction():
            if manifest_set_sha256 is not None and manifest_set_sha256 != self.manifest_set_sha256():
                raise ValueError("worker manifest set does not match controller")
            self._reconcile_expired_leases(now_ns)
            existing = self._connection.execute("SELECT lease_id,experiment_id,expires_ns FROM leases WHERE worker_id=? AND capability=?", (worker_id, capability)).fetchone()
            if existing:
                experiment_id = str(existing[1])
                return JobLease(str(existing[0]), experiment_id, worker_id, capability, int(existing[2]), self._job(experiment_id), self._publication(experiment_id), self._parent_publication(experiment_id), self._evaluation_matrix(experiment_id, capability))
            active = self._connection.execute("SELECT COUNT(*) FROM leases").fetchone()[0]
            cap_active = self._connection.execute("SELECT COUNT(*) FROM leases WHERE capability=?", (capability,)).fetchone()[0]
            evaluator_active = self._connection.execute("SELECT COUNT(*) FROM leases WHERE capability IN ('evaluation','final_evaluation')").fetchone()[0]
            if active >= 3 or cap_active >= (2 if capability == "training" else 1) or (capability in {"evaluation", "final_evaluation"} and evaluator_active >= 1):
                return None
            if capability == "training":
                eval_backlog = int(self._connection.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('EVAL_READY','EVAL_RETRYABLE')").fetchone()[0])
                independent_ready = int(self._connection.execute("SELECT COUNT(*) FROM jobs j WHERE j.state IN ('READY','RETRYABLE') AND NOT EXISTS(SELECT 1 FROM dependencies d WHERE d.experiment_id=j.experiment_id)").fetchone()[0])
                if eval_backlog > 2 and independent_ready == 0:
                    return None
            if capability == "final_evaluation":
                rows = self._connection.execute("SELECT f.experiment_id,j.created_order FROM final_evaluations f JOIN jobs j ON j.experiment_id=f.experiment_id WHERE f.state IN ('READY','RETRYABLE') ORDER BY j.created_order").fetchall()
            else:
                states = ("READY", "RETRYABLE") if capability == "training" else ("EVAL_READY", "EVAL_RETRYABLE")
                rows = self._connection.execute("SELECT experiment_id,created_order FROM jobs WHERE state IN (?,?) ORDER BY created_order", states).fetchall()
            # Continuations get the freed slot ahead of unstarted baseline arms;
            # this is the explicit no-wave priority policy, with FIFO within rung.
            rows = sorted(
                rows,
                key=lambda row: (
                    -self._job(str(row[0])).training.target_step,
                    -int(self._job(str(row[0])).admission.get("kind") == "seed_repeat"),
                    int(row[1]),
                ),
            )
            for row in rows:
                experiment_id = str(row[0])
                publication = self._publication(experiment_id)
                if capability in {"evaluation", "final_evaluation"} and publication is None:
                    raise RuntimeError("evaluation-ready job has no verified publication")
                job = self._job(experiment_id)
                lease = JobLease(uuid.uuid4().hex, experiment_id, worker_id, capability, now_ns + lease_ns, job, publication, self._parent_publication(experiment_id) if capability == "training" else None, self._evaluation_matrix(experiment_id, capability))
                if capability == "training" and not self._reserve_budget(lease.lease_id, lease.job, now_ns, lease_ns):
                    self._connection.execute("UPDATE jobs SET state='BLOCKED_BUDGET' WHERE experiment_id=?", (experiment_id,))
                    self._event(experiment_id, "BLOCKED_BUDGET", None, now_ns, "budget admission")
                    continue
                if capability in {"evaluation", "final_evaluation"} and not self._reserve_evaluation_budget(lease.lease_id, experiment_id, now_ns, lease_ns):
                    state = "BLOCKED_BUDGET" if capability == "final_evaluation" else "EVAL_BLOCKED_BUDGET"
                    if capability == "final_evaluation":
                        self._connection.execute("UPDATE final_evaluations SET state=? WHERE experiment_id=?", (state, experiment_id))
                    else:
                        self._connection.execute("UPDATE jobs SET state=? WHERE experiment_id=?", (state, experiment_id))
                    self._event(experiment_id, state, None, now_ns, "budget admission")
                    continue
                self._connection.execute("INSERT INTO leases VALUES(?,?,?,?,?)", (lease.lease_id, experiment_id, worker_id, capability, lease.expires_ns))
                if capability == "final_evaluation":
                    self._connection.execute("UPDATE final_evaluations SET state='LEASED' WHERE experiment_id=?", (experiment_id,))
                else:
                    self._connection.execute("UPDATE jobs SET state='LEASED' WHERE experiment_id=?", (experiment_id,))
                self._event(experiment_id, "LEASED", worker_id, now_ns)
                return lease
            return None

    def heartbeat(self, worker_id: str, lease_id: str, now_ns: int, lease_ns: int) -> JobLease:
        if type(worker_id) is not str or type(lease_id) is not str or type(now_ns) is not int or type(lease_ns) is not int or lease_ns <= 0:
            raise ValueError("heartbeat is invalid")
        budget_blocked = False
        result: JobLease | None = None
        with self._lock:
            with self._transaction():
                self._reconcile_expired_leases(now_ns)
                row = self._connection.execute("SELECT experiment_id,capability FROM leases WHERE lease_id=? AND worker_id=?", (lease_id, worker_id)).fetchone()
            if row is None:
                raise ValueError("lease does not belong to worker")
            with self._transaction():
                # A second controller process may have transitioned this lease
                # after the committed reconciliation above.  Recheck before
                # extending its budget or expiry; a heartbeat never recreates
                # an ownership row.
                if self._connection.execute(
                    "SELECT 1 FROM leases WHERE lease_id=? AND worker_id=?",
                    (lease_id, worker_id),
                ).fetchone() is None:
                    raise ValueError("lease does not belong to worker")
                experiment_id, capability, expires = str(row[0]), str(row[1]), now_ns + lease_ns
                if capability == "training" and not self._extend_training_budget(lease_id, expires):
                    # Do not leave a paid training process with a renewable lease
                    # once its next interval cannot be reserved.  The worker loses
                    # authority immediately and can only resume under a newly
                    # budgeted campaign decision.
                    self._connection.execute("DELETE FROM leases WHERE lease_id=?", (lease_id,))
                    self._release_budget(lease_id, now_ns)
                    self._connection.execute("UPDATE jobs SET state='BLOCKED_BUDGET' WHERE experiment_id=?", (experiment_id,))
                    self._event(experiment_id, "BLOCKED_BUDGET", worker_id, now_ns, "training heartbeat budget")
                    budget_blocked = True
                elif capability in {"evaluation", "final_evaluation"} and not self._extend_evaluation_budget(lease_id, expires):
                    raise RuntimeError("evaluation heartbeat exceeds the campaign GPU budget")
                elif not budget_blocked:
                    self._connection.execute("UPDATE leases SET expires_ns=? WHERE lease_id=?", (expires, lease_id))
                    self._event(experiment_id, "TRAINING" if capability == "training" else ("FINAL_EVALUATING" if capability == "final_evaluation" else "EVALUATING"), worker_id, now_ns, "heartbeat")
                    result = JobLease(lease_id, experiment_id, worker_id, capability, expires, self._job(experiment_id), self._publication(experiment_id), self._parent_publication(experiment_id) if capability == "training" else None, self._evaluation_matrix(experiment_id, capability))
        if budget_blocked:
            raise RuntimeError("training heartbeat exceeds the campaign GPU budget")
        if result is None:
            raise RuntimeError("heartbeat did not produce a lease")
        return result

    def lease_for(self, lease_id: str, experiment_id: str, worker_id: str, *, now_ns: int) -> JobLease:
        """Rehydrate one active lease for a signed HTTP transition request."""
        if not all(type(value) is str and value for value in (lease_id, experiment_id, worker_id)) or type(now_ns) is not int:
            raise ValueError("lease identity is invalid")
        with self._lock:
            with self._transaction():
                self._reconcile_expired_leases(now_ns)
                row = self._connection.execute("SELECT capability,expires_ns FROM leases WHERE lease_id=? AND experiment_id=? AND worker_id=?", (lease_id, experiment_id, worker_id)).fetchone()
            if row is None:
                raise ValueError("invalid lease")
            capability = str(row[0])
            return JobLease(lease_id, experiment_id, worker_id, capability, int(row[1]), self._job(experiment_id), self._publication(experiment_id), self._parent_publication(experiment_id) if capability == "training" else None, self._evaluation_matrix(experiment_id, capability))

    def _finish(self, lease: JobLease, state: str, now_ns: int, detail: str) -> None:
        if lease.capability == "final_evaluation":
            raise ValueError("final evaluation requires its dedicated transition")
        row = self._connection.execute("SELECT worker_id,capability FROM leases WHERE lease_id=? AND experiment_id=?", (lease.lease_id, lease.experiment_id)).fetchone()
        if row is None or row[0] != lease.worker_id or row[1] != lease.capability:
            raise ValueError("invalid lease")
        self._connection.execute("DELETE FROM leases WHERE lease_id=?", (lease.lease_id,))
        self._release_budget(lease.lease_id, now_ns, consume_steps=lease.capability == "training" and state == "PUBLISHING")
        self._connection.execute("UPDATE jobs SET state=? WHERE experiment_id=?", (state, lease.experiment_id))
        self._event(lease.experiment_id, state, lease.worker_id, now_ns, detail)

    def _existing_terminal_receipt_state(
        self, lease_id: str, experiment_id: str, worker_id: str, receipt: str,
    ) -> str | None:
        """Return an exact already-settled receipt state, never new authority."""
        row = self._connection.execute(
            "SELECT receipt_sha256,completion_lease_id,completion_worker_id FROM artifacts WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return None
        if str(row[0]) != receipt:
            raise ValueError("terminal receipt mismatch")
        if str(row[1]) != lease_id or str(row[2]) != worker_id:
            raise ValueError("terminal receipt ownership mismatch")
        state = self._connection.execute(
            "SELECT state FROM jobs WHERE experiment_id=?", (experiment_id,)
        ).fetchone()
        if state is None:
            raise ValueError("unknown experiment")
        return str(state[0])

    def _reconcile_completion_grace(
        self, lease_id: str, experiment_id: str, worker_id: str, receipt: str, now_ns: int,
    ) -> str | None:
        """Consume an exact expired-training receipt during its bounded handoff.

        This helper is called under the immediate transaction that expires
        leases and accepts terminal receipts.  There is therefore no visible
        READY interval in which another trainer can lease the same work.
        """
        handoff = self._connection.execute(
            "SELECT lease_id,worker_id,grace_deadline_ns FROM terminal_handoffs WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()
        if handoff is None:
            return None
        if str(handoff[0]) != lease_id or str(handoff[1]) != worker_id:
            raise ValueError("terminal receipt ownership mismatch")
        if int(handoff[2]) < now_ns:
            raise ValueError("invalid lease")
        state = self._connection.execute(
            "SELECT state FROM jobs WHERE experiment_id=?", (experiment_id,)
        ).fetchone()
        if state is None or str(state[0]) != "COMPLETION_GRACE":
            raise ValueError("terminal receipt handoff is invalid")
        # This consumes the same authenticated lineage delta that a live
        # ``_finish(..., PUBLISHING, ...)`` would have consumed.  It also
        # verifies that expiry retained exactly that delta, preventing a late
        # receipt from bypassing the campaign ceiling after other leases run.
        self._consume_completion_grace_gradient(lease_id, experiment_id, now_ns)
        self._connection.execute(
            "DELETE FROM terminal_handoffs WHERE experiment_id=?", (experiment_id,)
        )
        self._connection.execute(
            "UPDATE jobs SET state='PUBLISHING' WHERE experiment_id=?", (experiment_id,)
        )
        self._event(experiment_id, "PUBLISHING", worker_id, now_ns, receipt)
        self._connection.execute(
            "INSERT INTO artifacts(experiment_id,receipt_sha256,completion_lease_id,completion_worker_id,publication,verified) VALUES(?,?,?,?,NULL,0)",
            (experiment_id, receipt, lease_id, worker_id),
        )
        return "PUBLISHING"

    def reconcile_terminal_receipt(self, lease: JobLease, terminal_receipt_sha256: str, now_ns: int) -> str:
        """Atomically settle one immutable terminal receipt exactly once.

        A worker writes this receipt before sending its completion request.  If
        the controller commits while the HTTP response is lost, the original
        lease no longer exists.  The exact same receipt is still safe to
        replay: it is evidence of the already-committed transition, not a new
        authority to mutate the job.  A different receipt remains a hard
        mismatch, and an absent receipt still requires the original live lease.
        """
        if lease.capability != "training" or type(terminal_receipt_sha256) is not str or len(terminal_receipt_sha256) != 64:
            raise ValueError("terminal receipt is invalid")
        with self._lock, self._transaction():
            self._reconcile_expired_leases(now_ns)
            state = self._existing_terminal_receipt_state(
                lease.lease_id, lease.experiment_id, lease.worker_id, terminal_receipt_sha256,
            )
            if state is not None:
                return state
            state = self._reconcile_completion_grace(
                lease.lease_id, lease.experiment_id, lease.worker_id, terminal_receipt_sha256, now_ns,
            )
            if state is not None:
                return state
            # An overlapping request can only make the exact receipt
            # idempotent under the single writer transaction.  No operation
            # here recreates a stale lease authority.
            self._finish(lease, "PUBLISHING", now_ns, terminal_receipt_sha256)
            self._connection.execute(
                "INSERT INTO artifacts(experiment_id,receipt_sha256,completion_lease_id,completion_worker_id,publication,verified) VALUES(?,?,?,?,NULL,0)",
                (lease.experiment_id, terminal_receipt_sha256, lease.lease_id, lease.worker_id),
            )
            return "PUBLISHING"

    def reconcile_terminal_receipt_by_identity(
        self, lease_id: str, experiment_id: str, worker_id: str,
        terminal_receipt_sha256: str, now_ns: int,
    ) -> str:
        """HTTP-safe form of :meth:`reconcile_terminal_receipt`.

        It deliberately checks the immutable receipt before rehydrating a
        lease.  The first completion still requires the exact live owner; only
        a receipt that is already durable may outlive that lease.
        """
        if (
            not all(type(value) is str and value for value in (lease_id, experiment_id, worker_id))
            or type(terminal_receipt_sha256) is not str
            or len(terminal_receipt_sha256) != 64
            or type(now_ns) is not int
        ):
            raise ValueError("terminal receipt identity is invalid")
        with self._lock, self._transaction():
            self._reconcile_expired_leases(now_ns)
            state = self._existing_terminal_receipt_state(
                lease_id, experiment_id, worker_id, terminal_receipt_sha256,
            )
            if state is not None:
                return state
            state = self._reconcile_completion_grace(
                lease_id, experiment_id, worker_id, terminal_receipt_sha256, now_ns,
            )
            if state is not None:
                return state
        lease = self.lease_for(lease_id, experiment_id, worker_id, now_ns=now_ns)
        return self.reconcile_terminal_receipt(lease, terminal_receipt_sha256, now_ns)

    def complete(self, lease: JobLease, terminal_receipt_sha256: str, now_ns: int) -> None:
        self.reconcile_terminal_receipt(lease, terminal_receipt_sha256, now_ns)

    def publication_verified(self, experiment_id: str, envelope: Mapping[str, object], now_ns: int) -> str:
        with self._lock, self._transaction():
            job = self._job(experiment_id)
            row = self._connection.execute(
                "SELECT receipt_sha256,publication,verified FROM artifacts WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                raise ValueError("publication receipt mismatch")
            publication = bind_checkpoint_publication(job, str(row[0]), envelope)
            if int(row[2]) == 1:
                if type(row[1]) is not str:
                    raise RuntimeError("verified publication is missing its canonical envelope")
                existing = parse_checkpoint_publication(json.loads(row[1])).canonical
                if dict(existing) != dict(publication.canonical):
                    raise ValueError("publication replay mismatch")
                state = self._connection.execute(
                    "SELECT state FROM jobs WHERE experiment_id=?", (experiment_id,),
                ).fetchone()
                if state is None:
                    raise ValueError("unknown experiment")
                return str(state[0])
            self._connection.execute("UPDATE artifacts SET verified=1, publication=? WHERE experiment_id=?", (json.dumps(dict(publication.canonical), sort_keys=True, separators=(",", ":")), experiment_id))
            self._connection.execute("UPDATE jobs SET state='EVAL_READY', receipt_sha256=? WHERE experiment_id=?", (publication.receipt_sha256, experiment_id))
            self._event(experiment_id, "EVAL_READY", None, now_ns, publication.receipt_sha256)
            return "EVAL_READY"

    def retryable(self, lease: JobLease, reason: str, now_ns: int) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("retry reason is invalid")
        with self._lock:
            self._require_live_lease(lease, now_ns)
            with self._transaction():
                if lease.capability == "final_evaluation":
                    self._finish_final_evaluation(lease, "RETRYABLE", now_ns, reason)
                else:
                    self._finish(lease, "EVAL_RETRYABLE" if lease.capability == "evaluation" else "RETRYABLE", now_ns, reason)

    def block_infrastructure(self, lease: JobLease, reason: str, now_ns: int) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("block reason is invalid")
        with self._lock:
            self._require_live_lease(lease, now_ns)
            with self._transaction():
                if lease.capability == "final_evaluation":
                    self._finish_final_evaluation(lease, "BLOCKED_INFRA", now_ns, reason)
                else:
                    job = self._job(lease.experiment_id)
                    self._finish(lease, "BLOCKED_INFRA", now_ns, reason)
                    self._reconcile_terminal_rung(job, now_ns)
                    self._materialize_pending_candidates(now_ns)

    def _is_controller_authorized_finalist(self, experiment_id: str) -> bool:
        """Return whether a 2K child carries the complete unseen-20 proof.

        Final evaluation is intentionally not an operator-selected arbitrary
        checkpoint lane.  It may consume only a 2K continuation materialized
        from the controller's immutable 1K promotion record, after its own
        checkpoint and safe promotion-matrix result were verified.
        """
        try:
            job = self._job(experiment_id)
            if job.training.target_step != 2000 or job.admission.get("kind") != "continuation":
                return False
            row = self._connection.execute(
                "SELECT pc.parent_experiment_id FROM promotion_children pc "
                "JOIN promotion_candidates candidate "
                "ON candidate.parent_experiment_id=pc.parent_experiment_id "
                "AND candidate.kind='step_2000' AND candidate.state='ADMITTED' "
                "WHERE pc.experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if row is None or type(row[0]) is not str:
                return False
            parent_id = str(row[0])
            parent = self._job(parent_id)
            if (
                parent.training.target_step != 1000
                or parent.admission.get("kind") != "continuation"
                or job.admission.get("source_experiment_id") != parent_id
            ):
                return False
            # A real 1K rung must itself have been admitted through the
            # immutable promotion tables.  Do not infer lineage from an arm
            # label or publication prefix.
            parent_lineage = self._connection.execute(
                "SELECT 1 FROM promotion_children pc "
                "JOIN promotion_candidates candidate "
                "ON candidate.parent_experiment_id=pc.parent_experiment_id "
                "AND candidate.kind='step_1000' AND candidate.state='ADMITTED' "
                "WHERE pc.experiment_id=?",
                (parent_id,),
            ).fetchone()
            if parent_lineage is None or dict(job.raw) != dict(self._generated_child(parent_id, "step_2000").raw):
                return False
            state = self._connection.execute(
                "SELECT state FROM jobs WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
            if state is None or str(state[0]) != "COMPLETED":
                return False
            publication = self._publication(experiment_id)
            evaluation = self._connection.execute(
                "SELECT report FROM evaluations WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
            if publication is None or evaluation is None or type(evaluation[0]) is not str:
                return False
            parsed = parse_experiment_evaluation(json.loads(str(evaluation[0])))
            if (
                parsed.experiment_id != experiment_id
                or parsed.checkpoint_receipt_sha256 != publication["receipt_sha256"]
                or parsed.policy_digest != publication["artifact_sha256"]
                or parsed.matrix_sha256 != job.evaluation.matrix_sha256
                or parsed.pairing_metrics.get("status") != "available"
                or parsed.safety_failure
                # Promotion evaluation is the frozen balanced unseen-20: five
                # runs in each of the four categories.  Final comparison must
                # never be queued from a different report size.
                or parsed.overall_episodes != 20
                or any(score.episodes != 5 for score in parsed.categories.values())
            ):
                return False
            return True
        except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
            return False

    def _controller_selected_finalist_ids(self) -> tuple[str, ...]:
        """Return the one primary finalist and its explicit tied runner.

        The final queue consumes the durable promotion decision, not a caller
        supplied subset of otherwise plausible 2K checkpoints.  The
        ``tied_runner`` bit is stored both with the candidate and materialized
        child so a partially written or tampered lineage fails closed.
        """
        rows = self._connection.execute(
            "SELECT pc.experiment_id,pc.tied_runner,candidate.tied_runner "
            "FROM promotion_children pc JOIN promotion_candidates candidate "
            "ON candidate.parent_experiment_id=pc.parent_experiment_id "
            "AND candidate.kind='step_2000' AND candidate.state='ADMITTED' "
            "ORDER BY pc.tied_runner,pc.experiment_id"
        ).fetchall()
        if not rows or len(rows) > 2:
            raise ValueError("controller finalist selection is invalid")
        finalists: dict[int, str] = {}
        for experiment_id, child_tied_runner, candidate_tied_runner in rows:
            if (
                type(experiment_id) is not str
                or child_tied_runner not in (0, 1)
                or candidate_tied_runner not in (0, 1)
                or child_tied_runner != candidate_tied_runner
                or int(child_tied_runner) in finalists
            ):
                raise ValueError("controller finalist selection is invalid")
            finalists[int(child_tied_runner)] = str(experiment_id)
        if 0 not in finalists or set(finalists) not in ({0}, {0, 1}):
            raise ValueError("controller finalist selection is invalid")
        return tuple(finalists[index] for index in sorted(finalists))

    def enqueue_finalists(self, experiment_ids: Sequence[str], *, matrix_sha256: str, now_ns: int) -> int:
        """Place the exact controller-selected finalist set on unseen-80.

        The external endpoint repeats the set only as a stale-request guard;
        it cannot cherry-pick a primary, omit its tied runner, or append a
        different continuation.  All inserts happen in one transaction.
        """
        if (
            not isinstance(experiment_ids, Sequence)
            or isinstance(experiment_ids, (str, bytes))
            or not experiment_ids
            or any(type(value) is not str or len(value) != 64 for value in experiment_ids)
            or len(set(experiment_ids)) != len(experiment_ids)
            or type(matrix_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", matrix_sha256) is None
            or type(now_ns) is not int
        ):
            raise ValueError("finalist queue request is invalid")
        with self._lock, self._transaction():
            expected = self._controller_selected_finalist_ids()
            if set(experiment_ids) != set(expected):
                raise ValueError("finalist queue must contain the exact controller-selected set")
            if not all(self._is_controller_authorized_finalist(experiment_id) for experiment_id in expected):
                raise ValueError("controller-selected finalists are not all eligible")
            rows = self._connection.execute(
                "SELECT experiment_id,matrix_sha256 FROM final_evaluations "
                "WHERE experiment_id IN (%s)" % ",".join("?" for _ in expected),
                expected,
            ).fetchall()
            if rows and len(rows) != len(expected):
                raise ValueError("finalist queue is partially enqueued")
            if rows:
                if any(str(row[1]) != matrix_sha256 for row in rows):
                    raise ValueError("finalist is already bound to a different final matrix")
                return 0
            for experiment_id in expected:
                self._connection.execute("INSERT INTO final_evaluations(experiment_id,matrix_sha256,state,report,received_ns) VALUES(?,?, 'READY',NULL,NULL)", (experiment_id, matrix_sha256))
                self._event(experiment_id, "FINAL_EVAL_READY", None, now_ns, matrix_sha256)
            return len(expected)

    def final_evaluation_state(self, experiment_id: str) -> str | None:
        if type(experiment_id) is not str:
            raise ValueError("finalist identity is invalid")
        row = self._connection.execute("SELECT state FROM final_evaluations WHERE experiment_id=?", (experiment_id,)).fetchone()
        return None if row is None else str(row[0])

    def _finish_final_evaluation(self, lease: JobLease, state: str, now_ns: int, detail: str, report: Mapping[str, object] | None = None) -> None:
        if lease.capability != "final_evaluation":
            raise ValueError("final evaluation lease is invalid")
        row = self._connection.execute("SELECT worker_id,capability FROM leases WHERE lease_id=? AND experiment_id=?", (lease.lease_id, lease.experiment_id)).fetchone()
        if row is None or row[0] != lease.worker_id or row[1] != "final_evaluation":
            raise ValueError("invalid final evaluation lease")
        existing = self._connection.execute("SELECT matrix_sha256,state FROM final_evaluations WHERE experiment_id=?", (lease.experiment_id,)).fetchone()
        if existing is None or str(existing[1]) != "LEASED":
            raise ValueError("final evaluation queue state is invalid")
        self._connection.execute("DELETE FROM leases WHERE lease_id=?", (lease.lease_id,))
        self._release_budget(lease.lease_id, now_ns)
        encoded = None if report is None else json.dumps(dict(report), sort_keys=True, separators=(",", ":"))
        self._connection.execute("UPDATE final_evaluations SET state=?,report=?,received_ns=? WHERE experiment_id=?", (state, encoded, now_ns if report is not None else None, lease.experiment_id))
        self._event(lease.experiment_id, "FINAL_EVAL_" + state, lease.worker_id, now_ns, detail)

    def submit_final_evaluation(self, lease: JobLease, report: Mapping[str, object], now_ns: int) -> None:
        """Persist one read-back verified unseen-80 receipt in the final lane."""
        if type(now_ns) is not int:
            raise ValueError("final evaluation timestamp is invalid")
        from lehome_train.groot.experiment_winner import validate_final_unseen80_report

        with self._lock:
            self._require_live_lease(lease, now_ns)
            with self._transaction():
                if lease.capability != "final_evaluation" or lease.publication is None:
                    raise ValueError("final evaluation lease is invalid")
                queue = self._connection.execute("SELECT matrix_sha256 FROM final_evaluations WHERE experiment_id=?", (lease.experiment_id,)).fetchone()
                if queue is None:
                    raise ValueError("final evaluation was not queued")
                publication = parse_checkpoint_publication(lease.publication)
                parsed = validate_final_unseen80_report(report)
                if (
                    parsed["experiment_id"] != lease.experiment_id
                    or parsed["candidate_id"] != lease.experiment_id
                    or parsed["matrix_sha256"] != str(queue[0])
                    or parsed["checkpoint_receipt_sha256"] != publication.receipt_sha256
                    or parsed["policy_digest"] != publication.artifact_sha256
                ):
                    raise ValueError("final evaluation report does not bind queued finalist publication")
                self._finish_final_evaluation(lease, "COMPLETED", now_ns, str(parsed["report_sha256"]), report)

    def final_winner_decision(
        self,
        *,
        baseline_report: Mapping[str, object] | None,
        matrix_sha256: str,
        now_ns: int,
    ) -> dict[str, object]:
        """Compute the winner only from completed separate final-lane receipts."""
        if type(now_ns) is not int:
            raise ValueError("final winner timestamp is invalid")
        from lehome_train.groot.experiment_winner import select_async_final_winner

        with self._lock, self._transaction():
            expected = self._controller_selected_finalist_ids()
            rows = self._connection.execute(
                "SELECT experiment_id,matrix_sha256,state,report FROM final_evaluations "
                "WHERE experiment_id IN (%s) ORDER BY experiment_id"
                % ",".join("?" for _ in expected),
                expected,
            ).fetchall()
            # A late tied-runner receipt or a stale matrix is not a partial
            # comparison: final selection has no winner until the exact
            # controller-selected set is complete on one immutable matrix.
            if (
                len(rows) != len(expected)
                or {str(row[0]) for row in rows} != set(expected)
                or any(str(row[1]) != matrix_sha256 or str(row[2]) != "COMPLETED" or type(row[3]) is not str for row in rows)
            ):
                return {"decision": "finalists_pending"}
            finalists = {str(experiment_id): json.loads(str(encoded)) for experiment_id, _matrix, _state, encoded in rows}
            result = select_async_final_winner(
                finalists,
                baseline_report=baseline_report,
                original_12k_checkpoint_digest=self.original_12k_checkpoint_digest,
                final_matrix_sha256=matrix_sha256,
            )
            if result.get("decision") == "winner":
                self._event(str(result["experiment_id"]), "FINAL_WINNER", None, now_ns, str(result["report_sha256"]))
            return result

    def _validate_evaluation(self, lease: JobLease, report: Mapping[str, object]):
        if lease.capability != "evaluation" or report.get("experiment_id") != lease.experiment_id or lease.publication is None:
            raise ValueError("evaluation experiment mismatch")
        publication = parse_checkpoint_publication(lease.publication)
        job = self._job(lease.experiment_id)
        parsed = parse_experiment_evaluation(report)
        if parsed.experiment_id != lease.experiment_id or parsed.checkpoint_receipt_sha256 != publication.receipt_sha256 or parsed.matrix_sha256 != job.evaluation.matrix_sha256 or parsed.policy_digest != publication.artifact_sha256:
            raise ValueError("evaluation report does not bind lease publication")
        return parsed

    def _score(self, experiment_id: str, report: Mapping[str, object]) -> EvaluationScore:
        parsed = parse_experiment_evaluation(report)
        if parsed.experiment_id != experiment_id:
            raise ValueError("evaluation score experiment is invalid")
        return to_evaluation_score(parsed)

    def _candidate_count(self, kind: str) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM promotion_candidates WHERE kind=? AND state IN ('PENDING','ADMITTED')", (kind,)).fetchone()[0])

    def _candidate(self, parent_id: str, kind: str, now_ns: int, *, tied_runner: bool = False) -> None:
        self._connection.execute("INSERT OR IGNORE INTO promotion_candidates(parent_experiment_id,kind,state,created_ns,tied_runner) VALUES(?,?, 'PENDING', ?,?)", (parent_id, kind, now_ns, int(tied_runner)))
        self._event(parent_id, "PENDING_PROMOTION", None, now_ns, kind)

    def _rung_scores(self, target_step: int) -> list[EvaluationScore]:
        rows = self._connection.execute("SELECT j.experiment_id,e.report FROM jobs j JOIN evaluations e ON e.experiment_id=j.experiment_id WHERE j.state IN ('COMPLETED','PROMOTED')").fetchall()
        scores: list[EvaluationScore] = []
        for experiment_id, encoded in rows:
            job = self._job(str(experiment_id))
            if job.training.target_step == target_step:
                scores.append(self._score(str(experiment_id), json.loads(encoded)))
        return scores

    def _candidate_parents(self, kind: str) -> set[str]:
        return {str(row[0]) for row in self._connection.execute("SELECT parent_experiment_id FROM promotion_candidates WHERE kind=? AND state IN ('PENDING','ADMITTED')", (kind,)).fetchall()}

    def _ranked_initial_500(self, current: EvaluationScore | None = None) -> list[EvaluationScore]:
        values: list[EvaluationScore] = []
        for score in self._rung_scores(500):
            job = self._job(score.experiment_id)
            if job.admission.get("kind") != "seed_repeat" and self.state(score.experiment_id) in {"COMPLETED", "PROMOTED"}:
                values.append(score)
        if current is not None:
            current_job = self._job(current.experiment_id)
            if current_job.admission.get("kind") != "seed_repeat":
                values.append(current)
        return sorted((score for score in values if not score.safety_failure), key=rank_key, reverse=True)

    def _admit_ranked_initial_candidates(self, now_ns: int, current: EvaluationScore | None = None) -> None:
        scores = self._ranked_initial_500(current)
        # The first two valid initial results define one immutable seed-check
        # pair.  Both independent repeats are admitted together; mutating the
        # pair later as more arms complete would turn completion order into a
        # hidden experimental variable.
        if len(scores) < 2:
            return
        seed_parents = self._candidate_parents("seed_repeat")
        for score in scores:
            if len(seed_parents) >= 2:
                break
            if score.experiment_id not in seed_parents:
                self._candidate(score.experiment_id, "seed_repeat", now_ns)
                seed_parents.add(score.experiment_id)

        # The second seed is independent 500-step evidence, not a dashboard
        # side run.  Do not select a 1K continuation until both reports can
        # confirm the initial ordering or demonstrate a reversal.
        required = self._seed_repeat_guarded_parents(seed_parents)
        if required is None:
            return
        desired_1k = max(min(3, len(scores) // 2), len(required))
        selected = self._candidate_parents("step_1000")
        ordered = [score for score in scores if score.experiment_id in required]
        # A stable repeat excludes the losing member of the fixed seed pair;
        # a reversal retains both.  Other arms still compete asynchronously
        # for remaining ASHA slots, so this guard does not introduce a wave.
        ordered.extend(
            score
            for score in scores
            if score.experiment_id not in required and score.experiment_id not in seed_parents
        )
        for score in ordered:
            if len(selected) >= desired_1k:
                break
            if score.experiment_id not in selected:
                self._candidate(score.experiment_id, "step_1000", now_ns)
                selected.add(score.experiment_id)

    def _seed_repeat_guarded_parents(self, seed_parents: set[str]) -> set[str] | None:
        """Return the 1K parents permitted by the immutable repeat check.

        ``None`` means one or both repeats have not produced terminal,
        authenticated 500-step evidence yet.  Once both reports exist, the
        original and repeat rankings either agree (retain the original leader)
        or reverse (retain both), as specified for the sweep's lucky-seed
        guard.
        """
        if len(seed_parents) != 2:
            return None
        initial: dict[str, EvaluationScore] = {}
        repeated: dict[str, EvaluationScore] = {}
        failed_repeat = False
        for parent_id in sorted(seed_parents):
            parent_row = self._connection.execute(
                "SELECT report FROM evaluations WHERE experiment_id=?", (parent_id,)
            ).fetchone()
            child_rows = self._connection.execute(
                "SELECT pc.experiment_id,e.report,j.state FROM promotion_children pc "
                "JOIN jobs j ON j.experiment_id=pc.experiment_id "
                "LEFT JOIN evaluations e ON e.experiment_id=pc.experiment_id "
                "WHERE pc.parent_experiment_id=?",
                (parent_id,),
            ).fetchall()
            repeats = [
                (str(experiment_id), report, str(state))
                for experiment_id, report, state in child_rows
                if self._seed_repeat_matches(self._job(parent_id), self._job(str(experiment_id)), parent_id)
            ]
            if (
                parent_row is None
                or type(parent_row[0]) is not str
                or len(repeats) != 1
            ):
                return None
            initial[parent_id] = self._score(parent_id, json.loads(str(parent_row[0])))
            repeat_id, repeat_report, repeat_state = repeats[0]
            if type(repeat_report) is str:
                score = self._score(repeat_id, json.loads(repeat_report))
                if score.safety_failure:
                    failed_repeat = True
                else:
                    repeated[parent_id] = score
            elif repeat_state in {"REJECTED", "BLOCKED_INFRA"}:
                failed_repeat = True
            else:
                return None
        # A terminally unsafe or infrastructure-blocked repeat invalidates the
        # fixed pair, but it must not freeze other independently scored arms.
        # Returning an empty guard releases those arms while excluding both
        # members of the failed seed pair from 1K selection.
        if failed_repeat:
            return set()
        initial_leader = max(initial, key=lambda parent_id: rank_key(initial[parent_id]))
        repeat_leader = max(repeated, key=lambda parent_id: rank_key(repeated[parent_id]))
        return set(seed_parents) if initial_leader != repeat_leader else {initial_leader}

    def _all_admitted_one_k_terminal(self, current_id: str | None = None) -> bool:
        """Close the 2K rung only after the six-result initial field closes.

        The 2/4/6 ASHA closures intentionally let 1K continuations start
        early.  Those early results are not, however, a complete finalist
        field: a later initial arm can still claim the third continuation
        slot.  The 2K comparison is therefore meaningful only after six valid
        initial evaluations have selected all three continuations.
        """
        if len(self._ranked_initial_500()) < 6:
            return False
        rows = self._connection.execute("SELECT pc.experiment_id,j.state FROM promotion_children pc JOIN jobs j ON j.experiment_id=pc.experiment_id WHERE j.capability='training'").fetchall()
        admitted = [(str(experiment_id), str(state)) for experiment_id, state in rows if self._job(str(experiment_id)).training.target_step == 1000]
        if len(admitted) != 3:
            return False
        terminal = {"COMPLETED", "PROMOTED", "REJECTED", "BLOCKED_INFRA"}
        return all(experiment_id == current_id or state in terminal for experiment_id, state in admitted)

    def _admit_closed_two_k_field(self, current: EvaluationScore | None, now_ns: int) -> None:
        """Select finalists when the last admitted 1K reaches any terminal result.

        The current report has not been persisted yet, so a safe result is
        added explicitly.  A safety failure closes the field but is never
        included in finalist ranking.
        """
        if not self._all_admitted_one_k_terminal(None if current is None else current.experiment_id):
            return
        scores = self._rung_scores(1000)
        if current is not None and not current.safety_failure:
            scores.append(current)
        for index, finalist in enumerate(select_2k_finalists(scores)):
            self._candidate(finalist.experiment_id, "step_2000", now_ns, tied_runner=index == 1)

    def _reconcile_terminal_rung(self, job: ExperimentJob, now_ns: int) -> None:
        """Advance unaffected candidates after a terminal non-report outcome."""
        if job.training.target_step == 500:
            self._admit_ranked_initial_candidates(now_ns)
        elif job.training.target_step == 1000:
            self._admit_closed_two_k_field(None, now_ns)

    def _admit_async_candidates(self, job: ExperimentJob, score: EvaluationScore, report: Mapping[str, object], now_ns: int) -> str:
        if job.training.target_step == 1000:
            # Closing the rung is a terminal-event concern, not a successful-
            # score concern.  A safety-rejected last reporter must still let
            # the remaining safe 1K policies advance.
            self._admit_closed_two_k_field(score, now_ns)
        # A repeat result may be the second half of the fixed ranking check,
        # so it must re-run admission even though it is not itself a candidate
        # for 1K.  ``_ranked_initial_500`` excludes it from the ranking.
        if job.training.target_step == 500:
            self._admit_ranked_initial_candidates(now_ns, score)
        if score.safety_failure:
            return "REJECTED"
        return "COMPLETED"

    @staticmethod
    def _seed_repeat_matches(parent: ExperimentJob, child: ExperimentJob, parent_experiment_id: str) -> bool:
        if child.admission.get("kind") != "seed_repeat" or child.admission.get("source_experiment_id") != parent_experiment_id:
            return False
        if child.training.target_step != 500 or parent.training.target_step != 500 or child.training.seed == parent.training.seed:
            return False
        if child.training.action_horizon != parent.training.action_horizon or child.training.batch_size != parent.training.batch_size or child.training.save_steps != parent.training.save_steps:
            return False
        if dict(child.parent_checkpoint) != dict(parent.parent_checkpoint) or dict(child.trainer) != dict(parent.trainer) or child.mixture != parent.mixture or child.evaluation != parent.evaluation:
            return False
        parent_sources = tuple(item for item in parent.data_sources if item.kind != "runtime_request_set")
        child_sources = tuple(item for item in child.data_sources if item.kind != "runtime_request_set")
        return parent_sources == child_sources and sum(item.kind == "runtime_request_set" for item in parent.data_sources) == sum(item.kind == "runtime_request_set" for item in child.data_sources)

    def _generated_child(self, parent_id: str, kind: str) -> ExperimentJob:
        """Pure deterministic materialization; no Hub/API call is needed here."""
        parent = self._job(parent_id)
        document = json.loads(json.dumps(dict(parent.raw)))
        document.pop("experiment_id", None)
        recovery_dependencies = self._recovery_dependency_digests(parent)
        if any(item.kind == "recovery" for item in parent.data_sources) and not recovery_dependencies:
            raise RuntimeError("automatic recovery continuation lacks verified source receipt")
        document["dependencies"] = [*recovery_dependencies, parent_id]
        if kind == "seed_repeat":
            seed = (parent.training.seed + int(parent_id[:8], 16)) % 2_147_483_647
            if seed == parent.training.seed:
                seed = (seed + 1) % 2_147_483_647
            document["arm"] = parent.arm + "-seed-repeat"
            document["training"]["seed"] = seed
            document["admission"] = {"kind": "seed_repeat", "source_experiment_id": parent_id}
            document["publication"]["prefix"] = parent.publication.prefix + "-seed-" + str(seed)
        else:
            target = {"step_1000": 1000, "step_2000": 2000}[kind]
            publication = self._publication(parent_id)
            if publication is None:
                raise RuntimeError("automatic continuation lacks parent readback publication")
            document["arm"] = parent.arm + "-" + str(target)
            document["training"]["target_step"] = target
            document["parent_checkpoint"] = {"repository": publication["repository"], "revision": publication["immutable_revision"], "subpath": publication["remote_prefix"], "artifact_sha256": publication["artifact_sha256"], "receipt_sha256": publication["receipt_sha256"]}
            document["admission"] = {"kind": "continuation", "source_experiment_id": parent_id}
            document["publication"]["prefix"] = parent.publication.prefix + "-to-" + str(target)
        document["experiment_id"] = experiment_identity(document)
        return _parse(document)

    def _materialize_pending_candidates(self, now_ns: int) -> None:
        rows = self._connection.execute("SELECT parent_experiment_id,kind FROM promotion_candidates WHERE state='PENDING' ORDER BY created_ns,parent_experiment_id,kind").fetchall()
        for parent_id, kind in rows:
            self.promote(str(parent_id), self._generated_child(str(parent_id), str(kind)), now_ns)

    def reconcile_pending_candidates(self, now_ns: int) -> None:
        """Restart repair for a process killed before an older controller commit."""
        with self._lock, self._transaction():
            self._materialize_pending_candidates(now_ns)

    def submit_evaluation(self, lease: JobLease, report: Mapping[str, object], now_ns: int) -> None:
        if type(now_ns) is not int:
            raise ValueError("evaluation timestamp is invalid")
        with self._lock:
            self._require_live_lease(lease, now_ns)
            with self._transaction():
                parsed = self._validate_evaluation(lease, report)
                row = self._connection.execute("SELECT verified FROM artifacts WHERE experiment_id=?", (lease.experiment_id,)).fetchone()
                if row is None or row[0] != 1:
                    raise ValueError("evaluation requires publication readback")
                if parsed.pairing_metrics.get("status") == "baseline_evaluation_required":
                    # A missing original-12K pairing is not a neutral score and not
                    # a safety failure.  Preserve the authenticated report for
                    # audit, release the GPU lease, and wait for the evaluator to
                    # supply the exact shared-matrix baseline evidence.
                    self._finish(lease, "EVAL_WAITING_BASELINE", now_ns, "baseline_evaluation_required")
                    self._connection.execute("INSERT OR REPLACE INTO evaluations VALUES(?,?,?)", (lease.experiment_id, json.dumps(dict(report), sort_keys=True, separators=(",", ":")), now_ns))
                    return
                job = self._job(lease.experiment_id)
                score = self._score(lease.experiment_id, report)
                # The current report is part of a seed-pair decision.  Persist it
                # before admission, within the same transaction, so a second
                # repeat finishing out of order sees both authenticated reports.
                self._connection.execute("INSERT OR REPLACE INTO evaluations VALUES(?,?,?)", (lease.experiment_id, json.dumps(dict(report), sort_keys=True, separators=(",", ":")), now_ns))
                state = self._admit_async_candidates(job, score, report, now_ns)
                self._finish(lease, state, now_ns, "evaluation submitted")
                self._materialize_pending_candidates(now_ns)

    def promote(self, parent_experiment_id: str, child: ExperimentJob, now_ns: int) -> None:
        with self._lock, self._maybe_transaction():
            parent = self._connection.execute("SELECT receipt_sha256 FROM jobs WHERE experiment_id=? AND state IN ('COMPLETED','PROMOTED')", (parent_experiment_id,)).fetchone()
            evaluation = self._connection.execute("SELECT 1 FROM evaluations WHERE experiment_id=?", (parent_experiment_id,)).fetchone()
            if parent is None or evaluation is None or parent_experiment_id not in child.dependencies:
                raise ValueError("promotion parent is not fully verified")
            parent_job = self._job(parent_experiment_id)
            is_seed_repeat = child.admission.get("kind") == "seed_repeat"
            kind = "seed_repeat" if is_seed_repeat else {1000: "step_1000", 2000: "step_2000"}.get(child.training.target_step)
            if kind is None:
                raise ValueError("promotion target rung is invalid")
            candidate = self._connection.execute("SELECT state,tied_runner FROM promotion_candidates WHERE parent_experiment_id=? AND kind=?", (parent_experiment_id, kind)).fetchone()
            if candidate is None or candidate[0] != "PENDING":
                raise ValueError("promotion has no pending, controller-approved admission")
            if child.evaluation != parent_job.evaluation:
                raise ValueError("promoted child does not preserve the original baseline evaluation binding")
            if is_seed_repeat:
                if not self._seed_repeat_matches(parent_job, child, parent_experiment_id):
                    raise ValueError("seed repeat must preserve the original baseline configuration")
            elif child.admission.get("kind") != "continuation" or child.admission.get("source_experiment_id") != parent_experiment_id or child.training.target_step <= parent_job.training.target_step:
                raise ValueError("continuation rung does not advance parent")
            if not is_seed_repeat:
                publication = self._publication(parent_experiment_id)
                if publication is None:
                    raise ValueError("promotion parent publication is not readback verified")
                expected_parent = {
                    "repository": publication["repository"],
                    "revision": publication["immutable_revision"],
                    "subpath": publication["remote_prefix"],
                    "artifact_sha256": publication["artifact_sha256"],
                    "receipt_sha256": publication["receipt_sha256"],
                }
                if dict(child.parent_checkpoint) != expected_parent:
                    raise ValueError("promoted child does not exact-bind parent readback publication")
            expected_child = self._generated_child(parent_experiment_id, kind)
            if dict(child.raw) != dict(expected_child.raw):
                raise ValueError("promoted child is not controller-generated")
            order = self._connection.execute("SELECT COALESCE(MAX(created_order), -1) + 1 FROM jobs").fetchone()[0]
            canonical = json.dumps(dict(child.raw), sort_keys=True, separators=(",", ":"))
            try:
                self._connection.execute("INSERT INTO jobs VALUES(?,?,?,?,?,NULL)", (child.experiment_id, canonical, self._initial_state(child), "training", order))
            except sqlite3.IntegrityError as error:
                raise ValueError("duplicate promoted child") from error
            self._jobs[child.experiment_id] = child
            for dependency in child.dependencies:
                self._connection.execute("INSERT INTO dependencies VALUES(?,?)", (child.experiment_id, dependency))
            self._connection.execute("INSERT INTO promotion_children(experiment_id,parent_experiment_id,tied_runner) VALUES(?,?,?)", (child.experiment_id, parent_experiment_id, candidate[1]))
            self._connection.execute("UPDATE promotion_candidates SET state='ADMITTED' WHERE parent_experiment_id=? AND kind=?", (parent_experiment_id, kind))
            self._event(parent_experiment_id, "PROMOTED", None, now_ns, child.experiment_id)
            self._connection.execute("UPDATE jobs SET state='PROMOTED' WHERE experiment_id=?", (parent_experiment_id,))
            self._event(child.experiment_id, self._initial_state(child), None, now_ns, parent_experiment_id)

    def state(self, experiment_id: str) -> str:
        row = self._connection.execute("SELECT state FROM jobs WHERE experiment_id=?", (experiment_id,)).fetchone()
        if row is None:
            raise ValueError("unknown experiment")
        return str(row[0])

    def close(self) -> None:
        self._connection.close()
