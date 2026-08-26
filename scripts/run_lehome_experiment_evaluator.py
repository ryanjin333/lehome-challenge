#!/usr/bin/env python3
"""Serial, one-policy-at-a-time asynchronous evaluation adapter."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import time
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
from lehome_train.groot.experiment_worker import LeaseHeartbeatGuard, is_retryable_transport, run_subprocess_cancellable, run_with_cancellation
from lehome_train.groot.experiment_deployment_gate import PRODUCTION_EVALUATOR_WORKER_ID


_EVALUATION_CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")
_EVALUATION_ROW_KEYS = {"trial_id", "category", "garment_name", "release_stage", "seed"}


def validate_frozen_evaluation_matrix(path: Path, expected_sha256: str, mode: str) -> list[dict[str, object]]:
    """Fail before controller construction, and therefore before GPU lease."""
    expected_count = 20 if mode == "promotion" else 80
    if mode not in {"promotion", "final-unseen80"} or not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("frozen evaluation matrix path or mode is invalid")
    try:
        payload = path.read_bytes()
    except OSError:
        raise ValueError("frozen evaluation matrix is unreadable") from None
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256) or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("frozen evaluation matrix SHA-256 mismatch")
    try:
        rows = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("frozen evaluation matrix is invalid JSON") from None
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"frozen evaluation matrix must contain exactly {expected_count} trials")
    trial_ids: set[str] = set()
    identities: set[tuple[object, ...]] = set()
    normalized: list[dict[str, object]] = []
    for value in rows:
        if not isinstance(value, dict) or set(value) != _EVALUATION_ROW_KEYS:
            raise ValueError("frozen evaluation trial schema is invalid")
        trial_id, category = value.get("trial_id"), value.get("category")
        garment, stage, seed = value.get("garment_name"), value.get("release_stage"), value.get("seed")
        if type(trial_id) is not str or not trial_id or category not in _EVALUATION_CATEGORIES or type(garment) is not str or not garment or stage != "public_unseen" or type(seed) is not int or seed < 0:
            raise ValueError("frozen evaluation trial identity is invalid")
        identity = (category, garment, seed)
        if trial_id in trial_ids or identity in identities:
            raise ValueError("frozen evaluation matrix contains duplicate trials")
        trial_ids.add(trial_id); identities.add(identity); normalized.append(value)
    expected_per_category = expected_count // len(_EVALUATION_CATEGORIES)
    if Counter(row["category"] for row in normalized) != Counter({category: expected_per_category for category in _EVALUATION_CATEGORIES}):
        raise ValueError("frozen evaluation matrix is not category-balanced")
    return normalized


def load_finalist_seen_regression_handoff(
    root: Path,
    experiment_id: str,
    checkpoint_receipt_sha256: str,
) -> dict[str, object]:
    """Read only the descriptor bound to the currently leased finalist.

    Finalists do not exist at image/bootstrap time.  A later seen-evaluation
    producer must atomically materialize one immutable descriptor at
    ``<root>/<experiment_id>/<checkpoint_receipt>.json``.  The final evaluator
    never accepts a shared campaign-wide seen-regression receipt.
    """
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or (root.stat().st_mode & 0o777) != 0o555
        or len(experiment_id) != 64
        or len(checkpoint_receipt_sha256) != 64
        or any(character not in "0123456789abcdef" for character in experiment_id + checkpoint_receipt_sha256)
    ):
        raise ValueError("finalist seen-regression handoff root or identity is unsafe")
    finalist_root = root / experiment_id
    descriptor_path = finalist_root / f"{checkpoint_receipt_sha256}.json"
    if (
        finalist_root.is_symlink()
        or not finalist_root.is_dir()
        or descriptor_path.is_symlink()
        or not descriptor_path.is_file()
        or (descriptor_path.stat().st_mode & 0o777) != 0o444
    ):
        raise ValueError("finalist seen-regression handoff is missing or unsafe")
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("finalist seen-regression handoff is invalid") from error
    required = {
        "schema_version", "kind", "experiment_id", "checkpoint_receipt_sha256",
        "evidence", "evidence_sha256",
    }
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != required
        or descriptor.get("schema_version") != 1
        or descriptor.get("kind") != "lehome_finalist_seen_regression_handoff"
        or descriptor.get("experiment_id") != experiment_id
        or descriptor.get("checkpoint_receipt_sha256") != checkpoint_receipt_sha256
        or not isinstance(descriptor.get("evidence"), dict)
    ):
        raise ValueError("finalist seen-regression handoff does not bind the leased finalist")
    evidence = dict(descriptor["evidence"])
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    if (
        descriptor.get("evidence_sha256") != hashlib.sha256(encoded).hexdigest()
        or evidence.get("candidate_checkpoint_receipt_sha256") != checkpoint_receipt_sha256
    ):
        raise ValueError("finalist seen-regression handoff evidence is tampered or misbound")
    return evidence

def run_evaluation_loop(controller: Any, adapter: Any, *, matrix: str, matrix_sha256: str, max_jobs: int | None = None, idle_timeout_seconds: float = 600.0, poll_seconds: float = 2.0, evaluation_capability: str = "evaluation") -> int:
    if len(matrix_sha256) != 64: raise ValueError("matrix digest is invalid")
    if evaluation_capability not in {"evaluation", "final_evaluation"}: raise ValueError("evaluator capability is invalid")
    if idle_timeout_seconds < 0 or poll_seconds <= 0: raise ValueError("evaluator idle configuration is invalid")
    count = 0
    leased = 0
    idle_since = time.monotonic()
    while max_jobs is None or leased < max_jobs:
        lease = controller.lease_next(
            PRODUCTION_EVALUATOR_WORKER_ID,
            evaluation_capability,
            now_ns=time.time_ns(),
            lease_ns=60_000_000_000,
        )
        if lease is None:
            if time.monotonic() - idle_since >= idle_timeout_seconds: return count
            time.sleep(min(poll_seconds, max(0.0, idle_timeout_seconds - (time.monotonic() - idle_since))))
            continue
        idle_since = time.monotonic()
        leased += 1
        if getattr(lease, "evaluation_matrix_sha256", None) != matrix_sha256:
            controller.block_infrastructure(lease, "evaluation_matrix_mismatch", time.time_ns())
            continue
        try:
            with LeaseHeartbeatGuard(controller, lease) as heartbeat:
                def campaign(_: object, *, cancellation: threading.Event | None = None):
                    return adapter.run(lease, matrix, matrix_sha256, 4, cancellation=cancellation)
                report = run_with_cancellation(campaign, None, heartbeat.cancelled)
                heartbeat.assert_owned()
            if not isinstance(report, dict) or report.get("experiment_id") != lease.experiment_id or report.get("matrix_sha256") != matrix_sha256: raise ValueError("evaluation report binding mismatch")
            heartbeat.assert_owned()
            if evaluation_capability == "final_evaluation":
                submit_final = getattr(controller, "submit_final_evaluation", None)
                if callable(submit_final):
                    submit_final(lease, report, time.time_ns())
                elif hasattr(controller, "_post"):
                    controller._post("/final-evaluation", {"lease_id": lease.lease_id, "experiment_id": lease.experiment_id, "worker_id": lease.worker_id, "report": report, "now_ns": time.time_ns()})
                else:
                    raise ValueError("controller does not support final evaluation submission")
            else:
                controller.submit_evaluation(lease, report, time.time_ns())
            count += 1
        except ValueError:
            # Policy outcomes are submitted as reports; malformed identities are deterministic infrastructure failures.
            if hasattr(controller, "block_infrastructure"): controller.block_infrastructure(lease, "evaluation_identity", time.time_ns())
            else: raise
        except Exception as error:
            if is_retryable_transport(error):
                if hasattr(controller, "retryable"): controller.retryable(lease, type(error).__name__, time.time_ns())
                else: raise
            elif hasattr(controller, "block_infrastructure"):
                controller.block_infrastructure(lease, type(error).__name__, time.time_ns())
            else: raise
    return count

class PersistentFourWorkerAdapter:
    """Run exactly one checkpoint through the existing persistent appliance."""
    def __init__(self, *, campaign_script: str = "/opt/lehome/rollout_appliance/run_12k_campaign.sh", campaign_root: Path, runner: Any = subprocess.run, summarizer: Any, mode: str = "promotion", baseline_policy: Mapping[str, object] | None = None, baseline_evidence_path: Path | None = None, seen_regression_handoff_root: Path | None = None, final_report_transport: Any | None = None, final_report_repository: str | None = None, final_report_prefix: str = "") -> None:
        if mode not in {"promotion", "final-unseen80"}:
            raise ValueError("evaluation mode is invalid")
        self.campaign_script, self.campaign_root, self.runner, self.summarizer = campaign_script, campaign_root, runner, summarizer
        self.mode = mode
        self.baseline_policy = None if baseline_policy is None else dict(baseline_policy)
        self.baseline_evidence_path = baseline_evidence_path
        self.seen_regression_handoff_root = seen_regression_handoff_root
        self.final_report_transport = final_report_transport
        self.final_report_repository = final_report_repository
        self.final_report_prefix = final_report_prefix

    @staticmethod
    def _read_json(path: Path, label: str) -> dict[str, object]:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} is missing or unsafe")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"{label} is invalid") from error
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an object")
        return value

    def _run_campaign(self, *, root: Path, rows: list[object], matrix: str, matrix_sha256: str, policy: Mapping[str, object], run_id: str, cancellation: threading.Event | None, evaluation_terminal_upload: bool) -> None:
        required = {"repository", "immutable_revision", "target_step", "artifact_sha256"}
        if set(policy) != required or type(policy.get("repository")) is not str or type(policy.get("immutable_revision")) is not str or type(policy.get("target_step")) is not int or type(policy.get("artifact_sha256")) is not str:
            raise ValueError("evaluation policy identity is invalid")
        env = dict(
            os.environ,
            LEHOME_WORKER_COUNT="4",
            LEHOME_MAX_ATTEMPTS=str(len(rows)),
            LEHOME_TARGET_ACCEPTED=str(len(rows)),
            LEHOME_ENABLE_HF_UPLOAD="1" if evaluation_terminal_upload else "0",
            LEHOME_EVALUATION_TERMINAL_UPLOAD="1" if evaluation_terminal_upload else "0",
            LEHOME_SKIP_ROUND_SEAL="0",
            LEHOME_CONTROLLED_RECOVERY_SMOKE="0",
            LEHOME_SNAPSHOT_SOURCE_BOOTSTRAP="0",
            LEHOME_RESUME_PREEMPTED_ROLLOUT="0",
            LEHOME_ATTEMPT_MATRIX=matrix,
            LEHOME_ATTEMPT_MATRIX_SHA256=matrix_sha256,
            LEHOME_CAMPAIGN_ROOT=str(root),
            LEHOME_RUN_ID=run_id,
            LEHOME_ROUND_ID="experiment-evaluation-" + run_id,
            LEHOME_POLICY_REPO=str(policy["repository"]),
            LEHOME_POLICY_REVISION=str(policy["immutable_revision"]),
            LEHOME_POLICY_STEP=str(policy["target_step"]),
            LEHOME_POLICY_ARTIFACT_SHA256=str(policy["artifact_sha256"]),
        )
        if evaluation_terminal_upload:
            # The policy server still owns CUDA; only cloth simulation is CPU.
            env["LEHOME_SIMULATOR_DEVICE"] = "cpu"
        if cancellation is not None and cancellation.is_set():
            raise RuntimeError("evaluation cancelled before campaign start")
        if cancellation is not None and self.runner is subprocess.run:
            run_subprocess_cancellable([self.campaign_script], env=env, cancellation=cancellation)
        else:
            self.runner([self.campaign_script], env=env, check=True)
        if cancellation is not None and cancellation.is_set():
            raise RuntimeError("evaluation cancelled after campaign")

    def _baseline_evidence(self, *, rows: list[object], matrix: str, matrix_sha256: str, expected_policy_digest: str, cancellation: threading.Event | None) -> Mapping[str, object] | None:
        """Reuse a verified same-matrix baseline, otherwise collect it once."""
        if self.mode != "promotion":
            return None
        if self.baseline_evidence_path is not None and self.baseline_evidence_path.exists():
            evidence = self._read_json(self.baseline_evidence_path, "paired baseline evidence")
            if evidence.get("policy_digest") != expected_policy_digest:
                raise ValueError("paired baseline evidence does not bind the pinned original parent")
            return evidence
        if self.baseline_policy is None:
            # The summarizer emits this typed state and the controller parks
            # the evaluation rather than treating it as a zero paired score.
            return None
        if self.baseline_policy.get("artifact_sha256") != expected_policy_digest:
            raise ValueError("baseline policy does not bind the pinned original parent")
        root = self.campaign_root / "paired-baselines" / matrix_sha256 / str(self.baseline_policy["artifact_sha256"])
        self._run_campaign(root=root, rows=rows, matrix=matrix, matrix_sha256=matrix_sha256, policy=self.baseline_policy, run_id="baseline-" + matrix_sha256[:16], cancellation=cancellation, evaluation_terminal_upload=True)
        from scripts.summarize_groot_persistent_evaluation import build_paired_baseline_evidence
        evidence = build_paired_baseline_evidence(
            campaign_root=root, matrix_path=Path(matrix), matrix_sha256=matrix_sha256,
            policy_repo=str(self.baseline_policy["repository"]), policy_revision=str(self.baseline_policy["immutable_revision"]),
            policy_step=int(self.baseline_policy["target_step"]), policy_artifact_sha256=str(self.baseline_policy["artifact_sha256"]),
        )
        if self.baseline_evidence_path is not None:
            target = self.baseline_evidence_path
            if target.exists() or target.is_symlink():
                raise ValueError("paired baseline evidence output already exists")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            target.chmod(0o444)
        return evidence
    def run(self, lease: Any, matrix: str, matrix_sha256: str, workers: int, *, cancellation: threading.Event | None = None) -> dict[str, object]:
        if workers != 4: raise ValueError("exactly four persistent workers are required")
        rows = json.loads(Path(matrix).read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows: raise ValueError("frozen matrix is invalid")
        job = lease.job
        publication = getattr(lease, "publication", None)
        if not isinstance(publication, dict) or publication.get("readback_verified") is not True: raise ValueError("evaluation requires verified terminal publication")
        seen_regression_evidence = None
        if self.mode == "final-unseen80":
            receipt = publication.get("receipt_sha256")
            if self.seen_regression_handoff_root is None or type(receipt) is not str:
                raise ValueError("final evaluation requires a finalist-specific seen-regression handoff root")
            seen_regression_evidence = load_finalist_seen_regression_handoff(
                self.seen_regression_handoff_root, lease.experiment_id, receipt,
            )
        evaluation = getattr(job, "evaluation", None)
        expected_baseline_policy = getattr(evaluation, "policy_digest", None)
        if not isinstance(expected_baseline_policy, str) or len(expected_baseline_policy) != 64:
            raise ValueError("evaluation job has no pinned original baseline policy")
        baseline_evidence = self._baseline_evidence(rows=rows, matrix=matrix, matrix_sha256=matrix_sha256, expected_policy_digest=expected_baseline_policy, cancellation=cancellation)
        root = self.campaign_root / lease.experiment_id
        policy = {"repository": publication["repository"], "immutable_revision": publication["immutable_revision"], "target_step": publication["target_step"], "artifact_sha256": publication["artifact_sha256"]}
        self._run_campaign(root=root, rows=rows, matrix=matrix, matrix_sha256=matrix_sha256, policy=policy, run_id=lease.experiment_id, cancellation=cancellation, evaluation_terminal_upload=True)
        if self.mode == "final-unseen80" and (self.final_report_transport is None or not isinstance(self.final_report_repository, str) or not self.final_report_repository):
            raise ValueError("final evaluation requires an injected Hugging Face report transport")
        report_path = Path(self.summarizer(experiment_job=job, publication=publication, campaign_root=root, matrix=Path(matrix), matrix_sha256=matrix_sha256, baseline_evidence=baseline_evidence, mode=self.mode, candidate_id=lease.experiment_id, seen_regression_evidence=seen_regression_evidence, final_report_transport=self.final_report_transport, final_report_repository=self.final_report_repository, final_report_path=self.final_report_prefix + lease.experiment_id + ".json"))
        sidecar = report_path.with_suffix(".json.sha256")
        if not report_path.is_file() or not sidecar.is_file(): raise ValueError("strict evaluation report or sidecar is missing")
        if self.mode == "final-unseen80":
            from lehome_train.groot.experiment_winner import validate_final_unseen80_report
            strict_final = validate_final_unseen80_report(json.loads(report_path.read_text(encoding="utf-8")))
            if strict_final["experiment_id"] != lease.experiment_id or strict_final["checkpoint_receipt_sha256"] != publication["receipt_sha256"] or strict_final["matrix_sha256"] != matrix_sha256 or strict_final["policy_digest"] != publication["artifact_sha256"]:
                raise ValueError("final evaluation report does not bind lease publication")
            return json.loads(report_path.read_text(encoding="utf-8"))
        from lehome_train.groot.experiment_evaluation import load_experiment_evaluation
        strict = load_experiment_evaluation(report_path)
        if strict.experiment_id != lease.experiment_id or strict.checkpoint_receipt_sha256 != publication["receipt_sha256"] or strict.matrix_sha256 != matrix_sha256 or strict.policy_digest != publication["artifact_sha256"]:
            raise ValueError("strict evaluation report does not bind lease publication")
        trial_ids = {row.get("trial_id") for row in rows if isinstance(row, dict)}
        if trial_ids != {row["trial_id"] for row in strict.episode_artifacts}:
            raise ValueError("strict evaluation report does not bind frozen episode identities")
        expected_trainer = getattr(job, "trainer", None)
        expected_sources = getattr(job, "data_sources", None)
        if not isinstance(expected_trainer, Mapping) or not isinstance(expected_sources, tuple):
            raise ValueError("evaluation job has no immutable provenance bindings")
        sources = [{"kind": source.kind, "repository": source.repository, "revision": source.revision, "prefix": source.prefix, "manifest_sha256": source.manifest_sha256, "tree_sha256": source.tree_sha256} for source in expected_sources]
        if dict(strict.provenance["trainer"]) != dict(expected_trainer) or [dict(source) for source in strict.provenance["data_sources"]] != sources:
            raise ValueError("strict evaluation report provenance does not bind job")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return report

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--controller-url", required=True); parser.add_argument("--controller-ca-file", type=Path, required=True); parser.add_argument("--matrix", type=Path, required=True); parser.add_argument("--matrix-sha256", required=True); parser.add_argument("--manifest-set-sha256", required=True); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--token-file", type=Path, required=True); parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("promotion", "final-unseen80"), default="promotion")
    parser.add_argument("--baseline-policy", type=Path)
    parser.add_argument("--baseline-evidence", type=Path)
    parser.add_argument("--seen-regression-handoff-root", type=Path)
    parser.add_argument("--final-report-repository")
    parser.add_argument("--final-hf-token-file", type=Path)
    parser.add_argument("--final-report-prefix", default="final-unseen80/")
    args = parser.parse_args(argv)
    if args.workers != 4 or not args.matrix.is_absolute() or not args.campaign_root.is_absolute() or args.token_file.is_symlink() or not args.token_file.is_file() or (args.token_file.stat().st_mode & 0o777) != 0o600: raise ValueError("evaluator inputs are unsafe")
    validate_frozen_evaluation_matrix(args.matrix, args.matrix_sha256, args.mode)
    from run_lehome_experiment_worker import HttpControllerClient
    if not args.controller_url.startswith(("http://", "https://")) or len(args.matrix_sha256) != 64 or len(args.manifest_set_sha256) != 64:
        raise ValueError("evaluator controller or matrix identity is invalid")
    controller = HttpControllerClient(args.controller_url, args.token_file, args.manifest_set_sha256, args.controller_ca_file)
    baseline_policy = None
    if args.baseline_policy is not None:
        if args.baseline_policy.is_symlink() or not args.baseline_policy.is_file():
            raise ValueError("baseline policy identity is unsafe")
        baseline_policy = json.loads(args.baseline_policy.read_text(encoding="utf-8"))
        if not isinstance(baseline_policy, dict):
            raise ValueError("baseline policy identity is invalid")
    final_transport = None
    if args.mode == "final-unseen80":
        if args.final_report_repository is None or args.final_hf_token_file is None or args.seen_regression_handoff_root is None:
            raise ValueError("final evaluation requires Hugging Face publication and a finalist-specific seen-regression handoff root")
        from scripts.summarize_groot_persistent_evaluation import HuggingFaceFinalReportTransport
        final_transport = HuggingFaceFinalReportTransport(args.final_hf_token_file)
    def write_strict_report(*, experiment_job: object, campaign_root: Path, matrix: Path, matrix_sha256: str, publication: dict[str, object], baseline_evidence: Mapping[str, object] | None = None, mode: str = "promotion", candidate_id: str = "", seen_regression_evidence: Mapping[str, object] | None = None, final_report_transport: object | None = None, final_report_repository: object | None = None, final_report_path: object | None = None, **_: object) -> Path:
        from scripts.summarize_groot_persistent_evaluation import build_experiment_report, build_final_unseen80_report, write_experiment_report, write_final_unseen80_report
        if mode == "final-unseen80":
            if seen_regression_evidence is None:
                raise ValueError("final evaluation requires sealed seen-regression evidence")
            if final_report_transport is None or type(final_report_repository) is not str or type(final_report_path) is not str:
                raise ValueError("final evaluation report publication transport is missing")
            return write_final_unseen80_report(
                campaign_root / "final-unseen80-report.json",
                build_final_unseen80_report(experiment_job=experiment_job, checkpoint_publication=publication, campaign_root=campaign_root, matrix_path=matrix, matrix_sha256=matrix_sha256, candidate_id=candidate_id, seen_regression_evidence=seen_regression_evidence),
                transport=final_report_transport,
                repository=final_report_repository,
                remote_path=final_report_path,
            )
        return write_experiment_report(
            campaign_root / "experiment-report.json",
            build_experiment_report(experiment_job=experiment_job, checkpoint_publication=publication, campaign_root=campaign_root, matrix_path=matrix, matrix_sha256=matrix_sha256, baseline_evidence=baseline_evidence),
        )
    adapter = PersistentFourWorkerAdapter(campaign_root=args.campaign_root, summarizer=write_strict_report, mode=args.mode, baseline_policy=baseline_policy, baseline_evidence_path=args.baseline_evidence, seen_regression_handoff_root=args.seen_regression_handoff_root, final_report_transport=final_transport, final_report_repository=args.final_report_repository, final_report_prefix=args.final_report_prefix)
    return run_evaluation_loop(controller, adapter, matrix=str(args.matrix), matrix_sha256=args.matrix_sha256, evaluation_capability="final_evaluation" if args.mode == "final-unseen80" else "evaluation")
if __name__ == "__main__": raise SystemExit(main())
