"""Local-only, receipt-bound lifecycle controller for one capped paid smoke."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from enum import Enum
from typing import Any, Protocol

from .dockerhub import DockerImageRelease
from .huggingface import CheckpointBucket, HubProbeReceipt, HubRepository, ReleaseDestinations
from .vast import CappedVastController, PROTECTED_INSTANCE_IDS


_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_RUN_ID_RE = re.compile(r"^b1k-smoke-([0-9a-f]{32})$")
_PAYLOAD_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MIN_DISK_GB = 100
_MIN_RAM_GB = 16
_MIN_NETWORK_MBPS = 100


def _runtime_operation_id(run_id: str, classification: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id) or classification not in {"smoke-model", "success-fixture", "failure-fixture"}:
        raise SmokeError("runtime artifact requires one current smoke run UUID and artifact kind")
    return f"{run_id}:{classification}"


def _runtime_probe_prefix(run_id: str, classification: str) -> str:
    match = _RUN_ID_RE.fullmatch(run_id)
    if match is None or classification not in {"smoke-model", "success-fixture", "failure-fixture"}:
        raise SmokeError("runtime artifact requires one current smoke run UUID and artifact kind")
    return f"b1k-bootstrap-{match.group(1)}-{classification}"


class SmokeError(ValueError):
    """Raised with controller-owned, credential-free text only."""


class SmokeState(str, Enum):
    PLANNED = "planned"
    RENTED = "rented"
    SSH_READY = "ssh-ready"
    RUNTIME_READY = "runtime-ready"
    READBACK_VERIFIED = "readback-verified"
    DESTROYED = "destroyed"
    DISAPPEARANCE_VERIFIED = "disappearance-verified"
    FAILED = "failed"


@dataclass(frozen=True)
class SmokeTimeouts:
    create_timeout_seconds: int = 30
    reconcile_timeout_seconds: int = 30
    ssh_timeout_seconds: int = 45
    runtime_timeout_seconds: int = 55
    contract_timeout_seconds: int = 55
    destroy_attempt_timeout_seconds: int = 30
    disappearance_timeout_seconds: int = 30
    poll_interval_seconds: int = 5

    def validate(self) -> None:
        for value in (
            self.create_timeout_seconds,
            self.reconcile_timeout_seconds,
            self.destroy_attempt_timeout_seconds,
            self.poll_interval_seconds,
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value >= 60:
                raise ValueError("provider tool timeouts and poll intervals must be positive and under 60 seconds")
        for value in (self.ssh_timeout_seconds, self.runtime_timeout_seconds, self.contract_timeout_seconds, self.disappearance_timeout_seconds):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > 21_600:
                raise ValueError("smoke lifecycle deadlines must be positive and at most six hours")

    def worst_case_seconds(self) -> int:
        # rent can reconcile once, and controller recovery can reconcile in
        # both except and finally.  Runtime also performs one bounded marker
        # SSH after the readiness deadline.
        return self.create_timeout_seconds + (3 * self.reconcile_timeout_seconds) + self.ssh_timeout_seconds + self.runtime_timeout_seconds + min(self.runtime_timeout_seconds, 55) + self.contract_timeout_seconds + (2 * self.destroy_attempt_timeout_seconds) + self.disappearance_timeout_seconds


@dataclass(frozen=True)
class SmokeCompatibility:
    verified_datacenter: bool
    gpu_compatible: bool
    disk_gb: int
    ram_gb: int
    network_mbps: int
    maximum_duration_minutes: int
    selection: str

    def validate(self) -> None:
        if not self.verified_datacenter or not self.gpu_compatible:
            raise SmokeError("smoke offer must be a verified compatible datacenter offer")
        if self.selection != "cheapest-compatible-verified":
            raise SmokeError("smoke offer selection must be cheapest-compatible-verified")
        for value in (self.disk_gb, self.ram_gb, self.network_mbps, self.maximum_duration_minutes):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SmokeError("smoke offer compatibility values must be positive integers")
        if self.disk_gb < _MIN_DISK_GB or self.ram_gb < _MIN_RAM_GB or self.network_mbps < _MIN_NETWORK_MBPS:
            raise SmokeError("smoke offer lacks sufficient disk, RAM, or network capacity")

    def receipt(self) -> dict[str, bool | int | str]:
        return {
            "verified_datacenter": self.verified_datacenter,
            "gpu_compatible": self.gpu_compatible,
            "disk_gb": self.disk_gb,
            "ram_gb": self.ram_gb,
            "network_mbps": self.network_mbps,
            "maximum_duration_minutes": self.maximum_duration_minutes,
            "selection": self.selection,
        }


@dataclass(frozen=True)
class SmokeOfferSelectionReceipt:
    """Provider-selected offer receipt, not a mutable caller-provided mapping."""

    offer_id: str
    hourly_rate_usd: Decimal | str | int | float
    gpu_name: str
    compatibility: SmokeCompatibility

    def ledger_offer(self) -> dict[str, Any]:
        return {
            "offer_id": self.offer_id,
            "hourly_rate_usd": self.hourly_rate_usd,
            "verified": self.compatibility.verified_datacenter,
            "gpu_name": self.gpu_name,
        }


@dataclass(frozen=True)
class SmokeTemplatePublicationReceipt:
    """Read-back publication receipt binding one immutable image to one template."""

    template_id: str
    image_release: DockerImageRelease
    payload_hash: str


@dataclass(frozen=True)
class SmokeArtifactReceipt:
    """Secret-free projection of one immutable private-Hub probe and its deletion proof."""

    classification: str
    repo_id: str
    repo_type: str
    prefix: str
    key: str
    upload_commit: str
    delete_commit: str
    absence_verified: bool

    @classmethod
    def from_hub_probe(cls, classification: str, receipt: HubProbeReceipt) -> "SmokeArtifactReceipt":
        if not isinstance(classification, str) or classification not in {"smoke-model", "success-fixture", "failure-fixture"}:
            raise SmokeError("smoke artifact classification is invalid")
        if not isinstance(receipt, HubProbeReceipt) or receipt.key != f"{receipt.prefix}/probe.json" or not receipt.prefix.startswith("b1k-bootstrap-"):
            raise SmokeError("smoke artifact receipt is not an exact bootstrap probe")
        if not _COMMIT_RE.fullmatch(receipt.upload_commit) or not _COMMIT_RE.fullmatch(receipt.delete_commit):
            raise SmokeError("smoke artifact receipt requires immutable upload and deletion commits")
        return cls(classification, receipt.repo_id, receipt.repo_type, receipt.prefix, receipt.key, receipt.upload_commit, receipt.delete_commit, True)


@dataclass(frozen=True)
class RuntimeArtifactReceipt:
    """An immutable private-Hub artifact produced by this exact rented smoke run."""

    run_id: str
    operation_id: str
    smoke_label: str
    artifact: SmokeArtifactReceipt

    @classmethod
    def from_hub_probe(cls, run_id: str, classification: str, receipt: HubProbeReceipt) -> "RuntimeArtifactReceipt":
        prefix = _runtime_probe_prefix(run_id, classification)
        if receipt.prefix != prefix or receipt.key != f"{prefix}/probe.json":
            raise SmokeError("runtime artifact probe must use the deterministic run prefix and exact key")
        return cls(run_id, _runtime_operation_id(run_id, classification), "smoke", SmokeArtifactReceipt.from_hub_probe(classification, receipt))


@dataclass(frozen=True)
class SmokeReadinessReceipt:
    """Pre-rent image and destination readiness only; it is never runtime output."""

    image_release: DockerImageRelease
    destinations: ReleaseDestinations
    operation_id: str


@dataclass(frozen=True)
class TrainingRuntimeEvidence:
    image_release: DockerImageRelease
    runtime_uid: int
    token_file_uid: int
    token_file_mode: int
    gpu_count: int
    optimizer_steps: int
    lifecycle_preflight: str
    artifact_label: str
    artifact: RuntimeArtifactReceipt
    checkpoint_bucket_probe: str = "passed"


@dataclass(frozen=True)
class RolloutRuntimeEvidence:
    image_release: DockerImageRelease
    gpu_count: int
    eula_environment: str
    warp_runtime: str
    headless_loads: int
    resets: int
    policy_health: str
    rgb_observation_count: int
    action_mapping_count: int
    evaluator_outcome: str
    fixtures: tuple[RuntimeArtifactReceipt, ...]


@dataclass(frozen=True)
class SmokePlan:
    purpose: str
    offer: SmokeOfferSelectionReceipt
    template: SmokeTemplatePublicationReceipt
    destination_readiness: SmokeReadinessReceipt
    declared_projected_spend_usd: Decimal | str | int | float | None = None


@dataclass(frozen=True)
class SmokeFailureEvidence:
    code: str
    state: SmokeState
    instance_id: str | None


@dataclass(frozen=True)
class SmokeReceipt:
    run_id: str
    purpose: str
    instance_id: str | None
    template: SmokeTemplatePublicationReceipt
    image_release: DockerImageRelease
    artifacts: tuple[RuntimeArtifactReceipt, ...]
    offer: SmokeOfferSelectionReceipt
    compatibility: Mapping[str, bool | int | str]
    reservation_active: bool
    worst_case_seconds: int
    projected_spend_usd: Decimal
    states: tuple[SmokeState, ...]
    failure: SmokeFailureEvidence | None


class SmokeRunFailed(SmokeError):
    def __init__(self, receipt: SmokeReceipt):
        self.receipt = receipt
        if receipt.instance_id is None:
            message = "paid smoke failed before an exact instance ID was recorded"
        elif receipt.failure is not None and receipt.failure.code == "cleanup-disappearance-failed":
            message = f"paid smoke cleanup failed for exact instance ID {receipt.instance_id}"
        else:
            message = f"paid smoke failed for exact instance ID {receipt.instance_id}"
        super().__init__(message)


class SmokeRemote(Protocol):
    """Remote operations are injected; this module ships no provider, SSH, or shell client."""

    def wait_for_ssh(self, instance_id: str, timeout_seconds: int, poll_interval_seconds: int) -> str: ...
    def wait_for_runtime(self, instance_id: str, purpose: str, timeout_seconds: int, poll_interval_seconds: int) -> str: ...
    def run_training_contract(self, run_id: str, instance_id: str, timeout_seconds: int) -> TrainingRuntimeEvidence: ...
    def run_rollout_contract(self, run_id: str, instance_id: str, timeout_seconds: int) -> RolloutRuntimeEvidence: ...
    def list_instance_ids(self, timeout_seconds: float) -> tuple[str, ...]: ...
    def ssh_endpoint_unreachable(self, instance_id: str, endpoint: str | None, timeout_seconds: float, poll_interval_seconds: int) -> bool: ...


class SmokeController:
    """Capped external state machine that reconciles a durable exact ID before cleanup."""

    def __init__(self, rental: CappedVastController, remote: SmokeRemote, *, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self._rental = rental
        self._remote = remote
        self._clock = clock
        self._sleep = sleep
        self.last_receipt: SmokeReceipt | None = None

    def run(self, plan: SmokePlan, *, timeouts: SmokeTimeouts | None = None) -> SmokeReceipt:
        limits = timeouts or SmokeTimeouts()
        limits.validate()
        projected = self._validate_plan(plan, limits)
        run_id = self._rental.ledger.new_smoke_run_id()
        states: list[SmokeState] = [SmokeState.PLANNED]
        instance_id: str | None = None
        endpoint: str | None = None
        runtime_artifacts: tuple[RuntimeArtifactReceipt, ...] = ()
        failure: SmokeFailureEvidence | None = None
        primary: BaseException | None = None
        cleanup_failure: SmokeFailureEvidence | None = None
        cleanup_interrupt: BaseException | None = None
        try:
            instance_id = self._rental.rent(
                run_id=run_id,
                purpose=plan.purpose,
                offer=plan.offer.ledger_offer(),
                projected_spend_usd=projected,
                request={"offer_id": plan.offer.offer_id, "template_id": plan.template.template_id, "image_reference": plan.template.image_release.reference, "payload_hash": plan.template.payload_hash, "purpose": plan.purpose, "disk_gb": 100 if plan.purpose == "training-smoke" else 300, "hourly_rate_usd": plan.offer.hourly_rate_usd},
                create_timeout_seconds=limits.create_timeout_seconds,
                reconcile_timeout_seconds=limits.reconcile_timeout_seconds,
            )
            self._validate_instance_id(instance_id)
            self._record_state(run_id, states, SmokeState.RENTED, instance_id)
            endpoint = self._call("SSH readiness", self._remote.wait_for_ssh, instance_id, limits.ssh_timeout_seconds, limits.poll_interval_seconds)
            if not isinstance(endpoint, str) or not endpoint:
                raise SmokeError("SSH readiness did not return one endpoint")
            self._record_state(run_id, states, SmokeState.SSH_READY, instance_id)
            runtime = self._call("runtime readiness", self._remote.wait_for_runtime, instance_id, plan.purpose, limits.runtime_timeout_seconds, limits.poll_interval_seconds)
            if runtime != "ready":
                raise SmokeError("runtime readiness did not complete")
            self._record_state(run_id, states, SmokeState.RUNTIME_READY, instance_id)
            evidence = self._run_contract(plan.purpose, run_id, instance_id, limits.contract_timeout_seconds)
            runtime_artifacts = self._validate_runtime(run_id, plan, evidence)
            self._record_state(run_id, states, SmokeState.READBACK_VERIFIED, instance_id)
        except BaseException as error:
            primary = error
            instance_id = instance_id or self._recover_instance_id(run_id, limits.reconcile_timeout_seconds)
            failure = SmokeFailureEvidence(self._failure_code(error), states[-1], instance_id)
            if instance_id is not None:
                self._record_failure(run_id, failure)
                states.append(SmokeState.FAILED)
        finally:
            instance_id = instance_id or self._recover_instance_id(run_id, limits.reconcile_timeout_seconds)
            if instance_id is not None:
                cleanup_failure, cleanup_interrupt = self._cleanup(run_id, instance_id, endpoint, states, limits)
                if cleanup_failure is not None:
                    failure = cleanup_failure
                    states.append(SmokeState.FAILED)
                    self._record_failure(run_id, cleanup_failure)
        receipt = SmokeReceipt(
            run_id=run_id,
            purpose=plan.purpose,
            instance_id=instance_id,
            template=plan.template,
            image_release=plan.template.image_release,
            artifacts=runtime_artifacts,
            offer=plan.offer,
            compatibility=plan.offer.compatibility.receipt(),
            reservation_active=self._reservation_active(run_id),
            worst_case_seconds=limits.worst_case_seconds(),
            projected_spend_usd=projected,
            states=tuple(states),
            failure=failure,
        )
        self.last_receipt = receipt
        if primary is not None:
            if isinstance(primary, (KeyboardInterrupt, SystemExit)):
                raise primary
            raise SmokeRunFailed(receipt) from None
        if cleanup_interrupt is not None:
            raise cleanup_interrupt
        if cleanup_failure is not None:
            raise SmokeRunFailed(receipt) from None
        return receipt

    @staticmethod
    def _validate_plan(plan: SmokePlan, limits: SmokeTimeouts) -> Decimal:
        if not isinstance(plan, SmokePlan) or plan.purpose not in {"training-smoke", "rollout-smoke"}:
            raise SmokeError("smoke purpose must be training-smoke or rollout-smoke")
        if not isinstance(plan.offer, SmokeOfferSelectionReceipt) or not isinstance(plan.template, SmokeTemplatePublicationReceipt):
            raise SmokeError("smoke plan requires typed offer selection and template publication receipts")
        if not isinstance(plan.template.template_id, str) or not plan.template.template_id or not isinstance(plan.template.image_release, DockerImageRelease) or not isinstance(plan.template.payload_hash, str) or not _PAYLOAD_HASH_RE.fullmatch(plan.template.payload_hash):
            raise SmokeError("smoke plan requires one typed template and image release")
        release = plan.template.image_release
        if release.reference != f"{release.repository}@{release.digest}" or not re.fullmatch(r"sha256:[0-9a-f]{64}", release.digest):
            raise SmokeError("smoke plan requires one canonical digest-qualified image release")
        expected_repository = "docker.io/ryanjin333/behavior1k-groot-n17-trainer" if plan.purpose == "training-smoke" else "docker.io/ryanjin333/behavior1k-groot-n17-rollout"
        if release.repository != expected_repository:
            raise SmokeError("smoke image release does not match the selected template purpose")
        if not isinstance(plan.offer.offer_id, str) or not plan.offer.offer_id or not isinstance(plan.offer.gpu_name, str) or not plan.offer.gpu_name:
            raise SmokeError("smoke offer selection receipt is invalid")
        plan.offer.compatibility.validate()
        if limits.worst_case_seconds() > plan.offer.compatibility.maximum_duration_minutes * 60:
            raise SmokeError("smoke total worst-case duration exceeds the selected offer duration")
        SmokeController._validate_destination_readiness(release, plan.destination_readiness)
        projected = SmokeController._derive_projected_spend(plan.offer.ledger_offer(), limits.worst_case_seconds())
        if plan.declared_projected_spend_usd is not None and SmokeController._money(plan.declared_projected_spend_usd) != projected:
            raise SmokeError("caller-declared projected spend must equal the internally derived worst-case cost")
        return projected

    @staticmethod
    def _validate_destination_readiness(image_release: DockerImageRelease, readiness: SmokeReadinessReceipt) -> None:
        if not isinstance(readiness, SmokeReadinessReceipt) or readiness.image_release != image_release or not isinstance(readiness.operation_id, str) or not re.fullmatch(r"preflight-[a-z0-9-]{1,80}", readiness.operation_id):
            raise SmokeError("smoke destination readiness must be one typed pre-rent image receipt")
        destinations = readiness.destinations
        if not isinstance(destinations, ReleaseDestinations) or not isinstance(destinations.model, HubRepository) or not isinstance(destinations.dataset, HubRepository) or not isinstance(destinations.checkpoint_bucket, CheckpointBucket):
            raise SmokeError("smoke destination readiness must use typed Hub and bucket destinations")
        if (destinations.model.repo_id, destinations.model.repo_type) != ("ryanjin333/behavior1k-groot-n17-models", "model") or (destinations.dataset.repo_id, destinations.dataset.repo_type) != ("ryanjin333/behavior1k-groot-n17-rollouts", "dataset") or destinations.checkpoint_bucket.bucket_id != "ryanjin333/behavior1k-groot-n17-checkpoints":
            raise SmokeError("smoke destination readiness does not match the pinned private release destinations")

    @staticmethod
    def _validate_runtime_artifacts(run_id: str, purpose: str, artifacts: tuple[RuntimeArtifactReceipt, ...]) -> tuple[RuntimeArtifactReceipt, ...]:
        if not isinstance(artifacts, tuple) or not all(
            isinstance(item, RuntimeArtifactReceipt)
            and isinstance(item.artifact, SmokeArtifactReceipt)
            and item.run_id == run_id
            and item.operation_id == _runtime_operation_id(run_id, item.artifact.classification)
            and item.smoke_label == "smoke"
            and item.artifact.absence_verified
            and item.artifact.prefix == _runtime_probe_prefix(run_id, item.artifact.classification)
            and item.artifact.key == f"{_runtime_probe_prefix(run_id, item.artifact.classification)}/probe.json"
            and _COMMIT_RE.fullmatch(item.artifact.upload_commit)
            and _COMMIT_RE.fullmatch(item.artifact.delete_commit)
            for item in artifacts
        ):
            raise SmokeError("runtime artifacts require current-run immutable deletion-and-absence receipts")
        if purpose == "training-smoke":
            valid = len(artifacts) == 1 and artifacts[0].artifact.classification == "smoke-model" and (artifacts[0].artifact.repo_id, artifacts[0].artifact.repo_type) == ("ryanjin333/behavior1k-groot-n17-models", "model")
        else:
            valid = len(artifacts) == 2 and {item.artifact.classification for item in artifacts} == {"success-fixture", "failure-fixture"} and all((item.artifact.repo_id, item.artifact.repo_type) == ("ryanjin333/behavior1k-groot-n17-rollouts", "dataset") for item in artifacts)
        if not valid or len({item.artifact.key for item in artifacts}) != len(artifacts) or len({item.operation_id for item in artifacts}) != len(artifacts):
            raise SmokeError("runtime artifacts do not match the exact private publication contract")
        return artifacts

    @staticmethod
    def _derive_projected_spend(offer: Mapping[str, Any], worst_case_seconds: int) -> Decimal:
        try:
            rate = Decimal(str(offer.get("hourly_rate_usd")))
        except (InvalidOperation, ValueError):
            raise SmokeError("smoke offer hourly rate is invalid") from None
        if not rate.is_finite() or rate <= 0 or rate.as_tuple().exponent < -6:
            raise SmokeError("smoke offer hourly rate is invalid")
        return (rate * Decimal(worst_case_seconds) / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_CEILING)

    @staticmethod
    def _money(value: Decimal | str | int | float) -> Decimal:
        try:
            amount = Decimal(str(value))
        except (InvalidOperation, ValueError):
            raise SmokeError("caller-declared projected spend is invalid") from None
        if not amount.is_finite() or amount < 0 or amount.as_tuple().exponent < -2:
            raise SmokeError("caller-declared projected spend is invalid")
        return amount

    def _validate_runtime(self, run_id: str, plan: SmokePlan, evidence: object) -> tuple[RuntimeArtifactReceipt, ...]:
        if plan.purpose == "training-smoke":
            if not isinstance(evidence, TrainingRuntimeEvidence) or evidence.image_release != plan.template.image_release or evidence.runtime_uid <= 0 or evidence.token_file_uid != evidence.runtime_uid or evidence.token_file_mode != 0o600 or evidence.gpu_count < 1 or evidence.optimizer_steps != 1 or evidence.lifecycle_preflight != "passed" or evidence.artifact_label != "smoke" or evidence.checkpoint_bucket_probe != "passed":
                raise SmokeError("training runtime evidence did not satisfy the typed smoke contract")
            return self._validate_runtime_artifacts(run_id, plan.purpose, (evidence.artifact,))
        if not isinstance(evidence, RolloutRuntimeEvidence) or evidence.image_release != plan.template.image_release or evidence.gpu_count < 1 or evidence.eula_environment != "OMNI_KIT_ACCEPT_EULA=YES" or evidence.warp_runtime != "bundled-compatible" or evidence.headless_loads < 1 or evidence.resets < 1 or evidence.policy_health != "ok" or evidence.rgb_observation_count < 1 or evidence.action_mapping_count < 1 or evidence.evaluator_outcome not in {"terminal", "quarantined"}:
            raise SmokeError("rollout runtime evidence did not satisfy the typed smoke contract")
        return self._validate_runtime_artifacts(run_id, plan.purpose, evidence.fixtures)

    def _run_contract(self, purpose: str, run_id: str, instance_id: str, timeout_seconds: int) -> object:
        callback = self._remote.run_training_contract if purpose == "training-smoke" else self._remote.run_rollout_contract
        return self._call("runtime contract", callback, run_id, instance_id, timeout_seconds)

    def _recover_instance_id(self, run_id: str, reconcile_timeout_seconds: int) -> str | None:
        try:
            recorded = self._rental.ledger.recorded_instance_id(run_id)
        except Exception:
            return None
        if recorded is not None:
            try:
                self._validate_instance_id(recorded)
            except SmokeError:
                return None
            return recorded
        try:
            authorized = any(record.get("event") == "rental-authorized" and record.get("run_id") == run_id for record in self._rental.ledger.records())
        except Exception:
            return None
        if not authorized:
            return None
        try:
            recovered = self._rental.reconcile_pending(run_id, timeout_seconds=reconcile_timeout_seconds)
            self._validate_instance_id(recovered)
            return recovered
        except Exception:
            return None

    def _reservation_active(self, run_id: str) -> bool:
        try:
            events = [record.get("event") for record in self._rental.ledger.records() if record.get("run_id") == run_id]
        except Exception:
            return False
        return "rental-authorized" in events and "actual-spend-recorded" not in events and "reservation-released" not in events

    @staticmethod
    def _validate_instance_id(instance_id: object) -> None:
        if not isinstance(instance_id, str) or not re.fullmatch(r"[1-9][0-9]*", instance_id) or instance_id in PROTECTED_INSTANCE_IDS:
            raise SmokeError("smoke cleanup requires one non-protected exact recorded instance ID")

    def _cleanup(self, run_id: str, instance_id: str, endpoint: str | None, states: list[SmokeState], limits: SmokeTimeouts) -> tuple[SmokeFailureEvidence | None, BaseException | None]:
        destroyed = False
        deferred_interrupt: BaseException | None = None
        for _attempt in range(2):
            try:
                self._rental.destroy(run_id, instance_id, timeout_seconds=limits.destroy_attempt_timeout_seconds)
            except BaseException as error:
                if not isinstance(error, Exception):
                    deferred_interrupt = deferred_interrupt or error
                continue
            destroyed = True
            self._record_state(run_id, states, SmokeState.DESTROYED, instance_id)
            break
        deadline = self._disappearance_deadline(limits.disappearance_timeout_seconds)
        list_budget = self._remaining_disappearance_seconds(deadline, limits.disappearance_timeout_seconds)
        if list_budget is None:
            vast_absent = False
            ssh_unreachable = False
        else:
            vast_absent = False
            while True:
                try:
                    listed = self._remote.list_instance_ids(min(55.0, list_budget))
                    vast_absent = isinstance(listed, tuple) and instance_id not in listed
                except BaseException as error:
                    if not isinstance(error, Exception):
                        deferred_interrupt = deferred_interrupt or error
                        # Preserve the interrupt for the caller, but do not
                        # spend the whole deadline polling after cancellation:
                        # SSH disappearance is independent cleanup evidence.
                        break
                    vast_absent = False
                if vast_absent:
                    break
                list_budget = self._remaining_disappearance_seconds(deadline, limits.disappearance_timeout_seconds)
                if list_budget is None:
                    break
                self._sleep(min(float(limits.poll_interval_seconds), list_budget))
            ssh_budget = self._remaining_disappearance_seconds(deadline, limits.disappearance_timeout_seconds)
            if ssh_budget is None:
                ssh_unreachable = False
            else:
                try:
                    ssh_unreachable = self._remote.ssh_endpoint_unreachable(instance_id, endpoint, ssh_budget, limits.poll_interval_seconds) is True
                except BaseException as error:
                    if not isinstance(error, Exception):
                        deferred_interrupt = deferred_interrupt or error
                    ssh_unreachable = False
        self._rental.ledger.record_lifecycle_evidence(run_id, "cleanup-checked", {"vast_absent": vast_absent, "ssh_unreachable": ssh_unreachable, "instance_id": instance_id})
        if destroyed and vast_absent and ssh_unreachable:
            self._record_state(run_id, states, SmokeState.DISAPPEARANCE_VERIFIED, instance_id)
            return None, deferred_interrupt
        return SmokeFailureEvidence("cleanup-disappearance-failed", states[-1], instance_id), deferred_interrupt

    def _disappearance_deadline(self, timeout_seconds: int) -> float | None:
        try:
            started = float(self._clock())
        except Exception:
            return None
        if started != started or started in {float("inf"), float("-inf")}:
            return None
        return started + float(timeout_seconds)

    def _remaining_disappearance_seconds(self, deadline: float | None, configured_timeout_seconds: int) -> float | None:
        if deadline is None:
            return None
        try:
            now = float(self._clock())
        except Exception:
            return None
        if now != now or now in {float("inf"), float("-inf")}:
            return None
        remaining = min(deadline - now, float(configured_timeout_seconds))
        return remaining if remaining > 0 else None

    def _record_state(self, run_id: str, states: list[SmokeState], state: SmokeState, instance_id: str) -> None:
        states.append(state)
        self._rental.ledger.record_lifecycle_evidence(run_id, state.value, {"status": state.value, "instance_id": instance_id})

    def _record_failure(self, run_id: str, failure: SmokeFailureEvidence) -> None:
        self._rental.ledger.record_lifecycle_evidence(run_id, SmokeState.FAILED.value, {"status": failure.code, "instance_id": failure.instance_id or "not-recorded"})

    @staticmethod
    def _call(operation: str, callback: Any, *args: object) -> Any:
        try:
            return callback(*args)
        except Exception:
            raise SmokeError(f"{operation} failed") from None

    @staticmethod
    def _failure_code(error: BaseException) -> str:
        if isinstance(error, KeyboardInterrupt):
            return "interrupted"
        if isinstance(error, SmokeError):
            return "runtime-contract-failed" if "runtime" in str(error) else "rental-or-readiness-failed"
        return "rental-or-readiness-failed"

    @staticmethod
    def _receipt_offer(offer: Mapping[str, Any]) -> dict[str, Any]:
        return {key: offer[key] for key in ("offer_id", "hourly_rate_usd", "verified", "gpu_name") if key in offer}
