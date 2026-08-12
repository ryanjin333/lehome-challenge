from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from b1k_deploy.dockerhub import DockerImageRelease
from b1k_deploy.huggingface import CheckpointBucket, HubProbeReceipt, HubRepository, ReleaseDestinations
from b1k_deploy.ledger import RentalLedger
from b1k_deploy.smoke import (
    RolloutRuntimeEvidence,
    RuntimeArtifactReceipt,
    SmokeArtifactReceipt,
    SmokeCompatibility,
    SmokeController,
    SmokeError,
    SmokePlan,
    SmokeOfferSelectionReceipt,
    SmokeReadinessReceipt,
    SmokeRunFailed,
    SmokeState,
    SmokeTimeouts,
    SmokeTemplatePublicationReceipt,
    TrainingRuntimeEvidence,
)
from b1k_deploy.vast import CappedVastController, ProviderCreatedButSetupFailed, VastAdapter


class RecordingVast:
    def __init__(self, instance_id: str = "9123456"):
        self.instance_id = instance_id
        self.created = []
        self.destroyed = []
        self.destroy_timeouts = []
        self.reconciled_by_key = {}
        self.live_instance_ids = set()

    def create_instance(self, request, *, timeout_seconds):
        self.created.append(request)
        self.live_instance_ids.add(self.instance_id)
        return self.instance_id

    def find_instance_by_idempotency_key(self, key, *, timeout_seconds):
        return self.reconciled_by_key.get(key)

    def destroy_instance(self, instance_id, *, timeout_seconds):
        self.destroyed.append(instance_id)
        self.destroy_timeouts.append(timeout_seconds)
        self.live_instance_ids.discard(instance_id)


class FakeRemote:
    def __init__(self):
        self.calls = []
        self.training_factory = None
        self.rollout_factory = None
        self.interrupt_at = None
        self.disappears = True
        self.endpoint_disappears = True

    def _interrupt(self, stage):
        if self.interrupt_at == stage:
            raise KeyboardInterrupt()

    def wait_for_ssh(self, instance_id, timeout_seconds, poll_interval_seconds):
        self.calls.append(("ssh", instance_id, timeout_seconds, poll_interval_seconds))
        self._interrupt("ssh")
        return "ssh://smoke-endpoint"

    def wait_for_runtime(self, instance_id, purpose, timeout_seconds, poll_interval_seconds):
        self.calls.append(("runtime", instance_id, timeout_seconds, poll_interval_seconds))
        self._interrupt("runtime")
        return "ready"

    def run_training_contract(self, run_id, instance_id, timeout_seconds):
        self.calls.append(("training", run_id, instance_id, timeout_seconds))
        self._interrupt("contract")
        return self.training_factory(run_id)

    def run_rollout_contract(self, run_id, instance_id, timeout_seconds):
        self.calls.append(("rollout", run_id, instance_id, timeout_seconds))
        self._interrupt("contract")
        return self.rollout_factory(run_id)

    def list_instance_ids(self, timeout_seconds):
        self.calls.append(("list", timeout_seconds))
        return () if self.disappears else ("9123456",)

    def ssh_endpoint_unreachable(self, instance_id, endpoint, timeout_seconds, poll_interval_seconds):
        self.calls.append(("ssh-gone", instance_id, endpoint, timeout_seconds, poll_interval_seconds))
        return self.endpoint_disappears


class FakeMonotonicClock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def offer(rate="0.40", gpu_bid="0.20"):
    return SmokeOfferSelectionReceipt("90210", rate, "RTX 4090", SmokeCompatibility(True, True, 200, 64, 1000, 15, "cheapest-compatible-verified"), gpu_bid)


def test_offer_ledger_snapshot_preserves_six_decimal_provider_rate() -> None:
    assert offer(Decimal("0.077778")).ledger_offer()["hourly_rate_usd"] == "0.077778"


def image(purpose="training-smoke"):
    repository = "docker.io/ryanjin333/behavior1k-groot-n17"
    digest = "sha256:" + ("a" if purpose == "training-smoke" else "b") * 64
    role = "training" if purpose == "training-smoke" else "rollout"
    source_commit = "a" * 40
    tag = f"trainer-{source_commit}" if role == "training" else f"rollout-{source_commit}"
    return DockerImageRelease(role, repository, tag, source_commit, digest, f"{repository}@{digest}")


def destinations():
    return ReleaseDestinations(
        HubRepository("ryanjin333/behavior1k-groot-n17-models", "model"),
        CheckpointBucket(),
        HubRepository("ryanjin333/behavior1k-groot-n17-rollouts", "dataset"),
    )


def artifact(role, classification, char):
    repo_id, repo_type = ("ryanjin333/behavior1k-groot-n17-models", "model") if role == "model" else ("ryanjin333/behavior1k-groot-n17-rollouts", "dataset")
    prefix = "b1k-bootstrap-" + char * 32
    return SmokeArtifactReceipt.from_hub_probe(classification, HubProbeReceipt(role, repo_id, repo_type, prefix, f"{prefix}/probe.json", char * 40, ("f" if char != "f" else "e") * 40))


def runtime_artifact(run_id, role, classification, char):
    prefix = f"b1k-bootstrap-{run_id.removeprefix('b1k-smoke-')}-{classification}"
    repo_id, repo_type = ("ryanjin333/behavior1k-groot-n17-models", "model") if role == "model" else ("ryanjin333/behavior1k-groot-n17-rollouts", "dataset")
    probe = HubProbeReceipt(role, repo_id, repo_type, prefix, f"{prefix}/probe.json", char * 40, ("f" if char != "f" else "e") * 40)
    return RuntimeArtifactReceipt.from_hub_probe(run_id, classification, probe)


def plan(purpose="training-smoke"):
    release = image(purpose)
    return SmokePlan(
        purpose=purpose,
        offer=offer(),
        template=SmokeTemplatePublicationReceipt("123", release, "c" * 64),
        destination_readiness=SmokeReadinessReceipt(release, destinations(), "preflight-release-ready"),
    )


def training_evidence(candidate, run_id):
    return TrainingRuntimeEvidence(candidate.template.image_release, runtime_uid=1000, token_file_uid=1000, token_file_mode=0o600, gpu_count=1, optimizer_steps=1, lifecycle_preflight="passed", artifact_label="smoke", artifact=runtime_artifact(run_id, "model", "smoke-model", "1"))


def rollout_evidence(candidate, run_id):
    fixtures = (runtime_artifact(run_id, "dataset", "success-fixture", "2"), runtime_artifact(run_id, "dataset", "failure-fixture", "3"))
    return RolloutRuntimeEvidence(candidate.template.image_release, gpu_count=1, eula_environment="OMNI_KIT_ACCEPT_EULA=YES", warp_runtime="bundled-compatible", headless_loads=1, resets=1, policy_health="ok", rgb_observation_count=1, action_mapping_count=1, evaluator_outcome="terminal", fixtures=fixtures)


def controller(tmp_path, remote=None, vast=None, clock=None):
    vast = vast or RecordingVast()
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    arguments = {} if clock is None else {"clock": clock}
    return SmokeController(CappedVastController(ledger, VastAdapter(vast)), remote or FakeRemote(), **arguments), ledger, vast


def test_training_smoke_final_receipt_uses_current_runtime_artifact_not_prerent_readiness(tmp_path):
    remote = FakeRemote()
    candidate = plan()
    remote.training_factory = lambda run_id: training_evidence(candidate, run_id)
    smoke, ledger, vast = controller(tmp_path, remote)

    receipt = smoke.run(candidate)

    assert receipt.states == (SmokeState.PLANNED, SmokeState.RENTED, SmokeState.SSH_READY, SmokeState.RUNTIME_READY, SmokeState.READBACK_VERIFIED, SmokeState.DESTROYED, SmokeState.DISAPPEARANCE_VERIFIED)
    assert receipt.artifacts[0].run_id == receipt.run_id
    assert receipt.artifacts[0].operation_id == f"{receipt.run_id}:smoke-model"
    assert receipt.artifacts[0].artifact.upload_commit == "1" * 40
    assert receipt.projected_spend_usd == Decimal("0.05")
    assert list(ledger.records())[0]["projected_spend_usd"] == "0.05"
    assert list(ledger.records())[0]["offer"]["hourly_rate_usd"] == "0.40"
    assert vast.created[0]["hourly_rate_usd"] == "0.40"
    assert vast.created[0]["gpu_bid_usd"] == "0.20"
    assert vast.destroyed == ["9123456"]
    assert vast.destroy_timeouts == [30]


def test_rollout_requires_current_run_typed_eula_warp_and_both_runtime_fixture_receipts(tmp_path):
    remote = FakeRemote()
    candidate = plan("rollout-smoke")
    remote.rollout_factory = lambda run_id: rollout_evidence(candidate, run_id)
    smoke, _ledger, vast = controller(tmp_path, remote)

    receipt = smoke.run(candidate)

    assert {item.artifact.classification for item in receipt.artifacts} == {"success-fixture", "failure-fixture"}
    assert all(item.run_id == receipt.run_id and item.smoke_label == "smoke" for item in receipt.artifacts)
    assert vast.destroyed == ["9123456"]


def test_rollout_accepts_one_executed_nonterminal_evaluator_step(tmp_path):
    remote = FakeRemote()
    candidate = plan("rollout-smoke")
    remote.rollout_factory = lambda run_id: replace(
        rollout_evidence(candidate, run_id), evaluator_outcome="advanced"
    )
    smoke, _ledger, vast = controller(tmp_path, remote)

    receipt = smoke.run(candidate)

    assert receipt.states[-3:] == (
        SmokeState.READBACK_VERIFIED,
        SmokeState.DESTROYED,
        SmokeState.DISAPPEARANCE_VERIFIED,
    )
    assert vast.destroyed == ["9123456"]


def test_prerent_readiness_receipt_cannot_be_substituted_for_a_runtime_artifact_receipt(tmp_path):
    remote = FakeRemote()
    candidate = plan()
    preflight = candidate.destination_readiness
    remote.training_factory = lambda _run_id: TrainingRuntimeEvidence(candidate.template.image_release, 1000, 1000, 0o600, 1, 1, "passed", "smoke", preflight)
    smoke, _ledger, vast = controller(tmp_path, remote)

    with pytest.raises(SmokeRunFailed):
        smoke.run(candidate)

    assert vast.destroyed == ["9123456"]
    assert smoke.last_receipt.artifacts == ()


def test_rejects_any_caller_cost_declaration_that_understates_or_mismatches_the_internal_worst_case_cost(tmp_path):
    candidate = replace(plan(), declared_projected_spend_usd=Decimal("0.01"))
    smoke, ledger, vast = controller(tmp_path)

    with pytest.raises(SmokeError, match="derived"):
        smoke.run(candidate)

    assert list(ledger.records()) == []
    assert vast.created == []


@pytest.mark.parametrize("gpu_bid", ("not-a-rate", "0", "0.400001", "0.2000001"))
def test_rejects_invalid_gpu_bid_before_reserving_budget(tmp_path, gpu_bid):
    candidate = replace(plan(), offer=offer("0.40", gpu_bid))
    smoke, ledger, vast = controller(tmp_path)

    with pytest.raises(SmokeError, match="GPU bid"):
        smoke.run(candidate)

    assert list(ledger.records()) == []
    assert vast.created == []


def test_total_budget_must_fit_the_offer_duration(tmp_path):
    original = offer("5.00")
    candidate = replace(plan(), offer=replace(original, compatibility=replace(original.compatibility, maximum_duration_minutes=1)))
    smoke, _ledger, vast = controller(tmp_path)

    with pytest.raises(SmokeError, match="duration"):
        smoke.run(candidate)

    assert vast.created == []


@pytest.mark.parametrize("boundary", ["ssh", "runtime", "contract"])
def test_interrupt_at_each_post_rent_boundary_still_destroys_the_exact_recorded_id(tmp_path, boundary):
    remote = FakeRemote()
    candidate = plan()
    remote.training_factory = lambda run_id: training_evidence(candidate, run_id)
    remote.interrupt_at = boundary
    smoke, _ledger, vast = controller(tmp_path, remote)

    with pytest.raises(KeyboardInterrupt):
        smoke.run(candidate)

    assert vast.destroyed == ["9123456"]
    assert smoke.last_receipt.instance_id == "9123456"


def test_interrupt_after_provider_create_before_ledger_id_reconciles_only_authorized_exact_instance_and_destroys_it(tmp_path):
    class InterruptAfterProviderCreate(CappedVastController):
        def rent(self, *, run_id, purpose, offer, projected_spend_usd, request, create_timeout_seconds, reconcile_timeout_seconds):
            self.ledger.authorize_rental(run_id, purpose, offer, projected_spend_usd)
            self.ledger.record_creation_attempt(run_id, run_id)
            self.vast.create_instance(request, run_id, timeout_seconds=create_timeout_seconds)
            raise KeyboardInterrupt()

    remote = FakeRemote()
    vast = RecordingVast()
    vast.live_instance_ids.update({"47198086", "9123457"})
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    smoke = SmokeController(InterruptAfterProviderCreate(ledger, VastAdapter(vast)), remote)
    original_new_run_id = ledger.new_smoke_run_id

    def new_run_id():
        run_id = original_new_run_id()
        vast.reconciled_by_key[run_id] = "9123456"
        return run_id

    ledger.new_smoke_run_id = new_run_id
    with pytest.raises(KeyboardInterrupt):
        smoke.run(plan())

    assert ledger.recorded_instance_id(smoke.last_receipt.run_id) == "9123456"
    assert vast.destroyed == ["9123456"]
    assert vast.live_instance_ids == {"47198086", "9123457"}
    assert "47198086" not in vast.destroyed
    assert "9123457" not in vast.destroyed


def test_post_create_ssh_setup_failure_destroys_the_exact_known_instance(tmp_path):
    class SetupFailedVast(RecordingVast):
        def create_instance(self, request, *, timeout_seconds):
            self.created.append(request)
            self.live_instance_ids.add(self.instance_id)
            raise ProviderCreatedButSetupFailed(self.instance_id)

        def find_instance_by_idempotency_key(self, key, *, timeout_seconds):
            raise AssertionError("known instance ID must not depend on reconciliation")

    smoke, ledger, vast = controller(tmp_path, FakeRemote(), SetupFailedVast())

    with pytest.raises(SmokeRunFailed) as error:
        smoke.run(plan("rollout-smoke"))

    assert ledger.recorded_instance_id(error.value.receipt.run_id) == "9123456"
    assert error.value.receipt.instance_id == "9123456"
    assert vast.destroyed == ["9123456"]
    assert vast.live_instance_ids == set()


def test_untyped_runtime_evidence_is_rejected_after_rent_with_exact_cleanup(tmp_path):
    remote = FakeRemote()
    remote.training_factory = lambda _run_id: {"private_digest_pull": True}
    smoke, _ledger, vast = controller(tmp_path, remote)

    with pytest.raises(SmokeRunFailed):
        smoke.run(plan())

    assert vast.destroyed == ["9123456"]


def test_wrong_image_reference_or_forged_destination_readiness_is_rejected_before_rent(tmp_path):
    candidate = replace(plan(), template=replace(plan().template, image_release=image("rollout-smoke")))
    smoke, _ledger, vast = controller(tmp_path)
    with pytest.raises(SmokeError, match="image"):
        smoke.run(candidate)
    assert vast.created == []

    forged = SmokeReadinessReceipt(candidate.template.image_release, ReleaseDestinations(HubRepository("not-the-private-model", "model"), CheckpointBucket(), HubRepository("ryanjin333/behavior1k-groot-n17-rollouts", "dataset")), "preflight-release-ready")
    with pytest.raises(SmokeError, match="destination"):
        smoke.run(replace(plan(), destination_readiness=forged))
    assert vast.created == []


@pytest.mark.parametrize(
    "changes",
    (
        {"tag": "trainer-" + "b" * 40},
        {"source_commit": "b" * 40},
    ),
)
def test_noncanonical_tag_or_source_receipt_is_rejected_before_rent(tmp_path, changes):
    candidate = plan()
    forged_release = replace(candidate.template.image_release, **changes)
    forged = replace(
        candidate,
        template=replace(candidate.template, image_release=forged_release),
        destination_readiness=replace(candidate.destination_readiness, image_release=forged_release),
    )
    smoke, _ledger, vast = controller(tmp_path)

    with pytest.raises(SmokeError, match="image release"):
        smoke.run(forged)

    assert vast.created == []


def test_two_provider_destroy_timeouts_are_bounded_and_finish_as_cleanup_failure(tmp_path):
    class TimeoutDestroyVast(RecordingVast):
        def destroy_instance(self, instance_id, *, timeout_seconds):
            super().destroy_instance(instance_id, timeout_seconds=timeout_seconds)
            raise TimeoutError("provider destroy exceeded injected deadline")

    remote = FakeRemote()
    candidate = plan()
    remote.training_factory = lambda run_id: training_evidence(candidate, run_id)
    smoke, _ledger, vast = controller(tmp_path, remote, TimeoutDestroyVast())

    with pytest.raises(SmokeRunFailed) as error:
        smoke.run(candidate)

    assert vast.destroyed == ["9123456", "9123456"]
    assert vast.destroy_timeouts == [30, 30]
    assert error.value.receipt.worst_case_seconds == 420
    assert error.value.receipt.failure.code == "cleanup-disappearance-failed"


def test_disappearance_probes_share_one_deadline_and_ssh_receives_only_the_remainder(tmp_path):
    class SlowListRemote(FakeRemote):
        def list_instance_ids(self, timeout_seconds):
            result = super().list_instance_ids(timeout_seconds)
            clock.now += 29.5
            return result

    clock = FakeMonotonicClock()
    remote = SlowListRemote()
    candidate = plan()
    remote.training_factory = lambda run_id: training_evidence(candidate, run_id)
    smoke, _ledger, _vast = controller(tmp_path, remote, clock=clock)

    receipt = smoke.run(candidate)

    list_call, ssh_call = remote.calls[-2:]
    assert list_call == ("list", 30.0)
    assert ssh_call[0] == "ssh-gone"
    assert ssh_call[3] == pytest.approx(0.5)
    assert receipt.worst_case_seconds == 420
    assert receipt.projected_spend_usd == Decimal("0.05")


def test_disappearance_expiry_after_list_fails_closed_without_calling_ssh(tmp_path):
    class ExpiringListRemote(FakeRemote):
        def list_instance_ids(self, timeout_seconds):
            result = super().list_instance_ids(timeout_seconds)
            clock.now += 30
            return result

    clock = FakeMonotonicClock()
    remote = ExpiringListRemote()
    candidate = plan()
    remote.training_factory = lambda run_id: training_evidence(candidate, run_id)
    smoke, _ledger, _vast = controller(tmp_path, remote, clock=clock)

    with pytest.raises(SmokeRunFailed) as error:
        smoke.run(candidate)

    assert remote.calls[-1] == ("list", 30.0)
    assert not any(call[0] == "ssh-gone" for call in remote.calls)
    assert error.value.receipt.failure.code == "cleanup-disappearance-failed"


def test_disappearance_polls_only_the_recorded_id_until_absent_under_the_300_second_deadline(tmp_path):
    class PollingRemote(FakeRemote):
        def __init__(self):
            super().__init__(); self.polls = 0
        def list_instance_ids(self, timeout_seconds):
            self.polls += 1
            self.calls.append(("list", timeout_seconds))
            return ("9123456",) if self.polls == 1 else ()

    clock = FakeMonotonicClock()
    remote = PollingRemote()
    candidate = plan()
    remote.training_factory = lambda run_id: training_evidence(candidate, run_id)
    vast = RecordingVast(); ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    smoke = SmokeController(CappedVastController(ledger, VastAdapter(vast)), remote, clock=clock, sleep=lambda seconds: setattr(clock, "now", clock.now + seconds))

    receipt = smoke.run(candidate, timeouts=SmokeTimeouts(disappearance_timeout_seconds=300))

    assert receipt.states[-1] is SmokeState.DISAPPEARANCE_VERIFIED
    assert remote.polls == 2
    assert [call for call in remote.calls if call[0] == "list"] == [("list", 55.0), ("list", 55.0)]


def test_runtime_ready_is_durably_recorded_before_contract_validation_fails(tmp_path):
    remote = FakeRemote()
    remote.training_factory = lambda _run_id: {"not": "typed evidence"}
    smoke, _ledger, _vast = controller(tmp_path, remote)

    with pytest.raises(SmokeRunFailed) as error:
        smoke.run(plan())

    assert error.value.receipt.failure.state == SmokeState.RUNTIME_READY


def test_runtime_failure_preserves_only_an_allowlisted_remote_diagnostic_code(tmp_path):
    remote = FakeRemote()
    remote.training_factory = lambda _run_id: (_ for _ in ()).throw(
        SmokeError("SSH command failed: remote-cuda-out-of-memory")
    )
    smoke, ledger, _vast = controller(tmp_path, remote)

    with pytest.raises(SmokeRunFailed) as error:
        smoke.run(plan())

    assert error.value.receipt.failure.code == "remote-cuda-out-of-memory"
    failed = [row for row in ledger.records() if row.get("lifecycle") == "failed"]
    assert failed[-1]["evidence"]["status"] == "remote-cuda-out-of-memory"


def test_runtime_failure_drops_non_allowlisted_remote_error_text(tmp_path):
    remote = FakeRemote()
    remote.training_factory = lambda _run_id: (_ for _ in ()).throw(
        SmokeError("SSH command failed: custom-secret-must-not-leak")
    )
    smoke, _ledger, _vast = controller(tmp_path, remote)

    with pytest.raises(SmokeRunFailed) as error:
        smoke.run(plan())

    assert error.value.receipt.failure.code == "runtime-contract-failed"
    assert "custom-secret-must-not-leak" not in str(error.value)


@pytest.mark.parametrize("boundary", ["destroy", "list", "ssh-gone"])
def test_cleanup_boundary_interrupt_is_deferred_until_exact_cleanup_receipt_is_persisted(tmp_path, boundary):
    class InterruptVast(RecordingVast):
        def destroy_instance(self, instance_id, *, timeout_seconds):
            super().destroy_instance(instance_id, timeout_seconds=timeout_seconds)
            if boundary == "destroy" and len(self.destroyed) == 1:
                raise KeyboardInterrupt()

    class InterruptRemote(FakeRemote):
        def list_instance_ids(self, timeout_seconds):
            result = super().list_instance_ids(timeout_seconds)
            if boundary == "list":
                raise KeyboardInterrupt()
            return result

        def ssh_endpoint_unreachable(self, instance_id, endpoint, timeout_seconds, poll_interval_seconds):
            result = super().ssh_endpoint_unreachable(instance_id, endpoint, timeout_seconds, poll_interval_seconds)
            if boundary == "ssh-gone":
                raise KeyboardInterrupt()
            return result

    remote = InterruptRemote()
    candidate = plan()
    remote.training_factory = lambda run_id: training_evidence(candidate, run_id)
    smoke, ledger, vast = controller(tmp_path, remote, InterruptVast())

    with pytest.raises(KeyboardInterrupt):
        smoke.run(candidate)

    assert ledger.recorded_instance_id(smoke.last_receipt.run_id) == "9123456"
    assert vast.destroyed == (["9123456", "9123456"] if boundary == "destroy" else ["9123456"])
    assert vast.live_instance_ids == set()
    assert any(call[0] == "list" for call in remote.calls)
    assert any(call[0] == "ssh-gone" for call in remote.calls)


def test_runtime_artifact_rejects_a_stale_prefix_even_when_its_run_and_operation_are_relabelled(tmp_path):
    remote = FakeRemote()
    candidate = plan()

    def stale_evidence(run_id):
        stale = SmokeArtifactReceipt("smoke-model", "ryanjin333/behavior1k-groot-n17-models", "model", "b1k-bootstrap-stale", "b1k-bootstrap-stale/probe.json", "1" * 40, "f" * 40, True)
        return TrainingRuntimeEvidence(candidate.template.image_release, 1000, 1000, 0o600, 1, 1, "passed", "smoke", RuntimeArtifactReceipt(run_id, f"{run_id}:smoke-model", "smoke", stale))

    remote.training_factory = stale_evidence
    smoke, _ledger, vast = controller(tmp_path, remote)

    with pytest.raises(SmokeRunFailed):
        smoke.run(candidate)

    assert vast.destroyed == ["9123456"]


def test_create_timeout_reconciles_exact_authorized_instance_with_bounded_provider_calls(tmp_path):
    class TimeoutThenReconcileVast(RecordingVast):
        def __init__(self):
            super().__init__()
            self.create_timeouts = []
            self.reconcile_timeouts = []
            self.pending = {}

        def create_instance(self, request, *, timeout_seconds):
            self.create_timeouts.append(timeout_seconds)
            self.created.append(request)
            self.live_instance_ids.add(self.instance_id)
            self.pending[request["idempotency_key"]] = self.instance_id
            raise TimeoutError("create timed out after provider accepted the request")

        def find_instance_by_idempotency_key(self, key, *, timeout_seconds):
            self.reconcile_timeouts.append(timeout_seconds)
            return self.pending.get(key)

    remote = FakeRemote()
    candidate = plan()
    remote.training_factory = lambda run_id: training_evidence(candidate, run_id)
    smoke, ledger, vast = controller(tmp_path, remote, TimeoutThenReconcileVast())

    receipt = smoke.run(candidate)

    assert ledger.recorded_instance_id(receipt.run_id) == "9123456"
    assert vast.create_timeouts == [30]
    assert vast.reconcile_timeouts == [30]
    assert vast.destroyed == ["9123456"]


def test_smoke_plan_rejects_forged_raw_offer_or_template_mappings_before_rent(tmp_path):
    smoke, _ledger, vast = controller(tmp_path)
    with pytest.raises(SmokeError, match="typed"):
        smoke.run(replace(plan(), offer={"offer_id": "90210"}))
    with pytest.raises(SmokeError, match="typed"):
        smoke.run(replace(plan(), template={"template_id": "123"}))

    assert vast.created == []
