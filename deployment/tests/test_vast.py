import pytest

from b1k_deploy.ledger import LedgerError, RentalLedger
from b1k_deploy.vast import (
    PROTECTED_INSTANCE_IDS,
    CappedVastController,
    InstanceTargetRejected,
    VastAdapter,
    VastCreateAmbiguous,
    VastCreateFailed,
    ProviderNotCreated,
    ProviderCreatedButSetupFailed,
    VastPostCreateSetupFailed,
)


class RecordingVast:
    def __init__(self, instance_id: str = "9123456"):
        self.instance_id = instance_id
        self.created: list[dict[str, object]] = []
        self.destroyed: list[str] = []

    def create_instance(self, request: dict[str, object], *, timeout_seconds: int) -> str:
        self.created.append(request)
        return self.instance_id

    def destroy_instance(self, instance_id: str, *, timeout_seconds: int) -> None:
        self.destroyed.append(instance_id)


def offer() -> dict[str, object]:
    return {"offer_id": "90210", "hourly_rate_usd": "0.40", "verified": True}


def test_rental_is_authorized_and_written_before_injected_adapter_mutates(tmp_path):
    ledger_path = tmp_path / "cost-ledger.jsonl"
    ledger = RentalLedger(ledger_path)
    remote = RecordingVast()
    controller = CappedVastController(ledger, VastAdapter(remote))
    run_id = ledger.new_smoke_run_id()

    instance_id = controller.rent(
        run_id=run_id,
        purpose="training-smoke",
        offer=offer(),
        projected_spend_usd="0.80",
        request={"offer_id": "90210", "template_id": "123"},
    )

    assert instance_id == "9123456"
    assert remote.created == [{"offer_id": "90210", "template_id": "123", "idempotency_key": run_id}]
    assert ledger.recorded_instance_id(run_id) == "9123456"
    assert [record["event"] for record in ledger.records()] == [
        "rental-authorized",
        "creation-attempt-recorded",
        "instance-recorded",
    ]
    assert remote.created[0]["idempotency_key"] == run_id


def test_ambiguous_create_reconciles_by_run_id_and_keeps_exact_cleanup_identity(tmp_path):
    class AmbiguousVast(RecordingVast):
        def create_instance(self, request, *, timeout_seconds):
            self.created.append(request)
            return None

        def find_instance_by_idempotency_key(self, key, *, timeout_seconds):
            return "9123456" if key.startswith("b1k-smoke-") else None

    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    remote = AmbiguousVast()
    controller = CappedVastController(ledger, VastAdapter(remote))
    run_id = ledger.new_smoke_run_id()

    assert controller.rent(run_id=run_id, purpose="training-smoke", offer=offer(), projected_spend_usd="0.80", request={"offer_id": "90210"}) == "9123456"
    assert ledger.recorded_instance_id(run_id) == "9123456"


def test_unreconciled_ambiguous_create_keeps_reservation(tmp_path):
    class AmbiguousVast(RecordingVast):
        def create_instance(self, request, *, timeout_seconds):
            return None

        def find_instance_by_idempotency_key(self, key, *, timeout_seconds):
            return None

    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    controller = CappedVastController(ledger, VastAdapter(AmbiguousVast()))
    run_id = ledger.new_smoke_run_id()

    with pytest.raises(VastCreateAmbiguous):
        controller.rent(run_id=run_id, purpose="training-smoke", offer=offer(), projected_spend_usd="3.00", request={"offer_id": "90210"})
    assert ledger.reserved_spend_usd() == 3


def test_generic_create_timeout_reconciles_or_remains_ambiguous_without_release(tmp_path):
    class TimeoutVast(RecordingVast):
        def create_instance(self, request, *, timeout_seconds):
            raise TimeoutError("provider did not answer")

        def find_instance_by_idempotency_key(self, key, *, timeout_seconds):
            return None

    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    controller = CappedVastController(ledger, VastAdapter(TimeoutVast()))
    run_id = ledger.new_smoke_run_id()
    with pytest.raises(VastCreateAmbiguous):
        controller.rent(run_id=run_id, purpose="training-smoke", offer=offer(), projected_spend_usd="1.00", request={"offer_id": "90210"})
    assert ledger.reserved_spend_usd() == 1


def test_record_append_failure_rejects_reconciliation_to_a_different_identity(tmp_path, monkeypatch):
    class ReconciledVast(RecordingVast):
        def find_instance_by_idempotency_key(self, key, *, timeout_seconds):
            return "9123457"

    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    controller = CappedVastController(ledger, VastAdapter(ReconciledVast("9123456")))
    original = ledger.record_instance

    def fail_first(run_id, instance_id):
        monkeypatch.setattr(ledger, "record_instance", original)
        raise OSError("durable append interrupted")

    monkeypatch.setattr(ledger, "record_instance", fail_first)
    run_id = ledger.new_smoke_run_id()
    with pytest.raises(VastCreateAmbiguous, match="differs"):
        controller.rent(run_id=run_id, purpose="training-smoke", offer=offer(), projected_spend_usd="1.00", request={"offer_id": "90210"})
    assert ledger.recorded_instance_id(run_id) is None


def test_only_provider_not_created_releases_the_reservation(tmp_path):
    class MissingVast(RecordingVast):
        def create_instance(self, request, *, timeout_seconds):
            raise ProviderNotCreated()

    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    controller = CappedVastController(ledger, VastAdapter(MissingVast()))
    run_id = ledger.new_smoke_run_id()
    with pytest.raises(VastCreateFailed):
        controller.rent(run_id=run_id, purpose="training-smoke", offer=offer(), projected_spend_usd="1.00", request={"offer_id": "90210"})
    assert ledger.reserved_spend_usd() == 0


def test_post_create_setup_failure_records_the_known_instance_before_raising(tmp_path):
    class SetupFailedVast(RecordingVast):
        def create_instance(self, request, *, timeout_seconds):
            self.created.append(request)
            raise ProviderCreatedButSetupFailed(self.instance_id)

        def find_instance_by_idempotency_key(self, key, *, timeout_seconds):
            raise AssertionError("known instance ID must not depend on reconciliation")

    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    remote = SetupFailedVast()
    controller = CappedVastController(ledger, VastAdapter(remote))
    run_id = ledger.new_smoke_run_id()

    with pytest.raises(VastPostCreateSetupFailed, match="post-create setup failed"):
        controller.rent(
            run_id=run_id,
            purpose="rollout-smoke",
            offer=offer(),
            projected_spend_usd="0.80",
            request={"offer_id": "90210", "template_id": "123"},
        )

    assert ledger.recorded_instance_id(run_id) == "9123456"


def test_reconcile_pending_recovers_a_restart_without_a_second_create(tmp_path):
    class PendingVast(RecordingVast):
        def find_instance_by_idempotency_key(self, key, *, timeout_seconds):
            return "9123456"

    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    run_id = ledger.new_smoke_run_id()
    ledger.authorize_rental(run_id, "training-smoke", offer(), "1.00")
    ledger.record_creation_attempt(run_id, run_id)
    assert CappedVastController(ledger, VastAdapter(PendingVast())).reconcile_pending(run_id) == "9123456"


@pytest.mark.parametrize(
    ("offer_id", "request_offer_id"),
    [
        ("90210", "90211"),
        ("90210", 90210),
        (90210, "90210"),
        ("90210", None),
    ],
)
def test_rental_rejects_unbound_or_invalid_offer_identity_before_any_mutation(
    tmp_path, offer_id, request_offer_id
):
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    remote = RecordingVast()
    controller = CappedVastController(ledger, VastAdapter(remote))

    with pytest.raises(LedgerError, match="offer_id"):
        controller.rent(
            run_id=ledger.new_smoke_run_id(),
            purpose="training-smoke",
            offer={**offer(), "offer_id": offer_id},
            projected_spend_usd="0.80",
            request={"offer_id": request_offer_id, "template_id": "123"},
        )

    assert remote.created == []
    assert list(ledger.records()) == []


def test_cap_rejection_prevents_vast_creation(tmp_path):
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    remote = RecordingVast()
    controller = CappedVastController(ledger, VastAdapter(remote))
    existing = ledger.new_smoke_run_id()
    ledger.authorize_rental(existing, "training-smoke", offer(), "5.00")
    ledger.record_actual_spend(existing, "4.95")

    with pytest.raises(Exception, match="5.00"):
        controller.rent(
            run_id=ledger.new_smoke_run_id(),
            purpose="rollout-smoke",
            offer=offer(),
            projected_spend_usd="0.06",
            request={"offer_id": "90210"},
        )

    assert remote.created == []


@pytest.mark.parametrize(
    "target",
    ["", "*", "$VAST_INSTANCE_ID", "${VAST_INSTANCE_ID}", "status=running", "9123456,9123457", "47198086"],
)
def test_destroy_rejects_broad_empty_environment_and_protected_targets(tmp_path, target):
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    remote = RecordingVast()
    controller = CappedVastController(ledger, VastAdapter(remote))
    run_id = ledger.new_smoke_run_id()
    ledger.authorize_rental(run_id, "training-smoke", offer(), "0.25")
    ledger.record_instance(run_id, "9123456")

    with pytest.raises(InstanceTargetRejected):
        controller.destroy(run_id, target, timeout_seconds=30)

    assert remote.destroyed == []


def test_destroy_requires_the_exact_instance_id_recorded_for_its_run(tmp_path):
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    remote = RecordingVast()
    controller = CappedVastController(ledger, VastAdapter(remote))
    run_id = ledger.new_smoke_run_id()
    other_run = ledger.new_smoke_run_id()
    ledger.authorize_rental(run_id, "training-smoke", offer(), "0.25")
    ledger.record_instance(run_id, "9123456")
    ledger.authorize_rental(other_run, "rollout-smoke", offer(), "0.25")
    ledger.record_instance(other_run, "9123457")

    with pytest.raises(InstanceTargetRejected, match="recorded"):
        controller.destroy(run_id, "9123457", timeout_seconds=30)

    controller.destroy(run_id, "9123456", timeout_seconds=30)
    assert remote.destroyed == ["9123456"]
    assert PROTECTED_INSTANCE_IDS == frozenset({"47198086"})


def test_destroy_requires_a_bounded_provider_deadline(tmp_path):
    class DeadlineRecordingVast(RecordingVast):
        def __init__(self):
            super().__init__()
            self.timeouts = []

        def destroy_instance(self, instance_id: str, *, timeout_seconds: int) -> None:
            self.timeouts.append(timeout_seconds)
            super().destroy_instance(instance_id, timeout_seconds=timeout_seconds)

    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    remote = DeadlineRecordingVast()
    controller = CappedVastController(ledger, VastAdapter(remote))
    run_id = ledger.new_smoke_run_id()
    ledger.authorize_rental(run_id, "training-smoke", offer(), "0.25")
    ledger.record_instance(run_id, "9123456")

    with pytest.raises(InstanceTargetRejected, match="timeout"):
        controller.destroy(run_id, "9123456", timeout_seconds=60)
    controller.destroy(run_id, "9123456", timeout_seconds=30)

    assert remote.timeouts == [30]


def test_create_and_reconciliation_deadlines_are_rejected_before_reserving_or_calling_provider(tmp_path):
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    remote = RecordingVast()
    controller = CappedVastController(ledger, VastAdapter(remote))

    with pytest.raises(VastCreateAmbiguous, match="timeout"):
        controller.rent(
            run_id=ledger.new_smoke_run_id(),
            purpose="training-smoke",
            offer=offer(),
            projected_spend_usd="0.25",
            request={"offer_id": "90210"},
            create_timeout_seconds=60,
        )

    assert remote.created == []
    assert list(ledger.records()) == []
