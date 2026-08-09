import json
import multiprocessing
from decimal import Decimal

import pytest

import b1k_deploy.ledger as ledger_module
from b1k_deploy.ledger import (
    CostCapExceeded,
    LedgerError,
    RentalLedger,
    SecretReceiptValue,
)


def offer() -> dict[str, object]:
    return {
        "offer_id": "90210",
        "hourly_rate_usd": "0.40",
        "gpu_name": "RTX 4090",
        "verified": True,
    }


def _concurrent_authorize(path: str, gate, outcomes) -> None:
    ledger = RentalLedger(path)
    gate.wait()
    try:
        ledger.authorize_rental(ledger.new_smoke_run_id(), "training-smoke", offer(), "3.00")
    except CostCapExceeded:
        outcomes.put("rejected")
    else:
        outcomes.put("authorized")


def test_plan_records_a_fsynced_append_only_offer_receipt(tmp_path):
    ledger_path = tmp_path / "cost-ledger.jsonl"
    ledger = RentalLedger(ledger_path)

    run_id = ledger.new_smoke_run_id()
    ledger.authorize_rental(
        run_id=run_id,
        purpose="training-smoke",
        offer=offer(),
        projected_spend_usd="1.20",
    )

    lines = ledger_path.read_text().splitlines()
    assert len(lines) == 1
    receipt = json.loads(lines[0])
    assert receipt["event"] == "rental-authorized"
    assert receipt["run_id"] == run_id
    assert receipt["offer"]["offer_id"] == "90210"
    assert receipt["offer"]["hourly_rate_usd"] == "0.40"
    assert receipt["projected_spend_usd"] == "1.20"
    assert receipt["cumulative_actual_spend_usd"] == "0.00"

    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write("{\"external\": true}\n")
    ledger.record_lifecycle_evidence(run_id, "destroyed", {"vast_absent": True})
    assert len(ledger_path.read_text().splitlines()) == 3


def test_smoke_run_ids_are_unique_and_ledger_rejects_reuse(tmp_path):
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    first = ledger.new_smoke_run_id()
    second = ledger.new_smoke_run_id()

    assert first != second
    ledger.authorize_rental(first, "training-smoke", offer(), "0.25")
    with pytest.raises(ValueError, match="already exists"):
        ledger.authorize_rental(first, "training-smoke", offer(), "0.25")


def test_cumulative_actual_spend_cap_rejects_before_a_receipt_is_mutated(tmp_path):
    ledger_path = tmp_path / "cost-ledger.jsonl"
    ledger = RentalLedger(ledger_path)
    first = ledger.new_smoke_run_id()
    ledger.authorize_rental(first, "training-smoke", offer(), "5.00")
    ledger.record_actual_spend(first, "4.90")
    before = ledger_path.read_bytes()

    with pytest.raises(CostCapExceeded, match="5.00"):
        ledger.authorize_rental(
            ledger.new_smoke_run_id(), "rollout-smoke", offer(), "0.11"
        )

    assert ledger_path.read_bytes() == before


def test_outstanding_authorizations_reserve_the_cumulative_cap_before_append(tmp_path):
    ledger_path = tmp_path / "cost-ledger.jsonl"
    ledger = RentalLedger(ledger_path)
    ledger.authorize_rental(ledger.new_smoke_run_id(), "training-smoke", offer(), "3.00")
    before = ledger_path.read_bytes()

    with pytest.raises(CostCapExceeded, match="reserved"):
        ledger.authorize_rental(
            ledger.new_smoke_run_id(), "rollout-smoke", offer(), "2.01"
        )

    assert ledger_path.read_bytes() == before


def test_final_actual_spend_replaces_its_reservation_for_the_next_smoke(tmp_path):
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    first = ledger.new_smoke_run_id()
    ledger.authorize_rental(first, "training-smoke", offer(), "3.00")
    assert ledger.record_actual_spend(first, "2.25") is False

    ledger.authorize_rental(
        ledger.new_smoke_run_id(), "rollout-smoke", offer(), "2.75"
    )

    records = list(ledger.records())
    assert records[1]["released_projected_spend_usd"] == "3.00"
    assert records[1]["cumulative_reserved_spend_usd"] == "2.25"
    assert ledger.reserved_spend_usd() == Decimal("5.00")


def test_over_cap_actual_spend_is_durably_recorded_and_closes_future_rentals(tmp_path):
    ledger_path = tmp_path / "cost-ledger.jsonl"
    ledger = RentalLedger(ledger_path)
    first = ledger.new_smoke_run_id()
    second = ledger.new_smoke_run_id()
    ledger.authorize_rental(first, "training-smoke", offer(), "3.00")
    ledger.authorize_rental(second, "rollout-smoke", offer(), "2.00")

    assert ledger.record_actual_spend(second, "2.01") is True

    actual_receipt = list(ledger.records())[-1]
    assert actual_receipt["actual_spend_usd"] == "2.01"
    assert actual_receipt["released_projected_spend_usd"] == "2.00"
    assert actual_receipt["cumulative_actual_spend_usd"] == "2.01"
    assert actual_receipt["cumulative_reserved_spend_usd"] == "5.01"
    assert actual_receipt["cap_breached"] is True

    with pytest.raises(CostCapExceeded, match="cap has already been breached"):
        ledger.authorize_rental(
            ledger.new_smoke_run_id(), "rollout-smoke", offer(), "0.00"
        )


def test_actual_cumulative_over_cap_closes_future_rentals_even_without_reservations(tmp_path):
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    run_id = ledger.new_smoke_run_id()
    ledger.authorize_rental(run_id, "training-smoke", offer(), "5.00")

    assert ledger.record_actual_spend(run_id, "5.01") is True
    assert ledger.accumulated_actual_spend_usd() == Decimal("5.01")
    assert ledger.reserved_spend_usd() == Decimal("5.01")

    with pytest.raises(CostCapExceeded, match="cap has already been breached"):
        ledger.authorize_rental(
            ledger.new_smoke_run_id(), "rollout-smoke", offer(), "0.00"
        )


def test_receipts_track_exact_instance_actual_spend_and_lifecycle_evidence(tmp_path):
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    run_id = ledger.new_smoke_run_id()
    ledger.authorize_rental(run_id, "rollout-smoke", offer(), "0.80")
    ledger.record_instance(run_id, "9123456")
    ledger.record_actual_spend(run_id, Decimal("0.47"))
    ledger.record_lifecycle_evidence(
        run_id,
        "disappearance-verified",
        {"vast_absent": True, "ssh_unreachable": True},
    )

    records = list(ledger.records())
    assert [record["event"] for record in records] == [
        "rental-authorized",
        "instance-recorded",
        "actual-spend-recorded",
        "lifecycle-evidence-recorded",
    ]
    assert ledger.recorded_instance_id(run_id) == "9123456"
    assert ledger.accumulated_actual_spend_usd() == Decimal("0.47")
    assert records[-1]["evidence"] == {
        "vast_absent": True,
        "ssh_unreachable": True,
    }


def test_secret_bearing_receipt_values_are_rejected(tmp_path):
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")

    with pytest.raises(SecretReceiptValue):
        ledger.authorize_rental(
            ledger.new_smoke_run_id(),
            "training-smoke",
            {**offer(), "api_key": "do-not-record"},
            "0.25",
        )


@pytest.mark.parametrize(
    "credential",
    [
        ("hf_" + "abcdefghijklmnopqrstuvwxyz0123456789"),
        ("vast_" + "abcdefghijklmnopqrstuvwxyz0123456789"),
        ("dckr_pat_" + "abcdefghijklmnopqrstuvwxyz0123456789"),
        ("ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789AB"),
        ("github_pat_" + "abcdefghijklmnopqrstuvwxyz0123456789AB"),
        ("glpat-" + "abcdefghijklmnopqrstuvwxyz0123456789"),
        ("AKIA" + "1234567890ABCDEF"),
        ("xoxb-" + "1234567890-abcdefghijklmnopqrstuvwxyz"),
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
    ],
)
def test_common_raw_credential_shapes_are_rejected_under_innocuous_evidence_keys(
    tmp_path, credential
):
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    run_id = ledger.new_smoke_run_id()
    ledger.authorize_rental(run_id, "training-smoke", offer(), "0.25")

    with pytest.raises(SecretReceiptValue):
        ledger.record_lifecycle_evidence(run_id, "runtime-ready", {"stderr": credential})


def test_first_ledger_creation_fsyncs_the_parent_directory(tmp_path, monkeypatch):
    calls: list[int] = []
    original_fsync = ledger_module.os.fsync

    def record_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(ledger_module.os, "fsync", record_fsync)
    ledger = RentalLedger(tmp_path / "ledger" / "cost-ledger.jsonl")
    ledger.authorize_rental(ledger.new_smoke_run_id(), "training-smoke", offer(), "0.25")

    assert len(calls) >= 2


def test_ledger_path_must_not_be_a_directory(tmp_path):
    ledger_path = tmp_path / "cost-ledger.jsonl"
    ledger_path.mkdir()
    ledger = RentalLedger(ledger_path)

    with pytest.raises(LedgerError, match="regular file"):
        ledger.authorize_rental(ledger.new_smoke_run_id(), "training-smoke", offer(), "0.25")


@pytest.mark.parametrize("amount", ["0.001", "1.999"])
def test_spend_values_with_more_than_two_decimal_places_are_rejected(tmp_path, amount):
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    with pytest.raises(LedgerError, match="two decimal"):
        ledger.authorize_rental(ledger.new_smoke_run_id(), "training-smoke", offer(), amount)


def test_cap_cannot_be_overridden_above_the_five_dollar_limit(tmp_path):
    with pytest.raises(LedgerError, match="cannot exceed"):
        RentalLedger(tmp_path / "cost-ledger.jsonl", cap_usd="5.01")


def test_interprocess_lock_allows_only_one_concurrent_three_dollar_authorization(tmp_path):
    context = multiprocessing.get_context("fork")
    gate = context.Event()
    outcomes = context.Queue()
    path = str(tmp_path / "cost-ledger.jsonl")
    workers = [context.Process(target=_concurrent_authorize, args=(path, gate, outcomes)) for _ in range(2)]
    for worker in workers:
        worker.start()
    gate.set()
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0
    assert sorted(outcomes.get(timeout=1) for _ in workers) == ["authorized", "rejected"]


def test_lifecycle_evidence_is_allowlisted_and_scans_assignments_before_rejecting(tmp_path):
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    run_id = ledger.new_smoke_run_id()
    ledger.authorize_rental(run_id, "training-smoke", offer(), "0.25")

    with pytest.raises(SecretReceiptValue):
        ledger.record_lifecycle_evidence(run_id, "runtime-ready", {"stderr": "VAST_API_KEY=plain-secret"})
    with pytest.raises(LedgerError, match="allowlisted"):
        ledger.record_lifecycle_evidence(run_id, "runtime-ready", {"stderr": "normal text"})


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_money_is_rejected(tmp_path, amount):
    with pytest.raises(LedgerError):
        RentalLedger(tmp_path / "cost-ledger.jsonl", cap_usd=amount)


@pytest.mark.parametrize(
    "assignment",
    [
        "OPENAI_API_KEY=plain-secret",
        "API_KEY=plain-secret",
        "TOKEN=plain-secret",
        "PASSWORD=plain-secret",
        "SECRET=plain-secret",
        "CREDENTIAL=plain-secret",
    ],
)
def test_generic_secret_assignment_is_rejected_in_permitted_evidence_field(
    tmp_path, assignment
):
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    run_id = ledger.new_smoke_run_id()
    ledger.authorize_rental(run_id, "training-smoke", offer(), "0.25")
    with pytest.raises(SecretReceiptValue):
        ledger.record_lifecycle_evidence(run_id, "runtime-ready", {"status": assignment})


def test_late_actual_spend_after_release_remains_truthful(tmp_path):
    ledger = RentalLedger(tmp_path / "cost-ledger.jsonl")
    run_id = ledger.new_smoke_run_id()
    ledger.authorize_rental(run_id, "training-smoke", offer(), "1.00")
    ledger.release_reservation(run_id, "provider-confirmed-not-created")
    ledger.record_actual_spend(run_id, "0.35")
    assert ledger.accumulated_actual_spend_usd() == Decimal("0.35")


def test_descriptor_walk_rejects_an_intermediate_symlink(tmp_path):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    (trusted / "jump").symlink_to(tmp_path / "elsewhere")
    ledger = RentalLedger(trusted / "jump" / "ledger.jsonl")
    with pytest.raises(LedgerError):
        ledger.authorize_rental(ledger.new_smoke_run_id(), "training-smoke", offer(), "0.25")
