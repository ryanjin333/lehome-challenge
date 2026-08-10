"""A narrow, injectable Vast boundary for individually recorded smoke IDs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Protocol

from .ledger import LedgerError, RentalLedger


PROTECTED_INSTANCE_IDS = frozenset({"47198086"})
_EXACT_INSTANCE_ID_RE = re.compile(r"^[1-9][0-9]*$")


class InstanceTargetRejected(LedgerError):
    """Raised before a deletion target is allowed to reach the Vast adapter."""


class VastCreateAmbiguous(LedgerError):
    """A create may have succeeded remotely and must remain reserved until reconciled."""


class VastCreateFailed(LedgerError):
    """A deterministic create failure released its reservation."""


class VastPostCreateSetupFailed(LedgerError):
    """A known paid instance was recorded after its required setup failed."""


class ProviderNotCreated(Exception):
    """Provider guarantees this request created no remote instance."""


class ProviderCreatedButSetupFailed(Exception):
    """Provider returned an exact instance ID before local post-create setup failed."""

    def __init__(self, instance_id: str) -> None:
        if (
            not isinstance(instance_id, str)
            or _EXACT_INSTANCE_ID_RE.fullmatch(instance_id) is None
            or instance_id in PROTECTED_INSTANCE_IDS
        ):
            raise ValueError("post-create failure requires one safe exact instance ID")
        self.instance_id = instance_id
        super().__init__("provider instance post-create setup failed")


class VastClient(Protocol):
    """The only remote operations this boundary allows a caller to inject.

    Implementations must make create, reconciliation, and destroy return or
    raise within their supplied deadlines; this controller intentionally
    creates no cancellation thread or subprocess that could outlive a paid
    reservation or cleanup attempt.
    """

    def create_instance(self, request: Mapping[str, Any], *, timeout_seconds: int) -> str: ...

    def find_instance_by_idempotency_key(self, key: str, *, timeout_seconds: int) -> str | None: ...

    def destroy_instance(self, instance_id: str, *, timeout_seconds: int) -> None: ...


class VastAdapter:
    """Thin adapter deliberately free of credentials, CLI calls, and discovery APIs."""

    def __init__(self, client: VastClient):
        self._client = client

    def create_instance(self, request: Mapping[str, Any], idempotency_key: str, *, timeout_seconds: int) -> str | None:
        _validate_provider_timeout(timeout_seconds, "create")
        payload = dict(request)
        payload["idempotency_key"] = idempotency_key
        instance_id = self._client.create_instance(payload, timeout_seconds=timeout_seconds)
        if instance_id is None:
            return None
        if not isinstance(instance_id, str) or not _EXACT_INSTANCE_ID_RE.fullmatch(instance_id):
            raise InstanceTargetRejected("Vast create did not return one exact numeric ID")
        if instance_id in PROTECTED_INSTANCE_IDS:
            raise InstanceTargetRejected("Vast create returned a protected LeHome instance ID")
        return instance_id

    def reconcile_instance(self, idempotency_key: str, *, timeout_seconds: int) -> str | None:
        _validate_provider_timeout(timeout_seconds, "reconciliation")
        instance_id = self._client.find_instance_by_idempotency_key(idempotency_key, timeout_seconds=timeout_seconds)
        if instance_id is None:
            return None
        if not isinstance(instance_id, str) or not _EXACT_INSTANCE_ID_RE.fullmatch(instance_id):
            raise VastCreateAmbiguous("Vast reconciliation did not return one exact numeric ID")
        if instance_id in PROTECTED_INSTANCE_IDS:
            raise VastCreateAmbiguous("Vast reconciliation returned a protected LeHome instance ID")
        return instance_id

    def destroy_instance(self, instance_id: str, *, timeout_seconds: int) -> None:
        _validate_destroy_target(instance_id)
        _validate_destroy_timeout(timeout_seconds)
        # The injected provider boundary owns transport cancellation.  Requiring
        # its bounded timeout avoids orphan worker threads/processes here.
        self._client.destroy_instance(instance_id, timeout_seconds=timeout_seconds)


class CappedVastController:
    """Coordinates pre-rental accounting with the intentionally small Vast adapter."""

    def __init__(self, ledger: RentalLedger, vast: VastAdapter):
        self.ledger = ledger
        self.vast = vast

    def rent(
        self,
        *,
        run_id: str,
        purpose: str,
        offer: Mapping[str, Any],
        projected_spend_usd: Decimal | str | int | float,
        request: Mapping[str, Any],
        create_timeout_seconds: int = 30,
        reconcile_timeout_seconds: int = 30,
    ) -> str:
        """Append a cap-approved receipt before making the injected create call."""
        _validate_provider_timeout(create_timeout_seconds, "create")
        _validate_provider_timeout(reconcile_timeout_seconds, "reconciliation")
        _validate_offer_binding(offer, request)
        self.ledger.authorize_rental(run_id, purpose, offer, projected_spend_usd)
        self.ledger.record_creation_attempt(run_id, run_id)
        try:
            instance_id = self.vast.create_instance(request, run_id, timeout_seconds=create_timeout_seconds)
        except ProviderNotCreated as error:
            self.ledger.release_reservation(run_id, "deterministic-create-failure")
            raise VastCreateFailed("Vast create failed before an ambiguous response") from error
        except ProviderCreatedButSetupFailed as error:
            self._record_instance(
                run_id,
                error.instance_id,
                reconcile_timeout_seconds=reconcile_timeout_seconds,
            )
            raise VastPostCreateSetupFailed(
                "Vast post-create setup failed after the exact instance was recorded"
            ) from error
        except Exception as error:
            instance_id = self.vast.reconcile_instance(run_id, timeout_seconds=reconcile_timeout_seconds)
            if instance_id is None:
                raise VastCreateAmbiguous("Vast create failed ambiguously; reservation remains") from error
        if instance_id is None:
            instance_id = self.vast.reconcile_instance(run_id, timeout_seconds=reconcile_timeout_seconds)
            if instance_id is None:
                raise VastCreateAmbiguous("Vast create is ambiguous; reservation remains until reconciliation")
        instance_id = self._record_instance(
            run_id,
            instance_id,
            reconcile_timeout_seconds=reconcile_timeout_seconds,
        )
        return instance_id

    def _record_instance(
        self,
        run_id: str,
        instance_id: str,
        *,
        reconcile_timeout_seconds: int,
    ) -> str:
        try:
            self.ledger.record_instance(run_id, instance_id)
        except Exception as append_error:
            recovered_id = self.vast.reconcile_instance(
                run_id, timeout_seconds=reconcile_timeout_seconds
            )
            if recovered_id is None:
                raise VastCreateAmbiguous(
                    "instance ID is ambiguous; reservation remains until reconciliation"
                ) from append_error
            if recovered_id != instance_id:
                raise VastCreateAmbiguous(
                    "reconciliation ID differs from create response"
                ) from append_error
            self.ledger.record_instance(run_id, recovered_id)
            instance_id = recovered_id
        return instance_id

    def reconcile_pending(self, run_id: str, *, timeout_seconds: int = 30) -> str:
        _validate_provider_timeout(timeout_seconds, "reconciliation")
        recorded = self.ledger.recorded_instance_id(run_id)
        if recorded is not None:
            return recorded
        instance_id = self.vast.reconcile_instance(run_id, timeout_seconds=timeout_seconds)
        if instance_id is None:
            raise VastCreateAmbiguous("pending create remains unresolved; reservation is retained")
        self.ledger.record_instance(run_id, instance_id)
        return instance_id

    def destroy(self, run_id: str, instance_id: str, *, timeout_seconds: int) -> None:
        """Destroy only the one numeric ID previously recorded for this run."""
        _validate_destroy_target(instance_id)
        _validate_destroy_timeout(timeout_seconds)
        recorded_instance_id = self.ledger.recorded_instance_id(run_id)
        if recorded_instance_id is None or instance_id != recorded_instance_id:
            raise InstanceTargetRejected(
                "destruction requires the exact instance ID recorded for this run"
            )
        self.vast.destroy_instance(instance_id, timeout_seconds=timeout_seconds)
        self.ledger.record_lifecycle_evidence(
            run_id,
            "destroy-requested",
            {"instance_id": instance_id},
        )


def _validate_destroy_target(instance_id: str) -> None:
    if not isinstance(instance_id, str) or not _EXACT_INSTANCE_ID_RE.fullmatch(instance_id):
        raise InstanceTargetRejected(
            "destruction target must be one explicit numeric instance ID; "
            "queries, globs, empty values, and environment expansions are forbidden"
        )
    if instance_id in PROTECTED_INSTANCE_IDS:
        raise InstanceTargetRejected("protected LeHome instance IDs can never be destroyed")


def _validate_destroy_timeout(timeout_seconds: int) -> None:
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or not 0 < timeout_seconds < 60:
        raise InstanceTargetRejected("destroy timeout must be a positive bounded integer under 60 seconds")


def _validate_provider_timeout(timeout_seconds: int, operation: str) -> None:
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or not 0 < timeout_seconds < 60:
        raise VastCreateAmbiguous(f"{operation} timeout must be a positive bounded integer under 60 seconds")


def _validate_offer_binding(offer: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    if not isinstance(offer, Mapping) or not isinstance(request, Mapping):
        raise LedgerError("offer and create request must be objects with an offer_id")
    offer_id = offer.get("offer_id")
    request_offer_id = request.get("offer_id")
    if not isinstance(offer_id, str) or not offer_id:
        raise LedgerError("offer_id in the offer snapshot must be a non-empty string")
    if not isinstance(request_offer_id, str) or not request_offer_id:
        raise LedgerError("offer_id in the create request must be a non-empty string")
    if request_offer_id != offer_id:
        raise LedgerError("create request offer_id must exactly match the offer snapshot")
