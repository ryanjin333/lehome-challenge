"""Append-only, secret-free accounting for paid BEHAVIOR-1K smoke runs."""

from __future__ import annotations

import json
import os
import re
import stat
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import fcntl


CAP_USD = Decimal("5.00")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_INSTANCE_ID_RE = re.compile(r"^[1-9][0-9]*$")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth(?:orization)?|credential|"
    r"password|private[_-]?key|secret|ssh[_-]?key|token)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE_RE = re.compile(
    r"""(?ix)
    (?:
        (?<![A-Z0-9_])(?:[A-Z][A-Z0-9_]*)?(?:TOKEN|API_KEY|PASSWORD|SECRET|CREDENTIAL)\s*=
        | -----BEGIN[ ]+[A-Z ]*PRIVATE[ ]+KEY-----
        | \bBearer\s+
        | (?<![A-Z0-9_])hf_[A-Z0-9]{20,}(?![A-Z0-9_])
        | (?<![A-Z0-9_])(?:vast|vastai|vst)_[A-Z0-9_-]{16,}(?![A-Z0-9_-])
        | (?<![A-Z0-9_])(?:dckr_pat|docker_pat)_[A-Z0-9_-]{16,}(?![A-Z0-9_-])
        | (?<![A-Z0-9_])gh[pousr]_[A-Z0-9]{20,}(?![A-Z0-9_])
        | (?<![A-Z0-9_])github_pat_[A-Z0-9_]{20,}(?![A-Z0-9_])
        | (?<![A-Z0-9_-])glpat-[A-Z0-9_-]{20,}(?![A-Z0-9_-])
        | \bAKIA[0-9A-Z]{16}\b
        | (?<![A-Z0-9_-])xox(?:b|p|a|s)-[A-Z0-9-]{20,}(?![A-Z0-9_-])
        | \beyJ[A-Z0-9_-]{5,}\.[A-Z0-9_-]{2,}\.[A-Z0-9_-]{2,}\b
    )
    """,
)


class LedgerError(ValueError):
    """Raised when a receipt would be invalid or unsafe."""


class CostCapExceeded(LedgerError):
    """Raised before a new rental authorization would exceed the USD cap."""


class SecretReceiptValue(LedgerError):
    """Raised instead of persisting a credential-bearing value."""


class LedgerIntegrityError(LedgerError):
    """Raised when this controller's persisted receipt stream is malformed."""


def _money(value: Decimal | str | int | float) -> Decimal:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise LedgerError(f"invalid USD amount: {value!r}") from error
    if not amount.is_finite() or amount < Decimal("0.00"):
        raise LedgerError("USD amount must be a finite non-negative value")
    if amount.as_tuple().exponent < -2:
        raise LedgerError("USD amount cannot have more than two decimal places")
    return amount


def _rate(value: Decimal | str | int | float) -> str:
    try:
        rate = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise LedgerError(f"invalid hourly rate: {value!r}") from error
    if not rate.is_finite() or rate < 0 or rate.as_tuple().exponent < -6:
        raise LedgerError("hourly rate must be non-negative with at most six decimal places")
    return format(rate, "f")


def _locked_transition(method):
    def wrapped(self, *args, **kwargs):
        with self._locked():
            return method(self, *args, **kwargs)

    return wrapped


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_receipt_value(value: Any, path: str = "receipt") -> Any:
    """Return a JSON-safe value, rejecting credential-shaped content first."""
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LedgerError(f"{path} keys must be strings")
            if _SENSITIVE_KEY_RE.search(key):
                raise SecretReceiptValue(f"secret-bearing receipt key rejected: {key}")
            safe[key] = _safe_receipt_value(item, f"{path}.{key}")
        return safe
    if isinstance(value, (list, tuple)):
        return [_safe_receipt_value(item, f"{path}[]") for item in value]
    if isinstance(value, str):
        if _SENSITIVE_VALUE_RE.search(value):
            raise SecretReceiptValue(f"secret-bearing receipt value rejected at {path}")
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Decimal):
        return format(value, ".2f")
    raise LedgerError(f"{path} is not JSON serializable")


_LIFECYCLE_EVIDENCE_KEYS = frozenset({
    "vast_absent", "ssh_unreachable", "instance_id", "template_id", "image_digest", "hub_commit", "status",
})


def _safe_lifecycle_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    safe = _safe_receipt_value(evidence, "evidence")
    if not isinstance(safe, dict):
        raise LedgerError("lifecycle evidence must be an object")
    unexpected = set(safe) - _LIFECYCLE_EVIDENCE_KEYS
    if unexpected:
        raise LedgerError("lifecycle evidence keys must be allowlisted")
    if not safe:
        raise LedgerError("lifecycle evidence cannot be empty")
    for key, value in safe.items():
        if key in {"vast_absent", "ssh_unreachable"} and not isinstance(value, bool):
            raise LedgerError("boolean lifecycle evidence has an invalid type")
        if key not in {"vast_absent", "ssh_unreachable"} and not isinstance(value, str):
            raise LedgerError("string lifecycle evidence has an invalid type")
    return safe


class RentalLedger:
    """A local JSONL ledger whose only write operation is a durable append."""

    def __init__(self, path: str | Path, *, cap_usd: Decimal | str = CAP_USD):
        self.path = Path(path).absolute()
        self.cap_usd = _money(cap_usd)
        if self.cap_usd > CAP_USD:
            raise LedgerError("ledger cap cannot exceed USD 5.00")

    def new_smoke_run_id(self) -> str:
        """Create a collision-resistant identifier for one isolated smoke run."""
        while True:
            run_id = f"b1k-smoke-{uuid.uuid4().hex}"
            if run_id not in self._known_run_ids():
                return run_id

    def records(self) -> Iterator[dict[str, Any]]:
        """Yield the append stream in order without rewriting or compacting it."""
        self._validate_storage_path()
        try:
            parent_descriptor = self._open_parent_directory(create=False)
        except FileNotFoundError:
            return
        try:
            try:
                descriptor = os.open(self.path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_descriptor)
            except FileNotFoundError:
                return
        finally:
            os.close(parent_descriptor)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise LedgerError("ledger path must be a regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise LedgerIntegrityError(f"blank JSONL receipt at line {line_number}")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise LedgerIntegrityError(
                        f"invalid JSONL receipt at line {line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise LedgerIntegrityError(
                        f"JSONL receipt at line {line_number} is not an object"
                    )
                yield record

    @_locked_transition
    def authorize_rental(
        self,
        run_id: str,
        purpose: str,
        offer: Mapping[str, Any],
        projected_spend_usd: Decimal | str | int | float,
    ) -> None:
        """Durably authorize one rental before any remote create call happens."""
        self._validate_new_run(run_id)
        if purpose not in {"training-smoke", "rollout-smoke"}:
            raise LedgerError("rental purpose must be training-smoke or rollout-smoke")
        if not isinstance(offer, Mapping):
            raise LedgerError("offer snapshot must be an object")
        safe_offer = _safe_receipt_value(offer, "offer")
        if not isinstance(safe_offer, dict):  # Defensive; Mapping always produces a dict.
            raise LedgerError("offer snapshot must be an object")
        allowed_offer_keys = {"offer_id", "hourly_rate_usd", "verified", "gpu_name"}
        if set(safe_offer) - allowed_offer_keys:
            raise LedgerError("offer snapshot contains unsupported fields")
        if not isinstance(safe_offer.get("offer_id"), str) or not safe_offer["offer_id"]:
            raise LedgerError("offer snapshot requires a non-empty offer_id")
        if not isinstance(safe_offer.get("verified"), bool):
            raise LedgerError("offer snapshot requires a boolean verified field")
        hourly_rate = _rate(safe_offer.get("hourly_rate_usd", ""))
        projected = _money(projected_spend_usd)
        accumulated = self.accumulated_actual_spend_usd()
        reserved = self.reserved_spend_usd()
        if accumulated > self.cap_usd or reserved > self.cap_usd:
            raise CostCapExceeded(
                f"USD {self.cap_usd:.2f} cap has already been breached: "
                f"actual USD {accumulated:.2f}, reserved USD {reserved:.2f}"
            )
        if reserved + projected > self.cap_usd:
            raise CostCapExceeded(
                f"USD {self.cap_usd:.2f} cumulative cap would be exceeded: "
                f"reserved USD {reserved:.2f} + projected USD {projected:.2f}"
            )
        safe_offer["hourly_rate_usd"] = hourly_rate
        self._append(
            {
                "event": "rental-authorized",
                "run_id": run_id,
                "purpose": purpose,
                "offer": safe_offer,
                "projected_spend_usd": format(projected, ".2f"),
                "cumulative_actual_spend_usd": format(accumulated, ".2f"),
                "cumulative_reserved_spend_usd": format(reserved + projected, ".2f"),
                "cap_usd": format(self.cap_usd, ".2f"),
            }
        )

    @_locked_transition
    def record_instance(self, run_id: str, instance_id: str) -> None:
        """Bind a newly created run to one exact Vast instance ID."""
        self._require_known_run(run_id)
        self._validate_instance_id(instance_id)
        if self.recorded_instance_id(run_id) is not None:
            raise LedgerError(f"run {run_id!r} already has a recorded instance ID")
        self._append(
            {
                "event": "instance-recorded",
                "run_id": run_id,
                "instance_id": instance_id,
            }
        )

    @_locked_transition
    def record_actual_spend(
        self, run_id: str, actual_spend_usd: Decimal | str | int | float
    ) -> bool:
        """Persist a final charge and report whether it breached the cap.

        Actual charges have already occurred, so this method never discards a
        valid charge because it is over budget.  The returned flag is visible
        only after the receipt's durable append succeeds.
        """
        self._require_known_run(run_id)
        actual_total, outstanding = self._spend_state()
        if any(record.get("event") == "actual-spend-recorded" and record.get("run_id") == run_id for record in self.records()):
            raise LedgerError(f"run {run_id!r} already has actual spend recorded")
        actual = _money(actual_spend_usd)
        released = self._projection_for(run_id)
        outstanding.pop(run_id, None)
        new_reserved = actual_total + sum(outstanding.values(), Decimal("0.00")) + actual
        new_actual_total = actual_total + actual
        cap_breached = new_actual_total > self.cap_usd or new_reserved > self.cap_usd
        self._append(
            {
                "event": "actual-spend-recorded",
                "run_id": run_id,
                "actual_spend_usd": format(actual, ".2f"),
                "released_projected_spend_usd": format(released, ".2f"),
                "cumulative_actual_spend_usd": format(new_actual_total, ".2f"),
                "cumulative_reserved_spend_usd": format(new_reserved, ".2f"),
                "cap_breached": cap_breached,
            }
        )
        return cap_breached

    @_locked_transition
    def record_lifecycle_evidence(
        self, run_id: str, lifecycle: str, evidence: Mapping[str, Any]
    ) -> None:
        """Append cleanup/readiness evidence without exposing credentials."""
        self._require_known_run(run_id)
        if not isinstance(lifecycle, str) or not lifecycle.strip():
            raise LedgerError("lifecycle state must be a non-empty string")
        safe_evidence = _safe_lifecycle_evidence(evidence)
        self._append(
            {
                "event": "lifecycle-evidence-recorded",
                "run_id": run_id,
                "lifecycle": lifecycle,
                "evidence": safe_evidence,
            }
        )

    @_locked_transition
    def record_creation_attempt(self, run_id: str, idempotency_key: str) -> None:
        self._require_known_run(run_id)
        if idempotency_key != run_id:
            raise LedgerError("creation idempotency key must equal the smoke run ID")
        if any(
            record.get("event") == "creation-attempt-recorded" and record.get("run_id") == run_id
            for record in self.records()
        ):
            raise LedgerError("creation attempt already recorded for this run")
        self._append({"event": "creation-attempt-recorded", "run_id": run_id, "idempotency_key": idempotency_key})

    @_locked_transition
    def release_reservation(self, run_id: str, reason: str) -> None:
        self._require_known_run(run_id)
        if not isinstance(reason, str) or not reason:
            raise LedgerError("reservation release requires a reason")
        if self.recorded_instance_id(run_id) is not None:
            raise LedgerError("reservation cannot be released after an instance is recorded")
        _, outstanding = self._spend_state()
        if run_id not in outstanding:
            raise LedgerError("only outstanding reservations can be released")
        self._append({"event": "reservation-released", "run_id": run_id, "reason": reason})

    def recorded_instance_id(self, run_id: str) -> str | None:
        for record in self.records():
            if record.get("event") == "instance-recorded" and record.get("run_id") == run_id:
                instance_id = record.get("instance_id")
                if not isinstance(instance_id, str):
                    raise LedgerIntegrityError("recorded instance ID is not a string")
                self._validate_instance_id(instance_id)
                return instance_id
        return None

    def accumulated_actual_spend_usd(self) -> Decimal:
        actual_total, _ = self._spend_state()
        return actual_total

    def reserved_spend_usd(self) -> Decimal:
        """Return final actual spend plus unfinalized rental reservations."""
        actual_total, outstanding = self._spend_state()
        return (actual_total + sum(outstanding.values(), Decimal("0.00"))).quantize(
            Decimal("0.01")
        )

    def _spend_state(self) -> tuple[Decimal, dict[str, Decimal]]:
        authorizations: dict[str, Decimal] = {}
        final_actuals: dict[str, Decimal] = {}
        released: set[str] = set()
        for record in self.records():
            event = record.get("event")
            run_id = record.get("run_id")
            if event == "rental-authorized":
                value = record.get("projected_spend_usd")
                if not isinstance(run_id, str) or not isinstance(value, str):
                    raise LedgerIntegrityError("rental authorization is missing run ID or projection")
                if run_id in authorizations:
                    raise LedgerIntegrityError("duplicate rental authorization run ID")
                authorizations[run_id] = _money(value)
            elif event == "actual-spend-recorded":
                value = record.get("actual_spend_usd")
                if not isinstance(run_id, str) or not isinstance(value, str):
                    raise LedgerIntegrityError("actual spend receipt is missing run ID or amount")
                if run_id not in authorizations or run_id in final_actuals:
                    raise LedgerIntegrityError("actual spend receipt has no unique authorization")
                final_actuals[run_id] = _money(value)
            elif event == "reservation-released":
                if not isinstance(run_id, str) or run_id not in authorizations:
                    raise LedgerIntegrityError("reservation release has no authorization")
                released.add(run_id)
        outstanding = {
            run_id: projection
            for run_id, projection in authorizations.items()
            if run_id not in final_actuals and run_id not in released
        }
        return sum(final_actuals.values(), Decimal("0.00")).quantize(Decimal("0.01")), outstanding

    def _projection_for(self, run_id: str) -> Decimal:
        for record in self.records():
            if record.get("event") == "rental-authorized" and record.get("run_id") == run_id:
                value = record.get("projected_spend_usd")
                if isinstance(value, str):
                    return _money(value)
        raise LedgerIntegrityError("actual spend receipt has no authorization")

    def _known_run_ids(self) -> set[str]:
        return {
            record["run_id"]
            for record in self.records()
            if record.get("event") == "rental-authorized"
            and isinstance(record.get("run_id"), str)
        }

    def _validate_new_run(self, run_id: str) -> None:
        if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
            raise LedgerError("run ID must be a safe, explicit identifier")
        if run_id in self._known_run_ids():
            raise LedgerError(f"run ID {run_id!r} already exists in this ledger")

    def _require_known_run(self, run_id: str) -> None:
        if run_id not in self._known_run_ids():
            raise LedgerError(f"run ID {run_id!r} is not authorized in this ledger")

    def _validate_storage_path(self) -> None:
        if self.path.name in {"", ".", ".."}:
            raise LedgerError("ledger path must name a regular file")

    def _prepare_parent(self) -> None:
        descriptor = self._open_parent_directory(create=True)
        os.close(descriptor)

    @contextmanager
    def _locked(self):
        self._prepare_parent()
        lock_name = f"{self.path.name}.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor = self._open_parent_directory(create=True)
        try:
            descriptor = os.open(lock_name, flags, 0o600, dir_fd=parent_descriptor)
        finally:
            os.close(parent_descriptor)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
                raise LedgerError("ledger lock must be a private regular file owned by this user")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    @staticmethod
    def _validate_instance_id(instance_id: str) -> None:
        if not isinstance(instance_id, str) or not _INSTANCE_ID_RE.fullmatch(instance_id):
            raise LedgerError("instance ID must be an explicit positive numeric Vast ID")

    def _append(self, payload: Mapping[str, Any]) -> None:
        record = {
            "timestamp": _utc_now(),
            **_safe_receipt_value(payload),
        }
        encoded = (
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        self._prepare_parent()
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        created = False
        parent_descriptor = self._open_parent_directory(create=True)
        try:
            try:
                descriptor = os.open(self.path.name, flags | os.O_EXCL, 0o600, dir_fd=parent_descriptor)
                created = True
            except FileExistsError:
                descriptor = os.open(self.path.name, flags, 0o600, dir_fd=parent_descriptor)
            try:
                written = 0
                while written < len(encoded):
                    count = os.write(descriptor, encoded[written:])
                    if count <= 0:
                        raise OSError("could not append complete ledger record")
                    written += count
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if created:
                os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)

    def _open_parent_directory(self, *, create: bool) -> int:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path.anchor, flags)
        except OSError as error:
            raise LedgerError(f"could not open trusted ledger root: {error}") from error
        try:
            for component in self.path.parent.parts[1:]:
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                    child = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            return descriptor
        except FileNotFoundError:
            os.close(descriptor)
            raise
        except OSError as error:
            os.close(descriptor)
            raise LedgerError(f"unsafe ledger parent path: {error}") from error
        except Exception:
            os.close(descriptor)
            raise
