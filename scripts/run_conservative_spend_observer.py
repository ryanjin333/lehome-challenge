#!/usr/bin/env python3
"""Write a local conservative upper-bound spend receipt without provider calls."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import json
import math
import os
from pathlib import Path
import signal
import stat
import tempfile
import threading
from typing import Any


_RECEIPT_KIND = "lehome_spend_observation_v1"
_STATE_KIND = "lehome_conservative_spend_observer_state_v1"
_OWNER_ONLY = 0o600
_CENT = Decimal("0.01")


class ObserverError(ValueError):
    """The observer cannot safely issue another receipt."""


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ObserverError(f"{label} is malformed")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ObserverError(f"{label} is malformed") from error
    if parsed.tzinfo != UTC:
        raise ObserverError(f"{label} is malformed")
    return parsed


def _decimal(value: object, *, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ObserverError(f"{label} is malformed")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ObserverError(f"{label} is malformed") from error
    if not parsed.is_finite() or parsed < 0:
        raise ObserverError(f"{label} is malformed")
    return parsed


def validate_interval(value: object) -> float:
    if isinstance(value, bool):
        raise ObserverError("interval is malformed")
    try:
        interval = float(value)
    except (TypeError, ValueError) as error:
        raise ObserverError("interval is malformed") from error
    if not math.isfinite(interval) or interval <= 0 or interval > 30:
        raise ObserverError("interval must be in (0, 30] seconds")
    return interval


def _safe_parent(path: Path) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.parent.is_dir():
        raise ObserverError("output parent is unsafe")
    for current in (path.parent, *path.parent.parents):
        try:
            info = current.lstat()
        except OSError as error:
            raise ObserverError("output parent is unsafe") from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ObserverError("output parent is unsafe")


def _regular_owned(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise ObserverError(f"{label} is missing or unsafe") from error
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != _OWNER_ONLY):
        raise ObserverError(f"{label} is missing or unsafe")


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    _safe_parent(path)
    if path.exists() or path.is_symlink():
        _regular_owned(path, label="output")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), _OWNER_ONLY)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _safe_parent(path)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _state_path(output: Path) -> Path:
    return output.with_name(output.name + ".state.json")


def _state_payload(*, baseline: Decimal, baseline_at: datetime, rate: Decimal, observer_name: str) -> dict[str, object]:
    return {
        "schema_version": 1, "kind": _STATE_KIND, "observer": observer_name,
        "baseline_usd": format(baseline, "f"), "baseline_observed_at_utc": _format_utc(baseline_at),
        "max_hourly_burn_usd": format(rate, "f"),
    }


def _read_state(path: Path) -> dict[str, object]:
    _regular_owned(path, label="state")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ObserverError("state is malformed") from error
    if not isinstance(payload, dict):
        raise ObserverError("state is malformed")
    return payload


def _read_receipt(path: Path) -> dict[str, object]:
    _regular_owned(path, label="output")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ObserverError("output is malformed") from error
    required = {"schema_version", "kind", "observer", "observed_at_utc", "spent_usd"}
    if (not isinstance(payload, dict) or set(payload) != required or payload.get("schema_version") != 1
            or payload.get("kind") != _RECEIPT_KIND or not isinstance(payload.get("observer"), str)
            or not payload["observer"]):
        raise ObserverError("output is malformed")
    _parse_utc(payload["observed_at_utc"], label="output timestamp")
    _decimal(payload["spent_usd"], label="output spend")
    return payload


def _elapsed_seconds(start: datetime, end: datetime) -> Decimal:
    delta = end - start
    if delta.total_seconds() < 0:
        raise ObserverError("clock regressed before provider baseline")
    return Decimal(delta.days * 86_400 + delta.seconds) + Decimal(delta.microseconds) / Decimal(1_000_000)


def _conservative_spend(*, baseline: Decimal, baseline_at: datetime, rate: Decimal, now: datetime) -> Decimal:
    amount = baseline + _elapsed_seconds(baseline_at, now) * rate / Decimal(3600)
    # The extra cent keeps the JSON float representation strictly above the
    # exact estimate even at an exact cent boundary.
    return amount.quantize(_CENT, rounding=ROUND_CEILING) + _CENT


def write_observation(
    *, output: Path, baseline_usd: object, baseline_observed_at: datetime,
    max_hourly_burn_usd: object, now: datetime, observer_name: str,
) -> dict[str, object]:
    """Durably write one receipt, never reducing an existing conservative bound."""
    if not isinstance(observer_name, str) or not observer_name:
        raise ObserverError("observer is malformed")
    if baseline_observed_at.tzinfo != UTC or now.tzinfo != UTC:
        raise ObserverError("timestamps must be UTC")
    baseline, rate = _decimal(baseline_usd, label="baseline"), _decimal(max_hourly_burn_usd, label="rate")
    if rate <= 0:
        raise ObserverError("rate is malformed")
    if baseline_observed_at > now:
        raise ObserverError("provider baseline is in the future")
    _elapsed_seconds(baseline_observed_at, now)
    _safe_parent(output)
    state_path = _state_path(output)
    expected_state = _state_payload(baseline=baseline, baseline_at=baseline_observed_at, rate=rate, observer_name=observer_name)
    if state_path.exists() or state_path.is_symlink():
        if _read_state(state_path) != expected_state:
            raise ObserverError("baseline state changed or regressed")
    elif output.exists() or output.is_symlink():
        raise ObserverError("output exists without durable baseline state")
    else:
        _atomic_write(state_path, _canonical(expected_state))

    conservative = _conservative_spend(baseline=baseline, baseline_at=baseline_observed_at, rate=rate, now=now)
    payload: dict[str, object] = {
        "schema_version": 1, "kind": _RECEIPT_KIND, "observer": observer_name,
        "observed_at_utc": _format_utc(now), "spent_usd": float(conservative),
    }
    if output.exists() or output.is_symlink():
        previous = _read_receipt(output)
        if previous["observer"] != observer_name:
            raise ObserverError("observer changed")
        if (_parse_utc(previous["observed_at_utc"], label="output timestamp") > now
                or _decimal(previous["spent_usd"], label="output spend") > conservative):
            raise ObserverError("output would regress")
    _atomic_write(output, _canonical(payload))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-usd", required=True)
    parser.add_argument("--baseline-observed-at-utc", required=True)
    parser.add_argument("--max-hourly-burn-usd", required=True)
    parser.add_argument("--interval-seconds", type=float, default=30)
    parser.add_argument("--observer", default="lehome-conservative-local-upper-bound-v1")
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        interval = validate_interval(args.interval_seconds)
        baseline_at = _parse_utc(args.baseline_observed_at_utc, label="baseline timestamp")
        stopped = threading.Event()
        def request_stop(_signum: int, _frame: Any) -> None: stopped.set()
        signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
        while not stopped.is_set():
            write_observation(
                output=args.output, baseline_usd=args.baseline_usd, baseline_observed_at=baseline_at,
                max_hourly_burn_usd=args.max_hourly_burn_usd, now=datetime.now(UTC), observer_name=args.observer,
            )
            if args.once: break
            stopped.wait(interval)
    except ObserverError as error:
        _parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
