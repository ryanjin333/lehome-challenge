"""Deterministic, strict, and crash-safe local artifact I/O."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Mapping, TypeVar

from lehome_train.models import StrictModel, model_from_mapping


T = TypeVar("T", bound=StrictModel)
_DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024


def _canonical_value(value: object) -> object:
    if isinstance(value, StrictModel):
        return value.to_dict()
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {
            key: _canonical_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical JSON numbers must be finite")
        return value
    raise TypeError("value is not supported by the canonical JSON contract")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize supported values as deterministic canonical UTF-8 JSON."""

    normalized = _canonical_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Return the SHA-256 identity of a canonical JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_non_finite_json(_: str) -> object:
    raise ValueError("JSON numbers must be finite")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate field in JSON object")
        value[key] = item
    return value


def parse_json(
    model_type: type[T],
    payload: str | bytes | bytearray | Mapping[object, object],
) -> T:
    """Strictly parse a typed JSON contract.

    Malformed JSON, non-finite numbers, missing fields, unknown fields, and
    mismatched nested types all fail closed.
    """

    if isinstance(payload, Mapping):
        decoded: object = payload
    else:
        try:
            if isinstance(payload, (bytes, bytearray)):
                text = bytes(payload).decode("utf-8")
            elif isinstance(payload, str):
                text = payload
            else:
                raise TypeError("payload must be JSON text, bytes, or an object")
            decoded = json.loads(
                text,
                parse_constant=_reject_non_finite_json,
                object_pairs_hook=_strict_object,
            )
        except UnicodeDecodeError as error:
            raise ValueError("invalid UTF-8 JSON") from error
        except json.JSONDecodeError as error:
            raise ValueError("invalid JSON") from error

    if not isinstance(decoded, Mapping):
        raise ValueError("typed JSON root must be an object")
    return model_from_mapping(model_type, decoded)


def load_json(model_type: type[T], path: str | os.PathLike[str]) -> T:
    """Load one strict typed UTF-8 JSON file."""

    return parse_json(model_type, Path(path).read_bytes())


def sha256_file(
    path: str | os.PathLike[str],
    *,
    chunk_size: int = _DEFAULT_HASH_CHUNK_SIZE,
) -> str:
    """Hash a file incrementally without loading it into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_parent(parent: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(
    destination: str | os.PathLike[str],
    value: object,
) -> None:
    """Atomically replace a JSON artifact after fully syncing a sibling temp.

    A failure before ``os.replace`` leaves any existing destination untouched,
    and every failure path removes the temporary file.
    """

    path = Path(destination)
    parent = path.parent
    if not parent.is_dir():
        raise FileNotFoundError("atomic JSON destination parent does not exist")

    payload = canonical_json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent(parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
