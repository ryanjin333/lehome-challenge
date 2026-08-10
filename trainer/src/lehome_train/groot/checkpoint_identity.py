"""Content identity for the exact safetensors weights loaded by GR00T."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


_SHARD_NAME = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")


def _sha256_regular_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint artifact must be a materialized regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_artifact_sha256(policy_path: str | Path) -> str:
    """Hash the monolithic or complete indexed weights the policy loads."""

    root = Path(policy_path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("parent checkpoint root must be a materialized directory")
    monolithic = root / "model.safetensors"
    index = root / "model.safetensors.index.json"
    if monolithic.exists() or monolithic.is_symlink():
        if index.exists() or index.is_symlink() or any(
            _SHARD_NAME.fullmatch(path.name) for path in root.iterdir()
        ):
            raise ValueError("parent checkpoint has ambiguous weight layouts")
        return _sha256_regular_file(monolithic)
    if index.is_symlink() or not index.is_file():
        raise ValueError("parent checkpoint index is invalid")
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise TypeError
        shard_names = sorted(set(weight_map.values()))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        raise ValueError("parent checkpoint index is invalid") from None
    parsed: list[tuple[int, int]] = []
    for name in shard_names:
        if not isinstance(name, str):
            raise ValueError("parent checkpoint index has an invalid shard name")
        match = _SHARD_NAME.fullmatch(name)
        if match is None or Path(name).name != name:
            raise ValueError("parent checkpoint index has an unsafe shard name")
        parsed.append((int(match.group(1)), int(match.group(2))))
    totals = {total for _, total in parsed}
    if len(totals) != 1:
        raise ValueError("parent checkpoint index has inconsistent shard totals")
    total = totals.pop()
    if (
        total <= 0
        or len(shard_names) != total
        or {number for number, _ in parsed} != set(range(1, total + 1))
    ):
        raise ValueError("parent checkpoint index has an incomplete shard set")
    discovered = {
        path.name for path in root.iterdir() if _SHARD_NAME.fullmatch(path.name)
    }
    referenced = set(shard_names)
    if discovered != referenced:
        raise ValueError("parent checkpoint shard set differs from its index")
    manifest = {
        "schema_version": 1,
        "files": [
            {"path": name, "sha256": _sha256_regular_file(root / name)}
            for name in (index.name, *shard_names)
        ],
    }
    return hashlib.sha256(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


__all__ = ["policy_artifact_sha256"]
