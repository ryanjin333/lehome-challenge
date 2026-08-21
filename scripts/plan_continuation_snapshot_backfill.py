#!/usr/bin/env python3
"""Report whether a sealed recovery audit can feed the H16 snapshot contract.

This is deliberately a planner, not a migration: it never synthesizes a
physical Snapshot from annotations or a replay prefix.  Pre-v3 audits are
historical evidence only and require fresh autonomous source collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


_CONTRACT = "authenticated_full_snapshot_at_fresh_h16_policy_boundary_before_action"


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _read(path: Path) -> Mapping[str, object]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("recovery audit must be an absolute regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("recovery audit is malformed") from error
    if not isinstance(value, Mapping):
        raise ValueError("recovery audit is malformed")
    return value


def plan_continuation_snapshot_backfill(*, audit_path: str | Path) -> dict[str, object]:
    """Return a fail-closed source-collection plan for an existing audit."""

    audit_file = Path(audit_path)
    audit = _read(audit_file)
    admitted = audit.get("admitted_episodes")
    if not isinstance(admitted, list):
        raise ValueError("recovery audit admitted episode list is malformed")
    existing_ids = [row.get("source_episode_id") for row in admitted if isinstance(row, Mapping)]
    if len(existing_ids) != len(admitted) or any(not isinstance(value, str) or not value for value in existing_ids):
        raise ValueError("recovery audit admitted episode identity is malformed")
    if len(set(existing_ids)) != len(existing_ids):
        raise ValueError("recovery audit admitted episode IDs collide")

    v3 = (
        audit.get("schema_version") == 3
        and audit.get("kind") == "lehome_successful_recovery_audit"
        and audit.get("continuation_contract") == _CONTRACT
    )
    # A sealed v2 audit has no selected snapshot file/hash binding.  Even if a
    # later filesystem happens to contain a similarly named JSON file, it is
    # not immutable evidence for those historic rows and remains ineligible.
    eligible = len(audit.get("selected_recoveries", [])) if v3 and isinstance(audit.get("selected_recoveries"), list) else 0
    document: dict[str, object] = {
        "schema_version": 1,
        "kind": "lehome_continuation_snapshot_backfill_plan",
        "audit_file_sha256": hashlib.sha256(audit_file.read_bytes()).hexdigest(),
        "audit_schema_version": audit.get("schema_version"),
        "existing_accepted_success_count": len(admitted),
        "eligible_snapshot_contract_source_count": eligible,
        "legacy_audit_ineligible": not v3,
        "action": "fresh_autonomous_success_collection_required" if not v3 else "audit_v3_sources_may_be_materialized",
        "required_source_artifact": {
            "path_template": "snapshots/continuations/{step:06d}.json",
            "capture_timing": "before_action_at_every_positive_h16_policy_boundary",
            "authentication": "SHA256SUMS.json+sealed_release+audit_snapshot_sha256",
            "annotation_binding": "source_seed+category+garment+step+policy_request_id+annotation_state_equals_snapshot_robot_position",
        },
    }
    document["semantic_sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args(argv)
    print(_canonical(plan_continuation_snapshot_backfill(audit_path=args.audit)).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
