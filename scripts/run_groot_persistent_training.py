#!/usr/bin/env python3
"""Explicit, dry-run-by-default lifecycle gate for persistent corrective RFT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol

ORGANIZER_SOURCE = {
    "repository": "lehome/dataset_challenge_merged",
    "revision": "17e8dee8fac294ffd21d250501d3b31bf8679042",
    "subdir": "four_types_merged",
    "mirror_repository": "kunhsiang/lehome-four-types-merged",
    "mirror_revision": "2ebcccf528dec91cefac0c94a9214a83028ae6cc",
    "manifest_sha256": "bf8fbae82002a33ff304b9a70993bdfe1c678ba9e8f798c1ad370d58969435eb",
}
CORRECTIVE_SOURCE = {
    "revision": "e6cd1c182514c15271c805d03a646e7a4f95b17c",
    "prefix": "corrective-rft/b96be3db22174a12dab62a8a673f7c7d083f87aa7b50c4e03ee43e064da56c35",
}


class Provider(Protocol):
    def rent(self, request: dict[str, object]) -> object: ...


def _load_request(path: str) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("lifecycle request must be an object")
    return value


def destroy(*, instance_id: int, training_receipt: dict[str, object]) -> dict[str, object]:
    if training_receipt.get("instance_id") != instance_id or training_receipt.get("immutable_checkpoint_steps") != [1000, 2000]:
        raise ValueError("instance-bound disposal requires two immutable checkpoints")
    return {"paid_action": False, "destroy_authorized": True, "instance_id": instance_id}


def main_for_test(argv: list[str], *, provider: Provider | object | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "capture-offers", "rent", "stage", "tune", "train", "status", "resume", "destroy"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    request = _load_request(args.request)
    if args.action == "prepare":
        return {"paid_action": False, "action": "prepare", "organizer_source": ORGANIZER_SOURCE, "corrective_source": CORRECTIVE_SOURCE, "request": request}
    if not args.execute:
        return {"paid_action": False, "action": args.action, "dry_run": True, "request": request}
    if args.action != "rent":
        raise ValueError("provider actions are intentionally explicit and unsupported by this local gate")
    hourly = request.get("hourly_price")
    account_total = request.get("account_hourly_total")
    if type(hourly) not in (int, float) or type(account_total) not in (int, float) or hourly >= 1 or account_total > 2:
        raise ValueError("rent requires interruptible RTX PRO 6000 under $1/hr and account total at most $2/hr")
    if provider is None:
        raise ValueError("rent requires an explicit provider adapter")
    return {"paid_action": True, "action": "rent", "provider_result": provider.rent(request)}  # type: ignore[union-attr]


if __name__ == "__main__":
    print(json.dumps(main_for_test(__import__("sys").argv[1:]), sort_keys=True))
