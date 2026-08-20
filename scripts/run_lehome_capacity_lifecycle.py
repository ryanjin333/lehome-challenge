#!/usr/bin/env python3
"""Run the opt-in fixed-VM capacity daemon.

This command never creates, replaces, or deletes compute.  It is inert unless
an operator supplies ``--execute`` and a root-owned config containing the
three predeclared instance IDs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from lehome_train.groot.experiment_capacity import (
    CapacityLifecycle,
    HttpCapacityController,
    NebiusCliInstanceRunner,
    load_root_owned_capacity_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--controller-url", required=True)
    parser.add_argument("--controller-ca-file", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--receipt-log", type=Path, required=True)
    parser.add_argument("--nebius-config-file", type=Path, required=True)
    parser.add_argument("--nebius-timeout-seconds", type=int, default=30)
    parser.add_argument("--nebius-max-attempts", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-backoff-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing capacity actions without --execute")
    if (
        args.poll_seconds < 1
        or not 1 <= args.max_backoff_seconds <= 300
        or not 1 <= args.nebius_timeout_seconds <= 120
        or not 1 <= args.nebius_max_attempts <= 3
    ):
        raise ValueError("capacity lifecycle arguments are invalid")
    if not all(path.is_absolute() for path in (args.config, args.controller_ca_file, args.token_file, args.receipt_log, args.nebius_config_file)):
        raise ValueError("capacity lifecycle paths must be absolute")

    lifecycle = CapacityLifecycle(
        load_root_owned_capacity_config(args.config),
        HttpCapacityController(args.controller_url, args.token_file, args.controller_ca_file),
        # The production adapter's complete provider surface is
        # instance_state/start_instance/stop_instance; it has no provisioning
        # or deletion entry point.
        NebiusCliInstanceRunner(
            timeout_seconds=args.nebius_timeout_seconds,
            max_attempts=args.nebius_max_attempts,
            provider_config_file=args.nebius_config_file,
            provider_config_owner_uid=0,
        ),
    )
    backoff = 1
    while True:
        receipt = lifecycle.reconcile(now_ns=time.time_ns())
        lifecycle.append_receipt(args.receipt_log, receipt)
        if args.once:
            return
        if receipt.status == "ok":
            backoff = 1
            time.sleep(args.poll_seconds)
        else:
            time.sleep(backoff)
            backoff = min(args.max_backoff_seconds, backoff * 2)


if __name__ == "__main__":
    main()
