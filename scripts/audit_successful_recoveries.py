#!/usr/bin/env python3
"""Run the sealed successful-recovery audit without network or cloud access."""

from __future__ import annotations

import argparse
import json

from lehome_train.groot.recovery_audit import RecoveryThresholds, audit_successful_recoveries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-root", action="append", required=True)
    parser.add_argument("--receipts-root", action="append", required=True)
    parser.add_argument("--round-seal", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nontrivial-progress", type=float, default=0.15)
    parser.add_argument("--minimum-drawdown", type=float, default=0.05)
    parser.add_argument("--minimum-stall-steps", type=int, default=16)
    parser.add_argument("--minimum-recovery-gain", type=float, default=0.10)
    parser.add_argument("--improvement-epsilon", type=float, default=1e-6)
    parser.add_argument("--per-category-minimum", type=int, default=5)
    args = parser.parse_args()
    result = audit_successful_recoveries(
        accepted_roots=args.accepted_root, receipt_roots=args.receipts_root,
        round_seal_paths=args.round_seal, output_path=args.output,
        thresholds=RecoveryThresholds(
            nontrivial_progress=args.nontrivial_progress, minimum_drawdown=args.minimum_drawdown,
            minimum_stall_steps=args.minimum_stall_steps, minimum_recovery_gain=args.minimum_recovery_gain,
            improvement_epsilon=args.improvement_epsilon,
        ), per_category_minimum=args.per_category_minimum,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
