#!/usr/bin/env python3
"""Emit a secret-free Isaac Sim 5.1 host compatibility receipt."""

from __future__ import annotations

import json

from lehome.flywheel.runtime_preflight import inspect_isaac_sim_5_1_runtime


def main() -> int:
    receipt = inspect_isaac_sim_5_1_runtime()
    print(json.dumps(receipt.as_dict(), sort_keys=True, separators=(",", ":")))
    return 0 if receipt.compatible else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
