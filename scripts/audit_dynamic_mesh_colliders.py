#!/usr/bin/env python3
"""CPU/static USD audit for the PhysX dynamic triangle-mesh restriction."""

from __future__ import annotations

import argparse
import json
import sys

from lehome.assets.collider_audit import audit_usd_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd_path", nargs="+", help="USD asset(s) to inspect")
    args = parser.parse_args()
    results = [audit_usd_file(path) for path in args.usd_path]
    print(json.dumps({"audits": results}, sort_keys=True))
    return 0 if all(result.get("healthy") is True for result in results) else 2


if __name__ == "__main__":
    sys.exit(main())
