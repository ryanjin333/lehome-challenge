#!/usr/bin/env python3
"""Thin CLI for the shared fresh visual-only replay evidence contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_SOURCE = Path(__file__).resolve().parents[1] / "source" / "lehome"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from lehome.flywheel.fresh_replay_evidence import validate_exact_fresh_visual_only


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, required=True)
    parser.add_argument("--source-reports-json", required=True)
    parser.add_argument("--source-matrices-json", required=True)
    args = parser.parse_args(argv)
    try:
        validate_exact_fresh_visual_only(
            matrix_path=args.matrix,
            max_attempts=args.max_attempts,
            source_reports_json=args.source_reports_json,
            source_matrices_json=args.source_matrices_json,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
