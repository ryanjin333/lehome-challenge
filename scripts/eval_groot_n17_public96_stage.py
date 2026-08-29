"""Stage entrypoint that installs the raw checker before importing Isaac evaluation."""

from __future__ import annotations

import argparse
import sys

from scripts.groot_n17_public96_raw_checker import RAW_CHECKER_OVERLAY_ID, install_raw_checker_overlay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--public96_raw_checker_overlay", required=True)
    parsed, remaining = parser.parse_known_args(argv)
    if parsed.public96_raw_checker_overlay != RAW_CHECKER_OVERLAY_ID:
        raise SystemExit("public96 raw checker overlay identity is invalid")
    install_raw_checker_overlay()
    from scripts import eval as evaluator

    previous = sys.argv
    try:
        sys.argv = [previous[0], *remaining]
        evaluator.main()
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
