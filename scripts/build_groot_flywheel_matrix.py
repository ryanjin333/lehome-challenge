"""Build or check the immutable 280-trial GR00T public matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from lehome.flywheel.matrix import (
    build_public_matrix,
    canonical_matrix_json,
    matrix_sha256,
    validate_release_assets,
    write_public_matrix,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("Assets/objects/Challenge_Garment/Release"),
        help="Release directory containing per-category garment lists",
    )
    parser.add_argument("--check", action="store_true", help="fail if output is absent or differs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    matrix = build_public_matrix()
    validate_release_assets(args.assets_root, matrix)
    expected = canonical_matrix_json(matrix)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != expected:
            print(f"matrix differs: {args.output}", file=sys.stderr)
            return 1
    else:
        write_public_matrix(args.output, matrix)
    print(f"sha256={matrix_sha256(matrix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
