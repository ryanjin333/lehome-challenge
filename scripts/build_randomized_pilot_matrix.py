#!/usr/bin/env python3
"""Build the deterministic geometry-randomized rollout pilot matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


CATEGORY_PREFIXES = {
    "top_long": "Top_Long",
    "top_short": "Top_Short",
    "pant_long": "Pant_Long",
    "pant_short": "Pant_Short",
}
GEOMETRY_STRATEGIES = ("mild_geometry", "strong_geometry")
PILOT_SEEDS = (
    211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
    269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
)


def build_matrix(
    *,
    category: str,
    garment_count: int = 10,
    strategies: Sequence[str] = GEOMETRY_STRATEGIES,
    seeds: Sequence[int] = PILOT_SEEDS,
) -> list[dict[str, object]]:
    """Return one immutable row per garment/profile with unique perturbation seeds."""

    try:
        prefix = CATEGORY_PREFIXES[category]
    except KeyError as error:
        raise ValueError(f"unsupported pilot category: {category}") from error
    if not isinstance(garment_count, int) or isinstance(garment_count, bool) or garment_count <= 0:
        raise ValueError("garment_count must be a positive integer")
    if tuple(strategies) != GEOMETRY_STRATEGIES:
        raise ValueError("pilot strategies must be the pinned mild/strong geometry profiles")
    required_seeds = garment_count * len(strategies)
    if len(seeds) != required_seeds or any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds
    ):
        raise ValueError("pilot requires one unique non-negative seed per attempt")
    if len(set(seeds)) != len(seeds):
        raise ValueError("pilot seeds must be unique")

    rows: list[dict[str, object]] = []
    seed_index = 0
    for garment_index in range(garment_count):
        garment = f"{prefix}_Seen_{garment_index}"
        garment_slug = garment.lower().replace("_", "-")
        for strategy in strategies:
            seed = int(seeds[seed_index])
            seed_index += 1
            strategy_slug = strategy.replace("_", "-")
            attempt_id = f"{garment_slug}-{strategy_slug}-seed-{seed}"
            rows.append({
                "attempt_id": attempt_id,
                "trial_id": attempt_id,
                "garment": garment,
                "garment_name": garment,
                "category": category,
                "release_stage": "seen",
                "difficulty": "randomized",
                "strategy": strategy,
                "seed": seed,
            })
    return rows


def encode_matrix(rows: Sequence[dict[str, object]]) -> bytes:
    """Canonical on-disk representation used by the checked-in SHA-256 pin."""

    return (json.dumps(list(rows), indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_matrix(output: Path, rows: Sequence[dict[str, object]]) -> tuple[Path, Path]:
    """Write a new matrix and adjacent digest, refusing silent replacement."""

    encoded = encode_matrix(rows)
    digest = hashlib.sha256(encoded).hexdigest()
    digest_path = output.with_name(output.name + ".sha256")
    for path in (output, digest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable pilot artifact: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    digest_path.write_text(digest + "\n", encoding="ascii")
    return output, digest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=tuple(CATEGORY_PREFIXES), default="top_short")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_matrix(args.output, build_matrix(category=args.category))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
