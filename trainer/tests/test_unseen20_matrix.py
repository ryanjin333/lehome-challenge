"""Frozen development matrix is isolated from the final public-unseen rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_dev20_is_canonical_disjoint_and_balanced() -> None:
    dev_path = ROOT / "configs/eval_groot_n17_unseen20_dev.json"
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    final = json.loads((ROOT / "configs/eval_groot_n17_public_280.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(dev_path.read_bytes()).hexdigest() == (dev_path.with_suffix(".json.sha256")).read_text().strip()
    assert len(dev["trials"]) == 20
    assert {row["category"] for row in dev["trials"]} == {"top_long", "top_short", "pant_long", "pant_short"}
    assert {(row["garment_name"], row["seed"]) for row in dev["trials"]}.isdisjoint({(row["garment_name"], row["seed"]) for row in final["trials"] if row["release_stage"] == "public_unseen"})
