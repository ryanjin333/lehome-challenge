"""Strict immutable training-experiment identity for the LeHome flywheel."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Mapping

from lehome_train.io import canonical_json_sha256


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_OCI = re.compile(r"^ghcr\.io/ryanjin333/lehome-groot-n17-trainer@sha256:[0-9a-f]{64}$")
_HELD_OUTS = (
    "Top_Long_Unseen_1", "Top_Short_Unseen_1",
    "Pant_Long_Unseen_1", "Pant_Short_Unseen_1",
)
_CAMPAIGN = {
    "isaac_groot_revision": "23ace64f17aa5015259b8609d371eb61a357c776",
    "trainer_oci": "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746",
    "parent_repository": "ryanjin333/lehome-groot-n17-models",
    "parent_revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
    "parent_subpath": "policies/step-12000",
    "archive_sha256": "0ddd4e7ce351dd2172cd1edd967293a50d02c15c0f2c21ca39db94692a57e0b5",
    "artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
}


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _reject_constant(_: str) -> object:
    raise ValueError("non-finite JSON number")


def _exact(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} has unknown or missing field")
    return value


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be SHA-256")
    return value


def _revision(value: object, label: str) -> str:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        raise ValueError(f"{label} must be an immutable revision")
    return value


def _relative(value: object, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{label} must be a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a safe relative path")
    return value


def _positive(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class BundleBinding:
    repository: str
    revision: str
    prefix: str
    tree_sha256: str
    manifest_sha256: str
    garment_index_path: str | None = None
    garment_index_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    identity_sha256: str
    bc_bundle: BundleBinding
    rollout_bundle: BundleBinding
    mixture_manifest_sha256: str
    train_lineage_sha256: str
    validation_lineage_sha256: str
    weights: Mapping[str, int]
    quotas: Mapping[str, int]
    held_out_garments: tuple[str, ...]
    raw: Mapping[str, object]


def _bundle(value: object, *, expected_prefix: str, label: str, garment_index: bool = False) -> BundleBinding:
    keys = {"repository", "revision", "prefix", "tree_sha256", "manifest_sha256"}
    if garment_index:
        keys |= {"garment_index_path", "garment_index_sha256"}
    item = _exact(value, keys, label)
    repository = item["repository"]
    if type(repository) is not str or repository != "ryanjin333/lehome-groot-n17-data":
        raise ValueError(f"{label} repository is not approved")
    prefix = _relative(item["prefix"], f"{label} prefix")
    if prefix != expected_prefix:
        raise ValueError(f"{label} prefix drift")
    return BundleBinding(
        str(repository), _revision(item["revision"], f"{label} revision"), prefix,
        _sha(item["tree_sha256"], f"{label} tree hash"), _sha(item["manifest_sha256"], f"{label} manifest hash"),
        _relative(item["garment_index_path"], f"{label} garment index path") if garment_index else None,
        _sha(item["garment_index_sha256"], f"{label} garment index hash") if garment_index else None,
    )


def _weights(value: object) -> tuple[Mapping[str, int], Mapping[str, int]]:
    item = _exact(value, {"bc", "rollout", "dagger"}, "mixture weights")
    bc, rollout, dagger = item["bc"], item["rollout"], item["dagger"]
    if type(bc) is not int or type(rollout) is not int or type(dagger) is not int:
        raise ValueError("mixture weights must be integers")
    if bc <= 0 or rollout <= 0 or dagger != 0 or bc + rollout != 100:
        raise ValueError("mixture weights must be BC/rollout positive, sum to 100, and DAgger zero")
    weights = MappingProxyType({"bc": bc, "rollout": rollout, "dagger": 0})
    quotas = MappingProxyType(batch64_quotas(weights))
    return weights, quotas


def batch64_quotas(weights: Mapping[str, int]) -> dict[str, int]:
    """Largest-remainder per-global-batch source slots, with stable ties."""

    kinds = ("bc", "rollout", "dagger")
    base = {kind: weights[kind] * 64 // 100 for kind in kinds}
    remaining = 64 - sum(base.values())
    for kind in sorted(kinds, key=lambda item: (-(weights[item] * 64 % 100), kinds.index(item)))[:remaining]:
        base[kind] += 1
    return base


def _training(value: object) -> None:
    item = _exact(value, {"action_horizon", "global_batch_size", "physical_batch_size", "max_steps", "local_checkpoint_steps", "hf_checkpoint_steps", "loader_candidates"}, "training")
    fixed = {"action_horizon": 16, "global_batch_size": 64, "physical_batch_size": 64, "max_steps": 2000}
    if any(item[key] != expected or type(item[key]) is not int for key, expected in fixed.items()):
        raise ValueError("training campaign invariant drift")
    expected_lists = {
        "local_checkpoint_steps": [500, 1000, 1500, 2000],
        "hf_checkpoint_steps": [1000, 2000],
        "loader_candidates": [0, 4, 8, 12, 16],
    }
    if any(
        not isinstance(item[key], list)
        or any(type(number) is not int for number in item[key])
        or item[key] != expected
        for key, expected in expected_lists.items()
    ):
        raise ValueError("training checkpoint or loader invariant drift")


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    """Load one immutable flywheel experiment manifest with exact campaign pins."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("experiment manifest is missing or unsafe")
    try:
        document = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("experiment manifest is malformed") from error
    item = _exact(document, {"schema_version", "kind", "campaign", "bc_bundle", "rollout_bundle", "mixture_manifest_sha256", "lineage", "mixture_weights", "training", "held_out_garments", "destinations"}, "experiment manifest")
    if (
        type(item["schema_version"]) is not int
        or item["schema_version"] != 1
        or type(item["kind"]) is not str
        or item["kind"] != "lehome_training_experiment"
    ):
        raise ValueError("experiment manifest schema is unsupported")
    campaign = _exact(item["campaign"], set(_CAMPAIGN), "campaign")
    if campaign != _CAMPAIGN or _OCI.fullmatch(str(campaign["trainer_oci"])) is None:
        raise ValueError("campaign pin drift")
    bc = _bundle(item["bc_bundle"], expected_prefix="bc/full", label="BC bundle", garment_index=True)
    rollout = _bundle(item["rollout_bundle"], expected_prefix="rollouts/round-1", label="rollout bundle")
    lineage = _exact(item["lineage"], {"train_sha256", "validation_sha256"}, "lineage")
    train, validation = _sha(lineage["train_sha256"], "train lineage hash"), _sha(lineage["validation_sha256"], "validation lineage hash")
    if train == validation:
        raise ValueError("train and validation lineages must be separate")
    weights, quotas = _weights(item["mixture_weights"])
    _training(item["training"])
    if item["held_out_garments"] != list(_HELD_OUTS):
        raise ValueError("held-out garment set drift")
    destinations = _exact(item["destinations"], {"hf_checkpoints_repository", "hf_model_repository"}, "destinations")
    if any(type(value) is not str or not value or "/" not in value or value.startswith("/") for value in destinations.values()):
        raise ValueError("HF destinations must be non-empty repositories")
    # The semantic canonical identity ignores insignificant whitespace but binds
    # every immutable input, including the runtime scheduling choice.
    identity = canonical_json_sha256(item)
    return ExperimentManifest(identity, bc, rollout, _sha(item["mixture_manifest_sha256"], "mixture manifest hash"), train, validation, weights, quotas, tuple(_HELD_OUTS), MappingProxyType(dict(item)))
