from __future__ import annotations

import json
from pathlib import Path

import pytest


def _manifest(*, bc: int = 70, rollout: int = 30) -> dict[str, object]:
    digest = "a" * 64
    revision = "b" * 40
    return {
        "schema_version": 1,
        "kind": "lehome_training_experiment",
        "campaign": {
            "isaac_groot_revision": "23ace64f17aa5015259b8609d371eb61a357c776",
            "trainer_oci": "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746",
            "parent_repository": "ryanjin333/lehome-groot-n17-models",
            "parent_revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
            "parent_subpath": "policies/step-12000",
            "archive_sha256": "0ddd4e7ce351dd2172cd1edd967293a50d02c15c0f2c21ca39db94692a57e0b5",
            "artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
        },
        "bc_bundle": {"repository": "ryanjin333/lehome-groot-n17-data", "revision": revision, "prefix": "bc/full", "tree_sha256": digest, "manifest_sha256": digest, "garment_index_path": "garment-index.json", "garment_index_sha256": digest},
        "rollout_bundle": {"repository": "ryanjin333/lehome-groot-n17-rollouts", "revision": "c" * 40, "prefix": "rollouts/round-1", "tree_sha256": digest, "manifest_sha256": digest},
        "mixture_manifest_sha256": digest,
        "lineage": {"train_sha256": digest, "validation_sha256": "d" * 64},
        "mixture_weights": {"bc": bc, "rollout": rollout, "dagger": 0},
        "training": {"action_horizon": 16, "global_batch_size": 64, "physical_batch_size": 64, "max_steps": 2000, "local_checkpoint_steps": [500, 1000, 1500, 2000], "hf_checkpoint_steps": [1000, 2000], "loader_candidates": [0, 4, 8, 12, 16]},
        "held_out_garments": ["Top_Long_Unseen_1", "Top_Short_Unseen_1", "Pant_Long_Unseen_1", "Pant_Short_Unseen_1"],
        "destinations": {"hf_checkpoints_repository": "ryanjin333/lehome-rft-checkpoints", "hf_model_repository": "ryanjin333/lehome-rft-model"},
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def test_manifest_admits_campaign_pins_and_normalizes_the_70_30_schedule(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_manifest import load_experiment_manifest

    path = tmp_path / "experiment.json"
    _write(path, _manifest())

    manifest = load_experiment_manifest(path)

    assert manifest.weights == {"bc": 70, "rollout": 30, "dagger": 0}
    assert manifest.quotas == {"bc": 45, "rollout": 19, "dagger": 0}
    assert manifest.identity_sha256 == manifest.identity_sha256


def test_manifest_change_to_80_20_changes_identity_and_derives_batch64_schedule(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_manifest import load_experiment_manifest

    first, second = tmp_path / "70-30.json", tmp_path / "80-20.json"
    _write(first, _manifest())
    _write(second, _manifest(bc=80, rollout=20))

    original, changed = load_experiment_manifest(first), load_experiment_manifest(second)

    assert changed.quotas == {"bc": 51, "rollout": 13, "dagger": 0}
    assert original.identity_sha256 != changed.identity_sha256


def test_manifest_admits_the_pure_bc_sweep_control(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_manifest import load_experiment_manifest

    path = tmp_path / "experiment.json"
    _write(path, _manifest(bc=100, rollout=0))

    manifest = load_experiment_manifest(path)

    assert manifest.weights == {"bc": 100, "rollout": 0, "dagger": 0}
    assert manifest.quotas == {"bc": 64, "rollout": 0, "dagger": 0}


def test_manifest_accepts_a_fresh_immutable_rollout_round_prefix(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_manifest import load_experiment_manifest

    value = _manifest()
    value["rollout_bundle"]["prefix"] = "rollouts/round-2"  # type: ignore[index]
    path = tmp_path / "experiment.json"
    _write(path, value)

    assert load_experiment_manifest(path).rollout_bundle.prefix == "rollouts/round-2"


@pytest.mark.parametrize(
    ("weights", "quotas"),
    [
        ({"bc": 100, "rollout": 0, "dagger": 0}, {"bc": 64, "rollout": 0, "dagger": 0}),
        ({"bc": 95, "rollout": 5, "dagger": 0}, {"bc": 61, "rollout": 3, "dagger": 0}),
        ({"bc": 90, "rollout": 10, "dagger": 0}, {"bc": 58, "rollout": 6, "dagger": 0}),
        ({"bc": 85, "rollout": 15, "dagger": 0}, {"bc": 54, "rollout": 10, "dagger": 0}),
        ({"bc": 80, "rollout": 20, "dagger": 0}, {"bc": 51, "rollout": 13, "dagger": 0}),
        ({"bc": 70, "rollout": 30, "dagger": 0}, {"bc": 45, "rollout": 19, "dagger": 0}),
    ],
)
def test_batch64_sweep_profiles_include_the_bc_control(
    weights: dict[str, int], quotas: dict[str, int]
) -> None:
    from lehome_train.groot.experiment_manifest import batch64_quotas

    assert batch64_quotas(weights) == quotas


@pytest.mark.parametrize("target_step", [500, 1000, 2000])
def test_sweep_runtime_profile_accepts_only_exact_rungs(tmp_path: Path, target_step: int) -> None:
    from lehome_train.groot.experiment_manifest import load_sweep_runtime_profile

    document = {
        "schema_version": 1,
        "kind": "lehome_sweep_runtime_profile",
        "mixture_weights": {"bc": 100, "rollout": 0, "dagger": 0},
        "training": {"action_horizon": 16, "global_batch_size": 64, "target_step": target_step, "save_steps": 500, "terminal_publish": True},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    assert load_sweep_runtime_profile(path).target_step == target_step


@pytest.mark.parametrize("mutation", ["float", "dagger", "heldout", "unsafe-prefix", "duplicate"])
def test_manifest_fails_closed_on_invalid_or_drifting_input(tmp_path: Path, mutation: str) -> None:
    from lehome_train.groot.experiment_manifest import load_experiment_manifest

    path = tmp_path / "experiment.json"
    value = _manifest()
    if mutation == "float":
        value["mixture_weights"]["bc"] = 70.0  # type: ignore[index]
    elif mutation == "dagger":
        value["mixture_weights"]["dagger"] = 1  # type: ignore[index]
    elif mutation == "heldout":
        value["held_out_garments"] = value["held_out_garments"][:-1]  # type: ignore[index]
    elif mutation == "unsafe-prefix":
        value["bc_bundle"]["prefix"] = "../bc"  # type: ignore[index]
    else:
        path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    if mutation != "duplicate":
        _write(path, value)

    with pytest.raises(ValueError):
        load_experiment_manifest(path)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True),
    ("schema_version", 1.0),
    ("kind", True),
    ("training.global_batch_size", 64.0),
    ("training.physical_batch_size", True),
    ("training.local_checkpoint_steps", [500.0, 1000, 1500, 2000]),
    ("training.hf_checkpoint_steps", [True, 1000]),
    ("training.loader_candidates", [0, 4, 8, 12, 16.0]),
])
def test_manifest_rejects_bool_and_float_integer_contracts(
    tmp_path: Path, field: str, value: object,
) -> None:
    from lehome_train.groot.experiment_manifest import load_experiment_manifest

    manifest = _manifest()
    if field.startswith("training."):
        manifest["training"][field.removeprefix("training.")] = value  # type: ignore[index]
    else:
        manifest[field] = value
    path = tmp_path / "experiment.json"
    _write(path, manifest)

    with pytest.raises(ValueError):
        load_experiment_manifest(path)
