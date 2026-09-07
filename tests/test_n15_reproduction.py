"""Offline tests for the immutable public GR00T N1.5 contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest


SOURCE_FILES = {
    "configs/train_groot.yaml": "eb0c82d4a9960a072e454389d82a618d81a79b789c2f19b1733dba4c629e9f75",
    "shs/train/train_groot.sh": "2a49d25a1bbde7a54e6027fcbd490cb0334132b0f628eccad69413e19a1481b5",
    "scripts/utils/evaluation.py": "9a9d9e28008405ead892fdf1d115cd83f3d2be7d806381dbc92486d2e6d966a7",
    "shs/harvest/harvest_groot_until_success_00.sh": "3ac3aefefe7eea057d3df6d336a958552d276efb8dad365557a20dccc211b034",
}


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _fixture_wheel_bytes() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in {
            "lerobot/__init__.py": b'__version__ = "0.4.3"\n',
            "lerobot/policy.py": b"POLICY = 'groot'\n",
            "lerobot/policies/groot/configuration_groot.py": (
                b"@dataclass\n"
                b"class GrootConfig:\n"
                b"    warmup_ratio: float = 0.05\n"
                b"    use_bf16: bool = True\n"
                b"\n"
                b"    def get_scheduler_preset(self):\n"
                b"        return Scheduler(\n"
                b"            num_warmup_steps=int(10000 * self.warmup_ratio),  # 5% warmup by default\n"
                b"            num_decay_steps=10000,  # Adjust based on training steps\n"
                b"            peak_lr=self.optimizer_lr,\n"
                b"            decay_lr=self.optimizer_lr * 0.1,\n"
                b"        )\n"
            ),
            "lerobot-0.4.3.dist-info/METADATA": b"Name: lerobot\nVersion: 0.4.3\n",
            "lerobot-0.4.3.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            "lerobot-0.4.3.dist-info/RECORD": b"",
        }.items():
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return stream.getvalue()


def _compatibility_fixture_wheel_bytes() -> bytes:
    """A minimal, otherwise-upstream-shaped wheel for compatibility sealing."""
    return _fixture_wheel_bytes()


def test_compatibility_wheel_builder_only_adds_the_public_scheduler_fields_and_is_deterministic(
    tmp_path: Path,
) -> None:
    """The N1.5 checkpoint needs only its two proven config/scheduler fields."""
    from lehome.n15_reproduction import (
        build_compatible_lerobot_wheel,
        compatibility_wheel_identity,
    )

    upstream = tmp_path / "lerobot-0.4.3-py3-none-any.whl"
    upstream.write_bytes(_compatibility_fixture_wheel_bytes())
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    first_receipt = tmp_path / "first.json"
    second_receipt = tmp_path / "second.json"

    first_identity = build_compatible_lerobot_wheel(
        upstream_wheel=upstream,
        output_wheel=first,
        receipt_output=first_receipt,
        expected_upstream_sha256=_sha(upstream.read_bytes()),
    )
    second_identity = build_compatible_lerobot_wheel(
        upstream_wheel=upstream,
        output_wheel=second,
        receipt_output=second_receipt,
        expected_upstream_sha256=_sha(upstream.read_bytes()),
    )

    assert first_identity["wheel_path"] == str(first.resolve())
    assert second_identity["wheel_path"] == str(second.resolve())
    assert first.read_bytes() == second.read_bytes()
    first_sealed = {key: value for key, value in first_identity.items() if not key.endswith("_path")}
    second_sealed = {key: value for key, value in second_identity.items() if not key.endswith("_path")}
    assert first_sealed == second_sealed == compatibility_wheel_identity(
        wheel=first,
        receipt=first_receipt,
        upstream_wheel=upstream,
        expected_upstream_sha256=_sha(upstream.read_bytes()),
    )
    assert first_identity["upstream_wheel_sha256"] == _sha(upstream.read_bytes())
    assert first_identity["transformation"] == {
        "kind": "lehome_lerobot_043_groot_scheduler_compatibility_v1",
        "fields": {
            "num_decay_steps": 10000,
            "decay_lr_ratio": 0.1,
        },
    }
    with zipfile.ZipFile(first) as archive:
        config = archive.read("lerobot/policies/groot/configuration_groot.py").decode("utf-8")
        assert "num_decay_steps: int = 10000" in config
        assert "decay_lr_ratio: float = 0.1" in config
        assert "int(self.num_decay_steps * self.warmup_ratio)" in config
        assert "num_decay_steps=self.num_decay_steps" in config
        assert "decay_lr=self.optimizer_lr * self.decay_lr_ratio" in config


def test_compatibility_wheel_builder_rolls_back_wheel_when_receipt_publish_fails(
    tmp_path: Path,
) -> None:
    from lehome.n15_reproduction import ReproductionError, build_compatible_lerobot_wheel

    upstream = tmp_path / "lerobot-0.4.3-py3-none-any.whl"
    upstream.write_bytes(_compatibility_fixture_wheel_bytes())
    wheel = tmp_path / "compatible.whl"
    receipt = tmp_path / "compatible.json"
    receipt.write_bytes(b"immutable-existing-receipt\n")

    with pytest.raises(ReproductionError, match="already exists"):
        build_compatible_lerobot_wheel(
            upstream_wheel=upstream,
            output_wheel=wheel,
            receipt_output=receipt,
            expected_upstream_sha256=_sha(upstream.read_bytes()),
        )

    assert not wheel.exists() and not wheel.is_symlink()
    assert receipt.read_bytes() == b"immutable-existing-receipt\n"


def test_compatibility_wheel_identity_rejects_tampering(tmp_path: Path) -> None:
    from lehome.n15_reproduction import (
        ReproductionError,
        build_compatible_lerobot_wheel,
        compatibility_wheel_identity,
    )

    upstream = tmp_path / "lerobot-0.4.3-py3-none-any.whl"
    upstream.write_bytes(_compatibility_fixture_wheel_bytes())
    wheel = tmp_path / "compatible.whl"
    receipt = tmp_path / "compatible.json"
    build_compatible_lerobot_wheel(
        upstream_wheel=upstream,
        output_wheel=wheel,
        receipt_output=receipt,
        expected_upstream_sha256=_sha(upstream.read_bytes()),
    )
    tampered = bytearray(wheel.read_bytes())
    tampered[-1] ^= 1
    wheel.chmod(0o644)
    wheel.write_bytes(tampered)
    with pytest.raises(ReproductionError, match="expected derived wheel"):
        compatibility_wheel_identity(
            wheel=wheel,
            receipt=receipt,
            upstream_wheel=upstream,
            expected_upstream_sha256=_sha(upstream.read_bytes()),
        )


def test_compatibility_wheel_identity_rejects_a_self_consistent_extra_mutation(
    tmp_path: Path,
) -> None:
    """A receipt cannot bless a changed policy file beyond the two-field patch."""
    from lehome.n15_reproduction import (
        ReproductionError,
        build_compatible_lerobot_wheel,
        compatibility_wheel_identity,
        wheel_lerobot_tree_identity,
    )

    upstream = tmp_path / "lerobot-0.4.3-py3-none-any.whl"
    upstream.write_bytes(_compatibility_fixture_wheel_bytes())
    wheel = tmp_path / "compatible.whl"
    receipt = tmp_path / "compatible.json"
    identity = build_compatible_lerobot_wheel(
        upstream_wheel=upstream,
        output_wheel=wheel,
        receipt_output=receipt,
        expected_upstream_sha256=_sha(upstream.read_bytes()),
    )
    forged = tmp_path / "forged.whl"
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(forged, "w") as output:
        for name in source.namelist():
            payload = source.read(name)
            if name == "lerobot/policy.py":
                payload = b"POLICY = 'forged'\n"
            output.writestr(name, payload)
    forged_bytes = forged.read_bytes()
    count, tree = wheel_lerobot_tree_identity(forged_bytes)
    forged_receipt = dict(identity)
    forged_receipt.pop("wheel_path")
    forged_receipt.pop("receipt_path")
    forged_receipt["derived_wheel_sha256"] = _sha(forged_bytes)
    forged_receipt["derived_package_file_count"] = count
    forged_receipt["derived_package_tree_sha256"] = tree
    receipt.chmod(0o644)
    receipt.write_bytes(_canonical(forged_receipt))

    with pytest.raises(ReproductionError, match="compatibility wheel receipt identity"):
        compatibility_wheel_identity(
            wheel=forged,
            receipt=receipt,
            upstream_wheel=upstream,
            expected_upstream_sha256=_sha(upstream.read_bytes()),
        )


def test_pinned_scheduler_yaml_resolves_the_public_12k_schedule() -> None:
    from lehome.n15_reproduction import resolve_groot_scheduler_from_yaml

    values = resolve_groot_scheduler_from_yaml(
        """steps: 12000
policy:
  optimizer_lr: 2e-4
  warmup_ratio: 0.05
  num_decay_steps: 12000
  decay_lr_ratio: 0.1
"""
    )
    assert values == {
        "num_warmup_steps": 600,
        "num_decay_steps": 12000,
        "peak_lr": 2e-4,
        "decay_lr": 2e-5,
    }


def _siblings(root: Path, *, lfs_paths: set[str] | None = None) -> list[dict[str, object]]:
    lfs_paths = set() if lfs_paths is None else lfs_paths
    result = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        lfs_sha256 = _sha(payload) if relative in lfs_paths else None
        result.append(
            {
                "path": relative,
                "blob_id": "f" * 40 if lfs_sha256 else _git_blob(payload),
                "size": len(payload),
                "lfs_sha256": lfs_sha256,
            }
        )
    return result


def _git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args],
        text=True,
    ).strip()


def _manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _materialize_source(tmp_path: Path) -> tuple[Path, Path]:
    from lehome.n15_reproduction import CONTRACT

    checkout = tmp_path / "source"
    checkout.mkdir()
    (checkout / ".gitignore").write_text("Datasets*\noutputs/\n", encoding="utf-8")
    wheel_sha256 = _sha(_fixture_wheel_bytes())
    (checkout / "uv.lock").write_text(
        f'hash = "sha256:{wheel_sha256}"\n',
        encoding="utf-8",
    )
    for relative in SOURCE_FILES:
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            (
                b"steps: 12000\npolicy:\n  optimizer_lr: 2e-4\n  warmup_ratio: 0.05\n"
                b"  num_decay_steps: 12000\n  decay_lr_ratio: 0.1\n"
                if relative == "configs/train_groot.yaml"
                else relative.encode("utf-8")
            )
        )
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-q", "-m", "fixture"], check=True)
    actual_files = {
        relative: _sha((checkout / relative).read_bytes()) for relative in SOURCE_FILES
    }
    receipt = {
        "schema_version": 1,
        "kind": "lehome_public_n15_source_v1",
        "repository": CONTRACT.source_repository,
        "revision": _git(checkout, "rev-parse", "HEAD"),
        "tree": _git(checkout, "rev-parse", "HEAD^{tree}"),
        "files": actual_files,
    }
    receipt_path = tmp_path / "source-receipt.json"
    receipt_path.write_bytes(_canonical(receipt))
    return checkout, receipt_path


def _materialize_snapshots(tmp_path: Path, checkout: Path) -> tuple[Path, Path, Path]:
    from lehome.n15_reproduction import CONTRACT

    hub = tmp_path / "hub"
    model = (
        hub
        / "models--nvidia--GR00T-N1.5-3B"
        / "snapshots"
        / CONTRACT.base_model_revision
    )
    dataset = checkout / "Datasets/example/four_types_merged"
    model.mkdir(parents=True)
    dataset.mkdir(parents=True)
    model_blobs = model.parent.parent / "blobs"
    model_blobs.mkdir()
    for relative, payload, lfs in (
        ("config.json", b"model config", False),
        ("model.safetensors", b"model weights", True),
    ):
        identity = _sha(payload) if lfs else _git_blob(payload)
        blob = model_blobs / identity
        blob.write_bytes(payload)
        (model / relative).symlink_to(Path("../../blobs") / identity)
    model_refs = model.parent.parent / "refs"
    model_refs.mkdir()
    (model_refs / "main").write_text(CONTRACT.base_model_revision + "\n", encoding="ascii")
    dataset_snapshot = (
        hub
        / "datasets--lehome--dataset_challenge_merged"
        / "snapshots"
        / CONTRACT.dataset_revision
    )
    (dataset_snapshot / "meta").mkdir(parents=True)
    (dataset_snapshot / "data").mkdir()
    dataset_blobs = dataset_snapshot.parent.parent / "blobs"
    dataset_blobs.mkdir()
    for relative, payload, lfs in (
        ("meta/info.json", b"dataset metadata", False),
        ("data/chunk-000.parquet", b"dataset rows", True),
    ):
        identity = _sha(payload) if lfs else _git_blob(payload)
        blob = dataset_blobs / identity
        blob.write_bytes(payload)
        (dataset_snapshot / relative).symlink_to(
            Path("../../../blobs") / identity
        )
    for relative in ("meta/info.json", "data/chunk-000.parquet"):
        target = dataset / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = dataset_snapshot / relative
        if relative.startswith("data/"):
            blob = source.resolve(strict=True)
            target.symlink_to(Path(os.path.relpath(blob, target.parent)))
        else:
            target.write_bytes(source.read_bytes())
    receipt = {
        "schema_version": 1,
        "kind": "lehome_public_n15_resolved_snapshots_v1",
        "base_model": {
            "repository": CONTRACT.base_model_repository,
            "revision": CONTRACT.base_model_revision,
            "root": str(model.resolve()),
            "siblings": _siblings(model, lfs_paths={"model.safetensors"}),
        },
        "dataset": {
            "repository": CONTRACT.dataset_repository,
            "revision": CONTRACT.dataset_revision,
            "root": str(dataset.resolve()),
            "snapshot_root": str(dataset_snapshot.resolve()),
            "siblings": _siblings(
                dataset_snapshot,
                lfs_paths={"data/chunk-000.parquet"},
            ),
        },
        "vm_id": CONTRACT.vm_id,
        "disk_id": CONTRACT.disk_id,
    }
    receipt_path = tmp_path / "resolved-snapshots-receipt.json"
    receipt_path.write_bytes(_canonical(receipt))
    return model, dataset, receipt_path


def test_resolve_dataset_blobs_mount_returns_verified_canonical_path(tmp_path: Path) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, _ = _materialize_source(tmp_path)
    _, dataset, receipt_path = _materialize_snapshots(tmp_path, checkout)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    snapshot = Path(receipt["dataset"]["snapshot_root"])

    assert reproduction.resolve_dataset_blobs_mount(
        resolved_snapshots_receipt=receipt_path,
        expected_dataset_root=dataset,
        protected_root=tmp_path,
    ) == snapshot.parent.parent / "blobs"


@pytest.mark.parametrize("alias_kind", ["traversal", "symlink_ancestor"])
def test_resolve_dataset_blobs_mount_rejects_noncanonical_aliases(
    tmp_path: Path, alias_kind: str
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, _ = _materialize_source(tmp_path)
    _, dataset, receipt_path = _materialize_snapshots(tmp_path, checkout)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    snapshot = Path(receipt["dataset"]["snapshot_root"])
    if alias_kind == "traversal":
        receipt["dataset"]["snapshot_root"] = str(
            snapshot.parent / ".." / snapshot.parent.name / snapshot.name
        )
    else:
        alias = tmp_path / "hub-alias"
        alias.symlink_to(snapshot.parents[2], target_is_directory=True)
        receipt["dataset"]["snapshot_root"] = str(
            alias.joinpath(*snapshot.relative_to(snapshot.parents[2]).parts)
        )
    receipt_path.write_bytes(_canonical(receipt))

    with pytest.raises(reproduction.ReproductionError, match="canonical"):
        reproduction.resolve_dataset_blobs_mount(
            resolved_snapshots_receipt=receipt_path,
            expected_dataset_root=dataset,
            protected_root=tmp_path,
        )


def _fixture_contract(checkout: Path):
    from dataclasses import replace
    from lehome.n15_reproduction import (
        CONTRACT,
        hub_metadata_sha256,
        wheel_lerobot_tree_identity,
    )

    snapshots_receipt = checkout.parent / "resolved-snapshots-receipt.json"
    snapshots = json.loads(snapshots_receipt.read_text(encoding="utf-8"))
    model_siblings = snapshots["base_model"]["siblings"]
    dataset_siblings = snapshots["dataset"]["siblings"]
    wheel_count, wheel_tree_sha256 = wheel_lerobot_tree_identity(_fixture_wheel_bytes())

    return replace(
        CONTRACT,
        source_revision=_git(checkout, "rev-parse", "HEAD"),
        source_tree=_git(checkout, "rev-parse", "HEAD^{tree}"),
        dependency_lock_sha256=_sha((checkout / "uv.lock").read_bytes()),
        lerobot_wheel_sha256=_sha(_fixture_wheel_bytes()),
        lerobot_package_file_count=wheel_count,
        lerobot_package_tree_sha256=wheel_tree_sha256,
        base_model_metadata_count=len(model_siblings),
        base_model_metadata_sha256=hub_metadata_sha256(model_siblings),
        dataset_metadata_count=len(dataset_siblings),
        dataset_metadata_sha256=hub_metadata_sha256(dataset_siblings),
        trusted_source_files={
            relative: _sha((checkout / relative).read_bytes())
            for relative in SOURCE_FILES
        },
    )


def test_contract_encodes_exact_public_recipe_and_is_immutable() -> None:
    from lehome.n15_reproduction import CONTRACT

    assert CONTRACT.source_repository == "theo-zhou/lehome-groot-submission-4"
    assert CONTRACT.source_revision == "d384fe00508acd96ab1c3c5dc265e08261f94b3b"
    assert CONTRACT.source_tree == "8bb4ff37d03762f8c4bc4bce5783e7d811991a3e"
    assert CONTRACT.dependency_lock_sha256 == "d0e6e3cb472cea3d04b0bc2d79b9d929bf498a392d5c155fa635f413fa092313"
    assert CONTRACT.lerobot_wheel_sha256 == "b08c1c15b2356bd4e658122deabfb9dacd2d7447de4a4327720991723d4edf2c"
    assert CONTRACT.lerobot_package_file_count == 289
    assert CONTRACT.lerobot_package_tree_sha256 == "db3b4e18b166d4bb7fb4354cec82a7fbd15bb24230f9d71269a017c774e0852f"
    assert CONTRACT.base_model_metadata_count == 13
    assert CONTRACT.base_model_metadata_sha256 == "b49d2e9f419064cbe31fcc877263f5a1af4ca1ec10acd723b3c325dc0d6fc70d"
    assert CONTRACT.dataset_metadata_count == 67
    assert CONTRACT.dataset_metadata_sha256 == "152e3b0e3da178fba9d29ddb1858df95a4c20fe8118aa36b57bde71b0ee25b9a"
    assert CONTRACT.base_model_repository == "nvidia/GR00T-N1.5-3B"
    assert CONTRACT.base_model_revision == "869830fc749c35f34771aa5209f923ac57e4564e"
    assert CONTRACT.dataset_repository == "lehome/dataset_challenge_merged"
    assert CONTRACT.dataset_revision == "17e8dee8fac294ffd21d250501d3b31bf8679042"
    assert dict(CONTRACT.trusted_source_files) == SOURCE_FILES
    assert CONTRACT.vm_id == "computeinstance-u00t6xfqhadrcmssa2"
    assert CONTRACT.disk_id == "computedisk-u00pbe55crxy7jr56x"
    assert CONTRACT.python_version == "3.11"
    assert CONTRACT.lerobot_version == "0.4.3"
    assert CONTRACT.training_command == (
        "lerobot-train",
        "--config_path=configs/train_groot.yaml",
        "--wandb.mode=offline",
    )
    assert CONTRACT.training == {
        "batch_size": 64,
        "steps": 12000,
        "optimizer_lr": 2e-4,
        "optimizer_beta1": 0.95,
        "optimizer_beta2": 0.999,
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 1e-5,
        "warmup_ratio": 0.05,
        "num_decay_steps": 12000,
        "decay_lr_ratio": 0.1,
        "use_bf16": True,
        "tune_llm": False,
        "tune_visual": False,
        "tune_projector": True,
        "tune_diffusion_model": True,
        "image_transforms": False,
        "state_normalization": "mean_std",
        "action_normalization": "mean_std",
        "policy_image_size": 224,
        "state_dimension": 12,
        "action_dimension": 12,
        "save_freq": 1500,
        "log_freq": 500,
    }
    with pytest.raises(FrozenInstanceError):
        CONTRACT.source_revision = "0" * 40  # type: ignore[misc]


def test_hub_metadata_hash_is_sorted_tab_separated_and_binds_all_fields() -> None:
    from lehome.n15_reproduction import hub_metadata_sha256

    siblings = [
        {"path": "z.bin", "blob_id": "f" * 40, "size": 3, "lfs_sha256": "a" * 64},
        {"path": "a.json", "blob_id": "b" * 40, "size": 2, "lfs_sha256": None},
    ]
    canonical = (
        f"a.json\t{'b' * 40}\t2\t\n"
        f"z.bin\t{'f' * 40}\t3\t{'a' * 64}\n"
    ).encode("utf-8")

    assert hub_metadata_sha256(siblings) == _sha(canonical)


def test_production_lerobot_wheel_tree_uses_the_audited_algorithm_identity() -> None:
    """This test intentionally uses CONTRACT directly, never the fixture override."""
    from lehome.n15_reproduction import CONTRACT

    assert (
        CONTRACT.lerobot_wheel_sha256,
        CONTRACT.lerobot_package_file_count,
        CONTRACT.lerobot_package_tree_sha256,
    ) == (
        "b08c1c15b2356bd4e658122deabfb9dacd2d7447de4a4327720991723d4edf2c",
        289,
        "db3b4e18b166d4bb7fb4354cec82a7fbd15bb24230f9d71269a017c774e0852f",
    )


def test_verify_inputs_binds_regular_source_files_snapshots_and_resources(
    tmp_path: Path,
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    model, dataset, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)

    verified = reproduction.verify_inputs(
        checkout=checkout,
        source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=reproduction.CONTRACT.vm_id,
        disk_id=reproduction.CONTRACT.disk_id,
        contract=contract,
    )

    assert verified.checkout == checkout.resolve()
    assert verified.base_model_root == model.resolve()
    assert verified.dataset_root == dataset.resolve()
    assert verified.source_receipt_sha256 == _sha(source_receipt.read_bytes())
    assert verified.resolved_snapshots_receipt_sha256 == _sha(
        snapshots_receipt.read_bytes()
    )


@pytest.mark.parametrize("unsafe", ["checkout", "source_file", "model", "receipt"])
def test_verify_inputs_rejects_symlinks(
    tmp_path: Path,
    unsafe: str,
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    model, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    if unsafe == "checkout":
        target = checkout
        checkout = tmp_path / "source-link"
        checkout.symlink_to(target, target_is_directory=True)
    elif unsafe == "source_file":
        path = checkout / "configs/train_groot.yaml"
        replacement = tmp_path / "replacement"
        replacement.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(replacement)
    elif unsafe == "model":
        target = model
        linked = tmp_path / "model-link"
        linked.symlink_to(target, target_is_directory=True)
        payload = json.loads(snapshots_receipt.read_text(encoding="utf-8"))
        payload["base_model"]["root"] = str(linked)
        snapshots_receipt.write_bytes(_canonical(payload))
    else:
        target = source_receipt
        linked = tmp_path / "receipt-link"
        linked.symlink_to(target)
        source_receipt = linked

    with pytest.raises(reproduction.ReproductionError, match="unsafe"):
        reproduction.verify_inputs(
            checkout=checkout,
            source_receipt=source_receipt,
            resolved_snapshots_receipt=snapshots_receipt,
            vm_id=reproduction.CONTRACT.vm_id,
            disk_id=reproduction.CONTRACT.disk_id,
            contract=contract,
        )


@pytest.mark.parametrize("field", ["revision", "file_digest", "vm", "disk"])
def test_verify_inputs_rejects_mismatched_identity_receipts(
    tmp_path: Path,
    field: str,
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    vm_id = reproduction.CONTRACT.vm_id
    disk_id = reproduction.CONTRACT.disk_id
    if field in {"revision", "file_digest"}:
        payload = json.loads(source_receipt.read_text(encoding="utf-8"))
        if field == "revision":
            payload["revision"] = "0" * 40
        else:
            payload["files"]["configs/train_groot.yaml"] = "0" * 64
        source_receipt.write_bytes(_canonical(payload))
    elif field == "vm":
        vm_id = "computeinstance-wrong"
    else:
        disk_id = "computedisk-wrong"

    with pytest.raises(reproduction.ReproductionError, match="mismatch|not accepted"):
        reproduction.verify_inputs(
            checkout=checkout,
            source_receipt=source_receipt,
            resolved_snapshots_receipt=snapshots_receipt,
            vm_id=vm_id,
            disk_id=disk_id,
            contract=contract,
        )


def test_verify_inputs_rejects_checkout_head_mismatch(tmp_path: Path) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    (checkout / "new-tracked-file").write_text("different tree\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "new-tracked-file"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-q", "-m", "different head"],
        check=True,
    )

    with pytest.raises(reproduction.ReproductionError, match="Git HEAD"):
        reproduction.verify_inputs(
            checkout=checkout,
            source_receipt=source_receipt,
            resolved_snapshots_receipt=snapshots_receipt,
            vm_id=contract.vm_id,
            disk_id=contract.disk_id,
            contract=contract,
        )


@pytest.mark.parametrize("snapshot", ["model", "dataset"])
def test_verify_inputs_rejects_snapshots_not_staged_at_upstream_paths(
    tmp_path: Path,
    snapshot: str,
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    payload = json.loads(snapshots_receipt.read_text(encoding="utf-8"))
    wrong = tmp_path / f"wrong-{snapshot}"
    wrong.mkdir()
    if snapshot == "model":
        payload["base_model"]["root"] = str(wrong)
    else:
        payload["dataset"]["root"] = str(wrong)
    snapshots_receipt.write_bytes(_canonical(payload))

    with pytest.raises(reproduction.ReproductionError, match="staged path"):
        reproduction.verify_inputs(
            checkout=checkout,
            source_receipt=source_receipt,
            resolved_snapshots_receipt=snapshots_receipt,
            vm_id=contract.vm_id,
            disk_id=contract.disk_id,
            contract=contract,
        )


@pytest.mark.parametrize("problem", ["missing_ref", "mismatched_ref", "empty_model", "tampered_model", "tampered_dataset"])
def test_verify_inputs_rejects_unproven_or_tampered_snapshot_content(
    tmp_path: Path,
    problem: str,
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    model, dataset, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    model_ref = model.parent.parent / "refs/main"
    payload = json.loads(snapshots_receipt.read_text(encoding="utf-8"))
    if problem == "missing_ref":
        model_ref.unlink()
    elif problem == "mismatched_ref":
        model_ref.write_text("0" * 40 + "\n", encoding="ascii")
    elif problem == "empty_model":
        for path in model.iterdir():
            path.unlink()
        payload["base_model"]["siblings"] = []
        snapshots_receipt.write_bytes(_canonical(payload))
    elif problem == "tampered_model":
        (model / "model.safetensors").write_bytes(b"tampered")
    else:
        (dataset / "data/chunk-000.parquet").write_bytes(b"tampered")

    with pytest.raises(reproduction.ReproductionError, match="ref|metadata|content"):
        reproduction.verify_inputs(
            checkout=checkout,
            source_receipt=source_receipt,
            resolved_snapshots_receipt=snapshots_receipt,
            vm_id=contract.vm_id,
            disk_id=contract.disk_id,
            contract=contract,
        )


@pytest.mark.parametrize("problem", ["escape", "dangling", "chain", "directory"])
def test_verify_inputs_rejects_unsafe_hub_snapshot_symlinks(
    tmp_path: Path,
    problem: str,
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    model, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    link = model / "config.json"
    original_blob = link.resolve(strict=True)
    link.unlink()
    if problem == "escape":
        outside = tmp_path / "outside-config"
        outside.write_bytes(original_blob.read_bytes())
        link.symlink_to(outside)
    elif problem == "dangling":
        link.symlink_to(Path("../../blobs/does-not-exist"))
    elif problem == "chain":
        terminal = original_blob.with_name("terminal-blob")
        terminal.write_bytes(original_blob.read_bytes())
        original_blob.unlink()
        original_blob.symlink_to(terminal.name)
        link.symlink_to(Path("../../blobs") / original_blob.name)
    else:
        link.symlink_to(Path("../../blobs"), target_is_directory=True)

    with pytest.raises(reproduction.ReproductionError, match="symlink"):
        reproduction.verify_inputs(
            checkout=checkout,
            source_receipt=source_receipt,
            resolved_snapshots_receipt=snapshots_receipt,
            vm_id=contract.vm_id,
            disk_id=contract.disk_id,
            contract=contract,
        )


def test_atomic_receipt_reconciles_crash_after_no_clobber_link(tmp_path: Path) -> None:
    from lehome import n15_reproduction as reproduction

    output = tmp_path / "receipt.json"
    payload = _canonical({"durable": True})
    temporary = tmp_path / ".receipt.json.crashed"
    temporary.write_bytes(payload)
    temporary.chmod(0o444)
    os.link(temporary, output)

    path, digest = reproduction._write_atomic_bytes(output, payload, "test receipt")

    assert path == output
    assert digest == _sha(payload)
    assert output.read_bytes() == payload
    assert not temporary.exists()


def test_render_training_writes_an_atomic_offline_manifest_without_execution(
    tmp_path: Path,
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    model, dataset, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout,
        source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id,
        disk_id=contract.disk_id,
        contract=contract,
    )
    output = tmp_path / "execution-manifest.json"

    receipt = reproduction.render_training(
        verified=verified,
        output=output,
        contract=contract,
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["kind"] == "lehome_public_n15_training_execution_v1"
    assert manifest["execution"] == {
        "argv": [
            "lerobot-train",
            "--config_path=configs/train_groot.yaml",
            "--wandb.mode=offline",
        ],
        "cwd": str(checkout.resolve()),
        "container": {
            "image_id": "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7",
            "python_executable": "/opt/lehome-challenge/.venv/bin/python",
            "pythonpath": "/flash/site-packages:/deps/peft-0.18.1-py3-none-any.whl",
        },
        "env": {
            "HF_HUB_CACHE": str((tmp_path / "hub").resolve()),
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": "/flash/site-packages:/deps/peft-0.18.1-py3-none-any.whl",
        },
        "shell_argv": "lerobot-train --config_path=configs/train_groot.yaml --wandb.mode=offline",
    }
    assert manifest["inputs"]["base_model_root"] == str(model.resolve())
    assert manifest["inputs"]["dataset_root"] == str(dataset.resolve())
    assert manifest["contract"]["training"] == dict(contract.training)
    assert receipt["manifest_sha256"] == _sha(output.read_bytes())
    assert output.stat().st_mode & 0o777 == 0o444
    with pytest.raises(reproduction.ReproductionError, match="already exists"):
        reproduction.render_training(
            verified=verified,
            output=output,
            contract=contract,
        )


def _materialize_training_output(
    tmp_path: Path,
    *,
    verified,
    contract,
) -> Path:
    from lehome import n15_reproduction as reproduction

    root = tmp_path / "training-output"
    checkpoint = root / "checkpoints/012000"
    pretrained = checkpoint / "pretrained_model"
    training_state = checkpoint / "training_state"
    evidence = root / "evidence"
    logs = root / "logs"
    runtime = root / "runtime"
    pretrained.mkdir(parents=True)
    training_state.mkdir()
    (checkpoint.parent / "last").symlink_to("012000", target_is_directory=True)
    evidence.mkdir()
    logs.mkdir()
    runtime.mkdir()
    for name, payload in {
        "config.json": _canonical({"type": "groot"}),
        "model.safetensors": b"realistic model weights",
        "train_config.json": _canonical({"batch_size": 64, "steps": 12000}),
        "policy_preprocessor.json": _canonical({"name": "policy_preprocessor"}),
        "policy_postprocessor.json": _canonical({"name": "policy_postprocessor"}),
        "policy_preprocessor_step_2_groot_pack_inputs_v3.safetensors": b"preprocessor state",
        "policy_postprocessor_step_0_groot_action_unpack_unnormalize_v1.safetensors": b"postprocessor state",
    }.items():
        (pretrained / name).write_bytes(payload)
    for name, payload in {
        "optimizer_param_groups.json": _canonical({"groups": [0]}),
        "optimizer_state.safetensors": b"optimizer state",
        "rng_state.safetensors": b"rng state",
        "scheduler_state.json": _canonical({"last_epoch": 12000}),
        "training_step.json": _canonical({"step": 12000}),
    }.items():
        (training_state / name).write_bytes(payload)
    (evidence / "source-receipt.json").write_bytes(verified.source_receipt.read_bytes())
    (evidence / "resolved-snapshots-receipt.json").write_bytes(
        verified.resolved_snapshots_receipt.read_bytes()
    )
    (evidence / "execution-manifest.json").write_bytes(
        _canonical(reproduction.build_training_manifest(verified=verified, contract=contract))
    )
    (evidence / "uv.lock").write_bytes((verified.checkout / "uv.lock").read_bytes())
    (evidence / "peft-overlay-receipt.json").write_bytes(
        _canonical(
            {
                "schema_version": 1,
                "kind": "lehome_native_reference_peft_overlay_v1",
                "wheel_path": "/mnt/lehome/reference-native/dependencies/peft-0.18.1-py3-none-any.whl",
                "wheel_filename": "peft-0.18.1-py3-none-any.whl",
                "wheel_url": "https://files.pythonhosted.org/packages/b3/14/b4e3f574acf349ae6f61f9c000a77f97a3b315b4bb6ad03791e79ae4a568/peft-0.18.1-py3-none-any.whl",
                "wheel_size": 556960,
                "wheel_sha256": "0bf06847a3551e3019fc58c440cffc9a6b73e6e2962c95b52e224f77bbdb50f1",
                "distribution_name": "peft",
                "peft_version": "0.18.1",
                "peft_origin": "/mnt/lehome/reference-native/dependencies/peft-0.18.1-py3-none-any.whl/peft/__init__.py",
                "required_symbols": ["LoraConfig", "get_peft_model"],
            }
        )
    )
    (evidence / "flash-attention-overlay-receipt.json").write_bytes(
        _canonical(
            {
                "schema_version": 1,
                "kind": "lehome_native_reference_flash_attention_overlay_v1",
                "wheel_path": "/mnt/lehome/reference-native/dependencies/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
                "wheel_filename": "flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
                "wheel_url": "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
                "wheel_size": 256027206,
                "wheel_sha256": "cd1a45ebfc1731a13e55ad68e0c9ad92390ddfffba306f9222be67c6d5a805af",
                "distribution_name": "flash_attn",
                "flash_attn_version": "2.8.3",
                "wheel_tag": "cp311-cp311-linux_x86_64",
            }
        )
    )
    (evidence / "flash-attention-runtime-receipt.json").write_bytes(
        _canonical(
            {
                "schema_version": 1,
                "kind": "lehome_native_reference_flash_attention_runtime_v1",
                "torch_version": "2.7.0+cu128",
                "torch_cuda_version": "12.8",
                "torch_cxx11_abi": True,
                "cuda_capability": [12, 0],
                "flash_attn_version": "2.8.3",
                "flash_attn_origin": str((runtime / "site-packages/flash_attn/__init__.py").resolve()),
                "kernel": {"shape": [1, 2, 4, 64], "dtype": "float16", "finite": True},
            }
        )
    )
    (evidence / "training-container-runtime-receipt.json").write_bytes(
        _canonical(
            {
                "schema_version": 1,
                "kind": "lehome_public_n15_training_container_runtime_v1",
                "image_id": "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7",
                "python_executable": "/opt/lehome-challenge/.venv/bin/python",
                "python_version": [3, 11, 13],
                "pythonpath": "/flash/site-packages:/deps/peft-0.18.1-py3-none-any.whl",
                "lerobot_origin": "/flash/site-packages/lerobot/__init__.py",
                "peft_origin": "/deps/peft-0.18.1-py3-none-any.whl/peft/__init__.py",
                "flash_attn_origin": "/flash/site-packages/flash_attn/__init__.py",
                "torch_version": "2.7.0+cu128",
                "torch_cuda_version": "12.8",
                "cuda_capability": [12, 0],
            }
        )
    )
    (evidence / "runtime-image-receipt.json").write_bytes(
        _canonical(
            {
                "schema_version": 1,
                "kind": "lehome_public_n15_training_runtime_image_v1",
                "image_id": "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7",
            }
        )
    )
    python_candidate = shutil.which("python3.11")
    assert python_candidate is not None
    python = Path(python_candidate).resolve()
    upstream_wheel = evidence / "upstream/lerobot-0.4.3-py3-none-any.whl"
    upstream_wheel.parent.mkdir()
    upstream_wheel.write_bytes(_fixture_wheel_bytes())
    wheel = evidence / "compatibility/lerobot-0.4.3-py3-none-any.whl"
    wheel.parent.mkdir()
    compatibility_receipt = wheel.parent / "lerobot-compatibility-receipt.json"
    compatibility = reproduction.build_compatible_lerobot_wheel(
        upstream_wheel=upstream_wheel,
        output_wheel=wheel,
        receipt_output=compatibility_receipt,
        expected_upstream_sha256=contract.lerobot_wheel_sha256,
    )
    package_root = runtime / "site-packages/lerobot"
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not name.startswith("lerobot/") or name.endswith("/"):
                continue
            target = runtime / "site-packages" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    (evidence / "runtime-receipt.json").write_bytes(
        _canonical(
            {
                "schema_version": 1,
                "kind": "lehome_public_n15_training_runtime_v1",
                "python_executable": str(python.resolve()),
                "upstream_lerobot_wheel_path": str(upstream_wheel.resolve()),
                "upstream_lerobot_wheel_sha256": contract.lerobot_wheel_sha256,
                "compatibility_wheel_path": str(wheel.resolve()),
                "compatibility_wheel_sha256": compatibility["derived_wheel_sha256"],
                "compatibility_wheel_receipt_path": str(compatibility_receipt.resolve()),
                "compatibility_wheel_receipt_sha256": _sha(compatibility_receipt.read_bytes()),
                "lerobot_package_root": str(package_root.resolve()),
                "dependency_lock_path": str((evidence / "uv.lock").resolve()),
                "dependency_lock_sha256": contract.dependency_lock_sha256,
                "scheduler": reproduction.resolve_groot_scheduler_from_yaml(
                    (verified.checkout / "configs/train_groot.yaml").read_text(encoding="utf-8")
                ),
            }
        )
    )
    (logs / "train.log").write_text(
        "Checkpoint policy after step 12000\nEnd of training\n",
        encoding="utf-8",
    )
    _write_training_checksums(root)
    return root


def _write_training_checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            relative = path.relative_to(root).as_posix()
            lines.append(f"{_sha(path.read_bytes())}  {relative}\n")
    (root / "checksums.sha256").write_text("".join(lines), encoding="ascii")


def _materialize_partial_training(
    tmp_path: Path,
    *,
    verified,
    contract,
    step: int = 1500,
) -> tuple[Path, Path, Path]:
    """Turn the complete fixture into the two real pre-finalization roots."""
    training_root = _materialize_training_output(
        tmp_path, verified=verified, contract=contract
    )
    upstream_output = (
        verified.checkout / "outputs/train/groot_four_types_merged_batch64_lr2e-4"
    )
    staging_root = Path(f"{training_root}.evidence-staging")
    upstream_output.mkdir(parents=True)
    staging_root.mkdir()
    (training_root / "checkpoints").rename(upstream_output / "checkpoints")
    (training_root / "evidence").rename(staging_root / "evidence")
    (training_root / "logs").rename(staging_root / "logs")
    (training_root / "runtime").rename(staging_root / "runtime")
    (training_root / "checksums.sha256").unlink()
    training_root.rmdir()
    checkpoint = upstream_output / f"checkpoints/{step:06d}"
    (upstream_output / "checkpoints/012000").rename(checkpoint)
    last = upstream_output / "checkpoints/last"
    last.unlink()
    last.symlink_to(f"{step:06d}", target_is_directory=True)
    state = checkpoint / "training_state/training_step.json"
    state.write_bytes(_canonical({"step": step}))
    (checkpoint / "training_state/scheduler_state.json").write_bytes(
        _canonical({"last_epoch": step})
    )
    config = checkpoint / "pretrained_model/train_config.json"
    # Authentic LeRobot 0.4.3 TrainPipelineConfig serialization from the
    # public 12K recipe.  Only the source YAML's lora_rank and run-specific
    # paths/id are resolved here.
    train_config = json.loads(_AUTHENTIC_PUBLIC_12K_TRAIN_CONFIG)
    train_config["dataset"]["root"] = str(verified.dataset_root)
    train_config["output_dir"] = str(upstream_output)
    train_config["wandb"]["run_id"] = "a1b2c3d4"
    train_config["wandb"]["mode"] = "offline"
    config.write_bytes(_canonical(train_config))
    return training_root, staging_root, upstream_output


_AUTHENTIC_PUBLIC_12K_TRAIN_CONFIG = r'''{"dataset":{"repo_id":"repo_groot","root":"Datasets/example/four_types_merged","episodes":null,"image_transforms":{"enable":false,"max_num_transforms":3,"random_order":false,"tfs":{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}},"revision":null,"use_imagenet_stats":true,"video_backend":"torchcodec","streaming":false},"env":null,"policy":{"type":"groot","n_obs_steps":1,"input_features":{"observation.state":{"type":"STATE","shape":[12]},"observation.images.top_rgb":{"type":"VISUAL","shape":[3,480,640]},"observation.images.left_rgb":{"type":"VISUAL","shape":[3,480,640]},"observation.images.right_rgb":{"type":"VISUAL","shape":[3,480,640]},"observation.images.top_depth":{"type":"STATE","shape":[1,480,640]}},"output_features":{"action":{"type":"ACTION","shape":[12]}},"device":"cuda","use_amp":false,"use_peft":false,"push_to_hub":false,"repo_id":null,"private":null,"tags":null,"license":null,"pretrained_path":null,"chunk_size":50,"n_action_steps":50,"max_state_dim":64,"max_action_dim":32,"normalization_mapping":{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"},"image_size":[224,224],"base_model_path":"nvidia/GR00T-N1.5-3B","tokenizer_assets_repo":"lerobot/eagle2hg-processor-groot-n1p5","embodiment_tag":"new_embodiment","tune_llm":false,"tune_visual":false,"tune_projector":true,"tune_diffusion_model":true,"lora_rank":0,"lora_alpha":16,"lora_dropout":0.05,"lora_full_model":false,"optimizer_lr":0.0002,"optimizer_betas":[0.95,0.999],"optimizer_eps":1e-08,"optimizer_weight_decay":1e-05,"warmup_ratio":0.05,"num_decay_steps":12000,"decay_lr_ratio":0.1,"use_bf16":true,"video_backend":"decord","balance_dataset_weights":true,"balance_trajectory_weights":true,"dataset_paths":null,"output_dir":"./tmp/gr00t","save_steps":1000,"max_steps":10000,"batch_size":32,"dataloader_num_workers":8,"report_to":"wandb","resume":false},"output_dir":"outputs/train/groot_four_types_merged_batch64_lr2e-4","job_name":"groot","resume":false,"seed":1000,"num_workers":4,"batch_size":64,"steps":12000,"eval_freq":20000,"log_freq":500,"tolerance_s":0.0001,"save_checkpoint":true,"save_freq":1500,"use_policy_training_preset":true,"optimizer":{"type":"adamw","lr":0.0002,"weight_decay":1e-05,"grad_clip_norm":10.0,"betas":[0.95,0.999],"eps":1e-08},"scheduler":{"type":"cosine_decay_with_warmup","num_warmup_steps":600,"num_decay_steps":12000,"peak_lr":0.0002,"decay_lr":2e-05},"eval":{"n_episodes":50,"batch_size":50,"use_async_envs":false},"wandb":{"enable":true,"disable_artifact":true,"project":"lehome-challenge","entity":null,"notes":null,"run_id":"iqfjc8st","mode":null},"peft":null,"use_rabc":false,"rabc_progress_path":null,"rabc_kappa":0.01,"rabc_epsilon":1e-06,"rabc_head_mode":"sparse","rename_map":{},"checkpoint_path":null}'''


def test_public_12k_golden_config_has_independent_origin_and_fixture_digest() -> None:
    from lehome import n15_reproduction as reproduction

    identity = reproduction.public_12k_train_config_identity()

    assert identity == {
        "schema_version": 1,
        "kind": "lehome_public_n15_train_config_golden_v1",
        "source_repository": "theo-zhou/lehome-groot-submission-4",
        "source_revision": "d384fe00508acd96ab1c3c5dc265e08261f94b3b",
        "source_path": "pretrained_model/train_config.json",
        "source_artifact_sha256": "8fed45ce6356ca2ab3a44ee16f58efcba65274666074d253fa252ef6e826052f",
        "fixture_path": "n15_public_12k_train_config.golden.json",
        "fixture_sha256": "a3130a1b796ecc0da6bb1c51b82b6ee04e2ecc761e4c2ae530f07281613f18ee",
        "resolved_recipe_sha256": "14db86649a124aedcfd8b88e2f2c668dfe7b628f6e3191d2a5150084a9c58fd6",
        "source_derivation": {
            "fixture_policy_lora_rank": 0,
            "source_policy_lora_rank": 8,
            "serialization": {
                "ensure_ascii": True,
                "indent": 4,
                "sort_keys": False,
                "trailing_newline": False,
            },
        },
        "allowed_resolutions": [
            "dataset.root",
            "output_dir",
            "wandb.run_id",
            "wandb.mode=offline",
        ],
    }
    golden_path = Path(reproduction.__file__).with_name(
        "n15_public_12k_train_config.golden.json"
    )
    derived_source = json.loads(golden_path.read_text(encoding="ascii"))
    assert derived_source["policy"]["lora_rank"] == 0
    derived_source["policy"]["lora_rank"] = 8
    source_artifact = json.dumps(
        derived_source, ensure_ascii=True, indent=4, sort_keys=False
    ).encode("ascii")
    assert _sha(source_artifact) == (
        "8fed45ce6356ca2ab3a44ee16f58efcba65274666074d253fa252ef6e826052f"
    )


def test_verify_resume_checkpoint_accepts_complete_001500_and_renders_exact_command(
    tmp_path: Path,
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout,
        source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id,
        disk_id=contract.disk_id,
        contract=contract,
    )
    training_root, staging_root, upstream_output = _materialize_partial_training(
        tmp_path, verified=verified, contract=contract
    )
    (upstream_output / "checkpoints/001500/regular-extra.bin").write_bytes(b"extra\n")

    receipt = reproduction.verify_resume_checkpoint(
        verified=verified,
        training_root=training_root,
        staging_root=staging_root,
        upstream_output=upstream_output,
        requested_step=1500,
        attempt_id="attempt-a",
        contract=contract,
    )

    config = upstream_output / "checkpoints/001500/pretrained_model/train_config.json"
    assert receipt["requested_step"] == 1500
    assert receipt["attempt_id"] == "attempt-a"
    assert receipt["resume_argv"] == [
        "/opt/lehome-challenge/.venv/bin/lerobot-train",
        f"--config_path={config}",
        "--resume=true",
        "--wandb.mode=offline",
    ]
    assert receipt["pythonpath"] == (
        "/flash/site-packages:/deps/peft-0.18.1-py3-none-any.whl"
    )
    assert receipt["checkpoint_files"]
    assert receipt["evidence_files"]
    assert receipt["train_config_origin"] == reproduction.public_12k_train_config_identity()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_last", "last-checkpoint"),
        ("wrong_last", "last-checkpoint"),
        ("symlink_file", "unsafe"),
        ("symlink_extra", "unsafe"),
        ("missing_file", "incomplete"),
        ("empty_file", "empty"),
        ("empty_extra", "empty"),
        ("wrong_step", "training-step"),
        ("stale_scheduler", "scheduler"),
        ("modified_recipe", "recipe"),
        ("unknown_recipe_field", "recipe"),
        ("wrong_output", "output"),
        ("completed_identity", "canonical training"),
        ("completed_publication", "canonical training"),
        ("source_evidence", "source receipt"),
        ("runtime_image", "runtime image"),
        ("compatibility", "compatibility wheel"),
    ],
)
def test_verify_resume_checkpoint_fails_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout,
        source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id,
        disk_id=contract.disk_id,
        contract=contract,
    )
    training_root, staging_root, upstream_output = _materialize_partial_training(
        tmp_path, verified=verified, contract=contract
    )
    checkpoint = upstream_output / "checkpoints/001500"
    if mutation == "missing_last":
        (upstream_output / "checkpoints/last").unlink()
    elif mutation == "wrong_last":
        (upstream_output / "checkpoints/last").unlink()
        (upstream_output / "checkpoints/last").symlink_to("003000")
    elif mutation in {"symlink_file", "symlink_extra"}:
        target = checkpoint / "pretrained_model/model.safetensors"
        payload = target.read_bytes(); target.unlink()
        outside = tmp_path / "outside"; outside.write_bytes(payload)
        if mutation == "symlink_file":
            target.symlink_to(outside)
        else:
            target.write_bytes(payload)
            (checkpoint / "extra.bin").symlink_to(outside)
    elif mutation == "missing_file":
        (checkpoint / "training_state/rng_state.safetensors").unlink()
    elif mutation == "empty_file":
        (checkpoint / "training_state/scheduler_state.json").write_bytes(b"")
    elif mutation == "empty_extra":
        (checkpoint / "extra.bin").write_bytes(b"")
    elif mutation == "wrong_step":
        (checkpoint / "training_state/training_step.json").write_bytes(
            _canonical({"step": 1499})
        )
    elif mutation == "stale_scheduler":
        (checkpoint / "training_state/scheduler_state.json").write_bytes(
            _canonical({"last_epoch": 1499})
        )
    elif mutation in {"modified_recipe", "unknown_recipe_field", "wrong_output"}:
        path = checkpoint / "pretrained_model/train_config.json"
        value = json.loads(path.read_text())
        if mutation == "modified_recipe":
            value["batch_size"] = 32
        elif mutation == "unknown_recipe_field":
            value["unreviewed_flag"] = True
        else:
            value["output_dir"] = "/wrong/output"
        path.write_bytes(_canonical(value))
    elif mutation == "completed_identity":
        training_root.mkdir(); (training_root / "training-identity.json").write_bytes(b"{}\n")
    elif mutation == "completed_publication":
        training_root.mkdir(); (training_root / "training-publication.json").write_bytes(b"{}\n")
    elif mutation == "source_evidence":
        (staging_root / "evidence/source-receipt.json").write_bytes(b"{}\n")
    elif mutation == "runtime_image":
        path = staging_root / "evidence/runtime-image-receipt.json"
        value = json.loads(path.read_text()); value["image_id"] = "sha256:" + "0" * 64
        path.write_bytes(_canonical(value))
    else:
        path = staging_root / "evidence/compatibility/lerobot-0.4.3-py3-none-any.whl"
        path.chmod(0o644)
        path.write_bytes(b"tampered")

    with pytest.raises(reproduction.ReproductionError, match=message):
        reproduction.verify_resume_checkpoint(
            verified=verified,
            training_root=training_root,
            staging_root=staging_root,
            upstream_output=upstream_output,
            requested_step=1500,
            attempt_id="attempt-a",
            contract=contract,
        )


@pytest.mark.parametrize("step", [0, 1499, 12000, 13500])
def test_verify_resume_checkpoint_rejects_non_boundary_steps(
    tmp_path: Path, step: int
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout, source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id, disk_id=contract.disk_id, contract=contract,
    )
    training_root, staging_root, upstream_output = _materialize_partial_training(
        tmp_path, verified=verified, contract=contract
    )
    with pytest.raises(reproduction.ReproductionError, match="resume step"):
        reproduction.verify_resume_checkpoint(
            verified=verified, training_root=training_root, staging_root=staging_root,
            upstream_output=upstream_output, requested_step=step,
            attempt_id="attempt-a", contract=contract,
        )


def _complete_resumed_training(
    *, training_root: Path, staging_root: Path, upstream_output: Path,
    receipt: dict[str, object], advance: bool = True,
) -> Path:
    attempt_id = str(receipt["attempt_id"])
    lineage = staging_root / f"evidence/resume-attempts/{attempt_id}.json"
    lineage.parent.mkdir(parents=True, exist_ok=True)
    lineage.write_bytes(_canonical(receipt))
    checkpoint_1500 = upstream_output / "checkpoints/001500"
    shutil.copytree(checkpoint_1500, upstream_output / "checkpoints/012000")
    last = upstream_output / "checkpoints/last"; last.unlink(); last.symlink_to("012000")
    (upstream_output / "checkpoints/012000/training_state/training_step.json").write_bytes(
        _canonical({"step": 12000})
    )
    if advance:
        final = upstream_output / "checkpoints/012000"
        for relative in (
            "pretrained_model/model.safetensors",
            "training_state/optimizer_state.safetensors",
            "training_state/rng_state.safetensors",
        ):
            path = final / relative
            path.write_bytes(path.read_bytes() + b" resumed-through-step-12000")
        (final / "training_state/scheduler_state.json").write_bytes(
            _canonical({"last_epoch": 12000})
        )
    (staging_root / "logs/train.log").write_text(
        "Checkpoint policy after step 1500\n", encoding="utf-8"
    )
    (staging_root / f"logs/train-resume-{attempt_id}.log").write_text(
        "Checkpoint policy after step 12000\nEnd of training\n", encoding="utf-8"
    )
    upstream_output.rename(training_root)
    (staging_root / "evidence").rename(training_root / "evidence")
    (staging_root / "logs").rename(training_root / "logs")
    (staging_root / "runtime").rename(training_root / "runtime")
    staging_root.rmdir()
    _write_training_checksums(training_root)
    return lineage


def _materialize_native_resume_completion(
    *, upstream_output: Path, staging_root: Path, receipt: dict[str, object]
) -> None:
    attempt_id = str(receipt["attempt_id"])
    attempts = staging_root / "evidence/resume-attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    (attempts / f"{attempt_id}.json").write_bytes(_canonical(receipt))
    source = upstream_output / "checkpoints/001500"
    final = upstream_output / "checkpoints/012000"
    shutil.copytree(source, final)
    last = upstream_output / "checkpoints/last"
    last.unlink(); last.symlink_to("012000")
    (final / "training_state/training_step.json").write_bytes(
        _canonical({"step": 12000})
    )
    (final / "training_state/scheduler_state.json").write_bytes(
        _canonical({"last_epoch": 12000})
    )
    for relative in (
        "pretrained_model/model.safetensors",
        "training_state/optimizer_state.safetensors",
        "training_state/rng_state.safetensors",
    ):
        path = final / relative
        path.write_bytes(path.read_bytes() + b" resumed-through-step-12000")
    (staging_root / "logs/train.log").write_text(
        "Checkpoint policy after step 1500\n", encoding="utf-8"
    )
    (staging_root / f"logs/train-resume-{attempt_id}.log").write_text(
        "Checkpoint policy after step 12000\nEnd of training\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    "fault_after",
    [
        "manifest", "manifest-linked", "upstream", "evidence", "logs", "runtime",
        "checksums-temporary", "checksums", "identity-temporary", "identity",
        "before-rename", "after-rename",
    ],
)
def test_training_finalization_recovers_every_boundary_with_one_atomic_publish(
    tmp_path: Path, fault_after: str,
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout, source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id, disk_id=contract.disk_id, contract=contract,
    )
    training_root, staging_root, upstream_output = _materialize_partial_training(
        tmp_path, verified=verified, contract=contract
    )
    lineage = reproduction.verify_resume_checkpoint(
        verified=verified, training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, requested_step=1500,
        attempt_id=f"attempt-finalize-{fault_after}", contract=contract,
    )
    _materialize_native_resume_completion(
        upstream_output=upstream_output, staging_root=staging_root, receipt=lineage
    )

    with pytest.raises(reproduction.ReproductionError, match="injected finalization fault"):
        reproduction.finalize_training_output(
            verified=verified, training_root=training_root, staging_root=staging_root,
            upstream_output=upstream_output, contract=contract,
            fault_after=fault_after,
        )
    if fault_after == "checksums":
        checksum = Path(f"{training_root}.finalizing") / "checksums.sha256"
        os.link(checksum, checksum.with_name(".checksums.sha256.crash-orphan"))
    result = reproduction.finalize_training_output(
        verified=verified, training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, contract=contract,
    )

    assert result["training_root"] == str(training_root.resolve())
    assert training_root.is_dir()
    assert not Path(f"{training_root}.finalizing").exists()
    assert not staging_root.exists() and not upstream_output.exists()
    assert not list(training_root.glob(".checksums.sha256.*"))
    assert not list(training_root.rglob(".training-finalization.json.*"))
    identity_path = training_root / "training-identity.json"
    assert identity_path.is_file() and not identity_path.is_symlink()
    assert json.loads(identity_path.read_text(encoding="ascii")) == result


def test_training_finalization_rejects_noncanonical_upstream_before_moving_state(
    tmp_path: Path,
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout, source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id, disk_id=contract.disk_id, contract=contract,
    )
    training_root, staging_root, upstream_output = _materialize_partial_training(
        tmp_path, verified=verified, contract=contract
    )
    wrong_upstream = tmp_path / "wrong-upstream"
    shutil.copytree(upstream_output, wrong_upstream, symlinks=True)

    with pytest.raises(reproduction.ReproductionError, match="canonical"):
        reproduction.finalize_training_output(
            verified=verified, training_root=training_root, staging_root=staging_root,
            upstream_output=wrong_upstream, contract=contract,
        )

    assert wrong_upstream.is_dir() and staging_root.is_dir()
    assert not Path(f"{training_root}.finalizing").exists()


def test_training_finalization_rejects_ambiguous_completed_and_finalizing_roots(
    tmp_path: Path,
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout, source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id, disk_id=contract.disk_id, contract=contract,
    )
    training_root, staging_root, upstream_output = _materialize_partial_training(
        tmp_path, verified=verified, contract=contract
    )
    lineage = reproduction.verify_resume_checkpoint(
        verified=verified, training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, requested_step=1500,
        attempt_id="attempt-ambiguous", contract=contract,
    )
    _materialize_native_resume_completion(
        upstream_output=upstream_output, staging_root=staging_root, receipt=lineage
    )
    reproduction.finalize_training_output(
        verified=verified, training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, contract=contract,
    )
    finalizing = Path(f"{training_root}.finalizing")
    finalizing.mkdir()

    with pytest.raises(reproduction.ReproductionError, match="ambiguous"):
        reproduction.finalize_training_output(
            verified=verified, training_root=training_root, staging_root=staging_root,
            upstream_output=upstream_output, contract=contract,
        )


def test_resumed_final_identity_authenticates_resume_lineage(tmp_path: Path) -> None:
    from lehome import n15_reproduction as reproduction
    from rollout_appliance.native_reference_site.training_identity import (
        TrainingIdentityError,
        validate_training_identity_receipt,
    )

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout, source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id, disk_id=contract.disk_id, contract=contract,
    )
    training_root, staging_root, upstream_output = _materialize_partial_training(
        tmp_path, verified=verified, contract=contract
    )
    lineage = reproduction.verify_resume_checkpoint(
        verified=verified, training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, requested_step=1500,
        attempt_id="attempt-a", contract=contract,
    )
    _complete_resumed_training(
        training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, receipt=lineage,
    )

    identity = reproduction.verify_training_output(
        verified=verified, training_root=training_root, contract=contract
    )
    assert identity["resume_lineage"] == [
        {
            "attempt_id": "attempt-a",
            "requested_step": 1500,
            "receipt": "evidence/resume-attempts/attempt-a.json",
            "receipt_sha256": _sha(
                (training_root / "evidence/resume-attempts/attempt-a.json").read_bytes()
            ),
            "log": "logs/train-resume-attempt-a.log",
            "log_sha256": _sha(
                (training_root / "logs/train-resume-attempt-a.log").read_bytes()
            ),
        }
    ]
    identity_path = training_root / "training-identity.json"
    identity_path.write_bytes(_canonical(identity))
    admitted = validate_training_identity_receipt(
        identity_path, expected_contract=contract,
        expected_pretrained_root=training_root / "checkpoints/012000/pretrained_model",
    )
    assert admitted["resume_lineage"] == identity["resume_lineage"]

    lineage_path = training_root / "evidence/resume-attempts/attempt-a.json"
    altered_lineage = json.loads(lineage_path.read_text(encoding="ascii"))
    altered_lineage["train_config_origin"]["source_artifact_sha256"] = "0" * 64
    lineage_path.write_bytes(_canonical(altered_lineage))
    identity_path.unlink()
    identity["resume_lineage"][0]["receipt_sha256"] = _sha(lineage_path.read_bytes())
    _rewrite_task1_identity(training_root, identity, identity_path)
    with pytest.raises(TrainingIdentityError, match="resume lineage receipt"):
        validate_training_identity_receipt(
            identity_path, expected_contract=contract,
            expected_pretrained_root=training_root / "checkpoints/012000/pretrained_model",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("model", "advance"), ("optimizer", "advance"), ("rng", "advance"),
        ("scheduler", "advance"), ("scheduler-step", "advance"),
        ("source-scheduler-step", "source scheduler"),
    ],
)
def test_task2_resumed_identity_enforces_final_state_advancement_parity(
    tmp_path: Path, mutation: str, message: str,
) -> None:
    from lehome import n15_reproduction as reproduction
    from rollout_appliance.native_reference_site.training_identity import (
        TrainingIdentityError,
        validate_training_identity_receipt,
    )

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout, source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id, disk_id=contract.disk_id, contract=contract,
    )
    training_root, staging_root, upstream_output = _materialize_partial_training(
        tmp_path, verified=verified, contract=contract
    )
    lineage = reproduction.verify_resume_checkpoint(
        verified=verified, training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, requested_step=1500,
        attempt_id=f"attempt-task2-{mutation}", contract=contract,
    )
    _complete_resumed_training(
        training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, receipt=lineage,
    )
    receipt = reproduction.verify_training_output(
        verified=verified, training_root=training_root, contract=contract
    )
    relative = {
        "model": "pretrained_model/model.safetensors",
        "optimizer": "training_state/optimizer_state.safetensors",
        "rng": "training_state/rng_state.safetensors",
        "scheduler": "training_state/scheduler_state.json",
    }.get(mutation)
    final = training_root / "checkpoints/012000"
    source = training_root / "checkpoints/001500"
    if mutation == "source-scheduler-step":
        source_scheduler = source / "training_state/scheduler_state.json"
        source_scheduler.write_bytes(_canonical({"last_epoch": 1499}))
        lineage_path = training_root / f"evidence/resume-attempts/attempt-task2-{mutation}.json"
        lineage_value = json.loads(lineage_path.read_text(encoding="ascii"))
        lineage_value["checkpoint_files"]["training_state/scheduler_state.json"] = _sha(
            source_scheduler.read_bytes()
        )
        lineage_path.write_bytes(_canonical(lineage_value))
        receipt["resume_lineage"][0]["receipt_sha256"] = _sha(lineage_path.read_bytes())
    elif relative is not None:
        (final / relative).write_bytes((source / relative).read_bytes())
    else:
        (final / "training_state/scheduler_state.json").write_bytes(
            _canonical({"last_epoch": 11999})
        )
    identity = tmp_path / f"task2-resume-{mutation}.json"
    _rewrite_task1_identity(training_root, receipt, identity)

    with pytest.raises(TrainingIdentityError, match=message):
        validate_training_identity_receipt(
            identity, expected_contract=contract,
            expected_pretrained_root=final / "pretrained_model",
        )


def test_same_checkpoint_boundary_supports_distinct_authenticated_attempts(
    tmp_path: Path,
) -> None:
    """Preemption before a new checkpoint must not poison the 001500 boundary."""
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout, source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id, disk_id=contract.disk_id, contract=contract,
    )
    training_root, staging_root, upstream_output = _materialize_partial_training(
        tmp_path, verified=verified, contract=contract
    )
    first = reproduction.verify_resume_checkpoint(
        verified=verified, training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, requested_step=1500,
        attempt_id="attempt-first", contract=contract,
    )
    attempts = staging_root / "evidence/resume-attempts"
    attempts.mkdir(parents=True)
    (attempts / "attempt-first.json").write_bytes(_canonical(first))
    # A process can die after authenticating the checkpoint but before its
    # launch log is created.  That immutable orphan remains valid lineage.
    preempted = reproduction.verify_resume_checkpoint(
        verified=verified, training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, requested_step=1500,
        attempt_id="attempt-preempted", contract=contract,
    )
    (attempts / "attempt-preempted.json").write_bytes(_canonical(preempted))
    (staging_root / "logs/train-resume-attempt-preempted.log").write_text(
        "resume admitted at step 1500\npreempted\n", encoding="utf-8"
    )

    second = reproduction.verify_resume_checkpoint(
        verified=verified, training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, requested_step=1500,
        attempt_id="attempt-second", contract=contract,
    )
    _complete_resumed_training(
        training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, receipt=second,
    )
    identity = reproduction.verify_training_output(
        verified=verified, training_root=training_root, contract=contract
    )

    assert [item["attempt_id"] for item in identity["resume_lineage"]] == [
        "attempt-first", "attempt-preempted", "attempt-second"
    ]
    assert [item["requested_step"] for item in identity["resume_lineage"]] == [
        1500, 1500, 1500
    ]
    assert identity["resume_lineage"][0]["log_sha256"] is None
    assert all(item["log_sha256"] for item in identity["resume_lineage"][1:])


def test_resumed_output_rejects_relabelled_checkpoint_without_state_advancement(
    tmp_path: Path,
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout, source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id, disk_id=contract.disk_id, contract=contract,
    )
    training_root, staging_root, upstream_output = _materialize_partial_training(
        tmp_path, verified=verified, contract=contract
    )
    lineage = reproduction.verify_resume_checkpoint(
        verified=verified, training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, requested_step=1500,
        attempt_id="attempt-no-advancement", contract=contract,
    )
    _complete_resumed_training(
        training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, receipt=lineage, advance=False,
    )

    with pytest.raises(reproduction.ReproductionError, match="did not advance"):
        reproduction.verify_training_output(
            verified=verified, training_root=training_root, contract=contract
        )


@pytest.mark.parametrize(
    "setup_point",
    ["compatibility", "runtime-image", "overlays", "runtime-receipt", "execution-manifest"],
)
def test_resume_retry_cleans_authenticated_scratch_after_each_setup_preemption(
    tmp_path: Path, setup_point: str,
) -> None:
    from lehome.n15_reproduction import cleanup_resume_scratch, prepare_resume_scratch

    staging = tmp_path / "training.evidence-staging"
    staging.mkdir(mode=0o700)
    first = prepare_resume_scratch(staging_root=staging, attempt_id="attempt-first")
    interrupted = first / setup_point
    if "." in setup_point:
        interrupted.write_bytes(b"partial\n")
    else:
        interrupted.mkdir()
        (interrupted / "partial.bin").write_bytes(b"partial\n")

    second = prepare_resume_scratch(staging_root=staging, attempt_id="attempt-second")

    assert not first.exists()
    assert second.is_dir() and not second.is_symlink()
    owner = json.loads((second / "owner.json").read_text(encoding="ascii"))
    assert owner == {
        "schema_version": 1,
        "kind": "lehome_public_n15_resume_scratch_v1",
        "attempt_id": "attempt-second",
    }
    cleanup_resume_scratch(staging_root=staging, attempt_id="attempt-second")
    assert not second.exists()


def test_resume_scratch_cleanup_fails_closed_on_an_unsafe_stale_entry(
    tmp_path: Path,
) -> None:
    from lehome.n15_reproduction import ReproductionError, prepare_resume_scratch

    staging = tmp_path / "training.evidence-staging"
    staging.mkdir(mode=0o700)
    outside = tmp_path / "outside"; outside.mkdir()
    (staging / ".resume-scratch-attempt-stale").symlink_to(outside)

    with pytest.raises(ReproductionError, match="scratch"):
        prepare_resume_scratch(staging_root=staging, attempt_id="attempt-next")
    assert outside.is_dir()


@pytest.mark.parametrize(
    "mutation",
    ["missing_receipt", "wrong_receipt", "missing_log", "missing_checkpoint_hash", "missing_evidence_hash"],
)
def test_resumed_final_verification_rejects_missing_or_wrong_resume_evidence(
    tmp_path: Path, mutation: str
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout, source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id, disk_id=contract.disk_id, contract=contract,
    )
    training_root, staging_root, upstream_output = _materialize_partial_training(
        tmp_path, verified=verified, contract=contract
    )
    lineage = reproduction.verify_resume_checkpoint(
        verified=verified, training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, requested_step=1500,
        attempt_id="attempt-a", contract=contract,
    )
    _complete_resumed_training(
        training_root=training_root, staging_root=staging_root,
        upstream_output=upstream_output, receipt=lineage,
    )
    if mutation == "missing_receipt":
        (training_root / "evidence/resume-attempts/attempt-a.json").unlink()
    elif mutation == "wrong_receipt":
        path = training_root / "evidence/resume-attempts/attempt-a.json"
        value = json.loads(path.read_text()); value["requested_step"] = 3000
        path.write_bytes(_canonical(value))
    elif mutation == "missing_log":
        (training_root / "logs/train-resume-attempt-a.log").unlink()
    else:
        path = training_root / "evidence/resume-attempts/attempt-a.json"
        value = json.loads(path.read_text())
        mapping = value[
            "checkpoint_files" if mutation == "missing_checkpoint_hash" else "evidence_files"
        ]
        mapping.pop(next(iter(mapping)))
        path.write_bytes(_canonical(value))
    _write_training_checksums(training_root)
    with pytest.raises(reproduction.ReproductionError, match="resume"):
        reproduction.verify_training_output(
            verified=verified, training_root=training_root, contract=contract
        )


def test_verify_training_output_requires_step_12000_receipts_logs_and_checksums(
    tmp_path: Path,
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout,
        source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id,
        disk_id=contract.disk_id,
        contract=contract,
    )
    training_root = _materialize_training_output(
        tmp_path,
        verified=verified,
        contract=contract,
    )

    receipt = reproduction.verify_training_output(
        verified=verified,
        training_root=training_root,
        contract=contract,
    )

    assert receipt["kind"] == "lehome_public_n15_verified_training_output_v1"
    assert receipt["step"] == 12000
    assert receipt["checkpoint_root"] == str(
        (training_root / "checkpoints/012000").resolve()
    )
    assert receipt["artifact_count"] >= 19

    (training_root / "training-identity.json").write_text("{}\n", encoding="ascii")
    (training_root / "training-publication.json").write_text("{}\n", encoding="ascii")
    resumed = reproduction.verify_training_output(
        verified=verified,
        training_root=training_root,
        contract=contract,
    )
    assert resumed == receipt


def test_verify_training_output_accepts_a_venv_python_executable_symlink(
    tmp_path: Path,
) -> None:
    from lehome import n15_reproduction as reproduction
    from rollout_appliance.native_reference_site.training_identity import (
        validate_training_identity_receipt,
    )

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout,
        source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id,
        disk_id=contract.disk_id,
        contract=contract,
    )
    root = _materialize_training_output(
        tmp_path,
        verified=verified,
        contract=contract,
    )
    runtime_receipt = root / "evidence/runtime-receipt.json"
    runtime = json.loads(runtime_receipt.read_text(encoding="ascii"))
    interpreter = Path(runtime["python_executable"])
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(interpreter)
    runtime["python_executable"] = str(venv_python)
    runtime_receipt.write_bytes(_canonical(runtime))
    _write_training_checksums(root)

    receipt = reproduction.verify_training_output(
        verified=verified,
        training_root=root,
        contract=contract,
    )
    identity_path = root / "training-identity.json"
    identity_path.write_bytes(_canonical(receipt))
    admitted = validate_training_identity_receipt(
        identity_path,
        expected_contract=contract,
        expected_pretrained_root=root / "checkpoints/012000/pretrained_model",
    )

    assert receipt["step"] == 12000
    assert admitted["step"] == 12000


def _rewrite_task1_identity(root: Path, receipt: dict[str, object], output: Path) -> None:
    _write_training_checksums(root)
    files = {
        path.relative_to(root).as_posix(): _sha(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name != "checksums.sha256"
    }
    receipt.update(
        {
            "checkpoint_files": {
                relative: digest for relative, digest in files.items()
                if relative.startswith("checkpoints/012000/")
            },
            "artifact_count": len(files),
            "checksums_sha256": _sha((root / "checksums.sha256").read_bytes()),
            "source_receipt_sha256": files["evidence/source-receipt.json"],
            "resolved_snapshots_receipt_sha256": files[
                "evidence/resolved-snapshots-receipt.json"
            ],
        }
    )
    output.write_bytes(_canonical(receipt))


@pytest.mark.parametrize(
    "mutation",
    (
        "pretrained",
        "training_state",
        "source_evidence",
        "execution_manifest",
        "uv_lock",
        "runtime_receipt",
        "peft_overlay",
        "flash_overlay",
        "container_runtime",
        "wheel",
        "installed_package",
        "training_log",
    ),
)
def test_task2_identity_admission_has_exact_task1_output_parity(
    tmp_path: Path, mutation: str
) -> None:
    from lehome import n15_reproduction as reproduction
    from rollout_appliance.native_reference_site.training_identity import (
        TrainingIdentityError,
        validate_training_identity_receipt,
    )

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout,
        source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id,
        disk_id=contract.disk_id,
        contract=contract,
    )
    root = _materialize_training_output(tmp_path, verified=verified, contract=contract)
    receipt = reproduction.verify_training_output(
        verified=verified, training_root=root, contract=contract
    )
    if mutation == "pretrained":
        (root / "checkpoints/012000/pretrained_model/train_config.json").write_bytes(
            _canonical({"batch_size": 64, "steps": 1})
        )
    elif mutation == "training_state":
        (root / "checkpoints/012000/training_state/optimizer_state.safetensors").unlink()
    elif mutation == "source_evidence":
        (root / "evidence/source-receipt.json").write_bytes(b"{}\n")
    elif mutation == "execution_manifest":
        value = json.loads((root / "evidence/execution-manifest.json").read_text())
        value["execution"]["argv"] = ["different"]
        (root / "evidence/execution-manifest.json").write_bytes(_canonical(value))
    elif mutation == "uv_lock":
        (root / "evidence/uv.lock").write_bytes(b"different lock\n")
    elif mutation == "runtime_receipt":
        value = json.loads((root / "evidence/runtime-receipt.json").read_text())
        value["dependency_lock_path"] = "/wrong/path"
        (root / "evidence/runtime-receipt.json").write_bytes(_canonical(value))
    elif mutation == "peft_overlay":
        (root / "evidence/peft-overlay-receipt.json").write_bytes(b"{}\n")
    elif mutation == "flash_overlay":
        (root / "evidence/flash-attention-runtime-receipt.json").write_bytes(b"{}\n")
    elif mutation == "container_runtime":
        value = json.loads(
            (root / "evidence/training-container-runtime-receipt.json").read_text()
        )
        value["image_id"] = "sha256:" + "0" * 64
        (root / "evidence/training-container-runtime-receipt.json").write_bytes(
            _canonical(value)
        )
    elif mutation == "wheel":
        wheel = root / "evidence/compatibility/lerobot-0.4.3-py3-none-any.whl"
        wheel.chmod(0o644)
        wheel.write_bytes(b"not a wheel")
    elif mutation == "installed_package":
        (root / "runtime/site-packages/lerobot/policy.py").write_bytes(b"tampered\n")
    else:
        (root / "logs/train.log").write_text("still running\n")
    identity = tmp_path / f"identity-{mutation}.json"
    _rewrite_task1_identity(root, receipt, identity)
    with pytest.raises(TrainingIdentityError):
        validate_training_identity_receipt(
            identity,
            expected_contract=contract,
            expected_pretrained_root=root / "checkpoints/012000/pretrained_model",
        )


def test_task2_identity_admission_accepts_the_actual_task1_valid_fixture(tmp_path: Path) -> None:
    from lehome import n15_reproduction as reproduction
    from rollout_appliance.native_reference_site.training_identity import (
        validate_training_identity_receipt,
    )

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout, source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id, disk_id=contract.disk_id, contract=contract,
    )
    root = _materialize_training_output(tmp_path, verified=verified, contract=contract)
    receipt = reproduction.verify_training_output(
        verified=verified, training_root=root, contract=contract
    )
    identity = root / "training-identity.json"
    identity.write_bytes(_canonical(receipt))
    (root / "training-publication.json").write_bytes(_canonical({"published": True}))
    admitted = validate_training_identity_receipt(
        identity,
        expected_contract=contract,
        expected_pretrained_root=root / "checkpoints/012000/pretrained_model",
    )
    assert admitted["artifact_count"] == receipt["artifact_count"]


@pytest.mark.parametrize(
    "problem",
    [
        "checkpoint",
        "log",
        "source_receipt",
        "checksum",
        "symlink",
        "execution_manifest",
        "runtime",
        "fake_step",
        "fake_log",
        "tampered_runtime_file",
        "invented_step_fields",
        "wrong_python",
    ],
)
def test_verify_training_output_rejects_incomplete_or_unsafe_artifacts(
    tmp_path: Path,
    problem: str,
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout,
        source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id,
        disk_id=contract.disk_id,
        contract=contract,
    )
    root = _materialize_training_output(
        tmp_path,
        verified=verified,
        contract=contract,
    )
    if problem == "checkpoint":
        (root / "checkpoints/012000/pretrained_model/model.safetensors").unlink()
    elif problem == "log":
        (root / "logs/train.log").write_bytes(b"")
    elif problem == "source_receipt":
        (root / "evidence/source-receipt.json").write_bytes(b"{}\n")
    elif problem == "checksum":
        (root / "checksums.sha256").write_text("0" * 64 + "  logs/train.log\n")
    elif problem == "symlink":
        model = root / "checkpoints/012000/pretrained_model/model.safetensors"
        outside = tmp_path / "outside"
        outside.write_bytes(model.read_bytes())
        model.unlink()
        model.symlink_to(outside)
    elif problem == "execution_manifest":
        (root / "evidence/execution-manifest.json").write_bytes(b"{}\n")
    elif problem == "runtime":
        receipt = json.loads((root / "evidence/runtime-receipt.json").read_text())
        receipt["lerobot_wheel_sha256"] = "0" * 64
        (root / "evidence/runtime-receipt.json").write_bytes(_canonical(receipt))
    elif problem == "fake_step":
        (root / "checkpoints/012000/training_state/training_step.json").write_bytes(
            _canonical({"step": 11999})
        )
    elif problem == "fake_log":
        (root / "logs/train.log").write_text("step 12000 complete\n", encoding="utf-8")
    elif problem == "tampered_runtime_file":
        (root / "runtime/site-packages/lerobot/__init__.py").write_bytes(
            b'__version__ = "tampered"\n'
        )
    elif problem == "invented_step_fields":
        (root / "checkpoints/012000/training_state/training_step.json").write_bytes(
            _canonical({"batch_size": 64, "num_processes": 1, "step": 12000})
        )
    else:
        receipt = json.loads((root / "evidence/runtime-receipt.json").read_text())
        receipt["python_executable"] = str(Path("/usr/bin/python3").resolve())
        (root / "evidence/runtime-receipt.json").write_bytes(_canonical(receipt))
    if problem != "checksum":
        _write_training_checksums(root)

    with pytest.raises(reproduction.ReproductionError):
        reproduction.verify_training_output(
            verified=verified,
            training_root=root,
            contract=contract,
        )
