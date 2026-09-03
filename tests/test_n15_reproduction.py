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
    (checkout / ".gitignore").write_text("Datasets*\n", encoding="utf-8")
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
