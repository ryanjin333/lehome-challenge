"""Offline tests for the immutable public GR00T N1.5 contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import io
import json
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
            "lerobot-0.4.3.dist-info/METADATA": b"Name: lerobot\nVersion: 0.4.3\n",
        }.items():
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return stream.getvalue()


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
        path.write_bytes(relative.encode("utf-8"))
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
    (model / "config.json").write_bytes(b"model config")
    (model / "model.safetensors").write_bytes(b"model weights")
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
    (dataset_snapshot / "meta/info.json").write_bytes(b"dataset metadata")
    (dataset_snapshot / "data/chunk-000.parquet").write_bytes(b"dataset rows")
    for relative, digest in _manifest(dataset_snapshot).items():
        target = dataset / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((dataset_snapshot / relative).read_bytes())
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
    assert CONTRACT.base_model_metadata_sha256 == "6da0e2fe38e294ca938d0540d0a23363446e54f941d16ce953612c443e43474a"
    assert CONTRACT.dataset_metadata_count == 67
    assert CONTRACT.dataset_metadata_sha256 == "63d5f109d26950a5091f82161750406ddbb7461cf24dfa1e7909897cbca4b71a"
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
        "argv": ["lerobot-train", "--config_path=configs/train_groot.yaml"],
        "cwd": str(checkout.resolve()),
        "env": {
            "HF_HUB_CACHE": str((tmp_path / "hub").resolve()),
            "HF_HUB_OFFLINE": "1",
        },
        "shell_argv": "lerobot-train --config_path=configs/train_groot.yaml",
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
    python_candidate = shutil.which("python3.11")
    assert python_candidate is not None
    python = Path(python_candidate).resolve()
    wheel = evidence / "lerobot-0.4.3-py3-none-any.whl"
    wheel.write_bytes(_fixture_wheel_bytes())
    package_root = runtime / "site-packages/lerobot"
    with zipfile.ZipFile(io.BytesIO(_fixture_wheel_bytes())) as archive:
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
                "lerobot_wheel_path": str(wheel.resolve()),
                "lerobot_wheel_sha256": contract.lerobot_wheel_sha256,
                "lerobot_package_root": str(package_root.resolve()),
                "dependency_lock_path": str((evidence / "uv.lock").resolve()),
                "dependency_lock_sha256": contract.dependency_lock_sha256,
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
