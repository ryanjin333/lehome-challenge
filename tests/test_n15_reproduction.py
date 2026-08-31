"""Offline tests for the immutable public GR00T N1.5 contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

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


def _materialize_source(tmp_path: Path) -> tuple[Path, Path]:
    from lehome.n15_reproduction import CONTRACT

    checkout = tmp_path / "source"
    for relative in SOURCE_FILES:
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    actual_files = {
        relative: _sha((checkout / relative).read_bytes()) for relative in SOURCE_FILES
    }
    receipt = {
        "schema_version": 1,
        "kind": "lehome_public_n15_source_v1",
        "repository": CONTRACT.source_repository,
        "revision": CONTRACT.source_revision,
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
    receipt = {
        "schema_version": 1,
        "kind": "lehome_public_n15_resolved_snapshots_v1",
        "base_model": {
            "repository": CONTRACT.base_model_repository,
            "revision": CONTRACT.base_model_revision,
            "root": str(model.resolve()),
        },
        "dataset": {
            "repository": CONTRACT.dataset_repository,
            "revision": CONTRACT.dataset_revision,
            "root": str(dataset.resolve()),
        },
        "vm_id": CONTRACT.vm_id,
        "disk_id": CONTRACT.disk_id,
    }
    receipt_path = tmp_path / "resolved-snapshots-receipt.json"
    receipt_path.write_bytes(_canonical(receipt))
    return model, dataset, receipt_path


def _fixture_contract(checkout: Path):
    from dataclasses import replace
    from lehome.n15_reproduction import CONTRACT

    return replace(
        CONTRACT,
        trusted_source_files={
            relative: _sha((checkout / relative).read_bytes())
            for relative in SOURCE_FILES
        },
    )


def test_contract_encodes_exact_public_recipe_and_is_immutable() -> None:
    from lehome.n15_reproduction import CONTRACT

    assert CONTRACT.source_repository == "theo-zhou/lehome-groot-submission-4"
    assert CONTRACT.source_revision == "d384fe00508acd96ab1c3c5dc265e08261f94b3b"
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
    source_receipt: Path,
    snapshots_receipt: Path,
) -> Path:
    root = tmp_path / "training-output"
    checkpoint = root / "checkpoints/012000"
    evidence = root / "evidence"
    logs = root / "logs"
    checkpoint.mkdir(parents=True)
    evidence.mkdir()
    logs.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    (evidence / "source-receipt.json").write_bytes(source_receipt.read_bytes())
    (evidence / "resolved-snapshots-receipt.json").write_bytes(
        snapshots_receipt.read_bytes()
    )
    (logs / "train.log").write_text("step 12000 complete\n", encoding="utf-8")
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            lines.append(f"{_sha(path.read_bytes())}  {relative}\n")
    (root / "checksums.sha256").write_text("".join(lines), encoding="ascii")
    return root


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
        source_receipt=source_receipt,
        snapshots_receipt=snapshots_receipt,
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
    assert receipt["artifact_count"] == 4


@pytest.mark.parametrize("problem", ["checkpoint", "log", "source_receipt", "checksum", "symlink"])
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
        source_receipt=source_receipt,
        snapshots_receipt=snapshots_receipt,
    )
    if problem == "checkpoint":
        (root / "checkpoints/012000/model.safetensors").unlink()
    elif problem == "log":
        (root / "logs/train.log").write_bytes(b"")
    elif problem == "source_receipt":
        (root / "evidence/source-receipt.json").write_bytes(b"{}\n")
    elif problem == "checksum":
        (root / "checksums.sha256").write_text("0" * 64 + "  logs/train.log\n")
    else:
        model = root / "checkpoints/012000/model.safetensors"
        outside = tmp_path / "outside"
        outside.write_bytes(model.read_bytes())
        model.unlink()
        model.symlink_to(outside)

    with pytest.raises(reproduction.ReproductionError):
        reproduction.verify_training_output(
            verified=verified,
            training_root=root,
            contract=contract,
        )
