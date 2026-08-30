from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lehome_train.commands.memorize import (
    CHECKPOINT_INTERVAL,
    EVALUATION_INTERVAL,
    MAX_SAMPLE_PRESENTATIONS,
    PHYSICAL_BATCH_SIZE,
    ChunkReceipt,
    run_memorization,
    select_training_episode,
)
from lehome_train.offline_eval import OfflineEvaluation


SHA_A = "a" * 64
SHA_B = "b" * 64


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "output_format": "groot_lerobot_v2.1_per_episode",
                "train_episode_ids": ["11", "3"],
                "validation_episode_ids": ["7"],
            }
        ),
        encoding="utf-8",
    )
    return dataset


def _manifest_sha256(dataset: Path) -> str:
    return hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()


def _validation_report(dataset: Path) -> dict[str, object]:
    return {
        "valid": True,
        "dataset_manifest_sha256": _manifest_sha256(dataset),
        "train_episode_count": 2,
        "validation_episode_count": 1,
    }


def _evaluation(normalized_mse: float, dimensions: tuple[float, ...]) -> OfflineEvaluation:
    return OfflineEvaluation(
        normalized_mse=normalized_mse,
        dimension_mse=dimensions,
        frame_count=4,
        action_dimension=len(dimensions),
    )


def test_episode_selection_is_deterministic_and_restricted_to_training_split(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)

    assert select_training_episode(dataset) == "3"
    assert select_training_episode(dataset, requested_episode_id="11") == "11"
    with pytest.raises(ValueError, match="training split"):
        select_training_episode(dataset, requested_episode_id="7")


def test_episode_selection_requires_a_nonempty_validation_split(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    manifest["validation_episode_ids"] = []
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="episode split"):
        select_training_episode(dataset)


def test_failed_gate_uses_the_fixed_budget_and_exact_cadences(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    training_calls: list[tuple[str, int, int]] = []
    evaluation_calls: list[tuple[str, int]] = []
    checkpoint_calls: list[tuple[str, int]] = []

    optimizer_step = 0

    def trainer(
        *, episode_id: str, optimizer_steps: int, physical_batch_size: int
    ) -> ChunkReceipt:
        nonlocal optimizer_step
        training_calls.append((episode_id, optimizer_steps, physical_batch_size))
        receipt = ChunkReceipt(
            start_optimizer_step=optimizer_step,
            end_optimizer_step=optimizer_step + optimizer_steps,
            sample_presentations=optimizer_steps,
            physical_batch_size=physical_batch_size,
            finite_loss=True,
        )
        optimizer_step += optimizer_steps
        return receipt

    def evaluator(*, episode_id: str, sample_presentations: int) -> OfflineEvaluation:
        evaluation_calls.append((episode_id, sample_presentations))
        return _evaluation(
            1.0 if sample_presentations == 0 else 0.2,
            (1.0, 1.0) if sample_presentations == 0 else (0.2, 0.2),
        )

    result = run_memorization(
        dataset_path=dataset,
        experiment_id="memorize-001",
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=_manifest_sha256(dataset),
        trainer=trainer,
        evaluator=evaluator,
        checkpointer=lambda *, episode_id, sample_presentations: checkpoint_calls.append(
            (episode_id, sample_presentations)
        ),
        prepared_validator=lambda _dataset: _validation_report(dataset),
    )

    assert PHYSICAL_BATCH_SIZE == 1
    assert EVALUATION_INTERVAL == 500
    assert CHECKPOINT_INTERVAL == 1_000
    assert MAX_SAMPLE_PRESENTATIONS == 10_000
    assert training_calls == [
        ("3", EVALUATION_INTERVAL, PHYSICAL_BATCH_SIZE)
        for _ in range(MAX_SAMPLE_PRESENTATIONS // EVALUATION_INTERVAL)
    ]
    assert evaluation_calls == [
        ("3", step)
        for step in range(0, MAX_SAMPLE_PRESENTATIONS + 1, EVALUATION_INTERVAL)
    ]
    assert checkpoint_calls == [
        ("3", step)
        for step in range(CHECKPOINT_INTERVAL, MAX_SAMPLE_PRESENTATIONS + 1, CHECKPOINT_INTERVAL)
    ]
    assert result.sample_presentations == MAX_SAMPLE_PRESENTATIONS
    assert result.initialized_normalized_mse == 1.0
    assert result.final_normalized_mse == 0.2
    assert result.offline_gate_passed is False
    assert result.promotable is False
    assert result.pending_gate == "simulator_expert_replay"


def test_early_stopping_requires_two_consecutive_qualifying_evaluations(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    evaluations = {
        0: _evaluation(1.0, (0.01, 1.99)),
        500: _evaluation(0.01, (0.005, 0.015)),
        1_000: _evaluation(0.01, (0.01, 0.01)),
        1_500: _evaluation(0.01, (0.005, 0.015)),
        2_000: _evaluation(0.01, (0.004, 0.016)),
    }
    trained_steps: list[int] = []
    checkpoints: list[int] = []
    optimizer_step = 0

    def trainer(**values: object) -> ChunkReceipt:
        nonlocal optimizer_step
        optimizer_steps = values["optimizer_steps"]
        physical_batch_size = values["physical_batch_size"]
        assert type(optimizer_steps) is int
        assert type(physical_batch_size) is int
        trained_steps.append(optimizer_steps)
        receipt = ChunkReceipt(
            optimizer_step,
            optimizer_step + optimizer_steps,
            optimizer_steps,
            physical_batch_size,
            True,
        )
        optimizer_step += optimizer_steps
        return receipt

    result = run_memorization(
        dataset_path=dataset,
        requested_episode_id="11",
        experiment_id="memorize-002",
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=_manifest_sha256(dataset),
        trainer=trainer,
        evaluator=lambda **values: evaluations[values["sample_presentations"]],
        checkpointer=lambda **values: checkpoints.append(values["sample_presentations"]),
        prepared_validator=lambda _dataset: _validation_report(dataset),
    )

    assert trained_steps == [500, 500, 500, 500]
    assert checkpoints == [1_000, 2_000]
    assert result.episode_id == "11"
    assert result.sample_presentations == 2_000
    assert result.initialized_dimension_mse == (0.01, 1.99)
    assert result.final_dimension_mse == (0.004, 0.016)
    assert result.final_normalized_mse <= result.initialized_normalized_mse * 0.1
    assert all(
        final < initialized
        for initialized, final in zip(
            result.initialized_dimension_mse,
            result.final_dimension_mse,
            strict=True,
        )
    )
    assert result.offline_gate_passed is True
    assert result.promotable is False
    assert result.pending_gate == "simulator_expert_replay"


@pytest.mark.parametrize(
    ("receipt", "message"),
    [
        (ChunkReceipt(1, 501, 500, 1, True), "start"),
        (ChunkReceipt(0, 499, 500, 1, True), "optimizer"),
        (ChunkReceipt(0, 501, 500, 1, True), "optimizer"),
        (ChunkReceipt(0, 500, 499, 1, True), "presentations"),
        (ChunkReceipt(0, 500, 501, 1, True), "presentations"),
        (ChunkReceipt(0, 500, 500, 2, True), "batch"),
        (ChunkReceipt(0, 500, 500, 1, False), "finite loss"),
    ],
)
def test_chunk_receipts_must_prove_exact_finite_progress(
    tmp_path: Path,
    receipt: ChunkReceipt,
    message: str,
) -> None:
    dataset = _dataset(tmp_path)
    evaluation_steps: list[int] = []

    with pytest.raises(ValueError, match=message):
        run_memorization(
            dataset_path=dataset,
            experiment_id="memorize-invalid-receipt",
            experiment_config_sha256=SHA_A,
            dataset_manifest_sha256=_manifest_sha256(dataset),
            trainer=lambda **_values: receipt,
            evaluator=lambda **values: (
                evaluation_steps.append(values["sample_presentations"])
                or _evaluation(1.0, (1.0, 1.0))
            ),
            checkpointer=lambda **_values: pytest.fail("invalid chunks cannot checkpoint"),
            prepared_validator=lambda _dataset: _validation_report(dataset),
        )

    assert evaluation_steps == [0]


@pytest.mark.parametrize("finite_loss", [float("nan"), float("inf"), 1, "true"])
def test_chunk_receipt_rejects_non_boolean_finite_loss(finite_loss: object) -> None:
    with pytest.raises(ValueError, match="finite loss"):
        ChunkReceipt(0, 500, 500, 1, finite_loss)  # type: ignore[arg-type]


def test_chunk_receipts_must_advance_cumulative_optimizer_progress(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    receipts = iter(
        (
            ChunkReceipt(0, 500, 500, 1, True),
            ChunkReceipt(0, 500, 500, 1, True),
        )
    )

    with pytest.raises(ValueError, match="monotonic"):
        run_memorization(
            dataset_path=dataset,
            experiment_id="memorize-repeated-receipt",
            experiment_config_sha256=SHA_A,
            dataset_manifest_sha256=_manifest_sha256(dataset),
            trainer=lambda **_values: next(receipts),
            evaluator=lambda **_values: _evaluation(1.0, (1.0, 1.0)),
            checkpointer=lambda **_values: None,
            prepared_validator=lambda _dataset: _validation_report(dataset),
        )


def test_memorization_rejects_expected_manifest_digest_mismatch(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    with pytest.raises(ValueError, match="manifest digest"):
        run_memorization(
            dataset_path=dataset,
            experiment_id="memorize-digest-mismatch",
            experiment_config_sha256=SHA_A,
            dataset_manifest_sha256=SHA_B,
            trainer=lambda **_values: pytest.fail("training must not start"),
            evaluator=lambda **_values: pytest.fail("evaluation must not start"),
            checkpointer=lambda **_values: pytest.fail("checkpointing must not start"),
            prepared_validator=lambda _dataset: pytest.fail("validation must not start"),
        )


def test_memorization_rejects_manifest_mutation_during_validation(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    expected = _manifest_sha256(dataset)

    def mutate_manifest(_dataset: Path) -> dict[str, object]:
        report = _validation_report(dataset)
        (dataset / "manifest.json").write_bytes(b"mutated after initial digest")
        return report

    with pytest.raises(ValueError, match="changed during validation"):
        run_memorization(
            dataset_path=dataset,
            experiment_id="memorize-mutated-manifest",
            experiment_config_sha256=SHA_A,
            dataset_manifest_sha256=expected,
            trainer=lambda **_values: pytest.fail("training must not start"),
            evaluator=lambda **_values: pytest.fail("evaluation must not start"),
            checkpointer=lambda **_values: pytest.fail("checkpointing must not start"),
            prepared_validator=mutate_manifest,
        )


@pytest.mark.parametrize(
    "report_override",
    [
        {"valid": False},
        {"validation_episode_count": 0},
        {"dataset_manifest_sha256": SHA_B},
    ],
)
def test_memorization_rejects_invalid_prepared_dataset_report(
    tmp_path: Path,
    report_override: dict[str, object],
) -> None:
    dataset = _dataset(tmp_path)

    def invalid_report(_dataset: Path) -> dict[str, object]:
        report = _validation_report(dataset)
        report.update(report_override)
        return report

    with pytest.raises(ValueError, match="prepared dataset validation"):
        run_memorization(
            dataset_path=dataset,
            experiment_id="memorize-invalid-prepared",
            experiment_config_sha256=SHA_A,
            dataset_manifest_sha256=_manifest_sha256(dataset),
            trainer=lambda **_values: pytest.fail("training must not start"),
            evaluator=lambda **_values: pytest.fail("evaluation must not start"),
            checkpointer=lambda **_values: pytest.fail("checkpointing must not start"),
            prepared_validator=invalid_report,
        )


def test_memorization_uses_real_prepared_validator_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path)
    calls: list[Path] = []

    def validator(path: str | Path, **_kwargs: object) -> dict[str, object]:
        calls.append(Path(path))
        return _validation_report(dataset)

    monkeypatch.setattr("lehome_train.data.validate.validate_prepared_dataset", validator)
    step = 0

    def trainer(**values: object) -> ChunkReceipt:
        nonlocal step
        requested = values["optimizer_steps"]
        assert type(requested) is int
        receipt = ChunkReceipt(step, step + requested, requested, 1, True)
        step += requested
        return receipt

    run_memorization(
        dataset_path=dataset,
        experiment_id="memorize-default-validation",
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=_manifest_sha256(dataset),
        trainer=trainer,
        evaluator=lambda **_values: _evaluation(0.0, (0.0, 0.0)),
        checkpointer=lambda **_values: None,
    )

    assert calls == [dataset]
