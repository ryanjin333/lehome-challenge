from __future__ import annotations

from pathlib import Path

from lehome_train.b1k.finalize import FinalEvidence, Finalizer


def test_finalizer_uploads_readbacks_private_run_branch_without_deleting_remote_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-15000"; checkpoint.mkdir(); (checkpoint / "trainer_state.json").write_text('{"global_step":15000}')
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"): (checkpoint / name).write_bytes(b"model")
    (checkpoint / "config.json").write_text("{}")
    evidence_dir = tmp_path / "evidence"; evidence_dir.mkdir()
    evidence_paths = []
    for name in ("run-contract.json", "selection.json", "materialized.json", "modality.py", "stats.json", "derivation.json", "revisions.json", "argv.json", "train.log", "receipt.json"):
        path = evidence_dir / name; path.write_text("{}"); evidence_paths.append(path)
    uploaded: dict[str, Path] = {}
    def upload(branch: str, path: str, source: Path) -> str:
        uploaded[f"{branch}/{path}"] = source
        return "f" * 40
    def download(branch: str, path: str, commit: str, destination: Path) -> None:
        destination.write_bytes(uploaded[f"{branch}/{path}"].read_bytes())
    evidence = FinalEvidence(*evidence_paths[:8], logs=(evidence_paths[8],), rolling_receipts=(evidence_paths[9],), world_size=1)
    receipt = Finalizer(upload_file=upload, download_file=download).finalize(run_id="b1k-run-001", checkpoint=checkpoint, evidence=evidence, final_dir=tmp_path / "final")
    assert receipt["branch"] == "runs/b1k-run-001"
    assert receipt["immutable_commit"] == "f" * 40
    assert (tmp_path / "final" / "final-receipt.json").exists()


def test_finalizer_creates_the_exact_run_branch_before_uploading(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-15000"; checkpoint.mkdir(); (checkpoint / "trainer_state.json").write_text('{"global_step":15000}')
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"): (checkpoint / name).write_bytes(b"model")
    (checkpoint / "config.json").write_text("{}")
    evidence_dir = tmp_path / "evidence"; evidence_dir.mkdir(); evidence_paths = []
    for name in ("run-contract.json", "selection.json", "materialized.json", "modality.py", "stats.json", "derivation.json", "revisions.json", "argv.json", "train.log", "receipt.json"):
        path = evidence_dir / name; path.write_text("{}"); evidence_paths.append(path)
    events: list[tuple[str, str]] = []; uploaded: dict[str, Path] = {}
    def ensure(branch: str) -> None: events.append(("ensure", branch))
    def upload(branch: str, path: str, source: Path) -> str:
        events.append(("upload", branch)); uploaded[f"{branch}/{path}"] = source; return "f" * 40
    def download(branch: str, path: str, _commit: str, destination: Path) -> None: destination.write_bytes(uploaded[f"{branch}/{path}"].read_bytes())
    evidence = FinalEvidence(*evidence_paths[:8], logs=(evidence_paths[8],), rolling_receipts=(evidence_paths[9],), world_size=1)

    Finalizer(upload_file=upload, download_file=download, ensure_branch=ensure).finalize(run_id="b1k-run-001", checkpoint=checkpoint, evidence=evidence, final_dir=tmp_path / "final")

    assert events[0] == ("ensure", "runs/b1k-run-001")
