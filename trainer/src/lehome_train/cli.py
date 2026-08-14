"""CPU-safe command-line entry point for portable training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import typer

from lehome_train.io import canonical_json_bytes


app = typer.Typer(help="LeHome GR00T N1.7 trainer.", no_args_is_help=True)
data_app = typer.Typer(help="Dataset inspection and transfer commands.", no_args_is_help=True)
model_app = typer.Typer(help="Pinned model hydration commands.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")


def _emit(value: object) -> None:
    typer.echo(canonical_json_bytes(value).decode("utf-8"))


def _fail_closed(operation) -> None:
    try:
        _emit(operation())
    except Exception as error:
        raise click.ClickException(str(error)) from None


@app.command("validate-training-capability")
def validate_training_capability(
    image_digest: str = typer.Option(..., "--image-digest"),
    one_step: bool = typer.Option(False, "--one-step"),
) -> None:
    """Run the bounded one-step Blackwell training capability probe."""
    if not one_step:
        raise typer.BadParameter("--one-step is required for the bounded capability probe")

    def operation() -> object:
        from lehome_train.release_manifest import capture_training_capability

        return capture_training_capability(image_digest=image_digest)

    _fail_closed(operation)


@data_app.command("inspect")
def data_inspect(
    source: Path = typer.Option(..., "--source"),
    output: Optional[Path] = typer.Option(None, "--output"),
) -> None:
    """Inspect an organizer dataset and write the proposed checked mapping."""

    def operation() -> object:
        from lehome_train.data.inspect import inspect_dataset

        return inspect_dataset(source, output_path=output)

    _fail_closed(operation)


@data_app.command("convert")
def data_convert(
    source: Path = typer.Option(..., "--source"),
    output: Path = typer.Option(..., "--output"),
    mapping: Path = typer.Option(..., "--mapping"),
    source_repository: str = typer.Option(..., "--source-repository"),
    source_revision: str = typer.Option(..., "--source-revision"),
    converter_commit: str = typer.Option(..., "--converter-commit"),
    container_digest: str = typer.Option(..., "--container-digest"),
    groot_root: Path = typer.Option(..., "--groot-root"),
    persistent_staging_root: Optional[Path] = typer.Option(None, "--persistent-staging-root"),
    unbound_staging_data_adoption_root: Optional[Path] = typer.Option(None, "--unbound-staging-data-adoption-root"),
) -> None:
    """Convert deterministically and compute pinned train-only statistics."""

    def operation() -> object:
        from lehome_train.data.convert import (
            convert_dataset,
            persistent_destination_operation_lock,
        )
        from lehome_train.data.stats import write_train_statistics

        def convert_and_write_statistics() -> dict[str, object]:
            manifest = convert_dataset(
                source,
                output,
                mapping_path=mapping,
                source_repository=source_repository,
                source_revision=source_revision,
                converter_commit=converter_commit,
                converter_container_digest=container_digest,
                persistent_staging_root=persistent_staging_root,
                unbound_staging_data_adoption_root=unbound_staging_data_adoption_root,
            )
            statistics = write_train_statistics(output, groot_root=groot_root)
            return {"manifest": manifest, "statistics": statistics}

        if persistent_staging_root is None:
            return convert_and_write_statistics()
        with persistent_destination_operation_lock(output):
            return convert_and_write_statistics()

    _fail_closed(operation)


@data_app.command("validate")
def data_validate(
    dataset: Path = typer.Option(..., "--dataset"),
    groot_root: Path = typer.Option(..., "--groot-root"),
) -> None:
    """Validate schema, hashes, statistics, and one pinned loader batch."""

    def operation() -> object:
        from lehome_train.data.validate import validate_prepared_dataset

        return validate_prepared_dataset(dataset, groot_root=groot_root)

    _fail_closed(operation)


@data_app.command("publish")
def data_publish(
    dataset: Path = typer.Option(..., "--dataset"),
    repository: str = typer.Option(..., "--repo"),
    revision: str = typer.Option(..., "--revision"),
    staging_root: Path = typer.Option(..., "--staging-root"),
    timeout_seconds: float = typer.Option(30.0, "--timeout-seconds"),
) -> None:
    """Publish the validated allowlist and verify its immutable readback."""

    def operation() -> object:
        from lehome_train.data.publish import publish_prepared_dataset
        from lehome_train.hub import HuggingFaceHubTransport

        published = publish_prepared_dataset(
            dataset,
            repository=repository,
            revision=revision,
            transport=HuggingFaceHubTransport(timeout_seconds=timeout_seconds),
            staging_root=staging_root,
        )
        return {
            "repository": published.repository,
            "revision": published.revision,
            "dataset_manifest_sha256": published.dataset_manifest_sha256,
            "entries": [entry.to_dict() for entry in published.entries],
        }

    _fail_closed(operation)


@data_app.command("retrieve")
def data_retrieve(
    destination: Path = typer.Option(..., "--destination"),
    repository: str = typer.Option(..., "--repo"),
    revision: str = typer.Option(..., "--revision"),
    manifest_sha256: str = typer.Option(..., "--manifest-sha256"),
    timeout_seconds: float = typer.Option(30.0, "--timeout-seconds"),
    max_attempts: int = typer.Option(3, "--max-attempts"),
    snapshot_manifest: Optional[Path] = typer.Option(None, "--snapshot-manifest"),
) -> None:
    """Hydrate one immutable prepared dataset after complete hash verification."""

    def operation() -> object:
        from lehome_train.data.publish import (
            download_prepared_dataset,
            write_prepared_snapshot_manifest,
        )
        from lehome_train.hub import HuggingFaceHubTransport

        restored = download_prepared_dataset(
            destination,
            repository=repository,
            revision=revision,
            expected_manifest_sha256=manifest_sha256,
            transport=HuggingFaceHubTransport(timeout_seconds=timeout_seconds),
            max_attempts=max_attempts,
        )
        evidence = write_prepared_snapshot_manifest(
            restored,
            (
                destination / "lehome_dataset_snapshot.json"
                if snapshot_manifest is None
                else snapshot_manifest
            ),
            revision=revision,
        )
        return {
            "schema_version": 1,
            "status": "dataset_restored",
            "destination": str(restored),
            "repository": repository,
            "revision": revision,
            "dataset_manifest_sha256": manifest_sha256,
            "snapshot_manifest": str(evidence),
        }

    _fail_closed(operation)


@model_app.command("retrieve")
def model_retrieve(
    destination: Path = typer.Option(..., "--destination"),
    repository: str = typer.Option("nvidia/GR00T-N1.7-3B", "--repo"),
    revision: str = typer.Option(..., "--revision"),
    staging_root: Path = typer.Option(..., "--staging-root"),
    timeout_seconds: float = typer.Option(30.0, "--timeout-seconds"),
) -> None:
    """Hydrate the exact pinned complete base-model snapshot beneath /cache."""

    def operation() -> object:
        cache = Path("/cache")
        for path, label in (
            (destination, "model destination"),
            (staging_root, "model staging root"),
        ):
            resolved = path.resolve(strict=False)
            if resolved != cache and cache not in resolved.parents:
                raise ValueError(f"{label} must stay beneath /cache")
        from lehome_train.groot.model_snapshot import (
            HuggingFaceModelSnapshotTransport,
            download_base_model,
        )

        restored = download_base_model(
            destination,
            repository=repository,
            revision=revision,
            transport=HuggingFaceModelSnapshotTransport(
                timeout_seconds=timeout_seconds
            ),
            staging_root=staging_root,
        )
        return {
            "schema_version": 1,
            "status": "model_restored",
            "destination": str(restored),
            "repository": repository,
            "revision": revision,
        }

    _fail_closed(operation)


def _gpu_command(command: str, request: Path, runtime_factory: Optional[str]) -> None:
    def operation() -> object:
        from lehome_train.runtime import dispatch_runtime_request

        return dispatch_runtime_request(
            command,
            request,
            factory_spec=runtime_factory,
        )

    _fail_closed(operation)


@app.command("build-runtime-mixture")
def build_runtime_mixture(
    request: Path = typer.Option(..., "--request"),
) -> None:
    """Stage immutable 70/30 bytes pending explicit private-repo publication."""

    def operation() -> object:
        from lehome_train.groot.runtime_mixture_builder import build_from_request

        return build_from_request(request)

    _fail_closed(operation)


@app.command("pilot-runtime-mixture")
def pilot_runtime_mixture(
    request: Path = typer.Option(..., "--request"),
) -> None:
    """Run the model-free authenticated runtime-mixture loader pilot."""

    def operation() -> object:
        from lehome_train.groot.runtime_mixture_builder import pilot_from_request

        return pilot_from_request(request)

    _fail_closed(operation)


@app.command("publish-runtime-source")
def publish_runtime_source(
    request: Path = typer.Option(..., "--request"),
) -> None:
    """Publish one BC or rollout source through the injected Hub boundary."""

    def operation() -> object:
        from lehome_train.groot.runtime_mixture_publish import publish_source_from_request
        from lehome_train.hub import HuggingFaceHubTransport

        return publish_source_from_request(request, transport=HuggingFaceHubTransport())

    _fail_closed(operation)


@app.command("publish-runtime-mixture")
def publish_runtime_mixture(
    request: Path = typer.Option(..., "--request"),
) -> None:
    """Publish builder-pending mixture bytes through the injected Hub boundary."""

    def operation() -> object:
        from lehome_train.groot.runtime_mixture_publish import publish_pending_mixture_from_request
        from lehome_train.hub import HuggingFaceHubTransport

        return publish_pending_mixture_from_request(request, transport=HuggingFaceHubTransport())

    _fail_closed(operation)


@app.command("finalize-runtime-mixture")
def finalize_runtime_mixture(
    request: Path = typer.Option(..., "--request"),
) -> None:
    """Publish verified final bytes and write the local runtime deployment mount."""

    def operation() -> object:
        from lehome_train.groot.runtime_mixture_publish import finalize_pending_mixture_from_request
        from lehome_train.hub import HuggingFaceHubTransport

        return {"destination": str(finalize_pending_mixture_from_request(request, transport=HuggingFaceHubTransport()))}

    _fail_closed(operation)


@app.command("prepare")
def prepare(
    request: Path = typer.Option(..., "--request"),
    runtime_factory: Optional[str] = typer.Option(None, "--runtime-factory"),
) -> None:
    """Run checked preflight through the image-provided runtime adapter."""

    _gpu_command("prepare", request, runtime_factory)


@app.command("memorize")
def memorize(
    request: Path = typer.Option(..., "--request"),
    runtime_factory: Optional[str] = typer.Option(None, "--runtime-factory"),
) -> None:
    """Run checked one-episode memorization through the runtime adapter."""

    _gpu_command("memorize", request, runtime_factory)


@app.command("smoke")
def smoke(
    request: Path = typer.Option(..., "--request"),
    runtime_factory: Optional[str] = typer.Option(None, "--runtime-factory"),
) -> None:
    """Run checked batch smoke tests through the runtime adapter."""

    _gpu_command("smoke", request, runtime_factory)


@app.command("tune")
def tune(
    request: Path = typer.Option(..., "--request"),
    runtime_factory: Optional[str] = typer.Option(None, "--runtime-factory"),
) -> None:
    """Measure the bounded corrective loader and batch candidates."""

    _gpu_command("tune", request, runtime_factory)


@app.command("train")
def train(
    request: Path = typer.Option(..., "--request"),
    runtime_factory: Optional[str] = typer.Option(None, "--runtime-factory"),
) -> None:
    """Run fixed-exposure training through the runtime adapter."""

    _gpu_command("train", request, runtime_factory)


@app.command("continuous-train")
def continuous_train(
    request: Path = typer.Option(..., "--request"),
    runtime_factory: Optional[str] = typer.Option(None, "--runtime-factory"),
) -> None:
    """Run sealed corrective training continuously through the runtime adapter."""

    _gpu_command("continuous-train", request, runtime_factory)


@app.command("runtime-mixture-train")
def runtime_mixture_train(
    request: Path = typer.Option(..., "--request"),
    runtime_factory: Optional[str] = typer.Option(None, "--runtime-factory"),
) -> None:
    """Run the explicit authenticated runtime-mixture production path."""
    _gpu_command("runtime-mixture-train", request, runtime_factory)


@app.command("report")
def report(request: Path = typer.Option(..., "--request")) -> None:
    """Build and write a complete provenance report from checked JSON."""

    def operation() -> object:
        from lehome_train.runtime import execute_report_request

        return execute_report_request(request)

    _fail_closed(operation)


@app.command("sync")
def sync(request: Path = typer.Option(..., "--request")) -> None:
    """Stage, upload, and immutable-readback verify the experiment."""

    def operation() -> object:
        from lehome_train.runtime import execute_sync_request

        return execute_sync_request(request)

    _fail_closed(operation)


@app.command("restore")
def restore(
    sync_result: Path = typer.Option(..., "--sync-result"),
    destination: Path = typer.Option(..., "--destination"),
    staging_root: Path = typer.Option(..., "--staging-root"),
    timeout_seconds: float = typer.Option(30.0, "--timeout-seconds"),
    max_attempts: int = typer.Option(3, "--max-attempts"),
) -> None:
    """Hydrate one immutable experiment snapshot for compatible resume."""

    def operation() -> object:
        from lehome_train.commands.restore import restore_experiment_snapshot
        from lehome_train.commands.sync import load_sync_result
        from lehome_train.hub import HuggingFaceHubTransport

        evidence = load_sync_result(sync_result)
        restored = restore_experiment_snapshot(
            destination,
            sync_result=evidence,
            transport=HuggingFaceHubTransport(timeout_seconds=timeout_seconds),
            staging_root=staging_root,
            max_attempts=max_attempts,
        )
        return {
            "schema_version": 1,
            "status": "experiment_restored",
            "destination": str(restored),
            "repository": evidence.repository,
            "immutable_revision": evidence.immutable_revision,
            "remote_prefix": evidence.manifest.remote_prefix,
        }

    _fail_closed(operation)


def main() -> None:
    """Run the trainer CLI."""

    app()


if __name__ == "__main__":
    main()
