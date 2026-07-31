"""CPU-safe command-line entry point for portable training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import typer

from lehome_train.io import canonical_json_bytes


app = typer.Typer(help="LeHome GR00T N1.7 trainer.", no_args_is_help=True)
data_app = typer.Typer(help="Dataset inspection and transfer commands.", no_args_is_help=True)
app.add_typer(data_app, name="data")


def _emit(value: object) -> None:
    typer.echo(canonical_json_bytes(value).decode("utf-8"))


def _fail_closed(operation) -> None:
    try:
        _emit(operation())
    except Exception as error:
        raise click.ClickException(str(error)) from None


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
) -> None:
    """Convert deterministically and compute pinned train-only statistics."""

    def operation() -> object:
        from lehome_train.data.convert import convert_dataset
        from lehome_train.data.stats import write_train_statistics

        manifest = convert_dataset(
            source,
            output,
            mapping_path=mapping,
            source_repository=source_repository,
            source_revision=source_revision,
            converter_commit=converter_commit,
            converter_container_digest=container_digest,
        )
        statistics = write_train_statistics(output, groot_root=groot_root)
        return {"manifest": manifest, "statistics": statistics}

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


def _gpu_command(command: str, request: Path, runtime_factory: Optional[str]) -> None:
    def operation() -> object:
        from lehome_train.runtime import dispatch_runtime_request

        return dispatch_runtime_request(
            command,
            request,
            factory_spec=runtime_factory,
        )

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


@app.command("train")
def train(
    request: Path = typer.Option(..., "--request"),
    runtime_factory: Optional[str] = typer.Option(None, "--runtime-factory"),
) -> None:
    """Run fixed-exposure training through the runtime adapter."""

    _gpu_command("train", request, runtime_factory)


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


def main() -> None:
    """Run the trainer CLI."""

    app()


if __name__ == "__main__":
    main()
