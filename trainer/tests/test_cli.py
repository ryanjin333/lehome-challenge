from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.utils import strip_ansi
from typer.testing import CliRunner

from lehome_train.cli import app
from lehome_train.constants import DEFAULT_SETTINGS


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "LeHome GR00T N1.7 trainer" in result.stdout


@pytest.mark.parametrize(
    "command",
    [
        "data",
        "model",
        "prepare",
        "memorize",
        "smoke",
        "train",
        "continuous-train",
        "validate-training-capability",
        "report",
        "sync",
        "restore",
        "build-runtime-mixture",
        "pilot-runtime-mixture",
        "runtime-gpu-warmup",
        "hydrate-runtime-mixture",
        "publish-runtime-source",
        "verify-uploaded-runtime-source",
        "adopt-uploaded-runtime-source",
        "publish-runtime-mixture",
        "finalize-runtime-mixture",
    ],
)
def test_cli_registers_command_group(command: str) -> None:
    result = CliRunner().invoke(
        app,
        [command, "--help"],
        prog_name="lehome-train",
        color=True,
    )

    assert result.exit_code == 0
    assert f"Usage: lehome-train {command}" in strip_ansi(result.stdout)


def test_runtime_gpu_warmup_cli_dispatches_the_exact_request_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lehome_train.groot import runtime_mixture_warmup

    request = tmp_path / "runtime-warmup.json"
    request.write_text("{}", encoding="utf-8")
    observed: list[Path] = []
    monkeypatch.setattr(
        runtime_mixture_warmup,
        "warmup_from_request",
        lambda path: observed.append(path) or {"selected_loader_workers": 4},
    )

    result = CliRunner().invoke(
        app, ["runtime-gpu-warmup", "--request", str(request)]
    )

    assert result.exit_code == 0
    assert observed == [request]
    assert '"selected_loader_workers":4' in result.stdout


@pytest.mark.parametrize("subcommand", ["inspect", "convert", "validate", "publish", "retrieve"])
def test_cli_registers_operational_data_subcommands(subcommand: str) -> None:
    result = CliRunner().invoke(app, ["data", subcommand, "--help"], color=True)

    assert result.exit_code == 0
    assert f"Usage: root data {subcommand}" in strip_ansi(result.stdout)


def test_data_commands_invoke_real_data_adapters(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []
    injected = {
        "lehome_train.data.inspect": SimpleNamespace(
            inspect_dataset=lambda source, *, output_path=None: calls.append(
                ("inspect", (source, output_path))
            )
            or {"valid": True}
        ),
        "lehome_train.data.convert": SimpleNamespace(
            convert_dataset=lambda source, output, **kwargs: calls.append(
                ("convert", (source, output, kwargs))
            )
            or {"converted": True},
            persistent_destination_operation_lock=lambda _output: nullcontext(),
        ),
        "lehome_train.data.stats": SimpleNamespace(
            write_train_statistics=lambda dataset, *, groot_root=None: calls.append(
                ("stats", (dataset, groot_root))
            )
            or {"statistics": True}
        ),
        "lehome_train.data.validate": SimpleNamespace(
            validate_prepared_dataset=lambda dataset, *, groot_root=None: calls.append(
                ("validate", (dataset, groot_root))
            )
            or {"valid": True}
        ),
        "lehome_train.data.publish": SimpleNamespace(
            publish_prepared_dataset=lambda dataset, **kwargs: calls.append(
                ("publish", (dataset, kwargs))
            )
            or SimpleNamespace(
                repository=kwargs["repository"],
                revision="a" * 40,
                dataset_manifest_sha256="b" * 64,
                entries=(),
            )
        ),
        "lehome_train.hub": SimpleNamespace(
            HuggingFaceHubTransport=lambda **_kwargs: object()
        ),
    }
    previous = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)

    try:
        inspect_result = CliRunner().invoke(
            app,
            ["data", "inspect", "--source", str(tmp_path), "--output", str(tmp_path / "inspection.json")],
        )
        convert_result = CliRunner().invoke(
            app,
            [
                "data",
                "convert",
                "--source",
                str(tmp_path),
                "--output",
                str(tmp_path / "prepared"),
                "--mapping",
                str(tmp_path / "mapping.json"),
                "--source-repository",
                "owner/source",
                "--source-revision",
                "a" * 40,
                "--converter-commit",
                "b" * 40,
                "--container-digest",
                "sha256:" + "c" * 64,
                "--groot-root",
                str(tmp_path / "groot"),
                "--persistent-staging-root",
                str(tmp_path / "persistent-stage"),
                "--unbound-staging-data-adoption-root",
                str(tmp_path / "orphan"),
            ],
        )
        validate_result = CliRunner().invoke(
            app,
            ["data", "validate", "--dataset", str(tmp_path), "--groot-root", str(tmp_path / "groot")],
        )
        publish_result = CliRunner().invoke(
            app,
            [
                "data",
                "publish",
                "--dataset",
                str(tmp_path),
                "--repo",
                DEFAULT_SETTINGS.data_repo,
                "--revision",
                "prepared-v1",
                "--staging-root",
                str(tmp_path),
            ],
        )
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    assert all(
        result.exit_code == 0
        for result in (inspect_result, convert_result, validate_result, publish_result)
    )
    assert [name for name, _ in calls] == [
        "inspect",
        "convert",
        "stats",
        "validate",
        "publish",
    ]
    assert calls[1][1][2]["persistent_staging_root"] == tmp_path / "persistent-stage"
    assert calls[1][1][2]["unbound_staging_data_adoption_root"] == tmp_path / "orphan"


def test_persistent_data_convert_holds_destination_operation_lock_through_statistics(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    output = tmp_path / "prepared"

    @contextmanager
    def operation_lock(destination: Path):
        lock = destination.parent / f".{destination.name}.data-convert-operation.lock"
        try:
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise ValueError("persistent conversion destination is already owned") from None
        os.close(descriptor)
        calls.append("locked")
        try:
            yield
        finally:
            lock.unlink(missing_ok=True)
            calls.append("released")

    fail_statistics = False

    def write_statistics(*_args, **_kwargs):
        calls.append("stats")
        if fail_statistics:
            raise RuntimeError("statistics failed")
        return {"statistics": True}

    injected = {
        "lehome_train.data.convert": SimpleNamespace(
            convert_dataset=lambda *_args, **_kwargs: calls.append("convert") or {"converted": True},
            persistent_destination_operation_lock=operation_lock,
        ),
        "lehome_train.data.stats": SimpleNamespace(
            write_train_statistics=write_statistics
        ),
    }
    previous = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)
    try:
        with operation_lock(output):
            second = CliRunner().invoke(
                app,
                [
                    "data", "convert", "--source", str(tmp_path), "--output", str(output),
                    "--mapping", str(tmp_path / "mapping.json"), "--source-repository", "owner/source",
                    "--source-revision", "a" * 40, "--converter-commit", "b" * 40,
                    "--container-digest", "sha256:" + "c" * 64, "--groot-root", str(tmp_path / "groot"),
                    "--persistent-staging-root", str(tmp_path / "stage"),
                ],
            )
        assert second.exit_code == 1
        assert "already owned" in second.output
        assert calls == ["locked", "released"]

        fail_statistics = True
        statistics_failure = CliRunner().invoke(
            app,
            [
                "data", "convert", "--source", str(tmp_path), "--output", str(output),
                "--mapping", str(tmp_path / "mapping.json"), "--source-repository", "owner/source",
                "--source-revision", "a" * 40, "--converter-commit", "b" * 40,
                "--container-digest", "sha256:" + "c" * 64, "--groot-root", str(tmp_path / "groot"),
                "--persistent-staging-root", str(tmp_path / "stage"),
            ],
        )
        assert statistics_failure.exit_code == 1
        assert "statistics failed" in statistics_failure.output
        assert not (output.parent / ".prepared.data-convert-operation.lock").exists()
        fail_statistics = False
        completed = CliRunner().invoke(
            app,
            [
                "data", "convert", "--source", str(tmp_path), "--output", str(output),
                "--mapping", str(tmp_path / "mapping.json"), "--source-repository", "owner/source",
                "--source-revision", "a" * 40, "--converter-commit", "b" * 40,
                "--container-digest", "sha256:" + "c" * 64, "--groot-root", str(tmp_path / "groot"),
                "--persistent-staging-root", str(tmp_path / "stage"),
            ],
        )
        assert completed.exit_code == 0
        assert calls == [
            "locked", "released",  # competing CLI operation
            "locked", "convert", "stats", "released",  # stats failure releases
            "locked", "convert", "stats", "released",  # corrected retry owns anew
        ]
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_image_native_retrieve_and_restore_dispatch_checked_transports(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []
    injected = {
        "lehome_train.data.publish": SimpleNamespace(
            download_prepared_dataset=lambda destination, **kwargs: calls.append(
                ("retrieve", (destination, kwargs))
            )
            or destination,
            write_prepared_snapshot_manifest=lambda dataset, destination, **kwargs: calls.append(
                ("dataset-evidence", (dataset, destination, kwargs))
            )
            or destination,
        ),
        "lehome_train.commands.restore": SimpleNamespace(
            restore_experiment_snapshot=lambda destination, **kwargs: calls.append(
                ("restore", (destination, kwargs))
            )
            or destination
        ),
        "lehome_train.commands.sync": SimpleNamespace(
            load_sync_result=lambda path: SimpleNamespace(
                repository=DEFAULT_SETTINGS.model_repo,
                immutable_revision="c" * 40,
                manifest=SimpleNamespace(remote_prefix="experiments/example/" + "d" * 64),
                source=path,
            )
        ),
        "lehome_train.hub": SimpleNamespace(
            HuggingFaceHubTransport=lambda **kwargs: ("transport", kwargs)
        ),
    }
    previous = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)
    try:
        retrieve = CliRunner().invoke(
            app,
            [
                "data",
                "retrieve",
                "--destination",
                str(tmp_path / "dataset"),
                "--repo",
                DEFAULT_SETTINGS.data_repo,
                "--revision",
                "a" * 40,
                "--manifest-sha256",
                "b" * 64,
            ],
        )
        restore = CliRunner().invoke(
            app,
            [
                "restore",
                "--sync-result",
                str(tmp_path / "sync-result.json"),
                "--destination",
                str(tmp_path / "experiment"),
                "--staging-root",
                str(tmp_path),
            ],
        )
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    assert retrieve.exit_code == 0
    assert restore.exit_code == 0
    assert [name for name, _value in calls] == [
        "retrieve",
        "dataset-evidence",
        "restore",
    ]
    retrieve_kwargs = calls[0][1][1]
    assert retrieve_kwargs["revision"] == "a" * 40
    assert retrieve_kwargs["expected_manifest_sha256"] == "b" * 64
    dataset_evidence = calls[1][1]
    assert dataset_evidence[1] == tmp_path / "dataset" / "lehome_dataset_snapshot.json"


def test_image_native_model_retrieve_is_pinned_beneath_cache() -> None:
    calls: list[tuple[object, dict[str, object]]] = []
    injected = {
        "lehome_train.groot.model_snapshot": SimpleNamespace(
            HuggingFaceModelSnapshotTransport=lambda **kwargs: (
                "transport",
                kwargs,
            ),
            download_base_model=lambda destination, **kwargs: calls.append(
                (destination, kwargs)
            )
            or destination,
        )
    }
    previous = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)
    try:
        result = CliRunner().invoke(
            app,
            [
                "model",
                "retrieve",
                "--destination",
                "/cache/models/groot-n17",
                "--revision",
                DEFAULT_SETTINGS.model_revision,
                "--staging-root",
                "/cache/staging",
            ],
        )
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    assert result.exit_code == 0
    assert calls[0][0] == Path("/cache/models/groot-n17")
    assert calls[0][1]["revision"] == DEFAULT_SETTINGS.model_revision


@pytest.mark.parametrize("command", ["prepare", "memorize", "smoke", "train", "continuous-train", "runtime-mixture-train"])
def test_gpu_commands_dispatch_checked_request_to_runtime_factory(
    command: str,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class Runtime:
        def prepare(self, request: dict[str, object]) -> dict[str, object]:
            calls.append(("prepare", request))
            return {"status": "prepared"}

        def memorize(self, request: dict[str, object]) -> dict[str, object]:
            calls.append(("memorize", request))
            return {"status": "memorized"}

        def smoke(self, request: dict[str, object]) -> dict[str, object]:
            calls.append(("smoke", request))
            return {"status": "smoked"}

        def tune(self, request: dict[str, object]) -> dict[str, object]:
            calls.append(("tune", request))
            return {"status": "tuned"}

        def train(self, request: dict[str, object]) -> dict[str, object]:
            calls.append(("train", request))
            return {"status": "trained"}

        def continuous_train(self, request: dict[str, object]) -> dict[str, object]:
            calls.append(("continuous-train", request))
            return {"status": "continuous-trained"}

        def runtime_mixture_train(self, request: dict[str, object]) -> dict[str, object]:
            calls.append(("runtime-mixture-train", request))
            return {"status": "runtime-mixture-trained"}

    sys.modules["test_lehome_runtime"] = SimpleNamespace(create=lambda: Runtime())
    request = tmp_path / f"{command}.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "command": command,
                "arguments": {"experiment_root": "/output/experiment"},
            }
        ),
        encoding="utf-8",
    )
    try:
        result = CliRunner().invoke(
            app,
            [
                command,
                "--request",
                str(request),
                "--runtime-factory",
                "test_lehome_runtime:create",
            ],
        )
    finally:
        del sys.modules["test_lehome_runtime"]

    assert result.exit_code == 0
    assert calls == [(command, {"experiment_root": "/output/experiment"})]
    assert '"status":' in result.stdout


def test_gpu_command_fails_closed_without_runtime_factory(tmp_path: Path) -> None:
    request = tmp_path / "prepare.json"
    request.write_text(
        json.dumps({"schema_version": 1, "command": "prepare", "arguments": {}}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["prepare", "--request", str(request)])

    assert result.exit_code != 0
    assert "runtime factory" in result.output


def test_report_and_sync_commands_reach_default_request_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lehome_train.runtime as runtime_module

    report_request = tmp_path / "report.json"
    sync_request = tmp_path / "sync.json"
    report_request.write_text("{}", encoding="utf-8")
    sync_request.write_text("{}", encoding="utf-8")
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        runtime_module,
        "execute_report_request",
        lambda path: calls.append(("report", Path(path))) or {"status": "reported"},
    )
    monkeypatch.setattr(
        runtime_module,
        "execute_sync_request",
        lambda path: calls.append(("sync", Path(path))) or {"status": "synced"},
    )

    report_result = CliRunner().invoke(app, ["report", "--request", str(report_request)])
    sync_result = CliRunner().invoke(app, ["sync", "--request", str(sync_request)])

    assert report_result.exit_code == sync_result.exit_code == 0
    assert calls == [("report", report_request), ("sync", sync_request)]


def test_report_request_wires_local_sync_and_pruning_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lehome_train.runtime as runtime_module

    request = tmp_path / "report-request.json"
    arguments = {
        "experiment_config": str(tmp_path / "config.json"),
        "isaac_groot_revision": "a" * 40,
        "smoke_result": str(tmp_path / "smoke.json"),
        "checkpoint_descriptors": [str(tmp_path / "checkpoint.json")],
        "local_artifact_root": str(tmp_path / "experiment"),
        "sync_result": str(tmp_path / "sync-result.json"),
        "pruning_receipts": [str(tmp_path / "pruning-receipt.json")],
        "instance_started_at": "2026-07-31T10:00:00Z",
        "generated_at": "2026-07-31T11:00:00Z",
        "provider_hourly_price": 1.0,
        "output": str(tmp_path / "report.json"),
    }
    request.write_text(
        json.dumps({"schema_version": 1, "command": "report", "arguments": arguments}),
        encoding="utf-8",
    )
    calls: list[tuple[str, object]] = []
    report = SimpleNamespace(to_dict=lambda: {"status": "reported"})
    fake_report_module = SimpleNamespace(
        build_training_report=lambda **kwargs: calls.append(("build", kwargs)) or report,
        write_training_report=lambda path, value: calls.append(
            ("write", (path, value))
        ),
    )
    fake_report_evidence_module = SimpleNamespace(
        load_checkpoint_pruning_receipt=lambda path: f"receipt:{path}"
    )
    fake_sync_module = SimpleNamespace(
        load_sync_result=lambda path: f"sync:{path}"
    )
    monkeypatch.setitem(sys.modules, "lehome_train.commands.report", fake_report_module)
    monkeypatch.setitem(
        sys.modules,
        "lehome_train.report_evidence",
        fake_report_evidence_module,
    )
    monkeypatch.setitem(sys.modules, "lehome_train.commands.sync", fake_sync_module)
    monkeypatch.setattr(
        runtime_module,
        "load_json",
        lambda model, path: f"{model.__name__}:{path}",
    )
    monkeypatch.setattr(
        runtime_module,
        "load_checkpoint_descriptor",
        lambda path: f"checkpoint:{path}",
    )

    result = runtime_module.execute_report_request(request)

    assert result == {"status": "reported"}
    built = calls[0][1]
    assert built["local_artifact_root"] == arguments["local_artifact_root"]
    assert built["sync_evidence"] == f"sync:{arguments['sync_result']}"
    assert built["pruning_receipts"] == (
        f"receipt:{arguments['pruning_receipts'][0]}",
    )
    assert calls[1] == ("write", (arguments["output"], report))


def test_sync_request_persists_result_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lehome_train.runtime as runtime_module

    request = tmp_path / "sync-request.json"
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    arguments = {
        "experiment_root": str(experiment),
        "experiment_id": "experiment-001",
        "experiment_config_sha256": "a" * 64,
        "repository": DEFAULT_SETTINGS.model_repo,
        "revision": "experiment-001",
        "staging_root": str(tmp_path / "staging"),
        "timeout_seconds": 30,
        "max_attempts": 5,
        "output": str(tmp_path / "sync-result.json"),
    }
    request.write_text(
        json.dumps({"schema_version": 1, "command": "sync", "arguments": arguments}),
        encoding="utf-8",
    )
    calls: list[tuple[str, object]] = []
    result = SimpleNamespace(to_dict=lambda: {"status": "synced"})
    fake_sync_module = SimpleNamespace(
        sync_experiment=lambda *args, **kwargs: calls.append(
            ("sync", (args, kwargs))
        )
        or result,
        write_sync_result=lambda path, value: calls.append(
            ("write", (path, value))
        ),
    )
    fake_hub_module = SimpleNamespace(
        HuggingFaceHubTransport=lambda **kwargs: ("transport", kwargs)
    )
    monkeypatch.setitem(sys.modules, "lehome_train.commands.sync", fake_sync_module)
    monkeypatch.setitem(sys.modules, "lehome_train.hub", fake_hub_module)

    observed = runtime_module.execute_sync_request(request)

    assert observed == {"status": "synced"}
    assert calls[-1] == ("write", (arguments["output"], result))


@pytest.mark.parametrize("output_kind", ["same", "nested", "symlink_escape"])
def test_sync_request_rejects_output_inside_experiment_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_kind: str,
) -> None:
    import lehome_train.runtime as runtime_module

    experiment = tmp_path / "experiment"
    experiment.mkdir()
    if output_kind == "same":
        output = experiment
    elif output_kind == "nested":
        output = experiment / "reports" / "sync-result.json"
    else:
        external = tmp_path / "external"
        external.mkdir()
        link = experiment / "escape"
        link.symlink_to(external, target_is_directory=True)
        output = link / "sync-result.json"
    request = tmp_path / f"sync-{output_kind}.json"
    arguments = {
        "experiment_root": str(experiment),
        "experiment_id": "experiment-001",
        "experiment_config_sha256": "a" * 64,
        "repository": DEFAULT_SETTINGS.model_repo,
        "revision": "experiment-001",
        "staging_root": str(tmp_path / "staging"),
        "timeout_seconds": 30,
        "max_attempts": 5,
        "output": str(output),
    }
    request.write_text(
        json.dumps({"schema_version": 1, "command": "sync", "arguments": arguments}),
        encoding="utf-8",
    )
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "lehome_train.commands.sync",
        SimpleNamespace(
            sync_experiment=lambda *_args, **_kwargs: calls.append("sync"),
            write_sync_result=lambda *_args: calls.append("write"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "lehome_train.hub",
        SimpleNamespace(
            HuggingFaceHubTransport=lambda **_kwargs: calls.append("transport")
        ),
    )

    with pytest.raises(ValueError, match="outside experiment root"):
        runtime_module.execute_sync_request(request)

    assert calls == []


def test_settings_use_exact_immutable_pins() -> None:
    settings = DEFAULT_SETTINGS

    assert settings.python_version == "3.10.18"
    assert (
        settings.cuda_base_digest
        == "sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719"
    )
    assert settings.isaac_groot_revision == "23ace64f17aa5015259b8609d371eb61a357c776"
    assert settings.model_revision == "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
    assert settings.data_repo == "ryanjin333/lehome-groot-n17-data"
    assert settings.model_repo == "ryanjin333/lehome-groot-n17-models"

    with pytest.raises(AttributeError):
        settings.model_revision = "explicit-override"
