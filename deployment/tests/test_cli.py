from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

import b1k_deploy.cli as cli
from b1k_deploy.cli import main
from b1k_deploy.dockerhub import DockerImageRelease
from b1k_deploy.production_smoke import ProductionSmokeError
from b1k_deploy.smoke import SmokeTemplatePublicationReceipt


def test_smoke_campaign_is_dry_run_by_default_and_does_not_load_provider_settings(tmp_path: Path, capsys) -> None:
    result = main([
        "smoke-campaign",
        "--training-image", "docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64,
        "--rollout-image", "docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64,
        "--training-template-id", "123", "--rollout-template-id", "456",
        "--training-template-payload-sha256", "c" * 64, "--rollout-template-payload-sha256", "d" * 64,
        "--ledger", str(tmp_path / "ledger.jsonl"), "--receipt", str(tmp_path / "receipt.json"), "--known-hosts", str(tmp_path / "known_hosts"),
    ])

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    assert output["provider_calls"] == 0
    assert not (tmp_path / "ledger.jsonl").exists()
    assert not (tmp_path / "receipt.json").exists()


def _execute_smoke_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        training_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64,
        rollout_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64,
        training_template_id="123", rollout_template_id="456",
        training_template_payload_sha256="c" * 64, rollout_template_payload_sha256="d" * 64,
        ledger=tmp_path / "ledger.jsonl", receipt=tmp_path / "receipt.json",
        known_hosts=tmp_path / "known_hosts", ssh_identity=tmp_path / "identity",
        publication_receipt=tmp_path / "publication.json", execute=True,
    )


def _stub_execute_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, create_error: Exception | None = None,
    runtime_error: Exception | None = None, cleanup_error: Exception | None = None,
) -> list[dict[str, object]]:
    import b1k_deploy.dockerhub as dockerhub
    import b1k_deploy.huggingface as huggingface
    import b1k_deploy.ledger as ledger
    import b1k_deploy.production as production
    import b1k_deploy.production_smoke as production_smoke
    import b1k_deploy.smoke as smoke
    import b1k_deploy.vast as vast

    writes: list[dict[str, object]] = []
    monkeypatch.setenv("B1K_VAST_PRIVATE_PULL_READY", "verified")
    monkeypatch.setenv("B1K_CHECKPOINT_BUCKET_HELPER", "/tmp/b1k-helper")
    checkpoint_probe_root = tmp_path / "checkpoint-probe"
    checkpoint_probe_root.mkdir(mode=0o700)
    monkeypatch.setenv("B1K_CHECKPOINT_PROBE_ROOT", str(checkpoint_probe_root))
    training = DockerImageRelease("training", "docker.io/ryanjin333/behavior1k-groot-n17", "trainer-" + "a" * 40, "a" * 40, "sha256:" + "a" * 64, "docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64)
    rollout = DockerImageRelease("rollout", "docker.io/ryanjin333/behavior1k-groot-n17", "rollout-" + "a" * 40, "a" * 40, "sha256:" + "b" * 64, "docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64)
    monkeypatch.setattr(cli, "AtomicCampaignReceiptStore", lambda _path: SimpleNamespace(read=lambda: SimpleNamespace(status="complete", images=(training, rollout), templates=(1, 2))))
    monkeypatch.setattr(cli, "_verify_smoke_bindings_against_publication", lambda *_args: None)
    monkeypatch.setattr(cli, "_write_receipt", lambda _path, payload: writes.append(dict(payload)))
    monkeypatch.setattr(
        production.ConfiguredPublicationSettings,
        "from_environment",
        lambda: SimpleNamespace(
            vastai_executable=tmp_path / "vastai",
            vast_api_key_file=tmp_path / "vast-key",
            docker_username="ryanjin333",
            docker_token_file=tmp_path / "docker-token",
        ),
    )
    monkeypatch.setattr(production, "_required_private_file", lambda *_args: tmp_path / "token")
    monkeypatch.setattr(dockerhub, "TokenSource", SimpleNamespace(from_token_file=lambda _path: object()))

    class Hub:
        def __init__(self, *_args: object) -> None: pass
        def verify_private_repositories(self, *_args: object) -> None: pass
        def bootstrap_checkpoint_bucket_probe(self, *_args: object, **kwargs: object) -> None:
            assert kwargs["files"]._root == checkpoint_probe_root

    class Vast:
        def new_ephemeral_smoke_template_name(self, role: str) -> str:
            return f"behavior1k-groot-n17-{'trainer' if role == 'training' else 'rollout'}-smoke-" + "e" * 32

        def create_ephemeral_smoke_template(self, _role: str, production_template: SmokeTemplatePublicationReceipt, *, name: str) -> SmokeTemplatePublicationReceipt:
            if create_error is not None:
                raise create_error
            return SmokeTemplatePublicationReceipt("999", production_template.image_release, "f" * 64)

        def select_offer(self, _purpose: str) -> object:
            return object()

        def destroy_ephemeral_smoke_template(self, _template: SmokeTemplatePublicationReceipt) -> None:
            if cleanup_error is not None:
                raise cleanup_error

    class Controller:
        def __init__(self, *_args: object) -> None: pass
        def run(self, *_args: object, **_kwargs: object) -> object:
            if runtime_error is not None:
                raise runtime_error
            raise AssertionError("test requires a typed runtime failure")

    monkeypatch.setattr(huggingface, "HuggingFaceReleaseVerifier", Hub)
    monkeypatch.setattr(huggingface, "HuggingFaceHubClient", lambda: object())
    monkeypatch.setattr(huggingface, "CheckpointBucketHelperClient", lambda *_args: object())
    monkeypatch.setattr(production_smoke, "VastCliSmokeClient", lambda **_kwargs: Vast())
    monkeypatch.setattr(production_smoke, "SshSmokeRemote", lambda **_kwargs: object())
    monkeypatch.setattr(ledger, "RentalLedger", lambda _path: object())
    monkeypatch.setattr(vast, "VastAdapter", lambda _vast: object())
    monkeypatch.setattr(vast, "CappedVastController", lambda *_args: object())
    monkeypatch.setattr(smoke, "SmokeController", Controller)
    monkeypatch.setattr(smoke, "SmokePlan", lambda **_kwargs: object())
    monkeypatch.setattr(smoke, "SmokeReadinessReceipt", lambda *_args: object())
    return writes


def test_execute_smoke_campaign_preserves_typed_template_create_failure_after_receipt_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _stub_execute_dependencies(tmp_path, monkeypatch, create_error=ProductionSmokeError("typed create failed"))

    with pytest.raises(ProductionSmokeError, match="typed create failed"):
        cli._smoke_campaign(_execute_smoke_args(tmp_path))

    assert len(writes) >= 2
    assert writes[-1]["ephemeral_templates"]["training"]["status"] == "intent-recorded"


def test_execute_smoke_campaign_preserves_typed_runtime_failure_and_chains_template_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _stub_execute_dependencies(
        tmp_path, monkeypatch,
        runtime_error=ProductionSmokeError("typed runtime failed"),
        cleanup_error=ProductionSmokeError("template cleanup failed"),
    )

    with pytest.raises(ProductionSmokeError, match="typed runtime failed") as error:
        cli._smoke_campaign(_execute_smoke_args(tmp_path))

    assert isinstance(error.value.__cause__, ProductionSmokeError)
    assert str(error.value.__cause__) == "template cleanup failed"
    assert writes[-1]["template_cleanup_failure"] is True
