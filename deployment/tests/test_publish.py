from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path

import pytest

from b1k_deploy.cli import main
from b1k_deploy.dockerhub import CommandResult, DockerImageRelease
from b1k_rollout.template import ROLLOUT_ONSTART, render_vast_template
from b1k_deploy.publish import (
    AtomicCampaignReceiptStore,
    CampaignPublicationReceipt,
    CampaignPublisher,
    ImagePublicationPlan,
    PublicationAdapters,
    PublicationError,
    PublicationPartialError,
    TemplatePublicationPlan,
    TemplateSchemaPlan,
    TemplatePublisher,
    campaign_final_plan_hash,
    campaign_preplan_hash,
    canonical_payload_hash,
    load_canonical_template,
    load_publication_adapters,
    publish_images,
)

WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "a" * 40
TRAINING_TAG = f"trainer-{SOURCE_COMMIT}"
ROLLOUT_TAG = f"rollout-{SOURCE_COMMIT}"

def release(purpose: str, source_commit: str = "a" * 40) -> DockerImageRelease:
    repository = "docker.io/ryanjin333/behavior1k-groot-n17"
    digest = "sha256:" + ("a" if purpose == "training" else "b") * 64
    tag = f"trainer-{source_commit}" if purpose == "training" else f"rollout-{source_commit}"
    return DockerImageRelease(purpose, repository, tag, source_commit, digest, f"{repository}@{digest}")


def payload(purpose="training", *, model_commit="c" * 40, checkpoint_artifact_sha256="d" * 64, gpu_ids=(0,)) -> dict[str, object]:
    if purpose == "rollout":
        return json.loads(
            render_vast_template(
                image_digest="sha256:" + "0" * 64,
                model_commit=model_commit,
                checkpoint_artifact_sha256=checkpoint_artifact_sha256,
                gpu_ids=gpu_ids,
            )
        )
    return load_canonical_template("training", source_root=WORKSPACE)


def rollout_cli_arguments() -> list[str]:
    return ["--model-commit", "c" * 40, "--checkpoint-artifact-sha256", "d" * 64, "--gpu-ids", "0", "--source-root", str(WORKSPACE)]


class FakeBuilder:
    def __init__(self):
        self.calls = []

    def build_and_push(self, repository, tag, source_commit):
        self.calls.append((repository, tag, source_commit))


class FakeVerifier:
    def __init__(self):
        self.calls = []

    def verify_private_image(self, repository, tag):
        self.calls.append((repository, tag))
        return release("training" if tag.startswith("trainer-") else "rollout", tag.split("-", 1)[1])


class PassingReleaseContext:
    def __init__(self):
        self.calls = []

    def verify(self, workspace, source_commit):
        self.calls.append((workspace, source_commit))


class FakeTemplates:
    def __init__(self):
        self.created = []
        self.templates = {}

    def find_private_template(self, name, image_reference):
        for template_id, item in self.templates.items():
            if item["name"] == name and item["image"] == image_reference:
                return template_id
        return None

    def create_private_template(self, template):
        template_id = str(100 + len(self.templates))
        self.created.append(template)
        self.templates[template_id] = template
        return template_id

    def get_template(self, template_id):
        return self.templates[template_id]


class FailOnVerifierCall(FakeVerifier):
    def __init__(self, call_number):
        super().__init__()
        self.call_number = call_number

    def verify_private_image(self, repository, tag):
        self.calls.append((repository, tag))
        if len(self.calls) == self.call_number:
            raise RuntimeError("simulated registry readback failure")
        return release("training" if tag.startswith("trainer-") else "rollout", tag.split("-", 1)[1])


class MalformedVerifierOnCall(FakeVerifier):
    def __init__(self, call_number):
        super().__init__()
        self.call_number = call_number

    def verify_private_image(self, repository, tag):
        self.calls.append((repository, tag))
        if len(self.calls) == self.call_number:
            return object()
        return release("training" if tag.startswith("trainer-") else "rollout", tag.split("-", 1)[1])


class FailTemplateCreate(FakeTemplates):
    def __init__(self):
        super().__init__()
        self.fail = True

    def create_private_template(self, template):
        if self.fail:
            self.fail = False
            raise RuntimeError("simulated template create failure")
        return super().create_private_template(template)


class FailSecondTemplateCreate(FakeTemplates):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def create_private_template(self, template):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated second template create failure")
        return super().create_private_template(template)


class FailTemplateReadbackOnce(FakeTemplates):
    def __init__(self):
        super().__init__()
        self.fail = True

    def get_template(self, template_id):
        if self.fail:
            self.fail = False
            raise RuntimeError("simulated template readback failure")
        return super().get_template(template_id)


def campaign_plans():
    images = ImagePublicationPlan("a" * 40, "trainer-" + "a" * 40, "rollout-" + "a" * 40)
    templates = (
        TemplateSchemaPlan(images.source_commit, "training", payload("training")),
        TemplateSchemaPlan(images.source_commit, "rollout", payload("rollout")),
    )
    return images, templates


def campaign_publisher(tmp_path: Path, builder=None, verifier=None, templates=None):
    adapters = PublicationAdapters(builder or FakeBuilder(), verifier or FakeVerifier(), templates or FakeTemplates())
    return CampaignPublisher(adapters, AtomicCampaignReceiptStore(tmp_path / "receipt.json")), adapters


class RecordingReceiptStore(AtomicCampaignReceiptStore):
    def __init__(self, path):
        super().__init__(path)
        self.writes = []

    def write_locked(self, receipt):
        self.writes.append(receipt)
        super().write_locked(receipt)


class FailFinalReceiptStore(AtomicCampaignReceiptStore):
    def write_locked(self, receipt):
        if receipt.status == "complete":
            raise OSError("simulated final receipt write failure")
        super().write_locked(receipt)


def test_dry_run_image_publication_is_plan_only_and_never_calls_builder_or_registry():
    builder, verifier = FakeBuilder(), FakeVerifier()
    receipt = publish_images(ImagePublicationPlan(SOURCE_COMMIT, TRAINING_TAG, ROLLOUT_TAG), builder, verifier, execute=False)

    assert receipt.dry_run is True
    assert builder.calls == []
    assert verifier.calls == []
    assert {item.repository for item in receipt.images} == {"docker.io/ryanjin333/behavior1k-groot-n17"}


def test_shared_repository_image_releases_remain_distinct_by_purpose_and_canonical_tag():
    source_commit = "a" * 40
    plan = ImagePublicationPlan(source_commit, f"trainer-{source_commit}", f"rollout-{source_commit}")
    builder, verifier = FakeBuilder(), FakeVerifier()

    receipt = publish_images(plan, builder, verifier, execute=True)

    assert [item.repository for item in receipt.images] == [
        "docker.io/ryanjin333/behavior1k-groot-n17",
        "docker.io/ryanjin333/behavior1k-groot-n17",
    ]
    assert [item.purpose for item in receipt.images] == ["training", "rollout"]
    assert [item.tag for item in receipt.images] == [plan.training_tag, plan.rollout_tag]
    assert builder.calls == [
        ("docker.io/ryanjin333/behavior1k-groot-n17", plan.training_tag, source_commit),
        ("docker.io/ryanjin333/behavior1k-groot-n17", plan.rollout_tag, source_commit),
    ]


def test_execute_builds_exact_repositories_then_requires_private_digest_readback():
    builder, verifier = FakeBuilder(), FakeVerifier()
    receipt = publish_images(ImagePublicationPlan(SOURCE_COMMIT, TRAINING_TAG, ROLLOUT_TAG), builder, verifier, execute=True)

    assert receipt.dry_run is False
    assert builder.calls == [
        ("docker.io/ryanjin333/behavior1k-groot-n17", TRAINING_TAG, SOURCE_COMMIT),
        ("docker.io/ryanjin333/behavior1k-groot-n17", ROLLOUT_TAG, SOURCE_COMMIT),
    ]
    assert verifier.calls == [
        ("docker.io/ryanjin333/behavior1k-groot-n17", TRAINING_TAG),
        ("docker.io/ryanjin333/behavior1k-groot-n17", ROLLOUT_TAG),
    ]
    assert all(item.reference.endswith(item.digest) for item in receipt.images)


def test_template_publication_substitutes_only_digest_reference_is_private_and_is_idempotent():
    client = FakeTemplates()
    publisher = TemplatePublisher(client)
    plan = TemplatePublicationPlan("a" * 40, "training", release("training"), payload())

    first = publisher.publish(plan, execute=True)
    second = publisher.publish(plan, execute=True)

    assert first.template_id == second.template_id
    assert len(client.created) == 1
    created = client.created[0]
    assert created["image"] == release("training").reference
    assert created["private"] is True
    assert created["name"] == first.name
    assert first.payload_hash == canonical_payload_hash(created)


def test_template_publication_rejects_unrelated_image_public_templates_and_secret_shaped_payloads():
    client = FakeTemplates()
    publisher = TemplatePublisher(client)

    with pytest.raises(PublicationError, match="placeholder"):
        publisher.publish(TemplatePublicationPlan("a" * 40, "training", release("training"), {**payload(), "image": "docker.io/elsewhere@sha256:" + "c" * 64}), execute=True)
    with pytest.raises(PublicationError, match="private"):
        publisher.publish(TemplatePublicationPlan("a" * 40, "training", release("training"), {**payload(), "private": False}), execute=True)
    with pytest.raises(PublicationError, match="secret"):
        publisher.publish(TemplatePublicationPlan("a" * 40, "training", release("training"), {**payload(), "env": "-e TOKEN=not-allowed"}), execute=True)


def test_receipt_is_secret_free_canonical_json():
    client = FakeTemplates()
    receipt = TemplatePublisher(client).publish(TemplatePublicationPlan("a" * 40, "rollout", release("rollout"), payload("rollout")), execute=True)

    encoded = json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":"))
    assert "TOKEN" not in encoded
    assert receipt.image_reference == release("rollout").reference


@pytest.mark.parametrize("purpose", ["training", "rollout"])
def test_actual_canonical_template_contract_renders_image_and_container_digest_together(purpose):
    client = FakeTemplates()
    rendered = TemplatePublisher(client).publish(TemplatePublicationPlan("a" * 40, purpose, release(purpose), payload(purpose)), execute=True)

    template = client.templates[rendered.template_id]
    assert template["image"] == release(purpose).reference
    assert f"CONTAINER_DIGEST={release(purpose).digest}" in template["env"]
    assert "B1K_HF_TOKEN_FILE=/workspace/.cache/huggingface/token" in template["env"]


def test_structured_secret_validation_allows_token_file_paths_but_rejects_actual_credential_values():
    client = FakeTemplates()
    publisher = TemplatePublisher(client)
    allowed = payload()
    publisher.publish(TemplatePublicationPlan("a" * 40, "training", release("training"), allowed), execute=True)

    with pytest.raises(PublicationError, match="credential"):
        publisher.publish(TemplatePublicationPlan("a" * 40, "training", release("training"), {**payload(), "metadata": "Bearer simulated-credential"}), execute=True)


@pytest.mark.parametrize(
    "malicious",
    [
        {**payload(), "password": 123456},
        {**payload(), "env": payload()["env"] + " --env PASSWORD=hunter2"},
        {**payload(), "env": payload()["env"] + " --env=PASSWORD=hunter2"},
        {**payload(), "env": payload()["env"] + " -e PASSWORD=hunter2"},
        {**payload(), "env": payload()["env"] + " -e=PASSWORD=hunter2"},
    ],
)
def test_structured_secret_validation_rejects_secret_keys_for_all_value_types_and_docker_env_forms(malicious):
    with pytest.raises(PublicationError, match="secret"):
        TemplatePublisher(FakeTemplates()).publish(TemplatePublicationPlan("a" * 40, "training", release("training"), malicious), execute=True)


def test_loads_only_an_injected_trusted_adapter_name_without_constructing_live_clients():
    builder, verifier, templates = FakeBuilder(), FakeVerifier(), FakeTemplates()

    adapters = load_publication_adapters("test", workspace=WORKSPACE, factories={"test": lambda: PublicationAdapters(builder, verifier, templates)})
    receipt = publish_images(ImagePublicationPlan(SOURCE_COMMIT, TRAINING_TAG, ROLLOUT_TAG), adapters.builder, adapters.verifier, execute=True)

    assert receipt.dry_run is False
    assert len(builder.calls) == 2


def test_wheel_installed_cli_plans_templates_from_the_explicit_source_root(tmp_path: Path) -> None:
    """Non-editable installs must not infer a repository from module parents."""
    rollout_dist = tmp_path / "rollout-dist"
    deployment_dist = tmp_path / "deployment-dist"
    virtualenv = tmp_path / "venv"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    def run(arguments: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(arguments, cwd=cwd, env=environment, text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        return result

    run(["uv", "build", "--wheel", "--out-dir", str(rollout_dist)], cwd=WORKSPACE / "rollout")
    run(["uv", "build", "--wheel", "--out-dir", str(deployment_dist)], cwd=WORKSPACE / "deployment")
    run(["uv", "venv", str(virtualenv)], cwd=tmp_path)
    python = virtualenv / "bin" / "python"
    run(["uv", "pip", "install", "--python", str(python), "--no-deps", *map(str, rollout_dist.glob("*.whl")), *map(str, deployment_dist.glob("*.whl"))], cwd=tmp_path)

    result = run(
        [
            str(virtualenv / "bin" / "b1k-deploy"), "publish-campaign",
            "--source-root", str(WORKSPACE), "--source-commit", "a" * 40,
            "--training-tag", TRAINING_TAG, "--rollout-tag", ROLLOUT_TAG,
            "--model-commit", "b" * 40, "--checkpoint-artifact-sha256", "c" * 64,
            "--gpu-ids", "0", "--receipt", str(tmp_path / "plan.json"),
        ],
        cwd=tmp_path,
    )

    assert json.loads(result.stdout)["dry_run"] is True


def test_cli_execute_uses_campaign_receipt_and_only_an_injected_trusted_adapter(tmp_path, capsys):
    builder, verifier, templates = FakeBuilder(), FakeVerifier(), FakeTemplates()

    assert main(
        [
            "publish-campaign", "--source-commit", SOURCE_COMMIT, "--training-tag", TRAINING_TAG, "--rollout-tag", ROLLOUT_TAG, *rollout_cli_arguments(),
            "--receipt", str(tmp_path / "receipt.json"), "--execute", "--adapter", "test",
        ],
        adapter_factories={"test": lambda: PublicationAdapters(builder, verifier, templates)},
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "complete"
    assert AtomicCampaignReceiptStore(tmp_path / "receipt.json").read().to_dict() == output
    assert len(builder.calls) == len(verifier.calls) == 2


def test_cli_campaign_is_dry_run_by_default_and_rejects_execute_without_a_trusted_adapter(tmp_path, capsys):
    arguments = [
        "publish-campaign", "--source-commit", SOURCE_COMMIT, "--training-tag", TRAINING_TAG, "--rollout-tag", ROLLOUT_TAG, *rollout_cli_arguments(),
        "--receipt", str(tmp_path / "receipt.json"),
    ]
    assert main(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    assert len(output["preplan_hash"]) == 64
    assert output["final_plan_hash"] is None
    assert not (tmp_path / "receipt.json").exists()

    with pytest.raises(SystemExit):
        main([*arguments, "--execute"])

    with pytest.raises(SystemExit):
        main([
            "publish-campaign", "--source-commit", SOURCE_COMMIT, "--training-tag", TRAINING_TAG, "--rollout-tag", ROLLOUT_TAG,
            "--receipt", str(tmp_path / "missing-rollout-inputs.json"),
        ])


def test_cli_requires_and_renders_nonfixture_rollout_inputs_before_any_vast_operation(tmp_path, monkeypatch):
    import b1k_deploy.cli as cli

    calls = []
    renderer = cli.render_vast_template

    def record_renderer(**kwargs):
        calls.append(kwargs)
        return renderer(**kwargs)

    monkeypatch.setattr(cli, "render_vast_template", record_renderer)
    builder, verifier, templates = FakeBuilder(), FakeVerifier(), FakeTemplates()
    assert main(
        [
            "publish-campaign", "--source-commit", SOURCE_COMMIT, "--training-tag", TRAINING_TAG, "--rollout-tag", ROLLOUT_TAG,
            *rollout_cli_arguments(), "--receipt", str(tmp_path / "receipt.json"), "--execute", "--adapter", "test",
        ],
        adapter_factories={"test": lambda: PublicationAdapters(builder, verifier, templates)},
    ) == 0

    assert calls == [{
        "image_digest": "sha256:" + "0" * 64,
        "model_commit": "c" * 40,
        "checkpoint_artifact_sha256": "d" * 64,
        "gpu_ids": (0,),
    }]
    rollout = templates.created[1]
    assert "MODEL_COMMIT=" + "c" * 40 in rollout["env"]
    assert "CHECKPOINT_ARTIFACT_SHA256=" + "d" * 64 in rollout["env"]


def test_zero_rollout_schema_fixture_is_rejected_before_vast_or_registry_calls(tmp_path):
    images, schemas = campaign_plans()
    builder, verifier, templates = FakeBuilder(), FakeVerifier(), FakeTemplates()
    publisher, _ = campaign_publisher(tmp_path, builder=builder, verifier=verifier, templates=templates)
    zero_fixture = TemplateSchemaPlan(images.source_commit, "rollout", load_canonical_template("rollout", source_root=WORKSPACE))

    with pytest.raises(PublicationError, match="rollout template"):
        publisher.publish(images, (schemas[0], zero_fixture))

    assert builder.calls == verifier.calls == templates.created == []


def test_campaign_failure_after_first_verified_image_persists_only_the_verified_artifact(tmp_path):
    images, templates = campaign_plans()
    publisher, adapters = campaign_publisher(tmp_path, verifier=FailOnVerifierCall(2))

    with pytest.raises(PublicationPartialError):
        publisher.publish(images, templates)

    receipt = AtomicCampaignReceiptStore(tmp_path / "receipt.json").read()
    assert receipt is not None
    assert receipt.status == "ambiguous"
    assert [item.purpose for item in receipt.images] == ["training"]
    assert receipt.templates == ()
    assert len(adapters.builder.calls) == 2


def test_campaign_malformed_second_verifier_readback_preserves_the_first_verified_image(tmp_path):
    images, templates = campaign_plans()
    publisher, _ = campaign_publisher(tmp_path, verifier=MalformedVerifierOnCall(2))

    with pytest.raises(PublicationPartialError):
        publisher.publish(images, templates)

    receipt = AtomicCampaignReceiptStore(tmp_path / "receipt.json").read()
    assert receipt is not None
    assert [item.purpose for item in receipt.images] == ["training"]


def test_campaign_invalid_template_pair_is_rejected_before_any_remote_adapter_is_called(tmp_path):
    images, templates = campaign_plans()
    publisher, adapters = campaign_publisher(tmp_path)

    with pytest.raises(PublicationError, match="exactly one training"):
        publisher.publish(images, (templates[0], templates[0]))

    assert adapters.builder.calls == adapters.verifier.calls == adapters.templates.created == []


def test_campaign_rejects_template_source_identity_that_diverges_from_the_image_plan_before_remote_calls(tmp_path):
    images, templates = campaign_plans()
    divergent = TemplateSchemaPlan("c" * 40, templates[0].purpose, templates[0].template)
    publisher, adapters = campaign_publisher(tmp_path)

    with pytest.raises(PublicationError, match="source commit"):
        publisher.publish(images, (divergent, templates[1]))

    assert adapters.builder.calls == adapters.verifier.calls == adapters.templates.created == []


def test_campaign_preflight_identity_changes_for_tags_and_schema_before_a_digest_exists():
    images, templates = campaign_plans()
    changed_source = "b" * 40
    changed_tags = ImagePublicationPlan(changed_source, f"trainer-{changed_source}", f"rollout-{changed_source}")
    changed_source_templates = tuple(TemplateSchemaPlan(changed_source, item.purpose, item.template, item.source_root) for item in templates)
    changed_templates = (templates[0], TemplateSchemaPlan(images.source_commit, "rollout", payload("rollout", model_commit="e" * 40)))

    assert campaign_preplan_hash(images, templates) != campaign_preplan_hash(changed_tags, changed_source_templates)
    assert campaign_preplan_hash(images, templates) != campaign_preplan_hash(images, changed_templates)


def test_campaign_final_identity_is_only_available_after_verified_digests_are_frozen():
    images, schemas = campaign_plans()
    preplan = campaign_preplan_hash(images, schemas)
    frozen = tuple(
        TemplatePublicationPlan(images.source_commit, schema.purpose, release(schema.purpose), schema.template)
        for schema in schemas
    )

    assert campaign_final_plan_hash(images, preplan, frozen) != preplan


def test_campaign_retry_does_not_reuse_a_partial_receipt_for_different_image_tags(tmp_path):
    images, templates = campaign_plans()
    builder, verifier = FakeBuilder(), FailOnVerifierCall(2)
    publisher, _ = campaign_publisher(tmp_path, builder=builder, verifier=verifier)

    with pytest.raises(PublicationPartialError):
        publisher.publish(images, templates)
    verifier.call_number = 99
    changed_source = "b" * 40
    changed_tags = ImagePublicationPlan(changed_source, f"trainer-{changed_source}", f"rollout-{changed_source}")
    changed_templates = tuple(TemplateSchemaPlan(changed_source, item.purpose, item.template, item.source_root) for item in templates)
    publisher.publish(changed_tags, changed_templates)

    assert [call[0] for call in builder.calls].count("docker.io/ryanjin333/behavior1k-groot-n17") == 4


def test_campaign_retry_reuses_verified_first_image_and_completes_atomically(tmp_path):
    images, templates = campaign_plans()
    builder, verifier = FakeBuilder(), FailOnVerifierCall(2)
    publisher, adapters = campaign_publisher(tmp_path, builder=builder, verifier=verifier)

    with pytest.raises(PublicationPartialError):
        publisher.publish(images, templates)
    verifier.call_number = 99

    completed = publisher.publish(images, templates)
    stored = AtomicCampaignReceiptStore(tmp_path / "receipt.json").read()

    assert completed.status == stored.status == "complete"
    assert {item.purpose for item in completed.images} == {"training", "rollout"}
    assert {item.purpose for item in completed.templates} == {"training", "rollout"}
    assert [call[0] for call in builder.calls].count("docker.io/ryanjin333/behavior1k-groot-n17") == 3
    assert adapters.templates.created and len(adapters.templates.created) == 2


def test_partial_receipt_drift_is_rejected_before_retry_can_rebuild_or_publish_templates(tmp_path):
    class PartialDriftVerifier(FakeVerifier):
        def verify_private_image(self, repository, tag):
            self.calls.append((repository, tag))
            if len(self.calls) == 2:
                raise RuntimeError("simulated interrupted rollout verification")
            if len(self.calls) > 2 and tag.startswith("trainer-"):
                changed = "sha256:" + "c" * 64
                return DockerImageRelease("training", repository, f"trainer-{'a' * 40}", "a" * 40, changed, f"{repository}@{changed}")
            return release("training" if tag.startswith("trainer-") else "rollout", tag.split("-", 1)[1])

    images, schemas = campaign_plans()
    builder, verifier, client = FakeBuilder(), PartialDriftVerifier(), FakeTemplates()
    publisher, _ = campaign_publisher(tmp_path, builder=builder, verifier=verifier, templates=client)
    with pytest.raises(PublicationPartialError):
        publisher.publish(images, schemas)

    with pytest.raises(PublicationError, match="registry readback drifted"):
        publisher.publish(images, schemas)

    assert len(builder.calls) == 2
    assert client.created == []


def test_campaign_template_create_failure_after_both_images_is_recorded_and_retry_reuses_images(tmp_path):
    images, templates = campaign_plans()
    builder, verifier, client = FakeBuilder(), FakeVerifier(), FailTemplateCreate()
    publisher, _ = campaign_publisher(tmp_path, builder=builder, verifier=verifier, templates=client)

    with pytest.raises(PublicationError, match="template publication failed"):
        publisher.publish(images, templates)
    partial = AtomicCampaignReceiptStore(tmp_path / "receipt.json").read()
    assert partial is not None and partial.status == "ambiguous"
    assert len(partial.images) == 2
    assert partial.templates == ()

    client.fail = False
    completed = publisher.publish(images, templates)

    assert completed.status == "complete"
    assert len(builder.calls) == 2
    assert len(verifier.calls) == 4  # retry re-verifies both immutable receipts


def test_campaign_template_readback_failure_is_reconciled_without_new_image_or_template_create(tmp_path):
    images, templates = campaign_plans()
    builder, verifier, client = FakeBuilder(), FakeVerifier(), FailTemplateReadbackOnce()
    publisher, _ = campaign_publisher(tmp_path, builder=builder, verifier=verifier, templates=client)

    with pytest.raises(PublicationError, match="template readback failed"):
        publisher.publish(images, templates)
    partial = AtomicCampaignReceiptStore(tmp_path / "receipt.json").read()
    assert partial is not None and len(partial.images) == 2 and partial.templates == ()
    assert len(client.created) == 1

    completed = publisher.publish(images, templates)

    assert completed.status == "complete"
    assert len(builder.calls) == 2
    assert len(verifier.calls) == 4  # retry re-verifies both immutable receipts
    assert len(client.created) == 2


def test_ambiguous_one_template_receipt_freshly_verifies_before_skipping_a_deleted_template(tmp_path):
    images, schemas = campaign_plans()
    builder, verifier, client = FakeBuilder(), FakeVerifier(), FailSecondTemplateCreate()
    publisher, _ = campaign_publisher(tmp_path, builder=builder, verifier=verifier, templates=client)

    with pytest.raises(PublicationError, match="template publication failed"):
        publisher.publish(images, schemas)
    partial = AtomicCampaignReceiptStore(tmp_path / "receipt.json").read()
    assert partial is not None and partial.status == "ambiguous" and partial.phase == "templates"
    assert len(partial.templates) == 1
    client.templates.pop("100")

    with pytest.raises(PublicationError, match="template readback failed"):
        publisher.publish(images, schemas)

    assert len(builder.calls) == 2
    assert len(client.created) == 1
    assert AtomicCampaignReceiptStore(tmp_path / "receipt.json").read() == partial


def test_final_receipt_write_failure_keeps_two_template_receipt_that_freshly_rejects_drift(tmp_path):
    images, schemas = campaign_plans()
    builder, verifier, client = FakeBuilder(), FakeVerifier(), FakeTemplates()
    store = FailFinalReceiptStore(tmp_path / "receipt.json")
    publisher = CampaignPublisher(PublicationAdapters(builder, verifier, client), store)

    with pytest.raises(OSError, match="final receipt write failure"):
        publisher.publish(images, schemas)
    partial = AtomicCampaignReceiptStore(store.path).read()
    assert partial is not None and partial.status == "ambiguous" and partial.phase == "templates"
    assert len(partial.templates) == 2
    client.templates["100"] = {**client.templates["100"], "env": client.templates["100"]["env"] + " -e DRIFT=1"}

    with pytest.raises(PublicationError, match="template readback drifted"):
        publisher.publish(images, schemas)

    assert len(builder.calls) == 2
    assert len(client.created) == 2
    assert AtomicCampaignReceiptStore(store.path).read() == partial


def test_campaign_durably_records_each_verified_remote_boundary_before_complete_receipt(tmp_path):
    images, templates = campaign_plans()
    store = RecordingReceiptStore(tmp_path / "receipt.json")
    completed = CampaignPublisher(PublicationAdapters(FakeBuilder(), FakeVerifier(), FakeTemplates()), store).publish(images, templates)

    assert [(item.status, item.phase, len(item.images), len(item.templates), item.final_plan_hash is not None) for item in store.writes] == [
        ("ambiguous", "preflight", 0, 0, False),
        ("ambiguous", "images", 1, 0, False),
        ("ambiguous", "images", 2, 0, True),
        ("ambiguous", "templates", 2, 1, True),
        ("ambiguous", "templates", 2, 2, True),
        ("complete", "complete", 2, 2, True),
    ]
    assert all(item.preplan_hash == completed.preplan_hash for item in store.writes)
    assert AtomicCampaignReceiptStore(store.path).read() == completed


def test_atomic_receipt_never_replaces_last_known_good_receipt_when_durable_write_fails(tmp_path, monkeypatch):
    store = AtomicCampaignReceiptStore(tmp_path / "receipt.json")
    images, templates = campaign_plans()
    complete = CampaignPublisher(PublicationAdapters(FakeBuilder(), FakeVerifier(), FakeTemplates()), store).publish(images, templates)
    before = store.path.read_bytes()

    monkeypatch.setattr("b1k_deploy.publish.os.fsync", lambda descriptor: (_ for _ in ()).throw(OSError("simulated sync failure")))
    with pytest.raises(OSError, match="simulated sync failure"):
        store.write(complete)

    assert store.path.read_bytes() == before
    assert not list(tmp_path.glob(".receipt.json.tmp"))


def test_atomic_receipt_cleans_the_unique_temp_file_when_its_mode_hardening_fails(tmp_path, monkeypatch):
    store = AtomicCampaignReceiptStore(tmp_path / "receipt.json")
    images, templates = campaign_plans()
    receipt = CampaignPublisher(PublicationAdapters(FakeBuilder(), FakeVerifier(), FakeTemplates()), store).publish(images, templates)
    original = __import__("b1k_deploy.publish", fromlist=["os"]).os.fchmod
    calls = []

    def fail_temporary_mode(descriptor, mode):
        calls.append((descriptor, mode))
        if len(calls) == 2:
            raise OSError("simulated fchmod failure")
        return original(descriptor, mode)

    monkeypatch.setattr("b1k_deploy.publish.os.fchmod", fail_temporary_mode)
    with pytest.raises(OSError, match="simulated fchmod failure"):
        store.write(receipt)

    assert not list(tmp_path.glob(".receipt.json.*.tmp"))


def test_atomic_receipt_write_never_follows_a_hostile_stale_temp_symlink(tmp_path):
    store = AtomicCampaignReceiptStore(tmp_path / "receipt.json")
    images, templates = campaign_plans()
    receipt = CampaignPublisher(PublicationAdapters(FakeBuilder(), FakeVerifier(), FakeTemplates()), store).publish(images, templates)
    target = tmp_path / "must-not-change"
    target.write_text("original", encoding="utf-8")
    (tmp_path / ".receipt.json.tmp").symlink_to(target)

    store.write(receipt)

    assert target.read_text(encoding="utf-8") == "original"
    assert AtomicCampaignReceiptStore(store.path).read() == receipt
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / ".receipt.json.lock").stat().st_mode & 0o777 == 0o600


def test_atomic_receipt_store_rejects_a_hostile_lock_symlink(tmp_path):
    store = AtomicCampaignReceiptStore(tmp_path / "receipt.json")
    target = tmp_path / "must-not-lock"
    target.write_text("original", encoding="utf-8")
    (tmp_path / ".receipt.json.lock").symlink_to(target)
    images, templates = campaign_plans()
    receipt = CampaignPublicationReceipt(images.source_commit, "ambiguous", "preflight", "a" * 64, None, (), ())

    with pytest.raises(PublicationError, match="lock is unsafe"):
        store.write(receipt)

    assert target.read_text(encoding="utf-8") == "original"


def test_atomic_receipt_store_serializes_concurrent_writers_without_losing_valid_json(tmp_path):
    store = AtomicCampaignReceiptStore(tmp_path / "receipt.json")
    images, templates = campaign_plans()
    receipt = CampaignPublisher(PublicationAdapters(FakeBuilder(), FakeVerifier(), FakeTemplates()), store).publish(images, templates)
    barrier = Barrier(4)

    def write() -> None:
        barrier.wait()
        store.write(receipt)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(write) for _ in range(4)]
        for future in futures:
            future.result()

    assert AtomicCampaignReceiptStore(store.path).read() == receipt


def test_campaign_lock_serializes_concurrent_publishers_so_remote_images_are_not_rebuilt(tmp_path):
    images, templates = campaign_plans()
    builder, verifier, client = FakeBuilder(), FakeVerifier(), FakeTemplates()
    store = AtomicCampaignReceiptStore(tmp_path / "receipt.json")
    adapters = PublicationAdapters(builder, verifier, client)
    first = CampaignPublisher(adapters, store)
    second = CampaignPublisher(adapters, AtomicCampaignReceiptStore(store.path))
    barrier = Barrier(2)

    def publish(publisher):
        barrier.wait()
        return publisher.publish(images, templates)

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = [future.result() for future in (executor.submit(publish, first), executor.submit(publish, second))]

    assert [receipt.status for receipt in receipts] == ["complete", "complete"]
    assert len(builder.calls) == 2
    assert len(verifier.calls) == 4  # the second caller fresh-verifies the completed release
    assert len(client.created) == 2


def test_matching_complete_receipt_is_freshly_verified_without_rebuilding_or_template_mutation(tmp_path):
    images, schemas = campaign_plans()
    builder, verifier, client = FakeBuilder(), FakeVerifier(), FakeTemplates()
    publisher, _ = campaign_publisher(tmp_path, builder=builder, verifier=verifier, templates=client)

    first = publisher.publish(images, schemas)
    second = publisher.publish(images, schemas)

    assert second == first
    assert len(builder.calls) == 2
    assert len(verifier.calls) == 4
    assert len(client.created) == 2


def test_matching_complete_receipt_rejects_deleted_template_without_mutating_or_rebuilding(tmp_path):
    images, schemas = campaign_plans()
    builder, verifier, client = FakeBuilder(), FakeVerifier(), FakeTemplates()
    publisher, _ = campaign_publisher(tmp_path, builder=builder, verifier=verifier, templates=client)
    publisher.publish(images, schemas)
    client.templates.pop("100")

    with pytest.raises(PublicationError, match="template readback"):
        publisher.publish(images, schemas)

    assert len(builder.calls) == 2
    assert len(client.created) == 2
    assert AtomicCampaignReceiptStore(tmp_path / "receipt.json").read().status == "complete"


def test_matching_complete_receipt_rejects_registry_digest_drift_without_rebuilding(tmp_path):
    class DriftVerifier(FakeVerifier):
        drift = False

        def verify_private_image(self, repository, tag):
            if self.drift:
                expected = release("training" if repository.endswith("trainer") else "rollout")
                changed = "sha256:" + "c" * 64
                return DockerImageRelease(expected.purpose, expected.repository, expected.tag, expected.source_commit, changed, f"{expected.repository}@{changed}")
            return super().verify_private_image(repository, tag)

    images, schemas = campaign_plans()
    builder, verifier, client = FakeBuilder(), DriftVerifier(), FakeTemplates()
    publisher, _ = campaign_publisher(tmp_path, builder=builder, verifier=verifier, templates=client)
    publisher.publish(images, schemas)
    verifier.drift = True

    with pytest.raises(PublicationError, match="registry readback drifted"):
        publisher.publish(images, schemas)

    assert len(builder.calls) == 2
    assert len(client.created) == 2


def test_cli_invalid_plan_is_rejected_before_any_trusted_adapter_factory_is_called(tmp_path):
    called = []

    with pytest.raises(SystemExit):
        main(
            [
                "publish-campaign", "--source-commit", "NOT-A-COMMIT", "--training-tag", TRAINING_TAG, "--rollout-tag", ROLLOUT_TAG, *rollout_cli_arguments(),
                "--receipt", str(tmp_path / "receipt.json"), "--execute", "--adapter", "test",
            ],
            adapter_factories={"test": lambda: called.append(True)},
        )

    assert called == []


def test_cli_rejects_arbitrary_adapter_names_without_evaluating_them(tmp_path):
    called = []
    arguments = [
        "publish-campaign", "--source-commit", SOURCE_COMMIT, "--training-tag", TRAINING_TAG, "--rollout-tag", ROLLOUT_TAG, *rollout_cli_arguments(),
        "--receipt", str(tmp_path / "receipt.json"), "--execute", "--adapter", "not-an-import:factory",
    ]

    with pytest.raises(SystemExit):
        main(arguments, adapter_factories={"test": lambda: called.append(True)})

    assert called == []


def test_fixed_buildx_builder_uses_private_inline_auth_and_exact_amd64_release_arguments(tmp_path):
    from b1k_deploy.production import DockerBuildxImageBuilder, buildx_release_command

    token_file = tmp_path / "docker-token"
    token_file.write_text("token-never-in-arguments", encoding="utf-8")
    token_file.chmod(0o600)
    resources = tmp_path / "Resources"
    docker_executable = resources / "bin" / "docker"
    docker_executable.parent.mkdir(parents=True)
    docker_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    docker_executable.chmod(0o700)
    buildx_plugin = resources / "cli-plugins" / "docker-buildx"
    buildx_plugin.parent.mkdir()
    buildx_plugin.write_text("#!/bin/sh\n", encoding="utf-8")
    buildx_plugin.chmod(0o700)

    class Runner:
        def __init__(self):
            self.calls = []
            self.configs = []

        def run(self, arguments, *, stdin, env, timeout):
            self.calls.append((arguments, stdin, dict(env), timeout))
            config = Path(env["DOCKER_CONFIG"]) / "config.json"
            self.configs.append((config.read_text(encoding="utf-8"), config.stat().st_mode & 0o777))
            linked_plugin = config.parent / "cli-plugins" / "docker-buildx"
            assert linked_plugin.is_symlink()
            assert linked_plugin.resolve(strict=True) == buildx_plugin
            assert linked_plugin.parent.stat().st_mode & 0o777 == 0o700
            return CommandResult(0, "", "")

    runner = Runner()
    builder = DockerBuildxImageBuilder(
        username="ryanjin333",
        token_file=token_file,
        docker_executable=docker_executable,
        workspace=Path(__file__).resolve().parents[2],
        runner=runner,
        release_context=PassingReleaseContext(),
    )
    builder.build_and_push(release("training").repository, TRAINING_TAG, SOURCE_COMMIT)

    [build] = runner.calls
    assert build[1] is None
    assert "token-never-in-arguments" not in build[0]
    assert runner.configs == [(runner.configs[0][0], 0o600)]
    assert '"auths":{"https://index.docker.io/v1/":{"auth":' in runner.configs[0][0]
    assert "token-never-in-arguments" not in str(build)
    assert build[0] == buildx_release_command(
        docker_executable,
        Path(__file__).resolve().parents[2],
        release("training").repository,
        TRAINING_TAG,
        "a" * 40,
    )
    assert ("--target", "training-runtime") in zip(build[0], build[0][1:])
    assert ("--label", "io.lehome.release-mode=release") in zip(build[0], build[0][1:])
    assert build[3] == 7200.0


def test_buildx_plugin_discovery_derives_the_homebrew_prefix_from_the_resolved_docker_executable(tmp_path):
    from b1k_deploy.production import _discover_buildx_plugin

    prefix = tmp_path / "homebrew"
    docker_executable = prefix / "Cellar" / "docker" / "29.0.1" / "bin" / "docker"
    docker_executable.parent.mkdir(parents=True)
    docker_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    docker_executable.chmod(0o700)
    buildx_plugin = prefix / "lib" / "docker" / "cli-plugins" / "docker-buildx"
    buildx_plugin.parent.mkdir(parents=True)
    buildx_plugin.write_text("#!/bin/sh\n", encoding="utf-8")
    buildx_plugin.chmod(0o700)

    assert _discover_buildx_plugin(docker_executable) == buildx_plugin.resolve(strict=True)


def test_fixed_buildx_builder_rejects_workspace_head_mismatch_before_docker_login_or_build(tmp_path):
    from b1k_deploy.production import DockerBuildxImageBuilder, GitReleaseContext

    token_file = tmp_path / "docker-token"
    token_file.write_text("token", encoding="utf-8")
    token_file.chmod(0o600)
    docker_executable = tmp_path / "docker"
    docker_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    docker_executable.chmod(0o700)

    class DockerRunner:
        calls = []

        def run(self, arguments, *, stdin, env, timeout):
            self.calls.append(arguments)
            return CommandResult(0, "", "")

    git_calls = []

    def git(arguments):
        git_calls.append(arguments)
        return "b" * 40 + "\n"

    runner = DockerRunner()
    builder = DockerBuildxImageBuilder(
        username="ryanjin333", token_file=token_file, docker_executable=docker_executable,
        workspace=Path(__file__).resolve().parents[2], runner=runner, release_context=GitReleaseContext(git),
    )
    with pytest.raises(PublicationError, match="does not match workspace HEAD"):
        builder.build_and_push(release("training").repository, TRAINING_TAG, SOURCE_COMMIT)

    assert runner.calls == []
    assert len(git_calls) == 1 and git_calls[0][-2:] == ("rev-parse", "HEAD")


@pytest.mark.parametrize("status", [" M trainer/Dockerfile\n", "?? rollout/generated-input\n"])
def test_fixed_buildx_builder_rejects_tracked_or_untracked_release_context_before_docker_calls(tmp_path, status):
    from b1k_deploy.production import DockerBuildxImageBuilder, GitReleaseContext

    token_file = tmp_path / "docker-token"
    token_file.write_text("token", encoding="utf-8")
    token_file.chmod(0o600)
    docker_executable = tmp_path / "docker"
    docker_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    docker_executable.chmod(0o700)

    class DockerRunner:
        calls = []

        def run(self, arguments, *, stdin, env, timeout):
            self.calls.append(arguments)
            return CommandResult(0, "", "")

    outputs = iter(["a" * 40 + "\n", status])
    runner = DockerRunner()
    builder = DockerBuildxImageBuilder(
        username="ryanjin333", token_file=token_file, docker_executable=docker_executable,
        workspace=Path(__file__).resolve().parents[2], runner=runner,
        release_context=GitReleaseContext(lambda arguments: next(outputs)),
    )
    with pytest.raises(PublicationError, match="tracked or untracked changes"):
        builder.build_and_push(release("training").repository, TRAINING_TAG, SOURCE_COMMIT)

    assert runner.calls == []


def test_fixed_vast_client_uses_raw_bounded_commands_and_template_readback_without_exposing_api_key(tmp_path):
    from b1k_deploy.production import VastCliTemplateClient

    vastai_executable = tmp_path / "vastai"
    vastai_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    vastai_executable.chmod(0o700)
    api_key_file = tmp_path / "vast_api_key"
    api_key_file.write_text("vast-secret-never-in-command", encoding="utf-8")
    api_key_file.chmod(0o600)
    expected_client = FakeTemplates()
    TemplatePublisher(expected_client).publish(TemplatePublicationPlan("a" * 40, "training", release("training"), payload()), execute=True)
    rendered = expected_client.created[0]
    provider_row = {
        **rendered,
        "id": 101,
        "extra_filters": json.dumps(rendered["extra_filters"]),
        "recommended_disk_space": 2048.0,
        "hash_id": "provider-hash",
        "creator_id": 99,
        "created_at": 1234567890,
        "count_created": 7,
        "default_tag": False,
        "docker_login_repo": "docker.io",
        "recommended": True,
        "recent_create_date": "provider-owned",
        "tag": "provider-owned",
    }

    class Runner:
        def __init__(self):
            self.calls = []
            self.responses = [[], "New Template: 101", [provider_row], [provider_row]]

        def run(self, arguments, *, stdin, env, timeout):
            self.calls.append((arguments, stdin, dict(env), timeout))
            return CommandResult(0, json.dumps(self.responses.pop(0)), "vast-secret-never-in-command")

    runner = Runner()
    client = VastCliTemplateClient(vastai_executable=vastai_executable, api_key_file=api_key_file, runner=runner)
    receipt = TemplatePublisher(client).publish(TemplatePublicationPlan("a" * 40, "training", release("training"), payload()), execute=True)

    search, create, creation_readback, readback = runner.calls
    assert receipt.template_id == "101"
    assert search[0][:4] == (str(vastai_executable), "--raw", "search", "templates")
    assert create[0][:4] == (str(vastai_executable), "--raw", "create", "template")
    assert "--ssh" in create[0] and "--direct" in create[0] and "--no-default" in create[0]
    assert ("--login", "docker.io") in zip(create[0], create[0][1:])
    search_params = create[0][create[0].index("--search_params") + 1]
    assert "cpu_ram >= 128" in search_params
    assert "gpu_ram >= 96" in search_params
    assert "cpu_ram >= 128000" not in search_params
    assert "gpu_ram >= 96000" not in search_params
    assert rendered["extra_filters"]["cpu_ram"] == {"gte": 128000}
    assert rendered["extra_filters"]["gpu_ram"] == {"gte": 96000}
    assert creation_readback[0][-1].startswith("name==")
    assert readback[0][-1] == "id==101"
    assert all("vast-secret-never-in-command" not in " ".join(call[0]) for call in runner.calls)
    assert all(call[1] is None and call[3] == 30.0 for call in runner.calls)


def test_fixed_vast_client_permits_only_the_canonical_rollout_onstart(tmp_path):
    from b1k_deploy.production import VastCliTemplateClient

    vastai_executable = tmp_path / "vastai"
    vastai_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    vastai_executable.chmod(0o700)
    api_key_file = tmp_path / "vast_api_key"
    api_key_file.write_text("not-in-command", encoding="utf-8")
    api_key_file.chmod(0o600)

    class Runner:
        def run(self, arguments, *, stdin, env, timeout):
            return CommandResult(0, "[]", "")

    client = VastCliTemplateClient(vastai_executable=vastai_executable, api_key_file=api_key_file, runner=Runner())
    rollout_plan = TemplatePublicationPlan("a" * 40, "rollout", release("rollout"), payload("rollout"))
    rollout_templates = FakeTemplates()
    TemplatePublisher(rollout_templates).publish(rollout_plan, execute=True)
    rollout_command = client._create_command(rollout_templates.created[0])
    assert ("--onstart-cmd", ROLLOUT_ONSTART) in zip(rollout_command, rollout_command[1:])
    assert "B1K_ACCEPT_DATASET_TOS=YES" in rollout_templates.created[0]["env"]

    with pytest.raises(PublicationError, match="training template"):
        client._create_command({**payload("training"), "onstart": ""})
    for onstart in ("", "bash /usr/local/bin/b1k-rollout-entrypoint", "rollout command"):
        with pytest.raises(PublicationError, match="rollout template"):
            client._create_command({**rollout_templates.created[0], "onstart": onstart})


def test_vast_cli_filter_renderer_rejects_noncanonical_raw_ram_units() -> None:
    from b1k_deploy.production import _search_query

    with pytest.raises(PublicationError, match="1000-unit"):
        _search_query({"gpu_ram": {"gte": 24576}})


def test_fixed_vast_client_rejects_unknown_readback_fields_instead_of_hashing_provider_shape(tmp_path):
    from b1k_deploy.production import VastCliTemplateClient

    vastai_executable = tmp_path / "vastai"
    vastai_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    vastai_executable.chmod(0o700)
    api_key_file = tmp_path / "vast_api_key"
    api_key_file.write_text("not-in-arguments", encoding="utf-8")
    api_key_file.chmod(0o600)
    plan = TemplatePublicationPlan("a" * 40, "training", release("training"), payload())
    rendered_templates = FakeTemplates()
    TemplatePublisher(rendered_templates).publish(plan, execute=True)
    rendered_payload = rendered_templates.created[0]

    class Runner:
        def run(self, arguments, *, stdin, env, timeout):
            return CommandResult(0, json.dumps([{**rendered_payload, "id": 101, "docker_login_repo": "docker.io", "unclassified_provider_value": "drift"}]), "")

    client = VastCliTemplateClient(vastai_executable=vastai_executable, api_key_file=api_key_file, runner=Runner())
    with pytest.raises(PublicationError, match="unknown"):
        client.get_template("101")


def test_fixed_vast_client_accepts_current_vast_provider_only_readback_fields(tmp_path):
    from b1k_deploy.production import VastCliTemplateClient

    vastai_executable = tmp_path / "vastai"
    vastai_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    vastai_executable.chmod(0o700)
    api_key_file = tmp_path / "vast_api_key"
    api_key_file.write_text("not-in-arguments", encoding="utf-8")
    api_key_file.chmod(0o600)
    plan = TemplatePublicationPlan("a" * 40, "training", release("training"), payload())
    rendered_templates = FakeTemplates()
    TemplatePublisher(rendered_templates).publish(plan, execute=True)
    rendered_payload = rendered_templates.created[0]
    provider_metadata = {
        "id": 101,
        "args_str": None,
        "autoscaler": False,
        "cached": False,
        "command": None,
        "created_from": None,
        "created_from_id": None,
        "deleted_at": None,
        "desc_count": 0,
        "docker_login_pass": None,
        "docker_login_repo": "docker.io",
        "docker_login_user": None,
        "jupyter_tested": None,
        "jupyterlab_tested": None,
        "lang_utf8": None,
        "max_cuda": None,
        "min_cuda": None,
        "python_utf8": None,
        "readme_hash": None,
        "sort_order": None,
        "vm": False,
        "volume_info": None,
    }

    class Runner:
        def run(self, arguments, *, stdin, env, timeout):
            return CommandResult(0, json.dumps([{**rendered_payload, **provider_metadata}]), "")

    client = VastCliTemplateClient(vastai_executable=vastai_executable, api_key_file=api_key_file, runner=Runner())

    assert client.get_template("101") == rendered_payload


def test_fixed_vast_client_normalizes_current_vast_filter_readback_types(tmp_path):
    from b1k_deploy.production import VastCliTemplateClient

    vastai_executable = tmp_path / "vastai"
    vastai_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    vastai_executable.chmod(0o700)
    api_key_file = tmp_path / "vast_api_key"
    api_key_file.write_text("not-in-arguments", encoding="utf-8")
    api_key_file.chmod(0o600)
    plan = TemplatePublicationPlan("a" * 40, "training", release("training"), payload())
    rendered_templates = FakeTemplates()
    TemplatePublisher(rendered_templates).publish(plan, execute=True)
    rendered_payload = rendered_templates.created[0]
    provider_filters = json.loads(json.dumps(rendered_payload["extra_filters"]))
    for field, constraints in provider_filters.items():
        for operator, value in constraints.items():
            if isinstance(value, int) and not isinstance(value, bool):
                constraints[operator] = float(value) if field in {"cpu_ram", "gpu_ram"} else str(value)
    provider_row = {
        **rendered_payload,
        "id": 101,
        "docker_login_repo": "docker.io",
        "extra_filters": json.dumps(provider_filters),
    }

    class Runner:
        def run(self, arguments, *, stdin, env, timeout):
            return CommandResult(0, json.dumps([provider_row]), "")

    client = VastCliTemplateClient(vastai_executable=vastai_executable, api_key_file=api_key_file, runner=Runner())

    assert client.get_template("101") == rendered_payload


def test_fixed_vast_client_rejects_template_without_private_pull_repo(tmp_path):
    from b1k_deploy.production import VastCliTemplateClient

    vastai_executable = tmp_path / "vastai"
    vastai_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    vastai_executable.chmod(0o700)
    api_key_file = tmp_path / "vast_api_key"
    api_key_file.write_text("not-in-arguments", encoding="utf-8")
    api_key_file.chmod(0o600)
    plan = TemplatePublicationPlan("a" * 40, "training", release("training"), payload())
    rendered_templates = FakeTemplates()
    TemplatePublisher(rendered_templates).publish(plan, execute=True)
    rendered_payload = rendered_templates.created[0]
    provider_row = {**rendered_payload, "id": 101, "docker_login_repo": None}

    class Runner:
        def run(self, arguments, *, stdin, env, timeout):
            return CommandResult(0, json.dumps([provider_row]), "")

    client = VastCliTemplateClient(vastai_executable=vastai_executable, api_key_file=api_key_file, runner=Runner())

    assert client.find_private_template(rendered_payload["name"], rendered_payload["image"]) is None
    with pytest.raises(PublicationError, match="private pull"):
        client.get_template("101")


def test_fixed_subprocess_clients_redact_failing_provider_output(tmp_path):
    from b1k_deploy.production import VastCliTemplateClient

    vastai_executable = tmp_path / "vastai"
    vastai_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    vastai_executable.chmod(0o700)
    api_key_file = tmp_path / "vast_api_key"
    api_key_file.write_text("vast-secret-never-in-error", encoding="utf-8")
    api_key_file.chmod(0o600)

    class Runner:
        def run(self, arguments, *, stdin, env, timeout):
            return CommandResult(1, "", "vast-secret-never-in-error")

    client = VastCliTemplateClient(vastai_executable=vastai_executable, api_key_file=api_key_file, runner=Runner())
    with pytest.raises(PublicationError) as error:
        client.find_private_template("b1k-training-aaaaaaaaaaaaaaaa", release("training").reference)

    assert "vast-secret-never-in-error" not in str(error.value)


def test_configured_adapter_settings_accept_only_private_token_pointers_and_absolute_executables(tmp_path):
    from b1k_deploy.production import ConfiguredPublicationSettings

    token_file = tmp_path / "docker-token"
    token_file.write_text("token-value-never-serialized", encoding="utf-8")
    token_file.chmod(0o600)
    home = tmp_path / "home"
    api_key_file = home / ".config" / "vastai" / "vast_api_key"
    api_key_file.parent.mkdir(parents=True)
    api_key_file.write_text("vast-value-never-serialized", encoding="utf-8")
    api_key_file.chmod(0o600)
    docker_executable = tmp_path / "docker"
    vastai_executable = tmp_path / "vastai"
    for executable in (docker_executable, vastai_executable):
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o700)

    settings = ConfiguredPublicationSettings.from_environment(
        {
            "B1K_DOCKER_USERNAME": "ryanjin333",
            "B1K_DOCKER_TOKEN_FILE": str(token_file),
            "B1K_DOCKER_EXECUTABLE": str(docker_executable),
            "B1K_VASTAI_EXECUTABLE": str(vastai_executable),
        },
        home=home,
        workspace=Path(__file__).resolve().parents[2],
    )

    assert settings.docker_token_file == token_file
    assert settings.vast_api_key_file == api_key_file
    assert "token-value-never-serialized" not in repr(settings)
    assert "vast-value-never-serialized" not in repr(settings)
