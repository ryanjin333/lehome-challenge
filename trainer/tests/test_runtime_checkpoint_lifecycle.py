from __future__ import annotations

from pathlib import Path

import pytest

from lehome_train.checkpoints import CheckpointDescriptor, write_checkpoint_descriptor
from lehome_train.io import canonical_json_bytes, sha256_file
from lehome_train.models import ArtifactIdentity, CheckpointRecord
from lehome_train.groot.runtime_checkpoint_lifecycle import (
    RuntimeMixtureTrainingIdentity,
    authorize_runtime_mixture_disposal,
    build_runtime_checkpoint_anchor,
    discover_runtime_checkpoint_anchor,
    discover_runtime_mixture_resume,
    provider_interruption_terminal,
    runtime_mixture_completion_terminal,
    publish_runtime_mixture_checkpoint,
)


REPOSITORY = "ryanjin333/lehome-groot-n17-models"


class FakeHub:
    """Literal in-memory private Hub: immutable commits and fresh readback."""

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, bytes]] = {}
        self.refs: dict[str, str] = {}

    def publish(self, *, checkpoint: CheckpointDescriptor, artifact_root: Path, identity: RuntimeMixtureTrainingIdentity) -> dict[str, object]:
        step = checkpoint.record.optimizer_step
        archive = artifact_root / checkpoint.record.artifact.relative_path
        descriptor_relative = f"checkpoints/step-{step}.json"
        descriptor = artifact_root / descriptor_relative
        revision = f"{step:040x}"
        prefix = f"runtime-mixture/{identity.mixture_id}/{checkpoint.record.artifact.sha256}"
        self.objects[revision] = {
            f"{prefix}/{checkpoint.record.artifact.relative_path}": archive.read_bytes(),
            f"{prefix}/{descriptor_relative}": descriptor.read_bytes(),
        }
        return {
            "optimizer_step": step,
            "repository": REPOSITORY,
            "immutable_revision": revision,
            "remote_prefix": prefix,
            "relative_path": checkpoint.record.artifact.relative_path,
            "artifact_sha256": sha256_file(archive),
            "artifact_byte_size": archive.stat().st_size,
            "descriptor_relative_path": descriptor_relative,
            "descriptor_sha256": sha256_file(descriptor),
            "descriptor_byte_size": descriptor.stat().st_size,
            "readback_verified": True,
        }

    def list_tree(self, *, repository: str, revision: str) -> tuple[str, ...]:
        assert repository == REPOSITORY
        return tuple(sorted(self.objects[revision]))

    def download_files(self, *, repository: str, revision: str, destination: Path, relative_paths: tuple[str, ...], remote_prefix: str) -> None:
        assert repository == REPOSITORY
        for relative in relative_paths:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.objects[revision][f"{remote_prefix}/{relative}"])

    def resolve_approved_ref(self, *, repository: str, ref: str) -> str:
        assert repository == REPOSITORY
        assert ref == "main"
        return self.refs[ref]

    def write_anchor(self, *, anchor: dict[str, object], revision: str, ref: str = "main") -> str:
        path = f"checkpoints/{anchor['experiment_id']}/latest.json"
        self.objects[revision] = {path: canonical_json_bytes(anchor)}
        self.refs[ref] = revision
        return sha256_file_from_bytes(self.objects[revision][path])


def sha256_file_from_bytes(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


def _identity() -> RuntimeMixtureTrainingIdentity:
    return RuntimeMixtureTrainingIdentity(
        mixture_id="a" * 64,
        deployment_receipt_sha256="b" * 64,
        source_revisions=(
            ("organizer", "c" * 40, "bc/full", "d" * 64),
            ("rollout", "e" * 40, "rollouts/round-1", "f" * 64),
        ),
        schedule_seed=17,
        code_bundle_sha256="1" * 64,
        code_bundle_revision="2" * 40,
        oci_image="sha256:" + "3" * 64,
        parent_step12000_artifact_sha256="4" * 64,
    )


def test_runtime_training_identity_binds_optional_awr_evidence_and_configuration() -> None:
    identity = RuntimeMixtureTrainingIdentity(
        mixture_id="a" * 64,
        deployment_receipt_sha256="b" * 64,
        source_revisions=(
            ("organizer", "c" * 40, "bc/full", "d" * 64),
            ("rollout", "e" * 40, "rollouts/round-1", "f" * 64),
        ),
        schedule_seed=17,
        code_bundle_sha256="1" * 64,
        code_bundle_revision="2" * 40,
        oci_image="sha256:" + "3" * 64,
        parent_step12000_artifact_sha256="4" * 64,
        awr_evidence_sha256="8" * 64,
        awr_config_sha256="9" * 64,
    )

    assert identity.to_dict()["awr_evidence_sha256"] == "8" * 64
    assert identity.to_dict()["awr_config_sha256"] == "9" * 64
    assert identity.sha256 != _identity().sha256


def test_runtime_training_identity_rejects_a_partial_awr_binding() -> None:
    with pytest.raises(ValueError, match="AWR.*together"):
        RuntimeMixtureTrainingIdentity(
            mixture_id="a" * 64,
            deployment_receipt_sha256="b" * 64,
            source_revisions=(
                ("organizer", "c" * 40, "bc/full", "d" * 64),
                ("rollout", "e" * 40, "rollouts/round-1", "f" * 64),
            ),
            schedule_seed=17,
            code_bundle_sha256="1" * 64,
            code_bundle_revision="2" * 40,
            oci_image="sha256:" + "3" * 64,
            parent_step12000_artifact_sha256="4" * 64,
            awr_evidence_sha256="8" * 64,
        )


def _checkpoint(root: Path, *, step: int, identity: RuntimeMixtureTrainingIdentity) -> CheckpointDescriptor:
    archive = root / "checkpoints" / f"step-{step}.tar"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(f"archive-{step}".encode())
    descriptor = CheckpointDescriptor(
        record=CheckpointRecord(
            experiment_id="runtime-mixture-70-30",
            optimizer_step=step,
            sample_presentations=step * 64,
            experiment_config_sha256="5" * 64,
            dataset_manifest_sha256=identity.mixture_id,
            schedule_sha256="6" * 64,
            artifact=ArtifactIdentity(
                relative_path=f"checkpoints/step-{step}.tar",
                sha256=sha256_file(archive),
                byte_size=archive.stat().st_size,
            ),
            resumable=True,
            remotely_verified=False,
        ),
        normalization_sha256="7" * 64,
        schedule_sha256="6" * 64,
        locally_verified=True,
    )
    write_checkpoint_descriptor(root / "checkpoints" / f"step-{step}.json", descriptor)
    return descriptor


def test_fake_hub_interrupt_discover_hydrate_resume_publish_and_complete(tmp_path: Path) -> None:
    identity, hub = _identity(), FakeHub()
    one = publish_runtime_mixture_checkpoint(
        identity=identity, checkpoint=_checkpoint(tmp_path / "first", step=1000, identity=identity),
        artifact_root=tmp_path / "first", publisher=hub.publish, hub=hub,
    )
    interrupted = provider_interruption_terminal(
        identity=identity, instance_id="instance-a", publications=(one,),
        provider_loss={"kind": "instance_absent", "evidence_sha256": "8" * 64},
    )
    resume = discover_runtime_mixture_resume(
        terminal=interrupted, identity=identity, hub=hub, destination=tmp_path / "hydrate",
    )
    assert resume.optimizer_step == 1000
    assert resume.global_sample_offset == 64_000
    assert resume.dataset_kwargs() == {"global_sample_offset": 64_000, "expected_global_step": 1000, "global_batch_size": 64}
    two_root = tmp_path / "second"
    two = publish_runtime_mixture_checkpoint(
        identity=identity, checkpoint=_checkpoint(two_root, step=2000, identity=identity),
        artifact_root=two_root, publisher=hub.publish, hub=hub,
    )
    complete = runtime_mixture_completion_terminal(identity=identity, instance_id="instance-b", publications=(one, two))
    authorization = authorize_runtime_mixture_disposal(instance_id="instance-b", terminal=complete, identity=identity, hub=hub)
    assert authorization["kind"] == "runtime_mixture_disposal_authorization"
    assert authorization["publication_steps"] == [1000, 2000]


def test_durable_anchor_discovers_lost_host_one_k_checkpoint_at_immutable_ref(tmp_path: Path) -> None:
    identity, hub = _identity(), FakeHub()
    root = tmp_path / "first"
    one = publish_runtime_mixture_checkpoint(
        identity=identity, checkpoint=_checkpoint(root, step=1000, identity=identity),
        artifact_root=root, publisher=hub.publish, hub=hub,
    )
    anchor = build_runtime_checkpoint_anchor(
        publication=one, identity=identity, experiment_id="runtime-mixture-70-30",
        experiment_config_sha256="5" * 64, anchor_ref="main", previous_anchor=None,
    )
    hub.write_anchor(anchor=anchor, revision="a" * 40)

    # The lost GPU has no terminal receipt or process output.  The controller
    # resolves the known ref twice, downloads it at the full commit, and then
    # fresh-readbacks the referenced immutable checkpoint.
    discovered = discover_runtime_checkpoint_anchor(
        identity=identity, experiment_id="runtime-mixture-70-30",
        experiment_config_sha256="5" * 64, anchor_ref="main", hub=hub,
        destination=tmp_path / "replacement",
    )

    assert discovered.immutable_anchor_revision == "a" * 40
    assert discovered.resume.optimizer_step == 1000
    assert discovered.resume.checkpoint_archive.read_bytes() == b"archive-1000"


def test_durable_two_k_anchor_requires_and_freshly_verifies_its_one_k_link(tmp_path: Path) -> None:
    identity, hub = _identity(), FakeHub()
    one_root = tmp_path / "one"
    one = publish_runtime_mixture_checkpoint(
        identity=identity, checkpoint=_checkpoint(one_root, step=1000, identity=identity),
        artifact_root=one_root, publisher=hub.publish, hub=hub,
    )
    first = build_runtime_checkpoint_anchor(
        publication=one, identity=identity, experiment_id="runtime-mixture-70-30",
        experiment_config_sha256="5" * 64, anchor_ref="main", previous_anchor=None,
    )
    first_sha = hub.write_anchor(anchor=first, revision="a" * 40)
    two_root = tmp_path / "two"
    two = publish_runtime_mixture_checkpoint(
        identity=identity, checkpoint=_checkpoint(two_root, step=2000, identity=identity),
        artifact_root=two_root, publisher=hub.publish, hub=hub,
    )
    second = build_runtime_checkpoint_anchor(
        publication=two, identity=identity, experiment_id="runtime-mixture-70-30",
        experiment_config_sha256="5" * 64, anchor_ref="main",
        previous_anchor={"immutable_anchor_revision": "a" * 40, "anchor_sha256": first_sha},
    )
    hub.write_anchor(anchor=second, revision="b" * 40)

    discovered = discover_runtime_checkpoint_anchor(
        identity=identity, experiment_id="runtime-mixture-70-30",
        experiment_config_sha256="5" * 64, anchor_ref="main", hub=hub,
        destination=tmp_path / "replacement",
    )
    assert discovered.resume.optimizer_step == 2000

    second["previous_anchor_sha256"] = "0" * 64
    hub.write_anchor(anchor=second, revision="c" * 40)
    with pytest.raises(ValueError, match="previous link hash"):
        discover_runtime_checkpoint_anchor(
            identity=identity, experiment_id="runtime-mixture-70-30",
            experiment_config_sha256="5" * 64, anchor_ref="main", hub=hub,
            destination=tmp_path / "tampered-replacement",
        )


@pytest.mark.parametrize("mutator", ["tamper", "missing_descriptor", "wrong_identity", "unpublished", "double_cursor"])
def test_resume_rejects_lost_or_ambiguous_state(tmp_path: Path, mutator: str) -> None:
    identity, hub = _identity(), FakeHub()
    root = tmp_path / "checkpoint"
    published = publish_runtime_mixture_checkpoint(
        identity=identity, checkpoint=_checkpoint(root, step=1000, identity=identity), artifact_root=root, publisher=hub.publish, hub=hub,
    )
    terminal = provider_interruption_terminal(
        identity=identity, instance_id="instance-a", publications=(published,),
        provider_loss={"kind": "preempted", "evidence_sha256": "8" * 64},
    )
    if mutator == "tamper":
        remote = next(path for path in hub.objects[published["immutable_revision"]] if path.endswith(".tar"))
        hub.objects[published["immutable_revision"]][remote] = b"tampered"
    elif mutator == "missing_descriptor":
        remote = next(path for path in hub.objects[published["immutable_revision"]] if path.endswith(".json"))
        del hub.objects[published["immutable_revision"]][remote]
    elif mutator == "wrong_identity":
        terminal["identity"]["mixture_id"] = "9" * 64
    elif mutator == "unpublished":
        terminal["immutable_checkpoint_publications"] = []
        terminal["resumable_checkpoint_step"] = 1000
    elif mutator == "double_cursor":
        terminal["runtime_cursor"]["global_sample_offset"] = 1
    with pytest.raises(ValueError):
        discover_runtime_mixture_resume(terminal=terminal, identity=identity, hub=hub, destination=tmp_path / "hydrate")


def test_terminals_enforce_exact_steps_and_authenticated_provider_loss(tmp_path: Path) -> None:
    identity, hub = _identity(), FakeHub()
    one = publish_runtime_mixture_checkpoint(identity=identity, checkpoint=_checkpoint(tmp_path, step=1000, identity=identity), artifact_root=tmp_path, publisher=hub.publish, hub=hub)
    with pytest.raises(ValueError, match="1000 and 2000"):
        runtime_mixture_completion_terminal(identity=identity, instance_id="x", publications=(one,))
    with pytest.raises(ValueError, match="provider loss"):
        provider_interruption_terminal(identity=identity, instance_id="x", publications=(one,), provider_loss={"kind": "trainer_error", "evidence_sha256": "8" * 64})


def test_resume_rejects_a_terminal_without_authenticated_provider_loss(tmp_path: Path) -> None:
    identity, hub = _identity(), FakeHub()
    root = tmp_path / "checkpoint"
    one = publish_runtime_mixture_checkpoint(identity=identity, checkpoint=_checkpoint(root, step=1000, identity=identity), artifact_root=root, publisher=hub.publish, hub=hub)
    terminal = provider_interruption_terminal(identity=identity, instance_id="x", publications=(one,), provider_loss={"kind": "preempted", "evidence_sha256": "8" * 64})
    terminal["provider_loss"] = None
    with pytest.raises(ValueError, match="provider loss"):
        discover_runtime_mixture_resume(terminal=terminal, identity=identity, hub=hub, destination=tmp_path / "hydrate")


def test_provider_unreachable_is_not_a_resumable_runtime_terminal(tmp_path: Path) -> None:
    identity, hub = _identity(), FakeHub()
    root = tmp_path / "checkpoint"
    one = publish_runtime_mixture_checkpoint(
        identity=identity, checkpoint=_checkpoint(root, step=1000, identity=identity),
        artifact_root=root, publisher=hub.publish, hub=hub,
    )
    with pytest.raises(ValueError, match="provider loss"):
        provider_interruption_terminal(
            identity=identity, instance_id="instance-a", publications=(one,),
            provider_loss={"kind": "provider_unreachable", "evidence_sha256": "8" * 64},
        )


def test_resume_rejects_a_terminal_outside_the_stable_receipt_schema(tmp_path: Path) -> None:
    identity, hub = _identity(), FakeHub()
    root = tmp_path / "checkpoint"
    one = publish_runtime_mixture_checkpoint(identity=identity, checkpoint=_checkpoint(root, step=1000, identity=identity), artifact_root=root, publisher=hub.publish, hub=hub)
    terminal = provider_interruption_terminal(identity=identity, instance_id="x", publications=(one,), provider_loss={"kind": "preempted", "evidence_sha256": "8" * 64})
    terminal["untrusted"] = True
    with pytest.raises(ValueError, match="schema"):
        discover_runtime_mixture_resume(terminal=terminal, identity=identity, hub=hub, destination=tmp_path / "hydrate")
