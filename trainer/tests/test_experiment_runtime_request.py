"""Verified runtime-request-set handoff for experiment training workers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import tarfile
from types import SimpleNamespace
import shlex
from pathlib import Path


def test_request_set_rejects_non_runtime_source_before_hydration(tmp_path) -> None:
    from lehome_train.groot.experiment_runtime_request import materialize_request_set
    from test_experiment_controller import _job

    job = _job(tmp_path, "a")
    try:
        materialize_request_set(job, tmp_path / "cache", hydrator=object())
    except ValueError as error:
        assert "runtime_request_set" in str(error)
    else:
        raise AssertionError("ordinary data source was accepted as a runtime request set")


def _runtime_job_and_bundle(tmp_path: Path):
    from lehome_train.groot.experiment_job import dump_experiment_job
    from lehome_train.io import canonical_json_sha256
    from test_experiment_job import _document, REV

    bundle = tmp_path / "bundle"; (bundle / "prepared").mkdir(parents=True)
    paths = {
        "prepared/experiment.json": b"{}",
        "prepared/hydrate.json": b"{}",
        "prepared/pilot.json": b"{}",
        "prepared/warmup.json": b"{}",
        "prepared/train.json": b"{}",
        "code.bundle": b"bundle",
    }
    for relative, payload in paths.items():
        (bundle / relative).write_bytes(payload)
    sha = {relative: hashlib.sha256(payload).hexdigest() for relative, payload in paths.items()}
    environment = {
        "LEHOME_EXPERIMENT_MANIFEST": "prepared/experiment.json", "LEHOME_EXPERIMENT_MANIFEST_SHA256": sha["prepared/experiment.json"],
        "LEHOME_CODE_BUNDLE": "code.bundle", "LEHOME_CODE_BUNDLE_SHA256": sha["code.bundle"], "LEHOME_CODE_REVISION": REV,
        "LEHOME_RUNTIME_HYDRATE_REQUEST": "prepared/hydrate.json", "LEHOME_RUNTIME_HYDRATE_REQUEST_SHA256": sha["prepared/hydrate.json"],
        "LEHOME_RUNTIME_PILOT_REQUEST": "prepared/pilot.json", "LEHOME_RUNTIME_PILOT_REQUEST_SHA256": sha["prepared/pilot.json"],
        "LEHOME_RUNTIME_WARMUP_REQUEST": "prepared/warmup.json", "LEHOME_RUNTIME_WARMUP_REQUEST_SHA256": sha["prepared/warmup.json"],
        "LEHOME_RUNTIME_TRAIN_REQUEST": "prepared/train.json", "LEHOME_RUNTIME_TRAIN_REQUEST_SHA256": sha["prepared/train.json"],
    }
    entries = [{"path": relative, "byte_size": len(payload), "sha256": sha[relative]} for relative, payload in sorted(paths.items())]
    tree = canonical_json_sha256(entries)
    document = _document()
    document["data_sources"] = [{"kind": "runtime_request_set", "repository": "owner/requests", "revision": REV, "prefix": "runtime/a", "manifest_sha256": "0" * 64, "tree_sha256": tree}]
    document["publication"]["prefix"] = "experiments/a"
    profile = dict(document); profile.pop("experiment_id"); profile["data_sources"] = []
    manifest = {"schema_version": 1, "kind": "lehome_runtime_request_set", "runtime_profile": profile, "runtime_profile_sha256": canonical_json_sha256(profile), "environment": environment, "result_output": "output/result.json", "files": entries, "tree_sha256": tree}
    (bundle / "bundle-manifest.json").write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    document["data_sources"][0]["manifest_sha256"] = hashlib.sha256((bundle / "bundle-manifest.json").read_bytes()).hexdigest()
    job = dump_experiment_job(tmp_path / "job.json", document)
    return job, bundle


def test_production_runner_uses_verified_request_set_and_guest_script(tmp_path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/run_lehome_experiment_worker.py"
    spec = importlib.util.spec_from_file_location("experiment_worker_cli", script); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    job, bundle = _runtime_job_and_bundle(tmp_path)
    token = tmp_path / "hf-token"; token.write_text("hf_abcdefghijklmnopqrst\n"); token.chmod(0o600)
    class Hydrator:
        def hydrate(self, _source, destination):
            shutil.copytree(bundle, destination, dirs_exist_ok=True)
            return destination
    calls = []
    def guest(argv, *, env, check):
        assert argv == ["/fake/lehome-training.sh"] and check is True
        calls.append(env)
        runtime = {}
        for line in Path(env["LEHOME_RUNTIME_ENV"]).read_text().splitlines():
            key, value = line.split("=", 1); runtime[key] = value.strip("'")
        output = Path(runtime["LEHOME_OUTPUT_ROOT"]) / "result.json"
        payload = {"immutable_checkpoint_publications": [{"optimizer_step": 500, "repository": job.publication.checkpoint_repository, "immutable_revision": "a" * 40, "remote_prefix": job.publication.prefix + "/step-500", "relative_path": "checkpoints/step-500.tar", "artifact_sha256": "d" * 64, "artifact_byte_size": 1, "descriptor_relative_path": "checkpoints/step-500.json", "descriptor_sha256": "e" * 64, "descriptor_byte_size": 1, "readback_verified": True}]}
        output.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    runner = module.ProductionRuntimeExperimentRunner(tmp_path / "cache", tmp_path / "output", token, hydrator=Hydrator(), training_script=Path("/fake/lehome-training.sh"), process_runner=guest)
    result = runner.run(job)
    assert result["publication"]["target_step"] == 500
    assert len(calls) == 1 and "LEHOME_RUNTIME_ENV" in calls[0]


def test_production_runner_rejects_tampered_request_set_before_guest_script(tmp_path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/run_lehome_experiment_worker.py"
    spec = importlib.util.spec_from_file_location("experiment_worker_cli_tamper", script); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    job, bundle = _runtime_job_and_bundle(tmp_path)
    (bundle / "prepared/train.json").write_text('{"tampered":true}')
    token = tmp_path / "hf-token"; token.write_text("hf_abcdefghijklmnopqrst\n"); token.chmod(0o600)
    class Hydrator:
        def hydrate(self, _source, destination):
            shutil.copytree(bundle, destination, dirs_exist_ok=True)
            return destination
    called = []
    runner = module.ProductionRuntimeExperimentRunner(tmp_path / "cache", tmp_path / "output", token, hydrator=Hydrator(), training_script=Path("/fake/lehome-training.sh"), process_runner=lambda *_args, **_kwargs: called.append(True))
    try:
        runner.run(job)
    except ValueError as error:
        assert "digest" in str(error)
    else:
        raise AssertionError("tampered request set reached the guest script")
    assert called == []


def test_compatibility_overlay_binds_only_the_promotable_job_coordinates(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_runtime_request import (
        build_sweep_train_overlay,
        runtime_compatibility_profile_document,
    )

    job, bundle = _runtime_job_and_bundle(tmp_path)
    profile = runtime_compatibility_profile_document(job)
    overlay = build_sweep_train_overlay(
        job,
        workspace=bundle,
        base_train_request=bundle / "prepared/train.json",
        compatibility_profile_sha256=__import__("hashlib").sha256(
            json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    value = json.loads(overlay.read_text(encoding="utf-8"))

    assert value["kind"] == "lehome_sweep_train_request_overlay"
    assert value["job_digest"] == job.experiment_id
    assert value["parent_checkpoint"] == dict(job.parent_checkpoint)
    assert value["training"] == {
        "seed": job.training.seed,
        "target_step": job.training.target_step,
        "save_steps": 500,
    }
    assert value["base_train_request"] == "prepared/train.json"


def test_v2_request_set_accepts_only_the_reusable_compatibility_projection(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_job import dump_experiment_job
    from lehome_train.groot.experiment_runtime_request import (
        materialize_request_set,
        runtime_compatibility_profile_document,
        runtime_compatibility_profile_sha256,
    )

    job, bundle = _runtime_job_and_bundle(tmp_path)
    manifest_path = bundle / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("runtime_profile")
    manifest.pop("runtime_profile_sha256")
    manifest["runtime_compatibility_profile"] = runtime_compatibility_profile_document(job)
    manifest["runtime_compatibility_profile_sha256"] = runtime_compatibility_profile_sha256(job)
    manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    document = json.loads((tmp_path / "job.json").read_text(encoding="utf-8"))
    document["data_sources"][0]["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    job = dump_experiment_job(tmp_path / "v2-job.json", document)

    class Hydrator:
        def hydrate(self, _source, destination):
            shutil.copytree(bundle, destination, dirs_exist_ok=True)
            return destination

    request_set = materialize_request_set(job, tmp_path / "cache", hydrator=Hydrator())
    assert request_set.compatibility_profile_sha256 == runtime_compatibility_profile_sha256(job)


def test_v2_parent_request_set_accepts_controller_generated_children_and_rejects_contract_drift(
    tmp_path: Path,
) -> None:
    """Children inherit the request set, not the parent's mutable job identity."""
    from lehome_train.groot.experiment_job import dump_experiment_job
    from lehome_train.groot.experiment_manifest import batch64_quotas
    from lehome_train.groot.experiment_runtime_request import (
        materialize_request_set,
        runtime_compatibility_profile_document,
        runtime_compatibility_profile_sha256,
    )

    job, bundle = _runtime_job_and_bundle(tmp_path)
    parent_document = json.loads((tmp_path / "job.json").read_text(encoding="utf-8"))
    # A real sweep has non-runtime data bindings.  Those, rather than an arm
    # label, keep data-distinct arms from sharing an incompatible request set.
    parent_document["data_sources"].insert(0, {
        "kind": "bc", "repository": "owner/data", "revision": "b" * 40,
        "prefix": "bc/full", "manifest_sha256": "a" * 64, "tree_sha256": "c" * 64,
    })
    parent = dump_experiment_job(tmp_path / "parent.json", parent_document)

    manifest_path = bundle / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("runtime_profile")
    manifest.pop("runtime_profile_sha256")
    manifest["runtime_compatibility_profile"] = runtime_compatibility_profile_document(parent)
    manifest["runtime_compatibility_profile_sha256"] = runtime_compatibility_profile_sha256(parent)
    manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    parent_document["data_sources"][1]["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    parent = dump_experiment_job(tmp_path / "parent-v2.json", parent_document)

    class Hydrator:
        def hydrate(self, _source, destination):
            shutil.copytree(bundle, destination, dirs_exist_ok=True)
            return destination

    def child_document(*, kind: str, arm: str, seed: int, target_step: int) -> dict[str, object]:
        child = json.loads(json.dumps(dict(parent.raw)))
        child["arm"] = arm
        child["training"]["seed"] = seed
        child["training"]["target_step"] = target_step
        child["publication"]["prefix"] = f"experiments/{arm}"
        # Every rung remains paired against the same original 12K baseline;
        # orchestration coordinates may change, but baseline identity may not.
        child["evaluation"]["policy_digest"] = parent.evaluation.policy_digest
        child["dependencies"] = [parent.experiment_id]
        child["admission"] = {"kind": kind, "source_experiment_id": parent.experiment_id}
        if kind == "continuation":
            child["parent_checkpoint"] = {
                "repository": "owner/checkpoints", "revision": "d" * 40,
                "subpath": "experiments/parent/step-500", "artifact_sha256": "e" * 64,
                "receipt_sha256": "f" * 64,
            }
        return child

    # A deterministic seed repeat and the controller's 1K/2K continuations
    # differ in every excluded orchestration coordinate but must consume the
    # same verified parent request set.
    descendants = (
        child_document(kind="seed_repeat", arm="repeat-2", seed=2, target_step=500),
        child_document(kind="continuation", arm="promoted-1k", seed=3, target_step=1000),
        child_document(kind="continuation", arm="promoted-2k", seed=4, target_step=2000),
    )
    for index, document in enumerate(descendants):
        child = dump_experiment_job(tmp_path / f"child-{index}.json", document)
        request_set = materialize_request_set(child, tmp_path / f"cache-{index}", hydrator=Hydrator())
        assert request_set.compatibility_profile_sha256 == runtime_compatibility_profile_sha256(parent)

    def assert_rejected(name: str, mutate) -> None:
        document = child_document(kind="seed_repeat", arm=f"other-{name}", seed=5, target_step=500)
        mutate(document)
        drifted = dump_experiment_job(tmp_path / f"drift-{name}.json", document)
        try:
            materialize_request_set(drifted, tmp_path / f"cache-drift-{name}", hydrator=Hydrator())
        except ValueError as error:
            assert "manifest is invalid" in str(error)
        else:
            raise AssertionError(f"{name} drift reused the parent request set")

    # The arm string itself is intentionally unbound.  A genuinely different
    # arm is rejected through its data/mixture contract, while matrix and
    # trainer drift are each independently bound.
    assert_rejected("other-arm-mixture", lambda document: document.update({
        "mixture": {
            "bc_percent": 95, "added_percent": 5,
            "batch64_quotas": batch64_quotas({"bc": 95, "rollout": 5, "dagger": 0}),
            "sampling_strategy": "unweighted",
        },
    }))
    assert_rejected("data", lambda document: document["data_sources"][0].update({"tree_sha256": "1" * 64}))
    assert_rejected("matrix", lambda document: document["evaluation"].update({"matrix_id": "unseen80"}))
    assert_rejected("trainer", lambda document: document["trainer"].update({"code_revision": "2" * 40}))


def test_sweep_runtime_request_is_canonical_and_job_overlay_bound(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_runtime_request import (
        build_sweep_runtime_request,
        build_sweep_train_overlay,
        runtime_compatibility_profile_sha256,
    )

    job, bundle = _runtime_job_and_bundle(tmp_path)
    base_launch = bundle / "prepared" / "launch.json"
    base_experiment = bundle / "prepared" / "experiment-config.json"
    base_train = bundle / "prepared" / "train.json"
    base_launch.write_bytes(json.dumps({
        "base_model_path": "/cache/parent", "max_steps": 2000,
        "save_steps": 500, "output_dir": "/output", "experiment_name": "base",
    }, sort_keys=True, separators=(",", ":")).encode())
    base_experiment.write_bytes(json.dumps({"sample_presentations": 128000}, sort_keys=True, separators=(",", ":")).encode())
    base_train.write_bytes(json.dumps({
        "schema_version": 1, "command": "runtime-mixture-train",
        "arguments": {"launch_config": "/prepared/config/launch.json", "experiment_config": "/prepared/config/experiment.json"},
    }, sort_keys=True, separators=(",", ":")).encode())
    overlay = build_sweep_train_overlay(
        job, workspace=bundle, base_train_request=base_train,
        compatibility_profile_sha256=runtime_compatibility_profile_sha256(job),
    )
    generated = build_sweep_runtime_request(
        job, workspace=bundle, base_train_request=base_train,
        base_launch_config=base_launch, base_experiment_config=base_experiment,
        overlay=overlay, promoted_parent=None,
    )
    launch = json.loads(generated.launch_config.read_text(encoding="utf-8"))
    train = json.loads(generated.train_request.read_text(encoding="utf-8"))
    binding = json.loads(generated.binding.read_text(encoding="utf-8"))
    assert launch["max_steps"] == 500 and launch["runtime_sweep_profile"]["training"]["target_step"] == 500
    assert train["arguments"]["result_output"] == f"/output/sweep/{job.experiment_id}/runtime-train-result.json"
    assert binding["experiment_id"] == job.experiment_id and binding["overlay_sha256"] == hashlib.sha256(overlay.read_bytes()).hexdigest()


def test_v2_runner_passes_canonical_dynamic_sweep_request_to_fake_guest(tmp_path: Path) -> None:
    """CPU integration: request set -> runner -> guest env uses sweep bytes."""
    script = Path(__file__).resolve().parents[2] / "scripts/run_lehome_experiment_worker.py"
    spec = importlib.util.spec_from_file_location("experiment_worker_cli_v2", script); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    from lehome_train.groot.experiment_job import dump_experiment_job
    from lehome_train.groot.experiment_runtime_request import (
        runtime_compatibility_profile_document, runtime_compatibility_profile_sha256,
    )

    job, bundle = _runtime_job_and_bundle(tmp_path)
    (bundle / "prepared" / "config").mkdir()
    launch = bundle / "prepared" / "config" / "launch.json"
    experiment = bundle / "prepared" / "config" / "experiment.json"
    launch.write_bytes(json.dumps({"base_model_path": "/cache/parent", "max_steps": 2000, "save_steps": 500, "output_dir": "/output", "experiment_name": "base"}, sort_keys=True, separators=(",", ":")).encode())
    experiment.write_bytes(json.dumps({"sample_presentations": 128000}, sort_keys=True, separators=(",", ":")).encode())
    train = bundle / "prepared" / "train.json"
    train.write_bytes(json.dumps({"schema_version": 1, "command": "runtime-mixture-train", "arguments": {}}, sort_keys=True, separators=(",", ":")).encode())
    manifest_path = bundle / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_files = sorted(path for path in bundle.rglob("*") if path.is_file() and path.name != "bundle-manifest.json")
    manifest["files"] = [{"path": path.relative_to(bundle).as_posix(), "byte_size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in all_files]
    manifest["tree_sha256"] = __import__("lehome_train.io", fromlist=["canonical_json_sha256"]).canonical_json_sha256(manifest["files"])
    manifest.pop("runtime_profile"); manifest.pop("runtime_profile_sha256")
    manifest["runtime_compatibility_profile"] = runtime_compatibility_profile_document(job)
    manifest["runtime_compatibility_profile_sha256"] = runtime_compatibility_profile_sha256(job)
    manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    document = json.loads((tmp_path / "job.json").read_text(encoding="utf-8"))
    document["data_sources"][0]["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    document["data_sources"][0]["tree_sha256"] = manifest["tree_sha256"]
    job = dump_experiment_job(tmp_path / "v2-runner-job.json", document)
    token = tmp_path / "hf-token"; token.write_text("hf_abcdefghijklmnopqrst\n"); token.chmod(0o600)
    class Hydrator:
        def hydrate(self, _source, destination):
            shutil.copytree(bundle, destination, dirs_exist_ok=True); return destination
    observed: dict[str, str] = {}
    def guest(_argv, *, env, check):
        assert check is True
        runtime = {}
        for line in Path(env["LEHOME_RUNTIME_ENV"]).read_text().splitlines():
            key, value = line.split("=", 1); runtime[key] = shlex.split(value)[0]
        assert "LEHOME_SWEEP_TRAIN_REQUEST" in runtime
        dynamic = json.loads(Path(runtime["LEHOME_SWEEP_TRAIN_REQUEST"]).read_text(encoding="utf-8"))
        observed["target"] = str(dynamic["arguments"]["result_output"])
        output = Path(runtime["LEHOME_OUTPUT_ROOT"]) / "sweep" / job.experiment_id / "runtime-train-result.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"immutable_checkpoint_publications": [{"optimizer_step": 500, "repository": job.publication.checkpoint_repository, "immutable_revision": "a" * 40, "remote_prefix": job.publication.prefix + "/step-500", "relative_path": "checkpoints/step-500.tar", "artifact_sha256": "d" * 64, "artifact_byte_size": 1, "descriptor_relative_path": "checkpoints/step-500.json", "descriptor_sha256": "e" * 64, "descriptor_byte_size": 1, "readback_verified": True}]}
        output.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    runner = module.ProductionRuntimeExperimentRunner(tmp_path / "cache", tmp_path / "output", token, hydrator=Hydrator(), training_script=Path("/fake/lehome-training.sh"), process_runner=guest)
    result = runner.run(job)
    assert result["publication"]["schema_version"] == 2
    assert observed["target"] == f"/output/sweep/{job.experiment_id}/runtime-train-result.json"


def test_promoted_parent_hydration_fails_closed_without_exact_archive_and_descriptor(
    tmp_path: Path, monkeypatch,
) -> None:
    from types import SimpleNamespace
    from lehome_train.groot.experiment_runtime_request import verify_promoted_parent_hydration

    job, _bundle = _runtime_job_and_bundle(tmp_path)
    archive = tmp_path / "parent.tar"; archive.write_bytes(b"immutable-parent")
    descriptor = tmp_path / "parent.json"; descriptor.write_bytes(b"descriptor")
    publication = {
        "schema_version": 2,
        "experiment_id": "f" * 64,
        "job_digest": "f" * 64,
        "target_step": 500,
        "repository": job.parent_checkpoint["repository"],
        "immutable_revision": job.parent_checkpoint["revision"],
        "remote_prefix": "experiments/parent-500",
        "relative_path": "checkpoints/step-500.tar",
        "artifact_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "artifact_byte_size": archive.stat().st_size,
        "descriptor_relative_path": "checkpoints/step-500.json",
        "descriptor_sha256": hashlib.sha256(descriptor.read_bytes()).hexdigest(),
        "descriptor_byte_size": descriptor.stat().st_size,
        "receipt_sha256": "d" * 64,
        "readback_verified": True,
    }
    promoted = SimpleNamespace(
        parent_checkpoint={
            "repository": publication["repository"],
            "revision": publication["immutable_revision"],
            "subpath": publication["remote_prefix"],
            "artifact_sha256": publication["artifact_sha256"],
            "receipt_sha256": publication["receipt_sha256"],
        },
        training=SimpleNamespace(target_step=1000),
    )
    monkeypatch.setattr(
        "lehome_train.groot.experiment_runtime_request.load_checkpoint_descriptor",
        lambda _path: SimpleNamespace(record=SimpleNamespace(
            optimizer_step=500, resumable=True,
            artifact=SimpleNamespace(
                sha256=publication["artifact_sha256"], byte_size=archive.stat().st_size,
            ),
        )),
    )

    try:
        verify_promoted_parent_hydration(promoted, publication=publication, archive=None, descriptor_path=None)
    except ValueError as error:
        assert "archive" in str(error)
    else:
        raise AssertionError("unhydrated promoted parent was accepted")
    verify_promoted_parent_hydration(
        promoted, publication=publication, archive=archive, descriptor_path=descriptor,
    )


def test_promoted_parent_hydrator_fresh_lists_verifies_and_atomically_caches(
    tmp_path: Path, monkeypatch,
) -> None:
    from lehome_train.groot.experiment_runtime_request import hydrate_promoted_parent

    archive = tmp_path / "published.tar"
    source = tmp_path / "source" / "run" / "checkpoint-500"
    source.mkdir(parents=True)
    (source / "model.safetensors").write_bytes(b"published-model")
    (source / "trainer_state.json").write_text('{"global_step":500}', encoding="utf-8")
    with tarfile.open(archive, "w") as bundle:
        bundle.add(source.parent, arcname="run")
    descriptor = tmp_path / "published.json"; descriptor.write_bytes(b"descriptor")
    publication = {
        "schema_version": 2, "experiment_id": "f" * 64, "job_digest": "f" * 64,
        "target_step": 500, "repository": "owner/models",
        "immutable_revision": "a" * 40, "remote_prefix": "experiments/a-500",
        "relative_path": "checkpoints/step-500.tar",
        "artifact_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "artifact_byte_size": archive.stat().st_size,
        "descriptor_relative_path": "checkpoints/step-500.json",
        "descriptor_sha256": hashlib.sha256(descriptor.read_bytes()).hexdigest(),
        "descriptor_byte_size": descriptor.stat().st_size,
        "receipt_sha256": "d" * 64,
        "readback_verified": True,
    }
    job = SimpleNamespace(
        parent_checkpoint={
            "repository": publication["repository"],
            "revision": publication["immutable_revision"],
            "subpath": publication["remote_prefix"],
            "artifact_sha256": publication["artifact_sha256"],
            "receipt_sha256": publication["receipt_sha256"],
        },
        training=SimpleNamespace(target_step=1000),
    )
    monkeypatch.setattr(
        "lehome_train.groot.experiment_runtime_request.load_checkpoint_descriptor",
        lambda _path: SimpleNamespace(record=SimpleNamespace(
            optimizer_step=500, resumable=True,
            artifact=SimpleNamespace(
                sha256=publication["artifact_sha256"], byte_size=archive.stat().st_size,
            ),
        )),
    )

    class Hub:
        calls = 0
        def list_tree(self, **_kwargs):
            self.calls += 1
            return [
                SimpleNamespace(relative_path="experiments/a-500/checkpoints/step-500.tar", entry_type="file"),
                SimpleNamespace(relative_path="experiments/a-500/checkpoints/step-500.json", entry_type="file"),
            ]
        def download_files(self, *, destination, relative_paths, **_kwargs):
            for relative in relative_paths:
                target = destination / relative; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(archive if relative.endswith(".tar") else descriptor, target)

    hub = Hub()
    materialized = hydrate_promoted_parent(
        job, publication=publication, cache_root=tmp_path / "cache", hub=hub,
    )
    assert materialized is not None
    assert materialized.cursor["optimizer_step"] == 500
    assert (materialized.checkpoint_path / "model.safetensors").read_bytes() == b"published-model"
    again = hydrate_promoted_parent(
        job, publication=publication, cache_root=tmp_path / "cache", hub=hub,
    )
    assert again is not None and hub.calls == 1


def test_controller_lease_client_preserves_full_v2_parent_publication(tmp_path: Path) -> None:
    """The worker must not reduce the controller's canonical parent envelope."""
    script = Path(__file__).resolve().parents[2] / "scripts/run_lehome_experiment_worker.py"
    spec = importlib.util.spec_from_file_location("experiment_worker_cli_lease", script); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    from lehome_train.groot.experiment_job import dump_experiment_job

    job, _bundle = _runtime_job_and_bundle(tmp_path)
    document = json.loads((tmp_path / "job.json").read_text(encoding="utf-8"))
    document["training"]["target_step"] = 1000
    publication = {
        "schema_version": 2, "experiment_id": "f" * 64, "job_digest": "f" * 64,
        "target_step": 500, "repository": "owner/models", "immutable_revision": "a" * 40,
        "remote_prefix": "experiments/parent-500", "relative_path": "checkpoints/step-500.tar",
        "artifact_sha256": "b" * 64, "artifact_byte_size": 1,
        "descriptor_relative_path": "checkpoints/step-500.json", "descriptor_sha256": "c" * 64,
        "descriptor_byte_size": 1, "receipt_sha256": "d" * 64, "readback_verified": True,
    }
    document["parent_checkpoint"] = {
        "repository": publication["repository"], "revision": publication["immutable_revision"],
        "subpath": publication["remote_prefix"], "artifact_sha256": publication["artifact_sha256"],
        "receipt_sha256": publication["receipt_sha256"],
    }
    document["admission"] = {
        "kind": "continuation",
        "source_experiment_id": publication["experiment_id"],
    }
    job = dump_experiment_job(tmp_path / "child.json", document)
    client = object.__new__(module.HttpControllerClient)
    client.manifest_set_sha256 = "0" * 64
    client._post = lambda _endpoint, _payload: {
        "lease": True, "lease_id": "lease", "experiment_id": job.experiment_id,
        "worker_id": "worker", "expires_ns": 1, "job": dict(job.raw),
        "publication": None, "parent_publication": publication,
    }
    lease = client.lease_next("worker", "training", now_ns=0, lease_ns=60)
    assert lease is not None
    assert dict(lease.parent_publication) == publication
