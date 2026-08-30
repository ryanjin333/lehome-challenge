from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import Event, Lock
from concurrent.futures import ThreadPoolExecutor

import pytest

from lehome_train.b1k.bootstrap import BootstrapWorkflow, BucketNotFound, HfHubAdapter, ModelSmokeObjectNotFound, ProductionHubAccess, WorkspacePaths, bootstrap_remote, dataset_snapshot_patterns, preflight_remote_access, require_hardware, read_hf_token
from lehome_train.b1k.dataset import MaterializedTrainingManifest
from lehome_train.constants import BEHAVIOR_1K_CHECKPOINT_BUCKET, BEHAVIOR_1K_DATASET_REPOSITORY, BEHAVIOR_1K_DATASET_REVISION, BEHAVIOR_1K_FINAL_MODEL_REPOSITORY, COSMOS_REPOSITORY, COSMOS_REVISION, MODEL_REVISION
from lehome_train.groot.model_snapshot import BASE_MODEL_REPOSITORY
from lehome_train.io import canonical_json_sha256
from lehome_train.b1k.snapshot_integrity import build_remote_manifest


@pytest.fixture(autouse=True)
def _local_smoke_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lehome_train.b1k.bootstrap._SMOKE_LOCAL_ROOT", tmp_path)


def test_workspace_paths_are_fixed_to_one_workspace_disk() -> None:
    paths = WorkspacePaths.from_root("/workspace", run_id="b1k-run-001")
    assert paths.dataset == Path("/workspace/data/b1k")
    assert paths.derived_model == Path("/workspace/models/groot")
    assert paths.output == Path("/workspace/outputs/b1k-run-001")
    assert paths.checkpoints == Path("/workspace/checkpoints")


def test_hardware_gate_requires_supported_gpu_and_1point5tb() -> None:
    smi = "NVIDIA RTX PRO 6000 Blackwell Server, 98304\n"
    require_hardware(smi, free_bytes=1_500_000_000_000)
    with pytest.raises(ValueError): require_hardware("NVIDIA A6000, 49152", free_bytes=2_000_000_000_000)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_hardware_gate_accepts_each_template_gpu_count(count: int) -> None:
    require_hardware("NVIDIA RTX PRO 6000 Blackwell Server, 98304\n" * count, free_bytes=2_000_000_000_000)


def test_hardware_gate_rejects_more_than_four_template_gpus() -> None:
    with pytest.raises(ValueError, match="one to four"):
        require_hardware("NVIDIA RTX PRO 6000 Blackwell Server, 98304\n" * 5, free_bytes=2_000_000_000_000)


@pytest.mark.parametrize("name", ["NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"])
def test_hardware_gate_parses_real_nvidia_smi_csv_names(name: str) -> None:
    require_hardware(f"{name}, 98304\n", free_bytes=1_500_000_000_000)


def test_token_is_regular_private_current_owner_and_never_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token = tmp_path / "token"; token.write_text("hf_test"); token.chmod(0o600)
    monkeypatch.setenv("HF_TOKEN", "forbidden")
    assert read_hf_token(token) == "hf_test"
    token.chmod(0o644)
    with pytest.raises(ValueError): read_hf_token(token)


def test_token_rejects_symlink_and_wrong_runtime_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token = tmp_path / "token"; token.write_text("hf_test"); token.chmod(0o600)
    link = tmp_path / "link"; link.symlink_to(token)
    with pytest.raises(ValueError, match="unsafe"):
        read_hf_token(link)
    monkeypatch.setattr("lehome_train.b1k.bootstrap.os.getuid", lambda: token.stat().st_uid + 1)
    with pytest.raises(ValueError, match="unsafe"):
        read_hf_token(token)


def test_token_requires_exact_0600_mode_without_special_bits(tmp_path: Path) -> None:
    token = tmp_path / "token"; token.write_text("hf_test"); token.chmod(0o4600)
    with pytest.raises(ValueError, match="unsafe"):
        read_hf_token(token)


def test_dataset_snapshot_patterns_are_metadata_first_then_rgb_only_without_depth() -> None:
    metadata, full = dataset_snapshot_patterns()
    assert metadata == ("meta/**", "annotations/skill_summary.csv", "annotations/skill_type_summary.csv")
    assert "data/**" in full and "annotations/**" in full
    assert len([item for item in full if item.startswith("videos/observation.rgb.")]) == 3
    assert all("depth" not in item for item in full)


def test_bootstrap_downloads_metadata_before_selection_then_full_rgb_snapshot(tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []
    payload = b"metadata"

    def snapshot(_repo: str, _revision: str, local: Path, patterns: tuple[str, ...], _token: str) -> None:
        calls.append(("snapshot", (local, tuple(patterns))))
        local.mkdir(parents=True, exist_ok=True); (local / "meta").mkdir(exist_ok=True)
        (local / "meta/info.json").write_bytes(payload)

    workflow = BootstrapWorkflow(
        snapshot=snapshot,
        remote_manifest=lambda repository, revision, patterns, _token: build_remote_manifest(repository=repository, revision=revision, resolved_revision=revision, entries=({"path": "meta/info.json", "size": len(payload), "blob_id": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(), "lfs": None},), allow_patterns=patterns),
        build_selection=lambda local: calls.append(("selection", local)), materialize=lambda local, **_kwargs: calls.append(("materialize", local)),
    )
    destination = tmp_path / "dataset"
    workflow.dataset(local_dir=destination, token="in-memory")
    staging = destination.with_name(".dataset.incomplete")
    assert calls == [("snapshot", (staging, ("meta/**", "annotations/skill_summary.csv", "annotations/skill_type_summary.csv"))), ("selection", staging), ("snapshot", (staging, dataset_snapshot_patterns()[1])), ("selection", staging), ("materialize", staging)]
    assert (destination / ".b1k-snapshot-receipt.json").is_file()


def test_dataset_workflow_serializes_same_destination_and_reuses_the_promoted_snapshot(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []
    calls_lock = Lock()
    payload = b"metadata"

    def snapshot(_repo: str, _revision: str, local: Path, patterns: tuple[str, ...], _token: str) -> None:
        with calls_lock:
            calls.append(tuple(patterns))
        local.mkdir(parents=True, exist_ok=True); (local / "meta").mkdir(exist_ok=True)
        (local / "meta/info.json").write_bytes(payload)

    workflow = BootstrapWorkflow(
        snapshot=snapshot,
        remote_manifest=lambda repository, revision, patterns, _token: build_remote_manifest(repository=repository, revision=revision, resolved_revision=revision, entries=({"path": "meta/info.json", "size": len(payload), "blob_id": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(), "lfs": None},), allow_patterns=patterns),
        build_selection=lambda _local: {"selected": ["episode"]}, materialize=lambda _local, **_kwargs: {"materialized": ["episode"]},
    )
    destination = tmp_path / "dataset"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(workflow.dataset, local_dir=destination, token="in-memory") for _ in range(2)]
        for future in futures:
            assert future.result(timeout=5) is None
    assert calls == [dataset_snapshot_patterns()[0], dataset_snapshot_patterns()[1]]
    assert (destination / ".b1k-snapshot-receipt.json").is_file()


def test_model_snapshot_refuses_an_empty_download_without_the_authoritative_remote_payload(tmp_path: Path) -> None:
    from lehome_train.b1k.bootstrap import _ensure_model_snapshot

    expected = b"complete Cosmos weights"

    class EmptyCosmosHub:
        def snapshot_download(self, _repository: str, *, revision: str, local_dir: Path, allow_patterns: tuple[str, ...] | None, token: str) -> object:
            assert revision == COSMOS_REVISION and allow_patterns is None and token == "in-memory"
            local_dir.mkdir(parents=True, exist_ok=True); (local_dir / "cosmos.bin").write_bytes(b"")
            return local_dir

        def remote_manifest(self, repository: str, *, revision: str, allow_patterns: tuple[str, ...] | None, token: str):
            return build_remote_manifest(
                repository=repository,
                revision=revision,
                resolved_revision=revision,
                allow_patterns=allow_patterns,
                entries=({
                    "path": "cosmos.bin", "size": len(expected),
                    "blob_id": hashlib.sha1(f"blob {len(expected)}\0".encode() + expected).hexdigest(),
                    "lfs": {"size": len(expected), "sha256": hashlib.sha256(expected).hexdigest()},
                },),
            )

    with pytest.raises(ValueError, match="size|identity|remote manifest"):
        _ensure_model_snapshot(
            hub=EmptyCosmosHub(), repository=COSMOS_REPOSITORY, revision=COSMOS_REVISION,
            destination=tmp_path / "cosmos", token="in-memory",
        )
    assert not (tmp_path / "cosmos").exists()


class _Hub:
    def __init__(self, *, bucket: object | None = None) -> None:
        self.calls: list[str] = []; self.bucket = bucket; self.model_objects: dict[str, bytes] = {}; self.bucket_objects: dict[str, bytes] = {}
    def repo_info(self, repo: str, *, revision: str | None, token: str) -> object:
        self.calls.append(f"repo:{repo}:{revision}"); return {"private": repo == BEHAVIOR_1K_FINAL_MODEL_REPOSITORY, "sha": revision}
    def bucket_info(self, _bucket: str, *, token: str) -> object:
        self.calls.append("bucket")
        if self.bucket is None: raise BucketNotFound("missing")
        return self.bucket
    def create_bucket(self, _bucket: str, *, private: bool, token: str) -> object:
        self.calls.append("create"); self.bucket = {"private": private}; return self.bucket
    def snapshot_download(self, *_args: object, **_kwargs: object) -> object: pytest.fail("downloads must not start in access test")
    def upload_model_file(self, _repo: str, key: str, data: bytes, *, token: str) -> None:
        self.calls.append(f"model-upload:{key}"); self.model_objects[key] = data
    def download_model_file(self, _repo: str, key: str, *, token: str) -> bytes:
        self.calls.append(f"model-download:{key}"); return self.model_objects[key]
    def delete_model_file(self, _repo: str, key: str, *, token: str) -> None:
        self.calls.append(f"model-delete:{key}"); self.model_objects.pop(key, None)
    def list_model_files(self, _repo: str, prefix: str = "", *, token: str) -> tuple[str, ...]:
        self.calls.append(f"model-list:{prefix}"); return tuple(path for path in self.model_objects if path.startswith(prefix))
    def upload_bucket_file(self, _bucket: str, source: Path, key: str, *, token: str) -> None:
        self.calls.append(f"bucket-upload:{key}"); self.bucket_objects[key] = source.read_bytes()
    def download_bucket_file(self, _bucket: str, key: str, destination: Path, *, token: str) -> None:
        self.calls.append(f"bucket-download:{key}"); destination.write_bytes(self.bucket_objects[key])
    def delete_bucket_file(self, _bucket: str, key: str, *, token: str) -> None:
        self.calls.append(f"bucket-delete:{key}"); self.bucket_objects.pop(key, None)
    def list_bucket_files(self, _bucket: str, prefix: str, *, token: str) -> tuple[str, ...]:
        self.calls.append(f"bucket-list:{prefix}"); return tuple(path for path in self.bucket_objects if path.startswith(prefix))


def test_remote_access_checks_all_repositories_before_any_download_and_creates_bucket_only_when_enabled() -> None:
    hub = _Hub(bucket=None)
    with pytest.raises(ValueError, match="bucket"): preflight_remote_access("memory-token", "0", hub=hub)
    assert not any(call.startswith("snapshot") for call in hub.calls)
    hub = _Hub(bucket=None); preflight_remote_access("memory-token", "1", hub=hub)
    assert hub.calls[4:6] == ["bucket", "create"]


def test_remote_access_probes_b1k_private_outputs_with_exact_readback_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lehome_train.b1k.bootstrap._SMOKE_LOCAL_ROOT", tmp_path)
    hub = _Hub(bucket={"private": True})
    preflight_remote_access("memory-token", "0", hub=hub)
    model_keys = [call.split(":", 1)[1] for call in hub.calls if call.startswith(("model-upload:", "model-download:", "model-delete:"))]
    bucket_keys = [call.split(":", 1)[1] for call in hub.calls if call.startswith(("bucket-upload:", "bucket-download:", "bucket-delete:"))]
    assert len(set(model_keys)) == 1 and len(set(bucket_keys)) == 1
    assert model_keys[0].startswith("smoke/") and bucket_keys[0].startswith("smoke/")
    assert model_keys[0] != bucket_keys[0]
    assert hub.model_objects == {} and hub.bucket_objects == {}
    assert BEHAVIOR_1K_FINAL_MODEL_REPOSITORY in hub.calls[3]
    assert BEHAVIOR_1K_CHECKPOINT_BUCKET == "ryanjin333/behavior1k-groot-n17-checkpoints"


@pytest.mark.parametrize("failure", ["model-readback", "bucket-readback"])
def test_remote_access_probe_cleanup_runs_after_readback_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    monkeypatch.setattr("lehome_train.b1k.bootstrap._SMOKE_LOCAL_ROOT", tmp_path)

    class BrokenReadbackHub(_Hub):
        def download_model_file(self, repository: str, key: str, *, token: str) -> bytes:
            value = super().download_model_file(repository, key, token=token)
            return b"wrong" if failure == "model-readback" else value
        def download_bucket_file(self, bucket: str, key: str, destination: Path, *, token: str) -> None:
            super().download_bucket_file(bucket, key, destination, token=token)
            if failure == "bucket-readback": destination.write_bytes(b"wrong")

    hub = BrokenReadbackHub(bucket={"private": True})
    with pytest.raises(ValueError, match="smoke readback"):
        preflight_remote_access("memory-token", "0", hub=hub)
    if failure == "model-readback":
        assert any(call.startswith("model-delete:smoke/") for call in hub.calls)
        assert hub.model_objects == {}
    else:
        assert any(call.startswith("bucket-delete:smoke/") for call in hub.calls)
        assert hub.bucket_objects == {}


@pytest.mark.parametrize("failure", ["model-upload", "bucket-upload"])
def test_remote_access_cleanup_runs_after_an_ambiguous_upload_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    monkeypatch.setattr("lehome_train.b1k.bootstrap._SMOKE_LOCAL_ROOT", tmp_path)

    class AmbiguousUploadHub(_Hub):
        def upload_model_file(self, repository: str, key: str, data: bytes, *, token: str) -> None:
            super().upload_model_file(repository, key, data, token=token)
            if failure == "model-upload": raise OSError("lost model response")
        def upload_bucket_file(self, bucket: str, source: Path, key: str, *, token: str) -> None:
            super().upload_bucket_file(bucket, source, key, token=token)
            if failure == "bucket-upload": raise OSError("lost bucket response")

    hub = AmbiguousUploadHub(bucket={"private": True})
    with pytest.raises(OSError, match="lost"):
        preflight_remote_access("memory-token", "0", hub=hub)
    if failure == "model-upload":
        assert any(call.startswith("model-delete:smoke/") for call in hub.calls)
        assert hub.model_objects == {}
    else:
        assert any(call.startswith("bucket-delete:smoke/") for call in hub.calls)
        assert hub.bucket_objects == {}


@pytest.mark.parametrize("failure", ["model", "bucket"])
def test_remote_access_tolerates_typed_not_found_cleanup_after_prewrite_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    monkeypatch.setattr("lehome_train.b1k.bootstrap._SMOKE_LOCAL_ROOT", tmp_path)

    class PrewriteFailureHub(_Hub):
        def upload_model_file(self, repository: str, key: str, data: bytes, *, token: str) -> None:
            if failure == "model": raise OSError("prewrite model failure")
            super().upload_model_file(repository, key, data, token=token)
        def delete_model_file(self, repository: str, key: str, *, token: str) -> None:
            if failure == "model":
                self.calls.append(f"model-delete:{key}")
                raise ModelSmokeObjectNotFound("missing exact model key")
            super().delete_model_file(repository, key, token=token)
        def upload_bucket_file(self, bucket: str, source: Path, key: str, *, token: str) -> None:
            if failure == "bucket": raise OSError("prewrite bucket failure")
            super().upload_bucket_file(bucket, source, key, token=token)
        def delete_bucket_file(self, bucket: str, key: str, *, token: str) -> None:
            if failure == "bucket":
                self.calls.append(f"bucket-delete:{key}")
                raise BucketNotFound("missing exact bucket key")
            super().delete_bucket_file(bucket, key, token=token)

    hub = PrewriteFailureHub(bucket={"private": True})
    with pytest.raises(OSError, match="prewrite"):
        preflight_remote_access("memory-token", "0", hub=hub)
    expected = "model-delete:smoke/" if failure == "model" else "bucket-delete:smoke/"
    assert any(call.startswith(expected) for call in hub.calls)


def test_remote_access_propagates_generic_cleanup_failure_after_ambiguous_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lehome_train.b1k.bootstrap._SMOKE_LOCAL_ROOT", tmp_path)

    class BrokenCleanupHub(_Hub):
        def upload_model_file(self, repository: str, key: str, data: bytes, *, token: str) -> None:
            super().upload_model_file(repository, key, data, token=token)
            raise OSError("lost model response")
        def delete_model_file(self, _repository: str, _key: str, *, token: str) -> None:
            raise RuntimeError("cleanup authentication failure")

    with pytest.raises(RuntimeError, match="authentication"):
        preflight_remote_access("memory-token", "0", hub=BrokenCleanupHub(bucket={"private": True}))


@pytest.mark.parametrize("target", ["model", "bucket"])
def test_remote_access_rejects_typed_missing_cleanup_when_the_exact_key_remains_listed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    monkeypatch.setattr("lehome_train.b1k.bootstrap._SMOKE_LOCAL_ROOT", tmp_path)
    monkeypatch.setattr("lehome_train.b1k.bootstrap._SMOKE_CLEANUP_INTERVAL_SECONDS", 0)

    class LeakedTypedMissingHub(_Hub):
        def upload_model_file(self, repository: str, key: str, data: bytes, *, token: str) -> None:
            if target == "model": raise OSError("model upload interrupted")
            super().upload_model_file(repository, key, data, token=token)
        def delete_model_file(self, repository: str, key: str, *, token: str) -> None:
            if target == "model": raise ModelSmokeObjectNotFound("missing")
            super().delete_model_file(repository, key, token=token)
        def list_model_files(self, repository: str, prefix: str = "", *, token: str) -> tuple[str, ...]:
            if target == "model": return (f"{prefix}model.bin",)
            return super().list_model_files(repository, prefix, token=token)
        def upload_bucket_file(self, bucket: str, source: Path, key: str, *, token: str) -> None:
            if target == "bucket": raise OSError("bucket upload interrupted")
            super().upload_bucket_file(bucket, source, key, token=token)
        def delete_bucket_file(self, bucket: str, key: str, *, token: str) -> None:
            if target == "bucket": raise BucketNotFound("missing")
            super().delete_bucket_file(bucket, key, token=token)
        def list_bucket_files(self, bucket: str, prefix: str, *, token: str) -> tuple[str, ...]:
            if target == "bucket": return (f"{prefix}checkpoint.bin",)
            return super().list_bucket_files(bucket, prefix, token=token)

    with pytest.raises(ValueError, match="cleanup"):
        preflight_remote_access("memory-token", "0", hub=LeakedTypedMissingHub(bucket={"private": True}))


def test_remote_access_accepts_model_and_bucket_cleanup_that_become_consistent_within_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lehome_train.b1k.bootstrap._SMOKE_LOCAL_ROOT", tmp_path)
    monkeypatch.setattr("lehome_train.b1k.bootstrap._SMOKE_CLEANUP_INTERVAL_SECONDS", 0)

    class EventuallyConsistentHub(_Hub):
        def __init__(self) -> None:
            super().__init__(bucket={"private": True}); self.model_lists = 0; self.bucket_lists = 0
        def list_model_files(self, repository: str, prefix: str = "", *, token: str) -> tuple[str, ...]:
            self.model_lists += 1
            return (f"{prefix}model.bin",) if self.model_lists == 1 else ()
        def list_bucket_files(self, bucket: str, prefix: str, *, token: str) -> tuple[str, ...]:
            self.bucket_lists += 1
            return (f"{prefix}checkpoint.bin",) if self.bucket_lists == 1 else ()

    hub = EventuallyConsistentHub()
    preflight_remote_access("memory-token", "0", hub=hub)
    assert hub.model_lists == 2 and hub.bucket_lists == 2


def test_hub_adapter_translates_only_model_entry_not_found_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub.errors import EntryNotFoundError
    import huggingface_hub

    class MissingApi:
        def delete_file(self, *_args: object, **_kwargs: object) -> None:
            raise EntryNotFoundError("missing")

    monkeypatch.setattr(huggingface_hub, "HfApi", MissingApi)
    with pytest.raises(ModelSmokeObjectNotFound, match="exact smoke object"):
        HfHubAdapter().delete_model_file(BEHAVIOR_1K_FINAL_MODEL_REPOSITORY, "smoke/abc/model.bin", token="memory-token")


def test_remote_access_does_not_treat_generic_bucket_failures_as_create_permission() -> None:
    class BrokenHub(_Hub):
        def bucket_info(self, _bucket: str, *, token: str) -> object:
            self.calls.append("bucket")
            raise RuntimeError("network authentication failure")

    hub = BrokenHub()
    with pytest.raises(RuntimeError, match="network"):
        preflight_remote_access("memory-token", "1", hub=hub)
    assert "create" not in hub.calls


def test_remote_access_rejects_a_repository_request_that_resolves_to_another_commit() -> None:
    class MismatchedHub(_Hub):
        def repo_info(self, repo: str, *, revision: str | None, token: str) -> object:
            self.calls.append(f"repo:{repo}:{revision}")
            return {"private": True, "sha": "0" * 40}

    hub = MismatchedHub()
    with pytest.raises(ValueError, match="pinned revision"):
        preflight_remote_access("memory-token", "0", hub=hub)
    assert hub.calls == [f"repo:{BEHAVIOR_1K_DATASET_REPOSITORY}:{BEHAVIOR_1K_DATASET_REVISION}"]


class _BucketClient:
    def __init__(self, *, missing: bool = False) -> None: self.missing = missing; self.calls: list[tuple[str, dict[str, object]]] = []

    def request(self, operation: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append((operation, payload))
        if self.missing and operation == "info": raise BucketNotFound("missing")
        return {"private": True}


def test_production_hub_preserves_typed_bucket_missing_error_for_preflight_create_path() -> None:
    client = _BucketClient(missing=True)
    hub = ProductionHubAccess(client)  # type: ignore[arg-type]
    with pytest.raises(BucketNotFound): hub.bucket_info("owner/checkpoints", token="memory-token")


def _temporary_paths(root: Path) -> WorkspacePaths:
    return WorkspacePaths(
        root=root,
        dataset=root / "data/b1k",
        groot_upstream=root / "models/groot-upstream",
        cosmos=root / "models/cosmos",
        derived_model=root / "models/groot",
        output=root / "outputs/run-001",
        checkpoints=root / "checkpoints",
        logs=root / "logs",
        final=root / "final",
    )


def _ensure_remote_stats(destination: Path) -> None:
    stats = destination / "meta/stats.json"
    if not stats.exists():
        stats.write_text("{}")


class _BootstrapHub:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []; self.model_objects: dict[str, bytes] = {}; self.bucket_objects: dict[str, bytes] = {}
        self.model_submissions = Event(); self._model_submissions = 0; self._lock = Lock()
        self._snapshot_roots: dict[str, Path] = {}; self._remote_manifests: dict[tuple[str, tuple[str, ...] | None], object] = {}

    def repo_info(self, repository: str, *, revision: str | None, token: str) -> object:
        self.events.append(("repo", repository, revision, token))
        return {"private": repository == BEHAVIOR_1K_FINAL_MODEL_REPOSITORY, "sha": revision}

    def bucket_info(self, bucket: str, *, token: str) -> object:
        self.events.append(("bucket", bucket, token))
        return {"private": True}

    def create_bucket(self, *_args: object, **_kwargs: object) -> object: pytest.fail("existing bucket must not be recreated")

    def snapshot_download(self, repository: str, *, revision: str, local_dir: Path, allow_patterns: tuple[str, ...] | None, token: str) -> object:
        self.events.append(("snapshot", repository, revision, local_dir, allow_patterns, token))
        local_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_roots[repository] = local_dir
        if repository == BEHAVIOR_1K_DATASET_REPOSITORY and allow_patterns == dataset_snapshot_patterns()[1]:
            assert self.model_submissions.wait(timeout=2), "both model snapshots must be submitted before full data work completes"
        elif repository in {BASE_MODEL_REPOSITORY, COSMOS_REPOSITORY}:
            with self._lock:
                self._model_submissions += 1
                if self._model_submissions == 2: self.model_submissions.set()
        if repository == BASE_MODEL_REPOSITORY:
            (local_dir / "config.json").write_text('{"model_name":"upstream","other":1}')
            (local_dir / "weights.safetensors").write_bytes(b"weights")
        if repository == COSMOS_REPOSITORY:
            (local_dir / "cosmos.bin").write_bytes(b"cosmos")
        if repository == BEHAVIOR_1K_DATASET_REPOSITORY and allow_patterns == dataset_snapshot_patterns()[1]:
            (local_dir / "meta").mkdir(exist_ok=True)
            (local_dir / "meta/stats.json").write_text("{}")
        return local_dir

    def remote_manifest(self, repository: str, *, revision: str, allow_patterns: tuple[str, ...] | None, token: str):
        self.events.append(("remote-manifest", repository, revision, allow_patterns, token))
        key = (repository, allow_patterns)
        if key not in self._remote_manifests:
            root = self._snapshot_roots[repository]
            entries: list[dict[str, object]] = []
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(root).as_posix()
                if relative in {".b1k-snapshot-intent.json", ".b1k-snapshot-receipt.json", "meta/modality.json"} or relative.startswith(".cache/"):
                    continue
                data = path.read_bytes()
                entries.append({
                    "path": relative, "size": len(data),
                    "blob_id": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(), "lfs": None,
                })
            self._remote_manifests[key] = build_remote_manifest(
                repository=repository, revision=revision, resolved_revision=revision,
                entries=entries, allow_patterns=allow_patterns,
            )
        return self._remote_manifests[key]

    def upload_model_file(self, _repository: str, key: str, data: bytes, *, token: str) -> None:
        self.events.append(("model-upload", key)); self.model_objects[key] = data
    def download_model_file(self, _repository: str, key: str, *, token: str) -> bytes:
        self.events.append(("model-download", key)); return self.model_objects[key]
    def delete_model_file(self, _repository: str, key: str, *, token: str) -> None:
        self.events.append(("model-delete", key)); self.model_objects.pop(key, None)
    def list_model_files(self, _repository: str, prefix: str = "", *, token: str) -> tuple[str, ...]:
        self.events.append(("model-list", prefix)); return tuple(path for path in self.model_objects if path.startswith(prefix))
    def upload_bucket_file(self, _bucket: str, source: Path, key: str, *, token: str) -> None:
        self.events.append(("bucket-upload", key)); self.bucket_objects[key] = source.read_bytes()
    def download_bucket_file(self, _bucket: str, key: str, destination: Path, *, token: str) -> None:
        self.events.append(("bucket-download", key)); destination.write_bytes(self.bucket_objects[key])
    def delete_bucket_file(self, _bucket: str, key: str, *, token: str) -> None:
        self.events.append(("bucket-delete", key)); self.bucket_objects.pop(key, None)
    def list_bucket_files(self, _bucket: str, prefix: str, *, token: str) -> tuple[str, ...]:
        self.events.append(("bucket-list", prefix)); return tuple(path for path in self.bucket_objects if path.startswith(prefix))


def test_bootstrap_remote_full_contract_is_ordered_concurrent_pinned_and_secret_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _temporary_paths(tmp_path / "workspace"); paths.create()
    hub = _BootstrapHub()
    monkeypatch.setattr("lehome_train.b1k.bootstrap.sha256_file", lambda path: "ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641" if Path(path).name == "modality.json" else "b" * 64)

    def deploy(destination: Path) -> Path:
        modality = destination / "meta/modality.json"; modality.parent.mkdir(exist_ok=True); modality.write_bytes(b"modality")
        _ensure_remote_stats(destination)
        return modality

    result = bootstrap_remote(
        paths=paths,
        token="hf_super_secret",
        create_bucket_flag="0",
        hub=hub,
        build_selection=lambda _destination: {"selected": ["episode-1"]},
        materialize=lambda _destination, **_kwargs: {"materialized": ["episode-1"]},
        deploy_modality=deploy,
        stats_path=paths.dataset / "meta/stats.json",
    )
    remote = [event for event in hub.events if event[0] in {"repo", "bucket"}]
    snapshots = [event for event in hub.events if event[0] == "snapshot"]
    assert len(remote) == 5
    assert max(hub.events.index(event) for event in remote) < hub.events.index(snapshots[0])
    dataset_staging = paths.dataset.with_name(".b1k.incomplete")
    assert snapshots[0][1:] == (BEHAVIOR_1K_DATASET_REPOSITORY, BEHAVIOR_1K_DATASET_REVISION, dataset_staging, dataset_snapshot_patterns()[0], "hf_super_secret")
    assert {(event[1], event[2], event[3]) for event in snapshots[1:]} == {
        (BEHAVIOR_1K_DATASET_REPOSITORY, BEHAVIOR_1K_DATASET_REVISION, dataset_staging),
        (BASE_MODEL_REPOSITORY, MODEL_REVISION, paths.groot_upstream.with_name(".groot-upstream.incomplete")),
        (COSMOS_REPOSITORY, COSMOS_REVISION, paths.cosmos.with_name(".cosmos.incomplete")),
    }
    assert json.loads(result.selection_manifest.read_text()) == {"selected": ["episode-1"]}
    assert json.loads(result.materialized_manifest.read_text()) == {"materialized": ["episode-1"]}
    assert result.offline_environment() == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline"}
    assert (paths.derived_model / "weights.safetensors").read_bytes() == b"weights"
    assert json.loads((paths.derived_model / "config.json").read_text())["model_name"] == "/workspace/models/cosmos"
    serialized = "\n".join(path.read_text() for path in paths.output.glob("*.json"))
    assert "hf_super_secret" not in serialized


def test_bootstrap_serializes_a_real_materialized_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _temporary_paths(tmp_path / "workspace"); paths.create()
    monkeypatch.setattr("lehome_train.b1k.bootstrap.sha256_file", lambda path: "ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641" if Path(path).name == "modality.json" else "b" * 64)
    payload = {
        "schema_version": 1, "selection_fingerprint": "a" * 64,
        "artifacts": [{"path": "meta/stats.json", "byte_size": 2, "sha256": "b" * 64}],
        "feature_schema": {"action": {"dtype": "float32", "shape": [23]}},
    }
    materialized = MaterializedTrainingManifest(
        selection_fingerprint=payload["selection_fingerprint"], artifacts=tuple(payload["artifacts"]),
        feature_schema=payload["feature_schema"], fingerprint=canonical_json_sha256(payload),
    )

    def deploy(destination: Path) -> Path:
        modality = destination / "meta/modality.json"; modality.parent.mkdir(exist_ok=True); modality.write_bytes(b"modality")
        _ensure_remote_stats(destination)
        return modality

    result = bootstrap_remote(
        paths=paths, token="hf_super_secret", create_bucket_flag="0", hub=_BootstrapHub(),
        build_selection=lambda _destination: {"selected": ["episode-1"]},
        materialize=lambda _destination, **_kwargs: materialized, deploy_modality=deploy,
        stats_path=paths.dataset / "meta/stats.json",
    )
    assert json.loads(result.materialized_manifest.read_text()) == materialized.to_dict()


def test_bootstrap_resumes_the_matching_sibling_stage_and_reuses_completed_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _temporary_paths(tmp_path / "workspace"); paths.create()
    monkeypatch.setattr("lehome_train.b1k.bootstrap.sha256_file", lambda path: "ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641" if Path(path).name == "modality.json" else "b" * 64)

    class InterruptedHub(_BootstrapHub):
        interrupted = True
        def snapshot_download(self, repository: str, *, revision: str, local_dir: Path, allow_patterns: tuple[str, ...] | None, token: str) -> object:
            result = super().snapshot_download(repository, revision=revision, local_dir=local_dir, allow_patterns=allow_patterns, token=token)
            if repository == BEHAVIOR_1K_DATASET_REPOSITORY and allow_patterns == dataset_snapshot_patterns()[1] and self.interrupted:
                raise OSError("interrupted full download")
            return result

    hub = InterruptedHub()

    def deploy(destination: Path) -> Path:
        modality = destination / "meta/modality.json"; modality.parent.mkdir(exist_ok=True); modality.write_bytes(b"modality")
        _ensure_remote_stats(destination)
        return modality

    arguments = dict(
        paths=paths, token="hf_super_secret", create_bucket_flag="0", hub=hub,
        build_selection=lambda _destination: {"selected": ["episode-1"]},
        materialize=lambda _destination, **_kwargs: {"materialized": ["episode-1"]},
        deploy_modality=deploy, stats_path=paths.dataset / "meta/stats.json",
    )
    with pytest.raises(OSError, match="interrupted"):
        bootstrap_remote(**arguments)
    staging = paths.dataset.with_name(".b1k.incomplete")
    assert staging.is_dir() and not paths.dataset.exists()
    assert (staging / ".b1k-snapshot-intent.json").is_file()

    hub.interrupted = False
    result = bootstrap_remote(**arguments)
    assert result.dataset == paths.dataset
    assert (paths.dataset / ".b1k-snapshot-receipt.json").is_file()
    snapshots_after_promotion = len([event for event in hub.events if event[0] == "snapshot"])
    import lehome_train.b1k.snapshot_integrity as integrity
    hash_calls = 0
    original_hash = integrity._hash_file_once

    def counted_hash(path: Path):
        nonlocal hash_calls
        hash_calls += 1
        return original_hash(path)

    monkeypatch.setattr(integrity, "_hash_file_once", counted_hash)
    bootstrap_remote(**arguments)
    assert len([event for event in hub.events if event[0] == "snapshot"]) == snapshots_after_promotion
    assert hash_calls == 4  # dataset stats, two upstream files, and Cosmos; no second pass per snapshot
    assert all(json.loads((root / ".b1k-snapshot-receipt.json").read_text())["hash_passes"] == 1 for root in (paths.dataset, paths.groot_upstream, paths.cosmos))


def test_bootstrap_fails_closed_for_a_tampered_completed_snapshot_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _temporary_paths(tmp_path / "workspace"); paths.create()
    monkeypatch.setattr("lehome_train.b1k.bootstrap.sha256_file", lambda path: "ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641" if Path(path).name == "modality.json" else "b" * 64)
    hub = _BootstrapHub()

    def deploy(destination: Path) -> Path:
        modality = destination / "meta/modality.json"; modality.parent.mkdir(exist_ok=True); modality.write_bytes(b"modality")
        _ensure_remote_stats(destination)
        return modality

    arguments = dict(
        paths=paths, token="hf_super_secret", create_bucket_flag="0", hub=hub,
        build_selection=lambda _destination: {"selected": ["episode-1"]},
        materialize=lambda _destination, **_kwargs: {"materialized": ["episode-1"]},
        deploy_modality=deploy, stats_path=paths.dataset / "meta/stats.json",
    )
    bootstrap_remote(**arguments)
    receipt = paths.dataset / ".b1k-snapshot-receipt.json"
    receipt_value = json.loads(receipt.read_text())
    receipt_value["revision"] = "0" * 40
    receipt.write_text(json.dumps(receipt_value))
    snapshots_before = len([event for event in hub.events if event[0] == "snapshot"])
    with pytest.raises(ValueError, match="receipt"):
        bootstrap_remote(**arguments)
    assert len([event for event in hub.events if event[0] == "snapshot"]) == snapshots_before


def test_bootstrap_rejects_a_mismatched_modality_before_model_derivation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _temporary_paths(tmp_path / "workspace"); paths.create()
    monkeypatch.setattr("lehome_train.b1k.bootstrap.sha256_file", lambda _path: "0" * 64)

    def deploy(destination: Path) -> Path:
        modality = destination / "meta/modality.json"; modality.parent.mkdir(exist_ok=True); modality.write_bytes(b"incorrect")
        _ensure_remote_stats(destination)
        return modality

    with pytest.raises(ValueError, match="modality hash"):
        bootstrap_remote(
            paths=paths,
            token="hf_super_secret",
            create_bucket_flag="0",
            hub=_BootstrapHub(),
            build_selection=lambda _destination: {},
            materialize=lambda _destination, **_kwargs: {},
            deploy_modality=deploy,
            stats_path=paths.dataset / "meta/stats.json",
        )
    assert not (paths.derived_model / "config.json").exists()


@pytest.mark.parametrize("which", ["modality", "stats"])
def test_bootstrap_rejects_noncanonical_loader_artifact_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str) -> None:
    paths = _temporary_paths(tmp_path / "workspace"); paths.create()
    monkeypatch.setattr("lehome_train.b1k.bootstrap.sha256_file", lambda _path: "ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641")

    def deploy(destination: Path) -> Path:
        modality = destination / "meta/modality.json"; modality.parent.mkdir(exist_ok=True); modality.write_bytes(b"modality")
        _ensure_remote_stats(destination)
        return tmp_path / "other.modality" if which == "modality" else modality

    with pytest.raises(ValueError, match="loader path"):
        bootstrap_remote(
            paths=paths,
            token="hf_super_secret",
            create_bucket_flag="0",
            hub=_BootstrapHub(),
            build_selection=lambda _destination: {},
            materialize=lambda _destination, **_kwargs: {},
            deploy_modality=deploy,
            stats_path=tmp_path / "other.stats" if which == "stats" else paths.dataset / "meta/stats.json",
        )
