from __future__ import annotations

import inspect

import pytest

from b1k_deploy.huggingface import (
    CheckpointBucket,
    CheckpointBucketHelperClient,
    CheckpointBucketProbeReceipt,
    HubAbsenceProof,
    HubProbeOperation,
    HuggingFaceHubClient,
    HuggingFaceReleaseVerifier,
    HubProbeError,
    HubRepository,
    TokenSource,
    ReleaseDestinations,
    WorkspaceCheckpointProbeFiles,
)


class FakeHub:
    def __init__(self):
        self.private: dict[tuple[str, str], bool] = {}
        self.files: dict[tuple[str, str, str, str], bytes] = {}
        self.calls: list[tuple[object, ...]] = []
        self.next_commit = 0
        self.delete_returns_empty = False

    def repo_info(self, repo_id: str, repo_type: str, token: str):
        self.calls.append(("repo_info", repo_id, repo_type, token))
        return {"private": self.private[(repo_id, repo_type)]}

    def upload_bytes(self, repo_id: str, repo_type: str, path: str, content: bytes, token: str):
        self.next_commit += 1
        commit = f"{self.next_commit:040x}"
        self.files[(repo_id, repo_type, commit, path)] = content
        self.calls.append(("upload_bytes", repo_id, repo_type, path, token))
        return commit

    def read_file(self, repo_id: str, repo_type: str, path: str, revision: str, token: str):
        self.calls.append(("read_file", repo_id, repo_type, path, revision, token))
        return self.files[(repo_id, repo_type, revision, path)]

    def delete_file(self, repo_id: str, repo_type: str, path: str, expected_commit: str, token: str):
        self.calls.append(("delete_file", repo_id, repo_type, path, expected_commit, token))
        if self.delete_returns_empty:
            return ""
        self.next_commit += 1
        return f"{self.next_commit:040x}"

    def file_exists(self, repo_id: str, repo_type: str, path: str, revision: str, token: str):
        self.calls.append(("file_exists", repo_id, repo_type, path, revision, token))
        return (repo_id, repo_type, revision, path) in self.files

    def absence_proof(self, repo_id: str, repo_type: str, path: str, revision: str, token: str):
        self.calls.append(("absence_proof", repo_id, repo_type, path, revision, token))
        return HubAbsenceProof(repo_id, repo_type, path, revision, True, True, not self.file_exists(repo_id, repo_type, path, revision, token))

    def resolve_exact_file_head(self, repo_id: str, repo_type: str, path: str, content: bytes, token: str):
        self.calls.append(("resolve_exact_file_head", repo_id, repo_type, path, content, token))
        matches = [revision for (candidate_repo, candidate_type, revision, candidate_path), value in self.files.items() if (candidate_repo, candidate_type, candidate_path, value) == (repo_id, repo_type, path, content)]
        return max(matches) if matches else None

    def current_revision(self, repo_id: str, repo_type: str, token: str):
        self.calls.append(("current_revision", repo_id, repo_type, token))
        return f"{self.next_commit:040x}" if self.next_commit else None


def token_store(name: str) -> str:
    assert name == "huggingface"
    return "hf_abcdefghijklmnopqrstuvwxyz0123456789"


def repositories() -> dict[str, HubRepository]:
    return {
        "model": HubRepository("ryanjin333/behavior1k-groot-n17-models", "model"),
        "dataset": HubRepository("ryanjin333/behavior1k-groot-n17-rollouts", "dataset"),
    }


def private_hub() -> FakeHub:
    hub = FakeHub()
    for repository in repositories().values():
        hub.private[(repository.repo_id, repository.repo_type)] = True
    return hub


def test_all_required_hub_destinations_must_be_private():
    hub = private_hub()
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))

    verified = verifier.verify_private_repositories(repositories())

    assert verified == repositories()
    hub.private[("ryanjin333/behavior1k-groot-n17-rollouts", "dataset")] = False
    with pytest.raises(HubProbeError, match="private"):
        verifier.verify_private_repositories(repositories())


def test_bootstrap_probe_uses_exact_unique_key_and_immutable_readback_then_exact_cleanup(monkeypatch):
    hub = private_hub()
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))
    monkeypatch.setattr("b1k_deploy.huggingface.uuid.uuid4", lambda: type("U", (), {"hex": "0123456789abcdef" * 2})())

    receipt = verifier.bootstrap_probe("dataset", repositories()["dataset"])

    assert receipt.key == "b1k-bootstrap-0123456789abcdef0123456789abcdef/probe.json"
    assert receipt.upload_commit == f"{1:040x}"
    assert receipt.delete_commit == f"{2:040x}"
    assert all(receipt.key in call for call in hub.calls if call[0] in {"upload_bytes", "delete_file", "file_exists"})
    assert ("read_file", "ryanjin333/behavior1k-groot-n17-rollouts", "dataset", receipt.key, receipt.upload_commit, "hf_abcdefghijklmnopqrstuvwxyz0123456789") in hub.calls
    assert ("file_exists", "ryanjin333/behavior1k-groot-n17-rollouts", "dataset", receipt.key, receipt.delete_commit, "hf_abcdefghijklmnopqrstuvwxyz0123456789") in hub.calls


def test_remote_probe_requires_an_image_created_exact_upload_then_performs_controller_readback_cleanup() -> None:
    hub = private_hub()
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))
    prefix = "b1k-bootstrap-" + "a" * 32 + "-smoke-model"
    key = f"{prefix}/probe.json"
    upload_commit = "1" * 40
    hub.next_commit = 1
    hub.files[("ryanjin333/behavior1k-groot-n17-models", "model", upload_commit, key)] = b'{"purpose":"b1k-private-release-bootstrap"}\n'

    receipt = verifier.verify_remote_probe("model", repositories()["model"], prefix=prefix, upload_commit=upload_commit)

    assert receipt.key == key
    assert receipt.upload_commit == upload_commit
    assert receipt.delete_commit == f"{2:040x}"
    assert not any(call[0] == "upload_bytes" for call in hub.calls)
    assert any(call[0] == "read_file" for call in hub.calls)
    assert any(call[0] == "delete_file" for call in hub.calls)


def test_lost_remote_probe_evidence_reconciles_only_the_exact_expected_key() -> None:
    hub = private_hub()
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))
    prefix = "b1k-bootstrap-" + "d" * 32 + "-smoke-model"
    key = f"{prefix}/probe.json"
    upload_commit = "1" * 40
    hub.next_commit = 1
    hub.files[("ryanjin333/behavior1k-groot-n17-models", "model", upload_commit, key)] = b'{"purpose":"b1k-private-release-bootstrap"}\n'

    verifier.reconcile_remote_probe("model", repositories()["model"], prefix=prefix)

    assert ("resolve_exact_file_head", "ryanjin333/behavior1k-groot-n17-models", "model", key, b'{"purpose":"b1k-private-release-bootstrap"}\n', "hf_abcdefghijklmnopqrstuvwxyz0123456789") in hub.calls
    assert ("delete_file", "ryanjin333/behavior1k-groot-n17-models", "model", key, upload_commit, "hf_abcdefghijklmnopqrstuvwxyz0123456789") in hub.calls
    assert any(call[0] == "absence_proof" and call[3] == key for call in hub.calls)


def test_lost_remote_probe_evidence_proves_exact_key_absent_at_the_current_revision() -> None:
    hub = private_hub()
    hub.next_commit = 1
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))
    prefix = "b1k-bootstrap-" + "e" * 32 + "-smoke-model"
    key = f"{prefix}/probe.json"

    verifier.reconcile_remote_probe("model", repositories()["model"], prefix=prefix)

    assert not any(call[0] == "delete_file" for call in hub.calls)
    assert ("absence_proof", "ryanjin333/behavior1k-groot-n17-models", "model", key, f"{1:040x}", "hf_abcdefghijklmnopqrstuvwxyz0123456789") in hub.calls


@pytest.mark.parametrize(
    ("role", "prefix"),
    [
        ("model", "b1k-bootstrap-" + "f" * 32 + "-success-fixture"),
        ("dataset", "b1k-bootstrap-" + "f" * 32 + "-smoke-model"),
        ("model", "b1k-bootstrap-" + "f" * 31 + "-smoke-model"),
    ],
)
def test_lost_remote_probe_reconciliation_rejects_any_non_exact_role_specific_prefix(
    role: str, prefix: str
) -> None:
    hub = private_hub()
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))

    with pytest.raises(HubProbeError, match="role-specific"):
        verifier.reconcile_remote_probe(role, repositories()[role], prefix=prefix)

    assert hub.calls == []


def test_remote_probe_readback_mismatch_still_cleans_up_and_preserves_the_primary_error() -> None:
    hub = private_hub()
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))
    prefix = "b1k-bootstrap-" + "b" * 32 + "-smoke-model"
    key = f"{prefix}/probe.json"
    upload_commit = "1" * 40
    hub.next_commit = 1
    hub.files[("ryanjin333/behavior1k-groot-n17-models", "model", upload_commit, key)] = b"wrong bytes"
    hub.current_revision = lambda *_args: upload_commit
    hub.resolve_exact_file_head = lambda *_args: upload_commit

    with pytest.raises(HubProbeError, match="immutable readback"):
        verifier.verify_remote_probe("model", repositories()["model"], prefix=prefix, upload_commit=upload_commit)

    assert any(call[0] == "delete_file" and call[3] == key for call in hub.calls)
    assert any(call[0] == "absence_proof" and call[3] == key for call in hub.calls)


def test_remote_probe_reports_cleanup_failure_without_losing_the_readback_failure() -> None:
    hub = private_hub()
    hub.delete_returns_empty = True
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))
    prefix = "b1k-bootstrap-" + "c" * 32 + "-smoke-model"
    key = f"{prefix}/probe.json"
    upload_commit = "1" * 40
    hub.next_commit = 1
    hub.files[("ryanjin333/behavior1k-groot-n17-models", "model", upload_commit, key)] = b"wrong bytes"
    hub.current_revision = lambda *_args: upload_commit
    hub.resolve_exact_file_head = lambda *_args: upload_commit

    with pytest.raises(HubProbeError, match="immutable readback.*cleanup failed"):
        verifier.verify_remote_probe("model", repositories()["model"], prefix=prefix, upload_commit=upload_commit)


def test_probe_readback_mismatch_still_deletes_only_the_uploaded_exact_key_and_verifies_absence():
    hub = private_hub()
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))

    original = hub.read_file

    def stale_read(*args):
        original(*args)
        return b"not the uploaded bytes"

    hub.read_file = stale_read
    with pytest.raises(HubProbeError, match="immutable"):
        verifier.bootstrap_probe("model", repositories()["model"], operation_id="e" * 32)

    upload = next(call for call in hub.calls if call[0] == "upload_bytes")
    delete = next(call for call in hub.calls if call[0] == "delete_file")
    absence = next(call for call in hub.calls if call[0] == "file_exists")
    assert delete[3] == upload[3]
    assert delete[4] == f"{1:040x}"
    assert absence[3] == upload[3]
    assert absence[4] == f"{2:040x}"

    with pytest.raises(HubProbeError, match="lacks immutable readback proof"):
        verifier.bootstrap_probe("model", repositories()["model"], operation_id="e" * 32)


def test_probe_reconciles_delete_response_lost_to_current_head_absence_and_retries_idempotently():
    hub = private_hub()
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))
    original_delete = hub.delete_file
    failed = False

    def committed_then_raised(*args):
        nonlocal failed
        commit = original_delete(*args)
        if not failed:
            failed = True
            raise RuntimeError("delete response lost")
        return commit

    hub.delete_file = committed_then_raised
    receipt = verifier.bootstrap_probe("dataset", repositories()["dataset"], operation_id="f" * 32)
    retry = verifier.bootstrap_probe("dataset", repositories()["dataset"], operation_id="f" * 32)

    assert receipt.delete_commit == f"{2:040x}"
    assert retry == receipt
    assert [call[0] for call in hub.calls].count("delete_file") == 1


def test_probe_fails_closed_if_delete_has_no_immutable_commit_and_never_deletes_another_key():
    hub = private_hub()
    hub.delete_returns_empty = True
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))

    with pytest.raises(HubProbeError, match="immutable"):
        verifier.bootstrap_probe("model", repositories()["model"])

    delete = next(call for call in hub.calls if call[0] == "delete_file")
    assert delete[3].endswith("/probe.json")
    assert delete[3].startswith("b1k-bootstrap-")


def test_probe_preserves_readback_failure_and_loudly_surfaces_exact_cleanup_failure():
    hub = private_hub()
    hub.delete_returns_empty = True
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))

    original = hub.read_file

    def stale_read(*args):
        original(*args)
        return b"not the uploaded bytes"

    hub.read_file = stale_read
    with pytest.raises(HubProbeError, match="immutable readback.*cleanup failed"):
        verifier.bootstrap_probe("model", repositories()["model"])

    upload = next(call for call in hub.calls if call[0] == "upload_bytes")
    delete = next(call for call in hub.calls if call[0] == "delete_file")
    assert delete[3] == upload[3]
    assert delete[4] == f"{1:040x}"


def test_probe_preserves_safe_primary_and_cleanup_labels_without_callback_text():
    hub = private_hub()
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))
    arbitrary_secret = "unrecognized hub secret: rose-garnet-29"
    provider_message = "Hub diagnostic that must never be returned"

    def broken_read(*_args):
        raise HubProbeError(f"{provider_message}; {arbitrary_secret}")

    def broken_delete(*_args):
        raise HubProbeError(f"{provider_message}; {arbitrary_secret}")

    hub.read_file = broken_read
    hub.delete_file = broken_delete
    with pytest.raises(HubProbeError) as error:
        verifier.bootstrap_probe("model", repositories()["model"])

    assert str(error.value) == "immutable readback failed; exact probe cleanup failed"
    assert arbitrary_secret not in str(error.value)
    assert provider_message not in str(error.value)


def test_injected_hub_domain_and_other_callback_errors_are_operation_only():
    arbitrary_secret = "unrecognized hub secret: indigo-cedar-11"
    provider_message = "Hub raw provider diagnostic"

    class DomainFailure(FakeHub):
        def repo_info(self, repo_id: str, repo_type: str, token: str):
            raise HubProbeError(f"{provider_message}; {arbitrary_secret}")

    hub = DomainFailure()
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))
    with pytest.raises(HubProbeError) as domain_error:
        verifier.verify_private_repositories(repositories())
    assert str(domain_error.value) == "repository privacy lookup failed"
    assert arbitrary_secret not in str(domain_error.value)
    assert provider_message not in str(domain_error.value)

    class OtherFailure(FakeHub):
        def __init__(self):
            super().__init__()
            for repository in repositories().values():
                self.private[(repository.repo_id, repository.repo_type)] = True

        def upload_bytes(self, repo_id: str, repo_type: str, path: str, content: bytes, token: str):
            raise RuntimeError(f"{provider_message}; {arbitrary_secret}")

    verifier = HuggingFaceReleaseVerifier(
        OtherFailure(), TokenSource.from_credential_store("huggingface", token_store)
    )
    with pytest.raises(HubProbeError) as callback_error:
        verifier.bootstrap_probe("dataset", repositories()["dataset"])
    assert str(callback_error.value) == "probe upload failed"
    assert arbitrary_secret not in str(callback_error.value)
    assert provider_message not in str(callback_error.value)


@pytest.mark.parametrize(
    "repos",
    [
        {"model": repositories()["model"]},
        {**repositories(), "checkpoint": HubRepository("ryanjin333/behavior1k-groot-n17-checkpoints", "dataset")},
        {**repositories(), "model": HubRepository("https://token@example.test/model", "model")},
        {**repositories(), "model": HubRepository("ryanjin333/lehome-groot-n17-models", "model")},
        {**repositories(), "dataset": HubRepository("ryanjin333/behavior1k-groot-n17-models", "dataset")},
    ],
)
def test_required_roles_types_and_credential_bearing_repository_ids_are_rejected_without_remote_calls(repos):
    hub = private_hub()
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))

    with pytest.raises(HubProbeError):
        verifier.verify_private_repositories(repos)

    assert hub.calls == []


def test_public_readback_api_never_accepts_a_raw_credential_argument():
    parameters = inspect.signature(HuggingFaceReleaseVerifier.readback_immutable).parameters

    assert "token" not in parameters


def test_public_immutable_readback_rechecks_that_its_destination_is_private():
    hub = private_hub()
    repository = repositories()["model"]
    hub.private[(repository.repo_id, repository.repo_type)] = False
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))

    with pytest.raises(HubProbeError, match="private"):
        verifier.readback_immutable(repository, "release/manifest.json", f"{1:040x}", b"expected")

    assert [call[0] for call in hub.calls] == ["repo_info"]


def test_absence_proof_requires_private_repository_existing_immutable_revision_and_exact_key_absence():
    hub = private_hub()
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))
    repository = repositories()["dataset"]
    proof = verifier.verify_absence(repository, "b1k-bootstrap-operation/probe.json", f"{2:040x}")
    assert proof.repository_private is True
    assert proof.revision_exists is True
    assert proof.key_absent is True

    class AmbiguousHub(FakeHub):
        def absence_proof(self, repo_id, repo_type, path, revision, token):
            return HubAbsenceProof(repo_id, repo_type, path, revision, True, False, True)

    with pytest.raises(HubProbeError, match="absence"):
        HuggingFaceReleaseVerifier(AmbiguousHub(), TokenSource.from_credential_store("huggingface", token_store)).verify_absence(repository, "b1k-bootstrap-operation/probe.json", f"{2:040x}")


def test_probe_reconciles_committed_upload_response_failure_and_reuses_exact_operation_on_retry():
    hub = private_hub()
    operation_id = "a" * 32
    repository = repositories()["model"]
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))
    operation = verifier.begin_probe_operation("model", repository, operation_id=operation_id)
    assert isinstance(operation, HubProbeOperation)

    original_upload = hub.upload_bytes
    failed = False

    def committed_then_raised(*args):
        nonlocal failed
        commit = original_upload(*args)
        if not failed:
            failed = True
            raise RuntimeError("response lost after remote commit")
        return commit

    hub.upload_bytes = committed_then_raised
    receipt = verifier.bootstrap_probe("model", repository, operation_id=operation_id)
    retry = verifier.bootstrap_probe("model", repository, operation_id=operation_id)

    assert receipt.key == operation.key
    assert retry == receipt
    assert [call[0] for call in hub.calls].count("upload_bytes") == 1
    assert any(call[0] == "resolve_exact_file_head" and call[3] == operation.key for call in hub.calls)


def test_probe_reconciles_a_malformed_post_commit_upload_response_without_creating_a_second_key():
    hub = private_hub()
    repository = repositories()["dataset"]
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))
    original_upload = hub.upload_bytes

    def committed_then_malformed(*args):
        original_upload(*args)
        return "not-an-immutable-commit"

    hub.upload_bytes = committed_then_malformed
    receipt = verifier.bootstrap_probe("dataset", repository, operation_id="d" * 32)

    assert receipt.key == "b1k-bootstrap-" + "d" * 32 + "/probe.json"
    assert [call[0] for call in hub.calls].count("upload_bytes") == 1
    assert [call[0] for call in hub.calls].count("resolve_exact_file_head") == 1


def test_probe_retries_delete_at_current_head_only_when_exact_key_content_is_unchanged():
    hub = private_hub()
    repository = repositories()["model"]
    verifier = HuggingFaceReleaseVerifier(hub, TokenSource.from_credential_store("huggingface", token_store))
    original_delete = hub.delete_file
    calls = 0

    def stale_parent(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            key = args[2]
            hub.next_commit += 1
            hub.files[(repository.repo_id, repository.repo_type, f"{hub.next_commit:040x}", key)] = b'{"purpose":"b1k-private-release-bootstrap"}\n'
            raise RuntimeError("parent stale")
        return original_delete(*args)

    hub.delete_file = stale_parent
    receipt = verifier.bootstrap_probe("model", repository, operation_id="b" * 32)
    assert receipt.delete_commit == f"{3:040x}"
    assert calls == 2

    changed = private_hub()
    verifier = HuggingFaceReleaseVerifier(changed, TokenSource.from_credential_store("huggingface", token_store))
    original_delete = changed.delete_file
    changed_calls = 0

    def changed_parent(*args):
        nonlocal changed_calls
        changed_calls += 1
        key = args[2]
        changed.next_commit += 1
        changed.files[(repository.repo_id, repository.repo_type, f"{changed.next_commit:040x}", key)] = b"changed"
        raise RuntimeError("parent stale")

    changed.delete_file = changed_parent
    with pytest.raises(HubProbeError, match="exact probe deletion"):
        verifier.bootstrap_probe("model", repository, operation_id="c" * 32)
    assert changed_calls == 1


def test_concrete_hub_client_uses_pinned_api_surface_and_returns_strict_absence_proof():
    class MissingFile(Exception):
        pass

    class Api:
        def __init__(self):
            self.calls = []

        def repo_info(self, **kwargs):
            self.calls.append(("repo_info", kwargs))
            return type("Info", (), {"private": True})()

        def upload_file(self, **kwargs):
            self.calls.append(("upload_file", kwargs))
            return type("Commit", (), {"oid": "a" * 40})()

        def delete_file(self, **kwargs):
            self.calls.append(("delete_file", kwargs))
            return type("Commit", (), {"oid": "b" * 40})()

        def list_repo_commits(self, **kwargs):
            self.calls.append(("list_repo_commits", kwargs))
            return [type("Commit", (), {"commit_id": "a" * 40})()]

    api = Api()

    def download(**kwargs):
        raise MissingFile()

    client = HuggingFaceHubClient(api_factory=lambda token: api, download=download, missing_file_errors=(MissingFile,))
    repository = repositories()["dataset"]
    assert client.repo_info(repository.repo_id, repository.repo_type, "hf_token") == {"private": True}
    assert client.upload_bytes(repository.repo_id, repository.repo_type, "release/x.json", b"x", "hf_token") == "a" * 40
    assert client.delete_file(repository.repo_id, repository.repo_type, "release/x.json", "a" * 40, "hf_token") == "b" * 40
    proof = client.absence_proof(repository.repo_id, repository.repo_type, "release/x.json", "b" * 40, "hf_token")
    assert proof == HubAbsenceProof(repository.repo_id, repository.repo_type, "release/x.json", "b" * 40, True, True, True)
    assert api.calls[2] == ("delete_file", {"path_in_repo": "release/x.json", "repo_id": repository.repo_id, "repo_type": "dataset", "parent_commit": "a" * 40, "token": "hf_token"})


def test_checkpoint_bucket_helper_keeps_bucket_protocol_separate_from_hub_repositories():
    calls = []

    def runner(arguments, **kwargs):
        calls.append((arguments, kwargs))
        operation = __import__("json").loads(kwargs["input"])["operation"]
        result = {"private": True} if operation == "info" else {"files": []} if operation == "list" else {}
        return __import__("subprocess").CompletedProcess(arguments, 0, stdout=__import__("json").dumps({"ok": True, "result": result}), stderr="")

    bucket = CheckpointBucket()
    client = CheckpointBucketHelperClient("/workspace/b1k-bucket-helper", "/workspace/.cache/huggingface/token", runner=runner)
    assert client.info(bucket) == {"private": True}
    client.upload(bucket, "/workspace/checkpoints/probe.json", "b1k-bootstrap/op/probe.json")
    client.download(bucket, "b1k-bootstrap/op/probe.json", "/workspace/checkpoints/readback.json")
    client.delete(bucket, "b1k-bootstrap/op/probe.json")
    assert client.list(bucket, "b1k-bootstrap/op") == ()
    assert [__import__("json").loads(call[1]["input"])["operation"] for call in calls] == ["info", "upload", "download", "delete", "list"]
    assert calls[1][1]["env"]["B1K_HF_TOKEN_FILE"] == "/workspace/.cache/huggingface/token"
    assert __import__("json").loads(calls[3][1]["input"])["payload"]["paths"] == ["b1k-bootstrap/op/probe.json"]

    verifier = HuggingFaceReleaseVerifier(private_hub(), TokenSource.from_credential_store("huggingface", token_store))
    destinations = ReleaseDestinations(repositories()["model"], bucket, repositories()["dataset"])
    assert verifier.verify_private_destinations(destinations, client) == destinations


def test_checkpoint_bucket_bootstrap_probe_uploads_reads_compares_and_removes_only_its_unique_key_with_reconciliation():
    class ProbeFiles:
        def __init__(self):
            self.files = {}

        def write_bytes(self, path, content):
            self.files[path] = content

        def read_bytes(self, path):
            return self.files[path]

        def remove(self, path):
            self.files.pop(path, None)

    local = ProbeFiles()
    remote = {}
    calls = []
    delete_response_lost = True

    def runner(arguments, **kwargs):
        nonlocal delete_response_lost
        request = __import__("json").loads(kwargs["input"])
        calls.append(request)
        payload = request["payload"]
        operation = request["operation"]
        if operation == "info":
            result = {"private": True}
        elif operation == "upload":
            remote[payload["remote_path"]] = local.files[payload["local_path"]]
            result = {}
        elif operation == "list":
            result = {"files": [{"path": path, "size": len(content), "xet_hash": None, "type": "file"} for path, content in remote.items() if path.startswith(payload["prefix"])]}
        elif operation == "download":
            local.files[payload["local_path"]] = remote[payload["remote_path"]]
            result = {}
        elif operation == "delete":
            for path in payload["paths"]:
                remote.pop(path, None)
            if delete_response_lost:
                delete_response_lost = False
                return __import__("subprocess").CompletedProcess(arguments, 1, stdout='{"ok":false,"error":"operation_failed"}', stderr="raw provider response")
            result = {}
        else:
            pytest.fail(f"unexpected operation {operation}")
        return __import__("subprocess").CompletedProcess(arguments, 0, stdout=__import__("json").dumps({"ok": True, "result": result}), stderr="")

    client = CheckpointBucketHelperClient("/workspace/b1k-bucket-helper", "/workspace/.cache/huggingface/token", runner=runner)
    verifier = HuggingFaceReleaseVerifier(private_hub(), TokenSource.from_credential_store("huggingface", token_store))

    receipt = verifier.bootstrap_checkpoint_bucket_probe(client, CheckpointBucket(), operation_id="1" * 32, files=local)
    retry = verifier.bootstrap_checkpoint_bucket_probe(client, CheckpointBucket(), operation_id="1" * 32, files=local)

    assert isinstance(receipt, CheckpointBucketProbeReceipt)
    assert retry == receipt
    assert receipt.key == "b1k-bootstrap-" + "1" * 32 + "/probe.json"
    assert remote == {}
    assert "hf_abcdefghijklmnopqrstuvwxyz0123456789" not in repr(receipt)
    assert all("hf_abcdefghijklmnopqrstuvwxyz0123456789" not in str(call) for call in calls)
    assert [call["operation"] for call in calls] == ["info", "list", "upload", "list", "download", "delete", "list", "info"]
    assert all(call["payload"]["bucket_id"] == "ryanjin333/behavior1k-groot-n17-checkpoints" for call in calls)


def test_checkpoint_bucket_probe_rejects_a_bucket_that_is_not_explicitly_private_before_staging_or_remote_mutation():
    class PublicBucket:
        def info(self, _bucket):
            return {"private": False}

        def list(self, *_args):
            pytest.fail("must not list a public bucket")

        def upload(self, *_args):
            pytest.fail("must not upload to a public bucket")

    verifier = HuggingFaceReleaseVerifier(private_hub(), TokenSource.from_credential_store("huggingface", token_store))

    with pytest.raises(HubProbeError, match="explicitly private"):
        verifier.bootstrap_checkpoint_bucket_probe(PublicBucket(), CheckpointBucket(), operation_id="2" * 32)


def test_workspace_checkpoint_probe_files_reject_a_symlinked_staging_parent_without_writing_outside_its_root(tmp_path, monkeypatch):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    (checkpoint_root / ".b1k-release-probes").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(WorkspaceCheckpointProbeFiles, "_root", checkpoint_root)
    monkeypatch.setattr(WorkspaceCheckpointProbeFiles, "_logical_root", "/workspace/checkpoints", raising=False)
    files = WorkspaceCheckpointProbeFiles()
    operation_id = "3" * 32
    path = f"/workspace/checkpoints/.b1k-release-probes/{operation_id}.upload.json"

    assert files._target(path) == checkpoint_root / ".b1k-release-probes" / f"{operation_id}.upload.json"
    with pytest.raises(HubProbeError, match="local staging"):
        files.write_bytes(path, b"probe bytes")
    assert not (outside / f"{operation_id}.upload.json").exists()


def test_checkpoint_probe_retry_after_remote_deletion_retries_local_cleanup_before_issuing_a_receipt():
    class ProbeFiles:
        def __init__(self):
            self.files = {}
            self.fail_cleanup_once = True

        def write_bytes(self, path, content):
            self.files[path] = content

        def read_bytes(self, path):
            return self.files[path]

        def remove(self, path):
            if path.endswith(".readback.json") and path in self.files and self.fail_cleanup_once:
                self.fail_cleanup_once = False
                raise RuntimeError("local filesystem busy")
            self.files.pop(path, None)

    local = ProbeFiles()
    remote = {}
    operations = []

    def runner(arguments, **kwargs):
        request = __import__("json").loads(kwargs["input"])
        operations.append(request["operation"])
        payload = request["payload"]
        operation = request["operation"]
        if operation == "info":
            result = {"private": True}
        elif operation == "list":
            result = {"files": [{"path": path, "size": len(content), "xet_hash": None, "type": "file"} for path, content in remote.items() if path.startswith(payload["prefix"])]}
        elif operation == "upload":
            remote[payload["remote_path"]] = local.files[payload["local_path"]]
            result = {}
        elif operation == "download":
            local.files[payload["local_path"]] = remote[payload["remote_path"]]
            result = {}
        elif operation == "delete":
            for path in payload["paths"]:
                remote.pop(path, None)
            result = {}
        else:
            pytest.fail(f"unexpected operation {operation}")
        return __import__("subprocess").CompletedProcess(arguments, 0, stdout=__import__("json").dumps({"ok": True, "result": result}), stderr="")

    client = CheckpointBucketHelperClient("/workspace/b1k-bucket-helper", "/workspace/.cache/huggingface/token", runner=runner)
    verifier = HuggingFaceReleaseVerifier(private_hub(), TokenSource.from_credential_store("huggingface", token_store))

    with pytest.raises(HubProbeError, match="local cleanup"):
        verifier.bootstrap_checkpoint_bucket_probe(client, CheckpointBucket(), operation_id="4" * 32, files=local)
    assert remote == {}
    assert any(path.endswith(".readback.json") for path in local.files)

    receipt = verifier.bootstrap_checkpoint_bucket_probe(client, CheckpointBucket(), operation_id="4" * 32, files=local)

    assert receipt.key == "b1k-bootstrap-" + "4" * 32 + "/probe.json"
    assert remote == {}
    assert local.files == {}
    assert operations[-2:] == ["info", "list"]
