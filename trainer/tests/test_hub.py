from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lehome_train.hub as hub_module
from lehome_train.hub import (
    HubAccess,
    HuggingFaceHubTransport,
    HubRateLimitError,
    HubTransientError,
    download_files,
    ensure_approved_private_repository,
    require_access,
    upload_files,
)
from lehome_train.models import SyncEntry


class FakeTransport:
    def __init__(self) -> None:
        self.access = HubAccess(can_read=True, can_write=True)
        self.upload_calls: list[dict[str, object]] = []
        self.upload_failures = 0
        self.download_calls: list[dict[str, object]] = []
        self.list_calls: list[dict[str, object]] = []

    def check_access(self, *, repository: str, token: str) -> HubAccess:
        return self.access

    def upload_files(
        self,
        *,
        repository: str,
        revision: str,
        source: Path,
        entries: tuple[SyncEntry, ...],
        token: str,
    ) -> str:
        self.upload_calls.append(
            {
                "repository": repository,
                "revision": revision,
                "source": source,
                "entries": entries,
                "token": token,
            }
        )
        if self.upload_failures:
            self.upload_failures -= 1
            raise HubTransientError("hf_sensitive_retry_token transport detail")
        return "a" * 40

    def download_files(
        self,
        *,
        repository: str,
        revision: str,
        destination: Path,
        relative_paths: tuple[str, ...],
        token: str,
    ) -> str:
        self.download_calls.append(
            {
                "repository": repository,
                "revision": revision,
                "destination": destination,
                "relative_paths": relative_paths,
                "token": token,
            }
        )
        return revision

    def list_tree(
        self,
        *,
        repository: str,
        revision: str,
        token: str,
    ) -> tuple[hub_module.HubTreeEntry, ...]:
        self.list_calls.append(
            {
                "repository": repository,
                "revision": revision,
                "token": token,
            }
        )
        return (hub_module.HubTreeEntry("payload.bin", "file"),)


class FakeRepositoryTransport(FakeTransport):
    def __init__(self) -> None:
        super().__init__()
        self.ensure_calls: list[dict[str, object]] = []

    def ensure_private_repository(
        self,
        *,
        repository: str,
        repo_type: str,
        token: str,
        create: bool,
        timeout_seconds: float,
    ) -> HubAccess:
        self.ensure_calls.append(
            {
                "repository": repository,
                "repo_type": repo_type,
                "token": token,
                "create": create,
                "timeout_seconds": timeout_seconds,
            }
        )
        return HubAccess(can_read=True, can_write=True, private_repository=True)


def test_upload_requires_process_token_and_passes_it_explicitly(tmp_path: Path) -> None:
    transport = FakeTransport()
    entry = SyncEntry("payload.bin", "0" * 64, 0)

    resolved = upload_files(
        transport=transport,
        repository="owner/private-data",
        revision="prepared-v1",
        source=tmp_path,
        entries=(entry,),
        environ={"HF_TOKEN": "hf_process_memory_only"},
    )

    assert resolved == "a" * 40
    assert transport.upload_calls == [
        {
            "repository": "owner/private-data",
            "revision": "prepared-v1",
            "source": tmp_path,
            "entries": (entry,),
            "token": "hf_process_memory_only",
        }
    ]

    with pytest.raises(ValueError, match="HF_TOKEN"):
        upload_files(
            transport=transport,
            repository="owner/private-data",
            revision="prepared-v1",
            source=tmp_path,
            entries=(entry,),
            environ={},
        )


@pytest.mark.parametrize(
    ("access", "read", "write", "message"),
    [
        (HubAccess(can_read=False, can_write=True), True, False, "read"),
        (HubAccess(can_read=True, can_write=False), False, True, "write"),
    ],
)
def test_permission_failures_are_fail_closed_and_redacted(
    access: HubAccess,
    read: bool,
    write: bool,
    message: str,
) -> None:
    transport = FakeTransport()
    transport.access = access
    token = "hf_sensitive_permission_token"

    with pytest.raises(PermissionError, match=message) as error:
        require_access(
            transport=transport,
            repository="owner/private-data",
            read=read,
            write=write,
            environ={"HF_TOKEN": token},
        )

    assert token not in str(error.value)


def test_hub_private_repo_without_permission_metadata_requires_token_write_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = HuggingFaceHubTransport()
    monkeypatch.setattr(
        transport,
        "_repo_info",
        lambda **_kwargs: SimpleNamespace(private=True),
    )
    monkeypatch.setattr(
        transport,
        "_api",
        lambda _token: SimpleNamespace(
            whoami=lambda **_kwargs: {
                "name": "RyanJin333",
                "auth": {"accessToken": {"role": "write"}}
            }
        ),
    )

    access = transport.check_access(
        repository="ryanjin333/lehome-groot-n17-models",
        token="hf_explicit_permission_probe",
    )

    assert access.can_read is True
    assert access.can_write is True
    assert access.private_repository is True


def test_hub_fine_grained_owner_scope_grants_repo_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = HuggingFaceHubTransport()
    monkeypatch.setattr(
        transport,
        "_repo_info",
        lambda **_kwargs: SimpleNamespace(private=True),
    )
    monkeypatch.setattr(
        transport,
        "_api",
        lambda _token: SimpleNamespace(
            whoami=lambda **_kwargs: {
                "name": "RyanJin333",
                "auth": {
                    "accessToken": {
                        "role": "fineGrained",
                        "fineGrained": {
                            "scoped": [
                                {
                                    "entity": {"type": "user", "name": "ryanjin333"},
                                    "permissions": ["repo.content.read", "repo.write"],
                                }
                            ]
                        },
                    }
                },
            }
        ),
    )

    access = transport.check_access(
        repository="ryanjin333/lehome-groot-n17-models",
        token="hf_fine_grained_permission_probe",
    )

    assert access.can_read is True
    assert access.can_write is True


def test_hub_fine_grained_scope_for_another_owner_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = HuggingFaceHubTransport()
    monkeypatch.setattr(
        transport,
        "_repo_info",
        lambda **_kwargs: SimpleNamespace(private=True),
    )
    monkeypatch.setattr(
        transport,
        "_api",
        lambda _token: SimpleNamespace(
            whoami=lambda **_kwargs: {
                "name": "RyanJin333",
                "auth": {
                    "accessToken": {
                        "role": "fineGrained",
                        "fineGrained": {
                            "scoped": [
                                {
                                    "entity": {"type": "user", "name": "someone-else"},
                                    "permissions": ["repo.write"],
                                }
                            ]
                        },
                    }
                },
            }
        ),
    )

    access = transport.check_access(
        repository="ryanjin333/lehome-groot-n17-models",
        token="hf_fine_grained_permission_probe",
    )

    assert access.can_read is True
    assert access.can_write is False


@pytest.mark.parametrize(
    ("identity", "expected_write"),
    [
        (
            {
                "name": "ryanjin333",
                "auth": {"accessToken": {"role": "write"}},
            },
            True,
        ),
        (
            {
                "name": "someone-else",
                "auth": {"accessToken": {"role": "write"}},
            },
            False,
        ),
        ({"auth": {"accessToken": {"role": "write"}}}, False),
        (
            {
                "name": "ryanjin333",
                "auth": {"accessToken": {"role": "read"}},
            },
            False,
        ),
        ({}, False),
        ({"auth": None}, False),
        ({"auth": {"accessToken": "malformed"}}, False),
        ({"auth": {"accessToken": {"role": ["write"]}}}, False),
    ],
)
def test_hub_token_capability_is_explicit_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    identity: object,
    expected_write: bool,
) -> None:
    transport = HuggingFaceHubTransport()
    monkeypatch.setattr(
        transport,
        "_repo_info",
        lambda **_kwargs: SimpleNamespace(private=True),
    )
    monkeypatch.setattr(
        transport,
        "_api",
        lambda _token: SimpleNamespace(whoami=lambda **_kwargs: identity),
    )

    access = transport.check_access(
        repository="ryanjin333/lehome-groot-n17-models",
        token="hf_explicit_capability_probe",
    )

    assert access.can_read is True
    assert access.can_write is expected_write


def test_hub_token_capability_failure_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = HuggingFaceHubTransport()
    token = "hf_sensitive_capability_token"
    monkeypatch.setattr(
        transport,
        "_repo_info",
        lambda **_kwargs: SimpleNamespace(private=True),
    )
    monkeypatch.setattr(
        transport,
        "_api",
        lambda _token: SimpleNamespace(
            whoami=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(token))
        ),
    )

    with pytest.raises(PermissionError, match="access check failed") as error:
        transport.check_access(
            repository="ryanjin333/lehome-groot-n17-models",
            token=token,
        )

    assert token not in str(error.value)
    assert error.value.__cause__ is None


def test_upload_retries_only_to_the_explicit_limit_and_redacts_failures(
    tmp_path: Path,
) -> None:
    transport = FakeTransport()
    transport.upload_failures = 4
    token = "hf_sensitive_retry_token"

    with pytest.raises(RuntimeError, match="3 attempts") as error:
        upload_files(
            transport=transport,
            repository="owner/private-data",
            revision="prepared-v1",
            source=tmp_path,
            entries=(SyncEntry("payload.bin", "0" * 64, 0),),
            environ={"HF_TOKEN": token},
            max_attempts=3,
        )

    assert len(transport.upload_calls) == 3
    assert token not in str(error.value)


def test_retry_backoff_is_bounded_and_uses_the_injected_sleeper(
    tmp_path: Path,
) -> None:
    transport = FakeTransport()
    transport.upload_failures = 2
    observed_delays: list[float] = []

    resolved = upload_files(
        transport=transport,
        repository="owner/private-data",
        revision="prepared-v1",
        source=tmp_path,
        entries=(SyncEntry("payload.bin", "0" * 64, 0),),
        environ={"HF_TOKEN": "hf_retry_process_token"},
        max_attempts=3,
        sleeper=observed_delays.append,
    )

    assert resolved == "a" * 40
    assert observed_delays == [0.25, 0.5]
    assert len(transport.upload_calls) == 3


def test_real_transport_recovers_branch_head_after_post_commit_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transport = HuggingFaceHubTransport()
    revision = "d" * 40
    monkeypatch.setattr(
        transport,
        "_api",
        lambda _token: SimpleNamespace(
            upload_folder=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("post-commit response failed")
            )
        ),
    )
    monkeypatch.setattr(
        transport,
        "_repo_info",
        lambda **_kwargs: SimpleNamespace(sha=revision),
    )

    resolved = transport.upload_files(
        repository="ryanjin333/lehome-groot-n17-data",
        revision="lehome-groot-n17-v1",
        source=tmp_path,
        entries=(SyncEntry("manifest.json", "0" * 64, 0),),
        token="hf_post_commit_recovery_probe",
    )

    assert resolved == revision


def test_download_requires_and_preserves_an_explicit_immutable_revision(
    tmp_path: Path,
) -> None:
    transport = FakeTransport()
    revision = "b" * 40

    observed = download_files(
        transport=transport,
        repository="owner/private-data",
        revision=revision,
        destination=tmp_path,
        relative_paths=("manifest.json",),
        environ={"HF_TOKEN": "hf_download_process_token"},
    )

    assert observed == revision
    assert transport.download_calls[0]["revision"] == revision
    assert transport.download_calls[0]["token"] == "hf_download_process_token"

    with pytest.raises(ValueError, match="immutable"):
        download_files(
            transport=transport,
            repository="owner/private-data",
            revision="prepared-v1",
            destination=tmp_path,
            relative_paths=("manifest.json",),
            environ={"HF_TOKEN": "hf_download_process_token"},
        )
    assert len(transport.download_calls) == 1


def test_real_transport_downloads_an_allowlist_with_bounded_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transport = HuggingFaceHubTransport()
    revision = "e" * 40
    files = tuple(f"data/episode_{index:06d}.bin" for index in range(32))

    class FakeLibrary:
        @staticmethod
        def snapshot_download(**kwargs: object) -> str:
            snapshot = Path(str(kwargs["local_dir"]))
            for filename in kwargs["allow_patterns"]:
                cached = snapshot / str(filename)
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(str(filename).encode("utf-8"))
            return str(snapshot)

    monkeypatch.setattr(
        transport,
        "_repo_info",
        lambda **_kwargs: SimpleNamespace(sha=revision),
    )
    monkeypatch.setattr(transport, "_library", lambda: FakeLibrary())

    resolved = transport.download_files(
        repository="ryanjin333/lehome-groot-n17-data",
        revision=revision,
        destination=tmp_path / "readback",
        relative_paths=files,
        token="hf_parallel_readback_probe",
    )

    assert resolved == revision
    assert {
        str(path.relative_to(tmp_path / "readback"))
        for path in (tmp_path / "readback").rglob("*.bin")
    } == set(files)


def test_real_transport_uses_exact_nested_release_prefix_allowlist_then_copies_exact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transport = HuggingFaceHubTransport()
    revision = "e" * 40
    destination = tmp_path / "readback"
    remote_prefix = "releases/corrective/immutable-001"
    files = ("manifest.json", "meta/rft-selection.json", "data/episode-000001.bin")
    calls: list[dict[str, object]] = []

    class FakeLibrary:
        @staticmethod
        def snapshot_download(**kwargs: object) -> str:
            calls.append(dict(kwargs))
            snapshot = Path(str(kwargs["local_dir"]))
            for relative_path in (*files, "unrequested/debug.json"):
                target = snapshot / remote_prefix / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(relative_path, encoding="utf-8")
            return str(snapshot)

    monkeypatch.setattr(transport, "_repo_info", lambda **_kwargs: SimpleNamespace(sha=revision))
    monkeypatch.setattr(transport, "_library", lambda: FakeLibrary())

    resolved = transport.download_files(
        repository="ryanjin333/lehome-groot-n17-data",
        revision=revision,
        destination=destination,
        relative_paths=files,
        remote_prefix=remote_prefix,
        token="hf_prefix_filter_probe",
    )

    assert resolved == revision
    assert calls[0]["allow_patterns"] == [f"{remote_prefix}/{path}" for path in files]
    assert {str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()} == set(files)
    assert not (destination / "releases").exists()


def test_real_transport_preserves_partial_prefixed_bytes_when_a_snapshot_rate_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    transport = HuggingFaceHubTransport()
    revision = "e" * 40
    destination = tmp_path / "readback"
    remote_prefix = "bc/full"
    files = ("shards/0000.bin", "shards/0001.bin")

    class FakeLibrary:
        @staticmethod
        def snapshot_download(**kwargs: object) -> str:
            snapshot = Path(str(kwargs["local_dir"]))
            partial = snapshot / remote_prefix / files[0]
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b"partial-but-complete")
            (snapshot / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
            error = RuntimeError("rate limit")
            error.status_code = 429  # type: ignore[attr-defined]
            raise error

    monkeypatch.setattr(transport, "_repo_info", lambda **_kwargs: SimpleNamespace(sha=revision))
    monkeypatch.setattr(transport, "_library", lambda: FakeLibrary())

    with pytest.raises(HubRateLimitError):
        transport.download_files(
            repository="ryanjin333/lehome-groot-n17-data", revision=revision,
            destination=destination, relative_paths=files, remote_prefix=remote_prefix,
            token="hf_partial_readback_probe",
        )

    assert (destination / files[0]).read_bytes() == b"partial-but-complete"
    assert not (destination / "bc").exists()
    assert not (destination / ".cache").exists()


def test_real_transport_reuses_destination_for_snapshot_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    transport = HuggingFaceHubTransport()
    revision = "e" * 40
    destination = tmp_path / "readback"
    calls: list[Path] = []

    class FakeLibrary:
        @staticmethod
        def snapshot_download(**kwargs: object) -> str:
            local = Path(str(kwargs["local_dir"]))
            calls.append(local)
            (local / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
            target = local / str(kwargs["allow_patterns"][0])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"payload")
            return str(local)

    monkeypatch.setattr(transport, "_repo_info", lambda **_kwargs: SimpleNamespace(sha=revision))
    monkeypatch.setattr(transport, "_library", lambda: FakeLibrary())
    transport.download_files(
        repository="ryanjin333/lehome-groot-n17-data", revision=revision,
        destination=destination, relative_paths=("data/episode.bin",), token="hf_resume_probe",
    )

    assert calls == [destination]
    assert not (destination / ".cache").exists()


def test_tree_listing_requires_and_preserves_an_explicit_immutable_revision() -> None:
    transport = FakeTransport()
    revision = "c" * 40

    observed = hub_module.list_repository_tree(
        transport=transport,
        repository="owner/private-data",
        revision=revision,
        environ={"HF_TOKEN": "hf_tree_process_token"},
    )

    assert observed == (hub_module.HubTreeEntry("payload.bin", "file"),)
    assert transport.list_calls == [
        {
            "repository": "owner/private-data",
            "revision": revision,
            "token": "hf_tree_process_token",
        }
    ]
    with pytest.raises(ValueError, match="immutable"):
        hub_module.list_repository_tree(
            transport=transport,
            repository="owner/private-data",
            revision="mutable-branch",
            environ={"HF_TOKEN": "hf_tree_process_token"},
        )
    assert len(transport.list_calls) == 1


def test_approved_private_repository_creation_is_explicit_and_never_public() -> None:
    transport = FakeRepositoryTransport()

    ensure_approved_private_repository(
        transport=transport,
        repository="ryanjin333/lehome-groot-n17-data",
        create=True,
        environ={"HF_TOKEN": "hf_process_memory_only"},
        timeout_seconds=12.0,
    )

    assert transport.ensure_calls == [
        {
            "repository": "ryanjin333/lehome-groot-n17-data",
            "repo_type": "dataset",
            "token": "hf_process_memory_only",
            "create": True,
            "timeout_seconds": 12.0,
        }
    ]
    with pytest.raises(ValueError, match="approved"):
        ensure_approved_private_repository(
            transport=transport,
            repository="owner/public-or-unapproved",
            create=True,
            environ={"HF_TOKEN": "hf_process_memory_only"},
        )


def test_real_transport_is_lazy_and_requires_finite_timeout() -> None:
    transport = HuggingFaceHubTransport(timeout_seconds=15.0)

    assert transport.timeout_seconds == 15.0
    with pytest.raises(ValueError, match="finite positive"):
        HuggingFaceHubTransport(timeout_seconds=float("inf"))


def test_real_transport_configures_a_finite_default_for_every_http_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: list[object] = []

    class FakeSession:
        def request(self, method: str, url: str, **kwargs: object) -> object:
            return kwargs["timeout"]

    fake_hub = SimpleNamespace(
        configure_http_backend=lambda *, backend_factory: configured.append(
            backend_factory
        )
    )

    def fake_import(name: str) -> object:
        if name == "huggingface_hub":
            return fake_hub
        if name == "requests":
            return SimpleNamespace(Session=FakeSession)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("lehome_train.hub.importlib.import_module", fake_import)
    transport = HuggingFaceHubTransport(timeout_seconds=17.0)

    assert transport._library() is fake_hub
    session = configured[0]()
    assert session.request("GET", "https://example.invalid") == 17.0
    assert session.request("GET", "https://example.invalid", timeout=None) == 17.0
    assert session.request("GET", "https://example.invalid", timeout=3.0) == 3.0


def test_real_transport_lists_complete_tree_at_explicit_commit_with_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "d" * 40
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeApi:
        def list_repo_tree(self, **kwargs: object) -> list[object]:
            calls.append(("tree", kwargs))
            return [
                SimpleNamespace(path="experiments", tree_id="tree-root"),
                SimpleNamespace(
                    path="experiments/run/payload.bin",
                    size=7,
                    blob_id="blob",
                ),
            ]

        def list_repo_files(self, **kwargs: object) -> list[str]:
            calls.append(("files", kwargs))
            return ["experiments/run/payload.bin"]

    transport = HuggingFaceHubTransport(timeout_seconds=19.0)
    monkeypatch.setattr(transport, "_api", lambda token: FakeApi())
    monkeypatch.setattr(
        transport,
        "_repo_info",
        lambda **_kwargs: SimpleNamespace(sha=revision),
    )

    observed = transport.list_tree(
        repository="ryanjin333/lehome-groot-n17-models",
        revision=revision,
        token="hf_explicit_tree_token",
    )

    assert observed == (
        hub_module.HubTreeEntry("experiments", "directory"),
        hub_module.HubTreeEntry("experiments/run/payload.bin", "file"),
    )
    expected_common = {
        "repo_id": "ryanjin333/lehome-groot-n17-models",
        "repo_type": "model",
        "revision": revision,
        "token": "hf_explicit_tree_token",
    }
    assert calls == [
        ("tree", {**expected_common, "recursive": True, "expand": True}),
        ("files", expected_common),
    ]


def test_hub_tree_entry_allows_repository_dotfiles_but_rejects_aliases() -> None:
    assert hub_module.HubTreeEntry(".gitattributes", "file").relative_path == ".gitattributes"
    assert hub_module.HubTreeEntry("nested/.metadata", "file").relative_path == "nested/.metadata"
    for unsafe in ("./payload", "nested/../payload", "/payload", "nested//payload"):
        with pytest.raises(ValueError, match="canonical and relative"):
            hub_module.HubTreeEntry(unsafe, "file")
