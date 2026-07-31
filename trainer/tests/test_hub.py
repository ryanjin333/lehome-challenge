from __future__ import annotations

from pathlib import Path

import pytest

from lehome_train.hub import (
    HubAccess,
    HubTransientError,
    download_files,
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
