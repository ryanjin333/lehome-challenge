from __future__ import annotations

import os
import math
from pathlib import Path

import pytest

from b1k_rollout.provenance import (
    ProvenanceAuthenticationError,
    ProvenanceAuthenticator,
    load_local_provenance_key,
)


def test_key_loader_rejects_a_symlinked_parent_directory(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    key = real / "key"
    key.write_bytes(b"k" * 32)
    os.chmod(key, 0o600)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(ProvenanceAuthenticationError, match="key file is invalid"):
        load_local_provenance_key(alias / "key")


@pytest.mark.parametrize("issuer", ["contains space", ("Bearer hf_" + "abcdefghijklmnopqrstuvwxyzABCDEFGH"), "line\nbreak"])
def test_authenticator_rejects_unsafe_issuer_without_echoing_it(issuer: str) -> None:
    with pytest.raises(ProvenanceAuthenticationError) as raised:
        ProvenanceAuthenticator(b"k" * 32, issuer=issuer)

    assert issuer not in str(raised.value)


def test_authenticator_deterministically_attests_positive_infinity_payloads() -> None:
    authenticator = ProvenanceAuthenticator(b"i" * 32, issuer="infinity-test")
    payload = {"metrics": {"normalized_time": math.inf}, "schema_version": 3}

    attestation = authenticator.sign(payload)

    authenticator.verify(payload, attestation)
    with pytest.raises(ProvenanceAuthenticationError):
        authenticator.verify({"metrics": {"normalized_time": 1.0}, "schema_version": 3}, attestation)
