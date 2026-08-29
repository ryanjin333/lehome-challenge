"""Focused offline contract tests for the public96 evidence publisher."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "publish_groot_n17_public96.py"
MATRIX = ROOT / "configs" / "eval_groot_n17_public96_reference.json"
MATRIX_SHA256 = ROOT / "configs" / "eval_groot_n17_public96_reference.json.sha256"
COMMIT = "a" * 40


def test_public96_publisher_module_exists() -> None:
    """The dedicated publisher is an explicit post-run boundary."""
    assert SCRIPT.is_file()


def _module():
    assert SCRIPT.is_file()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("public96_publisher", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Entry:
    def __init__(self, relative_path: str, entry_type: str = "file") -> None:
        self.relative_path = relative_path
        self.entry_type = entry_type


class Transport:
    def __init__(self) -> None:
        self.store: dict[str, dict[str, bytes]] = {}
        self.upload_calls = 0
        self.authenticated_downloads = 0
        self.anonymous_downloads = 0
        self.download_revisions: list[str] = []
        self.mutate_authenticated = False
        self.mutate_anonymous = False

    def resolve_approved_ref(self, *, repository, ref, token, **_kwargs):
        assert repository == "owner/public96" and ref == "main" and token == "token"
        return COMMIT

    def list_tree(self, *, repository, revision, token, remote_prefix=None, **_kwargs):
        assert repository == "owner/public96" and revision == COMMIT and token == "token"
        return tuple(Entry(path) for path in sorted(self.store.get(str(remote_prefix), {})))

    def upload_files(self, *, repository, revision, source, entries, token, remote_prefix=None, parent_commit=None, **_kwargs):
        assert repository == "owner/public96" and revision == "main" and token == "token" and parent_commit == COMMIT
        self.upload_calls += 1
        bucket = self.store.setdefault(str(remote_prefix), {})
        for entry in entries:
            bucket[f"{remote_prefix}/{entry.relative_path}"] = (Path(source) / entry.relative_path).read_bytes()
        return COMMIT

    def download_files(self, *, repository, revision, destination, relative_paths, token, remote_prefix=None, **_kwargs):
        assert repository == "owner/public96" and revision == COMMIT
        self.download_revisions.append(revision)
        if token is None:
            self.anonymous_downloads += 1
        else:
            self.authenticated_downloads += 1
        bucket = self.store[str(remote_prefix)]
        for relative in relative_paths:
            raw = bucket[f"{remote_prefix}/{relative}"]
            if (token is not None and self.mutate_authenticated) or (token is None and self.mutate_anonymous):
                raw = b"mismatch"
            target = Path(destination) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        return COMMIT


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import scripts.eval_groot_n17_public96 as evaluator
    fixture_spec = importlib.util.spec_from_file_location("public96_evaluator_fixture", ROOT / "tests" / "test_eval_groot_n17_public96.py")
    assert fixture_spec and fixture_spec.loader
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    sys.modules[fixture_spec.name] = fixture_module
    fixture_spec.loader.exec_module(fixture_module)

    run_root = tmp_path / "closed-run"; run_root.mkdir()
    policy = fixture_module._policy_root_with_canonical_artifact(tmp_path)
    stages = evaluator.load_frozen_matrix(MATRIX, MATRIX_SHA256)
    result = fixture_module._valid_result(stages, run_root, policy)
    _write_json(run_root / "result.json", result)
    result_descriptor = {"relative_path": "result.json", "sha256": hashlib.sha256((run_root / "result.json").read_bytes()).hexdigest()}
    receipt = {
        "kind": "lehome_groot_n17_public96_verifier_receipt_v1",
        "result": result_descriptor,
        "policy_server_log": {"relative_path": "policy-server.log", "sha256": ""},
        "summary": result["summary"], "matrix_sha256": result["matrix_sha256"],
        "checkpoint": result["checkpoint"], "raw_checker_overlay": result["raw_checker_overlay"],
        "publication": result["publication"],
    }
    (run_root / "policy-server.log").write_text("server complete\n", encoding="utf-8")
    receipt["policy_server_log"]["sha256"] = hashlib.sha256((run_root / "policy-server.log").read_bytes()).hexdigest()
    _write_json(run_root / "verifier-receipt.json", receipt)
    monkeypatch.setattr(evaluator, "canonical_policy_artifact_sha256", lambda _: evaluator.CHECKPOINT["artifact_sha256"])
    return run_root


def _publish(module, root: Path, transport: Transport, **kwargs):
    return module.publish_public96(
        root, matrix=MATRIX, matrix_sha256_path=MATRIX_SHA256, repository="owner/public96",
        ref="main", token="token", transport=transport, **kwargs,
    )


def test_publisher_stages_exact_allowlist_manifest_and_two_readbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport()

    result = _publish(module, root, transport)

    assert result.entry_count == 389  # 4 root artifacts + 48 logs + 48 receipts + 288 videos + manifest
    assert result.remote_prefix == f"public96/results/{result.matrix_sha256[:16]}-{result.result_sha256[:16]}"
    assert "garment-config" not in {entry.relative_path for entry in result.entries}
    assert "SHA256SUMS.json" in {entry.relative_path for entry in result.entries}
    assert transport.upload_calls == 1 and transport.authenticated_downloads == 1 and transport.anonymous_downloads == 1
    receipt = json.loads((root / "public96-publication-receipt.json").read_text(encoding="utf-8"))
    assert receipt["immutable_revision"] == COMMIT
    assert receipt["authenticated_readback_verified"] is True and receipt["anonymous_readback_verified"] is True


@pytest.mark.parametrize("mutation, message", (("receipt", "verifier receipt"), ("result", "result verifier")))
def test_publisher_rejects_result_or_verifier_receipt_mismatch_before_transport(mutation: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport()
    if mutation == "receipt":
        receipt = json.loads((root / "verifier-receipt.json").read_text(encoding="utf-8")); receipt["summary"] = {"bad": True}
        _write_json(root / "verifier-receipt.json", receipt)
    else:
        result = json.loads((root / "result.json").read_text(encoding="utf-8")); result["summary"]["overall"]["successes"] = 0
        _write_json(root / "result.json", result)

    with pytest.raises(module.Public96PublicationError, match=message):
        _publish(module, root, transport)
    assert transport.upload_calls == 0


@pytest.mark.parametrize("mutation", ("symlink", "traversal"))
def test_publisher_rejects_unsafe_referenced_paths_before_transport(mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport()
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    descriptor = result["episodes"][0]["artifacts"]["videos"]["top_rgb"]
    if mutation == "symlink":
        video = root / descriptor["relative_path"]; outside = tmp_path / "outside.mp4"; outside.write_bytes(video.read_bytes())
        video.unlink(); video.symlink_to(outside)
    else:
        descriptor["relative_path"] = "../outside.mp4"; (tmp_path / "outside.mp4").write_bytes(b"outside")
    _write_json(root / "result.json", result)

    with pytest.raises(module.Public96PublicationError):
        _publish(module, root, transport)
    assert transport.upload_calls == 0


def test_publisher_rejects_immutable_prefix_collision_and_resumes_exact_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport()
    first = _publish(module, root, transport)
    second = _publish(module, root, transport)
    assert second.immutable_revision == first.immutable_revision and transport.upload_calls == 1
    transport.store[first.remote_prefix][f"{first.remote_prefix}/unexpected.txt"] = b"no"
    with pytest.raises(module.Public96PublicationError, match="collision"):
        _publish(module, root, transport, receipt_output=root / "second-receipt.json")


@pytest.mark.parametrize("which", ("authenticated", "anonymous"))
def test_publisher_fails_closed_when_independent_readback_bytes_differ(which: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport()
    setattr(transport, f"mutate_{which}", True)
    with pytest.raises(module.Public96PublicationError, match="readback"):
        _publish(module, root, transport)
    assert not (root / "public96-publication-receipt.json").exists()


def test_publisher_binds_every_remote_operation_to_returned_immutable_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport()
    result = _publish(module, root, transport)
    assert transport.download_revisions == [result.immutable_revision, result.immutable_revision]


def test_token_file_requires_owner_only_regular_file(tmp_path: Path) -> None:
    module = _module(); token = tmp_path / "token"; token.write_text("token", encoding="utf-8"); token.chmod(0o644)
    with pytest.raises(module.Public96PublicationError, match="owner-only"):
        module.load_token(token)
    token.chmod(0o600)
    assert module.load_token(token) == "token"
    token.unlink(); token.symlink_to(tmp_path / "replacement")
    with pytest.raises(module.Public96PublicationError, match="owner-only|unavailable"):
        module.load_token(token)


def test_publication_receipt_never_overwrites_and_identical_resume_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport()
    _publish(module, root, transport)
    before = (root / "public96-publication-receipt.json").read_bytes()
    _publish(module, root, transport)
    assert (root / "public96-publication-receipt.json").read_bytes() == before
    (root / "other-receipt.json").write_text("untrusted", encoding="utf-8")
    with pytest.raises(module.Public96PublicationError, match="already exists"):
        _publish(module, root, transport, receipt_output=root / "other-receipt.json")
