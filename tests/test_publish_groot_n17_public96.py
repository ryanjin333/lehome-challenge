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


class HubTransientError(RuntimeError):
    """Named for the canonical bounded-retry classifier."""


class Transport:
    def __init__(self) -> None:
        self.store: dict[str, dict[str, bytes]] = {}
        self.upload_calls = 0
        self.authenticated_downloads = 0
        self.anonymous_downloads = 0
        self.download_revisions: list[str] = []
        self.mutate_authenticated = False
        self.mutate_anonymous = False
        self.transient: dict[str, int] = {}
        self.lost_upload_response = False
        self.ambiguous_upload_mismatch = False

    def _transient(self, operation: str) -> None:
        if self.transient.get(operation, 0):
            self.transient[operation] -= 1
            raise HubTransientError(f"transient {operation}")

    def resolve_approved_ref(self, *, repository, ref, token, **_kwargs):
        assert repository == "owner/public96" and ref == "main" and token == "token"
        self._transient("resolve")
        return COMMIT

    def list_tree(self, *, repository, revision, token, remote_prefix=None, **_kwargs):
        assert repository == "owner/public96" and revision == COMMIT and token == "token"
        self._transient("list")
        return tuple(Entry(path) for path in sorted(self.store.get(str(remote_prefix), {})))

    def upload_files(self, *, repository, revision, source, entries, token, remote_prefix=None, parent_commit=None, **_kwargs):
        assert repository == "owner/public96" and revision == "main" and token == "token" and parent_commit == COMMIT
        self.upload_calls += 1
        bucket = self.store.setdefault(str(remote_prefix), {})
        for entry in entries:
            bucket[f"{remote_prefix}/{entry.relative_path}"] = (Path(source) / entry.relative_path).read_bytes()
        if self.ambiguous_upload_mismatch:
            bucket[f"{remote_prefix}/result.json"] = b"different"
            raise ConnectionError("unknown upload result")
        if self.lost_upload_response:
            self.lost_upload_response = False
            raise ConnectionError("unknown upload result")
        return COMMIT

    def download_files(self, *, repository, revision, destination, relative_paths, token, remote_prefix=None, **_kwargs):
        assert repository == "owner/public96" and revision == COMMIT
        self.download_revisions.append(revision)
        if token is None:
            self.anonymous_downloads += 1
            self._transient("anonymous_download")
        else:
            self.authenticated_downloads += 1
            self._transient("authenticated_download")
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
    published_paths = {entry.relative_path for entry in result.entries}
    result_body = json.loads((root / "result.json").read_text(encoding="utf-8"))
    expected_paths = {"result.json", "verifier-receipt.json", "policy-server-readiness.json", "policy-server.log", "SHA256SUMS.json"}
    expected_paths.update(f"{stage['stage_id']}/stage.log" for stage in result_body["episodes"][::2])
    expected_paths.update(f"{stage['stage_id']}/stage-receipt.json" for stage in result_body["episodes"][::2])
    expected_paths.update(
        descriptor["relative_path"]
        for episode in result_body["episodes"]
        for descriptor in episode["artifacts"]["videos"].values()
    )
    assert published_paths == expected_paths and len(expected_paths) == 389
    assert "garment-config" not in published_paths
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


@pytest.mark.parametrize("operation", ("resolve", "list", "authenticated_download", "anonymous_download"))
def test_publisher_retries_bounded_transient_transport_operations(operation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport(); transport.transient[operation] = 1
    result = _publish(module, root, transport)
    assert result.immutable_revision == COMMIT and transport.transient[operation] == 0


def test_publisher_reconciles_lost_upload_response_only_after_exact_immutable_readback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport(); transport.lost_upload_response = True
    result = _publish(module, root, transport)
    assert result.immutable_revision == COMMIT and transport.upload_calls == 1 and transport.authenticated_downloads >= 2


def test_publisher_rejects_mismatching_ambiguous_upload_without_reupload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport(); transport.ambiguous_upload_mismatch = True
    with pytest.raises(module.Public96PublicationError, match="ambiguous upload"):
        _publish(module, root, transport)
    assert transport.upload_calls == 1 and not (root / "public96-publication-receipt.json").exists()


@pytest.mark.parametrize("relative", ("top-long-seen-0/stage.log", "top-long-seen-0/videos/success/episode0_observation_images_top_rgb.mp4"))
def test_publisher_rejects_credential_shaped_public_bytes_before_transport(relative: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport()
    (root / relative).write_bytes(b"credential=secretvalue123")
    monkeypatch.setattr(module.evaluator, "verify_result", lambda result, **_kwargs: result["summary"])
    with pytest.raises(module.Public96PublicationError, match="credential|publication source") as error:
        _publish(module, root, transport)
    assert "secretvalue123" not in str(error.value) + capsys.readouterr().out + capsys.readouterr().err
    assert transport.upload_calls == 0


def test_publisher_rejects_root_artifact_mutation_after_verification_before_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport()
    original = module._verify_verifier_receipt

    def mutate_after_receipt(*args, **kwargs):
        verified = original(*args, **kwargs)
        (root / "policy-server.log").write_text("mutated after verification\n", encoding="utf-8")
        return verified

    monkeypatch.setattr(module, "_verify_verifier_receipt", mutate_after_receipt)
    with pytest.raises(module.Public96PublicationError, match="changed|descriptor"):
        _publish(module, root, transport)
    assert transport.upload_calls == 0 and not (root / "public96-publication-receipt.json").exists()


def test_token_file_requires_owner_only_regular_file_and_descriptor_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); token = tmp_path / "token"; token.write_text("token", encoding="utf-8"); token.chmod(0o644)
    with pytest.raises(module.Public96PublicationError, match="owner-only"):
        module.load_token(token)
    token.chmod(0o600)
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("token must be read from an opened descriptor")))
    assert module.load_token(token) == "token"
    token.unlink(); token.symlink_to(tmp_path / "replacement")
    with pytest.raises(module.Public96PublicationError, match="owner-only|unavailable"):
        module.load_token(token)


def test_token_load_keeps_open_descriptor_when_name_is_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); token = tmp_path / "token"; token.write_text("original", encoding="utf-8"); token.chmod(0o600)
    replacement = tmp_path / "replacement"; replacement.write_text("replacement", encoding="utf-8"); replacement.chmod(0o600)
    real_open = module.os.open
    opened = False

    def open_then_replace(path, flags, *args, **kwargs):
        nonlocal opened
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == token:
            opened = True
            os.replace(replacement, token)
        return descriptor

    monkeypatch.setattr(module.os, "open", open_then_replace)
    assert module.load_token(token) == "original" and opened


def test_existing_publication_receipt_requires_exact_schema_and_timestamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport()
    _publish(module, root, transport)
    receipt_path = root / "public96-publication-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")); receipt["unexpected"] = True
    _write_json(receipt_path, receipt)
    with pytest.raises(module.Public96PublicationError, match="already exists"):
        _publish(module, root, transport)
    receipt.pop("unexpected"); receipt["published_at_utc"] = "not-a-rfc3339-time"
    _write_json(receipt_path, receipt)
    with pytest.raises(module.Public96PublicationError, match="already exists"):
        _publish(module, root, transport)


def test_receipt_rejects_symlinked_parent_and_never_overwrites_race(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport()
    real_parent = tmp_path / "real"; real_parent.mkdir(); alias = tmp_path / "alias"; alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(module.Public96PublicationError, match="receipt.*symlink|receipt parent"):
        _publish(module, root, transport, receipt_output=alias / "receipt.json")
    target = real_parent / "receipt.json"; target.write_text("other process receipt", encoding="utf-8")
    with pytest.raises(module.Public96PublicationError, match="already exists"):
        _publish(module, root, transport, receipt_output=target)
    assert target.read_text(encoding="utf-8") == "other process receipt"


def test_receipt_parent_fd_cannot_escape_when_an_ancestor_is_swapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport()
    ancestor = tmp_path / "receipt-ancestor"; parent = ancestor / "out"; parent.mkdir(parents=True)
    outside = tmp_path / "outside"; (outside / "out").mkdir(parents=True)
    moved = tmp_path / "moved-ancestor"; receipt = parent / "receipt.json"
    real_open = module.os.open; swapped = False; parent_opens = 0

    def swap_before_parent_open(path, flags, *args, **kwargs):
        nonlocal swapped, parent_opens
        # The old implementation re-opened the full parent path after its
        # lstat walk. The fixed one opens `out` under a pinned ancestor fd.
        if Path(path) == parent and "dir_fd" not in kwargs:
            parent_opens += 1
            if parent_opens == 2:
                os.replace(ancestor, moved); ancestor.symlink_to(outside, target_is_directory=True); swapped = True
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == "out" and "dir_fd" in kwargs:
            parent_opens += 1
            if parent_opens == 2:
                os.replace(ancestor, moved); ancestor.symlink_to(outside, target_is_directory=True); swapped = True
        return descriptor

    monkeypatch.setattr(module.os, "open", swap_before_parent_open)
    try:
        _publish(module, root, transport, receipt_output=receipt)
    except module.Public96PublicationError:
        pass
    assert swapped and not (outside / "out" / "receipt.json").exists()


def test_receipt_directory_fsync_failure_is_not_reported_as_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport()
    original_fsync = module.os.fsync
    def fail_parent_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_parent_fsync)
    with pytest.raises(module.Public96PublicationError, match="receipt"):
        _publish(module, root, transport)


def test_cli_normalizes_transport_or_dependency_exceptions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module = _module(); token = tmp_path / "token"; token.write_text("secret-token", encoding="utf-8"); token.chmod(0o600)
    monkeypatch.setattr(module, "publish_public96", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("network secret-token detail")))
    assert module.main(["--run-root", str(tmp_path), "--matrix", str(MATRIX), "--matrix-sha256", str(MATRIX_SHA256), "--token-file", str(token)]) == 2
    output = capsys.readouterr()
    assert output.out == "" and output.err == "public96 publication failed: transport or dependency error\n"


def test_publication_receipt_never_overwrites_and_identical_resume_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); root = _fixture(module, tmp_path, monkeypatch); transport = Transport()
    _publish(module, root, transport)
    before = (root / "public96-publication-receipt.json").read_bytes()
    _publish(module, root, transport)
    assert (root / "public96-publication-receipt.json").read_bytes() == before
    (root / "other-receipt.json").write_text("untrusted", encoding="utf-8")
    with pytest.raises(module.Public96PublicationError, match="already exists"):
        _publish(module, root, transport, receipt_output=root / "other-receipt.json")
