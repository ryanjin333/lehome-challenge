"""Fail-closed operational gates for the digest-pinned rollout image."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from pathlib import PurePosixPath

from b1k_rollout.checkpoint import FinalPolicyMaterializer, MaterializationError
from b1k_rollout.contracts import RolloutContract
from b1k_rollout.controller import (
    RolloutController,
    SubprocessGroupLauncher,
    bounded_http_health_probe,
)
from b1k_rollout.identity import DATASET_REPO, MODEL_REPO, require_image_digest, require_immutable_commit, require_sha256
from b1k_rollout.provenance import ProvenanceAuthenticator
from b1k_rollout.publisher import publish_release
from b1k_rollout.task_manifest import load_task_manifest


_BEHAVIOR_ASSET_VERSION = "3.9.0"
_MINIMUM_ROBOT_ASSET_VERSION = "3.8.2"
_PUBLIC_INSTANCE_IDS = tuple(range(301, 311))
_DEFAULT_TASK_MANIFEST_PATH = Path("/opt/rollout/task-manifest.json")
_ASSET_DATASET_REPOSITORY = "behavior-1k/zipped-datasets"
_ASSET_DATASET_REVISION = "9f0d57d465726976ed98138d3f8b8ca3e2186775"
_EXPECTED_CAMPAIGN_CONTENT_ROOT = "14a87fdfeefa04e9d1cc2035bc6eff424ea5f916b84d67b04d141a6366b20aec"
_ASSET_RECEIPTS_DIRECTORY = ".b1k-asset-receipts"


@dataclass(frozen=True, slots=True)
class PinnedAssetArchive:
    directory: str
    filename: str
    sha256: str


_PINNED_ASSET_ARCHIVES = (
    PinnedAssetArchive("omnigibson-robot-assets", "omnigibson-robot-assets-3.8.2.zip", "3d813b2181e0581cf2300a40892de70f8475fe59346ceeea4fb9bf7ff21ce126"),
    PinnedAssetArchive("behavior-1k-assets", "behavior-1k-assets-3.9.0.zip", "09e9fce600f841dc611aa96c0b1b9f9074f56f0e67898b37b39bd00c38a0095e"),
    PinnedAssetArchive("2026-challenge-task-instances", "2026-challenge-task-instances.zip", "d4ac3d72dd585178e85d542f28d1b48233b886c80e196e2f9d3aa5993ba21f81"),
)


class CliError(RuntimeError):
    """An operator-visible preflight failure that contains no credential material."""


@dataclass(frozen=True, slots=True)
class PreflightResult:
    token_file: Path
    data_path: Path
    image_digest: str

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def preflight(*, environment: dict[str, str] | None = None) -> PreflightResult:
    """Validate only non-secret runtime identity before any remote operation."""

    env = dict(os.environ if environment is None else environment)
    if env.get("AUTO_DESTROY") != "0":
        raise CliError("AUTO_DESTROY must be exactly 0")
    if env.get("OMNI_KIT_ACCEPT_EULA") != "YES":
        raise CliError("OMNI_KIT_ACCEPT_EULA must be YES")
    token_file = _private_regular_file(env.get("B1K_HF_TOKEN_FILE"), label="B1K_HF_TOKEN_FILE")
    data_path = _directory(env.get("OMNIGIBSON_DATA_PATH"), label="OMNIGIBSON_DATA_PATH")
    _validate_official_assets(
        data_path,
        task_manifest_path=Path(env.get("TASK_MANIFEST_PATH", _DEFAULT_TASK_MANIFEST_PATH)),
    )
    image_digest = env.get("CONTAINER_DIGEST")
    try:
        require_image_digest(image_digest)
    except ValueError as error:
        raise CliError("CONTAINER_DIGEST must be an immutable image digest") from error
    return PreflightResult(token_file=token_file, data_path=data_path, image_digest=image_digest)


def bootstrap_assets(*, environment: dict[str, str] | None = None) -> None:
    """Fetch every official licensed asset set, then prove the pinned evaluator layout."""

    env = dict(os.environ if environment is None else environment)
    data_path = _directory(env.get("OMNIGIBSON_DATA_PATH"), label="OMNIGIBSON_DATA_PATH")
    if env.get("B1K_ROLLOUT_VERIFY_PRIVILEGE_DROP") == "1":
        return
    token_file = _private_regular_file(env.get("B1K_HF_TOKEN_FILE"), label="B1K_HF_TOKEN_FILE")
    _bootstrap_behavior_license_key()
    for archive in _PINNED_ASSET_ARCHIVES:
        _install_pinned_asset(data_path, archive=archive, token_file=token_file)
    _validate_official_assets(
        data_path,
        task_manifest_path=Path(env.get("TASK_MANIFEST_PATH", _DEFAULT_TASK_MANIFEST_PATH)),
    )


def bootstrap_checkpoint(*, environment: dict[str, str] | None = None) -> Path:
    """Materialize only the final-manifest-selected immutable checkpoint tree."""

    env = dict(os.environ if environment is None else environment)
    ready = preflight(environment=env)
    if env.get("HF_MODEL_REPO") != MODEL_REPO:
        raise CliError("HF_MODEL_REPO must be the approved B1K private model repository")
    try:
        model_commit = require_immutable_commit(env.get("MODEL_COMMIT"), label="MODEL_COMMIT")
        artifact_sha256 = require_sha256(
            env.get("CHECKPOINT_ARTIFACT_SHA256"), label="CHECKPOINT_ARTIFACT_SHA256"
        )
    except ValueError as error:
        raise CliError("checkpoint identity must use immutable commit and SHA-256 values") from error
    try:
        destination = Path(env.get("CHECKPOINT_DIR", "/workspace/campaign/checkpoint"))
        receipt = FinalPolicyMaterializer(
            hub=HuggingFaceModelAdapter(ready.token_file), expected_manifest_sha256=artifact_sha256
        ).download(repository=MODEL_REPO, revision=model_commit, destination=destination)
    except (MaterializationError, OSError) as error:
        raise CliError("immutable final checkpoint materialization failed") from error
    return receipt.local_path


def healthcheck(*, environment: dict[str, str] | None = None) -> None:
    """Probe the local policy process without logging requests or credentials."""

    env = dict(os.environ if environment is None else environment)
    url = env.get("POLICY_HEALTH_URL", "http://127.0.0.1:8000/healthz")
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            if response.status != 200:
                raise CliError("policy health endpoint did not return HTTP 200")
    except (OSError, urllib.error.URLError) as error:
        raise CliError("policy health endpoint is unavailable") from error


class HuggingFaceDatasetAdapter:
    """Concrete private dataset transport used only by the production CLI."""

    def __init__(self, token_file: Path) -> None:
        self._token_file = token_file
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except Exception as error:  # pragma: no cover - image dependency gate
            raise CliError("rollout image is missing the Hub publication dependency") from error
        self._api = HfApi(token=self._token())
        self._download = hf_hub_download

    def get_dataset_info(self, repo_id: str) -> object:
        self._require_repo(repo_id)
        try:
            info = self._api.repo_info(repo_id=repo_id, repo_type="dataset", token=self._token(), timeout=30)
            return {"private": getattr(info, "private", None), "sha": getattr(info, "sha", None)}
        except Exception as error:
            raise CliError("private rollout dataset lookup failed") from error

    def list_tree(self, repo_id: str, *, revision: str, prefix: str) -> dict[str, str]:
        self._require_repo(repo_id)
        try:
            entries = self._api.list_repo_tree(repo_id=repo_id, repo_type="dataset", revision=revision, recursive=True, token=self._token())
            paths = [getattr(entry, "path", None) for entry in entries]
            selected = sorted(path for path in paths if isinstance(path, str) and path.startswith(prefix + "/"))
            return {path: _file_sha256(Path(self._download(repo_id=repo_id, repo_type="dataset", filename=path, revision=revision, token=self._token(), force_download=True))) for path in selected}
        except Exception as error:
            raise CliError("immutable rollout dataset tree readback failed") from error

    def upload_tree(self, repo_id: str, *, local_dir: Path, remote_prefix: str, commit_message: str) -> str:
        self._require_repo(repo_id)
        try:
            receipt = self._api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(local_dir), path_in_repo=remote_prefix, commit_message=commit_message, token=self._token())
            return _commit_id(receipt)
        except Exception as error:
            raise CliError("rollout dataset staging upload failed") from error

    def promote_prefix(self, repo_id: str, *, staging_prefix: str, release_prefix: str, commit_message: str) -> str:
        self._require_repo(repo_id)
        try:
            from huggingface_hub import CommitOperationAdd, CommitOperationDelete
            entries = list(self._api.list_repo_tree(repo_id=repo_id, repo_type="dataset", revision="main", recursive=True, token=self._token()))
            staged = sorted(getattr(entry, "path") for entry in entries if isinstance(getattr(entry, "path", None), str) and getattr(entry, "path").startswith(staging_prefix + "/"))
            if not staged:
                raise CliError("rollout dataset staging tree is absent")
            operations = []
            for path in staged:
                local = self._download(repo_id=repo_id, repo_type="dataset", filename=path, revision="main", token=self._token(), force_download=True)
                operations.append(CommitOperationAdd(path_in_repo=release_prefix + path.removeprefix(staging_prefix), path_or_fileobj=local))
                operations.append(CommitOperationDelete(path_in_repo=path))
            return _commit_id(self._api.create_commit(repo_id=repo_id, repo_type="dataset", operations=operations, commit_message=commit_message, token=self._token()))
        except CliError:
            raise
        except Exception as error:
            raise CliError("rollout dataset promotion failed") from error

    def delete_prefix(self, repo_id: str, *, prefix: str) -> str:
        self._require_repo(repo_id)
        try:
            from huggingface_hub import CommitOperationDelete
            entries = list(self._api.list_repo_tree(repo_id=repo_id, repo_type="dataset", revision="main", recursive=True, token=self._token()))
            paths = [getattr(entry, "path") for entry in entries if isinstance(getattr(entry, "path", None), str) and getattr(entry, "path").startswith(prefix + "/")]
            return _commit_id(self._api.create_commit(repo_id=repo_id, repo_type="dataset", operations=[CommitOperationDelete(path_in_repo=path) for path in paths], commit_message=f"remove incomplete B1K rollout {prefix}", token=self._token()))
        except Exception as error:
            raise CliError("rollout dataset staging cleanup failed") from error

    def download_file_to_path(self, repo_id: str, *, revision: str, path: str, destination: Path) -> None:
        self._require_repo(repo_id)
        try:
            source = Path(self._download(repo_id=repo_id, repo_type="dataset", filename=path, revision=revision, token=self._token(), force_download=True))
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        except Exception as error:
            raise CliError("immutable rollout dataset file readback failed") from error

    def _token(self) -> str:
        try:
            token = self._token_file.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise CliError("rollout token file became unreadable") from error
        if not token:
            raise CliError("rollout token file is empty")
        return token

    @staticmethod
    def _require_repo(repo_id: str) -> None:
        if repo_id != DATASET_REPO:
            raise CliError("rollout dataset repository is invalid")


class HuggingFaceModelAdapter:
    """Stream exact immutable model-repository files without broad snapshots."""

    def __init__(self, token_file: Path) -> None:
        self._token_file = token_file
        try:
            from huggingface_hub import hf_hub_download
        except Exception as error:  # pragma: no cover - image dependency gate
            raise CliError("rollout image is missing the Hub model download dependency") from error
        self._download = hf_hub_download

    def open_file(self, repository: str, *, revision: str, path: str):
        if repository != MODEL_REPO:
            raise CliError("rollout model repository is invalid")
        if path != "final-manifest.json" and not path.startswith("checkpoint/"):
            raise CliError("rollout model path is invalid")
        try:
            source = Path(
                self._download(
                    repo_id=repository,
                    repo_type="model",
                    filename=path,
                    revision=revision,
                    token=self._token(),
                    force_download=True,
                )
            )
            return source.open("rb")
        except CliError:
            raise
        except Exception as error:
            raise CliError("immutable rollout model file download failed") from error

    def _token(self) -> str:
        try:
            token = self._token_file.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise CliError("rollout token file became unreadable") from error
        if not token:
            raise CliError("rollout token file is empty")
        return token


def run_campaign(environment: dict[str, str] | None = None) -> object | None:
    """Compose the controller, concrete local checkpoint transport, and Hub publisher."""

    env = dict(os.environ if environment is None else environment)
    ready = preflight(environment=env)
    try:
        contract = RolloutContract.from_mapping({
            "behavior_revision": env["BEHAVIOR_REVISION"], "groot_revision": env["GROOT_REVISION"],
            "model_repository": env["HF_MODEL_REPO"], "model_commit": env["MODEL_COMMIT"],
            "dataset_repository": env["HF_DATASET_REPO"], "image_digest": env["CONTAINER_DIGEST"],
            "run_id": env["RUN_ID"], "cycle_id": env["CYCLE_ID"], "campaign_id": env["CAMPAIGN_ID"],
            "evaluator_mode": env["EVALUATOR_MODE"], "task_manifest_sha256": env["TASK_MANIFEST_SHA256"],
            "checkpoint_artifact_sha256": env["CHECKPOINT_ARTIFACT_SHA256"], "auto_destroy": env["AUTO_DESTROY"],
        })
    except (KeyError, ValueError) as error:
        raise CliError("campaign contract is incomplete or invalid") from error
    checkpoint_dir = Path(env.get("CHECKPOINT_DIR", "/workspace/campaign/checkpoint"))
    gpu_ids = _gpu_ids(env.get("GPU_IDS"))
    key = _provenance_key(Path(env.get("B1K_PROVENANCE_KEY_FILE", "/workspace/.b1k-provenance.key")))
    controller = RolloutController(
        contract=contract, task_manifest=load_task_manifest(Path(env.get("TASK_MANIFEST_PATH", "/opt/rollout/task-manifest.json"))),
        checkpoint_source=FinalPolicyMaterializer(
            hub=HuggingFaceModelAdapter(ready.token_file),
            expected_manifest_sha256=contract.checkpoint_artifact_sha256,
        ),
        checkpoint_dir=checkpoint_dir, workspace=Path(env.get("CAMPAIGN_WORKSPACE", "/workspace/campaign")),
        gpu_ids=gpu_ids, launcher=SubprocessGroupLauncher(), health_probe=bounded_http_health_probe,
        policy_command=(sys.executable, "-m", "b1k_rollout.policy_server"),
        provenance_authenticator=ProvenanceAuthenticator(key, issuer=f"b1k-{contract.campaign_id}"),
    )
    hub = HuggingFaceDatasetAdapter(ready.token_file)
    return controller.run(publish=lambda **kwargs: publish_release(hub=hub, **kwargs))


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="b1k-rollout")
    command = parser.add_subparsers(dest="command", required=True)
    command.add_parser("assets-bootstrap")
    command.add_parser("preflight")
    smoke_runtime = command.add_parser("smoke-runtime")
    smoke_runtime.add_argument("--success-prefix", required=True)
    smoke_runtime.add_argument("--failure-prefix", required=True)
    command.add_parser("checkpoint-bootstrap")
    command.add_parser("healthcheck")
    command.add_parser("campaign")
    parsed = parser.parse_args(arguments)
    try:
        if parsed.command == "assets-bootstrap":
            bootstrap_assets()
        elif parsed.command == "preflight":
            preflight()
        elif parsed.command == "smoke-runtime":
            from b1k_rollout.smoke_runtime import main as smoke_runtime_main
            return smoke_runtime_main(["--success-prefix", parsed.success_prefix, "--failure-prefix", parsed.failure_prefix])
        elif parsed.command == "checkpoint-bootstrap":
            bootstrap_checkpoint()
        elif parsed.command == "healthcheck":
            healthcheck()
        else:
            run_campaign()
    except CliError as error:
        print(f"b1k-rollout: {error}", file=sys.stderr)
        return 64
    return 0


def _private_regular_file(value: str | None, *, label: str) -> Path:
    if not value:
        raise CliError(f"{label} must name a regular file")
    path = Path(value)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise CliError(f"{label} must be a regular file") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise CliError(f"{label} must be a regular file")
    if metadata.st_mode & 0o077:
        raise CliError(f"{label} must not be readable by group or other")
    try:
        if not path.read_text(encoding="utf-8").strip():
            raise CliError(f"{label} must not be empty")
    except OSError as error:
        raise CliError(f"{label} must be readable by the rollout user") from error
    return path


def _directory(value: str | None, *, label: str) -> Path:
    if not value:
        raise CliError(f"{label} must name an existing directory")
    path = Path(value)
    if not path.is_dir() or path.is_symlink():
        raise CliError(f"{label} must name an existing directory")
    return path


def _bootstrap_behavior_license_key() -> None:
    """Keep BEHAVIOR's license-key flow without delegating any archive download."""

    try:
        from omnigibson.utils.asset_utils import download_key, get_key_path

        if not Path(get_key_path()).is_file():
            download_key()
    except Exception as error:
        raise CliError("BEHAVIOR license-key bootstrap failed") from error


def _install_pinned_asset(data_path: Path, *, archive: PinnedAssetArchive, token_file: Path) -> None:
    target = data_path / archive.directory
    receipt = data_path / _ASSET_RECEIPTS_DIRECTORY / f"{archive.directory}.json"
    if target.is_dir() and _receipt_is_valid(receipt, archive):
        return
    downloaded = _download_pinned_archive(archive, token_file=token_file)
    _verify_archive_sha256(downloaded, expected=archive.sha256)
    staging_parent = data_path / ".b1k-asset-staging"
    staging_parent.mkdir(mode=0o700, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{archive.directory}.", dir=staging_parent))
    backup = target.with_name(f".{target.name}.unverified-backup")
    try:
        _safe_extract_zip(downloaded, destination=staging)
        if backup.exists():
            raise CliError(f"stale asset backup prevents atomic install of {archive.directory}")
        if target.exists():
            os.replace(target, backup)
        os.replace(staging, target)
        _write_receipt(receipt, archive)
        if backup.exists():
            shutil.rmtree(backup)
    except (OSError, zipfile.BadZipFile, CliError) as error:
        if not target.exists() and backup.exists():
            os.replace(backup, target)
        raise CliError(f"pinned asset install failed for {archive.directory}") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _download_pinned_archive(archive: PinnedAssetArchive, *, token_file: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=_ASSET_DATASET_REPOSITORY,
            repo_type="dataset",
            filename=archive.filename,
            revision=_ASSET_DATASET_REVISION,
            token=_read_token_file(token_file),
        )
    except Exception as error:
        raise CliError(f"pinned archive retrieval failed for {archive.directory}") from error
    return Path(downloaded)


def _read_token_file(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise CliError("rollout token file became unreadable") from error
    if not value:
        raise CliError("rollout token file is empty")
    return value


def _verify_archive_sha256(path: Path, *, expected: str) -> None:
    if not path.is_file() or path.is_symlink() or _file_sha256(path) != expected:
        raise CliError("pinned asset archive SHA-256 mismatch")


def _safe_extract_zip(archive: Path, *, destination: Path) -> None:
    with zipfile.ZipFile(archive) as reader:
        for member in reader.infolist():
            relative = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(mode):
                raise CliError("pinned asset archive contains an unsafe member")
            if member.is_dir():
                continue
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise CliError("pinned asset archive contains a non-regular member")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with reader.open(member) as source, target.open("xb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)


def _receipt_is_valid(path: Path, archive: PinnedAssetArchive) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return payload == {
        "dataset_repository": _ASSET_DATASET_REPOSITORY,
        "dataset_revision": _ASSET_DATASET_REVISION,
        "filename": archive.filename,
        "sha256": archive.sha256,
    }


def _write_receipt(path: Path, archive: PinnedAssetArchive) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {
        "dataset_repository": _ASSET_DATASET_REPOSITORY,
        "dataset_revision": _ASSET_DATASET_REVISION,
        "filename": archive.filename,
        "sha256": archive.sha256,
    }
    descriptor, temporary = tempfile.mkstemp(prefix=f".{archive.directory}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as writer:
            json.dump(payload, writer, sort_keys=True, separators=(",", ":"))
            writer.write("\n")
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_official_assets(data_path: Path, *, task_manifest_path: Path) -> None:
    """Validate the exact versioned layouts used by the pinned 2026 evaluator."""

    robot_version = _asset_version(data_path / "omnigibson-robot-assets" / "VERSION", label="OmniGibson robot assets")
    if _version_tuple(robot_version) < _version_tuple(_MINIMUM_ROBOT_ASSET_VERSION):
        raise CliError(
            f"OmniGibson robot assets version must be >= {_MINIMUM_ROBOT_ASSET_VERSION}"
        )
    behavior_version = _asset_version(data_path / "behavior-1k-assets" / "VERSION", label="BEHAVIOR assets")
    if behavior_version != _BEHAVIOR_ASSET_VERSION:
        raise CliError(f"BEHAVIOR assets version must be exactly {_BEHAVIOR_ASSET_VERSION}")

    metadata_path = data_path / "2026-challenge-task-instances" / "metadata" / "B100_task_misc.csv"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise CliError("2026 challenge assets are missing B100_task_misc.csv")
    try:
        manifest = load_task_manifest(task_manifest_path)
        with metadata_path.open(newline="", encoding="utf-8") as reader:
            rows = list(csv.DictReader(reader))
    except (OSError, ValueError, csv.Error) as error:
        raise CliError("2026 challenge asset metadata is unreadable or stale") from error
    if not rows or not {"Task", "Task ID"}.issubset(rows[0]):
        raise CliError("2026 challenge asset metadata must contain Task and Task ID columns")
    metadata: dict[str, int] = {}
    try:
        for row in rows:
            task_name = row["Task"]
            task_id = int(row["Task ID"])
            if not task_name or task_name in metadata:
                raise ValueError("duplicate or empty task")
            metadata[task_name] = task_id
    except (KeyError, TypeError, ValueError) as error:
        raise CliError("2026 challenge asset metadata has invalid task identifiers") from error

    tasks = manifest["tasks"]
    expected = {
        str(task["task_name"]): int(task["source_task_id"])
        for task in tasks  # type: ignore[union-attr]
    }
    if metadata != expected:
        raise CliError("2026 challenge asset metadata does not match the canonical 100-task campaign")

    public_root = data_path / "2026-challenge-task-instances" / "scene_test" / "public"
    if not public_root.is_dir() or public_root.is_symlink():
        raise CliError("2026 challenge assets are missing the public scene_test layout")
    task_root = data_path / "2026-challenge-task-instances"
    matches: dict[tuple[str, int], list[Path]] = {}
    for candidate in public_root.rglob("*-tro_state.json"):
        name = candidate.name.removesuffix("-tro_state.json")
        if "_task_" not in name or "_0_" not in name:
            continue
        _, task_and_instance = name.split("_task_", 1)
        task_name, instance_suffix = task_and_instance.rsplit("_0_", 1)
        instance_text, separator, template = instance_suffix.partition("_template")
        if separator != "_template" or not instance_text.isdecimal():
            continue
        instance_id = int(instance_text)
        if task_name in expected and instance_id in _PUBLIC_INSTANCE_IDS:
            matches.setdefault((task_name, instance_id), []).append(candidate)

    campaign_files = [metadata_path]
    invalid = []
    for task_name in expected:
        for instance_id in _PUBLIC_INSTANCE_IDS:
            candidates = matches.get((task_name, instance_id), [])
            if len(candidates) != 1 or not candidates[0].is_file() or candidates[0].is_symlink():
                invalid.append(f"{task_name}:{instance_id}")
            else:
                campaign_files.append(candidates[0])
    if invalid:
        raise CliError("2026 challenge assets are incomplete or ambiguous for the canonical public campaign")
    if _campaign_content_root(task_root, campaign_files) != _EXPECTED_CAMPAIGN_CONTENT_ROOT:
        raise CliError("2026 challenge assets do not match the pinned public campaign content root")


def _campaign_content_root(task_root: Path, files: list[Path]) -> str:
    root = hashlib.sha256()
    for path in sorted(files, key=lambda candidate: candidate.relative_to(task_root).as_posix()):
        relative = path.relative_to(task_root).as_posix()
        root.update(relative.encode("utf-8"))
        root.update(b"\0")
        root.update(_file_sha256(path).encode("ascii"))
        root.update(b"\n")
    return root.hexdigest()


def _asset_version(path: Path, *, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise CliError(f"{label} are missing the required VERSION marker")
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise CliError(f"{label} VERSION marker is unreadable") from error
    _version_tuple(value)
    return value


def _version_tuple(value: str) -> tuple[int, ...]:
    pieces = value.split(".")
    if not pieces or any(not piece.isascii() or not piece.isdecimal() for piece in pieces):
        raise CliError("asset VERSION marker is invalid")
    return tuple(int(piece) for piece in pieces)


def _gpu_ids(value: str | None) -> tuple[int, ...]:
    if value in (None, ""):
        raise CliError("GPU_IDS must explicitly name one through four GPU IDs")
    raw = value
    try:
        ids = tuple(int(item) for item in raw.split(","))
    except ValueError as error:
        raise CliError("CUDA_VISIBLE_DEVICES must contain explicit numeric GPU IDs") from error
    if not 1 <= len(ids) <= 4 or any(item < 0 for item in ids) or len(set(ids)) != len(ids):
        raise CliError("campaign requires one through four unique explicit GPUs")
    return ids


def _provenance_key(path: Path) -> bytes:
    """Create a local owner-only campaign key once, without serializing its value."""

    path = Path(path)
    try:
        if path.exists():
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
                raise CliError("campaign provenance key is unsafe")
            key = path.read_bytes()
        else:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            key = os.urandom(32)
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as writer:
                writer.write(key)
                writer.flush()
                os.fsync(writer.fileno())
    except OSError as error:
        raise CliError("campaign provenance key is unavailable") from error
    if len(key) != 32:
        raise CliError("campaign provenance key is invalid")
    return key


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while chunk := reader.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _commit_id(value: object) -> str:
    commit = getattr(value, "oid", getattr(value, "commit_id", None))
    try:
        return require_immutable_commit(commit, label="Hub commit")
    except ValueError as error:
        raise CliError("Hub operation did not return an immutable commit") from error


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
