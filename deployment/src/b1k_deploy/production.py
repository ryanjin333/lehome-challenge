"""Concrete, fixed subprocess transports for the trusted publication adapter."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .dockerhub import (
    CommandResult,
    DockerCommandRunner,
    DockerHubClient,
    DockerHubReleaseVerifier,
    SubprocessDockerRunner,
    TokenSource,
)
from .publish import PublicationAdapters, PublicationError


_REPOSITORY = "docker.io/ryanjin333/behavior1k-groot-n17"
_DOCKER_LOGIN_REPOSITORY = "docker.io"
_SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_TEMPLATE_ID_RE = re.compile(r"^[1-9][0-9]*$")
_BUILD_TIMEOUT_SECONDS = 1800.0
_VAST_TIMEOUT_SECONDS = 30.0
_TEMPLATE_FIELDS = frozenset({"env", "extra_filters", "image", "jup_direct", "name", "onstart", "private", "recommended_disk_space", "runtype", "ssh_direct", "use_ssh"})
# Search responses include these provider-owned fields.  They never describe a
# configurable template value, so excluding them keeps receipt hashes stable.
_VAST_PROVIDER_METADATA = frozenset({
    "id", "hash_id", "creator_id", "created_at", "count_created", "default_tag",
    "docker_login_repo", "docker_login_user", "docker_login_pass", "recommended",
    "recent_create_date", "tag", "image_tag", "href", "repo", "jupyter_dir",
    "readme", "readme_hash", "readme_visible", "desc", "desc_count",
    "use_jupyter_lab", "date_created", "date_updated", "status", "args_str",
    "autoscaler", "cached", "command", "created_from", "created_from_id",
    "deleted_at", "jupyter_tested", "jupyterlab_tested", "lang_utf8", "max_cuda",
    "min_cuda", "python_utf8", "sort_order", "vm", "volume_info",
})


class ReleaseContext(Protocol):
    """Proves that the Docker build context is the requested release source."""

    def verify(self, workspace: Path, source_commit: str) -> None: ...


class GitReleaseContext:
    """The production equivalent of trainer/scripts/build-image.sh's gate."""

    def __init__(self, command: Callable[[tuple[str, ...]], str] | None = None):
        self._command = command or self._run_git

    def verify(self, workspace: Path, source_commit: str) -> None:
        head = self._command(("git", "-C", str(workspace), "rev-parse", "HEAD")).strip()
        if head != source_commit:
            raise PublicationError("release source commit does not match workspace HEAD")
        changes = self._command(
            ("git", "-C", str(workspace), "status", "--porcelain=v1", "--untracked-files=all", "--", ".dockerignore", "trainer", "rollout")
        )
        if changes.strip():
            raise PublicationError("release build context has tracked or untracked changes")

    @staticmethod
    def _run_git(arguments: tuple[str, ...]) -> str:
        try:
            result = subprocess.run(arguments, capture_output=True, check=False, text=True, timeout=10.0)
        except (OSError, subprocess.SubprocessError):
            raise PublicationError("release context inspection failed") from None
        if result.returncode != 0:
            raise PublicationError("release context inspection failed")
        return result.stdout


@dataclass(frozen=True)
class ConfiguredPublicationSettings:
    docker_username: str
    docker_token_file: Path
    docker_executable: Path
    vastai_executable: Path
    vast_api_key_file: Path
    workspace: Path

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        home: Path | None = None,
        workspace: Path | None = None,
    ) -> "ConfiguredPublicationSettings":
        values = os.environ if environment is None else environment
        username = values.get("B1K_DOCKER_USERNAME", "")
        if not _USERNAME_RE.fullmatch(username):
            raise PublicationError("configured Docker username is invalid")
        docker_token_file = _required_private_file(values, "B1K_DOCKER_TOKEN_FILE")
        docker_executable = _required_executable(values, "B1K_DOCKER_EXECUTABLE")
        vastai_executable = _required_executable(values, "B1K_VASTAI_EXECUTABLE")
        vast_api_key_file = (Path.home() if home is None else home) / ".config" / "vastai" / "vast_api_key"
        _validate_private_file(vast_api_key_file, "Vast API key file")
        fixed_workspace = Path(__file__).resolve().parents[3] if workspace is None else workspace
        fixed_workspace = _validate_workspace(fixed_workspace)
        return cls(username, docker_token_file, docker_executable, vastai_executable, vast_api_key_file, fixed_workspace)


class DockerBuildxImageBuilder:
    """Build the two fixed linux/amd64 release images after an ephemeral login."""

    def __init__(
        self,
        *,
        username: str,
        token_file: Path,
        docker_executable: str | Path,
        workspace: Path,
        runner: DockerCommandRunner | None = None,
        release_context: ReleaseContext | None = None,
        timeout_seconds: float = _BUILD_TIMEOUT_SECONDS,
    ):
        if not _USERNAME_RE.fullmatch(username):
            raise PublicationError("configured Docker username is invalid")
        if timeout_seconds <= 0 or timeout_seconds > _BUILD_TIMEOUT_SECONDS:
            raise PublicationError("Docker build timeout is invalid")
        self._username = username
        self._token_source = TokenSource.from_token_file(_validated_private_file(token_file, "Docker token file"))
        self._docker_executable = _validate_executable_path(Path(docker_executable), "Docker executable")
        self._workspace = _validate_workspace(Path(workspace))
        self._runner = runner or SubprocessDockerRunner()
        self._release_context = release_context or GitReleaseContext()
        self._timeout = timeout_seconds

    def build_and_push(self, repository: str, tag: str, source_commit: str) -> None:
        purpose = _purpose_for_image(repository, tag, source_commit)
        if not _TAG_RE.fullmatch(tag) or not _SOURCE_COMMIT_RE.fullmatch(source_commit):
            raise PublicationError("Docker image build inputs are invalid")
        self._release_context.verify(self._workspace, source_commit)
        token = self._token_source.resolve()
        with tempfile.TemporaryDirectory(prefix="b1k-buildx-config-") as config:
            os.chmod(config, 0o700)
            environment = DockerHubClient._docker_environment(config)
            DockerHubClient._write_ephemeral_auth_config(config, self._username, token)
            plugin = _discover_buildx_plugin(self._docker_executable)
            if plugin is not None:
                plugin_directory = Path(config) / "cli-plugins"
                plugin_directory.mkdir(mode=0o700)
                plugin_directory.chmod(0o700)
                (plugin_directory / "docker-buildx").symlink_to(plugin)
            self._run(
                buildx_release_command(self._docker_executable, self._workspace, repository, tag, source_commit),
                None,
                environment,
                "Docker buildx publication",
            )

    def _run(self, arguments: tuple[str, ...], stdin: str | None, environment: Mapping[str, str], operation: str) -> None:
        try:
            result = self._runner.run(arguments, stdin=stdin, env=environment, timeout=self._timeout)
        except Exception:
            raise PublicationError(f"{operation} failed") from None
        if not isinstance(result, CommandResult) or result.returncode != 0:
            raise PublicationError(f"{operation} failed")


def _discover_buildx_plugin(docker_executable: Path) -> Path | None:
    candidates = [
        docker_executable.parent.parent / "cli-plugins" / "docker-buildx",
        Path("/usr/local/lib/docker/cli-plugins/docker-buildx"),
        Path("/usr/local/libexec/docker/cli-plugins/docker-buildx"),
        Path("/usr/lib/docker/cli-plugins/docker-buildx"),
        Path("/usr/libexec/docker/cli-plugins/docker-buildx"),
    ]
    homebrew_cellar = next((parent for parent in docker_executable.parents if parent.name == "Cellar"), None)
    if homebrew_cellar is not None:
        candidates.extend(
            (
                homebrew_cellar.parent / "lib" / "docker" / "cli-plugins" / "docker-buildx",
                homebrew_cellar.parent / "libexec" / "docker" / "cli-plugins" / "docker-buildx",
            )
        )
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            metadata = resolved.stat()
        except OSError:
            continue
        if (
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_mode & 0o111
            and not metadata.st_mode & 0o022
            and metadata.st_uid in {0, os.getuid()}
        ):
            return resolved
    return None


class VastCliTemplateClient:
    """A raw-JSON Vast CLI boundary using only the local configured API-key file."""

    def __init__(
        self,
        *,
        vastai_executable: str | Path,
        api_key_file: Path,
        runner: DockerCommandRunner | None = None,
        timeout_seconds: float = _VAST_TIMEOUT_SECONDS,
    ):
        if timeout_seconds <= 0 or timeout_seconds > _VAST_TIMEOUT_SECONDS:
            raise PublicationError("Vast CLI timeout is invalid")
        self._vastai_executable = _validate_executable_path(Path(vastai_executable), "Vast CLI executable")
        _validate_private_file(api_key_file, "Vast API key file")
        self._runner = runner or SubprocessDockerRunner()
        self._timeout = timeout_seconds

    def find_private_template(self, name: str, image_reference: str) -> str | None:
        matches = [template for template in self._search(f"name=={_query_value(name)}") if _is_private_template(template, name, image_reference)]
        if len(matches) > 1:
            raise PublicationError("Vast template lookup is ambiguous")
        return _template_id(matches[0].get("id")) if matches else None

    def create_private_template(self, template: Mapping[str, Any]) -> str:
        command = self._create_command(template)
        self._run_quiet(command)
        name = _required_string(template, "name")
        image = _required_string(template, "image")
        template_id = self.find_private_template(name, image)
        if template_id is None:
            raise PublicationError("Vast template publication readback is missing")
        return template_id

    def get_template(self, template_id: str) -> Mapping[str, Any]:
        expected = _template_id(template_id)
        matches = [template for template in self._search(f"id=={expected}") if _template_id(template.get("id")) == expected]
        if len(matches) != 1:
            raise PublicationError("Vast template readback is missing or ambiguous")
        if matches[0].get("docker_login_repo") != _DOCKER_LOGIN_REPOSITORY:
            raise PublicationError("Vast template private pull repository is missing")
        return _project_template_readback(matches[0])

    def _search(self, query: str) -> list[Mapping[str, Any]]:
        response = self._raw_json((str(self._vastai_executable), "--raw", "search", "templates", query))
        rows = response.get("templates") if isinstance(response, Mapping) else response
        if not isinstance(rows, list) or not all(isinstance(item, Mapping) for item in rows):
            raise PublicationError("Vast template readback is invalid")
        return [dict(item) for item in rows]

    def _create_command(self, template: Mapping[str, Any]) -> tuple[str, ...]:
        try:
            name = _required_string(template, "name")
            image = _required_string(template, "image")
            environment = _required_string(template, "env")
            onstart = template["onstart"]
            disk_space = template["recommended_disk_space"]
            filters = template["extra_filters"]
        except (KeyError, PublicationError):
            raise PublicationError("Vast template payload is invalid") from None
        if not isinstance(onstart, str):
            raise PublicationError("Vast template payload is invalid")
        if image.startswith(_REPOSITORY + "@sha256:") and "b1k-rollout-" in name:
            if onstart != "":
                raise PublicationError("Vast rollout template must use the canonical empty onstart")
        elif not onstart:
            raise PublicationError("Vast training template must use a nonempty onstart")
        if template.get("private") is not True or template.get("runtype") != "ssh" or template.get("use_ssh") is not True or template.get("ssh_direct") is not True or template.get("jup_direct") is not False or not isinstance(disk_space, int) or isinstance(disk_space, bool) or disk_space <= 0 or not isinstance(filters, Mapping):
            raise PublicationError("Vast template payload is invalid")
        return (
            str(self._vastai_executable), "--raw", "create", "template",
            "--name", name, "--image", image, "--env", environment,
            "--login", _DOCKER_LOGIN_REPOSITORY,
            "--ssh", "--direct", "--onstart-cmd", onstart,
            "--search_params", _search_query(filters), "--no-default",
            "--disk_space", str(disk_space),
        )

    def _raw_json(self, arguments: tuple[str, ...]) -> Any:
        result = self._run(arguments)
        try:
            return json.loads(result.stdout)
        except (TypeError, ValueError):
            raise PublicationError("Vast CLI returned invalid raw JSON") from None

    def _run_quiet(self, arguments: tuple[str, ...]) -> None:
        self._run(arguments)

    def _run(self, arguments: tuple[str, ...]) -> CommandResult:
        try:
            result = self._runner.run(arguments, stdin=None, env=_vast_environment(), timeout=self._timeout)
        except Exception:
            raise PublicationError("Vast CLI operation failed") from None
        if not isinstance(result, CommandResult) or result.returncode != 0:
            raise PublicationError("Vast CLI operation failed")
        return result


def configured_publication_adapters(*, workspace: Path) -> PublicationAdapters:
    settings = ConfiguredPublicationSettings.from_environment(workspace=workspace)
    runner = SubprocessDockerRunner()
    token_source = TokenSource.from_token_file(settings.docker_token_file)
    registry = DockerHubClient(settings.docker_username, runner=runner, docker_executable=str(settings.docker_executable))
    return PublicationAdapters(
        builder=DockerBuildxImageBuilder(
            username=settings.docker_username,
            token_file=settings.docker_token_file,
            docker_executable=settings.docker_executable,
            workspace=settings.workspace,
            runner=runner,
        ),
        verifier=DockerHubReleaseVerifier(registry, token_source),
        templates=VastCliTemplateClient(vastai_executable=settings.vastai_executable, api_key_file=settings.vast_api_key_file, runner=runner),
    )


def buildx_release_command(
    docker_executable: str | Path,
    workspace: Path,
    repository: str,
    tag: str,
    source_commit: str,
) -> tuple[str, ...]:
    """The sole canonical local/CI release-build argument contract."""
    purpose = _purpose_for_image(repository, tag, source_commit)
    if not _SOURCE_COMMIT_RE.fullmatch(source_commit):
        raise PublicationError("Docker image build inputs are invalid")
    dockerfile = workspace / ("trainer" if purpose == "training" else "rollout") / "Dockerfile"
    if not dockerfile.is_file():
        raise PublicationError("fixed Dockerfile is unavailable")
    target = "training-runtime" if purpose == "training" else "rollout-runtime"
    return (
        str(docker_executable), "buildx", "build", "--platform", "linux/amd64", "--push",
        "--target", target,
        "--build-arg", f"REPOSITORY_COMMIT={source_commit}",
        "--label", "io.lehome.release-mode=release",
        "--label", f"io.lehome.image-role={purpose}",
        "--tag", f"{repository}:{tag}", "--file", str(dockerfile), str(workspace),
    )


def _purpose_for_image(repository: str, tag: str, source_commit: str) -> str:
    if repository != _REPOSITORY or not _SOURCE_COMMIT_RE.fullmatch(source_commit):
        raise PublicationError("Docker repository is not the approved B1K release target")
    if tag == f"trainer-{source_commit}":
        return "training"
    if tag == f"rollout-{source_commit}":
        return "rollout"
    raise PublicationError("Docker tag must be the canonical purpose-prefixed source revision")


def _required_executable(values: Mapping[str, str], name: str) -> Path:
    value = values.get(name, "")
    if not isinstance(value, str) or not value:
        raise PublicationError(f"{name} must point to an absolute executable")
    return _validate_executable_path(Path(value), name)


def _validate_executable_path(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError:
        raise PublicationError(f"{label} is unavailable") from None
    if not path.is_absolute() or not stat.S_ISREG(metadata.st_mode) or not metadata.st_mode & 0o111:
        raise PublicationError(f"{label} must be an absolute executable")
    return resolved


def _required_private_file(values: Mapping[str, str], name: str) -> Path:
    value = values.get(name, "")
    if not isinstance(value, str) or not value:
        raise PublicationError(f"{name} must point to a private file")
    return _validated_private_file(Path(value), name)


def _validate_private_file(path: Path, label: str) -> None:
    _validated_private_file(path, label)


def _validated_private_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise PublicationError(f"{label} must be an absolute private file")
    try:
        resolved = path.resolve(strict=True)
        # Reuse the secure component-by-component, no-follow validation already
        # used by Docker verification.  The value is immediately discarded.
        TokenSource.from_token_file(resolved).resolve()
    except Exception:
        raise PublicationError(f"{label} must be a nonempty 0600 file owned by the current user") from None
    return resolved


def _validate_workspace(workspace: Path) -> Path:
    try:
        resolved = workspace.resolve(strict=True)
    except OSError:
        raise PublicationError("fixed B1K workspace is unavailable") from None
    if not (resolved / "trainer" / "Dockerfile").is_file() or not (resolved / "rollout" / "Dockerfile").is_file():
        raise PublicationError("fixed B1K Dockerfiles are unavailable")
    return resolved


def _vast_environment() -> dict[str, str]:
    environment = {"HOME": str(Path.home())}
    if "PATH" in os.environ:
        environment["PATH"] = os.environ["PATH"]
    return environment


def _template_id(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not _TEMPLATE_ID_RE.fullmatch(value):
        raise PublicationError("Vast template did not return one exact ID")
    return value


def _is_private_template(template: Mapping[str, Any], name: str, image_reference: str) -> bool:
    return (
        template.get("name") == name
        and template.get("image") == image_reference
        and template.get("private") is True
        and template.get("docker_login_repo") == _DOCKER_LOGIN_REPOSITORY
    )


def _project_template_readback(row: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only canonical configurable template fields from a Vast row."""
    unknown = set(row) - _TEMPLATE_FIELDS - _VAST_PROVIDER_METADATA
    if unknown or not _TEMPLATE_FIELDS.issubset(row):
        raise PublicationError("Vast template readback contains unknown or missing configurable fields")
    projected = {field: row[field] for field in _TEMPLATE_FIELDS}
    filters = projected["extra_filters"]
    if isinstance(filters, str):
        try:
            filters = json.loads(filters)
        except ValueError:
            raise PublicationError("Vast template filters readback is invalid") from None
    if not isinstance(filters, Mapping):
        raise PublicationError("Vast template filters readback is invalid")
    normalized_filters: dict[str, dict[str, Any]] = {}
    for field, constraints in filters.items():
        if not isinstance(field, str) or not isinstance(constraints, Mapping):
            raise PublicationError("Vast template filters readback is invalid")
        normalized_constraints: dict[str, Any] = {}
        for operator, value in constraints.items():
            if isinstance(value, str) and re.fullmatch(r"(?:0|[1-9][0-9]*)", value):
                value = int(value)
            elif isinstance(value, float) and value.is_integer():
                value = int(value)
            normalized_constraints[operator] = value
        normalized_filters[field] = normalized_constraints
    try:
        _search_query(normalized_filters)
    except PublicationError:
        raise PublicationError("Vast template filters readback is invalid") from None
    projected["extra_filters"] = normalized_filters
    disk = projected["recommended_disk_space"]
    if isinstance(disk, float) and disk.is_integer():
        disk = int(disk)
    if not isinstance(disk, int) or isinstance(disk, bool) or disk <= 0:
        raise PublicationError("Vast template disk readback is invalid")
    projected["recommended_disk_space"] = disk
    return projected


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise PublicationError("Vast template payload is invalid")
    return result


def _query_value(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise PublicationError("Vast template query value is invalid")
    return value


def _search_query(filters: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    operators = {"eq": "==", "gte": ">=", "lte": "<=", "in": "in"}
    for field in sorted(filters):
        constraints = filters[field]
        if not isinstance(field, str) or not re.fullmatch(r"[a-z_]+", field) or not isinstance(constraints, Mapping) or not constraints:
            raise PublicationError("Vast template filters are invalid")
        for name in sorted(constraints):
            value = constraints[name]
            if name not in operators:
                raise PublicationError("Vast template filters are invalid")
            if name == "in":
                if not isinstance(value, list) or not value or not all(isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9_]+", item.replace(" ", "_")) for item in value):
                    raise PublicationError("Vast template filters are invalid")
                rendered = "[" + ",".join(item.replace(" ", "_") for item in value) + "]"
            elif isinstance(value, bool):
                rendered = "True" if value else "False"
            elif isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                # Canonical template payload/readback uses Vast raw MiB-like
                # RAM values (e.g. 96000, 128000). Vast CLI 1.5.2 search
                # parameters expect GB and multiplies these fields by 1000.
                # Convert only at CLI rendering; never rewrite the receipt.
                if field in {"gpu_ram", "cpu_ram"}:
                    if value % 1000:
                        raise PublicationError("Vast template RAM filters must be exact 1000-unit values")
                    rendered = str(value // 1000)
                else:
                    rendered = str(value)
            else:
                raise PublicationError("Vast template filters are invalid")
            pieces.append(f"{field} {operators[name]} {rendered}")
    return " ".join(pieces)
