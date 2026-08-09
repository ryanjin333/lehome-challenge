"""Private Docker Hub release verification with ephemeral authenticated pulls."""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


_TAG_RE = re.compile(r"^(trainer|rollout)-([0-9a-f]{40})$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPOSITORY = "docker.io/ryanjin333/behavior1k-groot-n17"
_DOCKER_HUB_AUTH_KEY = "https://index.docker.io/v1/"
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:[A-Z][A-Z0-9_]*_)?(?:TOKEN|API_KEY|PASSWORD|SECRET|CREDENTIAL)\s*=\s*[^\s,'\"]+"),
    re.compile(r"(?i)\bBearer\s+[^\s,'\"]+"),
    re.compile(r"(?i)\b(?:hf|dckr_pat|docker_pat|vast|vastai|vst)_[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)(https?://)[^\s/@:]+:[^\s/@]+@"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


class DockerReleaseError(ValueError):
    """Raised when an image cannot meet the private immutable-release contract."""


class CredentialSourceError(DockerReleaseError):
    """Raised when a credential source is unsafe, absent, or unreadable."""


class DockerRegistryClient(Protocol):
    """Injectable release-verification boundary."""

    def repository_info(self, repository: str, token: str) -> Mapping[str, Any]: ...

    def registry_manifest(self, repository: str, tag: str, token: str) -> Mapping[str, Any]: ...

    def authenticated_pull(self, reference: str, token: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class HttpTransport(Protocol):
    def request(
        self, method: str, url: str, headers: Mapping[str, str], *, timeout: float, body: bytes | None = None
    ) -> HttpResponse: ...


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class DockerCommandRunner(Protocol):
    def run(
        self,
        arguments: tuple[str, ...],
        *,
        stdin: str | None,
        env: Mapping[str, str],
        timeout: float,
    ) -> CommandResult: ...


class UrllibTransport:
    """Small stdlib transport; callers receive only status, headers, and bytes."""

    def request(self, method: str, url: str, headers: Mapping[str, str], *, timeout: float, body: bytes | None = None) -> HttpResponse:
        try:
            with urlopen(Request(url, data=body, method=method, headers=dict(headers)), timeout=timeout) as response:
                return HttpResponse(response.status, dict(response.headers.items()), response.read())
        except HTTPError as error:
            return HttpResponse(error.code, dict(error.headers.items()) if error.headers else {}, error.read())
        except (URLError, OSError):
            raise DockerReleaseError("Docker HTTP request failed") from None


class SubprocessDockerRunner:
    """No-output subprocess boundary for the ephemeral Docker CLI verification path."""

    def run(
        self,
        arguments: tuple[str, ...],
        *,
        stdin: str | None,
        env: Mapping[str, str],
        timeout: float,
    ) -> CommandResult:
        try:
            completed = subprocess.run(
                arguments,
                input=stdin,
                text=True,
                capture_output=True,
                check=False,
                env=dict(env),
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise DockerReleaseError("Docker CLI operation failed") from None
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def sanitize_output(value: object, *, known_secrets: tuple[str, ...] = ()) -> str:
    """Redact explicitly supplied output; provider exceptions are never rendered."""
    text = str(value)
    for secret in known_secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


class TokenSource:
    """A credential locator, never a serializable raw Docker or Hub token."""

    def __init__(
        self,
        *,
        credential_name: str | None = None,
        credential_store: Callable[[str], str | None] | None = None,
        token_file: str | Path | None = None,
    ):
        use_store = credential_name is not None or credential_store is not None
        use_file = token_file is not None
        if use_store == use_file:
            raise CredentialSourceError("credentials must come from exactly one process credential store or token file")
        if use_store:
            if not credential_name or credential_store is None:
                raise CredentialSourceError("credential store name and reader are both required")
            self._credential_name = credential_name
            self._credential_store = credential_store
            self._token_file: Path | None = None
        else:
            self._credential_name = None
            self._credential_store = None
            self._token_file = Path(token_file).absolute()  # type: ignore[arg-type]

    @classmethod
    def from_credential_store(cls, name: str, reader: Callable[[str], str | None]) -> "TokenSource":
        return cls(credential_name=name, credential_store=reader)

    @classmethod
    def from_token_file(cls, path: str | Path) -> "TokenSource":
        return cls(token_file=path)

    def resolve(self) -> str:
        if self._credential_store is not None:
            try:
                token = self._credential_store(self._credential_name or "")
            except Exception:
                raise CredentialSourceError("credential store lookup failed") from None
            return self._validate_token(token)
        return self._read_private_token_file()

    @staticmethod
    def _validate_token(token: str | None) -> str:
        if not isinstance(token, str) or not token or token != token.strip():
            raise CredentialSourceError("credential source returned no usable token")
        return token

    def _read_private_token_file(self) -> str:
        assert self._token_file is not None
        parts = self._token_file.parts
        if not self._token_file.is_absolute() or not parts or parts[0] != os.sep or any(part in {"", ".", ".."} for part in parts[1:]):
            raise CredentialSourceError("token file cannot be opened")
        descriptors: list[int] = []
        try:
            descriptor = os.open(os.sep, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
            descriptors.append(descriptor)
            for index, component in enumerate(parts[1:]):
                is_final = index == len(parts[1:]) - 1
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
                if not is_final:
                    flags |= getattr(os, "O_DIRECTORY", 0)
                descriptor = os.open(component, flags, dir_fd=descriptors[-1])
                descriptors.append(descriptor)
            metadata = os.fstat(descriptors[-1])
            if not stat.S_ISREG(metadata.st_mode):
                raise CredentialSourceError("token file must be a regular file")
            if metadata.st_uid != os.getuid():
                raise CredentialSourceError("token file must be owned by the current user")
            if stat.S_IMODE(metadata.st_mode) != 0o600:
                raise CredentialSourceError("token file mode must be exactly 0600")
            data = os.read(descriptors[-1], 8193)
            if len(data) > 8192:
                raise CredentialSourceError("token file cannot be read")
            try:
                return self._validate_token(data.decode("utf-8").strip())
            except UnicodeDecodeError:
                raise CredentialSourceError("token file cannot be read") from None
        except CredentialSourceError:
            raise
        except OSError:
            raise CredentialSourceError("token file cannot be opened") from None
        finally:
            for descriptor in reversed(descriptors):
                try:
                    os.close(descriptor)
                except OSError:
                    pass


class DockerHubClient:
    """Concrete Docker Hub/Registry plus ephemeral Docker CLI client for later CLI use."""

    def __init__(
        self,
        username: str,
        *,
        transport: HttpTransport | None = None,
        runner: DockerCommandRunner | None = None,
        docker_executable: str | None = None,
        context_resolver: Callable[[str, Mapping[str, str]], str] | None = None,
        timeout_seconds: float = 30.0,
        pull_timeout_seconds: float = 180.0,
    ):
        if not isinstance(username, str) or not username or any(char.isspace() for char in username):
            raise DockerReleaseError("Docker username is invalid")
        if timeout_seconds <= 0 or pull_timeout_seconds <= 0:
            raise DockerReleaseError("Docker timeouts must be positive")
        self._username = username
        self._transport = transport or UrllibTransport()
        self._runner = runner or SubprocessDockerRunner()
        self._docker_executable = docker_executable or shutil.which("docker")
        if not self._docker_executable or not os.path.isabs(self._docker_executable):
            raise DockerReleaseError("Docker executable is unavailable")
        self._timeout = timeout_seconds
        self._pull_timeout = pull_timeout_seconds
        self._context_resolver = context_resolver

    def repository_info(self, repository: str, token: str) -> Mapping[str, Any]:
        access_token = self._hub_access_token(token)
        namespace, name = repository.split("/", 1)
        response = self._request(
            "Docker Hub repository lookup",
            "GET",
            f"https://hub.docker.com/v2/namespaces/{namespace}/repositories/{name}",
            {"Authorization": f"Bearer {access_token}"},
        )
        if response.status != 200:
            raise DockerReleaseError("Docker Hub repository lookup failed")
        body = self._json(response.body, "Docker Hub repository lookup")
        private = body.get("is_private")
        if not isinstance(private, bool):
            raise DockerReleaseError("Docker Hub repository lookup failed")
        return {"is_private": private}

    def registry_manifest(self, repository: str, tag: str, token: str) -> Mapping[str, Any]:
        scope = urlencode({"service": "registry.docker.io", "scope": f"repository:{repository}:pull"})
        authorization = self._request(
            "Docker registry authorization",
            "GET",
            f"https://auth.docker.io/token?{scope}",
            self._basic_headers(token),
        )
        if authorization.status != 200:
            raise DockerReleaseError("Docker registry authorization failed")
        token_value = self._json(authorization.body, "Docker registry authorization").get("token")
        if not isinstance(token_value, str) or not token_value:
            raise DockerReleaseError("Docker registry authorization failed")
        response = self._request(
            "Docker registry manifest lookup",
            "GET",
            f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}",
            {
                "Authorization": f"Bearer {token_value}",
                "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json",
            },
        )
        digest = response.headers.get("Docker-Content-Digest") or response.headers.get("docker-content-digest")
        if response.status != 200 or not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
            raise DockerReleaseError("Docker registry manifest lookup failed")
        return {"digest": digest}

    def authenticated_pull(self, reference: str, token: str) -> Mapping[str, Any]:
        DockerHubReleaseVerifier.require_digest_reference(reference)
        digest = reference.rsplit("@", 1)[1]
        with tempfile.TemporaryDirectory(prefix="b1k-docker-config-") as config:
            os.chmod(config, 0o700)
            environment = self._docker_environment(config)
            if "DOCKER_CONTEXT" in environment:
                environment["DOCKER_HOST"] = self._resolve_context_endpoint(environment["DOCKER_CONTEXT"])
                environment.pop("DOCKER_CONTEXT", None)
            self._write_ephemeral_auth_config(config, self._username, token)
            self._run((self._docker_executable, "pull", reference), None, environment)
            labels = self._inspect_labels(reference, environment)
        return {"digest": digest, "labels": labels}

    @staticmethod
    def _write_ephemeral_auth_config(config: str, username: str, token: str) -> None:
        encoded = base64.b64encode(f"{username}:{token}".encode("utf-8")).decode("ascii")
        path = Path(config) / "config.json"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                payload = json.dumps(
                    {"auths": {_DOCKER_HUB_AUTH_KEY: {"auth": encoded}}},
                    separators=(",", ":"),
                ).encode("utf-8")
                written = 0
                while written < len(payload):
                    count = os.write(descriptor, payload[written:])
                    if count <= 0:
                        raise OSError("ephemeral Docker auth write made no progress")
                    written += count
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)
        except OSError:
            raise DockerReleaseError("ephemeral Docker authentication configuration failed") from None

    def _inspect_labels(self, reference: str, environment: Mapping[str, str]) -> Mapping[str, str]:
        try:
            result = self._runner.run(
                (self._docker_executable, "image", "inspect", "--format", "{{json .Config.Labels}}", reference),
                stdin=None,
                env=environment,
                timeout=self._pull_timeout,
            )
        except Exception:
            raise DockerReleaseError("Docker image label inspection failed") from None
        if not isinstance(result, CommandResult) or result.returncode != 0 or len(result.stdout.encode("utf-8")) > 16_384:
            raise DockerReleaseError("Docker image label inspection failed")
        try:
            labels = json.loads(result.stdout)
        except (TypeError, ValueError):
            raise DockerReleaseError("Docker image label inspection failed") from None
        if not isinstance(labels, Mapping) or not all(isinstance(key, str) and isinstance(value, str) for key, value in labels.items()):
            raise DockerReleaseError("Docker image label inspection failed")
        return dict(labels)

    def _hub_access_token(self, token: str) -> str:
        response = self._request(
            "Docker Hub authentication",
            "POST",
            "https://hub.docker.com/v2/auth/token",
            {"Content-Type": "application/json"},
            json.dumps({"identifier": self._username, "secret": token}, separators=(",", ":")).encode("utf-8"),
        )
        value = self._json(response.body, "Docker Hub authentication").get("access_token") if response.status == 200 else None
        if not isinstance(value, str) or not value:
            raise DockerReleaseError("Docker Hub authentication failed")
        return value

    def _request(self, operation: str, method: str, url: str, headers: Mapping[str, str], body: bytes | None = None) -> HttpResponse:
        try:
            return self._transport.request(method, url, headers, timeout=self._timeout, body=body)
        except Exception:
            raise DockerReleaseError(f"{operation} failed") from None

    def _run(self, arguments: tuple[str, ...], stdin: str | None, environment: Mapping[str, str]) -> None:
        try:
            result = self._runner.run(arguments, stdin=stdin, env=environment, timeout=self._pull_timeout)
        except Exception:
            raise DockerReleaseError("Docker authenticated pull failed") from None
        if not isinstance(result, CommandResult) or result.returncode != 0:
            raise DockerReleaseError("Docker authenticated pull failed")

    def _resolve_context_endpoint(self, context: str) -> str:
        try:
            if self._context_resolver is not None:
                endpoint = self._context_resolver(context, self._original_context_environment())
            else:
                result = self._runner.run(
                    (self._docker_executable, "context", "inspect", context, "--format", "{{.Endpoints.docker.Host}}"),
                    stdin=None,
                    env=self._original_context_environment(),
                    timeout=self._timeout,
                )
                endpoint = result.stdout.strip() if isinstance(result, CommandResult) and result.returncode == 0 else ""
        except Exception:
            raise DockerReleaseError("Docker context resolution failed") from None
        if not isinstance(endpoint, str) or not endpoint or endpoint != endpoint.strip():
            raise DockerReleaseError("Docker context resolution failed")
        return endpoint

    def _basic_headers(self, token: str) -> Mapping[str, str]:
        encoded = base64.b64encode(f"{self._username}:{token}".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {encoded}"}

    @staticmethod
    def _docker_environment(config: str) -> dict[str, str]:
        """Keep endpoint/context semantics but never inherit an existing auth config."""
        allowed = ("PATH", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")
        environment = {key: os.environ[key] for key in allowed if key in os.environ}
        environment["DOCKER_CONFIG"] = config
        return environment

    @staticmethod
    def _original_context_environment() -> dict[str, str]:
        allowed = ("PATH", "DOCKER_CONFIG", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")
        return {key: os.environ[key] for key in allowed if key in os.environ}

    @staticmethod
    def _json(body: bytes, operation: str) -> Mapping[str, Any]:
        try:
            decoded = json.loads(body)
        except (TypeError, ValueError):
            raise DockerReleaseError(f"{operation} failed") from None
        if not isinstance(decoded, Mapping):
            raise DockerReleaseError(f"{operation} failed")
        return decoded


@dataclass(frozen=True)
class DockerImageRelease:
    purpose: str
    repository: str
    tag: str
    source_commit: str
    digest: str
    reference: str


class DockerHubReleaseVerifier:
    """Validate private registry state before a digest reaches a Vast template."""

    def __init__(self, registry: DockerRegistryClient, credentials: TokenSource):
        self._registry = registry
        self._credentials = credentials

    def verify_private_image(self, repository: str, tag: str) -> DockerImageRelease:
        self._validate_repository(repository)
        purpose, source_commit = self.image_identity(tag)
        registry_repository = repository.removeprefix("docker.io/")
        token = self._credentials.resolve()
        info = self._call("repository lookup", token, self._registry.repository_info, registry_repository, token)
        if not isinstance(info, Mapping) or info.get("is_private") is not True:
            raise DockerReleaseError("Docker Hub repository must be explicitly private")
        manifest = self._call("registry manifest lookup", token, self._registry.registry_manifest, registry_repository, tag, token)
        digest = self._registry_digest(manifest)
        reference = f"{repository}@{digest}"
        pull = self._call("authenticated pull", token, self._registry.authenticated_pull, reference, token)
        if not isinstance(pull, Mapping) or pull.get("digest") != digest:
            raise DockerReleaseError("authenticated pull did not verify the registry-reported digest")
        labels = pull.get("labels")
        if not isinstance(labels, Mapping) or labels.get("io.lehome.image-role") != purpose or labels.get("org.opencontainers.image.revision") != source_commit:
            raise DockerReleaseError("authenticated pull image labels do not match the canonical tag identity")
        return DockerImageRelease(purpose, repository, tag, source_commit, digest, reference)

    @staticmethod
    def require_digest_reference(reference: str) -> str:
        if not isinstance(reference, str) or reference.count("@") != 1:
            raise DockerReleaseError("template image must be a digest-qualified repository reference")
        repository, digest = reference.split("@", 1)
        DockerHubReleaseVerifier._validate_repository(repository)
        if not _DIGEST_RE.fullmatch(digest):
            raise DockerReleaseError("template image requires an exact sha256 digest")
        return reference

    @staticmethod
    def _validate_repository(repository: str) -> None:
        if repository != _REPOSITORY:
            raise DockerReleaseError("repository must be the exact private docker.io B1K release repository")

    @staticmethod
    def _validate_tag(tag: str) -> None:
        DockerHubReleaseVerifier.image_identity(tag)

    @staticmethod
    def image_identity(tag: str) -> tuple[str, str]:
        if not isinstance(tag, str) or "@" in tag:
            raise DockerReleaseError("tag must be one exact canonical role-prefixed source revision")
        match = _TAG_RE.fullmatch(tag)
        if match is None:
            raise DockerReleaseError("tag must be one exact canonical role-prefixed source revision")
        return ("training" if match.group(1) == "trainer" else "rollout", match.group(2))

    @staticmethod
    def _registry_digest(manifest: object) -> str:
        if not isinstance(manifest, Mapping):
            raise DockerReleaseError("registry manifest response is invalid")
        digest = manifest.get("digest")
        if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
            raise DockerReleaseError("registry-reported sha256 digest is required; local or tag-inferred digests are rejected")
        return digest

    @staticmethod
    def _call(operation: str, token: str, callback: Callable[..., Any], *args: object) -> Any:
        try:
            return callback(*args)
        except Exception:
            raise DockerReleaseError(f"{operation} failed") from None
