from __future__ import annotations

from types import SimpleNamespace

import pytest

import b1k_deploy.dockerhub as dockerhub_module

from b1k_deploy.dockerhub import (
    CommandResult,
    CredentialSourceError,
    DockerHubClient,
    DockerHubReleaseVerifier,
    DockerReleaseError,
    HttpResponse,
    TokenSource,
    sanitize_output,
)


TRAINER_REPOSITORY = "docker.io/ryanjin333/behavior1k-groot-n17-trainer"
ROLLOUT_REPOSITORY = "docker.io/ryanjin333/behavior1k-groot-n17-rollout"
_DOCKER_TEST_TOKEN = "dckr_pat_" + "abcdefghijklmnopqrstuvwxyz0123456789"
_HF_TEST_TOKEN = "hf_" + "abcdefghijklmnopqrstuvwxyz0123456789"


class FakeDockerRegistry:
    def __init__(self, *, private: bool = True, digest: str | None = None):
        self.private = private
        self.digest = digest or "sha256:" + "a" * 64
        self.calls: list[tuple[object, ...]] = []
        self.fail_pull = False

    def repository_info(self, repository: str, token: str):
        self.calls.append(("repository_info", repository, token))
        return {"is_private": self.private}

    def registry_manifest(self, repository: str, tag: str, token: str):
        self.calls.append(("registry_manifest", repository, tag, token))
        return {"digest": self.digest}

    def authenticated_pull(self, reference: str, token: str):
        self.calls.append(("authenticated_pull", reference, token))
        if self.fail_pull:
            raise RuntimeError(f"docker login failed with token {token}")
        return {"digest": self.digest}


def credential_store(name: str) -> str:
    assert name == "dockerhub"
    return _DOCKER_TEST_TOKEN


def test_private_registry_digest_and_authenticated_pull_are_required():
    registry = FakeDockerRegistry()
    verifier = DockerHubReleaseVerifier(registry, TokenSource.from_credential_store("dockerhub", credential_store))

    release = verifier.verify_private_image(TRAINER_REPOSITORY, "release-123")

    assert release.reference == TRAINER_REPOSITORY + "@sha256:" + "a" * 64
    assert release.digest == "sha256:" + "a" * 64
    assert registry.calls[-1] == (
        "authenticated_pull",
        TRAINER_REPOSITORY + "@sha256:" + "a" * 64,
        _DOCKER_TEST_TOKEN,
    )
    assert [call[1] for call in registry.calls[:2]] == [
        "ryanjin333/behavior1k-groot-n17-trainer",
        "ryanjin333/behavior1k-groot-n17-trainer",
    ]


@pytest.mark.parametrize(
    "repository, tag",
    [
        (TRAINER_REPOSITORY + ":latest", "release-123"),
        ("https://user:password@docker.io/ryanjin333/behavior1k-groot-n17-trainer", "release-123"),
        (TRAINER_REPOSITORY, ""),
        (TRAINER_REPOSITORY, "latest@sha256:" + "a" * 64),
        ("ryanjin333/behavior1k-groot-n17-trainer", "release-123"),
        ("docker.io/ryanjin333/lehome-groot-n17", "release-123"),
    ],
)
def test_tag_only_and_credential_bearing_image_inputs_are_rejected_before_registry_calls(repository, tag):
    registry = FakeDockerRegistry()
    verifier = DockerHubReleaseVerifier(registry, TokenSource.from_credential_store("dockerhub", credential_store))

    with pytest.raises(DockerReleaseError):
        verifier.verify_private_image(repository, tag)

    assert registry.calls == []


def test_public_repositories_and_non_registry_digests_are_rejected_before_pull():
    public = FakeDockerRegistry(private=False)
    verifier = DockerHubReleaseVerifier(public, TokenSource.from_credential_store("dockerhub", credential_store))
    with pytest.raises(DockerReleaseError, match="private"):
        verifier.verify_private_image(TRAINER_REPOSITORY, "release-123")
    assert public.calls == [("repository_info", "ryanjin333/behavior1k-groot-n17-trainer", ("dckr_pat_" + "abcdefghijklmnopqrstuvwxyz0123456789"))]

    local = FakeDockerRegistry(digest="local-build-sha256")
    verifier = DockerHubReleaseVerifier(local, TokenSource.from_credential_store("dockerhub", credential_store))
    with pytest.raises(DockerReleaseError, match="registry-reported"):
        verifier.verify_private_image(TRAINER_REPOSITORY, "release-123")
    assert [call[0] for call in local.calls] == ["repository_info", "registry_manifest"]


def test_authenticated_pull_must_report_the_registry_digest_and_redacts_errors():
    registry = FakeDockerRegistry()
    registry.fail_pull = True
    verifier = DockerHubReleaseVerifier(registry, TokenSource.from_credential_store("dockerhub", credential_store))

    with pytest.raises(DockerReleaseError) as error:
        verifier.verify_private_image(TRAINER_REPOSITORY, "release-123")

    assert ("dckr_pat_" + "abcdefghijklmnopqrstuvwxyz0123456789") not in str(error.value)
    assert "docker login failed" not in str(error.value)
    assert str(error.value) == "authenticated pull failed"


def test_credential_store_helper_failure_omits_arbitrary_unresolved_secret_and_helper_text():
    arbitrary_secret = "unrecognized helper secret: peony-cinder-47"
    helper_message = "credential helper threw an unexpected diagnostic"

    def broken_store(_name: str) -> str:
        raise RuntimeError(f"{helper_message}; {arbitrary_secret}")

    with pytest.raises(CredentialSourceError) as error:
        TokenSource.from_credential_store("dockerhub", broken_store).resolve()

    assert str(error.value) == "credential store lookup failed"
    assert arbitrary_secret not in str(error.value)
    assert helper_message not in str(error.value)


def test_injected_docker_domain_and_other_callback_errors_are_operation_only():
    arbitrary_secret = "unrecognized provider secret: violet-ember-83"
    provider_message = "registry supplied a raw diagnostic"

    class DomainFailure(FakeDockerRegistry):
        def repository_info(self, repository: str, token: str):
            raise DockerReleaseError(f"{provider_message}; {arbitrary_secret}")

    verifier = DockerHubReleaseVerifier(
        DomainFailure(), TokenSource.from_credential_store("dockerhub", credential_store)
    )
    with pytest.raises(DockerReleaseError) as domain_error:
        verifier.verify_private_image(TRAINER_REPOSITORY, "release-123")
    assert str(domain_error.value) == "repository lookup failed"
    assert arbitrary_secret not in str(domain_error.value)
    assert provider_message not in str(domain_error.value)

    class OtherFailure(FakeDockerRegistry):
        def authenticated_pull(self, reference: str, token: str):
            raise RuntimeError(f"{provider_message}; {arbitrary_secret}")

    verifier = DockerHubReleaseVerifier(
        OtherFailure(), TokenSource.from_credential_store("dockerhub", credential_store)
    )
    with pytest.raises(DockerReleaseError) as callback_error:
        verifier.verify_private_image(TRAINER_REPOSITORY, "release-123")
    assert str(callback_error.value) == "authenticated pull failed"
    assert arbitrary_secret not in str(callback_error.value)
    assert provider_message not in str(callback_error.value)


def test_digest_qualified_template_references_are_required():
    digest_reference = TRAINER_REPOSITORY + "@sha256:" + "a" * 64
    assert DockerHubReleaseVerifier.require_digest_reference(digest_reference) == digest_reference

    with pytest.raises(DockerReleaseError):
        DockerHubReleaseVerifier.require_digest_reference(TRAINER_REPOSITORY + ":release-123")
    with pytest.raises(DockerReleaseError):
        DockerHubReleaseVerifier.require_digest_reference(
            "ryanjin333/behavior1k-groot-n17-trainer@sha256:" + "a" * 64
        )
    with pytest.raises(DockerReleaseError):
        DockerHubReleaseVerifier.require_digest_reference(
            "docker.io/ryanjin333/lehome-groot-n17@sha256:" + "a" * 64
        )


def test_only_the_two_pinned_docker_repositories_can_be_released():
    registry = FakeDockerRegistry()
    verifier = DockerHubReleaseVerifier(registry, TokenSource.from_credential_store("dockerhub", credential_store))

    rollout = verifier.verify_private_image(ROLLOUT_REPOSITORY, "release-123")

    assert rollout.reference.startswith(ROLLOUT_REPOSITORY + "@sha256:")


def test_token_file_is_private_and_raw_credentials_are_not_constructor_inputs(tmp_path):
    token_file = tmp_path / "docker.token"
    token_file.write_text(_DOCKER_TEST_TOKEN + "\n", encoding="utf-8")
    token_file.chmod(0o600)
    token = TokenSource.from_token_file(token_file)
    assert token.resolve() == _DOCKER_TEST_TOKEN

    token_file.chmod(0o644)
    with pytest.raises(CredentialSourceError, match="0600"):
        TokenSource.from_token_file(token_file).resolve()


def test_token_file_open_failures_omit_arbitrary_path_text_before_any_token_is_resolved(tmp_path):
    arbitrary_secret = "unrecognized-path-secret-amber-31"
    token_file = tmp_path / arbitrary_secret

    with pytest.raises(CredentialSourceError) as error:
        TokenSource.from_token_file(token_file).resolve()

    assert str(error.value) == "token file cannot be opened"
    assert arbitrary_secret not in str(error.value)


def test_sanitize_output_redacts_assignments_urls_bearers_and_known_secrets():
    raw = (
        f"HF_TOKEN={_HF_TEST_TOKEN} "
        f"Bearer {_DOCKER_TEST_TOKEN} "
        "https://user:password@example.test/x custom-secret"
    )
    safe = sanitize_output(raw, known_secrets=("custom-secret",))

    assert "abcdefghijklmnopqrstuvwxyz" not in safe
    assert "password" not in safe
    assert "custom-secret" not in safe
    assert safe.count("[REDACTED]") >= 4


def test_concrete_docker_client_uses_current_hub_bearer_api_and_ephemeral_cli_config(monkeypatch):
    class Transport:
        def __init__(self):
            self.calls = []
            self.responses = [
                HttpResponse(200, {}, b'{"access_token": "hub-token"}'),
                HttpResponse(200, {}, b'{"is_private": true}'),
                HttpResponse(200, {}, b'{"token": "registry-token"}'),
                HttpResponse(200, {"Docker-Content-Digest": "sha256:" + "b" * 64}, b"{}"),
            ]

        def request(self, method, url, headers, *, timeout, body=None):
            self.calls.append((method, url, headers, timeout, body))
            return self.responses.pop(0)

    class Runner:
        def __init__(self):
            self.calls = []

        def run(self, arguments, *, stdin, env, timeout):
            self.calls.append((arguments, stdin, env, timeout))
            return CommandResult(0, "", "")

    transport = Transport()
    runner = Runner()
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    monkeypatch.setenv("DOCKER_HOST", "unix:///Users/user/.docker/run/docker.sock")
    monkeypatch.setenv("DOCKER_CONTEXT", "desktop-linux")
    context_calls = []

    def resolve_context(name, environment):
        context_calls.append((name, environment))
        return "unix:///Users/user/.docker/run/docker.sock"

    client = DockerHubClient("ryanjin333", transport=transport, runner=runner, docker_executable="/usr/local/bin/docker", context_resolver=resolve_context, timeout_seconds=11, pull_timeout_seconds=17)
    token = _DOCKER_TEST_TOKEN

    assert client.repository_info("ryanjin333/behavior1k-groot-n17-trainer", token) == {"is_private": True}
    manifest = client.registry_manifest("ryanjin333/behavior1k-groot-n17-trainer", "release-123", token)
    assert manifest["digest"] == "sha256:" + "b" * 64
    assert client.authenticated_pull(TRAINER_REPOSITORY + "@sha256:" + "b" * 64, token)["digest"] == "sha256:" + "b" * 64

    assert transport.calls[0][0:2] == ("POST", "https://hub.docker.com/v2/auth/token")
    expected_auth_body = ('{"identifier":"ryanjin333","secret":"' + token + '"}').encode()
    assert transport.calls[0][4] == expected_auth_body
    assert transport.calls[1][1].endswith("/v2/namespaces/ryanjin333/repositories/behavior1k-groot-n17-trainer")
    assert transport.calls[1][2] == {"Authorization": "Bearer hub-token"}
    assert transport.calls[2][1].startswith("https://auth.docker.io/token?")
    assert transport.calls[3][1].endswith("/v2/ryanjin333/behavior1k-groot-n17-trainer/manifests/release-123")
    assert runner.calls[0][0] == ("/usr/local/bin/docker", "login", "--username", "ryanjin333", "--password-stdin", "docker.io")
    assert runner.calls[0][1] == token
    assert runner.calls[1][0] == ("/usr/local/bin/docker", "pull", TRAINER_REPOSITORY + "@sha256:" + "b" * 64)
    assert runner.calls[0][2]["DOCKER_CONFIG"] == runner.calls[1][2]["DOCKER_CONFIG"]
    assert runner.calls[0][2]["PATH"] == "/usr/local/bin:/usr/bin"
    assert runner.calls[0][2]["DOCKER_HOST"] == "unix:///Users/user/.docker/run/docker.sock"
    assert "DOCKER_CONTEXT" not in runner.calls[0][2]
    assert context_calls == [("desktop-linux", {"PATH": "/usr/local/bin:/usr/bin", "DOCKER_HOST": "unix:///Users/user/.docker/run/docker.sock", "DOCKER_CONTEXT": "desktop-linux"})]
    assert runner.calls[0][3] == 17
    assert runner.calls[1][3] == 17


def test_concrete_docker_client_rejects_failed_transport_or_cli_without_output_leakage():
    class FailedTransport:
        def __init__(self):
            self.responses = [HttpResponse(200, {}, b'{"access_token":"hub-token"}'), HttpResponse(500, {}, b"provider diagnostic with secret meadow-pine-17")]

        def request(self, method, url, headers, *, timeout, body=None):
            return self.responses.pop(0)

    client = DockerHubClient("ryanjin333", transport=FailedTransport(), runner=None, docker_executable="/usr/bin/docker")
    with pytest.raises(DockerReleaseError) as error:
        client.repository_info("ryanjin333/behavior1k-groot-n17-trainer", "token")
    assert str(error.value) == "Docker Hub repository lookup failed"


def test_ephemeral_docker_environment_preserves_linux_endpoint_semantics_without_old_config(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("DOCKER_HOST", "tcp://127.0.0.1:2376")
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    monkeypatch.setenv("DOCKER_CERT_PATH", "/run/user/1000/docker-certs")
    monkeypatch.setenv("DOCKER_CONFIG", "/home/user/.docker-with-credentials")

    environment = DockerHubClient._docker_environment("/tmp/b1k-ephemeral-config")

    assert environment == {
        "PATH": "/usr/bin:/bin",
        "DOCKER_HOST": "tcp://127.0.0.1:2376",
        "DOCKER_TLS_VERIFY": "1",
        "DOCKER_CERT_PATH": "/run/user/1000/docker-certs",
        "DOCKER_CONFIG": "/tmp/b1k-ephemeral-config",
    }


def test_authenticated_pull_on_linux_default_endpoint_never_resolves_or_preserves_a_context(monkeypatch):
    class Runner:
        def __init__(self):
            self.calls = []

        def run(self, arguments, *, stdin, env, timeout):
            self.calls.append((arguments, stdin, env, timeout))
            return CommandResult(0, "", "")

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("DOCKER_HOST", "tcp://127.0.0.1:2376")
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    runner = Runner()
    client = DockerHubClient(
        "ryanjin333",
        runner=runner,
        docker_executable="/usr/bin/docker",
        context_resolver=lambda *_args: pytest.fail("a Linux default endpoint must not resolve a Docker context"),
    )

    client.authenticated_pull(TRAINER_REPOSITORY + "@sha256:" + "a" * 64, ("dckr_pat_" + "abcdefghijklmnopqrstuvwxyz0123456789"))

    assert runner.calls[0][2]["DOCKER_HOST"] == "tcp://127.0.0.1:2376"
    assert "DOCKER_CONTEXT" not in runner.calls[0][2]


def test_desktop_context_inspection_uses_the_original_config_before_ephemeral_login(monkeypatch):
    class Runner:
        def __init__(self):
            self.calls = []

        def run(self, arguments, *, stdin, env, timeout):
            self.calls.append((arguments, stdin, env, timeout))
            if arguments[1:3] == ("context", "inspect"):
                return CommandResult(0, "unix:///Users/user/.docker/run/docker.sock\n", ("diagnostic with dckr_pat_" + "abcdefghijklmnopqrstuvwxyz0123456789"))
            return CommandResult(0, "", "")

    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    monkeypatch.setenv("DOCKER_CONFIG", "/Users/user/.docker")
    monkeypatch.setenv("DOCKER_CONTEXT", "desktop-linux")
    runner = Runner()
    client = DockerHubClient("ryanjin333", runner=runner, docker_executable="/usr/local/bin/docker", timeout_seconds=11)

    client.authenticated_pull(TRAINER_REPOSITORY + "@sha256:" + "a" * 64, ("dckr_pat_" + "abcdefghijklmnopqrstuvwxyz0123456789"))

    inspect, login = runner.calls[:2]
    assert inspect[0] == ("/usr/local/bin/docker", "context", "inspect", "desktop-linux", "--format", "{{.Endpoints.docker.Host}}")
    assert inspect[1] is None
    assert inspect[2] == {"PATH": "/usr/local/bin:/usr/bin", "DOCKER_CONFIG": "/Users/user/.docker", "DOCKER_CONTEXT": "desktop-linux"}
    assert inspect[3] == 11
    assert login[2]["DOCKER_HOST"] == "unix:///Users/user/.docker/run/docker.sock"
    assert "DOCKER_CONTEXT" not in login[2]


def test_token_file_walk_requires_exact_0600_owned_regular_file_and_rejects_symlink_component(tmp_path):
    token = ("dckr_pat_" + "abcdefghijklmnopqrstuvwxyz0123456789")
    token_file = tmp_path / "tokens" / "docker.token"
    token_file.parent.mkdir()
    token_file.write_text(token, encoding="utf-8")
    token_file.chmod(0o400)
    with pytest.raises(CredentialSourceError, match="0600"):
        TokenSource.from_token_file(token_file).resolve()

    token_file.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(token_file.parent, target_is_directory=True)
    with pytest.raises(CredentialSourceError):
        TokenSource.from_token_file(link / "docker.token").resolve()


def test_token_file_walk_rejects_a_regular_0600_file_not_owned_by_the_current_user(tmp_path, monkeypatch):
    token_file = tmp_path / "docker.token"
    token_file.write_text(("dckr_pat_" + "abcdefghijklmnopqrstuvwxyz0123456789"), encoding="utf-8")
    token_file.chmod(0o600)
    original_fstat = dockerhub_module.os.fstat

    def wrong_owner(descriptor: int):
        metadata = original_fstat(descriptor)
        return SimpleNamespace(st_mode=metadata.st_mode, st_uid=metadata.st_uid + 1)

    monkeypatch.setattr(dockerhub_module.os, "fstat", wrong_owner)
    with pytest.raises(CredentialSourceError, match="owned"):
        TokenSource.from_token_file(token_file).resolve()
