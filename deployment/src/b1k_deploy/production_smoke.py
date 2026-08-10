"""Concrete, dry-run-first Vast and SSH transports for the capped smoke controller.

This module deliberately keeps provider credentials in Vast's configured local
key-file boundary.  It never shells out through a string, never discovers a
destroy target, and only reconciles a create by its exact run label.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import stat
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from .dockerhub import (
    CommandResult,
    CredentialSourceError,
    DockerCommandRunner,
    HttpTransport,
    SubprocessDockerRunner,
    TokenSource,
    UrllibTransport,
)
from .huggingface import HubProbeReceipt, HubRepository, HuggingFaceReleaseVerifier
from .production import _validate_executable_path, _validated_private_file, _vast_environment
from .publish import canonical_payload_hash
from .production import _project_template_readback
from .smoke import (
    RolloutRuntimeEvidence,
    RuntimeArtifactReceipt,
    SmokeCompatibility,
    SmokeError,
    SmokeOfferSelectionReceipt,
    TrainingRuntimeEvidence,
    SmokeTemplatePublicationReceipt,
    _canonical_image_release_identity,
)
from .vast import PROTECTED_INSTANCE_IDS, ProviderNotCreated


_ID_RE = re.compile(r"^[1-9][0-9]*$")
_LABEL_RE = re.compile(r"^b1k-smoke-[0-9a-f]{32}$")
_DIGEST_IMAGE_RE = re.compile(r"^docker\.io/ryanjin333/behavior1k-groot-n17@sha256:[0-9a-f]{64}$")
_REGISTRY_USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_REGISTRY_TOKEN_RE = re.compile(r"^[^\s'\"\\]{8,8192}$")
_HOST_RE = re.compile(r"^(?:[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?|\d{1,3}(?:\.\d{1,3}){3})$")
_MAX_TOOL_TIMEOUT = 55


def _remote_failure_category(stdout: object, stderr: object) -> str:
    """Map untrusted remote output to one fixed, secret-free diagnostic code."""
    output = f"{stdout}\n{stderr}".lower()
    categories = (
        (("cuda out of memory", "torch.outofmemoryerror"), "remote-cuda-out-of-memory"),
        (("gatedrepoerror", "401 unauthorized", "403 forbidden", "access denied"), "remote-access-denied"),
        (("no space left on device",), "remote-disk-full"),
        (("modulenotfounderror",), "remote-python-module-missing"),
        (("filenotfounderror",), "remote-file-missing"),
        (("timeoutexpired", "timed out"), "remote-timeout"),
        (("calledprocesserror",), "remote-subprocess-failed"),
        (("valueerror",), "remote-value-error"),
        (("runtimeerror",), "remote-runtime-error"),
        (("assertionerror",), "remote-assertion-failed"),
    )
    return next((category for needles, category in categories if any(needle in output for needle in needles)), "remote-command-failed")
_TEMPLATE_CREATE_RESULT_RE = re.compile(r"^New Template: ([1-9][0-9]*)$")
_MAX_TEMPLATE_CREATE_RESULT_BYTES = 65536
_MAX_TEMPLATE_DELETE_RESULT_BYTES = 4096
_TEMPLATE_RECONCILIATION_ATTEMPTS = 3
_VAST_CREATE_URL = "https://console.vast.ai/api/v0/asks/{offer_id}/"


class ProductionSmokeError(SmokeError):
    """A credential-free production transport failure."""


@dataclass(frozen=True)
class VastInstanceEndpoint:
    instance_id: str
    host: str
    port: int
    username: str = "root"


class VastCliSmokeClient:
    """Raw-JSON Vast CLI client with exact-ID creation and reconciliation only."""

    def __init__(
        self,
        *,
        vastai_executable: str | Path,
        api_key_file: Path,
        registry_username: str,
        registry_token_file: Path,
        create_transport: HttpTransport | None = None,
        runner: DockerCommandRunner | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self._vastai = _validate_executable_path(Path(vastai_executable), "Vast CLI executable")
        self._vast_api_key = TokenSource.from_token_file(
            _validated_private_file(api_key_file, "Vast API key file")
        )
        if not _REGISTRY_USERNAME_RE.fullmatch(registry_username):
            raise ProductionSmokeError("Docker registry username is invalid")
        self._registry_username = registry_username
        self._registry_token = TokenSource.from_token_file(
            _validated_private_file(registry_token_file, "Docker registry token file")
        )
        if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or not 0 < timeout_seconds <= _MAX_TOOL_TIMEOUT:
            raise ProductionSmokeError("Vast CLI timeout is invalid")
        self._runner = runner or SubprocessDockerRunner()
        self._create_transport = create_transport or UrllibTransport()
        self._timeout = timeout_seconds

    def select_offer(self, purpose: str) -> SmokeOfferSelectionReceipt:
        if purpose not in {"training-smoke", "rollout-smoke"}:
            raise ProductionSmokeError("smoke purpose is invalid")
        # Smoke contracts deliberately use bounded canary preflight, never the
        # production 15k-step hardware gate.  These are role-specific image
        # floors: CUDA GR00T canary (12 GB/100 GB) and headless Isaac rollout
        # (24 GB/300 GB).
        minimum_vram_gb = 24
        minimum_vram_mib = minimum_vram_gb * 1024
        requested_disk = 100 if purpose == "training-smoke" else 300
        # Vast CLI 1.5.2 returns cpu_ram in MiB, duration in seconds, float
        # disk_space, `verification`/`vericode`, cuda_max_good, and dph_total.
        # Keep the provider-side predicate simple and revalidate those exact
        # raw fields below before any offer can enter a receipt.
        # CLI query values are expressed in GB (the client multiplies them by
        # 1000), while raw offer rows report MiB.  Query in GB, then enforce
        # the stricter raw-MiB threshold before accepting a receipt.
        query = f"num_gpus=1 cpu_arch=amd64 gpu_ram>={minimum_vram_gb} compute_cap>=750 cuda_max_good>=12 rentable=True"
        rows = self._rows((str(self._vastai), "search", "offers", query, "-i", "--storage", str(requested_disk), "--order", "dph_total", "--raw"))
        candidates: list[tuple[Decimal, Mapping[str, Any], SmokeCompatibility]] = []
        for row in rows:
            try:
                compatibility = SmokeCompatibility(
                    verified_datacenter=_verified_datacenter(row),
                    gpu_compatible=_gpu_compatible(row, minimum_vram_mib),
                    disk_gb=_floor_units(row, "disk_space", 1),
                    ram_gb=_floor_units(row, "cpu_ram", 1024),
                    network_mbps=_floor_units(row, "inet_down", 1),
                    maximum_duration_minutes=_floor_units(row, "duration", 60),
                    selection="cheapest-compatible-verified",
                )
                compatibility.validate()
                rate = _rate(row.get("dph_total", row.get("dph_base")))
                if rate <= 0 or _exact_id(row.get("id")) in PROTECTED_INSTANCE_IDS:
                    continue
            except (ProductionSmokeError, SmokeError):
                continue
            candidates.append((rate, row, compatibility))
        if not candidates:
            raise ProductionSmokeError("no compatible verified one-GPU Vast offer is available")
        rate, row, compatibility = min(candidates, key=lambda item: (item[0], _exact_id(item[1].get("id"))))
        return SmokeOfferSelectionReceipt(_exact_id(row.get("id")), rate, _string(row, "gpu_name", fallback="gpu"), compatibility)

    def _template_readback(self, template_id: str) -> Mapping[str, Any]:
        exact = _exact_id(template_id)
        rows = self._rows((str(self._vastai), "--raw", "search", "templates", f"id=={exact}"))
        matches = [row for row in rows if _exact_id(row.get("id")) == exact]
        if len(matches) != 1:
            raise ProductionSmokeError("exact Vast template readback is missing or ambiguous")
        try:
            return _project_template_readback(matches[0])
        except Exception:
            raise ProductionSmokeError("exact Vast template readback is incomplete") from None

    def attest_template_binding(
        self,
        *,
        template_id: str,
        image_reference: str,
        payload_hash: str,
        required_smoke_environment: str | None = None,
    ) -> str:
        """Bind the exact provider template to a durable canonical receipt."""
        if not _DIGEST_IMAGE_RE.fullmatch(image_reference) or not re.fullmatch(r"[0-9a-f]{64}", payload_hash):
            raise ProductionSmokeError("template binding is invalid")
        readback = self._template_readback(template_id)
        if canonical_payload_hash(readback) != payload_hash or readback.get("image") != image_reference:
            raise ProductionSmokeError("exact Vast template readback does not match the publication receipt")
        digest = image_reference.rsplit("@", 1)[1]
        environment = readback.get("env")
        if not isinstance(environment, str) or f"CONTAINER_DIGEST={digest}" not in environment:
            raise ProductionSmokeError("exact Vast template does not bind the container digest")
        if required_smoke_environment is not None:
            if required_smoke_environment not in {"B1K_TRAINING_SMOKE_RUNTIME", "B1K_ROLLOUT_SMOKE_RUNTIME"} or re.search(
                rf"(?:^| )-e {re.escape(required_smoke_environment)}=1(?: |$)", environment
            ) is None:
                raise ProductionSmokeError("exact Vast template lacks the required smoke-mode environment")
        provider_hash = self._rows((str(self._vastai), "--raw", "search", "templates", f"id=={_exact_id(template_id)}"))[0].get("hash_id")
        if not isinstance(provider_hash, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", provider_hash):
            raise ProductionSmokeError("exact Vast template readback lacks one provider template hash")
        return provider_hash

    def create_instance(self, request: Mapping[str, Any], *, timeout_seconds: int) -> str:
        self._tool_timeout(timeout_seconds)
        offer_id = _exact_id(request.get("offer_id"))
        template_id = _exact_id(request.get("template_id"))
        label = _label(request.get("idempotency_key"))
        if offer_id in PROTECTED_INSTANCE_IDS:
            raise ProductionSmokeError("protected instance cannot be selected")
        purpose = request.get("purpose")
        if purpose not in {"training-smoke", "rollout-smoke"}:
            raise ProductionSmokeError("smoke create purpose is invalid")
        disk = _positive_int(request.get("disk_gb", 100 if purpose == "training-smoke" else 300), "disk")
        bid = _rate(request.get("hourly_rate_usd"))
        image_reference = request.get("image_reference")
        payload_hash = request.get("payload_hash")
        if not isinstance(image_reference, str) or not isinstance(payload_hash, str):
            raise ProductionSmokeError("create request lacks a publication-bound template identity")
        required_environment = "B1K_TRAINING_SMOKE_RUNTIME" if purpose == "training-smoke" else "B1K_ROLLOUT_SMOKE_RUNTIME"
        template_hash = self.attest_template_binding(
            template_id=template_id,
            image_reference=image_reference,
            payload_hash=payload_hash,
            required_smoke_environment=required_environment,
        )
        try:
            registry_token = self._registry_token.resolve()
        except CredentialSourceError:
            raise ProductionSmokeError("Docker registry credential is unavailable") from None
        if not _REGISTRY_TOKEN_RE.fullmatch(registry_token):
            raise ProductionSmokeError("Docker registry token is invalid")
        try:
            api_key = self._vast_api_key.resolve()
        except CredentialSourceError:
            raise ProductionSmokeError("Vast API credential is unavailable") from None
        if not _REGISTRY_TOKEN_RE.fullmatch(api_key):
            raise ProductionSmokeError("Vast API credential is invalid")
        payload = {
            "client_id": "me",
            "image": None,
            # Any per-instance env replaces the template's complete Docker
            # options on Vast.  The ephemeral smoke template already carries
            # its smoke-mode flag, so preserve that template verbatim here.
            "env": {},
            "price": float(bid),
            "disk": disk,
            "label": label,
            "extra": None,
            "onstart": None,
            "image_login": f"-u {self._registry_username} -p {registry_token} docker.io",
            "python_utf8": False,
            "lang_utf8": False,
            "use_jupyter_lab": False,
            "jupyter_dir": None,
            "force": False,
            "cancel_unavail": True,
            "template_hash_id": template_hash,
            "user": None,
        }
        try:
            response = self._create_transport.request(
                "PUT",
                _VAST_CREATE_URL.format(offer_id=offer_id),
                {
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "b1k-deploy/1",
                },
                timeout=timeout_seconds,
                body=json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            )
        except Exception:
            raise ProductionSmokeError("Vast create request failed ambiguously") from None
        if not 200 <= response.status < 300:
            raise ProductionSmokeError("Vast create request failed ambiguously")
        try:
            result = json.loads(response.body)
        except (TypeError, ValueError):
            raise ProductionSmokeError("Vast create returned invalid JSON") from None
        if not isinstance(result, Mapping):
            raise ProductionSmokeError("Vast create returned invalid raw JSON")
        if result.get("success") is False:
            raise ProviderNotCreated("Vast rejected create before provisioning")
        instance_id = result.get("new_contract")
        if instance_id is None:
            # A timeout or malformed success could have created a billable VM;
            # the controller must reconcile by the exact label.
            raise ProductionSmokeError("Vast create has no exact instance ID")
        return _non_protected_id(instance_id)

    @staticmethod
    def new_ephemeral_smoke_template_name(purpose: str) -> str:
        if purpose not in {"training", "rollout"}:
            raise ProductionSmokeError("ephemeral smoke template purpose is invalid")
        stem = "trainer" if purpose == "training" else "rollout"
        return f"b1k-{'training' if purpose == 'training' else 'rollout'}-smoke-{uuid.uuid4().hex}"

    def create_ephemeral_smoke_template(self, purpose: str, production: SmokeTemplatePublicationReceipt, *, name: str) -> SmokeTemplatePublicationReceipt:
        """Create one separately named low-resource smoke template from readback.

        The production template is only a digest/onstart/env source.  Resource
        relaxation is confined to this ephemeral template and never mutates
        the production payload or its receipt.
        """
        if purpose not in {"training", "rollout"} or production.template_id is None or not re.fullmatch(rf"b1k-{'training' if purpose == 'training' else 'rollout'}-smoke-[0-9a-f]{{32}}", name):
            raise ProductionSmokeError("production template receipt is invalid")
        self.attest_template_binding(template_id=production.template_id, image_reference=production.image_release.reference, payload_hash=production.payload_hash)
        payload = dict(self._template_readback(production.template_id))
        environment = "B1K_TRAINING_SMOKE_RUNTIME" if purpose == "training" else "B1K_ROLLOUT_SMOKE_RUNTIME"
        template_env = payload.get("env")
        if not isinstance(template_env, str) or f"-e {environment}=" in template_env:
            raise ProductionSmokeError("production template environment cannot be specialized for smoke")
        payload["env"] = f"{template_env} -e {environment}=1"
        filters = dict(payload["extra_filters"])
        payload["name"] = name
        payload["recommended_disk_space"] = 100 if purpose == "training" else 300
        filters["gpu_ram"] = {"gte": 12000 if purpose == "training" else 24000}
        filters["cpu_ram"] = {"gte": 16000 if purpose == "training" else 32000}
        filters["cpu_cores_effective"] = {"gte": 4 if purpose == "training" else 8}
        filters.pop("gpu_name", None)
        payload["extra_filters"] = filters
        docker_login_repo = self._template_registry_repo(production.template_id)
        command = self._template_create_command(payload, docker_login_repo=docker_login_repo)
        created_ids: set[str] = set()
        create_error: Exception | None = None
        try:
            created_ids.add(self._create_template_id(command))
        except Exception as error:
            create_error = error
        rows: list[Mapping[str, Any]] = []
        try:
            rows = self._ephemeral_templates_by_name(name)
        except Exception as error:
            if create_error is None:
                create_error = error
        try:
            created_ids.update(_exact_id(row.get("id")) for row in rows)
        except Exception as error:
            if create_error is None:
                create_error = ProductionSmokeError("ephemeral smoke template reconciliation returned an invalid ID")
                create_error.__cause__ = error

        primary_error: Exception | None = create_error
        template_id: str | None = None
        payload_hash: str | None = None
        if primary_error is None:
            if len(created_ids) != 1:
                primary_error = ProductionSmokeError("ephemeral smoke template readback is missing or ambiguous")
            else:
                template_id = next(iter(created_ids))
                try:
                    readback = self._template_readback(template_id)
                    payload_hash = canonical_payload_hash(readback)
                    if readback != payload or readback.get("image") != production.image_release.reference:
                        raise ProductionSmokeError("ephemeral smoke template readback drifted")
                    if self._template_registry_repo(template_id) != docker_login_repo:
                        raise ProductionSmokeError("ephemeral smoke template private registry reference drifted")
                except Exception as error:
                    primary_error = ProductionSmokeError("ephemeral smoke template readback failed")
                    primary_error.__cause__ = error

        if primary_error is not None:
            if created_ids:
                try:
                    self._cleanup_ephemeral_templates(name, created_ids)
                except Exception as cleanup_error:
                    raise primary_error from cleanup_error
            raise primary_error

        if template_id is None or payload_hash is None:  # pragma: no cover - guarded above.
            raise ProductionSmokeError("ephemeral smoke template reconciliation did not produce a receipt")
        return SmokeTemplatePublicationReceipt(template_id, production.image_release, payload_hash)

    def destroy_ephemeral_smoke_template(self, template: SmokeTemplatePublicationReceipt) -> None:
        template_id = _exact_id(template.template_id)
        row = self._template_readback(template_id)
        name = row.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"b1k-(?:training|rollout)-smoke-[0-9a-f]{32}", name):
            raise ProductionSmokeError("only an exact ephemeral smoke template may be deleted")
        if canonical_payload_hash(row) != template.payload_hash or row.get("image") != template.image_release.reference:
            raise ProductionSmokeError("ephemeral smoke template receipt drifted before deletion")
        self._delete_template(template_id)
        matches = [item for item in self._rows((str(self._vastai), "--raw", "search", "templates", f"id=={template_id}")) if _exact_id(item.get("id")) == template_id]
        if matches:
            raise ProductionSmokeError("ephemeral smoke template absence was not verified")

    def _ephemeral_templates_by_name(self, name: str) -> list[Mapping[str, Any]]:
        """Bound a provider's eventually-consistent exact-name readback."""
        command = (str(self._vastai), "--raw", "search", "templates", f"name=={name}")
        for _ in range(_TEMPLATE_RECONCILIATION_ATTEMPTS):
            rows = [row for row in self._rows(command) if row.get("name") == name]
            if rows:
                return rows
        return []

    def _cleanup_ephemeral_templates(self, name: str, template_ids: set[str]) -> None:
        """Delete only exact recovered IDs, then prove both ID and name absence."""
        cleanup_error: Exception | None = None

        def attempt(operation: Callable[[], None]) -> None:
            nonlocal cleanup_error
            try:
                operation()
            except Exception as error:
                if cleanup_error is None:
                    cleanup_error = error

        exact_ids = tuple(sorted(_exact_id(template_id) for template_id in template_ids))
        for template_id in exact_ids:
            attempt(lambda template_id=template_id: self._delete_template(template_id))
        for template_id in exact_ids:
            def prove_id_absent(template_id: str = template_id) -> None:
                rows = self._rows((str(self._vastai), "--raw", "search", "templates", f"id=={template_id}"))
                matches = [row for row in rows if _exact_id(row.get("id")) == template_id]
                if matches:
                    raise ProductionSmokeError("ephemeral smoke template ID cleanup absence was not verified")
            attempt(prove_id_absent)

        def prove_name_absent() -> None:
            if self._ephemeral_templates_by_name(name):
                raise ProductionSmokeError("ephemeral smoke template name cleanup absence was not verified")
        attempt(prove_name_absent)
        if cleanup_error is not None:
            raise cleanup_error

    def _delete_template(self, template_id: str) -> None:
        """Delete one exact template without assuming Vast's text output is JSON."""
        exact = _exact_id(template_id)
        arguments = (str(self._vastai), "--raw", "delete", "template", "--template-id", exact)
        try:
            result = self._runner.run(arguments, stdin=None, env=_vast_environment(), timeout=self._timeout)
        except Exception:
            raise ProductionSmokeError("Vast template deletion failed") from None
        if (
            not isinstance(result, CommandResult)
            or result.returncode != 0
            or not isinstance(result.stdout, str)
            or not isinstance(result.stderr, str)
            or bool(result.stderr.strip())
        ):
            raise ProductionSmokeError("Vast template deletion failed")
        line = result.stdout[:-1] if result.stdout.endswith("\n") else result.stdout
        if line.endswith("\r"):
            line = line[:-1]
        if (
            not line
            or "\n" in line
            or len(line.encode("utf-8")) > _MAX_TEMPLATE_DELETE_RESULT_BYTES
            or line.casefold().startswith(("error", "the response is not valid json"))
        ):
            raise ProductionSmokeError("Vast template deletion failed")

    def _template_registry_repo(self, template_id: str) -> str:
        expected = _exact_id(template_id)
        rows = self._rows((str(self._vastai), "--raw", "search", "templates", f"id=={expected}"))
        matches = [row for row in rows if _exact_id(row.get("id")) == expected]
        if len(matches) != 1 or matches[0].get("docker_login_repo") != "docker.io":
            raise ProductionSmokeError("exact Vast template lacks the approved private registry reference")
        return "docker.io"

    def _template_create_command(self, payload: Mapping[str, Any], *, docker_login_repo: str) -> tuple[str, ...]:
        from .production import _search_query
        required = ("name", "image", "env", "onstart", "recommended_disk_space", "extra_filters")
        if docker_login_repo != "docker.io" or any(key not in payload for key in required) or payload.get("private") is not True or payload.get("runtype") != "ssh" or payload.get("use_ssh") is not True or payload.get("ssh_direct") is not True or payload.get("jup_direct") is not False:
            raise ProductionSmokeError("ephemeral smoke template payload is invalid")
        return (str(self._vastai), "--raw", "create", "template", "--name", str(payload["name"]), "--image", str(payload["image"]), "--login", docker_login_repo, "--env", str(payload["env"]), "--ssh", "--direct", "--onstart-cmd", str(payload["onstart"]), "--search_params", _search_query(payload["extra_filters"]), "--no-default", "--disk_space", str(payload["recommended_disk_space"]))

    def find_instance_by_idempotency_key(self, key: str, *, timeout_seconds: int) -> str | None:
        self._tool_timeout(timeout_seconds)
        label = _label(key)
        matches = [row for row in self._instances(timeout_seconds) if row.get("label") == label]
        if not matches:
            return None
        if len(matches) != 1:
            raise ProductionSmokeError("Vast label reconciliation is ambiguous")
        return _non_protected_id(matches[0].get("id"))

    def destroy_instance(self, instance_id: str, *, timeout_seconds: int) -> None:
        self._tool_timeout(timeout_seconds)
        exact = _non_protected_id(instance_id)
        # No query, label, or discovery is ever accepted here.
        arguments = (str(self._vastai), "destroy", "instance", exact, "--yes")
        try:
            result = self._runner.run(arguments, stdin=None, env=_vast_environment(), timeout=timeout_seconds)
        except Exception:
            raise ProductionSmokeError("Vast instance destruction failed") from None
        if (
            not isinstance(result, CommandResult)
            or result.returncode != 0
            or not isinstance(result.stdout, str)
            or not isinstance(result.stderr, str)
            or bool(result.stderr.strip())
        ):
            raise ProductionSmokeError("Vast instance destruction failed")
        line = result.stdout[:-1] if result.stdout.endswith("\n") else result.stdout
        if line.endswith("\r"):
            line = line[:-1]
        if line != f"destroying instance {exact}.":
            raise ProductionSmokeError("Vast instance destruction failed")

    def list_instance_ids(self, *, timeout_seconds: float) -> tuple[str, ...]:
        # Listing is read-only: a protected existing instance may be visible but
        # must never cause another run's cleanup proof to fail.
        return tuple(_exact_id(row.get("id")) for row in self._instances(_tool_timeout_float(timeout_seconds)))

    def endpoint(self, instance_id: str, *, timeout_seconds: int) -> VastInstanceEndpoint:
        self._tool_timeout(timeout_seconds)
        exact = _non_protected_id(instance_id)
        matches = [row for row in self._instances(timeout_seconds) if _exact_id(row.get("id")) == exact]
        if len(matches) != 1:
            raise ProductionSmokeError("exact Vast instance readback is missing or ambiguous")
        row = matches[0]
        host = row.get("ssh_host") or row.get("public_ipaddr") or row.get("public_ip")
        port = row.get("ssh_port") or row.get("port")
        username = row.get("ssh_user") or "root"
        if not isinstance(host, str) or not _HOST_RE.fullmatch(host) or not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535 or username != "root":
            raise ProductionSmokeError("exact Vast instance readback lacks a safe SSH endpoint")
        return VastInstanceEndpoint(exact, host, port, username)

    def _instances(self, timeout_seconds: int) -> list[Mapping[str, Any]]:
        return self._rows((str(self._vastai), "--raw", "show", "instances"), timeout_seconds)

    def _rows(self, arguments: tuple[str, ...], timeout_seconds: int | None = None) -> list[Mapping[str, Any]]:
        value = self._json(arguments, self._timeout if timeout_seconds is None else timeout_seconds)
        rows = value.get("instances") if isinstance(value, Mapping) and "instances" in value else value
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise ProductionSmokeError("Vast CLI returned invalid raw JSON rows")
        return [dict(row) for row in rows]

    def _json(self, arguments: tuple[str, ...], timeout_seconds: int) -> Any:
        try:
            result = self._runner.run(arguments, stdin=None, env=_vast_environment(), timeout=timeout_seconds)
        except Exception:
            raise ProductionSmokeError("Vast CLI operation failed") from None
        if not isinstance(result, CommandResult) or result.returncode != 0:
            raise ProductionSmokeError("Vast CLI operation failed")
        try:
            parsed = json.loads(result.stdout)
        except (TypeError, ValueError):
            raise ProductionSmokeError("Vast CLI returned invalid raw JSON") from None
        return parsed

    def _create_template_id(self, arguments: tuple[str, ...]) -> str:
        """Accept only Vast's documented one-line raw create-template result."""
        try:
            result = self._runner.run(arguments, stdin=None, env=_vast_environment(), timeout=self._timeout)
        except Exception:
            raise ProductionSmokeError("Vast template creation failed") from None
        if not isinstance(result, CommandResult) or result.returncode != 0 or not isinstance(result.stdout, str):
            raise ProductionSmokeError("Vast template creation failed")
        line = result.stdout[:-1] if result.stdout.endswith("\n") else result.stdout
        if line.endswith("\r"):
            line = line[:-1]
        match = _TEMPLATE_CREATE_RESULT_RE.fullmatch(line)
        if match is not None:
            return _exact_id(match.group(1))
        prefix = "New Template: "
        if not line.startswith(prefix) or len(line.encode("utf-8")) > _MAX_TEMPLATE_CREATE_RESULT_BYTES:
            raise ProductionSmokeError("Vast template creation returned no exact template ID")
        try:
            payload = ast.literal_eval(line.removeprefix(prefix))
        except (MemoryError, RecursionError, SyntaxError, ValueError):
            raise ProductionSmokeError("Vast template creation returned no exact template ID") from None
        if not isinstance(payload, Mapping):
            raise ProductionSmokeError("Vast template creation returned no exact template ID")
        return _exact_id(payload.get("id"))

    @staticmethod
    def _tool_timeout(timeout_seconds: int) -> int:
        if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or not 0 < timeout_seconds <= _MAX_TOOL_TIMEOUT:
            raise ProductionSmokeError("Vast tool timeout is invalid")
        return timeout_seconds


class SshSmokeRemote:
    """Strict-host-key SSH implementation of the controller's runtime protocol.

    Runtime commands remain long-running *inside* the remote Docker daemon,
    while every local ssh/keyscan subprocess has a short, finite deadline.
    """

    def __init__(
        self,
        *,
        vast: VastCliSmokeClient,
        identity_file: Path,
        known_hosts: Path,
        training_image: str,
        rollout_image: str,
        training_release: DockerImageRelease | None = None,
        rollout_release: DockerImageRelease | None = None,
        hub_verifier: HuggingFaceReleaseVerifier,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._vast = vast
        self._identity = _private_identity(identity_file)
        self._known_hosts = _campaign_known_hosts(known_hosts)
        if not _DIGEST_IMAGE_RE.fullmatch(training_image):
            raise ProductionSmokeError("training smoke image must be one canonical trainer digest")
        if not _DIGEST_IMAGE_RE.fullmatch(rollout_image):
            raise ProductionSmokeError("rollout smoke image must be one canonical rollout digest")
        if not _canonical_image_release_identity(training_release, "training") or training_release.reference != training_image:
            raise ProductionSmokeError("training smoke release receipt is invalid")
        if not _canonical_image_release_identity(rollout_release, "rollout") or rollout_release.reference != rollout_image:
            raise ProductionSmokeError("rollout smoke release receipt is invalid")
        if training_release.source_commit != rollout_release.source_commit:
            raise ProductionSmokeError("training and rollout smoke releases must share one source commit")
        self._training_image, self._rollout_image = training_image, rollout_image
        self._training_release, self._rollout_release = training_release, rollout_release
        self._hub = hub_verifier
        self._runner, self._clock, self._sleep = runner, clock, sleep
        self._endpoints: dict[str, VastInstanceEndpoint] = {}

    def wait_for_ssh(self, instance_id: str, timeout_seconds: int, poll_interval_seconds: int) -> str:
        endpoint = self._wait_endpoint(instance_id, timeout_seconds, poll_interval_seconds, require_ssh=True)
        return f"ssh://{endpoint.host}:{endpoint.port}"

    def wait_for_runtime(self, instance_id: str, purpose: str, timeout_seconds: int, poll_interval_seconds: int) -> str:
        if purpose not in {"training-smoke", "rollout-smoke"}:
            raise ProductionSmokeError("runtime purpose is invalid")
        exact = _non_protected_id(instance_id)
        deadline = self._deadline(timeout_seconds)
        if exact not in self._endpoints:
            self._wait_endpoint(exact, timeout_seconds, poll_interval_seconds, require_ssh=True)
        # Vast direct SSH enters the template container itself.  Probe the
        # already-running image process directly; a nested Docker daemon/socket
        # is neither assumed nor required.
        marker = "/workspace/smoke-canary/training-ready" if purpose == "training-smoke" else "/workspace/smoke-canary/rollout-ready"
        image = self._training_image if purpose == "training-smoke" else self._rollout_image
        digest = image.rsplit("@", 1)[1]
        command = ("/bin/sh", "-c", f"test \"$(stat -c '%u' {marker})\" = 10001 && test \"$(stat -c '%a' {marker})\" = 600 && /usr/bin/grep -zFxq 'CONTAINER_DIGEST={digest}' /proc/1/environ")
        last_error: Exception | None = None
        while self._remaining(deadline) > 0:
            try:
                self._ssh(exact, command, min(_MAX_TOOL_TIMEOUT, max(1, int(self._remaining(deadline)))))
                return "ready"
            except Exception as error:
                last_error = error
                self._sleep(min(float(poll_interval_seconds), self._remaining(deadline)))
        raise ProductionSmokeError("runtime readiness timed out") from last_error

    def run_training_contract(self, run_id: str, instance_id: str, timeout_seconds: int) -> TrainingRuntimeEvidence:
        _label(run_id)
        try:
            payload = self._runtime_json(instance_id, _training_command(self._training_image, f"b1k-bootstrap-{_run_hex(run_id)}-smoke-model"), timeout_seconds, self._training_image)
            _require_fields(payload, {"runtime_uid", "token_file_uid", "token_file_mode", "gpu_count", "optimizer_steps", "lifecycle_preflight", "container_digest", "checkpoint_bucket_probe"})
            artifact = RuntimeArtifactReceipt.from_hub_probe(run_id, "smoke-model", self._remote_probe("model", "smoke-model", run_id, payload))
            return TrainingRuntimeEvidence(
                self._training_release,
                runtime_uid=_positive_int(payload["runtime_uid"], "runtime uid"),
                token_file_uid=_positive_int(payload["token_file_uid"], "token uid"),
                token_file_mode=_mode(payload["token_file_mode"]),
                gpu_count=_positive_int(payload["gpu_count"], "gpu count"),
                optimizer_steps=_positive_int(payload["optimizer_steps"], "optimizer steps"),
                lifecycle_preflight=_exact_string(payload["lifecycle_preflight"], "passed"),
                artifact_label="smoke",
                artifact=artifact,
                checkpoint_bucket_probe=_exact_string(payload["checkpoint_bucket_probe"], "passed"),
            )
        except Exception as runtime_failure:
            self._raise_after_runtime_reconciliation("model", run_id, ("smoke-model",), runtime_failure)

    def run_rollout_contract(self, run_id: str, instance_id: str, timeout_seconds: int) -> RolloutRuntimeEvidence:
        _label(run_id)
        run_hex = _run_hex(run_id)
        try:
            payload = self._runtime_json(instance_id, _rollout_command(self._rollout_image, f"b1k-bootstrap-{run_hex}-success-fixture", f"b1k-bootstrap-{run_hex}-failure-fixture"), timeout_seconds, self._rollout_image)
            _require_fields(payload, {"gpu_count", "eula_environment", "warp_runtime", "headless_loads", "resets", "rgb_observation_count", "action_mapping_count", "evaluator_outcome", "container_digest"})
            fixture_receipts: list[RuntimeArtifactReceipt] = []
            primary_failure: Exception | None = None
            later_failure: Exception | None = None
            for classification in ("success-fixture", "failure-fixture"):
                try:
                    receipt = self._remote_probe("dataset", classification, run_id, _fixture_payload(payload, classification))
                    fixture_receipts.append(RuntimeArtifactReceipt.from_hub_probe(run_id, classification, receipt))
                except Exception as error:
                    if primary_failure is None:
                        primary_failure = error
                    else:
                        later_failure = error
            if primary_failure is not None:
                if later_failure is not None:
                    raise primary_failure from later_failure
                raise primary_failure
            fixtures = tuple(fixture_receipts)
            return RolloutRuntimeEvidence(
                self._rollout_release, _positive_int(payload["gpu_count"], "gpu count"),
                _exact_string(payload["eula_environment"], "OMNI_KIT_ACCEPT_EULA=YES"),
                _exact_string(payload["warp_runtime"], "bundled-compatible"),
                _positive_int(payload["headless_loads"], "headless loads"), _positive_int(payload["resets"], "resets"), "ok",
                _positive_int(payload["rgb_observation_count"], "RGB observations"), _positive_int(payload["action_mapping_count"], "action mappings"),
                _exact_one_of(payload["evaluator_outcome"], {"terminal", "quarantined"}), fixtures,
            )
        except Exception as runtime_failure:
            self._raise_after_runtime_reconciliation("dataset", run_id, ("success-fixture", "failure-fixture"), runtime_failure)

    def list_instance_ids(self, timeout_seconds: float) -> tuple[str, ...]:
        return self._vast.list_instance_ids(timeout_seconds=timeout_seconds)

    def ssh_endpoint_unreachable(self, instance_id: str, endpoint: str | None, timeout_seconds: float, poll_interval_seconds: int) -> bool:
        if endpoint is None:
            return True
        deadline = self._deadline(timeout_seconds)
        while True:
            try:
                self._ssh(instance_id, ("true",), min(_MAX_TOOL_TIMEOUT, max(1, int(self._remaining(deadline)))))
            except ProductionSmokeError:
                return True
            if self._remaining(deadline) <= 0:
                return False
            self._sleep(min(float(poll_interval_seconds), self._remaining(deadline)))

    def _wait_endpoint(self, instance_id: str, timeout_seconds: int, poll_interval_seconds: int, *, require_ssh: bool) -> VastInstanceEndpoint:
        exact = _non_protected_id(instance_id)
        deadline = self._deadline(timeout_seconds)
        last_error: Exception | None = None
        while self._remaining(deadline) > 0:
            try:
                endpoint = self._vast.endpoint(exact, timeout_seconds=min(_MAX_TOOL_TIMEOUT, max(1, int(self._remaining(deadline)))))
                self._endpoints[exact] = endpoint
                self._keyscan(endpoint, min(_MAX_TOOL_TIMEOUT, max(1, int(self._remaining(deadline)))))
                if not require_ssh:
                    return endpoint
                self._ssh(exact, ("true",), min(_MAX_TOOL_TIMEOUT, max(1, int(self._remaining(deadline)))))
                return endpoint
            except Exception as error:
                last_error = error
                self._sleep(min(float(poll_interval_seconds), self._remaining(deadline)))
        raise ProductionSmokeError("SSH readiness timed out") from last_error

    def _keyscan(self, endpoint: VastInstanceEndpoint, timeout_seconds: int) -> None:
        # Seed only the campaign-local file, then every SSH connection is strict.
        command = ("ssh-keyscan", "-T", str(min(timeout_seconds, 15)), "-p", str(endpoint.port), endpoint.host)
        completed = self._run(command, timeout=min(timeout_seconds, 20))
        if completed.returncode != 0 or not completed.stdout.strip():
            raise ProductionSmokeError("SSH host key scan failed")
        try:
            descriptor = os.open(self._known_hosts, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
                raise ProductionSmokeError("campaign known_hosts file is unsafe")
            os.write(descriptor, completed.stdout.encode("utf-8"))
            os.fsync(descriptor)
        except OSError:
            raise ProductionSmokeError("campaign known_hosts file is unsafe") from None
        finally:
            try:
                os.close(descriptor)
            except UnboundLocalError:
                pass

    def _ssh(self, instance_id: str, remote: tuple[str, ...], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        endpoint = self._endpoints.get(_non_protected_id(instance_id))
        if endpoint is None:
            endpoint = self._vast.endpoint(instance_id, timeout_seconds=min(timeout_seconds, _MAX_TOOL_TIMEOUT))
            self._endpoints[instance_id] = endpoint
        command = (
            "ssh", "-F", "/dev/null", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={self._known_hosts}", "-o", "GlobalKnownHostsFile=/dev/null", "-o", f"ConnectTimeout={min(timeout_seconds, 20)}",
            "-i", str(self._identity), "-p", str(endpoint.port), f"{endpoint.username}@{endpoint.host}", "--", shlex.join(remote),
        )
        completed = self._run(command, timeout=timeout_seconds)
        if completed.returncode != 0:
            category = _remote_failure_category(completed.stdout, completed.stderr)
            raise ProductionSmokeError(f"SSH command failed: {category}")
        return completed

    def _runtime_json(self, instance_id: str, remote: tuple[str, ...], timeout_seconds: int, image: str) -> Mapping[str, Any]:
        # This is the one image-local execution deadline.  Readiness, provider,
        # and cleanup probes remain individually bounded below one minute.
        completed = self._ssh(instance_id, self._as_runtime_user(remote, image), timeout_seconds)
        try:
            payload = json.loads(completed.stdout.splitlines()[-1])
        except (IndexError, ValueError):
            raise ProductionSmokeError("image-local runtime smoke returned invalid evidence") from None
        if not isinstance(payload, Mapping):
            raise ProductionSmokeError("image-local runtime smoke returned invalid evidence")
        if payload.get("container_digest") != image.rsplit("@", 1)[1]:
            raise ProductionSmokeError("image-local runtime did not attest the expected container digest")
        return dict(payload)

    def _as_runtime_user(self, command: tuple[str, ...], image: str) -> tuple[str, ...]:
        simulator_environment = (
            "OMNI_KIT_ACCEPT_EULA=YES",
            "OMNIGIBSON_DATA_PATH=/workspace/omnigibson-data",
        ) if image == self._rollout_image else ()
        return (
            "setpriv", "--reuid=10001", "--regid=10001", "--init-groups", "env", "-i",
            "PATH=/opt/runtime/bin:/opt/conda/envs/behavior/bin:/usr/bin:/bin",
            "HOME=/workspace", "B1K_HF_TOKEN_FILE=/workspace/.cache/huggingface/token",
            "HF_HOME=/workspace/.cache/huggingface", "HF_HUB_CACHE=/workspace/.cache/huggingface/hub",
            *simulator_environment, f"CONTAINER_DIGEST={image.rsplit('@', 1)[1]}", *command,
        )

    def _remote_probe(self, role: str, classification: str, run_id: str, payload: Mapping[str, Any]) -> HubProbeReceipt:
        run_hex = _run_hex(run_id)
        repository = self._remote_probe_repository(role)
        prefix = f"b1k-bootstrap-{run_hex}-{classification}"
        upload_commit = payload.get("remote_probe_upload_commit")
        if not isinstance(upload_commit, str):
            raise ProductionSmokeError("remote image did not return an immutable probe upload commit")
        return self._hub.verify_remote_probe(role, repository, prefix=prefix, upload_commit=upload_commit)

    @staticmethod
    def _remote_probe_repository(role: str) -> HubRepository:
        if role not in {"model", "dataset"}:
            raise ProductionSmokeError("remote probe role is invalid")
        return HubRepository(
            "ryanjin333/behavior1k-groot-n17-models" if role == "model" else "ryanjin333/behavior1k-groot-n17-rollouts",
            "model" if role == "model" else "dataset",
        )

    def _reconcile_lost_runtime_probes(self, role: str, run_id: str, classifications: tuple[str, ...]) -> None:
        repository = self._remote_probe_repository(role)
        run_hex = _run_hex(run_id)
        cleanup_failure: Exception | None = None
        for classification in classifications:
            try:
                self._hub.reconcile_remote_probe(role, repository, prefix=f"b1k-bootstrap-{run_hex}-{classification}")
            except Exception as error:
                cleanup_failure = cleanup_failure or error
        if cleanup_failure is not None:
            raise cleanup_failure

    def _raise_after_runtime_reconciliation(
        self, role: str, run_id: str, classifications: tuple[str, ...], primary: Exception
    ) -> None:
        try:
            self._reconcile_lost_runtime_probes(role, run_id, classifications)
        except Exception as cleanup_failure:
            raise primary from cleanup_failure
        raise primary

    def _run(self, command: tuple[str, ...], *, timeout: int) -> subprocess.CompletedProcess[str]:
        try:
            return self._runner(command, text=True, capture_output=True, check=False, timeout=timeout, env={"PATH": os.environ.get("PATH", os.defpath)})
        except Exception:
            raise ProductionSmokeError("bounded SSH subprocess failed") from None

    def _deadline(self, timeout_seconds: float) -> float:
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise ProductionSmokeError("SSH deadline is invalid")
        return self._clock() + float(timeout_seconds)

    def _remaining(self, deadline: float) -> float:
        return max(0.0, deadline - self._clock())


def _training_command(image: str, prefix: str) -> tuple[str, ...]:
    del image
    return ("/opt/runtime/bin/python", "/opt/b1k-launchkit/training_smoke.py", "--prefix", prefix)


def _rollout_command(image: str, success_prefix: str, failure_prefix: str) -> tuple[str, ...]:
    del image
    return ("/opt/conda/envs/behavior/bin/python", "-m", "b1k_rollout.cli", "smoke-runtime", "--success-prefix", success_prefix, "--failure-prefix", failure_prefix)


def _exact_id(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool): value = str(value)
    if not isinstance(value, str) or not _ID_RE.fullmatch(value): raise ProductionSmokeError("provider did not return one exact numeric ID")
    return value


def _non_protected_id(value: object) -> str:
    result = _exact_id(value)
    if result in PROTECTED_INSTANCE_IDS: raise ProductionSmokeError("protected LeHome instance ID is forbidden")
    return result


def _label(value: object) -> str:
    if not isinstance(value, str) or not _LABEL_RE.fullmatch(value): raise ProductionSmokeError("smoke label is invalid")
    return value


def _rate(value: object) -> Decimal:
    try: result = Decimal(str(value))
    except (InvalidOperation, ValueError): raise ProductionSmokeError("Vast offer rate is invalid") from None
    if not result.is_finite() or result <= 0: raise ProductionSmokeError("Vast offer rate is invalid")
    if result.as_tuple().exponent < -6:
        try: result = result.quantize(Decimal("0.000001"), rounding=ROUND_CEILING)
        except InvalidOperation: raise ProductionSmokeError("Vast offer rate is invalid") from None
    return result


def _floor_units(row: Mapping[str, Any], key: str, divisor: int) -> int:
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not value > 0:
        raise ProductionSmokeError(f"{key} is invalid")
    result = int(float(value) // divisor)
    if result <= 0:
        raise ProductionSmokeError(f"{key} is insufficient")
    return result
def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0: raise ProductionSmokeError(f"{label} is invalid")
    return value
def _string(row: Mapping[str, Any], key: str, *, fallback: str) -> str:
    value = row.get(key, fallback)
    return value if isinstance(value, str) and value else fallback
def _mode(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else (_ for _ in ()).throw(ProductionSmokeError("token mode is invalid"))
def _exact_string(value: object, expected: str) -> str:
    if value != expected: raise ProductionSmokeError("runtime evidence is invalid")
    return expected
def _exact_one_of(value: object, allowed: set[str]) -> str:
    if not isinstance(value, str) or value not in allowed: raise ProductionSmokeError("runtime evidence is invalid")
    return value
def _require_fields(payload: Mapping[str, Any], fields: set[str]) -> None:
    if set(payload) - fields - {"infrastructure_smoke", "remote_probe_upload_commit", "remote_probe_upload_commits"} or not fields <= set(payload): raise ProductionSmokeError("runtime evidence schema is invalid")
def _fixture_payload(payload: Mapping[str, Any], classification: str) -> Mapping[str, Any]:
    commits = payload.get("remote_probe_upload_commits")
    if not isinstance(commits, Mapping) or not isinstance(commits.get(classification), str):
        raise ProductionSmokeError("remote image did not return both immutable fixture upload commits")
    return {"remote_probe_upload_commit": commits[classification]}
def _verified_datacenter(row: Mapping[str, Any]) -> bool:
    return row.get("verification") == "verified" and row.get("vericode") == 1
def _gpu_compatible(row: Mapping[str, Any], minimum_vram_mib: int) -> bool:
    cuda = row.get("cuda_max_good")
    capability = row.get("compute_cap")
    vram = row.get("gpu_ram")
    driver = row.get("driver_version")
    return (
        row.get("cpu_arch") in {"amd64", "x86_64"}
        and row.get("num_gpus") == 1
        and isinstance(cuda, (int, float)) and not isinstance(cuda, bool) and cuda >= 12
        and isinstance(capability, (int, float)) and not isinstance(capability, bool) and capability >= 800
        and isinstance(vram, (int, float)) and not isinstance(vram, bool) and vram >= minimum_vram_mib
        and isinstance(driver, str) and re.fullmatch(r"[0-9]{3,4}\.[0-9]{1,3}\.[0-9]{1,3}", driver) is not None
    )
def _tool_timeout_float(value: float) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value <= _MAX_TOOL_TIMEOUT: raise ProductionSmokeError("Vast tool timeout is invalid")
    return max(1, int(value))
def _run_hex(run_id: str) -> str: return _label(run_id).removeprefix("b1k-smoke-")
def _private_identity(path: Path) -> Path:
    try:
        metadata = path.lstat()
    except OSError: raise ProductionSmokeError("SSH identity file is unavailable") from None
    if not path.is_absolute() or path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077: raise ProductionSmokeError("SSH identity file must be a current-user private regular file")
    return path
def _campaign_known_hosts(path: Path) -> Path:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]): raise ProductionSmokeError("campaign known_hosts path is invalid")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for parent in (path.parent, *path.parent.parents):
        if parent == parent.parent:
            break
        try:
            metadata = parent.lstat()
        except OSError:
            raise ProductionSmokeError("campaign known_hosts parent is unavailable") from None
        # System-owned sticky ancestors (for example macOS's temporary
        # hierarchy) are safe traversal parents; the campaign directory and
        # final file themselves remain current-user private and no-follow.
        writable_by_others = bool(metadata.st_mode & 0o022) and not bool(metadata.st_mode & stat.S_ISVTX)
        if parent.is_symlink() or not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid not in {os.getuid(), 0} or writable_by_others:
            raise ProductionSmokeError("campaign known_hosts parent is unsafe")
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise ProductionSmokeError("campaign known_hosts path is invalid")
        os.fchmod(descriptor, 0o600)
    except OSError:
        raise ProductionSmokeError("campaign known_hosts path is invalid") from None
    finally:
        try:
            os.close(descriptor)
        except UnboundLocalError:
            pass
    return path
