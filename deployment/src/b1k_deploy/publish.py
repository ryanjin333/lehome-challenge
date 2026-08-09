"""Dry-run-first, secret-free publication of the two immutable B1K templates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shlex
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import fcntl

from .dockerhub import DockerHubReleaseVerifier, DockerImageRelease


_SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SECRET_KEY_RE = re.compile(r"(?:token|password|secret|credential|api[_-]?key|authorization)", re.IGNORECASE)
_CREDENTIAL_VALUE_RE = re.compile(r"(?:\bhf_[A-Za-z0-9_-]{20,}\b|\b(?:dckr_pat|docker_pat)_[A-Za-z0-9_-]{16,}\b|\bBearer\s+\S+)", re.IGNORECASE)
_REPOSITORIES = {
    "training": "docker.io/ryanjin333/behavior1k-groot-n17-trainer",
    "rollout": "docker.io/ryanjin333/behavior1k-groot-n17-rollout",
}
_ZERO_DIGEST = "sha256:" + "0" * 64
_NOFOLLOW = os.O_NOFOLLOW


class PublicationError(ValueError):
    """A release plan or returned provider readback is unsafe or incomplete."""


class PublicationPartialError(PublicationError):
    def __init__(self, images: tuple[DockerImageRelease, ...]):
        self.images = images
        super().__init__("publication reached an ambiguous partial state")


class ImageBuilder(Protocol):
    def build_and_push(self, repository: str, tag: str, source_commit: str) -> None: ...


class TemplateClient(Protocol):
    def find_private_template(self, name: str, image_reference: str) -> str | None: ...
    def create_private_template(self, template: Mapping[str, Any]) -> str: ...
    def get_template(self, template_id: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PublicationAdapters:
    """Explicitly injected authenticated transports; never constructed by the CLI."""

    builder: ImageBuilder
    verifier: DockerHubReleaseVerifier
    templates: TemplateClient


def _configured_publication_adapters(workspace: Path) -> PublicationAdapters:
    """Load only the fixed local production boundary; never import a user target."""
    from .production import configured_publication_adapters

    return configured_publication_adapters(workspace=workspace)


TRUSTED_ADAPTER_FACTORIES: Mapping[str, Callable[[Path], PublicationAdapters]] = {
    "configured": _configured_publication_adapters,
}


@dataclass(frozen=True)
class ImagePublicationPlan:
    source_commit: str
    training_tag: str
    rollout_tag: str


@dataclass(frozen=True)
class PlannedImageRelease:
    repository: str
    tag: str


@dataclass(frozen=True)
class ImagePublicationReceipt:
    source_commit: str
    dry_run: bool
    images: tuple[PlannedImageRelease | DockerImageRelease, ...]


@dataclass(frozen=True)
class TemplatePublicationPlan:
    source_commit: str
    purpose: Literal["training", "rollout"]
    image: DockerImageRelease
    template: Mapping[str, Any]
    source_root: Path | None = None


@dataclass(frozen=True)
class TemplateSchemaPlan:
    """Pre-build canonical template contract; image digests are intentionally absent."""

    source_commit: str
    purpose: Literal["training", "rollout"]
    template: Mapping[str, Any]
    source_root: Path | None = None


@dataclass(frozen=True)
class TemplatePublicationReceipt:
    source_commit: str
    purpose: Literal["training", "rollout"]
    dry_run: bool
    name: str
    template_id: str | None
    image_reference: str
    payload_hash: str

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "source_commit": self.source_commit,
            "purpose": self.purpose,
            "dry_run": self.dry_run,
            "name": self.name,
            "template_id": self.template_id,
            "image_reference": self.image_reference,
            "payload_hash": self.payload_hash,
        }


@dataclass(frozen=True)
class CampaignPublicationReceipt:
    source_commit: str
    status: Literal["complete", "ambiguous"]
    phase: Literal["preflight", "images", "templates", "complete"]
    preplan_hash: str
    final_plan_hash: str | None
    images: tuple[DockerImageRelease, ...]
    templates: tuple[TemplatePublicationReceipt, ...]

    def to_dict(self) -> dict[str, object]:
        return {"source_commit": self.source_commit, "status": self.status, "phase": self.phase, "preplan_hash": self.preplan_hash, "final_plan_hash": self.final_plan_hash, "images": [item.__dict__ for item in self.images], "templates": [item.to_dict() for item in self.templates]}


class AtomicCampaignReceiptStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock_path = path.with_name(f".{path.name}.lock")

    def write(self, receipt: CampaignPublicationReceipt) -> None:
        with self.locked():
            self.write_locked(receipt)

    def write_locked(self, receipt: CampaignPublicationReceipt) -> None:
        _validate_campaign_receipt(receipt)
        encoded = (json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        temporary: Path | None = None
        try:
            temporary, descriptor = self._open_unique_temporary()
            try:
                written = 0
                while written < len(encoded):
                    count = os.write(descriptor, encoded[written:])
                    if count <= 0:
                        raise OSError("receipt write made no progress")
                    written += count
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, self.path)
            self._fsync_directory()
        except Exception:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise

    def read(self) -> CampaignPublicationReceipt | None:
        with self.locked():
            return self.read_locked()

    def read_locked(self) -> CampaignPublicationReceipt | None:
        try:
            descriptor = self._open_existing_private_file(self.path)
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            images = tuple(DockerImageRelease(**item) for item in data["images"])
            templates = tuple(TemplatePublicationReceipt(**item) for item in data["templates"])
            receipt = CampaignPublicationReceipt(data["source_commit"], data["status"], data["phase"], data["preplan_hash"], data["final_plan_hash"], images, templates)
            _validate_campaign_receipt(receipt)
            return receipt
        except FileNotFoundError:
            return None
        except (KeyError, TypeError, ValueError):
            raise PublicationError("campaign receipt is invalid") from None

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = self._open_lock_file()
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _open_lock_file(self) -> int:
        flags = os.O_RDWR | os.O_CREAT | _NOFOLLOW
        try:
            descriptor = os.open(self._lock_path, flags, 0o600)
        except OSError:
            raise PublicationError("campaign receipt lock is unsafe") from None
        try:
            os.fchmod(descriptor, 0o600)
            _require_private_regular_file(descriptor)
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    def _open_unique_temporary(self) -> tuple[Path, int]:
        for _ in range(32):
            temporary = self.path.with_name(f".{self.path.name}.{secrets.token_hex(16)}.tmp")
            try:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o600)
            except FileExistsError:
                continue
            except OSError:
                raise PublicationError("campaign receipt temporary file is unsafe") from None
            try:
                os.fchmod(descriptor, 0o600)
                return temporary, descriptor
            except Exception:
                os.close(descriptor)
                temporary.unlink(missing_ok=True)
                raise
        raise PublicationError("campaign receipt temporary file could not be allocated safely")

    def _open_existing_private_file(self, path: Path) -> int:
        try:
            descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW)
        except FileNotFoundError:
            raise
        except OSError:
            raise PublicationError("campaign receipt file is unsafe") from None
        try:
            _require_private_regular_file(descriptor)
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    def _fsync_directory(self) -> None:
        descriptor = os.open(self.path.parent, os.O_RDONLY | _NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class CampaignPublisher:
    def __init__(self, adapters: PublicationAdapters, store: AtomicCampaignReceiptStore):
        self._adapters, self._store = adapters, store

    def publish(self, images: ImagePublicationPlan, templates: tuple[TemplateSchemaPlan, TemplateSchemaPlan]) -> CampaignPublicationReceipt:
        with self._store.locked():
            return self._publish_locked(images, templates)

    def _publish_locked(self, images: ImagePublicationPlan, templates: tuple[TemplateSchemaPlan, TemplateSchemaPlan]) -> CampaignPublicationReceipt:
        preplan_hash = campaign_preplan_hash(images, templates)
        prior = self._store.read_locked()
        reuse = prior is not None and prior.source_commit == images.source_commit and prior.preplan_hash == preplan_hash
        if reuse:
            _validate_campaign_receipt_matches_preplan(prior, images, templates)
        published_images: tuple[DockerImageRelease, ...] = prior.images if reuse else ()
        published_templates: list[TemplatePublicationReceipt] = list(prior.templates) if reuse else []

        if not reuse:
            self._store.write_locked(CampaignPublicationReceipt(images.source_commit, "ambiguous", "preflight", preplan_hash, None, (), ()))

        def persist(phase: Literal["images", "templates"], releases: tuple[DockerImageRelease, ...] | None = None, final_plan_hash: str | None = None) -> None:
            self._store.write_locked(CampaignPublicationReceipt(images.source_commit, "ambiguous", phase, preplan_hash, final_plan_hash, releases if releases is not None else published_images, tuple(published_templates)))

        # Readback failure is not a new publication phase.  Keep the durable
        # receipt intact so the next invocation must fresh-check it again.
        if reuse and prior is not None:
            self._verify_reusable_images(prior.images, images)
            if prior.templates:
                self._verify_persisted_template_state(prior, images, templates)

        try:
            if len(published_images) != 2:
                def on_verified(releases: tuple[DockerImageRelease, ...]) -> None:
                    frozen = _freeze_template_plans(images, templates, releases) if len(releases) == 2 else None
                    persist("images", releases, campaign_final_plan_hash(images, preplan_hash, frozen) if frozen is not None else None)

                result = publish_images(images, self._adapters.builder, self._adapters.verifier, execute=True, existing=published_images, on_verified=on_verified)
                published_images = tuple(result.images)
            frozen_templates = _freeze_template_plans(images, templates, published_images)
            final_plan_hash = campaign_final_plan_hash(images, preplan_hash, frozen_templates)
            if reuse and prior is not None and prior.final_plan_hash not in {None, final_plan_hash}:
                raise PublicationError("campaign receipt final identity does not match verified images")
            existing_templates = {item.purpose: item for item in published_templates}
            for template in frozen_templates:
                if template.purpose in existing_templates:
                    continue
                published_templates.append(TemplatePublisher(self._adapters.templates).publish(template, execute=True))
                persist("templates", final_plan_hash=final_plan_hash)
        except PublicationPartialError as error:
            published_images = error.images
            receipt = CampaignPublicationReceipt(images.source_commit, "ambiguous", "images", preplan_hash, None, published_images, tuple(published_templates))
            self._store.write_locked(receipt)
            raise
        except Exception:
            phase: Literal["images", "templates"] = "templates" if published_templates else "images"
            final = campaign_final_plan_hash(images, preplan_hash, _freeze_template_plans(images, templates, published_images)) if len(published_images) == 2 else None
            receipt = CampaignPublicationReceipt(images.source_commit, "ambiguous", phase, preplan_hash, final, published_images, tuple(published_templates))
            self._store.write_locked(receipt)
            raise
        receipt = CampaignPublicationReceipt(images.source_commit, "complete", "complete", preplan_hash, final_plan_hash, published_images, tuple(published_templates))
        self._store.write_locked(receipt)
        return receipt

    def _verify_reusable_images(
        self,
        releases: tuple[DockerImageRelease, ...],
        images: ImagePublicationPlan,
    ) -> None:
        for purpose, tag in (("training", images.training_tag), ("rollout", images.rollout_tag)):
            expected = next((item for item in releases if item.repository == _REPOSITORIES[purpose]), None)
            if expected is None:
                continue
            actual = _call("private registry digest readback", self._adapters.verifier.verify_private_image, expected.repository, tag)
            if actual != expected or not _valid_image_release(actual, expected.repository):
                raise PublicationError("campaign receipt registry readback drifted")

    def _verify_persisted_template_state(
        self,
        receipt: CampaignPublicationReceipt,
        images: ImagePublicationPlan,
        schemas: tuple[TemplateSchemaPlan, TemplateSchemaPlan],
    ) -> None:
        """A receipt authorizes idempotence, never stale remote state."""
        frozen = _freeze_template_plans(images, schemas, receipt.images)
        expected_templates = {plan.purpose: plan for plan in frozen}
        for template in receipt.templates:
            expected = expected_templates[template.purpose]
            rendered = _render_template(expected)
            readback = _call("template readback", self._adapters.templates.get_template, template.template_id)
            if (
                not isinstance(readback, Mapping)
                or canonical_payload_hash(readback) != template.payload_hash
                or template.payload_hash != canonical_payload_hash(rendered)
            ):
                raise PublicationError("persisted campaign receipt template readback drifted")
            _validate_rendered_template(readback, template.name, template.image_reference)


def publish_images(plan: ImagePublicationPlan, builder: ImageBuilder, verifier: DockerHubReleaseVerifier, *, execute: bool, existing: tuple[DockerImageRelease, ...] = (), on_verified: Callable[[tuple[DockerImageRelease, ...]], None] | None = None) -> ImagePublicationReceipt:
    validate_image_plan(plan)
    candidates = tuple(PlannedImageRelease(repository, tag) for repository, tag in ((_REPOSITORIES["training"], plan.training_tag), (_REPOSITORIES["rollout"], plan.rollout_tag)))
    if not execute:
        return ImagePublicationReceipt(plan.source_commit, True, candidates)
    releases: list[DockerImageRelease] = list(existing)
    for candidate in candidates:
        if any(item.repository == candidate.repository for item in releases):
            continue
        try:
            _call("image build and push", builder.build_and_push, candidate.repository, candidate.tag, plan.source_commit)
            release = _call("private registry digest readback", verifier.verify_private_image, candidate.repository, candidate.tag)
            if not _valid_image_release(release, candidate.repository):
                raise PublicationError("private registry digest readback is invalid")
        except PublicationError:
            raise PublicationPartialError(tuple(releases)) from None
        releases.append(release)
        if on_verified is not None:
            on_verified(tuple(releases))
    return ImagePublicationReceipt(plan.source_commit, False, tuple(releases))


class TemplatePublisher:
    def __init__(self, client: TemplateClient):
        self._client = client

    def publish(self, plan: TemplatePublicationPlan, *, execute: bool) -> TemplatePublicationReceipt:
        _validate_template_plan(plan)
        rendered = _render_template(plan)
        name = rendered["name"]
        assert isinstance(name, str)
        digest_reference = plan.image.reference
        payload_hash = canonical_payload_hash(rendered)
        if not execute:
            return TemplatePublicationReceipt(plan.source_commit, plan.purpose, True, name, None, digest_reference, payload_hash)
        existing = _call("template lookup", self._client.find_private_template, name, digest_reference)
        if existing is None:
            template_id = _call("template publication", self._client.create_private_template, rendered)
        else:
            template_id = existing
        if not isinstance(template_id, str) or not re.fullmatch(r"[1-9][0-9]*", template_id):
            raise PublicationError("template publication did not return one exact template ID")
        readback = _call("template readback", self._client.get_template, template_id)
        if not isinstance(readback, Mapping) or canonical_payload_hash(readback) != payload_hash:
            raise PublicationError("template readback payload hash does not match the published payload")
        _validate_rendered_template(readback, name, digest_reference)
        return TemplatePublicationReceipt(plan.source_commit, plan.purpose, False, name, template_id, digest_reference, payload_hash)


def canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    canonical = _canonical_payload(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def campaign_preplan_hash(images: ImagePublicationPlan, templates: tuple[TemplateSchemaPlan, TemplateSchemaPlan]) -> str:
    """Hash source, tags, and immutable schemas before any remote operation."""
    validate_image_plan(images)
    if len(templates) != 2 or {template.purpose for template in templates} != set(_REPOSITORIES):
        raise PublicationError("campaign requires exactly one training and one rollout template plan")
    schemas: list[dict[str, str]] = []
    for template in sorted(templates, key=lambda item: item.purpose):
        validate_template_schema(template)
        if template.source_commit != images.source_commit:
            raise PublicationError("campaign template source commit must match the image plan")
        schemas.append(
            {
                "purpose": template.purpose,
                "schema_hash": canonical_payload_hash(template.template),
            }
        )
    return canonical_payload_hash(
        {
            "source_commit": images.source_commit,
            "training_tag": images.training_tag,
            "rollout_tag": images.rollout_tag,
            "templates": schemas,
        }
    )


def campaign_final_plan_hash(
    images: ImagePublicationPlan,
    preplan_hash: str,
    templates: tuple[TemplatePublicationPlan, TemplatePublicationPlan],
) -> str:
    return canonical_payload_hash(
        {
            "preplan_hash": preplan_hash,
            "source_commit": images.source_commit,
            "images": [
                {"repository": template.image.repository, "tag": images.training_tag if template.purpose == "training" else images.rollout_tag, "reference": template.image.reference}
                for template in sorted(templates, key=lambda item: item.purpose)
            ],
            "rendered_templates": [
                {"purpose": template.purpose, "payload_hash": canonical_payload_hash(_render_template(template))}
                for template in sorted(templates, key=lambda item: item.purpose)
            ],
        }
    )


# Compatibility name for callers that need the preflight (digest-free) identity.
campaign_plan_hash = campaign_preplan_hash


def load_canonical_template(purpose: Literal["training", "rollout"], *, source_root: Path) -> dict[str, Any]:
    if purpose not in _REPOSITORIES:
        raise PublicationError("template purpose is invalid")
    workspace = _validate_template_source_root(source_root)
    path = workspace / ("trainer" if purpose == "training" else "rollout") / "vast-template.example.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise PublicationError("canonical template contract could not be read") from None
    if not isinstance(payload, dict):
        raise PublicationError("canonical template contract must be an object")
    return payload


def _validate_template_source_root(source_root: Path) -> Path:
    if not isinstance(source_root, Path):
        raise PublicationError("canonical template source root must be an explicit path")
    try:
        workspace = source_root.resolve(strict=True)
    except OSError:
        raise PublicationError("canonical template source root is unavailable") from None
    required = (
        workspace / "trainer" / "Dockerfile",
        workspace / "trainer" / "vast-template.example.json",
        workspace / "rollout" / "Dockerfile",
        workspace / "rollout" / "vast-template.example.json",
    )
    if not all(path.is_file() for path in required):
        raise PublicationError("canonical template source root is incomplete")
    return workspace


def load_publication_adapters(
    name: str,
    *,
    workspace: Path,
    factories: Mapping[str, Callable[[], PublicationAdapters]] | None = None,
) -> PublicationAdapters:
    factory_map = TRUSTED_ADAPTER_FACTORIES if factories is None else factories
    if not isinstance(name, str) or name not in factory_map:
        raise PublicationError("adapter name is not in the trusted publication adapter allowlist")
    try:
        adapters = factory_map[name](workspace) if factories is None else factory_map[name]()
    except Exception:
        raise PublicationError("adapter loading failed") from None
    if not isinstance(adapters, PublicationAdapters):
        raise PublicationError("adapter factory must return PublicationAdapters")
    return adapters


def _validate_source_and_tags(plan: ImagePublicationPlan) -> None:
    if not isinstance(plan, ImagePublicationPlan) or not _SOURCE_COMMIT_RE.fullmatch(plan.source_commit):
        raise PublicationError("source commit must be one exact 40-character lowercase revision")
    for tag in (plan.training_tag, plan.rollout_tag):
        if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", tag):
            raise PublicationError("image release tags are invalid")


def validate_image_plan(plan: ImagePublicationPlan) -> None:
    _validate_source_and_tags(plan)


def validate_template_plan(plan: TemplatePublicationPlan) -> None:
    _validate_template_plan(plan)


def validate_template_schema(plan: TemplateSchemaPlan) -> None:
    _validate_template_schema(plan)


def _validate_template_plan(plan: TemplatePublicationPlan) -> None:
    if not isinstance(plan, TemplatePublicationPlan):
        raise PublicationError("template publication plan is invalid")
    _validate_template_schema(TemplateSchemaPlan(plan.source_commit, plan.purpose, plan.template, plan.source_root))
    if not _valid_image_release(plan.image, _REPOSITORIES[plan.purpose]):
        raise PublicationError("template requires the purpose-specific private digest-qualified image")


def _validate_template_schema(plan: TemplateSchemaPlan) -> None:
    if not isinstance(plan, TemplateSchemaPlan) or plan.purpose not in _REPOSITORIES or not _SOURCE_COMMIT_RE.fullmatch(plan.source_commit):
        raise PublicationError("template publication plan is invalid")
    if not isinstance(plan.template, Mapping):
        raise PublicationError("template payload must be an object")
    _reject_secrets(plan.template)
    template_image = plan.template.get("image")
    if template_image not in {"${IMAGE_REFERENCE}", f"{_REPOSITORIES[plan.purpose]}@{_ZERO_DIGEST}"}:
        raise PublicationError("template image must be the exact IMAGE_REFERENCE or canonical zero-digest placeholder")
    if plan.template.get("private") is not True:
        raise PublicationError("template payload must explicitly request a private template")
    environment = plan.template.get("env", "")
    if plan.purpose == "training":
        if plan.source_root is not None:
            canonical = load_canonical_template("training", source_root=plan.source_root)
            if set(plan.template) != set(canonical) or any(
                plan.template[key] != canonical[key] for key in canonical if key not in {"name", "image", "env"}
            ):
                raise PublicationError("template payload must be the canonical purpose-specific contract")
            if environment != canonical.get("env"):
                raise PublicationError("template environment must be the canonical controlled-substitution contract")
    else:
        _validate_production_rollout_schema(plan.template)
    if "AUTO_DESTROY=0" not in str(environment) or f"CONTAINER_DIGEST={_ZERO_DIGEST}" not in str(environment):
        raise PublicationError("template payload must preserve production AUTO_DESTROY=0")


def _validate_production_rollout_schema(template: Mapping[str, Any]) -> None:
    """Accept only the rollout renderer's non-fixture production schema."""
    try:
        from b1k_rollout.template import render_vast_template

        environment = template["env"]
        if not isinstance(environment, str):
            raise ValueError
        assignments = _environment_assignments(environment)
        model_commit = assignments["MODEL_COMMIT"]
        checkpoint = assignments["CHECKPOINT_ARTIFACT_SHA256"]
        gpu_ids = tuple(int(value) for value in assignments["GPU_IDS"].split(","))
        expected = json.loads(
            render_vast_template(
                image_digest=_ZERO_DIGEST,
                model_commit=model_commit,
                checkpoint_artifact_sha256=checkpoint,
                gpu_ids=gpu_ids,
            )
        )
    except (ImportError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise PublicationError("rollout template must be rendered from nonzero immutable rollout inputs") from None
    if template != expected:
        raise PublicationError("rollout template must match the canonical production renderer")


def _environment_assignments(environment: str) -> dict[str, str]:
    words = shlex.split(environment)
    assignments: dict[str, str] = {}
    for index, word in enumerate(words):
        candidate = words[index + 1] if word in {"-e", "--env"} and index + 1 < len(words) else word.removeprefix("-e=").removeprefix("--env=")
        if "=" in candidate:
            key, value = candidate.split("=", 1)
            assignments[key] = value
    return assignments


def _freeze_template_plans(
    images: ImagePublicationPlan,
    schemas: tuple[TemplateSchemaPlan, TemplateSchemaPlan],
    releases: tuple[DockerImageRelease, ...],
) -> tuple[TemplatePublicationPlan, TemplatePublicationPlan]:
    if len(releases) != 2:
        raise PublicationError("campaign requires both verified image readbacks before templates can be frozen")
    by_repository = {release.repository: release for release in releases}
    frozen: list[TemplatePublicationPlan] = []
    schemas_by_purpose = {schema.purpose: schema for schema in schemas}
    for purpose in ("training", "rollout"):
        schema = schemas_by_purpose[purpose]
        release = by_repository.get(_REPOSITORIES[schema.purpose])
        if release is None or not _valid_image_release(release, _REPOSITORIES[schema.purpose]):
            raise PublicationError("campaign image receipt does not match template purpose")
        frozen.append(TemplatePublicationPlan(images.source_commit, schema.purpose, release, schema.template, schema.source_root))
    return frozen[0], frozen[1]


def _render_template(plan: TemplatePublicationPlan) -> dict[str, Any]:
    rendered = json.loads(_canonical_payload(plan.template))
    assert isinstance(rendered, dict)
    rendered["image"] = plan.image.reference
    rendered["name"] = f"b1k-{plan.purpose}-{plan.image.digest.removeprefix('sha256:')[:16]}"
    rendered["env"] = str(rendered["env"]).replace(f"CONTAINER_DIGEST={_ZERO_DIGEST}", f"CONTAINER_DIGEST={plan.image.digest}")
    _validate_rendered_template(rendered, rendered["name"], plan.image.reference)
    return rendered


def _validate_rendered_template(payload: Mapping[str, Any], name: str, image_reference: str) -> None:
    if payload.get("name") != name or payload.get("image") != image_reference or payload.get("private") is not True:
        raise PublicationError("template readback is not the exact private name and digest payload")
    environment = str(payload.get("env", ""))
    if "AUTO_DESTROY=0" not in environment or f"CONTAINER_DIGEST={image_reference.rsplit('@', 1)[1]}" not in environment:
        raise PublicationError("template readback does not preserve AUTO_DESTROY=0")
    _reject_secrets(payload)


def _canonical_payload(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError):
        raise PublicationError("template payload must be canonical JSON") from None


def _reject_secrets(value: object, path: str = "template", key: str | None = None) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PublicationError("template payload keys must be strings")
            if _SECRET_KEY_RE.search(key) and not (_is_token_file_key(key) and isinstance(item, str) and item.startswith("/")):
                raise PublicationError("template payload contains a secret-shaped key")
            _reject_secrets(item, f"{path}.{key}", key)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_secrets(item, path, key)
    elif isinstance(value, str):
        if _CREDENTIAL_VALUE_RE.search(value):
            raise PublicationError("template payload contains an actual credential value")
        if key == "env":
            _validate_environment(value)
        elif key is not None and _SECRET_KEY_RE.search(key) and not (_is_token_file_key(key) and value.startswith("/")):
            raise PublicationError("template payload contains a secret-shaped key")


def _validate_environment(environment: str) -> None:
    try:
        words = shlex.split(environment)
    except ValueError:
        raise PublicationError("template environment is invalid") from None
    for index, word in enumerate(words):
        if word in {"-e", "--env"}:
            if index + 1 >= len(words):
                raise PublicationError("template environment assignment is invalid")
            assignment = words[index + 1]
        elif word.startswith("-e="):
            assignment = word.removeprefix("-e=")
        elif word.startswith("--env="):
            assignment = word.removeprefix("--env=")
        else:
            continue
        if "=" not in assignment:
            raise PublicationError("template environment assignment is invalid")
        name, value = assignment.split("=", 1)
        if _CREDENTIAL_VALUE_RE.search(value):
            raise PublicationError("template payload contains an actual credential value")
        if _SECRET_KEY_RE.search(name) and not (_is_token_file_key(name) and value.startswith("/")):
            raise PublicationError("template payload contains a secret-shaped key")


def _is_token_file_key(key: str) -> bool:
    return key.upper().endswith(("_TOKEN_FILE", "_TOKEN_PATH", "_CREDENTIAL_FILE"))


def _valid_image_release(release: object, repository: str) -> bool:
    if not isinstance(release, DockerImageRelease) or release.repository != repository:
        return False
    try:
        return DockerHubReleaseVerifier.require_digest_reference(release.reference) == release.reference and release.reference == f"{release.repository}@{release.digest}"
    except ValueError:
        return False


def _require_private_regular_file(descriptor: int) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise PublicationError("campaign receipt file must be a private regular file")


def _validate_campaign_receipt(receipt: CampaignPublicationReceipt) -> None:
    if not isinstance(receipt, CampaignPublicationReceipt) or not _SOURCE_COMMIT_RE.fullmatch(receipt.source_commit):
        raise PublicationError("campaign receipt source commit is invalid")
    if receipt.status not in {"complete", "ambiguous"} or receipt.phase not in {"preflight", "images", "templates", "complete"}:
        raise PublicationError("campaign receipt state is invalid")
    if not isinstance(receipt.preplan_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", receipt.preplan_hash):
        raise PublicationError("campaign receipt preflight identity is invalid")
    if receipt.final_plan_hash is not None and (not isinstance(receipt.final_plan_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", receipt.final_plan_hash)):
        raise PublicationError("campaign receipt final identity is invalid")
    expected_repositories = (_REPOSITORIES["training"], _REPOSITORIES["rollout"])
    if len(receipt.images) > 2 or tuple(item.repository for item in receipt.images) != expected_repositories[: len(receipt.images)]:
        raise PublicationError("campaign receipt images are not the canonical verified prefix")
    image_by_repository: dict[str, DockerImageRelease] = {}
    for image in receipt.images:
        if not _valid_image_release(image, image.repository):
            raise PublicationError("campaign receipt images are invalid")
        image_by_repository[image.repository] = image
    templates_by_purpose: dict[str, TemplatePublicationReceipt] = {}
    expected_purposes = ("training", "rollout")
    if len(receipt.templates) > 2 or tuple(item.purpose for item in receipt.templates) != expected_purposes[: len(receipt.templates)]:
        raise PublicationError("campaign receipt templates are not the canonical verified prefix")
    for template in receipt.templates:
        if (
            template.purpose in templates_by_purpose
            or template.purpose not in _REPOSITORIES
            or template.source_commit != receipt.source_commit
            or template.dry_run
            or not isinstance(template.name, str)
            or not isinstance(template.template_id, str)
            or not re.fullmatch(r"[1-9][0-9]*", template.template_id)
            or not isinstance(template.payload_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", template.payload_hash)
        ):
            raise PublicationError("campaign receipt templates are invalid")
        image = image_by_repository.get(_REPOSITORIES[template.purpose])
        if image is None or template.image_reference != image.reference:
            raise PublicationError("campaign receipt does not bind templates to verified images")
        templates_by_purpose[template.purpose] = template
    image_count, template_count = len(receipt.images), len(receipt.templates)
    if receipt.phase == "preflight" and (receipt.status != "ambiguous" or image_count or template_count or receipt.final_plan_hash is not None):
        raise PublicationError("preflight receipt must not contain remote artifacts")
    if receipt.phase == "images" and (receipt.status != "ambiguous" or template_count or image_count not in {1, 2} or (image_count == 1 and receipt.final_plan_hash is not None) or (image_count == 2 and receipt.final_plan_hash is None)):
        raise PublicationError("image receipt has an invalid phase boundary")
    if receipt.phase == "templates" and (receipt.status != "ambiguous" or image_count != 2 or template_count not in {1, 2} or receipt.final_plan_hash is None):
        raise PublicationError("template receipt has an invalid phase boundary")
    if receipt.phase == "complete" and (receipt.status != "complete" or image_count != 2 or template_count != 2 or receipt.final_plan_hash is None):
        raise PublicationError("complete campaign receipt requires both images, both templates, and final identity")


def _validate_campaign_receipt_matches_preplan(
    receipt: CampaignPublicationReceipt,
    images: ImagePublicationPlan,
    schemas: tuple[TemplateSchemaPlan, TemplateSchemaPlan],
) -> None:
    if receipt.source_commit != images.source_commit or receipt.preplan_hash != campaign_preplan_hash(images, schemas):
        raise PublicationError("campaign receipt does not match the preflight identity")
    if len(receipt.images) == 2:
        frozen = _freeze_template_plans(images, schemas, receipt.images)
        expected_final = campaign_final_plan_hash(images, receipt.preplan_hash, frozen)
        if receipt.final_plan_hash != expected_final:
            raise PublicationError("campaign receipt final identity does not match verified images")
        expected_hashes = {template.purpose: canonical_payload_hash(_render_template(template)) for template in frozen}
    else:
        if receipt.final_plan_hash is not None:
            raise PublicationError("partial campaign receipt cannot have a final identity")
        expected_hashes = {}
    for template in receipt.templates:
        image = next(item for item in receipt.images if item.repository == _REPOSITORIES[template.purpose])
        if template.image_reference != image.reference or template.payload_hash != expected_hashes[template.purpose]:
            raise PublicationError("campaign receipt template does not match the campaign identity")


def _call(operation: str, callback: Any, *args: object) -> Any:
    try:
        return callback(*args)
    except Exception:
        raise PublicationError(f"{operation} failed") from None
