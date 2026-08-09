"""Dry-run-first command line for one durable, identity-bound B1K publication campaign."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Sequence

from .publish import (
    AtomicCampaignReceiptStore,
    CampaignPublisher,
    ImagePublicationPlan,
    PublicationAdapters,
    PublicationError,
    TemplateSchemaPlan,
    campaign_preplan_hash,
    load_canonical_template,
    load_publication_adapters,
)
from b1k_rollout.template import render_vast_template


_ZERO_DIGEST = "sha256:" + "0" * 64


def main(
    argv: Sequence[str] | None = None,
    *,
    adapter_factories: Mapping[str, Callable[[], PublicationAdapters]] | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="b1k-deploy")
    commands = parser.add_subparsers(dest="command", required=True)
    campaign = commands.add_parser("publish-campaign")
    campaign.add_argument("--source-commit", required=True)
    campaign.add_argument("--training-tag", required=True)
    campaign.add_argument("--rollout-tag", required=True)
    campaign.add_argument("--model-commit", required=True)
    campaign.add_argument("--checkpoint-artifact-sha256", required=True)
    campaign.add_argument("--gpu-ids", required=True)
    campaign.add_argument("--source-root", type=Path, required=True)
    campaign.add_argument("--receipt", type=Path, required=True)
    campaign.add_argument("--execute", action="store_true")
    campaign.add_argument("--adapter")
    smoke = commands.add_parser("smoke-campaign")
    smoke.add_argument("--training-image", required=True)
    smoke.add_argument("--rollout-image", required=True)
    smoke.add_argument("--training-template-id", required=True)
    smoke.add_argument("--rollout-template-id", required=True)
    smoke.add_argument("--training-template-payload-sha256", required=True)
    smoke.add_argument("--rollout-template-payload-sha256", required=True)
    smoke.add_argument("--ledger", type=Path, required=True)
    smoke.add_argument("--receipt", type=Path, required=True)
    smoke.add_argument("--known-hosts", type=Path, required=True)
    smoke.add_argument("--ssh-identity", type=Path)
    smoke.add_argument("--publication-receipt", type=Path)
    smoke.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "smoke-campaign":
        try:
            return _smoke_campaign(args)
        except PublicationError as error:
            # A paid run always has a typed receipt written before this point.
            # Do not erase that evidence behind argparse's generic error path.
            print(json.dumps({"smoke_campaign_failed": True, "reason": str(error), "receipt": str(args.receipt)}, sort_keys=True, separators=(",", ":")))
            return 2
    try:
        images = ImagePublicationPlan(args.source_commit, args.training_tag, args.rollout_tag)
        if args.execute and args.adapter == "configured":
            from .production import GitReleaseContext

            GitReleaseContext().verify(args.source_root, args.source_commit)
        training_template = load_canonical_template("training", source_root=args.source_root)
        rollout_source_template = load_canonical_template("rollout", source_root=args.source_root)
        rollout_template = json.loads(
            render_vast_template(
                image_digest=_ZERO_DIGEST,
                model_commit=args.model_commit,
                checkpoint_artifact_sha256=args.checkpoint_artifact_sha256,
                gpu_ids=_gpu_ids(args.gpu_ids),
            )
        )
        if not isinstance(rollout_template, dict) or set(rollout_template) != set(rollout_source_template):
            raise PublicationError("rollout source template contract is incompatible with the production renderer")
        templates = (
            TemplateSchemaPlan(args.source_commit, "training", training_template, args.source_root),
            TemplateSchemaPlan(
                args.source_commit,
                "rollout",
                rollout_template,
                args.source_root,
            ),
        )
        preplan_hash = campaign_preplan_hash(images, templates)
        if not args.execute:
            print(
                json.dumps(
                    {
                        "source_commit": images.source_commit,
                        "preplan_hash": preplan_hash,
                        "final_plan_hash": None,
                        "dry_run": True,
                        "receipt": str(args.receipt),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if not args.adapter:
            raise PublicationError("--execute requires a trusted adapter name")
        adapters = load_publication_adapters(args.adapter, workspace=args.source_root, factories=adapter_factories)
        receipt = CampaignPublisher(adapters, AtomicCampaignReceiptStore(args.receipt)).publish(images, templates)
        print(json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":")))
    except (OSError, ValueError, PublicationError):
        parser.error("publication operation failed")
    return 0


def _gpu_ids(value: str) -> tuple[int, ...]:
    pieces = value.split(",")
    if not 1 <= len(pieces) <= 4 or any(not piece.isdigit() for piece in pieces):
        raise PublicationError("--gpu-ids must be one through four comma-separated nonnegative IDs")
    ids = tuple(int(piece) for piece in pieces)
    if len(set(ids)) != len(ids):
        raise PublicationError("--gpu-ids must not repeat IDs")
    return ids


def _smoke_campaign(args: argparse.Namespace) -> int:
    """Run two sequential paid smokes only after an explicit --execute.

    The default output intentionally performs no provider, Hub, Docker, or SSH
    operation; operators can inspect the exact immutable bindings first.
    """
    bindings = {
        "training": {"image": args.training_image, "template_id": args.training_template_id, "payload_hash": args.training_template_payload_sha256},
        "rollout": {"image": args.rollout_image, "template_id": args.rollout_template_id, "payload_hash": args.rollout_template_payload_sha256},
    }
    _validate_smoke_bindings(bindings)
    if not args.execute:
        print(json.dumps({"dry_run": True, "ledger": str(args.ledger), "receipt": str(args.receipt), "bindings": bindings, "provider_calls": 0}, sort_keys=True, separators=(",", ":")))
        return 0
    if os.environ.get("B1K_VAST_PRIVATE_PULL_READY") != "verified":
        raise PublicationError("--execute requires a separately verified private-pull readiness gate")
    if args.publication_receipt is None:
        raise PublicationError("--execute requires the complete publication receipt that binds both template identities")
    publication = AtomicCampaignReceiptStore(args.publication_receipt).read()
    if publication is None or publication.status != "complete" or len(publication.images) != 2 or len(publication.templates) != 2:
        raise PublicationError("publication receipt is incomplete or unsafe")
    _verify_smoke_bindings_against_publication(bindings, publication)
    from .dockerhub import TokenSource
    from .huggingface import CheckpointBucket, CheckpointBucketHelperClient, HubRepository, HuggingFaceHubClient, HuggingFaceReleaseVerifier, ReleaseDestinations
    from .ledger import RentalLedger
    from .production import ConfiguredPublicationSettings, _required_private_file
    from .production_smoke import SshSmokeRemote, VastCliSmokeClient
    from .smoke import SmokeController, SmokePlan, SmokeReadinessReceipt, SmokeTemplatePublicationReceipt, SmokeTimeouts
    from .vast import CappedVastController, VastAdapter

    settings = ConfiguredPublicationSettings.from_environment()
    identity = args.ssh_identity or _required_private_file(os.environ, "B1K_SMOKE_SSH_IDENTITY")
    hub = HuggingFaceReleaseVerifier(HuggingFaceHubClient(), TokenSource.from_token_file(_required_private_file(os.environ, "B1K_HF_TOKEN_FILE")))
    destinations = ReleaseDestinations(HubRepository("ryanjin333/behavior1k-groot-n17-models", "model"), CheckpointBucket(), HubRepository("ryanjin333/behavior1k-groot-n17-rollouts", "dataset"))
    hub.verify_private_repositories({"model": destinations.model, "dataset": destinations.dataset})
    helper = os.environ.get("B1K_CHECKPOINT_BUCKET_HELPER")
    if not isinstance(helper, str) or not helper.startswith("/"):
        raise PublicationError("--execute requires B1K_CHECKPOINT_BUCKET_HELPER for the exact private bucket pre-rent probe")
    hub.bootstrap_checkpoint_bucket_probe(CheckpointBucketHelperClient(helper, str(_required_private_file(os.environ, "B1K_HF_TOKEN_FILE"))), destinations.checkpoint_bucket)
    vast = VastCliSmokeClient(vastai_executable=settings.vastai_executable, api_key_file=settings.vast_api_key_file)
    releases = {item.purpose: item for item in publication.images}
    remote = SshSmokeRemote(
        vast=vast,
        identity_file=identity,
        known_hosts=args.known_hosts.absolute(),
        training_image=args.training_image,
        rollout_image=args.rollout_image,
        training_release=releases["training"],
        rollout_release=releases["rollout"],
        hub_verifier=hub,
    )
    controller = SmokeController(CappedVastController(RentalLedger(args.ledger), VastAdapter(vast)), remote)
    timeouts = SmokeTimeouts(ssh_timeout_seconds=900, runtime_timeout_seconds=1800, contract_timeout_seconds=1800, disappearance_timeout_seconds=300)
    result: dict[str, object] = {"dry_run": False, "runs": [], "ephemeral_templates": {}}
    for role, purpose in (("training", "training-smoke"), ("rollout", "rollout-smoke")):
        image = releases[role]
        production_template = SmokeTemplatePublicationReceipt(bindings[role]["template_id"], image, bindings[role]["payload_hash"])
        name = vast.new_ephemeral_smoke_template_name(role)
        result["ephemeral_templates"][role] = {"name": name, "status": "intent-recorded"}
        smoke_template = None
        primary: Exception | None = None
        try:
            _write_receipt(args.receipt, result)
            smoke_template = vast.create_ephemeral_smoke_template(role, production_template, name=name)
            result["ephemeral_templates"][role].update({"template_id": smoke_template.template_id, "payload_hash": smoke_template.payload_hash, "image_reference": smoke_template.image_release.reference, "status": "created"})
            _write_receipt(args.receipt, result)
            plan = SmokePlan(
                purpose=purpose,
                offer=vast.select_offer(purpose),
                template=smoke_template,
                destination_readiness=SmokeReadinessReceipt(image, destinations, f"preflight-{role}-release-ready"),
            )
            receipt = controller.run(plan, timeouts=timeouts)
        except Exception as error:
            primary = error
            failed = getattr(error, "receipt", None)
            if failed is not None:
                result["runs"].append(_smoke_receipt(failed))
                result["failed_run_id"] = failed.run_id
                result["failure"] = failed.failure.code if failed.failure is not None else "smoke-run-failed"
                _write_receipt(args.receipt, result)
        finally:
            if smoke_template is not None:
                try:
                    vast.destroy_ephemeral_smoke_template(smoke_template)
                    result["ephemeral_templates"][role]["absence_verified"] = True
                except Exception as cleanup_failure:
                    result["ephemeral_templates"][role]["absence_verified"] = False
                    result["template_cleanup_failure"] = True
                    _write_receipt(args.receipt, result)
                    if primary is not None:
                        raise primary from cleanup_failure
                    raise
        if primary is not None:
            _write_receipt(args.receipt, result)
            raise primary
        result["runs"].append(_smoke_receipt(receipt))
        _write_receipt(args.receipt, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _validate_smoke_bindings(bindings: Mapping[str, Mapping[str, str]]) -> None:
    for role, row in bindings.items():
        if not re.fullmatch(r"docker\.io/ryanjin333/behavior1k-groot-n17@sha256:[0-9a-f]{64}", row["image"]):
            raise PublicationError("smoke image must be an exact published digest")
        if not re.fullmatch(r"[1-9][0-9]*", row["template_id"]) or not re.fullmatch(r"[0-9a-f]{64}", row["payload_hash"]):
            raise PublicationError("smoke must use exact publication template IDs and payload hashes")


def _verify_smoke_bindings_against_publication(bindings: Mapping[str, Mapping[str, str]], publication: object) -> None:
    for role in ("training", "rollout"):
        image = next((item for item in publication.images if item.purpose == role), None)
        template = next((item for item in publication.templates if item.purpose == role), None)
        bound = bindings[role]
        if image is None or template is None or template.template_id is None or image.reference != bound["image"] or template.template_id != bound["template_id"] or template.payload_hash != bound["payload_hash"] or template.image_reference != image.reference:
            raise PublicationError("raw smoke binding does not exactly match the complete publication receipt")


def _smoke_receipt(receipt: object) -> dict[str, object]:
    return {
        "run_id": receipt.run_id, "purpose": receipt.purpose, "instance_id": receipt.instance_id,
        "projected_spend_usd": str(receipt.projected_spend_usd), "states": [state.value for state in receipt.states],
        "artifacts": [{"classification": item.artifact.classification, "key": item.artifact.key, "upload_commit": item.artifact.upload_commit, "delete_commit": item.artifact.delete_commit, "absence_verified": item.artifact.absence_verified} for item in receipt.artifacts],
    }


def _write_receipt(path: Path, value: Mapping[str, object]) -> None:
    path = path.absolute()
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise PublicationError("smoke receipt path is unsafe")
    descriptor = _open_private_receipt_parent(path.parent)
    temporary = f".{path.name}.{os.getpid()}.tmp"
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=descriptor)
        try:
            encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            os.write(fd, encoded); os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, path.name, src_dir_fd=descriptor, dst_dir_fd=descriptor)
        os.fsync(descriptor)
    finally:
        try:
            os.unlink(temporary, dir_fd=descriptor)
        except FileNotFoundError:
            pass
        os.close(descriptor)


def _open_private_receipt_parent(parent: Path) -> int:
    """Traverse every receipt ancestor with dir-fd + no-follow semantics."""
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parent.parts[1:]:
            if part in {"", ".", ".."}:
                raise PublicationError("smoke receipt path is unsafe")
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(part, 0o700, dir_fd=descriptor)
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor)
            metadata = os.fstat(child)
            if metadata.st_uid not in {0, os.getuid()} or (metadata.st_mode & 0o022 and not metadata.st_mode & stat.S_ISVTX):
                os.close(child)
                raise PublicationError("smoke receipt ancestor is unsafe")
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise PublicationError("smoke receipt parent must be a current-user private directory")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise
if __name__ == "__main__":
    raise SystemExit(main())
