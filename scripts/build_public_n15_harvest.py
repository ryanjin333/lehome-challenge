#!/usr/bin/env python3
"""Build and verify the offline public-N1.5 native-harvest contracts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "source/lehome"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lehome.n15_harvest import (  # noqa: E402
    HarvestError,
    HarvestProvenance,
    admission_smoke_schedule,
    assess_worker_memory,
    assess_worker_admission,
    admit_workers,
    build_manifest,
    canonical_bytes,
    collect_native_outcomes,
    evaluate_first_100,
    measure_runtime_contract,
    inspect_success_datasets,
    native_worker_plan,
    publish_harvest_bundle,
    provider_stop_receipt_from_response,
    terminal_receipt,
    validate_manifest,
    validate_collected_outcomes,
    validate_provider_stop_receipt,
    verify_manifest_receipt,
    write_observational_site,
    write_manifest_bundle,
)


def _path(parser: argparse.ArgumentParser, name: str, *, required: bool = True) -> None:
    parser.add_argument(name, type=Path, required=required)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build")
    build.add_argument("--checkpoint-tree-sha256", required=True)
    build.add_argument("--checkpoint-receipt-sha256", required=True)
    build.add_argument("--runtime-receipt-sha256", required=True)
    build.add_argument("--source-tree-sha256", required=True)
    build.add_argument("--dataset-snapshot-sha256", required=True)
    build.add_argument("--rollout-image-sha256", required=True)
    build.add_argument("--base-seed", type=int, default=100_000)
    _path(build, "--manifest")
    _path(build, "--receipt")

    verify = commands.add_parser("verify")
    _path(verify, "--manifest")
    _path(verify, "--receipt")

    first = commands.add_parser("first-100")
    _path(first, "--manifest")
    _path(first, "--outcomes")
    _path(first, "--output")

    runtime = commands.add_parser("verify-runtime")
    _path(runtime, "--source-root")
    runtime.add_argument("--source-revision", required=True)
    _path(runtime, "--checkpoint-root")
    _path(runtime, "--training-identity-receipt")
    _path(runtime, "--rollout-image-receipt")
    _path(runtime, "--docker-inspect-receipt")
    _path(runtime, "--output")

    collect = commands.add_parser("collect-outcomes")
    _path(collect, "--manifest")
    _path(collect, "--process-status")
    _path(collect, "--harvest-root")
    collect.add_argument("--expected-attempt-count", type=int, required=True)
    _path(collect, "--success-dataset-receipt")
    _path(collect, "--output")

    inspect_datasets = commands.add_parser("inspect-success-datasets")
    _path(inspect_datasets, "--manifest")
    _path(inspect_datasets, "--harvest-root")
    inspect_datasets.add_argument("--expected-attempt-count", type=int, required=True)
    _path(inspect_datasets, "--output")

    observational_site = commands.add_parser("write-observational-site")
    _path(observational_site, "--output-dir")

    status = commands.add_parser("build-process-status")
    _path(status, "--tsv")
    status.add_argument("--expected-process-count", type=int, required=True)
    _path(status, "--output")

    admission = commands.add_parser("admit-workers")
    _path(admission, "--manifest")
    _path(admission, "--runtime-receipt")
    _path(admission, "--four-worker-evidence-root")
    _path(admission, "--two-worker-evidence-root", required=False)
    _path(admission, "--output")

    memory = commands.add_parser("assess-memory")
    _path(memory, "--evidence-root")
    memory.add_argument("--worker-count", type=int, choices=(2, 4), required=True)
    _path(memory, "--output")

    schedule = commands.add_parser("admission-schedule")
    schedule.add_argument("--worker-count", type=int, choices=(2, 4), required=True)

    assess_admission = commands.add_parser("assess-admission")
    _path(assess_admission, "--manifest")
    _path(assess_admission, "--runtime-receipt")
    _path(assess_admission, "--evidence-root")
    assess_admission.add_argument("--worker-count", type=int, choices=(2, 4), required=True)
    _path(assess_admission, "--output")

    plan = commands.add_parser("render-worker-plan")
    _path(plan, "--manifest")
    _path(plan, "--admission")
    _path(plan, "--source-root")
    _path(plan, "--checkpoint-root")
    _path(plan, "--output-root")
    _path(plan, "--output")

    terminal = commands.add_parser("verify-terminal")
    _path(terminal, "--manifest")
    _path(terminal, "--manifest-receipt")
    _path(terminal, "--publication-receipt")
    _path(terminal, "--provider-receipt")
    _path(terminal, "--output")

    publish = commands.add_parser("publish-hf")
    _path(publish, "--bundle-root")
    _path(publish, "--manifest")
    _path(publish, "--manifest-receipt")
    _path(publish, "--final-outcomes")
    publish.add_argument("--repository", required=True)
    publish.add_argument("--revision", default="main")
    publish.add_argument("--token-env", default="HF_TOKEN")
    _path(publish, "--output")

    observe = commands.add_parser("observe-provider-stop")
    _path(observe, "--response")
    _path(observe, "--output")
    validate_stop = commands.add_parser("validate-provider-stop")
    _path(validate_stop, "--provider-receipt")
    return parser


def _load_json(path: Path, *, label: str, require_canonical: bool = True) -> object:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise HarvestError(f"{label} path is unsafe or unavailable")
    try:
        raw = path.read_bytes()
        def reject_duplicates(pairs: list[tuple[object, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if not isinstance(key, str) or key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result

        value = json.loads(raw, object_pairs_hook=reject_duplicates)
        if require_canonical and raw != canonical_bytes(value):
            raise ValueError("non-canonical JSON")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise HarvestError(f"{label} is malformed") from error
    return value


def _write_output(path: Path, value: object, *, label: str) -> None:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise HarvestError(f"{label} path is unsafe")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise HarvestError(f"{label} already exists")
    created = False
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        created = True
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o444)
    except OSError as error:
        if created:
            path.unlink(missing_ok=True)
        raise HarvestError(f"{label} could not be written atomically") from error


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            value = build_manifest(
                provenance=HarvestProvenance(
                    checkpoint_tree_sha256=args.checkpoint_tree_sha256,
                    checkpoint_receipt_sha256=args.checkpoint_receipt_sha256,
                    runtime_receipt_sha256=args.runtime_receipt_sha256,
                    source_tree_sha256=args.source_tree_sha256,
                    dataset_snapshot_sha256=args.dataset_snapshot_sha256,
                    rollout_image_sha256=args.rollout_image_sha256,
                ),
                base_seed=args.base_seed,
            )
            result = write_manifest_bundle(
                manifest=value,
                manifest_path=args.manifest,
                receipt_path=args.receipt,
            )
        elif args.command == "verify":
            manifest = _load_json(args.manifest, label="manifest")
            receipt = _load_json(args.receipt, label="manifest receipt")
            result = verify_manifest_receipt(manifest=manifest, receipt=receipt)
        elif args.command == "first-100":
            loaded = _load_json(args.outcomes, label="first-100 outcomes")
            if isinstance(loaded, dict):
                loaded = validate_collected_outcomes(
                    loaded,
                    manifest=_load_json(args.manifest, label="manifest"),
                    expected_attempt_count=100,
                )["outcomes"]
            result = evaluate_first_100(
                manifest=_load_json(args.manifest, label="manifest"),
                outcomes=loaded,
            )
            _write_output(args.output, result, label="first-100 gate receipt")
        elif args.command == "verify-runtime":
            try:
                import lerobot
            except ImportError as error:
                raise HarvestError("installed LeRobot package is unavailable") from error
            package_root = Path(lerobot.__file__).resolve(strict=True).parent
            result = measure_runtime_contract(
                source_root=args.source_root,
                source_revision=args.source_revision,
                checkpoint_root=args.checkpoint_root,
                training_identity_receipt=args.training_identity_receipt,
                rollout_image_receipt=args.rollout_image_receipt,
                docker_inspect_receipt=args.docker_inspect_receipt,
                python_executable=Path(sys.executable),
                python_version=".".join(str(value) for value in sys.version_info[:3]),
                lerobot_package_root=package_root,
            )
            _write_output(args.output, result, label="harvest runtime receipt")
        elif args.command == "collect-outcomes":
            result = collect_native_outcomes(
                manifest=_load_json(args.manifest, label="manifest"),
                process_status=_load_json(args.process_status, label="process status receipt"),
                harvest_root=args.harvest_root,
                expected_attempt_count=args.expected_attempt_count,
                success_dataset_receipt=_load_json(
                    args.success_dataset_receipt, label="success dataset receipt"
                ),
            )
            _write_output(args.output, result, label="collected outcome receipt")
        elif args.command == "inspect-success-datasets":
            result = inspect_success_datasets(
                manifest=_load_json(args.manifest, label="manifest"),
                harvest_root=args.harvest_root,
                expected_attempt_count=args.expected_attempt_count,
            )
            _write_output(args.output, result, label="success dataset receipt")
        elif args.command == "write-observational-site":
            result = write_observational_site(args.output_dir)
        elif args.command == "build-process-status":
            if args.expected_process_count not in {4, 40}:
                raise HarvestError("expected process count must be 4 or 40")
            try:
                lines = args.tsv.read_text(encoding="ascii").splitlines()
            except (OSError, UnicodeError) as error:
                raise HarvestError("process status TSV is unreadable") from error
            processes = []
            for line in lines:
                fields = line.split("\t")
                if len(fields) != 5:
                    raise HarvestError("process status TSV is malformed")
                process_id, category, garment, seed, exit_code = fields
                try:
                    process_seed = int(seed)
                    code = int(exit_code)
                except ValueError as error:
                    raise HarvestError("process status TSV is malformed") from error
                processes.append({
                    "process_id": process_id, "category": category, "garment": garment,
                    "process_seed": process_seed, "exit_code": code,
                })
            if len(processes) != args.expected_process_count:
                raise HarvestError("process status TSV count is incomplete")
            result = {
                "schema_version": 1, "kind": "lehome_public_n15_process_status_v1",
                "processes": processes,
            }
            _write_output(args.output, result, label="process status receipt")
        elif args.command == "admit-workers":
            result = admit_workers(
                manifest=_load_json(args.manifest, label="manifest"),
                runtime_receipt=args.runtime_receipt,
                four_worker_evidence_root=args.four_worker_evidence_root,
                two_worker_evidence_root=args.two_worker_evidence_root,
            )
            _write_output(args.output, result, label="worker selection receipt")
        elif args.command == "assess-memory":
            result = assess_worker_memory(
                evidence_root=args.evidence_root, worker_count=args.worker_count,
            )
            _write_output(args.output, result, label="worker memory measurement receipt")
        elif args.command == "admission-schedule":
            result = admission_smoke_schedule(args.worker_count)
        elif args.command == "assess-admission":
            result = assess_worker_admission(
                manifest=_load_json(args.manifest, label="manifest"),
                runtime_receipt=args.runtime_receipt,
                evidence_root=args.evidence_root,
                worker_count=args.worker_count,
            )
            _write_output(args.output, result, label="worker admission receipt")
        elif args.command == "render-worker-plan":
            result = native_worker_plan(
                manifest=_load_json(args.manifest, label="manifest"),
                admission=_load_json(args.admission, label="worker selection receipt"),
                source_root=args.source_root,
                checkpoint_root=args.checkpoint_root,
                output_root=args.output_root,
            )
            _write_output(args.output, result, label="worker plan")
        elif args.command == "verify-terminal":
            result = terminal_receipt(
                manifest=_load_json(args.manifest, label="manifest"),
                manifest_receipt=_load_json(args.manifest_receipt, label="manifest receipt"),
                publication_receipt=_load_json(args.publication_receipt, label="publication receipt"),
                provider_receipt=_load_json(args.provider_receipt, label="provider stopped receipt"),
            )
            _write_output(args.output, result, label="terminal receipt")
        elif args.command == "publish-hf":
            try:
                from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
            except ImportError as error:
                raise HarvestError("huggingface_hub is unavailable") from error
            token = os.environ.get(args.token_env, "")
            result = publish_harvest_bundle(
                bundle_root=args.bundle_root,
                manifest=_load_json(args.manifest, label="manifest"),
                manifest_receipt_value=_load_json(args.manifest_receipt, label="manifest receipt"),
                final_outcomes=_load_json(args.final_outcomes, label="final outcomes"),
                repository=args.repository,
                revision=args.revision,
                token=token,
                authenticated_api=HfApi(token=token),
                anonymous_api=HfApi(token=False),
                downloader=hf_hub_download,
                operation_factory=lambda **kwargs: CommitOperationAdd(**kwargs),
            )
            _write_output(args.output, result, label="publication readback receipt")
        elif args.command == "observe-provider-stop":
            result = provider_stop_receipt_from_response(
                _load_json(args.response, label="provider response", require_canonical=False)
            )
            _write_output(args.output, result, label="provider stopped receipt")
        elif args.command == "validate-provider-stop":
            result = validate_provider_stop_receipt(
                _load_json(args.provider_receipt, label="provider stopped receipt")
            )
        else:  # pragma: no cover - argparse owns this branch.
            raise AssertionError("unreachable command")
    except (HarvestError, OSError, ValueError) as error:
        print(f"public N1.5 harvest gate failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
