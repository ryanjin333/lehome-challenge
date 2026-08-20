#!/usr/bin/env python3
"""Build a portable controlled-recovery schedule plus local hydration receipt."""
from __future__ import annotations

import argparse, hashlib, json, math, os, stat, tempfile
from pathlib import Path
from typing import Mapping, Sequence

_CATEGORIES = ("pant_long", "top_long", "top_short")
_CAPS = {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}
_PROFILE = {"cloth_displacement_m": 0.002, "cloth_velocity_mps": 0.01, "gripper_offset_rad": 0.02}
_HORIZON = 16

def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()

def _json(path: Path, *, label: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file(): raise ValueError(f"{label} is missing or unsafe")
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error: raise ValueError(f"{label} is malformed") from error
    if not isinstance(value, Mapping): raise ValueError(f"{label} is malformed")
    return value

def _safe_file(root: Path, relative: str) -> Path:
    path = root
    for part in Path(relative).parts:
        if part in {"", ".", ".."}: raise ValueError("accepted source path is unsafe")
        path /= part
        try:
            if stat.S_ISLNK(path.lstat().st_mode): raise ValueError("accepted source path is unsafe")
        except OSError as error: raise ValueError("accepted source path is missing or unsafe") from error
    if not stat.S_ISREG(path.lstat().st_mode): raise ValueError("accepted source path is missing or unsafe")
    return path

def _root_for(roots: Sequence[Path], row: Mapping[str, object]) -> Path:
    episode = row.get("source_episode_id")
    if not isinstance(episode, str) or not episode or "/" in episode or "\\" in episode: raise ValueError("audit source episode ID is unsafe")
    matches = [root / episode for root in roots if (root / episode).is_dir() and not (root / episode).is_symlink()]
    if len(matches) != 1: raise ValueError("portable audit source does not map to exactly one accepted root")
    return matches[0]

def _annotations(path: Path) -> tuple[list[dict[str, object]], str]:
    text = path.read_text(encoding="utf-8")
    try: rows = [json.loads(line) for line in text.splitlines()]
    except json.JSONDecodeError as error: raise ValueError("accepted annotations are malformed") from error
    if not rows or not all(isinstance(row, dict) for row in rows): raise ValueError("accepted annotations are malformed")
    return rows, hashlib.sha256(text.encode()).hexdigest()


def _validate_policy_chunk_trace(rows: Sequence[Mapping[str, object]]) -> None:
    """Re-authenticate the complete H=16 cached-action trace before replay."""

    seen_request_ids: set[str] = set()
    active_request: str | None = None
    for index, row in enumerate(rows):
        request_id, offset = row.get("policy_request_id"), row.get("policy_chunk_offset")
        if (row.get("step") != index or row.get("action_source") != "policy"
                or not isinstance(request_id, str) or not request_id.strip()
                or type(offset) is not int or not 0 <= offset < _HORIZON):
            raise ValueError("accepted annotations have malformed policy chunk trace")
        expected_offset = index % _HORIZON
        if offset != expected_offset:
            raise ValueError("accepted annotations have an invalid policy chunk trace offset transition")
        if expected_offset == 0:
            if request_id in seen_request_ids:
                raise ValueError("accepted annotations reuse a policy chunk trace request ID")
            seen_request_ids.add(request_id)
            active_request = request_id
        elif request_id != active_request:
            raise ValueError("accepted annotations change policy chunk trace request before offset 15")


def _state_fingerprint(*, category: str, garment: str, state: list[object]) -> str:
    if len(state) != 12:
        raise ValueError("audit continuation state must be a finite 12-D vector")
    rounded: list[str] = []
    for value in state:
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError("audit continuation state must be a finite 12-D vector")
        number = float(value)
        rounded.append("0.000000" if number == 0.0 else format(number, ".6f"))
    return hashlib.sha256(json.dumps(
        {"category": category, "garment": garment, "state_rounding": "fixed_6dp", "state": rounded},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _continuation(
    *, row: Mapping[str, object], annotation_rows: list[dict[str, object]], category: str, garment: str,
) -> tuple[int, str, list[object]]:
    """Verify the audit's exact reset-safe, fresh-policy start against source data."""

    evidence = row.get("continuation_start")
    if not isinstance(evidence, Mapping):
        raise ValueError("audit selected recovery lacks continuation provenance")
    index, step = evidence.get("annotation_index"), evidence.get("step")
    request_id, chunk_offset = evidence.get("policy_request_id"), evidence.get("policy_chunk_offset")
    state, fingerprint = evidence.get("state"), evidence.get("state_fingerprint")
    if (type(index) is not int or type(step) is not int or index <= 0 or step != index
            or not isinstance(request_id, str) or not request_id.strip() or chunk_offset != 0
            or not isinstance(state, list) or not isinstance(fingerprint, str) or len(fingerprint) != 64):
        raise ValueError("audit continuation provenance is malformed")
    if index >= len(annotation_rows):
        raise ValueError("audit continuation is outside accepted annotations")
    annotation = annotation_rows[index]
    if (annotation.get("step") != step or annotation.get("policy_request_id") != request_id
            or annotation.get("policy_chunk_offset") != 0 or annotation.get("state") != state):
        raise ValueError("audit continuation does not match accepted chunk-boundary evidence")
    event = row.get("recovery_event")
    if not isinstance(event, Mapping):
        raise ValueError("audit continuation lacks recovery-event evidence")
    adverse, confirmation = event.get("adverse_start"), event.get("recovery_confirmation")
    if type(adverse) is not int or type(confirmation) is not int or not adverse <= index < confirmation:
        raise ValueError("audit continuation is not inside the corrective interval")
    actual_fingerprint = _state_fingerprint(category=category, garment=garment, state=state)
    if fingerprint != actual_fingerprint or row.get("fingerprint") != actual_fingerprint:
        raise ValueError("audit continuation state fingerprint mismatch")
    return index, actual_fingerprint, list(state)

def _source(row: Mapping[str, object], roots: Sequence[Path]) -> dict[str, object]:
    category = row.get("category")
    if category not in _CATEGORIES: raise ValueError("audit selected a category without a controlled recovery cap")
    package, episode = _root_for(roots, row), str(row["source_episode_id"])
    raw = package / "raw" / episode
    reset, annotations, episode_path, raw_manifest = (_safe_file(raw, "snapshots/reset.json"), _safe_file(raw, "annotations.jsonl"), _safe_file(raw, "episode.json"), _safe_file(raw, "SHA256SUMS.json"))
    artifacts = row.get("source_artifacts")
    if not isinstance(artifacts, Mapping) or artifacts.get("package_sync_digest") != row.get("source_episode_digest"): raise ValueError("audit source package digest does not match its portable episode identity")
    if artifacts.get("raw_checksum_manifest_sha256") != _sha256(raw_manifest): raise ValueError("accepted source checksum manifest SHA-256 mismatch")
    manifest = _json(raw_manifest, label="accepted source checksum manifest")
    for path, relative in ((reset, "snapshots/reset.json"), (annotations, "annotations.jsonl"), (episode_path, "episode.json")):
        record = manifest.get(relative)
        if not isinstance(record, Mapping) or record.get("sha256") != _sha256(path) or record.get("size") != path.stat().st_size: raise ValueError("accepted source checksum manifest does not authenticate source artifacts")
    if artifacts.get("annotations_sha256") != _sha256(annotations) or artifacts.get("episode_manifest_sha256") != _sha256(episode_path): raise ValueError("accepted source artifact SHA-256 mismatch")
    episode_doc = _json(episode_path, label="accepted source episode")
    identity = episode_doc.get("identity")
    if not isinstance(identity, Mapping) or episode_doc.get("episode_id") != episode or episode_doc.get("accepted_success") is not True or episode_doc.get("outcome") != "success" or episode_doc.get("terminal_reason") != "success" or identity.get("category") != category or identity.get("garment_name") != row.get("garment"): raise ValueError("audit source is not a terminal successful accepted episode")
    annotation_rows, annotation_hash = _annotations(annotations)
    _validate_policy_chunk_trace(annotation_rows)
    actions, first_success = [], None
    for index, annotation in enumerate(annotation_rows):
        if annotation.get("step") != index or not isinstance(annotation.get("action"), list): raise ValueError("accepted annotations lack ordered recorded actions")
        action = [float(value) for value in annotation["action"]]
        if len(action) != 12 or not all(math.isfinite(value) for value in action): raise ValueError("accepted annotations contain invalid 12D actions")
        actions.append(action)
        if annotation.get("success") is True and first_success is None: first_success = index
    if first_success is None or first_success < 2: raise ValueError("accepted source has no usable pre-success prefix")
    stop, fingerprint, continuation_state = _continuation(
        row=row, annotation_rows=annotation_rows, category=str(category), garment=str(row["garment"]),
    )
    if stop >= first_success:
        raise ValueError("audit continuation must precede the first successful source record")
    portable = {key: row[key] for key in ("source_round_id", "source_round_ordinal", "source_episode_id", "source_episode_digest", "source_immutable_revision", "source_receipt_file_name", "source_receipt_remote_prefix", "source_receipt_publication_ref", "source_receipt_sha256", "receipt_immutable_revision", "garment", "category") if key in row}
    portable.update({"source_artifacts": dict(artifacts), "source_continuation_state": continuation_state, "source_state_fingerprint": fingerprint, "source_reset_sha256": _sha256(reset), "source_annotations_sha256": annotation_hash, "source_first_success_step": first_success, "prefix_stop": stop, "action_prefix_sha256": hashlib.sha256(_canonical(actions[:stop])).hexdigest()})
    return {"portable": portable, "source_reset": str(reset), "source_annotations": str(annotations)}

def _write_absent(path: Path, payload: bytes) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink(): raise FileExistsError("output must be an absent absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir(): raise ValueError("output parent is unsafe")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent); temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload); stream.flush(); os.fchmod(stream.fileno(), 0o644); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally: temporary.unlink(missing_ok=True)

def _preflight(paths: Sequence[Path]) -> None:
    if any(not path.is_absolute() or path.exists() or path.is_symlink() for path in paths): raise FileExistsError("output must be an absent absolute path")

def build_controlled_recovery_matrix(*, audit_path: Path | str, accepted_roots: Sequence[Path | str], output: Path | str, max_attempts: int = 96) -> dict[str, object]:
    if type(max_attempts) is not int or not 8 <= max_attempts <= 96: raise ValueError("max_attempts must be in 8..96")
    audit_file, sidecar = Path(audit_path), Path(str(audit_path) + ".sha256")
    if not audit_file.is_absolute() or audit_file.is_symlink(): raise ValueError("recovery audit must be an absolute regular file")
    audit, audit_sha = _json(audit_file, label="recovery audit"), _sha256(audit_file)
    if sidecar.is_symlink() or not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != audit_sha: raise ValueError("recovery audit SHA-256 sidecar mismatch")
    audit_semantic = {key: value for key, value in audit.items() if key != "semantic_sha256"}
    if audit.get("schema_version") != 2 or audit.get("kind") != "lehome_successful_recovery_audit" or audit.get("semantic_sha256") != hashlib.sha256(json.dumps(audit_semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest(): raise ValueError("recovery audit semantic identity mismatch")
    roots = tuple(Path(root) for root in accepted_roots)
    if not roots or any(not root.is_absolute() or root.is_symlink() or not root.is_dir() for root in roots): raise ValueError("accepted roots must be real absolute directories")
    selected, shortfalls = audit.get("selected_recoveries"), audit.get("shortfalls")
    if not isinstance(selected, list) or not isinstance(shortfalls, Mapping): raise ValueError("recovery audit selection is malformed")
    sources = [_source(row, roots) for row in selected if isinstance(row, Mapping) and row.get("category") in _CATEGORIES]
    by_category = {category: [source for source in sources if source["portable"]["category"] == category] for category in _CATEGORIES}
    for category, cap in _CAPS.items():
        if int(shortfalls.get(category, 0)) != cap: raise ValueError("audit shortfall does not match immutable controlled recovery caps")
        if cap and not by_category[category]: raise ValueError("controlled recovery shortfall lacks an audited source")
    semantic_rows, hydrated_rows = [], []
    # The first eight rows must make every immutable acceptance cap reachable
    # even if the caller requests the minimum valid attempt bound.  Thereafter
    # choose the least represented category (with canonical tuple order for
    # ties), which preserves deterministic retry scheduling and makes 96 rows
    # exactly balanced at 32 attempts per active category.
    schedule = ["pant_long"] * 4 + ["top_long"] + ["top_short"] * 3
    while len(schedule) < max_attempts:
        counts = {category: schedule.count(category) for category in _CATEGORIES}
        least_count = min(counts.values())
        schedule.append(next(category for category in _CATEGORIES if counts[category] == least_count))
    for index, category in enumerate(schedule):
        source = by_category[category][index % len(by_category[category])]; portable = source["portable"]; seed = 71_000 + index
        perturbation = hashlib.sha256(_canonical({**_PROFILE, "seed": seed, "source_episode_digest": portable["source_episode_digest"], "prefix_sha256": portable["action_prefix_sha256"]})).hexdigest()
        source_perturbation = hashlib.sha256(_canonical({"source_state_fingerprint": portable["source_state_fingerprint"], "perturbation_fingerprint": perturbation})).hexdigest()
        attempt_id = f"controlled-{category.replace('_', '-')}-{index:03d}-{perturbation[:16]}"
        row = {"attempt_id": attempt_id, "trial_id": attempt_id, "garment": portable["garment"], "garment_name": portable["garment"], "category": category, "release_stage": "seen", "seed": seed, "strategy": "canonical", "recovery_kind": "controlled_success_recovery_v1", "category_acceptance_cap": _CAPS[category], **portable, "perturbation_profile": dict(_PROFILE), "perturbation_seed": seed, "perturbation_fingerprint": perturbation, "source_state_perturbation_fingerprint": source_perturbation}
        semantic_rows.append(row); hydrated_rows.append({**row, "source_reset": source["source_reset"], "source_annotations": source["source_annotations"]})
    matrix = {"schema_version": 1, "kind": "controlled_success_recovery_matrix_v1", "audit": {"file_sha256": audit_sha, "semantic_sha256": audit["semantic_sha256"]}, "target_accepted": 8, "category_acceptance_caps": dict(_CAPS), "rows": semantic_rows}
    target, materialization = Path(output), Path(str(output) + ".materialization.json")
    receipt, materialization_receipt = Path(str(target) + ".sha256"), Path(str(materialization) + ".sha256")
    _preflight((target, receipt, materialization, materialization_receipt))
    payload = _canonical(matrix); matrix_sha = hashlib.sha256(payload).hexdigest(); hydration_payload = _canonical({"schema_version": 1, "kind": "controlled_success_recovery_materialization_v1", "matrix_sha256": matrix_sha, "target_accepted": 8, "category_acceptance_caps": dict(_CAPS), "rows": [{**row, "controlled_matrix_sha256": matrix_sha} for row in hydrated_rows]})
    _write_absent(target, payload); _write_absent(receipt, (matrix_sha + "\n").encode()); _write_absent(materialization, hydration_payload); _write_absent(materialization_receipt, (hashlib.sha256(hydration_payload).hexdigest() + "\n").encode())
    return {"matrix_path": str(target), "matrix_sha256": matrix_sha, "materialization_path": str(materialization), "materialization_sha256": hashlib.sha256(hydration_payload).hexdigest(), "attempt_count": max_attempts, "category_acceptance_caps": dict(_CAPS), "target_accepted": 8}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--audit", type=Path, required=True); parser.add_argument("--accepted-root", type=Path, action="append", required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--max-attempts", type=int, default=96); args = parser.parse_args(argv)
    print(json.dumps(build_controlled_recovery_matrix(audit_path=args.audit, accepted_roots=args.accepted_root, output=args.output, max_attempts=args.max_attempts), sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
