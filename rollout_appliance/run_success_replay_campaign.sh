#!/usr/bin/env bash
# Replay checksum-verified original-12K success resets through the proven 12K appliance.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CAMPAIGN="${SCRIPT_DIR}/run_12k_campaign.sh"
WORKSPACE="${LEHOME_WORKSPACE:-/mnt/lehome}"
MATRIX="${LEHOME_SUCCESS_REPLAY_MATRIX:-}"
EXPECTED_SHA256="${LEHOME_SUCCESS_REPLAY_MATRIX_SHA256:-}"
WORKER_COUNT="${LEHOME_WORKER_COUNT:-4}"
MAX_ATTEMPTS="${LEHOME_MAX_ATTEMPTS:-200}"
TARGET_ACCEPTED="${LEHOME_TARGET_ACCEPTED:-150}"
MAX_WORKER_RESTARTS="${LEHOME_MAX_WORKER_RESTARTS:-8}"
CAMPAIGN_ROOT="${LEHOME_CAMPAIGN_ROOT:-${WORKSPACE}/eval/campaign-success-replay-12k-round-1}"
ROUND_ID="${LEHOME_ROUND_ID:-success-replay-12k-round-1}"
ORIGINAL_12K_POLICY_SHA256="e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa"
ORIGINAL_12K_POLICY_REPO="ryanjin333/lehome-groot-n17-models"
ORIGINAL_12K_POLICY_REVISION="30ac1a84da67b099e115ad147bcd61e9d60046d3"
ORIGINAL_12K_POLICY_ARTIFACT_SHA256="3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06"
ORIGINAL_12K_CHECKPOINT="${LEHOME_ORIGINAL_12K_CHECKPOINT:-${WORKSPACE}/eval/policies/original_baseline}"
FRESH_SOURCE_REPORTS="${LEHOME_FRESH_SOURCE_REPORTS_JSON:-}"
FRESH_SOURCE_MATRICES="${LEHOME_FRESH_SOURCE_MATRICES_JSON:-}"

if [ ! -x "${BASE_CAMPAIGN}" ] && [ ! -f "${BASE_CAMPAIGN}" ]; then
  echo "missing reusable 12K campaign appliance" >&2
  exit 2
fi
if [ -z "${MATRIX}" ]; then
  echo "LEHOME_SUCCESS_REPLAY_MATRIX is required" >&2
  exit 2
fi
if [ -z "${EXPECTED_SHA256}" ]; then
  echo "LEHOME_SUCCESS_REPLAY_MATRIX_SHA256 is required" >&2
  exit 2
fi
case "${MATRIX}" in
  /*) ;;
  *) echo "LEHOME_SUCCESS_REPLAY_MATRIX must be an absolute path" >&2; exit 2 ;;
esac
if [ -L "${MATRIX}" ] || [ ! -f "${MATRIX}" ] || [ -L "${MATRIX}.sha256" ] || [ ! -f "${MATRIX}.sha256" ]; then
  echo "success replay matrix and receipt must be regular files" >&2
  exit 2
fi
if ! [[ "${EXPECTED_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "LEHOME_SUCCESS_REPLAY_MATRIX_SHA256 must be a lowercase 64-character SHA-256" >&2
  exit 2
fi
if ! [[ "${TARGET_ACCEPTED}" =~ ^([1-9]|[1-9][0-9]|1[0-9][0-9]|200)$ ]]; then
  echo "LEHOME_TARGET_ACCEPTED must be an integer in 1..200" >&2
  exit 2
fi
if (( 10#${TARGET_ACCEPTED} > 150 )) && [ "${TARGET_ACCEPTED}" != "200" ]; then
  echo "LEHOME_TARGET_ACCEPTED must be in 1..150 unless using the exact fresh visual-only tuple" >&2
  exit 2
fi
ACTUAL_SHA256="$(sha256sum "${MATRIX}" | awk '{print $1}')"
RECEIPT_SHA256="$(tr -d '\r\n' < "${MATRIX}.sha256")"
if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ] || [ "${RECEIPT_SHA256}" != "${EXPECTED_SHA256}" ]; then
  echo "success replay matrix SHA-256 mismatch" >&2
  exit 2
fi
python3 - "${MATRIX}" "${MAX_ATTEMPTS}" "${TARGET_ACCEPTED}" "${FRESH_SOURCE_REPORTS}" "${FRESH_SOURCE_MATRICES}" <<'PY'
import json
import re
import sys
from collections import Counter
from pathlib import Path

matrix, max_attempts, target_accepted = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
report_evidence, matrix_evidence = sys.argv[4], sys.argv[5]
try:
    rows = json.loads(matrix.read_text(encoding="utf-8"))
except (OSError, ValueError) as error:
    raise SystemExit(f"success replay matrix is malformed: {error}")
categories = ("top_long", "top_short", "pant_long", "pant_short")
if not isinstance(rows, list) or not 1 <= len(rows) <= 400 or len(rows) != max_attempts:
    raise SystemExit("success replay matrix must match LEHOME_MAX_ATTEMPTS in 1..400")
counts = Counter(row.get("category") for row in rows if isinstance(row, dict))
if any(category not in categories for category in counts):
    raise SystemExit("success replay matrix contains an invalid category")
has_caps = ["category_acceptance_cap" in row for row in rows if isinstance(row, dict)]
if any(has_caps) and not all(has_caps):
    raise SystemExit("success replay matrix category caps must be present on every row")
cap_mode = all(has_caps)
if not cap_mode and (len(rows) != 200 or counts != Counter({key: 50 for key in categories})):
    raise SystemExit("legacy success replay matrix must contain 50 attempts per category")
caps = {}
fresh_fields = {
    "source_episode_sha256", "source_episode_root", "source_episode_path", "source_reset_sha256", "source_annotations_sha256",
    "source_continuation_snapshot_sha256", "source_state_fingerprint",
    "source_report_sha256", "source_matrix_sha256", "source_receipt_sha256", "source_receipt_path",
    "source_remote_prefix", "source_immutable_revision", "source_round_id", "source_run_id",
    "source_report_path", "source_matrix_path",
}
for row in rows:
    if not isinstance(row, dict):
        raise SystemExit("success replay matrix row is malformed")
    replay_kind = row.get("replay_kind")
    if replay_kind not in {"verified_success_reset_v1", "verified_success_early_snapshot_v1"}:
        raise SystemExit("success replay matrix row has an invalid replay kind")
    if (
        replay_kind == "verified_success_early_snapshot_v1"
        and row.get("restore_snapshot_step") != 16
    ) or (
        replay_kind == "verified_success_reset_v1"
        and "restore_snapshot_step" in row
    ):
        raise SystemExit("success replay matrix row has an invalid restore boundary")
    if not isinstance(row.get("restore_snapshot"), str) or not row["restore_snapshot"].startswith("/"):
        raise SystemExit("success replay matrix row has an unsafe restore snapshot")
    if not isinstance(row.get("restore_snapshot_sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", row["restore_snapshot_sha256"]) is None:
        raise SystemExit("success replay matrix row has an invalid restore digest")
    if row.get("restore_snapshot_cloth_frame") not in {"usd_local_points_v1", "physx_cloth_view_world_v1"}:
        raise SystemExit("success replay matrix row has an invalid cloth frame")
    if not isinstance(row.get("parent_episode_id"), str) or not row["parent_episode_id"] or row.get("lineage_id") != row["parent_episode_id"]:
        raise SystemExit("success replay matrix row has inconsistent lineage")
    if cap_mode:
        cap = row.get("category_acceptance_cap")
        category = row.get("category")
        if type(cap) is not int or not 0 <= cap <= 150 or (category in caps and caps[category] != cap):
            raise SystemExit("success replay matrix category cap is invalid")
        caps[category] = cap
if cap_mode and (any(caps[category] > counts[category] for category in caps) or sum(caps.values()) != target_accepted):
    raise SystemExit("success replay matrix caps do not match LEHOME_TARGET_ACCEPTED")
if target_accepted == 200:
    if (
        len(rows) != 400
        or counts != Counter({key: 100 for key in categories})
        or caps != {key: 50 for key in categories}
        or any(row.get("strategy") != "visual_only" for row in rows)
    ):
        raise SystemExit("200 accepted requires the exact fresh visual-only matrix tuple")
    for row in rows:
        if not fresh_fields <= set(row):
            raise SystemExit("200 accepted requires bound fresh-source provenance")
        if any(not isinstance(row[field], str) or re.fullmatch(r"[0-9a-f]{64}", row[field]) is None for field in fresh_fields - {"source_episode_root", "source_episode_path", "source_report_path", "source_matrix_path", "source_receipt_path", "source_remote_prefix", "source_immutable_revision", "source_round_id", "source_run_id"}):
            raise SystemExit("200 accepted fresh-source hashes are invalid")
        if (
            not isinstance(row["source_round_id"], str)
            or re.fullmatch(r"fresh-12k-[a-z0-9-]{1,112}", row["source_round_id"]) is None
            or row["source_remote_prefix"] != f"rollout-rounds/{row['source_round_id']}/{row['parent_episode_id']}"
            or re.fullmatch(r"[0-9a-f]{40}", str(row["source_immutable_revision"])) is None
        ):
            raise SystemExit("200 accepted fresh-source receipt binding is invalid")
elif sum(caps.values()) > 150:
    raise SystemExit("legacy success replay acceptance caps must remain at most 150")
if target_accepted == 200:
    def strict_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value: raise ValueError("duplicate JSON field")
            value[key] = item
        return value
    def evidence(raw, label):
        try: values = json.loads(raw, object_pairs_hook=strict_pairs)
        except ValueError as error: raise SystemExit(f"fresh {label} evidence is malformed: {error}")
        if not isinstance(values, list) or not values: raise SystemExit(f"fresh {label} evidence is required")
        found = {}
        for item in values:
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}: raise SystemExit(f"fresh {label} evidence is malformed")
            path, digest = Path(item.get("path", "")), item.get("sha256")
            if (not path.is_absolute() or path.is_symlink() or not path.is_file() or not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                    or __import__("hashlib").sha256(path.read_bytes()).hexdigest() != digest
                    or str(path) in found): raise SystemExit(f"fresh {label} evidence is missing or tampered")
            found[str(path)] = (digest, path)
        return found
    reports, matrices = evidence(report_evidence, "source report"), evidence(matrix_evidence, "source matrix")
    parsed_matrices, parsed_reports = {}, {}
    for path, (digest, file_path) in matrices.items():
        try: source_rows = json.loads(file_path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
        except ValueError as error: raise SystemExit(f"fresh source matrix is malformed: {error}")
        if not isinstance(source_rows, list): raise SystemExit("fresh source matrix is malformed")
        parsed_matrices[path] = {row.get("attempt_id"): row for row in source_rows if isinstance(row, dict)}
    for path, (digest, file_path) in reports.items():
        try: report = json.loads(file_path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
        except ValueError as error: raise SystemExit(f"fresh source report is malformed: {error}")
        if not isinstance(report, dict): raise SystemExit("fresh source report is malformed")
        body = dict(report); declared = body.pop("report_sha256", None)
        if declared != __import__("hashlib").sha256((json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest(): raise SystemExit("fresh source report authentication failed")
        parsed_reports[path] = report
    for row in rows:
        report_path, matrix_path = row.get("source_report_path"), row.get("source_matrix_path")
        if (not isinstance(report_path, str) or not isinstance(matrix_path, str)
                or report_path not in reports or matrix_path not in matrices
                or row.get("source_report_sha256") != reports[report_path][0]
                or row.get("source_matrix_sha256") != matrices[matrix_path][0]): raise SystemExit("fresh source report or matrix binding is invalid")
        report, source_row = parsed_reports[report_path], parsed_matrices[matrix_path].get(row.get("parent_episode_id"))
        trials = report.get("trials") if isinstance(report.get("trials"), list) else []
        trial = next((item for item in trials if isinstance(item, dict) and item.get("attempt_id") == row.get("parent_episode_id")), None)
        if (
            report.get("matrix_sha256") != row.get("source_matrix_sha256")
            or report.get("round_id") != row.get("source_round_id") or report.get("run_id") != row.get("source_run_id")
            or not isinstance(source_row, dict) or source_row.get("category") != row.get("category")
            or source_row.get("garment_name") != row.get("garment")
            or source_row.get("campaign_round_id") != row.get("source_round_id")
            or source_row.get("campaign_run_id") != row.get("source_run_id")
            or not isinstance(trial, dict) or trial.get("accepted_success") is not True or trial.get("outcome") != "success"
            or trial.get("artifact_sha256") != row.get("source_episode_sha256")
            or trial.get("campaign_round_id") != row.get("source_round_id") or trial.get("campaign_run_id") != row.get("source_run_id")
        ): raise SystemExit("fresh source report/matrix row is not authenticated")
        episode = Path(row.get("source_episode_path", "")); snapshot = Path(row.get("restore_snapshot", "")); receipt = Path(row.get("source_receipt_path", ""))
        if (not episode.is_absolute() or episode.is_symlink() or not episode.is_file()
                or not snapshot.is_absolute() or snapshot.is_symlink() or not snapshot.is_file()
                or not receipt.is_absolute() or receipt.is_symlink() or not receipt.is_file()): raise SystemExit("fresh source episode, snapshot, or receipt is missing")
        root = Path(row.get("source_episode_root", "")); attempt = row.get("parent_episode_id")
        if not root.is_absolute() or root.is_symlink() or not root.is_dir() or not isinstance(attempt, str): raise SystemExit("fresh source artifact root is missing")
        reset = root / "raw" / attempt / "snapshots" / "reset.json"
        annotations = root / "raw" / attempt / "annotations.jsonl"
        expected_paths = {"episode": root / "raw" / attempt / "episode.json", "reset": reset, "annotations": annotations, "h16": snapshot}
        manifest = root / "SHA256SUMS.json"
        try: manifest_rows = json.loads(manifest.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
        except (OSError, ValueError) as error: raise SystemExit(f"fresh source checksum manifest is malformed: {error}")
        if episode != expected_paths["episode"] or not isinstance(manifest_rows, dict): raise SystemExit("fresh source episode path is not canonical")
        checked = {}
        for name, item_path in expected_paths.items():
            relative = item_path.relative_to(root).as_posix(); record = manifest_rows.get(relative)
            if (item_path.is_symlink() or not item_path.is_file() or not isinstance(record, dict)
                    or set(record) != {"sha256", "size"} or record.get("sha256") != __import__("hashlib").sha256(item_path.read_bytes()).hexdigest()
                    or record.get("size") != item_path.stat().st_size): raise SystemExit("fresh source checksum manifest does not bind required evidence")
            checked[name] = record["sha256"]
        entries = []
        for item_path in sorted(root.rglob("*")):
            if item_path.is_symlink(): raise SystemExit("fresh source artifact contains a symlink")
            if item_path.is_file() and item_path.name != "SHA256SUMS.json":
                entries.append({"relative_path": item_path.relative_to(root).as_posix(), "sha256": __import__("hashlib").sha256(item_path.read_bytes()).hexdigest(), "byte_size": item_path.stat().st_size})
        artifact = __import__("hashlib").sha256((json.dumps(entries, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
        if (artifact != row.get("source_episode_sha256") or checked["reset"] != row.get("source_reset_sha256")
                or checked["annotations"] != row.get("source_annotations_sha256") or checked["h16"] != row.get("source_continuation_snapshot_sha256")):
            raise SystemExit("fresh source artifact hashes are not authenticated")
        if (__import__("hashlib").sha256(snapshot.read_bytes()).hexdigest() != row.get("restore_snapshot_sha256")
                or row.get("restore_snapshot_sha256") != row.get("source_continuation_snapshot_sha256")):
            raise SystemExit("fresh source continuation snapshot binding is invalid")
        try:
            episode_json = json.loads(episode.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
            snapshot_json = json.loads(snapshot.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
        except ValueError as error: raise SystemExit(f"fresh source episode or snapshot is malformed: {error}")
        try: receipt_json = json.loads(receipt.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
        except ValueError as error: raise SystemExit(f"fresh source receipt is malformed: {error}")
        identity = episode_json.get("identity") if isinstance(episode_json, dict) else None
        state = snapshot_json.get("robot_position") if isinstance(snapshot_json, dict) else None
        if (not isinstance(state, list) or len(state) != 12
                or any(type(value) not in (int, float) for value in state)):
            raise SystemExit("fresh source continuation state is invalid")
        state_fingerprint = __import__("hashlib").sha256(json.dumps({"category": row.get("category"), "garment": row.get("garment"), "state_rounding": "fixed_6dp", "state": ["0.000000" if float(value) == 0 else format(float(value), ".6f") for value in state]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (not isinstance(identity, dict) or episode_json.get("randomization") != {"strategy": "canonical"}
            or identity.get("campaign_round_id") != row.get("source_round_id") or identity.get("campaign_run_id") != row.get("source_run_id")
            or snapshot_json.get("randomization") != {"strategy": "canonical", "continuation_step": 16}
            or __import__("hashlib").sha256(receipt.read_bytes()).hexdigest() != row.get("source_receipt_sha256")
            or not isinstance(receipt_json, dict) or receipt_json.get("readback_verified") is not True
            or receipt_json.get("round_id") != row.get("source_round_id") or receipt_json.get("run_id") != row.get("source_run_id")
            or receipt_json.get("episode_sha256") != row.get("source_episode_sha256")
            or receipt_json.get("remote_prefix") != row.get("source_remote_prefix")
            or receipt_json.get("immutable_revision") != row.get("source_immutable_revision")
            or state_fingerprint != row.get("source_state_fingerprint")): raise SystemExit("fresh source canonical replay boundary is invalid")
PY

exec env \
  LEHOME_POLICY_SHA256="${ORIGINAL_12K_POLICY_SHA256}" \
  LEHOME_POLICY_REPO="${ORIGINAL_12K_POLICY_REPO}" \
  LEHOME_POLICY_REVISION="${ORIGINAL_12K_POLICY_REVISION}" \
  LEHOME_POLICY_STEP="12000" \
  LEHOME_POLICY_ARTIFACT_SHA256="${ORIGINAL_12K_POLICY_ARTIFACT_SHA256}" \
  LEHOME_CHECKPOINT_DIR="${ORIGINAL_12K_CHECKPOINT}" \
  LEHOME_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  LEHOME_ATTEMPT_MATRIX="${MATRIX}" \
  LEHOME_ATTEMPT_MATRIX_SHA256="${EXPECTED_SHA256}" \
  LEHOME_WORKER_COUNT="${WORKER_COUNT}" \
  LEHOME_SIMULATOR_DEVICE="cpu" \
  LEHOME_SUCCESS_REPLAY_CAMPAIGN="1" \
  LEHOME_FRESH_SOURCE_REPORTS_JSON="${FRESH_SOURCE_REPORTS}" \
  LEHOME_FRESH_SOURCE_MATRICES_JSON="${FRESH_SOURCE_MATRICES}" \
  LEHOME_MAX_ATTEMPTS="${MAX_ATTEMPTS}" \
  LEHOME_MAX_WORKER_RESTARTS="${MAX_WORKER_RESTARTS}" \
  LEHOME_TARGET_ACCEPTED="${TARGET_ACCEPTED}" \
  LEHOME_ROUND_ID="${ROUND_ID}" \
  bash "${BASE_CAMPAIGN}"
