from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "rollout_appliance" / "run_success_replay_campaign.sh"
BASE_CAMPAIGN = REPO_ROOT / "rollout_appliance" / "run_12k_campaign.sh"


def _matrix() -> list[dict[str, object]]:
    categories = ("top_long", "top_short", "pant_long", "pant_short")
    return [
        {
            "attempt_id": f"replay-{category}-{index}", "trial_id": f"replay-{category}-{index}",
            "garment": f"{category}-garment", "garment_name": f"{category}-garment",
            "category": category, "release_stage": "seen", "difficulty": "randomized",
            "seed": 50_000 + category_index * 50 + index,
            "strategy": "mild_geometry" if index % 2 == 0 else "strong_geometry",
            "restore_snapshot": "/verified/continuations/000016.json", "restore_snapshot_sha256": "a" * 64,
            "restore_snapshot_cloth_frame": "usd_local_points_v1",
            "restore_snapshot_step": 16,
            "parent_episode_id": f"parent-{category}", "lineage_id": f"parent-{category}",
            "replay_kind": "verified_success_early_snapshot_v1",
        }
        for category_index, category in enumerate(categories)
        for index in range(50)
    ]


def _fresh_visual_only_matrix() -> list[dict[str, object]]:
    return [
        {
            **_matrix()[category_index * 50 + index % 50],
            "attempt_id": f"fresh-{category}-{index}",
            "trial_id": f"fresh-{category}-{index}",
            "seed": 90_000 + category_index * 100 + index,
            "strategy": "visual_only",
            "category_acceptance_cap": 50,
            "source_episode_sha256": "b" * 64,
            "source_reset_sha256": "c" * 64,
            "source_annotations_sha256": "d" * 64,
            "source_continuation_snapshot_sha256": "a" * 64,
            "source_state_fingerprint": "e" * 64,
            "source_report_sha256": "f" * 64,
            "source_matrix_sha256": "1" * 64,
            "source_receipt_sha256": "2" * 64,
            "source_remote_prefix": f"rollout-rounds/fresh-12k-source-20260827/parent-{category}",
            "source_immutable_revision": "a" * 40,
            "source_round_id": "fresh-12k-source-20260827",
            "source_run_id": "fresh-run-20260827-a",
        }
        for category_index, category in enumerate(("top_long", "top_short", "pant_long", "pant_short"))
        for index in range(100)
    ]


def _fresh_evidence_matrix(tmp_path: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Create actual report/matrix bytes bound to the fresh replay rows."""

    rows = _fresh_visual_only_matrix()
    source_rows = [
        {
            "attempt_id": f"parent-{category}",
            "trial_id": f"parent-{category}",
            "category": category,
            "garment_name": f"{category}-garment",
            "release_stage": "seen",
            "campaign_round_id": "fresh-12k-source-20260827",
            "campaign_run_id": "fresh-run-20260827-a",
        }
        for category in ("top_long", "top_short", "pant_long", "pant_short")
    ]
    source_matrix = tmp_path / "fresh-source-matrix.json"
    source_matrix_bytes = (json.dumps(source_rows, sort_keys=True, separators=(",", ":")) + "\n").encode()
    source_matrix.write_bytes(source_matrix_bytes)
    source_matrix_sha = hashlib.sha256(source_matrix_bytes).hexdigest()
    source_episodes: dict[str, Path] = {}
    source_snapshots: dict[str, Path] = {}
    source_receipts: dict[str, Path] = {}
    source_roots: dict[str, Path] = {}
    source_digests: dict[str, str] = {}
    for category in ("top_long", "top_short", "pant_long", "pant_short"):
        attempt = f"parent-{category}"
        root = tmp_path / f"accepted-{category}"
        raw = root / "raw" / attempt
        snapshots = raw / "snapshots"
        snapshots.mkdir(parents=True)
        episode = raw / "episode.json"
        episode.write_text(
            json.dumps(
                {
                    "identity": {
                        "campaign_round_id": "fresh-12k-source-20260827",
                        "campaign_run_id": "fresh-run-20260827-a",
                    },
                    "randomization": {"strategy": "canonical"},
                }
            ),
            encoding="utf-8",
        )
        reset = snapshots / "reset.json"
        reset.write_text(json.dumps({"randomization": {"strategy": "canonical"}}), encoding="utf-8")
        annotations = raw / "annotations.jsonl"
        annotations.write_text('{"step":0,"action":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"success":true}\n', encoding="utf-8")
        snapshot = snapshots / "continuations" / "000016.json"
        snapshot.parent.mkdir()
        snapshot.write_text(
            json.dumps({"robot_position": [0.0] * 12, "randomization": {"strategy": "canonical", "continuation_step": 16}}),
            encoding="utf-8",
        )
        checksums = {}
        for item in (episode, reset, annotations, snapshot):
            checksums[item.relative_to(root).as_posix()] = {"sha256": hashlib.sha256(item.read_bytes()).hexdigest(), "size": item.stat().st_size}
        (root / "SHA256SUMS.json").write_text(json.dumps(checksums), encoding="utf-8")
        entries = [
            {"relative_path": item.relative_to(root).as_posix(), "sha256": hashlib.sha256(item.read_bytes()).hexdigest(), "byte_size": item.stat().st_size}
            for item in sorted((episode, reset, annotations, snapshot))
        ]
        source_digests[category] = hashlib.sha256((json.dumps(entries, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
        source_episodes[category], source_snapshots[category], source_roots[category] = episode, snapshot, root
        receipt = tmp_path / f"parent-{category}.sync.json"
        receipt.write_text(
            json.dumps(
                {
                    "readback_verified": True,
                    "round_id": "fresh-12k-source-20260827",
                    "run_id": "fresh-run-20260827-a",
                    "episode_sha256": source_digests[category],
                    "remote_prefix": f"rollout-rounds/fresh-12k-source-20260827/parent-{category}",
                    "immutable_revision": "a" * 40,
                }
            ),
            encoding="utf-8",
        )
        source_receipts[category] = receipt
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "lehome_fresh_12k_success_source_report_v1",
        "campaign_kind": "fresh_12k_success_source_v1",
        "round_id": "fresh-12k-source-20260827",
        "run_id": "fresh-run-20260827-a",
        "matrix_sha256": source_matrix_sha,
        "identity": {
            "policy_repo": "ryanjin333/lehome-groot-n17-models",
            "policy_revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
            "policy_step": 12000,
            "policy_artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
        },
        "trials": [
            {
                "attempt_id": f"parent-{category}", "category": category,
                "garment_name": f"{category}-garment", "accepted_success": True,
                "outcome": "success", "simulator_device": "cpu", "cloth_device": "cpu",
                "safety_failure": False, "numerical_failure": False, "cloth_failure": False,
                "artifact_sha256": source_digests[category],
                "hub_sync_receipt_sha256": hashlib.sha256(source_receipts[category].read_bytes()).hexdigest(),
                "remote_prefix": f"rollout-rounds/fresh-12k-source-20260827/parent-{category}",
                "campaign_round_id": "fresh-12k-source-20260827",
                "campaign_run_id": "fresh-run-20260827-a",
            }
            for category in ("top_long", "top_short", "pant_long", "pant_short")
        ],
        "safety_failure": False,
    }
    body = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    report["report_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    source_report = tmp_path / "fresh-source-report.json"
    source_report_bytes = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
    source_report.write_bytes(source_report_bytes)
    source_report_sha = hashlib.sha256(source_report_bytes).hexdigest()
    for row in rows:
        category = str(row["category"])
        row["source_report_path"] = str(source_report)
        row["source_matrix_path"] = str(source_matrix)
        row["source_report_sha256"] = source_report_sha
        row["source_matrix_sha256"] = source_matrix_sha
        row["source_episode_path"] = str(source_episodes[category])
        row["source_episode_root"] = str(source_roots[category])
        row["restore_snapshot"] = str(source_snapshots[category])
        row["restore_snapshot_sha256"] = hashlib.sha256(source_snapshots[category].read_bytes()).hexdigest()
        row["source_continuation_snapshot_sha256"] = row["restore_snapshot_sha256"]
        row["source_receipt_path"] = str(source_receipts[category])
        row["source_receipt_sha256"] = hashlib.sha256(source_receipts[category].read_bytes()).hexdigest()
        row["source_episode_sha256"] = source_digests[category]
        row["source_reset_sha256"] = hashlib.sha256((source_roots[category] / "raw" / f"parent-{category}" / "snapshots" / "reset.json").read_bytes()).hexdigest()
        row["source_annotations_sha256"] = hashlib.sha256((source_roots[category] / "raw" / f"parent-{category}" / "annotations.jsonl").read_bytes()).hexdigest()
        row["source_state_fingerprint"] = hashlib.sha256(json.dumps({"category": category, "garment": f"{category}-garment", "state_rounding": "fixed_6dp", "state": ["0.000000"] * 12}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return rows, {
        "LEHOME_FRESH_SOURCE_REPORTS_JSON": json.dumps([{"path": str(source_report), "sha256": source_report_sha}]),
        "LEHOME_FRESH_SOURCE_MATRICES_JSON": json.dumps([{"path": str(source_matrix), "sha256": source_matrix_sha}]),
    }


def test_success_replay_wrapper_is_pinned_to_the_four_worker_original_12k_campaign() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "run_12k_campaign.sh" in text
    assert 'WORKER_COUNT="${LEHOME_WORKER_COUNT:-4}"' in text
    assert 'MAX_ATTEMPTS="${LEHOME_MAX_ATTEMPTS:-200}"' in text
    assert 'MAX_WORKER_RESTARTS="${LEHOME_MAX_WORKER_RESTARTS:-8}"' in text
    assert 'LEHOME_MAX_WORKER_RESTARTS="${MAX_WORKER_RESTARTS}"' in text
    assert "campaign-success-replay-12k-round-1" in text
    assert "success-replay-12k-round-1" in text
    assert "LEHOME_ATTEMPT_MATRIX_SHA256" in text


def test_success_replay_wrapper_fails_closed_for_missing_or_inconsistent_matrix_inputs(tmp_path: Path) -> None:
    missing = subprocess.run(["bash", str(WRAPPER)], capture_output=True, text=True, check=False)
    assert missing.returncode != 0
    assert "LEHOME_SUCCESS_REPLAY_MATRIX" in missing.stderr

    matrix = tmp_path / "matrix.json"
    encoded = (json.dumps(_matrix(), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    matrix.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    (tmp_path / "matrix.json.sha256").write_text(digest + "\n", encoding="ascii")
    environment = os.environ | {
        "LEHOME_SUCCESS_REPLAY_MATRIX": str(matrix),
        "LEHOME_SUCCESS_REPLAY_MATRIX_SHA256": "0" * 64,
        "LEHOME_WORKSPACE": str(tmp_path),
        "LEHOME_CAMPAIGN_ROOT": str(tmp_path / "campaign"),
        "LEHOME_VALIDATE_MATRIX_ONLY": "1",
    }
    inconsistent = subprocess.run(["bash", str(WRAPPER)], env=environment, capture_output=True, text=True, check=False)
    assert inconsistent.returncode != 0
    assert "mismatch" in inconsistent.stderr


def test_success_replay_wrapper_rejects_an_ambiguous_legacy_snapshot_frame(tmp_path: Path) -> None:
    rows = _matrix()
    rows[0].pop("restore_snapshot_cloth_frame")
    matrix = tmp_path / "matrix.json"
    encoded = (json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    matrix.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    (tmp_path / "matrix.json.sha256").write_text(digest + "\n", encoding="ascii")

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        env=os.environ | {
            "LEHOME_SUCCESS_REPLAY_MATRIX": str(matrix),
            "LEHOME_SUCCESS_REPLAY_MATRIX_SHA256": digest,
            "LEHOME_WORKSPACE": str(tmp_path),
            "LEHOME_CAMPAIGN_ROOT": str(tmp_path / "campaign"),
            "LEHOME_VALIDATE_MATRIX_ONLY": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "cloth frame" in result.stderr


def test_success_replay_wrapper_passes_task_ledger_safe_accepted_target_to_12k_appliance(
    tmp_path: Path,
) -> None:
    wrapper = tmp_path / "run_success_replay_campaign.sh"
    appliance = tmp_path / "run_12k_campaign.sh"
    shutil.copy2(WRAPPER, wrapper)
    appliance.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s:%s' \"${LEHOME_MAX_ATTEMPTS}\" \"${LEHOME_TARGET_ACCEPTED}\" > \"${LEHOME_CAPTURE}\"\n",
        encoding="utf-8",
    )
    appliance.chmod(0o755)
    matrix = tmp_path / "matrix.json"
    encoded = (json.dumps(_matrix(), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    matrix.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    (tmp_path / "matrix.json.sha256").write_text(digest + "\n", encoding="ascii")
    capture = tmp_path / "effective-env.txt"
    environment = os.environ | {
        "LEHOME_SUCCESS_REPLAY_MATRIX": str(matrix),
        "LEHOME_SUCCESS_REPLAY_MATRIX_SHA256": digest,
        "LEHOME_CAPTURE": str(capture),
    }

    default = subprocess.run(["bash", str(wrapper)], env=environment, capture_output=True, text=True, check=False)
    assert default.returncode == 0, default.stderr
    assert capture.read_text(encoding="utf-8") == "200:150"

    override = subprocess.run(
        ["bash", str(wrapper)],
        env=environment | {"LEHOME_TARGET_ACCEPTED": "123"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert override.returncode == 0, override.stderr
    assert capture.read_text(encoding="utf-8") == "200:123"

    invalid = subprocess.run(
        ["bash", str(wrapper)],
        env=environment | {"LEHOME_TARGET_ACCEPTED": "151"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode != 0
    assert "1..150" in invalid.stderr


def test_success_replay_wrapper_runs_an_exact_capped_matrix_with_cpu_cloth(tmp_path: Path) -> None:
    wrapper = tmp_path / "run_success_replay_campaign.sh"
    appliance = tmp_path / "run_12k_campaign.sh"
    shutil.copy2(WRAPPER, wrapper)
    appliance.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "printf '%s:%s:%s:%s' \"${LEHOME_MAX_ATTEMPTS}\" \"${LEHOME_TARGET_ACCEPTED}\" "
        "\"${LEHOME_SIMULATOR_DEVICE}\" \"${LEHOME_SUCCESS_REPLAY_CAMPAIGN}\" > \"${LEHOME_CAPTURE}\"\n",
        encoding="utf-8",
    )
    appliance.chmod(0o755)
    rows = _matrix()[:40]
    caps = {"top_long": 7, "top_short": 3, "pant_long": 9, "pant_short": 0}
    # Make this a deliberately asymmetric exact-cap matrix.
    rows = (
        [dict(_matrix()[index], category_acceptance_cap=7) for index in range(12)]
        + [dict(_matrix()[50 + index], category_acceptance_cap=3) for index in range(4)]
        + [dict(_matrix()[100 + index], category_acceptance_cap=9) for index in range(24)]
    )
    matrix = tmp_path / "matrix.json"
    encoded = (json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    matrix.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    (tmp_path / "matrix.json.sha256").write_text(digest + "\n", encoding="ascii")
    capture = tmp_path / "effective-env.txt"

    result = subprocess.run(
        ["bash", str(wrapper)],
        env=os.environ | {
            "LEHOME_SUCCESS_REPLAY_MATRIX": str(matrix),
            "LEHOME_SUCCESS_REPLAY_MATRIX_SHA256": digest,
            "LEHOME_MAX_ATTEMPTS": "40",
            "LEHOME_TARGET_ACCEPTED": str(sum(caps.values())),
            "LEHOME_CAPTURE": str(capture),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8") == "40:19:cpu:1"


def test_success_replay_wrapper_admits_200_only_for_exact_fresh_visual_only_tuple(
    tmp_path: Path,
) -> None:
    wrapper = tmp_path / "run_success_replay_campaign.sh"
    appliance = tmp_path / "run_12k_campaign.sh"
    shutil.copy2(WRAPPER, wrapper)
    appliance.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "printf '%s:%s:%s:%s:%s' \"${LEHOME_MAX_ATTEMPTS}\" \"${LEHOME_TARGET_ACCEPTED}\" "
        "\"${LEHOME_SIMULATOR_DEVICE}\" \"${LEHOME_POLICY_REPO}\" \"${LEHOME_POLICY_REVISION}\" > \"${LEHOME_CAPTURE}\"\n",
        encoding="utf-8",
    )
    appliance.chmod(0o755)
    rows, source_environment = _fresh_evidence_matrix(tmp_path)
    matrix = tmp_path / "fresh-matrix.json"
    encoded = (json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    matrix.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    (tmp_path / "fresh-matrix.json.sha256").write_text(digest + "\n", encoding="ascii")
    capture = tmp_path / "effective-env.txt"
    environment = os.environ | {
        "LEHOME_SUCCESS_REPLAY_MATRIX": str(matrix),
        "LEHOME_SUCCESS_REPLAY_MATRIX_SHA256": digest,
        "LEHOME_MAX_ATTEMPTS": "400",
        "LEHOME_TARGET_ACCEPTED": "200",
        "LEHOME_CAPTURE": str(capture),
        **source_environment,
    }

    accepted = subprocess.run(["bash", str(wrapper)], env=environment, capture_output=True, text=True, check=False)
    assert accepted.returncode == 0, accepted.stderr
    assert capture.read_text(encoding="utf-8") == (
        "400:200:cpu:ryanjin333/lehome-groot-n17-models:30ac1a84da67b099e115ad147bcd61e9d60046d3"
    )

    base_only = subprocess.run(
        ["bash", str(BASE_CAMPAIGN)],
        env=environment | {
            "LEHOME_ATTEMPT_MATRIX": str(matrix),
            "LEHOME_ATTEMPT_MATRIX_SHA256": digest,
            "LEHOME_WORKER_COUNT": "4",
            "LEHOME_ENABLE_HF_UPLOAD": "1",
            "LEHOME_SIMULATOR_DEVICE": "cpu",
            "LEHOME_SUCCESS_REPLAY_CAMPAIGN": "1",
            "LEHOME_VALIDATE_MATRIX_ONLY": "1",
            "LEHOME_CAMPAIGN_ROOT": str(tmp_path / "base-campaign"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert base_only.returncode == 0, base_only.stderr

    report = Path(json.loads(source_environment["LEHOME_FRESH_SOURCE_REPORTS_JSON"])[0]["path"])
    original_report = report.read_bytes()
    report.write_text("{}", encoding="utf-8")
    tampered = subprocess.run(
        ["bash", str(wrapper)], env=environment, capture_output=True, text=True, check=False
    )
    assert tampered.returncode != 0
    assert "source report" in tampered.stderr
    direct_tampered = subprocess.run(
        ["bash", str(BASE_CAMPAIGN)],
        env=environment | {
            "LEHOME_ATTEMPT_MATRIX": str(matrix),
            "LEHOME_ATTEMPT_MATRIX_SHA256": digest,
            "LEHOME_WORKER_COUNT": "4",
            "LEHOME_ENABLE_HF_UPLOAD": "1",
            "LEHOME_SIMULATOR_DEVICE": "cpu",
            "LEHOME_SUCCESS_REPLAY_CAMPAIGN": "1",
            "LEHOME_VALIDATE_MATRIX_ONLY": "1",
            "LEHOME_CAMPAIGN_ROOT": str(tmp_path / "base-campaign-tampered"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert direct_tampered.returncode != 0
    assert "source report" in direct_tampered.stderr
    report.write_bytes(original_report)
    source_matrix = Path(json.loads(source_environment["LEHOME_FRESH_SOURCE_MATRICES_JSON"])[0]["path"])
    source_matrix.write_text("[]", encoding="utf-8")
    matrix_tampered = subprocess.run(
        ["bash", str(wrapper)], env=environment, capture_output=True, text=True, check=False
    )
    assert matrix_tampered.returncode != 0
    assert "source matrix" in matrix_tampered.stderr

    rejected = subprocess.run(
        ["bash", str(wrapper)],
        env=environment | {"LEHOME_TARGET_ACCEPTED": "199"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "exact" in rejected.stderr
