from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from scripts.eval_groot_n17_public96 import (
    CATEGORIES,
    CHECKPOINT,
    CheckpointIdentityError,
    Public96ContractError,
    await_authenticated_policy_server_ready,
    build_stage_command,
    load_frozen_matrix,
    run,
    tree_sha256,
    validate_policy_server_startup_timeout,
    validate_checkpoint_identity,
    validate_output_path,
    verify_result,
    video_filename_for_key,
    _summary_from_episodes,
)
from scripts.groot_n17_public96_raw_checker import (
    RAW_CHECKER_OVERLAY_ID,
    overlay_sha256,
    raw_success_checker,
)


ROOT = Path(__file__).parents[1]
MATRIX = ROOT / "configs" / "eval_groot_n17_public96_reference.json"
MATRIX_SHA256 = ROOT / "configs" / "eval_groot_n17_public96_reference.json.sha256"


def _identity_receipt(policy_root: Path) -> dict[str, object]:
    return {
        "kind": "lehome_groot_n17_checkpoint_identity_v1",
        "repository": "ryanjin333/lehome-groot-n17-models",
        "revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
        "subpath": "policies/step-12000",
        "step": 12000,
        "artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
        "runtime_policy_sha256": "e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa",
        "cache_path": str(policy_root.resolve()),
        "cache_tree_sha256": tree_sha256(policy_root),
    }


def _policy_root_with_canonical_artifact(tmp_path: Path) -> Path:
    """A tiny cache shape is deliberately not a valid original-12K artifact."""
    policy_root = tmp_path / "policy"
    policy_root.mkdir()
    (policy_root / "config.json").write_text("{}", encoding="utf-8")
    return policy_root


def _runtime_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, output_name: str) -> tuple[SimpleNamespace, Path, dict[str, object]]:
    """Build CPU-safe public96 runner inputs without importing a policy runtime."""
    import scripts.eval_groot_n17_public96 as evaluator

    policy = _policy_root_with_canonical_artifact(tmp_path)
    identity = tmp_path / "identity.json"
    identity_value = _identity_receipt(policy)
    identity.write_text(json.dumps(identity_value), encoding="utf-8")
    release = tmp_path / "assets" / "Release"
    for prefix in ("Top_Long", "Top_Short", "Pant_Long", "Pant_Short"):
        directory = release / prefix
        directory.mkdir(parents=True)
        directory.joinpath(f"{prefix}.txt").write_text(
            "\n".join([f"{prefix}_Seen_{index}" for index in range(10)] + [f"{prefix}_Unseen_{index}" for index in range(2)]),
            encoding="utf-8",
        )
    monkeypatch.setattr(evaluator, "canonical_policy_artifact_sha256", lambda _: CHECKPOINT["artifact_sha256"])
    monkeypatch.setenv("LEHOME_GROOT_N17_PUBLIC96_POLICY_TOKEN", "x" * 32)
    output_root = tmp_path / output_name
    return (
        SimpleNamespace(
            matrix=MATRIX,
            matrix_sha256=MATRIX_SHA256,
            policy_path=policy,
            checkpoint_identity_receipt=identity,
            asset_root=tmp_path / "assets",
            output_root=output_root,
            policy_server_port=9117,
            policy_server_token_env="LEHOME_GROOT_N17_PUBLIC96_POLICY_TOKEN",
            dry_run=False,
        ),
        output_root,
        identity_value,
    )


def _external_readiness_payload(policy_root: Path) -> dict[str, object]:
    return {
        "kind": "lehome_groot_n17_public96_policy_server_readiness_v1",
        "artifact_sha256": CHECKPOINT["artifact_sha256"],
        "runtime_policy_sha256": CHECKPOINT["runtime_policy_sha256"],
        "model_path": str(policy_root.resolve()),
        "device": "cuda:0",
        "adapter": "nvidia_gr00t_policy_server_public96_v1",
        "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()},
    }


def _write_external_readiness_receipt(path: Path, policy_root: Path) -> Path:
    path.write_text(json.dumps(_external_readiness_payload(policy_root), indent=1), encoding="utf-8")
    return path


def _assert_pre_stage_invalid_evidence(output_root: Path, identity: dict[str, object], reason: str) -> None:
    stages = load_frozen_matrix(MATRIX, MATRIX_SHA256)
    result_path = output_root / "result.json"
    receipt_path = output_root / "verifier-receipt.json"
    server_log = output_root / "policy-server.log"
    assert result_path.is_file() and receipt_path.is_file() and server_log.is_file()

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["kind"] == "lehome_groot_n17_public96_result_v1"
    assert result["matrix_sha256"] == hashlib.sha256(MATRIX.read_bytes()).hexdigest()
    assert result["checkpoint"] == identity
    assert result["raw_checker_overlay"] == {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()}
    assert result["status"] == "invalid"
    assert result["failure_reason"] == reason
    assert result["summary"] == _summary_from_episodes(result["episodes"], assigned_episodes=96, status="invalid")
    assert result["summary"]["overall"]["scored_episodes"] == 0
    assert result["summary"]["overall"]["success_rate"] is None
    assert result["publication"] == {"status": "not_attempted", "vm_stop": "not_attempted"}
    assert result["invalid_stages"] == [{"stage_id": stage.stage_id, "reason": reason} for stage in stages]
    assert result["episodes"] == [
        {
            "stage_id": stage.stage_id,
            "category": stage.category,
            "garment_name": stage.garment_name,
            "release_stage": stage.release_stage,
            "seed": stage.seed,
            "episode_index": episode_index,
            "outcome": "infrastructure_invalid",
            "success": False,
            "invalid_reason": reason,
            "artifacts": {},
        }
        for stage in stages
        for episode_index in stage.episode_indices
    ]
    assert not any((output_root / stage.stage_id).exists() for stage in stages)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["kind"] == "lehome_groot_n17_public96_verifier_receipt_v1"
    assert receipt["status"] == "invalid"
    assert receipt["failure_reason"] == reason
    assert receipt["invalid_stages"] == result["invalid_stages"]
    assert receipt["matrix_sha256"] == result["matrix_sha256"]
    assert receipt["checkpoint"] == identity
    assert receipt["raw_checker_overlay"] == result["raw_checker_overlay"]
    assert receipt["summary"] == result["summary"]
    assert receipt["publication"] == {"status": "not_attempted", "vm_stop": "not_attempted"}
    assert receipt["result"] == {"relative_path": "result.json", "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()}
    assert receipt["policy_server_log"] == {"relative_path": "policy-server.log", "sha256": hashlib.sha256(server_log.read_bytes()).hexdigest()}


def _result_identity(policy_root: Path) -> dict[str, object]:
    return {"kind": "lehome_groot_n17_checkpoint_identity_v1", **CHECKPOINT, "cache_path": str(policy_root.resolve()), "cache_tree_sha256": tree_sha256(policy_root)}


def _artifact(path: Path, root: Path) -> dict[str, str]:
    return {"relative_path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _outcomes(stages: tuple[object, ...], root: Path, policy_root: Path) -> list[dict[str, object]]:
    outcomes: list[dict[str, object]] = []
    root.mkdir(parents=True, exist_ok=True)
    readiness = root / "policy-server-readiness.json"
    readiness_payload = {
        "kind": "lehome_groot_n17_public96_policy_server_readiness_v1",
        "artifact_sha256": CHECKPOINT["artifact_sha256"], "runtime_policy_sha256": CHECKPOINT["runtime_policy_sha256"],
        "model_path": str(policy_root.resolve()), "device": "cuda:0",
        "adapter": "nvidia_gr00t_policy_server_public96_v1",
        "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()},
    }
    readiness.write_text(json.dumps(readiness_payload, sort_keys=True), encoding="utf-8")
    for stage in stages:
        stage_root = root / stage.stage_id
        stage_root.mkdir(parents=True, exist_ok=True)
        log = stage_root / "stage.log"; log.write_text("complete\n", encoding="utf-8")
        stage_entries = []
        for episode_index in stage.episode_indices:
            videos = {}
            directory = stage_root / "videos" / ("success" if episode_index == 1 else "failure")
            directory.mkdir(parents=True, exist_ok=True)
            for camera in ("top_rgb", "left_rgb", "right_rgb"):
                video = directory / f"episode{episode_index - 1}_observation_images_{camera}.mp4"
                video.write_bytes(f"{stage.stage_id}:{episode_index}:{camera}".encode())
                videos[camera] = {"relative_path": video.relative_to(root).as_posix(), "sha256": hashlib.sha256(video.read_bytes()).hexdigest()}
            stage_entries.append((episode_index, videos))
        receipt = stage_root / "stage-receipt.json"
        command = build_stage_command(stage, repo_root=ROOT, policy_path=policy_root, output_root=root, policy_server_port=9117, token_env="LEHOME_GROOT_N17_PUBLIC96_POLICY_TOKEN")
        receipt.write_text(json.dumps({
            "kind": "lehome_groot_n17_public96_stage_receipt_v1", "schema_version": 1,
            "stage": {"stage_id": stage.stage_id, "category": stage.category, "garment_name": stage.garment_name, "release_stage": stage.release_stage, "seed": stage.seed, "episode_indices": [1, 2]},
            "command": command,
            "child_completion_sentinel": {"raw_checker_overlay": {"overlay_id": RAW_CHECKER_OVERLAY_ID, "overlay_sha256": overlay_sha256()}, "runtime_policy_sha256": CHECKPOINT["runtime_policy_sha256"]},
            "log": _artifact(log, root),
            "episodes": [{"episode_index": episode_index, "videos": videos} for episode_index, videos in stage_entries],
            "policy_server_readiness": {"artifact": _artifact(readiness, root), "binding": readiness_payload},
        }, sort_keys=True), encoding="utf-8")
        for episode_index, videos in stage_entries:
            outcomes.append(
                {
                    "stage_id": stage.stage_id,
                    "category": stage.category,
                    "garment_name": stage.garment_name,
                    "release_stage": stage.release_stage,
                    "seed": stage.seed,
                    "episode_index": episode_index,
                    "outcome": "success" if episode_index == 1 else "policy_failure",
                    "success": episode_index == 1,
                    "return": 1.0 if episode_index == 1 else 0.0,
                    "length": 600,
                    "artifacts": {
                        "log": {"relative_path": log.relative_to(root).as_posix(), "sha256": hashlib.sha256(log.read_bytes()).hexdigest()},
                        "videos": videos,
                        "receipt": {"relative_path": receipt.relative_to(root).as_posix(), "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()},
                    },
                }
            )
    return outcomes


def _valid_result(stages: tuple[object, ...], root: Path, policy_root: Path) -> dict[str, object]:
    episodes = _outcomes(stages, root, policy_root)
    return {
        "kind": "lehome_groot_n17_public96_result_v1", "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        "checkpoint": _result_identity(policy_root), "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()},
        "episodes": episodes, "invalid_stages": [],
        "publication": {"status": "not_attempted", "vm_stop": "not_attempted"}, "status": "valid",
        "summary": _summary_from_episodes(episodes),
    }


def test_frozen_matrix_has_public_96_order_and_two_sequential_episodes() -> None:
    stages = load_frozen_matrix(MATRIX, MATRIX_SHA256)

    assert len(stages) == 48
    assert sum(len(stage.episode_indices) for stage in stages) == 96
    assert [stage.category for stage in stages] == [category for category in ("top_long", "top_short", "pant_long", "pant_short") for _ in range(12)]
    assert [(stage.garment_name, stage.release_stage) for stage in stages[:12]] == [
        (f"Top_Long_Seen_{index}", "seen") for index in range(10)
    ] + [(f"Top_Long_Unseen_{index}", "unseen") for index in range(2)]
    assert all(stage.seed == 42 and stage.episode_indices == (1, 2) for stage in stages)


def test_matrix_rejects_mutation_duplicate_and_missing_stage(tmp_path: Path) -> None:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    path = tmp_path / "matrix.json"
    digest = tmp_path / "matrix.json.sha256"
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest.write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "  matrix.json\n", encoding="ascii")

    payload["stages"][1]["stage_id"] = payload["stages"][0]["stage_id"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest.write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "  matrix.json\n", encoding="ascii")
    with pytest.raises(Public96ContractError, match="unique"):
        load_frozen_matrix(path, digest)

    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    payload["stages"].pop()
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest.write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "  matrix.json\n", encoding="ascii")
    with pytest.raises(Public96ContractError, match="48"):
        load_frozen_matrix(path, digest)

    path.write_text(MATRIX.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(Public96ContractError, match="digest"):
        load_frozen_matrix(path, MATRIX_SHA256)


def test_checkpoint_identity_rejects_runtime_digest_mismatch(tmp_path: Path) -> None:
    policy_root = _policy_root_with_canonical_artifact(tmp_path)
    receipt = _identity_receipt(policy_root)

    with pytest.raises(CheckpointIdentityError, match="artifact"):
        validate_checkpoint_identity(receipt, policy_root)


def test_checkpoint_identity_rejects_caller_self_attested_one_file_cache(tmp_path: Path) -> None:
    policy_root = _policy_root_with_canonical_artifact(tmp_path)
    with pytest.raises(CheckpointIdentityError, match="artifact"):
        validate_checkpoint_identity(_identity_receipt(policy_root), policy_root)


def test_safe_output_path_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "result"
    root.mkdir()
    safe = root / "stage-001" / "stage.log"
    safe.parent.mkdir(); safe.write_text("log", encoding="utf-8")
    outside = tmp_path / "outside.json"; outside.write_text("outside", encoding="utf-8")

    assert validate_output_path(root, safe) == safe
    with pytest.raises(Public96ContractError, match="escapes"):
        validate_output_path(root, outside)


def test_result_verification_counts_clean_failures_and_categories(tmp_path: Path) -> None:
    stages = load_frozen_matrix(MATRIX, MATRIX_SHA256)
    policy_root = _policy_root_with_canonical_artifact(tmp_path)
    result = _valid_result(stages, tmp_path, policy_root)

    verified = verify_result(result, stages=stages, matrix_sha256=result["matrix_sha256"], output_root=tmp_path, policy_artifact_verifier=lambda _: CHECKPOINT["artifact_sha256"])
    assert verified["overall"] == {"episodes": 96, "successes": 48, "policy_failures": 48, "infrastructure_invalid": 0, "fidelity_invalid": 0, "invalid_episodes": 0, "scored_episodes": 96, "success_rate": 0.5}
    assert verified["categories"]["top_long"] == {"episodes": 24, "successes": 12, "policy_failures": 12, "infrastructure_invalid": 0, "fidelity_invalid": 0, "invalid_episodes": 0, "scored_episodes": 24, "success_rate": 0.5}


def test_invalid_summary_counts_partial_rows_and_has_no_startup_score() -> None:
    episodes = [
        {"category": "top_long", "outcome": "success"},
        {"category": "top_long", "outcome": "policy_failure"},
        {"category": "top_long", "outcome": "infrastructure_invalid"},
        {"category": "top_long", "outcome": "fidelity_invalid"},
    ]
    summary = _summary_from_episodes(episodes, assigned_episodes=96, status="invalid")
    assert summary["overall"] == {"episodes": 4, "successes": 1, "policy_failures": 1, "infrastructure_invalid": 1, "fidelity_invalid": 1, "invalid_episodes": 2, "scored_episodes": 2, "success_rate": None}
    assert summary["categories"]["top_long"] == summary["overall"]
    assert all(summary["categories"][category]["success_rate"] is None for category in CATEGORIES)
    assert summary["assigned_episodes"] == 96 and summary["status"] == "invalid"


def test_result_verification_rejects_fabricated_policy_cache_path(tmp_path: Path) -> None:
    stages = load_frozen_matrix(MATRIX, MATRIX_SHA256)
    policy_root = _policy_root_with_canonical_artifact(tmp_path)
    result = _valid_result(stages, tmp_path, policy_root)
    result["checkpoint"] = {**result["checkpoint"], "cache_path": "/verified/cache", "cache_tree_sha256": "a" * 64}

    with pytest.raises(Public96ContractError, match="cache"):
        verify_result(result, stages=stages, matrix_sha256=result["matrix_sha256"], output_root=tmp_path)


def test_result_verification_rejects_invalid_or_missing_episode(tmp_path: Path) -> None:
    stages = load_frozen_matrix(MATRIX, MATRIX_SHA256)
    policy_root = _policy_root_with_canonical_artifact(tmp_path)
    result = _valid_result(stages, tmp_path, policy_root)
    result["episodes"][0]["outcome"] = "infrastructure_invalid"
    with pytest.raises(Public96ContractError, match="invalid"):
        verify_result(result, stages=stages, matrix_sha256=result["matrix_sha256"], output_root=tmp_path, policy_artifact_verifier=lambda _: CHECKPOINT["artifact_sha256"])

    result["episodes"] = result["episodes"][:-1]
    with pytest.raises(Public96ContractError, match="96"):
        verify_result(result, stages=stages, matrix_sha256=result["matrix_sha256"], output_root=tmp_path, policy_artifact_verifier=lambda _: CHECKPOINT["artifact_sha256"])


def test_result_verification_rejects_mutated_real_camera_artifact(tmp_path: Path) -> None:
    stages = load_frozen_matrix(MATRIX, MATRIX_SHA256)
    policy_root = _policy_root_with_canonical_artifact(tmp_path)
    result = _valid_result(stages, tmp_path, policy_root)
    first_video = tmp_path / result["episodes"][0]["artifacts"]["videos"]["top_rgb"]["relative_path"]
    first_video.write_bytes(b"tampered")
    with pytest.raises(Public96ContractError, match="digest"):
        verify_result(result, stages=stages, matrix_sha256=result["matrix_sha256"], output_root=tmp_path, policy_artifact_verifier=lambda _: CHECKPOINT["artifact_sha256"])


def test_result_verification_rejects_zero_byte_camera_even_when_its_digest_is_rebound(tmp_path: Path) -> None:
    stages = load_frozen_matrix(MATRIX, MATRIX_SHA256)
    policy_root = _policy_root_with_canonical_artifact(tmp_path)
    result = _valid_result(stages, tmp_path, policy_root)
    descriptor = result["episodes"][0]["artifacts"]["videos"]["top_rgb"]
    video = tmp_path / descriptor["relative_path"]
    video.write_bytes(b"")
    descriptor["sha256"] = hashlib.sha256(b"").hexdigest()

    _rewrite_stage_receipt(
        result,
        tmp_path,
        lambda receipt: receipt["episodes"][0]["videos"]["top_rgb"].update(sha256=descriptor["sha256"]),
    )

    with pytest.raises(Public96ContractError, match="empty"):
        verify_result(result, stages=stages, matrix_sha256=result["matrix_sha256"], output_root=tmp_path, policy_artifact_verifier=lambda _: CHECKPOINT["artifact_sha256"])


def test_result_verification_calls_the_production_canonical_verifier_and_rejects_its_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.eval_groot_n17_public96 as evaluator

    stages = load_frozen_matrix(MATRIX, MATRIX_SHA256)
    policy_root = _policy_root_with_canonical_artifact(tmp_path)
    result = _valid_result(stages, tmp_path, policy_root)
    called: list[Path] = []

    def mismatched_canonical_verifier(path: Path) -> str:
        called.append(path)
        return "0" * 64

    monkeypatch.setattr(evaluator, "canonical_policy_artifact_sha256", mismatched_canonical_verifier)
    with pytest.raises(Public96ContractError, match="cache"):
        verify_result(result, stages=stages, matrix_sha256=result["matrix_sha256"], output_root=tmp_path)
    assert called == [policy_root.resolve()]


def _rewrite_stage_receipt(result: dict[str, object], root: Path, mutate) -> None:
    receipt_path = root / "top-long-seen-0" / "stage-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutate(receipt)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    descriptor = _artifact(receipt_path, root)
    for episode in result["episodes"]:
        if episode["stage_id"] == "top-long-seen-0":
            episode["artifacts"]["receipt"] = descriptor


@pytest.mark.parametrize("mutation", [
    lambda receipt: receipt["command"].append("--tampered"),
    lambda receipt: receipt["log"].update(sha256="0" * 64),
    lambda receipt: receipt["episodes"][0]["videos"]["top_rgb"].update(relative_path="top-long-seen-0/videos/failure/episode0_observation_images_top_rgb.mp4"),
    lambda receipt: receipt["child_completion_sentinel"].update(runtime_policy_sha256="0" * 64),
    lambda receipt: receipt["policy_server_readiness"]["binding"].update(device="cpu"),
    lambda receipt: receipt.update(unexpected=True),
    lambda receipt: receipt.pop("log"),
])
def test_result_verification_rejects_mutated_stage_receipt_contract(tmp_path: Path, mutation) -> None:
    stages = load_frozen_matrix(MATRIX, MATRIX_SHA256)
    policy_root = _policy_root_with_canonical_artifact(tmp_path)
    result = _valid_result(stages, tmp_path, policy_root)
    _rewrite_stage_receipt(result, tmp_path, mutation)

    with pytest.raises(Public96ContractError):
        verify_result(result, stages=stages, matrix_sha256=result["matrix_sha256"], output_root=tmp_path, policy_artifact_verifier=lambda _: CHECKPOINT["artifact_sha256"])


def test_result_verification_rejects_checkpoint_digest_and_summary_mutations(tmp_path: Path) -> None:
    stages = load_frozen_matrix(MATRIX, MATRIX_SHA256)
    policy_root = _policy_root_with_canonical_artifact(tmp_path)
    result = _valid_result(stages, tmp_path, policy_root)
    result["checkpoint"]["artifact_sha256"] = "0" * 64
    with pytest.raises(Public96ContractError):
        verify_result(result, stages=stages, matrix_sha256=result["matrix_sha256"], output_root=tmp_path, policy_artifact_verifier=lambda _: CHECKPOINT["artifact_sha256"])

    result = _valid_result(stages, tmp_path / "second", policy_root)
    result["summary"]["overall"]["successes"] = 47
    with pytest.raises(Public96ContractError, match="summary"):
        verify_result(result, stages=stages, matrix_sha256=result["matrix_sha256"], output_root=tmp_path / "second", policy_artifact_verifier=lambda _: CHECKPOINT["artifact_sha256"])


@pytest.mark.parametrize("mutation", ("escape", "missing", "symlink"))
def test_result_verification_rejects_unsafe_artifact_paths(tmp_path: Path, mutation: str) -> None:
    stages = load_frozen_matrix(MATRIX, MATRIX_SHA256)
    policy_root = _policy_root_with_canonical_artifact(tmp_path)
    result = _valid_result(stages, tmp_path, policy_root)
    descriptor = result["episodes"][0]["artifacts"]["videos"]["top_rgb"]
    video = tmp_path / descriptor["relative_path"]
    if mutation == "escape":
        outside = tmp_path.parent / "public96-outside.mp4"
        outside.write_bytes(video.read_bytes())
        descriptor["relative_path"] = "../public96-outside.mp4"
        descriptor["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    elif mutation == "missing":
        video.unlink()
    else:
        outside = tmp_path / "outside.mp4"
        outside.write_bytes(video.read_bytes())
        video.unlink(); video.symlink_to(outside)

    with pytest.raises(Public96ContractError):
        verify_result(result, stages=stages, matrix_sha256=result["matrix_sha256"], output_root=tmp_path, policy_artifact_verifier=lambda _: CHECKPOINT["artifact_sha256"])


def test_dry_run_command_selects_groot_server_not_lerobot(tmp_path: Path) -> None:
    stage = load_frozen_matrix(MATRIX, MATRIX_SHA256)[0]
    command = build_stage_command(
        stage,
        repo_root=ROOT,
        policy_path=tmp_path / "policy",
        output_root=tmp_path / "output",
        policy_server_port=9117,
        token_env="LEHOME_GROOT_N17_PUBLIC96_POLICY_TOKEN",
    )

    assert command[command.index("--policy_type") + 1] == "groot_server"
    assert "lerobot" not in command
    assert command[command.index("--num_episodes") + 1] == "2"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--policy_server_endpoint") + 1] == "tcp://127.0.0.1:9117"
    assert command[command.index("--public96_raw_checker_overlay") + 1] == RAW_CHECKER_OVERLAY_ID


def test_real_video_filename_contract_uses_saver_dot_replacement() -> None:
    assert video_filename_for_key(0, "observation.images.top_rgb") == "episode0_observation_images_top_rgb.mp4"
    assert video_filename_for_key(1, "observation.images.left_rgb") == "episode1_observation_images_left_rgb.mp4"
    with pytest.raises(Public96ContractError):
        video_filename_for_key(0, "observation.top_rgb")


def test_policy_server_startup_timeout_uses_a_large_model_safe_default() -> None:
    assert validate_policy_server_startup_timeout(180.0) == 180.0
    assert validate_policy_server_startup_timeout("180") == 180.0
    assert validate_policy_server_startup_timeout("180.0") == 180.0


@pytest.mark.parametrize("timeout", (29.0, 601.0, 0.0, True, float("nan"), float("inf"), "NaN", "inf", "180 seconds", ""))
def test_policy_server_startup_timeout_rejects_unsafe_values(timeout: object) -> None:
    with pytest.raises(Public96ContractError, match="startup timeout"):
        validate_policy_server_startup_timeout(timeout)


def test_parser_accepts_literal_numeric_policy_server_startup_timeout() -> None:
    import scripts.eval_groot_n17_public96 as evaluator

    args = evaluator._parser().parse_args([
        "--matrix", "/tmp/matrix.json", "--matrix-sha256", "/tmp/matrix.sha256",
        "--policy-path", "/tmp/policy", "--checkpoint-identity-receipt", "/tmp/identity.json",
        "--asset-root", "/tmp/assets", "--output-root", "/tmp/output",
        "--policy-server-startup-timeout", "180",
    ])

    assert args.policy_server_startup_timeout == 180.0


def test_parser_accepts_external_policy_server_readiness_receipt() -> None:
    import scripts.eval_groot_n17_public96 as evaluator

    receipt = Path("/tmp/external-policy-server-readiness.json")
    args = evaluator._parser().parse_args([
        "--matrix", "/tmp/matrix.json", "--matrix-sha256", "/tmp/matrix.sha256",
        "--policy-path", "/tmp/policy", "--checkpoint-identity-receipt", "/tmp/identity.json",
        "--asset-root", "/tmp/assets", "--output-root", "/tmp/output",
        "--external-policy-server-readiness-receipt", str(receipt),
    ])

    assert args.external_policy_server_readiness_receipt == receipt


@pytest.mark.parametrize("mode", ("relative", "missing", "symlink"))
def test_external_readiness_receipt_must_be_absolute_existing_and_not_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    import scripts.eval_groot_n17_public96 as evaluator

    args, _, _ = _runtime_inputs(tmp_path, monkeypatch, output_name=f"unsafe-external-{mode}")
    if mode == "relative":
        receipt = Path("relative-readiness.json")
    elif mode == "missing":
        receipt = tmp_path / "missing-readiness.json"
    else:
        target = _write_external_readiness_receipt(tmp_path / "readiness.json", args.policy_path)
        receipt = tmp_path / "readiness-link.json"
        receipt.symlink_to(target)
    args.external_policy_server_readiness_receipt = receipt
    monkeypatch.setattr(evaluator.subprocess, "Popen", lambda *_, **__: (_ for _ in ()).throw(AssertionError("external mode must not start a child server")))

    with pytest.raises(Public96ContractError, match="external policy server readiness receipt"):
        run(args)


def test_external_readiness_receipt_must_bind_the_resolved_policy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.eval_groot_n17_public96 as evaluator

    args, _, _ = _runtime_inputs(tmp_path, monkeypatch, output_name="external-bad-binding")
    receipt = _write_external_readiness_receipt(tmp_path / "readiness.json", args.policy_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["model_path"] = "/different/policy"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    args.external_policy_server_readiness_receipt = receipt
    monkeypatch.setattr(evaluator.subprocess, "Popen", lambda *_, **__: (_ for _ in ()).throw(AssertionError("external mode must not start a child server")))

    with pytest.raises(Public96ContractError, match="readiness does not bind"):
        run(args)


def test_external_readiness_receipt_rejects_a_path_swapped_to_a_symlink_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.eval_groot_n17_public96 as evaluator

    policy = tmp_path / "policy"
    policy.mkdir()
    source = _write_external_readiness_receipt(tmp_path / "readiness.json", policy)
    replacement = _write_external_readiness_receipt(tmp_path / "replacement.json", policy)
    replacement.write_text(json.dumps(_external_readiness_payload(policy), indent=4), encoding="utf-8")
    real_open = evaluator.os.open
    open_flags: list[int] = []

    def swap_then_open(path: str | bytes | int, flags: int, *args: object) -> int:
        open_flags.append(flags)
        if Path(path) == source:
            source.unlink()
            source.symlink_to(replacement)
        return real_open(path, flags, *args)

    monkeypatch.setattr(evaluator.os, "open", swap_then_open)

    with pytest.raises(Public96ContractError, match="external policy server readiness receipt"):
        evaluator._load_external_readiness_receipt(source, policy_root=policy)

    assert source.is_symlink()
    assert open_flags and open_flags[0] & getattr(evaluator.os, "O_NOFOLLOW", 0)


def test_external_dry_run_validates_and_copies_the_readiness_receipt_without_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.eval_groot_n17_public96 as evaluator

    args, output_root, _ = _runtime_inputs(tmp_path, monkeypatch, output_name="external-dry-run")
    source = _write_external_readiness_receipt(tmp_path / "external-readiness.json", args.policy_path)
    args.external_policy_server_readiness_receipt = source
    args.dry_run = True
    monkeypatch.setattr(evaluator.subprocess, "Popen", lambda *_, **__: (_ for _ in ()).throw(AssertionError("dry run must not start a child server")))
    monkeypatch.setattr(evaluator, "await_authenticated_policy_server_ready", lambda **_: (_ for _ in ()).throw(AssertionError("dry run must not ping")))

    result = run(args)

    assert result["policy_server_mode"] == "external"
    assert result["policy_server_command"] is None
    assert result["external_policy_server_readiness_receipt_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (output_root / "policy-server-readiness.json").read_bytes() == source.read_bytes()
    assert (output_root / "validation-only-receipt.json").is_file()


def _successful_external_stage(command: list[str]) -> SimpleNamespace:
    stage_root = Path(command[command.index("--video_dir") + 1]).parent
    output = []
    for episode_index, success in ((1, True), (2, False)):
        folder = "success" if success else "failure"
        for camera in ("top_rgb", "left_rgb", "right_rgb"):
            video = stage_root / "videos" / folder / f"episode{episode_index - 1}_observation_images_{camera}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"{episode_index}:{camera}".encode())
        output.append(f"Episode {episode_index}/2: Return={1.0 if success else 0.0:.2f}, Length=600, Success={success}")
    output.append("PUBLIC96_STAGE_COMPLETE " + json.dumps({"raw_checker_overlay": {"overlay_id": RAW_CHECKER_OVERLAY_ID, "overlay_sha256": overlay_sha256()}, "runtime_policy_sha256": CHECKPOINT["runtime_policy_sha256"]}, sort_keys=True))
    return SimpleNamespace(returncode=0, stdout="\n".join(output))


def test_external_mode_requires_authenticated_admission_and_reprobes_before_every_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.eval_groot_n17_public96 as evaluator

    args, output_root, _ = _runtime_inputs(tmp_path, monkeypatch, output_name="external-success")
    source = _write_external_readiness_receipt(tmp_path / "external-readiness.json", args.policy_path)
    args.external_policy_server_readiness_receipt = source
    probes: list[dict[str, object]] = []
    stage_commands: list[list[str]] = []

    monkeypatch.setattr(evaluator.subprocess, "Popen", lambda *_, **__: (_ for _ in ()).throw(AssertionError("external mode must not start a child server")))
    monkeypatch.setattr(evaluator, "await_authenticated_policy_server_ready", lambda **kwargs: probes.append(kwargs))

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        assert len(probes) == len(stage_commands) + 2
        stage_commands.append(command)
        return _successful_external_stage(command)

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

    result = run(args)

    assert len(stage_commands) == 48
    assert len(probes) == 49  # admission plus an immediate probe before every Isaac stage
    assert all(probe["process"] is None for probe in probes)
    assert result["status"] == "valid"
    log = (output_root / "policy-server.log").read_text(encoding="utf-8")
    assert "source_mode=external" in log
    assert hashlib.sha256(source.read_bytes()).hexdigest() in log
    assert "sidecar process log" not in log


def test_external_mid_run_probe_failure_stops_later_stages_and_emits_complete_invalid_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.eval_groot_n17_public96 as evaluator

    args, output_root, identity = _runtime_inputs(tmp_path, monkeypatch, output_name="external-probe-failure")
    args.external_policy_server_readiness_receipt = _write_external_readiness_receipt(tmp_path / "external-readiness.json", args.policy_path)
    probe_count = 0
    stage_commands: list[list[str]] = []

    monkeypatch.setattr(evaluator.subprocess, "Popen", lambda *_, **__: (_ for _ in ()).throw(AssertionError("external mode must not start a child server")))

    def fake_probe(**_: object) -> None:
        nonlocal probe_count
        probe_count += 1
        if probe_count == 3:
            raise Public96ContractError("external policy server did not pass token-bound readiness")

    monkeypatch.setattr(evaluator, "await_authenticated_policy_server_ready", fake_probe)

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        stage_commands.append(command)
        return _successful_external_stage(command)

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

    with pytest.raises(Public96ContractError, match="infrastructure/fidelity invalid"):
        run(args)

    assert len(stage_commands) == 1
    result = json.loads((output_root / "result.json").read_text(encoding="utf-8"))
    assert result["checkpoint"] == identity
    assert result["status"] == "invalid"
    assert len(result["episodes"]) == 96
    assert result["summary"]["overall"]["scored_episodes"] == 2
    assert result["summary"]["overall"]["infrastructure_invalid"] == 94
    assert result["summary"]["overall"]["success_rate"] is None
    assert len(result["invalid_stages"]) == 47


def test_authenticated_readiness_uses_a_real_token_checked_loopback_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    zmq = pytest.importorskip("zmq")
    if "torch" not in sys.modules:
        torch_stub = SimpleNamespace(Tensor=object)
        monkeypatch.setitem(sys.modules, "torch", torch_stub)
    from scripts.eval_policy.groot_policy import pack_policy_server_message, unpack_policy_server_message

    context = zmq.Context()
    server = context.socket(zmq.REP)
    server.linger = 0
    server.bind("tcp://127.0.0.1:*")
    endpoint = server.getsockopt_string(zmq.LAST_ENDPOINT)
    port = int(endpoint.rsplit(":", 1)[1])
    token = "t" * 48
    received: list[object] = []

    def serve_one_ping() -> None:
        try:
            message = unpack_policy_server_message(server.recv())
            received.append(message)
            assert message == {"endpoint": "ping", "data": {}, "api_token": token}
            server.send(pack_policy_server_message({"status": "ok", "message": "Server is running"}))
        finally:
            server.close()
            context.term()

    thread = threading.Thread(target=serve_one_ping)
    thread.start()
    try:
        await_authenticated_policy_server_ready(port=port, token=token, readiness_timeout=2.0, request_timeout=1.0)
    finally:
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert received == [{"endpoint": "ping", "data": {}, "api_token": token}]


def test_raw_checker_uses_untransformed_mesh_and_unscaled_thresholds() -> None:
    transformed = [[0.0, 0.0, 0.0]] * 6
    raw = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0], [300.0, 0.0, 0.0], [400.0, 0.0, 0.0], [500.0, 0.0, 0.0]]
    particle = SimpleNamespace(
        check_points=[0, 1, 2, 3, 4, 5],
        success_distance=[1000.0, 1000.0, 1000.0, 0.0, 0.0],
        init_scale=[0.45],
        get_current_mesh_points=lambda: (transformed, raw, None, None),
    )

    result = raw_success_checker(particle, "top-long-sleeve")

    assert result["success"] is False
    assert result["thresholds"] == particle.success_distance
    assert result["mesh_source"] == "raw_mesh_points"


def test_default_checker_source_remains_unchanged_and_overlay_is_isolated() -> None:
    default_checker = ROOT / "source" / "lehome" / "lehome" / "utils" / "success_checker_chanllege.py"
    source = default_checker.read_text(encoding="utf-8")

    assert "transformed_mesh_points, _, _, _ = particle_object.get_current_mesh_points()" in source
    assert "success_distance = [d * current_scale for d in raw_success_distance]" in source
    assert "raw_mesh_points" not in source


def test_shared_eval_parser_is_not_widened_for_public96() -> None:
    parser_path = ROOT / "scripts" / "utils" / "parser.py"
    spec = importlib.util.spec_from_file_location("public96_parser", parser_path)
    assert spec and spec.loader
    parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_module)

    with pytest.raises(SystemExit):
        parser_module.setup_eval_parser().parse_args(
        [
            "--policy_type", "groot_server",
            "--policy_server_endpoint", "tcp://127.0.0.1:9117",
            "--policy_server_token_env", "LEHOME_GROOT_N17_PUBLIC96_POLICY_TOKEN",
            "--policy_server_request_timeout", "600",
        ]
        )


def test_validation_only_emits_all_48_n17_stage_assignments_without_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.eval_groot_n17_public96 as evaluator
    policy_root = tmp_path / "policy"
    policy_root.mkdir()
    (policy_root / "config.json").write_text("{}", encoding="utf-8")
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(_identity_receipt(policy_root)), encoding="utf-8")
    output_root = tmp_path / "validation-only"
    monkeypatch.setattr(evaluator, "canonical_policy_artifact_sha256", lambda _: CHECKPOINT["artifact_sha256"])
    release = tmp_path / "assets" / "Release"
    for prefix in ("Top_Long", "Top_Short", "Pant_Long", "Pant_Short"):
        directory = release / prefix; directory.mkdir(parents=True)
        directory.joinpath(f"{prefix}.txt").write_text("\n".join([f"{prefix}_Seen_{index}" for index in range(10)] + [f"{prefix}_Unseen_{index}" for index in range(2)]), encoding="utf-8")

    result = run(SimpleNamespace(
        matrix=MATRIX, matrix_sha256=MATRIX_SHA256, policy_path=policy_root,
        checkpoint_identity_receipt=identity, asset_root=tmp_path / "assets",
        output_root=output_root, policy_server_port=9117,
        policy_server_token_env="LEHOME_GROOT_N17_PUBLIC96_POLICY_TOKEN", dry_run=True,
    ))

    assert result["kind"] == "lehome_groot_n17_public96_validation_v1"
    assert len(result["stage_commands"]) == 48
    assert (output_root / "validation-only-receipt.json").is_file()


def test_validation_only_rejects_missing_release_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.eval_groot_n17_public96 as evaluator
    policy = _policy_root_with_canonical_artifact(tmp_path); identity = tmp_path / "identity.json"; identity.write_text(json.dumps(_identity_receipt(policy)), encoding="utf-8")
    monkeypatch.setattr(evaluator, "canonical_policy_artifact_sha256", lambda _: CHECKPOINT["artifact_sha256"])
    with pytest.raises(Public96ContractError, match="Release"):
        run(SimpleNamespace(matrix=MATRIX, matrix_sha256=MATRIX_SHA256, policy_path=policy, checkpoint_identity_receipt=identity, asset_root=tmp_path / "missing", output_root=tmp_path / "out", policy_server_port=9117, policy_server_token_env="TOKEN", dry_run=True))


def test_synthetic_successful_run_writes_complete_result_and_verifier_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.eval_groot_n17_public96 as evaluator
    policy = _policy_root_with_canonical_artifact(tmp_path); identity = tmp_path / "identity.json"; identity.write_text(json.dumps(_identity_receipt(policy)), encoding="utf-8")
    release = tmp_path / "assets" / "Release"
    for prefix in ("Top_Long", "Top_Short", "Pant_Long", "Pant_Short"):
        directory = release / prefix; directory.mkdir(parents=True)
        directory.joinpath(f"{prefix}.txt").write_text("\n".join([f"{prefix}_Seen_{index}" for index in range(10)] + [f"{prefix}_Unseen_{index}" for index in range(2)]), encoding="utf-8")
    monkeypatch.setattr(evaluator, "canonical_policy_artifact_sha256", lambda _: CHECKPOINT["artifact_sha256"])

    class Server:
        def poll(self): return None
        def terminate(self): pass
        def wait(self, timeout=None): return 0
        def kill(self): pass

    stage_commands: list[list[str]] = []
    readiness_receipt: Path | None = None
    readiness_probe_passed = False
    sleep_calls = 0

    def readiness_payload() -> dict[str, object]:
        return {"kind": "lehome_groot_n17_public96_policy_server_readiness_v1", "artifact_sha256": CHECKPOINT["artifact_sha256"], "runtime_policy_sha256": CHECKPOINT["runtime_policy_sha256"], "model_path": str(policy.resolve()), "device": "cuda:0", "adapter": "nvidia_gr00t_policy_server_public96_v1", "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()}}

    def fake_run(command, *, cwd, text, stdout, stderr, check):
        assert readiness_probe_passed
        stage_commands.append(command)
        stage_root = Path(command[command.index("--video_dir") + 1]).parent
        output = []
        for episode_index, success in ((1, True), (2, False)):
            folder = "success" if success else "failure"
            for camera in ("top_rgb", "left_rgb", "right_rgb"):
                video = stage_root / "videos" / folder / f"episode{episode_index - 1}_observation_images_{camera}.mp4"
                video.parent.mkdir(parents=True, exist_ok=True); video.write_bytes(f"{episode_index}:{camera}".encode())
            output.append(f"Episode {episode_index}/2: Return={1.0 if success else 0.0:.2f}, Length=600, Success={success}")
        output.append("PUBLIC96_STAGE_COMPLETE " + json.dumps({"raw_checker_overlay": {"overlay_id": RAW_CHECKER_OVERLAY_ID, "overlay_sha256": overlay_sha256()}, "runtime_policy_sha256": CHECKPOINT["runtime_policy_sha256"]}, sort_keys=True))
        return SimpleNamespace(returncode=0, stdout="\n".join(output))

    def fake_popen(command, *args, **kwargs):
        nonlocal readiness_receipt
        readiness_receipt = Path(command[command.index("--readiness-receipt") + 1])
        return Server()

    def delayed_readiness_sleep(_: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        assert stage_commands == []
        if sleep_calls == 25:
            assert readiness_receipt is not None
            readiness_receipt.write_text(json.dumps(readiness_payload()), encoding="utf-8")

    def fake_authenticated_readiness(**_: object) -> None:
        nonlocal readiness_probe_passed
        assert readiness_receipt is not None and readiness_receipt.is_file()
        assert sleep_calls == 25
        readiness_probe_passed = True

    monkeypatch.setattr(evaluator.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
    monkeypatch.setattr(evaluator, "await_authenticated_policy_server_ready", fake_authenticated_readiness)
    monkeypatch.setattr(evaluator.time, "sleep", delayed_readiness_sleep)
    monkeypatch.setenv("LEHOME_GROOT_N17_PUBLIC96_POLICY_TOKEN", "x" * 32)
    output_root = tmp_path / "run"
    result = run(SimpleNamespace(matrix=MATRIX, matrix_sha256=MATRIX_SHA256, policy_path=policy, checkpoint_identity_receipt=identity, asset_root=tmp_path / "assets", output_root=output_root, policy_server_port=9117, policy_server_token_env="LEHOME_GROOT_N17_PUBLIC96_POLICY_TOKEN", dry_run=False))
    assert len(result["episodes"]) == 96
    assert len(stage_commands) == 48 and sleep_calls == 25 and readiness_probe_passed
    assert (output_root / "result.json").is_file()
    assert (output_root / "verifier-receipt.json").is_file()


def test_run_blocks_all_stages_when_authenticated_readiness_probe_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.eval_groot_n17_public96 as evaluator

    args, output_root, identity = _runtime_inputs(tmp_path, monkeypatch, output_name="probe-failure")
    stage_commands: list[list[str]] = []

    class Server:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            pass

    def fake_popen(command: list[str], *popen_args: object, **kwargs: object) -> Server:
        receipt = Path(command[command.index("--readiness-receipt") + 1])
        receipt.write_text(json.dumps({
            "kind": "lehome_groot_n17_public96_policy_server_readiness_v1",
            "artifact_sha256": CHECKPOINT["artifact_sha256"],
            "runtime_policy_sha256": CHECKPOINT["runtime_policy_sha256"],
            "model_path": str(args.policy_path.resolve()), "device": "cuda:0",
            "adapter": "nvidia_gr00t_policy_server_public96_v1",
            "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()},
        }), encoding="utf-8")
        return Server()

    def no_stage(command: list[str], **kwargs: object) -> object:
        stage_commands.append(command)
        raise AssertionError("a failed authenticated readiness probe must not launch a stage")

    reason = "N1.7 policy server did not pass token-bound readiness"
    monkeypatch.setattr(evaluator.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(evaluator.subprocess, "run", no_stage)
    monkeypatch.setattr(evaluator, "await_authenticated_policy_server_ready", lambda **_: (_ for _ in ()).throw(Public96ContractError(reason)))

    with pytest.raises(Public96ContractError, match=reason):
        run(args)

    assert stage_commands == []
    _assert_pre_stage_invalid_evidence(output_root, identity, reason)


def test_policy_server_construction_failure_emits_all_unstarted_invalid_assignments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.eval_groot_n17_public96 as evaluator

    args, output_root, identity = _runtime_inputs(tmp_path, monkeypatch, output_name="popen-failure")
    stage_commands: list[list[str]] = []

    def fail_to_start(*args: object, **kwargs: object) -> object:
        raise OSError("policy server launch denied")

    def no_stage(command: list[str], **kwargs: object) -> object:
        stage_commands.append(command)
        raise AssertionError("startup failure must not launch a public96 stage")

    monkeypatch.setattr(evaluator.subprocess, "Popen", fail_to_start)
    monkeypatch.setattr(evaluator.subprocess, "run", no_stage)

    reason = "policy server startup failed: policy server launch denied"
    with pytest.raises(Public96ContractError, match=reason):
        run(args)

    assert stage_commands == []
    _assert_pre_stage_invalid_evidence(output_root, identity, reason)


@pytest.mark.parametrize("mode", ("exited", "invalid_readiness"))
def test_policy_server_pre_stage_failure_emits_all_unstarted_invalid_assignments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    import scripts.eval_groot_n17_public96 as evaluator

    args, output_root, identity = _runtime_inputs(tmp_path, monkeypatch, output_name=f"server-{mode}")
    stage_commands: list[list[str]] = []

    class Server:
        def poll(self) -> int | None:
            return 1 if mode == "exited" else None

        def terminate(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            pass

    def fake_popen(command: list[str], *args: object, **kwargs: object) -> Server:
        if mode == "invalid_readiness":
            readiness = Path(command[command.index("--readiness-receipt") + 1])
            readiness.write_text(json.dumps({"artifact_sha256": "not-the-pinned-policy"}), encoding="utf-8")
        return Server()

    def no_stage(command: list[str], **kwargs: object) -> object:
        stage_commands.append(command)
        raise AssertionError("pre-stage server failure must not launch a public96 stage")

    monkeypatch.setattr(evaluator.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(evaluator.subprocess, "run", no_stage)
    monkeypatch.setattr(evaluator, "await_authenticated_policy_server_ready", lambda **_: None)
    monkeypatch.setattr(evaluator.time, "sleep", lambda _: None)

    reason = "N1.7 policy server exited before public96 evaluation" if mode == "exited" else "policy server readiness does not bind the pinned N1.7 policy"
    with pytest.raises(Public96ContractError, match=reason):
        run(args)

    assert stage_commands == []
    _assert_pre_stage_invalid_evidence(output_root, identity, reason)
