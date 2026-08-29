from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.eval_groot_n17_public96 import (
    CHECKPOINT,
    CheckpointIdentityError,
    Public96ContractError,
    build_stage_command,
    load_frozen_matrix,
    run,
    tree_sha256,
    validate_checkpoint_identity,
    validate_output_path,
    verify_result,
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


def _result_identity() -> dict[str, object]:
    return {"kind": "lehome_groot_n17_checkpoint_identity_v1", **CHECKPOINT, "cache_path": "/verified/cache", "cache_tree_sha256": "a" * 64}


def _outcomes(stages: tuple[object, ...], root: Path) -> list[dict[str, object]]:
    outcomes: list[dict[str, object]] = []
    for stage in stages:
        stage_root = root / stage.stage_id
        stage_root.mkdir(parents=True, exist_ok=True)
        log = stage_root / "stage.log"; log.write_text("complete\n", encoding="utf-8")
        receipt = stage_root / "stage-receipt.json"
        stage_entries = []
        for episode_index in stage.episode_indices:
            videos = {}
            directory = stage_root / "videos" / ("success" if episode_index == 1 else "failure")
            directory.mkdir(parents=True, exist_ok=True)
            for camera in ("top_rgb", "left_rgb", "right_rgb"):
                video = directory / f"episode{episode_index - 1}_observation_{camera}.mp4"
                video.write_bytes(f"{stage.stage_id}:{episode_index}:{camera}".encode())
                videos[camera] = {"relative_path": video.relative_to(root).as_posix(), "sha256": hashlib.sha256(video.read_bytes()).hexdigest()}
            stage_entries.append((episode_index, videos))
        receipt.write_text(json.dumps({"kind": "lehome_groot_n17_public96_stage_receipt_v1", "stage_id": stage.stage_id, "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()}, "episodes": [{"episode_index": episode_index, "videos": videos} for episode_index, videos in stage_entries]}), encoding="utf-8")
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

    assert validate_output_path(root, root / "stage-001" / "stage.log") == root / "stage-001" / "stage.log"
    with pytest.raises(Public96ContractError, match="escapes"):
        validate_output_path(root, tmp_path / "outside.json")


def test_result_verification_counts_clean_failures_and_categories(tmp_path: Path) -> None:
    stages = load_frozen_matrix(MATRIX, MATRIX_SHA256)
    result = {
        "kind": "lehome_groot_n17_public96_result_v1",
        "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        "checkpoint": _result_identity(), "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()},
        "episodes": _outcomes(stages, tmp_path),
    }

    verified = verify_result(result, stages=stages, matrix_sha256=result["matrix_sha256"], output_root=tmp_path)
    assert verified["overall"] == {"episodes": 96, "successes": 48, "policy_failures": 48}
    assert verified["categories"]["top_long"] == {"episodes": 24, "successes": 12, "policy_failures": 12}


def test_result_verification_rejects_invalid_or_missing_episode(tmp_path: Path) -> None:
    stages = load_frozen_matrix(MATRIX, MATRIX_SHA256)
    result = {
        "kind": "lehome_groot_n17_public96_result_v1",
        "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        "checkpoint": _result_identity(), "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()},
        "episodes": _outcomes(stages, tmp_path),
    }
    result["episodes"][0]["outcome"] = "infrastructure_invalid"
    with pytest.raises(Public96ContractError, match="invalid"):
        verify_result(result, stages=stages, matrix_sha256=result["matrix_sha256"], output_root=tmp_path)

    result["episodes"] = result["episodes"][:-1]
    with pytest.raises(Public96ContractError, match="96"):
        verify_result(result, stages=stages, matrix_sha256=result["matrix_sha256"], output_root=tmp_path)


def test_result_verification_rejects_mutated_real_camera_artifact(tmp_path: Path) -> None:
    stages = load_frozen_matrix(MATRIX, MATRIX_SHA256)
    result = {"kind": "lehome_groot_n17_public96_result_v1", "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(), "checkpoint": _result_identity(), "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()}, "episodes": _outcomes(stages, tmp_path)}
    first_video = tmp_path / result["episodes"][0]["artifacts"]["videos"]["top_rgb"]["relative_path"]
    first_video.write_bytes(b"tampered")
    with pytest.raises(Public96ContractError, match="digest"):
        verify_result(result, stages=stages, matrix_sha256=result["matrix_sha256"], output_root=tmp_path)


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

    def fake_run(command, *, cwd, text, stdout, stderr, check):
        stage_root = Path(command[command.index("--video_dir") + 1]).parent
        output = []
        for episode_index, success in ((1, True), (2, False)):
            folder = "success" if success else "failure"
            for camera in ("top_rgb", "left_rgb", "right_rgb"):
                video = stage_root / "videos" / folder / f"episode{episode_index - 1}_observation_{camera}.mp4"
                video.parent.mkdir(parents=True, exist_ok=True); video.write_bytes(f"{episode_index}:{camera}".encode())
            output.append(f"Episode {episode_index}/2: Return={1.0 if success else 0.0:.2f}, Length=600, Success={success}")
        output.append("PUBLIC96_STAGE_COMPLETE " + json.dumps({"raw_checker_overlay": {"overlay_id": RAW_CHECKER_OVERLAY_ID, "overlay_sha256": overlay_sha256()}, "runtime_policy_sha256": CHECKPOINT["runtime_policy_sha256"]}, sort_keys=True))
        return SimpleNamespace(returncode=0, stdout="\n".join(output))

    def fake_popen(command, *args, **kwargs):
        receipt = Path(command[command.index("--readiness-receipt") + 1])
        receipt.write_text(json.dumps({"artifact_sha256": CHECKPOINT["artifact_sha256"], "runtime_policy_sha256": CHECKPOINT["runtime_policy_sha256"], "model_path": str(policy.resolve()), "device": "cuda:0"}), encoding="utf-8")
        return Server()
    monkeypatch.setattr(evaluator.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
    monkeypatch.setattr(evaluator.time, "sleep", lambda _: None)
    monkeypatch.setenv("LEHOME_GROOT_N17_PUBLIC96_POLICY_TOKEN", "x" * 32)
    output_root = tmp_path / "run"
    result = run(SimpleNamespace(matrix=MATRIX, matrix_sha256=MATRIX_SHA256, policy_path=policy, checkpoint_identity_receipt=identity, asset_root=tmp_path / "assets", output_root=output_root, policy_server_port=9117, policy_server_token_env="LEHOME_GROOT_N17_PUBLIC96_POLICY_TOKEN", dry_run=False))
    assert len(result["episodes"]) == 96
    assert (output_root / "result.json").is_file()
    assert (output_root / "verifier-receipt.json").is_file()
