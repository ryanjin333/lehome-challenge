from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts.run_official_lehome_comparison import (
    ASSET_REVISION,
    SOURCE_REVISION,
    ComparisonError,
    MatrixRow,
    N17_IDENTITY,
    COMPETITOR_FILES,
    PolicyDefinition,
    build_eval_command,
    compile_policy_result,
    smoke_matrix,
    validate_evaluation_assets,
    validate_reference_matrix,
    validate_n17_checkpoint,
    validate_competitor_checkpoint,
    validate_runtime_evidence,
    validate_smoke_prerequisite,
    metadata_identities,
    seal_execution_bundle,
    validate_sealed_execution,
    deterministic_remote_prefix,
    publish_comparison,
    _command_parity,
    _execution_env,
    load_release_matrix,
)


CATEGORIES = {
    "top_long": "Top_Long",
    "top_short": "Top_Short",
    "pant_long": "Pant_Long",
    "pant_short": "Pant_Short",
}


def test_execution_env_does_not_inherit_runtime_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "official-source"
    log_root = tmp_path / "logs"
    isaaclab_root = tmp_path / "isaaclab"
    isaaclab_tasks_root = tmp_path / "isaaclab-tasks"
    native_site_root = tmp_path / "native-site"
    monkeypatch.setenv("PYTHONPATH", "/runtime/source/lehome:/runtime")

    env = _execution_env(
        source_root=source_root,
        log_root=log_root,
        isaaclab_root=isaaclab_root,
        isaaclab_tasks_root=isaaclab_tasks_root,
        native_site_root=native_site_root,
        policy=PolicyDefinition("ours-12k", "docker", docker_url="http://127.0.0.1:8080"),
        sanitized_config_root=None,
        compatibility_receipt=None,
    )

    paths = env["PYTHONPATH"].split(":")
    assert paths[:2] == [str(source_root / "source/lehome"), str(source_root)]
    assert "/runtime" not in paths
    assert "/runtime/source/lehome" not in paths


def _assets(root: Path) -> Path:
    base = root / "objects" / "Challenge_Garment" / "Release"
    for category, directory in CATEGORIES.items():
        target = base / directory
        target.mkdir(parents=True)
        prefix = directory
        target.joinpath(f"{prefix}.txt").write_text(
            "\n".join(f"{prefix}_Garment_{index:02d}" for index in range(12)) + "\n",
            encoding="utf-8",
        )
    return root


def test_release_matrix_uses_native_lists_in_fixed_category_and_episode_order(tmp_path: Path) -> None:
    rows = load_release_matrix(_assets(tmp_path / "assets"), episodes_per_garment=2)

    assert len(rows) == 96
    assert rows[0].category == "top_long"
    assert rows[0].garment == "Top_Long_Garment_00"
    assert [row.episode_index for row in rows[:2]] == [1, 2]
    assert rows[-1].category == "pant_short"
    assert rows[-1].garment == "Pant_Short_Garment_11"
    assert all(row.seed == 42 for row in rows)


def test_smoke_matrix_is_exactly_two_top_long_seen_zero_episodes() -> None:
    rows = smoke_matrix()
    assert [(row.category, row.garment, row.episode_index, row.seed) for row in rows] == [
        ("custom", "Top_Long_Seen_0", 1, 42),
        ("custom", "Top_Long_Seen_0", 2, 42),
    ]


def test_release_matrix_rejects_duplicate_or_non_12_item_lists(tmp_path: Path) -> None:
    assets = _assets(tmp_path / "assets")
    path = assets / "objects/Challenge_Garment/Release/Top_Long/Top_Long.txt"
    path.write_text("\n".join(["duplicate"] * 12) + "\n", encoding="utf-8")

    with pytest.raises(ComparisonError, match="12 unique"):
        load_release_matrix(assets, episodes_per_garment=2)


def test_official_commands_have_identical_scoring_arguments_and_policy_specific_adapter(tmp_path: Path) -> None:
    common = dict(
        source_root=tmp_path / "source",
        assets_root=tmp_path / "assets",
        dataset_root=tmp_path / "metadata",
        video_dir=tmp_path / "videos",
        garment_type="top_long",
        python_bin="python3",
    )
    ours = build_eval_command(
        PolicyDefinition("ours-12k", "docker", docker_url="http://127.0.0.1:8080"),
        **common,
    )
    competitor = build_eval_command(
        PolicyDefinition(
            "competitor-n15", "lerobot", checkpoint_root=tmp_path / "checkpoint"
        ),
        **common,
    )

    for token in (
        "--headless", "--enable_cameras", "--device", "cpu", "--seed", "42",
        "--max_steps", "600", "--num_episodes", "2", "--save_video",
    ):
        assert token in ours and token in competitor
    assert ours[ours.index("--policy_type") + 1] == "docker"
    assert competitor[competitor.index("--policy_type") + 1] == "lerobot"
    assert "--docker_url" in ours and "--docker_url" not in competitor
    assert "--policy_path" in competitor and "--policy_path" not in ours


def test_smoke_uses_top_long_metadata_while_retaining_official_custom_garment_type(tmp_path: Path) -> None:
    command = build_eval_command(
        PolicyDefinition("ours-12k", "docker", docker_url="http://127.0.0.1:8080"),
        source_root=tmp_path / "source",
        assets_root=tmp_path / "assets",
        dataset_root=tmp_path / "metadata",
        video_dir=tmp_path / "videos",
        garment_type="custom",
        python_bin="python3",
    )
    assert command[command.index("--garment_type") + 1] == "custom"
    assert command[command.index("--dataset_root") + 1] == str(tmp_path / "metadata/top_long_merged")


def test_command_parity_ignores_only_policy_adapter_and_fresh_video_destination(tmp_path: Path) -> None:
    common = dict(
        source_root=tmp_path / "source",
        assets_root=tmp_path / "assets",
        dataset_root=tmp_path / "metadata",
        garment_type="custom",
        python_bin="python3",
    )
    commands = {
        "ours-12k-custom": build_eval_command(
            PolicyDefinition("ours-12k", "docker", docker_url="http://127.0.0.1:8080"),
            video_dir=tmp_path / "ours-videos",
            **common,
        ),
        "competitor-n15-custom": build_eval_command(
            PolicyDefinition("competitor-n15", "lerobot", checkpoint_root=tmp_path / "checkpoint"),
            video_dir=tmp_path / "competitor-videos",
            **common,
        ),
    }
    assert _command_parity(commands)["verified"] is True


def _write_log(path: Path, garments: list[str], *, success: bool = False) -> None:
    lines: list[str] = []
    for index, garment in enumerate(garments, start=1):
        lines.append(f"2026-08-30 - INFO - Evaluating: {garment} (Release) ({index}/{len(garments)})")
        for episode in (1, 2):
            lines.append(
                f"2026-08-30 - INFO - Episode {episode}/2: Return=1.25, Length=600, Success={success}"
            )
    lines.append("2026-08-30 - INFO - Evaluation completed successfully")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_videos(root: Path, command_id: str) -> None:
    target = root / command_id / "failure"
    target.mkdir(parents=True)
    for episode in (0, 1):
        for camera in ("top_rgb", "left_rgb", "right_rgb"):
            (target / f"episode{episode}_observation_images_{camera}.mp4").write_bytes(b"video")


def test_compile_policy_result_requires_completion_order_count_and_videos(tmp_path: Path) -> None:
    assets = _assets(tmp_path / "assets")
    matrix = load_release_matrix(assets, episodes_per_garment=2)
    logs = tmp_path / "logs"
    videos = tmp_path / "videos"
    logs.mkdir()
    for category, directory in CATEGORIES.items():
        garments = [f"{directory}_Garment_{index:02d}" for index in range(12)]
        _write_log(logs / f"ours-12k-{category}.log", garments)
        _write_videos(videos, f"ours-12k-{category}")

    result = compile_policy_result(
        policy_id="ours-12k", matrix=matrix, logs_root=logs, videos_root=videos
    )

    assert result["status"] == "valid"
    assert result["episode_count"] == 96
    assert result["success_count"] == 0
    assert len(result["outcomes"]) == 96
    assert result["outcomes"][0]["garment"] == "Top_Long_Garment_00"


def test_compile_policy_result_records_but_does_not_reject_stale_opposite_status_video(tmp_path: Path) -> None:
    assets = _assets(tmp_path / "assets")
    matrix = load_release_matrix(assets, episodes_per_garment=2)
    logs = tmp_path / "logs"
    videos = tmp_path / "videos"
    logs.mkdir()
    for category, directory in CATEGORIES.items():
        garments = [f"{directory}_Garment_{index:02d}" for index in range(12)]
        _write_log(logs / f"ours-12k-{category}.log", garments)
        _write_videos(videos, f"ours-12k-{category}")
    stale = videos / "ours-12k-top_long/success/episode0_observation_images_top_rgb.mp4"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    result = compile_policy_result(
        policy_id="ours-12k", matrix=matrix, logs_root=logs, videos_root=videos
    )

    top_video = next(
        row for row in result["retained_official_videos"]
        if row["path"].endswith("ours-12k-top_long/failure/episode0_observation_images_top_rgb.mp4")
    )
    assert top_video["stale_opposite_path"].endswith(
        "ours-12k-top_long/success/episode0_observation_images_top_rgb.mp4"
    )


def test_full_assets_must_be_the_canonical_root_and_smoke_rejects_symlink_drift(tmp_path: Path) -> None:
    canonical = _assets(tmp_path / "canonical")
    different = _assets(tmp_path / "different")
    with pytest.raises(ComparisonError, match="full evaluation assets"):
        validate_evaluation_assets(canonical, different, mode="full")

    view = tmp_path / "view"
    view.symlink_to(canonical, target_is_directory=True)
    with pytest.raises(ComparisonError, match="smoke evaluation assets"):
        validate_evaluation_assets(canonical, view, mode="smoke")


@pytest.mark.parametrize("bad_line", ["Traceback (most recent call last):", "CUDA nonfinite state"])
def test_compile_policy_result_fails_closed_on_traceback_or_nonfinite(tmp_path: Path, bad_line: str) -> None:
    assets = _assets(tmp_path / "assets")
    matrix = load_release_matrix(assets, episodes_per_garment=2)
    logs = tmp_path / "logs"
    videos = tmp_path / "videos"
    logs.mkdir()
    for category, directory in CATEGORIES.items():
        garments = [f"{directory}_Garment_{index:02d}" for index in range(12)]
        path = logs / f"ours-12k-{category}.log"
        _write_log(path, garments)
        path.write_text(path.read_text(encoding="utf-8") + bad_line + "\n", encoding="utf-8")
        _write_videos(videos, f"ours-12k-{category}")

    with pytest.raises(ComparisonError, match="infrastructure_invalid"):
        compile_policy_result(
            policy_id="ours-12k", matrix=matrix, logs_root=logs, videos_root=videos
        )


def test_publication_receipt_constants_are_immutable_revisions() -> None:
    assert SOURCE_REVISION == "a805ad2f7ab52a4583066fc4ee5180459a7f9d15"
    assert ASSET_REVISION == "bea65fd960ad5a1bb3bd3fa77164b28001c08ef9"
    assert len(SOURCE_REVISION) == len(ASSET_REVISION) == 40


def test_frozen_reference_matrix_bytes_bind_the_same_native_96_order() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "configs/eval_groot_n17_public96_reference.json").read_text(encoding="utf-8")
    )
    rows = [
        row
        for stage in payload["stages"]
        for row in (
            MatrixRow(stage["category"], stage["garment_name"], 1, 42),
            MatrixRow(stage["category"], stage["garment_name"], 2, 42),
        )
    ]
    digest = validate_reference_matrix(
        root / "configs/eval_groot_n17_public96_reference.json",
        root / "configs/eval_groot_n17_public96_reference.json.sha256",
        rows,
    )
    assert digest == "bb3c11ddb10eb53ba3cd2b189850d74bc8f2bfa45d15153812b806060b4b80b5"


def test_n17_identity_uses_the_existing_exact_validator_and_rejects_identity_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "n17"
    checkpoint.mkdir()
    receipt = tmp_path / "n17-identity.json"
    receipt.write_text("{}\n", encoding="utf-8")
    calls = []

    def validator(payload, root):
        calls.append((payload, root))
        return {"kind": "lehome_groot_n17_checkpoint_identity_v1", **N17_IDENTITY}

    result = validate_n17_checkpoint(checkpoint, receipt, validator=validator)
    assert result["artifact_sha256"] == N17_IDENTITY["artifact_sha256"]
    assert calls == [({}, checkpoint)]

    def drifted(_payload, _root):
        return {"kind": "lehome_groot_n17_checkpoint_identity_v1", **N17_IDENTITY, "step": 500}

    with pytest.raises(ComparisonError, match="identity drift"):
        validate_n17_checkpoint(checkpoint, receipt, validator=drifted)


def test_competitor_validator_is_closed_over_exact_seven_file_contract(tmp_path: Path) -> None:
    assert len(COMPETITOR_FILES) == 7
    assert COMPETITOR_FILES["model.safetensors"] == "d8d91b6c11cb5aa18fe9a48e7da88eae0ec7e5a227a315ba67ee167d645cde76"
    checkpoint = tmp_path / "competitor"
    checkpoint.mkdir()
    for name in COMPETITOR_FILES:
        (checkpoint / name).write_bytes(b"wrong")
    with pytest.raises(ComparisonError, match="digest mismatch"):
        validate_competitor_checkpoint(checkpoint)
    (checkpoint / "extra-directory").mkdir()
    with pytest.raises(ComparisonError, match="unsafe entry"):
        validate_competitor_checkpoint(checkpoint)


def test_runtime_evidence_binds_clean_revision_images_cuda_and_policy_readiness(tmp_path: Path) -> None:
    runtime = {"revision": "a" * 40, "tree_sha256": "b" * 64}
    rollout = {
        "kind": "lehome_official_image_inspection_v1",
        "reference": "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7",
        "image_id": "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7",
        "repo_digests": [],
        "docker_inspect_sha256": "c" * 64,
    }
    policy = {
        "kind": "lehome_official_image_inspection_v1",
        "reference": "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746",
        "image_id": "sha256:" + "d" * 64,
        "repo_digests": ["ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746"],
        "docker_inspect_sha256": "e" * 64,
    }
    cuda = {"cuda_available": True, "cuda_device_count": 1, "cuda_runtime": "12.8", "cuda_device_name": "RTX PRO 6000"}
    readiness = {
        "kind": "lehome_groot_n17_public96_policy_server_readiness_v1",
        "artifact_sha256": N17_IDENTITY["artifact_sha256"],
        "runtime_policy_sha256": N17_IDENTITY["runtime_policy_sha256"],
        "model_path": str(tmp_path / "n17"),
        "device": "cuda:0",
        "adapter": "nvidia_gr00t_policy_server_public96_v1",
        "raw_checker_overlay": {"id": "ignored-by-official-scorer", "sha256": "f" * 64},
    }
    result = validate_runtime_evidence(
        runtime_identity=runtime,
        rollout_image=rollout,
        policy_image=policy,
        cuda_receipt=cuda,
        readiness_receipt=readiness,
        n17_checkpoint=tmp_path / "n17",
    )
    assert result["policy_device"] == "cuda:0"
    assert result["runtime"]["revision"] == "a" * 40


def test_metadata_identity_binds_each_category_root(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    for category in CATEGORIES:
        root = metadata / f"{category}_merged"
        root.mkdir(parents=True)
        (root / "meta.json").write_text(category, encoding="utf-8")

    identity = metadata_identities(metadata)

    assert set(identity["category_tree_sha256"]) == set(CATEGORIES)
    assert len(set(identity["category_tree_sha256"].values())) == 4


def _valid_smoke_receipt() -> dict[str, object]:
    return {
        "kind": "lehome_official_policy_comparison_v1",
        "status": "valid",
        "mode": "smoke",
        "official_source": {"revision": SOURCE_REVISION, "tree_sha256": "1" * 64},
        "canonical_assets": {"revision": ASSET_REVISION, "tree_sha256": "2" * 64},
        "reviewed_runtime": {"revision": "a" * 40, "tree_sha256": "3" * 64, "adapter_sha256": {}},
        "runtime_evidence": {"rollout_image": {"image_id": "rollout"}, "policy_image": {"reference": "policy"}},
        "n17_checkpoint": {"artifact_sha256": "n17"},
        "competitor_checkpoint": {"tree_sha256": "competitor"},
        "metadata": {"tree_sha256": "4" * 64, "category_tree_sha256": {category: "5" * 64 for category in CATEGORIES}},
        "scorer_sha256": "6" * 64,
        "frozen_reference_matrix_sha256": "7" * 64,
        "simulator_device": "cpu",
        "policy_device": "cuda:0",
        "seed": 42,
        "max_steps": 600,
        "episodes_per_garment": 2,
        "results": [
            {"policy_id": "ours-12k", "status": "valid", "episode_count": 2, "outcomes": [
                {"category": "custom", "garment": "Top_Long_Seen_0", "episode_index": index, "seed": 42}
                for index in (1, 2)
            ]},
            {"policy_id": "competitor-n15", "status": "valid", "episode_count": 2, "outcomes": [
                {"category": "custom", "garment": "Top_Long_Seen_0", "episode_index": index, "seed": 42}
                for index in (1, 2)
            ]},
        ],
    }


def test_full_prerequisite_requires_identity_matched_two_outcome_smoke() -> None:
    smoke = _valid_smoke_receipt()
    expected = {key: smoke[key] for key in (
        "official_source", "canonical_assets", "reviewed_runtime", "runtime_evidence",
        "n17_checkpoint", "competitor_checkpoint", "metadata", "scorer_sha256",
        "frozen_reference_matrix_sha256", "simulator_device", "policy_device", "seed",
        "max_steps", "episodes_per_garment",
    )}
    validated = validate_smoke_prerequisite(smoke, expected=expected)
    assert validated["mode"] == "smoke"

    smoke["results"][0]["outcomes"].pop()
    with pytest.raises(ComparisonError, match="two valid outcomes"):
        validate_smoke_prerequisite(smoke, expected=expected)


def test_execution_seal_detects_added_or_changed_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    (root / "logs").mkdir(parents=True)
    (root / "logs/eval.log").write_text("complete\n", encoding="utf-8")
    (root / "comparison-receipt.json").write_text('{"status":"valid"}\n', encoding="utf-8")

    seal = seal_execution_bundle(root)
    validated = validate_sealed_execution(root / "comparison-receipt.json")
    assert validated["manifest_sha256"] == seal["manifest_sha256"]
    assert deterministic_remote_prefix(validated["receipt_sha256"], validated["manifest_sha256"]).startswith(
        "official-comparisons/"
    )

    (root / "logs/eval.log").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ComparisonError, match="execution manifest"):
        validate_sealed_execution(root / "comparison-receipt.json")


class _FakeHub:
    def __init__(self, *, commit_timeout: bool = False) -> None:
        self.files: dict[str, bytes] = {}
        self.commit_timeout = commit_timeout
        self.create_calls = 0
        self.revision = "a" * 40

    def api(self, token=False):
        hub = self

        class Api:
            def list_repo_files(self, **_kwargs):
                return sorted(hub.files)

            def repo_info(self, **_kwargs):
                return SimpleNamespace(sha=hub.revision)

            def create_commit(self, *, operations, **_kwargs):
                hub.create_calls += 1
                for operation in operations:
                    hub.files[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
                if hub.commit_timeout:
                    raise TimeoutError("commit response timed out")
                return SimpleNamespace(oid=hub.revision)

        return Api()

    def module(self, tmp_path: Path) -> ModuleType:
        hub = self
        module = ModuleType("huggingface_hub")

        class CommitOperationAdd:
            def __init__(self, *, path_in_repo, path_or_fileobj):
                self.path_in_repo = path_in_repo
                self.path_or_fileobj = path_or_fileobj

        module.CommitOperationAdd = CommitOperationAdd
        module.HfApi = lambda token=False: hub.api(token=token)

        def download(*, filename, local_dir, **_kwargs):
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(hub.files[filename])
            return str(target)

        module.hf_hub_download = download
        return module


def _sealed_full_bundle(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "bundle"
    root.mkdir()
    receipt = root / "comparison-receipt.json"
    receipt.write_text(
        json.dumps({"kind": "lehome_official_policy_comparison_v1", "status": "valid", "mode": "full"}) + "\n",
        encoding="utf-8",
    )
    (root / "payload.log").write_text("complete\n", encoding="utf-8")
    seal_execution_bundle(root)
    return receipt, validate_sealed_execution(receipt)


def _publication_args(receipt: Path, tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        receipt=receipt,
        repository="owner/public-dataset",
        remote_prefix=None,
        revision="main",
        token_env="TEST_HF_TOKEN",
        publication_receipt=tmp_path / "publication.json",
    )


def test_publication_fresh_upload_includes_payload_and_all_seal_files(tmp_path: Path, monkeypatch) -> None:
    receipt, sealed = _sealed_full_bundle(tmp_path)
    hub = _FakeHub()
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", hub.module(tmp_path))
    monkeypatch.setenv("TEST_HF_TOKEN", "secret")

    output = publish_comparison(_publication_args(receipt, tmp_path))

    prefix = deterministic_remote_prefix(sealed["receipt_sha256"], sealed["manifest_sha256"])
    expected = {f"{prefix}/{entry['path']}" for entry in sealed["entries"]} | {
        f"{prefix}/execution-manifest.json",
        f"{prefix}/comparison-receipt.sha256.json",
        f"{prefix}/status.json",
    }
    assert set(hub.files) == expected
    assert hub.create_calls == 1
    assert json.loads(output.read_text(encoding="utf-8"))["recovered_existing_prefix"] is False


def test_publication_recovers_exact_existing_prefix_after_commit_timeout(tmp_path: Path, monkeypatch) -> None:
    receipt, _sealed = _sealed_full_bundle(tmp_path)
    hub = _FakeHub(commit_timeout=True)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", hub.module(tmp_path))
    monkeypatch.setenv("TEST_HF_TOKEN", "secret")

    output = publish_comparison(_publication_args(receipt, tmp_path))

    publication = json.loads(output.read_text(encoding="utf-8"))
    assert publication["recovered_existing_prefix"] is True
    assert publication["immutable_revision"] == "a" * 40
    assert hub.create_calls == 1


def test_publication_rejects_existing_prefix_drift_without_reupload(tmp_path: Path, monkeypatch) -> None:
    receipt, sealed = _sealed_full_bundle(tmp_path)
    hub = _FakeHub()
    prefix = deterministic_remote_prefix(sealed["receipt_sha256"], sealed["manifest_sha256"])
    hub.files[f"{prefix}/comparison-receipt.json"] = b"drift"
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", hub.module(tmp_path))
    monkeypatch.setenv("TEST_HF_TOKEN", "secret")

    with pytest.raises(ComparisonError, match="missing, extra, or drifted"):
        publish_comparison(_publication_args(receipt, tmp_path))
    assert hub.create_calls == 0
