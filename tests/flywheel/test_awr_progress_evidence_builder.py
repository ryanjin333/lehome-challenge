from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from lehome_train.groot.awr_weighting import load_progress_evidence


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_awr_progress_evidence.py"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_awr_progress_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _outer_digest(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file() and path.name != "SHA256SUMS.json":
            entries.append({
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": _sha(path),
                "byte_size": path.stat().st_size,
            })
    return hashlib.sha256(_canonical(entries)).hexdigest()


def _manifest(root: Path) -> None:
    checksums = {
        path.relative_to(root).as_posix(): {"sha256": _sha(path), "size": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write(root / "SHA256SUMS.json", checksums)


def _episode(accepted: Path, episode_id: str, *, category: str, success_at: int) -> str:
    root = accepted / episode_id
    raw = root / "raw" / episode_id
    _write(raw / "episode.json", {
        "episode_id": episode_id,
        "accepted_success": True,
        "outcome": "success",
        "identity": {"episode_id": episode_id, "category": category, "release_stage": "seen"},
    })
    lines = []
    for step in range(4):
        lines.append(json.dumps({
            "step": step,
            "reward": float(step) / 3.0,
            "success": step >= success_at,
            "action_source": "policy",
        }, sort_keys=True, separators=(",", ":")))
    (raw / "annotations.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _manifest(root)
    return _outer_digest(root)


def _mixture(tmp_path: Path, episodes: list[str], *, ambiguous: bool = False) -> tuple[Path, str]:
    mixture_id = "d" * 64
    manifest = {
        "schema_version": 2,
        "kind": "lehome_runtime_mixture",
        "repository": "ryanjin333/lehome-groot-n17-rollouts",
        "safe_prefix": f"mixtures/{mixture_id}",
        "mixture_id": mixture_id,
        "sources": [
            {
                "source_id": "bc", "source_type": "bc", "quota": 7, "release_stage": "seen",
                "source_tree_sha256": "a" * 64, "artifact_receipt_path": "receipt.json",
                "artifact_receipt_sha256": "a" * 64, "acceptance_receipt_path": "acceptance.json",
                "acceptance_receipt_sha256": "a" * 64,
                "publication": {"repository": "ryanjin333/lehome-groot-n17-data", "revision": "a" * 40, "prefix": "bc/full", "readback_receipt_path": "/tmp/bc.json", "readback_receipt_sha256": "a" * 64},
                "source_identity": {"prepared_manifest_path": "manifest.json", "prepared_manifest_sha256": "a" * 64, "action_source": "organizer_expert"},
            },
            {
                "source_id": "replay", "source_type": "rollout", "quota": 3, "release_stage": "seen",
                "source_tree_sha256": "b" * 64, "artifact_receipt_path": "receipt.json",
                "artifact_receipt_sha256": "b" * 64, "acceptance_receipt_path": "acceptance.json",
                "acceptance_receipt_sha256": "b" * 64,
                "publication": {"repository": "ryanjin333/lehome-groot-n17-rollouts", "revision": "b" * 40, "prefix": "rollouts/round-1", "readback_receipt_path": "/tmp/replay.json", "readback_receipt_sha256": "b" * 64},
                "source_identity": {"round_manifest_path": "round-manifest.json", "round_manifest_sha256": "b" * 64, "action_source": "policy"},
            },
        ],
        "camera_schema": ["observation.images.top_rgb", "observation.images.left_rgb", "observation.images.right_rgb"],
        "image_shape": [480, 640, 3], "state_schema": {"dimension": 12, "storage": "absolute"},
        "action_schema": {"dimension": 12, "storage": "absolute"}, "fps": 30,
        "action_horizon": 16, "instruction": "fold the garment on the table",
        "schedule_seed": 17, "cycle_size": 10,
        "mixture_normalization": {"path": "normalization.json", "sha256": "c" * 64, "byte_size": 1},
        "window_index": {"path": "windows.json", "sha256": "", "byte_size": 0},
    }
    windows = [
        {
            "window_id": "bc-0", "source_id": "bc", "source_type": "bc", "source_episode_id": "bc-0",
            "start": 0, "stop": 16, "frame_ids": list(range(16)), "lineage_id": "bc-0", "split": "train",
            "source_locator": {"episode_id": "bc-0", "prepared_manifest_path": "manifest.json", "prepared_manifest_sha256": "a" * 64},
        },
        {
            "window_id": "validation-rollout", "source_id": "replay", "source_type": "rollout", "source_episode_id": "not-in-evidence",
            "start": 0, "stop": 16, "frame_ids": list(range(16)), "lineage_id": "validation-only", "split": "validation",
            "source_locator": {"attempt_root": "raw/not-in-evidence", "attempt_manifest_path": "raw/not-in-evidence/episode.json", "attempt_manifest_sha256": "b" * 64},
        },
    ]
    for index, episode_id in enumerate(episodes):
        windows.append({
            "window_id": f"replay-{index}", "source_id": "replay", "source_type": "rollout", "source_episode_id": episode_id,
            "start": 0, "stop": 16, "frame_ids": list(range(16)),
            "lineage_id": "shared-lineage" if ambiguous else f"lineage-{episode_id}", "split": "train",
            "source_locator": {"attempt_root": f"raw/{episode_id}", "attempt_manifest_path": f"raw/{episode_id}/episode.json", "attempt_manifest_sha256": "b" * 64},
        })
    index = {"schema_version": 2, "manifest_sha256": hashlib.sha256(_canonical(manifest)).hexdigest(), "windows": windows}
    index_path = tmp_path / "windows.json"
    _write(index_path, index)
    manifest["window_index"] = {"path": "windows.json", "sha256": _sha(index_path), "byte_size": index_path.stat().st_size}
    manifest_path = tmp_path / "mixture.json"
    _write(manifest_path, manifest)
    return manifest_path, mixture_id


def _sealed_round(
    tmp_path: Path,
    *,
    episode_ids: list[str],
    extra: bool = False,
    round_id: str = "success-replay-12k-round-1",
) -> tuple[Path, Path, Path]:
    accepted, receipts = tmp_path / "accepted", tmp_path / "receipts"
    digests = {
        episode_id: _episode(accepted, episode_id, category="top_short", success_at=index + 1)
        for index, episode_id in enumerate(episode_ids)
    }
    if extra:
        _episode(accepted, "unsealed", category="pant_short", success_at=2)
    revision = {episode_id: (str(index + 1) * 40) for index, episode_id in enumerate(episode_ids)}
    seal = {
        "schema_version": 2, "kind": "rollout_round_seal", "repository": "ryanjin333/lehome-groot-n17-rollouts",
        "round_id": round_id, "episode_count": len(episode_ids),
        "episode_sha256s": digests, "immutable_revisions": revision, "readback_verified": True,
    }
    seal["seal_sha256"] = hashlib.sha256(_canonical({
        "round_id": seal["round_id"],
        "repository": seal["repository"],
        "episode_sha256s": seal["episode_sha256s"],
        "immutable_revisions": seal["immutable_revisions"],
    })).hexdigest()
    seal_path = tmp_path / "sealed-round.json"
    _write(seal_path, seal)
    for episode_id in episode_ids:
        _write(receipts / f"{episode_id}.sync.json", {
            "schema_version": 1, "attempt_id": episode_id, "repository": seal["repository"],
            "round_id": seal["round_id"], "remote_prefix": f"rollout-rounds/{seal['round_id']}/{episode_id}",
            "publication_ref": "main", "immutable_revision": revision[episode_id], "entry_count": 3,
            "episode_sha256": digests[episode_id], "readback_verified": True,
        })
    return accepted, receipts, seal_path


def _refresh_sealed_round(accepted: Path, receipts: Path, seal_path: Path) -> None:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    episode_ids = sorted(seal["episode_sha256s"])
    seal["episode_sha256s"] = {episode_id: _outer_digest(accepted / episode_id) for episode_id in episode_ids}
    seal["seal_sha256"] = hashlib.sha256(_canonical({
        "round_id": seal["round_id"],
        "repository": seal["repository"],
        "episode_sha256s": seal["episode_sha256s"],
        "immutable_revisions": seal["immutable_revisions"],
    })).hexdigest()
    _write(seal_path, seal)
    for episode_id in episode_ids:
        path = receipts / f"{episode_id}.sync.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["episode_sha256"] = seal["episode_sha256s"][episode_id]
        _write(path, receipt)


def test_builder_writes_exact_rollout_train_evidence_with_category_normalized_progress(tmp_path: Path) -> None:
    builder = _load_builder()
    accepted, receipts, seal = _sealed_round(tmp_path, episode_ids=["episode-a", "episode-b"])
    mixture, mixture_id = _mixture(tmp_path, ["episode-a", "episode-b"])
    output = tmp_path / "awr-progress.json"

    result = builder.build_awr_progress_evidence(
        mixture_manifest=mixture, accepted_root=accepted, sync_receipts_root=receipts,
        round_seal=seal, output=output,
    )

    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert result["mixture_id"] == mixture_id
    assert result["evidence_sha256"] == _sha(output)
    assert {row["episode_id"] for row in evidence["episodes"]} == {"episode-a", "episode-b"}
    assert {row["lineage_id"] for row in evidence["episodes"]} == {"lineage-episode-a", "lineage-episode-b"}
    assert evidence["episodes"][0]["score"] > evidence["episodes"][1]["score"]
    assert not (tmp_path / "awr-progress.json.sha256").is_symlink()
    assert (tmp_path / "awr-progress.json.sha256").read_text(encoding="ascii").strip() == _sha(output)


@pytest.mark.parametrize("failure", ["unsealed", "tampered", "nonfinite", "ambiguous-lineage"])
def test_builder_fails_closed_for_untrusted_replay_evidence(tmp_path: Path, failure: str) -> None:
    builder = _load_builder()
    accepted, receipts, seal = _sealed_round(tmp_path, episode_ids=["episode-a", "episode-b"], extra=failure == "unsealed")
    mixture, _ = _mixture(tmp_path, ["episode-a", "episode-b"], ambiguous=failure == "ambiguous-lineage")
    if failure == "tampered":
        raw = accepted / "episode-a" / "raw" / "episode-a" / "annotations.jsonl"
        raw.write_text(raw.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    if failure == "nonfinite":
        raw = accepted / "episode-a" / "raw" / "episode-a" / "annotations.jsonl"
        raw.write_text(raw.read_text(encoding="utf-8").replace("0.0", "NaN", 1), encoding="utf-8")
        _manifest(accepted / "episode-a")
        _refresh_sealed_round(accepted, receipts, seal)

    with pytest.raises(ValueError):
        builder.build_awr_progress_evidence(
            mixture_manifest=mixture, accepted_root=accepted, sync_receipts_root=receipts,
            round_seal=seal, output=tmp_path / "should-not-exist.json",
        )
    assert not (tmp_path / "should-not-exist.json").exists()


def test_builder_rejects_symlinked_input_and_output_paths(tmp_path: Path) -> None:
    builder = _load_builder()
    accepted, receipts, seal = _sealed_round(tmp_path, episode_ids=["episode-a", "episode-b"])
    mixture, _ = _mixture(tmp_path, ["episode-a", "episode-b"])
    unsafe_input = tmp_path / "accepted-link"
    unsafe_input.symlink_to(accepted, target_is_directory=True)
    with pytest.raises(ValueError):
        builder.build_awr_progress_evidence(
            mixture_manifest=mixture, accepted_root=unsafe_input, sync_receipts_root=receipts,
            round_seal=seal, output=tmp_path / "input-link.json",
        )

    output_parent = tmp_path / "output-real"
    output_parent.mkdir()
    unsafe_output_parent = tmp_path / "output-link"
    unsafe_output_parent.symlink_to(output_parent, target_is_directory=True)
    with pytest.raises(ValueError):
        builder.build_awr_progress_evidence(
            mixture_manifest=mixture, accepted_root=accepted, sync_receipts_root=receipts,
            round_seal=seal, output=unsafe_output_parent / "evidence.json",
        )


def test_builder_publishes_neither_file_when_sidecar_commit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    accepted, receipts, seal = _sealed_round(
        tmp_path, episode_ids=["episode-a", "episode-b"],
    )
    mixture, _ = _mixture(tmp_path, ["episode-a", "episode-b"])
    output = tmp_path / "awr-progress.json"
    real_link = builder.os.link

    def fail_sidecar(source: object, destination: object) -> None:
        if str(destination).endswith(".sha256"):
            raise OSError("injected sidecar publication failure")
        real_link(source, destination)

    monkeypatch.setattr(builder.os, "link", fail_sidecar)

    with pytest.raises(OSError, match="injected sidecar"):
        builder.build_awr_progress_evidence(
            mixture_manifest=mixture, accepted_root=accepted,
            sync_receipts_root=receipts, round_seal=seal, output=output,
        )

    assert not output.exists()
    assert not Path(str(output) + ".sha256").exists()


def test_builder_binds_two_ordered_sealed_rounds_to_exact_train_coverage(tmp_path: Path) -> None:
    builder = _load_builder()
    first = _sealed_round(
        tmp_path / "first",
        episode_ids=["fresh-b", "fresh-a"],
        round_id="balanced-fresh-round",
    )
    second = _sealed_round(
        tmp_path / "second",
        episode_ids=["replay-b", "replay-a"],
        round_id="success-replay-round",
    )
    mixture, _ = _mixture(
        tmp_path / "mixture",
        ["replay-a", "fresh-a", "replay-b", "fresh-b"],
    )
    output = tmp_path / "awr-progress.json"

    result = builder.build_awr_progress_evidence(
        mixture_manifest=mixture,
        accepted_root=[first[0], second[0]],
        sync_receipts_root=[first[1], second[1]],
        round_seal=[first[2], second[2]],
        output=output,
    )

    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert [row["episode_id"] for row in evidence["episodes"]] == [
        "fresh-a", "fresh-b", "replay-a", "replay-b",
    ]
    assert all(
        row["provenance_path"].startswith("source-rounds/0000-balanced-fresh-round-")
        for row in evidence["episodes"][:2]
    )
    assert all(
        row["provenance_path"].startswith("source-rounds/0001-success-replay-round-")
        for row in evidence["episodes"][2:]
    )
    assert [source["round_id"] for source in result["source_rounds"]] == [
        "balanced-fresh-round", "success-replay-round",
    ]
    assert result["episode_count"] == 4
    loaded = load_progress_evidence(
        output,
        expected_sha256=result["evidence_sha256"],
        mixture_id=result["mixture_id"],
        mixture_manifest_sha256=result["mixture_manifest_sha256"],
    )
    assert tuple(loaded.episodes) == ("fresh-a", "fresh-b", "replay-a", "replay-b")


@pytest.mark.parametrize(
    "failure",
    [
        "empty-sources",
        "unequal-source-counts",
        "duplicate-round-id",
        "duplicate-episode-id",
        "duplicate-input-path",
        "output-inside-source",
        "missing-train-episode",
        "extra-sealed-episode",
        "duplicate-train-window-id",
        "colliding-train-path",
    ],
)
def test_multi_round_builder_rejects_global_collisions_and_nonexact_coverage(
    tmp_path: Path,
    failure: str,
) -> None:
    builder = _load_builder()
    first = _sealed_round(
        tmp_path / "first",
        episode_ids=["fresh-a"],
        round_id="balanced-fresh-round",
    )
    second_episode = "fresh-a" if failure == "duplicate-episode-id" else "replay-a"
    second_round_id = "balanced-fresh-round" if failure == "duplicate-round-id" else "success-replay-round"
    second = _sealed_round(
        tmp_path / "second",
        episode_ids=[second_episode],
        round_id=second_round_id,
    )
    mixture_episodes = ["fresh-a", second_episode]
    if failure == "missing-train-episode":
        mixture_episodes.append("not-sealed")
    if failure == "extra-sealed-episode":
        mixture_episodes = ["fresh-a"]
    mixture, _ = _mixture(tmp_path / "mixture", mixture_episodes)
    accepted_roots: list[Path] = [first[0], second[0]]
    receipt_roots: list[Path] = [first[1], second[1]]
    seals: list[Path] = [first[2], second[2]]
    if failure == "empty-sources":
        accepted_roots, receipt_roots, seals = [], [], []
    elif failure == "unequal-source-counts":
        receipt_roots.pop()
    elif failure == "duplicate-input-path":
        accepted_roots[1] = accepted_roots[0]
    elif failure in {"duplicate-train-window-id", "colliding-train-path"}:
        index_path = mixture.parent / "windows.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        rollout_windows = [window for window in index["windows"] if window["source_type"] == "rollout" and window["split"] == "train"]
        if failure == "duplicate-train-window-id":
            rollout_windows[1]["window_id"] = rollout_windows[0]["window_id"]
        else:
            rollout_windows[1]["source_locator"]["attempt_root"] = rollout_windows[0]["source_locator"]["attempt_root"]
        _write(index_path, index)
        manifest = json.loads(mixture.read_text(encoding="utf-8"))
        manifest["window_index"] = {
            "path": "windows.json",
            "sha256": _sha(index_path),
            "byte_size": index_path.stat().st_size,
        }
        _write(mixture, manifest)

    output = tmp_path / "should-not-exist.json"
    if failure == "output-inside-source":
        output = first[0] / "should-not-exist.json"
    with pytest.raises(ValueError):
        builder.build_awr_progress_evidence(
            mixture_manifest=mixture,
            accepted_root=accepted_roots,
            sync_receipts_root=receipt_roots,
            round_seal=seals,
            output=output,
        )
    assert not output.exists()
    assert not Path(str(output) + ".sha256").exists()


def test_multi_round_output_is_independent_of_directory_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    first = _sealed_round(
        tmp_path / "first",
        episode_ids=["fresh-b", "fresh-a"],
        round_id="balanced-fresh-round",
    )
    second = _sealed_round(
        tmp_path / "second",
        episode_ids=["replay-b", "replay-a"],
        round_id="success-replay-round",
    )
    mixture, _ = _mixture(tmp_path / "mixture", ["fresh-a", "fresh-b", "replay-a", "replay-b"])
    first_output = tmp_path / "first-evidence.json"
    second_output = tmp_path / "second-evidence.json"
    arguments = {
        "mixture_manifest": mixture,
        "accepted_root": [first[0], second[0]],
        "sync_receipts_root": [first[1], second[1]],
        "round_seal": [first[2], second[2]],
    }
    builder.build_awr_progress_evidence(output=first_output, **arguments)
    real_iterdir = builder.Path.iterdir

    def reverse_iterdir(path: Path):
        return iter(reversed(list(real_iterdir(path))))

    monkeypatch.setattr(builder.Path, "iterdir", reverse_iterdir)
    builder.build_awr_progress_evidence(output=second_output, **arguments)

    assert first_output.read_bytes() == second_output.read_bytes()


def test_cli_repeats_source_flags_but_preserves_single_source_form(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    builder = _load_builder()
    first = _sealed_round(tmp_path / "first", episode_ids=["fresh-a"], round_id="balanced-fresh-round")
    second = _sealed_round(tmp_path / "second", episode_ids=["replay-a"], round_id="success-replay-round")
    mixture, _ = _mixture(tmp_path / "mixture", ["fresh-a", "replay-a"])
    output = tmp_path / "multi.json"

    assert builder.main([
        "--mixture-manifest", str(mixture),
        "--accepted-root", str(first[0]), "--accepted-root", str(second[0]),
        "--sync-receipts-root", str(first[1]), "--sync-receipts-root", str(second[1]),
        "--round-seal", str(first[2]), "--round-seal", str(second[2]),
        "--output", str(output),
    ]) == 0
    multi_result = json.loads(capsys.readouterr().out)
    assert multi_result["episode_count"] == 2
    assert tuple(load_progress_evidence(
        output,
        expected_sha256=multi_result["evidence_sha256"],
        mixture_id=multi_result["mixture_id"],
        mixture_manifest_sha256=multi_result["mixture_manifest_sha256"],
    ).episodes) == ("fresh-a", "replay-a")

    single_mixture, _ = _mixture(tmp_path / "single-mixture", ["fresh-a"])
    single_output = tmp_path / "single.json"
    assert builder.main([
        "--mixture-manifest", str(single_mixture),
        "--accepted-root", str(first[0]),
        "--sync-receipts-root", str(first[1]),
        "--round-seal", str(first[2]),
        "--output", str(single_output),
    ]) == 0
    single_result = json.loads(capsys.readouterr().out)
    assert single_result["episode_count"] == 1
    assert tuple(load_progress_evidence(
        single_output,
        expected_sha256=single_result["evidence_sha256"],
        mixture_id=single_result["mixture_id"],
        mixture_manifest_sha256=single_result["mixture_manifest_sha256"],
    ).episodes) == ("fresh-a",)
