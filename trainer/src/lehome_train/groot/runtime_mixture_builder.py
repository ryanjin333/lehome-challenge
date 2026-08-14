"""Build the immutable, no-recut runtime 70/30 mixture contract.

This module deliberately consumes the already-frozen selection plan.  It never
copies, clips, or re-encodes source artifacts: windows authorize h16 reads from
the organizer or recorder trees at runtime.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterable, Mapping
import re

import pyarrow.parquet as pq

from lehome_train.groot.runtime_mixture import (
    ACTION_HORIZON, APPROVED_MIXTURE_REPOSITORY, CAMERAS, FPS, INSTRUCTION,
    source_tree_sha256,
)
from lehome_train.io import canonical_json_bytes, canonical_json_sha256, sha256_file


def _identity_processor(messages: object) -> object:
    """Pickle-safe loader-only processor used only by the Docker pilot."""
    return messages


def _load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is malformed") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _sha(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a SHA-256")
    return value


def _relative(value: object, label: str) -> str:
    if type(value) is not str or not value or value.startswith("/") or "\\" in value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{label} must be a safe relative path")
    return value


def _external_receipt(value: object, label: str) -> Path:
    if type(value) is not str or not value or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return Path(value)


def _source_publications(path: Path, roots: Mapping[str, Path]) -> dict[str, dict[str, object]]:
    """Authenticate separately published BC and rollout source trees.

    This intentionally accepts no all-purpose corrective prefix: every source
    names its own immutable repository revision, prefix, and readback receipt.
    """

    document = _load(path, "runtime mixture source publications")
    if set(document) != {"schema_version", "kind", "sources"} or document["schema_version"] != 1 or document["kind"] != "runtime_mixture_source_publications" or not isinstance(document["sources"], list):
        raise ValueError("runtime mixture source publications have an incompatible schema")
    result: dict[str, dict[str, object]] = {}
    expected = {"organizer": ("bc", "bc/full"), "rollout": ("rollout", None)}
    for item in document["sources"]:
        if not isinstance(item, dict) or set(item) != {"source_id", "source_type", "repository", "revision", "prefix", "readback_receipt_path", "readback_receipt_sha256"}:
            raise ValueError("runtime mixture source publication entry is malformed")
        source_id, source_type, prefix = item["source_id"], item["source_type"], item["prefix"]
        if source_id not in expected or source_id in result or source_type != expected[source_id][0] or item["repository"] != APPROVED_MIXTURE_REPOSITORY or type(item["revision"]) is not str or re.fullmatch(r"[0-9a-f]{40}", item["revision"]) is None or type(prefix) is not str:
            raise ValueError("runtime mixture source publication identity is invalid")
        expected_prefix = expected[source_id][1]
        if (expected_prefix is not None and prefix != expected_prefix) or (
            expected_prefix is None and re.fullmatch(r"rollouts/round-[1-9][0-9]*", prefix) is None
        ):
            raise ValueError("runtime mixture source publication prefix is invalid")
        receipt = _external_receipt(item["readback_receipt_path"], "runtime mixture source readback receipt path")
        if receipt.is_symlink() or not receipt.is_file() or sha256_file(receipt) != _sha(item["readback_receipt_sha256"], "runtime mixture source readback receipt hash"):
            raise ValueError("runtime mixture source publication readback receipt drift")
        value = _load(receipt, "runtime mixture source publication readback receipt")
        if set(value) != {"repository", "immutable_revision", "remote_prefix", "fresh_readback_verified", "tree_listing_verified"} or value.get("repository") != item["repository"] or value.get("immutable_revision") != item["revision"] or value.get("remote_prefix") != prefix or value.get("fresh_readback_verified") is not True or value.get("tree_listing_verified") is not True:
            raise ValueError("runtime mixture source publication readback is not authenticated")
        result[source_id] = dict(item)
    if set(result) != set(expected):
        raise ValueError("runtime mixture source publications are incomplete")
    return result


def _write(path: Path, value: object) -> None:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _finite_vector(row: object, *, label: str) -> list[float]:
    if not isinstance(row, list) or len(row) != 12:
        raise ValueError(f"{label} must be a finite 12D vector")
    values = [float(item) for item in row]
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in row):
        raise ValueError(f"{label} must be a finite 12D vector")
    return values


def _stats(rows: Iterable[list[float]], dimensions: int) -> dict[str, list[float]]:
    values = list(rows)
    if not values or any(len(row) != dimensions for row in values):
        raise ValueError("normalization has no consistent values")
    answer = {key: [] for key in ("min", "max", "mean", "std", "q01", "q99")}
    for dimension in range(dimensions):
        column = sorted(row[dimension] for row in values)
        mean = sum(column) / len(column)
        variance = sum((item - mean) ** 2 for item in column) / len(column)
        def quantile(probability: float) -> float:
            index = (len(column) - 1) * probability
            left, right = math.floor(index), math.ceil(index)
            return column[left] if left == right else column[left] * (right - index) + column[right] * (index - left)
        answer["min"].append(column[0]); answer["max"].append(column[-1])
        answer["mean"].append(mean); answer["std"].append(math.sqrt(variance))
        answer["q01"].append(quantile(.01)); answer["q99"].append(quantile(.99))
    return answer


def _joint_groups(statistics: Mapping[str, list[float]]) -> dict[str, dict[str, list[float]]]:
    """Split the 12-D LeHome vector into pinned GR00T joint-group statistics."""

    slices = {"left_arm": slice(0, 5), "left_gripper": slice(5, 6), "right_arm": slice(6, 11), "right_gripper": slice(11, 12)}
    return {
        group: {field: list(statistics[field][indices]) for field in ("min", "max", "mean", "std", "q01", "q99")}
        for group, indices in slices.items()
    }


def _accepted_from_campaign(receipt: Mapping[str, Any]) -> dict[str, str]:
    attempts = receipt.get("attempt_receipts")
    if not isinstance(attempts, list):
        raise ValueError("campaign receipt lacks canonical attempt ledger")
    accepted: dict[str, str] = {}
    for item in attempts:
        if not isinstance(item, dict):
            raise ValueError("campaign attempt receipt is malformed")
        attempt_id, episode_id = item.get("attempt_id"), item.get("episode_id")
        if type(attempt_id) is not str or type(episode_id) is not str or attempt_id != episode_id:
            raise ValueError("campaign attempt identity is malformed")
        if item.get("accepted_success") is True and item.get("release_stage") == "seen":
            accepted[attempt_id] = episode_id
    if not accepted:
        raise ValueError("campaign has no accepted seen rollouts")
    return accepted


def validate_plan_windows(plan: Mapping[str, Any], *, organizer_manifest: Path, accepted_rollouts: Mapping[str, str]) -> list[dict[str, Any]]:
    """Validate selection provenance and return unique non-overlapping windows.

    Repeated materialized flywheel windows are expected from the old physical
    oversampling snapshot.  Identical repeats collapse; a repeat with different
    split/source identity fails rather than silently changing the training set.
    """
    organizer = _load(organizer_manifest, "organizer manifest")
    train, validation = organizer.get("train_episode_ids"), organizer.get("validation_episode_ids")
    if not isinstance(train, list) or not isinstance(validation, list) or not all(type(item) is str for item in train + validation):
        raise ValueError("organizer split ledger is malformed")
    split_by_episode = {item: "train" for item in train} | {item: "validation" for item in validation}
    if len(split_by_episode) != len(train) + len(validation):
        raise ValueError("organizer train/validation split overlaps")
    selections = plan.get("selected_frame_ranges")
    if not isinstance(selections, list) or not selections:
        raise ValueError("mix plan has no selected frame ranges")
    unique: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for item in selections:
        if not isinstance(item, dict):
            raise ValueError("mix plan selection is malformed")
        kind, source_id, raw_id = item.get("source_kind"), item.get("source_episode_id"), item.get("raw_episode_id")
        start, stop, split = item.get("raw_frame_start"), item.get("raw_frame_stop"), item.get("split")
        if kind not in {"organizer", "flywheel"} or type(source_id) is not str or type(raw_id) is not str or type(start) is not int or type(stop) is not int or start < 0 or stop - start != ACTION_HORIZON or split not in {"train", "validation"}:
            raise ValueError("mix plan selection is not an exact h16 range")
        frames = item.get("raw_frame_ids")
        if frames != [str(number) for number in range(start, stop)]:
            raise ValueError("mix plan selection frame IDs drift")
        if kind == "organizer":
            # The frozen plan may reserve an original training lineage for its
            # mixture holdout, but it must never promote an original holdout.
            if source_id != raw_id or raw_id not in split_by_episode or (split == "train" and split_by_episode[raw_id] != "train"):
                raise ValueError("organizer selection crosses the authenticated split")
        else:
            if split != "train" or accepted_rollouts.get(raw_id) != raw_id:
                raise ValueError("rollout selection is not an accepted seen attempt")
        if kind == "organizer":
            source_start, source_stop = item.get("frame_start"), item.get("frame_stop")
            if type(source_start) is not int or type(source_stop) is not int or source_start < 0 or source_stop - source_start != ACTION_HORIZON:
                raise ValueError("organizer selection has an invalid source range")
            # A physical mixed snapshot repeated source ranges to express its
            # ratio.  Runtime retains one authorized raw h16 and the fixed 7/3
            # schedule supplies replacement without duplicating allowlist rows.
            key = (kind, raw_id, start, stop)
        else:
            # Flywheel materialization had physical oversampling; collapse only
            # exact reads from the same recorder trajectory.
            key = (kind, raw_id, start, stop)
        previous = unique.get(key)
        if previous is not None and (previous.get("split") != split or previous.get("source_manifest_sha256") != item.get("source_manifest_sha256")):
            raise ValueError("oversample duplicate conflicts with immutable selection")
        unique[key] = dict(item)
    return [unique[key] for key in sorted(unique)]


def _raw_attempt(root: Path, attempt_id: str, expected_checksum: str) -> tuple[list[list[float]], list[list[float]]]:
    attempt = root / "raw" / attempt_id
    sums = attempt / "SHA256SUMS.json"
    episode = _load(attempt / "episode.json", "raw episode")
    if sha256_file(sums) != expected_checksum or episode.get("accepted_success") is not True or episode.get("outcome") != "success" or episode.get("terminal_reason") != "success" or not isinstance(episode.get("identity"), dict) or episode["identity"].get("release_stage") != "seen":
        raise ValueError("raw rollout acceptance or checksum binding drift")
    rows: list[dict[str, Any]] = []
    for line in (attempt / "annotations.jsonl").read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("raw rollout annotation is malformed")
        rows.append(value)
    return ([_finite_vector(row.get("state"), label="rollout state") for row in rows], [_finite_vector(row.get("action"), label="rollout action") for row in rows])


def _normalization_statistics(windows: list[dict[str, Any]], *, organizer_root: Path, campaign_root: Path, selected: Mapping[str, str]) -> dict[str, Any]:
    states: list[list[float]] = []; actions: list[list[float]] = []; relative: dict[str, list[list[float]]] = {"left_arm": [], "left_gripper": [], "right_arm": [], "right_gripper": []}
    cache: dict[tuple[str, str], tuple[list[list[float]], list[list[float]]]] = {}
    for window in windows:
        if window["split"] != "train":
            continue
        kind, episode_id, start, stop = window["source_kind"], window["raw_episode_id"], window["raw_frame_start"], window["raw_frame_stop"]
        key = (kind, episode_id)
        if key not in cache:
            if kind == "organizer":
                episode = int(episode_id); table = pq.read_table(organizer_root / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet", columns=["observation.state", "action"])
                cache[key] = ([_finite_vector(value, label="BC state") for value in table["observation.state"].to_pylist()], [_finite_vector(value, label="BC action") for value in table["action"].to_pylist()])
            else:
                cache[key] = _raw_attempt(campaign_root, episode_id, selected[episode_id])
        state_rows, action_rows = cache[key]
        if stop > len(state_rows) or stop > len(action_rows):
            raise ValueError("selected range overflows authenticated episode")
        states.extend(state_rows[start:stop]); actions.extend(action_rows[start:stop])
        current = state_rows[start]
        for action in action_rows[start:stop]:
            relative["left_arm"].append([action[index] - current[index] for index in range(0, 5)])
            relative["left_gripper"].append([action[5]])
            relative["right_arm"].append([action[index] - current[index] for index in range(6, 11)])
            relative["right_gripper"].append([action[11]])
    return {"new_embodiment": {"state": _joint_groups(_stats(states, 12)), "action": _joint_groups(_stats(actions, 12)), "relative_action": {name: _stats(rows, 5 if name.endswith("arm") else 1) for name, rows in relative.items()}}}


def build_runtime_mixture(*, organizer_root: str | Path, campaign_root: str | Path, source_publications: str | Path, selected_bindings: str | Path, plan_state: str | Path, destination: str | Path) -> dict[str, Any]:
    """Create immutable local publication-pending bytes, never an authorized run."""
    organizer, campaign, destination = Path(organizer_root), Path(campaign_root), Path(destination)
    if Path(selected_bindings).name != "selected-150.json" or Path(source_publications).name != "source-publications.json":
        raise ValueError("runtime mixture requires explicit selected-150.json and source-publications.json")
    state = _load(Path(plan_state), "persisted mix-plan state"); plan = state.get("plan")
    if not isinstance(plan, dict) or plan.get("sha256") != state.get("plan_sha256") or canonical_json_sha256({key: value for key, value in plan.items() if key != "sha256"}) != plan.get("sha256"):
        raise ValueError("persisted plan hash is invalid")
    organizer_manifest = organizer / "manifest.json"; organizer_hash = sha256_file(organizer_manifest)
    campaign_receipt = campaign / "campaign-receipt.json"; receipt = _load(campaign_receipt, "campaign receipt")
    selected_document = _load(Path(selected_bindings), "selected rollout bindings")
    selected_rows = selected_document.get("selected_bindings")
    if not isinstance(selected_rows, list):
        raise ValueError("selected rollout bindings are malformed")
    selected = {item.get("attempt_id"): item.get("episode_manifest_sha256") for item in selected_rows if isinstance(item, dict) and type(item.get("attempt_id")) is str and type(item.get("episode_manifest_sha256")) is str}
    accepted = _accepted_from_campaign(receipt)
    if not set(selected).issubset(accepted) or len(selected) != len(selected_rows):
        raise ValueError("selected rollout bindings are not an accepted seen allowlist")
    windows = validate_plan_windows(plan, organizer_manifest=organizer_manifest, accepted_rollouts={key: key for key in selected})
    if destination.exists():
        raise FileExistsError("runtime mixture destination is immutable; choose an explicit new destination")
    publications = _source_publications(
        Path(source_publications), {"organizer": organizer, "rollout": campaign}
    )
    staging = destination.parent / f".{destination.name}.{plan['sha256'][:12]}.tmp"
    if staging.exists():
        raise FileExistsError("runtime mixture staging already exists")
    try:
        staging.mkdir(parents=True)
        statistics = _normalization_statistics(windows, organizer_root=organizer, campaign_root=campaign, selected=selected)
        source_entries = [
            {"source_id": "organizer", "source_type": "bc", "quota": 7, "release_stage": "seen", "source_tree_sha256": source_tree_sha256(organizer), "artifact_receipt_path": "manifest.json", "artifact_receipt_sha256": organizer_hash, "acceptance_receipt_path": "manifest.json", "acceptance_receipt_sha256": organizer_hash, "publication": {key: publications["organizer"][key] for key in ("repository", "revision", "prefix", "readback_receipt_path", "readback_receipt_sha256")}, "source_identity": {"prepared_manifest_path": "manifest.json", "prepared_manifest_sha256": organizer_hash, "action_source": "organizer_expert"}},
            {"source_id": "rollout", "source_type": "rollout", "quota": 3, "release_stage": "seen", "source_tree_sha256": source_tree_sha256(campaign), "artifact_receipt_path": "campaign-receipt.json", "artifact_receipt_sha256": sha256_file(campaign_receipt), "acceptance_receipt_path": "campaign-receipt.json", "acceptance_receipt_sha256": sha256_file(campaign_receipt), "publication": {key: publications["rollout"][key] for key in ("repository", "revision", "prefix", "readback_receipt_path", "readback_receipt_sha256")}, "source_identity": {"round_manifest_path": "campaign-receipt.json", "round_manifest_sha256": sha256_file(campaign_receipt), "action_source": "policy"}},
        ]
        converted = []
        for number, item in enumerate(windows):
            kind = "bc" if item["source_kind"] == "organizer" else "rollout"; episode = item["raw_episode_id"]
            locator = ({"episode_id": episode, "prepared_manifest_path": "manifest.json", "prepared_manifest_sha256": organizer_hash} if kind == "bc" else {"attempt_root": f"raw/{episode}", "attempt_manifest_path": f"raw/{episode}/episode.json", "attempt_manifest_sha256": sha256_file(campaign / "raw" / episode / "episode.json")})
            converted.append({"window_id": f"{kind}-{number:06d}", "source_id": "organizer" if kind == "bc" else "rollout", "source_type": kind, "source_episode_id": episode, "start": item["raw_frame_start"], "stop": item["raw_frame_stop"], "frame_ids": list(range(item["raw_frame_start"], item["raw_frame_stop"])), "lineage_id": f"{kind}:{episode}", "split": item["split"], "source_locator": locator})
        normalization = {"schema_version": 3, "train_only": True, "derivation": {"train_window_ids": [item["window_id"] for item in converted if item["split"] == "train"], "sample_count": sum(item["split"] == "train" for item in converted) * ACTION_HORIZON}, "statistics": statistics}
        _write(staging / "mixture-normalization.json", normalization)
        # This is a deterministic publication target, not proof of publication.
        # It deliberately is not a loadable runtime manifest: a publisher must
        # upload these bytes and produce an immutable mixture readback receipt.
        mixture_id = canonical_json_sha256({"plan_sha256": plan["sha256"], "sources": source_entries, "windows": converted, "normalization": normalization})
        _write(staging / "windows.json", {"schema_version": 3, "windows": converted})
        pending = {"schema_version": 1, "kind": "runtime_mixture_publication_pending", "repository": APPROVED_MIXTURE_REPOSITORY, "mixture_id": mixture_id, "prefix": f"mixtures/{mixture_id}", "sources": source_entries, "normalization_sha256": sha256_file(staging / "mixture-normalization.json"), "windows_sha256": sha256_file(staging / "windows.json"), "publication_pending": True}
        _write(staging / "publication-pending.json", pending)
        receipt_value = {"schema_version": 3, "kind": "runtime_mixture_generation", "publication_pending": True, "plan_sha256": plan["sha256"], "mixture_id": mixture_id, "prefix": pending["prefix"], "unique_windows": len(converted), "train_windows": sum(item["split"] == "train" for item in converted), "bc_train_windows": sum(item["source_type"] == "bc" and item["split"] == "train" for item in converted), "bc_validation_windows": sum(item["source_type"] == "bc" and item["split"] == "validation" for item in converted), "rollout_train_windows": sum(item["source_type"] == "rollout" and item["split"] == "train" for item in converted), "source_readback": {"source_publications_sha256": sha256_file(source_publications), "selected_bindings_sha256": sha256_file(selected_bindings)}}
        _write(staging / "generation-receipt.json", receipt_value)
        os.replace(staging, destination)
        return receipt_value
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_from_request(path: str | Path) -> dict[str, Any]:
    """Run the builder from its intentionally narrow, strict JSON envelope."""
    request = _load(Path(path), "runtime mixture build request")
    required = {"schema_version", "command", "arguments"}
    if set(request) != required or request["schema_version"] != 1 or request["command"] != "build-runtime-mixture" or not isinstance(request["arguments"], dict):
        raise ValueError("runtime mixture build request has an incompatible schema")
    arguments = request["arguments"]
    expected = {"organizer_root", "campaign_root", "source_publications", "selected_bindings", "plan_state", "destination"}
    if set(arguments) != expected or not all(type(arguments[key]) is str and arguments[key] for key in expected):
        raise ValueError("runtime mixture build request arguments are incomplete or unknown")
    return build_runtime_mixture(**arguments)


def pilot_from_request(path: str | Path) -> dict[str, Any]:
    """Run a CPU-only deterministic loader pilot; it never imports a model."""
    request = _load(Path(path), "runtime mixture pilot request")
    if set(request) != {"schema_version", "command", "arguments"} or request.get("schema_version") != 1 or request.get("command") != "pilot-runtime-mixture" or not isinstance(request.get("arguments"), dict):
        raise ValueError("runtime mixture pilot request has an incompatible schema")
    arguments = request["arguments"]
    expected = {"mixture_manifest", "mounts_descriptor", "sample_count", "worker_counts"}
    if set(arguments) != expected or type(arguments.get("mixture_manifest")) is not str or type(arguments.get("mounts_descriptor")) is not str or type(arguments.get("sample_count")) is not int or arguments["sample_count"] < 100 or arguments.get("worker_counts") != [0, 1, 4]:
        raise ValueError("runtime mixture pilot request arguments are incomplete or unknown")
    from lehome_train.groot.runtime_mixture import RangeSourceLoader, RuntimeMixtureDataset, load_runtime_contract

    contract = load_runtime_contract(arguments["mixture_manifest"], arguments["mounts_descriptor"])
    train = contract.training_windows
    bc = next((window for window in train if window.source_type == "bc"), None)
    rollout = next((window for window in train if window.source_type == "rollout"), None)
    if bc is None or rollout is None:
        raise ValueError("pilot requires both BC and rollout training windows")
    loader = RangeSourceLoader(contract)
    # Representative real decodes catch camera/h16 contract drift before timing.
    loader.load(bc); loader.load(rollout)
    timings: dict[str, dict[str, float | int]] = {}
    try:
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise RuntimeError("pilot must run inside the pinned trainer Docker image") from error
    for requested_workers in arguments["worker_counts"]:
        start = time.perf_counter()
        dataset = RuntimeMixtureDataset(contract, processor=_identity_processor, limit=arguments["sample_count"])
        loader_kwargs: dict[str, Any] = {"batch_size": None, "num_workers": requested_workers}
        if requested_workers:
            loader_kwargs["prefetch_factor"] = 2
        iterator = iter(DataLoader(dataset, **loader_kwargs))
        decoded = sum(1 for _ in range(arguments["sample_count"]) if next(iterator) is not None)
        elapsed = time.perf_counter() - start
        timings[str(requested_workers)] = {"seconds": elapsed, "decoded_samples": decoded, "samples_per_second": decoded / elapsed if elapsed else 0.0, "prefetch_factor": 2 if requested_workers else 0}
    return {"schema_version": 1, "kind": "runtime_mixture_loader_pilot", "model_loaded": False, "processor_contract": "pinned_processor_integration_required", "representative": {"bc_window_id": bc.window_id, "rollout_window_id": rollout.window_id, "three_cameras": True, "action_horizon": ACTION_HORIZON}, "sample_count_per_worker": arguments["sample_count"], "worker_counts": arguments["worker_counts"], "loader_throughput": timings, "cache_cap": loader.cache_cap, "native_x86_required": True, "throughput_verified": False}
