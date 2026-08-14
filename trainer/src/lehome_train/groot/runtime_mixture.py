"""Fail-closed runtime sampling for the Phase 1 GR00T multi-source mixture.

The immutable manifest is deliberately separate from local mount locations.
Nothing in this module materializes, slices, or re-encodes an episode: a window
only authorizes a range in an already hydrated full-episode source.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import subprocess
from typing import Any, Literal

from lehome_train.io import canonical_json_sha256, sha256_file

try:  # Torch is a runtime dependency of pinned GR00T, not of macOS import tests.
    from torch.utils.data import IterableDataset, get_worker_info
except ImportError:  # pragma: no cover - permits contract inspection without torch.
    class IterableDataset:  # type: ignore[no-redef]
        pass

    def get_worker_info() -> None:  # type: ignore[no-redef]
        return None


ACTION_HORIZON = 16
FPS = 30
CAMERAS = (
    "observation.images.top_rgb",
    "observation.images.left_rgb",
    "observation.images.right_rgb",
)
INSTRUCTION = "fold the garment on the table"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_TYPES = {"bc", "rollout", "dagger"}
APPROVED_MIXTURE_REPOSITORY = "ryanjin333/lehome-groot-n17-data"


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def _load_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}")
    return value


def _exact(value: Mapping[str, object], keys: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != keys:
        unknown = actual - keys
        missing = keys - actual
        detail = sorted(unknown or missing)[0]
        raise ValueError(f"{label} has unknown or missing field: {detail}")


def _relative(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts or "\\" in value:
        raise ValueError(f"{label} is unsafe")
    return value


def _digest(value: object, *, label: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer")
    return value


def _source_tree_entries(root: Path) -> list[dict[str, object]]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("mounted source root is unsafe")
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("mounted source contains symlink")
        if path.is_file():
            entries.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "byte_size": path.stat().st_size})
    return entries


def source_tree_sha256(root: str | os.PathLike[str]) -> str:
    """Stable full-tree identity used by the immutable source binding."""

    return canonical_json_sha256(_source_tree_entries(Path(root)))


@dataclass(frozen=True, slots=True)
class Source:
    source_id: str
    source_type: Literal["bc", "rollout", "dagger"]
    quota: int
    release_stage: str
    source_tree_sha256: str
    artifact_receipt_path: str
    artifact_receipt_sha256: str
    acceptance_receipt_path: str
    acceptance_receipt_sha256: str
    source_identity: dict[str, object]


@dataclass(frozen=True, slots=True)
class Window:
    window_id: str
    source_id: str
    source_type: Literal["bc", "rollout", "dagger"]
    source_episode_id: str
    start: int
    stop: int
    frame_ids: tuple[int, ...]
    lineage_id: str
    split: Literal["train", "validation"]
    source_locator: dict[str, object]


@dataclass(frozen=True, slots=True)
class MixtureManifest:
    repository: str
    revision: str
    safe_prefix: str
    sources: tuple[Source, ...]
    schedule_seed: int
    cycle_size: int
    window_index_path: str
    window_index_sha256: str
    window_index_byte_size: int
    normalization_path: str
    normalization_sha256: str
    normalization_byte_size: int
    raw: dict[str, object]

    @property
    def action_horizon(self) -> int:
        return ACTION_HORIZON


@dataclass(frozen=True, slots=True)
class RuntimeContract:
    manifest: MixtureManifest
    windows: tuple[Window, ...]
    mounts: dict[str, Path]
    normalization: dict[str, object]

    @property
    def training_windows(self) -> tuple[Window, ...]:
        return tuple(window for window in self.windows if window.split == "train")


def _validate_source_identity(kind: str, value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("source identity must be an object")
    if kind == "bc":
        _exact(value, {"prepared_manifest_path", "prepared_manifest_sha256", "action_source"}, label="BC source identity")
        _relative(value["prepared_manifest_path"], label="prepared manifest path")
        _digest(value["prepared_manifest_sha256"], label="prepared manifest hash")
        if value["action_source"] != "organizer_expert":
            raise ValueError("BC action source must be organizer expert")
    elif kind == "rollout":
        _exact(value, {"round_manifest_path", "round_manifest_sha256", "action_source"}, label="rollout source identity")
        _relative(value["round_manifest_path"], label="round manifest path")
        _digest(value["round_manifest_sha256"], label="round manifest hash")
        if value["action_source"] != "policy":
            raise ValueError("rollout action source must be policy")
    else:
        _exact(value, {"round_manifest_path", "round_manifest_sha256", "action_source"}, label="DAgger source identity")
        _relative(value["round_manifest_path"], label="DAgger manifest path")
        _digest(value["round_manifest_sha256"], label="DAgger manifest hash")
        if value["action_source"] != "organizer_expert":
            raise ValueError("DAgger action source is invalid")
    return dict(value)


def _parse_source(value: object) -> Source:
    if not isinstance(value, dict):
        raise ValueError("source must be an object")
    _exact(value, {
        "source_id", "source_type", "quota", "release_stage", "source_tree_sha256",
        "artifact_receipt_path", "artifact_receipt_sha256", "acceptance_receipt_path",
        "acceptance_receipt_sha256", "source_identity",
    }, label="source")
    source_id, kind = value["source_id"], value["source_type"]
    if type(source_id) is not str or not source_id or "/" in source_id or "\\" in source_id:
        raise ValueError("source ID is unsafe")
    if kind not in _SOURCE_TYPES:
        raise ValueError("source type is invalid")
    if value["release_stage"] != "seen":
        raise ValueError("unseen source is not eligible")
    return Source(
        source_id, kind, _integer(value["quota"], label="source quota"), str(value["release_stage"]),
        _digest(value["source_tree_sha256"], label="source tree hash"),
        _relative(value["artifact_receipt_path"], label="artifact receipt path"), _digest(value["artifact_receipt_sha256"], label="artifact receipt hash"),
        _relative(value["acceptance_receipt_path"], label="acceptance receipt path"), _digest(value["acceptance_receipt_sha256"], label="acceptance receipt hash"),
        _validate_source_identity(str(kind), value["source_identity"]),
    )


def _parse_window(value: object) -> Window:
    if not isinstance(value, dict):
        raise ValueError("window must be an object")
    _exact(value, {"window_id", "source_id", "source_type", "source_episode_id", "start", "stop", "frame_ids", "lineage_id", "split", "source_locator"}, label="window")
    kind = value["source_type"]
    if kind not in _SOURCE_TYPES or value["split"] not in {"train", "validation"}:
        raise ValueError("window type or split is invalid")
    for name in ("window_id", "source_id", "source_episode_id", "lineage_id"):
        if type(value[name]) is not str or not value[name]:
            raise ValueError(f"window {name} is invalid")
    start, stop = _integer(value["start"], label="window start"), _integer(value["stop"], label="window stop")
    frames = value["frame_ids"]
    if not isinstance(frames, list) or any(type(frame) is not int for frame in frames):
        raise ValueError("window frame IDs are invalid")
    if stop - start != ACTION_HORIZON or frames != list(range(start, stop)):
        raise ValueError("window must be one contiguous h16 range without tail overflow")
    identity = value["source_locator"]
    if not isinstance(identity, dict):
        raise ValueError("window source identity is invalid")
    expected = ({"episode_id", "prepared_manifest_path", "prepared_manifest_sha256"} if kind == "bc" else {"attempt_root", "attempt_manifest_path", "attempt_manifest_sha256"})
    _exact(identity, expected, label="window source locator")
    if kind == "bc":
        if identity["episode_id"] != value["source_episode_id"]:
            raise ValueError("BC episode locator drift")
        _relative(identity["prepared_manifest_path"], label="BC prepared manifest path")
        _digest(identity["prepared_manifest_sha256"], label="BC prepared manifest hash")
    else:
        _relative(identity["attempt_root"], label="raw attempt root")
        _relative(identity["attempt_manifest_path"], label="raw attempt manifest path")
        _digest(identity["attempt_manifest_sha256"], label="raw attempt manifest hash")
        if PurePosixPath(str(identity["attempt_manifest_path"])).parent.as_posix() != identity["attempt_root"]:
            raise ValueError("raw attempt root must equal manifest parent")
    return Window(str(value["window_id"]), str(value["source_id"]), kind, str(value["source_episode_id"]), start, stop, tuple(frames), str(value["lineage_id"]), value["split"], dict(identity))


def _manifest_digest_binding(document: dict[str, object]) -> str:
    binding = dict(document)
    index = dict(binding["window_index"])  # validated before this call
    index["sha256"] = ""
    index["byte_size"] = 0
    binding["window_index"] = index
    return canonical_json_sha256(binding)


def _parse_manifest(path: Path) -> MixtureManifest:
    document = _load_object(path, label="mixture manifest")
    required = {
        "schema_version", "kind", "repository", "revision", "safe_prefix", "sources", "camera_schema", "image_shape", "state_schema", "action_schema", "fps", "action_horizon", "instruction", "schedule_seed", "cycle_size", "mixture_normalization", "window_index",
    }
    if set(document) == required | {"self_sha256"}:
        declared = _digest(document["self_sha256"], label="manifest self hash")
        hashed = dict(document)
        hashed.pop("self_sha256")
        if declared != canonical_json_sha256(hashed):
            raise ValueError("manifest self hash mismatch")
    elif set(document) != required:
        _exact(document, required, label="mixture manifest")
    if document["schema_version"] != 2 or document["kind"] != "lehome_runtime_mixture":
        raise ValueError("mixture manifest version is unsupported")
    repository = document["repository"]
    if repository != APPROVED_MIXTURE_REPOSITORY:
        raise ValueError("repository is not an approved private repository")
    revision = document["revision"]
    if type(revision) is not str or not _REVISION.fullmatch(revision):
        raise ValueError("revision is floating or invalid")
    safe_prefix = _relative(document["safe_prefix"], label="safe prefix")
    sources_raw = document["sources"]
    if not isinstance(sources_raw, list):
        raise ValueError("sources must be an array")
    sources = tuple(_parse_source(item) for item in sources_raw)
    if len({source.source_id for source in sources}) != len(sources):
        raise ValueError("duplicate source IDs")
    if not sources or sum(source.quota for source in sources) == 0:
        raise ValueError("total training quota must be nonzero")
    if document["camera_schema"] != list(CAMERAS) or document["image_shape"] != [480, 640, 3]:
        raise ValueError("camera schema drift")
    if document["state_schema"] != {"dimension": 12, "storage": "absolute"} or document["action_schema"] != {"dimension": 12, "storage": "absolute"}:
        raise ValueError("state or action schema drift")
    if document["fps"] != FPS or document["action_horizon"] != ACTION_HORIZON or document["instruction"] != INSTRUCTION:
        raise ValueError("fixed modality contract drift")
    schedule_seed, cycle_size = _integer(document["schedule_seed"], label="schedule seed"), _integer(document["cycle_size"], label="cycle size", minimum=1)
    if cycle_size != sum(source.quota for source in sources):
        raise ValueError("quota and cycle size mismatch")
    quotas = {kind: sum(source.quota for source in sources if source.source_type == kind) for kind in _SOURCE_TYPES}
    if quotas != {"bc": 7, "rollout": 3, "dagger": 0}:
        raise ValueError("runtime quota contract must be exactly BC 7 rollout 3 dagger 0")
    normalization = document["mixture_normalization"]
    if not isinstance(normalization, dict):
        raise ValueError("mixture normalization binding is invalid")
    _exact(normalization, {"path", "sha256", "byte_size"}, label="mixture normalization binding")
    index = document["window_index"]
    if not isinstance(index, dict):
        raise ValueError("window index binding is invalid")
    _exact(index, {"path", "sha256", "byte_size"}, label="window index binding")
    return MixtureManifest(str(repository), str(revision), safe_prefix, sources, schedule_seed, cycle_size, _relative(index["path"], label="window index path"), _digest(index["sha256"], label="window index hash"), _integer(index["byte_size"], label="window index size"), _relative(normalization["path"], label="mixture normalization path"), _digest(normalization["sha256"], label="mixture normalization hash"), _integer(normalization["byte_size"], label="mixture normalization size"), document)


def _safe_file(root: Path, relative: str, *, label: str) -> Path:
    target = root / relative
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError, OSError) as error:
        raise ValueError(f"{label} is missing or unsafe") from error
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    return target


def _validate_mounts(path: Path, manifest: MixtureManifest) -> dict[str, Path]:
    document = _load_object(path, label="mount descriptor")
    _exact(document, {"schema_version", "mounts"}, label="mount descriptor")
    if document["schema_version"] != 1 or not isinstance(document["mounts"], list):
        raise ValueError("mount descriptor is invalid")
    required = {source.source_id: source for source in manifest.sources}
    mounts: dict[str, Path] = {}
    for entry in document["mounts"]:
        if not isinstance(entry, dict):
            raise ValueError("mount entry is invalid")
        _exact(entry, {"source_id", "root", "source_tree_sha256", "artifact_receipt_sha256"}, label="mount entry")
        source_id = entry["source_id"]
        if type(source_id) is not str or source_id not in required or source_id in mounts:
            raise ValueError("mounts have missing or extra source IDs")
        if type(entry["root"]) is not str or not Path(entry["root"]).is_absolute():
            raise ValueError("mount root must be absolute")
        source = required[source_id]
        if entry["source_tree_sha256"] != source.source_tree_sha256 or entry["artifact_receipt_sha256"] != source.artifact_receipt_sha256:
            raise ValueError("mount identity does not match immutable manifest")
        root = Path(entry["root"])
        if source_tree_sha256(root) != source.source_tree_sha256:
            raise ValueError("mounted source tree hash mismatch")
        receipt = _safe_file(root, source.artifact_receipt_path, label="artifact receipt")
        if sha256_file(receipt) != source.artifact_receipt_sha256:
            raise ValueError("artifact receipt drift")
        for relative, digest, label in ((source.acceptance_receipt_path, source.acceptance_receipt_sha256, "acceptance receipt"),):
            if sha256_file(_safe_file(root, relative, label=label)) != digest:
                raise ValueError(f"{label} drift")
        mounts[source_id] = root
    if set(mounts) != set(required):
        raise ValueError("mounts have missing or extra source IDs")
    return mounts


def load_runtime_contract(manifest_path: str | os.PathLike[str], mounts_path: str | os.PathLike[str]) -> RuntimeContract:
    """Authenticate immutable artifacts and operational mounts before any read."""

    manifest_file = Path(manifest_path)
    manifest = _parse_manifest(manifest_file)
    index_file = _safe_file(manifest_file.parent, manifest.window_index_path, label="window index")
    if index_file.stat().st_size != manifest.window_index_byte_size or sha256_file(index_file) != manifest.window_index_sha256:
        raise ValueError("window index hash or size mismatch")
    index = _load_object(index_file, label="window index")
    _exact(index, {"schema_version", "manifest_sha256", "windows"}, label="window index")
    if index["schema_version"] != 2 or index["manifest_sha256"] != _manifest_digest_binding(manifest.raw) or not isinstance(index["windows"], list):
        raise ValueError("window index immutable binding mismatch")
    windows = tuple(_parse_window(item) for item in index["windows"])
    if len({window.window_id for window in windows}) != len(windows):
        raise ValueError("duplicate window IDs")
    by_source = {source.source_id: source for source in manifest.sources}
    training_by_source: dict[str, int] = {source.source_id: 0 for source in manifest.sources}
    lineage_splits: dict[str, set[str]] = {}
    for window in windows:
        source = by_source.get(window.source_id)
        if source is None or source.source_type != window.source_type:
            raise ValueError("window source binding mismatch")
        lineage_splits.setdefault(window.lineage_id, set()).add(window.split)
        if window.split == "train":
            training_by_source[window.source_id] += 1
    if any(len(splits) > 1 for splits in lineage_splits.values()):
        raise ValueError("train validation lineage overlap")
    for source in manifest.sources:
        if source.quota and not training_by_source[source.source_id]:
            raise ValueError("quota source has no train windows")
    normalization = _safe_file(manifest_file.parent, manifest.normalization_path, label="mixture normalization")
    if normalization.stat().st_size != manifest.normalization_byte_size or sha256_file(normalization) != manifest.normalization_sha256:
        raise ValueError("mixture normalization hash or size mismatch")
    normalization_value = _load_object(normalization, label="mixture normalization")
    if normalization_value.get("train_only") is not True:
        raise ValueError("mixture normalization is not train-only")
    mounts = _validate_mounts(Path(mounts_path), manifest)
    for source in manifest.sources:
        identity = source.source_identity
        path_name = "prepared_manifest_path" if source.source_type == "bc" else "round_manifest_path"
        hash_name = "prepared_manifest_sha256" if source.source_type == "bc" else "round_manifest_sha256"
        if sha256_file(_safe_file(mounts[source.source_id], str(identity[path_name]), label="source identity manifest")) != identity[hash_name]:
            raise ValueError("source identity manifest drift")
    return RuntimeContract(manifest, windows, mounts, normalization_value)


@dataclass(frozen=True, slots=True)
class RuntimeSample:
    sample_id: str
    global_position: int
    source_id: str
    source_type: str
    window: Window


def _permutation(values: list[Any], seed: int, cycle: int, namespace: str) -> list[Any]:
    result = list(values)
    material = f"{seed}:{cycle}:{namespace}".encode("utf-8")
    random.Random(int.from_bytes(hashlib.sha256(material).digest()[:16], "big")).shuffle(result)
    return result


def _permuted_index(length: int, *, seed: int, cycle: int, namespace: str, ordinal: int) -> int:
    """O(1) affine permutation index; never copies/shuffles source windows."""
    if length <= 0:
        raise ValueError("source has no train windows")
    material = hashlib.sha256(f"{seed}:{cycle}:{namespace}".encode()).digest()
    offset = int.from_bytes(material[:8], "big") % length
    stride = int.from_bytes(material[8:16], "big") % length or 1
    while math.gcd(stride, length) != 1:
        stride = (stride + 1) % length or 1
    return (offset + ordinal * stride) % length


class RuntimeMixtureDataset(IterableDataset):
    """Deterministic infinite logical sample stream, partitioned by global position.

    Stock ``DataLoader`` workers are supported because workers independently
    take a stride of logical positions.  They do not independently shuffle or
    choose sources, so aggregating them preserves every complete 7/3 cycle.
    """

    def __init__(self, contract: RuntimeContract, *, processor: Callable[[Any], Any] | None = None, decoder: Callable[..., Any] | None = None, global_sample_offset: int = 0, limit: int | None = None, worker_id: int | None = None, worker_count: int | None = None, rank: int = 0, world_size: int = 1) -> None:
        if global_sample_offset < 0 or rank < 0 or world_size <= 0 or rank >= world_size:
            raise ValueError("invalid deterministic stream partition")
        if limit is not None and limit < 0:
            raise ValueError("limit must be nonnegative")
        self.contract, self.processor, self.decoder = contract, processor, decoder
        self.global_sample_offset, self.limit = global_sample_offset, limit
        self.explicit_worker_id, self.explicit_worker_count = worker_id, worker_count
        self.rank, self.world_size = rank, world_size
        self._train: dict[str, list[Window]] = {source.source_id: [] for source in contract.manifest.sources}
        for window in contract.training_windows:
            self._train[window.source_id].append(window)

    def _partition(self) -> tuple[int, int]:
        if self.explicit_worker_id is not None or self.explicit_worker_count is not None:
            if type(self.explicit_worker_id) is not int or type(self.explicit_worker_count) is not int or self.explicit_worker_count <= 0 or not 0 <= self.explicit_worker_id < self.explicit_worker_count:
                raise ValueError("invalid explicit worker partition")
            local_id, local_count = self.explicit_worker_id, self.explicit_worker_count
        else:
            worker = get_worker_info()
            local_id, local_count = (0, 1) if worker is None else (worker.id, worker.num_workers)
        return self.rank * local_count + local_id, self.world_size * local_count

    def _at(self, position: int) -> RuntimeSample:
        cycle, position_in_cycle = divmod(position, self.contract.manifest.cycle_size)
        slots = [source.source_id for source in self.contract.manifest.sources for _ in range(source.quota)]
        source_id = _permutation(slots, self.contract.manifest.schedule_seed, cycle, "slots")[position_in_cycle]
        # Occurrence index within this cycle drives the independently permuted source windows.
        occurrence = sum(item == source_id for item in _permutation(slots, self.contract.manifest.schedule_seed, cycle, "slots")[:position_in_cycle])
        choices = self._train[source_id]
        window = choices[_permuted_index(len(choices), seed=self.contract.manifest.schedule_seed, cycle=cycle, namespace=f"windows:{source_id}", ordinal=occurrence)]
        return RuntimeSample(f"{position}:{window.window_id}", position, source_id, window.source_type, window)

    def _render(self, sample: RuntimeSample) -> Any:
        if self.processor is None:
            return sample
        payload = RangeSourceLoader(self.contract, decoder=self.decoder).load(sample.window)
        return self.processor(pinned_processor_messages(payload))

    def __iter__(self) -> Iterator[Any]:
        partition_id, stride = self._partition()
        position = self.global_sample_offset + partition_id
        # ``limit`` is an exclusive global position, which lets a resumed
        # suffix be compared directly with an uninterrupted prefix in tests.
        stop = self.limit
        while stop is None or position < stop:
            yield self._render(self._at(position))
            position += stride

    def get_dataset_statistics(self) -> dict[str, object]:
        """Return authenticated train-only normalization metadata for GR00T callers."""
        return dict(self.contract.normalization["statistics"])  # pinned processor expects the statistics payload only

    def initial_actions(self) -> list[list[float]]:
        return [[0.0] * 12 for _ in range(ACTION_HORIZON)]

    def get_initial_actions(self) -> list[list[float]]:
        return self.initial_actions()


class PinnedGrootFrameDecoder:
    """Lazy production decoder boundary; tests inject a small deterministic decoder."""

    def __call__(self, video_path: Path, frame_ids: tuple[int, ...], fps: int) -> Any:
        try:
            import importlib
            module = importlib.import_module("gr00t.utils.video_utils")
            decode = getattr(module, "get_frames_by_indices")
        except (ImportError, AttributeError) as error:
            raise RuntimeError("pinned GR00T range frame decoder is unavailable") from error
        return decode(str(video_path), list(frame_ids), video_backend="torchcodec")


def pinned_processor_messages(payload: Mapping[str, object], *, backend: object | None = None) -> list[dict[str, object]]:
    """Build the exact N1.7 one-step processor message lazily."""
    if backend is None:
        import importlib
        import numpy as np

        types = importlib.import_module("gr00t.data.types")
        backend = types
    else:
        import numpy as np
    images = payload.get("images")
    state = payload.get("state")
    actions = payload.get("actions")
    if not isinstance(images, Mapping) or not isinstance(state, list) or not isinstance(actions, list):
        raise ValueError("invalid runtime VLA payload")
    if set(images) != {"top_rgb", "left_rgb", "right_rgb"} or len(state) != 12 or len(actions) != ACTION_HORIZON:
        raise ValueError("runtime VLA modality shape drift")
    state_value = np.asarray(state, dtype=np.float32)
    action_value = np.asarray(actions, dtype=np.float32)
    if state_value.shape != (12,) or action_value.shape != (ACTION_HORIZON, 12) or not np.isfinite(state_value).all() or not np.isfinite(action_value).all():
        raise ValueError("runtime VLA numeric drift")
    split = lambda array: {"left_arm": array[..., :5], "left_gripper": array[..., 5:6], "right_arm": array[..., 6:11], "right_gripper": array[..., 11:12]}
    data = backend.VLAStepData(images={key: [np.asarray(value)[0]] for key, value in images.items()}, states=split(state_value), actions=split(action_value), text=INSTRUCTION, embodiment=backend.EmbodimentTag.NEW_EMBODIMENT)
    return [{"type": backend.MessageType.EPISODE_STEP.value, "content": data}]


def _video_probe(path: Path, *, stop: int) -> None:
    try:
        completed = subprocess.run(("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,nb_frames", "-of", "json", str(path)), check=False, capture_output=True, text=True, timeout=15)
        data = json.loads(completed.stdout) if completed.returncode == 0 else {}
        stream = data.get("streams", [])[0]
        frames = int(stream["nb_frames"])
        rate = stream["avg_frame_rate"]
        numerator, denominator = (int(part) for part in rate.split("/"))
        if frames < stop or denominator == 0 or numerator / denominator != FPS:
            raise ValueError
    except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        raise ValueError("missing corrupt video or FPS drift") from error


class RangeSourceLoader:
    """Read exactly one authenticated h16 range from a full episode."""

    def __init__(self, contract: RuntimeContract, *, decoder: Callable[..., Any] | None = None) -> None:
        self.contract = contract
        self.decoder = decoder or PinnedGrootFrameDecoder()
        self._attempt_cache: OrderedDict[Path, None] = OrderedDict()
        self._video_cache: OrderedDict[tuple[Path, int], None] = OrderedDict()
        self._bc_rows_cache: OrderedDict[Path, list[Any]] = OrderedDict()
        self.cache_cap = 8

    def _cache(self, cache: OrderedDict[Any, Any], key: Any, value: Any) -> Any:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.cache_cap:
            cache.popitem(last=False)
        return value

    def _bc_rows(self, root: Path, window: Window) -> tuple[list[Any], list[Any], dict[str, Path]]:
        import pyarrow.parquet as pq

        source = next(item for item in self.contract.manifest.sources if item.source_id == window.source_id)
        prepared = _load_object(_safe_file(root, str(source.source_identity["prepared_manifest_path"]), label="prepared source manifest"), label="prepared source manifest")
        if prepared.get("fps") != FPS or prepared.get("fixed_language_instruction") != INSTRUCTION or prepared.get("future_actions", {}).get("horizon") != ACTION_HORIZON:
            raise ValueError("BC prepared source modality contract drift")
        episode = int(window.source_episode_id)
        parquet = _safe_file(root, f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet", label="BC episode parquet")
        try:
            rows = self._bc_rows_cache.get(parquet)
            if rows is None:
                rows = pq.read_table(parquet).to_pylist()
                self._cache(self._bc_rows_cache, parquet, rows)
            rows = rows[window.start:window.stop]
        except Exception as error:
            raise ValueError("invalid BC episode parquet") from error
        if len(rows) != ACTION_HORIZON or any(row.get("frame_index") != frame or row.get("episode_index") != episode for frame, row in zip(window.frame_ids, rows, strict=True)):
            raise ValueError("BC cross episode or short tail")
        videos = {camera: _safe_file(root, f"videos/chunk-{episode // 1000:03d}/{camera}/episode_{episode:06d}.mp4", label="BC video") for camera in CAMERAS}
        return [row.get("observation.state") for row in rows], [row.get("action") for row in rows], videos

    def _rollout_rows(self, root: Path, source: Source, window: Window) -> tuple[list[Any], list[Any], dict[str, Path]]:
        locator = window.source_locator
        attempt_root = _safe_file(root, str(locator["attempt_manifest_path"]), label="attempt manifest").parent
        attempt_path = _safe_file(root, str(locator["attempt_manifest_path"]), label="attempt manifest")
        episode = _load_object(attempt_path, label="attempt manifest")
        if sha256_file(attempt_path) != locator["attempt_manifest_sha256"] or episode.get("episode_id") != window.source_episode_id:
            raise ValueError("rollout attempt identity drift")
        identity = episode.get("identity")
        if not isinstance(identity, dict) or episode.get("accepted_success") is not True or episode.get("outcome") != "success" or episode.get("terminal_reason") != "success" or identity.get("release_stage") != "seen" or identity.get("instruction") != INSTRUCTION:
            raise ValueError("rollout attempt is not accepted successful seen policy data")
        if attempt_root not in self._attempt_cache:
            self._verify_checksums(attempt_root)
            self._cache(self._attempt_cache, attempt_root, None)
        annotations = _safe_file(attempt_root, "annotations.jsonl", label="rollout annotations")
        try:
            rows = [_strict_pairs(list(json.loads(line, object_pairs_hook=lambda pairs: pairs).items())) for line in annotations.read_text(encoding="utf-8").splitlines() if line]
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError) as error:
            raise ValueError("invalid rollout annotations") from error
        rows = rows[window.start:window.stop]
        if len(rows) != ACTION_HORIZON or any(row.get("step") != frame or row.get("action_source") != "policy" for frame, row in zip(window.frame_ids, rows, strict=True)):
            raise ValueError("rollout cross episode or short tail")
        videos = {camera: _safe_file(attempt_root, f"videos/{camera}_rgb.mp4", label="rollout video") for camera in ("top", "left", "right")}
        return [row.get("state") for row in rows], [row.get("action") for row in rows], {f"observation.images.{name}_rgb": path for name, path in videos.items()}

    def _verify_checksums(self, root: Path) -> None:
        sums = _safe_file(root, "SHA256SUMS.json", label="raw SHA256SUMS.json")
        receipt = _load_object(sums, label="raw SHA256SUMS.json")
        expected: dict[str, str] = {}
        for relative, identity in receipt.items():
            _relative(relative, label="raw checksum path")
            if not isinstance(identity, dict):
                raise ValueError("invalid raw SHA256SUMS.json")
            _exact(identity, {"sha256", "size"}, label="raw checksum identity")
            expected[relative] = _digest(identity["sha256"], label="raw checksum hash")
            if _integer(identity["size"], label="raw checksum size") != _safe_file(root, relative, label="raw checksum file").stat().st_size:
                raise ValueError("raw checksum size drift")
        actual = {path.relative_to(root).as_posix(): sha256_file(path) for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS.json"}
        if actual != expected:
            raise ValueError("raw checksum drift")

    def load(self, window: Window) -> dict[str, object]:
        source = next((item for item in self.contract.manifest.sources if item.source_id == window.source_id), None)
        if source is None or window.split != "train" or source.release_stage != "seen":
            raise ValueError("unseen or validation window cannot be iterated for training")
        root = self.contract.mounts[source.source_id]
        states, actions, videos = self._bc_rows(root, window) if source.source_type == "bc" else self._rollout_rows(root, source, window)
        for values, label in ((states, "state"), (actions, "action")):
            if any(not isinstance(row, list) or len(row) != 12 or any(type(number) not in (int, float) or not math.isfinite(float(number)) for number in row) for row in values):
                raise ValueError(f"{label} dimension or finite-value drift")
        for path in videos.values():
            key = (path, window.stop)
            if key not in self._video_cache:
                _video_probe(path, stop=window.stop)
                self._cache(self._video_cache, key, None)
        images = {camera: self.decoder(path, (window.start,), FPS) for camera, path in videos.items()}
        return {"images": {key.rsplit(".", 1)[-1]: value for key, value in images.items()}, "state": states[0], "actions": actions, "window_id": window.window_id}


def make_dataset_factory(*, mixture_manifest: str | os.PathLike[str], mounts_descriptor: str | os.PathLike[str], global_sample_offset: int = 0, decoder: Callable[..., Any] | None = None, expected_window_index: str | os.PathLike[str] | None = None) -> Callable[..., RuntimeMixtureDataset]:
    """Return the sole injected factory; arbitrary upstream args stay untouched."""

    def factory(*_args: Any, processor: Callable[[Any], Any] | None = None, **_kwargs: Any) -> RuntimeMixtureDataset:
        contract = load_runtime_contract(mixture_manifest, mounts_descriptor)
        if expected_window_index is not None:
            selected = (Path(mixture_manifest).parent / contract.manifest.window_index_path).resolve()
            if selected != Path(expected_window_index).resolve():
                raise ValueError("selected window index does not match immutable manifest")
        return RuntimeMixtureDataset(contract, processor=processor, decoder=decoder, global_sample_offset=global_sample_offset)

    return factory


def runtime_dataset_factory_class(*, mixture_manifest: str | os.PathLike[str], window_index: str | os.PathLike[str], mounts_descriptor: str | os.PathLike[str], global_sample_offset: int) -> type[object]:
    """Create the exact ``DatasetFactory(config).build(processor)`` replacement."""

    class RuntimeDatasetFactory:
        def __init__(self, config: object) -> None:
            self.config = config

        def build(self, processor: Any) -> tuple[RuntimeMixtureDataset, None]:
            factory = make_dataset_factory(mixture_manifest=mixture_manifest, mounts_descriptor=mounts_descriptor, global_sample_offset=global_sample_offset, expected_window_index=window_index)
            dataset = factory(processor=processor)
            processor.set_statistics(dataset.get_dataset_statistics(), override=True)
            return dataset, None

    return RuntimeDatasetFactory
