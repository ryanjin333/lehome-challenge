"""Read one prepared LeRobot v2.1 episode without converting the dataset.

The rollout image carries LeRobot v3, which intentionally refuses to open a
v2.1 dataset.  Training data in this repository is already materialized as one
Parquet file per episode, so replay only needs a small read-only adapter for
the requested episode rather than a second copy of the full dataset.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping


_V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"


@dataclass(frozen=True)
class GarmentEpisodeIdentity:
    garment_name: str
    garment_episode_index: int


class PreparedV21ReplayDataset:
    """Minimal dataset surface consumed by the simulator replay command."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        info = _json_object(self.root / "meta" / "info.json")
        if info.get("codebase_version") != "v2.1":
            raise ValueError("prepared replay adapter requires LeRobot v2.1")
        if info.get("data_path") != _V21_DATA_PATH:
            raise ValueError("prepared replay adapter requires per-episode Parquet")
        features = info.get("features")
        if not isinstance(features, dict) or "action" not in features:
            raise ValueError("prepared replay dataset is missing action metadata")
        self.num_episodes = _positive_int(info.get("total_episodes"), "total_episodes")
        self.num_frames = _positive_int(info.get("total_frames"), "total_frames")
        self.fps = float(info.get("fps", 0))
        if self.fps <= 0:
            raise ValueError("prepared replay dataset fps must be positive")
        self.meta = SimpleNamespace(features=features)

    @staticmethod
    def supports(root: str | Path) -> bool:
        try:
            info = _json_object(Path(root) / "meta" / "info.json")
        except (FileNotFoundError, ValueError):
            return False
        return (
            info.get("codebase_version") == "v2.1"
            and info.get("data_path") == _V21_DATA_PATH
        )

    def episode_path(self, episode_index: int) -> Path:
        if type(episode_index) is not int or not 0 <= episode_index < self.num_episodes:
            raise IndexError(f"episode index is out of range: {episode_index}")
        path = self.root / _V21_DATA_PATH.format(
            episode_chunk=episode_index // 1000,
            episode_index=episode_index,
        )
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError as error:
            raise ValueError("prepared replay episode path escapes dataset root") from error
        if not path.is_file():
            raise FileNotFoundError(f"prepared replay episode is missing: {path}")
        return path

    def load_episode_rows(self, episode_index: int) -> list[dict[str, Any]]:
        # PyArrow is deliberately lazy: normal rollout workers do not need it.
        import pyarrow.parquet as pq

        table = pq.read_table(
            self.episode_path(episode_index),
            columns=["observation.state", "action"],
        )
        if table.num_rows < 1:
            raise ValueError(f"prepared replay episode {episode_index} is empty")
        rows = table.to_pylist()
        if any(
            not isinstance(row, dict)
            or not isinstance(row.get("observation.state"), list)
            or not isinstance(row.get("action"), list)
            for row in rows
        ):
            raise ValueError("prepared replay episode has invalid state/action rows")
        return rows


def load_garment_episode_identity(
    garment_index_path: str | Path, episode_index: int
) -> GarmentEpisodeIdentity:
    """Resolve the garment and its local reset-pose ordinal for a BC episode."""

    document = _json_object(Path(garment_index_path))
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "lehome_bc_garment_index"
        or not isinstance(document.get("episodes"), list)
    ):
        raise ValueError("invalid BC garment index")

    target = str(episode_index)
    garments: dict[str, str] = {}
    for row in document["episodes"]:
        if not isinstance(row, Mapping):
            raise ValueError("invalid BC garment index row")
        episode_id = row.get("episode_id")
        garment_name = row.get("garment_name")
        if (
            not isinstance(episode_id, str)
            or not episode_id.isdigit()
            or not isinstance(garment_name, str)
            or not garment_name
            or episode_id in garments
        ):
            raise ValueError("invalid BC garment index row")
        garments[episode_id] = garment_name

    try:
        garment_name = garments[target]
    except KeyError as error:
        raise ValueError(f"BC garment index does not cover episode {episode_index}") from error
    local_index = sum(
        1
        for episode_id, candidate in garments.items()
        if int(episode_id) < episode_index and candidate == garment_name
    )
    return GarmentEpisodeIdentity(garment_name, local_index)


def configure_replay_timing(
    env_cfg: Any,
    *,
    render_every_actions: int,
    headless: bool,
) -> int:
    """Apply action cadence and safe sparse-render reset settings."""

    if type(render_every_actions) is not int or render_every_actions < 1:
        raise ValueError("render_every_actions must be a positive integer")
    action_decimation = env_cfg.decimation
    if type(action_decimation) is not int or action_decimation < 1:
        raise ValueError("environment decimation must be a positive integer")
    env_cfg.decimation = action_decimation
    env_cfg.sim.render_interval = action_decimation * render_every_actions
    if headless:
        env_cfg.wait_for_textures = False
    return action_decimation


@contextmanager
def suppress_unused_observations(env: Any, *, enabled: bool) -> Iterator[None]:
    """Avoid camera readback when action-only replay discards observations."""

    if not enabled:
        yield
        return
    original = env._get_observations
    env._get_observations = lambda: {}
    try:
        yield
    finally:
        env._get_observations = original


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value
