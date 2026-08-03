"""Canonical, deterministic public GR00T evaluation matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


CATEGORY_PREFIX = {
    "top_long": "Top_Long",
    "top_short": "Top_Short",
    "pant_long": "Pant_Long",
    "pant_short": "Pant_Short",
}
SEEN_SEEDS = (101, 211, 307, 401, 503)
UNSEEN_SEEDS = (601, 607, 613, 617, 619, 631, 641, 643, 647, 653)
PUBLIC_UNSEEN_HOLDOUTS = (
    "Top_Long_Unseen_1",
    "Top_Short_Unseen_1",
    "Pant_Long_Unseen_1",
    "Pant_Short_Unseen_1",
)


@dataclass(frozen=True, slots=True)
class Trial:
    category: str
    garment_name: str
    release_stage: str
    seed: int

    @property
    def trial_id(self) -> str:
        garment_index = self.garment_name.rsplit("_", 1)[-1]
        return f"{self.category.replace('_', '-')}-{self.release_stage.replace('_', '-')}-{garment_index}-seed-{self.seed}"

    def to_dict(self) -> dict[str, object]:
        return {"trial_id": self.trial_id, **asdict(self)}


@dataclass(frozen=True, slots=True)
class PublicMatrix:
    schema_version: int
    trials: tuple[Trial, ...]
    training_holdouts: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "training_holdouts": list(self.training_holdouts),
            "trials": [trial.to_dict() for trial in self.trials],
        }


def build_public_matrix() -> PublicMatrix:
    """Build the fixed 200 seen / 80 public-unseen evaluation assignments."""
    trials: list[Trial] = []
    for category, prefix in CATEGORY_PREFIX.items():
        for index in range(10):
            for seed in SEEN_SEEDS:
                trials.append(Trial(category, f"{prefix}_Seen_{index}", "seen", seed))
        for index in range(2):
            for seed in UNSEEN_SEEDS:
                trials.append(Trial(category, f"{prefix}_Unseen_{index}", "public_unseen", seed))
    return PublicMatrix(
        schema_version=1,
        trials=tuple(trials),
        training_holdouts=PUBLIC_UNSEEN_HOLDOUTS,
    )


def canonical_matrix_json(matrix: PublicMatrix) -> str:
    """Serialize the matrix in one stable byte representation for commits and hashes."""
    return json.dumps(matrix.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def matrix_sha256(matrix: PublicMatrix) -> str:
    return hashlib.sha256(canonical_matrix_json(matrix).encode("utf-8")).hexdigest()


def load_public_matrix(path: Path) -> PublicMatrix:
    """Load only the byte-for-byte canonical public matrix contract."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"canonical public matrix is missing: {path}")
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read canonical public matrix: {path}") from error
    matrix = build_public_matrix()
    if contents != canonical_matrix_json(matrix):
        raise ValueError(f"matrix does not match the committed canonical public contract: {path}")
    return matrix


def _read_release_names(path: Path) -> frozenset[str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required release asset list is missing: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"release asset list contains duplicate names: {path}")
    return frozenset(names)


def validate_release_assets(release_root: Path, matrix: PublicMatrix) -> None:
    """Require every generated garment to be listed in the immutable Release assets."""
    if release_root.is_symlink() or not release_root.is_dir():
        raise ValueError(f"required Release asset directory is missing: {release_root}")
    for category, prefix in CATEGORY_PREFIX.items():
        names = _read_release_names(release_root / prefix / f"{prefix}.txt")
        requested = {trial.garment_name for trial in matrix.trials if trial.category == category}
        missing = requested - names
        if missing:
            raise ValueError(f"generated release assets are missing for {category}: {sorted(missing)}")


def write_public_matrix(output: Path, matrix: PublicMatrix) -> None:
    """Create a matrix once; never silently rewrite a committed contract."""
    payload = canonical_matrix_json(matrix)
    if output.exists() or output.is_symlink():
        existing = output.read_text(encoding="utf-8") if output.is_file() else ""
        if existing != payload:
            raise ValueError(f"refusing to overwrite differing matrix: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
