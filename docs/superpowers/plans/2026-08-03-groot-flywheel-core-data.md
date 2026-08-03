# GR00T Flywheel Core Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the fail-closed raw-episode, expert-export, evaluation-matrix, and hard-mining contracts used by every later flywheel component.

**Architecture:** A dependency-light `lehome.flywheel` package owns immutable dataclasses, validation, atomic artifact finalization, expert-only selection, deterministic matrix generation, and hard-case ranking. Isaac Sim adapters and Hub transports consume these pure interfaces later; they do not redefine the data contract.

**Tech Stack:** Python 3.11, dataclasses, NumPy, JSON/JSONL, SHA-256, pytest.

---

## File structure

- Create `source/lehome/lehome/flywheel/__init__.py`: public core API.
- Create `source/lehome/lehome/flywheel/models.py`: episode, frame, provenance, outcome, and quality types.
- Create `source/lehome/lehome/flywheel/artifacts.py`: atomic episode writer and checksum manifest.
- Create `source/lehome/lehome/flywheel/export.py`: expert-only 16-step BC-window selector.
- Create `source/lehome/lehome/flywheel/matrix.py`: deterministic 280-trial public matrix and holdout rules.
- Create `source/lehome/lehome/flywheel/hard_mining.py`: diagnostic failure ranking.
- Create `scripts/build_groot_flywheel_matrix.py`: matrix CLI.
- Create `tests/flywheel/`: pure-Python contract tests.

### Task 1: Immutable episode and provenance schema

**Files:**
- Create: `source/lehome/lehome/flywheel/__init__.py`
- Create: `source/lehome/lehome/flywheel/models.py`
- Test: `tests/flywheel/test_models.py`

- [ ] **Step 1: Write the failing schema tests**

```python
from dataclasses import replace
import pytest

from lehome.flywheel.models import ActionSource, EpisodeFrame, EpisodeIdentity


def test_episode_identity_requires_pinned_artifacts() -> None:
    identity = EpisodeIdentity(
        episode_id="01JTEST0000000000000000000",
        policy_repo="ryanjin333/lehome-groot-n17-policy",
        policy_revision="a" * 40,
        policy_step=12000,
        code_revision="b" * 40,
        asset_revision="c" * 40,
        simulator_version="5.1.0",
        garment_name="Pant_Long_Seen_0",
        category="pant_long",
        release_stage="seen",
        seed=42,
        instruction="fold the garment on the table",
        strategy="canonical",
    )
    assert identity.policy_step == 12000
    with pytest.raises(ValueError, match="40-character"):
        replace(identity, policy_revision="main")


def test_frame_rejects_nonfinite_or_wrong_dimension_actions() -> None:
    with pytest.raises(ValueError, match="12 finite"):
        EpisodeFrame(
            step=0,
            monotonic_ns=1,
            wall_time_ns=2,
            state=(0.0,) * 12,
            action=(0.0,) * 11,
            action_source=ActionSource.EXPERT,
            reward=0.0,
            success=False,
            segment=1,
        )
```

- [ ] **Step 2: Run the test and verify the missing module failure**

Run: `uv run --offline pytest tests/flywheel/test_models.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'lehome.flywheel'`.

- [ ] **Step 3: Implement strict frozen types and validators**

```python
# source/lehome/lehome/flywheel/models.py
from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
import math
import re

PINNED = re.compile(r"^[0-9a-f]{40}$")
CATEGORIES = frozenset({"top_long", "top_short", "pant_long", "pant_short"})


class ActionSource(StrEnum):
    POLICY = "policy"
    EXPERT = "expert"
    HOLD = "hold"


class QualityGrade(StrEnum):
    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True, slots=True)
class EpisodeIdentity:
    episode_id: str
    policy_repo: str
    policy_revision: str
    policy_step: int
    code_revision: str
    asset_revision: str
    simulator_version: str
    garment_name: str
    category: str
    release_stage: str
    seed: int
    instruction: str
    strategy: str

    def __post_init__(self) -> None:
        for name in ("policy_revision", "code_revision", "asset_revision"):
            if not PINNED.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a pinned 40-character revision")
        if self.category not in CATEGORIES:
            raise ValueError("unsupported garment category")
        if self.release_stage not in {"seen", "public_unseen"}:
            raise ValueError("unsupported release stage")
        if self.strategy not in {"canonical", "mild", "strong"}:
            raise ValueError("unsupported collection strategy")


@dataclass(frozen=True, slots=True)
class EpisodeFrame:
    step: int
    monotonic_ns: int
    wall_time_ns: int
    state: tuple[float, ...]
    action: tuple[float, ...]
    action_source: ActionSource
    reward: float
    success: bool
    segment: int
    policy_request_id: str | None = None
    policy_chunk_offset: int | None = None
    expert_sequence: int | None = None
    expert_sample_age_ms: float | None = None

    def __post_init__(self) -> None:
        if len(self.state) != 12 or len(self.action) != 12:
            raise ValueError("state and action must contain 12 finite values")
        if not all(math.isfinite(v) for v in (*self.state, *self.action, self.reward)):
            raise ValueError("state and action must contain 12 finite values")
```

Export these names from `source/lehome/lehome/flywheel/__init__.py` and add `EpisodeOutcome`, `RandomizationRecord`, and rejection-reason fields using the same frozen, fail-closed pattern.

- [ ] **Step 4: Run the schema tests**

Run: `uv run --offline pytest tests/flywheel/test_models.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the schema boundary**

```bash
git add source/lehome/lehome/flywheel/__init__.py source/lehome/lehome/flywheel/models.py tests/flywheel/test_models.py
git commit -m "feat: define flywheel episode contracts"
```

### Task 2: Atomic raw episode artifacts

**Files:**
- Create: `source/lehome/lehome/flywheel/artifacts.py`
- Test: `tests/flywheel/test_artifacts.py`

- [ ] **Step 1: Write failure and completion tests**

```python
import json
from pathlib import Path
import pytest

from lehome.flywheel.artifacts import EpisodeArtifactWriter, verify_episode


def test_episode_is_visible_only_after_atomic_finalize(tmp_path: Path) -> None:
    writer = EpisodeArtifactWriter(tmp_path, "episode-001")
    writer.append_annotation({"step": 0, "action_source": "policy"})
    assert not (tmp_path / "raw" / "episode-001").exists()
    final = writer.finalize({"outcome": "timeout"})
    assert final == tmp_path / "raw" / "episode-001"
    assert verify_episode(final)["episode_id"] == "episode-001"


def test_finalize_refuses_missing_or_empty_video(tmp_path: Path) -> None:
    writer = EpisodeArtifactWriter(tmp_path, "episode-002")
    writer.append_annotation({"step": 0})
    with pytest.raises(ValueError, match="video"):
        writer.finalize({"outcome": "success"}, required_videos=("top.mp4",))
```

- [ ] **Step 2: Verify the tests fail**

Run: `uv run --offline pytest tests/flywheel/test_artifacts.py -v`

Expected: FAIL with missing `lehome.flywheel.artifacts`.

- [ ] **Step 3: Implement staging, fsync, hashes, and rename**

```python
# source/lehome/lehome/flywheel/artifacts.py
class EpisodeArtifactWriter:
    def __init__(self, run_root: Path, episode_id: str) -> None:
        self.run_root = run_root.resolve()
        self.episode_id = episode_id
        self.staging = self.run_root / ".pending" / episode_id
        self.staging.mkdir(parents=True, exist_ok=False)

    def append_annotation(self, value: Mapping[str, object]) -> None:
        with (self.staging / "annotations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def finalize(
        self,
        episode: Mapping[str, object],
        *,
        required_videos: tuple[str, ...] = (),
    ) -> Path:
        for name in required_videos:
            path = self.staging / "videos" / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"required video is missing or empty: {name}")
        atomic_write_json(self.staging / "episode.json", dict(episode))
        manifest = build_sha256_manifest(self.staging)
        atomic_write_json(self.staging / "SHA256SUMS.json", manifest)
        destination = self.run_root / "raw" / self.episode_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.staging.rename(destination)
        return destination
```

Reject symlinks, path traversal, duplicate final IDs, and a manifest that includes itself. `verify_episode()` must recompute every listed size/hash and reject unlisted regular files.

- [ ] **Step 4: Run artifact tests**

Run: `uv run --offline pytest tests/flywheel/test_artifacts.py -v`

Expected: PASS.

- [ ] **Step 5: Commit atomic artifact handling**

```bash
git add source/lehome/lehome/flywheel/artifacts.py tests/flywheel/test_artifacts.py
git commit -m "feat: finalize flywheel episodes atomically"
```

### Task 3: Expert-only behavior-cloning export

**Files:**
- Create: `source/lehome/lehome/flywheel/export.py`
- Test: `tests/flywheel/test_export.py`

- [ ] **Step 1: Write exact selection-boundary tests**

```python
from lehome.flywheel.export import select_expert_windows
from lehome.flywheel.models import ActionSource, EpisodeFrame


def frame(step: int, source: ActionSource) -> EpisodeFrame:
    return EpisodeFrame(step, step + 1, step + 2, (0.0,) * 12, (float(step),) * 12, source, 0.0, False, 1)


def test_dagger_exports_only_complete_expert_windows() -> None:
    frames = [frame(i, ActionSource.POLICY if i < 4 else ActionSource.EXPERT) for i in range(24)]
    selected = select_expert_windows(frames, horizon=16, accepted_success=True)
    assert [window.observation_step for window in selected] == [4, 5, 6, 7, 8]
    assert all(len(window.future_actions) == 16 for window in selected)
    assert all(value[0] >= 4 for window in selected for value in window.future_actions)


def test_failed_episode_and_hold_frames_export_nothing() -> None:
    frames = [frame(i, ActionSource.EXPERT) for i in range(20)]
    assert select_expert_windows(frames, horizon=16, accepted_success=False) == ()
```

- [ ] **Step 2: Confirm fail-first behavior**

Run: `uv run --offline pytest tests/flywheel/test_export.py -v`

Expected: FAIL with missing export module.

- [ ] **Step 3: Implement contiguous expert-window selection and rejection counts**

```python
@dataclass(frozen=True, slots=True)
class ExpertWindow:
    observation_step: int
    future_actions: tuple[tuple[float, ...], ...]


def select_expert_windows(
    frames: Sequence[EpisodeFrame], *, horizon: int, accepted_success: bool
) -> tuple[ExpertWindow, ...]:
    if not accepted_success:
        return ()
    windows: list[ExpertWindow] = []
    for start in range(len(frames)):
        segment = frames[start : start + horizon]
        if len(segment) != horizon:
            continue
        if any(frame.action_source is not ActionSource.EXPERT for frame in segment):
            continue
        if any(next_frame.step != frame.step + 1 for frame, next_frame in zip(segment, segment[1:])):
            continue
        windows.append(ExpertWindow(frames[start].step, tuple(frame.action for frame in segment)))
    return tuple(windows)
```

Add `build_selection_report()` with counts for `policy`, `hold`, `failed_episode`, `short_tail`, `stale_expert`, `holdout`, and `selected`. Make the later LeRobot exporter consume only returned windows.

- [ ] **Step 4: Run export tests and the existing GR00T policy tests**

Run: `uv run --offline pytest tests/flywheel/test_export.py trainer/tests/test_rollout_policy.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the expert-data firewall**

```bash
git add source/lehome/lehome/flywheel/export.py tests/flywheel/test_export.py
git commit -m "feat: export expert-only action windows"
```

### Task 4: Deterministic 280-trial matrix and holdout policy

**Files:**
- Create: `source/lehome/lehome/flywheel/matrix.py`
- Create: `scripts/build_groot_flywheel_matrix.py`
- Create: `configs/eval_groot_n17_public_280.json`
- Test: `tests/flywheel/test_matrix.py`

- [ ] **Step 1: Write the matrix invariants**

```python
from lehome.flywheel.matrix import build_public_matrix, matrix_sha256


def test_public_matrix_has_frozen_breadth_and_holdouts() -> None:
    matrix = build_public_matrix()
    assert len(matrix.trials) == 280
    assert sum(t.release_stage == "seen" for t in matrix.trials) == 200
    assert sum(t.release_stage == "public_unseen" for t in matrix.trials) == 80
    assert sum(t.category == "pant_long" for t in matrix.trials) == 70
    assert len({(t.garment_name, t.seed) for t in matrix.trials}) == 280
    assert matrix.training_holdouts == (
        "Top_Long_Unseen_1", "Top_Short_Unseen_1", "Pant_Long_Unseen_1", "Pant_Short_Unseen_1"
    )
    assert matrix_sha256(matrix) == matrix_sha256(build_public_matrix())
```

- [ ] **Step 2: Run the matrix test and observe failure**

Run: `uv run --offline pytest tests/flywheel/test_matrix.py -v`

Expected: FAIL with missing matrix module.

- [ ] **Step 3: Implement canonical IDs, seeds, serialization, and CLI**

```python
SEEN_SEEDS = (101, 211, 307, 401, 503)
UNSEEN_SEEDS = (601, 607, 613, 617, 619, 631, 641, 643, 647, 653)


def build_public_matrix() -> PublicMatrix:
    trials = []
    for category, prefix in CATEGORY_PREFIX.items():
        for index in range(10):
            for seed in SEEN_SEEDS:
                trials.append(Trial(category, f"{prefix}_Seen_{index}", "seen", seed))
        for index in range(2):
            for seed in UNSEEN_SEEDS:
                trials.append(Trial(category, f"{prefix}_Unseen_{index}", "public_unseen", seed))
    return PublicMatrix(schema_version=1, trials=tuple(trials), training_holdouts=PUBLIC_UNSEEN_HOLDOUTS)
```

The CLI writes canonical sorted JSON only when the generated asset names exist under `Assets/objects/Challenge_Garment/Release`; it prints the SHA-256 and refuses to overwrite a differing file without `--check` failing.

- [ ] **Step 4: Generate and verify the committed matrix**

Run: `uv run --offline python -m scripts.build_groot_flywheel_matrix --output configs/eval_groot_n17_public_280.json`

Expected: one `sha256=` line whose value matches `^[0-9a-f]{64}$`, and exit 0.

Run: `uv run --offline pytest tests/flywheel/test_matrix.py tests/test_eval_groot_n17_matrix.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the public evaluation contract**

```bash
git add source/lehome/lehome/flywheel/matrix.py scripts/build_groot_flywheel_matrix.py configs/eval_groot_n17_public_280.json tests/flywheel/test_matrix.py
git commit -m "feat: freeze 280-trial GR00T public matrix"
```

### Task 5: Automated hard-case ranking

**Files:**
- Create: `source/lehome/lehome/flywheel/hard_mining.py`
- Test: `tests/flywheel/test_hard_mining.py`

- [ ] **Step 1: Write ranking tests using official and diagnostic fields**

```python
from lehome.flywheel.hard_mining import FailureEvidence, rank_failures


def test_failures_rank_by_category_gap_stall_and_progress() -> None:
    failures = (
        FailureEvidence("pant", "pant_long", False, 0.1, 220, 400, True),
        FailureEvidence("shirt", "top_short", False, 0.7, 20, 400, False),
    )
    ranked = rank_failures(failures, category_success={"pant_long": 0.0, "top_short": 0.8})
    assert [item.episode_id for item in ranked] == ["pant", "shirt"]
    assert ranked[0].priority_reasons == ("category_gap", "low_progress", "stalled", "restorable")
```

- [ ] **Step 2: Verify the hard-mining test fails**

Run: `uv run --offline pytest tests/flywheel/test_hard_mining.py -v`

Expected: FAIL with missing hard-mining module.

- [ ] **Step 3: Implement deterministic scoring without changing official reward**

```python
def rank_failures(
    failures: Sequence[FailureEvidence], *, category_success: Mapping[str, float]
) -> tuple[RankedFailure, ...]:
    ranked = []
    for failure in failures:
        category_gap = 1.0 - category_success[failure.category]
        low_progress = 1.0 - min(max(failure.max_progress, 0.0), 1.0)
        stalled = min(failure.stalled_steps / max(failure.length, 1), 1.0)
        score = 4.0 * category_gap + 3.0 * low_progress + 2.0 * stalled + float(failure.restorable)
        ranked.append(RankedFailure.from_evidence(failure, score))
    return tuple(sorted(ranked, key=lambda item: (-item.score, item.episode_id)))
```

Keep `official_success` and `official_return` unchanged in the report. Store `max_progress`, `stalled_steps`, and ranking reasons under `diagnostics`, never `reward`.

- [ ] **Step 4: Run the complete pure core suite**

Run: `uv run --offline pytest tests/flywheel -v`

Expected: PASS with no Isaac Sim process started.

- [ ] **Step 5: Commit hard-mining selection**

```bash
git add source/lehome/lehome/flywheel/hard_mining.py tests/flywheel/test_hard_mining.py
git commit -m "feat: rank rollout failures for hard mining"
```

## Plan 1 completion gate

Run: `uv run --offline pytest tests/flywheel tests/test_eval_groot_n17_matrix.py trainer/tests/test_rollout_policy.py -v`

Expected: all tests pass; `git status --short` contains no changes from this plan. No GPU rental is authorized by completing this plan.
