# GR00T Flywheel Autonomous Rollouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record provenance-complete autonomous Isaac Sim episodes, restorable snapshots, controlled domain randomization, and measured multi-worker capacity reports.

**Architecture:** A simulator adapter translates live Isaac observations into the core episode contract without coupling pure validation to Isaac imports. One process owns one Isaac environment; a supervisor assigns immutable trials, observes heartbeats, and writes a capacity report. Policy servers remain separate GPU processes.

**Tech Stack:** Isaac Sim 5.1, IsaacLab, PyTorch, NumPy, GR00T N1.7, ffmpeg/H.264, multiprocessing/subprocess, pytest.

---

## File structure

- Create `source/lehome/lehome/flywheel/isaac_recorder.py`: per-step autonomous episode recorder.
- Create `source/lehome/lehome/flywheel/snapshots.py`: robot, cloth, RNG, and environment state capture/restore.
- Create `source/lehome/lehome/flywheel/randomization.py`: deterministic canonical/mild/strong parameter sampling.
- Create `source/lehome/lehome/flywheel/capacity.py`: finite 1/2/4/6/8 acceptance decisions.
- Create `scripts/run_groot_flywheel_trial.py`: one-trial process boundary.
- Create `scripts/run_groot_flywheel_campaign.py`: resumable supervisor.
- Modify `scripts/utils/evaluation.py`: opt-in recorder hooks and terminal reasons.
- Modify `source/lehome/lehome/tasks/bedroom/garment_bi_v2.py`: snapshot and randomization adapter methods.
- Create `tests/flywheel/test_isaac_recorder.py`, `test_snapshots.py`, `test_randomization.py`, and `test_capacity.py`.

### Task 1: Per-step autonomous recorder

**Files:**
- Create: `source/lehome/lehome/flywheel/isaac_recorder.py`
- Test: `tests/flywheel/test_isaac_recorder.py`

- [ ] **Step 1: Write a fake-environment recorder test**

```python
import numpy as np
from lehome.flywheel.isaac_recorder import AutonomousRecorder


def observation() -> dict[str, np.ndarray]:
    return {
        "observation.state": np.zeros(12, dtype=np.float32),
        "observation.images.top_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.left_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.right_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
    }


def test_autonomous_recorder_marks_policy_source_and_terminal_reason(tmp_path) -> None:
    recorder = AutonomousRecorder.for_test(tmp_path, policy_revision="a" * 40)
    recorder.record_step(observation(), np.ones(12), reward=0.2, success=False, request_id="r1", chunk_offset=0)
    final = recorder.finish(reason="horizon", accepted_success=False)
    assert final.episode["terminal_reason"] == "horizon"
    assert final.annotations[0]["action_source"] == "policy"
    assert final.annotations[0]["policy_request_id"] == "r1"
```

- [ ] **Step 2: Run the recorder test and see it fail**

Run: `uv run --offline pytest tests/flywheel/test_isaac_recorder.py -v`

Expected: FAIL with missing recorder module.

- [ ] **Step 3: Implement the recorder over the core artifact writer**

```python
class AutonomousRecorder:
    def record_step(
        self,
        observation: Mapping[str, object],
        action: object,
        *,
        reward: float,
        success: bool,
        request_id: str,
        chunk_offset: int,
    ) -> None:
        state = finite_vector(observation["observation.state"], size=12, name="state")
        applied = finite_vector(action, size=12, name="action")
        self.writer.append_annotation(
            EpisodeFrame(
                step=self.step,
                monotonic_ns=time.monotonic_ns(),
                wall_time_ns=time.time_ns(),
                state=state,
                action=applied,
                action_source=ActionSource.POLICY,
                reward=float(reward),
                success=bool(success),
                segment=0,
                policy_request_id=request_id,
                policy_chunk_offset=chunk_offset,
            ).to_dict()
        )
        self.video_sink.append(observation)
        self.step += 1
```

`finish()` must validate all three video frame counts against annotation count, encode browser-compatible H.264, write the official terminal result, and finalize atomically. A failure still finalizes diagnostic evidence and always has zero BC targets.

- [ ] **Step 4: Run recorder and policy adapter tests**

Run: `uv run --offline pytest tests/flywheel/test_isaac_recorder.py trainer/tests/test_rollout_policy.py -v`

Expected: PASS.

- [ ] **Step 5: Commit autonomous recording**

```bash
git add source/lehome/lehome/flywheel/isaac_recorder.py tests/flywheel/test_isaac_recorder.py
git commit -m "feat: record autonomous flywheel episodes"
```

### Task 2: Restorable simulator snapshots

**Files:**
- Create: `source/lehome/lehome/flywheel/snapshots.py`
- Modify: `source/lehome/lehome/tasks/bedroom/garment_bi_v2.py`
- Test: `tests/flywheel/test_snapshots.py`

- [ ] **Step 1: Write a round-trip snapshot test with a fake adapter**

```python
import numpy as np
from lehome.flywheel.snapshots import Snapshot, capture_snapshot, restore_snapshot


class FakeAdapter:
    robot_position = np.arange(12, dtype=np.float32)
    robot_velocity = np.zeros(12, dtype=np.float32)
    cloth_position = np.arange(30, dtype=np.float32).reshape(10, 3)
    cloth_velocity = np.ones((10, 3), dtype=np.float32)
    rng_state = {"seed": 42, "counter": 7}


def test_snapshot_round_trip_restores_every_state_group() -> None:
    env = FakeAdapter()
    snapshot = capture_snapshot(env, randomization={"strategy": "canonical"})
    env.robot_position[:] = -1
    restore_snapshot(env, snapshot)
    assert env.robot_position.tolist() == list(range(12))
    assert env.cloth_position.shape == (10, 3)
    assert env.rng_state == {"seed": 42, "counter": 7}
```

- [ ] **Step 2: Confirm the snapshot test fails**

Run: `uv run --offline pytest tests/flywheel/test_snapshots.py -v`

Expected: FAIL with missing snapshot module.

- [ ] **Step 3: Implement snapshot schema and narrow environment methods**

```python
@dataclass(frozen=True, slots=True)
class Snapshot:
    schema_version: int
    robot_position: tuple[float, ...]
    robot_velocity: tuple[float, ...]
    cloth_position: tuple[tuple[float, float, float], ...]
    cloth_velocity: tuple[tuple[float, float, float], ...]
    rng_state: dict[str, object]
    garment_name: str
    randomization: dict[str, object]


def capture_snapshot(adapter: SnapshotAdapter, *, randomization: Mapping[str, object]) -> Snapshot:
    return Snapshot(
        schema_version=1,
        robot_position=finite_tuple(adapter.get_robot_position()),
        robot_velocity=finite_tuple(adapter.get_robot_velocity()),
        cloth_position=finite_xyz(adapter.get_cloth_position()),
        cloth_velocity=finite_xyz(adapter.get_cloth_velocity()),
        rng_state=json_round_trip(adapter.get_rng_state()),
        garment_name=adapter.get_garment_name(),
        randomization=json_round_trip(randomization),
    )
```

Add `flywheel_capture_state()` and `flywheel_restore_state(snapshot)` to `GarmentEnv` using its existing robot articulation, garment particle object, and RNG objects. Do not change reward or reset semantics.

- [ ] **Step 4: Run pure tests, then one remote Isaac round-trip acceptance**

Run locally: `uv run --offline pytest tests/flywheel/test_snapshots.py tests/test_success_checker_challenge.py -v`

Expected: PASS.

Run only on an already-rented accepted host: `uv run python -m scripts.run_groot_flywheel_trial --snapshot-roundtrip-only --garment Pant_Long_Seen_0 --seed 42 --output-root /workspace/acceptance/snapshot`

Expected: exit 0 and `snapshot-acceptance.json` reports camera difference within the recorded tolerance and no simulation error.

- [ ] **Step 5: Commit snapshot support**

```bash
git add source/lehome/lehome/flywheel/snapshots.py source/lehome/lehome/tasks/bedroom/garment_bi_v2.py tests/flywheel/test_snapshots.py
git commit -m "feat: capture restorable garment snapshots"
```

### Task 3: Deterministic domain randomization

**Files:**
- Create: `source/lehome/lehome/flywheel/randomization.py`
- Modify: `source/lehome/lehome/tasks/bedroom/garment_bi_v2.py`
- Modify: `source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml`
- Test: `tests/flywheel/test_randomization.py`

- [ ] **Step 1: Write deterministic range and canonical tests**

```python
from lehome.flywheel.randomization import sample_randomization


def test_canonical_changes_nothing_and_mild_is_reproducible() -> None:
    canonical = sample_randomization("canonical", seed=99)
    assert canonical.values == {}
    first = sample_randomization("mild", seed=99)
    second = sample_randomization("mild", seed=99)
    assert first == second
    assert 0.85 <= first.values["light_intensity_scale"] <= 1.15
    assert abs(first.values["camera_translation_m"][0]) <= 0.01


def test_strong_stays_inside_physical_bounds() -> None:
    result = sample_randomization("strong", seed=123)
    assert abs(result.values["garment_yaw_deg"]) <= 15.0
    assert abs(result.values["robot_base_translation_m"][1]) <= 0.02
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run --offline pytest tests/flywheel/test_randomization.py -v`

Expected: FAIL with missing randomization module.

- [ ] **Step 3: Implement explicit strategy bounds and application receipts**

```python
BOUNDS = {
    "mild": RandomizationBounds(light=(0.85, 1.15), camera_m=0.01, garment_yaw_deg=5.0, base_m=0.005),
    "strong": RandomizationBounds(light=(0.65, 1.35), camera_m=0.02, garment_yaw_deg=15.0, base_m=0.02),
}


def sample_randomization(strategy: str, *, seed: int) -> RandomizationRecord:
    if strategy == "canonical":
        return RandomizationRecord(strategy, seed, {})
    bounds = BOUNDS[strategy]
    rng = np.random.default_rng(seed)
    values = {
        "light_intensity_scale": float(rng.uniform(*bounds.light)),
        "camera_translation_m": tuple(float(v) for v in rng.uniform(-bounds.camera_m, bounds.camera_m, 3)),
        "garment_yaw_deg": float(rng.uniform(-bounds.garment_yaw_deg, bounds.garment_yaw_deg)),
        "robot_base_translation_m": tuple(float(v) for v in rng.uniform(-bounds.base_m, bounds.base_m, 3)),
    }
    return RandomizationRecord(strategy, seed, values)
```

Implement table material, garment material, light, camera, garment reset, and robot-base application behind `apply_flywheel_randomization()`. Return the values read back from USD/Isaac; reject a mismatch rather than silently labeling canonical output as randomized. Leave YAML defaults disabled so normal challenge evaluation remains unchanged.

- [ ] **Step 4: Run local tests and render a remote sample sheet**

Run locally: `uv run --offline pytest tests/flywheel/test_randomization.py tests/test_success_checker_challenge.py -v`

Expected: PASS.

Run remotely: `uv run python -m scripts.run_groot_flywheel_trial --render-randomization-sheet --garment Pant_Long_Seen_0 --strategies canonical mild strong --seed 123 --output-root /workspace/acceptance/randomization`

Expected: nine labeled images plus `randomization-receipts.json`; visually reject occluded grippers, invisible garments, and camera framing outside the canonical workspace.

- [ ] **Step 5: Commit domain randomization**

```bash
git add source/lehome/lehome/flywheel/randomization.py source/lehome/lehome/tasks/bedroom/garment_bi_v2.py source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml tests/flywheel/test_randomization.py
git commit -m "feat: add reproducible rollout randomization"
```

### Task 4: Opt-in evaluation integration

**Files:**
- Create: `scripts/run_groot_flywheel_trial.py`
- Modify: `scripts/utils/evaluation.py`
- Modify: `scripts/eval_policy/groot_policy.py`
- Test: `tests/flywheel/test_trial_cli.py`
- Test: `trainer/tests/test_rollout_policy.py`

- [ ] **Step 1: Write CLI and chunk-provenance tests**

```python
def test_trial_cli_requires_pinned_policy_and_existing_matrix(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(["--policy-path", str(tmp_path / "missing"), "--policy-revision", "main"])
    with pytest.raises(ValueError, match="pinned"):
        validate_args(args)


def test_trial_cli_reads_revision_from_a_regular_file(tmp_path) -> None:
    revision = tmp_path / "revision.txt"
    revision.write_text("a" * 40 + "\n", encoding="utf-8")
    assert read_pinned_revision(revision) == "a" * 40


def test_action_queue_reports_request_and_offset() -> None:
    queue = ActionChunkQueue()
    queue.extend(np.zeros((16, 12), dtype=np.float32), request_id="req-7")
    item = queue.pop_with_provenance()
    assert item.request_id == "req-7"
    assert item.chunk_offset == 0
```

- [ ] **Step 2: Verify the focused tests fail**

Run: `uv run --offline pytest tests/flywheel/test_trial_cli.py trainer/tests/test_rollout_policy.py -v`

Expected: FAIL for missing CLI and provenance methods.

- [ ] **Step 3: Add opt-in hooks without changing legacy evaluation defaults**

```python
@dataclass(frozen=True, slots=True)
class QueuedAction:
    value: np.ndarray
    request_id: str
    chunk_offset: int


def select_action_with_provenance(self, observation: Mapping[str, Any]) -> QueuedAction:
    queued = self._action_queue.pop_with_provenance()
    if queued is not None:
        return queued
    request_id = uuid.uuid4().hex
    action, _ = self._policy.get_action(build_groot_observation(observation))
    self._action_queue.extend(flatten_groot_action_chunk(action), request_id=request_id)
    return self._action_queue.pop_with_provenance_required()
```

In `evaluation.py`, create the recorder only when `args.flywheel_manifest` is supplied. Record the action actually passed to `env.step()`, official reward, success-check result, and terminal reason. Existing `scripts.eval` behavior and output format must remain byte-compatible when the flag is absent.

- [ ] **Step 4: Run integration and regression tests**

Run: `uv run --offline pytest tests/flywheel/test_trial_cli.py trainer/tests/test_rollout_policy.py tests/test_eval_groot_n17_matrix.py -v`

Expected: PASS.

- [ ] **Step 5: Commit trial integration**

```bash
git add scripts/run_groot_flywheel_trial.py scripts/utils/evaluation.py scripts/eval_policy/groot_policy.py tests/flywheel/test_trial_cli.py trainer/tests/test_rollout_policy.py
git commit -m "feat: integrate provenance-complete GR00T trials"
```

### Task 5: Resumable campaign supervisor and capacity sweep

**Files:**
- Create: `source/lehome/lehome/flywheel/capacity.py`
- Create: `scripts/run_groot_flywheel_campaign.py`
- Test: `tests/flywheel/test_capacity.py`
- Test: `tests/flywheel/test_campaign.py`

- [ ] **Step 1: Write capacity-decision and resume tests**

```python
from lehome.flywheel.capacity import CapacitySample, choose_worker_count


def test_capacity_stops_at_six_when_eight_lacks_gain() -> None:
    samples = (
        CapacitySample(4, 120.0, 4, 0, 0.30, 0.25),
        CapacitySample(6, 82.0, 6, 0, 0.24, 0.20),
        CapacitySample(8, 77.0, 8, 0, 0.17, 0.13),
    )
    decision = choose_worker_count(samples, minimum_gain=0.15)
    assert decision.accepted_workers == 6
    assert decision.rejected[8] == ("render_vram_margin", "throughput_gain")


def test_campaign_resume_skips_checksum_verified_trials(tmp_path) -> None:
    state = campaign_state_with_completed_trial(tmp_path, "trial-001")
    assert pending_trial_ids(state) == ("trial-002",)
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `uv run --offline pytest tests/flywheel/test_capacity.py tests/flywheel/test_campaign.py -v`

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement finite sweep and supervised worker processes**

```python
def choose_worker_count(samples: Sequence[CapacitySample], *, minimum_gain: float = 0.15) -> CapacityDecision:
    accepted = 1
    rejected: dict[int, tuple[str, ...]] = {}
    prior_rate = None
    for sample in sorted(samples, key=lambda value: value.workers):
        reasons = sample.rejection_reasons(minimum_ram=0.20, minimum_vram=0.15)
        rate = sample.completed_trials / sample.elapsed_seconds
        if prior_rate is not None and rate / prior_rate - 1.0 < minimum_gain:
            reasons += ("throughput_gain",)
        if reasons:
            rejected[sample.workers] = tuple(dict.fromkeys(reasons))
            break
        accepted = sample.workers
        prior_rate = rate
    return CapacityDecision(accepted, rejected)
```

The supervisor uses explicit worker IDs, per-worker logs and heartbeat files, never shell globs for cleanup, verifies completed episode hashes before resume, and terminates a worker only after a finite timeout. Sweep order is exactly `1,2,4,6`, then `8` only if 6 passes. It never rents or destroys instances.

- [ ] **Step 4: Run local orchestration tests and the paid-host acceptance gate**

Run locally: `uv run --offline pytest tests/flywheel/test_capacity.py tests/flywheel/test_campaign.py -v`

Expected: PASS.

Remote acceptance sequence on one already-approved host:

The named matrix is checked byte-for-byte against the 280-trial public contract
in `lehome.flywheel.matrix` before the campaign launches any worker.

```bash
uv run python -m scripts.run_groot_flywheel_campaign \\
  --matrix configs/eval_groot_n17_public_280.json \\
  --policy-path /workspace/policies/step-12000 \\
  --policy-revision-file /workspace/policies/step-12000/revision.txt \\
  --policy-repo org/groot-policy \\
  --policy-step 12000 \\
  --code-revision "$(git rev-parse HEAD)" \\
  --asset-revision <40-character-release-assets-commit> \\
  --simulator-version isaac-sim-5.1 \\
  --policy-artifact-sha256 "$(sha256sum /workspace/policies/step-12000/model.safetensors | awk '{print $1}')" \\
  --image-identity <immutable-container-image-digest> \\
  --output-root /workspace/rollouts/capacity \\
  --capacity-sweep 1,2,4,6,8 \\
  --trials-per-worker 1
```

Expected: `capacity-report.json` names the accepted count, every tested process reaches a terminal trial, videos are nonempty, and the host keeps at least 20% RAM and 15% VRAM free. Do not launch the 280-trial campaign until this file passes validation.

- [ ] **Step 5: Commit campaign orchestration**

```bash
git add source/lehome/lehome/flywheel/capacity.py scripts/run_groot_flywheel_campaign.py tests/flywheel/test_capacity.py tests/flywheel/test_campaign.py
git commit -m "feat: supervise scalable rollout campaigns"
```

## Plan 2 completion gate

Run locally: `uv run --offline pytest tests/flywheel tests/test_eval_groot_n17_matrix.py tests/test_success_checker_challenge.py trainer/tests/test_rollout_policy.py -v`

Then run exactly one canonical remote acceptance trial with videos and snapshot restore. Only after that passes may the finite capacity sweep run. The 280-trial paid campaign remains a separate explicit execution decision.
