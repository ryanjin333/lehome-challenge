# Simple Curriculum and Success-Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a stopped-at-rest, one-VM collection workflow that produces 1,000 fresh original-12K outcomes using uniform calibration plus deterministic curriculum, then produces a bounded visual-only success-replay dataset from only those new successes.

**Architecture:** Reuse the existing persistent worker, SQLite task ledger, original-12K policy server, evaluation summarizer, and Hugging Face sync. Add pure deterministic matrix/report code and a thin resumable host orchestrator. Execute the logical 400-row calibration and 600-row curriculum as immutable physical partitions so the existing success-oriented ledger can still account for every valid outcome without a broad ledger rewrite. Explicitly bind-mount the checked-out source into the rollout containers; do not build another image or create another VM.

**Tech Stack:** Python 3.11, Bash, pytest, SQLite task ledger, Docker/Isaac Sim rollout appliance, Hugging Face Hub publication, SHA-256 receipts.

---

## Global constraints for every task

- Implementation is offline: do not start, create, resize, or delete a Nebius VM.
- Do not launch Isaac Sim, the policy server, a rollout, replay, training, or evaluation.
- Keep every GPU VM stopped during implementation verification.
- Do not modify the original step-12K checkpoint, the 40 seen-garment asset set, or prior datasets.
- Do not add hard-state mining, DAgger, A-500, geometry/physics perturbation, training, or another experimental arm.
- Preserve legacy campaign behavior unless `LEHOME_SIMPLE_CURRICULUM_COLLECTION=1` is set.
- Never publish credentials, organizer BC data, model caches, or temporary local files.

### Task 1: Add the deterministic fresh-collection matrix builder

**Files:**
- Create: `source/lehome/lehome/flywheel/simple_curriculum.py`
- Create: `scripts/build_simple_curriculum_matrix.py`
- Test: `tests/flywheel/test_simple_curriculum.py`
- Test: `tests/infrastructure/test_simple_curriculum_cli.py`

- [ ] **Step 1: Write RED tests for uniform calibration**

Test that a fixed 40-garment catalog produces exactly 400 canonical rows, ten rows per garment, 100 rows per category, 25 rows per category in rows 0-99, and globally unique attempt IDs, trial IDs, and seeds.

```python
def test_calibration_is_balanced_and_interleaved(seen_catalog):
    rows = build_calibration_rows(seen_catalog, seed_base=202608270000)
    assert len(rows) == 400
    assert Counter(row["category"] for row in rows) == {
        "pant_long": 100,
        "pant_short": 100,
        "top_long": 100,
        "top_short": 100,
    }
    assert Counter(row["category"] for row in rows[:100]) == {
        "pant_long": 25,
        "pant_short": 25,
        "top_long": 25,
        "top_short": 25,
    }
    assert set(Counter(row["garment"] for row in rows).values()) == {10}
    assert len({row["attempt_id"] for row in rows}) == 400
    assert len({row["trial_id"] for row in rows}) == 400
    assert len({row["seed"] for row in rows}) == 400
    assert {row["strategy"] for row in rows} == {"canonical"}
```

- [ ] **Step 2: Write RED tests for the curriculum formula and provenance**

Build a minimal authenticated report fixture with per-category and per-garment valid-outcome counts. Assert the exact released weighting formulas, deterministic sampling, 600 rows, no calibration-seed collisions, and fail-closed rejection of a report with the wrong policy, device, matrix hash, garment catalog, or incomplete 400 valid outcomes.

```python
def test_curriculum_weights_match_contract():
    assert type_weight(0.80) == pytest.approx(0.20)
    assert type_weight(1.00) == pytest.approx(0.05)
    assert garment_weight(0.50) == pytest.approx(1.0)
    expected = math.exp(-((0.10 - 0.50) ** 2) / (2 * 0.233**2))
    assert garment_weight(0.10) == pytest.approx(max(expected, 0.02))

def test_curriculum_is_deterministic(authenticated_calibration_report, calibration_rows):
    first = build_curriculum_rows(
        authenticated_calibration_report,
        calibration_rows=calibration_rows,
        count=600,
        rng_seed=20260827600,
    )
    second = build_curriculum_rows(
        authenticated_calibration_report,
        calibration_rows=calibration_rows,
        count=600,
        rng_seed=20260827600,
    )
    assert first == second
    assert len(first) == 600
    assert not ({row["seed"] for row in first} & {row["seed"] for row in calibration_rows})
```

- [ ] **Step 3: Run the focused RED tests**

```bash
PYTHONPATH=source/lehome uv run pytest -q \
  tests/flywheel/test_simple_curriculum.py \
  tests/infrastructure/test_simple_curriculum_cli.py
```

Expected: failures because the module and CLI do not exist.

- [ ] **Step 4: Implement the pure builder module**

Use immutable dictionaries and `random.Random(rng_seed)`. Keep report validation separate from sampling.

```python
SIGMA = 0.233
CATEGORIES = ("pant_long", "pant_short", "top_long", "top_short")

def type_weight(success_rate: float) -> float:
    return max(1.0 - success_rate, 0.05)

def garment_weight(success_rate: float) -> float:
    return max(math.exp(-((success_rate - 0.5) ** 2) / (2.0 * SIGMA**2)), 0.02)

def build_calibration_rows(
    catalog: Sequence[Mapping[str, object]], *, seed_base: int
) -> list[dict[str, object]]: ...

def validate_calibration_report(
    report: Mapping[str, object], *, matrix_sha256: str,
    policy_identity: Mapping[str, object], catalog: Sequence[Mapping[str, object]],
) -> None: ...

def build_curriculum_rows(
    report: Mapping[str, object], *, calibration_rows: Sequence[Mapping[str, object]],
    count: int, rng_seed: int,
) -> list[dict[str, object]]: ...
```

Every row must include `campaign_kind`, `logical_stage`, `attempt_id`, `trial_id`, `garment`, `garment_name`, `category`, `release_stage="seen"`, `seed`, `source_seed`, and `strategy="canonical"`. Curriculum rows also include `builder_rng_seed`, `calibration_matrix_sha256`, and the sampled category/garment weights.

- [ ] **Step 5: Implement an atomic CLI**

The CLI has two explicit subcommands:

```text
build-calibration --catalog ... --seed-base ... --output ... --receipt ...
build-curriculum --report ... --calibration-matrix ... --rng-seed ... --output ... --receipt ...
```

It refuses symlink inputs/outputs and existing output paths, writes through a same-directory temporary file plus `fsync`, then writes a SHA-256 receipt binding parameters and output bytes.

- [ ] **Step 6: Run tests and compile**

```bash
PYTHONPATH=source/lehome uv run pytest -q \
  tests/flywheel/test_simple_curriculum.py \
  tests/infrastructure/test_simple_curriculum_cli.py
PYTHONPATH=source/lehome python3 -m py_compile \
  source/lehome/lehome/flywheel/simple_curriculum.py \
  scripts/build_simple_curriculum_matrix.py
```

Expected: all focused tests pass and compilation exits zero.

- [ ] **Step 7: Commit Task 1**

```bash
git add source/lehome/lehome/flywheel/simple_curriculum.py \
  scripts/build_simple_curriculum_matrix.py \
  tests/flywheel/test_simple_curriculum.py \
  tests/infrastructure/test_simple_curriculum_cli.py
git commit -m "feat: build deterministic simple curriculum matrices"
```

### Task 2: Add an authenticated 100-outcome circuit-breaker report

**Files:**
- Create: `scripts/check_simple_curriculum_gate.py`
- Modify: `scripts/summarize_groot_persistent_evaluation.py`
- Test: `tests/flywheel/test_simple_curriculum_gate.py`
- Test: `tests/infrastructure/test_groot_persistent_summary.py`

- [ ] **Step 1: Write RED gate tests**

Cover pass, missing cloth, cloth flight, non-finite state, safety failure, more than 2% infrastructure-invalid executions, fewer than five official successes, mixed provenance, fewer/more than 100 valid outcomes, and duplicate assignment identity.

```python
@pytest.mark.parametrize("reason", [
    "missing_cloth", "cloth_flight", "nonfinite_cloth", "safety_failure",
])
def test_first_100_fidelity_signal_stops(reason, passing_report):
    passing_report["episodes"][0]["terminal_reason"] = reason
    result = evaluate_first_100_gate(passing_report)
    assert result.decision == "fidelity_stop"

def test_first_100_requires_five_official_successes(passing_report):
    for episode in passing_report["episodes"]:
        episode["official_success"] = False
    assert evaluate_first_100_gate(passing_report).decision == "insufficient_source_stop"
```

- [ ] **Step 2: Extend the summarizer without changing legacy fields**

Add the fields required for gate authentication while retaining existing output compatibility:

```python
report["valid_outcomes"] = len(valid_terminal_attempts)
report["infrastructure_invalid_executions"] = invalid_execution_count
report["execution_count"] = len(valid_terminal_attempts) + invalid_execution_count
report["runtime_identities"] = sorted(runtime_identity_digests)
report["fresh_assignment_ids"] = sorted(valid_assignment_ids)
```

Only ledger-terminal successes and failures count as valid outcomes. Expired leases, process crashes, missing receipts, malformed artifacts, and rejected publication do not.

- [ ] **Step 3: Implement pure gate evaluation and an atomic decision receipt**

```python
def evaluate_first_100_gate(report: Mapping[str, object]) -> GateDecision:
    if report["valid_outcomes"] != 100:
        return GateDecision("infrastructure_stop", "valid_outcome_count")
    if any_fidelity_failure(report):
        return GateDecision("fidelity_stop", "episode_fidelity")
    ratio = report["infrastructure_invalid_executions"] / report["execution_count"]
    if ratio > 0.02:
        return GateDecision("infrastructure_stop", "invalid_ratio")
    if report["official_successes"] < 5:
        return GateDecision("insufficient_source_stop", "official_success_floor")
    if len(report["runtime_identities"]) != 1:
        return GateDecision("fidelity_stop", "mixed_runtime_identity")
    return GateDecision("continue", "passed")
```

The CLI re-hashes the report and matrix, validates the pinned original-12K identity and CPU-cloth/CUDA-policy provenance, and writes exactly one immutable gate receipt.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=source/lehome uv run pytest -q \
  tests/flywheel/test_simple_curriculum_gate.py \
  tests/infrastructure/test_groot_persistent_summary.py
python3 -m py_compile scripts/check_simple_curriculum_gate.py \
  scripts/summarize_groot_persistent_evaluation.py
git diff --check
git add scripts/check_simple_curriculum_gate.py \
  scripts/summarize_groot_persistent_evaluation.py \
  tests/flywheel/test_simple_curriculum_gate.py \
  tests/infrastructure/test_groot_persistent_summary.py
git commit -m "feat: gate simple curriculum after 100 outcomes"
```

### Task 3: Admit exact fresh-outcome partitions without weakening legacy limits

**Files:**
- Modify: `source/lehome/lehome/flywheel/task_ledger.py`
- Modify: `rollout_appliance/run_12k_campaign.sh`
- Test: `tests/flywheel/test_task_ledger.py`
- Create: `tests/infrastructure/test_simple_curriculum_campaign.py`

- [ ] **Step 1: Write RED ledger and shell-contract tests**

The logical matrices are 400 calibration rows and 600 curriculum rows. Execute them as:

```text
calibration-head: 100 rows, target_terminal=100, lease budget=150
calibration-tail: 300 rows, target_terminal=300, lease budget=400
curriculum-a:     300 rows, target_terminal=300, lease budget=400
curriculum-b:     300 rows, target_terminal=300, lease budget=400
```

Test that successes and policy failures both advance terminal-outcome count, infrastructure failures requeue the same assignment, the retry budget cannot generate a replacement seed, a resumed ledger never reissues a terminal row, and default campaigns retain the 150 accepted-success cap.

```python
def test_exact_outcome_mode_finishes_on_successes_and_policy_failures(tmp_path, matrix):
    ledger = TaskLedger(
        tmp_path / "ledger.sqlite3", attempt_matrix=matrix,
        max_attempts=150, target_accepted=100, completion_metric="terminal_outcomes",
    )
    finish(ledger, matrix[:5], success=True)
    finish(ledger, matrix[5:], success=False)
    assert ledger.terminal_outcome_count == 100
    assert ledger.is_terminal
```

- [ ] **Step 2: Add an explicit ledger completion metric**

Do not overload `accepted_count`. Add one constructor parameter with a legacy default:

```python
CompletionMetric = Literal["accepted_successes", "terminal_outcomes"]

def __init__(..., target_accepted: int,
             completion_metric: CompletionMetric = "accepted_successes") -> None: ...

@property
def completion_count(self) -> int:
    return self.accepted_count if self.completion_metric == "accepted_successes" \
        else self.terminal_outcome_count
```

Persist the metric in ledger metadata and reject reopening the same database with a different metric, matrix hash, attempt budget, or target. Raise the absolute target limit only for `terminal_outcomes`; preserve the legacy accepted-success maximum.

- [ ] **Step 3: Add the exact shell allow-list**

Introduce `LEHOME_SIMPLE_CURRICULUM_COLLECTION=0` and require all of the following when it is `1`:

```text
LEHOME_SIMULATOR_DEVICE=cpu
LEHOME_WORKER_COUNT=4
LEHOME_ENABLE_HF_UPLOAD=1
LEHOME_SKIP_ROUND_SEAL=1
LEHOME_POLICY_STEP=12000
LEHOME_COMPLETION_METRIC=terminal_outcomes
partition rows/target/budget is exactly 100/100/150 or 300/300/400
matrix campaign_kind=simple_curriculum_source_v1
matrix strategy=canonical
all garments are approved seen garments
```

Make this mode mutually exclusive with evaluation, snapshot-source, success-replay, hard-state, and controlled-recovery modes. The wrapper must pass `--completion-metric terminal_outcomes` to every worker.

- [ ] **Step 4: Permit safe resume only for the same immutable partition**

Extend the existing preemption descriptor with `campaign_mode`, `completion_metric`, `partition_id`, `parent_matrix_sha256`, and `code_root_sha256`. Resume requires byte-identical matrix, identical ledger metadata, identical pinned policy and runtime identity, and an inactive prior context. It calls the existing `resume_after_preemption` and never rebuilds a matrix.

- [ ] **Step 5: Run tests and shell syntax**

```bash
PYTHONPATH=source/lehome uv run pytest -q \
  tests/flywheel/test_task_ledger.py \
  tests/infrastructure/test_simple_curriculum_campaign.py \
  tests/infrastructure/test_rollout_container.py
bash -n rollout_appliance/run_12k_campaign.sh
```

- [ ] **Step 6: Commit Task 3**

```bash
git add source/lehome/lehome/flywheel/task_ledger.py \
  rollout_appliance/run_12k_campaign.sh \
  tests/flywheel/test_task_ledger.py \
  tests/infrastructure/test_simple_curriculum_campaign.py
git commit -m "feat: collect exact fresh outcome partitions"
```

### Task 4: Add visual-only replay randomization

**Files:**
- Modify: `source/lehome/lehome/flywheel/models.py`
- Modify: `source/lehome/lehome/flywheel/randomization.py`
- Modify: `scripts/utils/evaluation.py`
- Modify: `source/lehome/lehome/tasks/bedroom/garment_bi_v2.py`
- Modify: `source/lehome/lehome/flywheel/recovery_collection.py`
- Test: `tests/flywheel/test_randomization.py`
- Test: `tests/flywheel/test_recovery_collection.py`
- Test: `tests/flywheel/test_persistent_worker.py`

- [ ] **Step 1: Write RED contract tests**

Assert that `visual_only` contains only camera, lighting, color, and texture values. Explicitly forbid garment yaw/pose/scale, robot base, cloth geometry/material/dynamics/friction, solver settings, and joint limits. Assert sampled values are deterministic and runtime readback covers every sampled field.

```python
def test_visual_only_never_mutates_physics():
    sample = sample_randomization("visual_only", np.random.default_rng(91))
    assert set(sample.values) == {
        "light_intensity_scale",
        "camera_translation",
        "table_texture_id",
        "garment_display_color",
    }
    assert not (set(sample.values) & PHYSICS_AFFECTING_FIELDS)
```

- [ ] **Step 2: Implement a narrow strategy**

Add `visual_only` to `STRATEGIES`. Define field sets centrally:

```python
VISUAL_ONLY_FIELDS = frozenset({
    "light_intensity_scale", "camera_translation",
    "table_texture_id", "garment_display_color",
})
PHYSICS_AFFECTING_FIELDS = frozenset({
    "garment_yaw", "robot_base_translation", "garment_scale",
    "cloth_friction", "cloth_stiffness", "cloth_damping", "solver_iterations",
})
```

`sample_randomization("visual_only", rng)` must never emit a physics-affecting field. `validate_randomization_receipt` must require observed readback for every visual field.

- [ ] **Step 3: Restrict the strategy to authenticated success replay**

In `_persistent_collection_strategy`, allow `visual_only` only when `LEHOME_SUCCESS_REPLAY_CAMPAIGN=1` and the row passes `validate_success_replay_descriptor`. Fresh source rows remain canonical. Legacy mild/strong behavior remains unchanged outside the new mode.

- [ ] **Step 4: Preserve restored cloth state while applying visual fields**

In `apply_flywheel_randomization`, capture cloth positions, velocities, and pose before visual changes; apply camera/light/material-display values; then verify those cloth arrays and pose are byte-close to their pre-randomization readback. A mismatch is a fidelity failure, not a retryable policy failure.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=source/lehome uv run pytest -q \
  tests/flywheel/test_randomization.py \
  tests/flywheel/test_recovery_collection.py \
  tests/flywheel/test_persistent_worker.py
python3 -m py_compile \
  source/lehome/lehome/flywheel/models.py \
  source/lehome/lehome/flywheel/randomization.py \
  source/lehome/lehome/flywheel/recovery_collection.py \
  scripts/utils/evaluation.py
git diff --check
git add source/lehome/lehome/flywheel/models.py \
  source/lehome/lehome/flywheel/randomization.py \
  scripts/utils/evaluation.py \
  source/lehome/lehome/tasks/bedroom/garment_bi_v2.py \
  source/lehome/lehome/flywheel/recovery_collection.py \
  tests/flywheel/test_randomization.py \
  tests/flywheel/test_recovery_collection.py \
  tests/flywheel/test_persistent_worker.py
git commit -m "feat: add physics-invariant visual replay"
```

### Task 5: Restrict success replay to the new source campaign and exact caps

**Files:**
- Modify: `scripts/build_success_replay_matrix.py`
- Modify: `rollout_appliance/run_success_replay_campaign.sh`
- Modify: `rollout_appliance/run_12k_campaign.sh`
- Test: `tests/flywheel/test_success_replay_matrix.py`
- Test: `tests/infrastructure/test_success_replay_campaign.py`

- [ ] **Step 1: Write RED source-eligibility tests**

Test rejection of old round IDs, wrong campaign matrices, non-successes, non-accepted episodes, missing or mismatched Hub receipts, wrong policy, non-CPU cloth provenance, missing step-16 snapshot, safety/numerical/cloth failures, and mixed garment identity.

```python
def test_replay_rejects_success_from_unapproved_campaign(tmp_path, verified_success):
    report = authenticated_source_report(allowed_attempt_ids={"fresh-001"})
    verified_success["attempt_id"] = "old-001"
    with pytest.raises(ValueError, match="not in authenticated fresh source report"):
        build_matrix(
            accepted_roots=[write_episode(tmp_path, verified_success)],
            source_reports=[report],
            strategy="visual_only",
        )
```

- [ ] **Step 2: Add explicit new-mode builder inputs**

Add:

```text
--source-report CALIBRATION_REPORT
--source-report CURRICULUM_REPORT
--source-matrix CALIBRATION_MATRIX
--source-matrix CURRICULUM_MATRIX
--strategy visual_only
--attempt-cap-per-category 100
--acceptance-cap-per-category 50
--max-attempts 400
--target-accepted 200
--rng-seed 20260827400
```

Require every source episode to appear in exactly one authenticated report and source matrix, and require its Hub sync receipt to bind the same artifact digest and immutable remote prefix.

- [ ] **Step 3: Implement weighted parent sampling**

Within each category, weight source garments by `max(1 - fresh_success_rate, 0.01)`, then sample states uniformly within the selected garment. Allow replacement. Freeze no more than 100 rows per category and 400 globally. A category with no eligible source gets zero rows plus an explicit shortage receipt; never borrow another category.

```python
def replay_garment_weight(success_rate: float) -> float:
    return max(1.0 - success_rate, 0.01)
```

Each row uses `strategy="visual_only"`, `category_acceptance_cap=50`, and source hashes for the episode, reset, annotations, step-16 continuation snapshot, state fingerprint, source report, and source matrix.

- [ ] **Step 4: Raise caps only in the exact visual-only mode**

Legacy success replay remains capped at 150 accepted successes. The new wrapper admits 200 only when all four category caps are exactly 50, attempts are at most 100 per category/400 total, simulator is CPU, policy is pinned original-12K, and every row is `visual_only`.

- [ ] **Step 5: Run tests and shell syntax**

```bash
PYTHONPATH=source/lehome uv run pytest -q \
  tests/flywheel/test_success_replay_matrix.py \
  tests/infrastructure/test_success_replay_campaign.py
bash -n rollout_appliance/run_success_replay_campaign.sh
bash -n rollout_appliance/run_12k_campaign.sh
```

- [ ] **Step 6: Commit Task 5**

```bash
git add scripts/build_success_replay_matrix.py \
  rollout_appliance/run_success_replay_campaign.sh \
  rollout_appliance/run_12k_campaign.sh \
  tests/flywheel/test_success_replay_matrix.py \
  tests/infrastructure/test_success_replay_campaign.py
git commit -m "feat: bind replay to fresh curriculum sources"
```

### Task 6: Build the resumable one-VM host orchestrator

**Files:**
- Create: `scripts/run_simple_curriculum_collection.py`
- Create: `rollout_appliance/run_simple_curriculum_collection.sh`
- Test: `tests/infrastructure/test_simple_curriculum_orchestrator.py`

- [ ] **Step 1: Write RED state-machine tests**

Use a fake subprocess runner and fake immutable publisher. Cover:

- no command starts a VM or creates cloud resources;
- phase order is calibration-head, gate, calibration-tail, calibration report, curriculum build, curriculum-a, curriculum-b, fresh report, replay build, replay, final publication;
- gate failure stops immediately;
- preemption resumes the same stage/root/ledger;
- restart validates existing receipts and does not repeat terminal stages;
- a report/hash mismatch stops rather than rebuilding;
- no old source path reaches the replay builder;
- replay shortage is a reported terminal result;
- successful completion invokes a configured stop hook exactly once.

```python
def test_gate_failure_never_launches_later_stages(tmp_path):
    runner = FakeRunner(gate_decision="fidelity_stop")
    result = run_collection(config(tmp_path), runner=runner)
    assert result == "fidelity_stop"
    assert runner.partitions == ["calibration-head"]
    assert runner.replay_calls == []
```

- [ ] **Step 2: Implement an immutable stage journal**

Use this explicit stage set:

```python
STAGES = (
    "calibration-matrix",
    "calibration-head",
    "first-100-gate",
    "calibration-tail",
    "calibration-report",
    "curriculum-matrix",
    "curriculum-a",
    "curriculum-b",
    "fresh-report",
    "replay-matrix",
    "success-replay",
    "final-publication",
    "gpu-stop",
)
```

Each completion receipt binds its input hashes, output hashes, pinned runtime identity, command version, and predecessor receipt. Existing receipt collision or mismatch is fatal. The orchestrator owns no database beyond these JSON receipts and reuses one ledger per physical partition.

- [ ] **Step 3: Explicitly mount the checked-out code; do not rebuild an image**

The shell wrapper requires a regular, non-symlinked `LEHOME_HOST_CODE_ROOT` containing the current commit. Add that variable to `run_12k_campaign.sh` and replace baked script/source mounts only in the exact new mode:

```bash
-v "${LEHOME_HOST_CODE_ROOT}/scripts:/opt/lehome/scripts:ro"
-v "${LEHOME_HOST_CODE_ROOT}/source/lehome:/opt/lehome/source/lehome:ro"
-v "${LEHOME_HOST_CODE_ROOT}/rollout_appliance:/opt/lehome/rollout_appliance:ro"
```

For the Isaac worker container, mount the same checkout paths at `/opt/lehome-challenge/scripts` and `/opt/lehome-challenge/source/lehome/lehome`. Compute and record the Git commit plus a deterministic source-tree SHA before the first partition; resume requires the same values. Legacy mode keeps `/opt/lehome` mounts unchanged.

- [ ] **Step 4: Add the four physical partitions**

The orchestrator slices the already-written logical matrices without changing row bytes, writes partition manifests containing parent hash and row index interval, and invokes the existing campaign appliance once per partition. The first 100 rows are never regenerated after the gate.

- [ ] **Step 5: Add one configurable stop hook**

The orchestrator itself does not contain Nebius create/delete logic. It requires `LEHOME_GPU_STOP_COMMAND` only for a paid run and invokes it on collection complete or any terminal stop. Tests use a harmless fake command. Failure to stop is reported separately as `infrastructure_stop_failure`; it never changes a data result into success.

- [ ] **Step 6: Run tests and commit**

```bash
PYTHONPATH=source/lehome uv run pytest -q \
  tests/infrastructure/test_simple_curriculum_orchestrator.py \
  tests/infrastructure/test_simple_curriculum_campaign.py \
  tests/infrastructure/test_success_replay_campaign.py
python3 -m py_compile scripts/run_simple_curriculum_collection.py
bash -n rollout_appliance/run_simple_curriculum_collection.sh
bash -n rollout_appliance/run_12k_campaign.sh
git diff --check
git add scripts/run_simple_curriculum_collection.py \
  rollout_appliance/run_simple_curriculum_collection.sh \
  rollout_appliance/run_12k_campaign.sh \
  tests/infrastructure/test_simple_curriculum_orchestrator.py
git commit -m "feat: orchestrate one-vm curriculum collection"
```

### Task 7: Publish immutable matrices, reports, results, and seals

**Files:**
- Create: `scripts/publish_simple_curriculum_collection.py`
- Modify: `source/lehome/lehome/flywheel/hub_sync.py`
- Test: `tests/flywheel/test_hub_sync.py`
- Create: `tests/infrastructure/test_simple_curriculum_publication.py`

- [ ] **Step 1: Write RED publication tests**

Test immutable remote prefixes, upload-then-download byte equality, hash receipt binding, collision rejection, transient retry, no credential paths in payloads, incomplete accepted-artifact rejection, and distinct final outcomes for fidelity stop, insufficient source, and collection complete.

```python
def test_publication_requires_fresh_download_readback(tmp_path, fake_hub):
    result = publish_collection(bundle(tmp_path), transport=fake_hub)
    assert result.readback_verified is True
    assert fake_hub.downloaded_after_upload
    assert result.bundle_sha256 == sha256_tree(fake_hub.readback_root)
```

- [ ] **Step 2: Reuse Hub transport and add collection-bundle publication**

Do not change accepted-episode publication semantics. Add a small publisher that uploads the matrices, builder receipts, authenticated reports, stage receipts, success/failure terminal artifacts, replay artifacts, and final summary beneath:

```text
collection-rounds/<run-id>/manifests/
collection-rounds/<run-id>/fresh/
collection-rounds/<run-id>/replay/
collection-rounds/<run-id>/reports/
collection-rounds/<run-id>/seals/
```

Each local file is regular and non-symlinked. Each remote path is immutable. Download to a fresh temporary directory, verify every byte and SHA-256, then atomically write the local readback receipt.

- [ ] **Step 3: Define final seals**

The final seal records, separately:

```json
{
  "terminal_outcome": "collection_complete",
  "fresh_valid_outcomes": 1000,
  "fresh_official_successes": 0,
  "replay_attempts": 0,
  "replay_accepted_successes": 0,
  "replay_shortages": {},
  "all_hub_readbacks_verified": true,
  "gpu_stop_verified": true
}
```

Counts are populated from authenticated reports; zero is valid data, not a placeholder. A fidelity/insufficient-source seal contains only completed stages and the stopping evidence. Never claim `collection_complete` unless all 1,000 fresh rows are terminal and every required remote readback passes.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=source/lehome uv run pytest -q \
  tests/flywheel/test_hub_sync.py \
  tests/infrastructure/test_simple_curriculum_publication.py \
  tests/infrastructure/test_simple_curriculum_orchestrator.py
python3 -m py_compile scripts/publish_simple_curriculum_collection.py
git diff --check
git add scripts/publish_simple_curriculum_collection.py \
  source/lehome/lehome/flywheel/hub_sync.py \
  tests/flywheel/test_hub_sync.py \
  tests/infrastructure/test_simple_curriculum_publication.py
git commit -m "feat: publish curriculum collection with readback"
```

### Task 8: Document the exact paid-run handoff and verify the complete offline implementation

**Files:**
- Create: `docs/experiments/2026-08-27-simple-curriculum-runbook.md`
- Modify: `docs/experiments/README.md`
- Modify: `README.md`
- Test: `tests/infrastructure/test_simple_curriculum_runbook.py`

- [ ] **Step 1: Write the runbook contract test**

Assert the runbook names the original-12K pinned identity, one VM/four workers, CPU cloth/CUDA policy, 40 garments, 400/600/400 limits, the first-100 gate, public Hub readback, resume rules, stop hook, and explicit exclusions. Assert it contains no token values and no command that creates, deletes, or starts a VM.

- [ ] **Step 2: Write the handoff runbook**

Include only these operational phases:

1. verify one existing rollout VM is stopped and no other GPU VM is running;
2. upload/checkout the reviewed commit onto that VM without rebuilding an image;
3. validate the 40-garment catalog, checkpoint, runtime tree hash, public Hub destination, and stop hook;
4. separately authorize and start that exact VM;
5. run the one orchestrator command;
6. inspect the first-100 decision receipt;
7. let the same orchestrator resume the remaining stages only after `continue`;
8. verify final Hub readback receipt and stopped GPU state.

The runbook must say that implementation completion is not paid-run authorization.

- [ ] **Step 3: Run the focused and regression suites**

```bash
PYTHONPATH=source/lehome uv run pytest -q \
  tests/flywheel/test_simple_curriculum.py \
  tests/flywheel/test_simple_curriculum_gate.py \
  tests/flywheel/test_task_ledger.py \
  tests/flywheel/test_randomization.py \
  tests/flywheel/test_success_replay_matrix.py \
  tests/flywheel/test_recovery_collection.py \
  tests/flywheel/test_persistent_worker.py \
  tests/flywheel/test_hub_sync.py \
  tests/infrastructure/test_simple_curriculum_cli.py \
  tests/infrastructure/test_simple_curriculum_campaign.py \
  tests/infrastructure/test_simple_curriculum_orchestrator.py \
  tests/infrastructure/test_simple_curriculum_publication.py \
  tests/infrastructure/test_success_replay_campaign.py \
  tests/infrastructure/test_rollout_container.py \
  tests/infrastructure/test_simple_curriculum_runbook.py
bash -n rollout_appliance/run_12k_campaign.sh
bash -n rollout_appliance/run_success_replay_campaign.sh
bash -n rollout_appliance/run_simple_curriculum_collection.sh
python3 -m py_compile \
  scripts/build_simple_curriculum_matrix.py \
  scripts/check_simple_curriculum_gate.py \
  scripts/build_success_replay_matrix.py \
  scripts/run_simple_curriculum_collection.py \
  scripts/publish_simple_curriculum_collection.py
git diff --check
```

Expected: all listed tests pass; all shells parse; Python compilation and diff checks exit zero; no Docker, Isaac, Nebius, training, rollout, or Hub command runs.

- [ ] **Step 4: Verify stopped-at-rest state without mutating cloud resources**

Use the existing read-only Nebius status command from the active runbook. Record instance IDs and statuses in local verification notes. Expected: every GPU instance is stopped. If authentication is unavailable, record that cloud stop-state verification remains pending; do not sign in, start, stop, or delete anything as part of implementation.

- [ ] **Step 5: Review the complete diff**

```bash
git status --short
git diff --stat 27cd5c8..HEAD
git diff --check 27cd5c8..HEAD
git log --oneline -8
```

Confirm there is no new Terraform root, Packer image, service, database, training path, hard-state path, A-500 input, old rollout source, or automatic paid-run launch.

- [ ] **Step 6: Commit documentation**

```bash
git add docs/experiments/2026-08-27-simple-curriculum-runbook.md \
  docs/experiments/README.md README.md \
  tests/infrastructure/test_simple_curriculum_runbook.py
git commit -m "docs: add simple curriculum collection runbook"
```

## Paid-run authorization boundary

After all eight implementation tasks pass, stop. Report:

- reviewed commit SHA;
- exact offline test results;
- whether live read-only GPU stop state was verified;
- estimated maximum paid workload: 1,000 valid fresh outcomes plus at most 400 replay attempts on one existing VM;
- the first-100 circuit breaker and stop behavior;
- the public Hugging Face destination that will receive immutable readback-verified artifacts.

Do not start the VM or collection until the user separately authorizes the paid run after reviewing this implementation evidence.
