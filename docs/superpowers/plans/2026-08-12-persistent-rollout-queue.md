# Persistent Rollout Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the next corrective rollout campaign with four long-lived Isaac workers that dynamically lease attempts and finalize artifacts in the background without duplicating evidence.

**Architecture:** Add an append-only SQLite task ledger, a pure deterministic scheduler adapter around the current corrective allocator, one persistent worker loop per GPU, and a bounded artifact-finalization queue. Persistent attempts have a schedule sequence independent of the actual execution worker; both identities are recorded instead of overloading the legacy wave `worker_slot`. The first version retains one GR00T server per worker. The current wave launcher and receipt schema remain unchanged as rollback.

**Tech Stack:** Python 3.11 standard-library SQLite/threads/subprocess, Isaac Sim 5.1, GR00T N1.7, Hugging Face Hub, pytest.

---

## File structure

- Create `source/lehome/lehome/flywheel/task_ledger.py`: durable attempts, leases, heartbeats, retries, and terminal transitions.
- Create `source/lehome/lehome/flywheel/artifact_queue.py`: bounded finalization workers and backpressure.
- Create `source/lehome/lehome/flywheel/persistent_worker.py`: reusable environment/policy worker state machine.
- Create `scripts/run_groot_persistent_campaign.py`: controller CLI and four-worker process lifecycle.
- Modify `scripts/run_groot_flywheel_trial.py`: expose the existing policy-server supervisor and one-episode trial setup as reusable context managers without changing its CLI path.
- Modify `scripts/run_groot_corrective_campaign.py`: expose the current deterministic next-attempt allocator as a reusable pure function.
- Modify `trainer/src/lehome_train/flywheel/corrective.py`: add typed persistent-attempt and campaign receipts without weakening legacy four-worker-wave validation.
- Modify `trainer/src/lehome_train/flywheel/publish.py`: consume ledger terminal receipts in the existing all-attempt release.
- Modify `scripts/run_groot_corrective_vast_lifecycle.py`: add opt-in persistent launch/sync actions without changing legacy wave actions.
- Create `tests/flywheel/test_task_ledger.py`, `test_artifact_queue.py`, and `test_persistent_worker.py`.
- Create `trainer/tests/test_persistent_campaign.py` and extend `test_corrective_publish.py` and `test_corrective_vast_lifecycle.py`.

### Task 1: Durable task ledger

**Files:**
- Create: `source/lehome/lehome/flywheel/task_ledger.py`
- Test: `tests/flywheel/test_task_ledger.py`

- [ ] **Step 1: Write transition, expiry, and duplicate tests**

```python
def test_lease_terminal_and_retry_are_append_only(tmp_path: Path) -> None:
    ledger = TaskLedger.create(tmp_path / "tasks.sqlite", generation_id="g1")
    ledger.add_attempt(attempt("a1"))
    lease = ledger.lease_next(worker_slot=0, now_unix=100, ttl_seconds=30)
    assert lease.attempt_id == "a1"
    ledger.heartbeat(lease.lease_id, now_unix=110, ttl_seconds=30)
    ledger.complete(lease.lease_id, terminal_sha256="a" * 64, now_unix=120)
    assert ledger.lease_next(worker_slot=1, now_unix=121, ttl_seconds=30) is None
    with pytest.raises(ValueError, match="terminal attempt"):
        ledger.add_attempt(attempt("a1"))


def test_expired_nonterminal_lease_creates_retry_record(tmp_path: Path) -> None:
    ledger = TaskLedger.create(tmp_path / "tasks.sqlite", generation_id="g1")
    ledger.add_attempt(attempt("a1"))
    first = ledger.lease_next(worker_slot=0, now_unix=100, ttl_seconds=10)
    second = ledger.lease_next(worker_slot=1, now_unix=111, ttl_seconds=10)
    assert second.attempt_id == "a1"
    assert second.retry_index == first.retry_index + 1
```

- [ ] **Step 2: Run and confirm missing module failure**

```bash
PYTHONPATH=source/lehome uv run --project trainer pytest -q \
  tests/flywheel/test_task_ledger.py
```

- [ ] **Step 3: Implement exact schema and transactions**

Use SQLite `BEGIN IMMEDIATE`, WAL mode, foreign keys, and these tables:

```sql
CREATE TABLE campaign (
  generation_id TEXT PRIMARY KEY,
  policy_sha256 TEXT NOT NULL,
  stopped_reason TEXT
);
CREATE TABLE attempts (
  attempt_id TEXT PRIMARY KEY,
  schedule_sequence INTEGER UNIQUE NOT NULL,
  payload_json BLOB NOT NULL,
  payload_sha256 TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN (
    'pending','leased','terminal_pending_validation',
    'accepted','rejected','infrastructure_abort')),
  terminal_sha256 TEXT UNIQUE
);
CREATE TABLE leases (
  lease_id TEXT PRIMARY KEY,
  attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
  worker_slot INTEGER NOT NULL,
  retry_index INTEGER NOT NULL,
  issued_at_unix INTEGER NOT NULL,
  expires_at_unix INTEGER NOT NULL,
  closed_at_unix INTEGER
);
CREATE TABLE events (
  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
  event_json BLOB NOT NULL,
  event_sha256 TEXT UNIQUE NOT NULL
);
```

Canonical JSON bytes and SHA-256 bind every transition. A terminal attempt can
never become pending again. `schedule_sequence` is immutable campaign order;
`leases.worker_slot` is the actual persistent worker that ran the attempt. They
must never be conflated with the legacy manifest's fixed `worker_slot`.

- [ ] **Step 4: Run ledger tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add source/lehome/lehome/flywheel/task_ledger.py \
  tests/flywheel/test_task_ledger.py
git commit -m "Add durable rollout task ledger"
```

### Task 2: Reuse the corrective allocator without wave barriers

**Files:**
- Modify: `trainer/src/lehome_train/flywheel/corrective.py`
- Modify: `scripts/run_groot_corrective_campaign.py`
- Create: `trainer/tests/test_persistent_campaign.py`

- [ ] **Step 1: Write completion-order independence tests**

```python
def test_next_attempts_do_not_depend_on_worker_completion_order() -> None:
    receipts_a = [receipt("a", "pant_long"), receipt("b", "top_short")]
    receipts_b = [receipt("b", "top_short"), receipt("a", "pant_long")]
    assert allocate_persistent_attempts(receipts_a, start_sequence=8, count=4) == \
        allocate_persistent_attempts(receipts_b, start_sequence=8, count=4)


def test_allocator_excludes_canonical_terminal_attempt_ids() -> None:
    allocated = allocate_persistent_attempts(
        [receipt("persistent-attempt-000000", "top_long")],
        start_sequence=1,
        count=4,
    )
    assert "persistent-attempt-000000" not in {
        item["attempt_id"] for item in allocated
    }


def test_schedule_identity_is_not_bound_to_execution_worker() -> None:
    attempt = allocate_persistent_attempts([], start_sequence=0, count=1)[0]
    assert "worker_slot" not in attempt
    assert attempt["schedule_sequence"] == 0
```

- [ ] **Step 2: Run the test and confirm the reusable API is missing**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_persistent_campaign.py
```

- [ ] **Step 3: Extract a pure allocator**

```python
def allocate_persistent_attempts(
    terminal_receipts: Iterable[Mapping[str, object]],
    *,
    start_sequence: int,
    count: int,
) -> tuple[dict[str, object], ...]:
    if type(count) is not int or count <= 0:
        raise ValueError("attempt allocation count must be positive")
    canonical = canonical_terminal_receipts(terminal_receipts)
    assessment = assess_corrective_campaign(
        canonical,
        policy=CorrectiveCampaignPolicy(),
        attempted_episodes=len(canonical),
        offered_hourly_cost_usd=2.0,
        rental_kind="on-demand",
    )
    categories = next_persistent_categories(assessment, count=count)
    return tuple(
        build_persistent_attempt(
            terminal_receipts=canonical,
            category=category,
            schedule_sequence=start_sequence + index,
        )
        for index, category in enumerate(categories)
    )
```

`build_persistent_attempt` derives deterministic attempt ID, seen garment, and
seed from `schedule_sequence`, but records no execution worker. Leasing adds the
actual worker in a separate receipt. Existing `_next_wave_categories`,
`_wave_manifest`, commands, and legacy output must remain byte-identical.

- [ ] **Step 4: Run persistent and legacy campaign tests**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_persistent_campaign.py \
  trainer/tests/test_corrective_campaign.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_groot_corrective_campaign.py \
  trainer/src/lehome_train/flywheel/corrective.py \
  trainer/tests/test_persistent_campaign.py
git commit -m "Expose corrective attempt allocator"
```

### Task 3: Bounded artifact finalization queue

**Files:**
- Create: `source/lehome/lehome/flywheel/artifact_queue.py`
- Test: `tests/flywheel/test_artifact_queue.py`

- [ ] **Step 1: Write overlap and backpressure tests**

```python
def test_finalization_runs_while_next_episode_starts(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    queue = ArtifactFinalizationQueue(
        max_pending_attempts=2,
        max_pending_bytes=1024,
        finalizer=blocking_finalizer(started, release),
    )
    queue.submit(finalized_tree(tmp_path, "a1", size=100))
    assert started.wait(1)
    queue.submit(finalized_tree(tmp_path, "a2", size=100))
    release.set()
    assert [item.attempt_id for item in queue.finish()] == ["a1", "a2"]


def test_queue_applies_backpressure_without_dropping_attempts(tmp_path: Path) -> None:
    queue = ArtifactFinalizationQueue(max_pending_attempts=1, max_pending_bytes=100, finalizer=noop)
    queue.submit(finalized_tree(tmp_path, "a1", size=100))
    assert queue.can_accept(size_bytes=1) is False
```

- [ ] **Step 2: Run and confirm failure**

```bash
PYTHONPATH=source/lehome uv run --project trainer pytest -q \
  tests/flywheel/test_artifact_queue.py
```

- [ ] **Step 3: Implement one bounded worker queue**

`submit()` accepts only closed real directories, records byte size before
enqueue, and returns immediately. The background worker calls the existing
`verify_episode_manifest`, hashes the closed tree, and invokes an injected
publisher. `finish()` returns results sorted by task submission sequence and
raises after preserving every failure receipt.

```python
@dataclass(frozen=True, slots=True)
class FinalizationResult:
    sequence: int
    attempt_id: str
    terminal_sha256: str
    accepted_success: bool
    publication: Mapping[str, object] | None
```

- [ ] **Step 4: Run artifact and existing artifact-verifier tests**

```bash
PYTHONPATH=source/lehome uv run --project trainer pytest -q \
  tests/flywheel/test_artifact_queue.py tests/flywheel/test_artifacts.py
```

- [ ] **Step 5: Commit**

```bash
git add source/lehome/lehome/flywheel/artifact_queue.py \
  tests/flywheel/test_artifact_queue.py
git commit -m "Finalize rollout artifacts asynchronously"
```

### Task 4: Persistent worker reset isolation

**Files:**
- Create: `source/lehome/lehome/flywheel/persistent_worker.py`
- Modify: `scripts/utils/evaluation.py`
- Modify: `scripts/run_groot_flywheel_trial.py`
- Test: `tests/flywheel/test_persistent_worker.py`

- [ ] **Step 1: Write garment-switch and reset tests with fakes**

```python
def test_worker_reuses_process_but_resets_episode_state() -> None:
    env = FakeEnvironment()
    policy = FakePolicyServer()
    worker = PersistentIsaacWorker(worker_slot=0, env=env, policy=policy)
    first = worker.run(attempt("a1", garment="Top_Long_Seen_0", seed=1))
    second = worker.run(attempt("a2", garment="Pant_Short_Seen_0", seed=2))
    assert env.start_count == 1
    assert env.switches == ["Top_Long_Seen_0", "Pant_Short_Seen_0"]
    assert policy.reset_count == 2
    assert first.output_root != second.output_root
    assert second.preflight["queued_actions"] == 0
```

- [ ] **Step 2: Run and confirm failure**

```bash
PYTHONPATH=source/lehome uv run --project trainer pytest -q \
  tests/flywheel/test_persistent_worker.py
```

- [ ] **Step 3: Implement the narrow worker state machine**

```python
class PersistentIsaacWorker:
    def run(self, attempt: Mapping[str, object]) -> WorkerTerminal:
        self._require_identity(attempt)
        self.env.switch_garment(attempt["garment_name"], "Release")
        self.env.cfg.seed = attempt["seed"]
        self.env.cfg.random_seed = attempt["seed"]
        self.env.garment_rng = np.random.RandomState(attempt["seed"])
        self.env.reset()
        self.policy.reset()
        self._require_clean_episode_state(attempt)
        return run_evaluation_loop(
            self.env,
            self.policy,
            self._episode_arguments(attempt),
            is_bimanual=True,
            garment_name=attempt["garment_name"],
        )
```

Add an opt-in evaluation hook that accepts an existing environment, existing
`GrootServerPolicy`, and an attempt-specific recorder manifest, then returns one
terminal outcome without closing the application. The hook must set and verify
the existing environment's seed fields and `garment_rng` before reset, invoke
the existing server `reset` endpoint through `policy.reset()`, and prove the
action queue is empty. There is no invented session-ID reset API. Legacy
single-episode evaluation still closes normally.

- [ ] **Step 4: Run worker, evaluation, and garment-switch tests**

```bash
PYTHONPATH=source/lehome uv run --project trainer pytest -q \
  tests/flywheel/test_persistent_worker.py \
  tests/flywheel/test_isaac_recorder.py \
  tests/flywheel/test_trial_cli.py
```

- [ ] **Step 5: Commit**

```bash
git add source/lehome/lehome/flywheel/persistent_worker.py \
  scripts/utils/evaluation.py scripts/run_groot_flywheel_trial.py \
  tests/flywheel/test_persistent_worker.py
git commit -m "Reuse Isaac workers across rollout attempts"
```

### Task 5: Persistent campaign controller CLI

**Files:**
- Create: `scripts/run_groot_persistent_campaign.py`
- Modify: `trainer/tests/test_persistent_campaign.py`

- [ ] **Step 1: Write slow-worker and stop-condition tests**

```python
def test_fast_worker_leases_again_before_slow_worker_finishes(tmp_path: Path) -> None:
    events = run_fake_campaign(tmp_path, durations={0: [1, 1], 1: [10], 2: [10], 3: [10]})
    assert event_index(events, "worker-0", "attempt-4", "leased") < event_index(
        events, "worker-1", "attempt-1", "terminal"
    )


def test_integrity_failure_stops_new_leases_but_preserves_active_attempts(tmp_path: Path) -> None:
    report = run_fake_campaign(tmp_path, fail_integrity_attempt="attempt-2")
    assert report["stop_reason"] == "integrity_failure"
    assert report["dropped_attempts"] == 0
```

- [ ] **Step 2: Run and confirm missing controller failure**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_persistent_campaign.py
```

- [ ] **Step 3: Implement controller dependencies and CLI**

The controller receives factories in tests and uses four spawned processes in
production. It fills the ledger to a configurable ready depth, starts one
worker per numeric GPU slot, consumes small terminal receipts, submits closed
trees to the artifact queue, and refills after each terminal.

```python
parser.add_argument("--mode", choices=("dry-run", "run"), default="dry-run")
parser.add_argument("--workers", type=int, choices=(4,), default=4)
parser.add_argument("--ready-depth", type=int, default=8)
parser.add_argument("--lease-ttl-seconds", type=int, default=1200)
parser.add_argument("--max-pending-attempts", type=int, default=8)
parser.add_argument("--max-pending-bytes", type=int, default=100 * 1024**3)
```

`dry-run` creates manifests and a ledger but starts no Isaac, policy, provider,
or Hub process.

- [ ] **Step 4: Run controller tests and compile**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_persistent_campaign.py \
  tests/flywheel/test_task_ledger.py \
  tests/flywheel/test_artifact_queue.py \
  tests/flywheel/test_persistent_worker.py
python3 -m py_compile scripts/run_groot_persistent_campaign.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_groot_persistent_campaign.py \
  trainer/tests/test_persistent_campaign.py
git commit -m "Add persistent corrective campaign controller"
```

### Task 6: Immutable publication from ledger evidence

**Files:**
- Modify: `trainer/src/lehome_train/flywheel/publish.py`
- Modify: `trainer/tests/test_corrective_publish.py`

- [ ] **Step 1: Write all-terminal and mutation tests**

```python
def test_persistent_release_contains_every_ledger_terminal(tmp_path: Path) -> None:
    bundle = persistent_bundle(tmp_path, terminal_count=8, success_count=3)
    staged = stage_release(bundle)
    assert json.loads((staged / "ledger/terminal-index.json").read_text())["count"] == 8


def test_publication_rejects_terminal_tree_mutated_after_ledger_close(tmp_path: Path) -> None:
    bundle = persistent_bundle(tmp_path, terminal_count=1, success_count=1)
    mutate_manifest_listed_file(bundle.attempt_roots[0])
    with pytest.raises(ValueError, match="terminal artifact"):
        stage_release(bundle)
```

- [ ] **Step 2: Run and confirm failure**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_corrective_publish.py
```

- [ ] **Step 3: Add a persistent ledger adapter, not a second publisher**

Build the existing `CorrectiveReleasePublicationBundle` from sorted ledger
terminal receipts. Re-run `verify_episode_manifest` for each tree at staging,
stage the closed ledger event log, and preserve the same immutable tree/readback
and disposal receipt fields. Do not upload SQLite WAL/SHM files.

- [ ] **Step 4: Run publisher and corrective RFT tests**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_corrective_publish.py trainer/tests/test_corrective_rft.py
```

- [ ] **Step 5: Commit**

```bash
git add trainer/src/lehome_train/flywheel/publish.py \
  trainer/tests/test_corrective_publish.py
git commit -m "Publish persistent rollout ledger evidence"
```

### Task 7: Vast persistent lifecycle actions

**Files:**
- Modify: `scripts/run_groot_corrective_vast_lifecycle.py`
- Modify: `trainer/tests/test_corrective_vast_lifecycle.py`

- [ ] **Step 1: Write opt-in lifecycle tests**

```python
def test_persistent_launch_starts_four_workers_once_and_syncs_incrementally(tmp_path: Path) -> None:
    commands = persistent_lifecycle_commands(tmp_path)
    remote = "\n".join(commands.ssh_scripts)
    assert remote.count("run_groot_persistent_campaign.py --mode run") == 1
    assert all(
        f"LEHOME_FLYWHEEL_WORKER_GPU={slot}" in remote
        for slot in range(4)
    )
    assert commands.sync_source.endswith("/campaign/")


def test_legacy_wave_action_is_byte_unchanged() -> None:
    assert legacy_remote_command(current_fixture()) == expected_legacy_command()
```

- [ ] **Step 2: Run and confirm persistent action is absent**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_corrective_vast_lifecycle.py
```

- [ ] **Step 3: Add `persistent-launch`, `persistent-sync`, and `persistent-stop`**

Reuse the current sanitized provider evidence, code-bundle hydration, R580 gate,
SSH safety, instance receipt, and destroy gate. `persistent-stop` stops new
leases and waits for active leases plus artifact finalization; it does not
destroy. Destruction continues to require the existing publisher disposal
receipt.

- [ ] **Step 4: Run lifecycle/campaign tests and shell compilation**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_corrective_vast_lifecycle.py \
  trainer/tests/test_persistent_campaign.py
python3 -m py_compile \
  scripts/run_groot_corrective_vast_lifecycle.py \
  scripts/run_groot_persistent_campaign.py
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_groot_corrective_vast_lifecycle.py \
  trainer/tests/test_corrective_vast_lifecycle.py
git commit -m "Add persistent rollout lifecycle actions"
```

### Task 8: Free acceptance and paid eight-episode smoke gate

**Files:**
- Modify: `trainer/tests/test_persistent_campaign.py`
- Create: `docs/groot_persistent_rollouts.md`

- [ ] **Step 1: Add a literal eight-episode acceptance fixture**

The fixture must prove four worker initializations, eight unique terminals,
worker 0 leasing its second task before the slowest first task terminates,
complete immutable publisher proof, and no secret-like fields.

- [ ] **Step 2: Run the complete free suite**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/flywheel/test_task_ledger.py \
  tests/flywheel/test_artifact_queue.py \
  tests/flywheel/test_persistent_worker.py \
  trainer/tests/test_persistent_campaign.py \
  trainer/tests/test_corrective_publish.py \
  trainer/tests/test_corrective_vast_lifecycle.py
python3 -m py_compile \
  scripts/run_groot_persistent_campaign.py \
  scripts/run_groot_corrective_vast_lifecycle.py
git diff --check
```

Expected: PASS.

- [ ] **Step 3: Document the exact paid gate**

Document commands for dry-run, capture fresh provider evidence, launch, monitor,
stop, publish, read back, and destroy. State that the current active wave is not
migrated; only a fresh next-run root may use persistent mode.

- [ ] **Step 4: Commit**

```bash
git add docs/groot_persistent_rollouts.md trainer/tests/test_persistent_campaign.py
git commit -m "Document persistent rollout acceptance"
```

- [ ] **Step 5: Run exactly one paid smoke after the current campaign ends**

Use a fresh manifest/root on one approved on-demand 4x3090 R580 host. Run eight
episodes with four workers. Compare startup seconds, total wall time, accepted
episodes/hour, CPU/GPU utilization, and artifact backlog against two legacy
four-episode waves. Do not promote persistent mode unless all eight attempts are
canonical, immutable-readback succeeds, and wall-clock improvement is material.

This paid step is operational and must not be simulated or claimed by unit
tests.
