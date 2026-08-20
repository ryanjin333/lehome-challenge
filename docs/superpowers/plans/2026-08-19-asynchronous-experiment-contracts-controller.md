# Asynchronous Experiment Contracts and Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build immutable multi-rung experiment manifests and a single-writer asynchronous controller that leases independent training/evaluation jobs without wave barriers.

**Architecture:** Generalize the existing GR00T runtime profile without weakening legacy 2K validation, then add focused `experiment_job`, `experiment_controller`, and `experiment_service` modules. The controller owns SQLite locally and exposes a narrow JSON-over-HTTP API; workers never open its database file.

**Tech Stack:** Python 3.10+, dataclasses, canonical JSON/SHA-256, SQLite WAL, standard-library HTTP, pytest.

---

### Task 1: Generalize immutable runtime profiles

**Files:**
- Modify: `trainer/src/lehome_train/groot/experiment_manifest.py`
- Modify: `trainer/src/lehome_train/groot/config.py`
- Test: `trainer/tests/test_experiment_manifest.py`
- Test: `trainer/tests/test_groot_config.py`

- [ ] **Step 1: Write failing quota/profile tests**

Add parameterized tests proving exact batch-64 quotas and the allowed rung budgets:

```python
@pytest.mark.parametrize(
    ("weights", "quotas"),
    [
        ({"bc": 100, "rollout": 0, "dagger": 0}, {"bc": 64, "rollout": 0, "dagger": 0}),
        ({"bc": 95, "rollout": 5, "dagger": 0}, {"bc": 61, "rollout": 3, "dagger": 0}),
        ({"bc": 90, "rollout": 10, "dagger": 0}, {"bc": 58, "rollout": 6, "dagger": 0}),
        ({"bc": 85, "rollout": 15, "dagger": 0}, {"bc": 54, "rollout": 10, "dagger": 0}),
        ({"bc": 80, "rollout": 20, "dagger": 0}, {"bc": 51, "rollout": 13, "dagger": 0}),
        ({"bc": 70, "rollout": 30, "dagger": 0}, {"bc": 45, "rollout": 19, "dagger": 0}),
    ],
)
def test_batch64_profiles(weights, quotas):
    assert batch64_quotas(weights) == quotas


@pytest.mark.parametrize("target_step", [500, 1000, 2000])
def test_sweep_profile_accepts_exact_rungs(profile_document, target_step):
    profile_document["training"]["target_step"] = target_step
    profile_document["training"]["terminal_publish"] = True
    assert load_sweep_runtime_profile(write_canonical(profile_document)).target_step == target_step
```

- [ ] **Step 2: Run the focused tests and observe RED**

Run:

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_experiment_manifest.py \
  trainer/tests/test_groot_config.py
```

Expected: failures showing 100/0 is rejected and `load_sweep_runtime_profile` does not exist.

- [ ] **Step 3: Add a versioned sweep profile without relaxing legacy v1**

Introduce a `SweepRuntimeProfile` dataclass and `load_sweep_runtime_profile`. Preserve
`load_experiment_manifest` and its exact legacy 2K invariants. The new schema must require:

```python
_SWEEP_STEPS = (500, 1000, 2000)
_SWEEP_WEIGHTS = ((100, 0), (95, 5), (90, 10), (85, 15), (80, 20), (70, 30))

@dataclass(frozen=True, slots=True)
class SweepRuntimeProfile:
    weights: Mapping[str, int]
    quotas: Mapping[str, int]
    target_step: int
    save_steps: int
    terminal_publish: bool
    action_horizon: int
    global_batch_size: int
```

Require canonical JSON, batch 64, horizon 16, save steps 500, terminal publication true,
DAgger zero, and exact approved ratios. Reject booleans where integers are required.

- [ ] **Step 4: Generalize launch configuration through an explicit sweep flag**

Add `runtime_sweep_profile: SweepRuntimeProfile | None` to `FineTuneLaunchConfig`. When
absent, retain the existing exact 2K runtime-mixture checks. When present, require its
target step to equal `max_steps`, batch 64, save steps 500, one GPU, and no gradient
accumulation.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the command from Step 2.

Expected: all tests pass, including unchanged legacy 70/30 and 80/20 fixtures.

- [ ] **Step 6: Commit the profile change**

```bash
git add trainer/src/lehome_train/groot/experiment_manifest.py \
  trainer/src/lehome_train/groot/config.py \
  trainer/tests/test_experiment_manifest.py \
  trainer/tests/test_groot_config.py
git commit -m "feat: add immutable sweep runtime profiles"
```

### Task 2: Define canonical experiment jobs

**Files:**
- Create: `trainer/src/lehome_train/groot/experiment_job.py`
- Create: `trainer/tests/test_experiment_job.py`

- [ ] **Step 1: Write failing manifest identity tests**

Cover a valid 500-step BC control, a promoted 1K recovery job, unknown fields, duplicate
JSON keys, unsafe prefixes, missing source hashes, invalid target steps, dependency cycles,
and a declared experiment ID that differs from the canonical digest.

```python
def test_job_id_is_sha256_of_identity_without_declared_id(tmp_path):
    document = valid_job_document()
    identity = dict(document)
    identity.pop("experiment_id")
    document["experiment_id"] = canonical_json_sha256(identity)
    job = load_experiment_job(write_canonical(tmp_path / "job.json", document))
    assert job.experiment_id == document["experiment_id"]
```

- [ ] **Step 2: Run the test and observe RED**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q trainer/tests/test_experiment_job.py
```

Expected: import failure for `lehome_train.groot.experiment_job`.

- [ ] **Step 3: Implement the exact schema**

Define immutable dataclasses for `ArtifactBinding`, `MixtureBinding`, `TrainingBudget`,
`EvaluationBinding`, `PublicationBinding`, and `ExperimentJob`. Implement:

```python
def load_experiment_job(path: str | Path) -> ExperimentJob: ...
def experiment_identity(document: Mapping[str, object]) -> str: ...
def dump_experiment_job(path: str | Path, identity: Mapping[str, object]) -> ExperimentJob: ...
```

The canonical identity excludes only `experiment_id`; every artifact path, revision,
SHA-256, image ID, OCI digest, code revision, ratio, quota, seed, rung, evaluation matrix,
publication prefix, and dependency participates in the digest.

- [ ] **Step 4: Add semantic validation**

Require:

- parent step-12K identity for root jobs;
- one authenticated prior-rung checkpoint for promoted jobs;
- `target_step` in 500/1000/2000;
- promoted target strictly greater than parent target;
- allowed mixtures and exact batch64 quotas;
- recovery jobs to depend on a recovery bundle receipt;
- arm G 80/20 jobs to declare `minimum_distinct_per_category >= 15`;
- unique dependencies and safe Hugging Face prefixes;
- no secret-shaped field names such as `token`, `password`, or `secret`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the command from Step 2.

Expected: all canonical identity and tamper tests pass.

- [ ] **Step 6: Commit canonical jobs**

```bash
git add trainer/src/lehome_train/groot/experiment_job.py trainer/tests/test_experiment_job.py
git commit -m "feat: define immutable experiment jobs"
```

### Task 3: Build the single-writer experiment ledger

**Files:**
- Create: `trainer/src/lehome_train/groot/experiment_controller.py`
- Create: `trainer/tests/test_experiment_controller.py`

- [ ] **Step 1: Write RED tests for schema and transitions**

Test bootstrap, reopen, exact schema verification, WAL enforcement, duplicate job
rejection, dependency blocking, worker-compatible leasing, idempotent same-worker lease,
heartbeat extension, expiry to `RETRYABLE`, deterministic failure to `BLOCKED_INFRA`, and
append-only event triggers.

```python
def test_workers_do_not_wait_for_a_wave(controller, ready_jobs):
    controller.add_jobs(ready_jobs)
    a = controller.lease_next("train-a", capability="training", now_ns=10, lease_ns=100)
    b = controller.lease_next("train-b", capability="training", now_ns=10, lease_ns=100)
    controller.complete(a, terminal_receipt_sha256="a" * 64, now_ns=20)
    c = controller.lease_next("train-a", capability="training", now_ns=21, lease_ns=100)
    assert {a.experiment_id, b.experiment_id, c.experiment_id} == {
        job.experiment_id for job in ready_jobs[:3]
    }
```

- [ ] **Step 2: Run the controller tests and observe RED**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q trainer/tests/test_experiment_controller.py
```

Expected: module import failure.

- [ ] **Step 3: Implement strict SQLite bootstrap**

Use `PRAGMA journal_mode=WAL`, `foreign_keys=ON`, `busy_timeout`, exact schema checks, and
`BEGIN IMMEDIATE` for every mutation. Tables:

```text
jobs
dependencies
leases
events
artifacts
evaluations
campaign_budget
```

Triggers must reject update/delete of events and mutation of canonical job identity bytes.
Expose a read-only audit connection using SQLite URI `mode=ro`.

- [ ] **Step 4: Implement transition methods**

Add:

```python
class ExperimentController:
    def add_jobs(self, jobs: Sequence[ExperimentJob]) -> None: ...
    def lease_next(self, worker_id: str, capability: str, now_ns: int, lease_ns: int) -> JobLease | None: ...
    def heartbeat(self, worker_id: str, lease_id: str, now_ns: int, lease_ns: int) -> JobLease: ...
    def complete(self, lease: JobLease, terminal_receipt_sha256: str, now_ns: int) -> None: ...
    def publication_verified(self, experiment_id: str, receipt_sha256: str, now_ns: int) -> None: ...
    def retryable(self, lease: JobLease, reason: str, now_ns: int) -> None: ...
    def block_infrastructure(self, lease: JobLease, reason: str, now_ns: int) -> None: ...
    def submit_evaluation(self, experiment_id: str, report: Mapping[str, object], now_ns: int) -> None: ...
```

Lease priority is: safety gates, baseline evaluation, 500-step arms, promoted 1K jobs,
second seeds, 2K finalists. Preserve FIFO order within one priority.

- [ ] **Step 5: Enforce budget admission**

Record configured GPU-seconds and gradient-step ceilings. `lease_next` must return no job
when admitting it could exceed either ceiling or the three-GPU global cap. Infrastructure
retries do not double-count completed GPU time, but every actual attempt contributes to
the spend ledger.

- [ ] **Step 6: Run tests and verify GREEN**

Run the command from Step 2.

Expected: all transition, concurrency, corruption, and budget tests pass.

- [ ] **Step 7: Commit the controller ledger**

```bash
git add trainer/src/lehome_train/groot/experiment_controller.py \
  trainer/tests/test_experiment_controller.py
git commit -m "feat: add asynchronous experiment controller"
```

### Task 4: Add promotion policy

**Files:**
- Create: `trainer/src/lehome_train/groot/experiment_promotion.py`
- Create: `trainer/tests/test_experiment_promotion.py`
- Modify: `trainer/src/lehome_train/groot/experiment_controller.py`

- [ ] **Step 1: Write failing ordering tests**

Cover safety rejection, minimum-category ordering, overall ordering, paired-progress
tie-break, GPU-time final tie-break, top-three 1K promotion, top-two seed replication,
and runner-up 2K promotion only when within one unseen-20 episode.

```python
def test_category_floor_beats_higher_overall_score():
    balanced = score(overall=11, category_successes=(3, 3, 3, 2))
    collapsed = score(overall=13, category_successes=(5, 5, 3, 0))
    assert rank_key(balanced) > rank_key(collapsed)
```

- [ ] **Step 2: Run the tests and observe RED**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q trainer/tests/test_experiment_promotion.py
```

- [ ] **Step 3: Implement pure promotion functions**

Define `EvaluationScore`, `rank_key`, `select_1k_promotions`, `select_seed_repeats`, and
`select_2k_finalists`. Reject non-finite metrics, duplicate policy digests, mismatched
matrix digests, and reports without all four categories.

- [ ] **Step 4: Integrate promotions transactionally**

The controller creates promoted jobs only from pre-generated immutable child manifests.
It verifies the child dependency references the exact readback-verified parent checkpoint
receipt before moving the child from `BLOCKED_DATA` to `READY`.

- [ ] **Step 5: Run focused tests and commit**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_experiment_promotion.py \
  trainer/tests/test_experiment_controller.py
git add trainer/src/lehome_train/groot/experiment_promotion.py \
  trainer/src/lehome_train/groot/experiment_controller.py \
  trainer/tests/test_experiment_promotion.py \
  trainer/tests/test_experiment_controller.py
git commit -m "feat: add asynchronous experiment promotions"
```

Expected: all tests pass.

### Task 5: Expose a narrow controller service and CLI

**Files:**
- Create: `trainer/src/lehome_train/groot/experiment_service.py`
- Create: `scripts/run_lehome_experiment_controller.py`
- Create: `trainer/tests/test_experiment_service.py`

- [ ] **Step 1: Write HTTP contract tests**

Test `/health`, `/lease`, `/heartbeat`, `/complete`, `/retryable`, `/publication`, and
`/evaluation`. Require exact JSON fields, bounded bodies, request IDs, bearer token file
authentication, idempotency keys, and no token value in responses or logs.

- [ ] **Step 2: Run RED test**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q trainer/tests/test_experiment_service.py
```

- [ ] **Step 3: Implement standard-library HTTP service**

Use `ThreadingHTTPServer` only for transport; all mutations remain serialized by the
controller transaction lock. Read the bearer token from an existing regular file with
mode 0600. Bind to a configured private address; reject wildcard binding unless an
explicit TLS-terminating proxy flag is supplied.

- [ ] **Step 4: Add CLI startup validation**

The CLI requires absolute safe paths for DB, token file, manifests directory, and audit
log. It verifies every job manifest before opening the listen socket and writes a durable
ready receipt containing only service address, schema version, controller DB identity,
and manifest-set digest.

- [ ] **Step 5: Run focused tests and commit**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_experiment_job.py \
  trainer/tests/test_experiment_controller.py \
  trainer/tests/test_experiment_promotion.py \
  trainer/tests/test_experiment_service.py
python3 -m py_compile \
  trainer/src/lehome_train/groot/experiment_job.py \
  trainer/src/lehome_train/groot/experiment_controller.py \
  trainer/src/lehome_train/groot/experiment_promotion.py \
  trainer/src/lehome_train/groot/experiment_service.py \
  scripts/run_lehome_experiment_controller.py
git diff --check
git add trainer/src/lehome_train/groot/experiment_service.py \
  scripts/run_lehome_experiment_controller.py \
  trainer/tests/test_experiment_service.py
git commit -m "feat: expose experiment controller service"
```

Expected: all focused tests, compilation, and diff checks pass.

### Task 6: Generate the seven-arm campaign manifest set

**Files:**
- Create: `scripts/build_lehome_experiment_sweep.py`
- Create: `trainer/tests/test_build_lehome_experiment_sweep.py`
- Create: `configs/experiments/lehome-recovery-sweep-v1/README.md`

- [ ] **Step 1: Write deterministic builder tests**

Require arms A-G, exact ratios, 500/1K/2K children, two seed-repeat children for the
top-two slots, 7,000 default and 8,000 tied-runner ceilings, recovery bundle dependencies,
and no credential fields.

- [ ] **Step 2: Run RED test**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q trainer/tests/test_build_lehome_experiment_sweep.py
```

- [ ] **Step 3: Implement canonical generation**

The builder accepts a canonical request containing the fixed parent, trainer image/OCI,
code revision, BC/ordinary-success/recovery artifact bindings, unseen-20 and unseen-80
matrix bindings, publication repositories, and campaign budget. It atomically writes one
directory of canonical job files plus `campaign.json` and SHA-256 sidecars. A recovery
binding may be declared pending; dependent jobs then begin as `BLOCKED_DATA`.

- [ ] **Step 4: Run builder and inspect output**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q trainer/tests/test_build_lehome_experiment_sweep.py
python3 -m py_compile scripts/build_lehome_experiment_sweep.py
git diff --check
```

Expected: deterministic rebuilds are byte-identical and every emitted job loads through
`load_experiment_job`.

- [ ] **Step 5: Commit the campaign builder**

```bash
git add scripts/build_lehome_experiment_sweep.py \
  trainer/tests/test_build_lehome_experiment_sweep.py \
  configs/experiments/lehome-recovery-sweep-v1/README.md
git commit -m "feat: build the recovery experiment sweep"
```
