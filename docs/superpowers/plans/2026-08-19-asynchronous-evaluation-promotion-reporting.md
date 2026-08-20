# Asynchronous Evaluation, Promotion, and Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate completed checkpoints on fixed all-category matrices, submit authenticated scores to the controller, promote candidates asynchronously, and produce a sealed final comparison plus a matched AWR-style ablation.

**Architecture:** Reuse the persistent four-worker rollout appliance and existing evaluation artifact validation. Add immutable dev/final matrix bindings, a queue adapter that loads one checkpoint at a time, a strict all-category report schema, and pure promotion inputs consumed by the controller.

**Tech Stack:** Python, canonical JSON/SHA-256, existing Isaac persistent workers and policy server, pytest.

---

### Task 1: Create a disjoint frozen unseen-20 development matrix

**Files:**
- Create: `configs/eval_groot_n17_unseen20_dev.json`
- Create: `configs/eval_groot_n17_unseen20_dev.json.sha256`
- Create: `trainer/tests/test_unseen20_matrix.py`

- [ ] **Step 1: Write matrix invariants**

Require exactly five episodes per category, all `public_unseen`, unique trial IDs, seeds
disjoint from `configs/eval_groot_n17_public_280.json`, all four held-out garment
categories, and no seen garment.

```python
def test_dev20_is_episode_disjoint_from_final80():
    dev = load_matrix(DEV20)
    final = public_unseen_trials(load_matrix(PUBLIC_280))
    assert {(t.garment_name, t.seed) for t in dev.trials}.isdisjoint(
        {(t.garment_name, t.seed) for t in final}
    )
```

- [ ] **Step 2: Generate canonical matrix bytes**

Use `Unseen_0` garments and five seeds absent from the final-80 seed set. Sort by category
order `top_long`, `top_short`, `pant_long`, `pant_short`, then seed. Write canonical
single-line JSON and a lowercase SHA-256 sidecar.

- [ ] **Step 3: Run tests and commit**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q trainer/tests/test_unseen20_matrix.py
git diff --check
git add configs/eval_groot_n17_unseen20_dev.json \
  configs/eval_groot_n17_unseen20_dev.json.sha256 \
  trainer/tests/test_unseen20_matrix.py
git commit -m "test: freeze disjoint unseen20 development matrix"
```

Expected: the matrix has exactly 20 trials and no episode identity overlaps final unseen80.

### Task 2: Define authenticated all-category evaluation reports

**Files:**
- Create: `trainer/src/lehome_train/groot/experiment_evaluation.py`
- Create: `trainer/tests/test_experiment_evaluation.py`
- Modify: `scripts/summarize_groot_persistent_evaluation.py`

- [ ] **Step 1: Write RED report tests**

Cover exact four-category counts, paired trial identities, safety events, continuous
progress, recovery events, policy/matrix/image/code/data/checkpoint digests, infrastructure
retry counts, GPU-seconds, duplicate rows, missing artifacts, and non-finite metrics.

```python
def test_report_requires_all_four_categories(valid_report):
    valid_report["categories"].pop("pant_short")
    with pytest.raises(ValueError, match="four categories"):
        load_experiment_evaluation(write_report(valid_report))
```

- [ ] **Step 2: Run RED tests**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_experiment_evaluation.py \
  tests/flywheel/test_persistent_evaluation_summary.py
```

- [ ] **Step 3: Implement immutable report loading**

Define `CategoryScore`, `ExperimentEvaluation`, `load_experiment_evaluation`, and
`build_experiment_evaluation`. Bind the report to the experiment ID and terminal
checkpoint publication receipt. Require every episode artifact to have a terminal
controller outcome and verified identity.

- [ ] **Step 4: Extend the existing summarizer**

Add an explicit `--experiment-job` and `--checkpoint-publication-receipt` mode. Preserve
legacy output when those flags are absent. In experiment mode, emit the strict report and
SHA-256 sidecar atomically with file and directory fsync.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_experiment_evaluation.py \
  tests/flywheel/test_persistent_evaluation_summary.py
python3 -m py_compile trainer/src/lehome_train/groot/experiment_evaluation.py \
  scripts/summarize_groot_persistent_evaluation.py
git diff --check
git add trainer/src/lehome_train/groot/experiment_evaluation.py \
  trainer/tests/test_experiment_evaluation.py \
  scripts/summarize_groot_persistent_evaluation.py
git commit -m "feat: add authenticated experiment evaluation reports"
```

### Task 3: Add the rollout-appliance evaluation queue adapter

**Files:**
- Create: `scripts/run_lehome_experiment_evaluator.py`
- Create: `rollout_appliance/run_experiment_evaluator.sh`
- Modify: `infrastructure/nebius/packer/scripts/install-rollout.sh`
- Create: `tests/flywheel/test_experiment_evaluator.py`
- Modify: `tests/infrastructure/test_rollout_container.py`

- [ ] **Step 1: Write queue-adapter tests**

Test controller lease capability `evaluation`, checkpoint readback verification, one
loaded policy at a time, four worker processes, identical matrix bytes, heartbeat during
evaluation, immediate next-checkpoint lease, infrastructure retry, and policy failures
remaining experimental outcomes.

- [ ] **Step 2: Run RED tests**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/flywheel/test_experiment_evaluator.py \
  tests/infrastructure/test_rollout_container.py
```

- [ ] **Step 3: Implement evaluator orchestration**

For each lease:

1. authenticate the experiment job and checkpoint publication receipt;
2. hydrate the exact checkpoint into a unique local directory;
3. verify the frozen matrix digest;
4. start one policy server and exactly four persistent Isaac workers;
5. drain the matrix without wave barriers;
6. finalize artifacts and build the strict evaluation report;
7. submit the report to the controller;
8. stop the policy server and lease the next checkpoint.

Do not run two policy checkpoints concurrently on one GPU.

- [ ] **Step 4: Package without enabling automatic collection**

Install the evaluator wrapper in the rollout image. Leave it disabled by default and
require an explicit experiment-controller environment file. Preserve the final12 teacher
probe and controlled collection gates unchanged.

- [ ] **Step 5: Run focused tests and commit**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/flywheel/test_experiment_evaluator.py \
  tests/flywheel/test_persistent_worker.py \
  tests/infrastructure/test_rollout_container.py \
  tests/infrastructure/test_packer_contract.py
bash -n rollout_appliance/run_experiment_evaluator.sh
python3 -m py_compile scripts/run_lehome_experiment_evaluator.py
git diff --check
git add scripts/run_lehome_experiment_evaluator.py \
  rollout_appliance/run_experiment_evaluator.sh \
  infrastructure/nebius/packer/scripts/install-rollout.sh \
  tests/flywheel/test_experiment_evaluator.py \
  tests/infrastructure/test_rollout_container.py
git commit -m "feat: add asynchronous checkpoint evaluator"
```

### Task 4: Wire evaluation submission to promotion

**Files:**
- Modify: `trainer/src/lehome_train/groot/experiment_controller.py`
- Modify: `trainer/src/lehome_train/groot/experiment_promotion.py`
- Modify: `trainer/tests/test_experiment_controller.py`
- Modify: `trainer/tests/test_experiment_promotion.py`

- [ ] **Step 1: Write end-to-end asynchronous promotion test**

Simulate two training workers and one evaluator. Complete jobs in a different order from
their schedule, submit evaluations incrementally, and prove the first free trainer leases
a promoted job as soon as its promotion is justified.

```python
def test_promotion_has_no_global_wave_barrier(campaign):
    campaign.complete_and_score("arm-d-500", score=strong_score())
    campaign.complete_and_score("arm-a-500", score=control_score())
    promoted = campaign.lease_training("train-a")
    assert promoted.parent_experiment_id == "arm-d-500"
```

- [ ] **Step 2: Implement transactional evaluation admission**

Require evaluation report experiment ID, policy digest, checkpoint receipt digest, and
matrix digest to match the job. Insert report and promotion events in one transaction.
Never promote an unpublished checkpoint.

- [ ] **Step 3: Add backpressure transitions**

Expose the count of `EVAL_READY` jobs. When the backlog exceeds two and no independent
training job is ready, return `idle_stop_recommended=true` to training workers. Do not
cancel active training.

- [ ] **Step 4: Run focused tests and commit**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_experiment_controller.py \
  trainer/tests/test_experiment_promotion.py \
  trainer/tests/test_experiment_evaluation.py
git diff --check
git add trainer/src/lehome_train/groot/experiment_controller.py \
  trainer/src/lehome_train/groot/experiment_promotion.py \
  trainer/tests/test_experiment_controller.py \
  trainer/tests/test_experiment_promotion.py
git commit -m "feat: promote candidates from verified evaluations"
```

### Task 5: Add final unseen-80 winner gate and baseline reuse proof

**Files:**
- Create: `trainer/src/lehome_train/groot/experiment_winner.py`
- Create: `trainer/tests/test_experiment_winner.py`
- Modify: `trainer/src/lehome_train/challenge_evaluation.py`

- [ ] **Step 1: Write winner and baseline tests**

Require 70% overall, 60% each category, no safety regression, no major seen regression,
exact final-80 matrix digest, and baseline reuse only with matching checkpoint/matrix
digests plus a sealed per-episode report.

- [ ] **Step 2: Run RED tests**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_experiment_winner.py \
  trainer/tests/test_challenge_evaluation.py
```

- [ ] **Step 3: Implement strict winner selection**

Add `verify_reusable_baseline`, `winner_gate`, and `select_final_winner`. A missing or
unsealed baseline returns `baseline_evaluation_required`; it never substitutes the
conversation's informal score.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_experiment_winner.py \
  trainer/tests/test_challenge_evaluation.py
git diff --check
git add trainer/src/lehome_train/groot/experiment_winner.py \
  trainer/tests/test_experiment_winner.py \
  trainer/src/lehome_train/challenge_evaluation.py
git commit -m "feat: enforce final experiment winner gate"
```

### Task 6: Add the matched AWR-style ablation builder

**Files:**
- Create: `trainer/src/lehome_train/groot/experiment_ablation.py`
- Create: `trainer/tests/test_experiment_ablation.py`

- [ ] **Step 1: Write exact-difference tests**

Given the best unweighted recovery job, generate one AWR-style child and assert every
field is identical except experiment ID, deterministic replay evidence/configuration,
publication prefix, and the dependency on the unweighted result.

- [ ] **Step 2: Run RED test**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q trainer/tests/test_experiment_ablation.py
```

- [ ] **Step 3: Implement matched ablation generation**

Require authenticated progress evidence and use existing `AwrReplayConfig`. Name the arm
`awr_style_weighted_replay`; reject `awr` as the canonical algorithm label. Preserve the
same parent step-12K, data, ratio, seed, target step, and evaluation matrices.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_experiment_ablation.py \
  trainer/tests/test_awr_weighting.py
git diff --check
git add trainer/src/lehome_train/groot/experiment_ablation.py \
  trainer/tests/test_experiment_ablation.py
git commit -m "feat: build matched AWR-style ablation"
```

### Task 7: Run the CPU-only end-to-end campaign simulation

**Files:**
- Create: `trainer/tests/test_asynchronous_experiment_campaign.py`
- Modify: `docs/nebius_training_rollout.md`

- [ ] **Step 1: Build an end-to-end fake campaign test**

The test must generate arms A-G, keep recovery arms blocked, run A-C on two fake trainers,
unblock D-G after a fake recovery seal, evaluate out of order on one fake evaluator,
promote top candidates, inject one preemption and one deterministic manifest failure,
run the seed check, select a 2K finalist, enforce the final winner gate, and assert at most
three concurrent GPU leases and at most 8,000 gradient-step equivalents.

- [ ] **Step 2: Document operator workflow**

Update the runbook with controller bootstrap, manifest-set verification, stopped-at-rest
pool validation, dry-run simulation, admission checks, rollout teacher probe, data
dependency release, asynchronous monitoring, idle-stop behavior, and final teardown.

- [ ] **Step 3: Run the complete CPU suite**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_experiment_job.py \
  trainer/tests/test_experiment_controller.py \
  trainer/tests/test_experiment_promotion.py \
  trainer/tests/test_experiment_service.py \
  trainer/tests/test_experiment_worker.py \
  trainer/tests/test_experiment_evaluation.py \
  trainer/tests/test_experiment_winner.py \
  trainer/tests/test_experiment_ablation.py \
  trainer/tests/test_asynchronous_experiment_campaign.py \
  tests/flywheel/test_experiment_evaluator.py \
  tests/infrastructure/test_experiment_pool_terraform.py \
  tests/infrastructure/test_experiment_pool_dry_run.py \
  tests/infrastructure/test_nebius_experiment_worker.py \
  tests/infrastructure/test_nebius_experiment_controller.py
git diff --check
```

Expected: all tests pass with no network, cloud, Docker GPU, Isaac, Packer build, Terraform
apply, VM start, or Hugging Face mutation.

- [ ] **Step 4: Commit integration and runbook**

```bash
git add trainer/tests/test_asynchronous_experiment_campaign.py docs/nebius_training_rollout.md
git commit -m "test: simulate asynchronous three-gpu experiment campaign"
```
