# Asynchronous Three-GPU Experiment Sweep Design

## Status

Approved conversational design, written for repository review before implementation planning.

## Objective

Replace the current one-experiment-at-a-time LeHome workflow with an asynchronous,
failure-isolated experiment system that uses at most three paid GPUs:

- one RTX PRO 6000 rollout/evaluation appliance;
- two RTX PRO 6000 training workers;
- one lightweight CPU control plane that does not count toward the GPU cap.

The system must determine which data type, mixture ratio, and training duration improve
the original step-12K policy. It must stop weak candidates early, keep every free worker
busy with the next compatible job, and never confuse infrastructure failures with policy
results.

## Non-goals

- Running every candidate for 2,000 steps regardless of early evidence.
- Training on unresolved hard-state failures or dead stalls.
- Treating deterministic weighted replay as canonical advantage-weighted regression.
- Mounting the protected 500 GiB block disk on multiple VMs simultaneously.
- Building a different VM image for each experiment.
- Starting four-worker recovery collection before the one-worker final12 teacher probe
  passes all fidelity and infrastructure gates.

## Chosen approach

Use an ASHA-style asynchronous successive-halving schedule implemented as a small,
repository-owned job controller rather than adopting a general-purpose tuning framework.
The search space is intentionally small and categorical, so a lightweight controller is
easier to audit and integrate with the existing immutable manifests, checkpoint receipts,
Hugging Face publication, preemption recovery, and Nebius lifecycle guards.

Rejected alternatives:

1. **Fixed full 2K grid.** Simple, but seven initial candidates would consume 14,000
   gradient steps before seed replication and would repeat the original waste pattern.
2. **Bayesian optimization.** The search space and evaluation sample count are too small
   and noisy to justify a probabilistic optimizer before the basic causal controls are
   established.

## Architecture

### CPU experiment controller

Run one small controller process on a CPU VM with a small protected system disk. It owns
the experiment database and exposes a narrow authenticated lease API to workers. The
controller stores metadata only: canonical job manifests, state transitions, hashes,
leases, heartbeats, retry classifications, evaluation summaries, and promotion decisions.
Datasets and checkpoints remain in Hugging Face or worker-local caches.

The controller is the single writer to a SQLite database. GPU workers communicate through
the controller API rather than sharing the database file. This preserves simple
transactions without relying on SQLite locking over a network filesystem.

Job states are:

```text
BLOCKED_DATA
READY
LEASED
TRAINING
PUBLISHING
EVAL_READY
EVALUATING
COMPLETED
PROMOTED
REJECTED
RETRYABLE
BLOCKED_INFRA
```

Every state change is append-only and carries a timestamp, worker identity, attempt
number, and immutable job digest. A worker heartbeat extends only its current lease. An
expired lease becomes `RETRYABLE`; it never creates a policy failure result.

### GPU 1: rollout and evaluation appliance

The existing final12 rollout VM remains the single simulator appliance. Its lifecycle is:

```text
one-worker zero-perturbation teacher probe
-> four-worker controlled-recovery collection if the probe passes
-> immutable recovery seal and Hugging Face readback
-> asynchronous checkpoint evaluation queue
-> stop when the evaluation queue is drained
```

Evaluation uses one loaded checkpoint and four persistent Isaac workers behind the
session-aware policy server. Each worker binds simulator, PhysX cloth, renderer,
cameras, and policy to one canonical CUDA device. Each worker has an isolated
shared-memory/IPC namespace, preparation watchdog, heartbeat, session ID, seed, and
attempt directory.

The appliance evaluates checkpoints one at a time to avoid multiple-policy VRAM pressure,
but executes four episodes concurrently. As soon as one checkpoint's matrix is complete,
it loads the next `EVAL_READY` checkpoint without waiting for a training wave.

### GPUs 2 and 3: training workers

Both workers use one generic training image. A worker:

1. leases the highest-priority compatible `READY` job;
2. verifies the image, code, parent policy, dataset, lineage, and job hashes;
3. hydrates immutable inputs into its local cache;
4. trains from the original step-12K parent or an authenticated promoted checkpoint;
5. writes local checkpoints every 500 steps;
6. publishes each terminal rung checkpoint immediately;
7. publishes 1K and 2K recovery checkpoints with fresh Hugging Face readback;
8. marks the checkpoint `EVAL_READY` and leases the next job.

The trainers never wait for each other. If no compatible job is ready, a worker enters a
short drain window. It stops after ten idle minutes rather than accruing open-ended GPU
cost. The controller retries the exact stopped VM when new work appears; it never creates
a duplicate as a retry side effect.

## Storage and artifact ownership

The protected disk `computedisk-u00pbe55crxy7jr56x` remains attached only to the rollout
VM. Nebius block disks are single-attach resources, so the two training VMs receive
independent local scratch/cache disks. The hot training dataset is decoded from local
storage, not a shared network filesystem.

Hugging Face is the durable canonical store for:

- the original step-12K parent and its pinned hashes;
- immutable BC, ordinary-success, and verified-recovery bundles;
- canonical experiment manifests;
- terminal-rung, 1K, and 2K checkpoints;
- evaluation reports and readback receipts.

No credential value may appear in a job manifest, image, disk snapshot, Git history,
controller event, or log. Workers receive narrowly scoped credentials at runtime.

## Immutable experiment contract

Each experiment is canonical JSON. Its experiment ID is the SHA-256 of the canonical
document. The required fields are:

```text
schema_version
experiment_id
parent_checkpoint { repository, revision, subpath, artifact_sha256 }
trainer { image_id, oci_digest, code_revision }
data_sources[] { kind, repository, revision, prefix, manifest_sha256, tree_sha256 }
mixture { bc_percent, added_percent, batch64_quotas, sampling_strategy }
training { action_horizon, batch_size, seed, target_step, save_steps }
evaluation { matrix_id, matrix_sha256, policy_digest }
publication { checkpoint_repository, result_repository, prefix }
dependencies[]
```

The controller rejects a document whose declared `experiment_id` does not equal its
canonical digest. A promoted job references the immutable prior-rung checkpoint receipt;
it cannot silently rediscover a different checkpoint.

The current runtime contract must be generalized before use. It presently requires a
positive rollout percentage and exactly 2,000 steps. The generalized contract must admit:

- 100/0 BC control jobs;
- approved 95/5, 90/10, 85/15, 80/20, and 70/30 profiles;
- terminal rungs at 500, 1,000, and 2,000 steps;
- multiple training seeds;
- immediate publication of every terminal rung.

Batch 64 uses the repository's stable largest-remainder rule:

| Mixture | BC slots | Added-data slots |
|---|---:|---:|
| 100/0 | 64 | 0 |
| 95/5 | 61 | 3 |
| 90/10 | 58 | 6 |
| 85/15 | 54 | 10 |
| 80/20 | 51 | 13 |
| 70/30 | 45 | 19 |

## Initial experiment set

All training candidates begin from the exact original step-12K checkpoint.

| Arm | Mixture | Added data | Question answered |
|---|---|---|---|
| A | 100/0 | none | Do additional gradient steps alone help or hurt? |
| B | 95/5 | ordinary successful rollout windows | Does a small ordinary-success dose help? |
| C | 70/30 | the prior ordinary-success mixture | Can the previous regression be reproduced early? |
| D | 95/5 | verified corrective recovery windows | Are recoveries better than ordinary successes at the same dose? |
| E | 90/10 | verified corrective recovery windows | Is a moderate recovery dose better? |
| F | 85/15 | verified corrective recovery windows | Is a larger recovery dose better? |
| G | 80/20 | verified corrective recovery windows | Does a high recovery dose help without overfitting? |

Arm G remains `BLOCKED_DATA` until there are at least 15 distinct successful recovery
trajectories in every category. Recovery datasets contain only authenticated corrective
`h=16` windows that end in successful recovery. They exclude unresolved failures, dead
stalls, and ordinary easy windows.

Arms A, B, and C are immediately runnable once the generalized profile contract exists.
Arms D through G depend on a readback-verified recovery bundle. This dependency does not
block the independent arms.

## Asynchronous promotion schedule

### Rung 1: 500 steps

Run every eligible arm for 500 steps with one seed. Seven full candidates consume 3,500
gradient steps rather than 14,000 steps for a blind seven-by-2K grid. Each terminal
checkpoint is published immediately and queued for frozen unseen-20 evaluation.

### Rung 2: 1,000 steps

After a candidate's unseen-20 result arrives, the controller updates its rung ranking.
At most three candidates are promoted to 1,000 total steps. Promotion is asynchronous: a
candidate need not wait for every slow or retrying peer if enough completed peers exist to
establish that it is in the current promotable set.

Ranking is lexicographic:

1. no safety failure;
2. highest minimum per-category success;
3. highest overall success;
4. highest paired recovery/progress improvement;
5. lower GPU time as the final tie-breaker.

A candidate is immediately rejected for a major safety failure. Infrastructure failures
do not participate in ranking.

### Seed check

Run a second 500-step seed for the top two configurations. If rankings reverse, retain
both through the 1K comparison rather than selecting on one lucky seed.

### Rung 3: 2,000 steps

Promote the best configuration to 2,000 total steps. Promote the runner-up only when the
1K paired evaluation remains tied within one overall unseen-20 episode and neither policy
has a category or safety advantage.

Evaluate finalists on the untouched unseen-80 matrix. A strict final winner must satisfy:

- at least 70% overall unseen-top success;
- at least 60% in each garment category;
- no major safety regression;
- no major seen-top regression.

Reuse an original step-12K baseline only when its checkpoint digest, exact evaluation
matrix digest, per-episode artifacts, and sealed report are all verified. Otherwise run the
baseline once on the same frozen matrix.

The expected training ceiling is 7,000 gradient-step equivalents: 3,500 initial steps,
1,500 promotion steps for three candidates, 1,000 steps for two second-seed canaries, and
1,000 final promotion steps. A tied runner-up raises the ceiling to 8,000. This is 43-50%
less training than taking all seven initial arms directly to 2K.

## Evaluation isolation

Unseen-20 is a development screen, not final proof. Five episodes per category means one
episode changes a category score by 20 percentage points. All candidates use the exact
same garments, seeds, randomization, success checks, and safety checks.

The untouched unseen-80 matrix is opened only for finalists. Reports include:

- overall and per-category success;
- paired wins, losses, and ties against step-12K;
- fold/progress measurements;
- recovery-event success;
- safety events;
- policy, matrix, image, code, dataset, and checkpoint digests;
- GPU time and infrastructure retry counts.

## AWR-style ablation

AWR-style weighted replay is excluded from the initial ratio sweep. After the best
unweighted recovery mixture is identified, run exactly one matched ablation with the same
parent, data, ratio, seed, steps, and evaluation matrix. The only changed field is the
deterministic replay weighting profile.

The result is called `AWR-style weighted replay`, not canonical AWR. Canonical AWR learns
a value function and uses estimated advantages to weight policy targets; the observed
GR00T path repeats selected windows deterministically instead.

## Failure isolation

| Failure class | Classification | Required action |
|---|---|---|
| Nebius capacity or API outage | infrastructure | Keep the exact job queued; retry the exact VM with bounded backoff; continue other lanes |
| Billing admission failure | infrastructure | Start no new GPU; preserve all jobs and artifacts |
| VM preemption after a verified checkpoint | retryable | Resume from the exact authenticated local or Hugging Face checkpoint |
| VM preemption before step 500 | retryable | Restart the short canary from its parent |
| Manifest, lineage, or continuation identity mismatch | deterministic infrastructure/configuration | Fail before training or simulation; do not retry unchanged bytes |
| Isaac preparation hang | infrastructure | Expire the worker lease, record `infrastructure_abort`, restart only that worker |
| Missing garment or USD asset | deterministic infrastructure/configuration | Reject during CPU preflight before paid simulation |
| HF publication transport failure | retryable publication | Keep local bytes, retry asynchronously, and block promotion until readback succeeds |
| Policy fails the task | experimental result | Record the failure; do not retry it as infrastructure |
| Recovery campaign misses category caps | data shortfall | Do not seal; keep only recovery-dependent arms blocked |

No single failure stops unrelated jobs. The controller may halt the entire system only for
a safety breach, corrupted canonical state, credential exposure, or spend-cap violation.

## Cost and backpressure controls

- Hard cap: three running GPU VMs.
- The rollout/evaluation GPU starts only for a pending smoke, collection, or evaluation
  job.
- Training workers stop after ten idle minutes.
- A trainer may take another training job while its background uploader drains, but the
  associated checkpoint cannot be promoted until fresh readback succeeds.
- If the evaluation backlog exceeds two checkpoints and no independent training arm is
  ready, the next idle trainer stops instead of waiting for promotion.
- The controller records estimated and actual GPU-seconds per job and rejects a lease that
  would exceed the configured campaign spend cap.

## Implementation decomposition

Implementation will be split into three independently testable plans:

1. **Experiment contracts and controller.** Generalized profiles, canonical job IDs,
   state machine, leases, retries, priorities, promotions, and CPU service.
2. **Nebius worker and storage integration.** Two elastic training workers, one existing
   rollout/evaluation worker, local caches, runtime secrets, lifecycle guards, and
   preemption recovery.
3. **Evaluation and experiment reporting.** Frozen matrices, four-worker evaluation queue,
   paired reports, promotion inputs, final unseen-80 gate, and the later matched AWR-style
   ablation.

Each plan must preserve the existing controlled-recovery smoke gate and must pass focused
CPU tests before any paid GPU is started.

## Acceptance criteria

The system is ready for paid use only when all of the following are demonstrated:

1. Canonical manifests represent every initial arm, including 100/0 and each 500-step
   rung, with stable content-derived IDs.
2. Two fake training workers lease different jobs and immediately take new work after
   completion without a wave barrier.
3. A simulated preemption retries only the affected job and resumes from the exact
   checkpoint receipt.
4. A deterministic manifest error fails before a GPU lease and is not retried forever.
5. An evaluation result promotes or rejects a candidate using the specified ordering.
6. The protected 500 GiB disk is never attached to more than the rollout VM.
7. Terminal-rung publication is readback-verified before promotion.
8. The final12 one-worker teacher probe passes before four-worker controlled collection.
9. The four-worker collection writes no strict seal unless exact category caps are met.
10. A dry-run campaign computes no more than three concurrent GPUs and the configured
    gradient-step and spend ceilings.

## References

- Li et al., *A System for Massively Parallel Hyperparameter Tuning*, MLSys 2020.
- Li et al., *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*,
  JMLR 2018.
- Peng et al., *Advantage-Weighted Regression: Simple and Scalable Off-Policy
  Reinforcement Learning*, 2019.
- Nebius AI Cloud, *Attaching and mounting Compute volumes to VMs*.
