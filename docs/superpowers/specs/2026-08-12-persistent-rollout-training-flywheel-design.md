# Persistent LeHome Rollout and Training Flywheel Design

**Status:** Approved design
**Date:** 2026-08-12

## Objective

Replace the current four-episode wave barrier and repeated machine setup with a
generation-based flywheel that spends most paid time simulating or optimizing:

- four long-lived Isaac Sim workers pull episodes from one dynamic queue;
- episode finalization, validation, and Hugging Face transfer run concurrently
  with later episodes;
- one RTX PRO 6000 trainer is measured and tuned for the largest efficient
  physical batch;
- one training process continues through the 1,000- and 2,000-step checkpoints;
- every training generation consumes one immutable, uncontaminated dataset;
- old and new policies are evaluated on identical fixed matrices; and
- paid instances remain alive only while they are useful and are destroyed only
  after immutable publication and readback.

The first implementation optimizes orchestration, not the learning algorithm.
It preserves success-only RFT, the organizer-data anchor, category-aware
sampling, current policy provenance, and the untouched unseen benchmark.

## Evidence and Current Bottlenecks

The current corrective collector safely binds every attempt to a policy, code
revision, simulator/runtime identity, provider receipt, garment, and seed. It
also supports retaining one Vast lease across waves. The expensive part is the
execution shape:

- a wave launches up to four new Isaac/controller/policy process groups;
- the controller waits for all four, even when one garment takes much longer;
- canonical campaign output is synchronized after the wave barrier;
- the next four attempts do not start until validation and lifecycle work ends;
- setup and model hydration are repeated when a new machine is rented; and
- four independent GR00T model copies do not share inference batches.

The checked-in trainer is reproducible and resumable, and checkpoint upload is
already placed on a bounded background queue. It is not yet throughput-optimal:

- single-GPU smoke tests only cover physical batches 16, 32, and 64;
- the four-GPU profile forces per-device batch 1, for global batch 4;
- training is driven in checkpoint-sized process chunks, which can repeat model,
  optimizer, dataloader, and compilation initialization;
- checkpoint tar creation and hashing remain synchronous; and
- dataloader workers are fixed at four rather than selected from measurements.

The public winning LeHome solution demonstrates the useful architecture, not a
drop-in GR00T configuration: it used persistent Isaac workers, a shared task
queue, background dataset writing, asynchronous Hub synchronization, batch 192
and eight loader workers on an H200 trainer, and mostly RTX PRO 6000 rollout
hardware. Its JAX pi-policy batch size cannot be copied directly into GR00T.

## Chosen Approach

Implement the optimization in two bounded stages.

### Stage 1: persistent collection with independent policy servers

Keep four policy servers, one per GPU, but keep each Isaac worker and policy
server alive across many episodes. Add the shared queue and asynchronous
artifact pipeline around them. This removes process startup, wave barriers, and
slow-worker idle time without introducing cross-worker policy-session coupling.

### Stage 2: optional shared batched GR00T inference

After Stage 1 has a measured throughput baseline, separately test one shared
GR00T server that waits for a short batching window, stacks concurrent requests,
and routes the returned action chunks to their originating sessions. Promote it
only if it improves accepted episodes per dollar while preserving byte-level
request/response identity and independent session state.

Shared batching is deliberately not part of the first implementation. It has a
larger failure radius, can suffer from poor batch occupancy when simulators are
out of phase, and needs explicit cancellation, timeout, request-routing, and
session-isolation proofs.

### Training: one measured 96 GB GPU

Use one RTX PRO 6000 96 GB for short corrective RFT training. Do not use the
existing four-GPU, batch-1-per-device profile unless a later benchmark proves it
is faster and cheaper. Run one long-lived optimization process through steps
1,000 and 2,000, with checkpoint publication occurring behind it.

This is the preferred configuration under the shared $2/hour account cap. An
H200 may be tested only as a separately approved cost experiment. Hardware is
selected by measured samples per second and measured dollars per training run,
not model name alone. The first persistent run keeps the verified global batch
of 64; larger batches are benchmark evidence for a later admitted run because
changing batch while retaining step-1,000/2,000 checkpoints also changes sample
exposure and optimization behavior.

### Host-driver policy

Training and simulation have different driver gates. They must not share one
blanket R580-only rule.

The RTX PRO 6000 Blackwell trainer may use the provider's newer host driver.
The pinned training container uses CUDA 12.8, and newer NVIDIA drivers support
older CUDA runtimes through backward compatibility. Training admission therefore
requires a minimum compatible driver, the exact OCI image, successful CUDA and
NCCL/library probes relevant to the selected topology, and a real optimizer-step
smoke. It does not impose the rollout runtime's `<590` ceiling.

Isaac Sim rollout admission is stricter because CUDA success does not prove
Vulkan/RTX rendering, CUDA interop, cloth contact behavior, or simulator parity.
Production collection initially retains the reviewed R580 window on 4x3090
hosts. An R590/595 Blackwell rollout host is `unverified`, not inherently
incompatible. It may enter the allowlist only after the exact pinned rollout
image passes a bounded canary covering renderer startup, cameras, policy
inference, visible cloth contact, canonical terminal artifacts, and comparison
with the admitted CPU-physics reference.

The controller never tries to downgrade or replace Vast's injected host driver
from inside a container. A failed driver gate selects a different host. A newly
validated driver branch is recorded as an explicit image-and-driver capability
tuple rather than widening the gate to every newer version.

## Architecture

```text
                     generation controller
                    /          |           \
                   /           |            \
        immutable policy   task ledger    artifact ledger
                |               |               |
                v               v               v
      +----------------+  leased attempts  validate/hash/upload
      | policy server 0|<-- worker 0 -------+       |
      | Isaac worker 0 |                    |       |
      +----------------+                    |       v
      +----------------+  leased attempts   |  private HF staging
      | policy server 1|<-- worker 1 -------+       |
      | Isaac worker 1 |                    |       |
      +----------------+                    |       |
      +----------------+  leased attempts   |       |
      | policy server 2|<-- worker 2 -------+       |
      | Isaac worker 2 |                    |       |
      +----------------+                    |       |
      +----------------+  leased attempts   |       |
      | policy server 3|<-- worker 3 -------+       |
      | Isaac worker 3 |                    |       |
      +----------------+                    |       |
                                             v       v
                                  immutable generation dataset
                                             |
                                             v
                                  persistent PRO 6000 trainer
                                      |               |
                                   step 1000       step 2000
                                      |               |
                                      +---- fixed evaluation matrix
```

The rollout instance and training instance are separate roles. They may overlap
only when a fresh provider query proves the complete account-wide hourly total,
including retained storage, remains below $2/hour. Otherwise the controller
finishes and disposes the rollout rental before renting the trainer.

## Persistent Rollout Worker

Each worker owns one Isaac application, one environment, one renderer GPU, and
one GR00T policy server for its lifetime. Current simulation parity remains:

- cloth physics, environment state, control, and success checks use the admitted
  CPU simulation path;
- RTX rendering uses the assigned GPU; and
- GR00T inference uses the assigned GPU.

GPU cloth physics is not enabled by this project. It requires a separate parity
and contact-behavior acceptance experiment.

For every task, the worker:

1. leases one immutable attempt identity from the queue;
2. resets or switches the garment using the existing environment interface;
3. resets all per-episode policy/session state and seeds;
4. runs the episode with its assigned worker/GPU identity;
5. closes the raw attempt into an immutable local terminal directory;
6. writes a small completion receipt to the ledger; and
7. immediately requests another task.

The worker never waits for another worker's episode. A worker process may be
restarted individually after a bounded failure or measured memory leak; the
other workers continue. The controller records restarts and will not reuse an
attempt identity whose canonical terminal artifact already exists.

Persistent execution must not leak state between episodes. The acceptance test
checks garment identity, simulator seed and garment RNG, the client's local
action queue, the server's existing reset endpoint, camera buffers, success
state, and output paths before each lease is admitted. The first release does
not invent multi-session policy-server semantics: each server remains private
to one worker.

## Dynamic Task Queue

The queue is an append-only local ledger, not an in-memory-only work list. Each
attempt moves monotonically through:

```text
pending -> leased -> terminal_pending_validation
        -> accepted | rejected | infrastructure_abort
```

A lease contains attempt ID, generation ID, policy digest, category, garment,
seed, worker slot, issue time, and expiry. Heartbeats renew an active lease. If a
worker dies, an expired nonterminal lease becomes retryable under a new retry
record; it does not erase the original evidence.

Scheduling remains deterministic from the campaign state. Category floors and
effort-deficit prioritization decide which attempt is created next. Dynamic
scheduling changes only which available worker receives it. This preserves the
same statistical campaign policy while eliminating the four-worker barrier.

The queue stops issuing work at the first terminal condition: all success and
uniqueness floors pass, the maximum attempt budget is reached, the cost ceiling
is reached, an integrity failure occurs, or the operator stops the campaign.

## Asynchronous Artifact Pipeline

During an episode, raw observations and actions stream to an attempt-scoped
pending directory. At terminal state the worker atomically closes that directory
and enqueues it for background work. The worker may then start the next episode.

A bounded background pipeline performs:

1. canonical schema validation;
2. manifest and file hashing;
3. success/category/uniqueness admission;
4. optional diagnostic video encoding;
5. content-addressed upload to private Hugging Face staging; and
6. immutable download/readback verification.

Backpressure is explicit. If local finalized bytes or queued attempts exceed a
configured bound, workers stop leasing new tasks until the writer catches up.
Nothing is dropped to maintain throughput.

Small terminal receipts reach the controller immediately. Large videos,
trajectories, and snapshots can upload in the background. Final campaign
publication still includes every attempt and the sorted accepted-success set in
one immutable release. The instance is disposable only after that release has a
fresh exact-tree readback and disposal receipt.

## Generation Semantics

Collection and training are asynchronous inside a generation but not across an
unbounded stream of changing policies.

```text
checkpoint N is frozen
        |
        v
collect one balanced generation with checkpoint N
        |
        v
freeze and publish generation dataset N
        |
        v
train checkpoints N+1@1000 and N+1@2000
        |
        v
evaluate baseline, 1000, and 2000 on identical matrices
        |
        v
promote only a non-regressing winner
```

The trainer never consumes a directory that is still receiving rollouts. New
episodes belong to the next immutable generation. This prevents an experiment
from silently changing its training distribution during optimization and avoids
unbounded stale/off-policy mixing.

For success-only RFT, generation-synchronous learning is preferred to a fully
asynchronous trainer. Ignoring wall-clock time, asynchronous arrival provides no
inherent score advantage and can reduce reproducibility. The speed benefit comes
from keeping simulation, validation, transfer, and checkpoint publication busy
concurrently—not from training on a moving dataset.

## Training Data Contract

The first corrective fine-tune preserves an exact post-split training mixture:

- 70% organizer/original BC frame windows;
- 30% verified, distinct, seen-garment RFT success windows; and
- zero public-unseen or private-unseen benchmark frames.

The RFT portion remains category-aware and episode-balanced so one easy garment
or one long episode cannot dominate through repeated frames. Validation is
split by immutable raw episode lineage, never by frames from the same episode.

Before any paid training, the repository must resolve the current action-window
contract. The pinned checkpoint evidence distinguishes two values: the GR00T
model's maximum action capacity is 40, while the `new_embodiment` processor
`delta_indices` and live policy wire contain 16 actions. Legacy RFT launch and
materialization metadata incorrectly use the ambiguous name `action_horizon`
for 40. The implementation must preserve model capacity 40 while enforcing
LeHome training targets and executed policy chunks at 16. Both values are
named and bound consistently in:

- rollout receipts;
- selected frame windows;
- prepared-dataset metadata;
- mixture planning;
- launcher configuration (`model_max_action_horizon=40` and
  `embodiment_action_horizon=16`); and
- one real forward/loss smoke.

No conversion shim may silently pad, truncate, reinterpret 16 targets as 40,
or change the pinned model architecture from 40. Training remains blocked until
this gate passes.

## Persistent Trainer

The trainer uses one immutable dataset generation and one immutable parent
checkpoint. It hydrates these once, warms caches once, and runs one optimizer
process to step 2,000.

Before the real run, a bounded benchmark selects settings:

1. measure dataloader worker counts 4, 8, and 12 while holding batch constant;
2. measure physical batches 64, 96, and 128 using the best loader setting;
3. use 100 steady optimizer steps for each admitted batch candidate;
4. stop increasing after a proven OOM or less than 10% physical VRAM headroom;
5. reject nonfinite loss or materially unstable step time; and
6. report the fastest stable candidate, using lower cost and more headroom as
   tie-breakers.

The first production run remains at the verified global batch 64 and uses the
benchmark to tune dataloader workers. Results for batches 96 and 128 are
exploratory: they cannot silently replace batch 64 because the agreed
step-1,000/2,000 milestones would then see more samples and change learning
semantics. A later run may admit a larger batch only with an explicit learning
rate/exposure plan and identical evaluation gates.

The production process saves at steps 1,000 and 2,000 without being relaunched.
A checkpoint observer waits for an upstream completion marker and validates the
complete directory. A separate packaging worker snapshots, archives, hashes,
uploads, and freshly reads back step 1,000 while optimization continues toward
2,000. It never reads a checkpoint directory still being written. Bounded disk
and upload queues pause future saving rather than deleting unverified state.

At step 2,000, the trainer waits for both checkpoint publications and writes the
disposal receipt. It is then destroyed unless an immediately approved next
generation is ready and the retained rental is still healthy and within budget.

## Evaluation and Promotion

Evaluate the original step-12000 baseline, the new step-1000 checkpoint, and the
new step-2000 checkpoint on identical immutable matrices. Evaluation uses no
training mixture frames and no changing seeds between candidates.

Promotion requires:

- at least 70% overall success on the required evaluation matrix;
- at least 60% in every required garment category;
- no major safety failure;
- no major seen-category regression; and
- complete immutable checkpoint and evaluation provenance.

If neither new checkpoint beats the parent under the gate, retain the parent and
do not collect the next generation from a regressed checkpoint. The failed
training round remains useful experiment evidence but does not become the new
policy.

## Rental Lifetime and Failure Handling

"Persistent" means for one active campaign or training block, not forever.

The rollout rental may remain alive across hundreds of attempts when it remains
healthy, productive, and under budget. Rotate it when the provider interrupts
it, worker recovery repeatedly fails, throughput degrades materially, or the
campaign ends. Before rotation or destruction, publish and read back all closed
attempts plus abort evidence.

The trainer rental may remain alive from preflight through the 2,000-step run
and evaluation preparation. An interruptible trainer is allowed because exact
checkpoint resume exists. Rollout rentals remain on-demand because simulator
episodes and local evidence are harder to recover economically.

Every destructive provider action requires an exact instance-bound disposal
receipt and an absence readback. A worker crash, upload failure, or integrity
failure stops new leases but never authorizes destruction by itself.

## Interfaces and Compatibility

The new mode is opt-in. The existing wave launcher remains available as a
rollback path until the persistent mode passes acceptance.

New components have narrow responsibilities:

- `PersistentCampaignController`: creates deterministic attempts and owns stop
  conditions, without running Isaac or uploading files.
- `TaskLedger`: provides durable lease, heartbeat, retry, and terminal state
  transitions.
- `PersistentIsaacWorker`: owns one environment and executes leased attempts.
- `ArtifactFinalizer`: validates and hashes finalized local attempt trees.
- `BackgroundArtifactPublisher`: stages verified bytes and records immutable
  readback evidence.
- `GenerationFreezer`: produces the complete immutable dataset generation.
- `PersistentTrainingController`: benchmarks the host, launches one continuous
  run, observes checkpoints, and gates disposal.

All process boundaries exchange canonical JSON receipts and content hashes.
No component reads provider credentials or Hugging Face tokens unless its
existing explicit authority requires them.

## Acceptance Plan

### Free tests

- task-ledger transition, lease-expiry, retry, and crash-recovery tests;
- deterministic scheduling independent of worker completion order;
- no duplicate attempt after canonical terminal evidence exists;
- persistent reset isolation across different garments and seeds;
- slow-worker test proving faster workers continue leasing;
- bounded artifact backpressure and recovery tests;
- mutation and token-trap tests before publication;
- moving-dataset rejection by the trainer;
- continuous 1,000-to-2,000 process test with background checkpoint packaging;
- batch/loader selection tests with OOM, low-headroom, and unstable candidates;
- exact 70/30 mixture and unseen-contamination rejection; and
- unified action-horizon loader/loss contract tests.

### Paid rollout acceptance

On one approved 4x3090 on-demand host, run an eight-episode mixed-duration smoke:

- four workers initialize exactly once;
- each worker completes at least one episode and at least one worker completes a
  second without waiting for the slowest first episode;
- all eight attempt identities are unique and terminal;
- every artifact validates and receives immutable private-Hub readback;
- policy/session/garment state does not cross episode boundaries;
- no token or provider secret enters synchronized evidence; and
- a report compares wall-clock time, startup time, GPU/CPU utilization, and
  accepted episodes/hour with the legacy two-wave path on the same host.

Promote persistent mode only if correctness is exact and it produces a material
wall-clock improvement. Throughput alone cannot waive an evidence failure.

### Paid training acceptance

On one RTX PRO 6000 96 GB host:

- accept the provider's newer Blackwell driver only after the pinned CUDA 12.8
  image passes the real CUDA/library/optimizer smoke; do not apply the Isaac
  rollout `<590` ceiling to this training-only machine;
- pass the unified action-horizon real loader/loss smoke;
- complete the loader-worker and batch-size benchmark;
- run one continuous 2,000-step process at the verified global batch 64;
- observe and immutably publish step 1,000 while later optimization progresses;
- immutably publish step 2,000;
- report optimizer steps/second, samples/second, GPU utilization, dataloader wait,
  checkpoint pause time, upload overlap, hourly price, and total cost; and
- destroy only after fresh readback and an instance-bound disposal receipt.

## Non-Goals

This design does not:

- enable GPU cloth physics;
- train on unseen benchmark garments;
- implement DAgger, hard-state recovery, value heads, AWR, or online RL;
- let the trainer consume a changing dataset;
- copy the winner's batch size or model-specific loss system;
- require shared batched GR00T inference in the first release;
- keep any rental alive indefinitely; or
- relax immutable publication, cost, safety, or disposal gates for speed.

## Expected Impact

The main gain should come from eliminating repeated Isaac/model startup and the
four-episode barrier. Dynamic leasing removes idle time caused by 2.5-to-7-minute
episode variation, while background finalization hides validation and transfer
behind later simulation.

Training gains should come from selecting the actual throughput-optimal physical
batch, avoiding the current tiny four-GPU profile, keeping one process alive
through both checkpoints, and overlapping step-1000 publication with later
optimization. Exact speedup remains a measurement result, not a design claim.
