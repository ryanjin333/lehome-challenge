# Simple Curriculum and Success-Replay Collection Design

**Status:** Approved design; implementation and paid execution remain separate gates.

## Objective

Collect one clean, interpretable dataset from the approved original step-12K
policy using the 40 seen LeHome garments, then derive a bounded success-replay
dataset from successful episodes in that new collection. The campaign ends
after immutable Hugging Face publication and readback. It does not train or
evaluate a new checkpoint.

## Fixed Scope

The campaign uses:

- the approved original step-12K checkpoint and its pinned artifact identity;
- one preemptible RTX PRO 6000 rollout VM;
- four persistent rollout workers;
- CPU cloth simulation and CUDA policy inference;
- the 40 seen garments: 10 top-long, 10 top-short, 10 pant-long, and
  10 pant-short;
- fresh canonical episode resets for the 1,000 source rollouts;
- public Hugging Face storage for durable artifacts and readback receipts.

The campaign excludes:

- A-500 and every other derived policy;
- prior rollout episodes as training or replay inputs;
- hard-state mining, failure-state restoration, and DAgger;
- geometry or cloth-physics perturbations;
- training, checkpoint selection, or seen/unseen evaluation;
- additional rollout or training VMs;
- automatic continuation into another experiment.

Old artifacts remain available for historical comparison, but no old episode
may enter the new source or replay dataset.

## Data Products

The campaign produces two explicitly separate data products:

1. **Fresh source collection:** exactly 1,000 valid fresh outcomes unless a
   fidelity or cost gate stops the campaign.
2. **Success replay:** at most 400 replay attempts derived only from successful
   episodes in the new source collection, targeting at most 200 accepted
   replay successes.

Replay attempts never count toward the 1,000 fresh outcomes and never update
per-garment curriculum success rates.

## Phase 1A: Uniform Calibration

The calibration matrix contains 400 immutable assignments:

- 10 fresh seeds for each of the 40 seen garments;
- 100 assignments for each garment category;
- unique attempt IDs, trial IDs, and seeds;
- canonical fresh resets with no restored snapshot;
- round-robin ordering so the first 100 valid outcomes contain 25 assignments
  from each category.

Success and failure are both valid outcomes. Infrastructure-invalid attempts do
not count toward the 400 and are retried using the same immutable assignment.
Policy failures are recorded and never converted into infrastructure retries.

Only valid fresh outcomes contribute to per-garment and per-category success
rates.

### First-100 circuit breaker

After 100 valid outcomes, the orchestrator stops before the remaining 300 when
any of these conditions holds:

- any episode reports missing cloth, cloth flight, non-finite cloth state, or a
  safety failure;
- infrastructure-invalid executions divided by all executions exceeds 0.02;
- fewer than 5 of the 100 valid outcomes are official successes;
- the checkpoint, code revision, asset revision, simulator version, image
  identity, or CPU-cloth device provenance is inconsistent across outcomes.

This gate prevents a second full 1,000-attempt campaign when the runtime is not
producing a usable policy distribution.

## Phase 1B: Curriculum Collection

After all 400 calibration outcomes pass verification, a deterministic builder
reads the authenticated calibration report and emits exactly 600 fresh
assignments.

The sampler matches the released winner-style two-level curriculum:

1. For each category `t`, compute
   `type_weight(t) = max(1 - success_rate(t), 0.05)`.
2. Select a category in proportion to its type weight.
3. Within that category, for each garment `g`, compute
   `garment_weight(g) = max(exp(-((success_rate(g) - 0.5)^2) / (2 * 0.233^2)), 0.02)`.
4. Select a garment in proportion to its within-category weight.
5. Assign a fresh seed that does not occur in the calibration matrix or an
   earlier curriculum assignment.

Sampling uses one recorded RNG seed, samples garments with replacement, and
freezes all 600 assignments before paid execution. There is no live schedule
mutation while workers are running.

The curriculum matrix uses the same canonical reset and runtime identity as the
calibration matrix. It adds no visual, geometry, cloth, or robot perturbation.

## Phase 2: Success Replay

Replay source eligibility is limited to fresh Phase 1A or Phase 1B episodes
that satisfy every condition below:

- official terminal success and `accepted_success == true`;
- seen-garment identity bound to the approved step-12K policy;
- complete checksum manifest and Hugging Face readback receipt;
- authenticated CPU-cloth continuation snapshot at step 16;
- no safety, numerical, or cloth-presence failure.

For each category, source garments are weighted by
`max(1 - fresh_success_rate(garment), 0.01)`. States within a selected garment
are sampled uniformly. Sampling with replacement is allowed when a category
has fewer source states than requested.

The replay matrix contains at most 400 attempts, with at most 100 attempts per
category. Replay execution:

- restores the authenticated step-16 snapshot;
- serves the same approved step-12K policy;
- varies only cameras, lighting, garment color, and garment texture;
- preserves cloth geometry, scale, material dynamics, friction, robot base,
  joint limits, and every other physics-affecting property;
- retains only official successful outcomes;
- stops accepting a category after 50 replay successes;
- stops globally after 200 accepted replay successes or after all 400 attempts
  are terminal.

If a category has no eligible fresh success state, its replay shortage is
reported explicitly. The builder must not substitute an old episode or another
category.

## Execution Topology

One host orchestrator owns the two immutable fresh matrices, the replay matrix,
and the stage transitions. Four workers pull assignments from the existing
persistent task ledger. The orchestrator does not start Phase 1B before the
Phase 1A report is authenticated and does not start replay before all fresh
source outcomes and their Hub receipts are verified.

No image rebuild is part of this design. New matrix-building and orchestration
code runs from the mounted repository while the existing rollout appliance
continues to own policy serving, simulation, task leases, artifact capture,
and Hugging Face synchronization.

Preemption resumes the same matrix and task ledger. It must never generate a
replacement seed, duplicate a terminal assignment, or create another VM.

## Publication Contract

The public Hugging Face dataset receives immutable, readback-verified copies of:

- the 400-row calibration matrix and SHA-256 receipt;
- the authenticated calibration report used by the curriculum builder;
- the 600-row curriculum matrix, builder parameters, RNG seed, and SHA-256
  receipt;
- every valid fresh success and failure artifact;
- the replay matrix and SHA-256 receipt;
- every accepted replay artifact and each terminal replay receipt;
- per-garment and per-category fresh success statistics;
- replay attempt and accepted-success counts by category;
- policy, code, image, simulator, asset, and device identities;
- final stage seals and Hugging Face download/readback receipts.

Credentials, organizer BC data, cached model weights, and temporary files are
never published.

## Terminal Outcomes

The campaign has only three terminal outcomes:

1. **Fidelity stop:** a cloth, numerical, identity, safety, or infrastructure
   gate fails. Publish the failure evidence and stop the GPU.
2. **Insufficient source stop:** the first-100 gate produces fewer than five
   successes. Publish the 100-outcome report and stop the GPU.
3. **Collection complete:** publish and read back the fresh and replay products,
   then stop the GPU.

Completion does not authorize training. Replay shortages do not make the
collection incomplete when the 400-attempt replay cap is exhausted; they are
reported as results.

## Implementation Boundaries

Implementation should add only:

- a deterministic curriculum matrix builder that consumes the existing
  authenticated evaluation-report schema;
- a success-replay matrix builder or a minimal extension of the existing
  builder to accept the new campaign report and exact category caps;
- a thin host orchestrator that runs Phase 1A, enforces the first-100 gate,
  freezes Phase 1B, runs replay, publishes seals, and stops the VM;
- focused unit and integration tests for matrix identity, weighting,
  deterministic sampling, seed uniqueness, gate behavior, replay eligibility,
  category caps, resume behavior, and publication readback.

Implementation must reuse the existing original-12K campaign appliance,
persistent worker, task ledger, artifact sync, evaluation summarizer, and
Hugging Face transport. It must not introduce a new controller, database,
service, image family, or cloud topology.

## Acceptance Criteria

The implementation is ready for a separately authorized paid run only when:

- the calibration builder proves 400 rows, 10 per garment, 100 per category,
  and 25 per category in the first 100 rows;
- curriculum sampling is deterministic and matches the exact formulas above;
- all 1,000 fresh seeds and attempt IDs are unique;
- replay cannot read prior-campaign episodes;
- replay cannot mutate physics-affecting parameters;
- replay enforces 100 attempts and 50 accepted successes per category, with
  global caps of 400 attempts and 200 accepted successes;
- CPU cloth and approved step-12K provenance fail closed;
- preemption resumes without duplicate terminal assignments;
- public Hub publication is immutable and download/readback verified;
- targeted tests, shell syntax checks, Python compilation, and repository diff
  checks pass;
- all GPU VMs remain stopped after implementation verification.
