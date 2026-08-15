# Nebius Preemptible Training and Rollout Design

**Status:** Approved design

**Date:** 2026-08-15

**Target:** LeHome GR00T corrective-RFT flywheel on Nebius

## Objective

Create reusable, versioned infrastructure templates for two separate Nebius
RTX PRO 6000 roles:

1. a portable single-GPU VLA training appliance; and
2. a LeHome-specific four-worker rollout appliance.

Both runtime machines are preemptible. They reuse one deletion-protected 500
GiB network SSD, attached to only one runtime machine at a time. Packer golden
images remove repeated environment setup, while immutable Hugging Face artifacts
and the shared disk make interruption recovery explicit.

The first training campaign starts from the pinned step-12K policy, trains for
2,000 gradient steps at global batch 64, saves local recovery checkpoints every
500 steps, and publishes immutable Hugging Face recovery checkpoints every
1,000 steps. The rollout appliance loads the policy once and serves four
persistent Isaac workers through one session-aware batching gateway.

The templates must be safe to validate locally without creating paid Nebius
resources. Packer builds, Terraform applies, and paid GPU smoke tests remain
explicit later operator actions.

## Relationship to Existing Flywheel Work

This design adapts the existing persistent rollout and training flywheel to one
Nebius RTX PRO 6000 with 24 vCPUs and 218 GB RAM. It supersedes the earlier
Nebius/Vast runtime assumption of one isolated GPU and one model copy per Isaac
worker for this deployment. It preserves the existing immutable attempt,
dataset, checkpoint, lineage, and evaluation contracts.

The shared policy server is required here rather than an optional later stage:
four workers share one 96 GB GPU, so the model is loaded once and inference
requests are batched and routed by session.

## Decisions

- Cloud: Nebius.
- GPU platform: `gpu-rtx6000`.
- Runtime preset: one RTX PRO 6000 96 GB, 24 vCPUs, 218 GB RAM.
- Runtime lifecycle: preemptible training and rollout VMs.
- Recovery policy: fail/stop on preemption; never silently replace a running
  campaign with an unbound fresh machine.
- Persistent storage: one standalone 500 GiB network SSD with deletion
  protection and Terraform `prevent_destroy`.
- Concurrency: the shared disk is attached to exactly one runtime VM at a time.
- Training and rollout never run concurrently on that disk.
- Rollout worker count: four by default. Three-worker staging is not the normal
  production topology.
- Cloth simulation: CPU only. The GPU is used for rendering/interoperability and
  policy inference, not CUDA cloth simulation.
- Training global batch: 64.
- Default training GPU: RTX PRO 6000. H200 is an opt-in, measured training-only
  profile and is never selected automatically.
- Training duration: exactly 2,000 optimizer steps.
- Local checkpoint cadence: every 500 steps.
- Hugging Face checkpoint cadence: every 1,000 steps.
- Durable source of truth: private, immutable Hugging Face artifacts plus
  verified local recovery state on the shared disk.

The initial implementation preserves these already-reviewed runtime pins:

```text
Isaac-GR00T revision:
  23ace64f17aa5015259b8609d371eb61a357c776

Training OCI image:
  ghcr.io/ryanjin333/lehome-groot-n17-trainer
  @sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746

Parent policy repository:
  ryanjin333/lehome-groot-n17-models
Parent policy revision:
  30ac1a84da67b099e115ad147bcd61e9d60046d3
Parent policy subpath:
  policies/step-12000
Parent policy archive SHA-256:
  0ddd4e7ce351dd2172cd1edd967293a50d02c15c0f2c21ca39db94692a57e0b5
Parent policy artifact SHA-256:
  3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06
```

The experiment manifest repeats the policy identity and hashes. The boot and
training admission paths compare the manifest against the downloaded tree; a
matching Hugging Face revision without matching content hashes is rejected.

## System Architecture

```text
                           image build time
                  +-----------------------------+
                  | temporary CPU Packer builder|
                  +--------------+--------------+
                                 |
                   +-------------+-------------+
                   |                           |
                   v                           v
        vla-training-base image       lehome-rollout image
                   |                           |
                   +-------------+-------------+
                                 |
                         Terraform templates
                                 |
              +------------------+------------------+
              |                                     |
              v                                     v
     preemptible training VM                preemptible rollout VM
     batch-64 GR00T trainer                 one shared policy server
                                            four Isaac workers
              |                                     |
              +------------------+------------------+
                                 |
                   deletion-protected 500 GiB SSD
                                 |
                                 v
                      private immutable HF repos
```

Packer builder VMs are temporary CPU machines used only when creating or
updating a golden image. They do not run training or production rollouts. Once
Packer saves a validated image, it deletes the builder VM.

The two runtime VMs are separate Terraform roles. An operator stops one role,
detaches the shared disk, and attaches it to the other role. Terraform and boot
guards prevent both roles from claiming the same campaign workspace.

## Image Families

### Portable training image: `vla-training-base`

The training image is not based on the LeHome challenge tarball. It contains
portable host and application prerequisites:

- the Nebius-compatible NVIDIA host image and driver contract;
- Docker and pinned container runtime tooling;
- the pinned GR00T/LeRobot training container;
- Hugging Face download, upload, hash, and readback tools;
- video-decode and dataset-loader dependencies;
- persistent-disk discovery and mounting;
- metrics and structured logging;
- preemption handling and resume admission; and
- the generic experiment-manifest launcher.

It contains no policy weights, training datasets, competition garments,
Hugging Face token, Nebius credentials, or experiment-specific mixture.

The same image can be reused by another GR00T/LeRobot competition by supplying a
different immutable experiment manifest. A different model family may require a
new application container, but the Packer host image, Terraform module, disk
contract, and recovery services remain reusable.

The training image family may produce platform-specific host variants from the
same Packer source when Nebius driver or base-image requirements differ. The
pinned training container, manifest contract, disk layout, and recovery services
remain identical across an RTX PRO 6000 variant and a later admitted H200
variant.

### Training GPU economics

Using the operator-observed preemptible prices of $1.08/hour for RTX PRO 6000
and $2.58/hour for H200, H200 must complete the same admitted workload at least
`2.58 / 1.08 = 2.39` times faster to reduce compute cost. That comparison uses
end-to-end admitted training time, including loader stalls and checkpoint work,
not peak tensor throughput.

The initial batch-64 run stays on RTX PRO 6000 because 96 GB VRAM is expected to
fit the agreed workload and its Nebius preset provides 24 vCPUs for video decode
and data loading. The H200 preset provides more VRAM and memory bandwidth but
only 16 vCPUs, so a CPU-fed workload may not realize the GPU's peak advantage.

The Terraform training module keeps the GPU platform explicit, but H200 remains
blocked until a separately authorized identical-workload benchmark proves one
of these operator decisions:

- cost promotion: at least 2.39x end-to-end throughput at the current prices; or
- deadline promotion: the operator explicitly accepts a higher total cost for a
  materially shorter wall-clock run.

The benchmark holds parent checkpoint, experiment manifest, batch 64, precision,
optimizer, sample ordering, checkpoint behavior, and measured steady-step range
constant. H200 is not a rollout candidate because the LeHome appliance depends
on the RTX rendering path and the four-worker design benefits from the
RTX PRO 6000 preset's larger CPU allocation.

### LeHome rollout image: `lehome-rollout`

The rollout image is intentionally competition-specific. Its Packer builder:

1. downloads the official `lehome-challenge.tar.gz` from the pinned Hugging Face
   dataset revision;
2. verifies exact byte length and LFS SHA-256 before use;
3. loads the official challenge Docker image;
4. builds a derived rollout image using the repository's companion dummy Docker
   policy interface as the protocol starting point;
5. installs the session-aware policy gateway, persistent worker launcher,
   controller, append-only ledgers, artifact finalizer, writer/video encoder,
   Hugging Face sync daemon, and boot supervisor; and
6. saves the loaded Docker layers and runtime as the rollout golden image.

The pinned official artifact is:

```text
repository: lehome/docker
revision: a914115729bb0bfd260971b9c8d4147bff38c1fb
file: lehome-challenge.tar.gz
size: 26676771349 bytes
LFS SHA-256: 1a85e389962909debc4ee9988d8a8c388f905fba60686ef78b1623e6872f7123
```

The tarball is the official challenge environment image, not the training
environment and not the policy checkpoint store. The derived image retains the
organizer environment while replacing the single-session dummy implementation
with the coordinated four-worker runtime.

The Packer build needs enough temporary boot storage for the compressed tarball,
loaded Docker layers, derived layer, and build overhead. The implementation
must calculate and document the final image footprint rather than assuming the
26.7 GB compressed size is sufficient.

## Build and Launch Boundaries

Image creation and GPU runtime validation are different gates:

- A CPU Packer builder can verify downloads, hashes, package installation,
  service definitions, and image creation.
- It cannot prove CUDA training, GR00T inference, Isaac rendering, camera output,
  CPU cloth behavior, or four-worker stability.
- Each golden image therefore requires a later bounded RTX PRO 6000 smoke test
  before production use.

Repository implementation creates and validates templates only. It does not run
`packer build`, create the shared disk, or run `terraform apply` until the user
explicitly authorizes paid provisioning.

## Terraform Layout

Terraform uses one common Nebius module with two role configurations:

```text
infrastructure/nebius/
  modules/runtime-vm/
  environments/training/
  environments/rollout/
  storage/
```

The common module owns:

- project, region, subnet, and security-group references;
- the `gpu-rtx6000` platform and one-GPU preset;
- preemptible scheduling and stop-on-preemption behavior;
- disposable boot disk from a pinned custom-image ID;
- optional attachment of the standalone shared disk;
- runtime service account and least-privilege metadata;
- cloud-init/bootstrap inputs; and
- outputs needed for SSH, lifecycle receipts, and disk handoff.

The storage configuration owns one 500 GiB network SSD. Deletion protection and
`prevent_destroy` are mandatory. Destroying a runtime VM must not destroy the
shared disk.

Role-specific inputs include the golden-image ID, role name, experiment or
campaign manifest URI and digest, and runtime startup command. ML hyperparameters
are not passed as a loose collection of Terraform variables; Terraform passes
one immutable manifest identity to prevent unrecorded configuration drift.

## Shared Disk Contract

The disk mounts at `/mnt/lehome` and contains:

```text
/mnt/lehome/
  workspace-manifest.json
  cache/
    huggingface/
    containers/
    video/
  datasets/
    bundles/
    bc/full/
    rollouts/round-1/
    manifests/
  checkpoints/
    local/
    published-receipts/
  rollouts/
    attempts/
    accepted/
    upload-queue/
  ledgers/
  logs/
  receipts/
```

`workspace-manifest.json` records disk identity, schema version, active role,
campaign or run identity, last clean handoff, and hashes of durable manifests.
Boot fails closed when the wrong disk, wrong role transition, incompatible
schema, or conflicting active lease is detected.

Closed terminal attempts and complete checkpoints are never rewritten. Scratch
decode caches, an interrupted checkpoint temporary directory, and an incomplete
episode directory may be cleaned only through explicit, receipt-producing
recovery rules.

## Secrets

Secrets are never baked into Packer images, Terraform state values, Docker
layers, experiment manifests, logs, checkpoints, or uploaded evidence.

At runtime, scoped credentials are injected through the approved Nebius secret
path or an operator-provided ephemeral environment file on the VM. The bootstrap
validates that required variables exist without printing their values. The HF
sync service receives only the private repositories and permissions needed for
its role.

## Immutable Training Experiment Manifest

The reusable training image accepts a pinned manifest such as
`lehome-rft-70-30-v1.yaml`. The manifest defines:

- exact parent policy repository, revision, subpath, and verified hash;
- exact BC and accepted-rollout bundle URIs and hashes;
- immutable mixture-manifest URI and hash;
- BC/rollout sampling weights;
- embodiment action horizon and model action capacity;
- global batch, optimizer steps, and loader settings;
- training/validation lineage assignments;
- evaluation-garment exclusions;
- local and Hugging Face checkpoint schedules; and
- output repository and immutable naming convention.

The initial experiment uses:

```yaml
mixture:
  bc_weight: 0.70
  accepted_rollout_weight: 0.30
training:
  embodiment_action_horizon: 16
  global_batch_size: 64
  gradient_steps: 2000
  local_checkpoint_interval: 500
  hf_checkpoint_interval: 1000
```

Changing the mixture to 80/20 requires a new immutable manifest, not a Packer
rebuild:

```text
lehome-rft-70-30-v1.yaml -> one run identity
lehome-rft-80-20-v1.yaml -> a different run identity
```

The manifest digest is part of the run ID and every checkpoint receipt. Resume
rejects any manifest-digest mismatch. An intentional 80/20 experiment starts a
new run, optionally from a newly pinned parent checkpoint; it never mutates a
70/30 run in place.

## Training Data Construction

The trainer performs the following admission sequence before optimization:

1. download the exact step-12K parent policy at its pinned revision;
2. verify the policy artifact hash;
3. download the immutable BC, accepted rollout, and mixture-manifest bundles;
4. verify every bundle and manifest hash;
5. unpack only to the manifest-bound locations `bc/full/` and
   `rollouts/round-1/`;
6. verify episode schema and raw-lineage identities;
7. reject every held-out evaluation garment before window generation;
8. split train and validation by raw episode lineage;
9. create horizon-16 windows in memory; and
10. sample the manifest-defined mixture without materializing a mutable mixed
    dataset.

The initial held-out set contains all four garment categories:

```text
Top_Long_Unseen_1
Top_Short_Unseen_1
Pant_Long_Unseen_1
Pant_Short_Unseen_1
```

No raw episode may contribute windows to both training and validation. Mixture
weights are enforced over sampled training windows while retaining episode and
category provenance in every batch receipt.

## Training Admission and Execution

Before the 2,000-step run, one short GPU sanity job verifies:

- CUDA, expected GPU identity, and available VRAM;
- the pinned model loads and matches the expected policy digest;
- optimizer construction and one real forward/backward/update step;
- video decoding from both dataset bundles;
- horizon-16 batch creation at global batch 64;
- finite loss and gradients; and
- DataLoader worker candidates `0`, `4`, `8`, `12`, and `16`.

The loader benchmark chooses the fastest stable count with bounded host RAM,
open-file usage, decode failures, and batch-wait variance. It records the result
in the admitted run receipt; it does not change mixture or sample ordering.

The production process remains alive through all 2,000 steps. It writes complete
local recovery checkpoints at steps 500, 1,000, 1,500, and 2,000. At steps
1,000 and 2,000 it also creates an immutable Hub release containing model,
optimizer, scheduler, scaler if used, RNG state, sampler position, run manifest,
source revision, and artifact hashes. Publication is not complete until a fresh
readback verifies the uploaded tree.

Local step-500 and step-1,500 checkpoints are recovery points, not promoted
candidate releases. An operator may publish them manually after an incident,
but normal candidate evaluation uses immutable step-1,000 and step-2,000
releases.

## Rollout Runtime Topology

The rollout VM boots this coordinated appliance:

```text
RTX PRO 6000 96 GB / 24 vCPU / 218 GB RAM
├── one session-aware policy server
│   └── one loaded checkpoint, bounded batched inference
├── Isaac worker 0
├── Isaac worker 1
├── Isaac worker 2
├── Isaac worker 3
└── CPU/disk services
    ├── controller and append-only attempt queue
    ├── heartbeat and retry-lease manager
    ├── success/failure/hard-state ledger
    ├── background writer and video encoder
    └── background Hugging Face sync daemon
```

Four workers is the default production configuration. A lower count may be used
for diagnosis, but the acceptance target and Terraform rollout profile remain
four.

The initial CPU-affinity budget is approximately five cores per Isaac worker
and four cores shared by the policy gateway, controller, writer, uploader, and
operating system. Affinity is a starting configuration, not an unmeasured
throughput claim. The paid smoke must report CPU saturation, renderer and policy
GPU pressure, memory, video backlog, simulator failures, inference latency, and
accepted episodes per hour.

## CPU Cloth Contract

The organizer runtime and this deployment use the admitted CPU simulation path.
No code path may enable GPU cloth merely because CUDA is available.

The runtime admission receipt records:

- simulation device and cloth device as CPU;
- renderer device and GPU identity;
- policy inference device;
- relevant Isaac/LeHome runtime versions; and
- a visible cloth-contact canary result.

A configuration that moves cloth simulation to CUDA fails before campaign
leases begin.

## Session-Aware Policy Gateway

The stock synchronous, session-unaware policy server is insufficient for four
persistent workers. The new gateway owns the model once and exposes a protocol
with:

- `session_id`;
- episode generation number;
- monotonically unique request ID;
- policy digest;
- observation/action schema version;
- camera/state/language payload;
- request deadline; and
- returned action-chunk metadata.

The server gathers ready requests for a short bounded batching window, collates
up to four sessions into one model batch, runs inference once, then splits and
routes action chunks by the ordered request identities. It rejects duplicate,
expired, cancelled, wrong-generation, or stale-session requests.

Each worker owns its local horizon-16 action queue. It requests another chunk
only when required, so the server is not called for every simulator step. Reset
increments the episode generation, clears the worker queue, invalidates old
requests, and prevents a late response from reaching a new episode.

Every inference receipt records policy digest, ordered session/request IDs,
batch occupancy, batching wait, model latency, returned chunk hashes, and the
applicable deterministic seed identity. The protocol is tested for out-of-order
completion, cancellation, timeout, worker restart, and stale-response rejection.

## Persistent Rollout Workers and Controller

Each worker owns one persistent simulator instance, one assigned episode at a
time, one session ID, one random-seed stream, camera/action buffers, and an
attempt-scoped output directory. A worker does not own a separate model server.

The lifecycle is:

```text
controller leases immutable attempt
-> free worker resets and runs it
-> worker atomically closes raw terminal episode
-> background pipeline validates, hashes, and uploads it
-> worker immediately leases another attempt
```

There are no wave barriers. A worker that finishes early immediately obtains
the next eligible lease.

The append-only attempt state machine is:

```text
pending -> leased -> terminal_pending_validation
        -> accepted | rejected | infrastructure_abort
```

Heartbeats renew leases. A preempted or crashed mid-episode lease is retried from
its immutable attempt identity with an explicit retry record. Opaque Isaac
process memory is never treated as resumable state. An already canonicalized
terminal attempt is never executed twice.

The campaign admits at most 400 attempts and targets 150 validated successful
seen/randomized/hard-state episodes. The accepted set, rejected set,
infrastructure aborts, policy provenance, and attempt ledger are published as
one immutable rollout round before training consumes it.

## Asynchronous Artifact Pipeline

Workers write active attempts under pending directories. At terminal state the
directory is atomically closed and placed on a bounded finalization queue. The
worker can then lease another episode while CPU/disk services:

1. validate schema and terminal completeness;
2. classify success, failure, and hard state;
3. hash raw artifacts;
4. encode diagnostic video when configured;
5. append ledger receipts;
6. upload content-addressed artifacts to the private Hub; and
7. perform immutable readback verification.

Backpressure stops new leases when finalized bytes, writer work, or upload work
exceeds configured bounds. It never drops evidence to preserve throughput.

## Preemption and Recovery

Nebius preemption can deliver a short termination window. The shutdown handler
is designed around the documented 60-second SIGTERM period:

1. stop issuing new training work or rollout leases;
2. mark active rollout leases interrupted and flush heartbeats/ledgers;
3. atomically close only already-terminal artifacts;
4. request a training checkpoint only when the trainer can safely create a
   complete bounded recovery point;
5. persist a preemption receipt to the shared disk;
6. start a time-bounded sync of small critical receipts; and
7. exit without deleting local durable state.

The shared disk survives VM loss. A replacement VM from the same golden image
mounts it, validates `workspace-manifest.json`, reads immutable local/HF state,
and resumes the exact run or campaign identity.

Training resumes from the newest complete local checkpoint whose parent policy,
dataset bundles, experiment manifest, code revision, and sampler identity all
match. Rollouts retry incomplete attempts; they do not resume simulator RAM.

## Hugging Face Durability Contract

Private Hugging Face repositories contain every artifact required to reproduce
or continue accepted work:

- pinned parent policy;
- immutable BC and rollout bundles;
- immutable mixture and experiment manifests;
- accepted terminal rollout episodes and complete ledgers;
- step-1,000 and step-2,000 training releases;
- optimizer/recovery provenance for those releases;
- evaluation matrices, results, and promotion receipts; and
- immutable next-round rollout bundles.

The local shared disk additionally keeps frequent recovery checkpoints, active
attempts, caches, and bounded upload queues. Incomplete scratch state is not
claimed as Hub-durable evidence.

Publication requires upload plus fresh readback and hash verification. A mutable
branch name or `latest` pointer is never sufficient provenance.

## Evaluation and Promotion

The existing top-only 40-episode diagnostic is not the final challenge gate.
LeHome contains four garment categories:

```text
top_long
top_short
pant_long
pant_short
```

The canonical public matrix contains 280 trials: 50 seen and 20 public-unseen
trials per category. Evaluation uses all categories and fixed candidate-neutral
seeds.

The five candidates are:

1. original step-12K baseline;
2. previous step-1K;
3. previous step-2K;
4. new step-1K; and
5. new step-2K.

Every candidate runs the same untouched 80 public-unseen episodes: 20 per
category. Promotion requires:

- at least 56/80 overall success (70%);
- at least 12/20 in each of the four categories (60%);
- no major safety failure;
- no integrity or provenance failure; and
- no material seen-category regression.

All five candidates also run a fixed, predeclared seen-development screen. The
proposed winner and original baseline then run the complete 200-trial seen
matrix before promotion. The full comparison must cover all four categories,
not only tops. The seen screen, regression tolerance, tie-breakers, and safety
taxonomy are fixed in the evaluation manifest before candidate results are
opened.

The winner is the passing candidate with the highest overall unseen success.
Predeclared tie-breakers use category floor margin, seen-regression result,
safety, then the earlier checkpoint. No post-hoc seed or garment substitution
is allowed.

If no candidate passes but one improves without a disqualifying regression, the
best admissible candidate becomes the policy for the next rollout campaign. The
campaign collects up to 400 attempts, retains 150 verified successful episodes,
publishes the next immutable rollout bundle, and starts a new experiment from
the data-ingestion step. Physical testing begins only after the complete gate
passes.

## Observability

Training records:

- optimizer and sample throughput;
- batch and loader wait times;
- GPU utilization and VRAM;
- host RAM and decoder failures;
- loss and gradient finiteness;
- checkpoint pause and upload overlap; and
- preemption and exact-resume receipts.

Rollout records:

- leases and worker heartbeats;
- worker resets/restarts and simulator failures;
- CPU utilization per worker;
- GPU utilization and VRAM;
- batching occupancy, wait, and policy latency;
- renderer and camera failures;
- writer/video/upload queue depths;
- attempts and accepted episodes per hour; and
- preemption/retry outcomes.

Metrics contain identities and hashes but no credential values or raw secrets.

## Acceptance Plan

### Free repository checks

- `packer fmt` and `packer validate` for both image templates;
- `terraform fmt -check` and `terraform validate` for storage, training, and
  rollout configurations;
- static checks proving paid resource commands are not run by validation tests;
- manifest-schema and digest tests, including 70/30 versus 80/20 run identity;
- exact tarball revision, byte-length, and hash validation tests;
- shared-disk single-owner, wrong-role, and recovery admission tests;
- training lineage, four-category exclusion, mixture, and horizon tests;
- checkpoint cadence, atomic completion, resume, and HF readback tests;
- session batching, routing, cancellation, timeout, stale-response, and restart
  tests;
- dynamic leasing, no-wave-barrier, retry, and duplicate-terminal tests;
- CPU-cloth configuration rejection tests;
- bounded writer/upload backpressure tests; and
- four-category evaluation matrix and threshold tests.

### Later paid training smoke

On one preemptible RTX PRO 6000 VM from `vla-training-base`:

- mount and validate the shared disk;
- verify CUDA and the exact pinned GPU/runtime tuple;
- verify model and dataset downloads/readback;
- run video decode, batch-64 construction, and one optimizer step;
- benchmark loader workers `0`, `4`, `8`, `12`, and `16`;
- prove a bounded interruption and exact checkpoint resume; and
- record throughput, utilization, storage, and cost evidence.

### Later paid rollout smoke

On one preemptible RTX PRO 6000 VM from `lehome-rollout`:

- prove the official challenge image is already present;
- prove cloth simulation remains on CPU;
- load one checkpoint exactly once;
- start four persistent Isaac workers;
- route simultaneous requests to the correct sessions;
- show a fast worker leases again without waiting for a slow worker;
- close, validate, upload, and read back terminal attempts;
- measure CPU/GPU/VRAM, policy latency, batching occupancy, simulator stability,
  and writer/upload backlog; and
- prove a bounded preemption/restart retries only incomplete attempts.

Four workers remain the default unless the paid acceptance evidence shows the
24-vCPU machine cannot run them correctly. The system fails closed and reports
the capacity problem; it does not silently change the approved production
topology.

## Non-Goals

This design does not:

- create paid Nebius resources during repository implementation;
- run training and rollout simultaneously on one VM;
- attach the shared disk to both runtime VMs;
- base the training image on the LeHome simulator tarball;
- bake experiment data, model weights, or credentials into images;
- enable CUDA cloth simulation;
- train on any held-out evaluation garment;
- treat local scratch files as immutable Hub publications;
- silently change 70/30 to another mixture during resume;
- make the LeHome rollout image portable to unrelated simulator competitions;
  or
- begin physical testing before the complete evaluation gate passes.

## Expected Outcome

After implementation, the repository will contain two reproducible Packer image
definitions, reusable Terraform modules and role templates, immutable training
and rollout manifest contracts, a session-aware four-worker rollout runtime,
and tests for recovery, lineage, evaluation, and infrastructure safety.

Preemptions should require replacement compute, not rebuilding environments or
reconstructing accepted state. A later 80/20 experiment should require only a
new immutable manifest. A later GR00T-based competition should reuse the
training image and infrastructure while supplying its own profile, whereas the
LeHome rollout image remains deliberately tied to the official challenge
environment.
