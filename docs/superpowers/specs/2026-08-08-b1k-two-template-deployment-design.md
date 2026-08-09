# BEHAVIOR-1K GR00T Two-Template Deployment Design

## Objective

Publish and verify two separate, immutable Vast.ai templates for the 2026
BEHAVIOR-1K 100-task R1Pro track before paying for the full training run:

1. A headless GR00T N1.7 training template that produces recoverable rolling
   checkpoints and publishes a verified final policy to a private Hugging Face
   model repository.
2. A headless Isaac Sim rollout template that consumes an explicitly selected
   verified policy and publishes every closed episode to a private Hugging Face
   dataset repository, split into evaluator-confirmed `success` and `failure`
   partitions.

Both OCI images are published to Docker Hub, referenced by digest, and contain
no credentials. Both templates are smoke-tested on automatically rented cheap
GPU instances. Smoke rentals have a cumulative five-dollar cap and are always
destroyed and independently confirmed gone.

This design is exclusively for BEHAVIOR-1K. It must not read from, write to, or
modify the separate LeHome garment repositories, templates, worktrees, rollout
matrices, or active instances.

## Settled Scope

- Benchmark: 2026 BEHAVIOR-1K, exactly 100 R1Pro tasks.
- Policy: GR00T N1.7 using the already approved RGB-only training contract.
- Training: 15,000 optimizer steps, checkpoint every 1,000 steps, and two
  remotely retained rolling checkpoints.
- Rollout runtime: headless only. There is no desktop, VNC, noVNC, Xfce, or GUI
  dependency in the rollout image or Vast template.
- Rollout output: observations, actions, evaluator outcome, task and instance
  identity, checkpoint identity, logs, videos when produced, provenance, and
  checksums.
- Later RFT consumes only verified successful episodes. Failures remain
  immutable diagnostic evidence.
- Hard-state generation, saved-state recovery collection, expert takeover,
  teleoperation, DAgger, and TTT are excluded. Hard-state/DAgger gets a third
  template later. TTT is applied only after selecting the strongest base policy
  produced by the demonstration and RFT cycles.

## External Resource Boundaries

The B1K resources are new and campaign-specific:

| Purpose | Resource |
| --- | --- |
| Training OCI | `docker.io/ryanjin333/behavior1k-groot-n17:trainer-<source-commit>` |
| Rollout OCI | `docker.io/ryanjin333/behavior1k-groot-n17:rollout-<source-commit>` |
| Final policies | `ryanjin333/behavior1k-groot-n17-models` (private model repo) |
| Rolling checkpoints | `ryanjin333/behavior1k-groot-n17-checkpoints` (private bucket) |
| Rollout episodes | `ryanjin333/behavior1k-groot-n17-rollouts` (private dataset repo) |

The existing `ryanjin333/lehome-groot-n17-*` resources are explicitly forbidden
from B1K configuration defaults and tests. Runtime configuration may override a
repository only when the override is validated as B1K-specific; a LeHome prefix
is rejected fail-closed.

The shared Docker Hub repository is private; role-prefixed immutable tags and
independent OCI labels keep training and rollout identities distinct. Vast receives registry pull credentials
through its account-level secret mechanism; credentials are never included in
the image, template payload, command line, logs, or repository.

## Architecture

### Shared release contracts

Training and rollout use a small shared library for:

- exact run, image digest, source revision, dataset revision, and policy
  identity validation;
- streaming SHA-256 and byte-size manifests;
- secret redaction;
- atomic status and receipt writes;
- immutable Hugging Face upload/readback verification; and
- external smoke-instance cleanup receipts.

The two images do not share simulator or trainer dependencies. Shared code is
limited to release contracts and transports so an Isaac dependency cannot
break training and a trainer dependency cannot enlarge rollout startup.

### Training image and template

The training image contains the pinned Isaac-GR00T checkout, frozen Python
environment, dataset/model download tools, B1K bootstrap, lifecycle controller,
rolling-checkpoint publisher, and final-policy publisher. It does not contain
Isaac Sim or the rollout evaluator.

The template:

- supports one to four compatible GPUs through bounded, explicit per-rank batch plans; it never freely auto-scales the optimizer contract beyond those validated counts;
- requests 2 TB disk, at least 128 GB RAM, 24 effective CPU cores, direct SSH,
  and the approved network/disk floors;
- runs the container as root only long enough to materialize the inherited
  Hugging Face token as a runtime-user-owned `0600` file, then removes the
  token from the process environment and runs the workload as the non-root
  trainer user;
- keeps production `AUTO_DESTROY=0`; and
- starts the complete lifecycle automatically after preflight.

Normal lifecycle execution must perform bootstrap, exact data/model revision
validation, resume selection, training, stable checkpoint detection, rolling
publication, final publication, immutable readback, and receipt writing. It may
fall back to a smaller approved batch plan only for a recognized CUDA OOM before
any optimizer progress or remote checkpoint state exists. Any later failure
stops without silently changing the experiment.

Rolling checkpoints are mutable operational recovery objects with exactly two
verified generations retained. The final policy is an immutable Hugging Face
model-repository release containing the native checkpoint tree plus modality,
normalization, task selection, launch arguments, revisions, image digest, logs,
and a content manifest. A final release is not complete until a fresh immutable
revision readback matches every recorded hash.

### Rollout image and template

The rollout image contains pinned Isaac Sim/BEHAVIOR-1K, the matching evaluator,
the GR00T policy server, checkpoint hydrator, exact 100-task manifest, episode
recorder, classifier, release publisher, and headless health probes.

The template defaults to one simulator worker per GPU and supports explicit
worker scaling only after a capacity probe. It downloads a policy only from a
verified immutable model-repository release. Policy identity, model config,
action horizon, modality, normalization, evaluator revision, task ID, and
instance ID are bound into every episode manifest.

An episode enters `success` or `failure` only after the official evaluator has
closed it with a recognized terminal outcome. Crashes, missing files, partial
uploads, interrupted episodes, and unknown outcomes enter a local quarantine
and are not published as either class. The release fails if any expected
episode is missing, duplicated, open, or has an unrecognized outcome.

The dataset layout is content-addressed and immutable:

```text
releases/<policy-commit>/<run-id>/
  success/<episode-id>/...
  failure/<episode-id>/...
  reports/rollout-report.json
  task-instance-manifest.json
  release-manifest.json
  SHA256SUMS.json
```

The release manifest records exact counts and bytes for both partitions. The
publisher uploads one closed release, resolves its immutable Hugging Face
commit, lists the remote tree, downloads it to a fresh location, and verifies
all hashes before writing a publication receipt.

## Flywheel and Future Correction Data

The initial policy is trained on the official demonstrations. Each RFT cycle
then rolls out that policy, adds verified successes to an aggregated mixture,
and retrains a new checkpoint. Failures are analyzed but never treated as
behavior-cloning action targets.

The later hard-state/DAgger template will collect expert actions from states
induced by the current policy. A final correction-focused training stage must
not train on correction episodes alone. It uses a replay mixture containing
original demonstrations, balanced successful RFT episodes, and the new DAgger
corrections, with a lower learning rate and task-balanced sampling. Policy
actions before takeover are context or diagnostics, not supervised targets.
This preserves clean-start skills while improving recovery coverage and follows
DAgger's dataset-aggregation principle rather than replacing the accumulated
dataset with the last correction batch.

## Smoke Orchestrator and Spending Safety

Paid smoke testing is owned by an external local orchestrator, not by either
production template. The orchestrator persists an append-only rental ledger
before instance creation and tracks the exact instance ID, offer ID, hourly
rate, creation time, purpose, and cleanup state.

It enforces:

- a cumulative maximum authorized charge of five US dollars;
- the cheapest compatible verified offers that satisfy the individual smoke;
- one smoke instance at a time unless concurrency is required to prove a
  multi-GPU contract;
- bounded deadlines for image pull, bootstrap, and runtime probes;
- `try/finally` destruction of the exact recorded instance ID on every exit;
- a second destruction attempt for transient API failure; and
- final proof that the ID is absent from Vast and its former SSH endpoint is
  unreachable.

It never destroys an instance that is absent from its own ledger. In
particular, it cannot target the separate LeHome campaign instance.

### Training smoke

The cheapest compatible CUDA GPU may be used; a 96 GB RTX PRO 6000 is not
required for image-load validation. The smoke proves:

1. digest-pinned private Docker Hub pull works on Vast;
2. entrypoint, non-root transition, GPU visibility, imports, and local
   preflight complete;
3. token-file ownership and redaction are correct;
4. lifecycle dependency wiring reaches a bounded smoke adapter rather than the
   15,000-step training job; and
5. a tiny synthetic checkpoint-shaped artifact is uploaded to a namespaced
   smoke path in the B1K model repo and read back by immutable revision.

The synthetic artifact is explicitly labeled `smoke` and is never eligible for
rollout checkpoint selection.

### Rollout smoke

The cheapest GPU compatible with the pinned Isaac Sim stack is used. The smoke
proves:

1. digest-pinned private Docker Hub pull works;
2. headless Isaac Sim starts with the required EULA setting and correct bundled
   Warp runtime;
3. BEHAVIOR-1K imports, one official task reset, camera/state observations, the
   GR00T server, and at least one action request execute;
4. interrupted/short smoke output is quarantined rather than mislabeled; and
5. synthetic closed success and failure fixtures exercise the two-partition
   publisher and immutable dataset-repository readback without pretending those
   fixtures are real evaluation episodes.

## Error Handling

- Missing credentials, inaccessible gated models, wrong repo ownership,
  mutable image tags, revision drift, insufficient disk/RAM/GPU, or forbidden
  LeHome repository names stop before large downloads.
- Secrets are redacted from commands, status, exceptions, logs, manifests, and
  subprocess environments.
- Watcher or publication failure terminates training and preserves the instance
  for manual recovery; production templates never self-destroy.
- Rollout worker failure quarantines only the affected episode, stops release
  publication, and preserves all closed local episodes.
- Smoke cleanup failure is a stop-the-line condition. The orchestrator keeps
  retrying bounded safe cleanup and reports the exact still-live ID; it does not
  continue to a second rental.

## Verification and Acceptance

Local acceptance requires unit and integration tests for fresh training,
resume, corrupt remote state, watcher failure, OOM boundaries, final immutable
publication, exact task manifests, episode classification, quarantine,
success/failure release readback, secret scanning, template rendering, cost
accounting, and exact-ID cleanup.

OCI acceptance requires both linux/amd64 images to build, pass image-level
verification, publish under role-prefixed immutable tags in the shared private
Docker Hub repository, and resolve to recorded immutable digests. Vast acceptance requires both private templates
to read back exactly with those digests and no secret material.

Runtime acceptance requires successful paid training and rollout smoke receipts
plus verified destruction receipts for every smoke instance. Passing unit
tests, building images, creating templates, or merely launching Isaac Sim is
not sufficient by itself.

The release remains incomplete until a fresh Sol review returns `ship` after
the parent session has inspected the full diff and rerun the acceptance checks.
