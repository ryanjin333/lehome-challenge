# GR00T N1.7 Trainable Rollouts and DAgger Flywheel Design

## Objective

Extend the existing LeHome GR00T N1.7 baseline into a restart-safe data
flywheel that can:

- measure the step-12000 policy on a statistically useful public matrix;
- preserve autonomous rollouts as complete, provenance-bearing episodes;
- extract only valid expert labels into behavior-cloning datasets;
- collect full expert demonstrations and DAgger recoveries with two physical
  SO101 leader arms connected to the operator's Mac;
- preserve restorable simulator states for hard mining;
- collect canonical and domain-randomized experience without contaminating the
  fixed evaluation matrix;
- exchange immutable policy and dataset revisions asynchronously through
  Hugging Face Hub; and
- scale rollout concurrency only after measured acceptance gates.

This design targets the existing pinned GR00T N1.7 behavior-cloning stack. It
does not claim to reproduce the winning solution's AWR, RECAP, value heads,
best-of-N inference, or full reinforcement-learning algorithm.

## Evidence and Current Gaps

The checked-in baseline already provides:

- a pinned GR00T N1.7 training launcher;
- deterministic conversion of the organizer's 1,000 LeRobot v3 episodes into
  the canonical three-camera, 12D bimanual GR00T layout;
- a GR00T rollout adapter with a 16-action execution horizon;
- a deterministic matrix runner with checksum-bearing reports and videos;
- a successful 24-trial run using six Isaac Sim workers and two GR00T policy
  servers on an eight-GPU host; and
- organizer teleoperation and successful-policy dataset recorders.

The completed 24-trial matrix produced 12/24 successes: 10/16 seen, 2/8
public-unseen, and 0/6 long-pants. Six long-pants attempts are insufficient to
declare the category incapable.

The existing recorders are not a complete flywheel data contract:

- the evaluator saves only successful policy episodes and does not preserve
  action-source labels or failed trainable evidence;
- the teleoperation recorder does not combine policy prefixes with expert
  intervention;
- no collector records per-frame policy revision, intervention boundaries,
  randomization values, latency, or data quality;
- the current simulator has table-texture and light randomization hooks, but
  both are disabled and there is no garment appearance randomizer;
- the current training launcher does not explicitly select the stronger
  image-augmentation recipe; and
- the existing simulator metadata is not a full restorable cloth-particle
  snapshot suitable for hard-state replay.

## Design Principles

1. **Evaluation and training evidence remain separate.** Canonical evaluation
   episodes are never silently promoted into behavior-cloning targets.
2. **Expert labels dominate the first flywheel fine-tune.** Autonomous actions
   are recorded, but they are not selected as initial BC targets.
3. **Every artifact is immutable and attributable.** Episodes identify the
   policy, code, assets, dataset, simulator, randomization, and operator inputs
   that produced them.
4. **No raw failure action enters BC by accident.** Failed or unsafe attempts
   remain diagnostics and hard-mining evidence only.
5. **The fixed public matrix remains reproducible.** Randomized rollout
   strategies are additional training collections, not replacements for the
   score-estimation matrix.
6. **Hardware concurrency is measured.** A GPU count or VRAM total alone never
   authorizes a larger worker count.
7. **Paid hardware starts only after local contracts pass.** Unit tests, schema
   exports, manifests, and dry-run commands are verified before a rental is
   used for acceptance.

## System Architecture

```text
Physical SO101 leaders on Mac
        |
        | local serial reader, calibration hash, sequence/timestamp
        v
Authenticated local bridge
        |
        | SSH tunnel; no public teleoperation port
        v
Remote DAgger collector --------> Isaac Sim environment
        |                              |
        |                              +--> canonical or randomized cameras
        |                              +--> robot/cloth state and success
        |
        +--> GR00T policy server
        +--> raw episode store
        +--> diagnostic videos/snapshots
        +--> BC exporter
                         |
                         v
                immutable Hub dataset revision
                         |
                         v
                independent training worker
                         |
                         v
                immutable policy revision
```

The physical leaders remain connected to the Mac. Isaac Sim does not expect
remote `/dev/ttyACM*` devices. The Mac bridge reads both Feetech buses and sends
timestamped, sequenced joint samples through an SSH tunnel to a loopback-bound
remote endpoint. The remote collector performs the existing SO101-to-follower
joint-limit conversion and records the resulting absolute 12D radian command.

The bridge handshake includes protocol version, left/right serial identity,
calibration-file hashes, motor limits, expected sampling rate, and a short-lived
session nonce. The collector rejects duplicate arm identities, incompatible
calibration, reordered samples, stale samples, excessive jitter, or a dropped
connection. Secrets and raw environment values never enter episode metadata.

## Collection Modes and Intervention State Machine

The collector supports three explicit modes.

### Autonomous evaluation

GR00T owns the complete episode. The collector records all observations,
actions, outcome evidence, snapshots, and provenance. These episodes feed
measurement and hard-state selection. They are not initial BC targets.

### Full expert demonstration

The operator owns the episode from reset. Accepted successful episodes become
expert BC data. Practice mode uses the same controls but never creates a
training episode.

### DAgger recovery

GR00T creates the prefix. The operator performs a one-way takeover and controls
the remainder of the attempt. The raw episode retains both sources; the BC
exporter selects only expert-controlled observations paired with expert
actions.

```text
READY
  |-- practice --------------------------> PRACTICE_EXPERT
  |-- full expert -----------------------> EXPERT
  `-- policy rollout --------------------> POLICY

POLICY -- takeover --> EXPERT
POLICY -- success/timeout/error ---------> FINALIZE_DIAGNOSTIC

EXPERT -- accepted simulator success ----> SAVE_EXPERT_LABELS
EXPERT -- discard/timeout/unsafe --------> FINALIZE_DIAGNOSTIC

SAVE_EXPERT_LABELS ----------------------> RESET
FINALIZE_DIAGNOSTIC ---------------------> RESET or EXIT
```

Takeover is one-way within an episode. Before accepting the first expert
command, the collector displays the current simulated joint state, requires a
leader-arm synchronization tolerance, clears any queued GR00T actions, and
starts a new action-source segment. This prevents a discontinuity at takeover.

Controls provide activate, takeover, accept, discard, retry/reset, and safe
exit. Accept is enabled only after the challenge success checker passes. A
discard or failure clears the pending BC buffer without deleting the raw
diagnostic episode.

## Raw Episode Contract

Every raw episode contains synchronized 30 Hz records for:

- `observation.images.top_rgb`;
- `observation.images.left_rgb`;
- `observation.images.right_rgb`;
- absolute 12D `observation.state`;
- absolute 12D applied `action` in left-then-right organizer joint order;
- reward, geometric success, episode step, and wall/monotonic timestamps;
- action source: `policy`, `expert`, or `hold`;
- policy request/chunk identity and executed chunk offset;
- expert bridge sequence number and measured sample age;
- intervention segment and takeover step;
- garment category, name, release stage, and environment seed;
- canonical task instruction;
- policy repository, immutable revision, checkpoint step, and artifact hash;
- repository commit, container/image identity, asset revision, and simulator
  version;
- randomization strategy plus exact sampled parameters; and
- episode outcome, quality grade, rejection reasons, and operator disposition.

Raw metadata and frame annotations may use sidecar JSON/Parquet files. The raw
format may contain fields that the pinned GR00T loader does not consume. A
separate exporter materializes a canonical LeRobot dataset containing only the
three RGB cameras, 12D state/action, task instruction, and required metadata.
This prevents diagnostic fields from changing the pinned training schema.

## Behavior-Cloning Export Rules

The exporter operates fail-closed and never mutates the raw run.

- Full expert mode exports accepted expert frames.
- DAgger mode exports frames at or after takeover only when their applied
  action source is expert.
- Each exported frame must have a complete 16-step future expert-action window;
  incomplete correction tails are dropped.
- Policy-controlled, hold, stale, missing, clipped, unsafe, and post-disconnect
  commands are excluded.
- Failed attempts are excluded even if they contain expert frames.
- Evaluation-only garment/seed assignments are excluded.
- All three video streams must have exact frame count and FPS agreement with
  state/action data.
- The exporter emits selection counts and rejection counts by reason and proves
  that no non-expert target crossed the boundary.
- Train-only statistics and normalization are recomputed from the final mixed
  training snapshot; holdout episodes never influence them.

The first flywheel fine-tune samples approximately 70% from the immutable
organizer expert pool and 30% from accepted new full-expert and DAgger data.
Autonomous-policy actions receive 0% sampling weight initially. A later,
separately measured ablation may add at most 5--10% tightly filtered successful
self-imitation data.

## DAgger Quality and Rookie-Safe Collection

The collector grades expert attempts rather than requiring a perfect operator.

- **Grade A -- clean expert success:** smooth, coherent, accepted success with
  no transport or safety violations; normal or elevated sampling weight.
- **Grade B -- successful recovery:** includes hesitation, a corrected mistake,
  or a recovery maneuver but remains bounded and reaches accepted success;
  retained as lower-weight recovery data.
- **Grade C -- rejected diagnostic:** failed, unsafe, excessively erratic,
  transport-corrupted, or manually discarded; never exported to BC.

Automatic checks include joint limits, maximum step velocity and acceleration,
packet sequence, command age, latency/jitter, synchronization at takeover,
camera/frame alignment, and success-checker evidence. Thresholds are derived
from organizer expert statistics and recorded in the run manifest rather than
guessed per session.

The first session consists of roughly 10 practice episodes with no training
output, followed by 20--40 accepted long-pants full demonstrations and 20--40
accepted DAgger recoveries. Discarded attempts do not count against the target.
A North American rollout host is preferred for interactive collection because
transport latency matters for physical leader control; headless autonomous
evaluation remains region-independent.

## Public Evaluation Matrix

The first statistically useful baseline contains 280 canonical, unaugmented
episodes:

- 40 seen garment IDs times 5 seeds = 200 episodes;
- 8 public-unseen IDs times 10 seeds = 80 episodes; and
- 70 long-pants episodes in total.

The matrix is generated deterministically and committed with a checksum. One
public-unseen garment per category is permanently evaluation-only. The matrix
may evaluate those garments, but neither their autonomous episodes nor future
expert labels enter a training export. Seen, public-unseen, category, and
overall success are reported separately with confidence intervals.

The baseline does not enable lighting, texture, garment appearance, camera, or
arm-base randomization. It estimates submission behavior in the canonical
released environment.

## Domain Randomization and Training Augmentation

Randomized data is collected in separate strategy buckets after the fixed
matrix:

- 50% canonical environment;
- 30% mild environment randomization; and
- 20% stronger but physically plausible combined randomization.

Randomizable simulator properties include:

- dome-light intensity and color temperature/tint;
- table material/texture;
- garment base color and approved pattern/texture variants;
- garment initial translation/rotation and settling variation;
- small per-camera pose/FOV perturbations; and
- small robot-base placement perturbations.

Every sampled property is stored in the episode manifest. Randomization assets
are immutable and hashed. The collector rejects missing textures, invalid USD
prims, out-of-range camera geometry, and changes that make success criteria or
robot kinematics inconsistent.

Training-time image augmentation is independently sampled per camera and starts
conservatively:

- brightness, contrast, saturation, and small hue jitter;
- light blur and sensor noise;
- small crop, translation, scale, and rotation;
- modest cutout; and
- rare camera dropout.

Augmentation never changes action/state labels. A rendered sample sheet and a
fixed-seed loader test must pass before a paid fine-tune. Stronger augmentation
is introduced only after the canonical holdout shows no regression.

## Simulator Snapshots and Hard Mining

The raw collector stores restorable snapshots at reset, takeover, success,
timeout/failure, and configured progress checkpoints. A valid snapshot includes
robot joints, cloth particle positions and velocities, relevant simulator
state, garment identity/scale, RNG state, environment configuration, and exact
randomization values.

Snapshot restore is accepted only when a replay reproduces the same initial
camera observations within a recorded tolerance and advances without simulator
errors. Failed restores remain diagnostics and cannot seed expert collection.

Hard mining ranks states using observed failures and challenge geometry rather
than manual video impressions alone. Selected snapshots start targeted expert
or DAgger attempts while retaining the original episode and checkpoint
provenance.

## Asynchronous Flywheel Contract

Training, rollout, and DAgger workers communicate only through immutable Hub
commits and small manifests.

1. A training worker uploads policy revision `N` and its manifest.
2. A rollout worker adopts `N` only between episodes, never mid-episode.
3. Every episode records its generating policy revision.
4. Completed raw shards upload atomically with checksums and a terminal
   manifest.
5. The trainer starts an iteration by freezing exact input dataset commits.
6. Episodes arriving after that freeze wait for the next iteration.
7. A completed trainer uploads revision `N+1`; workers adopt it gradually.

The trainer does not assume that new data arrives forever. Iteration zero waits
for the 280-episode fixed baseline. Later fine-tunes require at least 40 new
accepted expert-labeled episodes or a configured equivalent number of eligible
expert frames. When the gate is not met, training pauses instead of repeatedly
optimizing unchanged data.

Policy staleness remains explicit. Old autonomous rollouts can still support
diagnostics, but metrics and any future advantage computation are grouped by
generating revision. Expert labels remain useful across revisions but retain
the policy/state provenance that elicited them.

## Worker Scaling

Six workers are the known-good production baseline for the prior eight-GPU
host: six GPUs handled Isaac rendering and two GPUs hosted GR00T servers.

The new runner performs a finite capacity sweep at 1, 2, 4, and 6 workers. It
tests 8 workers only if 6 passes. Counts above 8 are tested only when 8 provides
material throughput improvement and all safety margins pass. The production
gate requires:

- every trial reaches a first progress/success check and terminal record;
- no stuck synchronized wave or stale IPC state;
- no simulator, Vulkan, CUDA, policy-server, or video-encoding failure;
- bounded inference latency and queue depth;
- at least 20% free host RAM and acceptable swap behavior;
- at least 15% free rendering and inference VRAM;
- CPU utilization and run queue that do not starve simulator progress; and
- at least 15% throughput gain over the prior accepted worker count.

Sharing a rendering GPU among Isaac processes is an experimental assignment,
not the default. If 8 or 16 workers fail to improve throughput safely, the
system scales horizontally with another six-worker rollout machine.

## Artifact Layout and Hub Publishing

Each run is append-only and uses an unambiguous root:

```text
<run-root>/
  run-manifest.json
  matrix.json
  raw/<episode-id>/
    frames-and-actions/
    annotations.parquet
    episode.json
    snapshots/
    videos/
  exports/<export-id>/
    dataset/
    selection-report.json
    validation-report.json
    SHA256SUMS
  reports/
    rollout-report.json
    quality-report.json
    capacity-report.json
```

Raw episodes, BC exports, and policy checkpoints are uploaded to private Hub
repositories or private prefixes. An upload is considered durable only after a
fresh manifest read verifies the immutable commit, paths, hashes, sizes, and
expected episode counts. Local or remote instances are not stopped or deleted
merely because an upload command returned success.

## Failure Handling and Recovery

- A bridge disconnect holds the current simulated joint command, marks all
  subsequent frames ineligible, and requires explicit resynchronization.
- A policy-server failure finalizes the raw diagnostic episode but creates no
  BC output.
- A recorder error clears only the pending training buffer; completed raw
  episodes remain intact.
- Each episode writes into a temporary directory and is atomically renamed only
  after videos and metadata validate.
- Resume trusts only checksum-verified terminal episodes and never overwrites an
  incomplete attempt.
- Randomization failure falls back to no episode, not silently to canonical
  data under a randomized label.
- Hub failures retain local artifacts and retry without duplicating episode
  identities.
- Paid instances are stopped only after durable-copy verification; destruction
  remains a separately authorized action.

## Testing and Acceptance

### Local tests without Isaac Sim

- bridge protocol, authentication, sequencing, calibration identity, and stale
  sample rejection;
- intervention state transitions and one-way takeover;
- expert-only frame routing and complete 16-step future-action windows;
- A/B/C quality grading and rejection reasons;
- holdout exclusion and 70/30 dataset-mix manifests;
- canonical versus randomized strategy separation;
- deterministic matrix generation and confidence summaries;
- atomic episode finalization, resume, checksum verification, and upload
  manifests;
- simulator-snapshot schema validation; and
- capacity-report decision rules.

### Remote acceptance

1. One canonical autonomous episode completes with aligned trainable records,
   videos, and a restorable reset snapshot.
2. One deliberately failed autonomous episode is retained diagnostically and
   exports zero BC targets.
3. The Mac bridge reads the correct left/right physical leaders through the SSH
   tunnel and passes latency/calibration gates.
4. One practice episode creates no training data.
5. One full-expert success exports valid canonical data.
6. One DAgger attempt proves policy frames are excluded and expert frames are
   exported only after takeover.
7. One discarded/failed DAgger attempt creates no BC episode.
8. Mild lighting, table, garment, and camera randomization produces a manifest
   whose values match the render.
9. The 1/2/4/6 capacity sweep completes; 8 is attempted only if allowed by the
   finite gate.
10. A private Hub upload is verified from a fresh read before the host is
    stopped.

## Staged Delivery

1. Core episode schema, manifest, exporter, and unit tests.
2. Autonomous recorder, fixed 280-trial matrix, reports, and snapshot plumbing.
3. Capacity sweep and six-worker production orchestration.
4. Simulator randomization and training augmentation contracts.
5. Mac physical-leader bridge and remote receiver.
6. Full-expert and DAgger state machine with quality gates.
7. Immutable Hub synchronization and trainer new-data gates.
8. Remote acceptance, then paid collection.

The first paid collection remains blocked until the relevant stage's local tests
and one-worker remote acceptance pass. No stage requires a GUI for headless
autonomous evaluation; physical DAgger collection uses the interactive remote
view and a low-latency North American host.

## Non-Goals

- Reimplementing the winner's value/Q heads, AWR, RECAP, Thompson sampling, or
  best-of-N action selection in this phase.
- Treating public-unseen garments as a substitute for inaccessible private
  unseen identities.
- Training on failed or merely nonzero-reward actions as if they were expert.
- Automatically deleting Vast instances or local artifacts.
- Claiming more than six safe workers before measured acceptance.
