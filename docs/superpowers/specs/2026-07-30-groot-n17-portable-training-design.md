# Portable GR00T N1.7 Training System Design

**Status:** Approved design
**Date:** 2026-07-30

## Objective

Build a provider-independent training system for LeHome that can take a fresh
single-GPU Linux rental from authentication to the first GR00T N1.7 optimizer
step in 10–30 minutes when measured download bandwidth is at least 1 Gbps. The
system must train, checkpoint, and report without installing Isaac Sim on the
training machine.

Active model development moves from the existing π0.5 LoRA experiment to GR00T
N1.7. Existing π0.5 code, checkpoints, reports, and videos remain archived as
historical evidence; they are not deleted, silently converted, or used to
initialize GR00T.

## Why This Direction

The public LeHome GR00T N1.5 baseline provides a simpler starting point than
reproducing the winning π0.5-specific architecture and reinforcement-learning
system:

- It trained on the organizer's `four_types_merged` demonstrations.
- It trained the projector and diffusion action model while freezing the
  language model and visual encoder.
- It reported 70% success after 12,000 steps at batch 64, or 768,000 sample
  presentations.
- Later public iterations added successful and hard-case rollouts through short,
  low-learning-rate follow-up training, demonstrating a small data flywheel
  without reproducing the winning system's full RL stack.

The 70% result is self-reported, and its repository cannot be conclusively
mapped to a team in the official ranking. It is therefore an experiment
starting point, not a guaranteed competition score.

GR00T N1.7 is selected instead of N1.5 because it is the current model family
we want to develop. NVIDIA's new-embodiment guide provides a directly relevant
SO100 recipe: relative non-EEF arm actions, absolute gripper actions, and
training of the projector plus diffusion model with the language and visual
backbones frozen. There is no published LeHome ablation proving that
end-effector actions outperform relative joint actions, so the simpler official
SO100 representation is the primary path.

Evidence:

- <https://github.com/NVIDIA/Isaac-GR00T>
- <https://huggingface.co/nvidia/GR00T-N1.7-3B>
- <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md>
- <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md>
- <https://huggingface.co/theo-zhou/lehome-groot-submission-2>
- <https://huggingface.co/theo-zhou/lehome-groot-submission-4>

## Scope

This project includes:

- A public OCI training image that runs on Vast.ai, RunPod, or another
  compatible single-GPU Linux provider.
- A one-time converter from the organizer's LeHome dataset to the GR00T-flavored
  LeRobot schema.
- Prepared dataset statistics and metadata stored with the converted private
  dataset so new machines do not repeat preprocessing.
- GR00T N1.7 fine-tuning with the language and visual backbones frozen.
- One-episode memorization, batch-size smoke tests, full behavior-cloning
  training, checkpoint preservation, and machine-readable reports.
- A rollout adapter for evaluating trained GR00T checkpoints in the existing
  LeHome/Isaac Sim environment.
- A simple model-independent data flywheel for later successful rollouts and
  corrected hard states.
- A gated experimental relative end-effector representation.

This project does not include:

- Continuing active π0.5 training.
- Using the winner's learned weights, private data, advantage labels, or
  π0.5-specific auxiliary heads.
- Installing or running Isaac Sim inside the training image.
- Full tuning of GR00T's language or visual backbone.
- Reinforcement learning, advantage weighting, a value model, or automatic
  hard-state mining in the first implementation.
- Training on public-unseen evaluation garments.
- Treating failed policy actions as behavior-cloning targets.

## System Boundaries

The system has three independently replaceable parts:

```text
Organizer demonstrations
        |
        v
One-time dataset preparation -----> Private prepared dataset
                                           |
                                           v
Public training image -----------> Rental training GPU
                                           |
                                           v
                               Private checkpoints + reports
                                           |
                                           v
Existing LeHome/Isaac machine ---> Fixed rollout evaluation
                                           |
                                           v
                         Successes and human corrections
                                           |
                                           +----> next prepared-data version
```

### 1. Training image

The public image contains:

- A pinned CUDA/PyTorch runtime compatible with GR00T N1.7.
- A pinned checkout of NVIDIA Isaac-GR00T.
- FlashAttention and the remaining training dependencies.
- This repository's dataset, smoke-test, training, upload, and reporting tools.
- A `lehome-train` executable run as the image's non-root container user.

The image does not contain:

- Hugging Face credentials.
- Organizer data.
- Base-model weights.
- Fine-tuned checkpoints.
- Isaac Sim or Isaac Lab.

The default image target is:

```text
ghcr.io/ryanjin333/lehome-groot-n17-trainer:<git-sha>
```

The image is immutable by commit SHA. A human-readable release tag may point to
an already verified SHA, but experiments record the immutable digest.

The initial dependency lock is anchored to:

```text
Linux architecture: linux/amd64
Python: 3.10.18
CUDA base: nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
CUDA base digest: sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719
Isaac-GR00T commit: 23ace64f17aa5015259b8609d371eb61a357c776
GR00T-N1.7-3B revision: 2fc962b973bccdd5d8ce4f67cc63b264d6886495
```

PyTorch, FlashAttention, and every Python transitive dependency are resolved
once into a checked-in, hash-bearing `uv.lock` from those anchors. Image builds
must use `uv sync --frozen`; they may not resolve newer packages. A dedicated
lock-update command requires explicit replacement revisions, regenerates the
lock and compatibility report, and fails CI until both are committed and the
GPU integration test passes. Model downloads always provide the full Hugging
Face revision. The built OCI digest is added to the release manifest before the
image is used for an experiment.

### 2. Private artifacts

The default Hugging Face repositories are:

```text
ryanjin333/lehome-groot-n17-data
ryanjin333/lehome-groot-n17-models
```

The data repository contains versioned prepared datasets, manifests, metadata,
and statistics. The model repository contains checkpoints, training reports,
evaluation reports, and exact provenance. Repository creation and access remain
explicit user-owned operations. Containers receive `HF_TOKEN` only through the
provider's secret mechanism or process environment. The container refuses
interactive `hf auth login` so a token is never written into its cache or
output.

### 3. Rollout machine

Isaac Sim remains on a separate LeHome-capable machine. It loads a promoted
checkpoint, converts simulator observations into the GR00T input contract,
converts predicted actions back into LeHome joint commands, runs the official
success checker, and records videos and metrics.

Training and rollout machines communicate only through versioned artifacts.
Neither machine relies on an unrecorded provider snapshot.

## Dataset Contract

### Input

The source is the official LeHome `four_types_merged` organizer demonstration
dataset. Winner data, winner checkpoints, public-unseen garments, autonomous
failures, and operator corrections are excluded from version 1.

### Observation modalities

Each training sample contains:

- Three RGB camera observations, mapped from the organizer dataset through a
  checked modality manifest.
- Twelve current robot joint values: six for each arm, including each gripper.
- The task instruction `fold the garment on the table`.

The converter must refuse missing cameras, ambiguous camera mappings, incorrect
joint dimensions, non-finite values, inconsistent episode FPS, or mismatched
timestamps. Camera names are resolved once from the organizer schema and then
stored explicitly in the prepared dataset manifest; downstream code never
guesses them.

### Primary action representation

The first trainable representation seen by the model is:

- Five relative joint deltas for the left arm.
- One absolute left-gripper command.
- Five relative joint deltas for the right arm.
- One absolute right-gripper command.
- A fixed 16-step future action horizon matching NVIDIA's N1.7 SO100
  new-embodiment example.

For each non-gripper joint:

```text
relative_action[t] = target_joint_position[t] - current_joint_position[t]
```

The prepared LeRobot dataset preserves the organizer's absolute joint targets.
The pinned GR00T relative-action transform performs the subtraction above at
load time and produces `relative_stats.json`. The converter must not store
already-relative joints because that would make GR00T subtract the current state
twice.

The gripper stays absolute because its open/closed target has a stable physical
meaning and matches NVIDIA's SO100 new-embodiment recipe.

Training statistics are computed only from the training split and are packaged
with the prepared dataset and checkpoint processors. OpenPI `norm_stats.json`
is not reused.

### Splits and provenance

Splits occur at the episode level so frames from one episode cannot appear in
both training and validation. The prepared version records:

- Source repository and revision.
- Source and output file hashes.
- Converter commit and container digest.
- Episode IDs in each split.
- Modality and action schemas.
- FPS, frame counts, and episode counts.
- Per-modality statistics and finite-value validation.

Running preparation again with identical inputs must either reproduce the same
artifact hashes or fail with a report describing the difference.

### One-time preparation workflow

Dataset conversion is an explicit trusted-machine workflow, separate from
fresh-rental preflight:

```bash
lehome-train data inspect \
  --source /data/four_types_merged
lehome-train data convert \
  --source /data/four_types_merged \
  --output /prepared/lehome-groot-n17-v1
lehome-train data validate \
  --dataset /prepared/lehome-groot-n17-v1
lehome-train data publish \
  --dataset /prepared/lehome-groot-n17-v1 \
  --repo ryanjin333/lehome-groot-n17-data \
  --revision lehome-groot-n17-v1
```

`inspect` produces the proposed camera/joint mapping and requires an explicit
checked-in mapping before conversion. `convert` is deterministic and writes no
remote state. `validate` runs the full schema, finite-value, split, statistics,
and hash checks. Only `publish` needs write credentials, and it publishes
exactly the validated manifest allowlist. The immutable dataset revision and
manifest hash then become inputs to rental-machine `prepare`; conversion is not
repeated on training rentals.

## Optional Relative End-Effector Experiment

Relative end-effector control is an experimental adapter, not the default
training path. It is allowed to progress only after all of these checks pass:

1. Convert expert joint trajectories to end-effector poses with forward
   kinematics.
2. Express consecutive poses as relative Cartesian and rotational deltas in one
   documented coordinate frame.
3. Convert those deltas back to joint targets through the same inverse
   kinematics path that rollout will use.
4. Replay a fixed canary set of 12 expert episodes, three from each garment
   category. Across that set, require zero IK failures, zero joint-limit
   violations, Cartesian reconstruction p95 at or below 5 mm, rotation
   reconstruction p95 at or below 3 degrees, and joint-target reconstruction
   p95 at or below 2 degrees. Each replay must retain at least 95% of the
   expert's geometric score improvement from the identical initial state.
5. Pass a one-episode memorization test and a short equal-sample pilot against
   the relative-joint baseline.

The first implementation does not full-train both representations. If the
end-effector path passes replay validation, it receives at most a short
successive-halving pilot at 2,000, 5,000, and 10,000 steps. It replaces the
primary path only if it gains at least two successes on the 24-trial
`seen-development` matrix and no garment category loses more than one success
relative to the equal-sample relative-joint pilot.

## Training Workflow

### Stage 0: preflight

`lehome-train prepare` validates:

- Exactly one visible CUDA GPU.
- Supported CUDA capability and at least 40 GB VRAM for training.
- At least 200 GB of writable local disk.
- Hugging Face authentication and read/write access to the configured private
  repositories.
- Base-model and prepared-dataset revisions and hashes.
- Dataset schema, dimensions, splits, statistics, and finite values.
- Checkpoint upload access before paid training starts.

Preflight never silently falls back to CPU, another GPU, multiple GPUs, or a
different artifact revision.

### Stage 1: one-episode memorization

Before the full dataset, train a disposable checkpoint on one selected expert
episode. This test proves that the entire data/action/model path can learn.

The gate passes only when:

- Training loss and diagnostic action errors remain finite.
- Normalized action MSE on the selected episode falls to at most 10% of its
  initialized value, and every action dimension improves.
- An offline prediction replay has the correct shapes, action ranges, and
  temporal alignment.
- From the identical saved simulator state, the memorized rollout achieves at
  least 80% of the official geometric-score improvement produced by replaying
  the expert episode and triggers no joint-limit or simulator-safety abort.

This checkpoint is diagnostic and is never promoted as the competition model.

### Stage 2: batch smoke tests

Run 100 optimizer steps sequentially using the candidate list selected from
physical VRAM:

```text
40–63 GB  -> 8, 16, 32
64–95 GB  -> 16, 32, 64
96 GB+    -> 16, 32, 64
```

The RTX PRO 6000 96 GB acceptance run therefore tests 16, 32, and 64. Each test
uses the same data revision and training configuration and records:

- Model/data initialization time.
- Compilation or graph warm-up time.
- Steady-state optimizer steps per second.
- Samples per second.
- Peak allocated and reserved VRAM.
- GPU utilization, power, temperature, and host-memory use when available.
- Loss trajectory and finite-loss status.
- OOMs and all other errors.

The selected batch is the largest stable batch that leaves at least 10% of
physical VRAM free after warm-up. A failed batch stops tests at larger sizes
unless the failure is proven unrelated to memory. If the first candidate OOMs
or misses the headroom gate, the tool tests smaller powers of two down to batch
1. If batch 1 cannot leave 10% free, it aborts and reports that a larger GPU or
a separately approved memory optimization is required. No long run begins
until one batch passes.

For the single-GPU, no-accumulation smoke tests, GR00T's `global_batch_size` is
the physical per-device batch. The smoke report records gradient accumulation
separately and rejects comparisons that change it.

### Stage 3: initial behavior-cloning run

The first full run targets exactly 768,000 sample presentations:

```text
batch 64 -> 12,000 optimizer steps
batch 32 -> 24,000 optimizer steps
batch 16 -> 48,000 optimizer steps
batch 8  -> 96,000 optimizer steps
batch 4  -> 192,000 optimizer steps
batch 2  -> 384,000 optimizer steps
batch 1  -> 768,000 optimizer steps
```

Sample presentations are global batch size multiplied by optimizer steps. They
are used to keep the initial exposure constant across hardware-compatible batch
sizes; they do not claim mathematical equivalence between different models or
optimizers.

The run trains:

- The multimodal projector.
- The diffusion/flow-matching action transformer.

The run freezes:

- The language/VLM backbone.
- The SigLip2 visual tower.

The learning-rate schedule is expressed as fractions of the selected total
steps so changing batch size cannot accidentally truncate warm-up or decay.
Checkpoints are saved every 64,000 sample presentations, uploaded
asynchronously, and retained until rollout selection is complete. The lowest
training loss is not automatically promoted.

### Stage 4: fixed rollout evaluation

Checkpoint selection and unseen reporting use two separate, version-controlled
matrices:

- `seen-development`: eight organizer-seen garment instances with reset seeds
  42, 43, and 44, for 24 paired trials. Candidate checkpoints and flywheel
  iterations may be selected only on this matrix.
- `public-unseen-audit`: eight public-unseen garment instances with the same
  three reset seeds, for 24 trials. It is run once only after the promoted
  checkpoint and dataset version are locked. Its result is reported but cannot
  change checkpoint selection, promotion, training data, or hyperparameters.

Both matrices share:

- A fixed simulator, camera, control-frequency, action-horizon, and
  action-execution settings.
- Deterministic reset validation.

Reports separate seen and unseen success rates and include return, geometric
success measurements, failure category, checkpoint identity, seed, garment,
video path, and configuration hash. Public-unseen trajectories never enter a
training dataset.

Within the initial run, select the checkpoint with the most successes on the 24
`seen-development` trials; break ties by higher mean normalized geometric score,
then by fewer sample presentations. A later flywheel checkpoint replaces the
current champion only if either:

1. It gains at least two successes overall and no garment category loses more
   than one success; or
2. It gains zero or one success, improves mean normalized geometric score by at
   least 5%, and no garment category loses more than one success.

If any later design or data decision is influenced by an unseen-audit result,
that audit is marked development-exposed and cannot be described as blind
evaluation again; a new sequestered holdout is required for a final claim.

## Simple Data Flywheel

After the first BC policy records at least one success on the 24-trial
`seen-development` matrix, iteration uses:

```text
organizer demonstrations
  + successful policy rollouts from allowed training garments
  + successful human corrections from policy-induced hard states
        |
        v
short low-learning-rate fine-tune
        |
        v
fixed seen-development evaluation
        |
        +--> promote only when held-out results improve
        |
        +--> one public-unseen audit after the campaign is locked
```

Rules:

- A single garment category may start the flywheel; all four categories do not
  need to succeed first.
- Dataset sampling keeps organizer demonstrations present so a successful
  category cannot erase competence elsewhere. Organizer demonstrations receive
  at least 50% of samples; the remaining probability is divided evenly among
  available successful-rollout and human-correction sources and is recorded in
  the dataset manifest.
- Successful rollout replay is preferred before DAgger because it is cheaper
  and directly reinforces behavior already known to work.
- DAgger is used when the policy repeatedly reaches a recoverable hard state
  that is missing from demonstrations. "Repeatedly" means the same versioned
  failure category appears in at least three of the 24 seen-development trials,
  and a human confirms that takeover can still complete the task.
- Only expert-controlled actions after takeover become BC targets.
- Failed policy prefixes and unrecovered corrections remain diagnostics, not
  action labels.
- Each new dataset version records source proportions and episode identities.
- More collected data is not automatically better; every iteration must beat
  the frozen evaluation contract before promotion.

The first flywheel fine-tune uses 256,000 sample presentations at one tenth of
the initial BC run's peak learning rate, with checkpoints every 64,000
presentations. Later exposure or learning-rate changes require a new versioned
experiment configuration; they are not silently extended until loss reaches an
arbitrary value.

This flywheel is intentionally model-independent. It can later feed a different
VLA after that model's adapter and normalization are regenerated.

## Command-Line Interface

A fresh training rental uses the image interactively so all stages share the
mounted cache and output directory:

```bash
docker run --gpus '"device=0"' --rm -it \
  -v "$PWD/output:/output" \
  -v "$PWD/cache:/cache" \
  -e HF_TOKEN \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  ghcr.io/ryanjin333/lehome-groot-n17-trainer:<git-sha>

lehome-train prepare
lehome-train memorize --episode-id <training-episode-id>
lehome-train smoke --batches auto --steps 100
lehome-train train --sample-presentations 768000
lehome-train report
lehome-train sync
```

Provider templates may set the equivalent image, GPU, output mount, and secret
without using `docker run`. Every command writes both human-readable logs and a
machine-readable JSON status file under `/output`.

`/cache` contains downloaded public/base-model and prepared-dataset blobs only;
it is never uploaded. The CLI reads `HF_TOKEN` from the process environment,
passes it explicitly only to required Hub calls, and never invokes Hugging Face
login. The supervisor retains the token in memory for its download, upload, and
`sync` workers, but removes it from the training subprocess environment.
Container home is ephemeral and outside both mounts.

Commands are restart-safe:

- A completed stage is skipped only when its output manifest and hashes match.
- A partial stage resumes only from a verified compatible checkpoint.
- A mismatch creates a new experiment ID instead of overwriting prior output.
- `sync` considers only paths listed in the experiment's generated
  `sync-manifest.json`: checkpoints, redacted logs, resolved configuration,
  provenance, and JSON reports under that experiment directory. It rejects
  symlinks, dotfiles, caches, environment files, credential-store filenames,
  paths outside the experiment root, and content matching supported token
  formats. It verifies remote hashes before declaring local artifacts
  disposable.

## Fresh-Machine Time and Storage Budget

The recommended rental is:

```text
GPU: 1 x RTX PRO 6000 96 GB
CPU: 16 vCPU or more
RAM: 64 GB minimum, 128 GB preferred
Local ephemeral disk: 200 GB
Network: at least 1 Gbps preferred
```

The 10–30 minute target is time to the first optimizer step, not training
completion. It assumes the public image and private prepared dataset already
exist and the provider has normal registry/Hugging Face bandwidth. The report
separates image pull, model download, dataset download, preflight, and model
initialization so slower providers can be diagnosed rather than hidden.

All irreplaceable artifacts are uploaded and hash-verified before deleting a
rental. Local provider volumes are optional caches, not the sole copy.

## Error Handling and Safety

- Never print or store access tokens in an image, manifest, command report, or
  Git history.
- Refuse training if the visible GPU count is not exactly one.
- Refuse schema drift instead of padding, dropping, or reordering joints
  silently.
- Stop immediately on non-finite loss or gradients. Uploads retry five times
  with bounded exponential backoff. If uploads remain unavailable, training may
  continue only while free disk can hold two additional complete checkpoints
  plus 20 GB; otherwise it pauses before the next checkpoint boundary.
- Preserve the last verified resumable checkpoint before pruning older local
  checkpoints.
- Write final cost and runtime metadata when provider price and instance start
  time are supplied.
- Do not delete a rental until checkpoint, logs, configuration, and report
  hashes are verified remotely.
- Deletion remains an explicit provider operation scoped to the named rental;
  the training CLI does not delete cloud instances.

## Verification

### Automated tests

- Dataset conversion schema and deterministic split tests.
- Relative-joint/gripper transformation and inverse reconstruction tests.
- Statistics, finite-value, and hash-validation tests.
- Camera mapping and timestamp-alignment failure tests.
- Component-freezing assertions.
- Sample-budget and learning-rate-schedule tests.
- Batch smoke-result parser and 10% headroom selection tests.
- Checkpoint resume, incompatible-resume rejection, and upload verification
  tests.
- Evaluation manifest, holdout-boundary, and promotion-gate tests.
- Flywheel routing tests proving failed policy actions never become BC labels.

### Container integration tests

- Build the image from a clean checkout.
- Scan the image for secrets and unexpected large assets.
- Start it with exactly one GPU and validate pinned dependency revisions.
- Run preflight against a small synthetic dataset.
- Run one CPU-safe converter test and one GPU optimizer step.
- Verify `/output` contains the expected JSON status and provenance.

### Real acceptance run

On a fresh RTX PRO 6000 96 GB rental:

1. Pull the public image and authenticate to Hugging Face.
2. Reach the first optimizer step within 30 minutes under at least 1 Gbps
   measured download bandwidth.
3. Pass one-episode memorization.
4. Complete batch 16, 32, and 64 smoke tests unless a proven memory failure
   makes larger batches invalid.
5. Start or resume the 768,000-presentation run at the selected batch.
6. Upload and hash-verify a checkpoint, logs, configuration, and report.
7. Load that checkpoint on the rollout machine and complete one deterministic
   seen-garment smoke rollout.

## Acceptance Criteria

The implementation is complete when:

- A fresh compatible Vast.ai or RunPod machine can use the same immutable image
  without provider-specific installation commands.
- The prepared dataset is downloaded rather than regenerated on each rental.
- No Isaac Sim installation is required for training.
- The action representation, camera mapping, frozen/trainable components,
  sample budget, and artifact revisions are explicit in every experiment.
- One-episode memorization proves the end-to-end learning path.
- Batch smoke tests select the largest stable batch with at least 10% VRAM
  headroom.
- A restart can resume from a remotely verified checkpoint.
- A promoted checkpoint completes the fixed seen/unseen evaluation with videos
  and machine-readable results.
- The first flywheel iteration can add successful replay or human-corrected
  data without admitting failed policy actions or public-unseen data.
- Existing π0.5 artifacts remain intact and clearly separated from GR00T.
