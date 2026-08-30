# Portable GR00T N1.7 Trainer Phase 1 Implementation Plan

> **For Codex:** Execute this plan with the `superpowers:subagent-driven-development`
> skill. Use a fresh implementation subagent per task, then a spec review and
> code-quality review before moving to the next task.

**Goal:** Build a provider-independent, single-GPU GR00T N1.7 training image and
CLI that converts the organizer LeHome demonstrations once, validates and
publishes the prepared dataset, runs a diagnostic memorization job and
batch-size smoke tests, trains for exactly 768,000 sample presentations, and
hash-verifies private checkpoint/report uploads.

**Architecture:** Add an isolated Python 3.10 package under `trainer/`; do not
change the repository's Isaac Sim Python 3.11 environment. The package wraps
the pinned official Isaac-GR00T training entry point rather than reimplementing
its trainer. Commands exchange immutable manifests through `/prepared`,
`/cache`, and `/output`. Phase 1 contains no Isaac Sim rollout backend, so its
memorization result remains non-promotable until Phase 2 closes the simulator
replay gate.

**Pinned stack:** Python 3.10.18, CUDA 12.8.1 Ubuntu 22.04 base digest
`sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719`,
Isaac-GR00T commit `23ace64f17aa5015259b8609d371eb61a357c776`, and
GR00T-N1.7-3B revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`.

---

## Task 1: Scaffold the isolated trainer package and immutable configuration

**Files:**

- Create: `trainer/pyproject.toml`
- Create: `trainer/uv.lock`
- Create: `trainer/src/lehome_train/__init__.py`
- Create: `trainer/src/lehome_train/cli.py`
- Create: `trainer/src/lehome_train/constants.py`
- Create: `trainer/tests/test_cli.py`
- Modify: `.gitignore`

**Steps:**

1. Add failing CLI tests for `lehome-train --help`, the seven command groups
   (`data`, `prepare`, `memorize`, `smoke`, `train`, `report`, `sync`), and the
   exact pinned revisions.
2. Add a Python 3.10 package with a console entry point
   `lehome-train = lehome_train.cli:main`; use Typer only for argument parsing.
3. Keep all GR00T-specific imports lazy so schema/unit tests run on macOS
   without CUDA.
4. Store the pinned revisions and default private repository names in typed
   immutable settings. Require explicit overrides rather than falling back to
   `main` or `latest`.
5. Unignore `trainer/tests/` in `.gitignore`.
6. Generate and check in `trainer/uv.lock`; verify:

```bash
cd trainer
uv sync --frozen
uv run pytest tests/test_cli.py -q
uv lock --check
```

Expected: tests pass, the lock does not change, and importing the CLI does not
import Isaac Sim or CUDA.

## Task 2: Define manifest schemas, hashing, and secret-safe I/O

**Files:**

- Create: `trainer/src/lehome_train/models.py`
- Create: `trainer/src/lehome_train/io.py`
- Create: `trainer/src/lehome_train/redaction.py`
- Create: `trainer/tests/test_models.py`
- Create: `trainer/tests/test_io.py`
- Create: `trainer/tests/test_redaction.py`

**Steps:**

1. Write failing tests for strict JSON parsing, unknown-field rejection,
   canonical JSON hashes, atomic status writes, path traversal, symlinks,
   dotfiles, token-shaped content, and filenames associated with credentials.
2. Implement typed schemas for source inspection, camera/joint mappings,
   prepared-dataset provenance, experiment configuration, checkpoint records,
   smoke results, memorization results, and sync manifests.
3. Implement canonical UTF-8 JSON serialization and SHA-256 streaming file
   hashes. Atomic writes must use a temporary sibling followed by `os.replace`.
4. Implement a generated allowlist for uploads. Reject paths outside the
   experiment root, symlinks, caches, `.env`, Hugging Face token stores, and
   supported access-token patterns.
5. Verify:

```bash
cd trainer
uv run pytest tests/test_models.py tests/test_io.py tests/test_redaction.py -q
```

Expected: malformed manifests fail closed and tests never echo the synthetic
secret values.

## Task 3: Inspect and deterministically convert the organizer dataset

**Files:**

- Create: `trainer/src/lehome_train/data/inspect.py`
- Create: `trainer/src/lehome_train/data/convert.py`
- Create: `trainer/src/lehome_train/data/mapping.py`
- Create: `trainer/src/lehome_train/data/split.py`
- Create: `trainer/config/lehome_four_types_mapping.json`
- Create: `trainer/tests/fixtures/source_dataset.py`
- Create: `trainer/tests/test_data_inspect.py`
- Create: `trainer/tests/test_data_convert.py`

**Steps:**

1. Build a tiny synthetic two-arm, three-camera LeRobot fixture with multiple
   episodes, timestamps, FPS, and 12-dimensional state/actions.
2. Write failing tests proving that inspection reports proposed mappings but
   conversion refuses to guess them. Cover missing/extra cameras, ambiguous
   joint order, non-finite values, inconsistent FPS, timestamp drift, and wrong
   dimensions.
3. Add a checked mapping for the actual `four_types_merged` schema. `inspect`
   must print and save the observed schema and compare it to this mapping.
4. Split by episode ID using a fixed seed and stable sort; never split frames
   from one episode across training and validation.
5. Convert current state, three RGB streams, the fixed instruction
   `fold the garment on the table`, and the organizer's absolute joint targets
   into GR00T's LeRobot layout. Preserve absolute targets for all 12 action
   dimensions; the pinned GR00T relative-action transform, not the converter,
   converts the ten arm joints to relative targets while keeping each gripper
   absolute. Pad only at the episode tail using the pinned GR00T dataset
   convention and record the valid-action mask.
6. Preserve source IDs and write source/output hashes, converter commit, FPS,
   frame and episode counts, split IDs, and modality schemas.
7. Verify deterministic output by converting the fixture twice and comparing
   every manifest and payload hash:

```bash
cd trainer
uv run pytest tests/test_data_inspect.py tests/test_data_convert.py -q
```

Expected: identical inputs produce identical hashes; schema drift produces an
actionable refusal.

## Task 4: Compute GR00T statistics and validate the prepared dataset

**Files:**

- Create: `trainer/src/lehome_train/data/stats.py`
- Create: `trainer/src/lehome_train/data/validate.py`
- Create: `trainer/src/lehome_train/groot/modality.py`
- Create: `trainer/tests/test_data_stats.py`
- Create: `trainer/tests/test_data_validate.py`

**Steps:**

1. Write failing tests that statistics use training episodes only, contain
   finite values for all state/action dimensions, and never reuse OpenPI
   `norm_stats.json`.
2. Implement the custom GR00T modality configuration with three current RGB
   cameras, one current 12-dimensional state, the language instruction
   `fold the garment on the table`, and 16 future 12-dimensional actions. Mark
   ten arm joints relative and both grippers absolute so GR00T converts the
   stored absolute targets exactly once.
3. Call the pinned `gr00t/data/stats.py` APIs from the runtime integration
   path; keep a small pure-Python reference implementation for fixture tests.
4. Write `meta/stats.json`, `meta/relative_stats.json`, modality config,
   validation report, and hashes into the prepared version.
5. Run the pinned GR00T dataset loader against the fixture and consume one
   batch. The deterministic validation split is validated and offline-scored;
   do not pass it to the pinned Trainer, whose `DatasetFactory` requires
   `eval_strategy == "no"`.
6. Verify:

```bash
cd trainer
uv run pytest tests/test_data_stats.py tests/test_data_validate.py -q
```

Expected: exact dimensions and finite values pass; train/validation leakage or
missing relative statistics fails.

## Task 5: Publish and retrieve immutable prepared datasets

**Files:**

- Create: `trainer/src/lehome_train/hub.py`
- Create: `trainer/src/lehome_train/data/publish.py`
- Create: `trainer/tests/test_hub.py`
- Create: `trainer/tests/test_data_publish.py`

**Steps:**

1. Write tests using a fake Hub transport for explicit token passing, immutable
   revision selection, retry limits, remote hash verification, read/write
   permission failures, and redacted error messages.
2. Require `HF_TOKEN` in process memory for remote operations. Never call
   `hf auth login`, write a credential file, or include the token in a child
   process environment.
3. Publish only the validated manifest allowlist to
   `ryanjin333/lehome-groot-n17-data`; refuse dirty or unhashed payloads.
4. Download by explicit revision, verify every hash, and mark the local prepared
   directory complete only after verification.
5. Verify:

```bash
cd trainer
uv run pytest tests/test_hub.py tests/test_data_publish.py -q
```

Expected: remote mismatch leaves the dataset incomplete and non-trainable.

## Task 6: Add the GR00T training adapter and component-freezing contract

**Files:**

- Create: `trainer/src/lehome_train/groot/config.py`
- Create: `trainer/src/lehome_train/groot/launch.py`
- Create: `trainer/src/lehome_train/groot/metrics.py`
- Create: `trainer/tests/test_groot_config.py`
- Create: `trainer/tests/test_groot_launch.py`

**Steps:**

1. Write failing tests for a single visible GPU, global batch equal to physical
   batch, gradient accumulation exactly 1, action horizon 16, fractional
   warm-up/decay, component freeze flags, explicit base revision, and removal
   of `HF_TOKEN` from the trainer subprocess.
2. Generate arguments for the pinned official `launch_finetune.py`; do not copy
   or fork the trainer. Set `tune_llm=false`, `tune_visual=false`,
   `tune_projector=true`, and `tune_diffusion_model=true`.
3. Refuse CPU, zero GPUs, multiple visible GPUs, unpinned model/data revisions,
   or an existing incompatible experiment directory.
4. Parse official trainer logs into structured loss, steps/s, samples/s,
   checkpoint timing, and finite-loss records without making log text the
   source of experiment identity.
5. Verify:

```bash
cd trainer
uv run pytest tests/test_groot_config.py tests/test_groot_launch.py -q
```

Expected: generated command/config matches pinned N1.7 semantics and secrets
are absent from the recorded environment and command.

## Task 7: Implement preflight and restart-safe experiment identity

**Files:**

- Create: `trainer/src/lehome_train/preflight.py`
- Create: `trainer/src/lehome_train/experiment.py`
- Create: `trainer/src/lehome_train/commands/prepare.py`
- Create: `trainer/tests/test_preflight.py`
- Create: `trainer/tests/test_experiment.py`

**Steps:**

1. Write failing tests for minimum 40 GB VRAM, exactly one visible GPU, 200 GB
   writable disk, dataset/model revision verification, Hub permission checks,
   status resumption, and incompatible resume rejection.
2. Create an experiment ID from canonical resolved configuration and artifact
   hashes. A mismatch must create a new directory rather than overwrite output.
3. Split timed preflight stages into image/runtime verification, network
   measurement, model download, dataset download, schema/hash validation, and
   model initialization.
4. Require upload permission before paid training and create both readable log
   and machine-readable status files under `/output/<experiment-id>/`.
5. Verify:

```bash
cd trainer
uv run pytest tests/test_preflight.py tests/test_experiment.py -q
```

Expected: a completed matching stage skips safely; a partial compatible stage
resumes; an incompatible stage is preserved and superseded.

## Task 8: Implement offline one-episode memorization

**Files:**

- Create: `trainer/src/lehome_train/commands/memorize.py`
- Create: `trainer/src/lehome_train/offline_eval.py`
- Create: `trainer/tests/test_memorize.py`
- Create: `trainer/tests/test_offline_eval.py`

**Steps:**

1. Write failing tests for deterministic episode selection from the training
   split, initialized-versus-final normalized MSE, every-action-dimension
   improvement, finite shapes/ranges, temporal alignment, the fixed budget,
   evaluation cadence, and deterministic early stopping.
2. Train a disposable experiment on exactly one selected expert episode at
   physical batch 1 for at most 10,000 sample presentations (10,000 optimizer
   steps). Evaluate every 500 presentations and checkpoint every 1,000.
3. Pass and stop only when normalized action MSE is at most 10% of its
   initialized value and every dimension improves at two consecutive
   evaluations. Otherwise stop at the fixed budget and report the failed gate;
   never extend automatically.
4. Always emit `promotable: false` and
   `pending_gate: simulator_expert_replay` because Phase 1 excludes Isaac Sim.
   Do not claim the approved memorization gate is fully passed until Phase 2
   reaches at least 80% of expert geometric-score improvement from the same
   saved state.
5. Verify:

```bash
cd trainer
uv run pytest tests/test_memorize.py tests/test_offline_eval.py -q
```

Expected: an offline pass proves the data/model path can overfit but cannot
promote a competition checkpoint.

## Task 9: Implement sequential batch smoke tests and telemetry

**Files:**

- Create: `trainer/src/lehome_train/commands/smoke.py`
- Create: `trainer/src/lehome_train/telemetry.py`
- Create: `trainer/src/lehome_train/batch_select.py`
- Create: `trainer/tests/test_smoke.py`
- Create: `trainer/tests/test_batch_select.py`

**Steps:**

1. Write failing tests for VRAM-tier candidates, fallback, sequential launch,
   early stop after memory failure, 100 optimizer steps, 10% free-VRAM gate,
   non-finite loss, and fixed gradient accumulation.
2. Use candidates `8,16,32` for 40–63 GB and `16,32,64` for 64 GB or more.
   If the first candidate fails the headroom gate, try `8,4,2,1` as applicable.
   Never launch a larger batch after a proven OOM.
3. Do not introduce gradient checkpointing, accumulation, allocator overrides,
   offload, quantization, or a changed camera/action contract during the
   comparison.
4. Sample NVML telemetry for peak allocated/reserved memory, utilization,
   power, temperature, and host memory. Separate initialization/warm-up time
   from steady-state steps/s and samples/s.
5. Select the largest stable batch leaving at least 10% physical VRAM free.
6. Verify:

```bash
cd trainer
uv run pytest tests/test_smoke.py tests/test_batch_select.py -q
```

Expected: deterministic selection with a complete JSON record for every
attempt.

## Task 10: Implement the fixed-exposure training run and checkpoint lifecycle

**Files:**

- Create: `trainer/src/lehome_train/commands/train.py`
- Create: `trainer/src/lehome_train/schedule.py`
- Create: `trainer/src/lehome_train/checkpoints.py`
- Create: `trainer/tests/test_schedule.py`
- Create: `trainer/tests/test_checkpoints.py`
- Create: `trainer/tests/test_train.py`

**Steps:**

1. Write failing tests for exactly 768,000 presentations, optimizer-step
   derivation, fractional learning-rate scheduling, checkpoints every 64,000
   presentations, finite-loss aborts, two-checkpoint disk reserve, resumability,
   and incompatible-checkpoint rejection.
2. Require the selected smoke result. Derive
   `optimizer_steps = 768000 / physical_batch` and refuse a non-integral result.
3. Save checkpoints every `64000 / physical_batch` optimizer steps. Preserve
   the latest verified resumable checkpoint before pruning.
4. Upload asynchronously with five bounded retries. If uploads remain
   unavailable, continue only while disk can retain two additional complete
   checkpoints plus 20 GB; otherwise pause before the next boundary.
5. Record provider hourly price and instance-start time when provided, but do
   not delete the rental.
6. Verify:

```bash
cd trainer
uv run pytest tests/test_schedule.py tests/test_checkpoints.py tests/test_train.py -q
```

Expected: batch 64 produces 12,000 steps, batch 32 produces 24,000, batch 16
produces 48,000, and batch 8 produces 96,000, all with identical presentations.

## Task 11: Implement reports, verified sync, and operator documentation

**Files:**

- Create: `trainer/src/lehome_train/commands/report.py`
- Create: `trainer/src/lehome_train/commands/sync.py`
- Create: `trainer/tests/test_report.py`
- Create: `trainer/tests/test_sync.py`
- Create: `trainer/README.md`
- Create: `docs/groot_n17_training.md`
- Modify: `README.md`

**Steps:**

1. Write failing tests for provenance completeness, cost/runtime calculations,
   secret redaction, generated sync allowlists, remote hash verification, and
   refusal to mark unmatched artifacts disposable.
2. Report exact image digest, repository commit, GR00T/base revisions, prepared
   dataset revision and hash, resolved training config, smoke metrics,
   checkpoint hashes, runtime, and cost.
3. Implement `sync` against the model repository with explicit token passing.
   Mark an artifact remotely verified only after downloading or querying its
   immutable hash successfully.
4. Document the one-time trusted-machine flow:

```bash
lehome-train data inspect --source /data/four_types_merged
lehome-train data convert --source /data/four_types_merged --output /prepared/lehome-groot-n17-v1
lehome-train data validate --dataset /prepared/lehome-groot-n17-v1
lehome-train data publish --dataset /prepared/lehome-groot-n17-v1 --repo ryanjin333/lehome-groot-n17-data --revision lehome-groot-n17-v1
```

5. Document the rental flow:

```bash
lehome-train prepare
lehome-train memorize --episode-id <training-episode-id>
lehome-train smoke --batches auto --steps 100
lehome-train train --sample-presentations 768000
lehome-train report
lehome-train sync
```

6. Verify:

```bash
cd trainer
uv run pytest tests/test_report.py tests/test_sync.py -q
```

Expected: documentation can be followed without Isaac Sim and neither reports
nor upload payloads contain credentials.

## Task 12: Build and accept the immutable training image

**Files:**

- Create: `trainer/Dockerfile`
- Create: `trainer/docker/entrypoint.sh`
- Create: `trainer/scripts/build-image.sh`
- Create: `trainer/scripts/verify-image.sh`
- Create: `trainer/release-manifest.example.json`
- Create: `.github/workflows/groot-trainer-image.yml`
- Create: `trainer/tests/test_release_manifest.py`

**Steps:**

1. Write failing release-manifest tests for base digest, GR00T commit, model
   revision, dependency lock hash, repository commit, and final OCI digest.
2. Build on the pinned CUDA base, install with `uv sync --frozen`, clone and
   checkout the exact GR00T commit, install the local trainer, and run as a
   non-root user. Include no model, dataset, token, Isaac Sim, or Isaac Lab.
3. Make entrypoint refuse interactive Hugging Face login and create only
   `/cache` and `/output` state.
4. In CI, run CPU-safe tests and secret/large-file scans. On a GPU runner, verify
   exactly one GPU is visible, load a synthetic dataset, and execute one
   optimizer step.
5. Publish by immutable git-SHA tag, resolve the registry digest, and generate a
   release manifest. Do not use the image for paid training until this manifest
   is committed.
6. Run the fresh RTX PRO 6000 acceptance sequence: measure network, pull image,
   reach first optimizer step within 30 minutes at at least 1 Gbps, complete
   offline memorization, run batches 16/32/64 sequentially, and start or resume
   the selected 768,000-presentation training run.
7. Verify the complete local suite:

```bash
cd trainer
uv sync --frozen
uv run pytest tests -q
uv lock --check
docker buildx build --platform linux/amd64 --load -f Dockerfile ..
./scripts/verify-image.sh
```

Expected: all tests pass; the image contains no secrets or large artifacts; the
release manifest identifies the immutable image; Phase 1 ends with verified
training artifacts and a non-promotable offline memorization report ready for
the Phase 2 Isaac rollout adapter.

---

## Deferred Plans

Create separate reviewed plans before implementing:

1. Phase 2: GR00T-to-LeHome Isaac rollout adapter, simulator expert replay,
   fixed seen-development evaluation, checkpoint promotion, and one-time
   public-unseen audit.
2. Phase 3: model-independent success replay and targeted DAgger flywheel.

Do not start these phases, DAgger, public-unseen training, or full-backbone
fine-tuning as part of this plan.
