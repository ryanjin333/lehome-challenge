# Nebius Preemptible Training and Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build locally verifiable Nebius templates for a reusable RTX PRO 6000 training appliance and a LeHome-specific four-worker rollout appliance, with one protected 500 GiB shared disk and complete recovery from preemption.

**Architecture:** Packer creates two immutable golden images through temporary CPU builders. Terraform creates exactly one preemptible `gpu-rtx6000` runtime role at a time and attaches the same protected network SSD. Training is driven by an immutable experiment manifest so the initial 70/30 mixture can later become 80/20 without rebuilding an image. Rollout uses one loaded policy behind a session-aware batching gateway, four persistent CUDA-cloth Isaac workers, an append-only controller ledger, and background artifact/Hugging Face publication.

**Tech Stack:** Python 3.11 for the Isaac/runtime project, the trainer project's pinned Python 3.10.18 for training control-plane tools, pytest, GR00T N1.7, LeRobot, Isaac Sim 5.1, ZeroMQ/msgpack, SQLite WAL, Hugging Face Hub, Docker, Packer with `github.com/nebius/nebius` `0.0.7`, Terraform with `nebius/nebius` `0.6.42`, systemd.

**Cost boundary:** Repository implementation and validation must not run `packer build`, `terraform apply`, or create Nebius resources. A later operator-approved image build uses a short-lived on-demand CPU builder because the current official Packer plugin exposes no preemptible builder setting. Both paid RTX PRO 6000 runtime roles remain preemptible.

---

## Target file structure

```text
infrastructure/nebius/
  README.md
  validate.sh
  tools/bootstrap.sh
  guest/
    lehome_workspace.py
    lehome_preempt.py
    systemd/
  packer/
    plugins.pkr.hcl
    variables.pkr.hcl
    training.pkr.hcl
    rollout.pkr.hcl
    scripts/
  terraform/
    modules/runtime-vm/
    storage/
    runtime/
      training.tfvars.example
      rollout.tfvars.example
rollout_appliance/
  Dockerfile
  entrypoint.sh
  challenge-artifact.json
source/lehome/lehome/flywheel/
  policy_protocol.py
  policy_batcher.py
  task_ledger.py
  artifact_queue.py
  persistent_worker.py
trainer/src/lehome_train/groot/
  experiment_manifest.py
  local_recovery.py
trainer/src/lehome_train/
  challenge_evaluation.py
scripts/
  run_groot_batched_policy_server.py
  run_groot_rollout_controller.py
  run_groot_persistent_worker.py
  run_groot_rollout_appliance.py
  publish_groot_challenge_evaluation.py
  select_groot_challenge_winner.py
docs/
  nebius_training_rollout.md
```

Existing legacy wave rollout and top-40 diagnostic entrypoints remain intact as rollback paths.

### Task 1: Add a canonical immutable experiment manifest

**Files:**
- Create: `trainer/src/lehome_train/groot/experiment_manifest.py`
- Create: `trainer/config/experiments/lehome-rft-70-30-v1.example.json`
- Test: `trainer/tests/test_experiment_manifest.py`

- [ ] **Step 1: Write failing canonicalization and admission tests**

Cover canonical JSON hashing, exact parent policy identity, dataset bundle and manifest hashes, train/validation lineage identities, held-out garment exclusions, mixture weights, horizon, batch, steps, checkpoint targets, and Hugging Face destinations. Assert that unknown keys, floating-point ratios, missing hashes, a non-64 batch, or a parent hash mismatch fail closed.

```python
def test_manifest_identity_changes_when_70_30_becomes_80_20() -> None:
    first = manifest(mixture={"bc": 70, "rollout": 30})
    second = manifest(mixture={"bc": 80, "rollout": 20})
    assert first.identity_sha256 != second.identity_sha256


def test_admission_requires_all_four_unseen_garments() -> None:
    with pytest.raises(ValueError, match="held-out garments"):
        load_experiment_manifest(manifest_without("Pant_Short_Unseen_1"))
```

- [ ] **Step 2: Run and confirm the module is missing**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_experiment_manifest.py
```

- [ ] **Step 3: Implement canonical JSON and typed validation**

Use integer weights rather than decimal fractions. Require the four exact held-out garments:

```text
Top_Long_Unseen_1
Top_Short_Unseen_1
Pant_Long_Unseen_1
Pant_Short_Unseen_1
```

Bind the manifest to the pinned step-12K repository, revision, subpath, archive SHA-256, artifact SHA-256, Isaac-GR00T revision, OCI digest, and dataset bundle hashes. Keep Hugging Face tokens out of the file.

- [ ] **Step 4: Run the focused tests and commit**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_experiment_manifest.py
git add trainer/src/lehome_train/groot/experiment_manifest.py \
  trainer/config/experiments/lehome-rft-70-30-v1.example.json \
  trainer/tests/test_experiment_manifest.py
git commit -m "Add immutable training experiment manifest"
```

### Task 2: Parameterize the runtime mixture without weakening lineage gates

**Files:**
- Modify: `trainer/src/lehome_train/groot/runtime_mixture.py`
- Modify: `trainer/src/lehome_train/groot/runtime_mixture_builder.py`
- Test: `trainer/tests/test_runtime_mixture.py`
- Test: `trainer/tests/test_runtime_mixture_builder.py`

- [ ] **Step 1: Write failing 70/30 and 80/20 quota tests**

Assert deterministic quotas at batch 64, deterministic in-memory `h=16` windows, no raw-lineage overlap between train and validation, and rejection of any sample from all four held-out evaluation garments.

```python
@pytest.mark.parametrize(
    ("weights", "expected"),
    [(({"bc": 70, "rollout": 30}), (45, 19)),
     (({"bc": 80, "rollout": 20}), (51, 13))],
)
def test_batch_64_uses_largest_remainder_quotas(weights, expected) -> None:
    assert source_quotas(64, weights) == {"bc": expected[0], "rollout": expected[1]}
```

The exact quota algorithm must be documented and stable; do not silently alternate algorithms between runs.

- [ ] **Step 2: Run the focused tests and observe the hard-coded 7/3 failure**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_runtime_mixture.py \
  trainer/tests/test_runtime_mixture_builder.py
```

- [ ] **Step 3: Derive weights from the admitted manifest**

Remove hard-coded 7/3 quotas. Preserve the existing strict lineage checks and unseen-source rejection. Include the experiment-manifest digest in every runtime dataset receipt.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_runtime_mixture.py \
  trainer/tests/test_runtime_mixture_builder.py
git add trainer/src/lehome_train/groot/runtime_mixture.py \
  trainer/src/lehome_train/groot/runtime_mixture_builder.py \
  trainer/tests/test_runtime_mixture.py \
  trainer/tests/test_runtime_mixture_builder.py
git commit -m "Parameterize runtime training mixtures"
```

### Task 3: Lock the RTX PRO 6000 loader pilot and training admission

**Files:**
- Modify: `trainer/src/lehome_train/groot/throughput_tuning.py`
- Modify: `trainer/src/lehome_train/groot/production_runtime.py`
- Modify: `trainer/src/lehome_train/groot/runtime_mixture_builder.py`
- Test: `trainer/tests/test_throughput_tuning.py`
- Test: `trainer/tests/test_production_runtime.py`

- [ ] **Step 1: Write failing fixed-candidate tests**

Require loader candidates exactly `(0, 4, 8, 12, 16)`, global and physical batch exactly 64, finite-loss checks, CUDA availability, video decode, batch materialization, VRAM headroom, and a deterministic winner receipt. No batch-size sweep is allowed for this campaign.

- [ ] **Step 2: Run focused tests**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_throughput_tuning.py \
  trainer/tests/test_production_runtime.py
```

- [ ] **Step 3: Unify all candidate lists and fail closed on drift**

Remove the existing `24` worker candidate and any `96/128` production batch promotion. The pilot records host, GPU, CUDA, decoder, samples/second, latency, loss, VRAM, and chosen worker count. The production launcher must consume the signed pilot receipt for the same manifest identity.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_throughput_tuning.py \
  trainer/tests/test_production_runtime.py
git add trainer/src/lehome_train/groot/throughput_tuning.py \
  trainer/src/lehome_train/groot/production_runtime.py \
  trainer/src/lehome_train/groot/runtime_mixture_builder.py \
  trainer/tests/test_throughput_tuning.py \
  trainer/tests/test_production_runtime.py
git commit -m "Lock RTX PRO 6000 training admission"
```

### Task 4: Add 500-step local recovery and 1,000-step Hugging Face recovery

**Files:**
- Create: `trainer/src/lehome_train/groot/local_recovery.py`
- Modify: `trainer/src/lehome_train/groot/config.py`
- Modify: `trainer/src/lehome_train/groot/continuous_training.py`
- Modify: `trainer/src/lehome_train/groot/runtime_checkpoint_lifecycle.py`
- Modify: `trainer/src/lehome_train/groot/production_runtime.py`
- Modify: `scripts/run_groot_persistent_training.py`
- Test: `trainer/tests/test_local_recovery.py`
- Test: `trainer/tests/test_continuous_training.py`
- Test: `trainer/tests/test_runtime_checkpoint_lifecycle.py`

- [ ] **Step 1: Write failing cadence, completeness, and identity tests**

Assert local checkpoints at `500, 1000, 1500, 2000`; immutable Hugging Face package/upload/readback only at `1000, 2000`; newest-complete-local resume; rejection of partial checkpoints; rejection when manifest, parent policy, optimizer, scheduler, RNG, or dataset identity differs.

- [ ] **Step 2: Run focused tests**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_local_recovery.py \
  trainer/tests/test_continuous_training.py \
  trainer/tests/test_runtime_checkpoint_lifecycle.py
```

- [ ] **Step 3: Implement atomic local completion receipts**

Launch the official trainer once with `max_steps=2000`, `save_steps=500`, and `save_total_limit=5`. A checkpoint becomes resumable only after its directory and identity sidecar are fsynced and atomically marked complete. The background publisher authenticates every local boundary but queues only steps 1000 and 2000 for immutable Hugging Face publication and download/readback verification.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_local_recovery.py \
  trainer/tests/test_continuous_training.py \
  trainer/tests/test_runtime_checkpoint_lifecycle.py
git add trainer/src/lehome_train/groot/local_recovery.py \
  trainer/src/lehome_train/groot/config.py \
  trainer/src/lehome_train/groot/continuous_training.py \
  trainer/src/lehome_train/groot/runtime_checkpoint_lifecycle.py \
  trainer/src/lehome_train/groot/production_runtime.py \
  scripts/run_groot_persistent_training.py \
  trainer/tests/test_local_recovery.py \
  trainer/tests/test_continuous_training.py \
  trainer/tests/test_runtime_checkpoint_lifecycle.py
git commit -m "Add preemptible training recovery cadence"
```

### Task 5: Implement the shared-workspace and preemption guest contract

**Files:**
- Create: `infrastructure/nebius/guest/lehome_workspace.py`
- Create: `infrastructure/nebius/guest/lehome_preempt.py`
- Create: `infrastructure/nebius/guest/systemd/lehome-workspace.service`
- Create: `infrastructure/nebius/guest/systemd/lehome-preempt.service`
- Test: `tests/infrastructure/test_nebius_workspace.py`
- Test: `tests/infrastructure/test_nebius_preempt.py`

- [ ] **Step 1: Write failing mount, role-lock, and SIGTERM tests**

Use a temporary fake block device and mount table. Assert mount at `/mnt/lehome`, exact filesystem UUID, deletion-safe refusal to format a nonblank disk, atomic `workspace-manifest.json`, one active role lease, and role/manifest mismatch rejection. Simulate the 60-second termination path with an injected clock and subprocesses.

- [ ] **Step 2: Run focused tests**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/infrastructure/test_nebius_workspace.py \
  tests/infrastructure/test_nebius_preempt.py
```

- [ ] **Step 3: Implement bounded shutdown**

On preemption: stop issuing leases, mark in-flight Isaac attempts retryable, flush SQLite WAL and append-only ledgers, close only terminal artifacts, request a safe training checkpoint if possible, write a lifecycle receipt, perform bounded Hugging Face sync, then stop. Never claim to resume opaque Isaac process memory.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/infrastructure/test_nebius_workspace.py \
  tests/infrastructure/test_nebius_preempt.py
git add infrastructure/nebius/guest
git add -f tests/infrastructure/test_nebius_workspace.py \
  tests/infrastructure/test_nebius_preempt.py
git commit -m "Add shared disk and preemption guest services"
```

### Task 6: Define the session-aware policy protocol

**Files:**
- Create: `source/lehome/lehome/flywheel/policy_protocol.py`
- Modify: `scripts/eval_policy/groot_policy.py`
- Modify: `trainer/pyproject.toml`
- Modify: `trainer/uv.lock`
- Test: `tests/flywheel/test_policy_protocol.py`
- Test: `tests/flywheel/test_policy_server.py`

- [ ] **Step 1: Write failing protocol and stale-response tests**

Every request and response must bind `schema_version`, `session_id`, `episode_generation`, `request_id`, `policy_sha256`, and deadline. Cover reset/cancel, duplicate request IDs, stale generation responses, timeouts, gateway restart, and local `h=16` action-chunk caching.

- [ ] **Step 2: Run focused tests**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/flywheel/test_policy_protocol.py \
  tests/flywheel/test_policy_server.py
```

- [ ] **Step 3: Add typed canonical envelopes and a compatible client**

Preserve the existing stock client path for legacy evaluation. Add a new session client for the appliance. Do not silently add session behavior to the legacy synchronous REP server. Add the root runtime's exact `msgpack==1.1.0` and `pyzmq==27.0.1` pins to the trainer development/test environment and refresh `trainer/uv.lock`, because these tests run with `uv run --project trainer` rather than the root Isaac environment.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/flywheel/test_policy_protocol.py \
  tests/flywheel/test_policy_server.py
git add source/lehome/lehome/flywheel/policy_protocol.py \
  scripts/eval_policy/groot_policy.py \
  trainer/pyproject.toml trainer/uv.lock
git add -f \
  tests/flywheel/test_policy_protocol.py \
  tests/flywheel/test_policy_server.py
git commit -m "Add session-aware policy protocol"
```

### Task 7: Build one batched policy server for four workers

**Files:**
- Create: `source/lehome/lehome/flywheel/policy_batcher.py`
- Create: `scripts/run_groot_batched_policy_server.py`
- Test: `tests/flywheel/test_policy_batcher.py`
- Test: `tests/flywheel/test_batched_policy_server.py`

- [ ] **Step 1: Write failing pure-batcher tests**

With a fake model, enqueue four sessions, batch nested image/state observations along the batch dimension, call the model once, and route each action chunk to its originating request. Cover partial batches, deadline flush, cancellation, malformed shapes, one slow client, and policy digest mismatch.

- [ ] **Step 2: Run focused tests**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/flywheel/test_policy_batcher.py \
  tests/flywheel/test_batched_policy_server.py
```

- [ ] **Step 3: Implement ROUTER-based serving and one model load**

Use ZeroMQ `ROUTER` so four clients can be outstanding concurrently. Separate socket I/O from the pure batcher. Load `Gr00tPolicy` exactly once, enforce one policy digest for the server lifetime, cap each inference batch at four observations, and expose readiness/metrics without a second model service.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/flywheel/test_policy_batcher.py \
  tests/flywheel/test_batched_policy_server.py
git add source/lehome/lehome/flywheel/policy_batcher.py \
  scripts/run_groot_batched_policy_server.py
git add -f \
  tests/flywheel/test_policy_batcher.py \
  tests/flywheel/test_batched_policy_server.py
git commit -m "Add four-session batched policy server"
```

### Task 8: Add the append-only rollout controller and retry leases

**Files:**
- Create: `source/lehome/lehome/flywheel/task_ledger.py`
- Create: `scripts/run_groot_rollout_controller.py`
- Test: `tests/flywheel/test_task_ledger.py`
- Test: `tests/flywheel/test_rollout_controller.py`

- [ ] **Step 1: Write failing SQLite transition tests**

Cover deterministic attempt IDs, fixed attempt matrix, `BEGIN IMMEDIATE`, WAL mode, append-only events, worker heartbeats, lease expiry/retry, terminal idempotency, no duplicate accepted artifact, max 400 attempts, target 150 accepted successes, and no wave barrier.

- [ ] **Step 2: Run focused tests**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/flywheel/test_task_ledger.py \
  tests/flywheel/test_rollout_controller.py
```

- [ ] **Step 3: Implement controller leases**

Separate immutable schedule sequence from the worker that happens to execute an attempt. A free worker immediately leases the next attempt. An expired nonterminal lease appends a retry event; it never rewrites history. Stop at 150 validated accepted episodes or 400 attempted episodes.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/flywheel/test_task_ledger.py \
  tests/flywheel/test_rollout_controller.py
git add source/lehome/lehome/flywheel/task_ledger.py \
  scripts/run_groot_rollout_controller.py
git add -f \
  tests/flywheel/test_task_ledger.py \
  tests/flywheel/test_rollout_controller.py
git commit -m "Add persistent rollout controller"
```

### Task 9: Run four persistent CUDA-cloth Isaac workers

**Files:**
- Create: `source/lehome/lehome/flywheel/persistent_worker.py`
- Create: `scripts/run_groot_persistent_worker.py`
- Modify: `scripts/utils/evaluation.py`
- Test: `tests/flywheel/test_persistent_worker.py`
- Test: `tests/test_evaluation_session.py`

- [ ] **Step 1: Write failing persistent-process tests**

Inject fake simulator, policy client, and controller adapters. Assert one simulator launch per worker process, unique worker/session/seed/buffer/output identities, episode-generation increments, garment switching without simulator relaunch, locally cached action chunks, retry of interrupted episodes, and immediate next lease after terminal write.

- [ ] **Step 2: Run focused tests**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/flywheel/test_persistent_worker.py \
  tests/test_evaluation_session.py
```

- [ ] **Step 3: Extract a reusable evaluation session**

Preserve the current `eval` CLI behavior while making one `AppLauncher`/environment reusable across assigned episodes. Bind cloth simulation, rendering, camera interop, and policy inference to the same canonical CUDA device and validate that binding in the worker receipt. Reject CPU cloth and mismatched device indices before an attempt is leased.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/flywheel/test_persistent_worker.py \
  tests/test_evaluation_session.py
git add source/lehome/lehome/flywheel/persistent_worker.py \
  scripts/run_groot_persistent_worker.py \
  scripts/utils/evaluation.py
git add -f \
  tests/flywheel/test_persistent_worker.py \
  tests/test_evaluation_session.py
git commit -m "Add persistent CUDA-cloth Isaac workers"
```

### Task 10: Finalize, validate, and publish episodes in the background

**Files:**
- Create: `source/lehome/lehome/flywheel/artifact_queue.py`
- Create: `source/lehome/lehome/flywheel/hub_sync.py`
- Modify: `trainer/src/lehome_train/flywheel/publish.py`
- Test: `tests/flywheel/test_artifact_queue.py`
- Test: `tests/flywheel/test_hub_sync.py`

- [ ] **Step 1: Write failing queue and idempotency tests**

Cover bounded queue backpressure, raw terminal episode handoff, schema/video/hash validation, atomic accepted/rejected ledger event, immutable Hugging Face path, upload retry, remote readback hash, duplicate sync receipt, and shutdown drain deadline.

- [ ] **Step 2: Run focused tests**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/flywheel/test_artifact_queue.py \
  tests/flywheel/test_hub_sync.py
```

- [ ] **Step 3: Reuse existing validation and publication contracts**

Workers write raw terminal episodes locally and immediately request new work. CPU/disk background services encode video, validate, hash, update the success/failure/hard-state ledger, and upload. An episode counts toward 150 only after validation; Hugging Face publication status is tracked separately and must be complete before round sealing.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/flywheel/test_artifact_queue.py \
  tests/flywheel/test_hub_sync.py
git add source/lehome/lehome/flywheel/artifact_queue.py \
  source/lehome/lehome/flywheel/hub_sync.py \
  trainer/src/lehome_train/flywheel/publish.py
git add -f \
  tests/flywheel/test_artifact_queue.py \
  tests/flywheel/test_hub_sync.py
git commit -m "Add background rollout artifact sync"
```

### Task 11: Supervise the complete four-worker appliance

**Files:**
- Create: `scripts/run_groot_rollout_appliance.py`
- Test: `tests/flywheel/test_rollout_appliance.py`

- [ ] **Step 1: Write failing process-topology tests**

With fake child commands, assert one policy server, one controller, exactly four workers by default, one writer/finalizer pool, one uploader, readiness ordering, distinct output roots, initial CPU allocation compatible with 24 vCPUs, restart limits, child failure propagation, and coordinated SIGTERM.

- [ ] **Step 2: Run the focused test**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/flywheel/test_rollout_appliance.py
```

- [ ] **Step 3: Implement the supervisor**

Start services only after shared-disk and manifest admission. Start the model server before workers. Use four workers by default; any lower count requires an explicit debug flag and is recorded. Reserve CPU for the controller/writer/uploader and assign worker affinity without pretending 24 vCPUs provides the originally preferred 48-core topology.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/flywheel/test_rollout_appliance.py
git add scripts/run_groot_rollout_appliance.py
git add -f tests/flywheel/test_rollout_appliance.py
git commit -m "Supervise the LeHome rollout appliance"
```

### Task 12: Add challenge-complete evaluation and winner selection

**Files:**
- Create: `trainer/src/lehome_train/challenge_evaluation.py`
- Create: `scripts/publish_groot_challenge_evaluation.py`
- Create: `scripts/select_groot_challenge_winner.py`
- Test: `trainer/tests/test_challenge_evaluation.py`
- Test: `trainer/tests/test_challenge_winner.py`

- [ ] **Step 1: Write failing category and gate tests**

Require five candidates: original baseline, previous step-1K, previous step-2K, new step-1K, and new step-2K. Each uses the same 80 public-unseen episodes from `configs/eval_groot_n17_public_280.json`: 20 for top-long, top-short, pant-long, and pant-short. Gate at least 56/80 overall and 12/20 in every category.

Also require fixed seen-dev evaluation for all candidates, the full 200 seen episodes for the proposed winner versus baseline, no major safety regression, complete provenance, and predeclared deterministic tie-breakers.

- [ ] **Step 2: Run focused tests**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_challenge_evaluation.py \
  trainer/tests/test_challenge_winner.py
```

- [ ] **Step 3: Implement immutable reports and promotion receipts**

Keep `publish_groot_checkpoint_evaluation.py` and the legacy top-40 lifecycle unchanged as diagnostics. The new selector refuses physical-test promotion unless every gate passes. If no candidate passes but one improves, emit a next-round rollout manifest pointing to that immutable winner, capped at 400 attempts and 150 accepted episodes.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_challenge_evaluation.py \
  trainer/tests/test_challenge_winner.py
git add trainer/src/lehome_train/challenge_evaluation.py \
  scripts/publish_groot_challenge_evaluation.py \
  scripts/select_groot_challenge_winner.py \
  trainer/tests/test_challenge_evaluation.py \
  trainer/tests/test_challenge_winner.py
git commit -m "Add complete LeHome winner gate"
```

### Task 13: Build the LeHome-derived rollout container layer

**Files:**
- Create: `rollout_appliance/Dockerfile`
- Create: `rollout_appliance/entrypoint.sh`
- Create: `rollout_appliance/challenge-artifact.json`
- Create: `tests/infrastructure/test_rollout_container.py`

- [ ] **Step 1: Write failing static contract tests**

Verify the artifact manifest contains repository `lehome/docker`, revision `a914115729bb0bfd260971b9c8d4147bff38c1fb`, exact size `26676771349`, and SHA-256 `1a85e389962909debc4ee9988d8a8c388f905fba60686ef78b1623e6872f7123`. Verify the Dockerfile derives from the loaded official `lehome-challenge` image, copies only runtime code, runs non-secret entrypoint checks, and defaults to the appliance supervisor.

- [ ] **Step 2: Run the static test**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/infrastructure/test_rollout_container.py
```

- [ ] **Step 3: Implement the thin derived layer**

Do not duplicate the 26.7 GB tarball in the repository. Packer downloads the exact LFS object, checks byte length and SHA-256 before `docker load`, then builds this layer. Adapt the companion `dummy_docker_policy` HTTP/container contract where the challenge needs it, but keep the four-worker gateway internal and session-aware.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/infrastructure/test_rollout_container.py
git add rollout_appliance
git add -f tests/infrastructure/test_rollout_container.py
git commit -m "Add LeHome rollout container layer"
```

### Task 14: Create the two Packer golden-image templates

**Files:**
- Create: `infrastructure/nebius/packer/plugins.pkr.hcl`
- Create: `infrastructure/nebius/packer/variables.pkr.hcl`
- Create: `infrastructure/nebius/packer/training.pkr.hcl`
- Create: `infrastructure/nebius/packer/rollout.pkr.hcl`
- Create: `infrastructure/nebius/packer/scripts/install-common.sh`
- Create: `infrastructure/nebius/packer/scripts/install-training.sh`
- Create: `infrastructure/nebius/packer/scripts/install-rollout.sh`
- Test: `tests/infrastructure/test_packer_contract.py`

- [ ] **Step 1: Write failing template-contract tests**

Assert plugin source/version, CPU builder (`cpu-d3`, `16vcpu-64gb`), Ubuntu 24.04 driverless base, no baked secrets, unique training/rollout image names, sufficient rollout boot disk headroom, exact training OCI digest and code revision, exact LeHome tarball verification before use, service installation, and cleanup of downloaded tarball/build cache before image capture.

- [ ] **Step 2: Run static tests**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/infrastructure/test_packer_contract.py
```

- [ ] **Step 3: Implement reproducible builders**

Pin `github.com/nebius/nebius` to `= 0.0.7`. The portable training image must not load the LeHome tarball or bake any policy/dataset. The rollout image must load and derive from the verified challenge tarball. Install guest services in both. Document that the temporary builder is on-demand CPU due to plugin schema limitations; do not mislabel it preemptible.

- [ ] **Step 4: Run free Packer validation and commit**

```bash
infrastructure/nebius/tools/bootstrap.sh
infrastructure/nebius/.tools/packer init infrastructure/nebius/packer
infrastructure/nebius/.tools/packer fmt -check infrastructure/nebius/packer
infrastructure/nebius/.tools/packer validate \
  -var 'project_id=test-project' \
  -var 'subnet_id=test-subnet' \
  infrastructure/nebius/packer/training.pkr.hcl
infrastructure/nebius/.tools/packer validate \
  -var 'project_id=test-project' \
  -var 'subnet_id=test-subnet' \
  infrastructure/nebius/packer/rollout.pkr.hcl
git add infrastructure/nebius/packer
git add -f tests/infrastructure/test_packer_contract.py
git commit -m "Add Nebius Packer golden images"
```

### Task 15: Create protected storage and one-role-at-a-time Terraform

**Files:**
- Create: `infrastructure/nebius/terraform/modules/runtime-vm/main.tf`
- Create: `infrastructure/nebius/terraform/modules/runtime-vm/variables.tf`
- Create: `infrastructure/nebius/terraform/modules/runtime-vm/outputs.tf`
- Create: `infrastructure/nebius/terraform/storage/main.tf`
- Create: `infrastructure/nebius/terraform/storage/variables.tf`
- Create: `infrastructure/nebius/terraform/storage/outputs.tf`
- Create: `infrastructure/nebius/terraform/runtime/main.tf`
- Create: `infrastructure/nebius/terraform/runtime/variables.tf`
- Create: `infrastructure/nebius/terraform/runtime/outputs.tf`
- Create: `infrastructure/nebius/terraform/runtime/training.tfvars.example`
- Create: `infrastructure/nebius/terraform/runtime/rollout.tfvars.example`
- Test: `tests/infrastructure/test_terraform_contract.py`

- [ ] **Step 1: Write failing Terraform contract tests**

Require provider `nebius/nebius` exactly `0.6.42`; standalone 500 GiB `NETWORK_SSD`; `forbid_deletion=true`; Terraform `prevent_destroy`; runtime platform `gpu-rtx6000`; preset `1gpu-24vcpu-218gb`; `recovery_policy="FAIL"`; `preemptible = { on_preemption = "STOP" }`; disposable custom-image boot disk; one existing secondary disk attached `READ_WRITE` with stable `device_id`; and no secrets or model hyperparameters in state.

- [ ] **Step 2: Run static tests**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/infrastructure/test_terraform_contract.py
```

- [ ] **Step 3: Implement separate storage state and one runtime state**

Use one runtime root selected by an `active_role` variable rather than independent training and rollout states that could both try to attach the disk. The role tfvars differ in image ID, service name, and immutable manifest URI/digest. A role switch is an explicit stop/detach/attach/apply sequence. Runtime destroy must not touch storage.

- [ ] **Step 4: Run free Terraform validation and commit**

```bash
infrastructure/nebius/tools/bootstrap.sh
for root in infrastructure/nebius/terraform/storage infrastructure/nebius/terraform/runtime; do
  infrastructure/nebius/.tools/terraform -chdir="$root" init -backend=false
  infrastructure/nebius/.tools/terraform -chdir="$root" fmt -check -recursive
  infrastructure/nebius/.tools/terraform -chdir="$root" validate
done
git add infrastructure/nebius/terraform
git add -f tests/infrastructure/test_terraform_contract.py
git commit -m "Add protected Nebius runtime infrastructure"
```

### Task 16: Add pinned local tooling and repository-wide free validation

**Files:**
- Modify: `.gitignore`
- Create: `infrastructure/nebius/tools/bootstrap.sh`
- Create: `infrastructure/nebius/validate.sh`
- Create: `tests/infrastructure/test_tool_bootstrap.py`

- [ ] **Step 1: Write failing checksum and no-paid-command tests**

Assert the bootstrap downloads pinned platform-specific Terraform/Packer releases to ignored `infrastructure/nebius/.tools/`, verifies published SHA-256 checksums, and never invokes `terraform apply`, `packer build`, or Nebius resource-creation commands. Assert `.terraform/`, local tool binaries, tfstate, and secret tfvars are ignored while lockfiles remain trackable.

- [ ] **Step 2: Run the focused test**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  tests/infrastructure/test_tool_bootstrap.py
```

- [ ] **Step 3: Implement one free validation entrypoint**

`validate.sh` runs Python infrastructure tests, Terraform fmt/init-without-backend/validate, Packer init/fmt/validate, shell syntax checks, Dockerfile static checks, and secret-pattern scans. It prints paid next commands but never executes them.

- [ ] **Step 4: Run and commit**

```bash
infrastructure/nebius/validate.sh
git add .gitignore infrastructure/nebius/tools/bootstrap.sh \
  infrastructure/nebius/validate.sh
git add -f tests/infrastructure/test_tool_bootstrap.py
git commit -m "Add free Nebius template validation"
```

### Task 17: Document operations and run the final repository gate

**Files:**
- Create: `infrastructure/nebius/README.md`
- Create: `docs/nebius_training_rollout.md`
- Modify: `README.md`

- [ ] **Step 1: Write the operator runbook**

Document:

1. local/free validation;
2. explicit Packer image builds and expected temporary CPU-builder cost boundary;
3. creation of the protected 500 GiB disk;
4. launch and smoke test of the preemptible RTX PRO 6000 training role;
5. immutable download/hash verification of parent model and data bundles;
6. loader pilot, batch-64 2K training, local/HF recovery behavior;
7. safe role handoff from training to rollout;
8. four-worker CUDA-cloth rollout startup, monitoring, preemption, and restart;
9. immutable round sealing and Hugging Face readback;
10. five-candidate 80-unseen/200-seen evaluation and winner gate;
11. physical-test approval boundary; and
12. deletion procedure that requires removing `prevent_destroy` only through a separately reviewed change.

Include an explicit secrets table: Nebius credentials, Hugging Face token, and private repository access are injected at runtime and are never baked into images, committed, or passed through Terraform variables/state.

- [ ] **Step 2: Run the targeted Python suite**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_experiment_manifest.py \
  trainer/tests/test_runtime_mixture.py \
  trainer/tests/test_runtime_mixture_builder.py \
  trainer/tests/test_throughput_tuning.py \
  trainer/tests/test_production_runtime.py \
  trainer/tests/test_local_recovery.py \
  trainer/tests/test_continuous_training.py \
  trainer/tests/test_runtime_checkpoint_lifecycle.py \
  trainer/tests/test_challenge_evaluation.py \
  trainer/tests/test_challenge_winner.py \
  tests/flywheel/test_policy_protocol.py \
  tests/flywheel/test_policy_server.py \
  tests/flywheel/test_policy_batcher.py \
  tests/flywheel/test_batched_policy_server.py \
  tests/flywheel/test_task_ledger.py \
  tests/flywheel/test_rollout_controller.py \
  tests/flywheel/test_persistent_worker.py \
  tests/test_evaluation_session.py \
  tests/flywheel/test_artifact_queue.py \
  tests/flywheel/test_hub_sync.py \
  tests/flywheel/test_rollout_appliance.py \
  tests/infrastructure
```

- [ ] **Step 3: Run infrastructure validation and structural checks**

```bash
infrastructure/nebius/validate.sh
git diff --check
git status --short
```

Expected: all local tests and static validation pass; no paid resources exist; no secrets or generated state are tracked.

- [ ] **Step 4: Run outside review for the substantial diff**

```bash
/Users/user/.codex/bin/auto-review-claude --uncommitted
```

Classify findings as `ACCEPT`, `REJECT`, `NEEDS CLARIFICATION`, or `NIT`. Implement only accepted findings, rerun the affected checks, and record intentionally skipped paid smoke tests.

- [ ] **Step 5: Commit documentation**

```bash
git add infrastructure/nebius/README.md docs/nebius_training_rollout.md README.md
git commit -m "Document Nebius training and rollout operations"
```

---

## Paid validation gates after repository implementation

These gates are intentionally not executed by this plan without new user approval:

1. Build `vla-training-base` through its temporary on-demand CPU Packer builder.
2. Build `lehome-rollout` through its temporary on-demand CPU Packer builder and confirm final image footprint.
3. Create the protected 500 GiB network SSD.
4. Launch one preemptible RTX PRO 6000 training VM and prove CUDA, video decode, batch creation, loader selection, checkpoint/resume, and Hugging Face readback.
5. Stop/detach training and launch one preemptible RTX PRO 6000 rollout VM.
6. Prove CUDA cloth, four persistent workers, one model load, correct session routing, stable VRAM, acceptable policy latency, terminal artifact publication, and restart after forced interruption.

Production training and rollout begin only after their respective paid smoke gates pass.
