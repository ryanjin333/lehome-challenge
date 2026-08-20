# Nebius Experiment Worker Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provision and supervise two elastic training workers plus the existing rollout/evaluation appliance without multi-attaching the protected block disk or starting paid GPUs during implementation.

**Architecture:** Add a separate Terraform root for the CPU controller and two stopped-at-rest training workers. Each trainer uses independent local cache storage and a generic systemd worker that leases immutable jobs; the existing rollout Terraform root retains sole ownership of the protected 500 GiB disk.

**Tech Stack:** Terraform Nebius provider 0.6.42, Packer, cloud-init, systemd, Bash, Python worker client, pytest.

---

### Task 1: Add a dry-run-safe experiment-pool Terraform root

**Files:**
- Create: `infrastructure/nebius/terraform/experiment-pool/main.tf`
- Create: `infrastructure/nebius/terraform/experiment-pool/variables.tf`
- Create: `infrastructure/nebius/terraform/experiment-pool/outputs.tf`
- Create: `infrastructure/nebius/terraform/experiment-pool/example.tfvars`
- Test: `tests/infrastructure/test_experiment_pool_terraform.py`

- [ ] **Step 1: Write static Terraform contract tests**

Assert one CPU controller, exactly two training instances using `for_each`, all GPU VMs
`stopped = true`, preemptible stop behavior, 300 GiB independent boot/cache disks, no
reference to `computedisk-u00pbe55crxy7jr56x`, and no credential variables.

```python
def test_pool_never_attaches_protected_rollout_disk():
    source = POOL_MAIN.read_text()
    assert "computedisk-u00pbe55crxy7jr56x" not in source
    assert "secondary_disks" not in training_resource(source)
```

- [ ] **Step 2: Run RED tests**

```bash
uv run pytest -q tests/infrastructure/test_experiment_pool_terraform.py
```

- [ ] **Step 3: Implement the Terraform root**

Create:

- one small CPU controller VM with a 20 GiB protected boot disk;
- two `gpu-rtx6000` / `1gpu-24vcpu-218gb` training VMs;
- 300 GiB managed `NETWORK_SSD` boot/cache disk per training VM;
- `stopped = true`, `recovery_policy = "FAIL"`, and preemptible `STOP`;
- labels for role, worker slot, manifest-set SHA, and controller identity;
- private network interfaces and optional public IP controlled by a boolean defaulting
  false.

No Hugging Face token or controller bearer token enters Terraform state. Cloud-init writes
only non-secret controller URL, worker ID, cache path, and manifest-set digest.

- [ ] **Step 4: Validate without refreshing or applying**

```bash
terraform -chdir=infrastructure/nebius/terraform/experiment-pool fmt -check
terraform -chdir=infrastructure/nebius/terraform/experiment-pool init -backend=false
terraform -chdir=infrastructure/nebius/terraform/experiment-pool validate
```

Expected: formatting and validation pass; no cloud resource is created or refreshed.

- [ ] **Step 5: Commit Terraform root**

```bash
git add infrastructure/nebius/terraform/experiment-pool tests/infrastructure/test_experiment_pool_terraform.py
git commit -m "infra: define stopped three-gpu experiment pool"
```

### Task 2: Add the training worker client

**Files:**
- Create: `trainer/src/lehome_train/groot/experiment_worker.py`
- Create: `scripts/run_lehome_experiment_worker.py`
- Create: `trainer/tests/test_experiment_worker.py`

- [ ] **Step 1: Write worker lifecycle tests**

Test lease, heartbeat, immutable job hydration, exact resume checkpoint path, terminal
publication, retryable preemption, deterministic block, background publication, next-job
leasing, and ten-minute idle shutdown.

```python
def test_worker_leases_next_job_without_waiting_for_peer(fake_controller, runner):
    fake_controller.queue(job("a"), job("b"))
    worker = ExperimentWorker(fake_controller, runner=runner, idle_timeout_seconds=600)
    worker.run(max_jobs=2)
    assert runner.started == ["a", "b"]
```

- [ ] **Step 2: Run RED tests**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q trainer/tests/test_experiment_worker.py
```

- [ ] **Step 3: Implement a fail-closed worker**

The worker validates runtime paths, controller TLS/bearer configuration, local cache
ownership, GPU identity, image/code digest, and job manifest before invoking existing
`ProductionRuntime`. It starts a heartbeat thread before hydration and stops it after the
terminal controller transition.

Map errors explicitly:

```python
RETRYABLE = (PreemptionRequested, ControllerUnavailable, HubTransportError)
DETERMINISTIC = (ValueError, ManifestMismatch, LineageMismatch, UnsafePath)
```

Unknown exceptions report `retryable` with a bounded diagnostic code and terminate the
worker process; they do not fabricate a policy result.

- [ ] **Step 4: Connect terminal-rung publication**

For a 500-step terminal job, publish the terminal checkpoint immediately through the
existing authenticated archive/descriptor/readback path. For 1K and 2K, preserve the
existing recovery publication journals. Do not mark completion until the local terminal
receipt exists; do not mark publication verified until fresh Hub readback succeeds.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_experiment_worker.py \
  trainer/tests/test_production_runtime.py \
  trainer/tests/test_runtime_checkpoint_lifecycle.py
python3 -m py_compile trainer/src/lehome_train/groot/experiment_worker.py scripts/run_lehome_experiment_worker.py
git diff --check
git add trainer/src/lehome_train/groot/experiment_worker.py \
  scripts/run_lehome_experiment_worker.py trainer/tests/test_experiment_worker.py
git commit -m "feat: add asynchronous training worker"
```

Expected: focused worker and existing recovery tests pass.

### Task 3: Install and supervise the worker in the training image

**Files:**
- Create: `infrastructure/nebius/guest/bin/lehome-experiment-worker.sh`
- Create: `infrastructure/nebius/guest/systemd/lehome-experiment-worker.service`
- Modify: `infrastructure/nebius/packer/scripts/install-training.sh`
- Modify: `infrastructure/nebius/packer/training.pkr.hcl`
- Test: `tests/infrastructure/test_nebius_experiment_worker.py`
- Test: `tests/infrastructure/test_packer_contract.py`

- [ ] **Step 1: Write RED service/installer tests**

Assert the unit is enabled only in the training image, starts after network and Docker,
requires `/etc/lehome/experiment-worker.env` plus separate 0600 credential files, uses
`Restart=on-failure`, and calls the preemption control path on stop.

- [ ] **Step 2: Implement wrapper and unit**

The wrapper requires:

```text
LEHOME_CONTROLLER_URL
LEHOME_WORKER_ID
LEHOME_MANIFEST_SET_SHA256
LEHOME_CACHE_ROOT
LEHOME_OUTPUT_ROOT
LEHOME_CONTROLLER_TOKEN_FILE
HF_TOKEN_FILE
```

It rejects absent, symlinked, group/world-readable credential files and unsafe roots. It
executes the worker as the dedicated runtime user and never echoes environment values.

- [ ] **Step 3: Update training Packer installation**

Copy the worker script/module through the existing staged code bundle, install the unit,
and leave it inert until the non-secret environment file and secret files exist. Do not
enable the unit in the rollout image.

- [ ] **Step 4: Run infrastructure checks**

```bash
uv run pytest -q \
  tests/infrastructure/test_nebius_experiment_worker.py \
  tests/infrastructure/test_packer_contract.py \
  tests/infrastructure/test_nebius_training_guest.py
bash -n infrastructure/nebius/guest/bin/lehome-experiment-worker.sh
packer fmt -check infrastructure/nebius/packer
packer validate -syntax-only infrastructure/nebius/packer/training.pkr.hcl
git diff --check
```

Expected: all CPU-only checks pass; no image build occurs.

- [ ] **Step 5: Commit image integration**

```bash
git add infrastructure/nebius/guest/bin/lehome-experiment-worker.sh \
  infrastructure/nebius/guest/systemd/lehome-experiment-worker.service \
  infrastructure/nebius/packer/scripts/install-training.sh \
  infrastructure/nebius/packer/training.pkr.hcl \
  tests/infrastructure/test_nebius_experiment_worker.py \
  tests/infrastructure/test_packer_contract.py
git commit -m "infra: install experiment worker in training image"
```

### Task 4: Add CPU controller guest service

**Files:**
- Create: `infrastructure/nebius/guest/bin/lehome-experiment-controller.sh`
- Create: `infrastructure/nebius/guest/systemd/lehome-experiment-controller.service`
- Create: `infrastructure/nebius/packer/controller.pkr.hcl`
- Create: `infrastructure/nebius/packer/scripts/install-controller.sh`
- Test: `tests/infrastructure/test_nebius_experiment_controller.py`

- [ ] **Step 1: Write service hardening tests**

Require a dedicated user, `ProtectSystem=strict`, `PrivateTmp=true`, writable paths limited
to controller state/audit roots, token file mode 0600, private bind address, durable ready
receipt, and clean SQLite shutdown.

- [ ] **Step 2: Implement controller wrapper and unit**

The wrapper validates all paths before launching
`scripts/run_lehome_experiment_controller.py`. SIGTERM stops accepting leases, lets active
requests finish, checkpoints WAL, fsyncs the DB and parent directory, and writes a stopped
receipt.

- [ ] **Step 3: Add minimal controller image**

Base it on the existing CPU builder-compatible Ubuntu runtime. Install only Python,
controller code, CA certificates, systemd unit, and observability agent. Do not include
Isaac, GR00T model weights, Hugging Face data, or credentials.

- [ ] **Step 4: Run tests and commit**

```bash
uv run pytest -q tests/infrastructure/test_nebius_experiment_controller.py
bash -n infrastructure/nebius/guest/bin/lehome-experiment-controller.sh
packer fmt -check infrastructure/nebius/packer/controller.pkr.hcl
packer validate -syntax-only infrastructure/nebius/packer/controller.pkr.hcl
git diff --check
git add infrastructure/nebius/guest/bin/lehome-experiment-controller.sh \
  infrastructure/nebius/guest/systemd/lehome-experiment-controller.service \
  infrastructure/nebius/packer/controller.pkr.hcl \
  infrastructure/nebius/packer/scripts/install-controller.sh \
  tests/infrastructure/test_nebius_experiment_controller.py
git commit -m "infra: add experiment controller appliance"
```

### Task 5: Prove storage and concurrency safety without cloud mutation

**Files:**
- Modify: `infrastructure/nebius/validate.sh`
- Modify: `tests/infrastructure/test_terraform_contract.py`
- Create: `tests/infrastructure/test_experiment_pool_dry_run.py`

- [ ] **Step 1: Write a fake-provider plan test**

Prove the resolved desired topology has exactly three GPU VMs maximum, exactly two
training workers, one rollout VM reference, no duplicate runtime VM, and only the rollout
VM bound to the protected disk ID.

- [ ] **Step 2: Add pool validation to the repository validator**

Run fmt/validate and static contract tests for the new root. Do not run `terraform apply`,
`refresh`, Nebius CLI mutation, Packer build, or VM start.

- [ ] **Step 3: Run the complete infrastructure verification**

```bash
uv run pytest -q \
  tests/infrastructure/test_experiment_pool_terraform.py \
  tests/infrastructure/test_experiment_pool_dry_run.py \
  tests/infrastructure/test_terraform_contract.py \
  tests/infrastructure/test_nebius_experiment_worker.py \
  tests/infrastructure/test_nebius_experiment_controller.py
bash infrastructure/nebius/validate.sh
git diff --check
```

Expected: all tests pass and command logs contain no create/start/apply operation.

- [ ] **Step 4: Commit safety validation**

```bash
git add infrastructure/nebius/validate.sh \
  tests/infrastructure/test_terraform_contract.py \
  tests/infrastructure/test_experiment_pool_dry_run.py
git commit -m "test: prove experiment pool storage safety"
```
