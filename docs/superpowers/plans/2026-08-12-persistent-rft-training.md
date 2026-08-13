# Persistent RFT Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one immutable 70/30 corrective generation continuously to steps 1,000 and 2,000 at the verified global batch 64 on one measured RTX PRO 6000, while packaging and publishing checkpoints asynchronously.

**Architecture:** Add a host-throughput tuner for loader workers plus exploratory batch probes, a generation freeze receipt, one continuous official GR00T process with two checkpoint completion events, and a bounded background checkpoint publisher. The first production run keeps global batch 64 so steps 1,000/2,000 retain the agreed sample exposure; batch 96/128 results are recorded but cannot silently alter this run. Training-only Blackwell hosts use a minimum-driver capability smoke rather than the rollout R580 ceiling.

**Tech Stack:** Python 3.11, PyTorch/CUDA 12.8, GR00T N1.7, Hugging Face Hub, NVML, pytest.

---

## File structure

- Create `trainer/src/lehome_train/groot/throughput_tuning.py`: deterministic loader/batch benchmark selection.
- Create `trainer/src/lehome_train/groot/continuous_training.py`: one-process 1K/2K observer and background packager.
- Modify `trainer/src/lehome_train/groot/launch.py`: expose a continuous launch without checkpoint-chunk relaunch.
- Modify `trainer/src/lehome_train/groot/production_adapters.py`: package completed checkpoints from immutable snapshots.
- Modify `trainer/src/lehome_train/commands/train.py`: add generation-frozen continuous mode while preserving legacy fixed-exposure mode.
- Modify `trainer/src/lehome_train/groot/production_runtime.py`: add tune and continuous-train actions.
- Modify `trainer/src/lehome_train/release_manifest.py`: separate training driver capability from rollout R580 policy.
- Create `scripts/run_groot_persistent_training.py`: training rental lifecycle and disposal gate.
- Create `trainer/tests/test_throughput_tuning.py`, `test_continuous_training.py`, and `test_persistent_training_lifecycle.py`.
- Extend `test_groot_launch.py`, `test_production_adapters.py`, `test_train.py`, `test_production_runtime.py`, and `test_release_manifest.py`.

### Task 1: Freeze one training generation

**Files:**
- Modify: `trainer/src/lehome_train/flywheel/mix.py`
- Modify: `trainer/tests/test_flywheel_mix.py`

- [ ] **Step 1: Write moving-dataset and exact-mix tests**

```python
def test_generation_receipt_binds_exact_70_30_mix_and_artifacts(tmp_path: Path) -> None:
    result = materialize_mix(mix_plan(tmp_path), tmp_path / "generation")
    receipt = load_generation_receipt(result.root)
    assert receipt["organizer_training_frames"] * 3 == receipt["rft_training_frames"] * 7
    assert receipt["sealed"] is True
    assert len(receipt["output_manifest_sha256"]) == 64


def test_generation_changes_after_seal_are_rejected(tmp_path: Path) -> None:
    root = sealed_generation(tmp_path)
    mutate_manifest_listed_file(root)
    with pytest.raises(ValueError, match="sealed generation"):
        verify_generation(root)
```

- [ ] **Step 2: Run and confirm the seal API is missing**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_flywheel_mix.py
```

- [ ] **Step 3: Add a canonical sealed-generation receipt**

The receipt includes source revisions, mix-plan SHA-256, exact organizer/RFT
training frame counts, split seed, raw-lineage hashes, output artifact manifest
hash, statistics hash, and `sealed: true`. `verify_generation()` recomputes all
listed hashes and rejects symlinks or extra files. Because the prepared
snapshot already contains a manifest, write the generation receipt as a sibling
`<destination>.generation.json`; do not add an unlisted file inside the sealed
dataset after `artifact_identities()` and validation have run.

- [ ] **Step 4: Run mixture tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/src/lehome_train/flywheel/mix.py \
  trainer/tests/test_flywheel_mix.py
git commit -m "Seal immutable RFT training generations"
```

### Task 2: Measure loader workers and physical batch

**Files:**
- Create: `trainer/src/lehome_train/groot/throughput_tuning.py`
- Test: `trainer/tests/test_throughput_tuning.py`

- [ ] **Step 1: Write deterministic selection tests**

```python
def test_tuner_selects_loader_but_keeps_first_run_at_verified_batch() -> None:
    report = tune_training(
        loader_results=[loader(4, 10), loader(8, 14), loader(12, 13)],
        batch_results=[batch(64, 640), batch(96, 800), batch(128, 780)],
    )
    assert report.selected_loader_workers == 8
    assert report.fastest_stable_physical_batch == 96
    assert report.production_physical_batch == 64


def test_tuner_never_tries_larger_batch_after_proven_oom() -> None:
    attempted = []
    tune_on_host(run=lambda workers, batch: attempted.append(batch) or outcome(batch))
    assert attempted == [64, 96]
```

- [ ] **Step 2: Run and confirm missing module failure**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_throughput_tuning.py
```

- [ ] **Step 3: Implement the bounded benchmark**

```python
LOADER_CANDIDATES = (4, 8, 12)
BATCH_CANDIDATES = (64, 96, 128)
STEADY_STEPS = 100

def select_candidate(results: Sequence[TrainingProbe]) -> TrainingProbe:
    admitted = [
        item for item in results
        if item.finite_loss and item.stable and item.free_vram_percent >= 10.0
    ]
    if not admitted:
        raise NoStableTrainingCandidate("no stable candidate with 10% VRAM headroom")
    return max(admitted, key=lambda item: (item.samples_per_second, -item.hourly_cost, item.free_vram_percent))
```

Measure loaders first at batch 64, then batches using the selected loader count.
Persist every outcome including OOM. Do not run 128 after a proven 96 OOM. The
receipt separately records `fastest_stable_physical_batch` and the admitted
`production_physical_batch=64`; a later generation may promote 96/128 only with
an explicit learning-rate and sample-exposure plan.

- [ ] **Step 4: Run tuning tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/src/lehome_train/groot/throughput_tuning.py \
  trainer/tests/test_throughput_tuning.py
git commit -m "Tune GR00T training throughput"
```

### Task 3: Add a continuous official GR00T launch

**Files:**
- Modify: `trainer/src/lehome_train/groot/launch.py`
- Modify: `trainer/tests/test_groot_launch.py`

- [ ] **Step 1: Write one-process launch tests**

```python
def test_continuous_launch_runs_one_process_to_2000_with_save_1000(
    tmp_path: Path, official_checkout: Path
) -> None:
    calls = []
    launch_continuous_finetune(
        config(max_steps=2000, save_steps=1000, physical_batch_size=64),
        visible_devices="0",
        environment={},
        official_checkout=official_checkout,
        runner=lambda *args, **kwargs: calls.append((args, kwargs)) or CompletedProcess([], 0),
    )
    assert len(calls) == 1
    command = calls[0][0][0]
    assert command[command.index("--max-steps") + 1] == "2000"
    assert command[command.index("--save-steps") + 1] == "1000"
```

- [ ] **Step 2: Run and confirm missing launch function**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_groot_launch.py
```

- [ ] **Step 3: Implement continuous launch as a narrow wrapper**

```python
def launch_continuous_finetune(config: FineTuneLaunchConfig, **kwargs) -> CompletedProcess:
    if config.num_gpus != 1:
        raise ValueError("continuous corrective training requires one GPU")
    if config.global_batch_size != 64 or config.physical_batch_size != 64:
        raise ValueError("first continuous corrective run requires global batch 64")
    if config.max_steps != 2000 or config.save_steps != 1000:
        raise ValueError("continuous corrective training requires 1000/2000 checkpoints")
    return launch_finetune(config, **kwargs)
```

Do not call `launch_finetune_to_step`; the official process owns both saves.

- [ ] **Step 4: Run launch tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/src/lehome_train/groot/launch.py \
  trainer/tests/test_groot_launch.py
git commit -m "Launch continuous corrective GR00T training"
```

### Task 4: Observe and snapshot completed checkpoints

**Files:**
- Create: `trainer/src/lehome_train/groot/continuous_training.py`
- Modify: `trainer/src/lehome_train/groot/production_adapters.py`
- Test: `trainer/tests/test_continuous_training.py`
- Test: `trainer/tests/test_production_adapters.py`

- [ ] **Step 1: Write overlap and incomplete-directory tests**

```python
def test_step_1000_packages_while_training_continues_to_2000(tmp_path: Path) -> None:
    events = run_fake_continuous_training(tmp_path)
    assert events.index("train-step-1500") < events.index("upload-step-1000-finished")
    assert events.index("package-step-1000-started") < events.index("train-step-2000")


def test_observer_never_packages_checkpoint_without_completion_marker(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="complete checkpoint"):
        snapshot_checkpoint(checkpoint, optimizer_step=1000)
```

- [ ] **Step 2: Run and confirm missing module failure**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_continuous_training.py \
  trainer/tests/test_production_adapters.py
```

- [ ] **Step 3: Implement observer plus immutable local snapshot**

The observer watches only steps 1000 and 2000. It requires the upstream files
validated by `_verified_checkpoint_state_at`, then creates a reflink snapshot
when supported, falling back to a verified byte copy. Never use hardlinks:
later source writes would mutate the supposed snapshot through the shared inode.
It hashes the snapshot before submitting it to one bounded packaging worker.

```python
@dataclass(frozen=True, slots=True)
class CompletedCheckpoint:
    optimizer_step: int
    source_sha256: str
    snapshot_root: Path
    observed_at_unix: int
```

The training thread never reads or uploads Hub credentials. The publisher
thread uses the existing explicit-token `HubCheckpointUploader`.

- [ ] **Step 4: Run continuous/checkpoint tests**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_continuous_training.py \
  trainer/tests/test_production_adapters.py \
  trainer/tests/test_checkpoints.py
```

- [ ] **Step 5: Commit**

```bash
git add trainer/src/lehome_train/groot/continuous_training.py \
  trainer/src/lehome_train/groot/production_adapters.py \
  trainer/tests/test_continuous_training.py \
  trainer/tests/test_production_adapters.py
git commit -m "Publish checkpoints behind continuous training"
```

### Task 5: Add generation-frozen continuous training command

**Files:**
- Modify: `trainer/src/lehome_train/commands/train.py`
- Modify: `trainer/src/lehome_train/groot/production_runtime.py`
- Modify: `trainer/tests/test_train.py`
- Modify: `trainer/tests/test_production_runtime.py`

- [ ] **Step 1: Write immutable-input and terminal tests**

```python
def test_continuous_training_requires_sealed_generation(tmp_path: Path) -> None:
    request = continuous_request(tmp_path, sealed=False)
    with pytest.raises(ValueError, match="sealed generation"):
        run_continuous_training(request)


def test_continuous_training_finishes_only_after_both_readbacks(tmp_path: Path) -> None:
    result = run_fake_continuous_command(tmp_path, verified_steps=(1000, 2000))
    assert result["status"] == "completed"
    assert result["immutable_checkpoint_steps"] == [1000, 2000]
    assert result["disposable"] is True
```

- [ ] **Step 2: Run and confirm command is absent**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_train.py trainer/tests/test_production_runtime.py
```

- [ ] **Step 3: Implement `continuous-train` without changing legacy train**

Validate the sealed generation, horizon receipt, selected tuning receipt, parent
checkpoint identity, normalization, 70/30 counts, and zero unseen sources before
launch. Start the observer, run one official process, finish both publications,
then emit:

```python
{
    "schema_version": 1,
    "kind": "continuous_corrective_training_terminal",
    "generation_sha256": generation.sha256,
    "parent_checkpoint_sha256": parent.sha256,
    "optimizer_steps": 2000,
    "global_batch_size": 64,
    "sample_presentations": 128000,
    "checkpoint_steps": [1000, 2000],
    "immutable_checkpoint_steps": [1000, 2000],
    "disposable": True,
}
```

An interrupt or failed upload returns `disposable: false` and preserves the last
verified resumable checkpoint. This command intentionally does not reuse the
legacy `TOTAL_SAMPLE_PRESENTATIONS=768000` runtime action; the corrective run is
an explicit 128,000-presentation, batch-64 schedule.

- [ ] **Step 4: Run train/runtime tests**

Run the Step 2 command plus `trainer/tests/test_continuous_training.py`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/src/lehome_train/commands/train.py \
  trainer/src/lehome_train/groot/production_runtime.py \
  trainer/tests/test_train.py trainer/tests/test_production_runtime.py
git commit -m "Run frozen generation training continuously"
```

### Task 6: Separate Blackwell training driver admission

**Files:**
- Modify: `trainer/src/lehome_train/release_manifest.py`
- Modify: `trainer/tests/test_release_manifest.py`

- [ ] **Step 1: Write newer-driver training tests**

```python
def test_training_accepts_newer_blackwell_driver_after_capability_smoke() -> None:
    manifest = accepted_manifest(
        hardware="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        driver_version="595.71.05",
        cuda_optimizer_smoke={"passed": True, "image_digest": approved_digest()},
    )
    assert manifest.gpu_acceptance["passed"] is True


def test_training_rejects_newer_driver_without_real_smoke() -> None:
    with pytest.raises(ValueError, match="optimizer smoke"):
        accepted_manifest(driver_version="595.71.05", cuda_optimizer_smoke=None)
```

- [ ] **Step 2: Run and confirm schema failure**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_release_manifest.py
```

- [ ] **Step 3: Add a training-only capability tuple**

Require GPU name, driver version, exact OCI digest, CUDA runtime, Torch CUDA
version, compute capability, finite optimizer-step receipt, and NVML telemetry.
Do not import or reuse `_is_approved_r580`; that remains rollout-only.

- [ ] **Step 4: Run release tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/src/lehome_train/release_manifest.py \
  trainer/tests/test_release_manifest.py
git commit -m "Admit newer Blackwell training drivers by capability"
```

### Task 7: Training rental lifecycle

**Files:**
- Create: `scripts/run_groot_persistent_training.py`
- Create: `trainer/tests/test_persistent_training_lifecycle.py`

- [ ] **Step 1: Write dry-run, budget, interrupt, and disposal tests**

```python
def test_dry_run_never_calls_provider(tmp_path: Path) -> None:
    report = main_for_test(["prepare", "--request", request(tmp_path)], provider=FailProvider())
    assert report["paid_action"] is False


def test_destroy_requires_two_immutable_checkpoints_bound_to_instance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="instance-bound disposal"):
        destroy(instance_id=7, training_receipt=receipt(tmp_path, steps=[1000]))
```

- [ ] **Step 2: Run and confirm missing lifecycle failure**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_persistent_training_lifecycle.py
```

- [ ] **Step 3: Implement explicit actions**

Provide `prepare`, `capture-offers`, `rent`, `stage`, `tune`, `train`, `status`,
`resume`, and `destroy`. Rent one interruptible RTX PRO 6000 under $1/hour only
when the fresh account-wide total including storage is at most $2/hour. Stage
the exact sealed generation and parent checkpoint once. `resume` requires the
same generation/config identities. `destroy` requires immutable 1000/2000
readbacks and exact instance binding.

- [ ] **Step 4: Run lifecycle and training suites**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_persistent_training_lifecycle.py \
  trainer/tests/test_throughput_tuning.py \
  trainer/tests/test_continuous_training.py \
  trainer/tests/test_train.py \
  trainer/tests/test_production_runtime.py \
  trainer/tests/test_release_manifest.py
python3 -m py_compile scripts/run_groot_persistent_training.py
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_groot_persistent_training.py \
  trainer/tests/test_persistent_training_lifecycle.py
git commit -m "Add persistent corrective training lifecycle"
```

### Task 8: Free acceptance and paid training gate

**Files:**
- Create: `docs/groot_persistent_training.md`
- Verify all modified files.

- [ ] **Step 1: Run the complete free acceptance**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_horizon_gate.py \
  trainer/tests/test_flywheel_mix.py \
  trainer/tests/test_throughput_tuning.py \
  trainer/tests/test_continuous_training.py \
  trainer/tests/test_groot_launch.py \
  trainer/tests/test_production_adapters.py \
  trainer/tests/test_train.py \
  trainer/tests/test_production_runtime.py \
  trainer/tests/test_release_manifest.py \
  trainer/tests/test_persistent_training_lifecycle.py
python3 -m py_compile \
  trainer/src/lehome_train/groot/horizon_gate.py \
  trainer/src/lehome_train/groot/throughput_tuning.py \
  trainer/src/lehome_train/groot/continuous_training.py \
  scripts/run_groot_persistent_training.py
git diff --check
```

Expected: PASS.

- [ ] **Step 2: Document exact operational commands and rollback**

Document generation verification, newer-driver capability smoke, tuning,
continuous train, status/resume, immutable readback, evaluation handoff, and
destroy. State that legacy chunked training remains the rollback path.

- [ ] **Step 3: Commit documentation**

```bash
git add docs/groot_persistent_training.md
git commit -m "Document persistent corrective training"
```

- [ ] **Step 4: Run one paid trainer acceptance only after rollout publication**

On one interruptible RTX PRO 6000 under $1/hour, run the real horizon loader/loss
gate, loader/batch tuning, and continuous 2,000-step process at global batch 64.
Record samples/s,
GPU utilization, dataloader wait, checkpoint pause, upload overlap, total time,
and total cost. Destroy only after both checkpoints have immutable fresh
readbacks.

- [ ] **Step 5: Evaluate before promotion**

Evaluate the original step-12000 baseline and new steps 1000/2000 on identical
untouched matrices. Promote only at at least 70% overall, at least 60% per
required category, no major safety issue, and no major seen regression. A
regressed result does not become the next rollout policy.
