# Public GR00T N1.5 Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the pinned public GR00T N1.5 training recipe, admit it through a top-short/pant-long paired gate, then run a uniform 1,000-attempt seen-garment collection only if the gate passes.

**Architecture:** Add a thin fail-closed host controller around the unmodified pinned public repository. The controller verifies source, model, dataset, runtime, VM, disk, and artifact identities; invokes upstream training and evaluation commands; extends the existing official-comparison primitives with an explicit two-category paired profile; and dispatches a frozen uniform harvest manifest. It never imports the local N1.7 trainer or rewrites policy/data semantics.

**Tech Stack:** Python 3.11, Bash, pytest, Docker, LeRobot 0.4.3, GR00T N1.5, Nebius CLI, Hugging Face Hub, SHA-256 receipts.

---

## Global constraints

- Use only VM `computeinstance-u00t6xfqhadrcmssa2` and preserve disk
  `computedisk-u00pbe55crxy7jr56x`.
- Never create a VM, disk, image, or second rollout resource.
- Keep the VM stopped during implementation and local verification.
- Do not launch paid training, evaluation, or collection until all offline
  tests and reviews pass.
- Do not modify or call the local N1.7 trainer, policy bridge, runtime mixture,
  hard-state, or curriculum code.
- Use `apply_patch` for edits, TDD for behavior changes, and one fresh
  implementer plus spec and quality review per task.
- Keep weights, datasets, caches, and media off the local Mac.

### Task 1: Encode the immutable N1.5 reproduction contract

**Files:**
- Create: `source/lehome/lehome/n15_reproduction.py`
- Create: `scripts/run_public_n15_reproduction.py`
- Test: `tests/test_n15_reproduction.py`
- Test: `tests/infrastructure/test_public_n15_reproduction_cli.py`

- [ ] Write failing tests for exact upstream revision, four trusted file
  digests, base-model revision, dataset revision, training hyperparameters,
  accepted VM/disk IDs, and rejection of symlinks or mismatched receipts.
- [ ] Write failing CLI tests for `verify-inputs`, `render-training`, and
  `verify-training-output`; every command is offline and atomic.
- [ ] Implement immutable dataclasses and pure validators. The rendered
  training command must be exactly
  `lerobot-train --config_path=configs/train_groot.yaml` from the verified
  public checkout with `HF_HUB_OFFLINE=1`.
- [ ] Make `render-training` emit a JSON execution manifest and shell-safe argv,
  not execute training.
- [ ] Make `verify-training-output` require the step-12,000 checkpoint, source
  receipt, resolved-snapshot receipt, logs, and checksums.
- [ ] Run focused pytest and `py_compile` checks.
- [ ] Commit the task.

### Task 2: Add the two-category paired N1.5 gate

**Files:**
- Modify: `scripts/run_official_lehome_comparison.py`
- Modify: `tests/test_official_lehome_comparison.py`
- Modify: `tests/infrastructure/test_official_lehome_comparison_container.py`
- Create: `rollout_appliance/run_public_n15_focused_gate.sh`

- [ ] Write failing tests for an explicit `n15-focused` profile containing
  only `top_short` and `pant_long`, 12 Release garments per category, two
  episodes per garment, and identical seed matrices for candidate/reference.
- [ ] Write failing tests for paired promotion: candidate no more than two
  successes behind reference per category, candidate floors of 18/24 and
  13/24, complete provenance, and zero fidelity/infrastructure invalids.
- [ ] Refactor only the reusable matrix, command, parser, retained-video,
  execution-environment, sealing, and publication primitives. Preserve the
  current four-category comparison behavior as the default profile.
- [ ] Add a container wrapper that runs both N1.5 checkpoints sequentially
  through their native adapter and official scorer; reject the N1.7 server or
  action gateway.
- [ ] Require immutable receipt publication/readback before returning pass.
- [ ] Run focused unit/infrastructure tests and shell syntax checks.
- [ ] Commit the task.

### Task 3: Build the uniform 1,000-attempt native harvest manifest

**Files:**
- Create: `source/lehome/lehome/n15_harvest.py`
- Create: `scripts/build_public_n15_harvest.py`
- Create: `rollout_appliance/run_public_n15_harvest.sh`
- Test: `tests/test_n15_harvest.py`
- Test: `tests/infrastructure/test_public_n15_harvest.py`

- [ ] Write failing tests for exactly 40 seen garments, 10 per category, 25
  attempts per garment, 250 per category, 1,000 total, globally unique seeds
  and IDs, and deterministic byte-identical output.
- [ ] Write failing tests proving held-out Release evaluator garments/seeds,
  historical success episodes, hard states, perturbations, and curriculum
  weights cannot enter the manifest.
- [ ] Implement an atomic manifest/receipt builder with a frozen ordering and
  source/checkpoint/category provenance.
- [ ] Wrap the public source's native `scripts.eval --save_datasets` path;
  explicitly override its default unseen/two-category harvest filters.
- [ ] Add first-100 stopping rules: fewer than five official successes, any
  cloth fidelity failure, or more than 2% infrastructure-invalid attempts.
- [ ] Add four-worker memory/smoke admission with deterministic fallback to two
  workers; never alter policy/evaluator semantics and never create a VM.
- [ ] Require immutable Hugging Face upload/readback receipts and provider VM
  stop at terminal pass/fail.
- [ ] Run focused tests, shell syntax checks, and compilation.
- [ ] Commit the task.

### Task 4: Add the bounded remote lifecycle controller

**Files:**
- Modify: `scripts/run_public_n15_reproduction.py`
- Create: `rollout_appliance/run_public_n15_pipeline_remote.sh`
- Test: `tests/infrastructure/test_public_n15_pipeline_remote.py`

- [ ] Write failing state-machine tests for
  `STOPPED -> training -> publish/readback -> focused gate -> publish/readback
  -> harvest -> publish/readback -> STOPPED`.
- [ ] Prove every failed training, fidelity, evaluation, publication, budget,
  or provider gate stops the exact VM and forbids downstream stages.
- [ ] Add idempotent resume from immutable receipts without overwriting prior
  run prefixes or repeating a completed paid stage.
- [ ] Add pre-start budget estimation and the existing $100 hard cap.
- [ ] Verify exact VM image, attached protected disk, stopped/running state,
  cloud-init, workspace mount, GPU, and upstream checkpoint identity at each
  relevant transition.
- [ ] Run focused integration tests with mocked provider boundaries plus shell
  syntax/compile checks. Do not start the VM in tests.
- [ ] Commit the task.

### Task 5: Final offline review and paid execution

- [ ] Run all focused N1.5 reproduction, focused-gate, harvest, and lifecycle
  tests, then the relevant existing official-comparison regression suite.
- [ ] Inspect the complete diff and run a fresh read-only final review.
- [ ] Push the reviewed code before cloud execution.
- [ ] Verify Nebius Compute is operational, the exact VM is stopped, and the
  protected disk is attached before the first start.
- [ ] Execute and readback-publish exact N1.5 training evidence.
- [ ] Execute only the paired top-short/pant-long gate; stop and report any
  failed threshold, fidelity, teacher, or infrastructure gate.
- [ ] If and only if it passes, execute the frozen 1,000-attempt harvest with
  the first-100 circuit breaker.
- [ ] Read back the immutable public Hugging Face bundle and verify the exact VM
  is stopped.
- [ ] Report training result, focused scores, admitted/attempted collection
  counts, artifact receipts, spend, and provider state as separate facts.
