# N1.5 Provider RUNNING Receipt Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the bounded public N1.5 controller proceed after its first successful immutable RUNNING observation instead of attempting to create the same receipt twice.

**Architecture:** Keep the current fail-closed provider parser and immutable receipt contract. The polling loop owns creation and validation of the RUNNING observation; record that successful return explicitly, require the loop-created receipt to remain a regular non-symlink file, and continue to remote runtime validation without invoking the exclusive writer again. No provider, training, evaluator, or retry policy changes are in scope.

**Tech Stack:** Bash, Python/pytest, mocked Nebius CLI and SSH boundary.

---

### Task 1: Reuse the Successful RUNNING Observation

**Files:**
- Modify: `rollout_appliance/run_public_n15_pipeline_remote.sh:416-420`
- Test: `tests/infrastructure/test_public_n15_pipeline_remote.py`

- [x] **Step 1: Write the failing regression test**

Add a test that runs the real wrapper with a stateful fake `nebius` executable. The fake must return the exact STOPPED provider document before `start`, switch to RUNNING when `start` is called, and switch back to STOPPED when the EXIT trap calls `stop`. Use a fake `ssh` that fails immediately so the wrapper's expected terminal error after a successful provider transition is `runtime/cloud-init/workspace/GPU/upstream gate failed`. Assert that stderr does not contain `native reference receipt already exists` or `exact VM did not reach RUNNING`, and assert exactly one start and one cleanup stop were requested.

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/infrastructure/test_public_n15_pipeline_remote.py -k running_observation
```

Expected before the fix: FAIL because the wrapper tries to write the already-created RUNNING receipt again and reports `native reference receipt already exists` / `exact VM did not reach RUNNING`.

- [x] **Step 3: Implement the minimal lifecycle fix**

Track whether `capture_exact_provider_state RUNNING "$response"` returned success inside the poll loop. After the loop, require that success flag plus the loop-produced receipt as a regular non-symlink file, then remove the receipt and continue. Do not call the exclusive capture writer a second time, and do not weaken exact VM, image, disk, state, timeout, cleanup, or exclusivity checks.

- [x] **Step 4: Verify GREEN and regression coverage**

Run:

```bash
pytest -q tests/infrastructure/test_public_n15_pipeline_remote.py -k running_observation
pytest -q tests/infrastructure/test_public_n15_pipeline_remote.py
bash -n rollout_appliance/run_public_n15_pipeline_remote.sh
```

Expected: all selected tests pass and Bash syntax validation exits zero.

- [x] **Step 5: Run the existing focused N1.5 suite**

Run the repository's existing N1.5 controller/reproduction/focused/harvest tests that covered commit `6d258eba`, and require all to pass before commit.

- [x] **Step 6: Commit the fix**

```bash
git add docs/superpowers/plans/2026-08-31-n15-provider-running-receipt.md \
  rollout_appliance/run_public_n15_pipeline_remote.sh \
  tests/infrastructure/test_public_n15_pipeline_remote.py
git commit -m "fix: reuse n15 provider running receipt"
```
