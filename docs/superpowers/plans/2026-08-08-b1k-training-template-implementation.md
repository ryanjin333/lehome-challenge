# B1K Training Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish a headless, resumable B1K GR00T N1.7 training image and Vast template that train for 15,000 optimizer steps, retain two rolling checkpoints in a private Hugging Face bucket, and publish the final policy to a private Hugging Face model repository.

**Architecture:** Keep the existing non-root `lehome_train.b1k` controller, replace its LeHome repository identities with B1K-only contracts, and complete the injected lifecycle around immutable dataset/model revisions. The container never creates or destroys Vast instances; it validates the GPU, disk, token file, dataset identity, checkpoint state, and final immutable Hub readback before marking success.

**Tech Stack:** Python 3.10, pytest, Hugging Face Hub, PyTorch/torchrun, Isaac-GR00T, POSIX shell, Docker, Vast templates.

---

### Task 1: Make all training identities B1K-specific

**Files:**
- Modify: `trainer/src/lehome_train/constants.py`
- Modify: `trainer/src/lehome_train/b1k/contracts.py`
- Modify: `trainer/src/lehome_train/b1k/lifecycle.py`
- Modify: `trainer/release-manifest.example.json`
- Test: `trainer/tests/b1k/test_contracts.py`
- Test: `trainer/tests/b1k/test_lifecycle.py`
- Test: `trainer/tests/test_release_manifest.py`

- [ ] Add failing assertions that the accepted repositories are exactly:

```python
HF_DATASET_REPO = "behavior-1k/2026-challenge-demos"
HF_MODEL_REPO = "ryanjin333/behavior1k-groot-n17-models"
HF_CHECKPOINT_BUCKET = "ryanjin333/behavior1k-groot-n17-checkpoints"
```

- [ ] Add a regression test proving every `ryanjin333/lehome-*` value is rejected by `RunContract.from_environment`.
- [ ] Run `uv run pytest trainer/tests/b1k/test_contracts.py trainer/tests/b1k/test_lifecycle.py trainer/tests/test_release_manifest.py -q` and confirm the new tests fail for the old identities.
- [ ] Introduce named B1K constants and consume them from contracts, lifecycle assembly, and the example manifest without accepting arbitrary repositories.
- [ ] Run the targeted tests again and expect all to pass.
- [ ] Commit with `git commit -m "fix(b1k): isolate training repository identities"`.

### Task 2: Complete token and remote-access preflight

**Files:**
- Modify: `trainer/src/lehome_train/b1k/bootstrap.py`
- Modify: `trainer/b1k_launchkit/onstart.sh`
- Modify: `trainer/b1k_launchkit/README.md`
- Test: `trainer/tests/b1k/test_bootstrap.py`

- [ ] Add failing tests for a runtime-user-owned token file with mode `0600`, missing or group-readable files, private model/bucket repositories, and write/read/delete probes under a unique `smoke/<uuid>/` prefix.
- [ ] Assert probe cleanup deletes only the exact uploaded probe key and verifies it is absent afterward.
- [ ] Run `uv run pytest trainer/tests/b1k/test_bootstrap.py -q` and confirm the new security/integration-contract tests fail.
- [ ] Replace old LeHome Hub targets in `ProductionHubAccess`; keep the token out of dataclasses, manifests, logs, subprocess arguments, and generated Vast JSON.
- [ ] Keep `onstart.sh` responsible only for copying the account-level token to `/workspace/.cache/huggingface/token`, changing ownership to UID/GID 10001, setting `0600`, unsetting `HF_TOKEN`, and dropping privileges before Python starts.
- [ ] Run the targeted test and `bash -n trainer/b1k_launchkit/onstart.sh`; expect both to pass.
- [ ] Commit with `git commit -m "fix(b1k): harden training Hub preflight"`.

### Task 3: Make dataset bootstrap immutable and restart-safe

**Files:**
- Modify: `trainer/src/lehome_train/b1k/dataset.py`
- Modify: `trainer/src/lehome_train/b1k/models.py`
- Modify: `trainer/src/lehome_train/b1k/bootstrap.py`
- Test: `trainer/tests/b1k/test_dataset.py`
- Test: `trainer/tests/b1k/test_models.py`

- [ ] Add failing tests for revision-qualified snapshot downloads, atomic `.incomplete` staging, manifest/stats/modality fingerprint verification, cache reuse after restart, and rejection of a mismatched 100-task manifest.
- [ ] Require exactly 100 unique R1Pro tasks and reject duplicate task IDs or a manifest that omits the pinned dataset revision.
- [ ] Run `uv run pytest trainer/tests/b1k/test_dataset.py trainer/tests/b1k/test_models.py -q` and observe the expected failures.
- [ ] Implement idempotent snapshot/bootstrap functions that return verified local paths and never redownload a completed immutable snapshot.
- [ ] Keep RGB, actions/proprioception, task labels, episode metadata, and normalization statistics; do not materialize unused simulator assets in the training image.
- [ ] Run targeted tests and expect all to pass.
- [ ] Commit with `git commit -m "feat(b1k): add immutable training bootstrap"`.

### Task 4: Finish the production lifecycle

**Files:**
- Modify: `trainer/src/lehome_train/b1k/lifecycle.py`
- Modify: `trainer/src/lehome_train/b1k/rolling_checkpoints.py`
- Modify: `trainer/src/lehome_train/b1k/finalize.py`
- Modify: `trainer/src/lehome_train/b1k/launch.py`
- Test: `trainer/tests/b1k/test_lifecycle.py`
- Test: `trainer/tests/b1k/test_rolling_checkpoints.py`
- Test: `trainer/tests/b1k/test_finalize.py`
- Test: `trainer/tests/b1k/test_launch.py`

- [ ] Add failing end-to-end controller tests for fresh start, auto-resume from the newest valid remote checkpoint, step-0 CUDA OOM fallback, interruption after a published checkpoint, completion at exactly step 15,000, and immutable final-model readback.
- [ ] Assert checkpoints publish at steps 1,000 through 15,000, rolling retention leaves only the newest two checkpoint prefixes, and the final model release is not deleted by rolling cleanup.
- [ ] Assert `run-status.json` becomes `complete` only after the final model commit is freshly downloaded and hashes match the local release manifest.
- [ ] Run the four targeted test modules and confirm failures at the current deliberate `RuntimeError`.
- [ ] Replace the deliberate startup exception with orchestration that calls preflight, bootstrap, resume selection, `torchrun`, checkpoint watcher/publisher, final publisher, and atomic status recording.
- [ ] Preserve `AUTO_DESTROY=0`; failure and interruption must exit nonzero and leave enough identity metadata to resume without deleting the instance.
- [ ] Run the four targeted test modules and expect all to pass.
- [ ] Commit with `git commit -m "feat(b1k): complete resumable training lifecycle"`.

### Task 5: Publish a Docker Hub-compatible training image contract

**Files:**
- Modify: `trainer/Dockerfile`
- Modify: `trainer/scripts/verify-image.sh`
- Modify: `trainer/src/lehome_train/b1k/template.py`
- Modify: `trainer/tests/b1k/test_template.py`
- Modify: `.github/workflows/groot-trainer-image.yml`
- Create: `trainer/vast-template.example.json`

- [ ] Add failing tests requiring `docker.io/ryanjin333/behavior1k-groot-n17-trainer@sha256:<64hex>`, a private template, `AUTO_DESTROY=0`, one-to-four GPU portability, at least 2 TB disk, no credential values, and the B1K-only Hub targets.
- [ ] Add workflow assertions for Docker Hub login via `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`, multi-stage build, immutable digest output, and no GHCR publication.
- [ ] Run `uv run pytest trainer/tests/b1k/test_template.py -q` and confirm the existing GHCR/one-to-two-GPU contract fails.
- [ ] Update the template renderer and image verifier to accept only the canonical Docker Hub repository plus digest and to keep registry authentication outside template JSON.
- [ ] Keep the image training-only: assert `/isaac-sim`, `/IsaacLab`, and OmniGibson runtime assets are absent.
- [ ] Render `trainer/vast-template.example.json` with a zero digest placeholder used only as a schema fixture; production deployment must replace it with the registry-reported digest.
- [ ] Run the targeted tests, `bash -n trainer/scripts/verify-image.sh`, and a secret-name scan.
- [ ] Commit with `git commit -m "feat(b1k): define Docker Hub training template"`.

### Task 6: Run the local training acceptance gate

**Files:**
- Modify only if a verified defect is found in files owned by Tasks 1-5.

- [ ] Run `uv run pytest trainer/tests/b1k trainer/tests/test_release_manifest.py trainer/tests/test_cli.py -q`.
- [ ] Run `uv run python trainer/scripts/verify-b1k-cli.py`.
- [ ] Run `bash -n trainer/b1k_launchkit/onstart.sh trainer/scripts/verify-image.sh`.
- [ ] Run `rg -n "lehome-groot|ghcr\.io|HF_TOKEN=|hf_[A-Za-z0-9]" trainer .github/workflows/groot-trainer-image.yml` and resolve every B1K-scope hit or document why it is an intentional negative test.
- [ ] Run `git diff --check` and inspect the complete training diff.
- [ ] Commit any acceptance-only fixes with `git commit -m "test(b1k): close training acceptance gates"`.
