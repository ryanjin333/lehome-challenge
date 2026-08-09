# B1K Deployment and Paid Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish both B1K images to private Docker Hub repositories, create digest-pinned private Vast templates, run one cheap real GPU smoke per template under a cumulative USD 5 cap, verify immutable Hugging Face readback, and destroy every smoke instance with receipts.

**Architecture:** A local external deployment controller is the sole owner of paid instance creation and deletion. It records an append-only cost/instance ledger before rental, refuses any projected spend above the cap, names every resource with a unique smoke run ID, and performs cleanup in `finally`. Production container templates retain `AUTO_DESTROY=0` and cannot call the Vast API.

**Tech Stack:** Python 3.10, pytest, Docker Hub API/CLI, Hugging Face Hub, Vast REST/CLI, SSH, JSON release receipts.

---

### Task 1: Build the deployment ledger and Vast boundary

**Files:**
- Create: `deployment/pyproject.toml`
- Create: `deployment/src/b1k_deploy/__init__.py`
- Create: `deployment/src/b1k_deploy/ledger.py`
- Create: `deployment/src/b1k_deploy/vast.py`
- Create: `deployment/tests/test_ledger.py`
- Create: `deployment/tests/test_vast.py`

- [ ] Write failing tests for a USD 5 cumulative cap, unique smoke run IDs, exact instance IDs, offer snapshots, hourly rates, projected/actual spend, and append-only JSONL receipts.
- [ ] Assert rental is rejected before mutation when `accumulated_actual + projected_next > 5.00`.
- [ ] Assert destruction requires an explicit recorded instance ID and cannot accept a broad query, empty target, environment expansion, or unrelated active instance.
- [ ] Run `uv run --project deployment pytest deployment/tests/test_ledger.py deployment/tests/test_vast.py -q` and confirm failures.
- [ ] Implement an injectable Vast adapter and ledger; never store API keys, HF tokens, Docker credentials, or SSH private keys in receipts.
- [ ] Run targeted tests and expect all to pass.
- [ ] Commit with `git commit -m "feat(b1k): add capped Vast smoke controller"`.

### Task 2: Implement Docker Hub and Hugging Face release verification

**Files:**
- Create: `deployment/src/b1k_deploy/dockerhub.py`
- Create: `deployment/src/b1k_deploy/huggingface.py`
- Create: `deployment/tests/test_dockerhub.py`
- Create: `deployment/tests/test_huggingface.py`

- [ ] Write failing tests that require private Docker Hub repositories, registry-reported `sha256:` digests, authenticated pull verification, private HF model/checkpoint/dataset repositories, and fresh immutable readback.
- [ ] Reject tag-only template images, public repositories, embedded credentials, and a digest inferred only from local build output.
- [ ] Run targeted tests and confirm failures.
- [ ] Implement adapters that accept credentials only through the process credential store or token files and redact command/error output.
- [ ] Implement minimal private-repo bootstrap probes that upload, read back, and delete an exact unique key before any rental.
- [ ] Run targeted tests and expect all to pass.
- [ ] Commit with `git commit -m "feat(b1k): verify private release destinations"`.

### Task 3: Implement image and Vast template publication

**Files:**
- Create: `deployment/src/b1k_deploy/publish.py`
- Create: `deployment/src/b1k_deploy/cli.py`
- Create: `deployment/tests/test_publish.py`
- Modify: `trainer/vast-template.example.json`
- Modify: `rollout/vast-template.example.json`

- [ ] Write failing tests for build/push/pull verification, exact Docker Hub repository names, digest substitution into template payloads, private Vast templates, and returned template IDs.
- [ ] Assert template publication is idempotent by canonical name plus digest and never mutates an unrelated template.
- [ ] Run targeted tests and confirm failures.
- [ ] Implement `b1k-deploy publish-images` and `b1k-deploy publish-templates` commands with a dry-run default and explicit `--execute` mutation flag.
- [ ] Emit a secret-free publication receipt containing source commit, image digests, template IDs, and template payload hashes.
- [ ] Run targeted tests and expect all to pass.
- [ ] Commit with `git commit -m "feat(b1k): publish digest-pinned templates"`.

### Task 4: Implement the paid smoke state machine

**Files:**
- Create: `deployment/src/b1k_deploy/smoke.py`
- Create: `deployment/tests/test_smoke.py`

- [ ] Write failing state-machine tests for `planned -> rented -> ssh-ready -> runtime-ready -> readback-verified -> destroyed -> disappearance-verified` plus failure transitions that still destroy in `finally`.
- [ ] Require cheap verified datacenter offers with compatible NVIDIA GPU, sufficient disk/RAM/network for the selected image, expected maximum duration, and projected cost captured before rent.
- [ ] For training smoke, require image pull, non-root runtime, GPU visibility, one synthetic CUDA optimizer step, lifecycle preflight, tiny private model artifact publish, immutable readback, and exact probe cleanup.
- [ ] For rollout smoke, require image pull, OmniGibson/Isaac startup, one official headless local-policy evaluator episode or bounded load/reset smoke, one policy-server healthcheck, a terminal/quarantined episode envelope, private dataset publish, immutable readback, and exact probe cleanup.
- [ ] Assert cleanup verifies both the Vast instance list and SSH endpoint disappearance; a failed disappearance check leaves the smoke failed and loudly reports the exact instance ID.
- [ ] Run targeted tests and confirm failures.
- [ ] Implement the state machine with bounded readiness/runtime timeouts, polling intervals under 60 seconds, interrupt handling, and `finally` cleanup.
- [ ] Run targeted tests and expect all to pass.
- [ ] Commit with `git commit -m "feat(b1k): automate paid lifecycle smokes"`.

### Task 5: Pass all no-rent acceptance gates

**Files:**
- Modify only if verified defects are found in training, rollout, or deployment scope.

- [ ] Run the complete targeted suites:

```text
uv run pytest trainer/tests/b1k trainer/tests/test_release_manifest.py trainer/tests/test_cli.py -q
uv run --project rollout pytest rollout/tests -q
uv run --project deployment pytest deployment/tests -q
```

- [ ] Build both images locally or on a non-rented authenticated builder, run both image verifiers, and confirm registry credentials are available without printing them.
- [ ] Create or verify the three private HF repositories and perform exact-prefix upload/read/delete probes.
- [ ] Confirm the shared Docker Hub repository is private, push both role-prefixed image tags, pull each by registry digest, and record the digests.
- [ ] Publish the two private Vast templates from the digest-qualified payloads and fetch them back to verify payload hashes.
- [ ] Confirm the currently active LeHome campaign instance and every unrelated Vast instance ID are outside the smoke ledger and cleanup target set.
- [ ] Do not rent if any preceding checkbox fails.

### Task 6: Run and clean up the training smoke

**Files:**
- Create at runtime: `artifacts/smoke/<run_id>/training-receipt.json`
- Create at runtime: `artifacts/smoke/<run_id>/cost-ledger.jsonl`

- [ ] Snapshot eligible offers and select the cheapest compatible offer whose projected contribution keeps cumulative spend at or below USD 5.
- [ ] Record the offer and planned spend before calling the Vast create endpoint.
- [ ] Rent exactly one training-smoke instance from the published training template.
- [ ] Wait for SSH and runtime readiness; run the training smoke contract and verify the tiny model artifact by immutable HF readback.
- [ ] Destroy the exact recorded instance in `finally`, then verify it is absent from the Vast instance list and its SSH endpoint no longer accepts connections.
- [ ] Record actual elapsed time/cost, cleanup evidence, image digest, template ID, and Hub commit in the receipt.

### Task 7: Run and clean up the rollout smoke

**Files:**
- Create at runtime: `artifacts/smoke/<run_id>/rollout-receipt.json`
- Modify at runtime: `artifacts/smoke/<run_id>/cost-ledger.jsonl`

- [ ] Recompute remaining cap from recorded actual training-smoke spend before selecting an offer.
- [ ] Rent exactly one rollout-smoke instance from the published rollout template.
- [ ] Wait for image/runtime readiness; prove real GPU visibility, OmniGibson/Isaac headless startup, GR00T policy health, and bounded official evaluator loading.
- [ ] Publish the smoke episode envelope to the private dataset repo, verify the immutable Hub commit by fresh readback, and remove only the exact smoke probe release.
- [ ] Destroy the exact recorded instance in `finally`, then verify list absence and SSH endpoint disappearance.
- [ ] Record actual elapsed time/cost, cleanup evidence, image digest, template ID, and Hub commit in the receipt; assert cumulative spend is at most USD 5.

### Task 8: Final audit and handoff

**Files:**
- Create: `artifacts/deployment/b1k-two-template-release.json`
- Modify only if a verified defect is found.

- [ ] Fetch both Vast templates and both Docker Hub manifests again; verify IDs, privacy, canonical digest references, and payload hashes.
- [ ] Verify all three HF repositories are private and all smoke-only probe keys are absent.
- [ ] List current Vast instances and compare exact IDs against the smoke ledger; fail if any smoke instance remains.
- [ ] Run all local suites, `git diff --check`, shell syntax checks, secret scans, and both image verifiers.
- [ ] Write the final release record with source commit, image digests, template IDs, HF repo identities, smoke receipts, cumulative cost, and known limits.
- [ ] Request a fresh read-only Sol reviewer verdict of exactly `ship`, `fix-first`, or `rethink`; address every `fix-first` blocker before declaring completion.
