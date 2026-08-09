# B1K Rollout Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate headless B1K rollout image and Vast template that load a selected private GR00T checkpoint, run the pinned official R1Pro evaluator, and publish complete rollouts into immutable success and failure splits in a private Hugging Face dataset repository.

**Architecture:** Vendor no mutable simulator state. Build from the pinned BEHAVIOR-1K source identity, run the official OmniGibson evaluator against an isolated GR00T websocket policy process, and wrap results in a strict campaign controller. Only terminal evaluator records with a closed episode become success/failure; crashes, timeouts without evaluator closure, and partial uploads remain quarantined.

**Tech Stack:** Python 3.10/3.13, pytest, OmniGibson/Isaac Sim, Hugging Face Hub, GR00T websocket policy server, Docker, Vast templates.

---

### Task 1: Scaffold immutable rollout contracts

**Files:**
- Create: `rollout/pyproject.toml`
- Create: `rollout/src/b1k_rollout/__init__.py`
- Create: `rollout/src/b1k_rollout/contracts.py`
- Create: `rollout/src/b1k_rollout/identity.py`
- Create: `rollout/tests/test_contracts.py`
- Create: `rollout/tests/test_identity.py`

- [ ] Write failing tests for the pinned identities:

```python
BEHAVIOR_REVISION = "26f2c7ef7b9cf96bd0414f81e1e751e493762779"
GROOT_REVISION = "ace36d935b376fbf25cd56371e23877b95407c40"
MODEL_REPO = "ryanjin333/behavior1k-groot-n17-models"
DATASET_REPO = "ryanjin333/behavior1k-groot-n17-rollouts"
```

- [ ] Require a selected model commit, image digest, run/cycle/campaign IDs, evaluator mode, task-manifest hash, checkpoint artifact hash, and `AUTO_DESTROY=0`.
- [ ] Reject `hidden_test`, arbitrary task lists, mutable branch names, every LeHome repository, and credential material in serializable contracts.
- [ ] Run `uv run --project rollout pytest rollout/tests/test_contracts.py rollout/tests/test_identity.py -q` and confirm failures.
- [ ] Implement frozen, secret-free contract types with canonical JSON hashing.
- [ ] Run the targeted tests and expect all to pass.
- [ ] Commit with `git commit -m "feat(b1k): scaffold rollout contracts"`.

### Task 2: Pin and validate the official 100-task evaluator manifest

**Files:**
- Create: `rollout/src/b1k_rollout/task_manifest.py`
- Create: `rollout/scripts/build-task-manifest.py`
- Create: `rollout/task-manifest.json`
- Create: `rollout/tests/test_task_manifest.py`

- [ ] Write failing tests that derive task names from the pinned BEHAVIOR evaluator metadata, require exactly 100 unique R1Pro tasks, use only `train` or `public_test`, and map each requested instance index explicitly.
- [ ] Add a provenance record containing source repository, source commit, generator version, task count, and SHA-256.
- [ ] Run the targeted test and confirm the missing generator/manifest fails.
- [ ] Implement the generator against the pinned checkout; sort tasks deterministically and fail closed if the upstream count is not exactly 100.
- [ ] Generate and commit `rollout/task-manifest.json`; never hand-type or silently truncate the task list.
- [ ] Run the targeted tests and expect all to pass.
- [ ] Commit with `git commit -m "feat(b1k): pin official rollout task manifest"`.

### Task 3: Classify only verified closed outcomes

**Files:**
- Create: `rollout/src/b1k_rollout/outcomes.py`
- Create: `rollout/src/b1k_rollout/episodes.py`
- Create: `rollout/tests/fixtures/closed-success.json`
- Create: `rollout/tests/fixtures/closed-failure.json`
- Create: `rollout/tests/fixtures/incomplete.json`
- Create: `rollout/tests/test_outcomes.py`
- Create: `rollout/tests/test_episodes.py`

- [ ] Write failing table-driven tests for official evaluator records where `success is True` maps to `success`, `success is False` maps to `failure`, and missing/invalid terminal evidence maps to `quarantine`.
- [ ] Require task, resolved instance ID, rollout ID, steps, evaluator completion marker, final Q-score payload, model/image/task-manifest identities, and artifact hashes before classification.
- [ ] Reject policy-server crashes, simulator crashes, malformed JSON, zero-step records, duplicate episode keys, and files still ending in `.incomplete`.
- [ ] Run the targeted tests and confirm they fail.
- [ ] Implement atomic episode envelopes and strict outcome parsing without using Q-score thresholds to redefine official success.
- [ ] Run the targeted tests and expect all to pass.
- [ ] Commit with `git commit -m "feat(b1k): classify terminal rollout outcomes"`.

### Task 4: Publish immutable success/failure releases

**Files:**
- Create: `rollout/src/b1k_rollout/publish.py`
- Create: `rollout/src/b1k_rollout/release.py`
- Create: `rollout/tests/test_publish.py`
- Create: `rollout/tests/test_release.py`

- [ ] Write failing tests for the layout:

```text
campaigns/<campaign_id>/releases/<release_id>/
  success/<episode_id>/<artifact files>
  failure/<episode_id>/<artifact files>
  quarantine/<episode_id>/<artifact files>
  campaign-manifest.json
  release-manifest.json
  SHA256SUMS.json
```

- [ ] Assert `release_id` is content-derived, every classified episode appears exactly once, manifests include counts and provenance, and the dataset repo must be private.
- [ ] Assert publication uses `.incomplete` staging, verifies the immutable Hub commit via fresh download/readback, and deletes only the exact staging prefix after a failed publish.
- [ ] Run the targeted tests and confirm failures.
- [ ] Implement the publisher with an injectable Hub adapter and deterministic tree/hash verification.
- [ ] Run the targeted tests and expect all to pass.
- [ ] Commit with `git commit -m "feat(b1k): publish immutable rollout splits"`.

### Task 5: Orchestrate checkpoint, policy server, evaluator, and campaign

**Files:**
- Create: `rollout/src/b1k_rollout/checkpoint.py`
- Create: `rollout/src/b1k_rollout/policy.py`
- Create: `rollout/src/b1k_rollout/lifecycle.py`
- Create: `rollout/src/b1k_rollout/cli.py`
- Create: `rollout/tests/test_checkpoint.py`
- Create: `rollout/tests/test_policy.py`
- Create: `rollout/tests/test_lifecycle.py`

- [ ] Write failing controller tests for immutable checkpoint download/readback, policy `/healthz` readiness, evaluator command construction, per-episode timeout, process-group shutdown, resume after interruption, and publish only after all requested episodes are terminal or quarantined.
- [ ] Keep the official evaluator invocation equivalent to:

```text
python -m omnigibson.eval.eval --task-name <task> --robot-config /behavior-src/OmniGibson/omnigibson/eval/r1pro.yaml --mode <train|public_test> --host 127.0.0.1 --port 8000 --instance-indices <n> --num-rollouts <n> --output-dir <episode-dir> --headless --write-video
```

- [ ] Run targeted tests and confirm failures.
- [ ] Implement one local GR00T websocket server per assigned GPU and one evaluator worker per simulator GPU; make worker/GPU assignment explicit in the campaign manifest.
- [ ] Preserve official R1Pro action semantics and evaluator success; do not add DAgger, hard-state resets, task-group checkpoint routing, or RoboTTT.
- [ ] Run targeted tests and expect all to pass.
- [ ] Commit with `git commit -m "feat(b1k): orchestrate headless rollout campaigns"`.

### Task 6: Build and verify the headless rollout image

**Files:**
- Create: `rollout/Dockerfile`
- Create: `rollout/entrypoint.sh`
- Create: `rollout/scripts/verify-image.sh`
- Create: `rollout/tests/test_image_contract.py`
- Create: `rollout/README.md`

- [ ] Write failing static tests requiring the pinned BEHAVIOR and GR00T commits, NVIDIA/BEHAVIOR EULA acceptance at build/runtime, `OMNIGIBSON_DATA_PATH`, headless defaults, noVNC/X11 absence, non-embedded secrets, and an explicit Docker healthcheck.
- [ ] Base the runtime on the official pinned BEHAVIOR Docker recipe and verify the resolved parent digest in image labels and the release manifest.
- [ ] Install only the GR00T serving dependencies needed for rollout; keep training dependencies and checkpoint writers out of this image.
- [ ] Make `entrypoint.sh` perform token-file validation, immutable checkpoint bootstrap, simulator asset presence checks, policy readiness, and then `exec` the campaign CLI.
- [ ] Run `uv run --project rollout pytest rollout/tests/test_image_contract.py -q`, `bash -n rollout/entrypoint.sh rollout/scripts/verify-image.sh`, and a secret scan.
- [ ] Commit with `git commit -m "feat(b1k): build headless rollout image"`.

### Task 7: Define the Docker Hub rollout template

**Files:**
- Create: `rollout/src/b1k_rollout/template.py`
- Create: `rollout/tests/test_template.py`
- Create: `rollout/vast-template.example.json`
- Modify: `.github/workflows/groot-trainer-image.yml`

- [ ] Write failing tests requiring `docker.io/ryanjin333/behavior1k-groot-n17@sha256:<64hex>` with the rollout role label, private visibility, `AUTO_DESTROY=0`, one-to-four GPUs, at least 2 TB disk, account-level token-file consumption, and no registry/HF credential values.
- [ ] Require the template to expose SSH only for observability and to start headless without Jupyter, noVNC, desktop, or GUI ports.
- [ ] Extend the image workflow to build and push trainer and rollout as separate role-prefixed tags in the shared private Docker Hub repository and emit both immutable digests.
- [ ] Render the schema fixture, run all rollout tests, inspect the generated JSON, and scan for credentials.
- [ ] Commit with `git commit -m "feat(b1k): define Docker Hub rollout template"`.

### Task 8: Run the local rollout acceptance gate

**Files:**
- Modify only if a verified defect is found in Tasks 1-7.

- [ ] Run `uv run --project rollout pytest rollout/tests -q`.
- [ ] Run `bash -n rollout/entrypoint.sh rollout/scripts/verify-image.sh`.
- [ ] Run `rg -n "lehome-groot|hidden_test|novnc|x11vnc|HF_TOKEN=|hf_[A-Za-z0-9]" rollout .github/workflows/groot-trainer-image.yml` and resolve every unexpected hit.
- [ ] Run `git diff --check` and inspect the complete rollout diff.
- [ ] Commit any acceptance-only fixes with `git commit -m "test(b1k): close rollout acceptance gates"`.
