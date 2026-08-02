# LeHome GR00T N1.7 Training and Rollout Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce, evaluate, and iteratively improve a LeHome GR00T N1.7 policy using separate disposable Vast.ai training and rollout machines, selected by effective performance per dollar.

**Architecture:** The prepared dataset, checkpoints, reports, rollout shards, and promotion records are immutable Hugging Face artifacts. A single-GPU trainer never runs Isaac Sim; separate Isaac Sim workers consume promoted checkpoints and publish rollout evidence or approved training shards. Vast's listed DLP-per-dollar metric is used to shortlist rentals, while measured samples-per-dollar and completed-rollouts-per-dollar make the final decisions.

**Tech Stack:** GR00T N1.7, PyTorch, LeRobot v2.1 data, Isaac Sim 5.1/Isaac Lab, Docker/GHCR, Hugging Face Hub, Vast.ai, Git LFS, SHA-256 manifests.

---

## 1. Current evidence and immutable inputs

The one-time dataset conversion and normalization are complete. Do not recompute them on a paid trainer.

| Item | Immutable value |
|---|---|
| Dataset repository | `ryanjin333/lehome-groot-n17-data` |
| Dataset revision | `20017126d83cd31ee75e1efe32c0394d36de59be` |
| Dataset manifest SHA-256 | `a60b9e03869a6d38b3cc540e3a1fc0068ee0adc52fd29152188254decef3d8fc` |
| Dataset size | `3,105,604,931` bytes |
| Dataset entries | `4,012` |
| Training episodes | `900` |
| Validation episodes | `100` |
| Loader acceptance | `pinned_loader_one_batch` |
| Base model | `nvidia/GR00T-N1.7-3B` |
| Base model revision | `2fc962b973bccdd5d8ce4f67cc63b264d6886495` |
| Isaac-GR00T commit | `23ace64f17aa5015259b8609d371eb61a357c776` |
| Initial exposure | `768,000` sample presentations |
| Flywheel exposure | `256,000` sample presentations per accepted round |
| Checkpoint interval | `64,000` sample presentations |
| Gradient accumulation | exactly `1` |
| Action horizon | `16` |

The successful readback checked all 4,012 entries and all 3.1 GB against SHA-256. All Vast instances used for preparation have been destroyed. The temporary Vast API key was revoked and its local copies were removed.

### No-rent gates still outstanding

Do not rent the long-running trainer until all four are green:

- [ ] The six local trainer fixes are committed and pushed.
- [ ] CI publishes a clean `linux/amd64` image by immutable OCI digest.
- [ ] The image passes the fresh RTX PRO 6000 GPU acceptance sequence.
- [ ] The GR00T-to-LeHome rollout adapter passes a one-episode Isaac Sim smoke test before a rollout campaign is rented.

The present repository is Phase 1: training is implemented, but the GR00T-specific Isaac rollout adapter is not yet an accepted runtime. Generic custom-policy evaluation exists; that is not evidence that GR00T observation/action conversion is correct.

## 2. Why this strategy matches strong public solutions

The parts worth adopting are operational, not a claim that we can copy another team's private weights or full RL stack:

- The first-place online solution separated training, rollout workers, and a DAgger station, communicating through Hugging Face Hub. It trained on one H200 and collected most rollouts on RTX PRO 6000 workers.
- Its useful flywheel ideas were retained success replays, targeted hard-state recovery, strong environment/camera augmentation, and measured inference settings. Its AWR/RECAP/value-head system is outside this first GR00T baseline.
- The public GR00T submission used a base run followed by a short harvest fine-tune and generated deterministic rollout reports from raw logs.
- Another public submission used category-specialized checkpoints and longer training for a harder category, plus shader-warmup protection. Because LeHome does not reveal garment type at evaluation, category specialization is only safe after an image-based router is proven; it is not part of the first baseline.

Sources:

- <https://arxiv.org/abs/2606.27163>
- <https://huggingface.co/theo-zhou/lehome-groot-submission-4>
- <https://huggingface.co/HeidC/lehome-submission>

## 3. Machine economics: use DLP-per-dollar correctly

Vast shows both raw `DLPerf` and a value metric displayed as `DLP/$/hr`. Use the latter only to shortlist machines. It is a generic deep-learning benchmark, not a GR00T or Isaac Sim benchmark.

### Training shortlist

Reject an offer unless it satisfies every hard gate:

| Requirement | Training gate |
|---|---|
| GPU count | exactly `1` |
| VRAM | `96 GB` preferred; `40 GB` absolute minimum |
| GPU | RTX PRO 6000 96 GB preferred for the acceptance and baseline |
| PCIe | x16; PCIe 4.0 or 5.0 preferred |
| Disk | `300 GB` requested; `200 GB` absolute minimum |
| Disk speed | at least `1.5 GB/s` measured NVMe |
| Download | at least `1 Gbps` measured before model hydration |
| Reliability | at least `99.5%` |
| Max duration | at least `48 hours` for the baseline |
| Rental type | on-demand for the first accepted baseline |

Among offers that pass, rank by:

```text
listed_training_value = Vast DLP/$/hr
measured_training_value = smoke_samples_per_second / hourly_price
baseline_gpu_hours = 768000 / smoke_samples_per_second / 3600
baseline_compute_cost = baseline_gpu_hours * hourly_price
flywheel_gpu_hours = 256000 / smoke_samples_per_second / 3600
flywheel_compute_cost = flywheel_gpu_hours * hourly_price
```

The measured value wins. A machine with a lower Vast score but materially higher GR00T samples/second can be the better rental.

Reference only, from the August 1 screenshot—not a current quote: the strongest interruptible 1x RTX PRO 6000 listing showed `831.6 DLP/$/hr` at `$0.350/hr`; the next two showed `738.0` at `$0.397/hr` and `669.8` at `$0.434/hr`. Re-query immediately before renting.

### Why not 8x RTX 3090 for initial training

The current trainer deliberately enforces one GPU. An 8x3090 rental would pay for seven idle GPUs and each card still has only 24 GB VRAM. Do not use it for the baseline trainer.

An 8x3090 host becomes reasonable for rollout only when the rollout launcher can prove eight isolated Isaac processes, one per GPU, and the host has enough CPU and RAM. The better August 1 example was the `221.1 DLP/$/hr` host at `$1.272/hr`, not the `147.4 DLP/$/hr` host at `$1.234/hr`; even then, completed episodes per dollar is the governing metric.

### Rollout shortlist

Autonomous Isaac rollouts are affected by CPU, Vulkan/RTX compatibility, startup time, and policy inference. Rank them by a measured episode smoke, not DLPerf alone.

| Requirement | Autonomous rollout gate |
|---|---|
| GPU | 1x RTX A6000 48 GB, L40S 48 GB, RTX 4090 24 GB, or RTX PRO 6000 96 GB |
| VRAM | 48 GB preferred; allow 24 GB only after a real one-episode memory smoke |
| CPU | at least 16 fast physical cores per simultaneous Isaac process |
| RAM | at least 64 GB per process; 256 GB+ for four or more workers |
| Disk | 300–500 GB; 1 TB+ for multi-worker video/data collection |
| Driver | Isaac Sim 5.1-compatible Vulkan/RTX driver; reject on any `ERROR_DEVICE_LOST` |
| Rental type | on-demand for adapter acceptance; interruptible is allowed only for restart-safe autonomous batches |

Measure:

```text
rollout_value = completed_valid_episodes / (wall_hours * hourly_price)
rollout_hours = trial_count * mean_trial_minutes / 60 / parallel_workers
rollout_cost = rollout_hours * hourly_price
```

For interactive DAgger, latency matters. Use a nearby region, target under 80 ms round-trip latency, and use on-demand. For fully autonomous local inference, geographic latency is irrelevant after artifacts are downloaded.

## 4. Budget and timing model

Do not estimate the long run from DLPerf. Run the 100-step smoke, record `samples_per_second`, and fill this table before continuing:

| Measured samples/s | 768k baseline | One 256k flywheel | Baseline + three flywheels |
|---:|---:|---:|---:|
| 10 | 21.33 h | 7.11 h | 42.67 h |
| 20 | 10.67 h | 3.56 h | 21.33 h |
| 40 | 5.33 h | 1.78 h | 10.67 h |
| 64 | 3.33 h | 1.11 h | 6.67 h |

Add 30–90 minutes per fresh training instance for image/model/data restore and preflight. Add rollout time using the measured formula above. A practical account reserve is `$250` for the baseline, rollout evaluation, and three controlled flywheel rounds; set a `$500` hard ceiling so failed adapters, slow hosts, or repeated Isaac setup cannot drain the account silently. These are campaign controls, not predictions that all credits will be spent.

Stop conditions:

- Stop any setup that has not reached the first real optimizer step within 30 minutes after a verified 1 Gbps download test.
- Stop training immediately on non-finite loss, incompatible hashes, missing checkpoint upload verification, or a measured cost projection above the campaign ceiling.
- Stop a rollout host after two incompatible Vulkan/Isaac failures; changing to a faster GPU will not repair a driver mismatch.
- Never keep a paid machine idle waiting for another stage. Sync, verify, destroy, and restore on a new instance later.

## 5. Task 1 — Freeze and publish the trainer release

**Files:**

- Modify: `trainer/src/lehome_train/data/stats.py`
- Modify: `trainer/src/lehome_train/data/validate.py`
- Modify: `trainer/src/lehome_train/hub.py`
- Modify: `trainer/tests/test_data_stats.py`
- Modify: `trainer/tests/test_data_validate.py`
- Modify: `trainer/tests/test_hub.py`
- Modify after GPU acceptance: `trainer/release-manifest.example.json`

- [ ] **Step 1: Verify the focused fixes**

```bash
cd /workspace/lehome-challenge/trainer
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  tests/test_data_stats.py \
  tests/test_data_validate.py \
  tests/test_hub.py -q
```

Expected: zero failures. Inspect the collected test names if the count changes because tests were added.

- [ ] **Step 2: Verify the complete CPU-safe release suite**

```bash
cd /workspace/lehome-challenge/trainer
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  tests --ignore=tests/test_groot_runtime_gate.py -q
bash -n docker/entrypoint.sh scripts/build-image.sh scripts/verify-image.sh
```

Expected: zero failures and zero shell syntax errors.

- [ ] **Step 3: Commit and push the exact release source**

```bash
cd /workspace/lehome-challenge
git diff --check
git add \
  trainer/src/lehome_train/data/stats.py \
  trainer/src/lehome_train/data/validate.py \
  trainer/src/lehome_train/hub.py \
  trainer/tests/test_data_stats.py \
  trainer/tests/test_data_validate.py \
  trainer/tests/test_hub.py
git commit -m "fix: finalize immutable GR00T dataset transfer"
git push
git rev-parse HEAD
```

Expected: a clean 40-character commit. Record it as `TRAINER_COMMIT`.

- [ ] **Step 4: Build and resolve the immutable OCI digest**

Allow `.github/workflows/groot-trainer-image.yml` to publish:

```text
ghcr.io/ryanjin333/lehome-groot-n17-trainer:${TRAINER_COMMIT}
```

Resolve the registry digest and record the full `sha256:` value. A tag alone is not a runnable experiment identity.

- [ ] **Step 5: Run the fresh RTX PRO 6000 image acceptance**

Use an on-demand 1x RTX PRO 6000, 300 GB disk, and the highest passing DLP/$/hr offer. Complete the exact acceptance sequence in `trainer/README.md`: pull by digest, verify one GPU, reach a real optimizer step within 1,800 seconds, memorize one episode, smoke batches 16/32/64, sync and read back evidence. Destroy the acceptance machine after the external sync result says `disposable: true`.

## 6. Task 2 — Create the separate Vast templates

### Trainer template

- [ ] Use the public GHCR trainer image by exact digest.
- [ ] Use an SSH/headless container, not an Ubuntu VM.
- [ ] Request 300 GB container storage.
- [ ] Inject `HF_TOKEN` as an account secret/environment variable only.
- [ ] Do not write the token to `/workspace/.cache/huggingface/token` or run `hf auth login`.
- [ ] Mount writable `/cache`, `/prepared`, and `/output`.
- [ ] Keep `HF_HOME=/cache/huggingface` for model blobs, not credentials.

### Rollout template

- [ ] Use Ubuntu 22.04 plus the pinned Isaac Sim 5.1/Isaac Lab environment.
- [ ] Keep it separate from the trainer image; the trainer intentionally contains no Isaac Sim.
- [ ] Default to headless evaluation.
- [ ] Add noVNC only for interactive DAgger collection.
- [ ] Request 300–500 GB disk for a single worker and at least 1 TB for multi-worker video collection.

The initial BC trainer and later flywheel trainers may use the same trainer template. Each flywheel iteration may be a fresh rental: restore the exact promoted model/data commits, run the short fine-tune, sync, verify, and destroy.

## 7. Task 3 — Start the initial behavior-cloning trainer

- [ ] **Step 1: Rent using the training hard gates**

Choose the highest current DLP/$/hr 1x RTX PRO 6000 that passes reliability, PCIe, disk, bandwidth, and duration. Use on-demand for the first baseline.

- [ ] **Step 2: Verify the host before downloading weights**

```bash
nvidia-smi
df -h /cache /prepared /output
git lfs version
```

Require one visible 96 GB GPU and at least 200 GB free; 300 GB was requested.

- [ ] **Step 3: Restore the prepared dataset by immutable Git/LFS commit**

The 4,012-file dataset exceeds the Hub's 1,000-request/5-minute API window when downloaded one file per request. Use the proven Git/LFS bulk path, then validate locally. The authentication helper must read `HF_TOKEN` at process time and must be deleted after checkout; never place the token in the URL or Git config.

```bash
export DATASET_REPOSITORY=ryanjin333/lehome-groot-n17-data
export DATASET_REVISION=20017126d83cd31ee75e1efe32c0394d36de59be
export DATASET_MANIFEST_SHA256=a60b9e03869a6d38b3cc540e3a1fc0068ee0adc52fd29152188254decef3d8fc

test ! -e /cache/dataset-git/repository
test ! -e /prepared/lehome-groot-n17-v1
mkdir -p /cache/dataset-git /prepared/lehome-groot-n17-v1 /output/evidence
printf '%s\n' '#!/bin/sh' \
  'case "$1" in' \
  '*Username*) printf "%s\\n" "hf_user" ;;' \
  '*Password*) printf "%s\\n" "$HF_TOKEN" ;;' \
  'esac' > /cache/tmp/hf-askpass
chmod 700 /cache/tmp/hf-askpass

GIT_ASKPASS=/cache/tmp/hf-askpass GIT_TERMINAL_PROMPT=0 \
  git clone --no-checkout \
  "https://huggingface.co/datasets/${DATASET_REPOSITORY}" \
  /cache/dataset-git/repository
GIT_LFS_SKIP_SMUDGE=1 \
  git -C /cache/dataset-git/repository checkout --detach "$DATASET_REVISION"
test "$(git -C /cache/dataset-git/repository rev-parse HEAD)" = "$DATASET_REVISION"
GIT_ASKPASS=/cache/tmp/hf-askpass GIT_TERMINAL_PROMPT=0 \
  git -C /cache/dataset-git/repository lfs pull
rsync -a --exclude=.git/ \
  /cache/dataset-git/repository/ /prepared/lehome-groot-n17-v1/
rm -f /cache/tmp/hf-askpass

printf '%s  %s\n' "$DATASET_MANIFEST_SHA256" \
  /prepared/lehome-groot-n17-v1/manifest.json | sha256sum --check --strict

lehome-train data validate \
  --dataset /prepared/lehome-groot-n17-v1 \
  --groot-root /opt/isaac-groot

python -c 'from lehome_train.data.publish import write_prepared_snapshot_manifest; write_prepared_snapshot_manifest("/prepared/lehome-groot-n17-v1", "/prepared/lehome-groot-n17-v1/lehome_dataset_snapshot.json", revision="20017126d83cd31ee75e1efe32c0394d36de59be")'
```

Expected validation: `valid: true`, 900 train episodes, 100 validation episodes, and `pinned_loader_one_batch`.

- [ ] **Step 4: Restore the base model**

```bash
lehome-train model retrieve \
  --destination /cache/models/groot-n17 \
  --repo nvidia/GR00T-N1.7-3B \
  --revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --staging-root /cache/staging
```

- [ ] **Step 5: Run prepare, memorize, and smoke**

Generate strict request envelopes using the schemas in `docs/groot_n17_training.md`. Run:

```bash
lehome-train prepare --request /output/requests/prepare.json
lehome-train memorize --request /output/requests/memorize.json
lehome-train smoke --request /output/requests/smoke.json
```

Pass only if memorization reduces normalized MSE to at most 10% of its initialized value, every action dimension improves, loss is finite, and the selected smoke batch leaves at least 10% steady-state VRAM free.

- [ ] **Step 6: Calculate the schedule from the selected physical batch**

| Selected batch | Optimizer steps | Save every |
|---:|---:|---:|
| 64 | 12,000 | 1,000 |
| 32 | 24,000 | 2,000 |
| 16 | 48,000 | 4,000 |
| 8 | 96,000 | 8,000 |

Keep gradient accumulation at exactly 1. Do not add it to simulate a larger batch; it changes the tested configuration and the trainer rejects it.

- [ ] **Step 7: Run the fixed 768k exposure**

```bash
lehome-train train --request /output/requests/train.json
```

Checkpoint every 64,000 presentations. After the first steady 100 steps, calculate projected hours and cost from measured samples/second. Abort if the projection exceeds the campaign ceiling.

- [ ] **Step 8: Report, sync, verify, and destroy**

Use the two-sync shutdown gate in `docs/groot_n17_training.md`:

1. Create a local-evidence report with `sync_result: null`.
2. Sync and persist the result outside the mutable experiment tree.
3. Regenerate the report using that immutable sync result.
4. Sync again and inspect the external final result.
5. Destroy the rental only when `disposable: true` and every artifact is `remotely_verified: true`.

Do not merely stop the instance. Destroy it after the gate so storage billing also ends. Prune Docker cache only after remote verification; the HF model cache is disposable and no HF credential should be cached.

## 8. Task 4 — Implement and accept the GR00T rollout adapter before renting a campaign

**Files:**

- Create: `scripts/eval_policy/groot_n17_policy.py`
- Modify: `scripts/eval_policy/__init__.py`
- Create: `scripts/utils/groot_n17_adapter.py`
- Create: `scripts/eval_groot_n17_matrix.py`
- Create: `tests/test_groot_n17_rollout_adapter.py`
- Create: `tests/test_groot_n17_matrix.py`
- Create: `configs/eval_groot_n17_seen_dev.json`

The adapter contract is fixed:

- Input cameras: top, left wrist, right wrist, matching `meta/lehome_groot_modality.py`.
- Input state: 12 current joint values.
- Output: 16-step action chunk, 12 dimensions per step.
- Five non-gripper joints per arm are decoded as relative targets.
- Each gripper remains an absolute command.
- The action queue resets between episodes and on any safety abort.
- The checkpoint's dataset, normalization, processor, and configuration hashes must match before policy construction.

- [ ] Write CPU-safe tests for camera mapping, 12D state ordering, relative-joint decoding, absolute grippers, 16-step queueing, reset behavior, and incompatible-hash rejection.
- [ ] Generate and commit the seen-development matrix by taking the first two lexicographically sorted official `Seen` garment IDs from each of the four categories, then pairing each with seeds 42, 43, and 44. The committed JSON must contain exactly 24 unique trials and its SHA-256 must be recorded in every rollout report.
- [ ] Write matrix-runner tests that reject duplicate trials, public-unseen garment IDs, missing seeds, mutable checkpoint identifiers, and result/video paths outside the run root.
- [ ] Run the tests and require zero failures.
- [ ] On one compatible Isaac host, load one promoted checkpoint and run one deterministic seen-garment episode with video.
- [ ] Reject the host on Vulkan/RTX initialization errors instead of changing the policy or training machine.
- [ ] Record peak GPU memory, CPU utilization, episode wall time, success, geometric score, and all artifact hashes.

One non-zero deterministic seen success is the first celebration milestone and is enough to unlock flywheel collection. It is not enough to select the final checkpoint.

## 9. Task 5 — Evaluate baseline checkpoints on the fixed matrix

Use only organizer-seen development garments for checkpoint selection:

- Eight fixed seen garment instances.
- Seeds 42, 43, and 44.
- 24 trials per promoted candidate.
- Save video and machine-readable metrics for every trial.

Run an accepted checkpoint with:

```bash
python -m scripts.eval_groot_n17_matrix \
  --matrix configs/eval_groot_n17_seen_dev.json \
  --policy-path /workspace/checkpoints/promoted \
  --dataset-root /workspace/datasets/lehome-groot-n17-v1 \
  --output-root /workspace/rollouts/seen-development \
  --headless \
  --save-video \
  --device cpu
```

Expected: 24 unique completed trial records, 24 videos, no public-unseen garment ID, and one report tied to the checkpoint, matrix, simulator, and normalization hashes.

Public-unseen is a one-time audit after the campaign is locked. Never use public-unseen trajectories as training data or to choose a checkpoint.

Rank candidates by:

1. Most successes out of 24.
2. Higher mean normalized geometric score.
3. Fewer sample presentations.

Promote only the winner of that rule. Training loss alone cannot select a policy.

To reduce wasted rollout spend without corrupting the fixed gate:

- Run a one-episode adapter smoke first.
- Run the full 24-trial matrix only for checkpoints whose loader/hash gate passes.
- Parallelize only when each worker has an isolated seed/garment assignment and publishes an immutable result shard.
- Keep the official 24-trial promotion result separate from replay/hard-mining collection.

## 10. Task 6 — Run up to three flywheel iterations on fresh machines

Each iteration is independently restartable. A rollout machine produces an immutable shard; a new trainer restores the promoted checkpoint and approved shards.

### Data admission

- Always keep organizer demonstrations at 50% or more of training samples.
- Prefer successful policy replays.
- Add DAgger corrections only after the same recoverable failure category appears at least three times in the 24 seen-development trials and a human proves recovery.
- Save only expert-controlled actions after takeover.
- Keep failed policy prefixes and unrecovered corrections as diagnostics, never BC labels.
- Never add public-unseen data.

### Fine-tune contract

- Start from the latest promoted model weights, not a memorization checkpoint.
- Start a new optimizer deliberately and record that restore mode.
- Train exactly 256,000 sample presentations.
- Use peak learning rate `1e-5`, one tenth of the initial `1e-4`.
- Keep gradient accumulation at 1.
- Checkpoint every 64,000 presentations: four checkpoints per round.
- Sync and destroy the trainer after remote verification.
- Evaluate the candidates on the same 24-trial seen matrix.

Promote a flywheel checkpoint only if either:

1. It gains at least two successes overall and no category loses more than one; or
2. It gains zero or one success, improves mean normalized geometric score by at least 5%, and no category loses more than one.

Stop the flywheel early when an iteration fails promotion. Do not automatically spend all three rounds.

## 11. Campaign ledger

Create one row immediately after each rental or artifact promotion:

| Stage | Vast offer ID | GPU | DLP/$/hr | $/hr | Runtime h | Cost | Measured throughput | Artifact commit | Gate |
|---|---:|---|---:|---:|---:|---:|---|---|---|
| Image acceptance | | | | | | | first-step seconds | | |
| Initial BC | | | | | | | samples/s | | |
| Baseline rollout | | | | | | | valid episodes/$ | | |
| Flywheel 1 train | | | | | | | samples/s | | |
| Flywheel 1 rollout | | | | | | | valid episodes/$ | | |
| Flywheel 2 train | | | | | | | samples/s | | |
| Flywheel 2 rollout | | | | | | | valid episodes/$ | | |
| Flywheel 3 train | | | | | | | samples/s | | |
| Flywheel 3 rollout | | | | | | | valid episodes/$ | | |

Blank cells are operational records, not missing design decisions. Fill them from the actual Vast offer and immutable reports at execution time because prices and availability change continuously.

## 12. Definition of done

- [ ] The trainer source commit and OCI digest are immutable and GPU-accepted.
- [ ] The initial 768k run has remotely verified checkpoints and provenance.
- [ ] The GR00T rollout adapter passes its unit and one-episode Isaac smoke gates.
- [ ] Every evaluated candidate has a complete 24-trial seen-development report.
- [ ] A baseline champion is selected by success, geometric score, then exposure.
- [ ] Each admitted flywheel shard contains only allowed successes or post-takeover expert actions.
- [ ] Each flywheel round either passes the promotion rule or stops the campaign.
- [ ] The one-time public-unseen audit happens only after the final campaign lock.
- [ ] Every paid instance is destroyed after immutable remote verification.
- [ ] Actual campaign cost remains below the recorded hard ceiling.
