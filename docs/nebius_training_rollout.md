# Nebius Preemptible Training And Rollout

This is the operator runbook for the LeHome flywheel on Nebius. Repository
templates are free to validate. Images, the 500 GiB disk, and GPU VMs are
paid and are not created by `validate.sh`.

Smoke is not a third image. Training smoke is the first paid boot of
`vla-training-base` on the preemptible RTX PRO 6000 role. If that smoke
passes, the same VM stays up for the 2K train. Rollout smoke is the first
paid boot of `lehome-rollout` on the same GPU role after the disk is handed
off.

## 1. Local Free Validation

From the repository root:

```bash
infrastructure/nebius/validate.sh
git diff --check
git status --short
```

Expected: Packer and Terraform validate, infrastructure tests pass, no paid
commands run, and no secrets or tfstate are tracked.

Pinned local tools land in ignored `infrastructure/nebius/.tools/`:
Packer 1.11.2 and Terraform 1.5.7.

## 2. Packer Image Builds

Builders are temporary on-demand CPU VMs (`cpu-d3` / `16vcpu-64gb`). The
Nebius Packer plugin has no preemptible-builder setting, so these builders
are not preemptible. Packer deletes them after it captures the image.

Required Packer variables, supplied at build time only:

- `project_id`
- `subnet_id`
- `service_account_id`
- `service_account_public_key_id`
- `service_account_private_key_file`
- `image_version`

Neither image needs a Nebius OCI import link.

### Training image: `vla-training-base`

Current captured image: `vla-training-base-20260818-2`
(`computeimage-u00rgrf5r1frcrgdng`), family `vla-training-base`, version
`2026-08-18.2`. It was captured successfully on 2026-08-18; the temporary
builder VM and disk were deleted after capture.

```bash
infrastructure/nebius/.tools/packer init infrastructure/nebius/packer
infrastructure/nebius/.tools/packer build \
  -var-file=<gitignored-packer.pkrvars.hcl> \
  -only=vla-training-base.nebius-image.vla-training-base \
  infrastructure/nebius/packer
```

Packer `docker pull`s the already-pinned trainer digest:

`ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746`

If that GHCR repository is private, the builder needs a GHCR pull token in
its environment. That is registry auth, not a Nebius OCI link. The image
contains no policy weights, datasets, or Hugging Face tokens.

### Rollout image: `lehome-rollout`

```bash
infrastructure/nebius/tools/stage-rollout.sh
infrastructure/nebius/.tools/packer build \
  -var-file=<gitignored-packer.pkrvars.hcl> \
  -only=lehome-rollout.nebius-image.lehome-rollout \
  infrastructure/nebius/packer
```

Packer downloads the official challenge tarball, verifies size
`26676771349` and SHA-256
`1a85e389962909debc4ee9988d8a8c388f905fba60686ef78b1623e6872f7123`, then
`docker load`s it and builds the four-worker appliance layer. Source is
`lehome/docker` revision `a914115729bb0bfd260971b9c8d4147bff38c1fb`.

## 3. Protected 500 GiB Disk

```bash
infrastructure/nebius/.tools/terraform -chdir=infrastructure/nebius/terraform/storage init
infrastructure/nebius/.tools/terraform -chdir=infrastructure/nebius/terraform/storage apply \
  -var parent_id=project-u00tm34tpr00n783hz920t
```

The disk is `NETWORK_SSD`, 500 GiB, `forbid_deletion=true`, and
`lifecycle.prevent_destroy = true`. Record `shared_disk_id`. Destroying a
runtime VM never destroys this disk.

## 4. Training Role Smoke

Copy `infrastructure/nebius/terraform/runtime/training.tfvars.example` to a
gitignored `training.tfvars` and fill the training image id, disk id, subnet,
project, and experiment-manifest URI/digest.

```bash
infrastructure/nebius/.tools/terraform -chdir=infrastructure/nebius/terraform/runtime init
infrastructure/nebius/.tools/terraform -chdir=infrastructure/nebius/terraform/runtime apply \
  -var-file=training.tfvars
```

The VM is preemptible `gpu-rtx6000` / `1gpu-24vcpu-218gb`,
`recovery_policy=FAIL`, `on_preemption=STOP`. Prove on this same VM:

- shared-disk mount and role lease
- CUDA and the pinned GPU/runtime tuple
- parent model and dataset download plus hash readback
- video decode, batch-64 construction, one optimizer step
- loader candidates `0, 4, 8, 12, 16`
- bounded interruption and exact local resume

If any of those fail, stop. Do not start the 2K train.

## 5. Immutable Downloads

On the training VM, download and verify before training:

- parent policy `ryanjin333/lehome-groot-n17-models` revision
  `30ac1a84da67b099e115ad147bcd61e9d60046d3`, subpath `policies/step-12000`
- archive SHA-256 `0ddd4e7ce351dd2172cd1edd967293a50d02c15c0f2c21ca39db94692a57e0b5`
- artifact SHA-256 `3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06`
- BC bundle under `bc/full/`
- accepted rollout bundle under `rollouts/round-1/`
- immutable mixture manifest

A matching Hugging Face revision with the wrong content hash is rejected.

## 6. Batch-64 2K Training And Recovery

Train from the step-12K parent for 2,000 gradient steps, global and physical
batch 64, horizon 16, 70/30 BC/rollout (80/20 later by changing the
experiment manifest, not the image).

- local checkpoints every 500 steps
- Hugging Face publication at 1,000 and 2,000 with fresh readback
- preemption resume prefers verified local recovery, then HF 1K/2K

## 7. Training To Rollout Handoff

One runtime Terraform root, selected by `active_role`. Never attach the
shared disk to two VMs.

1. Stop training and wait for the guest preemption/shutdown handler.
2. Release the workspace role lease.
3. Destroy or stop the training instance. Storage state is untouched.
4. Apply the same runtime root with `rollout.tfvars` and the rollout image
   id. The disk attaches `READ_WRITE` with `device_id=lehome`.

## 8. Four-Worker CPU-Cloth Rollout

Default is four persistent Isaac workers plus one batched policy server.
Cloth simulation stays on CPU. The policy server owns the model once and
batches at most four sessions. The policy gateway binds `127.0.0.1:15555`
so DCGM can keep `localhost:5555`.

Do not boot another GPU until the frozen 1-episode smoke command is the
thing you will run. The current captured golden image is
`lehome-rollout-20260818-final7` (`computeimage-u00sfetv9456n1q937`), family
`lehome-rollout`, version `2026-08-18.7`. It has DCGM,
`/opt/lehome/pydeps`, writable `/eval/logs` and `/kitcache`, the CPU-cloth
contact fix, the active-lease worker supervisor guard, and
`smoke_one_episode.sh`. It also carries the reviewed, durable rollout
preemption path that pauses the campaign, makes active leases retryable,
checkpoints the SQLite WAL, and runs the finalizer before shutdown. The prior
images are fallbacks only.

Skip the old campaign 1K/2K (`rft/c87b1861…`). Eval candidates are only
`original_baseline` (original step-12K) and `new_step_2k`. Skip this-run 1K.

### 8a. Frozen 1-episode smoke

First paid boot of the rebuilt `lehome-rollout` image runs only:

```bash
sudo env \
  LEHOME_POLICY_SHA256=<loaded-weights-sha256> \
  LEHOME_POLICY_REVISION=<immutable-hf-revision> \
  LEHOME_POLICY_STEP=<step> \
  LEHOME_POLICY_ARTIFACT_SHA256=<artifact-sha256> \
  LEHOME_CHECKPOINT_DIR=<shared-disk-checkpoint> \
  /opt/lehome/rollout_appliance/smoke_one_episode.sh
```

That script is `rollout_appliance/smoke_one_episode.sh`. The operator supplies
the immutable policy revision, step, artifact hash, loaded-weights hash, and
checkpoint directory. For the completed run, the 2K revision is
`efc3d02605b0fa75c918fe094e335ec7475a8c54` and the loaded-weights SHA-256 is
`761b1caacc606466fdd5d5720b4b9da3f2baf1fe929ccc1f50ec7f82094861e5`.
The script starts the policy server on port 15555, then runs one Isaac worker
through `/isaac-sim/python.sh` inside the container as uid 1234, against
`Top_Long_Unseen_0` seed 601 with a unique session id. The host wrapper runs as
root because it owns Docker and the campaign directories; no host
`isaac-sim` user is required. Success means the worker leased that attempt and
finished. If it does not finish, stop the GPU. Do not leave the box up as a
debugger.

Pinned identities live in `rollout_appliance/one_episode_smoke.py`.

The `final6` predecessor passed this gate on 2026-08-18. The untouched boot
started the policy on `cuda:0`, created the simulator with CPU cloth, leased
the frozen episode, ran 600 steps, wrote all three videos and immutable
receipts, settled the terminal handoff, and exited 0. The task itself was a
normal policy failure; no infrastructure abort or unsettled handoff remained.
The episode JSON SHA-256 was
`6e24e7ba5ca9230def609ca3d2ccdedecfeb7dbfceee676ae13c805fd09d0806`.
`final7` is the code-reviewed successor to that live-smoked image. Its image
build and in-image syntax/compile checks passed, but it has not been booted for
a second paid GPU smoke because its changes are confined to shutdown safety
and image defaults.

### 8b. 80-unseen only after smoke

If and only if the 1-episode smoke leases and finishes, start the
80-unseen matrix on four workers for the three candidates above. Do not
download or evaluate the previous campaign 1K/2K.

On first boot prove:

- official challenge image already present
- cloth on CPU
- one checkpoint loaded once
- session routing, VRAM, and policy latency
- terminal artifact publication
- restart after forced interruption retries only incomplete attempts

No wave barriers. A finished worker leases the next attempt immediately.

## 9. Round Seal And Hugging Face Readback

Accepted episodes are validated, hashed, and uploaded in the background.
A round is sealed only after fresh readback. Mutable `latest` pointers are
not provenance. Cap the campaign at 400 attempts and 150 accepted episodes.

## 10. Winner Gate

This campaign evaluates two candidates only: original step-12K
(`original_baseline`) and this run's 2K (`new_step_2k`). Skip this-run 1K
and the old campaign 1K/2K. Each remaining candidate
runs the same 80 public-unseen episodes, 20 per category (`top_long`,
`top_short`, `pant_long`, `pant_short`), plus the 24-trial seen-dev
screen. The proposed winner and baseline then run the 200 seen matrix.

Physical promotion requires at least 56/80 overall, 12/20 in every
category, no major safety failure, no provenance failure, and no material
seen regression. If nobody passes but one improves safely, emit a
next-round rollout manifest instead.

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer \
  python scripts/select_groot_challenge_winner.py \
  --reports-dir <sealed-reports> \
  --evaluation-manifest <predeclared-manifest> \
  --output <receipt.json>
```

Exit 0 is physical approval. Exit 2 is next-round only. Exit 1 is rejected
or invalid input.

## 11. Physical-Test Boundary

Physical testing starts only after the complete winner gate passes. A
compiling image, a smoke, or a next-round improver is not approval.

### 2026-08-18 sealed result

The frozen 80-episode comparison is published in the private model repository
at revision `fa1c57edccc288fcf8c267a9d40cad0671607788` under
`evaluations/lehome-awr-2k-20260818-v1/unseen80/08370a18b44d9b583f4f9ac38cb6b881657bbbdb1523d65727cf545fd433f11d/`.

- recorded original step-12K: 30/80 (37.5%)
- new step-2K: 17/80 (21.25%)
- new step-2K categories: top-long 2/20, top-short 3/20, pant-long 4/20,
  pant-short 8/20
- selection: original step-12K wins; new step-2K is rejected
- physical testing: not approved

The strict new-step-2K report file SHA-256 is
`08370a18b44d9b583f4f9ac38cb6b881657bbbdb1523d65727cf545fd433f11d`.
Fresh Hugging Face readback also verified the comparison, recorded baseline,
and frozen matrix hashes.

## 12. Deletion

## Asynchronous experiment sweep (CPU preflight only)

The experiment controller is the single SQLite writer. Before any paid action,
generate and verify the canonical manifest set, validate the stopped-at-rest
experiment pool, and run the CPU campaign simulation. The two training workers
lease independently; the rollout appliance remains the only evaluation GPU and
does not begin evaluation or recovery collection until the existing final12
teacher-probe and sealed recovery-data gates pass. Recovery arms remain blocked
until category caps and Hub readback are verified. Stop idle trainers after ten
minutes and destroy no protected rollout storage through the experiment root.

### Async sweep operator sequence

Run the complete CPU-only preflight before starting any Nebius instance. This
creates no GPU, disk, Hub, or rollout artifacts:

```bash
PYTHONPATH=source/lehome:trainer/src:. uv run --project trainer pytest -q \
  trainer/tests/test_experiment_*.py trainer/tests/test_awr_weighting.py \
  trainer/tests/test_challenge_evaluation.py
PYTHONPATH=source/lehome:trainer/src:. uv run --project trainer \
  python scripts/build_lehome_experiment_sweep.py --help
git diff --check
```

Only after all canonical manifests, recovery dependency receipts, request-set
readbacks, and the final12 one-worker teacher probe are green may the three
lanes run:

| Lane | Machine | What it does | When it stops |
| --- | --- | --- | --- |
| Controller | stopped-at-rest CPU VM | SQLite lease writer, budget accounting, promotion, receipt checks | immediately on a bad immutable identity or strict gate |
| Trainer A/B | two independent preemptible RTX PRO 6000 VMs | 500 → 1K → 2K jobs; each worker immediately leases its next admissible job | ten idle minutes, preemption, budget cap, or controller block |
| Rollout/evaluator | one preemptible RTX PRO 6000 rollout VM using the protected 500 GiB disk | final12 probe, controlled recovery collection, then fixed-matrix evaluation | any smoke/fidelity/teacher/readback failure; never shares the disk with a trainer |

There is no wave barrier: a completed training lease publishes and readback
verifies its checkpoint, then becomes evaluation-ready while the other trainer
continues. The controller caps the topology at exactly three GPU leases and
keeps the sweep within its gradient/spend ceilings.

Classify a failure before retrying. An **infrastructure** failure (preemption,
capacity, mount, request-set hydration, GPU/Isaac process) is retryable only
for that lease after its local/HF receipt is checked. An **identity/fidelity**
failure (hash, matrix, parent publication, teacher probe, request-set profile)
is a hard block: do not spend GPU time. A **policy outcome** (ordinary task
failure or a successful recovery shortfall) is evidence, not an infrastructure
retry. A **sealed-artifact/readback** failure blocks publication and promotion.

The historical bad 1K/2K policies are never injected as named candidates. The
final async selector receives only dynamic finalist experiment IDs. It reuses
the original 12K baseline only when its sealed final-unseen80 receipt binds the
exact original checkpoint digest, the exact final matrix digest, all 80 unique
per-episode sealed/readback artifacts, the report digest, and its sidecar seal.
Otherwise it returns `baseline_evaluation_required` and starts no finalist
comparison on the assumption that an older score is comparable.

### AWR-style weighted-replay ablation

The first recovery experiment is unweighted. A deterministic **AWR-style
weighted replay** ablation is considered only after that exact recovery job is
a verified winner on the sealed final unseen-80 gate. It keeps the original
parent, train data, mixture ratio, seed, target step, save cadence, and matrix
unchanged. It changes replay frequency; it does not claim to apply native
per-sample loss weights or canonical AWR.

The AWR-style request set must be newly published. It may not reuse the
unweighted request-set manifest/tree. Its authenticated readback receipt binds
the pending-ablation digest, the new request-set identity, the child runtime
profile, the unchanged training/data identity, the progress-evidence digest and
its authenticated readback receipt, and the `AwrReplayConfig` digest. Until that external materialization and
receipt exist, state is `PENDING_MATERIALIZATION`, not `READY`; do not lease a
GPU for it. The controller admission path is the authenticated
`POST /awr-admission` endpoint; it consumes that exact receipt separately from
recovery-data dependencies before changing it to `READY`. Replaying the exact
receipt is idempotent; a different receipt for the same job is rejected.

The shared disk cannot be destroyed while `prevent_destroy` remains. Removal
requires a separately reviewed Terraform change that deletes that guard,
then a second apply. Runtime destroy must never be used to delete the disk.

## Secrets Table

| Secret | Injected at | Forbidden in |
| --- | --- | --- |
| Nebius service-account PEM | Packer build host | Images, git, Terraform state |
| Nebius project credentials | Operator shell for Terraform | Images, git, example tfvars |
| Hugging Face token | GPU VM environment | Images, Packer vars, Terraform |
| GHCR pull token | Packer builder environment | Git |
| Private repository access | Runtime only | `rollout-stage/` |

## Paid Commands Not Run By Validation

```text
infrastructure/nebius/.tools/packer build infrastructure/nebius/packer
infrastructure/nebius/.tools/terraform -chdir=infrastructure/nebius/terraform/storage apply
infrastructure/nebius/.tools/terraform -chdir=infrastructure/nebius/terraform/runtime apply
```
