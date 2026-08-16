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
batches at most four sessions.

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

Five candidates: original baseline, previous 1K, previous 2K, new 1K, new
2K. Each runs the same 80 public-unseen episodes, 20 per category
(`top_long`, `top_short`, `pant_long`, `pant_short`), plus the 24-trial
seen-dev screen. The proposed winner and baseline then run the 200 seen
matrix.

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

## 12. Deletion

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
