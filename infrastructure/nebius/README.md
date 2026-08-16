# Nebius Training And Rollout

This directory holds the free Packer and Terraform templates for the
preemptible LeHome flywheel. It does not create images, disks, or GPU VMs
until an operator runs the paid commands below.

Two golden images, one shared disk, one GPU role at a time:

- `vla-training-base` is the portable training image. Packer pulls
  `ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746`.
  It does not load the LeHome tarball and does not bake weights or datasets.
- `lehome-rollout` is the LeHome appliance. Packer downloads
  `lehome-challenge.tar.gz` from `lehome/docker` revision
  `a914115729bb0bfd260971b9c8d4147bff38c1fb`, checks size `26676771349` and
  SHA-256 `1a85e389962909debc4ee9988d8a8c388f905fba60686ef78b1623e6872f7123`,
  then `docker load`s it and builds the four-worker layer.
- Storage is one 500 GiB `NETWORK_SSD` with `forbid_deletion` and Terraform
  `prevent_destroy`.
- Runtime is one preemptible `gpu-rtx6000` / `1gpu-24vcpu-218gb` VM. Training
  and rollout never run together.

Smoke is not a third template. It is the first paid boot of the matching
golden image on that GPU. After training smoke passes, the same VM stays up
for the 2K train.

## Free Validation

From the repository root:

```bash
infrastructure/nebius/validate.sh
```

That script bootstraps pinned Packer 1.11.2 and Terraform 1.5.7 into the
ignored `infrastructure/nebius/.tools/` directory, runs infrastructure
pytest, Packer `fmt`/`validate`, and Terraform `fmt`/`validate`. It prints
paid next commands and never executes them.

## Secrets

| Secret | How it is supplied | Never |
| --- | --- | --- |
| Nebius service-account PEM and key id | Packer vars / `PKR_VAR_*` at image-build time | Baked into images, committed, or stored in Terraform state |
| Nebius IAM / CLI credentials | Operator environment for `terraform apply` | Committed or written into `.tfvars` examples |
| Hugging Face token | Runtime environment on the GPU VM | Baked into images or Packer variables |
| GHCR pull token | Packer builder environment if the trainer digest is private | Committed |
| Private Git access | Runtime only | Staged into `rollout-stage/` |

Copy the `*.tfvars.example` files to gitignored local files. Do not put
tokens, PEM paths, or model hyperparameters in Terraform state.

## Paid Sequence

1. Create a Nebius service account and local PEM. Packer needs
   `project_id`, `subnet_id`, `service_account_id`,
   `service_account_public_key_id`, and `service_account_private_key_file`.
2. Build `vla-training-base` with the on-demand CPU builder
   (`cpu-d3` / `16vcpu-64gb`). This builder is not preemptible.
3. Stage rollout code with `infrastructure/nebius/tools/stage-rollout.sh`,
   then build `lehome-rollout` the same way.
4. Apply `infrastructure/nebius/terraform/storage` to create the 500 GiB
   disk. Record `shared_disk_id`.
5. Apply `infrastructure/nebius/terraform/runtime` with
   `training.tfvars` and the training image id.
6. Run the training smoke on that VM, then the 2K train on the same VM.
7. Stop the training role, detach the disk, apply `rollout.tfvars`, and
   run the four-worker CPU-cloth rollout smoke.

The full operator procedure, recovery rules, winner gate, and deletion
review are in [docs/nebius_training_rollout.md](../../docs/nebius_training_rollout.md).
