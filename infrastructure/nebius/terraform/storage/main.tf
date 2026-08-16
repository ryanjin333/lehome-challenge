# One deletion-protected 500 GiB shared workspace disk, owned by its own
# Terraform state. Runtime roots only ever ATTACH this disk by id; destroying
# a runtime VM can never destroy it. Removal requires separately reviewed
# removal of prevent_destroy.

terraform {
  required_providers {
    nebius = {
      source  = "nebius/nebius"
      version = "0.6.42"
    }
  }
}

provider "nebius" {}

resource "nebius_compute_v1_disk" "shared_workspace" {
  parent_id      = var.parent_id
  name           = "lehome-shared-workspace"
  type           = "NETWORK_SSD"
  size_gibibytes = 500

  forbid_deletion = true

  lifecycle {
    prevent_destroy = true
  }
}
