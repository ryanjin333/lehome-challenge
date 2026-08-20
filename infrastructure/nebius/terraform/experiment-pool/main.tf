# Stopped-at-rest experiment capacity. This root deliberately never owns the
# rollout disk; the existing rollout runtime retains that exclusive attachment.
terraform {
  required_providers {
    nebius = { source = "nebius/nebius", version = "0.6.42" }
  }
}

provider "nebius" { profile = { name = var.nebius_profile } }

# Controller SQLite state has its own lifecycle and attachment. Replacing the
# disposable controller VM therefore never deletes controller leases, events,
# or publication receipts. This is distinct from, and never names, rollout
# storage.
resource "nebius_compute_v1_disk" "controller_state" {
  parent_id       = var.parent_id
  name            = "lehome-experiment-controller-state"
  type            = "NETWORK_SSD"
  size_gibibytes  = 20
  forbid_deletion = true

  lifecycle {
    prevent_destroy = true
  }
}

resource "nebius_compute_v1_instance" "controller" {
  parent_id = var.parent_id
  name      = "lehome-experiment-controller"
  resources = { platform = "cpu-d3", preset = "4vcpu-16gb" }
  stopped   = true
  boot_disk = { attach_mode = "READ_WRITE", managed_disk = { name = "lehome-experiment-controller-boot", spec = { type = "NETWORK_SSD", size_gibibytes = 20, source_image_id = var.controller_image_id } } }
  secondary_disks = [{
    attach_mode = "READ_WRITE"
    device_id   = "controller-state"
    existing_disk = {
      id = nebius_compute_v1_disk.controller_state.id
    }
  }]
  network_interfaces = [{ name = "eth0", subnet_id = var.subnet_id, ip_address = {} }]
  labels             = { lehome_role = "experiment-controller", lehome_manifest_set_sha256 = var.manifest_set_sha256 }
}

resource "nebius_compute_v1_instance" "training" {
  for_each           = toset(["1", "2"])
  parent_id          = var.parent_id
  name               = "lehome-experiment-training-${each.key}"
  resources          = { platform = "gpu-rtx6000", preset = "1gpu-24vcpu-218gb" }
  stopped            = true
  recovery_policy    = "FAIL"
  preemptible        = { on_preemption = "STOP" }
  boot_disk          = { attach_mode = "READ_WRITE", managed_disk = { name = "lehome-experiment-training-${each.key}-cache", spec = { type = "NETWORK_SSD", size_gibibytes = 300, source_image_id = var.training_image_id } } }
  network_interfaces = [{ name = "eth0", subnet_id = var.subnet_id, ip_address = {} }]
  # Runtime configuration and all credentials are injected only after boot as
  # root-owned local files.  Keeping this empty prevents the controller URL,
  # worker identity, and every secret from becoming Terraform state.
  labels = { lehome_role = "experiment-training", lehome_worker_slot = each.key, lehome_manifest_set_sha256 = var.manifest_set_sha256, lehome_controller_bind = var.controller_bind_address }
}
