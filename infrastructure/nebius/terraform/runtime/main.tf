# Exactly one preemptible RTX PRO 6000 runtime role at a time, selected by
# active_role. Training and rollout are NOT independent states that could both
# attach the shared disk; a role switch is an explicit stop/detach/attach/apply
# sequence on this single root. Destroying this root never touches the shared
# disk, which lives in the separate storage state.

terraform {
  required_providers {
    nebius = {
      source  = "nebius/nebius"
      version = "0.6.42"
    }
  }
}

provider "nebius" {}

resource "nebius_compute_v1_instance" "runtime" {
  parent_id = var.parent_id
  name      = "lehome-${var.active_role}"

  resources = {
    platform = "gpu-rtx6000"
    preset   = "1gpu-24vcpu-218gb"
  }

  recovery_policy = "FAIL"

  preemptible = {
    on_preemption = "STOP"
  }

  # Disposable boot disk derived from the role's golden image; it is deleted
  # with the instance and rebuilt on every launch.
  boot_disk = {
    attach_mode = "READ_WRITE"
    managed_disk = {
      name = "lehome-${var.active_role}-boot"
      spec = {
        type            = "NETWORK_SSD"
        size_gibibytes  = 128
        source_image_id = var.image_id
      }
    }
  }

  # The protected shared workspace disk, attached by stable id with a stable
  # device alias so the guest service mounts /dev/disk/by-id/virtio-lehome.
  secondary_disks = [
    {
      attach_mode = "READ_WRITE"
      device_id   = "lehome"
      existing_disk = {
        id = var.shared_disk_id
      }
    }
  ]

  network_interfaces = [
    {
      name       = "eth0"
      subnet_id  = var.subnet_id
      ip_address = {}
      public_ip_address = {
        static = false
      }
    }
  ]

  labels = {
    lehome_role            = var.active_role
    lehome_manifest_sha256 = var.manifest_sha256
  }
}

output "instance_id" {
  value = nebius_compute_v1_instance.runtime.id
}

output "instance_state" {
  value = nebius_compute_v1_instance.runtime.status.state
}

output "manifest_uri" {
  value = var.manifest_uri
}
