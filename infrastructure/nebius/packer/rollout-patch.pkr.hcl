# Fast code-only rollout rebuild based on an existing READY rollout image.
# The parent already contains the verified LeHome tarball, Docker base layers,
# Isaac prerequisites, NVIDIA drivers, DCGM, and guest services. This builder
# updates only the staged runtime code and recaptures a versioned image.

source "nebius-image" "lehome-rollout-patch" {
  communicator = "ssh"
  ssh_username = "ubuntu"

  service_account {
    private_key_file = var.service_account_private_key_file
    public_key_id    = var.service_account_public_key_id
    account_id       = var.service_account_id
  }

  disk {
    size_gibibytes = 192
    type           = "NETWORK_SSD"
  }

  base_image {
    id = var.rollout_parent_image_id
  }

  network {
    associate_public_ip_address = true
  }

  instance {
    platform = "cpu-d3"
    preset   = "16vcpu-64gb"
  }

  image {
    name                        = var.rollout_image_name
    version                     = var.image_version
    image_family                = "lehome-rollout"
    image_family_human_readable = "LeHome Rollout Appliance"
  }

  parent_id = var.project_id
}

build {
  name    = "lehome-rollout-patch"
  sources = ["source.nebius-image.lehome-rollout-patch"]

  provisioner "file" {
    source      = "${path.root}/scripts/install-rollout-patch.sh"
    destination = "/tmp/install-rollout-patch.sh"
  }

  provisioner "file" {
    source      = "${path.root}/../rollout-stage"
    destination = "/tmp/lehome-repo"
  }

  provisioner "shell" {
    environment_vars = [
      "LEHOME_ROLLOUT_CODE_REVISION=${var.rollout_code_revision}",
    ]
    inline = [
      "chmod +x /tmp/install-rollout-patch.sh",
      "sudo -E /tmp/install-rollout-patch.sh",
      "rm -f /tmp/install-rollout-patch.sh",
    ]
  }
}
