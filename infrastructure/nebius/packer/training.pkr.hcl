# Portable GR00T training golden image (vla-training-base).
#
# This uses a temporary, on-demand CPU builder. The Nebius Packer plugin does
# not expose a preemptible-builder setting, so this builder is explicitly NOT
# preemptible; it is deleted as soon as Packer captures the image. The
# resulting golden image is what the preemptible training runtime boots from.

source "nebius-image" "vla-training-base" {
  communicator = "ssh"
  ssh_username = "ubuntu"

  service_account {
    private_key_file = var.service_account_private_key_file
    public_key_id    = var.service_account_public_key_id
    account_id       = var.service_account_id
  }

  # Boot disk for the temporary builder only; disposable after capture.
  disk {
    size_gibibytes = 64
    type           = "NETWORK_SSD"
  }

  base_image {
    family = "ubuntu24.04-driverless"
  }

  network {
    associate_public_ip_address = true
  }

  instance {
    platform = "cpu-d3"
    preset   = "16vcpu-64gb"
  }

  image {
    name                        = var.training_image_name
    version                     = var.image_version
    image_family                = "vla-training-base"
    image_family_human_readable = "VLA Training Base"
  }

  parent_id = var.project_id
}

build {
  name    = "vla-training-base"
  sources = ["source.nebius-image.vla-training-base"]

  provisioner "file" {
    source      = "${path.root}/scripts/install-common.sh"
    destination = "/tmp/install-common.sh"
  }

  provisioner "file" {
    source      = "${path.root}/scripts/install-training.sh"
    destination = "/tmp/install-training.sh"
  }

  provisioner "file" {
    source      = "${path.root}/../guest"
    destination = "/tmp/lehome-guest"
  }

  provisioner "shell" {
    inline = [
      "chmod +x /tmp/install-common.sh /tmp/install-training.sh",
      "sudo -E /tmp/install-common.sh",
      "sudo -E TRAINING_OCI_IMAGE='${var.training_oci_image}' TRAINING_OCI_DIGEST='${var.training_oci_image}' TRAINER_CODE_REVISION='${var.trainer_code_revision}' /tmp/install-training.sh",
      "rm -f /tmp/install-common.sh /tmp/install-training.sh",
    ]
  }
}
