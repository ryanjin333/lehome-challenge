# LeHome-specific rollout golden image (lehome-rollout).
#
# Like the training template this uses a temporary, on-demand CPU builder and
# is therefore explicitly NOT preemptible. It downloads the exact official
# challenge tarball, verifies byte length and SHA-256 before docker load,
# builds the derived four-worker layer, then cleans the tarball and caches so
# the captured image carries runtime layers only. The boot disk is sized for
# the 26.7 GB tarball plus loaded and derived Docker layers.

variable "challenge_repository" {
  type    = string
  default = "lehome/docker"
}

variable "challenge_revision" {
  type    = string
  default = "a914115729bb0bfd260971b9c8d4147bff38c1fb"
}

variable "challenge_size" {
  type    = number
  default = 26676771349
}

variable "challenge_sha256" {
  type    = string
  default = "1a85e389962909debc4ee9988d8a8c388f905fba60686ef78b1623e6872f7123"
}

variable "challenge_url" {
  type    = string
  default = "https://huggingface.co/datasets/lehome/docker/resolve/a914115729bb0bfd260971b9c8d4147bff38c1fb/lehome-challenge.tar.gz"
}

source "nebius-image" "lehome-rollout" {
  communicator = "ssh"
  ssh_username = "ubuntu"

  service_account {
    private_key_file = var.service_account_private_key_file
    public_key_id    = var.service_account_public_key_id
    account_id       = var.service_account_id
  }

  # Large temporary boot disk: tarball + loaded layers + derived layer.
  disk {
    size_gibibytes = 160
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
    name                        = var.rollout_image_name
    version                     = var.image_version
    image_family                = "lehome-rollout"
    image_family_human_readable = "LeHome Rollout Appliance"
  }

  parent_id = var.project_id
}

build {
  name    = "lehome-rollout"
  sources = ["source.nebius-image.lehome-rollout"]

  provisioner "file" {
    source      = "${path.root}/scripts/install-common.sh"
    destination = "/tmp/install-common.sh"
  }

  provisioner "file" {
    source      = "${path.root}/scripts/install-rollout.sh"
    destination = "/tmp/install-rollout.sh"
  }

  provisioner "file" {
    source      = "${path.root}/../guest"
    destination = "/tmp/lehome-guest"
  }

  # Runtime code for the derived layer. This copies the repository checkout
  # staged by the operator; no model weights or datasets are included.
  provisioner "file" {
    source      = "${path.root}/../rollout-stage"
    destination = "/tmp/lehome-repo"
  }

  provisioner "shell" {
    inline = [
      "chmod +x /tmp/install-common.sh /tmp/install-rollout.sh",
      "sudo -E /tmp/install-common.sh",
      "sudo -E CHALLENGE_REPOSITORY='${var.challenge_repository}' CHALLENGE_REVISION='${var.challenge_revision}' CHALLENGE_SIZE='${var.challenge_size}' CHALLENGE_SHA256='${var.challenge_sha256}' CHALLENGE_URL='${var.challenge_url}' /tmp/install-rollout.sh",
      "rm -f /tmp/install-common.sh /tmp/install-rollout.sh",
    ]
  }
}
