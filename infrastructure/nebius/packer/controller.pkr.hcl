# CPU-only controller image: no Isaac, checkpoints, datasets, or credentials.
source "nebius-image" "lehome-experiment-controller" {
  communicator = "ssh"
  ssh_username = "ubuntu"
  service_account {
    private_key_file = var.service_account_private_key_file
    public_key_id    = var.service_account_public_key_id
    account_id       = var.service_account_id
  }
  disk {
    size_gibibytes = 20
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
    preset   = "4vcpu-16gb"
  }
  image {
    name                        = "lehome-experiment-controller"
    version                     = var.image_version
    image_family                = "lehome-experiment-controller"
    image_family_human_readable = "LeHome Experiment Controller"
  }
  parent_id = var.project_id
}
build {
  sources = ["source.nebius-image.lehome-experiment-controller"]
  provisioner "file" {
    source      = "${path.root}/scripts/install-controller.sh"
    destination = "/tmp/install-controller.sh"
  }
  provisioner "file" {
    source      = "${path.root}/../guest"
    destination = "/tmp/lehome-guest"
  }
  provisioner "file" {
    source      = "${path.root}/../../../scripts/run_lehome_experiment_controller.py"
    destination = "/tmp/run_lehome_experiment_controller.py"
  }
  provisioner "file" {
    source      = "${path.root}/../../../scripts/run_lehome_capacity_lifecycle.py"
    destination = "/tmp/run_lehome_capacity_lifecycle.py"
  }
  provisioner "file" {
    source      = "${path.root}/../../../trainer/src/lehome_train"
    destination = "/tmp/lehome_train"
  }
  provisioner "shell" {
    inline = [
      "sudo useradd --system --home /var/lib/lehome/controller --shell /usr/sbin/nologin lehome-controller || true",
      "sudo chmod +x /tmp/install-controller.sh",
      "sudo /tmp/install-controller.sh",
    ]
  }
}
