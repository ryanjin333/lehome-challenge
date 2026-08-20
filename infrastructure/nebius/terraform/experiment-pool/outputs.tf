output "controller_id" {
  value = nebius_compute_v1_instance.controller.id
}

output "controller_state_disk_id" {
  value = nebius_compute_v1_disk.controller_state.id
}

output "training_ids" {
  value = { for slot, instance in nebius_compute_v1_instance.training : slot => instance.id }
}

output "capacity_instance_ids" {
  description = "Write these exact IDs into the root-owned capacity configuration; rollout remains accounting-only and is not a resource/data reference."
  value = {
    training_1 = nebius_compute_v1_instance.training["1"].id
    training_2 = nebius_compute_v1_instance.training["2"].id
    rollout    = var.rollout_instance_id
  }
}

output "training_gpu_capacity" {
  value = length(nebius_compute_v1_instance.training)
}

# Compatibility name for the static dry-run guard.  It is deliberately an
# accounting value, not a reference to the rollout VM or its protected disk.
output "gpu_capacity" {
  value = length(nebius_compute_v1_instance.training) + var.existing_rollout_gpu_capacity
}

output "total_gpu_capacity_including_rollout" {
  value = length(nebius_compute_v1_instance.training) + var.existing_rollout_gpu_capacity
}
