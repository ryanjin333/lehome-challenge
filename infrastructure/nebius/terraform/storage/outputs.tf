output "shared_disk_id" {
  value       = nebius_compute_v1_disk.shared_workspace.id
  description = "Attach this exact disk to whichever runtime role currently leases it."
}
