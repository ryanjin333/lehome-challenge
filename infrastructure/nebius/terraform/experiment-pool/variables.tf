variable "nebius_profile" { type = string }
variable "parent_id" { type = string }
variable "subnet_id" { type = string }
variable "controller_image_id" { type = string }
variable "training_image_id" { type = string }
variable "controller_bind_address" {
  type        = string
  description = "Exact private controller address used by the separately operated TLS proxy; never a wildcard or public hostname."
  validation {
    condition     = can(regex("^[0-9a-fA-F:.]+$", var.controller_bind_address)) && !contains(["0.0.0.0", "::", "127.0.0.1", "localhost"], var.controller_bind_address)
    error_message = "controller_bind_address must be an explicit non-loopback private interface address."
  }
}
variable "manifest_set_sha256" { type = string }
variable "existing_rollout_gpu_capacity" {
  type        = number
  description = "Accounting-only external rollout appliance capacity; this pool never references its VM or protected disk."
  default     = 1
  validation {
    condition     = var.existing_rollout_gpu_capacity == 1
    error_message = "The approved experiment envelope assumes exactly one separately managed rollout GPU."
  }
}

variable "rollout_instance_id" {
  type        = string
  description = "Exact, separately operated rollout VM identity for capacity-accounting and root-owned lifecycle configuration. This root never reads or manages it."
  validation {
    condition     = can(regex("^computeinstance-[A-Za-z0-9]+$", var.rollout_instance_id))
    error_message = "rollout_instance_id must be an exact Nebius Compute instance ID."
  }
}
