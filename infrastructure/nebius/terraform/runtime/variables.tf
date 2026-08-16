variable "parent_id" {
  type        = string
  description = "Nebius project that owns the runtime instance."
}

variable "subnet_id" {
  type        = string
  description = "Subnet for the runtime instance network interface."
}

variable "active_role" {
  type        = string
  description = "Which golden image boots now. Switching roles is an explicit stop/detach/attach/apply."

  validation {
    condition     = contains(["training", "rollout"], var.active_role)
    error_message = "active_role must be training or rollout; exactly one role runs at a time."
  }
}

variable "image_id" {
  type        = string
  description = "Golden image id for the active role (vla-training-base or lehome-rollout)."
}

variable "shared_disk_id" {
  type        = string
  description = "Id of the protected shared workspace disk from the storage state."
}

variable "manifest_uri" {
  type        = string
  description = "Immutable experiment/campaign manifest URI for the active role."
}

variable "manifest_sha256" {
  type        = string
  description = "SHA-256 of the immutable manifest; recorded as an instance label."

  validation {
    condition     = can(regex("^[0-9a-f]{64}$", var.manifest_sha256))
    error_message = "manifest_sha256 must be a 64-char lowercase hex digest."
  }
}
