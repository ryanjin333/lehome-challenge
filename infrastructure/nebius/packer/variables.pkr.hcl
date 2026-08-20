# Credentials arrive only through environment variables (PKR_VAR_*) or the
# operator's Packer var-file at build time. They are never defaulted,
# committed, or baked into an image.

variable "project_id" {
  type        = string
  description = "Nebius project that owns temporary builders and golden images."
}

variable "subnet_id" {
  type        = string
  description = "Subnet for temporary builders; builders are deleted after capture."
}

variable "service_account_id" {
  type        = string
  description = "Service account used by temporary builders."
}

variable "service_account_public_key_id" {
  type        = string
  description = "Public key id of the builder service account."
}

variable "service_account_private_key_file" {
  type        = string
  description = "Host-local path to the builder private key; never uploaded."
  sensitive   = true
}

variable "ghcr_pull_token" {
  type        = string
  description = "Optional GHCR pull token used only during image build; never baked."
  sensitive   = true
  default     = ""
}

variable "image_version" {
  type        = string
  description = "Version label stamped into golden image metadata."
}

variable "training_image_name" {
  type    = string
  default = "vla-training-base"
}

variable "rollout_image_name" {
  type    = string
  default = "lehome-rollout"
}

variable "rollout_parent_image_id" {
  type        = string
  description = "Existing READY rollout image used for fast code-only rebuilds."
  default     = ""
}

variable "rollout_code_revision" {
  type        = string
  description = "Exact Git revision embedded in an incremental rollout OCI layer."
}

variable "training_oci_image" {
  type    = string
  default = "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746"
}

variable "training_oci_digest" {
  type        = string
  description = "Bare SHA-256 digest expected for the pinned trainer image."
  default     = "b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746"
}

variable "trainer_code_revision" {
  type        = string
  description = "Exact trainer code revision baked as image metadata, not source."
}
