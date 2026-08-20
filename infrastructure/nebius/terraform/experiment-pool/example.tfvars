# Non-secret examples only. All instances are defined stopped at rest.
nebius_profile          = "default"
parent_id               = "project-id"
subnet_id               = "subnet-id"
controller_image_id     = "controller-image-id"
training_image_id       = "training-image-id"
controller_bind_address = "10.0.0.2"
manifest_set_sha256     = "0000000000000000000000000000000000000000000000000000000000000000"
# Accounting and root-owned capacity configuration only. It creates no
# Terraform resource/data binding and never references the protected disk.
rollout_instance_id = "computeinstance-u00rv6yj0m1m7jen5q"
