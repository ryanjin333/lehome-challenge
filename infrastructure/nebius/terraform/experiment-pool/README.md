# LeHome async experiment pool

This root creates exactly one stopped CPU controller and two stopped,
preemptible RTX PRO 6000 training VMs. Their 300 GiB boot disks are separate
disposable caches. It does not create, attach, name, or read the rollout VM or
its protected shared disk.

The pool has no cloud-init data. After a VM is started, an operator injects a
non-secret environment file and separately owned `0600` token files locally.
Neither belongs in Terraform variables, state, image layers, logs, or Packer
inputs.

Workers must use an `https://` controller URL. The controller binds the exact
private `controller_bind_address`; an operator-managed TLS reverse proxy on
that private network terminates TLS and forwards only to that bind address.
Do not use a wildcard bind, a public listener, or plaintext `http://` URL. The
guest wrappers reject those configurations, so an absent proxy fails closed.

`rollout_instance_id` is required so the Terraform output can be copied into
the root-owned capacity configuration alongside the two generated training IDs.
It is an accounting-only identity: this root has no rollout resource or data
source, and never references the protected rollout disk.

`total_gpu_capacity_including_rollout` is an accounting-only dry-run value:
two training GPUs plus the separately managed one-GPU rollout appliance.

The controller image pins the official Nebius CLI and gives its capacity daemon
a dedicated service-account credential through systemd. The daemon first builds
an isolated root-owned CLI profile locally, then invokes only these remote
Nebius Compute commands for those exact three IDs: `compute instance get`,
`start`, and `stop`. It has no create, replacement, deletion, browser, or
broad-list command path. The generated CLI config and the capacity config must
remain root-owned and mode `0600`; do not substitute an ID from an untrusted
runtime request or rely on a user/default CLI profile.
