# Experiment-pool private bootstrap

Run this only after Terraform has created the stopped controller and two
training VMs, the controller state disk is mounted, and the immutable manifest
materialization has been readback-verified at
`/var/lib/lehome/controller/manifests`. The bootstrap does not upload, modify,
or reseal that manifest set.

From an operator machine with SSH access to the CPU controller, two training
VMs, and the existing rollout VM, use existing
local `0600` files for the controller bearer token and Hugging Face token. Never paste
either value into a shell command, Terraform variable, Packer input,
cloud-init payload, or chat.

```bash
infrastructure/nebius/tools/bootstrap-experiment-pool.sh \
  --controller-ssh ubuntu@CONTROLLER_ADMIN_HOST \
  --controller-ip CONTROLLER_RFC1918_IPV4 \
  --manifest-set-sha256 MANIFEST_SET_SHA256 \
  --controller-token-file /secure/path/controller-token \
  --hf-token-file /secure/path/hf-token \
  --nebius-private-key-file /secure/path/capacity-service-account.pem \
  --nebius-service-account-id SERVICE_ACCOUNT_ID \
  --nebius-public-key-id AUTHORIZED_PUBLIC_KEY_ID \
  --nebius-project-id PROJECT_ID \
  --rollout ubuntu@ROLLOUT_ADMIN_HOST \
  --promotion-matrix /secure/path/eval_groot_n17_unseen20_dev.json \
  --promotion-matrix-sha256 UNSEEN20_SOURCE_SHA256 \
  --final-matrix /secure/path/eval_groot_n17_public_280.json \
  --final-matrix-sha256 PUBLIC280_SOURCE_SHA256 \
  --promotion-baseline-evidence /secure/path/original-12k-unseen20.json \
  --promotion-baseline-evidence-sha256 BASELINE_EVIDENCE_SHA256 \
  --deployment-gate /secure/path/experiment-pool-deployment-gate.json \
  --deployment-gate-sha256 DEPLOYMENT_GATE_SHA256 \
  --final-report-repository OWNER/PRIVATE_REPORT_REPOSITORY \
  --worker 1=ubuntu@TRAINER_1_ADMIN_HOST \
  --worker 2=ubuntu@TRAINER_2_ADMIN_HOST
```

It generates a short-lived private CA locally, installs a TLS-only Nginx proxy
on the controller's exact RFC1918 address, and writes root-owned `0600`
environment and source-token files on the target VMs. The controller also gets
a dedicated, least-privilege Nebius service-account private key. Systemd copies
the root-only sources into per-service private runtime credentials; the capacity
daemon creates an isolated CLI profile from that key and passes its exact
root-owned config on every Compute request. It never uses an operator profile,
home-directory config, or VM metadata token. The script verifies
TLS health from every consumer, authenticated controller capacity from the
controller, the exact `lehome-experiment-training-1` and
`lehome-experiment-training-2` identities, and the rollout evaluator's distinct
pinned unseen-20 promotion matrix, unseen-80 final matrix, and original-12K
paired baseline evidence. The rollout environment starts in `promotion` mode.
The Hugging Face credential is installed up front, but no finalist-specific
seen-regression result is invented before finalists exist.

The two matrix arguments are authenticated source artifacts. Before any file
is copied to the rollout VM, bootstrap freezes them into canonical JSON arrays:
exactly 20 promotion trials (5/category) and the 80 `public_unseen` final
trials (20/category). It rejects duplicates, overlap between the two sets,
category imbalance, or an incompatible challenge envelope, then installs the
hashes of those frozen list-form artifacts. The paid evaluator never receives
the original envelope or the 200 seen rows.

The deployment gate is a separate immutable, readback-verified admission
record. It binds the exact READY successor controller, training, and rollout
image IDs to their already-created instance IDs, the training OCI digest
baked into both worker images, and the exact 40-character training-code
revision from each root-owned training-image manifest. Bootstrap validates
this record before installing any ready marker. Every marker contains the gate
SHA-256, and the root-owned capacity configuration rechecks the immutable gate
bytes before every possible start/stop reconciliation. A missing, mutable, or
changed gate therefore results in zero provider actions even if queue demand
exists.

Schema v1 additionally requires an accepted zero-perturbation
teacher-continuation probe and is full recovery admission. Schema v2 separates
that recovery-only condition: `recovery_admission.accepted` carries the same
teacher probe, Hub sync receipt, and no-strict-seal proof; alternatively,
`recovery_admission.unavailable` records a safe reason and immutable failure
receipt. The latter is sufficient to run the independent ordinary A/B/C
training and normal fixed-matrix evaluation. It is not permission to run
controlled recovery collection or any recovery-dependent D--G arm. Those stay
forbidden until a new immutable v2 gate records accepted recovery admission.
Bootstrap installs that exact gate on the rollout VM and gives its path and
SHA-256 to the evaluator environment. The controlled-recovery smoke and
four-worker campaign wrappers revalidate it before starting a base campaign.

The job manifests, deployment gate, and both baked training-image manifests
must carry the same code revision. A legacy gate without
`training_code_revision`, or a host built with a different revision, is
intentionally rejected before the controller admits a job or a worker sends
its first lease heartbeat. Rebuild/reissue those immutable artifacts together;
do not edit a captured image or bypass the gate.

At success, the script deliberately leaves the controller proxy, controller,
training workers, and rollout evaluator disabled. That is the safe at-rest
state. The ready marker permits
an explicit start only after the campaign operator has reviewed the immutable
manifest and capacity budget:

```bash
# Controller first; its private proxy starts as a dependency.
sudo systemctl enable --now lehome-experiment-controller.service

# The capacity daemon is deliberately a separate, explicit activation. Its
# only remote Compute operations are get/start/stop for the three IDs frozen in
# the root-owned capacity config. It refuses to start without its systemd
# service-account credential and isolated CLI config.
sudo systemctl enable --now lehome-experiment-capacity.service

# Start either available trainer independently. They do not wait for each
# other, so a completed worker can immediately lease the next experiment.
sudo systemctl enable --now lehome-experiment-worker.service

# On the existing rollout VM, ordinary fixed-matrix evaluation may run for
# A/B/C under a v2 unavailable-recovery gate. Controlled recovery collection
# and D--G remain forbidden until recovery admission is accepted:
sudo systemctl enable --now lehome-experiment-evaluator.service
```

Final evaluation is not only a mode switch. After the controller admits its
finalists, each finalist must first complete its seen-regression check. For
each result, use the exact controller experiment ID and checkpoint publication
receipt to create an immutable handoff:

```bash
sudo python3 /opt/lehome/scripts/materialize_finalist_seen_regression_handoff.py \
  --root /mnt/lehome/experiment-pool/evaluation/seen-regression-handoffs \
  --experiment-id FINALIST_EXPERIMENT_SHA256 \
  --checkpoint-receipt-sha256 FINALIST_CHECKPOINT_RECEIPT_SHA256 \
  --evidence /secure/readback/FINALIST-seen-regression.json
```

The evidence must already be sealed/readback-verified and bind the same
checkpoint receipt. The command writes only
`<root>/<experiment-id>/<checkpoint-receipt>.json`; the evaluator derives that
path from its authenticated lease and rejects a shared or differently bound
receipt. Once every admitted finalist has a handoff, seal the handoff root,
then change the root-owned evaluator mode and explicitly start it:

```bash
sudo chmod 0555 /mnt/lehome/experiment-pool/evaluation/seen-regression-handoffs
sudo sed -i 's/^LEHOME_EVALUATION_MODE=.*/LEHOME_EVALUATION_MODE=final-unseen80/' \
  /etc/lehome/experiment-evaluator.env
sudo systemctl enable --now lehome-experiment-evaluator.service
```

Final mode selects the separately pinned unseen-80 matrix. It never reuses the
unseen-20 promotion matrix. Do not run promotion and final evaluation modes
simultaneously on the one rollout GPU.

The capacity daemon is a systemd-only service. Do not invoke the Python entry
point manually: it requires the generated root-owned CLI config and systemd
credential directory, neither of which is an operator-shell credential.

```bash
sudo systemctl status lehome-experiment-capacity.service
sudo journalctl -u lehome-experiment-capacity.service --since '-15 min'
```

The TLS proxy never binds a public address and only forwards to the fixed,
private controller port. If bootstrap verification fails, it removes the
controller and rollout ready markers and disables the controller/proxy and
evaluator; do not bypass that failure by starting a worker manually.
