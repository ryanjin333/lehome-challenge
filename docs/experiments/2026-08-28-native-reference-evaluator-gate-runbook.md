# Native public-reference evaluator gate

This is an isolated compatibility gate, not a collection or training command.
It evaluates the public GR00T N1.5 checkpoint through the public submission's
native LeRobot 0.4.3 path before any ordinary N1.7 result is used for a
decision.

Do not start collection or training unless this gate completes with the exact
published `7/8` oracle and its result has been published and read back.

## Fixed contract

- Public source: `theo-zhou/lehome-groot-submission-4` at
  `d384fe00508acd96ab1c3c5dc265e08261f94b3b`.
- Source byte-tree SHA-256:
  `eada9f80b0dda1428177fe4551efa8059fe85845d4db5b32bb673f88a50c6bb2`.
- Policy path: the cached, unmodified `pretrained_model/`. The launcher pins
  the exact seven-file set: `config.json`, `model.safetensors`, both processor
  JSON files, both processor sidecars, and `train_config.json`. It verifies
  every SHA-256 (including model
  `d8d91b6c11cb5aa18fe9a48e7da88eae0ec7e5a227a315ba67ee167d645cde76`),
  rejects extra files and symlinks, and never downloads weights. Do not use an
  adapted checkpoint view or edit any checkpoint file.
- Runtime: the existing local `lehome-rollout:build` image, whose inspected
  immutable ID must be
  `sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7`.
  Its `/opt/lehome-challenge/.venv/bin/python` supplies Python 3.11, LeRobot
  0.4.3, torch/CUDA, CPU simulation, and CUDA policy inference on the one
  existing rollout VM. The saved processor must prove a 16-action horizon and
  12-D absolute bimanual action output.
- There are eight sequential 600-step episodes, all at seed 42:
  `Top_Long_Seen_0` twice, `Top_Short_Seen_0` twice,
  `Pant_Long_Seen_0` twice, then `Pant_Short_Seen_0` twice. The required
  result is seven successes: every attempt except the first pant-short one.
- If the first two top-long attempts are not both successes, stop after those
  two. That is a typed evaluator-compatibility stop, not an admission to run
  the other six attempts.

## Read-only identity check, then start the exact VM

Use read-only provider checks to confirm the exact existing rollout VM and
protected disk. The launcher itself does not create, start, stop, resize, or
delete provider resources, build an image, or upload to any service.

After that read-only identity check passes, the operator starts exactly
`computeinstance-u00t6xfqhadrcmssa2`; do not create a replacement or second
VM. Only after that exact VM reaches `RUNNING` may the operator stage source
and run cache inventory on it. Both operations are zero-evaluation admission
steps: they happen before the CUDA probe, simulator launch, or first episode.
This keeps the paid boundary to one existing VM and only the time needed for
the bounded gate; stop that exact VM after publication/readback or any gate
failure.

The operator must provide all inputs explicitly. Cache roots are on the VM
disk, never the local Mac. The provider's RUNNING/attached-disk state is an
external read-only gate; this launcher records the actual provider source-image
identity, inspected local runtime-image identity, and container
Python/LeRobot/CUDA facts, and never asks the provider to change state.

## Exact reviewed container boundary

Do not run the launcher on the bare VM host: that host has no admissible
LeRobot/Isaac runtime. Run the reviewed checkout inside the already-built image
with `--pull never` and `--gpus all`. The launcher still owns `--device cpu`, so
CUDA is used only for policy inference. Do not build or pull an image and do
not run another Isaac container in parallel.

Stage the exact reviewed Git revision at
`/mnt/lehome/runtime-code/<exact-reviewed-40-hex-revision>` on the protected
disk and export that revision to the wrapper. The wrapper validates that the
directory name equals `HEAD`, requires a clean tree, safely creates only the
four ignored `Assets/` mountpoint directories, and mounts the whole revision
read-only at the same path inside the container. The image's existing virtual
environment remains visible at `/opt/lehome-challenge/.venv/bin/python`.
Mount each canonical asset source directory read-only twice: once beneath the
canonical cache root and once beneath the reviewed runtime repository's
`Assets/`. The launcher proves each pair has the same device and inode before
preflight and before every simulator stage.

Capture the local image receipt on the VM host before any container execution:

```bash
export LEHOME_NATIVE_REFERENCE_RUNTIME_REVISION='<exact-reviewed-40-hex-revision>'
reviewed_runtime_checkout="/mnt/lehome/runtime-code/$LEHOME_NATIVE_REFERENCE_RUNTIME_REVISION"
runtime_image_receipt=/mnt/lehome/reference-native/runtime-image-receipt.json
python3 "$reviewed_runtime_checkout/scripts/verify_native_reference_evaluator_gate.py" \
  capture-runtime-image --receipt "$runtime_image_receipt"
```

Use only the executable host wrapper
`rollout_appliance/run_native_reference_evaluator_container.sh` for all four
modes. It constructs the complete Docker argv, explicitly places every
mode-specific `--env KEY=value` before the immutable image ID, applies the
reviewed runtime and dual authenticated-asset mounts, overrides the image
entrypoint with `bash`, and invokes the VM-local launcher only inside the
container. Do not invoke `run_native_reference_evaluator_gate.sh` on the bare
host. `--print-command` displays the exact shell-escaped argv without running
Docker.

The admitted checkpoint and metadata roots are the already-staged live cache
paths `/mnt/lehome/cache/reference-theo-d384fe0/repo/pretrained_model` and
`/mnt/lehome/cache/reference-theo-d384fe0/repo/dataset_meta`. The wrapper does
not substitute empty `reference-native` paths or download either cache.

Inspecting the fixed tag proves the operator's expected reference still names
that image; the wrapper launches by immutable ID to close the gap between
inspection and container creation.

The image receipt is required only for execution. `source-stage`,
`inventory-cache`, and validation-only remain zero-evaluation paths and do not
require that receipt, probe CUDA, or launch simulation.

Once the exact VM is running, first stage only the public evaluator source
into an empty source root. This
uses sparse Git with `GIT_LFS_SKIP_SMUDGE=1`, so it fetches no checkpoint
weight or simulator run:

```bash
bash rollout_appliance/run_native_reference_evaluator_container.sh source-stage
```

Before validation/execution, create and publish one immutable cache-trust
manifest from a zero-evaluation inventory on that running exact VM. The
launcher reads the manifest bytes anonymously and directly from the fixed public dataset
`ryanjin333/lehome-groot-n17-rollouts` at an immutable 40-hex revision and a
`reference-checks/...` path; it does not trust a local copy or a caller-supplied
digest. The manifest kind is
`lehome_native_reference_cache_trust_manifest_v2` and it binds the public
repository/path plus the observed checkpoint, metadata, and assets tree digests;
the immutable fetch URL itself binds the 40-hex revision. The checkpoint digest
is a logical file-name/SHA manifest: the
large model is fully SHA-256 verified at preflight, before every one of the
four stages, and once more after stage 4. If this exact readback does not yet
exist, create it with the zero-evaluation `inventory-cache` mode and publish
that one manifest rather than inventing a caller-supplied trust value.

```bash
export LEHOME_NATIVE_REFERENCE_OUTPUT_ROOT=/mnt/lehome/reference-native/native-reference-YYYYMMDDHHMMSS
export LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_REVISION='<published 40-char revision>'
export LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_PATH='reference-checks/native-cache-YYYYMMDDHHMMSS/cache-trust-manifest.json'
export LEHOME_NATIVE_REFERENCE_PROVIDER_RUNNING_RECEIPT=/mnt/lehome/reference-native/provider-running-receipt.json
export LEHOME_NATIVE_REFERENCE_RUNTIME_IMAGE_RECEIPT=/mnt/lehome/reference-native/runtime-image-receipt.json
```

### Cache inventory and immutable publication

On the now-running exact rollout VM, inventory only the already-cached source,
checkpoint, metadata, and assets. Before emitting any inventory digest, the
launcher authenticates every metadata file against the pinned public model
revision and every file under `objects/`, `robots/`, `scenes/`, and `textures/`
against the pinned public asset-dataset revision. This offline check rejects
mutations, extras, and symlinks. Inventory mode does not probe CUDA, launch the
simulator, or run an episode. The output path must be new; the remote path must
be a new `native-cache-*` prefix so the public manifest is immutable.

```bash
export LEHOME_NATIVE_REFERENCE_CACHE_MANIFEST_OUTPUT=/mnt/lehome/reference-native/cache-trust-manifest.json
export LEHOME_NATIVE_REFERENCE_CACHE_MANIFEST_PATH='reference-checks/native-cache-YYYYMMDDHHMMSS/cache-trust-manifest.json'
bash rollout_appliance/run_native_reference_evaluator_container.sh inventory-cache
```

Copy that one manifest to the operator host through a validated target, then
publish it there. The command uploads only this manifest, obtains the returned
immutable revision, and anonymously fetches its exact bytes before writing the
readback receipt.

```bash
scp -o ClearAllForwardings=yes -o BatchMode=yes -- \
  operator@validated-rollout-host:/mnt/lehome/reference-native/cache-trust-manifest.json \
  /operator/received/cache-trust-manifest.json
python3 scripts/verify_native_reference_evaluator_gate.py publish-cache-manifest \
  --manifest /operator/received/cache-trust-manifest.json \
  --token-file /operator/secrets/hf-token \
  --receipt /operator/received/cache-manifest-readback.json
```

Use `immutable_revision` and `path` from that receipt as
`LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_REVISION` and
`LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_PATH` for validation/execution.
The launcher anonymously fetches those bytes again; it does not accept the
operator's local manifest or readback receipt as cache trust.

The launcher computes cache digests itself and compares them to that immutable
public readback; it does not trust caller-provided cache-digest values. The
operator uses the restricted exact-instance Nebius adapter to capture a fresh
`RUNNING` observation before VM-local validation. The launcher never needs
Nebius credentials: it validates and copies that operator receipt into the
evidence bundle. The observation validates the pinned VM name/ID, protected
disk, and provider source-image ID, then records the nested CLI response hash.
It also proves `torch.cuda.is_available()`, records CUDA runtime/device count, and
binds all identities to `identity.json` and `preflight.json`. Simulation remains
CPU-only.

Capture RUNNING locally, validate the target VM path, then transfer the exact
receipt before VM-local validation. Do not write an operator-local command
directly to `/mnt/lehome`.

```bash
operator_receipt="$(mktemp -d)/provider-running-receipt.json"
python3 scripts/verify_native_reference_evaluator_gate.py capture-provider \
  --state RUNNING \
  --receipt "$operator_receipt"
scp -o ClearAllForwardings=yes -o BatchMode=yes -- "$operator_receipt" \
  operator@validated-rollout-host:/mnt/lehome/reference-native/provider-running-receipt.json
```

Run validation first:

```bash
bash rollout_appliance/run_native_reference_evaluator_container.sh validate-only
```

Validation refuses missing caches, an unpinned or changed source checkout,
wrong LeRobot version, a stale cache trust receipt, non-CPU simulation,
non-CUDA policy, unsafe paths, or an existing output root. It performs no
source download, simulator execution, CUDA probe, or external write in
validation-only mode.

## Approved execution only

After the provider reports the exact VM running and the validation result is
clean, execute only on that VM:

```bash
bash rollout_appliance/run_native_reference_evaluator_container.sh execute
```

Execution copies the exact public cache-manifest bytes, fresh provider
`RUNNING` receipt, and fresh host-captured runtime-image receipt into
`evidence/`. It cross-checks the exact `lehome-rollout:build` reference and
immutable image ID before CUDA or simulator access, then writes every raw stage log, exactly
three expected RGB videos (`left`, `right`, `top`) per episode in the matching
`success` or `failure` directory, and every
per-attempt receipt as regular immutable files. The result verifier SHA-256s
and sizes each of those files before it emits `execution-receipt.json`. Native
exceptions, known non-finite/cloth-flight/missing-cloth/safety text, an absent
video, a wrong/extra video directory, or any changed artifact fail immediately.
The public log parser accepts the canonical `Episode 1/2: ... Success=True`
lines only. A typed top-long or oracle-mismatch stop still writes its receipt,
then exits with status `3`. It does not invoke the ordinary N1.7 gateway.
The public evaluator runs as the pinned `scripts.eval` module from the reviewed
runtime working directory. `PYTHONDONTWRITEBYTECODE=1` prevents source-tree
bytecode, while reviewed `sitecustomize` redirects the evaluator's project logs
to `OUTPUT_ROOT/public-runtime/stage-N`; the complete pinned source is rehashed
before every stage and after stage 4.

## After execution

An exact local `7/8` result is only
`oracle_matched_pending_finalization`, never final `passed`. Before finalizing,
produce three independently reviewed JSON receipts, all bound to the execution
receipt SHA-256:

1. `lehome_native_reference_fidelity_review_v1`: explicit review for all eight
   attempts (manual video audit is acceptable) that cloth was present and that
   no cloth flight, non-finite state, or safety failure occurred. Include a
   digest of the exact per-attempt receipt/log/three-video manifest from the
   execution receipt for every attempt; arbitrary audit digests are rejected.
2. `lehome_native_reference_hf_readback_v2`: produced only by the Hugging Face
   publication command after it uploads the closed bundle to a new
   `reference-checks/native-...` prefix and anonymously lists/downloads every
   entry at the returned immutable revision. It records that revision, manifest
   digest, and publication time.
3. `lehome_native_reference_provider_observation_v1`: run the provided
   read-only command after stopping the VM to obtain an actual Nebius CLI
   `STOPPED` observation. It must bind the exact VM/disk/provider-source-image
   identity and nested provider-response hash. Do not hand-author this JSON.

First publish/read back from the operator host after it has received the
complete execution bundle and holds its owner-only HF token file:

```bash
python3 scripts/verify_native_reference_evaluator_gate.py publish-bundle \
  --bundle-root /operator/received/native-reference-YYYYMMDDHHMMSS \
  --execution /operator/received/native-reference-YYYYMMDDHHMMSS/execution-receipt.json \
  --token-file /operator/secrets/hf-token \
  --receipt /operator/received/publication-readback.json
```

Only after that successful readback, stop the VM and capture STOPPED locally;
the receipt must be newer than publication and no more than 15 minutes old:

```bash
python3 scripts/verify_native_reference_evaluator_gate.py capture-provider \
  --state STOPPED \
  --receipt /operator/received/provider-stopped-receipt.json
```

Then run the pure local finalizer:

```bash
python3 scripts/verify_native_reference_evaluator_gate.py finalize \
  --execution /operator/received/native-reference-YYYYMMDDHHMMSS/execution-receipt.json \
  --fidelity /operator/received/fidelity-review.json \
  --publication /operator/received/publication-readback.json \
  --stopped /operator/received/provider-stopped-receipt.json \
  --receipt /operator/received/final-gate-receipt.json
```

Only that receipt can have `status: passed`.

Any typed stop, missing artifact, mismatch, invalid physics outcome, failed
readback, or provider discrepancy keeps curriculum collection, success replay,
and training stopped. Preserve the local VM bundle for diagnosis; do not retry
using the ordinary evaluator or make the oracle easier.
