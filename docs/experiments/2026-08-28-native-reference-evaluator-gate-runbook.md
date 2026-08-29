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
- Runtime: CPU simulation and CUDA policy inference on the one existing
  rollout VM. The saved processor must prove a 16-action horizon and 12-D
  absolute bimanual action output.
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
identity plus host-Python/LeRobot/CUDA facts, and never asks the provider to
change state.

Once the exact VM is running, first stage only the public evaluator source
into an empty source root. This
uses sparse Git with `GIT_LFS_SKIP_SMUDGE=1`, so it fetches no checkpoint
weight or simulator run:

```bash
export LEHOME_NATIVE_REFERENCE_SOURCE_ROOT=/mnt/lehome/reference-native/source
LEHOME_NATIVE_REFERENCE_MODE=source-stage \
  bash rollout_appliance/run_native_reference_evaluator_gate.sh
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
export LEHOME_NATIVE_REFERENCE_VM_ID=computeinstance-u00t6xfqhadrcmssa2
export LEHOME_NATIVE_REFERENCE_DISK_ID=computedisk-u00pbe55crxy7jr56x
export LEHOME_NATIVE_REFERENCE_SOURCE_ROOT=/mnt/lehome/reference-native/source
export LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT=/mnt/lehome/reference-native/pretrained_model
export LEHOME_NATIVE_REFERENCE_METADATA_ROOT=/mnt/lehome/reference-native/dataset_meta
export LEHOME_NATIVE_REFERENCE_ASSETS_ROOT=/mnt/lehome/challenge-assets/Assets
export LEHOME_NATIVE_REFERENCE_OUTPUT_ROOT=/mnt/lehome/reference-native/native-reference-YYYYMMDDHHMMSS
export LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_REVISION='<published 40-char revision>'
export LEHOME_NATIVE_REFERENCE_CACHE_TRUST_MANIFEST_PATH='reference-checks/native-cache-YYYYMMDDHHMMSS/cache-trust-manifest.json'
export LEHOME_NATIVE_REFERENCE_PROVIDER_RUNNING_RECEIPT=/mnt/lehome/reference-native/provider-running-receipt.json
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
LEHOME_NATIVE_REFERENCE_MODE=inventory-cache \
  bash rollout_appliance/run_native_reference_evaluator_gate.sh
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
LEHOME_NATIVE_REFERENCE_VALIDATE_ONLY=1 \
  bash rollout_appliance/run_native_reference_evaluator_gate.sh
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
LEHOME_NATIVE_REFERENCE_VALIDATE_ONLY=0 \
  bash rollout_appliance/run_native_reference_evaluator_gate.sh
```

Execution copies the exact public cache-manifest bytes and fresh provider
`RUNNING` receipt into `evidence/`, then writes every raw stage log, exactly
three expected RGB videos (`left`, `right`, `top`) per episode in the matching
`success` or `failure` directory, and every
per-attempt receipt as regular immutable files. The result verifier SHA-256s
and sizes each of those files before it emits `execution-receipt.json`. Native
exceptions, known non-finite/cloth-flight/missing-cloth/safety text, an absent
video, a wrong/extra video directory, or any changed artifact fail immediately.
The public log parser accepts the canonical `Episode 1/2: ... Success=True`
lines only. A typed top-long or oracle-mismatch stop still writes its receipt,
then exits with status `3`. It does not invoke the ordinary N1.7 gateway.

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
