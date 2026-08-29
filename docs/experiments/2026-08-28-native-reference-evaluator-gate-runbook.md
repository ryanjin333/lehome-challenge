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
- Policy path: the cached, unmodified `pretrained_model/`.  Its config and
  primary model digests are verified by the launcher. Do not use the adapted
  reference-checkpoint view, edit `config.json`, or re-download model weights.
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

## Preflight while stopped

Use read-only provider checks to confirm the exact existing rollout VM and
protected disk. The launcher itself does not create, start, stop, resize, or
delete provider resources, build an image, or upload to any service.

The operator must provide all inputs explicitly. Cache roots are on the VM
disk, never the local Mac. `SOURCE_ROOT` may be empty for the initial source
stage; the other three roots must already contain trusted caches.

```bash
export LEHOME_NATIVE_REFERENCE_VM_ID=computeinstance-u00t6xfqhadrcmssa2
export LEHOME_NATIVE_REFERENCE_DISK_ID=computedisk-u00pbe55crxy7jr56x
export LEHOME_NATIVE_REFERENCE_IMAGE='registry/example@sha256:<64-lowercase-hex>'
export LEHOME_NATIVE_REFERENCE_SOURCE_ROOT=/mnt/lehome/reference-native/source
export LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT=/mnt/lehome/reference-native/pretrained_model
export LEHOME_NATIVE_REFERENCE_METADATA_ROOT=/mnt/lehome/reference-native/dataset_meta
export LEHOME_NATIVE_REFERENCE_ASSETS_ROOT=/mnt/lehome/challenge-assets/Assets
export LEHOME_NATIVE_REFERENCE_OUTPUT_ROOT=/mnt/lehome/reference-native/native-reference-YYYYMMDDHHMMSS
export LEHOME_NATIVE_REFERENCE_SOURCE_TREE_SHA256=eada9f80b0dda1428177fe4551efa8059fe85845d4db5b32bb673f88a50c6bb2
export LEHOME_NATIVE_REFERENCE_CHECKPOINT_TREE_SHA256='<trusted 64-char digest>'
export LEHOME_NATIVE_REFERENCE_METADATA_TREE_SHA256='<trusted 64-char digest>'
export LEHOME_NATIVE_REFERENCE_ASSETS_TREE_SHA256='<trusted 64-char digest>'
```

The three non-source tree digests must come from previously recorded trusted
cache manifests, not a value invented during this invocation. Checkpoint
preflight verifies the fixed public `config.json`, `model.safetensors`, saved
preprocessor, saved postprocessor, and processor schema independently.

Run validation first:

```bash
LEHOME_NATIVE_REFERENCE_VALIDATE_ONLY=1 \
  bash rollout_appliance/run_native_reference_evaluator_gate.sh
```

Validation refuses missing caches, an unpinned source checkout, changed source
files, wrong LeRobot version, non-CPU simulation, non-CUDA policy, a stale
digest, unsafe paths, or an existing output root. It performs no source
download in validation-only mode.

## Approved execution only

After the provider reports the exact VM running and the validation result is
clean, execute only on that VM:

```bash
LEHOME_NATIVE_REFERENCE_VALIDATE_ONLY=0 \
  bash rollout_appliance/run_native_reference_evaluator_gate.sh
```

The initial execution can sparse-checkout only missing public source/config
files at the pinned revision with `GIT_LFS_SKIP_SMUDGE=1`; it never requests
`pretrained_model` weights. It records stage logs, videos, per-attempt
immutable receipts, preflight evidence, `result.json`, and `gate-receipt.json`
under the new output root. It does not invoke the ordinary N1.7 gateway.

## After execution

For a `passed` receipt, publish the entire untouched output directory below a
new immutable `reference-checks/native-...` prefix on Hugging Face, then
independently read back every file and record the immutable revision and
readback receipt. Only after that receipt exists should the operator stop the
exact rollout VM and record the stopped provider state.

Any typed stop, missing artifact, mismatch, invalid physics outcome, failed
readback, or provider discrepancy keeps curriculum collection, success replay,
and training stopped. Preserve the local VM bundle for diagnosis; do not retry
using the ordinary evaluator or make the oracle easier.
