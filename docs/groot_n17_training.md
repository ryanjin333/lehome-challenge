# GR00T N1.7 portable training runbook

This runbook separates trusted dataset preparation, disposable GPU training,
and simulator rollout. The trainer needs one CUDA GPU but not Isaac Sim. Every
remote input and output is pinned by an immutable Hugging Face commit and a
local SHA-256 manifest.

## Artifact and credential boundary

Only these private repositories are approved:

- Dataset: `ryanjin333/lehome-groot-n17-data`
- Model artifacts: `ryanjin333/lehome-groot-n17-models`

Creating either repository is an explicit owner operation; see
[`trainer/README.md`](../trainer/README.md). Never create a public fallback.
Provide `HF_TOKEN` only through the current process or rental provider's secret
injection. Do not run `hf auth login`. The token must not enter the image,
command arguments, resolved configuration, child training environment,
credential store, output volume, report, or upload payload.

## One-time trusted-machine dataset flow

Inspect the organizer dataset, convert it deterministically, validate it, and
only then publish its validated allowlist:

```bash
lehome-train data inspect \
  --source /data/four_types_merged \
  --output /prepared/source-inspection.json

lehome-train data convert \
  --source /data/four_types_merged \
  --output /prepared/lehome-groot-n17-v1 \
  --mapping /app/trainer/config/lehome_four_types_mapping.json \
  --source-repository lehome/dataset_challenge_merged \
  --source-revision <full-source-commit> \
  --converter-commit <full-trainer-commit> \
  --container-digest sha256:<64-hex-digest> \
  --groot-root /opt/Isaac-GR00T

lehome-train data validate \
  --dataset /prepared/lehome-groot-n17-v1 \
  --groot-root /opt/Isaac-GR00T

HF_TOKEN='<write-scoped-token>' lehome-train data publish \
  --dataset /prepared/lehome-groot-n17-v1 \
  --repo ryanjin333/lehome-groot-n17-data \
  --revision lehome-groot-n17-v1 \
  --staging-root /prepared/staging \
  --timeout-seconds 30
```

Record the full 40-character Hub commit returned by publish and the SHA-256 of
the prepared `manifest.json`. Rental machines use those two immutable values;
they do not reconvert the source dataset.

### Normalization and modality contract

Normalization is generated only from training episodes. Validation episodes
never contribute statistics.

- `meta/stats.json` describes the 12-dimensional robot state and the
  12-dimensional absolute action stored by the prepared LeRobot dataset.
- `meta/relative_stats.json` contains the GR00T relative-action normalization
  for the 16-step horizon: a `16 x 5` non-gripper joint tensor for each arm.
  Each arm's gripper remains an absolute command.
- `meta/lehome_groot_modality.py` is the checked modality configuration for the
  three RGB cameras, 12 joint states, actions, and language instruction.
- `meta/validation_report.json` records split, schema, finite-value, timestamp,
  statistics, modality, and artifact-hash validation.
- OpenPI `norm_stats.json` is not read, converted, or reused.

The prepared dataset commit carries all statistics, modality configuration,
validation output, and hashes. Every checkpoint also carries the matching
normalization processor and normalization/provenance hashes. A checkpoint is
incompatible if those identities do not match the selected dataset commit and
manifest hash.

## Disposable rental flow

Start from the accepted image by exact `sha256:` digest. Supply the repository
commit, pinned Isaac-GR00T revision, base-model revision, dataset repository
commit, dataset manifest hash, provider start time, and hourly price as resolved
configuration. The accepted image sets `LEHOME_TRAIN_RUNTIME_FACTORY` to its
production `module:factory`. On a development image, pass the same value via
`--runtime-factory module:factory`; absence of a runtime factory is a hard,
actionable failure.

`prepare`, `memorize`, `smoke`, and `train` each accept a strict request envelope:

```json
{
  "schema_version": 1,
  "command": "prepare",
  "arguments": {
    "request_contract": "fields owned by the accepted Task 12 runtime adapter"
  }
}
```

The command identity must match the filename/stage. Unknown envelope fields,
duplicate JSON fields, non-finite values, and supported secret formats are
rejected before the runtime factory is loaded. Run the stages with:

```bash
lehome-train prepare --request /requests/prepare.json
lehome-train memorize --request /requests/memorize.json
lehome-train smoke --request /requests/smoke.json
lehome-train train --request /requests/train.json
lehome-train report --request /requests/report.json
lehome-train sync --request /requests/sync.json
```

The report request schema is fully implemented by this package:

```json
{
  "schema_version": 1,
  "command": "report",
  "arguments": {
    "experiment_config": "/output/experiment/resolved-config.json",
    "isaac_groot_revision": "<full-isaac-groot-commit>",
    "smoke_result": "/output/experiment/reports/smoke-result.json",
    "checkpoint_descriptors": [
      "/output/experiment/checkpoints/step-12000.json"
    ],
    "local_artifact_root": "/output/experiment",
    "sync_result": "/output/evidence/checkpoint-sync-result.json",
    "pruning_receipts": [],
    "instance_started_at": "2026-07-31T10:00:00Z",
    "generated_at": "2026-07-31T14:00:00Z",
    "provider_hourly_price": 1.25,
    "output": "/output/experiment/reports/training-report.json"
  }
}
```

The sync request schema is also fully implemented:

```json
{
  "schema_version": 1,
  "command": "sync",
  "arguments": {
    "experiment_root": "/output/experiment",
    "experiment_id": "<experiment-id>",
    "experiment_config_sha256": "<64-hex-config-hash>",
    "repository": "ryanjin333/lehome-groot-n17-models",
    "revision": "<explicit-upload-branch>",
    "staging_root": "/output/staging",
    "timeout_seconds": 30,
    "max_attempts": 5,
    "output": "/output/evidence/checkpoint-sync-result.json"
  }
}
```

For a pre-sync report, set `sync_result` to `null`; descriptor remote flags are
then recorded only as controller claims and never become verified facts. The
authoritative final flow is:

1. write a local-evidence report with `sync_result: null`;
2. run sync and persist its result outside the mutable experiment tree;
3. regenerate the report with that immutable sync-result path and any strict
   pruning-receipt paths; and
4. sync once more to archive the final evidence-backed report, persisting the
   shutdown-gate sync result outside the experiment tree.

Only the external result from step 4 authorizes shutdown. The report's
`sync_snapshot_disposable` field describes the earlier snapshot used to build
the report, while `shutdown_disposable` remains `false` because the report was
written after that snapshot.

All parsed report and pruning-receipt timestamps are serialized in canonical
UTC `Z` form.

`prepare` verifies one supported GPU, disk, immutable snapshots, local hashes,
and private Hub read/write access. `memorize` is an offline diagnostic and stays
non-promotable until the simulator expert-replay gate passes. `smoke` tests
physical batches sequentially at accumulation 1 and selects the largest stable
batch with at least 10% physical VRAM headroom. `train` processes exactly
768,000 sample presentations and retains a verified resumable checkpoint.

`report` records the exact image digest, trainer repository commit, Isaac-GR00T
revision, base-model repository and revision, prepared-dataset repository
revision and manifest SHA-256, resolved training configuration and its hash,
selected smoke metrics, and complete checkpoint provenance. Each checkpoint
entry includes artifact path/hash/size, optimizer step, sample presentations,
experiment/config/dataset hashes, normalization and schedule hashes, resumable
state, controller-reported claims, evidence-derived local and remote
verification flags, and explicitly labeled retention evidence. Sample
presentations must equal optimizer step times the resolved effective batch
(`physical_batch_size * gradient_accumulation_steps`). A local verified claim
requires opening the artifact beneath `local_artifact_root` and matching its
exact hash and size. A remote verified claim requires an exact path/hash/size
match in a compatible immutable sync result; descriptor booleans alone never
authorize disposal. A pruning claim remains `reported_only` unless a matching
deletion receipt is tied to that remotely verified commit. The report also
includes instance runtime, hourly price, calculated cost,
`sync_snapshot_disposable` for its input sync evidence, and
`shutdown_disposable: false`. Report generation repeats the central secret scan
before writing JSON.

`sync` generates `sync-manifest.json` from these closed artifact groups under
the experiment root:

- checkpoints;
- redacted logs;
- `resolved-config.json`;
- `provenance.json`; and
- JSON reports.

Dotfiles, symlinks, paths outside the experiment, caches, environment files,
credential-store filenames, and supported token formats are rejected. After
access is verified, sync capacity-checks the caller-selected staging filesystem,
byte-copies only the generated allowlist into a temporary immutable snapshot,
and re-scans every staged byte for exact hashes and secrets. It uploads that
snapshot, never the mutable experiment directory, beneath the content-addressed
prefix `experiments/{experiment_id}/{sync_manifest_sha256}/`; the generated
manifest records the exact prefix so multiple experiments can coexist. After
the upload resolves to a full commit, sync lists the complete immutable
repository tree. Unrelated paths outside the prefix are allowed, but any extra
file, unexpected directory, symlink-like entry, special entry, missing file, or
listing failure inside the prefix fails the disposal gate. Sync then cleans
upload staging, capacity-checks the same filesystem for readback, downloads the
allowlist from the same prefix and commit, and recomputes every SHA-256 and byte
size. Temporary upload and readback trees are always removed. An entry becomes
`remotely_verified: true` only after both the remote tree and byte readback
match.

## Shutdown gate

Do not stop, delete, or release the rental merely because upload returned
success. Inspect the external sync result produced by the second, post-report
sync and its final `sync-manifest.json`. The machine is disposable only when
that result says `disposable: true` and every entry is `remotely_verified:
true`. Neither disposal field in the report authorizes shutdown. A missing,
unreadable, or mismatched remote artifact keeps `disposable` false. Instance
deletion remains an explicit provider operation; the trainer never deletes
cloud instances.

## Fresh flywheel restore

For each new iteration, first choose and record all immutable inputs:

1. Select the exact prepared dataset repository commit and manifest SHA-256.
2. Select the exact model repository commit containing the latest promoted
   checkpoint. Promotion means all required offline and simulator gates passed;
   a memorization-only checkpoint is not eligible.
3. Publish the newly approved correction data as an immutable incremental
   rollout shard. Record its revision and hashes rather than mutating the prior
   dataset snapshot.
4. Build a new resolved dataset/config identity that references the baseline
   data plus the selected incremental shard.

There are two restore modes, and the choice must be explicit in provenance:

- **Fine-tune from promoted model weights:** load the latest promoted processor
  and weights, start a new optimizer, and lower the learning rate only when this
  mode was deliberately selected for the new rollout shard. Do not silently
  reset optimizer state or alter the learning rate.
- **Exact training resume:** load the full remotely verified checkpoint,
  including optimizer, scheduler, processor, and sample-presentation state.
  Keep the original schedule and learning rate exactly; all dataset,
  normalization, config, and schedule hashes must match.

After either path, repeat prepare, memorization, smoke, training, reporting,
and immutable sync verification. Always pin both dataset and model revisions;
`latest`, mutable branches, and provider-local volumes are not restoration
sources.
