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
lehome-train data inspect --source /data/four_types_merged
lehome-train data convert --source /data/four_types_merged --output /prepared/lehome-groot-n17-v1
lehome-train data validate --dataset /prepared/lehome-groot-n17-v1
lehome-train data publish --dataset /prepared/lehome-groot-n17-v1 --repo ryanjin333/lehome-groot-n17-data --revision lehome-groot-n17-v1
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
configuration. Then run:

```bash
lehome-train prepare
lehome-train memorize --episode-id <training-episode-id>
lehome-train smoke --batches auto --steps 100
lehome-train train --sample-presentations 768000
lehome-train report
lehome-train sync
```

`prepare` verifies one supported GPU, disk, immutable snapshots, local hashes,
and private Hub read/write access. `memorize` is an offline diagnostic and stays
non-promotable until the simulator expert-replay gate passes. `smoke` tests
physical batches sequentially at accumulation 1 and selects the largest stable
batch with at least 10% physical VRAM headroom. `train` processes exactly
768,000 sample presentations and retains a verified resumable checkpoint.

`report` records the exact image digest, trainer repository commit, Isaac-GR00T
revision, base-model repository and revision, prepared-dataset repository
revision and manifest SHA-256, resolved training configuration and its hash,
selected smoke metrics, every checkpoint path/hash/size, instance runtime,
hourly price, and calculated cost. Report generation repeats the central secret
scan before writing JSON.

`sync` generates `sync-manifest.json` from these closed artifact groups under
the experiment root:

- checkpoints;
- redacted logs;
- `resolved-config.json`;
- `provenance.json`; and
- JSON reports.

Dotfiles, symlinks, paths outside the experiment, caches, environment files,
credential-store filenames, and supported token formats are rejected. Sync
uploads only the generated entries to the private model repository, resolves
the upload to a full commit, downloads the same entries from that immutable
commit, and recomputes every SHA-256 and byte size. An entry becomes
`remotely_verified: true` only after this readback matches.

## Shutdown gate

Do not stop, delete, or release the rental merely because upload returned
success. Inspect the final `sync-manifest.json` and sync status. The machine is
disposable only when the sync result says `disposable: true` and every entry is
`remotely_verified: true`. A missing, unreadable, or mismatched remote artifact
keeps `disposable` false. Instance deletion remains an explicit provider
operation; the trainer never deletes cloud instances.

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
