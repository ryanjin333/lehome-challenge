# LeHome GR00T N1.7 trainer

This isolated Python 3.10 package prepares the organizer demonstrations and
runs NVIDIA GR00T N1.7 behavior-cloning training without Isaac Sim. Simulation
and rollout evaluation remain on a separate LeHome-capable machine.

The pinned environment is installed from this directory:

```bash
uv sync --locked
uv run lehome-train --help
```

Remote data is stored only in the approved private dataset repository
`ryanjin333/lehome-groot-n17-data`. Checkpoints, reports, redacted logs, and
provenance are stored only in the approved private model repository
`ryanjin333/lehome-groot-n17-models`. Remote operations require `HF_TOKEN` in
the current process environment. The trainer passes it explicitly to Hub calls,
does not invoke `hf auth login`, does not put it in a child environment, and
does not write it to a credential cache, report, manifest, or upload.

Repository creation is never implicit. An owner can explicitly create or
verify either approved private repository with the library API:

```bash
HF_TOKEN='<write-scoped-token>' uv run python - <<'PY'
from lehome_train.hub import (
    HuggingFaceHubTransport,
    ensure_approved_private_repository,
)

transport = HuggingFaceHubTransport(timeout_seconds=30.0)
for repository in (
    "ryanjin333/lehome-groot-n17-data",
    "ryanjin333/lehome-groot-n17-models",
):
    ensure_approved_private_repository(
        transport=transport,
        repository=repository,
        create=True,
        timeout_seconds=30.0,
    )
PY
```

Use `create=False` to verify existing repositories without creating them. Any
unapproved repository or repository that is not private is rejected.

Hydrate the immutable base model on a fresh host only beneath `/cache`, using
the exact approved revision and a caller-owned staging directory:

```bash
HF_TOKEN='<read-scoped-token>' lehome-train model retrieve \
  --destination /cache/models/groot-n17 \
  --repo nvidia/GR00T-N1.7-3B \
  --revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --staging-root /cache/staging
```

The complete operator workflow, request schemas, normalization contract,
restore choices, and shutdown gate are in
[the GR00T N1.7 training runbook](../docs/groot_n17_training.md).

The accepted Task 12 image sets
`LEHOME_TRAIN_RUNTIME_FACTORY=lehome_train.groot.production_runtime:create`.
Outside that image, pass `--runtime-factory module:factory` explicitly. The
factory must return the production adapter implementing `prepare`, `memorize`,
`smoke`, and `train`; missing GPU/runtime wiring is a hard error, never a
successful no-op.

The production adapter uses closed command-specific schemas. The exact
`arguments` keys are:

- `prepare`: `launch_config`, `experiment_config`, `model_snapshot_manifest`,
  `dataset_snapshot_manifest`, `network_measurement`, `artifact_repository`,
  `artifact_revision`, and `status_output`;
- `memorize`: `launch_config`, `experiment_config`,
  `dataset_manifest_sha256`, `requested_episode_id`, `result_output`, and
  `status_output`;
- `smoke`: `launch_config`, `experiment_config`, `report_output`,
  `selected_result_output`, and `status_output`; and
- `train`: `launch_config`, `experiment_config`, `selected_smoke_result`,
  `normalization_sha256`, `estimated_checkpoint_bytes`,
  `checkpoint_repository`, `checkpoint_revision`, `resume_checkpoint`,
  `provider_hourly_price`, `instance_start_time`, `result_output`, and
  `status_output`.

A launch config is the JSON form of `FineTuneLaunchConfig`. All referenced
paths must be below `/cache`, `/prepared`, or `/output`; the model and dataset
must already be downloaded at their manifest-verified revisions. Unknown or
missing keys fail closed. The runbook below gives complete request examples.

`launch_config.output_dir` is the canonical experiment and sync root. A
successful production `prepare` creates immutable `resolved-config.json` and
`provenance.json` there, plus the secret-checked `logs/prepare.json`. Production
`train` writes `logs/train.json` from its terminal controller result; it records
only allowlisted identities, status, and exposure, never request data or the
process token. Provenance keeps the operator-facing launch experiment name as
`experiment_id` and labels the restart-safe content hash separately as
`preflight_experiment_id`. The resolved config remains the exact
`ExperimentConfig` JSON so
the report command can parse it directly. The content-addressed child reported
by `prepare` is internal preflight evidence, not the sync root.

Every stage consumes a checked JSON request:

```bash
lehome-train prepare --request /requests/prepare.json
lehome-train memorize --request /requests/memorize.json
lehome-train smoke --request /requests/smoke.json
lehome-train train --request /requests/train.json
lehome-train restore \
  --sync-result /output/evidence/final-sync-result.json \
  --destination /output/restored-experiment \
  --staging-root /output/restore-staging
lehome-train report --request /requests/report.json
lehome-train sync --request /requests/sync.json
```

Final reports verify checkpoint bytes beneath the request's
`local_artifact_root` and accept remote-verification claims only from the
immutable sync-result artifact written by `sync`. Controller flags remain
reported claims; they do not make an artifact verified or disposable.
`sync_snapshot_disposable` describes only the sync evidence consumed while
building the report, and `shutdown_disposable` is always `false`. Only the
external result from the second, post-report sync can authorize shutdown.

Run the focused report and synchronization checks with:

```bash
uv run pytest tests/test_report.py tests/test_sync.py -q
```

These tests use injected runtimes and an in-memory transport. Creating
repositories or performing a real upload additionally requires a valid
`HF_TOKEN`; no token is bundled in this checkout.

Model sync stores each immutable artifact set beneath the manifest-recorded
`experiments/{experiment_id}/{sync_manifest_sha256}/` prefix and verifies the
complete target subtree at the resolved commit before permitting local cleanup.

## Immutable image build and structural verification

From a clean repository root:

```bash
trainer/scripts/build-image.sh
trainer/scripts/verify-image.sh
```

The build uses only `linux/amd64`, the digest-pinned CUDA 12.8.1 base, Python
3.10.18, uv 0.8.22 with a checked archive hash, the frozen trainer and upstream
locks, and Isaac-GR00T commit
`23ace64f17aa5015259b8609d371eb61a357c776`. The default tag is the full local
Git commit. A dirty checkout is rejected unless `ALLOW_DIRTY=1` is explicitly
used for a non-release diagnostic build.

The trainer lock is exported into the final GR00T virtual environment, with
shared exact dependencies aligned to the upstream lock, and `uv pip check`
rejects an incompatible composite runtime. Ubuntu packages are resolved only
from Canonical's timestamped `20260701T000000Z` Jammy snapshot. The build gives
APT a dedicated source list and disables all inherited source-part files, so
the CUDA base image's floating NVIDIA repository cannot participate in package
resolution. Separate builds are still not claimed to be byte-identical; only
the published OCI digest identifies a release candidate.

Every container invocation must mount writable `/cache`, `/prepared`, and
`/output` directories. The image runs as `trainer`, rejects `hf auth login` and
`huggingface-cli login`, and removes `HF_TOKEN` from every command except the
controller-owned remote operations `prepare`, `train`, `restore`,
`data retrieve`, `data publish`, `model retrieve`, and `sync`. Those
controllers use only the explicit process token for Hub operations; GR00T and
Git subprocesses independently receive a scrubbed child environment. Models,
datasets, outputs, and temporary state therefore remain on those mounts rather
than in the immutable image.

On a Linux NVIDIA host, the additional structural gate verifies exactly one
visible GPU and performs one real AdamW optimizer step over a synthetic
`TensorDataset`:

```bash
CUDA_VISIBLE_DEVICES=0 trainer/scripts/verify-image.sh --gpu <image>@sha256:<digest>
```

This one-step gate is not the fresh-machine release acceptance below.

## Fresh RTX PRO 6000 release acceptance

`release-manifest.example.json` is deliberately `unreleased`: it has neither a
claimed OCI digest nor GPU evidence. Do not use an image for paid training until
the registry digest and the following evidence have been recorded in a strict
manifest and that accepted manifest has been committed.

On a fresh Linux x86_64 RTX PRO 6000 96 GB rental:

1. Measure download bandwidth and require at least 1 Gbps. Record the result.
2. Pull the exact `ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:<digest>`
   and record a positive pull duration in seconds. Never accept a tag as
   experiment identity.
3. Mount fresh `/cache`, `/prepared`, and `/output` volumes. Inject `HF_TOKEN`
   only into the controller commands that retrieve data or the model, prepare,
   train/upload, restore, publish, or sync. The controller uses it for remote
   permission/readback checks while GR00T and Git children remain scrubbed.
4. Measure from the start of the fresh-machine procedure to the first real
   GR00T optimizer step. It must be no more than 1,800 seconds.
5. Complete the offline one-episode `memorize` gate.
6. Run physical batches 16, 32, and 64 sequentially with 100 steps per config;
   retain the telemetry and any proven CUDA OOM evidence.
7. Start, or resume only from a checkpoint whose dataset, normalization,
   configuration, schedule, and predecessor identity all verify, the selected
   768,000-sample-presentation training run. This is a full-state
   schedule/exposure resume, not a bit-exact sample-order resume: pinned GR00T
   restores model, optimizer, scheduler, and RNG state but sets
   `ignore_data_skip=True` and reseeds its dataset from the global step.
8. Sync and hash-verify the checkpoint, redacted logs, configuration, and
   report. Record the safe absolute HTTPS evidence URL, exact repository
   commit, actual `trainer/uv.lock` SHA-256, GR00T commit, model revision, base
   digest, full OCI digest, timing, hardware, memorization result, batch
   sequence, and 768k start/resume result in the release manifest. Also bind it
   to the private dataset repository, its resolved 40-character Hub commit,
   the prepared dataset manifest SHA-256, and the train-only normalization
   artifact SHA-256. Validate it with `load_release_manifest` before commit.

This macOS host can run the CPU structural checks and build an amd64 image with
Docker emulation, but it cannot execute or claim the NVIDIA GPU acceptance.
