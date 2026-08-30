# Official LeHome policy comparison

This is the clean comparison boundary for the original N1.7 12K policy and
the public N1.5 competitor. It runs the organizer's source at
`a805ad2f7ab52a4583066fc4ee5180459a7f9d15` with the public asset checkout at
`bea65fd960ad5a1bb3bd3fa77164b28001c08ef9`. Both checkouts are mounted
read-only. The harness revalidates their Git revisions and complete tree
digests before and after execution. It never edits organizer source, scoring,
or canonical assets.

The simulator is CPU (`--device cpu`). CUDA is policy inference and rendering
only; cloth physics is not CUDA. Both policies use seed 42, 600 maximum steps,
two episodes per garment, the same task text, and the same native Release
lists. The full matrix is 12 garments in each of four categories and 96
episodes per policy.

## Required cached inputs

Prepare these without starting a provider resource:

- clean official Git checkout at the exact source revision;
- clean Hugging Face asset Git checkout at the exact asset revision;
- N1.7 12K checkpoint, its existing exact checkpoint-identity receipt, and its
  policy metadata;
- the untouched competitor checkpoint;
- the existing native-reference compatibility artifacts produced by
  `scripts/verify_native_reference_evaluator_gate.py prepare-checkpoint-compatibility`:
  the sanitized config root and `lehome_native_reference_checkpoint_compatibility_v1`
  receipt. The receipt binds the original absolute checkpoint and sanitized
  paths, so both directories are mounted at those exact same absolute paths
  inside the evaluator container. The view omits only `num_decay_steps` and
  `decay_lr_ratio`; the checkpoint is unchanged;
- immutable rollout and N1.7 policy image IDs.

The competitor evaluator reuses the already-reviewed native-reference runtime
boundary exactly: PEFT 0.18.1 is a digest-checked zipimport overlay, and the
pinned FlashAttention 2.8.3, dm-tree 0.1.9, qwen-vl-utils 0.0.14, and
torchdiffeq 0.2.5 wheels are mounted read-only and installed offline/no-deps
only inside the disposable evaluator container. The established wheel
validators and CUDA/import/API probes must all pass. The evaluator interpreter
and `PYTHONEXE` are both `/opt/lehome-challenge/.venv/bin/python`; the receipt
binds all six native-reference dependency/backend receipts pre/post.

The runtime checkout must be clean at the exact
`LEHOME_OFFICIAL_RUNTIME_REVISION`. The wrapper permits only rollout image
`sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7`
and trainer RepoDigest
`ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746`.
It records fresh Docker image inspections, a CUDA probe, the authenticated
policy-server readiness receipt, and a token-free startup log captured through
authenticated readiness. The receipt does not claim that startup snapshot is a
complete lifetime log. Runtime tree and exact adapter hashes are rechecked
after evaluation.

Category-specific `${category}_merged` metadata is intentional and matches the
established official/native competitor construction path. It is authenticated
LeRobot construction metadata, not policy observation data: no garment or
category label is sent through the DockerPolicy HTTP bridge. The receipt binds
the whole metadata tree and each of the four category-root tree digests.

The required `LEHOME_OFFICIAL_*` paths are visible at the top of
`rollout_appliance/run_official_lehome_comparison_container.sh`. Every path is
an already-present cache. This boundary does not download weights, build an
image, create a VM, or start a VM.

## Smoke first

Use a fresh output path and run:

```bash
export LEHOME_OFFICIAL_OUTPUT_ROOT=/mnt/lehome/evaluations/official-smoke-YYYYMMDDHHMMSS
bash rollout_appliance/run_official_lehome_comparison_container.sh smoke
```

The smoke creates an external, read-only one-line custom asset-list view for
`Top_Long_Seen_0`. The canonical asset checkout remains unchanged. It runs two
episodes for each policy. Continue only when the receipt status is `valid`,
both policies have two completed outcomes, source/asset pre/post identities
match, and every expected retained video is decodable. Any traceback,
nonfinite value, transport/process failure, missing completion sentinel,
outcome drift, video ambiguity, or source/asset drift is
`infrastructure_invalid`. A clean completed `Success=False` episode is a
policy failure, not infrastructure failure.

The successful smoke is sealed with `execution-manifest.json`, a receipt SHA
companion, and `status.json`. Preserve that bundle. Full mode requires
`LEHOME_OFFICIAL_SMOKE_RECEIPT` to point to its sealed
`comparison-receipt.json`; the harness revalidates the complete smoke seal,
requires exactly two valid outcomes per policy, and identity-matches source,
assets, reviewed runtime/adapters, images, checkpoints, metadata, scorer,
reference matrix, and fixed evaluator configuration. Smoke mode rejects this
variable so a stale prerequisite cannot be silently accepted.

## Full native Release comparison

After the smoke is valid, use another fresh output path:

```bash
export LEHOME_OFFICIAL_OUTPUT_ROOT=/mnt/lehome/evaluations/official-full-YYYYMMDDHHMMSS
export LEHOME_OFFICIAL_SMOKE_RECEIPT=/mnt/lehome/evaluations/official-smoke-YYYYMMDDHHMMSS/comparison-receipt.json
bash rollout_appliance/run_official_lehome_comparison_container.sh full
```

The full run invokes the untouched official evaluator once per native category
for each policy. Each category log must contain the organizer's exact garment
order, 24 episode records, and one completion sentinel. The final receipt has
96 ordered outcomes per policy, the scorer digest, exact commands, a parity
digest, source/assets identities, and artifact hashes.

At completion, the harness writes an immutable execution manifest covering
every execution payload artifact. The manifest digest and comparison-receipt
digest are cross-bound by the SHA companion and valid status file. Added,
removed, changed, unsafe, or unlisted files invalidate publication.

The organizer video helper reuses `episode0` and `episode1` filenames for all
12 garments in a category. Therefore files are overwritten during an official
category invocation. The receipt explicitly labels the six retained videos
per category as representatives of the final garment only; they are not
claimed as per-episode videos. A fresh video directory is used for every
policy/category command. The final garment's logged success decides which
status path is authoritative. An opposite-status file left by an earlier
garment is explicitly recorded as stale overwrite evidence and is never
treated as the final garment's video.

## Explicit publication and anonymous readback

Execution never uploads automatically. Publication is explicit and is allowed
only for a reviewed, valid **full** comparison receipt:

```bash
python3 scripts/run_official_lehome_comparison.py publish \
  --receipt /mnt/lehome/evaluations/official-full-YYYYMMDDHHMMSS/comparison-receipt.json \
  --repository ryanjin333/lehome-groot-n17-rollouts \
  --token-env HF_TOKEN \
  --publication-receipt /mnt/lehome/evaluations/official-full-YYYYMMDDHHMMSS-publication.json
```

The publisher revalidates the receipt SHA companion, status, execution
manifest, exact file set, and every local byte. It derives the only permitted
fresh remote prefix from the receipt and manifest digests, rejects an existing
prefix, commits exactly the sealed publication set, then anonymously lists the
whole prefix. The remote set is the manifest payload plus
`execution-manifest.json`, `comparison-receipt.sha256.json`, and `status.json`,
so a reader can independently validate the complete seal. The publisher
requires that exact set before downloading and hashing every byte. If an
earlier commit succeeded but its response timed out, a retry recovers only
when anonymous listing and readback prove the complete exact bytes; any
missing, extra, or drifted file is rejected and is never overwritten. A
successful upload response without anonymous byte readback and
anonymous file-set verification is not publication success.

## Cleanup and next decision

The appliance owns and terminates the N1.7 policy server and HTTP bridge on
success, failure, or interruption. It intentionally contains no Nebius API
call. After valid publication/readback, or immediately after any
`infrastructure_invalid` stop, use the separate provider controller to stop the exact VM
and verify its `STOPPED` state. Do not create a replacement VM.

The 1,000-rollout collection remains off. Do not start it until the official
smoke and full public comparison are valid, published, anonymously read back,
and the result has been reviewed.
