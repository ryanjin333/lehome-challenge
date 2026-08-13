# Persistent corrective GR00T training

This run trains one sealed 70/30 corrective generation only: organizer/original
BC frames are 70%, accepted corrective RFT frames are 30%, and the corrective
snapshot remains horizon 16 while the GR00T model capacity is 40.  The legacy
chunked fixed-exposure `train` action remains the rollback path.

## Free preparation

Use only local evidence first:

```bash
PYTHONPATH=source/lehome:trainer/src uv run python \
  scripts/run_groot_persistent_training.py materialize --request materialize-request.json
PYTHONPATH=source/lehome:trainer/src uv run python \
  scripts/run_groot_persistent_training.py prepare --request prepare-request.json
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_flywheel_mix.py trainer/tests/test_throughput_tuning.py \
  trainer/tests/test_continuous_training.py trainer/tests/test_groot_config.py \
  trainer/tests/test_groot_launch.py trainer/tests/test_production_adapters.py \
  trainer/tests/test_train.py trainer/tests/test_production_runtime.py \
  trainer/tests/test_release_manifest.py trainer/tests/test_persistent_training_lifecycle.py
python3 -m py_compile \
  trainer/src/lehome_train/groot/config.py \
  trainer/src/lehome_train/groot/throughput_tuning.py \
  trainer/src/lehome_train/groot/continuous_training.py \
  scripts/run_groot_persistent_training.py
git diff --check
```

`materialize` is free and builds the generation itself with the canonical
`build_mix_plan` and `materialize_mixed_snapshot` functions from a verified
organizer root plus verified accepted corrective roots. It writes the sibling
sealed receipt; it does not accept a hand-written receipt. `prepare` verifies
the supplied local sibling generation receipt is sealed
before recording the exact organizer source (`lehome/dataset_challenge_merged`
at `17e8dee8fac294ffd21d250501d3b31bf8679042`, `four_types_merged`) and its
verified mirror/manifest, plus the private corrective revision and prefix. It
does not contact HF or Vast. Do not reuse the old horizon-40 mix tarball.

Verify a generation before use:

```python
from lehome_train.flywheel.mix import verify_generation
verify_generation('/prepared/generation')
```

## Paid gate and operation

Provider actions are dry-run unless `--execute` is explicit. `capture-offers`
uses the raw interruptible search and accepts only actual `RTX PRO 6000 WS` or
`RTX PRO 6000 S` one-GPU 96,000MB-plus offers. Rent only one
interruptible RTX PRO 6000 96GB at less than $1/hour when fresh instance plus
storage/volume account total is at most $1/hour. Run a capability receipt with
the exact image digest, CUDA/Torch CUDA, compute capability, a finite optimizer
step, and NVML telemetry. A newer Blackwell driver is accepted only by that
training capability gate; rollout driver policy is separate.

The two phases are deliberately separate. `bootstrap-canary --execute` may
rent only the historical structurally pinned image
`ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746`
for one bounded optimizer-step capability command. Its receipt binds the
capability result to the instance, image, and provider readback. Full
`tune`/`train`/`resume` actions require that exact instance-bound receipt;
they cannot promote an arbitrary digest or a receipt from another rental.
Use the explicit `promote --execute` action to recover and validate the receipt's bound SSH
instance before staging; it never rents a replacement instance.

If a staged train/resume SSH connection fails, the lifecycle performs a fresh
Vast instance readback.  Only `interrupted`, `terminated`, `stopped`, `offline`,
or an absent instance produces a `provider_interrupted` terminal; trainer/code
failures remain failures and cannot be resumed as a preemption.  Start an
explicit replacement bootstrap canary, promote its new instance-bound
capability receipt, then run `replacement-resume --execute` with that receipt
and the interruption terminal.  It selects the latest immutable readback
publication and refuses a reused instance or changed generation/config IDs.

Offer capture uses Vast `--storage 300`; the resulting `dph_total` is the
single all-in 300GB quote.  Any reported storage breakdown is retained as
evidence but is never added a second time to the account-wide `$1/hour` gate.

Stage verifies a complete local sealed-generation tree after SCP, safely
extracts the code bundle beneath `/prepared/code` and the approved parent
archive beneath `/cache/parent`, and keeps terminal/status artifacts beneath
`/output`. The staged trainer process is invoked with an explicit code
`PYTHONPATH`; its environment unsets `HF_TOKEN`. The chmod-600 token file is
read only by the asynchronous publisher parent and is never passed in the
trainer command line or trainer environment.
Stage uses strict port-bound SSH/SCP, transfers the complete sealed generation,
receipt, clean code bundle, parent artifact, config/modality, and token file,
then SHA-256-reads every remote file before a later tune/train action.

Measure loader workers at batch 64, then measure batches 64/96/128. Record all
outcomes and stop increasing batch after OOM. Production remains one GPU,
physical/global batch 64, one 2,000-step official process, saving at 1,000 and
2,000. Batches 96/128 are measurements, not a promotion.

Use `lehome-train continuous-train` only with the sealed generation, exact
parent digest, normalization/config identity, horizon 16/model capacity 40,
and no unseen source. It starts one official process and an observer that only
snapshots checkpoints after upstream completion evidence. A single background
publisher packages those independent copies and performs immutable Hub
readbacks; caller-supplied checkpoint step lists are not trusted. On
provider interruption, resume only with the same generation/config/image
identities and the last authenticated resumable checkpoint after an immutable
Hub download/readback. A code, data, or configuration failure is terminal but
not resumable.

Stage transfers require the exact clean code bundle hash, sealed generation and
receipt, parent artifact digest, config/modality, and token-file path; the
lifecycle never accepts arbitrary remote shell commands. Destroy only after
fresh immutable readbacks for steps 1,000 and 2,000 bound to the exact
instance and a post-destroy Vast absence readback. Then evaluate step-12000 and corrective step-1000/2000
on identical untouched matrices before any promotion.
