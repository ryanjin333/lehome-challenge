# Persistent corrective GR00T training

This run trains one sealed 70/30 corrective generation only: organizer/original
BC frames are 70%, accepted corrective RFT frames are 30%, and the corrective
snapshot remains horizon 16 while the GR00T model capacity is 40.  The legacy
chunked fixed-exposure `train` action remains the rollback path.

## Free preparation

Use only local evidence first:

```bash
python3 scripts/run_groot_persistent_training.py prepare --request request.json
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_flywheel_mix.py trainer/tests/test_throughput_tuning.py \
  trainer/tests/test_continuous_training.py trainer/tests/test_groot_config.py \
  trainer/tests/test_groot_launch.py trainer/tests/test_persistent_training_lifecycle.py
```

`prepare` verifies the supplied local sibling generation receipt is sealed
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
storage/volume account total is at most $2/hour. Run a capability receipt with
the exact image digest, CUDA/Torch CUDA, compute capability, a finite optimizer
step, and NVML telemetry. A newer Blackwell driver is accepted only by that
training capability gate; rollout driver policy is separate.

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
interruption, resume only with the same generation/config/image identities and
the last authenticated resumable checkpoint.

Stage transfers require the exact clean code bundle hash, sealed generation and
receipt, parent artifact digest, config/modality, and token-file path; the
lifecycle never accepts arbitrary remote shell commands. Destroy only after
fresh immutable readbacks for steps 1,000 and 2,000 bound to the exact
instance and a post-destroy Vast absence readback. Then evaluate step-12000 and corrective step-1000/2000
on identical untouched matrices before any promotion.
