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
  trainer/tests/test_continuous_training.py trainer/tests/test_groot_launch.py
```

`prepare` records the exact organizer source (`lehome/dataset_challenge_merged`
at `17e8dee8fac294ffd21d250501d3b31bf8679042`, `four_types_merged`) and its
verified mirror/manifest, plus the private corrective revision and prefix. It
does not contact HF or Vast. Do not reuse the old horizon-40 mix tarball.

Verify a generation before use:

```python
from lehome_train.flywheel.mix import verify_generation
verify_generation('/prepared/generation')
```

## Paid gate and operation

Provider actions are dry-run unless `--execute` is explicit. Rent only one
interruptible RTX PRO 6000 96GB at less than $1/hour when fresh instance plus
storage/volume account total is at most $2/hour. Run a capability receipt with
the exact image digest, CUDA/Torch CUDA, compute capability, a finite optimizer
step, and NVML telemetry. A newer Blackwell driver is accepted only by that
training capability gate; rollout driver policy is separate.

Measure loader workers at batch 64, then measure batches 64/96/128. Record all
outcomes and stop increasing batch after OOM. Production remains one GPU,
physical/global batch 64, one 2,000-step official process, saving at 1,000 and
2,000. Batches 96/128 are measurements, not a promotion.

Use `continuous-train` only with the sealed generation, exact parent digest,
normalization/config identity, and no unseen source. The training process has
no Hub credential path; publication packages immutable local snapshots in the
background. On interruption, resume only with the same generation/config
identity and the last verified resumable checkpoint.

Destroy only after fresh immutable readbacks for steps 1,000 and 2,000 bound
to the exact instance. Then evaluate step-12000 and corrective step-1000/2000
on identical untouched matrices before any promotion.
