# LeHome recovery sweep v1

Generate this immutable manifest set from a pinned request using
`scripts/build_lehome_experiment_sweep.py`. The generator performs no network,
GPU, Hugging Face, or cloud action. Recovery-dependent arms remain blocked until
their authenticated recovery seal is supplied to the controller.
