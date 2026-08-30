# Behavior 1K training-only rent gate

This Vast template is for training only. Rollouts require a separate future Isaac Sim GUI/noVNC template and are intentionally not supported here.

Do not create a paid instance until the image structural gate and preflight pass.
Use one verified datacenter instance with 1–4 RTX PRO 6000 Blackwell GPUs (96 GiB each), 2048 GB disk, disk bandwidth at least 4 GB/s, download at least 1 Gbps, upload at least 500 Mbps, 128 GiB RAM, and 24 CPU cores. GPU count is bounded to the explicit 1–4 launch plans; it never freely auto-scales the optimizer contract.

The onstart script receives the account token once, atomically writes `/workspace/.cache/huggingface/token` as trainer-owned `0600`, unsets it, exports only `B1K_HF_TOKEN_FILE`, and starts the controller as uid/gid `10001`. It never deletes an instance, model repository, or checkpoint bucket. `AUTO_DESTROY` is always `0`.

Before downloading the dataset or model inputs, bootstrap resolves the pinned dataset, base-model, and Cosmos revisions; confirms that `ryanjin333/behavior1k-groot-n17-models` and `ryanjin333/behavior1k-groot-n17-checkpoints` are private; then independently uploads, reads, verifies, and exactly deletes one temporary `smoke/<uuid>/` object in each output. The checkpoint bucket is created only when `CREATE_CHECKPOINT_BUCKET=1`.

Inspect `/workspace/logs/controller.log` and `/workspace/outputs/<RUN_ID>/run-status.json`. Resume uses the validated private bucket state; finalization writes a readback receipt under `/workspace/final`. A receipt is required before considering the run complete.
