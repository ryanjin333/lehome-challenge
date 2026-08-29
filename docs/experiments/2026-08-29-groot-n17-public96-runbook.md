# Original GR00T N1.7 public-96 evaluator

This is a standalone evidence contract for the original N1.7 12K checkpoint.
It is not the N1.5 calibration gate and must never load the checkpoint through
`LeRobotPolicy`.

The frozen matrix contains the 48 Release garments in category order
(`top_long`, `top_short`, `pant_long`, `pant_short`), Seen 0–9 then Unseen 0–1
for each. Every garment is one seed-42 stage with exactly two sequential
episodes, for 96 scored episodes total.

## Validation only (no paid run)

After an operator has made a local, read-only N1.7 checkpoint identity receipt
that binds the pinned cache tree, use a fresh absolute output path:

```bash
python3 -m scripts.eval_groot_n17_public96 \
  --matrix "$PWD/configs/eval_groot_n17_public96_reference.json" \
  --matrix-sha256 "$PWD/configs/eval_groot_n17_public96_reference.json.sha256" \
  --policy-path /mnt/lehome/cache/groot-n17/policies/step-12000 \
  --checkpoint-identity-receipt /mnt/lehome/cache/groot-n17/checkpoint-identity.json \
  --asset-root /mnt/lehome/assets/Challenge_Garment \
  --output-root /mnt/lehome/evaluations/n17-public96-plan-YYYYMMDDHHMMSS \
  --dry-run
```

Validation-only checks the matrix bytes/order, two-episode semantics, checkpoint
and immutable cache identity, raw-checker overlay digest, output safety, and the
exact sequential commands. It does not start CUDA, Isaac, a policy server, a
provider resource, upload, publication, readback, or VM stop. A successful
validation receipt is not an evaluation result.

## Later paid execution boundary

Only after separate provider admission and an explicit paid-run decision, omit
`--dry-run` and provide the already-present policy-server token environment
variable. The evaluator starts one pinned N1.7 GR00T PolicyServer on CUDA and
runs one CPU-cloth Isaac process at a time. Before the stage entrypoint imports
the evaluator task it installs the scoped raw-checker overlay: it requires the
second (`mesh_points`) value from `get_current_mesh_points()`, finite validated
indices, and the raw `success_distance` values with no `init_scale` multiplier.
It has no transformed-point fallback.

The policy-server receipt is written only after the loaded server accepts an
authenticated loopback ping. Its model-startup wait defaults to 180 seconds
and is bounded to 30–600 seconds through `--policy-server-startup-timeout`.
Use the default unless the provider logs show a legitimately slower cold model
load; this setting does not start CUDA during `--dry-run`.

Each valid clean policy failure remains a scored failure. Any missing video,
log, stage receipt, malformed metrics, policy-server failure, or cloth/fidelity
invalid makes the run invalid rather than reducing the denominator. The final
receipt states only local execution evidence: public publication/readback and
the exact VM-stop observation are still separate post-execution gates.
