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
variable. In the default legacy mode (with no external receipt argument), the
evaluator owns one pinned N1.7 GR00T PolicyServer child on CUDA and runs one
CPU-cloth Isaac process at a time. Before the stage entrypoint imports the
evaluator task it installs the scoped raw-checker overlay: it requires the
second (`mesh_points`) value from `get_current_mesh_points()`, finite validated
indices, and the raw `success_distance` values with no `init_scale` multiplier.
It has no transformed-point fallback.

The policy-server receipt is written only after the loaded server accepts an
authenticated loopback ping. Its model-startup wait defaults to 180 seconds
and is bounded to 30–600 seconds through `--policy-server-startup-timeout`.
Use the default unless the provider logs show a legitimately slower cold model
load; this setting does not start CUDA during `--dry-run`.

## Prestarted external policy server

External mode is only for a PolicyServer sidecar already running in the pinned
trainer image while the evaluator runs the CPU-cloth Isaac stages elsewhere.
The sidecar's model cache must be mounted at the exact same absolute
`--policy-path` in both environments. Supply its completed, pinned readiness
receipt with:

```text
--external-policy-server-readiness-receipt /absolute/path/policy-server-readiness.json
```

The evaluator validates and records that receipt, requires a token-bound
authenticated admission ping before any Isaac work, and repeats the same ping
immediately before every stage. In this mode it never starts, owns, terminates,
or claims the sidecar process log; the operator must stop the sidecar separately
after evaluation.

Each valid clean policy failure remains a scored failure. Any missing video,
log, stage receipt, malformed metrics, policy-server failure, or cloth/fidelity
invalid makes the run invalid rather than reducing the denominator. The final
receipt states only local execution evidence: public publication/readback and
the exact VM-stop observation are still separate post-execution gates.

## Post-run publication and readback gate

Only after `result.json` reports a completed valid 96-episode run and its
`verifier-receipt.json` is present, publish from the closed run root. The token
file must be a regular file owned by the invoking user with mode `0600`; its
contents are never recorded in the result, manifest, receipt, or command
output.

```bash
python3 -m scripts.publish_groot_n17_public96 \
  --run-root /mnt/lehome/evaluations/n17-public96-YYYYMMDDHHMMSS \
  --matrix "$PWD/configs/eval_groot_n17_public96_reference.json" \
  --matrix-sha256 "$PWD/configs/eval_groot_n17_public96_reference.json.sha256" \
  --token-file /run/secrets/lehome-hf-token
```

The publisher re-verifies the frozen matrix, every result artifact descriptor,
and the verifier receipt before opening a Hub transport. It stages only the
result/receipt, policy-server evidence, 48 stage logs, 48 stage receipts, and
the 288 referenced videos, then adds `SHA256SUMS.json`. Its immutable prefix
is `public96/results/<matrix-sha256-first16>-<result-sha256-first16>`.
Existing prefix data is resumable only when the exact tree and every byte
match. The returned immutable revision is then independently read back with
the owner token and anonymously before the local
`public96-publication-receipt.json` is written.

Do not stop the VM or external policy-server container merely because upload
returned. Stop them only after that publication receipt validates with both
`authenticated_readback_verified` and `anonymous_readback_verified` set to
`true`; stopping remains a separate operator/provider observation.
