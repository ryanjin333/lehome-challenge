# Native GR00T N1.5 Reference Evaluator Gate

## Purpose

Before any further LeHome collection or training, prove that the evaluator can
reproduce the pinned public reference submission through the checkpoint's
native runtime contract.  The earlier eight-attempt reference run is diagnostic
evidence only: it loaded a LeRobot GR00T N1.5 checkpoint through the local GR00T
N1.7 nested-modality, 16-action gateway, so its `1/8` score is not an admissible
checkpoint or evaluator result.

No training, curriculum collection, success replay, or large evaluation is
admitted until this gate passes.

## Immutable reference identity

- Repository: `theo-zhou/lehome-groot-submission-4`
- Revision: `d384fe00508acd96ab1c3c5dc265e08261f94b3b`
- Policy implementation: the pinned submission's `scripts.eval_policy.lerobot_policy.LeRobotPolicy`
- Runtime family: LeRobot `0.4.3`, GR00T N1.5
- Checkpoint directory: the pinned submission's unmodified `pretrained_model/`
- Dataset metadata: the pinned submission's per-category `dataset_meta/<category>_merged/meta/`
- Simulation: CPU
- Policy inference: CUDA on the one existing rollout GPU
- Task text: `fold the garment on the table`
- Maximum episode length: 600 policy steps

The gate must verify the exact source revision and trusted byte digests before
launch.  It must not rewrite `config.json`, remove fields, translate the flat
LeRobot observation into the GR00T N1.7 nested schema, or replace the saved
preprocessor/postprocessor.

## Native policy contract

The gate must preserve all of these checkpoint-owned behaviors:

1. Flat observations:
   - `observation.state`, shape `(12,)`
   - `observation.images.top_rgb`, HWC uint8
   - `observation.images.left_rgb`, HWC uint8
   - `observation.images.right_rgb`, HWC uint8
   - `observation.images.top_depth` when requested by the checkpoint
2. The saved LeRobot preprocessor and its normalization state.
3. The saved LeRobot postprocessor and its action unnormalization state.
4. `GrootPolicy.select_action` and its internal 16-action open-loop queue.  The
   top-level config's `chunk_size=50` is not the runtime authority; the saved
   preprocessor declares `action_horizon=16`, and the evaluator consumes one
   returned 12-D action per simulator step.
5. Absolute 12-D bimanual joint targets in the environment's native order.
6. The pinned submission's environment, garment loader, reset/stabilization,
   and success-checker source for this reproduction gate.  In particular, the
   checker uses the published raw `success_distance` values and the second
   (`mesh_points`) result from `get_current_mesh_points()`; it must not apply
   the local 0.45 garment-scale multiplier or switch to transformed vertices.

The ordinary N1.7 gateway remains unchanged and is not used by this gate.

## Bounded oracle

Run exactly the first seen garment from each category, two episodes each, in
this order and sequentially:

| Stage | Garment | Expected public outcomes |
|---|---|---|
| 1 | `Top_Long_Seen_0` | success, success |
| 2 | `Top_Short_Seen_0` | success, success |
| 3 | `Pant_Long_Seen_0` | success, success |
| 4 | `Pant_Short_Seen_0` | failure, success |

These eight outcomes are the pinned submission's published compatibility
oracle.  They do not reproduce the reported `72.92%`, which was `70/96` over
all 48 Release garments (two episodes per garment).  Stage 1 is
a fail-fast admission check: if it is not `2/2`, stop the GPU and do not spend
on the remaining six attempts.  Otherwise complete all eight.

The final gate passes only when:

- all eight attempts are present;
- the outcome vector exactly matches the published oracle (`7/8`);
- every attempt used the same pinned source/checkpoint/runtime identity;
- no missing cloth, cloth flight, non-finite cloth state, safety failure, or
  infrastructure-invalid outcome occurred;
- the result bundle and every video/log/receipt are uploaded to Hugging Face
  and independently read back; and
- the exact rollout VM is stopped after publication.

Any other result is a typed evaluator-compatibility or infrastructure stop.  It
does not admit training or collection.

## Execution constraints

- Use only the existing rollout VM and protected disk.
- Never run Isaac processes in parallel for this gate.
- Never create a replacement VM or image for the gate.
- Do not download checkpoint weights again when the pinned, verified cache is
  already present.
- Keep model weights and rollout artifacts off the local Mac.
- Keep total remaining-work spend below the existing `$100` cap.
- Publish under a new immutable `reference-checks/native-...` prefix; never
  overwrite the earlier invalid-compatibility diagnostic.

## Decision after the gate

- Pass: run the local 12K checkpoint on the same native evaluator boundary.
  If the reference remains `7/8` and the 12K policy is low, the training recipe
  is the problem; only then consider the targeted success-replay fine-tune.
- Fail: keep collection and training stopped and repair the exact native
  contract or runtime discrepancy shown by the receipt.  Do not gather more
  policy data against an uncalibrated evaluator.
