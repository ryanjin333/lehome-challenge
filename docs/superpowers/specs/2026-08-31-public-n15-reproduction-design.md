# Public GR00T N1.5 Reproduction and Focused Rollout Gate

## Purpose

Replace the current custom GR00T N1.7 path with a byte-pinned reproduction of
the public LeHome GR00T N1.5 submission. Stop the former four-category
comparison, train one N1.5 checkpoint using the public recipe, compare only
top-short and pant-long on the official evaluator, and begin fresh rollouts
only when the reproduced checkpoint is close to the public reference on both
categories.

This design deliberately removes the custom N1.7 trainer, action gateway,
relative-action conversion, augmentation, curriculum scheduler, hard-state
mining, and mixed historical replay data from the active path.

## Fixed upstream identities

- Public implementation: `theo-zhou/lehome-groot-submission-4`
- Source revision: `d384fe00508acd96ab1c3c5dc265e08261f94b3b`
- Base model: `nvidia/GR00T-N1.5-3B`
- Base-model revision: `869830fc749c35f34771aa5209f923ac57e4564e`
- Organizer dataset: `lehome/dataset_challenge_merged`
- Dataset revision: `17e8dee8fac294ffd21d250501d3b31bf8679042`
- LeRobot: `0.4.3`
- Python: `3.11`
- Training config: upstream `configs/train_groot.yaml`, SHA-256
  `eb0c82d4a9960a072e454389d82a618d81a79b789c2f19b1733dba4c629e9f75`
- Training shell: upstream `shs/train/train_groot.sh`, SHA-256
  `2a49d25a1bbde7a54e6027fcbd490cb0334132b0f628eccad69413e19a1481b5`
- Evaluation helper: upstream `scripts/utils/evaluation.py`, SHA-256
  `9a9d9e28008405ead892fdf1d115cd83f3d2be7d806381dbc92486d2e6d966a7`

The operator must verify every identity before paid work. The base model and
dataset are downloaded at their exact revisions before training, then Hub
access is disabled for the training process. The upstream training config is
not edited. Its relative model and dataset paths are satisfied by staging the
pinned snapshots at the paths expected by the public source.

## Exact training contract

Training invokes the public source directly:

```bash
lerobot-train --config_path=configs/train_groot.yaml
```

The effective public recipe is batch size 64, 12,000 steps, learning rate
`2e-4`, 5% warmup, cosine decay to 0.1 times the initial learning rate, bf16,
mean/std state and action normalization, 224-pixel policy input, frozen LLM
and visual encoder, and trainable projector and diffusion head. No image
transforms are enabled.

The training process writes checkpoints and caches only to the existing
protected cloud disk. It does not write weights or rollout media to the local
Mac. The final step-12,000 checkpoint, source/config receipts, dependency lock,
resolved Hub revisions, logs, and checksums are uploaded under a new immutable
prefix in the public Hugging Face model repository and read back before the
checkpoint is admitted to evaluation.

## Focused paired gate

The gate evaluates exactly two categories:

- `top_short`
- `pant_long`

For each category, use all 12 Release garments with two episodes per garment:
24 episodes per category and 48 episodes per policy. Run the reproduced N1.5
checkpoint and the pinned public submission checkpoint on the identical
garment/seed matrix through the public N1.5 policy adapter and official LeHome
success checker. Simulation is CPU; policy inference is CUDA. Runs are
sequential and use the one existing rollout VM.

"Close" is a predeclared paired criterion, not a judgment after seeing the
results. The reproduced checkpoint passes only when all of these hold:

1. It is no more than two successes behind the public reference in either
   category.
2. It scores at least 18/24 on top-short.
3. It scores at least 13/24 on pant-long.
4. Every expected episode is present and provenance-identical.
5. There is no missing cloth, cloth flight, non-finite cloth state, simulator
   crash, policy-load error, scorer mismatch, or other infrastructure-invalid
   episode.
6. The receipt, logs, outcome matrix, and retained videos are uploaded and
   independently read back.

The absolute floors bind the gate to existing evidence: the public reference
scored 20/24 top-short on the fixed evaluator, while its published pant-long
result was 15/24. Any failed criterion stops the VM and ends the run without
collection.

## Rollout collection after a pass

Passing the focused gate immediately admits one fresh 1,000-attempt collection
using the reproduced N1.5 checkpoint and the public submission's native
`scripts.eval`/harvest path.

- 40 seen garments: 10 per category.
- 25 fresh attempts per garment.
- 250 attempts per category; 1,000 total.
- All garment assignments and seeds are frozen before attempt 1.
- Uniform sampling only. There is no 400/600 curriculum split.
- Save successful trajectories for later success replay; retain failure logs
  and receipts, but do not turn failed attempts into training episodes.
- No old success pool, restored state, hard-state continuation, geometry
  perturbation, augmentation, or automatic replay fine-tuning is included.

The existing first-100 circuit breaker remains a cost and fidelity guardrail:
stop if fewer than five official successes are observed, if any cloth
fidelity failure occurs, or if infrastructure-invalid attempts exceed 2%.
This is not an optimization project; it is a bounded stop rule to avoid paying
for another invalid campaign.

Four category workers may run concurrently on the single GPU only after a
zero-episode memory check and one-episode-per-worker smoke prove that four
independent native N1.5 policy processes fit and produce valid cloth/scorer
outcomes. If that check fails, fall back to two workers without changing the
policy or evaluator. Do not create another VM.

The terminal collection bundle is uploaded to a new immutable prefix in the
public Hugging Face rollout repository and read back. The exact VM is then
stopped. Success-replay fine-tuning is a separate later decision and is not
started by this workflow.

## Infrastructure and cost constraints

- Reuse only rollout VM `computeinstance-u00t6xfqhadrcmssa2`.
- Preserve protected disk `computedisk-u00pbe55crxy7jr56x`.
- Do not create a VM, disk, image, or duplicate checkpoint cache.
- Use the existing public Hugging Face storage path; keep artifacts off the
  local Mac.
- Keep the previously declared $100 remaining-work cap.
- Start the VM only for a bounded training, focused evaluation, or admitted
  collection stage, and stop it at every terminal pass/fail boundary.
- A started process, uploaded file, or local checksum is not completion;
  provider state, task result, artifact receipt, and remote readback are
  reported separately.

## Non-goals

- No GR00T N1.7 repair or comparison.
- No learning-rate sweep or alternate training duration.
- No custom action chunk, policy server, normalization, augmentation, or data
  conversion.
- No hard-state mining.
- No curriculum.
- No four-category evaluation before collection.
- No automatic success-replay training after collection.
