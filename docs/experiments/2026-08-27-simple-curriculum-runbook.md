# Original-12K simple-curriculum collection runbook

This is the only operator handoff for the approved paid collection. It is not
authorization to run it. All implementation checks may run while every GPU is
stopped; a paid start requires a separate explicit authorization.

## Fixed boundary

- Use only `computeinstance-u00t6xfqhadrcmssa2` (`lehome-rollout`): never create an image or VM;
  do not use the stopped training VMs.
- Run four persistent workers on the original 12K policy only:
  revision `30ac1a84da67b099e115ad147bcd61e9d60046d3`, step `12000`, artifact
  SHA-256 `3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06`.
- Use 40 seen garments, 10 per category: top-long, top-short, pant-long, and
  pant-short. CPU cloth; CUDA policy and render only.
- Collect 400 uniform fresh outcomes, then 600 frozen curriculum fresh
  outcomes. Success replay only follows those fresh sources: at most 400 replay
  attempts and 200 accepted, with at most 100 attempts and 50 accepted
  per category.
- Exclude A-500, hard-state mining, old rollout inputs, training, geometry or
  physics perturbation, and any automatic next experiment.
- The hard ceiling is $100. The controller uses `99.00` as its strict internal
  cutoff so it cannot cross that ceiling.

No secret value belongs in this runbook. Keep all token paths, not token
contents, in the operator-only environment.

## 1. Read-only preflight

Do all of these before a paid start. Stop if any check differs.

1. With the existing approved Nebius CLI profile, issue this read-only lookup:

   ```bash
   nebius compute instance get computeinstance-u00t6xfqhadrcmssa2
   ```

   Record the returned ID, name, image, attached-disk IDs, and `STOPPED`
   state. It must be the only rollout VM. Do not use a create, delete,
   replacement, or broad-list command. Confirm manually that no other GPU VM
   is running.

2. On the stopped VM's mounted reviewed checkout, verify the exact committed
   source and its clean state, without checking out another revision:

   ```bash
   git -C /mnt/lehome/lehome-challenge rev-parse HEAD
   git -C /mnt/lehome/lehome-challenge status --porcelain
   ```

   The first value must be the reviewed commit and the second must be empty.
   Set `LEHOME_HOST_CODE_ROOT` to this absolute non-symlink checkout; the
   mounted code is used directly—there is no image rebuild.

3. Prepare a **new empty absolute** root such as
   `/mnt/lehome/eval/fresh-run-YYYYMMDD-01`. It must not be a symlink and must
   not contain an old campaign, receipt, ledger, or matrix. Use matching fresh
   identities `fresh-run-YYYYMMDD-01` and `fresh-12k-YYYYMMDD-01`; never reuse
   an ID or output root after a terminal outcome.

4. Prepare an owner-only, regular runtime-identity JSON file outside the
   public bundle. It must have exactly these fields: `policy_repo`,
   `policy_revision`, `policy_step`, `policy_artifact_sha256`,
   `simulator_device`, `cloth_device`, `policy_device`, `worker_count`,
   `rollout_image`, and `trainer_image`. The first eight bind the values above,
   `cpu`, `cpu`, `cuda:0`, and `4`. The two image values must be the exact
   current VM runtime strings, each digest-pinned as `@sha256:<64 lowercase
   hex>`; do not invent or replace them.

5. Check every path without printing secret bytes. The HF token file must be
   a non-symlink regular file, owner-only (`0600`), and nonempty. The spend
   observer is a current, typed JSON receipt with only
   `schema_version`, `kind`, `observer`, `observed_at_utc`, and `spent_usd`;
   its timestamp must be under five minutes old and spend must be below 99.00.
   The trusted stop hook must be exactly
   `/usr/local/libexec/lehome-stop-gpu`.

   ```bash
   test -f /mnt/lehome/secrets/hf_token && test ! -L /mnt/lehome/secrets/hf_token
   test "$(stat -c '%a' /mnt/lehome/secrets/hf_token)" = 600
   test -s /mnt/lehome/secrets/hf_token
   test -x /usr/local/libexec/lehome-stop-gpu
   ```

6. Confirm the public destination is
   `ryanjin333/lehome-groot-n17-rollouts`, and that anonymous access is
   possible to a harmless existing public object. The new immutable prefix
   will be `collection-rounds/<run-id>`; it must not already exist.

## 2. Offline dry run

Run this from the reviewed checkout before any paid start. It performs no
Nebius, Docker, Isaac, rollout, or Hub action.

```bash
PYTHONPATH=.:source/lehome:trainer/src pytest -q \
  tests/flywheel/test_simple_curriculum.py \
  tests/flywheel/test_simple_curriculum_gate.py \
  tests/flywheel/test_task_ledger.py \
  tests/flywheel/test_randomization.py \
  tests/flywheel/test_success_replay_matrix.py \
  tests/flywheel/test_recovery_collection.py \
  tests/flywheel/test_persistent_worker.py \
  tests/flywheel/test_hub_sync.py \
  tests/infrastructure/test_simple_curriculum_cli.py \
  tests/infrastructure/test_simple_curriculum_campaign.py \
  tests/infrastructure/test_simple_curriculum_orchestrator.py \
  tests/infrastructure/test_simple_curriculum_publication.py \
  tests/infrastructure/test_success_replay_campaign.py \
  tests/infrastructure/test_rollout_container.py \
  tests/infrastructure/test_simple_curriculum_runbook.py
bash -n rollout_appliance/run_12k_campaign.sh
bash -n rollout_appliance/run_success_replay_campaign.sh
bash -n rollout_appliance/run_simple_curriculum_collection.sh
python3 -m py_compile scripts/build_simple_curriculum_matrix.py \
  scripts/check_simple_curriculum_gate.py scripts/build_success_replay_matrix.py \
  scripts/run_simple_curriculum_collection.py scripts/publish_simple_curriculum_collection.py
git diff --check
```

## 3. One paid command

After explicit authorization, start **that exact stopped VM once** through the
approved operator path, wait for `RUNNING`, recheck its ID and mounted
checkout, then run only this command on it. Values in angle brackets are
operator-selected paths/identities, never secret contents.

```bash
sudo env -i \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LEHOME_PAID_COLLECTION=1 \
  LEHOME_GPU_STOP_COMMAND=/usr/local/libexec/lehome-stop-gpu \
  LEHOME_HOST_CODE_ROOT=/mnt/lehome/lehome-challenge \
  LEHOME_CAMPAIGN_ROOT=/mnt/lehome/eval/<fresh-run-id> \
  LEHOME_RUN_ID=<fresh-run-id> \
  LEHOME_ROUND_ID=<fresh-12k-id> \
  LEHOME_MAX_WALL_SECONDS=86399 \
  LEHOME_MAX_SPEND_USD=99.00 \
  LEHOME_RUNTIME_IDENTITY_JSON=/mnt/lehome/operator/runtime-identity.json \
  LEHOME_SPEND_OBSERVER=/mnt/lehome/operator/spend-observation.json \
  /mnt/lehome/lehome-challenge/rollout_appliance/run_simple_curriculum_collection.sh
```

The wrapper admits no provider lifecycle command. It delegates the trusted
exact-VM stop to the fixed hook only after a terminal outcome. Do not start a
second worker host or training VM.

## 4. State machine and immediate stops

`calibration-matrix` freezes 400 assignments. `calibration-head` settles the
first 100 (25/category), then `first-100-gate` checks typed cloth/safety
fidelity, identity consistency, exactly 100 valid outcomes, infrastructure
invalid ratio no greater than 2%, and at least five official successes.

- `continue`: only then run calibration-tail (remaining 300), authenticate the
  calibration report, freeze the 600-row curriculum matrix, and run its two
  300-row partitions.
- `fidelity_stop`, `infrastructure_stop`, or `insufficient_source_stop`:
  do not run the remaining 300+600. Stop/report immediately.
- A stale, malformed, regressing, or >=99.00 spend observation; wall-time
  limit; failed identity/receipt; preemption ambiguity; worker/process error;
  or stop-hook failure is fail closed. The status is reported as an
  infrastructure outcome, never a data success.

The controller polls the typed spend receipt before and after every paid stage
and while a child runs. It writes durable budget state. It does not pass the
budget check to final public publication because the VM is already verified
stopped at that point.

## 5. Fresh terminal evidence, replay, and publication

Every fresh success **and failure** remains terminal evidence. Success replay
is built only after all 1,000 fresh outcomes plus their receipts are
authenticated. It may use only accepted fresh successes with the verified
CPU-cloth step-16 continuation and changes only visual appearance; it has no
physics or geometry change. A category shortage is an honest replay result,
not permission to use old data.

Publication is immutable beneath `collection-rounds/<run-id>` and includes
only reviewed manifests, fresh artifacts, accepted replay artifacts, reports,
and seals. It requires byte-identical authenticated and anonymous downloads
at the recorded immutable revision. The terminal seals are:

- `fidelity_infrastructure_stop` for fidelity or infrastructure stopping
  evidence;
- `insufficient_fresh_source` for a source/replay shortage;
- `collection_complete` only for 1,000 authenticated fresh terminal outcomes,
  required replay/readbacks, and a verified stopped exact VM.

The exact VM must have a durable Nebius Compute `STOPPED` observation before
any final seal or collection completion claim. Report publication and readback
separately: an upload alone is not completion.

## 6. Preemption and crash recovery

Preemption resumes only the same immutable partition matrix and same task
ledger. It may retry incomplete leases; it may not create a seed, duplicate a
terminal assignment, alter a frozen matrix, or create another VM.

After any terminal stop, budget, fidelity, or infrastructure gate, there is no
automatic paid restart. The only retryable boundary is zero-compute final
publication after a verified stop. If a process crashed after the durable
`final-publication.json` but before its local readback receipt, rerun the
same controller against the same root. It invokes the publisher
`--reconcile` path: it pins the recorded immutable revision and manifest,
downloads all files authenticated and anonymously again, and writes only the
missing `final-publication-readback.json`. A malformed receipt, changed remote
byte, missing file, or non-public readback fails closed; it must not upload or
use a mutable branch head.

## 7. Status and cleanup checklist

Report these independently at every stop or operator handoff:

1. Runtime: exact instance ID/state, code revision, image digests, original
   12K identity, CPU cloth/CUDA policy-render, and active worker count.
2. Results: valid fresh outcomes/1,000, official fresh successes by category,
   replay attempts/400, accepted replay successes/200, and the first-100 or
   terminal decision.
3. Publication: immutable revision/prefix, authenticated readback result, and
   anonymous readback result.
4. Seals: which terminal seal exists and whether it is accepted.
5. Resource state: exact VM `STOPPED` observation and current spend/time.

Do not delete campaign roots, protected/shared state, images, disks, or local
evidence until public readback has succeeded and the applicable seal is
accepted. No destructive command is authorized by this runbook; resolve the
exact target and durable public evidence in a separate cleanup decision.
