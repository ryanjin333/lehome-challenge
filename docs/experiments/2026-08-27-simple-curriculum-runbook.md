# Original-12K simple-curriculum collection runbook

This is the only operator handoff for the approved paid collection. It is not
authorization to run it. Offline checks may run while every GPU is stopped; a
paid start requires separate explicit authorization.

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
  attempts and 200 accepted, with at most 100 attempts and 50 accepted per
  category.
- Exclude A-500, hard-state mining, old rollout inputs, training, geometry or
  physics perturbation, and any automatic next experiment.
- The hard ceiling is $100. The controller uses `99.00` as its strict internal
  cutoff so it cannot cross that ceiling.

No secret value belongs in this runbook. Keep token paths, not token contents,
in the operator-only environment.

## 1. Provider preflight while stopped

Do these **before** starting anything. They prove only provider state and
identity; they do not substitute for the post-start host checks below.

1. With the approved Nebius CLI profile, issue this read-only lookup:

   ```bash
   nebius compute instance get computeinstance-u00t6xfqhadrcmssa2
   ```

   Record the returned ID, name, image, attached-disk IDs, and `STOPPED`
   state. The result must identify `lehome-rollout` and the approved attached
   workspace. Do not use a create, delete, replacement, or broad-list command.

2. Confirm the public destination is `ryanjin333/lehome-groot-n17-rollouts`
   and that anonymous access works for a harmless existing public object. The
   new immutable prefix will be `collection-rounds/<run-id>` and must not
   already exist. Check the token path, not its contents: because the paid
   publisher runs as root through `sudo env -i`, it must be a root-owned,
   regular non-symlink `0600`, nonempty file at
   `/mnt/lehome/secrets/hf_token` once the workspace is mounted.

3. Confirm the provider spend view is below $99.00 and prepare a current typed
   spend receipt at `/mnt/lehome/operator/spend-observation.json`. The receipt
   contains only `schema_version`, `kind`, `observer`, `observed_at_utc`, and
   `spent_usd`; its timestamp must be under five minutes old. Confirm the
   trusted exact-VM stop hook is `/usr/local/libexec/lehome-stop-gpu`.

Stop and report if the exact ID, image, disk, state, public destination, or
budget evidence differs. Do not attempt the post-start checks on a stopped VM.

## 2. Start exactly the approved VM

Only after Section 1 passes and explicit authorization is present, use the
approved operator control to start **only**
`computeinstance-u00t6xfqhadrcmssa2`. Wait for the provider to report
`RUNNING`, then repeat the same read-only lookup and verify the exact ID,
image, and attached workspace before logging in. Do not start a second worker
host or training VM.

## 3. Post-start checkpoint and immutable input staging

These are required **after** `RUNNING` and before the collection command. A
cloud-init, mount, GPU, original-12K, CPU-cloth, image, or content mismatch is
an infrastructure stop: stop the exact VM through the trusted hook, report,
and do not launch collection.

```bash
set -euo pipefail
LEHOME_HOST_CODE_ROOT=/mnt/lehome/lehome-challenge
test -d "$LEHOME_HOST_CODE_ROOT" && test ! -L "$LEHOME_HOST_CODE_ROOT"
cloud-init status --wait
mountpoint -q /mnt/lehome
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
git -C "$LEHOME_HOST_CODE_ROOT" rev-parse HEAD
git -C "$LEHOME_HOST_CODE_ROOT" diff --quiet
test -d "$LEHOME_HOST_CODE_ROOT/source/lehome" && test ! -L "$LEHOME_HOST_CODE_ROOT/source/lehome"
test -d "$LEHOME_HOST_CODE_ROOT/trainer/src" && test ! -L "$LEHOME_HOST_CODE_ROOT/trainer/src"
test -x /usr/local/libexec/lehome-stop-gpu
test -f /mnt/lehome/secrets/hf_token && test ! -L /mnt/lehome/secrets/hf_token
test "$(stat -c '%u' /mnt/lehome/secrets/hf_token)" = 0
test "$(stat -c '%a' /mnt/lehome/secrets/hf_token)" = 600
test -s /mnt/lehome/secrets/hf_token
```

Read the runtime-identity JSON from
`/mnt/lehome/operator/runtime-identity.json` without changing it. It must have
exactly `policy_repo`, `policy_revision`, `policy_step`,
`policy_artifact_sha256`, `simulator_device`, `cloth_device`, `policy_device`,
`worker_count`, `rollout_image`, and `trainer_image`. Its policy fields must
match the revision, step, and SHA-256 in Fixed boundary; devices must be `cpu`,
`cpu`, and `cuda:0`; worker count must be `4`; both actual current image
strings must be digest-pinned as `@sha256:<64 lowercase hex>`. This binds the
mounted image/content and proves CPU cloth plus CUDA policy/render before any
collection.

### Create or reuse the one invocation identity

Run this once after the post-start checks. It creates an owner-only,
non-secret identity record. On a retry it reads and validates that record, so
the same IDs and exact command are reused rather than regenerated.

```bash
set -euo pipefail
LEHOME_HOST_CODE_ROOT=/mnt/lehome/lehome-challenge
LEHOME_INVOCATION_FILE=/mnt/lehome/operator/simple-curriculum-invocation.env
if test -e "$LEHOME_INVOCATION_FILE"; then
  test -f "$LEHOME_INVOCATION_FILE" && test ! -L "$LEHOME_INVOCATION_FILE"
  test "$(stat -c '%a' "$LEHOME_INVOCATION_FILE")" = 600
  . "$LEHOME_INVOCATION_FILE"
else
  LEHOME_RUN_ID="fresh-run-$(date -u +%Y%m%d%H%M%S)-01"
  LEHOME_ROUND_ID="fresh-12k-${LEHOME_RUN_ID#fresh-run-}"
  LEHOME_CAMPAIGN_ROOT="/mnt/lehome/eval/$LEHOME_RUN_ID"
  test ! -e "$LEHOME_CAMPAIGN_ROOT"
  umask 077
  printf 'LEHOME_RUN_ID=%q\nLEHOME_ROUND_ID=%q\n' "$LEHOME_RUN_ID" "$LEHOME_ROUND_ID" > "$LEHOME_INVOCATION_FILE"
fi
case "$LEHOME_RUN_ID" in fresh-run-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-01) ;; *) exit 2;; esac
case "$LEHOME_ROUND_ID" in fresh-12k-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-01) ;; *) exit 2;; esac
LEHOME_CAMPAIGN_ROOT="/mnt/lehome/eval/$LEHOME_RUN_ID"
export LEHOME_HOST_CODE_ROOT LEHOME_RUN_ID LEHOME_ROUND_ID LEHOME_CAMPAIGN_ROOT
```

For an initial invocation, that campaign root must not exist before this
section. Create only its `inputs` directory; it is not completely empty
because the immutable reviewed catalog is staged before controller-generated
files. On a preemption/resume, reuse the existing root and the record above;
do not copy an old rollout input or create a new identity.

The authoritative source is the reviewed checkout's
`configs/eval_groot_n17_public_280.json`, not either historical rollout
campaign file. The following first-stage command extracts only its `seen`
trials, rejects anything except 40 unique garments/10 per category, writes a
canonical catalog, records source provenance/hash, and reads both hashes back.

```bash
set -euo pipefail
CATALOG_SOURCE="$LEHOME_HOST_CODE_ROOT/configs/eval_groot_n17_public_280.json"
CATALOG_DIR="$LEHOME_CAMPAIGN_ROOT/inputs"
CATALOG="$CATALOG_DIR/seen-catalog.json"
if test ! -e "$LEHOME_CAMPAIGN_ROOT"; then
  mkdir -p "$CATALOG_DIR"
  python3 - "$CATALOG_SOURCE" "$CATALOG" <<'PY'
import json
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
payload = json.loads(source.read_text(encoding="utf-8"))
expected = {"top_long", "top_short", "pant_long", "pant_short"}
catalog = {category: set() for category in expected}
for trial in payload["trials"]:
    if trial.get("release_stage") == "seen" and trial.get("category") in expected:
        catalog[trial["category"]].add(trial["garment_name"])
if set(catalog) != expected or sum(map(len, catalog.values())) != 40 or any(len(items) != 10 for items in catalog.values()):
    raise SystemExit("expected exactly 40 unique seen garments, 10 per category")
destination.write_text(json.dumps({key: sorted(value) for key, value in sorted(catalog.items())}, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
PY
  (cd "$LEHOME_HOST_CODE_ROOT" && sha256sum configs/eval_groot_n17_public_280.json) > "$CATALOG_DIR/catalog-source.sha256"
  (cd "$CATALOG_DIR" && sha256sum seen-catalog.json) > seen-catalog.sha256
fi
test -f "$CATALOG" && test ! -L "$CATALOG"
(cd "$LEHOME_HOST_CODE_ROOT" && sha256sum --check "$CATALOG_DIR/catalog-source.sha256")
(cd "$CATALOG_DIR" && sha256sum --check seen-catalog.sha256)
python3 - "$CATALOG" <<'PY'
import json
import sys
catalog = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {"top_long", "top_short", "pant_long", "pant_short"}
if set(catalog) != expected or sum(map(len, catalog.values())) != 40 or any(len(set(items)) != 10 for items in catalog.values()):
    raise SystemExit("staged catalog is not exactly 40 unique seen garments, 10 per category")
PY
```

## 4. Offline dry run

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
env -i PATH="$PATH" PYTHONPATH="$PWD:$PWD/source/lehome:$PWD/trainer/src" \
  python3 scripts/run_simple_curriculum_collection.py --help
git diff --check
```

## 5. One paid command

After Sections 1–3 pass, run this exact command on the already-running exact
VM. It uses the exported identities from Section 3; retry the same command
with the same `LEHOME_INVOCATION_FILE`, never a regenerated ID.

```bash
sudo env -i \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  PYTHONPATH="$LEHOME_HOST_CODE_ROOT:$LEHOME_HOST_CODE_ROOT/source/lehome:$LEHOME_HOST_CODE_ROOT/trainer/src" \
  LEHOME_PAID_COLLECTION=1 \
  LEHOME_GPU_STOP_COMMAND=/usr/local/libexec/lehome-stop-gpu \
  LEHOME_HOST_CODE_ROOT="$LEHOME_HOST_CODE_ROOT" \
  LEHOME_CAMPAIGN_ROOT="$LEHOME_CAMPAIGN_ROOT" \
  LEHOME_RUN_ID="$LEHOME_RUN_ID" \
  LEHOME_ROUND_ID="$LEHOME_ROUND_ID" \
  LEHOME_MAX_WALL_SECONDS=86399 \
  LEHOME_MAX_SPEND_USD=99.00 \
  LEHOME_RUNTIME_IDENTITY_JSON=/mnt/lehome/operator/runtime-identity.json \
  LEHOME_SPEND_OBSERVER=/mnt/lehome/operator/spend-observation.json \
  /mnt/lehome/lehome-challenge/rollout_appliance/run_simple_curriculum_collection.sh
```

The wrapper admits no provider lifecycle command. It delegates the trusted
exact-VM stop to the fixed hook only after a terminal outcome. No secret is
passed on the command line.

## 6. State machine and immediate stops

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

## 7. Fresh terminal evidence, replay, and publication

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

## 8. Preemption and crash recovery

Preemption resumes only the same immutable partition matrix, task ledger,
campaign root, and invocation IDs. It may retry incomplete leases; it may not
create a seed, duplicate a terminal assignment, alter a frozen matrix, or
create another VM.

After any terminal stop, budget, fidelity, or infrastructure gate, there is no
automatic paid restart. The only retryable boundary is zero-compute final
publication after a verified stop. If a process crashed after the durable
`final-publication.json` but before its local readback receipt, rerun the same
controller against the same root. It invokes the publisher `--reconcile` path:
it pins the recorded immutable revision and manifest, downloads all files
authenticated and anonymously again, and writes only the missing
`final-publication-readback.json`. A malformed receipt, changed remote byte,
missing file, or non-public readback fails closed; it must not upload or use a
mutable branch head.

## 9. Status and cleanup checklist

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
