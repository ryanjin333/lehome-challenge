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
   already exist. Check token paths, not their contents. The remote
   **per-episode sync** token is separately consumed by the root-owned
   controller process and must be a root-owned, regular non-symlink `0600`,
   nonempty file at `/mnt/lehome/secrets/hf_token` once the workspace is
   mounted. The local operator finalizer uses a different, operator-owned
   `0600` token path and never sends that local token to the VM.

3. Confirm the provider spend view is below $99.00 and preserve its signed-in
   baseline observation for the local observer in Section 3. The observer,
   rather than this one-time provider observation, writes the current typed
   receipt at `/mnt/lehome/operator/spend-observation.json` with only
   `schema_version`, `kind`, `observer`, `observed_at_utc`, and `spent_usd`.
   Do not install or invoke a stop hook on the VM. Terminal stopping and
   final publication are performed by the local operator finalizer in Section
   7 after the remote controller returns its handoff.

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
an infrastructure stop: return the terminal handoff, then let the
operator-local finalizer stop the exact VM, report,
and do not launch collection.

Run this operator-side command against the already-approved RUNNING VM. It
prints a non-secret JSON receipt; it stages no provider resource and only
creates `/mnt/lehome/runtime-code/<reviewed-revision>` after all remote checks
pass.

```bash
LEHOME_REVIEWED_REVISION="$(git rev-parse HEAD)"
ssh -o ClearAllForwardings=yes "${LEHOME_VM_SSH_TARGET:?set approved SSH target}" 'mountpoint -q /mnt/lehome'
./scripts/stage_simple_curriculum_runtime_code.sh --ssh-target "${LEHOME_VM_SSH_TARGET:?set approved SSH target}"
printf 'staged reviewed revision: %s\n' "$LEHOME_REVIEWED_REVISION"
```

Open one persistent remote shell with that exact revision exported; run the
remaining Section 3 preparation blocks in this same shell. The one paid
controller invocation happens later from the operator machine through the
reviewed wrapper. Do not recompute the revision on the VM or open a fresh
unparameterized shell:

```bash
ssh -tt -o ClearAllForwardings=yes "${LEHOME_VM_SSH_TARGET:?set approved SSH target}" \
  "export LEHOME_REVIEWED_REVISION='$LEHOME_REVIEWED_REVISION'; exec bash -l"
```

In that same remote shell, run:

```bash
set -euo pipefail
LEHOME_REVIEWED_REVISION="${LEHOME_REVIEWED_REVISION:?carry the staged revision from the operator command}"
LEHOME_HOST_CODE_ROOT="/mnt/lehome/runtime-code/$LEHOME_REVIEWED_REVISION"
test -d "$LEHOME_HOST_CODE_ROOT" && test ! -L "$LEHOME_HOST_CODE_ROOT"
cloud-init status --wait
mountpoint -q /mnt/lehome
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
test "$(git -C "$LEHOME_HOST_CODE_ROOT" rev-parse HEAD)" = "$LEHOME_REVIEWED_REVISION"
git -C "$LEHOME_HOST_CODE_ROOT" diff --quiet
test -d "$LEHOME_HOST_CODE_ROOT/source/lehome" && test ! -L "$LEHOME_HOST_CODE_ROOT/source/lehome"
test -d "$LEHOME_HOST_CODE_ROOT/trainer/src" && test ! -L "$LEHOME_HOST_CODE_ROOT/trainer/src"
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
LEHOME_REVIEWED_REVISION="${LEHOME_REVIEWED_REVISION:?carry the staged revision from the operator command}"
LEHOME_HOST_CODE_ROOT="/mnt/lehome/runtime-code/$LEHOME_REVIEWED_REVISION"
LEHOME_INVOCATION_FILE=/mnt/lehome/operator/simple-curriculum-invocation.env
if test -e "$LEHOME_INVOCATION_FILE"; then
  test -f "$LEHOME_INVOCATION_FILE" && test ! -L "$LEHOME_INVOCATION_FILE"
  test "$(stat -c '%a' "$LEHOME_INVOCATION_FILE")" = 600
  test "$(stat -c '%u' "$LEHOME_INVOCATION_FILE")" = "$(id -u)"
  awk -F= 'BEGIN { split("LEHOME_REVIEWED_REVISION LEHOME_CAMPAIGN_ROOT LEHOME_RUN_ID LEHOME_ROUND_ID LEHOME_SPEND_BASELINE_USD LEHOME_SPEND_BASELINE_AT_UTC LEHOME_MAX_HOURLY_BURN_USD LEHOME_SPEND_OBSERVER_COMMAND", a, " "); for (i in a) ok[a[i]]=1 } /^[A-Z_]+=[A-Za-z0-9._:\/+\-]+$/ { if (!ok[$1] || seen[$1]++) exit 1; next } { exit 1 } END { for (i in ok) if (!seen[i]) exit 1 }' "$LEHOME_INVOCATION_FILE"
  LEHOME_REVIEWED_REVISION="$(sed -n 's/^LEHOME_REVIEWED_REVISION=//p' "$LEHOME_INVOCATION_FILE")"
  LEHOME_RUN_ID="$(sed -n 's/^LEHOME_RUN_ID=//p' "$LEHOME_INVOCATION_FILE")"
  LEHOME_ROUND_ID="$(sed -n 's/^LEHOME_ROUND_ID=//p' "$LEHOME_INVOCATION_FILE")"
  LEHOME_CAMPAIGN_ROOT="$(sed -n 's/^LEHOME_CAMPAIGN_ROOT=//p' "$LEHOME_INVOCATION_FILE")"
  LEHOME_SPEND_BASELINE_USD="$(sed -n 's/^LEHOME_SPEND_BASELINE_USD=//p' "$LEHOME_INVOCATION_FILE")"
  LEHOME_SPEND_BASELINE_AT_UTC="$(sed -n 's/^LEHOME_SPEND_BASELINE_AT_UTC=//p' "$LEHOME_INVOCATION_FILE")"
  LEHOME_MAX_HOURLY_BURN_USD="$(sed -n 's/^LEHOME_MAX_HOURLY_BURN_USD=//p' "$LEHOME_INVOCATION_FILE")"
  LEHOME_SPEND_OBSERVER_COMMAND="$(sed -n 's/^LEHOME_SPEND_OBSERVER_COMMAND=//p' "$LEHOME_INVOCATION_FILE")"
else
  LEHOME_RUN_ID="fresh-run-$(date -u +%Y%m%d%H%M%S)-01"
  LEHOME_ROUND_ID="fresh-12k-${LEHOME_RUN_ID#fresh-run-}"
  LEHOME_SPEND_BASELINE_USD=20.25
  LEHOME_SPEND_BASELINE_AT_UTC=2026-08-28T14:25:00Z
  LEHOME_MAX_HOURLY_BURN_USD=1.50
  LEHOME_SPEND_OBSERVER_COMMAND="$LEHOME_HOST_CODE_ROOT/scripts/run_conservative_spend_observer.py"
  LEHOME_CAMPAIGN_ROOT="/mnt/lehome/eval/$LEHOME_RUN_ID"
  test ! -e "$LEHOME_CAMPAIGN_ROOT"
  umask 077
  printf 'LEHOME_REVIEWED_REVISION=%q\nLEHOME_CAMPAIGN_ROOT=%q\nLEHOME_RUN_ID=%q\nLEHOME_ROUND_ID=%q\nLEHOME_SPEND_BASELINE_USD=%q\nLEHOME_SPEND_BASELINE_AT_UTC=%q\nLEHOME_MAX_HOURLY_BURN_USD=%q\nLEHOME_SPEND_OBSERVER_COMMAND=%q\n' \
    "$LEHOME_REVIEWED_REVISION" "$LEHOME_CAMPAIGN_ROOT" "$LEHOME_RUN_ID" "$LEHOME_ROUND_ID" "$LEHOME_SPEND_BASELINE_USD" "$LEHOME_SPEND_BASELINE_AT_UTC" "$LEHOME_MAX_HOURLY_BURN_USD" "$LEHOME_SPEND_OBSERVER_COMMAND" > "$LEHOME_INVOCATION_FILE"
fi
case "$LEHOME_RUN_ID" in fresh-run-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-01) ;; *) exit 2;; esac
case "$LEHOME_ROUND_ID" in fresh-12k-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-01) ;; *) exit 2;; esac
test "${LEHOME_SPEND_BASELINE_USD:-}" = 20.25
test "${LEHOME_SPEND_BASELINE_AT_UTC:-}" = 2026-08-28T14:25:00Z
test "${LEHOME_MAX_HOURLY_BURN_USD:-}" = 1.50
test "${LEHOME_SPEND_OBSERVER_COMMAND:-}" = "$LEHOME_HOST_CODE_ROOT/scripts/run_conservative_spend_observer.py"
test "${LEHOME_REVIEWED_REVISION:-}" = "$(git -C "$LEHOME_HOST_CODE_ROOT" rev-parse HEAD)"
test "${LEHOME_CAMPAIGN_ROOT:-}" = "/mnt/lehome/eval/$LEHOME_RUN_ID"
export LEHOME_HOST_CODE_ROOT LEHOME_RUN_ID LEHOME_ROUND_ID LEHOME_CAMPAIGN_ROOT LEHOME_SPEND_BASELINE_USD LEHOME_SPEND_BASELINE_AT_UTC LEHOME_MAX_HOURLY_BURN_USD LEHOME_SPEND_OBSERVER_COMMAND
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

### Start the conservative local spend observer

The signed-in billing observation for Aug 27–28 was $20.25, last updated at
`2026-08-28T14:25:00Z`. The invocation record above pins that amount and a
conservative $1.50/hour upper-bound burn rate; never replace either on retry
with a lower value. The observer makes **no** provider or Hub request. It
writes the controller's exact `lehome_spend_observation_v1` schema every 30
seconds and rounds the elapsed-time estimate upward with extra headroom. This
is deliberately an upper bound: it may stop early, but cannot silently spend
past the $99 controller cutoff.

Run this after catalog readback and immediately before the paid controller. It
persists the PID beside the invocation record, waits for a fresh receipt, and
terminates the local observer whenever the controller exits (including after a
terminal handoff). Do not run the paid controller if any command
below fails.

```bash
set -euo pipefail
LEHOME_SPEND_OBSERVER=/mnt/lehome/operator/spend-observation.json
LEHOME_SPEND_OBSERVER_PID_FILE="${LEHOME_INVOCATION_FILE}.spend-observer.pid"
python3 "$LEHOME_SPEND_OBSERVER_COMMAND" \
  --output "$LEHOME_SPEND_OBSERVER" \
  --baseline-usd "$LEHOME_SPEND_BASELINE_USD" \
  --baseline-observed-at-utc "$LEHOME_SPEND_BASELINE_AT_UTC" \
  --max-hourly-burn-usd "$LEHOME_MAX_HOURLY_BURN_USD" \
  --interval-seconds 30 \
  --observer lehome-conservative-local-upper-bound-v1 &
LEHOME_SPEND_OBSERVER_PID=$!
kill -0 "$LEHOME_SPEND_OBSERVER_PID"
(umask 077; printf '%s\n' "$LEHOME_SPEND_OBSERVER_PID" > "$LEHOME_SPEND_OBSERVER_PID_FILE")
for _ in 1 2 3 4 5; do test -s "$LEHOME_SPEND_OBSERVER" && break; sleep 1; done
python3 - "$LEHOME_SPEND_OBSERVER" <<'PY'
import json
import sys
from datetime import UTC, datetime, timedelta

receipt = json.load(open(sys.argv[1], encoding="utf-8"))
required = {"schema_version", "kind", "observer", "observed_at_utc", "spent_usd"}
observed = datetime.fromisoformat(receipt["observed_at_utc"].replace("Z", "+00:00"))
if set(receipt) != required or receipt["kind"] != "lehome_spend_observation_v1" or observed < datetime.now(UTC) - timedelta(seconds=60):
    raise SystemExit("spend observer receipt is missing, stale, or malformed")
PY
cleanup_spend_observer() {
  if kill -0 "$LEHOME_SPEND_OBSERVER_PID" 2>/dev/null; then
    kill -TERM "$LEHOME_SPEND_OBSERVER_PID"
    wait "$LEHOME_SPEND_OBSERVER_PID" || true
  fi
}
trap cleanup_spend_observer EXIT INT TERM
export LEHOME_SPEND_OBSERVER
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
  tests/infrastructure/test_simple_curriculum_terminal_handoff.py \
  tests/infrastructure/test_simple_curriculum_operator_finalizer.py \
  tests/infrastructure/test_success_replay_campaign.py \
  tests/infrastructure/test_rollout_container.py \
  tests/infrastructure/test_simple_curriculum_runbook.py
bash -n rollout_appliance/run_12k_campaign.sh
bash -n rollout_appliance/run_success_replay_campaign.sh
bash -n rollout_appliance/run_simple_curriculum_collection.sh
bash -n scripts/run_simple_curriculum_with_finalizer.sh
python3 -m py_compile scripts/build_simple_curriculum_matrix.py \
  scripts/check_simple_curriculum_gate.py scripts/build_success_replay_matrix.py \
  scripts/run_simple_curriculum_collection.py scripts/publish_simple_curriculum_collection.py \
  scripts/finalize_simple_curriculum_collection.py
env -i PATH="$PATH" PYTHONPATH="$PWD:$PWD/source/lehome:$PWD/trainer/src" \
  python3 scripts/run_simple_curriculum_collection.py --help
git diff --check
```

## 5. One paid command

After Sections 1–3 pass, run this exact command on the **operator machine**.
It is the only paid-controller invocation. It reads the guarded remote
invocation record as data, executes the fixed remote controller argv once, and
always invokes the local finalizer exactly once afterward. Retry the same
command with the same `LEHOME_INVOCATION_FILE`, never regenerated IDs.

```bash
LEHOME_OPERATOR_SSH_TARGET="${LEHOME_VM_SSH_TARGET:?approved user@host SSH target}" \
LEHOME_OPERATOR_SSH_PORT="${LEHOME_VM_SSH_PORT:-22}" \
LEHOME_OPERATOR_CAMPAIGN_ROOT="$LEHOME_CAMPAIGN_ROOT" \
LEHOME_OPERATOR_RUN_ID="$LEHOME_RUN_ID" \
LEHOME_OPERATOR_ROUND_ID="$LEHOME_ROUND_ID" \
LEHOME_OPERATOR_REVIEWED_REVISION="$LEHOME_REVIEWED_REVISION" \
LEHOME_OPERATOR_HF_TOKEN_FILE="${LEHOME_OPERATOR_HF_TOKEN_FILE:?operator-owned local 0600 token path}" \
./scripts/run_simple_curriculum_with_finalizer.sh
```

The wrapper admits no provider lifecycle command. It does not source the
invocation file, accept an arbitrary remote command, or print a token. A
terminal controller return is
`operator_stop_required` and leaves a compact immutable handoff at
`$LEHOME_CAMPAIGN_ROOT/reports/operator-stop-handoff.json`; it neither stops
the VM nor publishes. No secret is passed on the command line.

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
  or finalizer stop failure is fail closed. The status is reported as an
  infrastructure outcome, never a data success.

The controller polls the typed spend receipt before and after every paid stage
and while a child runs. Once a terminal outcome is known, while the VM is
still running it publishes only a compact immutable provisional JSON evidence
bundle (never rollout media) and records its pinned revision. The local
finalizer may promote/seal only after authoritative STOPPED.

## 7. Fresh terminal evidence, replay, and publication

When the paid controller returns `operator_stop_required`, do **not** run it
again, SSH for raw rollouts, or attempt publication from the VM. Section 5's
wrapper already invokes the local finalizer once with the persisted IDs. Its
safety trap means an SSH/controller failure cannot skip the exact-VM stop; the
finalizer fetches only the compact handoff into a temporary directory, then
downloads only the pinned provisional JSON bundle from Hub (never raw campaign
bytes or rollout media), and always calls exact-ID Compute `get`/`stop`/`get`
before any promotion upload.

The local finalizer pins `computeinstance-u00t6xfqhadrcmssa2`, name
`lehome-rollout`, and attached protected disk
`computedisk-u00pbe55crxy7jr56x`; it has no create/start/delete/list path. The
wrapper invokes the local finalizer after the remote controller returns or
fails. A handoff validation failure still stops that exact VM and reports
`infrastructure_stop_failure`; an HF failure after STOPPED leaves it stopped
and is a zero-compute publication retry. The token file must be regular,
non-symlink, owned by the effective operator UID, mode `0600`, and nonempty; it is never included in
arguments or receipts.

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
automatic paid restart. The only retryable boundary is the local finalizer's
zero-compute public readback after verified stop. If it crashes or loses the
response after uploading `final-publication.json`, rerun the same local
finalizer with the same persisted operator variables. It pins and re-reads the
existing immutable prefix authenticated and anonymously; it never reruns the
controller, creates compute, or uses a different run/round. A malformed
receipt, changed remote byte, missing file, or non-public readback fails
closed and leaves the VM stopped.

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
