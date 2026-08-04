# Physical SO101 DAgger collection

This workflow keeps the two leader arms on the operator's Mac. The remote Isaac
host receives authenticated samples only through an SSH loopback forward; do not
open a public teleoperation port or attach the leaders to the remote host.

## Before the first collection

1. Confirm the left and right physical buses, their calibration JSON files, and
   their USB identities. The bridge rejects duplicate identities and malformed
   calibration files.
2. Start on a user-approved North American GUI host. Do not use a paid host for
   this checklist until all local tests pass.
3. Create the remote session with the default practice mode. It creates a fresh
   owner-only session secret and a manifest which contains the nonce but never
   the secret or its path. Copy the secret through an already authenticated
   secure channel into the Mac bridge's fixed private state location while the
   session is active. The collector overwrites and removes only the secret it
   created when the session exits; do not reuse it. Do not put secret contents
   or a secret-file path in command arguments, logs, manifests, screenshots, or
   tickets.
4. Use the printed forwarding template after replacing its host placeholder:

   ```bash
   ssh -N -L 18080:127.0.0.1:18080 USER@APPROVED_NORTH_AMERICAN_HOST
   ```

   The listener and the Mac client both default to `127.0.0.1:18080`.

## Safe collection modes

- `practice` is the default. It records controls/diagnostics only and never
  creates `exports/`.
- `expert` begins with expert control. Training output is permitted only with a
  real, pinned organizer quality-threshold manifest.
- `dagger` begins with policy control. Takeover is one-way: the operator must
  synchronize both 12D leader commands to the simulated robot, queued policy
  actions are cleared, and only post-takeover expert actions may be exported.

Controls are `space` (activate/request takeover), `a` (accept only after
official success), `d` (discard), `r` (reset), and `escape` (safe exit). A
manual accept cannot turn an official failure, stale/disconnected bridge, or
safety rejection into a training example.

The remote collector requires these explicit conditions before it enables an
interactive expert or DAgger loop. Validation completes before importing Isaac
or constructing an app launcher; a failed command has not started a simulator.

```bash
python scripts/collect_groot_dagger.py \
  --mode dagger --enable-training-output \
  --quality-thresholds /secure/quality-thresholds.json \
  --organizer-dataset-revision <40-lowercase-hex-revision> \
  --organizer-dataset-sha256 <64-lowercase-hex-sha256> \
  --left-calibration-sha256 <64-lowercase-hex-sha256> \
  --right-calibration-sha256 <64-lowercase-hex-sha256> \
  --policy-path /secure/pinned-groot-checkpoint \
  --policy-revision <40-lowercase-hex-revision> \
  --policy-repo <policy-repository> \
  --policy-step <non-negative-step> \
  --policy-artifact-sha256 <64-lowercase-hex-sha256> \
  --image-identity <immutable-image-identity> \
  --code-revision <40-lowercase-hex-revision> \
  --asset-revision <40-lowercase-hex-revision> \
  --simulator-version <simulator-version> \
  --episode-id <unique-episode-id> \
  --garment <organizer-garment-name> \
  --category pant_long --release-stage seen \
  --interactive
```

The threshold file must be generated from organizer expert statistics and pin
the organizer dataset revision, dataset SHA-256, statistics SHA-256, sample
count, quantiles, and all hard/clean limits. This repository intentionally does
not supply guessed values.

Once the command prints its loopback forwarding instruction, press `space` to
start policy control (DAgger) or expert control (full expert). For DAgger,
press `space` again only after the receiver reports a fresh, synchronized 12D
command; this clears queued policy actions, records a takeover snapshot, and
makes each following applied action expert-labelled. Press `a` only after the
task's official success signal; otherwise press `d` or `escape`. Terminal input
is read in the background, so sampling never waits for a prompt. `r` resets the
task but finalizes the partial attempt as diagnostic-only; begin the next
immutable episode with a fresh invocation and episode ID.

Every attempt writes immutable raw episode evidence. Only an accepted,
quality-A/B non-practice attempt with reset/takeover/terminal snapshots, three
nonempty encoded camera videos, measured motion/jitter metrics, and eligible
post-takeover expert windows receives an atomic `exports/<episode-id>/` receipt.
That receipt contains `selection-report.json`, `expert-windows.json`, and its
own SHA-256 manifest. Holds, bridge faults, missing metric evidence, discarded
attempts, policy-only prefixes, and practice attempts remain diagnostic-only.

## Bridge operation

After the secret has been copied to the Mac's fixed private state location, run
the bridge with the remote nonce shown in the session manifest. This opens serial
devices only after it validates all options and the mode-0600 secret file:

```bash
lehome-bridge \
  --left-port /dev/cu.usbmodem-left \
  --right-port /dev/cu.usbmodem-right \
  --left-calibration /secure/left_so101_leader.json \
  --right-calibration /secure/right_so101_leader.json \
  --session-nonce <remote-session-nonce>
```

The version-2 bridge sends a canonical-JSON/HMAC-SHA256 handshake, nonce-bound
ping/ack probes, and strictly sequenced 30 Hz 12D samples. The RTT is measured
only on the Mac clock; arrival cadence and sender-clock deltas are separate
buffering/jitter checks, never one-way latency inferred from unrelated clocks.
Each sample must carry a fresh in-limit RTT measurement and lie within the
motor limits advertised by its handshake before conversion. It neither accepts
nor prints a secret-file option. Install the standalone `lehome-bridge` package
on the receiver host (or expose `bridge/src` beside `source/lehome`) before
connecting; the receiver imports its wire verifier only when a bridge client
connects.

The remote receiver holds the last safe command on any authentication, replay,
out-of-order, disconnect, stale-age, RTT, buffering, jitter, or raw-limit
failure. It records both valid raw leader values and converted follower commands.
After a fault, wait for fresh stable RTT and cadence, then explicitly
resynchronize; never continue through a hold. An unsafe command is never sent
to the environment and makes the entire attempt diagnostic-only regardless of
the configured quality threshold.

## Physical acceptance evidence (not yet satisfied by local tests)

On the approved host and after a successful loopback bridge health check:

1. Run 10 practice episodes and prove there is no `exports/` directory.
2. Record one accepted full-expert success.
3. Record one accepted DAgger recovery and prove every selected BC target is
   expert-controlled and after takeover.
4. Record one discarded DAgger attempt and prove it produces zero BC targets.
5. Copy the raw evidence off-host through the user-approved loopback SSH tunnel
   and checksum-verify it before stopping the host.

Do not claim these physical/remote gates from local unit-test results.
