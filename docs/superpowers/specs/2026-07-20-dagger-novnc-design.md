# Persistent noVNC and DAgger Collector Design

## Objective

Provide a restart-recoverable browser desktop for Isaac Sim and a dedicated
DAgger collector that lets a pi0.5 policy create a difficult garment state,
then hands control to a human using the existing dual-arm keyboard controller.
Only expert-controlled corrections become behavior-cloning targets.

## Scope

This change includes:

- Persistent scripts for installing and starting Xfce, X11, VNC, and noVNC on
  the RunPod A6000 after container-disk loss.
- A dedicated `scripts.dagger_sim` entry point for policy rollout, one-way
  keyboard intervention, diagnostic recording, and corrective dataset output.
- Unit tests for the intervention state machine and training-data boundary.
- A live A6000 smoke test covering the desktop, browser keyboard events,
  OpenPI policy connection, intervention, reset, and saved dataset schema.

This change does not include:

- Retraining pi0.5.
- Automatically merging corrective data into the original training dataset.
- Switching from expert control back to policy control within an episode.
- Retrospectively labeling policy-controlled frames with expert actions.
- Running Isaac Sim or teleoperation from a mobile device.

## Existing System

The LeHome repository already provides:

- Dual-arm keyboard control through `BiKeyboard`.
- LeRobot dataset recording helpers.
- Policy evaluation through `PolicyRegistry`.
- An OpenPI WebSocket adapter that requests pi0.5 action chunks.
- Simulator geometric success checks.

The A6000's `/workspace` data persists, but its container disk is reset when the
Pod is stopped. The previous desktop packages disappeared after a restart.
Therefore, installed packages are treated as disposable while scripts,
datasets, checkpoints, logs, and configuration live under `/workspace`.

## Architecture

### Persistent desktop runtime

The repository will contain focused runtime scripts under `scripts/runtime/`:

- `install_novnc.sh` installs only missing Ubuntu desktop packages and creates
  required runtime directories.
- `start_novnc.sh` starts Xvfb, Xfce, x11vnc, and websockify/noVNC in the correct
  order, writes PID files, and is safe to run repeatedly.
- `stop_novnc.sh` stops only processes started by this runtime.
- `status_novnc.sh` reports process, display, port, and HTTP health.

Runtime state and logs are written under `/workspace/runtime/novnc` and
`/workspace/logs`. The scripts detect a fresh container and reinstall missing
packages. noVNC is passwordless, matching the user's earlier request. The
RunPod proxy URL must therefore be treated as a secret because possession of
the URL grants desktop access.

The start script will use an HTTP port already exposed by the active RunPod
template. Its local upstream port will be selected after inspecting the nginx
proxy mapping. It will fail with a clear message if the required exposed port
is unavailable instead of silently starting an unreachable service.

### DAgger collector

`scripts/dagger_sim.py` launches Isaac Sim, imports the task and collection
implementation after `SimulationApp` starts, and closes the application on all
exit paths. `scripts/utils/dagger_record.py` owns the collection loop.

The collector creates both:

1. A diagnostic record for the complete episode, including the policy prefix,
   intervention step, success result, seed, garment, checkpoint identifier,
   replan interval, and video.
2. A LeRobot correction dataset containing only frames produced after expert
   takeover.

The correction dataset reuses the source training dataset's feature schema and
FPS so it can be merged safely later. Depth and camera features are not guessed;
they are derived from the source dataset metadata. The task description remains
`fold the garment on the table` unless explicitly overridden.

### Intervention state machine

An Isaac-independent state machine makes the control and recording rules
testable:

```text
POLICY --P--> EXPERT --N + simulator success--> SAVE_CORRECTION --> RESET
  |                 |--D-----------------------> DISCARD --> RESET
  |                 |--timeout/error-----------> ARCHIVE_FAILURE --> RESET
  |--timeout/error-----------------------------> ARCHIVE_FAILURE --> RESET
  |--Esc---------------------------------------> EXIT
EXPERT --Esc-----------------------------------> EXIT
```

Controls:

- `P`: one-way takeover from pi0.5 to the dual-arm keyboard controller.
- `N`: save the correction only when the simulator success checker is true.
- `D`: end the attempt, archive it as `operator_discarded` diagnostics, clear
  the correction buffer, and reset without adding training data.
- `Esc`: finalize pending metadata safely and exit.

`P` is unused by the existing dual-arm joint mappings. Once takeover occurs,
the OpenPI action queue is ignored for the rest of the episode. The expert
controller starts from the robot's current joint state, avoiding a command jump
at the intervention boundary.

## Data Boundaries

Policy actions are never written into the correction dataset. At takeover:

1. The current observation and intervention step are captured in diagnostic
   metadata.
2. The expert controller is synchronized to current robot joint positions.
3. The first expert command is applied.
4. Only the resulting expert-controlled observation/action pair is added to the
   LeRobot episode buffer.

If the expert cannot recover the garment, the diagnostic episode is retained as
`success=false`, while the correction buffer is cleared. This preserves failure
evidence without teaching behavior cloning to imitate a failure.

Successful correction metadata includes:

- `source: dagger_correction`
- policy config and checkpoint path
- garment name and version
- environment seed
- OpenPI replan interval
- policy prefix length
- expert correction length
- simulator success result
- timestamps and diagnostic video paths

## Output Layout

Default persistent paths:

```text
/workspace/lehome-challenge/Datasets/dagger/<run-name>/
  corrections/001/                # LeRobot dataset
  diagnostics/
    episodes.jsonl                 # episode metadata
    success/                       # complete diagnostic videos
    failure/                       # complete diagnostic videos

/workspace/runtime/novnc/
  pids/
  state/

/workspace/logs/
  novnc-*.log
  dagger-<run-name>.log
```

The collector refuses to overwrite or resume an existing run directory. Every
invocation creates a new numbered dataset directory using the repository's
existing convention.

## Error Handling

- Desktop installation is idempotent and stops on package-manager failures.
- Desktop startup validates each process and the proxied HTTP endpoint.
- The collector validates task/device compatibility, source dataset existence,
  policy connectivity, action shape, and output schema before starting.
- Policy inference failure archives diagnostics and stops the episode without
  adding correction frames.
- A simulator or recorder exception clears only the current LeRobot buffer,
  preserves completed episodes, finalizes metadata, and closes Isaac Sim.
- `N` before takeover or before simulator success is ignored with an explicit
  status message.
- A timeout after takeover archives failure diagnostics but does not save an
  incomplete correction as training data.

## Testing

### Automated tests

Tests run without Isaac Sim by isolating the state machine and frame-routing
rules. They prove:

- Collection starts in policy mode.
- `P` produces exactly one policy-to-expert transition.
- Policy frames never enter the correction sink.
- Expert frames enter the correction sink after takeover.
- `N` cannot save without takeover and simulator success.
- A successful recovery saves exactly one correction episode.
- Discard, timeout, inference failure, and exit clear or finalize the correct
  buffers.
- Repeated takeover input does not create duplicate transitions.

Shell scripts receive syntax checks and a dry-run/status test that does not
install packages locally.

### Live A6000 verification

The remote acceptance check requires:

1. `status_novnc.sh` reports healthy Xvfb, Xfce, x11vnc, websockify, and HTTP
   proxy connectivity.
2. The noVNC page loads from the RunPod proxy.
3. Isaac Sim opens on the remote display.
4. With the viewport focused, `B` enables the existing dual-arm controller and
   joint keys visibly move the appropriate simulated arm.
5. A collector smoke episode begins under pi0.5, `P` stops policy execution,
   and keyboard control continues without resetting the garment.
6. A discarded smoke episode writes diagnostics but no correction episode.
7. A deliberately completed or test-injected success writes a correction whose
   first saved frame occurs at or after the intervention step and whose schema
   matches the source dataset.

## Rollout and Operational Use

The initial live session uses the existing A6000, OpenPI checkpoint, and a
shorter replan interval of 10. The operator should first practice clean keyboard
folds, then collect 10 to 20 recoveries. The unseen evaluation garment remains
excluded from DAgger collection so it stays a valid holdout.

After a future Pod restart, recovery is:

```bash
cd /workspace/lehome-challenge
bash scripts/runtime/install_novnc.sh
bash scripts/runtime/start_novnc.sh
```

The status command provides the correct proxy endpoint and next collector
command, eliminating reliance on container-disk setup history.
