# GR00T Flywheel Physical-Arm DAgger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect two physical SO101 leader arms on macOS to remote Isaac Sim through SSH and collect practice, full-expert, and expert-only DAgger recovery episodes safely.

**Architecture:** A standalone Mac-compatible package reads both serial buses and emits authenticated, sequenced samples to a loopback-only remote receiver through an SSH tunnel. The simulator-side intervention controller owns synchronization, one-way takeover, quality grading, and the expert-only export boundary.

**Tech Stack:** Python 3.11, pyserial, Feetech serial protocol, standard-library sockets/HMAC, SSH port forwarding, NumPy, pytest, Isaac Sim adapter interfaces.

---

## File structure

- Create `bridge/pyproject.toml`: Mac-only package without Isaac/PyTorch dependencies.
- Create `bridge/src/lehome_bridge/protocol.py`: signed length-prefixed messages.
- Create `bridge/src/lehome_bridge/leaders.py`: dual serial-bus reader and calibration identity.
- Create `bridge/src/lehome_bridge/client.py`: fixed-rate bridge client.
- Create `bridge/src/lehome_bridge/cli.py`: `lehome-bridge` command.
- Create `bridge/tests/`: protocol, sequencing, and fake-bus tests.
- Create `source/lehome/lehome/flywheel/bridge_receiver.py`: loopback receiver and health gate.
- Create `source/lehome/lehome/flywheel/intervention.py`: practice/expert/DAgger state machine.
- Create `source/lehome/lehome/flywheel/quality.py`: A/B/C grading.
- Create `scripts/collect_groot_dagger.py`: interactive remote collector.
- Create `tests/flywheel/test_bridge_receiver.py`, `test_intervention.py`, and `test_quality.py`.

### Task 1: Standalone authenticated bridge protocol

**Files:**
- Create: `bridge/pyproject.toml`
- Create: `bridge/src/lehome_bridge/__init__.py`
- Create: `bridge/src/lehome_bridge/protocol.py`
- Test: `bridge/tests/test_protocol.py`

- [ ] **Step 1: Write signed-message and replay tests**

```python
import pytest
from lehome_bridge.protocol import BridgeMessage, MessageVerifier, encode_message


def test_signed_message_round_trip_and_replay_rejection() -> None:
    secret = b"x" * 32
    message = BridgeMessage.handshake(
        session_nonce="nonce-1",
        sequence=0,
        left_serial="left-001",
        right_serial="right-002",
        left_calibration_sha256="a" * 64,
        right_calibration_sha256="b" * 64,
        hz=30,
    )
    verifier = MessageVerifier(secret=secret, expected_nonce="nonce-1")
    wire = encode_message(message, secret=secret)
    assert verifier.verify(wire).sequence == 0
    with pytest.raises(ValueError, match="sequence"):
        verifier.verify(wire)


def test_tampered_message_fails_authentication() -> None:
    wire = bytearray(encode_message(BridgeMessage.sample("n", 1, 10, [0.0] * 12), secret=b"k" * 32))
    wire[-1] ^= 1
    with pytest.raises(ValueError, match="authentication"):
        MessageVerifier(secret=b"k" * 32, expected_nonce="n").verify(bytes(wire))
```

- [ ] **Step 2: Run the isolated bridge test and verify failure**

Run: `uv run --project bridge pytest bridge/tests/test_protocol.py -v`

Expected: FAIL because `lehome_bridge.protocol` does not exist.

- [ ] **Step 3: Implement canonical JSON, HMAC-SHA256, framing, and sequence checks**

```python
def encode_message(message: BridgeMessage, *, secret: bytes) -> bytes:
    payload = json.dumps(message.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    signature = hmac.digest(secret, payload, "sha256")
    body = signature + payload
    return struct.pack("!I", len(body)) + body


class MessageVerifier:
    def verify(self, wire: bytes) -> BridgeMessage:
        signature, payload = split_frame(wire)
        expected = hmac.digest(self.secret, payload, "sha256")
        if not hmac.compare_digest(signature, expected):
            raise ValueError("bridge message authentication failed")
        message = BridgeMessage.from_json(payload)
        if message.session_nonce != self.expected_nonce:
            raise ValueError("bridge session nonce mismatch")
        if message.sequence != self.next_sequence:
            raise ValueError("bridge sequence is stale, duplicate, or reordered")
        self.next_sequence += 1
        return message
```

Limit frames to 64 KiB, require protocol version 1, reject duplicate left/right serial identities, reject nonfinite 12D samples, and never serialize the secret. Configure the package with dependencies only on `pyserial` and the repository's extracted Feetech transport module.

- [ ] **Step 4: Run protocol tests**

Run: `uv run --project bridge pytest bridge/tests/test_protocol.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the bridge wire contract**

```bash
git add bridge/pyproject.toml bridge/src/lehome_bridge/__init__.py bridge/src/lehome_bridge/protocol.py bridge/tests/test_protocol.py
git commit -m "feat: define authenticated leader bridge protocol"
```

### Task 2: Dual physical leader reader on macOS

**Files:**
- Create: `bridge/src/lehome_bridge/leaders.py`
- Create: `bridge/src/lehome_bridge/client.py`
- Test: `bridge/tests/test_leaders.py`

- [ ] **Step 1: Write fake-bus identity, order, and calibration tests**

```python
def test_dual_reader_returns_left_then_right_joint_order(tmp_path) -> None:
    left = FakeBus(serial="L1", positions={name: index for index, name in enumerate(JOINTS)})
    right = FakeBus(serial="R1", positions={name: index + 10 for index, name in enumerate(JOINTS)})
    reader = DualLeaderReader(left, right, left_calibration=calibration(tmp_path, "left"), right_calibration=calibration(tmp_path, "right"))
    sample = reader.read()
    assert sample.positions == (0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15)
    assert sample.left_serial == "L1"
    assert sample.right_serial == "R1"


def test_reader_rejects_same_bus_or_wrong_calibration_hash(tmp_path) -> None:
    bus = FakeBus(serial="same", positions={name: 0 for name in JOINTS})
    with pytest.raises(ValueError, match="distinct"):
        DualLeaderReader(bus, bus, left_calibration=calibration(tmp_path, "left"), right_calibration=calibration(tmp_path, "right"))
```

- [ ] **Step 2: Verify fake-bus tests fail**

Run: `uv run --project bridge pytest bridge/tests/test_leaders.py -v`

Expected: FAIL with missing leaders module.

- [ ] **Step 3: Implement explicit bus injection and 30 Hz client loop**

```python
class DualLeaderReader:
    def read(self) -> LeaderSample:
        left = self.left_bus.sync_read("Present_Position")
        right = self.right_bus.sync_read("Present_Position")
        values = tuple(left[name] for name in JOINTS) + tuple(right[name] for name in JOINTS)
        return LeaderSample(
            monotonic_ns=time.monotonic_ns(),
            positions=finite_12(values),
            left_serial=self.left_bus.serial_identity,
            right_serial=self.right_bus.serial_identity,
        )


def stream(reader: DualLeaderReader, connection: BridgeConnection, *, hz: int = 30) -> None:
    period_ns = int(1_000_000_000 / hz)
    deadline = time.monotonic_ns()
    while not connection.stop_requested:
        connection.send_sample(reader.read())
        deadline += period_ns
        sleep_until_monotonic_ns(deadline)
```

Reuse the existing motor definitions and calibration JSON structure; do not instantiate keyboard listeners or import Isaac Sim. The client reads the session secret from a mode-0600 file, rejects insecure permissions, and defaults to the SSH-forwarded endpoint `127.0.0.1:18080`.

- [ ] **Step 4: Run the complete bridge suite with no hardware**

Run: `uv run --project bridge pytest bridge/tests -v`

Expected: PASS and no serial device is opened by the tests.

- [ ] **Step 5: Commit the Mac leader client**

```bash
git add bridge/src/lehome_bridge/leaders.py bridge/src/lehome_bridge/client.py bridge/tests/test_leaders.py
git commit -m "feat: stream dual SO101 leader samples"
```

### Task 3: Remote receiver and latency health gate

**Files:**
- Create: `source/lehome/lehome/flywheel/bridge_receiver.py`
- Test: `tests/flywheel/test_bridge_receiver.py`

- [ ] **Step 1: Write stale, jitter, disconnect, and hold tests**

```python
def test_receiver_holds_after_stale_sample_and_requires_resync() -> None:
    receiver = BridgeReceiver(max_age_ms=80.0, max_jitter_ms=30.0)
    receiver.accept_handshake(valid_handshake())
    receiver.accept_sample(valid_sample(sequence=1, age_ms=10.0))
    assert receiver.current().eligible is True
    receiver.accept_sample(valid_sample(sequence=2, age_ms=100.0))
    held = receiver.current()
    assert held.eligible is False
    assert held.reason == "stale_sample"
    assert held.command == receiver.last_safe_command
```

- [ ] **Step 2: Run and verify missing receiver failure**

Run: `uv run --offline pytest tests/flywheel/test_bridge_receiver.py -v`

Expected: FAIL with missing receiver module.

- [ ] **Step 3: Implement loopback-only server and fail-closed health state**

```python
class BridgeReceiver:
    def current(self, *, now_ns: int | None = None) -> ExpertCommand:
        now = time.monotonic_ns() if now_ns is None else now_ns
        if self.last_sample is None:
            return ExpertCommand(self.last_safe_command, False, "no_sample")
        age_ms = (now - self.last_sample.received_monotonic_ns) / 1_000_000
        if age_ms > self.max_age_ms or self.jitter_ms > self.max_jitter_ms:
            self.requires_resync = True
            return ExpertCommand(self.last_safe_command, False, "stale_or_jitter")
        if self.requires_resync:
            return ExpertCommand(self.last_safe_command, False, "resync_required")
        return ExpertCommand(self.convert(self.last_sample.positions), True, None)
```

Bind exactly to `127.0.0.1`, accept one client, cap read timeouts, and close on authentication/sequence failure. Convert leader positions using the same SO101 follower limits as the existing `input2action` path and record raw plus converted values.

- [ ] **Step 4: Run receiver and existing action-conversion tests**

Run: `uv run --offline pytest tests/flywheel/test_bridge_receiver.py tests/test_success_checker_challenge.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the remote receiver**

```bash
git add source/lehome/lehome/flywheel/bridge_receiver.py tests/flywheel/test_bridge_receiver.py
git commit -m "feat: gate remote expert commands by latency"
```

### Task 4: One-way intervention state machine and quality grades

**Files:**
- Create: `source/lehome/lehome/flywheel/intervention.py`
- Create: `source/lehome/lehome/flywheel/quality.py`
- Test: `tests/flywheel/test_intervention.py`
- Test: `tests/flywheel/test_quality.py`

- [ ] **Step 1: Write transition and grading tests**

```python
def test_dagger_takeover_is_one_way_and_clears_policy_queue() -> None:
    controller = InterventionController(mode="dagger", sync_tolerance_rad=0.08)
    controller.start_policy()
    controller.request_takeover()
    with pytest.raises(TransitionError, match="synchronization"):
        controller.accept_expert(current_robot=(0.0,) * 12, leader_command=(0.2,) * 12)
    controller.accept_expert(current_robot=(0.0,) * 12, leader_command=(0.01,) * 12)
    assert controller.state == "expert"
    assert controller.policy_queue_clear_requested is True
    with pytest.raises(TransitionError, match="one-way"):
        controller.start_policy()


def test_quality_grades_clean_recovery_and_rejected_attempts() -> None:
    assert grade_attempt(successful_stats()).grade == "A"
    assert grade_attempt(successful_stats(hesitations=2)).grade == "B"
    assert grade_attempt(successful_stats(stale_samples=1)).grade == "C"
```

Grade A is a clean expert success, Grade B is a successful but imperfect recovery, and Grade C is diagnostic-only and never trainable.

- [ ] **Step 2: Confirm state and quality tests fail**

Run: `uv run --offline pytest tests/flywheel/test_intervention.py tests/flywheel/test_quality.py -v`

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement explicit transitions and statistic-derived gates**

```python
ALLOWED = {
    "ready": {"practice", "expert", "policy"},
    "policy": {"takeover_pending", "diagnostic"},
    "takeover_pending": {"expert", "diagnostic"},
    "expert": {"accepted", "diagnostic"},
    "accepted": {"reset"},
    "diagnostic": {"reset", "exit"},
}


def grade_attempt(stats: AttemptStats, thresholds: QualityThresholds) -> QualityResult:
    rejection = stats.transport_rejections(thresholds) + stats.safety_rejections(thresholds)
    if not stats.official_success or rejection:
        return QualityResult("C", tuple(rejection or ("official_failure",)), 0.0)
    if stats.hesitations or stats.corrections or stats.velocity_p95 > thresholds.clean_velocity_p95:
        return QualityResult("B", ("successful_recovery",), 0.5)
    return QualityResult("A", ("clean_success",), 1.0)
```

Derive velocity, acceleration, and jitter thresholds from organizer expert statistics and store their dataset revision/hash in `quality-thresholds.json`. Manual accept cannot override official failure or a transport/safety rejection.

- [ ] **Step 4: Run state, quality, and exporter tests**

Run: `uv run --offline pytest tests/flywheel/test_intervention.py tests/flywheel/test_quality.py tests/flywheel/test_export.py -v`

Expected: PASS.

- [ ] **Step 5: Commit DAgger control and grading**

```bash
git add source/lehome/lehome/flywheel/intervention.py source/lehome/lehome/flywheel/quality.py tests/flywheel/test_intervention.py tests/flywheel/test_quality.py
git commit -m "feat: grade one-way DAgger interventions"
```

### Task 5: Interactive collector and physical acceptance

**Files:**
- Create: `bridge/src/lehome_bridge/cli.py`
- Create: `scripts/collect_groot_dagger.py`
- Create: `docs/groot_dagger_collection.md`
- Test: `bridge/tests/test_cli.py`
- Test: `tests/flywheel/test_dagger_cli.py`

- [ ] **Step 1: Write CLI safety tests**

```python
def test_collector_defaults_to_practice_and_loopback() -> None:
    args = build_parser().parse_args([])
    assert args.mode == "practice"
    assert args.listen_host == "127.0.0.1"
    assert args.enable_training_output is False


def test_training_output_requires_mode_and_threshold_manifest(tmp_path) -> None:
    args = arguments(mode="dagger", enable_training_output=True, quality_thresholds=tmp_path / "missing.json")
    with pytest.raises(ValueError, match="quality thresholds"):
        validate_args(args)
```

- [ ] **Step 2: Run CLI tests and confirm failure**

Run: `uv run --project bridge pytest bridge/tests/test_cli.py -v`

Run: `uv run --offline pytest tests/flywheel/test_dagger_cli.py -v`

Expected: both fail because the CLIs do not exist.

- [ ] **Step 3: Implement explicit controls and session manifest**

```python
CONTROLS = {
    "space": "activate_or_takeover",
    "a": "accept_after_official_success",
    "d": "discard",
    "r": "reset",
    "escape": "safe_exit",
}


def validate_args(args: argparse.Namespace) -> None:
    if args.listen_host != "127.0.0.1":
        raise ValueError("DAgger receiver must be loopback-only")
    if args.enable_training_output and args.mode not in {"expert", "dagger"}:
        raise ValueError("training output requires expert or dagger mode")
    if args.enable_training_output and not args.quality_thresholds.is_file():
        raise ValueError("quality thresholds manifest is required")
```

The remote CLI creates a mode-0600 session secret, prints the exact SSH forward command without the secret, shows bridge age/jitter/state in the interactive view, and records all controls. The Mac CLI receives the secret file through an already-authenticated secure copy or user-created local file; it never exposes it in process arguments.

- [ ] **Step 4: Run staged physical acceptance**

Run local tests:

```bash
uv run --project bridge pytest bridge/tests -v
uv run --offline pytest tests/flywheel/test_bridge_receiver.py tests/flywheel/test_intervention.py tests/flywheel/test_quality.py tests/flywheel/test_dagger_cli.py -v
```

Expected: PASS.

Then, on one user-approved North American GUI host: run 10 practice episodes that produce no `exports/` directory; one accepted full-expert episode; one accepted DAgger episode proving every selected target is expert and post-takeover; and one discarded DAgger episode proving zero BC targets. Stop the host only after the complete raw run is checksum-verified off-host.

- [ ] **Step 5: Commit collector and operator guide**

```bash
git add bridge/src/lehome_bridge/cli.py scripts/collect_groot_dagger.py docs/groot_dagger_collection.md bridge/tests/test_cli.py tests/flywheel/test_dagger_cli.py
git commit -m "feat: collect physical-arm DAgger recoveries"
```

## Plan 3 completion gate

The bridge suite passes on macOS without Isaac; simulator-side tests pass without serial hardware; and the four physical acceptance cases produce checksum-verified evidence. No autonomous policy frame, hold frame, failed attempt, practice attempt, stale sample, or unsafe command appears in a BC export.
