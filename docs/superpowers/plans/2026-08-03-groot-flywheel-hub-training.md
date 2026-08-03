# GR00T Flywheel Hub and Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize accepted expert labels into an immutable 70/30 training snapshot, recompute train-only normalization, apply gated image augmentation, publish exact Hub revisions, and advance the asynchronous flywheel only when genuinely new data exists.

**Architecture:** Existing secret-safe Hub transports and prepared-dataset validators remain the only remote boundary. A deterministic mix builder combines organizer expert episodes with accepted Grade A/B flywheel windows, creates a new prepared dataset and normalization identity, and freezes its Hub commit in an iteration manifest. Training publishes a policy commit that rollout workers adopt only between episodes.

**Tech Stack:** Python 3.11, LeRobot v3, Hugging Face Hub, GR00T N1.7 `launch_finetune.py`, JSON manifests, SHA-256, pytest.

---

## File structure

- Create `trainer/src/lehome_train/flywheel/__init__.py`: public trainer-side flywheel API.
- Create `trainer/src/lehome_train/flywheel/materialize.py`: raw accepted windows to canonical LeRobot episodes.
- Create `trainer/src/lehome_train/flywheel/mix.py`: deterministic 70/30 frame-weighted snapshot.
- Create `trainer/src/lehome_train/flywheel/augmentation.py`: supported conservative augmentation profiles and sample-sheet gate.
- Create `trainer/src/lehome_train/flywheel/iteration.py`: immutable revision and new-data state machine.
- Create `trainer/src/lehome_train/commands/flywheel.py`: validate, mix, freeze, and promote CLI.
- Modify `trainer/src/lehome_train/cli.py`: register flywheel commands.
- Modify `trainer/src/lehome_train/groot/config.py`: immutable augmentation identity.
- Modify `trainer/src/lehome_train/groot/launch.py`: official color-jitter arguments.
- Modify `trainer/src/lehome_train/data/publish.py`: publish flywheel prepared snapshots through existing verification.
- Create corresponding `trainer/tests/test_flywheel_*.py` tests.

### Task 1: Canonical LeRobot materialization from accepted windows

**Files:**
- Create: `trainer/src/lehome_train/flywheel/__init__.py`
- Create: `trainer/src/lehome_train/flywheel/materialize.py`
- Test: `trainer/tests/test_flywheel_materialize.py`

- [ ] **Step 1: Write acceptance and contamination tests**

```python
def test_materializer_writes_three_camera_12d_expert_episode(tmp_path) -> None:
    raw = accepted_grade_a_episode(tmp_path, frames=32, takeover_step=4)
    output = tmp_path / "materialized"
    report = materialize_episode(raw, output)
    assert report.selected_observations == 13
    assert report.rejected_by_reason["policy"] == 4
    assert read_feature_shape(output, "observation.state") == (12,)
    assert read_feature_shape(output, "action") == (12,)
    assert camera_keys(output) == ("top_rgb", "left_rgb", "right_rgb")


def test_materializer_rejects_grade_c_holdout_and_nonexpert_targets(tmp_path) -> None:
    with pytest.raises(ValueError, match="Grade C"):
        materialize_episode(grade_c_episode(tmp_path), tmp_path / "out-c")
    with pytest.raises(ValueError, match="holdout"):
        materialize_episode(holdout_episode(tmp_path), tmp_path / "out-h")
```

- [ ] **Step 2: Run the focused trainer test and verify failure**

Run: `cd trainer && uv run --offline pytest tests/test_flywheel_materialize.py -v`

Expected: FAIL with missing `lehome_train.flywheel`.

- [ ] **Step 3: Implement verified raw ingestion and canonical episode creation**

```python
def materialize_episode(raw_root: Path, output_root: Path) -> MaterializationReport:
    verified = verify_raw_episode(raw_root)
    if verified.quality_grade == "C":
        raise ValueError("Grade C episodes cannot enter training")
    if verified.garment_name in PUBLIC_UNSEEN_HOLDOUTS:
        raise ValueError("evaluation holdout cannot enter training")
    windows = select_expert_windows(
        verified.frames,
        horizon=16,
        accepted_success=verified.official_success,
    )
    if not windows:
        raise ValueError("accepted episode contains no complete expert windows")
    writer = CanonicalLeRobotWriter(output_root, fps=30, features=GROOT_FEATURES)
    for window in windows:
        writer.add_frame(
            cameras=verified.cameras_at(window.observation_step),
            state=verified.state_at(window.observation_step),
            action=window.future_actions[0],
            task="fold the garment on the table",
        )
    writer.save_episode(provenance=verified.training_provenance())
    return writer.report()
```

Use the existing converter's canonical feature names and video settings. Copy provenance into sidecar metadata, not GR00T feature columns. Verify every raw hash before decoding and every written video/frame count afterward.

- [ ] **Step 4: Run materialization and existing data-validation tests**

Run: `cd trainer && uv run --offline pytest tests/test_flywheel_materialize.py tests/test_data_convert.py tests/test_data_validate.py -v`

Expected: PASS.

- [ ] **Step 5: Commit canonical flywheel materialization**

```bash
git add trainer/src/lehome_train/flywheel/__init__.py trainer/src/lehome_train/flywheel/materialize.py trainer/tests/test_flywheel_materialize.py
git commit -m "feat: materialize accepted flywheel episodes"
```

### Task 2: Deterministic 70/30 expert mixture and train-only normalization

**Files:**
- Create: `trainer/src/lehome_train/flywheel/mix.py`
- Modify: `trainer/src/lehome_train/data/stats.py`
- Modify: `trainer/src/lehome_train/data/validate.py`
- Test: `trainer/tests/test_flywheel_mix.py`
- Test: `trainer/tests/test_normalization.py`

- [ ] **Step 1: Write ratio, grade-weight, and holdout tests**

```python
def test_mix_targets_seventy_thirty_by_training_frames(tmp_path) -> None:
    organizer = dataset_fixture(tmp_path / "organizer", frames=700, source="organizer")
    new = dataset_fixture(tmp_path / "new", frames=300, source="flywheel", grades=("A", "B"))
    plan = build_mix_plan(organizer, new, seed=20260803)
    assert plan.organizer_training_frames == 700
    assert plan.flywheel_training_frames == 300
    assert plan.source_weights == {"organizer": 0.7, "flywheel": 0.3}
    assert plan.grade_weights == {"A": 1.0, "B": 0.5}


def test_mix_rejects_policy_frames_and_public_unseen_holdout(tmp_path) -> None:
    with pytest.raises(ValueError, match="non-expert"):
        build_mix_plan(dataset_fixture(tmp_path / "org"), contaminated_dataset(tmp_path / "bad"), seed=1)
    with pytest.raises(ValueError, match="holdout"):
        build_mix_plan(dataset_fixture(tmp_path / "org2"), holdout_dataset(tmp_path / "held"), seed=1)
```

- [ ] **Step 2: Verify mix tests fail first**

Run: `cd trainer && uv run --offline pytest tests/test_flywheel_mix.py tests/test_normalization.py -v`

Expected: FAIL with missing mix module while existing normalization tests still pass.

- [ ] **Step 3: Implement deterministic sampling manifest and prepared snapshot**

```python
def build_mix_plan(
    organizer: Path,
    flywheel: Path,
    *,
    seed: int,
    organizer_fraction: float = 0.70,
) -> MixPlan:
    organizer_frames = eligible_frames(organizer, required_source="organizer")
    flywheel_frames = eligible_frames(flywheel, required_source="expert")
    reject_holdouts(flywheel_frames, PUBLIC_UNSEEN_HOLDOUTS)
    target_total = max(
        len(organizer_frames),
        math.ceil(len(flywheel_frames) / (1.0 - organizer_fraction)),
    )
    selected_organizer = deterministic_cycle(organizer_frames, round(target_total * organizer_fraction), seed)
    selected_flywheel = grade_weighted_cycle(flywheel_frames, target_total - len(selected_organizer), seed)
    return MixPlan.freeze(selected_organizer, selected_flywheel, seed=seed)
```

Materialize a single prepared LeRobot snapshot from the frozen frame/episode plan, then run the existing split and `compute_dataset_statistics()` only after holdout removal. Record source dataset commits, raw episode hashes, selected IDs, Grade A/B counts, rejected counts, and the mix-plan SHA-256 in `manifest.json`.

- [ ] **Step 4: Run mix, statistics, and validation suites**

Run: `cd trainer && uv run --offline pytest tests/test_flywheel_mix.py tests/test_normalization.py tests/test_data_stats.py tests/test_data_validate.py -v`

Expected: PASS and a tampered selection manifest fails before statistics are trusted.

- [ ] **Step 5: Commit deterministic mixing**

```bash
git add trainer/src/lehome_train/flywheel/mix.py trainer/src/lehome_train/data/stats.py trainer/src/lehome_train/data/validate.py trainer/tests/test_flywheel_mix.py trainer/tests/test_normalization.py
git commit -m "feat: build train-only flywheel mixtures"
```

### Task 3: Conservative, immutable image-augmentation profiles

**Files:**
- Create: `trainer/src/lehome_train/flywheel/augmentation.py`
- Modify: `trainer/src/lehome_train/groot/config.py`
- Modify: `trainer/src/lehome_train/groot/launch.py`
- Test: `trainer/tests/test_flywheel_augmentation.py`
- Test: `trainer/tests/test_groot_config.py`
- Test: `trainer/tests/test_groot_launch.py`

- [ ] **Step 1: Write profile and official-command tests**

```python
def test_mild_profile_matches_checked_nvidia_cli_contract() -> None:
    profile = augmentation_profile("mild")
    assert profile.color_jitter == {
        "brightness": 0.20,
        "contrast": 0.20,
        "saturation": 0.20,
        "hue": 0.05,
    }
    assert profile.sha256 == augmentation_profile("mild").sha256


def test_launch_passes_color_jitter_as_eight_official_cli_tokens(official_checkout, config) -> None:
    launch = build_launch(replace(config, augmentation_profile="mild"), visible_devices="0", environment={}, official_checkout=official_checkout)
    index = launch.command.index("--color-jitter-params")
    assert launch.command[index + 1:index + 9] == (
        "brightness", "0.2", "contrast", "0.2", "saturation", "0.2", "hue", "0.05"
    )
```

- [ ] **Step 2: Run augmentation and launch tests to verify failure**

Run: `cd trainer && uv run --offline pytest tests/test_flywheel_augmentation.py tests/test_groot_config.py tests/test_groot_launch.py -v`

Expected: FAIL because augmentation profiles are not yet represented.

- [ ] **Step 3: Implement `none`, `mild`, and gated `nvidia_reference` profiles**

```python
PROFILES = {
    "none": AugmentationProfile("none", {}),
    "mild": AugmentationProfile("mild", {"brightness": 0.20, "contrast": 0.20, "saturation": 0.20, "hue": 0.05}),
    "nvidia_reference": AugmentationProfile("nvidia_reference", {"brightness": 0.30, "contrast": 0.40, "saturation": 0.50, "hue": 0.08}),
}


def color_jitter_cli(profile: AugmentationProfile) -> tuple[str, ...]:
    if not profile.color_jitter:
        return ()
    return (
        "--color-jitter-params",
        "brightness", str(profile.color_jitter["brightness"]),
        "contrast", str(profile.color_jitter["contrast"]),
        "saturation", str(profile.color_jitter["saturation"]),
        "hue", str(profile.color_jitter["hue"]),
    )
```

Add `augmentation_profile` and its canonical hash to `FineTuneLaunchConfig.identity()`. Append the official CLI tokens in `build_launch()`. Refuse `nvidia_reference` unless the request contains a passing canonical-holdout comparison for `mild`. Keep blur, sensor noise, cutout, and camera dropout disabled in this first clean-checkout implementation; adding them requires a separately pinned upstream loader revision and a new reviewed plan rather than runtime monkey-patching.

- [ ] **Step 4: Render the sample sheet and run the complete launcher tests**

Run: `cd trainer && uv run --offline pytest tests/test_flywheel_augmentation.py tests/test_groot_config.py tests/test_groot_launch.py tests/test_production_runtime.py -v`

Expected: PASS.

On the accepted trainer image, run the new sample-sheet command against 32 fixed frames and three cameras. Expected: `augmentation-sample-sheet.png` and `augmentation-report.json` record profile/hash/seed; no sample loses the garment or either gripper.

- [ ] **Step 5: Commit conservative augmentation**

```bash
git add trainer/src/lehome_train/flywheel/augmentation.py trainer/src/lehome_train/groot/config.py trainer/src/lehome_train/groot/launch.py trainer/tests/test_flywheel_augmentation.py trainer/tests/test_groot_config.py trainer/tests/test_groot_launch.py
git commit -m "feat: gate GR00T image augmentation profiles"
```

### Task 4: Immutable Hub publication for datasets and policies

**Files:**
- Modify: `trainer/src/lehome_train/data/publish.py`
- Create: `trainer/src/lehome_train/flywheel/publication.py`
- Test: `trainer/tests/test_flywheel_publication.py`
- Test: `trainer/tests/test_data_publish.py`
- Test: `trainer/tests/test_hub.py`

- [ ] **Step 1: Write fresh-read verification tests**

```python
def test_publish_is_complete_only_after_fresh_tree_and_manifest_read(tmp_path) -> None:
    transport = FakeHubTransport(commit="d" * 40)
    result = publish_flywheel_snapshot(prepared_dataset(tmp_path), transport=transport, token="secret")
    assert result.revision == "d" * 40
    assert transport.calls[-2:] == ["list_tree", "download_manifest"]
    assert result.verified is True


def test_publication_never_persists_token(tmp_path) -> None:
    transport = FakeHubTransport(commit="e" * 40)
    publish_flywheel_snapshot(prepared_dataset(tmp_path), transport=transport, token="hf_sensitive")
    assert "hf_sensitive" not in all_text_files(tmp_path)
```

- [ ] **Step 2: Run publication regression tests and verify failure**

Run: `cd trainer && uv run --offline pytest tests/test_flywheel_publication.py tests/test_data_publish.py tests/test_hub.py -v`

Expected: FAIL only for the missing flywheel publication module.

- [ ] **Step 3: Reuse the approved transport and add terminal revision manifests**

```python
def publish_flywheel_snapshot(
    dataset: Path, *, transport: HubTransport, token: str
) -> PublishedSnapshot:
    validation = validate_prepared_dataset(dataset)
    revision = publish_prepared_dataset(
        dataset,
        repository=DEFAULT_DATA_REPO,
        transport=transport,
        token=token,
        remote_prefix=f"flywheel/{validation.manifest_sha256}",
    )
    if not COMMIT_REVISION.fullmatch(revision):
        raise ValueError("Hub did not return an immutable commit")
    verify_remote_snapshot_fresh(dataset, revision=revision, transport=transport, token=token)
    return PublishedSnapshot(revision, validation.manifest_sha256, True)
```

Publish raw diagnostic shards under content-addressed prefixes and prepared training snapshots under their manifest hash. Publish policy checkpoints through the existing model repository transport. Never consider a symbolic branch/tag a frozen input.

- [ ] **Step 4: Run all Hub and publication tests**

Run: `cd trainer && uv run --offline pytest tests/test_flywheel_publication.py tests/test_data_publish.py tests/test_hub.py tests/test_sync.py -v`

Expected: PASS with fake transports; no network call occurs.

- [ ] **Step 5: Commit immutable publication**

```bash
git add trainer/src/lehome_train/data/publish.py trainer/src/lehome_train/flywheel/publication.py trainer/tests/test_flywheel_publication.py trainer/tests/test_data_publish.py
git commit -m "feat: publish immutable flywheel snapshots"
```

### Task 5: Asynchronous iteration and new-data gate

**Files:**
- Create: `trainer/src/lehome_train/flywheel/iteration.py`
- Create: `trainer/src/lehome_train/commands/flywheel.py`
- Modify: `trainer/src/lehome_train/cli.py`
- Test: `trainer/tests/test_flywheel_iteration.py`
- Test: `trainer/tests/test_cli.py`

- [ ] **Step 1: Write freeze, staleness, and minimum-data tests**

```python
def test_iteration_freezes_exact_inputs_and_defers_late_shards() -> None:
    controller = IterationController(min_new_expert_episodes=40)
    frozen = controller.freeze(policy_revision="a" * 40, dataset_revision="b" * 40, eligible_shards=shards(40))
    controller.observe(shard("late", completed_after=frozen.frozen_at_ns + 1))
    assert frozen.new_expert_episodes == 40
    assert controller.pending_episode_ids == ("late",)


def test_iteration_pauses_when_only_unchanged_data_exists() -> None:
    controller = IterationController(min_new_expert_episodes=40)
    with pytest.raises(InsufficientNewData, match="40"):
        controller.freeze(policy_revision="a" * 40, dataset_revision="b" * 40, eligible_shards=shards(39))
```

- [ ] **Step 2: Run iteration and CLI tests to verify failure**

Run: `cd trainer && uv run --offline pytest tests/test_flywheel_iteration.py tests/test_cli.py -v`

Expected: FAIL for missing iteration controller and command.

- [ ] **Step 3: Implement a finite manifest state machine**

```python
class IterationState(StrEnum):
    COLLECTING = "collecting"
    FROZEN = "frozen"
    TRAINING = "training"
    PUBLISHED = "published"
    EVALUATING = "evaluating"
    PROMOTED = "promoted"
    REJECTED = "rejected"


def freeze_inputs(self, eligible: Sequence[EpisodeShard], *, now_ns: int) -> IterationManifest:
    new = tuple(shard for shard in eligible if shard.episode_id not in self.consumed_episode_ids)
    accepted = tuple(shard for shard in new if shard.quality_grade in {"A", "B"})
    if len(accepted) < self.min_new_expert_episodes:
        raise InsufficientNewData(f"need {self.min_new_expert_episodes} new accepted expert episodes")
    return IterationManifest.freeze(accepted, frozen_at_ns=now_ns)
```

The CLI exposes `flywheel validate-shards`, `flywheel freeze`, `flywheel mix`, `flywheel publish-dataset`, `flywheel record-policy`, and `flywheel promote`. Every command reads a strict request JSON and emits a strict result JSON. Rollout adoption writes a receipt only between episodes; it never swaps an active policy mid-episode.

- [ ] **Step 4: Run iteration, CLI, production-runtime, and redaction tests**

Run: `cd trainer && uv run --offline pytest tests/test_flywheel_iteration.py tests/test_cli.py tests/test_production_runtime.py tests/test_redaction.py -v`

Expected: PASS.

- [ ] **Step 5: Commit asynchronous iteration control**

```bash
git add trainer/src/lehome_train/flywheel/iteration.py trainer/src/lehome_train/commands/flywheel.py trainer/src/lehome_train/cli.py trainer/tests/test_flywheel_iteration.py trainer/tests/test_cli.py
git commit -m "feat: gate asynchronous flywheel iterations"
```

### Task 6: End-to-end dry run and paid fine-tune gate

**Files:**
- Create: `trainer/tests/test_flywheel_end_to_end.py`
- Modify: `docs/groot_n17_training.md`
- Modify: `trainer/README.md`

- [ ] **Step 1: Write a local end-to-end fixture test**

```python
def test_flywheel_dry_run_freezes_mix_and_launch_identity(tmp_path, fake_hub) -> None:
    run = execute_fixture_flywheel(
        organizer=organizer_fixture(tmp_path),
        flywheel=accepted_flywheel_fixture(tmp_path, episodes=40),
        hub=fake_hub,
        augmentation="mild",
    )
    assert run.dataset_revision == "d" * 40
    assert run.mix.source_weights == {"organizer": 0.7, "flywheel": 0.3}
    assert run.launch_identity["augmentation_profile"] == "mild"
    assert run.launch_identity["normalization_sha256"] == run.dataset.normalization_sha256
    assert run.policy_adoption_boundary == "between_episodes"
```

- [ ] **Step 2: Run the end-to-end test and observe failure**

Run: `cd trainer && uv run --offline pytest tests/test_flywheel_end_to_end.py -v`

Expected: FAIL until the fixture orchestration joins Tasks 1-5.

- [ ] **Step 3: Add the strict fixture orchestrator and operator commands**

```python
def execute_fixture_flywheel(...):
    materialized = materialize_verified_shards(flywheel)
    mix = build_mix_plan(organizer, materialized, seed=20260803)
    prepared = materialize_and_validate_mix(mix)
    published = publish_flywheel_snapshot(prepared, transport=hub, token="fixture-token")
    launch = build_flywheel_launch_identity(prepared, published, augmentation="mild")
    return FixtureFlywheelResult.from_parts(mix, prepared, published, launch)
```

Document exact request JSON examples with immutable revisions, local paths, augmentation profile, 4,000-step continuation schedule, checkpoint retention at final and comparison boundaries, and separate rollout/training machines. Commands must stop before any rental or destructive action.

- [ ] **Step 4: Run the full local quality gate and remote one-batch smoke**

Run locally:

```bash
cd trainer
uv run --offline pytest tests -q
uv run --offline python -m lehome_train.cli flywheel validate-shards --request /tmp/lehome-fixture-validate.json
```

Expected: all tests pass and fixture validation exits 0.

On a user-approved trainer only after immutable dataset upload verification, run one forward/backward optimizer step with `augmentation_profile=mild`; verify nonzero loss, finite gradients, the expected normalization hash, and no Hub token in the child environment. Stop the trainer after its evidence and checkpoint are verified durable.

- [ ] **Step 5: Commit the end-to-end gate and documentation**

```bash
git add trainer/tests/test_flywheel_end_to_end.py docs/groot_n17_training.md trainer/README.md
git commit -m "docs: add GR00T flywheel execution gate"
```

## Plan 4 completion gate

Run: `cd trainer && uv run --offline pytest tests -q`

Expected: PASS. A real fine-tune is authorized only after the new prepared dataset commit, manifest hash, normalization hash, augmentation sample sheet, one-step smoke evidence, available budget, and explicit rental decision are all recorded. Dataset and policy instances remain separate; no instance is destroyed by these commands.
