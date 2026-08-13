# GR00T Action-Horizon Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the pinned GR00T model's 40-action maximum while making rollout, RFT materialization, mixture construction, and LeHome training use the evidence-bound 16-action `new_embodiment` target contract before any paid training.

**Architecture:** Name the two distinct contracts: `MODEL_MAX_ACTION_HORIZON=40` from the pinned checkpoint model/processor capacity and `EMBODIMENT_ACTION_HORIZON=16` from its `new_embodiment.action.delta_indices` and live policy wire. Remove the ambiguous RFT 40-step target, preserve complete trajectories, and count/train complete 16-frame LeHome windows without changing the model architecture. A real loader/loss smoke is the paid-training admission gate.

**Tech Stack:** Python 3.11, PyArrow, LeRobot v2.1, GR00T N1.7, pytest.

---

## File structure

- Modify `trainer/src/lehome_train/flywheel/materialize.py`: select/count RFT windows with the canonical mapping horizon.
- Modify `trainer/src/lehome_train/flywheel/rft.py`: publish canonical horizon metadata.
- Create `trainer/src/lehome_train/groot/action_contract.py`: validate the pinned model maximum and `new_embodiment` target horizons.
- Modify `trainer/src/lehome_train/groot/config.py`: record both explicitly named horizon values.
- Modify `trainer/src/lehome_train/groot/modality.py`: generate only the 16-action LeHome modality while retaining model-capacity provenance.
- Modify `trainer/src/lehome_train/flywheel/mix.py`: continue consuming the canonical mapping constant and reject mismatched RFT inputs.
- Create `trainer/src/lehome_train/groot/horizon_gate.py`: inspect the prepared dataset and execute one real GR00T loader/loss batch.
- Modify `trainer/src/lehome_train/groot/production_runtime.py`: expose the horizon gate as a training preflight action.
- Modify `trainer/tests/test_flywheel_materialize.py`, `test_flywheel_rft.py`, `test_flywheel_mix.py`, `test_groot_config.py`, `test_groot_modality.py`, `test_models.py`, `test_smoke.py`, and `test_production_runtime.py`.

### Task 1: Establish the two-value action contract

**Files:**
- Create: `trainer/src/lehome_train/groot/action_contract.py`
- Test: `trainer/tests/test_action_contract.py`

- [ ] **Step 1: Write pinned-checkpoint contract tests**

```python
def test_pinned_checkpoint_has_40_model_capacity_and_16_lehome_targets(tmp_path: Path) -> None:
    checkpoint = pinned_checkpoint_fixture(tmp_path)
    receipt = inspect_action_contract(checkpoint)
    assert receipt.model_max_action_horizon == 40
    assert receipt.embodiment_action_horizon == 16
    assert receipt.embodiment_tag == "new_embodiment"


def test_contract_rejects_40_lehome_delta_indices(tmp_path: Path) -> None:
    checkpoint = pinned_checkpoint_fixture(tmp_path)
    mutate_new_embodiment_delta_indices(checkpoint, list(range(40)))
    with pytest.raises(ValueError, match="new_embodiment action horizon"):
        inspect_action_contract(checkpoint)
```

- [ ] **Step 2: Run and confirm missing module failure**

```bash
PYTHONPATH=trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_action_contract.py
```

- [ ] **Step 3: Implement exact inspection without importing GR00T**

```python
MODEL_MAX_ACTION_HORIZON = 40
EMBODIMENT_ACTION_HORIZON = 16

@dataclass(frozen=True, slots=True)
class ActionContract:
    model_max_action_horizon: int
    embodiment_action_horizon: int
    embodiment_tag: str
    model_config_sha256: str
    processor_config_sha256: str

def inspect_action_contract(checkpoint: str | Path) -> ActionContract:
    root = Path(checkpoint)
    model = strict_json(root / "config.json")
    processor = strict_json(root / "processor_config.json")
    model_horizon = model.get("action_horizon")
    maximum = processor.get("processor_kwargs", {}).get("max_action_horizon")
    indices = processor["processor_kwargs"]["modality_configs"]["new_embodiment"]["action"]["delta_indices"]
    if model_horizon != MODEL_MAX_ACTION_HORIZON or maximum != MODEL_MAX_ACTION_HORIZON:
        raise ValueError("pinned model maximum action horizon is incompatible")
    if indices != list(range(EMBODIMENT_ACTION_HORIZON)):
        raise ValueError("new_embodiment action horizon is incompatible")
    return ActionContract(
        MODEL_MAX_ACTION_HORIZON,
        EMBODIMENT_ACTION_HORIZON,
        "new_embodiment",
        sha256_file(root / "config.json"),
        sha256_file(root / "processor_config.json"),
    )
```

- [ ] **Step 4: Run action-contract tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/src/lehome_train/groot/action_contract.py \
  trainer/tests/test_action_contract.py
git commit -m "Name GR00T model and embodiment horizons"
```

### Task 2: Remove the independent RFT target horizon

**Files:**
- Modify: `trainer/src/lehome_train/flywheel/materialize.py`
- Modify: `trainer/src/lehome_train/flywheel/rft.py`
- Test: `trainer/tests/test_flywheel_materialize.py`
- Test: `trainer/tests/test_flywheel_rft.py`

- [ ] **Step 1: Write failing 16-frame RFT tests**

Add a success trajectory with exactly 16 contiguous policy frames and assert it
has one valid window. Assert 15 frames has none and 17 frames has two overlapping
windows:

```python
@pytest.mark.parametrize((frames, expected), [(15, 0), (16, 1), (17, 2)])
def test_rft_policy_window_count_uses_checked_mapping_horizon(
    tmp_path: Path, frames: int, expected: int
) -> None:
    raw = _raw_rft_episode(tmp_path, frame_count=frames)
    if expected == 0:
        with pytest.raises(ValueError, match="complete 16-frame policy window"):
            materialize_rft_episode(raw, tmp_path / "out")
    else:
        report = materialize_rft_episode(raw, tmp_path / "out")
        assert report.selected_observations == expected
        provenance = json.loads(
            (tmp_path / "out/meta/materialization-provenance.json").read_text()
        )
        assert provenance["selection_horizon"] == EMBODIMENT_ACTION_HORIZON == 16
```

Also assert `rft-selection.json` and `manifest.json/future_actions/horizon` are
both 16.

- [ ] **Step 2: Run the focused tests and verify the old 40-step behavior fails**

Run:

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_flywheel_materialize.py \
  trainer/tests/test_flywheel_rft.py
```

Expected: the new 16-frame RFT assertions fail because
`RFT_ACTION_HORIZON == 40`.

- [ ] **Step 3: Use only the checked mapping constant**

Import `EMBODIMENT_ACTION_HORIZON` from
`lehome_train.groot.action_contract`, delete the local `ACTION_HORIZON` and
`RFT_ACTION_HORIZON` definitions, and use the imported
constant in policy-window selection, rejection accounting, trajectory metadata,
and RFT snapshot metadata:

```python
from lehome_train.data.mapping import FIXED_INSTRUCTION, JOINT_NAMES
from lehome_train.groot.action_contract import EMBODIMENT_ACTION_HORIZON

rejected = {
    "incomplete_tail": min(len(parsed), EMBODIMENT_ACTION_HORIZON - 1),
    "discontinuity": 0,
}
for start in range(max(0, len(parsed) - EMBODIMENT_ACTION_HORIZON + 1)):
    window = parsed[start : start + EMBODIMENT_ACTION_HORIZON]
```

Keep the full successful trajectory once; do not duplicate each overlapping
window into separate stored episodes.

- [ ] **Step 4: Run the focused tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/src/lehome_train/flywheel/materialize.py \
  trainer/src/lehome_train/flywheel/rft.py \
  trainer/tests/test_flywheel_materialize.py \
  trainer/tests/test_flywheel_rft.py
git commit -m "Unify corrective RFT action horizon"
```

### Task 3: Bind mixtures and launches to both named horizons

**Files:**
- Modify: `trainer/src/lehome_train/groot/config.py`
- Modify: `trainer/src/lehome_train/groot/modality.py`
- Modify: `trainer/src/lehome_train/flywheel/mix.py`
- Test: `trainer/tests/test_groot_config.py`
- Test: `trainer/tests/test_groot_modality.py`
- Test: `trainer/tests/test_models.py`
- Test: `trainer/tests/test_smoke.py`
- Test: `trainer/tests/test_flywheel_mix.py`

- [ ] **Step 1: Write mismatch-rejection tests**

```python
def test_parent_checkpoint_records_model_40_and_embodiment_16() -> None:
    resolved = FineTuneLaunchConfig(**config_values(
        model_max_action_horizon=40,
        embodiment_action_horizon=16,
    ))
    assert resolved.model_max_action_horizon == 40
    assert resolved.embodiment_action_horizon == 16


def test_launch_rejects_ambiguous_40_step_embodiment_targets() -> None:
    with pytest.raises(ValueError, match="embodiment action horizon"):
        FineTuneLaunchConfig(**config_values(
            model_max_action_horizon=40,
            embodiment_action_horizon=40,
        ))


def test_mix_rejects_legacy_40_step_rft_manifest(tmp_path: Path) -> None:
    organizer, rft = prepared_sources(tmp_path)
    mutate_json(rft / "manifest.json", ("future_actions", "horizon"), 40)
    with pytest.raises(ValueError, match="incompatible action horizon"):
        build_mix_plan(organizer, rft, seed=7)
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_groot_config.py trainer/tests/test_groot_modality.py \
  trainer/tests/test_models.py trainer/tests/test_smoke.py \
  trainer/tests/test_flywheel_mix.py
```

Expected: the new named fields are absent and the old ambiguous
`action_horizon=40` special case remains.

- [ ] **Step 3: Replace the ambiguous field with two exact fields**

Use exact validation after parent identity checks:

```python
if self.model_max_action_horizon != MODEL_MAX_ACTION_HORIZON:
    raise ValueError("model maximum action horizon must remain 40")
if self.embodiment_action_horizon != EMBODIMENT_ACTION_HORIZON:
    raise ValueError("embodiment action horizon must be exactly 16")
```

The launcher identity records both fields. The generated LeHome modality config
uses `embodiment_action_horizon`; it never changes the pinned model config's
40-action capacity. Update existing configuration/model/smoke tests that refer
to the old `config.action_horizon` field so they assert both named values. Keep
`mix.py` bound to `EMBODIMENT_ACTION_HORIZON`; improve its error to report the
observed and required target horizons. Do not pass 40 as an action-target
horizon to `runtime_modality_config_source()`.

- [ ] **Step 4: Run focused configuration and mixture tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/src/lehome_train/groot/config.py \
  trainer/src/lehome_train/groot/modality.py \
  trainer/src/lehome_train/flywheel/mix.py \
  trainer/tests/test_groot_config.py trainer/tests/test_groot_modality.py \
  trainer/tests/test_models.py trainer/tests/test_smoke.py \
  trainer/tests/test_flywheel_mix.py
git commit -m "Bind GR00T training to canonical horizon"
```

### Task 4: Add the real loader/loss horizon gate

**Files:**
- Create: `trainer/src/lehome_train/groot/horizon_gate.py`
- Modify: `trainer/src/lehome_train/groot/production_runtime.py`
- Test: `trainer/tests/test_horizon_gate.py`
- Test: `trainer/tests/test_production_runtime.py`

- [ ] **Step 1: Write fail-closed gate tests with an injected adapter**

```python
def test_horizon_gate_requires_exact_loader_and_target_shapes(tmp_path: Path) -> None:
    receipt = run_horizon_gate(
        prepared_root=prepared_dataset(tmp_path, horizon=16),
        model_path=pinned_checkpoint_fixture(tmp_path),
        expected_model_max_horizon=40,
        expected_embodiment_horizon=16,
        batch_probe=lambda _root: {
            "observation_state_shape": [1, 12],
            "action_shape": [1, 16, 12],
            "loss": 0.25,
        },
    )
    assert receipt["passed"] is True
    assert receipt["action_shape"] == [1, 16, 12]


@pytest.mark.parametrize("shape", ([1, 40, 12], [1, 16, 11], [16, 12]))
def test_horizon_gate_rejects_any_other_target_shape(
    tmp_path: Path, shape: list[int]
) -> None:
    with pytest.raises(ValueError, match="action target shape"):
        run_horizon_gate(
            prepared_root=prepared_dataset(tmp_path, horizon=16),
            model_path=pinned_checkpoint_fixture(tmp_path),
            expected_model_max_horizon=40,
            expected_embodiment_horizon=16,
            batch_probe=lambda _root: {
                "observation_state_shape": [1, 12],
                "action_shape": shape,
                "loss": 0.25,
            },
        )
```

- [ ] **Step 2: Run tests and see the missing module/action failure**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_horizon_gate.py trainer/tests/test_production_runtime.py
```

Expected: FAIL because `horizon_gate` and its runtime action do not exist.

- [ ] **Step 3: Implement the pure validator and production adapter**

```python
def run_horizon_gate(
    *,
    prepared_root: str | Path,
    model_path: str | Path,
    expected_model_max_horizon: int,
    expected_embodiment_horizon: int,
    batch_probe: Callable[[Path], Mapping[str, object]],
) -> dict[str, object]:
    root = Path(prepared_root)
    manifest = load_prepared_manifest(root)
    contract = inspect_action_contract(model_path)
    observed = manifest["future_actions"]["horizon"]
    if contract.model_max_action_horizon != expected_model_max_horizon:
        raise ValueError("model maximum action horizon is incompatible")
    if observed != expected_embodiment_horizon or observed != contract.embodiment_action_horizon:
        raise ValueError("prepared embodiment action horizon is incompatible")
    probe = dict(batch_probe(root))
    if probe.get("action_shape") != [1, EMBODIMENT_ACTION_HORIZON, 12]:
        raise ValueError("GR00T action target shape is incompatible")
    loss = probe.get("loss")
    if type(loss) not in (int, float) or not math.isfinite(float(loss)):
        raise ValueError("GR00T horizon smoke loss is not finite")
    return {
        "schema_version": 1,
        "kind": "groot_action_horizon_gate",
        "model_max_action_horizon": MODEL_MAX_ACTION_HORIZON,
        "embodiment_action_horizon": EMBODIMENT_ACTION_HORIZON,
        "action_shape": probe["action_shape"],
        "finite_loss": True,
        "passed": True,
    }
```

The production probe must import the pinned official GR00T checkout, construct
its real dataset/transform loader, read one batch, execute one forward/loss call
on CUDA, and return only shapes plus finite scalar loss. It receives no Hub
token. Add a `horizon-gate` runtime action that writes the canonical receipt.

- [ ] **Step 4: Run pure tests, then the existing trainer suite**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_horizon_gate.py \
  trainer/tests/test_flywheel_materialize.py \
  trainer/tests/test_flywheel_rft.py \
  trainer/tests/test_flywheel_mix.py \
  trainer/tests/test_groot_config.py \
  trainer/tests/test_production_runtime.py
```

Expected: PASS. The real CUDA branch remains skipped on macOS and must run on
the paid trainer before training begins.

- [ ] **Step 5: Commit**

```bash
git add trainer/src/lehome_train/groot/horizon_gate.py \
  trainer/src/lehome_train/groot/production_runtime.py \
  trainer/tests/test_horizon_gate.py trainer/tests/test_production_runtime.py
git commit -m "Gate GR00T training on real horizon smoke"
```

### Task 5: Final horizon verification

**Files:**
- Verify only.

- [ ] **Step 1: Search for remaining independent 40-step training assumptions**

```bash
rg -n "RFT_ACTION_HORIZON|action_horizon.*40|horizon.*40" \
  trainer/src trainer/tests scripts
```

Expected: no live corrective materialization or LeHome modality target uses 40.
The pinned model maximum, processor maximum, exact contract assertions, and
explicit rejection fixtures must still contain 40.

- [ ] **Step 2: Run the focused regression suite**

```bash
PYTHONPATH=source/lehome:trainer/src uv run --project trainer pytest -q \
  trainer/tests/test_flywheel_materialize.py \
  trainer/tests/test_flywheel_rft.py \
  trainer/tests/test_flywheel_mix.py \
  trainer/tests/test_action_contract.py \
  trainer/tests/test_groot_config.py \
  trainer/tests/test_groot_modality.py \
  trainer/tests/test_models.py \
  trainer/tests/test_smoke.py \
  trainer/tests/test_horizon_gate.py \
  trainer/tests/test_production_runtime.py
python3 -m py_compile \
  trainer/src/lehome_train/flywheel/materialize.py \
  trainer/src/lehome_train/flywheel/rft.py \
  trainer/src/lehome_train/flywheel/mix.py \
  trainer/src/lehome_train/groot/action_contract.py \
  trainer/src/lehome_train/groot/config.py \
  trainer/src/lehome_train/groot/modality.py \
  trainer/src/lehome_train/groot/horizon_gate.py
git diff --check
```

Expected: all pass.

- [ ] **Step 3: Commit any test-only final adjustments**

```bash
git add trainer/tests trainer/src/lehome_train
git commit -m "Verify unified GR00T action horizon"
```

Skip this commit if the worktree is clean.
