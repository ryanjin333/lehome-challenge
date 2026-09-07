"""Offline boundary tests for the bounded public N1.5 remote pipeline."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import stat
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/run_public_n15_reproduction.py"
WRAPPER = ROOT / "rollout_appliance/run_public_n15_pipeline_remote.sh"
_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_public_n15_pipeline_fixtures", ROOT / "tests/test_n15_reproduction.py"
)
assert _FIXTURE_SPEC is not None and _FIXTURE_SPEC.loader is not None
_FIXTURES = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(_FIXTURES)


def _load_cli():
    spec = importlib.util.spec_from_file_location("public_n15_reproduction", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_host_stage_seal(
    pipeline: Path, *, run_id: str, stage: str, remote_paths: list[str],
) -> Path:
    path = pipeline / f"host-stage-{stage}-complete.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "lehome_public_n15_host_stage_completion_v1",
                "run_id": run_id,
                "stage": stage,
                "remote_receipts": [
                    {"path": remote_path, "sha256": hashlib.sha256(remote_path.encode()).hexdigest()}
                    for remote_path in remote_paths
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n",
        encoding="ascii",
    )
    path.chmod(0o444)
    return path


def _wrapper_env(tmp_path: Path, fake_bin: Path, run_id: str) -> dict[str, str]:
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir(exist_ok=True)
    controller_runtime = tmp_path / "controller-runtime"
    controller_runtime.mkdir(mode=0o700, exist_ok=True)
    controller_lock_test_root = tmp_path / "controller-lock-test-root"
    controller_lock_test_root.mkdir(mode=0o700, exist_ok=True)
    remote_pipeline = f"/mnt/lehome/runs/{run_id}"
    return {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TMPDIR": str(controller_runtime),
        "LEHOME_N15_CONTROLLER_LOCK_TEST_ROOT": str(controller_lock_test_root),
        "LEHOME_N15_RUN_ID": run_id,
        "LEHOME_N15_PIPELINE_ROOT": str(pipeline),
        "LEHOME_N15_SSH_TARGET": "operator@example",
        "LEHOME_N15_REMOTE_ROOT": "/mnt/lehome/runtime",
        "LEHOME_N15_REMOTE_RUNS_BASE": "/mnt/lehome/runs",
        "LEHOME_N15_REMOTE_PIPELINE_ROOT": remote_pipeline,
        "LEHOME_N15_PUBLIC_HF_REPOSITORY": "ryanjin333/public-n15",
        "LEHOME_OFFICIAL_ASSETS_ROOT": "/mnt/assets",
        "LEHOME_OFFICIAL_METADATA_ROOT": "/mnt/source",
        "LEHOME_N15_REFERENCE_CHECKPOINT": "/mnt/reference",
        "LEHOME_N15_REFERENCE_SANITIZED_CONFIG_ROOT": "/mnt/reference-config",
        "LEHOME_N15_REFERENCE_COMPATIBILITY_RECEIPT": "/mnt/reference-receipt",
        "LEHOME_N15_NATIVE_RUNTIME_EVIDENCE_ROOT": "/mnt/evidence",
        "LEHOME_N15_NATIVE_DEPENDENCIES_ROOT": "/mnt/deps",
        "LEHOME_N15_FOCUSED_HF_CACHE_ROOT": "/mnt/cache",
        "LEHOME_N15_ROLLOUT_IMAGE_RECEIPT": "/mnt/image.json",
        "LEHOME_N15_TRAINING_HF_CACHE_ROOT": "/mnt/train-cache",
        "LEHOME_N15_TRAINING_UV": "/mnt/uv",
        "LEHOME_N15_LEROBOT_WHEEL": "/mnt/lerobot.whl",
        "LEHOME_N15_TRAINING_ROOT": f"{remote_pipeline}/training",
    }


def _controller_lock_path(env: dict[str, str]) -> Path:
    return (
        Path(env["LEHOME_N15_CONTROLLER_LOCK_TEST_ROOT"])
        / f"lehome-public-n15-controller-{os.getuid()}"
        / "computeinstance-u00t6xfqhadrcmssa2.lock"
    )


def _write_executable(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o755)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def _remote_train_fixture(tmp_path: Path) -> tuple[dict[str, str], object, Path, Path, Path]:
    """Materialize a real resume tree with only OS/container transports doubled."""
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _FIXTURES._materialize_source(tmp_path)
    _, _, snapshots_receipt = _FIXTURES._materialize_snapshots(tmp_path, checkout)
    contract = _FIXTURES._fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout, source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id, disk_id=contract.disk_id, contract=contract,
    )
    training, staging, upstream = _FIXTURES._materialize_partial_training(
        tmp_path, verified=verified, contract=contract
    )
    installed_site = staging / "runtime/site-packages"
    fake_bin = tmp_path / "remote-bin"; fake_bin.mkdir()
    tool_bin = tmp_path / "training-tools/bin"; tool_bin.mkdir(parents=True)
    trace = tmp_path / "remote-train-trace.log"
    contract_path = tmp_path / "fixture-contract.json"
    contract_path.write_text(
        json.dumps(reproduction._contract_json(contract)), encoding="ascii"
    )
    driver = tmp_path / "contract-cli-driver.py"
    driver.write_text(
        """import importlib.util, json, os, signal, sys
from lehome.n15_reproduction import ReproductionContract
spec = importlib.util.spec_from_file_location('actual_n15_cli', os.environ['REAL_REPRO_CLI'])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
value = json.load(open(os.environ['TEST_CONTRACT_JSON'], encoding='ascii'))
value['training_command'] = tuple(value['training_command'])
contract = ReproductionContract(**value)
status = module.main(sys.argv[1:], contract=contract)
command = sys.argv[1] if len(sys.argv) > 1 else ''
point = os.environ.get('FAKE_INTERRUPT_POINT', '')
if status == 0 and ((point == 'compatibility' and command == 'build-compatible-wheel') or (point == 'execution-manifest' and command == 'render-training')):
    open(os.environ['FAKE_TRACE'], 'a').write(f'interrupt:{point}\\n')
    mode = os.environ['FAKE_INTERRUPT_SIGNAL']
    if mode == 'EXIT':
        sys.exit(23)
    os.kill(os.getppid(), signal.SIGINT if mode == 'INT' else signal.SIGTERM)
sys.exit(status)
""",
        encoding="utf-8",
    )
    _write_executable(
        fake_bin / "python3",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "$REAL_REPRO_CLI" ]]; then
  shift
  exec "$REAL_PYTHON" "$CONTRACT_CLI_DRIVER" "$@"
fi
exec "$REAL_PYTHON" "$@"
        """,
    )
    real_training_python = str(Path(shutil.which("python3.11") or sys.executable))
    _write_executable(
        tool_bin / "python",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == -I && "${3:-}" == *'import lerobot'* ]]; then exit 0; fi
if [[ "${1:-}" == - && $# -eq 4 && "$2" == */source/lehome ]]; then
  printf '%s\n' "$FAKE_DATASET_BLOBS"
  exit 0
fi
if [[ "${1:-}" == - && $# -eq 13 ]]; then
  cp "$FAKE_STAGING/evidence/runtime-receipt.json" "$4"
  chmod 0444 "$4"
  if [[ "${FAKE_INTERRUPT_POINT:-}" == runtime-receipt ]]; then
    printf 'interrupt:runtime-receipt\n' >> "$FAKE_TRACE"
    [[ "$FAKE_INTERRUPT_SIGNAL" == EXIT ]] && exit 23
    kill -"$FAKE_INTERRUPT_SIGNAL" "$PPID"
  fi
  exit 0
fi
"@REAL_TRAINING_PYTHON@" "$@"
status=$?
if [[ "$status" == 0 && "${FAKE_INTERRUPT_POINT:-}" == runtime-image && "${3:-}" == *runtime-image-receipt.json ]]; then
  printf 'interrupt:runtime-image\n' >> "$FAKE_TRACE"
  [[ "$FAKE_INTERRUPT_SIGNAL" == EXIT ]] && exit 23
  kill -"$FAKE_INTERRUPT_SIGNAL" "$PPID"
fi
exit "$status"
""".replace("@REAL_TRAINING_PYTHON@", real_training_python),
    )
    _write_executable(tool_bin / "lerobot-train", "#!/usr/bin/env bash\nexit 99\n")
    _write_executable(tool_bin / "uv", "#!/usr/bin/env bash\nexit 0\n")
    runtime_receipt = staging / "evidence/runtime-receipt.json"
    runtime_value = json.loads(runtime_receipt.read_text(encoding="ascii"))
    runtime_value["python_executable"] = str(tool_bin / "python")
    runtime_receipt.write_bytes(_FIXTURES._canonical(runtime_value))
    _write_executable(fake_bin / "findmnt", "#!/usr/bin/env bash\nprintf '1:1\\n'\n")
    _write_executable(fake_bin / "pgrep", "#!/usr/bin/env bash\nexit 1\n")
    _write_executable(
        fake_bin / "readlink",
        """#!/usr/bin/env bash
if [[ "${1:-}" == -f ]]; then
  exec "$REAL_PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "$2"
fi
exec /usr/bin/readlink "$@"
""",
    )
    _write_executable(
        fake_bin / "sudo",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${FAKE_SUDO_TRACE:-}" ]]; then
  python3 - "$FAKE_SUDO_TRACE" "$@" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]), encoding="utf-8")
PY
fi
[[ "${1:-}" == -n ]] && shift
if [[ "${1:-}" == chown ]]; then
  # Preserve the separately reviewed native-container handoff while checking
  # every argument. The privileged recovery itself must invoke Python, never
  # this path-recursive command.
  if [[ "$#" == 5 ]]; then
    [[ "${2:-}" == -R && "${3:-}" == --no-dereference && "${4:-}" == "$(id -u):$(id -g)" && "${5:-}" == "$FAKE_STAGING/eagle-home" ]] || { printf 'unexpected chown argv: %q\n' "$@" >&2; exit 97; }
    exit 0
  fi
  [[ "$#" == 7 && "${2:-}" == -R && "${3:-}" == --no-dereference ]] || { printf 'unexpected chown argv: %q\n' "$@" >&2; exit 97; }
  [[ "${4:-}" == "$(id -u):$(id -g)" && "${5:-}" == "$FAKE_UPSTREAM" ]] || { printf 'unexpected chown argv: %q\n' "$@" >&2; exit 97; }
  [[ "${6:-}" == "$FAKE_STAGING/eagle-home" && "${7:-}" == "$FAKE_STAGING" ]] || { printf 'unexpected chown argv: %q\n' "$@" >&2; exit 97; }
  exit 0
fi
exec "$@"
        """,
    )
    _write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == image && "${2:-}" == inspect ]]; then
  printf '[{"Id":"%s"}]\n' "${!#}"
  exit 0
fi
payload="$(mktemp)"
trap 'rm -f -- "$payload"' EXIT
cat > "$payload"
while (( $# )); do
  if [[ "$1" == -- ]]; then shift; break; fi
  shift
done
args=("$@")
if grep -q 'prepare-peft-overlay' "$payload"; then
  for index in 1 2 3 4; do
    target="${args[$index]}"
    cp "$FAKE_STAGING/evidence/$(basename "$target")" "$target"
    chmod 0444 "$target"
  done
  if [[ "${FAKE_INTERRUPT_POINT:-}" == overlays ]]; then
    printf 'interrupt:overlays\n' >> "$FAKE_TRACE"
    [[ "$FAKE_INTERRUPT_SIGNAL" == EXIT ]] && exit 23
    kill -"$FAKE_INTERRUPT_SIGNAL" "$PPID"
  fi
  exit 0
fi
grep -F 'PYTHONPATH="/flash/site-packages:/deps/peft-0.18.1-py3-none-any.whl" /opt/lehome-challenge/.venv/bin/lerobot-train --config_path="$resume_checkpoint/pretrained_model/train_config.json" --resume=true --wandb.mode=offline' "$payload" >/dev/null
checkpoint="${args[5]}"
printf 'native:/opt/lehome-challenge/.venv/bin/lerobot-train --config_path=%s/pretrained_model/train_config.json --resume=true --wandb.mode=offline\n' "$checkpoint" >> "$FAKE_TRACE"
[[ "${FAKE_TRAIN_COMPLETE:-0}" == 1 ]] || exit 17
"$FAKE_NATIVE_TRAINER" "--config_path=$checkpoint/pretrained_model/train_config.json" --resume=true --wandb.mode=offline
printf 'Checkpoint policy after step 12000\nEnd of training\n'
""",
    )
    trainer = tmp_path / "native-lerobot-train"
    _write_executable(
        trainer,
        """#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == "--config_path=$RESUME_CHECKPOINT/pretrained_model/train_config.json" ]]
[[ "$2" == --resume=true && "$3" == --wandb.mode=offline ]]
output="$(dirname "$(dirname "$RESUME_CHECKPOINT")")"
cp -R "$RESUME_CHECKPOINT" "$output/checkpoints/012000"
printf '{"step":12000}\n' > "$output/checkpoints/012000/training_state/training_step.json"
printf '{"last_epoch":12000}\n' > "$output/checkpoints/012000/training_state/scheduler_state.json"
for relative in pretrained_model/model.safetensors training_state/optimizer_state.safetensors training_state/rng_state.safetensors; do
  printf ' resumed-through-step-12000' >> "$output/checkpoints/012000/$relative"
done
rm "$output/checkpoints/last"
ln -s 012000 "$output/checkpoints/last"
        """,
    )
    snapshots_value = json.loads(snapshots_receipt.read_text(encoding="ascii"))
    hub_root = Path(snapshots_value["base_model"]["root"]).parents[3]
    eagle = hub_root / "models--lerobot--eagle2hg-processor-groot-n1p5"
    eagle_blobs = eagle / "blobs"
    eagle_snapshot = eagle / "snapshots/baf604d8a5caf26fda5cc545f141bc1814156237"
    eagle_blobs.mkdir(parents=True)
    eagle_snapshot.mkdir(parents=True)
    for name in (
        "vocab.json", "merges.txt", "added_tokens.json", "chat_template.json",
        "special_tokens_map.json", "config.json", "generation_config.json",
        "preprocessor_config.json", "processor_config.json", "tokenizer_config.json",
    ):
        blob = eagle_blobs / f"blob-{name}"
        blob.write_text(name, encoding="ascii")
        (eagle_snapshot / name).symlink_to(Path("../../blobs") / blob.name)
    env = _wrapper_env(tmp_path, fake_bin, "n15-actual-remote-train")
    env.update({
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PYTHONPATH": f"{ROOT / 'source/lehome'}:{installed_site}",
        "REAL_PYTHON": sys.executable,
        "REAL_REPRO_CLI": str(CLI),
        "CONTRACT_CLI_DRIVER": str(driver),
        "TEST_CONTRACT_JSON": str(contract_path),
        "FAKE_TRACE": str(trace),
        "FAKE_STAGING": str(staging),
        "FAKE_UPSTREAM": str(upstream),
        "FAKE_NATIVE_TRAINER": str(trainer),
        "FAKE_DATASET_BLOBS": str(
            Path(snapshots_value["dataset"]["snapshot_root"]).parents[1] / "blobs"
        ),
        # Production recovery requires Linux openat2.  The real remote-train
        # fixture is also exercised on macOS, where this narrowly scoped test
        # switch permits the portable descriptor/scope assertions below.
        "LEHOME_N15_TEST_ALLOW_UNSAFE_OPENAT2": "1",
        "LEHOME_N15_REMOTE_ROOT": str(ROOT),
        "LEHOME_N15_PUBLIC_SOURCE_ROOT": str(checkout),
        "LEHOME_N15_SOURCE_RECEIPT": str(source_receipt),
        "LEHOME_N15_RESOLVED_SNAPSHOTS_RECEIPT": str(snapshots_receipt),
        "LEHOME_N15_TRAINING_ROOT": str(training),
        "LEHOME_N15_TRAINING_HF_CACHE_ROOT": str(hub_root),
        "LEHOME_N15_TRAINING_PYTHON": str(tool_bin / "python"),
        "LEHOME_N15_TRAINING_UV": str(tool_bin / "uv"),
        "LEHOME_N15_LEROBOT_WHEEL": str(
            staging / "evidence/upstream/lerobot-0.4.3-py3-none-any.whl"
        ),
        "LEHOME_N15_RESUME_PARTIAL": "1",
        "LEHOME_N15_RESUME_STEP": "1500",
        "LEHOME_N15_RESUME_CHECKPOINT": str(upstream / "checkpoints/001500"),
        "RESUME_CHECKPOINT": str(upstream / "checkpoints/001500"),
    })
    return env, contract, training, staging, upstream


def _run_actual_remote_train_stage(
    env: dict[str, str], *, attempt_id: str, interrupt_point: str = "",
    interrupt_signal: str = "TERM", complete: bool = False,
) -> subprocess.CompletedProcess[str]:
    harness = r'''
source "$WRAPPER_PATH"
remote() { command "$@"; }
train_stage
'''
    return subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={
            **env,
            "WRAPPER_PATH": str(WRAPPER),
            "LEHOME_N15_RESUME_ATTEMPT_ID": attempt_id,
            "FAKE_INTERRUPT_POINT": interrupt_point,
            "FAKE_INTERRUPT_SIGNAL": interrupt_signal,
            "FAKE_TRAIN_COMPLETE": "1" if complete else "0",
        },
        text=True, capture_output=True, timeout=30,
    )


@pytest.mark.parametrize(
    ("setup_point", "interrupt_signal", "expected_status"),
    [
        ("compatibility", "EXIT", 23),
        ("runtime-image", "INT", 130),
        ("overlays", "TERM", 130),
        ("runtime-receipt", "INT", 130),
        ("execution-manifest", "TERM", 130),
    ],
)
def test_actual_remote_train_stage_cleans_scratch_on_setup_interruption_and_retries(
    tmp_path: Path, setup_point: str, interrupt_signal: str, expected_status: int,
) -> None:
    env, _contract, training, staging, upstream = _remote_train_fixture(tmp_path)

    interrupted = _run_actual_remote_train_stage(
        env, attempt_id=f"attempt-{setup_point}-first", interrupt_point=setup_point,
        interrupt_signal=interrupt_signal,
    )

    assert interrupted.returncode == expected_status, interrupted.stderr
    assert f"interrupt:{setup_point}" in Path(env["FAKE_TRACE"]).read_text()
    assert not list(staging.glob(".resume-scratch-*"))
    assert staging.is_dir() and upstream.is_dir() and not training.exists()

    retry = _run_actual_remote_train_stage(
        env, attempt_id=f"attempt-{setup_point}-retry", complete=False
    )
    assert retry.returncode == 17, retry.stderr
    assert not list(staging.glob(".resume-scratch-*"))
    assert staging.is_dir() and upstream.is_dir() and not training.exists()
    attempts = staging / "evidence/resume-attempts"
    assert (attempts / f"attempt-{setup_point}-first.json").is_file()
    assert (attempts / f"attempt-{setup_point}-retry.json").is_file()


def test_actual_remote_train_stage_runs_native_resume_and_production_finalization(
    tmp_path: Path,
) -> None:
    from rollout_appliance.native_reference_site.training_identity import (
        validate_training_identity_receipt,
    )

    env, contract, training, staging, upstream = _remote_train_fixture(tmp_path)

    preempted = _run_actual_remote_train_stage(
        env, attempt_id="attempt-real-01-preempted", complete=False
    )
    assert preempted.returncode == 17, preempted.stderr
    assert upstream.is_dir() and staging.is_dir() and not training.exists()

    result = _run_actual_remote_train_stage(
        env, attempt_id="attempt-real-02-completion", complete=True
    )

    assert result.returncode == 0, result.stderr
    assert training.is_dir() and not staging.exists() and not upstream.exists()
    identity_path = training / "training-identity.json"
    identity = json.loads(identity_path.read_text(encoding="ascii"))
    assert identity["kind"] == "lehome_public_n15_verified_training_output_v1"
    assert identity["step"] == 12000
    assert [item["attempt_id"] for item in identity["resume_lineage"]] == [
        "attempt-real-01-preempted", "attempt-real-02-completion",
    ]
    for attempt_id in ("attempt-real-01-preempted", "attempt-real-02-completion"):
        lineage_path = training / f"evidence/resume-attempts/{attempt_id}.json"
        lineage = json.loads(lineage_path.read_text(encoding="ascii"))
        assert lineage["kind"] == "lehome_public_n15_resume_lineage_v1"
        assert lineage["requested_step"] == 1500
        assert lineage["config_path"] == (
            f"{env['RESUME_CHECKPOINT']}/pretrained_model/train_config.json"
        )
        assert lineage["checkpoint_files"]
    admitted = validate_training_identity_receipt(
        identity_path, expected_contract=contract,
        expected_pretrained_root=training / "checkpoints/012000/pretrained_model",
    )
    assert admitted["resume_lineage"] == identity["resume_lineage"]
    assert admitted["identity_receipt_sha256"] == hashlib.sha256(
        identity_path.read_bytes()
    ).hexdigest()
    trace = Path(env["FAKE_TRACE"]).read_text(encoding="utf-8")
    assert trace.count(
        "native:/opt/lehome-challenge/.venv/bin/lerobot-train "
        f"--config_path={env['RESUME_CHECKPOINT']}/pretrained_model/train_config.json "
        "--resume=true --wandb.mode=offline"
    ) == 2


def test_completed_12k_upstream_is_authenticated_and_finalized_without_retraining(
    tmp_path: Path,
) -> None:
    env, _contract, training, staging, upstream = _remote_train_fixture(tmp_path)

    interrupted = _run_actual_remote_train_stage(
        env, attempt_id="attempt-after-trainer",
        interrupt_point="after-trainer", complete=True,
    )
    assert interrupted.returncode == 130, interrupted.stderr
    assert upstream.is_dir() and staging.is_dir() and not training.exists()
    assert (upstream / "checkpoints/last").readlink() == Path("012000")

    recovered = _run_actual_remote_train_stage(
        env, attempt_id="attempt-after-trainer-recovery", complete=False,
    )
    assert recovered.returncode == 0, recovered.stderr
    assert training.is_dir() and not upstream.exists() and not staging.exists()
    trace = Path(env["FAKE_TRACE"]).read_text(encoding="utf-8")
    assert trace.count("native:/opt/lehome-challenge/.venv/bin/lerobot-train") == 1


def test_completed_12k_recovery_repairs_unreadable_completed_tree_before_verification(
    tmp_path: Path,
) -> None:
    """The split recovery walks only authenticated 0700 roots through sudo."""
    env, _contract, training, staging, upstream = _remote_train_fixture(tmp_path)

    interrupted = _run_actual_remote_train_stage(
        env, attempt_id="attempt-root-owned-completion", interrupt_point="after-trainer",
        complete=True,
    )
    assert interrupted.returncode == 130, interrupted.stderr
    protected_log = staging / "logs/train-resume-attempt-root-owned-completion.log"
    assert protected_log.is_file()
    # 0700 roots match the failure topology. The shim executes the requested
    # Python walker unchanged and records its exact argv; it does not chmod or
    # recurse over the test directory.
    upstream.chmod(0o700)
    (upstream / "checkpoints/012000").chmod(0o700)
    staging.chmod(0o700)
    (staging / "logs").chmod(0o700)
    sudo_trace = tmp_path / "sudo-argv.json"
    recovered = _run_actual_remote_train_stage(
        {**env, "FAKE_SUDO_TRACE": str(sudo_trace)},
        attempt_id="attempt-root-owned-recovery", complete=False,
    )

    assert recovered.returncode == 0, recovered.stderr
    assert training.is_dir() and not upstream.exists() and not staging.exists()
    assert json.loads(sudo_trace.read_text(encoding="utf-8")) == [
        "-n", "python3", "-", str(Path(env["LEHOME_N15_PUBLIC_SOURCE_ROOT"])),
        str(training), str(staging), str(upstream), "split", str(os.getuid()), str(os.getgid()),
    ]
    trace = Path(env["FAKE_TRACE"]).read_text(encoding="utf-8")
    assert trace.count("native:/opt/lehome-challenge/.venv/bin/lerobot-train") == 1


def test_completed_12k_recovery_rejects_a_symlinked_mount_escape_before_mutating(
    tmp_path: Path,
) -> None:
    env, _contract, training, staging, upstream = _remote_train_fixture(tmp_path)
    interrupted = _run_actual_remote_train_stage(
        env, attempt_id="attempt-unsafe-symlink", interrupt_point="after-trainer", complete=True,
    )
    assert interrupted.returncode == 130, interrupted.stderr
    # A bind-mounted escape can only be reached through a directory entry; the
    # descriptor walker treats this symlinked mount root as unsafe before any
    # finalization or trainer invocation.
    (staging / "mount-escape").symlink_to("/dev")
    recovered = _run_actual_remote_train_stage(
        env, attempt_id="attempt-unsafe-symlink-recovery", complete=False,
    )
    assert recovered.returncode != 0
    assert training.exists() is False and upstream.is_dir() and staging.is_dir()
    trace = Path(env["FAKE_TRACE"]).read_text(encoding="utf-8")
    assert trace.count("native:/opt/lehome-challenge/.venv/bin/lerobot-train") == 1


def test_completed_12k_recovery_rejects_hardlinked_artifact_before_mutating(
    tmp_path: Path,
) -> None:
    env, _contract, training, staging, upstream = _remote_train_fixture(tmp_path)
    interrupted = _run_actual_remote_train_stage(
        env, attempt_id="attempt-hardlink", interrupt_point="after-trainer", complete=True,
    )
    assert interrupted.returncode == 130, interrupted.stderr
    source = upstream / "checkpoints/012000/pretrained_model/model.safetensors"
    external = tmp_path / "outside-hardlink"
    os.link(source, external)
    before = source.stat()
    recovered = _run_actual_remote_train_stage(
        env, attempt_id="attempt-hardlink-recovery", complete=False,
    )
    assert recovered.returncode != 0
    after = source.stat()
    assert (before.st_uid, before.st_gid, before.st_ino, before.st_nlink) == (
        after.st_uid, after.st_gid, after.st_ino, after.st_nlink
    )
    assert not training.exists() and upstream.is_dir() and staging.is_dir()
    trace = Path(env["FAKE_TRACE"]).read_text(encoding="utf-8")
    assert trace.count("native:/opt/lehome-challenge/.venv/bin/lerobot-train") == 1


def test_completed_12k_recovery_rejects_hardlinked_last_symlink_before_mutating(
    tmp_path: Path,
) -> None:
    env, _contract, training, staging, upstream = _remote_train_fixture(tmp_path)
    interrupted = _run_actual_remote_train_stage(
        env, attempt_id="attempt-last-hardlink", interrupt_point="after-trainer", complete=True,
    )
    assert interrupted.returncode == 130, interrupted.stderr
    last = upstream / "checkpoints/last"
    external = tmp_path / "outside-last-hardlink"
    try:
        os.link(last, external, follow_symlinks=False)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"filesystem cannot create a hardlink to a symlink: {error}")
    before = last.lstat()
    recovered = _run_actual_remote_train_stage(
        env, attempt_id="attempt-last-hardlink-recovery", complete=False,
    )
    assert recovered.returncode != 0
    after = last.lstat()
    assert (before.st_uid, before.st_gid, before.st_ino, before.st_nlink) == (
        after.st_uid, after.st_gid, after.st_ino, after.st_nlink
    )
    assert not training.exists() and upstream.is_dir() and staging.is_dir()
    assert Path(env["FAKE_TRACE"]).read_text(encoding="utf-8").count(
        "native:/opt/lehome-challenge/.venv/bin/lerobot-train"
    ) == 1


def test_completed_12k_recovery_rejects_a_late_unsafe_entry_before_any_mutation(
    tmp_path: Path,
) -> None:
    """The authenticated plan must finish before it changes even a safe leaf."""
    env, _contract, training, staging, upstream = _remote_train_fixture(tmp_path)
    interrupted = _run_actual_remote_train_stage(
        env, attempt_id="attempt-late-unsafe", interrupt_point="after-trainer", complete=True,
    )
    assert interrupted.returncode == 130, interrupted.stderr

    # `staging` is scanned after the completed upstream tree.  The entry is
    # deliberately late in its directory order so a mutating preorder walk
    # would have already touched the upstream output.
    (staging / "zz-late-unsafe-entry").symlink_to("/dev")
    trace = tmp_path / "ownership-order"
    upstream_before = upstream.stat()
    recovered = _run_actual_remote_train_stage(
        {**env, "LEHOME_N15_TEST_OWNERSHIP_ORDER_TRACE": str(trace)},
        attempt_id="attempt-late-unsafe-recovery", complete=False,
    )

    assert recovered.returncode != 0
    assert not trace.exists(), "no authenticated descriptor may be fchown'd"
    upstream_after = upstream.stat()
    assert (upstream_after.st_uid, upstream_after.st_gid, upstream_after.st_ino) == (
        upstream_before.st_uid, upstream_before.st_gid, upstream_before.st_ino,
    )
    assert not training.exists() and upstream.is_dir() and staging.is_dir()
    assert Path(env["FAKE_TRACE"]).read_text(encoding="utf-8").count(
        "native:/opt/lehome-challenge/.venv/bin/lerobot-train"
    ) == 1


def test_completed_12k_recovery_mutates_leaf_descriptors_before_directories(
    tmp_path: Path,
) -> None:
    env, _contract, training, staging, upstream = _remote_train_fixture(tmp_path)
    interrupted = _run_actual_remote_train_stage(
        env, attempt_id="attempt-order", interrupt_point="after-trainer", complete=True,
    )
    assert interrupted.returncode == 130, interrupted.stderr
    order = tmp_path / "ownership-order"
    recovered = _run_actual_remote_train_stage(
        {**env, "LEHOME_N15_TEST_OWNERSHIP_ORDER_TRACE": str(order)},
        attempt_id="attempt-order-recovery", complete=False,
    )
    assert recovered.returncode == 0, recovered.stderr
    rows = order.read_text(encoding="ascii").splitlines()
    directory_rows = [index for index, row in enumerate(rows) if row.startswith("directory:")]
    assert directory_rows
    assert all(not row.startswith("directory:") for row in rows[:directory_rows[0]])
    # Roots are encoded as an empty relative path and must be repaired last,
    # after nested 0700 directories have become accessible.
    assert rows[-2:] == ["directory:", "directory:"]
    assert any(row.startswith("directory:checkpoints/012000") for row in rows[:-2])


@pytest.mark.skipif(
    platform.system() != "Linux" or os.geteuid() != 0,
    reason="requires a Linux root mount namespace",
)
def test_linux_root_recovery_rejects_bind_mount_before_foreign_uid_mutation(
    tmp_path: Path,
) -> None:
    """openat2 must reject a same-device bind mount before any fchown."""
    if shutil.which("mount") is None or shutil.which("umount") is None:
        pytest.skip("mount tools are unavailable")
    env, _contract, training, staging, upstream = _remote_train_fixture(tmp_path)
    interrupted = _run_actual_remote_train_stage(
        env, attempt_id="attempt-linux-bind", interrupt_point="after-trainer", complete=True,
    )
    assert interrupted.returncode == 130, interrupted.stderr
    foreign_uid = 65534
    sentinel = upstream / "checkpoints/012000/pretrained_model/model.safetensors"
    external = tmp_path / "external-bind-source"; external.mkdir()
    external_sentinel = external / "must-not-chown"; external_sentinel.write_text("outside", encoding="ascii")
    escape = upstream / "same-device-bind"; escape.mkdir()
    sentinel.chown(foreign_uid, foreign_uid)
    upstream.chown(foreign_uid, foreign_uid); upstream.chmod(0o700)
    mounted = subprocess.run(["mount", "--bind", str(external), str(escape)], text=True, capture_output=True)
    if mounted.returncode:
        pytest.skip(f"bind mount unavailable: {mounted.stderr.strip()}")
    try:
        result = _run_actual_remote_train_stage(
            env, attempt_id="attempt-linux-bind-recovery", complete=False,
        )
        assert result.returncode != 0
        assert sentinel.stat().st_uid == foreign_uid
        assert external_sentinel.stat().st_uid == 0
        assert not training.exists()
        assert Path(env["FAKE_TRACE"]).read_text(encoding="utf-8").count(
            "native:/opt/lehome-challenge/.venv/bin/lerobot-train"
        ) == 1
    finally:
        subprocess.run(["umount", str(escape)], check=True)


def test_completed_12k_canonical_unsealed_topology_is_sealed_without_retraining(
    tmp_path: Path,
) -> None:
    env, _contract, training, staging, upstream = _remote_train_fixture(tmp_path)
    interrupted = _run_actual_remote_train_stage(
        env, attempt_id="attempt-canonical-unsealed", interrupt_point="after-trainer", complete=True,
    )
    assert interrupted.returncode == 130, interrupted.stderr
    # Reproduce the historical crash window: upstream and evidence components
    # were already assembled under the canonical name, but checksum/identity
    # sealing had not begun.
    shutil.rmtree(staging / "eagle-home")
    upstream.rename(training)
    for component in ("evidence", "logs", "runtime"):
        (staging / component).rename(training / component)
    staging.rmdir()
    training.chmod(0o700)
    wandb_run = "offline-run-20260903_070403-ybfzdp6h"
    wandb_logs = training / "wandb" / wandb_run / "logs"
    wandb_logs.mkdir(parents=True, exist_ok=True)
    (wandb_logs / "debug.log").write_text("debug", encoding="utf-8")
    (wandb_logs / "debug-internal.log").write_text("internal", encoding="utf-8")
    (training / "wandb/latest-run").symlink_to(wandb_run)
    (training / "wandb/debug.log").symlink_to(f"{wandb_run}/logs/debug.log")
    (training / "wandb/debug-internal.log").symlink_to(
        f"{wandb_run}/logs/debug-internal.log"
    )
    (wandb_logs / "debug-core.log").symlink_to(
        "/root/.cache/wandb/logs/core-debug-20260903_070405.log"
    )
    (training / "wandb").chmod(0o700)
    sudo_trace = tmp_path / "canonical-sudo-argv.json"

    recovered = _run_actual_remote_train_stage(
        {
                **env,
                "FAKE_SUDO_TRACE": str(sudo_trace),
                "LEHOME_N15_RECOVER_COMPLETED_12K": "1",
                "LEHOME_N15_RESUME_PARTIAL": "0",
                "LEHOME_N15_RESUME_STEP": "",
                "LEHOME_N15_RESUME_CHECKPOINT": "",
                "LEHOME_N15_RESUME_ATTEMPT_ID": "",
        },
        attempt_id="attempt-canonical-unsealed-recovery", complete=False,
    )

    assert recovered.returncode == 0, recovered.stderr
    assert training.is_dir() and not upstream.exists() and not staging.exists()
    assert (training / "checksums.sha256").is_file()
    assert (training / "training-identity.json").is_file()
    assert (wandb_logs / "debug.log").is_file()
    assert (wandb_logs / "debug-internal.log").is_file()
    for transient in (
        training / "wandb/latest-run",
        training / "wandb/debug.log",
        training / "wandb/debug-internal.log",
        wandb_logs / "debug-core.log",
    ):
        assert not transient.is_symlink()
    assert json.loads(sudo_trace.read_text(encoding="utf-8"))[7] == "canonical"
    trace = Path(env["FAKE_TRACE"]).read_text(encoding="utf-8")
    assert trace.count("native:/opt/lehome-challenge/.venv/bin/lerobot-train") == 1


def test_completed_12k_recovery_rejects_spoofed_wandb_symlink_before_mutation(
    tmp_path: Path,
) -> None:
    env, _contract, training, staging, upstream = _remote_train_fixture(tmp_path)
    interrupted = _run_actual_remote_train_stage(
        env, attempt_id="attempt-spoofed-wandb", interrupt_point="after-trainer",
        complete=True,
    )
    assert interrupted.returncode == 130, interrupted.stderr
    shutil.rmtree(staging / "eagle-home")
    upstream.rename(training)
    for component in ("evidence", "logs", "runtime"):
        (staging / component).rename(training / component)
    staging.rmdir()
    (training / "wandb").mkdir(exist_ok=True)
    (training / "wandb/debug.log").symlink_to("../../outside")
    order = tmp_path / "spoofed-wandb-order"

    recovered = _run_actual_remote_train_stage(
        {
            **env,
            "LEHOME_N15_TEST_OWNERSHIP_ORDER_TRACE": str(order),
            "LEHOME_N15_RECOVER_COMPLETED_12K": "1",
        },
        attempt_id="attempt-spoofed-wandb-recovery", complete=False,
    )

    assert recovered.returncode != 0
    assert not order.exists(), "unsafe W&B link must fail before any mutation"
    assert (training / "wandb/debug.log").is_symlink()
    assert not (training / "training-identity.json").exists()
    assert Path(env["FAKE_TRACE"]).read_text(encoding="utf-8").count(
        "native:/opt/lehome-challenge/.venv/bin/lerobot-train"
    ) == 1


def test_completed_12k_split_recovery_rejects_duplicate_wandb_sets_before_mutation(
    tmp_path: Path,
) -> None:
    env, _contract, training, staging, upstream = _remote_train_fixture(tmp_path)
    interrupted = _run_actual_remote_train_stage(
        env, attempt_id="attempt-duplicate-wandb", interrupt_point="after-trainer",
        complete=True,
    )
    assert interrupted.returncode == 130, interrupted.stderr

    wandb_run = "offline-run-20260903_070403-ybfzdp6h"
    for root in (upstream, staging):
        logs = root / "wandb" / wandb_run / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "debug.log").write_text("debug", encoding="utf-8")
        (logs / "debug-internal.log").write_text("internal", encoding="utf-8")
        (root / "wandb/latest-run").symlink_to(wandb_run)
        (root / "wandb/debug.log").symlink_to(f"{wandb_run}/logs/debug.log")
        (root / "wandb/debug-internal.log").symlink_to(
            f"{wandb_run}/logs/debug-internal.log"
        )
        (logs / "debug-core.log").symlink_to(
            "/root/.cache/wandb/logs/core-debug-20260903_070405.log"
        )
    order = tmp_path / "duplicate-wandb-order"

    recovered = _run_actual_remote_train_stage(
        {**env, "LEHOME_N15_TEST_OWNERSHIP_ORDER_TRACE": str(order)},
        attempt_id="attempt-duplicate-wandb-recovery", complete=False,
    )

    assert recovered.returncode != 0
    assert not order.exists(), "duplicate W&B sets must fail before any mutation"
    for root in (upstream, staging):
        assert (root / "wandb/latest-run").is_symlink()
        assert not (root / "training-identity.json").exists()
    assert Path(env["FAKE_TRACE"]).read_text(encoding="utf-8").count(
        "native:/opt/lehome-challenge/.venv/bin/lerobot-train"
    ) == 1


def test_recovery_only_mode_rejects_absent_canonical_output_without_trainer(
    tmp_path: Path,
) -> None:
    env, _contract, training, staging, upstream = _remote_train_fixture(tmp_path)
    # The normal partial tree is deliberately not an admissible recovery-only
    # topology.  A production restart must stop/fail rather than using it to
    # enter the ordinary resume/fresh trainer path.
    result = _run_actual_remote_train_stage(
        {**env, "LEHOME_N15_RECOVER_COMPLETED_12K": "1"},
        attempt_id="attempt-recovery-only-absent", complete=False,
    )
    assert result.returncode != 0
    assert not training.exists() and staging.is_dir() and upstream.is_dir()
    assert not Path(env["FAKE_TRACE"]).exists()


def test_recovery_only_rejects_generic_finalizing_state_without_finalizer_or_trainer(
    tmp_path: Path,
) -> None:
    env, _contract, training, staging, upstream = _remote_train_fixture(tmp_path)
    # This stale generic state used to be checked before recovery-only
    # admission.  It must not redirect an explicit canonical recovery into a
    # finalizer that may consume the split topology.
    finalizing = Path(f"{training}.finalizing")
    finalizing.mkdir()
    trace = tmp_path / "python-invocations"
    shim = tmp_path / "python-shim"; shim.mkdir()
    _write_executable(
        shim / "python3",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_PYTHON_TRACE\"\nexec \"$REAL_PYTHON\" \"$@\"\n",
    )
    result = _run_actual_remote_train_stage(
        {
            **env,
            "LEHOME_N15_RECOVER_COMPLETED_12K": "1",
            "LEHOME_N15_RESUME_PARTIAL": "0",
            "LEHOME_N15_RESUME_STEP": "",
            "LEHOME_N15_RESUME_CHECKPOINT": "",
            "LEHOME_N15_RESUME_ATTEMPT_ID": "",
            "PATH": f"{shim}:{env['PATH']}",
            "FAKE_PYTHON_TRACE": str(trace),
            "REAL_PYTHON": sys.executable,
        },
        attempt_id="attempt-recovery-finalizing", complete=False,
    )
    assert result.returncode != 0
    assert "canonical training root is absent" in result.stderr
    assert not trace.exists() or "finalize-training-output" not in trace.read_text(encoding="utf-8")
    assert not Path(env["FAKE_TRACE"]).exists()
    assert finalizing.is_dir() and staging.is_dir() and upstream.is_dir()


def test_recovery_only_main_bypasses_only_expired_train_stage_deadline(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, "n15-recovery-expired-train")
    env["LEHOME_N15_RECOVER_COMPLETED_12K"] = "1"
    pipeline = Path(env["LEHOME_N15_PIPELINE_ROOT"])
    module = _load_cli()
    plan = pipeline / "lifecycle-plan.json"
    assert module.main([
        "lifecycle-plan", "--run-id", env["LEHOME_N15_RUN_ID"],
        "--repository", env["LEHOME_N15_PUBLIC_HF_REPOSITORY"],
        "--remote-pipeline-root", env["LEHOME_N15_REMOTE_PIPELINE_ROOT"],
        "--budget-usd", "100", "--estimated-cost-usd", "72", "--output", str(plan),
    ]) == 0
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    now = int(time.time())
    paid = {
        "schema_version": 1, "kind": "lehome_public_n15_paid_deadline_v1",
        "run_id": env["LEHOME_N15_RUN_ID"], "lifecycle_plan_sha256": digest,
        "started_unix_seconds": now - 60, "deadline_unix_seconds": now + 86340,
    }
    train = {
        "schema_version": 1, "kind": "lehome_public_n15_stage_deadline_v1",
        "run_id": env["LEHOME_N15_RUN_ID"], "stage": "train",
        "lifecycle_plan_sha256": digest, "started_unix_seconds": now - 43201,
        "deadline_unix_seconds": now - 1,
    }
    for path, value in ((pipeline / "paid-deadline.json", paid), (pipeline / "stage-train-deadline.json", train)):
        path.write_bytes(_canonical_json_bytes(value)); path.chmod(0o444)
    trace = tmp_path / "trace"
    harness = r'''
source "$WRAPPER_PATH"
acquire_controller_lock() { :; }
release_controller_lock() { :; }
host_next_unfinished_stage() { printf 'train\n'; }
initialize_stage_deadline() { printf 'stage-deadline:%s\n' "$1" >> "$TRACE"; return 88; }
capture_exact_provider_state() { printf 'provider-read:%s\n' "$1" >> "$TRACE"; : > "$2"; }
nebius() { printf 'provider-start\n' >> "$TRACE"; }
run_aggregate_supervised_lifecycle() { printf 'recovery-lifecycle:%s\n' "$PRESTART_ADMITTED_STAGE" >> "$TRACE"; train_stage; }
train_stage() { printf 'recovery-authentication-failed-no-trainer\n' >> "$TRACE"; return 91; }
stop_exact_vm() { printf 'provider-stop\n' >> "$TRACE"; }
main
'''
    result = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={**env, "WRAPPER_PATH": str(WRAPPER), "TRACE": str(trace)},
        text=True, capture_output=True,
    )
    assert result.returncode != 0
    lines = trace.read_text(encoding="ascii").splitlines()
    assert "stage-deadline:train" not in lines
    assert "provider-start" in lines
    assert "recovery-lifecycle:completed_12k_recovery" in lines
    assert "recovery-authentication-failed-no-trainer" in lines
    assert lines[-1] == "provider-stop"
    assert (pipeline / "stage-train-deadline.json").read_bytes() == _canonical_json_bytes(train)


def test_lifecycle_plan_is_immutable_and_has_exact_paid_stage_order(tmp_path: Path) -> None:
    module = _load_cli()
    output = tmp_path / "pipeline-plan.json"
    assert module.main([
        "lifecycle-plan", "--run-id", "n15-20260831-a", "--repository", "ryanjin333/public-n15", "--remote-pipeline-root", "/mnt/lehome/public-n15-runs/n15-20260831-a", "--budget-usd", "100",
        "--estimated-cost-usd", "99.99", "--output", str(output),
    ]) == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["vm_id"] == "computeinstance-u00t6xfqhadrcmssa2"
    assert value["protected_disk_id"] == "computedisk-u00pbe55crxy7jr56x"
    assert value["budget_usd"] == 100.0
    assert value["provider_source_image_id"] == "computeimage-u00zf6w3yf72gakhcy"
    assert value["prefixes"]["harvest"] == "n15-public/n15-20260831-a/harvest"
    assert value["stages"] == [
        "verify_stopped", "start", "validate_runtime", "train", "train_publish_readback",
        "focused_gate", "focused_gate_publish_readback", "harvest",
        "harvest_publish_readback", "stop",
    ]
    assert output.stat().st_mode & 0o777 == 0o444
    assert module.main([
        "lifecycle-plan", "--run-id", "n15-20260831-a", "--repository", "ryanjin333/public-n15", "--remote-pipeline-root", "/mnt/lehome/public-n15-runs/n15-20260831-a", "--budget-usd", "100",
        "--estimated-cost-usd", "99.99", "--output", str(output),
    ]) == 2


def test_lifecycle_plan_refuses_over_budget_before_any_provider_action(tmp_path: Path, capsys) -> None:
    module = _load_cli()
    output = tmp_path / "pipeline-plan.json"
    assert module.main([
        "lifecycle-plan", "--run-id", "n15-20260831-b", "--repository", "ryanjin333/public-n15", "--remote-pipeline-root", "/mnt/lehome/public-n15-runs/n15-20260831-b", "--budget-usd", "100",
        "--estimated-cost-usd", "100.01", "--output", str(output),
    ]) == 2
    assert not output.exists()
    assert "budget" in capsys.readouterr().err.lower()


def test_lifecycle_plan_resume_requires_exact_canonical_run_and_prefixes(tmp_path: Path) -> None:
    module = _load_cli()
    output = tmp_path / "pipeline-plan.json"
    arguments = ["--run-id", "n15-20260831-c", "--repository", "ryanjin333/public-n15", "--remote-pipeline-root", "/mnt/lehome/public-n15-runs/n15-20260831-c", "--budget-usd", "100", "--estimated-cost-usd", "3", "--output", str(output)]
    assert module.main(["lifecycle-plan", *arguments]) == 0
    assert module.main(["verify-lifecycle-plan", *arguments]) == 0
    altered = json.loads(output.read_text(encoding="utf-8")); altered["prefixes"]["harvest"] = "shared/latest"
    output.chmod(0o644); output.write_text(json.dumps(altered), encoding="utf-8")
    assert module.main(["verify-lifecycle-plan", *arguments]) == 2


def test_remote_wrapper_is_single_vm_fail_closed_and_receipt_resumable() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert 'EXACT_VM_ID="computeinstance-u00t6xfqhadrcmssa2"' in text
    assert 'PROTECTED_DISK_ID="computedisk-u00pbe55crxy7jr56x"' in text
    assert 'EXACT_IMAGE_ID="computeimage-u00zf6w3yf72gakhcy"' in text
    assert 'RUNTIME_IMAGE_ID="sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7"' in text
    assert '"$LEROBOT_WHEEL" "$RUNTIME_IMAGE_ID" "$RESUME_PARTIAL" "$RESUME_CHECKPOINT" "$RESUME_STEP" "$RESUME_ATTEMPT_ID" "$RECOVER_COMPLETED_12K" <<\'SH\'' in text
    assert 'runtime_image_id="${12}"' in text
    assert 'recovery_only="${17}"' in text
    assert " LEROBOT_WHEEL RUNTIME_IMAGE_ID RESUME_PARTIAL RESUME_CHECKPOINT RESUME_STEP RESUME_ATTEMPT_ID RECOVER_COMPLETED_12K ASSETS_ROOT" in text
    assert "LEHOME_N15_EXPECTED_IMAGE_ID" not in text
    assert "nebius compute instance start --id" in text
    assert "nebius compute instance stop --id" in text
    assert "compute instance create" not in text
    assert "compute disk create" not in text
    assert "compute image create" not in text
    assert "trap controller_cleanup EXIT" in text
    assert text.count("StrictHostKeyChecking=accept-new") == 2
    assert "readonly SSH_READINESS_ATTEMPTS=24" in text
    assert "readonly SSH_READINESS_HARD_TIMEOUT_SECONDS=8" in text
    assert "readonly REMOTE_RUNTIME_ATTEMPTS=18" in text
    assert "attempt <= REMOTE_RUNTIME_ATTEMPTS" in text
    assert "LEHOME_N15_MAX_BUDGET_USD" in text
    assert "LEHOME_N15_ESTIMATED_COST_USD" not in text
    assert "PROVIDER_HOURLY_CEILING_USD=3" in text
    assert "run_public_n15_reproduction.py lifecycle-plan" in text
    assert 'readonly TRAINING_UV="${LEHOME_N15_TRAINING_UV:-}"' in text
    assert 'test -x "$uv_bin" && test ! -L "$uv_bin"' in text
    assert 'export UV_CACHE_DIR="$(dirname -- "$python_bin")/.uv-cache"' in text
    assert 'export TMPDIR="$(dirname -- "$python_bin")/.uv-tmp"' in text
    assert 'export UV_LINK_MODE=copy' in text
    assert 'sudo -n docker image inspect -- "$runtime_image_id"' in text
    assert 'sudo -n docker run --rm -i --pull never --gpus all --network none' in text
    assert text.count('--shm-size "32g"') == 1
    assert '--tmpfs "/flash:rw,exec,size=2g,mode=700,uid=$(id -u),gid=$(id -g)"' in text
    assert 'with zipfile.ZipFile(flash_wheel) as archive:' in text
    assert 'PYTHONPATH="$pythonpath" "$python_bin"' in text
    assert 'expected = "/flash/site-packages/flash_attn/__init__.py"' in text
    assert "grep -Eq '^(disk|part|lvm|crypt)$'" in text
    assert 'lsblk -ndo MAJ:MIN /dev/disk/by-id/virtio-lehome' in text
    assert '"$uv_bin" pip install --offline --no-deps --reinstall --python "$python_bin"' in text
    assert "materialize-lerobot-package" in text
    assert '--package-root "$staging_root/runtime/site-packages/lerobot"' in text
    assert '"$training_root/runtime/site-packages/lerobot" <<\'PY\'' in text
    assert 'package = Path(importlib.util.find_spec("lerobot").origin).parent' not in text
    compatibility_install = text.index('"$uv_bin" pip install --offline --no-deps --reinstall --python "$python_bin"')
    assert compatibility_install < text.index('test -x "$(dirname -- "$python_bin")/lerobot-train"')
    assert compatibility_install < text.index('"$python_bin" -I -c \'import lerobot; from pathlib import Path; assert Path(lerobot.__file__).is_file()\'')
    assert 'eagle_repository="$hf_cache/models--lerobot--eagle2hg-processor-groot-n1p5"' in text
    assert 'eagle_snapshot="$eagle_repository/snapshots/baf604d8a5caf26fda5cc545f141bc1814156237"' in text
    assert 'eagle_home="$staging_root/eagle-home"' in text
    assert 'export HF_HOME="$eagle_home" HF_HUB_OFFLINE=1 HF_HUB_CACHE="$hf_cache"' in text
    assert 'HF_LEROBOT_HOME="$eagle_home/lerobot"' in text
    assert 'export HF_HOME HF_LEROBOT_HOME HF_HUB_OFFLINE HF_HUB_CACHE' in text
    assert 'prepare-peft-overlay --receipt "$2"' in text
    assert '"$staging_root/evidence/peft-overlay-receipt.json"' in text
    assert 'peft_wheel="/mnt/lehome/reference-native/dependencies/peft-0.18.1-py3-none-any.whl"' in text
    assert 'PYTHONPATH="/flash/site-packages:/deps/peft-0.18.1-py3-none-any.whl"' in text
    assert 'prepare-flash-attention-overlay --receipt "$3"' in text
    assert '"$staging_root/evidence/flash-attention-overlay-receipt.json"' in text
    assert 'flash_wheel="/mnt/lehome/reference-native/dependencies/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"' in text
    assert 'dm_tree_wheel="/mnt/lehome/reference-native/dependencies/dm_tree-0.1.9-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"' in text
    assert '294dc1cecf87552a45cdd5ddb215e7f5295a5a47c46f1f0a0463c3dd02a527d7' in text
    pinned_preflight = text.index('sudo -n docker run --rm -i --pull never --gpus all --network none')
    preflight_command = text[pinned_preflight:text.index("<<'CONTAINER'", pinned_preflight)]
    assert '--user "$(id -u):$(id -g)"' not in preflight_command
    assert pinned_preflight < text.index('prepare-peft-overlay --receipt "$2"')
    assert pinned_preflight < text.index('prepare-flash-attention-overlay --receipt "$3"')
    assert '--mount "type=bind,src=$root,dst=$root,readonly"' in text
    assert '--mount "type=bind,src=/mnt/lehome/reference-native/dependencies,dst=/mnt/lehome/reference-native/dependencies,readonly"' in text
    host_preflight = text[text.index('"$python_bin" -I -c'):pinned_preflight]
    assert "prepare-peft-overlay" not in host_preflight
    assert "prepare-flash-attention-overlay" not in host_preflight
    assert 'flash_attn_2_cuda' in text
    assert 'top_level.startswith("flash_attn-")' in text
    assert 'top_level.endswith(".dist-info")' in text
    assert 'importlib.metadata.version("flash_attn")' in text
    assert 'lerobot_wheel = Path("/runtime/lerobot-0.4.3-py3-none-any.whl")' in text
    assert 'with zipfile.ZipFile(lerobot_wheel) as archive:' in text
    assert text.count('with zipfile.ZipFile(dm_tree_wheel) as archive:') == 2
    assert 'import tree' in text
    assert 'importlib.metadata.version("dm-tree")' in text
    assert 'expected_tree = "/flash/site-packages/tree/__init__.py"' in text
    assert 'tree.map_structure(lambda left, right: left + right, {"joint": 1}, {"joint": 2})' in text
    assert 'groot_n1.tree is not tree' in text
    assert '"dm_tree_version":' not in text
    assert '"dm_tree_origin":' not in text
    assert "import lerobot.scripts.lerobot_train" in text
    assert '--mount "type=bind,src=$hf_cache,dst=$hf_cache,readonly"' in text
    assert 'dataset_blobs="$(' in text
    assert "from lehome.n15_reproduction import resolve_dataset_blobs_mount" in text
    assert "print(resolve_dataset_blobs_mount(" in text
    assert '--mount "type=bind,src=$dataset_blobs,dst=$dataset_blobs,readonly"' in text
    assert '--mount "type=bind,src=$staging_root/evidence/compatibility/lerobot-0.4.3-py3-none-any.whl,dst=/runtime/lerobot-0.4.3-py3-none-any.whl,readonly"' in text
    assert 'find "$eagle_home" -depth -type f -delete' in text
    assert 'flash-attention-runtime-receipt.json' in text
    assert 'training-container-runtime-receipt.json' in text
    assert '"$runtime_image_id" -s --' in text
    assert 'lehome-rollout:build -s --' not in text
    training_command = text[text.index('sudo -n docker run --rm -i --pull never --gpus all --network none', pinned_preflight + 1):]
    training_command = training_command[:training_command.index("<<'CONTAINER'")]
    assert '--user "$(id -u):$(id -g)"' not in training_command
    ownership_handoff = 'sudo -n chown -R --no-dereference "$(id -u):$(id -g)" "$upstream_output" "$eagle_home" "$staging_root"'
    assert ownership_handoff in text
    assert text.index(ownership_handoff) > text.index('/opt/lehome-challenge/.venv/bin/lerobot-train --config_path=configs/train_groot.yaml')
    assert 'find "$upstream_output" "$eagle_home" "$staging_root" ! -user "$(id -u)" -print -quit' in text
    assert text.index('export HF_HOME="$eagle_home" HF_HUB_OFFLINE=1 HF_HUB_CACHE="$hf_cache"') < text.index('/opt/lehome-challenge/.venv/bin/lerobot-train --config_path=configs/train_groot.yaml')
    assert 'eagle_asset_source="$(readlink -f "$eagle_snapshot/$eagle_asset")"' in text
    assert '[[ "$eagle_asset_source" == "$eagle_repository/blobs/"* ]]' in text
    assert "run_public_n15_focused_gate.sh" in text
    assert "run_public_n15_harvest.sh" in text
    assert "immutable receipt" in text
    assert "anonymous" in text.lower()
    for required in (
        "LEHOME_OFFICIAL_RUNTIME_REVISION", "LEHOME_OFFICIAL_SOURCE_ROOT",
        "LEHOME_OFFICIAL_ASSETS_ROOT", "LEHOME_OFFICIAL_METADATA_ROOT",
        "LEHOME_N15_CANDIDATE_CHECKPOINT", "LEHOME_N15_CANDIDATE_IDENTITY_RECEIPT",
        "LEHOME_N15_FOCUSED_PROMOTION_RECEIPT", "LEHOME_N15_HARVEST_ROOT",
        "LEHOME_N15_TERMINAL_RECEIPT",
    ):
        assert required in text
    assert text.index("train_stage") < text.index("focused_stage") < text.index("harvest_stage")
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_remote_helper_preserves_empty_positional_arguments(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "ssh",
        "#!/usr/bin/env bash\n"
        "shift 5\n"
        "exec bash -c \"$*\"\n",
    )
    command = """
source "$1"
remote bash -s -- alpha "" omega <<'SH'
set -u
printf '<%s>|<%s>|<%s>\\n' "$1" "$2" "$3"
SH
"""
    result = subprocess.run(
        ["bash", "-c", command, "_", str(WRAPPER)],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "LEHOME_N15_SSH_TARGET": "operator@example",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "<alpha>|<>|<omega>\n"


def test_remote_wrapper_never_runs_downstream_after_a_failed_gate() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "verify_remote_training_chain" in text
    assert "verify_remote_focused_chain" in text
    assert "verify_remote_harvest_chain" in text
    assert "run_paid_stage focused_gate" in text
    assert "run_paid_stage harvest" in text
    assert "paid-deadline.json" in text
    assert text.index("verify_remote_focused_chain || fail") < text.rindex("run_paid_stage harvest")


def test_remote_wrapper_resume_is_explicit_exact_and_preserves_immutable_evidence() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert 'readonly RESUME_PARTIAL="${LEHOME_N15_RESUME_PARTIAL:-0}"' in text
    assert 'readonly RESUME_CHECKPOINT="${LEHOME_N15_RESUME_CHECKPOINT:-}"' in text
    assert 'readonly RESUME_STEP="${LEHOME_N15_RESUME_STEP:-}"' in text
    assert 'readonly RESUME_ATTEMPT_ID="${LEHOME_N15_RESUME_ATTEMPT_ID:-}"' in text
    assert '[[ "$RESUME_PARTIAL" == 0 || "$RESUME_PARTIAL" == 1 ]]' in text
    assert '[[ "$resume_checkpoint" == "$upstream_output/checkpoints/$resume_name" ]]' in text
    assert "verify-resume-checkpoint" in text
    assert '--resume-step "$resume_step"' in text
    assert '--attempt-id "$resume_attempt_id"' in text
    assert '--config_path="$resume_checkpoint/pretrained_model/train_config.json"' in text
    assert "--resume=true" in text
    assert (
        'PYTHONPATH="/flash/site-packages:/deps/peft-0.18.1-py3-none-any.whl" '
        '/opt/lehome-challenge/.venv/bin/lerobot-train'
    ) in text
    assert 'logs/train-resume-${resume_attempt_id}.log' in text
    assert 'evidence/resume-attempts/${resume_attempt_id}.json' in text
    assert 'cmp -s "$temporary_receipt" "$immutable_receipt"' in text
    assert 'mv -- "$temporary_receipt" "$immutable_receipt"' not in text
    assert "finalize-training-output" in text
    assert 'mv -- "$upstream_output" "$training_root"' not in text
    assert 'mv -- "$staging_root/evidence" "$training_root/evidence"' not in text
    assert "provider must be STOPPED before explicit partial resume" in text
    assert "acquire_controller_lock" in text
    assert "release_controller_lock" in text
    assert text.index('aggregate_deadline="$(initialize_deadline)"') < text.index("nebius compute instance start --id")
    assert "pgrep -f" in text
    assert 'findmnt -T "$staging_root" --noheadings --output MAJ:MIN' in text
    assert 'findmnt -T "$upstream_output" --noheadings --output MAJ:MIN' in text
    terminal_guard = '[[ "$RESUME_PARTIAL" == 1 && -f "$HARVEST_TERMINAL_RECEIPT" ]]'
    assert terminal_guard in text
    assert text.index(terminal_guard) < text.index('if [[ -f "$HARVEST_TERMINAL_RECEIPT" ]]')


def test_remote_wrapper_fresh_mode_does_not_discover_or_resume_a_checkpoint() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert 'if [[ "$resume_partial" == 1 ]]; then' in text
    fresh_branch = text[text.index('if [[ "$resume_partial" == 1 ]]; then'):]
    assert 'else\n  test ! -e "$training_root"' in fresh_branch
    assert "find \"$upstream_output/checkpoints\"" not in text
    assert "readlink \"$upstream_output/checkpoints/last\"" not in text


def test_runtime_gate_creates_the_fresh_run_directory_only_after_proving_the_workspace_mount() -> None:
    """The first verified-inputs receipt needs a real directory on the protected disk."""
    text = WRAPPER.read_text(encoding="utf-8")
    assert 'workspace_base="$(dirname -- "$pipeline_root")"' in text
    assert 'mkdir -m 0700 -- "$pipeline_root"' in text
    assert text.index('mkdir -m 0700 -- "$pipeline_root"') < text.index('verified_inputs="$(dirname -- "$training_root")/verified-inputs.json"')
    assert text.index('[[ "$(lsblk -ndo MAJ:MIN /dev/disk/by-id/virtio-lehome)" == "$(findmnt -T "$workspace_base" --noheadings --output MAJ:MIN)" ]]') < text.index('mkdir -m 0700 -- "$pipeline_root"')


def test_over_budget_plan_never_starts_the_mocked_exact_vm(tmp_path: Path) -> None:
    """A failing preflight may stop, but it must never make a start request."""
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    log = tmp_path / "nebius.log"
    raw = {
        "metadata": {"id": "computeinstance-u00t6xfqhadrcmssa2", "name": "lehome-rollout"},
        "status": {"state": "STOPPED"},
        "spec": {"boot_disk": {"managed_disk": {"spec": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}}}, "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}]},
    }
    (fake_bin / "nebius").write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\nopen(os.environ['FAKE_NEBIUS_LOG'], 'a').write(' '.join(sys.argv[1:]) + '\\n')\nprint(json.dumps(" + repr(raw) + "))\n",
        encoding="utf-8",
    )
    (fake_bin / "ssh").write_text("#!/usr/bin/env bash\nexit 97\n", encoding="utf-8")
    for command in (fake_bin / "nebius", fake_bin / "ssh"): command.chmod(0o755)
    pipeline = tmp_path / "pipeline"; pipeline.mkdir()
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "FAKE_NEBIUS_LOG": str(log),
           "LEHOME_N15_RUN_ID": "n15-over-budget", "LEHOME_N15_PIPELINE_ROOT": str(pipeline),
           "LEHOME_N15_MAX_BUDGET_USD": "71", "LEHOME_N15_SSH_TARGET": "operator@example", "LEHOME_N15_REMOTE_ROOT": "/mnt/lehome/runtime", "LEHOME_N15_REMOTE_RUNS_BASE": "/mnt/lehome/runs", "LEHOME_N15_REMOTE_PIPELINE_ROOT": "/mnt/lehome/runs/n15-over-budget", "LEHOME_N15_PUBLIC_HF_REPOSITORY": "ryanjin333/public-n15", "LEHOME_OFFICIAL_ASSETS_ROOT": "/mnt/assets", "LEHOME_OFFICIAL_METADATA_ROOT": "/mnt/source", "LEHOME_N15_REFERENCE_CHECKPOINT": "/mnt/reference", "LEHOME_N15_REFERENCE_SANITIZED_CONFIG_ROOT": "/mnt/reference-config", "LEHOME_N15_REFERENCE_COMPATIBILITY_RECEIPT": "/mnt/reference-receipt", "LEHOME_N15_NATIVE_RUNTIME_EVIDENCE_ROOT": "/mnt/evidence", "LEHOME_N15_NATIVE_DEPENDENCIES_ROOT": "/mnt/deps", "LEHOME_N15_FOCUSED_HF_CACHE_ROOT": "/mnt/cache", "LEHOME_N15_ROLLOUT_IMAGE_RECEIPT": "/mnt/image.json"}
    env.update({"LEHOME_N15_TRAINING_HF_CACHE_ROOT": "/mnt/train-cache", "LEHOME_N15_TRAINING_UV": "/mnt/uv", "LEHOME_N15_LEROBOT_WHEEL": "/mnt/lerobot.whl", "LEHOME_N15_TRAINING_ROOT": "/mnt/lehome/runs/n15-over-budget/training"})
    result = subprocess.run(["bash", str(WRAPPER)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert not log.exists()


@pytest.mark.parametrize("preemptible", [None, {"on_preemption": "DELETE"}])
def test_nonpreemptible_provider_never_starts_exact_vm(
    tmp_path: Path,
    preemptible: object,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    log = tmp_path / "nebius.log"
    raw = {
        "metadata": {"id": "computeinstance-u00t6xfqhadrcmssa2", "name": "lehome-rollout"},
        "status": {"state": "STOPPED"},
        "spec": {
            "boot_disk": {"managed_disk": {"spec": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}}},
            "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}],
        },
    }
    if preemptible is not None:
        raw["spec"]["preemptible"] = preemptible
    (fake_bin / "nebius").write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        "open(os.environ['FAKE_NEBIUS_LOG'], 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:4] == ['compute', 'instance', 'start']: raise SystemExit(97)\n"
        "print(json.dumps(" + repr(raw) + "))\n",
        encoding="utf-8",
    )
    (fake_bin / "ssh").write_text("#!/usr/bin/env bash\nexit 97\n", encoding="utf-8")
    for command in (fake_bin / "nebius", fake_bin / "ssh"): command.chmod(0o755)
    env = _wrapper_env(tmp_path, fake_bin, "n15-preemptible-admission")
    env["FAKE_NEBIUS_LOG"] = str(log)

    result = subprocess.run(
        ["bash", str(WRAPPER)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode != 0
    calls = log.read_text(encoding="utf-8").splitlines()
    assert any("compute instance get" in call for call in calls)
    assert all("compute instance start" not in call for call in calls)


@pytest.mark.parametrize("resume_step", [0, 1499, 1501, 12000])
def test_invalid_resume_boundary_fails_before_any_provider_call(
    tmp_path: Path, resume_step: int,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    provider_log = tmp_path / "provider.log"
    _write_executable(
        fake_bin / "nebius",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_PROVIDER_LOG\"\nexit 97\n",
    )
    _write_executable(fake_bin / "ssh", "#!/usr/bin/env bash\nexit 97\n")
    env = _wrapper_env(tmp_path, fake_bin, f"n15-invalid-step-{resume_step}")
    env.update({
        "FAKE_PROVIDER_LOG": str(provider_log),
        "LEHOME_N15_RESUME_PARTIAL": "1",
        "LEHOME_N15_RESUME_STEP": str(resume_step),
        "LEHOME_N15_RESUME_ATTEMPT_ID": "attempt-invalid-step",
        "LEHOME_N15_PUBLIC_SOURCE_ROOT": "/mnt/source",
        "LEHOME_N15_RESUME_CHECKPOINT": (
            "/mnt/source/outputs/train/groot_four_types_merged_batch64_lr2e-4/"
            f"checkpoints/{resume_step:06d}"
        ),
    })

    result = subprocess.run(
        ["bash", str(WRAPPER)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode != 0
    assert "checkpoint boundary" in result.stderr.lower()
    assert not provider_log.exists()


def test_resume_rejected_when_host_seal_places_pipeline_after_training_before_provider(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    provider_log = tmp_path / "provider.log"
    _write_executable(
        fake_bin / "nebius",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_PROVIDER_LOG\"\nexit 97\n",
    )
    _write_executable(fake_bin / "ssh", "#!/usr/bin/env bash\nexit 97\n")
    env = _wrapper_env(tmp_path, fake_bin, "n15-resume-after-training")
    env.update({
        "FAKE_PROVIDER_LOG": str(provider_log),
        "LEHOME_N15_RESUME_PARTIAL": "1",
        "LEHOME_N15_RESUME_STEP": "1500",
        "LEHOME_N15_RESUME_ATTEMPT_ID": "attempt-after-training",
        "LEHOME_N15_PUBLIC_SOURCE_ROOT": "/mnt/source",
        "LEHOME_N15_RESUME_CHECKPOINT": "/mnt/source/outputs/train/groot_four_types_merged_batch64_lr2e-4/checkpoints/001500",
    })
    pipeline = Path(env["LEHOME_N15_PIPELINE_ROOT"])
    seal = {
        "schema_version": 1,
        "kind": "lehome_public_n15_host_stage_completion_v1",
        "run_id": env["LEHOME_N15_RUN_ID"],
        "stage": "training",
        "remote_receipts": [
            {"path": env["LEHOME_N15_TRAINING_ROOT"] + "/training-identity.json", "sha256": "0" * 64},
            {"path": env["LEHOME_N15_TRAINING_ROOT"] + "/training-publication.json", "sha256": "1" * 64},
        ],
    }
    seal_path = pipeline / "host-stage-training-complete.json"
    seal_path.write_bytes(_canonical_json_bytes(seal)); seal_path.chmod(0o444)

    result = subprocess.run(
        ["bash", str(WRAPPER)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert result.returncode != 0
    assert "host-sealed next stage" in result.stderr
    assert not provider_log.exists()


@pytest.mark.parametrize("deadline_kind", ["invalid", "expired", "mutable"])
def test_invalid_immutable_deadline_fails_before_any_provider_action(
    tmp_path: Path, deadline_kind: str,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    provider_log = tmp_path / "provider.log"
    (fake_bin / "nebius").write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_PROVIDER_LOG\"\nexit 97\n",
        encoding="utf-8",
    )
    (fake_bin / "ssh").write_text("#!/usr/bin/env bash\nexit 97\n", encoding="utf-8")
    for command in (fake_bin / "nebius", fake_bin / "ssh"): command.chmod(0o755)
    env = _wrapper_env(tmp_path, fake_bin, "n15-invalid-deadline")
    env["FAKE_PROVIDER_LOG"] = str(provider_log)
    module = _load_cli()
    pipeline = Path(env["LEHOME_N15_PIPELINE_ROOT"])
    assert module.main([
        "lifecycle-plan", "--run-id", env["LEHOME_N15_RUN_ID"],
        "--repository", env["LEHOME_N15_PUBLIC_HF_REPOSITORY"],
        "--remote-pipeline-root", env["LEHOME_N15_REMOTE_PIPELINE_ROOT"],
        "--budget-usd", "100", "--estimated-cost-usd", "72",
        "--output", str(pipeline / "lifecycle-plan.json"),
    ]) == 0
    if deadline_kind == "invalid":
        deadline = {}
    else:
        plan_sha = hashlib.sha256((pipeline / "lifecycle-plan.json").read_bytes()).hexdigest()
        started = int(time.time()) if deadline_kind == "mutable" else 1
        deadline = {
            "schema_version": 1,
            "kind": "lehome_public_n15_paid_deadline_v1",
            "run_id": env["LEHOME_N15_RUN_ID"],
            "lifecycle_plan_sha256": plan_sha,
            "started_unix_seconds": started,
            "deadline_unix_seconds": started + 86400,
        }
    (pipeline / "paid-deadline.json").write_text(
        json.dumps(deadline, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii"
    )
    if deadline_kind != "mutable":
        (pipeline / "paid-deadline.json").chmod(0o444)

    result = subprocess.run(
        ["bash", str(WRAPPER)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode != 0
    assert "deadline" in result.stderr.lower()
    assert not provider_log.exists()


def test_conservative_budget_admission_needs_no_external_rate_receipt(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    provider_log = tmp_path / "provider.log"
    _write_executable(
        fake_bin / "nebius",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_PROVIDER_LOG\"\nexit 97\n",
    )
    _write_executable(fake_bin / "ssh", "#!/usr/bin/env bash\nexit 97\n")
    env = _wrapper_env(tmp_path, fake_bin, "n15-conservative-budget")
    env["FAKE_PROVIDER_LOG"] = str(provider_log)

    result = subprocess.run(
        ["bash", str(WRAPPER)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode != 0
    assert "rate receipt" not in result.stderr.lower()
    assert provider_log.exists()


@pytest.mark.parametrize("budget_case", ["maximum-window", "elapsed-window"])
def test_conservative_ceiling_budget_rejects_before_any_provider_action(
    tmp_path: Path, budget_case: str,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    provider_log = tmp_path / "provider.log"
    _write_executable(
        fake_bin / "nebius",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_PROVIDER_LOG\"\nexit 97\n",
    )
    _write_executable(fake_bin / "ssh", "#!/usr/bin/env bash\nexit 97\n")
    env = _wrapper_env(tmp_path, fake_bin, f"n15-budget-{budget_case}")
    env["FAKE_PROVIDER_LOG"] = str(provider_log)
    pipeline = Path(env["LEHOME_N15_PIPELINE_ROOT"])
    if budget_case == "maximum-window":
        env["LEHOME_N15_MAX_BUDGET_USD"] = "71"
    else:
        module = _load_cli()
        plan = pipeline / "lifecycle-plan.json"
        assert module.main([
            "lifecycle-plan", "--run-id", env["LEHOME_N15_RUN_ID"],
            "--repository", env["LEHOME_N15_PUBLIC_HF_REPOSITORY"],
            "--remote-pipeline-root", env["LEHOME_N15_REMOTE_PIPELINE_ROOT"],
            "--budget-usd", "100", "--estimated-cost-usd", "72",
            "--output", str(plan),
        ]) == 0
        plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
        started = int(time.time()) - 34 * 3600
        deadline = {
            "schema_version": 1,
            "kind": "lehome_public_n15_paid_deadline_v1",
            "run_id": env["LEHOME_N15_RUN_ID"],
            "lifecycle_plan_sha256": plan_sha,
            "started_unix_seconds": started,
            "deadline_unix_seconds": started + 86400,
        }
        deadline_path = pipeline / "paid-deadline.json"
        deadline_path.write_text(
            json.dumps(deadline, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
        deadline_path.chmod(0o444)

    result = subprocess.run(
        ["bash", str(WRAPPER)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode != 0
    assert "conservative provider budget" in result.stderr.lower()
    assert not provider_log.exists()


def test_stale_train_deadline_does_not_block_actual_next_unfinished_stage(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, "n15-next-unfinished-stage")
    pipeline = Path(env["LEHOME_N15_PIPELINE_ROOT"])
    stale_train = pipeline / "stage-train-deadline.json"
    stale_train.write_text("expired-train-deadline\n", encoding="ascii")
    stale_train.chmod(0o444)
    (pipeline / "harvest-terminal.json").write_text("{}\n", encoding="ascii")
    trace = tmp_path / "paid-stages.log"
    harness = r'''
source "$WRAPPER_PATH"
remote_file_exists() {
  case "$1" in
    "$TRAINING_IDENTITY_RECEIPT"|"$TRAINING_PUBLICATION_RECEIPT") return 0 ;;
    "$FOCUSED_PROMOTION_RECEIPT") return 1 ;;
  esac
  return 1
}
run_paid_stage() { printf '%s\n' "$1" >> "$FAKE_TRACE"; }
record_host_stage_completion() { :; }
advance_paid_stage_admission_from_host_seals() { PRESTART_ADMITTED_STAGE=""; }
verify_remote_training_chain() { return 0; }
verify_remote_training_publication() { return 0; }
verify_remote_focused_chain() { return 0; }
python3() { return 0; }
stop_exact_vm() { return 0; }
run_pipeline_after_runtime
'''

    result = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={**env, "WRAPPER_PATH": str(WRAPPER), "FAKE_TRACE": str(trace)},
        text=True, capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert trace.read_text(encoding="ascii").splitlines() == ["focused_gate"]
    assert stale_train.read_text(encoding="ascii") == "expired-train-deadline\n"


@pytest.mark.parametrize("deadline_kind", ["malformed", "expired"])
def test_main_rejects_actual_next_stage_deadline_before_any_provider_call(
    tmp_path: Path, deadline_kind: str,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    provider_log = tmp_path / "provider.log"
    _write_executable(
        fake_bin / "nebius",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_PROVIDER_LOG\"\nexit 97\n",
    )
    _write_executable(fake_bin / "ssh", "#!/usr/bin/env bash\nexit 97\n")
    env = _wrapper_env(tmp_path, fake_bin, f"n15-focused-deadline-{deadline_kind}")
    env["FAKE_PROVIDER_LOG"] = str(provider_log)
    pipeline = Path(env["LEHOME_N15_PIPELINE_ROOT"])
    module = _load_cli()
    plan = pipeline / "lifecycle-plan.json"
    assert module.main([
        "lifecycle-plan", "--run-id", env["LEHOME_N15_RUN_ID"],
        "--repository", env["LEHOME_N15_PUBLIC_HF_REPOSITORY"],
        "--remote-pipeline-root", env["LEHOME_N15_REMOTE_PIPELINE_ROOT"],
        "--budget-usd", "100", "--estimated-cost-usd", "72",
        "--output", str(plan),
    ]) == 0
    plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    started = int(time.time())
    paid = {
        "schema_version": 1,
        "kind": "lehome_public_n15_paid_deadline_v1",
        "run_id": env["LEHOME_N15_RUN_ID"],
        "lifecycle_plan_sha256": plan_sha,
        "started_unix_seconds": started,
        "deadline_unix_seconds": started + 86400,
    }
    paid_path = pipeline / "paid-deadline.json"
    paid_path.write_text(
        json.dumps(paid, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    paid_path.chmod(0o444)
    _write_host_stage_seal(
        pipeline,
        run_id=env["LEHOME_N15_RUN_ID"],
        stage="training",
        remote_paths=[
            f'{env["LEHOME_N15_TRAINING_ROOT"]}/training-identity.json',
            f'{env["LEHOME_N15_TRAINING_ROOT"]}/training-publication.json',
        ],
    )
    stage_path = pipeline / "stage-focused_gate-deadline.json"
    if deadline_kind == "malformed":
        stage = {}
    else:
        stage_started = started - 20000
        stage = {
            "schema_version": 1,
            "kind": "lehome_public_n15_stage_deadline_v1",
            "run_id": env["LEHOME_N15_RUN_ID"],
            "stage": "focused_gate",
            "lifecycle_plan_sha256": plan_sha,
            "started_unix_seconds": stage_started,
            "deadline_unix_seconds": stage_started + 14400,
        }
    stage_path.write_text(
        json.dumps(stage, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    stage_path.chmod(0o444)

    result = subprocess.run(
        ["bash", str(WRAPPER)], cwd=ROOT, env=env,
        text=True, capture_output=True,
    )

    assert result.returncode != 0
    assert "focused" in result.stderr.lower() and "deadline" in result.stderr.lower()
    assert not provider_log.exists()


@pytest.mark.parametrize("provider_state", ["STOPPED", "RUNNING"])
def test_main_processes_verified_terminal_before_expired_aggregate_deadline(
    tmp_path: Path, provider_state: str,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, f"n15-terminal-expired-{provider_state.lower()}")
    pipeline = Path(env["LEHOME_N15_PIPELINE_ROOT"])
    trace = tmp_path / "terminal-order.log"
    module = _load_cli()
    plan = pipeline / "lifecycle-plan.json"
    assert module.main([
        "lifecycle-plan", "--run-id", env["LEHOME_N15_RUN_ID"],
        "--repository", env["LEHOME_N15_PUBLIC_HF_REPOSITORY"],
        "--remote-pipeline-root", env["LEHOME_N15_REMOTE_PIPELINE_ROOT"],
        "--budget-usd", "100", "--estimated-cost-usd", "72",
        "--output", str(plan),
    ]) == 0
    plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    paid = {
        "schema_version": 1,
        "kind": "lehome_public_n15_paid_deadline_v1",
        "run_id": env["LEHOME_N15_RUN_ID"],
        "lifecycle_plan_sha256": plan_sha,
        "started_unix_seconds": 1,
        "deadline_unix_seconds": 86401,
    }
    paid_path = pipeline / "paid-deadline.json"
    paid_path.write_text(
        json.dumps(paid, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    paid_path.chmod(0o444)
    (pipeline / "harvest-terminal.json").write_text("{}\n", encoding="ascii")
    harness = r'''
source "$WRAPPER_PATH"
acquire_controller_lock() { return 0; }
release_controller_lock() { :; }
resolve_terminal_provider_receipt() { printf '%s\n' "$PROVIDER_STOPPED_RECEIPT"; }
python3() {
  if [[ "$*" == *"verify-terminal"* ]]; then return 0; fi
  command python3 "$@"
}
capture_exact_provider_state() {
  printf 'readback:%s\n' "$1" >> "$FAKE_TRACE"
  [[ "$FAKE_PROVIDER_STATE" == "$1" ]]
}
stop_exact_vm() { printf 'stop\n' >> "$FAKE_TRACE"; return 0; }
nebius() { printf 'paid-provider-command:%s\n' "$*" >> "$FAKE_TRACE"; return 97; }
main
'''

    result = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={
            **env,
            "WRAPPER_PATH": str(WRAPPER),
            "FAKE_TRACE": str(trace),
            "FAKE_PROVIDER_STATE": provider_state,
        },
        text=True, capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    expected = ["readback:STOPPED"] if provider_state == "STOPPED" else ["readback:STOPPED", "stop"]
    assert trace.read_text(encoding="ascii").splitlines() == expected


@pytest.mark.parametrize("completed_stage", ["training", "focused"])
def test_remote_completed_stage_is_adopted_and_existing_seal_rechecks_bytes(
    tmp_path: Path, completed_stage: str,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, f"n15-adopt-{completed_stage}")
    remote_receipts = tmp_path / "remote-receipts"; remote_receipts.mkdir()
    sources = {
        "training-identity.json": _canonical_json_bytes({"receipt": "training-identity"}),
        "training-publication.json": _canonical_json_bytes({"receipt": "training-publication"}),
        "comparison-receipt.json": _canonical_json_bytes({"receipt": "comparison"}),
        "publication.json": _canonical_json_bytes({"receipt": "focused-publication"}),
        "promotion.json": _canonical_json_bytes({"receipt": "promotion"}),
    }
    for name, payload in sources.items():
        if completed_stage == "training" and name in {
            "comparison-receipt.json", "publication.json", "promotion.json",
        }:
            continue
        (remote_receipts / name).write_bytes(payload)
    harness = r'''
source "$WRAPPER_PATH"
remote_file_exists() { [[ -f "$FAKE_REMOTE_RECEIPTS/${1##*/}" ]]; }
verify_remote_training_chain() { return 0; }
verify_remote_training_publication() { return 0; }
verify_remote_focused_chain() { return 0; }
fetch_remote_immutable() {
  cp "$FAKE_REMOTE_RECEIPTS/${1##*/}" "$2"
  chmod 0444 "$2"
}
reconcile_remote_stage_seals
'''
    command = ["bash", "-c", harness]
    command_env = {
        **env,
        "WRAPPER_PATH": str(WRAPPER),
        "FAKE_REMOTE_RECEIPTS": str(remote_receipts),
    }

    adopted = subprocess.run(
        command, cwd=ROOT, env=command_env, text=True, capture_output=True
    )

    assert adopted.returncode == 0, adopted.stderr
    training_seal = Path(env["LEHOME_N15_PIPELINE_ROOT"]) / "host-stage-training-complete.json"
    focused_seal = Path(env["LEHOME_N15_PIPELINE_ROOT"]) / "host-stage-focused-complete.json"
    assert training_seal.is_file() and not training_seal.is_symlink()
    assert focused_seal.exists() is (completed_stage == "focused")
    seal = json.loads(
        (focused_seal if completed_stage == "focused" else training_seal).read_text(
            encoding="ascii"
        )
    )
    for item in seal["remote_receipts"]:
        assert item["sha256"] == hashlib.sha256(
            (remote_receipts / Path(item["path"]).name).read_bytes()
        ).hexdigest()

    mutation = "promotion.json" if completed_stage == "focused" else "training-identity.json"
    (remote_receipts / mutation).write_bytes(_canonical_json_bytes({"tampered": True}))
    rejected = subprocess.run(
        command, cwd=ROOT, env=command_env, text=True, capture_output=True
    )
    assert rejected.returncode != 0
    assert "seal mismatch" in rejected.stderr.lower()


def test_training_identity_without_publication_is_verified_then_published_and_sealed(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, "n15-identity-only-restart")
    state = tmp_path / "publication-created"
    trace = tmp_path / "trace"
    harness = r'''
source "$WRAPPER_PATH"
remote_file_exists() {
  [[ "$1" == "$TRAINING_IDENTITY_RECEIPT" ]] && return 0
  [[ "$1" == "$TRAINING_PUBLICATION_RECEIPT" && -f "$PUBLICATION_STATE" ]] && return 0
  return 1
}
verify_remote_training_chain() { printf 'verify-identity\n' >> "$TRACE"; }
publish_training_readback() { printf 'publish\n' >> "$TRACE"; touch "$PUBLICATION_STATE"; }
verify_remote_training_publication() { test -f "$PUBLICATION_STATE"; printf 'verify-publication\n' >> "$TRACE"; }
record_host_stage_completion() { printf 'seal:%s\n' "$1" >> "$TRACE"; }
run_paid_stage() { printf 'paid:%s\n' "$1" >> "$TRACE"; return 88; }
reconcile_remote_stage_seals
run_pipeline_after_runtime
'''
    result = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={
            **env, "WRAPPER_PATH": str(WRAPPER), "TRACE": str(trace),
            "PUBLICATION_STATE": str(state),
        },
        text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert trace.exists(), result.stderr
    lines = trace.read_text(encoding="ascii").splitlines()
    assert "paid:train" not in lines
    assert "publish" in lines, (lines, result.stderr)
    assert lines[:5] == [
        "verify-identity", "verify-identity", "publish",
        "verify-publication", "seal:training",
    ]

    state.unlink()
    host_seal = Path(env["LEHOME_N15_PIPELINE_ROOT"]) / "host-stage-training-complete.json"
    host_seal.write_text("{}\n", encoding="ascii")
    rejected = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={
            **env, "WRAPPER_PATH": str(WRAPPER), "TRACE": str(trace),
            "PUBLICATION_STATE": str(state),
        },
        text=True, capture_output=True,
    )
    assert rejected.returncode != 0
    assert "seal lacks verified publication" in rejected.stderr


def test_main_explicit_resume_adopts_identity_only_completion_before_publication(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, "n15-main-identity-only-resume")
    env.update({
        "LEHOME_N15_RESUME_PARTIAL": "1",
        "LEHOME_N15_RESUME_STEP": "1500",
        "LEHOME_N15_RESUME_ATTEMPT_ID": "attempt-identity-only",
        "LEHOME_N15_PUBLIC_SOURCE_ROOT": "/mnt/source",
        "LEHOME_N15_RESUME_CHECKPOINT": "/mnt/source/outputs/train/groot_four_types_merged_batch64_lr2e-4/checkpoints/001500",
    })
    trace = tmp_path / "trace"; publication = tmp_path / "publication"
    focused = tmp_path / "focused"; harvest = tmp_path / "harvest"
    harness = r'''
source "$WRAPPER_PATH"
verify_conservative_task_budget() { :; }
initialize_deadline() { echo "$(( $(date +%s) + 3600 ))"; }
initialize_stage_deadline() { echo "$(( $(date +%s) + 3600 ))"; }
capture_exact_provider_state() { : > "$2"; return 0; }
nebius() { printf 'provider-start\n' >> "$TRACE"; }
wait_for_ssh_readiness() { :; }
wait_for_remote_runtime() { :; }
remote_file_exists() {
  [[ "$1" == "$TRAINING_IDENTITY_RECEIPT" ]] && return 0
  [[ "$1" == "$TRAINING_PUBLICATION_RECEIPT" && -f "$PUBLICATION" ]] && return 0
  [[ "$1" == "$FOCUSED_PROMOTION_RECEIPT" && -f "$FOCUSED_STATE" ]] && return 0
  [[ "$1" == "$REMOTE_PIPELINE_ROOT/harvest.publication.json" && -f "$HARVEST_STATE" ]] && return 0
  return 1
}
verify_remote_training_chain() { printf 'verify-identity\n' >> "$TRACE"; }
publish_training_readback() { printf 'publish\n' >> "$TRACE"; touch "$PUBLICATION"; }
verify_remote_training_publication() { test -f "$PUBLICATION"; printf 'verify-publication\n' >> "$TRACE"; }
focused_stage() { printf 'paid:focused_gate\n' >> "$TRACE"; touch "$FOCUSED_STATE"; }
harvest_stage() { printf 'paid:harvest\n' >> "$TRACE"; touch "$HARVEST_STATE"; }
train_stage() { printf 'paid:train\n' >> "$TRACE"; return 91; }
verify_remote_focused_chain() { test -f "$FOCUSED_STATE"; }
verify_remote_harvest_chain() { test -f "$HARVEST_STATE"; }
fetch_remote_immutable() { printf '{"remote":"%s"}\n' "$1" > "$2"; chmod 0444 "$2"; }
finalize_host_harvest_terminal() { touch "$HARVEST_TERMINAL_RECEIPT"; }
stop_exact_vm() { printf 'stop\n' >> "$TRACE"; }
main
'''
    result = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={
            **env, "WRAPPER_PATH": str(WRAPPER), "TRACE": str(trace),
            "PUBLICATION": str(publication), "FOCUSED_STATE": str(focused),
            "HARVEST_STATE": str(harvest),
        }, text=True, capture_output=True,
    )
    assert result.returncode == 0, (
        result.stderr, trace.read_text(encoding="ascii") if trace.exists() else "no trace"
    )
    assert trace.exists(), result.stderr
    lines = trace.read_text(encoding="ascii").splitlines()
    assert "paid:train" not in lines
    assert "publish" in lines, (lines, result.stderr)
    assert lines.count("paid:focused_gate") == 1
    assert lines.count("paid:harvest") == 1
    assert lines.index("publish") < lines.index("verify-publication") < lines.index("paid:focused_gate")
    assert lines.index("paid:focused_gate") < lines.index("paid:harvest")
    pipeline = Path(env["LEHOME_N15_PIPELINE_ROOT"])
    assert (pipeline / "host-stage-training-complete.json").is_file()
    assert (pipeline / "host-stage-focused-complete.json").is_file()


def test_completed_terminal_chain_is_processed_before_stale_stage_deadlines(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    provider_log = tmp_path / "provider.log"
    _write_executable(
        fake_bin / "nebius",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_PROVIDER_LOG\"\nexit 97\n",
    )
    _write_executable(fake_bin / "ssh", "#!/usr/bin/env bash\nexit 97\n")
    env = _wrapper_env(tmp_path, fake_bin, "n15-terminal-before-stage-deadline")
    env["FAKE_PROVIDER_LOG"] = str(provider_log)
    pipeline = Path(env["LEHOME_N15_PIPELINE_ROOT"])
    (pipeline / "harvest-terminal.json").write_text("{}\n", encoding="ascii")
    trace = tmp_path / "stage-deadlines.log"
    harness = r'''
source "$WRAPPER_PATH"
initialize_deadline() { echo "$(( $(date +%s) + 3600 ))"; }
initialize_stage_deadline() { printf 'stage:%s\n' "$1" >> "$FAKE_TRACE"; echo 1; }
acquire_controller_lock() { return 0; }
release_controller_lock() { :; }
python3() { return 0; }
capture_exact_provider_state() { test "$1" = STOPPED; : > "$2"; }
main
'''

    result = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={**env, "WRAPPER_PATH": str(WRAPPER), "FAKE_TRACE": str(trace)},
        text=True, capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert not trace.exists()
    assert not provider_log.exists()


def test_terminal_receipt_is_recomputed_and_requires_exact_immutable_bytes(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, "n15-terminal-canonical")
    pipeline = Path(env["LEHOME_N15_PIPELINE_ROOT"])
    provider = pipeline / "provider-stopped.json"
    provider.write_bytes(_canonical_json_bytes({"captured_unix_seconds": 1_788_150_400}))
    provider.chmod(0o444)
    terminal_value = {
        "terminal": True,
        "provider_receipt_name": provider.name,
        "provider_receipt_sha256": hashlib.sha256(provider.read_bytes()).hexdigest(),
        "provider_captured_unix_seconds": 1_788_150_400,
    }
    terminal = pipeline / "harvest-terminal.json"
    terminal.write_bytes(_canonical_json_bytes(terminal_value)); terminal.chmod(0o444)
    expected_terminal = tmp_path / "expected-terminal.json"
    expected_terminal.write_bytes(terminal.read_bytes())
    harness = r'''
source "$WRAPPER_PATH"
python3() {
  if [[ "$1" == "$HARVEST_BUILDER" ]]; then
    shift
    while [[ "$1" != --output ]]; do shift; done
    cp "$EXPECTED_TERMINAL" "$2"
    chmod 0444 "$2"
  else
    command python3 "$@"
  fi
}
verify_host_harvest_terminal
'''

    def verify() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", harness], cwd=ROOT,
            env={
                **env,
                "WRAPPER_PATH": str(WRAPPER),
                "EXPECTED_TERMINAL": str(expected_terminal),
            },
            text=True, capture_output=True,
        )

    assert verify().returncode == 0
    terminal.chmod(0o644)
    assert verify().returncode != 0
    terminal_value["terminal"] = False
    terminal.write_bytes(_canonical_json_bytes(terminal_value)); terminal.chmod(0o444)
    assert verify().returncode != 0


def test_terminal_restart_uses_exact_bound_provider_stop_receipt_and_rejects_substitution(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, "n15-terminal-provider-binding")
    pipeline = Path(env["LEHOME_N15_PIPELINE_ROOT"])
    canonical_provider = pipeline / "provider-stopped.json"
    final_provider = pipeline / "provider-stopped-n15-terminal-provider-binding-1788150401.json"
    old_provider_value = {"captured_unix_seconds": 1_788_150_400, "marker": "old"}
    final_provider_value = {"captured_unix_seconds": 1_788_150_401, "marker": "final"}
    canonical_provider.write_bytes(_canonical_json_bytes(old_provider_value))
    final_provider.write_bytes(_canonical_json_bytes(final_provider_value))
    canonical_provider.chmod(0o444); final_provider.chmod(0o444)
    terminal_value = {
        "terminal": True,
        "provider_receipt_name": final_provider.name,
        "provider_receipt_sha256": hashlib.sha256(final_provider.read_bytes()).hexdigest(),
        "provider_captured_unix_seconds": 1_788_150_401,
    }
    terminal = pipeline / "harvest-terminal.json"
    terminal.write_bytes(_canonical_json_bytes(terminal_value)); terminal.chmod(0o444)
    expected_terminal = tmp_path / "expected-terminal.json"
    expected_terminal.write_bytes(terminal.read_bytes())
    trace = tmp_path / "provider-paths"
    harness = r'''
source "$WRAPPER_PATH"
python3() {
  if [[ "$1" == "$HARVEST_BUILDER" ]]; then
    shift
    provider=""; output=""
    while (( $# )); do
      case "$1" in
        --provider-receipt) provider="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    printf '%s\n' "$provider" >> "$TRACE"
    [[ "$provider" == "$EXPECTED_PROVIDER" ]] || return 91
    cp "$EXPECTED_TERMINAL" "$output"
    chmod 0444 "$output"
  else
    command python3 "$@"
  fi
}
verify_host_harvest_terminal
'''

    def verify() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", harness], cwd=ROOT,
            env={
                **env,
                "WRAPPER_PATH": str(WRAPPER),
                "EXPECTED_PROVIDER": str(final_provider),
                "EXPECTED_TERMINAL": str(expected_terminal),
                "TRACE": str(trace),
            },
            text=True, capture_output=True,
        )

    accepted = verify()
    assert accepted.returncode == 0, accepted.stderr
    assert trace.read_text(encoding="ascii").splitlines() == [str(final_provider)]

    final_provider.chmod(0o644)
    final_provider.write_bytes(_canonical_json_bytes(old_provider_value))
    final_provider.chmod(0o444)
    rejected = verify()
    assert rejected.returncode != 0
    assert trace.read_text(encoding="ascii").splitlines() == [str(final_provider)]


def test_terminal_stop_failure_retains_singleton_until_stopped(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, "n15-terminal-stop-lock")
    terminal = Path(env["LEHOME_N15_PIPELINE_ROOT"]) / "harvest-terminal.json"
    terminal.write_text("{}\n", encoding="ascii")
    allow_stop = tmp_path / "allow-stop"; trace = tmp_path / "trace"
    harness = r'''
source "$WRAPPER_PATH"
verify_host_harvest_terminal() { return 0; }
capture_exact_provider_state() { return 1; }
stop_exact_vm() {
  [[ -f "$ALLOW_STOP" ]] || { printf 'stop-failed\n' >> "$TRACE"; return 1; }
  printf 'stopped\n' >> "$TRACE"
}
main
'''
    process = subprocess.Popen(
        ["bash", "-c", harness], cwd=ROOT,
        env={
            **env, "WRAPPER_PATH": str(WRAPPER), "ALLOW_STOP": str(allow_stop),
            "TRACE": str(trace),
        }, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    lock = _controller_lock_path(env)
    try:
        deadline = time.monotonic() + 5
        while (not trace.exists() or not lock.exists()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert lock.exists() and "stop-failed" in trace.read_text()
        contender = subprocess.run(
            ["bash", "-c", 'source "$WRAPPER_PATH"; acquire_controller_lock'],
            cwd=ROOT, env={**env, "WRAPPER_PATH": str(WRAPPER)},
            text=True, capture_output=True,
        )
        assert contender.returncode != 0
        allow_stop.touch()
        process.communicate(timeout=8)
        assert process.returncode != 0
        assert trace.read_text().splitlines()[-1] == "stopped"
        assert not lock.exists()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL); process.communicate(timeout=3)


def test_atomic_controller_singleton_rejects_second_live_controller_without_stopping_vm(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    state = tmp_path / "state"; state.write_text("STOPPED", encoding="ascii")
    trace = tmp_path / "trace"
    provider = {
        "metadata": {"id": "computeinstance-u00t6xfqhadrcmssa2", "name": "lehome-rollout"},
        "status": {"state": "STATE"},
        "spec": {"boot_disk": {"managed_disk": {"spec": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}}}, "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}], "preemptible": {"on_preemption": "STOP"}},
    }
    (fake_bin / "nebius").write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\n"
        "state=Path(os.environ['FAKE_STATE']); trace=Path(os.environ['FAKE_TRACE']); command=sys.argv[1:4]\n"
        "if command == ['compute','instance','start']: trace.open('a').write('start\\n'); state.write_text('RUNNING')\n"
        "elif command == ['compute','instance','stop']: trace.open('a').write('stop\\n'); state.write_text('STOPPED')\n"
        "else:\n value=" + repr(provider) + "; value['status']['state']=state.read_text().strip(); print(json.dumps(value))\n",
        encoding="utf-8",
    )
    (fake_bin / "ssh").write_text(
        "#!/usr/bin/env bash\nwhile :; do /bin/sleep 1; done\n", encoding="utf-8"
    )
    for command in (fake_bin / "nebius", fake_bin / "ssh"): command.chmod(0o755)
    env = _wrapper_env(tmp_path, fake_bin, "n15-singleton")
    env.update({"FAKE_STATE": str(state), "FAKE_TRACE": str(trace)})
    first = subprocess.Popen(
        ["bash", str(WRAPPER)], cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while (not trace.exists() or "start" not in trace.read_text()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert trace.exists() and trace.read_text().splitlines() == ["start"]
        lock_path = _controller_lock_path(env)
        owner = json.loads(lock_path.read_text(encoding="ascii"))
        assert owner == {
            "schema_version": 1,
            "kind": "lehome_public_n15_controller_lock_v1",
            "run_id": "n15-singleton",
            "controller_pid": first.pid,
            "acquisition_helper_pid": owner["acquisition_helper_pid"],
            "script_path": str(WRAPPER.resolve()),
        }
        with pytest.raises(ProcessLookupError):
            os.kill(owner["acquisition_helper_pid"], 0)

        second = subprocess.run(
            ["bash", str(WRAPPER)], cwd=ROOT, env=env, text=True,
            capture_output=True, timeout=5,
        )
        assert second.returncode != 0
        assert "controller" in second.stderr.lower()
        assert trace.read_text().splitlines() == ["start"]
        assert lock_path.is_file()
    finally:
        if first.poll() is None:
            os.killpg(first.pid, signal.SIGTERM)
        first.communicate(timeout=5)
    assert trace.read_text().splitlines() == ["start", "stop"]


def test_controller_singleton_contends_across_run_ids_and_pipeline_roots(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    first_root = tmp_path / "first"; first_root.mkdir()
    second_root = tmp_path / "second"; second_root.mkdir()
    first_env = _wrapper_env(first_root, fake_bin, "n15-lock-first")
    second_env = _wrapper_env(second_root, fake_bin, "n15-lock-second")
    second_env["LEHOME_N15_CONTROLLER_LOCK_TEST_ROOT"] = first_env[
        "LEHOME_N15_CONTROLLER_LOCK_TEST_ROOT"
    ]
    holder_script = 'source "$WRAPPER_PATH"; acquire_controller_lock; while :; do /bin/sleep 1; done'
    first = subprocess.Popen(
        ["bash", "-c", holder_script], cwd=ROOT,
        env={**first_env, "WRAPPER_PATH": str(WRAPPER)},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    lock_path = _controller_lock_path(first_env)
    try:
        deadline = time.monotonic() + 5
        while not lock_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert lock_path.is_file()
        second = subprocess.run(
            ["bash", "-c", 'source "$WRAPPER_PATH"; acquire_controller_lock'],
            cwd=ROOT, env={**second_env, "WRAPPER_PATH": str(WRAPPER)},
            capture_output=True, timeout=5,
        )
        assert second.returncode != 0
        assert json.loads(lock_path.read_text(encoding="ascii"))["run_id"] == "n15-lock-first"
    finally:
        if first.poll() is None:
            os.kill(first.pid, signal.SIGTERM)
        try:
            first.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(first.pid, signal.SIGKILL)
            first.communicate(timeout=5)


def test_abrupt_controller_death_leaves_safe_stale_file_and_releases_flock(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, "n15-orphan-lock")
    process = subprocess.Popen(
        ["bash", "-c", 'source "$WRAPPER_PATH"; acquire_controller_lock; while :; do /bin/sleep 1; done'],
        cwd=ROOT, env={**env, "WRAPPER_PATH": str(WRAPPER)},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    lock_path = _controller_lock_path(env)
    try:
        deadline = time.monotonic() + 5
        while not lock_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert lock_path.is_file()
        os.kill(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)
        assert lock_path.is_file()
        retry = subprocess.run(
            [
                "bash", "-c",
                'source "$WRAPPER_PATH"; acquire_controller_lock; release_controller_lock',
            ],
            cwd=ROOT, env={**env, "WRAPPER_PATH": str(WRAPPER)},
            text=True, capture_output=True, timeout=5,
        )
        assert retry.returncode == 0, retry.stderr
        assert not lock_path.exists()
    finally:
        if process.poll() is None:
            os.kill(process.pid, signal.SIGKILL)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)


def test_dead_acquisition_helper_does_not_release_controller_owned_flock(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    trace = tmp_path / "trace.log"
    env = _wrapper_env(tmp_path, fake_bin, "n15-holder-death")
    harness = r'''
source "$WRAPPER_PATH"
stop_exact_vm() { printf 'stop\n' >> "$FAKE_TRACE"; }
acquire_controller_lock
PROVIDER_CLEANUP_REQUIRED=1
trap controller_cleanup EXIT
printf 'working\n' >> "$FAKE_TRACE"
while :; do /bin/sleep 1; done
'''
    process = subprocess.Popen(
        ["bash", "-c", harness], cwd=ROOT,
        env={**env, "WRAPPER_PATH": str(WRAPPER), "FAKE_TRACE": str(trace)},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    lock_path = _controller_lock_path(env)
    try:
        deadline = time.monotonic() + 5
        while (not lock_path.is_file() or not trace.exists()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert lock_path.is_file() and trace.read_text().splitlines() == ["working"]
        owner = json.loads(lock_path.read_text(encoding="ascii"))
        with pytest.raises(ProcessLookupError):
            os.kill(owner["acquisition_helper_pid"], 0)
        contender = subprocess.run(
            ["bash", "-c", 'source "$WRAPPER_PATH"; acquire_controller_lock'],
            cwd=ROOT, env={**env, "WRAPPER_PATH": str(WRAPPER)},
            text=True, capture_output=True, timeout=5,
        )
        assert contender.returncode != 0
        assert process.poll() is None
        assert trace.read_text().splitlines() == ["working"]
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.communicate(timeout=3)

    assert trace.read_text().splitlines() == ["working", "stop"]
    assert not lock_path.exists()


def test_interrupted_paid_stage_is_reaped_and_stop_is_confirmed_before_unlock(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, "n15-paid-interrupt-cleanup")
    trace = tmp_path / "trace"
    stage_pid = tmp_path / "stage-pid"
    allow_stop = tmp_path / "allow-stop"
    harness = r'''
source "$WRAPPER_PATH"
initialize_deadline() { echo "$(( $(date +%s) + 60 ))"; }
initialize_stage_deadline() { echo "$(( $(date +%s) + 60 ))"; }
train_stage() { printf '%s\n' "$$" > "$STAGE_PID"; while :; do /bin/sleep 1; done; }
stop_exact_vm() {
  if [[ ! -e "$ALLOW_STOP" ]]; then printf 'stop-failed\n' >> "$TRACE"; return 1; fi
  printf 'stopped\n' >> "$TRACE"; return 0
}
acquire_controller_lock
PROVIDER_CLEANUP_REQUIRED=1
trap controller_cleanup EXIT
run_paid_stage train 60 train_stage
'''
    process = subprocess.Popen(
        ["bash", "-c", harness], cwd=ROOT,
        env={
            **env, "WRAPPER_PATH": str(WRAPPER), "TRACE": str(trace),
            "STAGE_PID": str(stage_pid), "ALLOW_STOP": str(allow_stop),
        },
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    lock_path = _controller_lock_path(env)
    try:
        deadline = time.monotonic() + 5
        while (not stage_pid.exists() or not lock_path.exists()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert stage_pid.exists() and lock_path.exists()
        os.kill(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while (not trace.exists() or "stop-failed" not in trace.read_text()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert trace.exists() and "stop-failed" in trace.read_text()
        assert lock_path.exists()
        contender = subprocess.run(
            ["bash", "-c", 'source "$WRAPPER_PATH"; acquire_controller_lock'],
            cwd=ROOT, env={**env, "WRAPPER_PATH": str(WRAPPER)},
            text=True, capture_output=True, timeout=5,
        )
        assert contender.returncode != 0
        allow_stop.touch()
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode != 0, (stdout, stderr)
        assert "stop confirmation failed" in stderr
        with pytest.raises(ProcessLookupError):
            os.kill(int(stage_pid.read_text()), 0)
        assert trace.read_text().splitlines()[-1] == "stopped"
        assert not lock_path.exists()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=3)


def test_interrupt_in_post_setsid_handshake_reaps_descendant_before_stop_and_unlock(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, "n15-paid-handshake-interrupt")
    window = tmp_path / "setsid-window"
    descendant_pid = tmp_path / "descendant-pid"
    descendant_stopped = tmp_path / "descendant-stopped"
    stage_entered = tmp_path / "stage-entered"
    stage_pid = tmp_path / "stage-pid"
    trace = tmp_path / "trace"
    harness = r'''
source "$WRAPPER_PATH"
initialize_deadline() { echo "$(( $(date +%s) + 60 ))"; }
initialize_stage_deadline() { echo "$(( $(date +%s) + 60 ))"; }
train_stage() { printf '%s\n' "$$" > "$STAGE_PID"; touch "$STAGE_ENTERED"; while :; do /bin/sleep 1; done; }
stop_exact_vm() {
  child="$(cat "$DESCENDANT_PID")"
  if kill -0 "$child" 2>/dev/null; then
    printf 'stop-before-descendant-reap\n' >> "$TRACE"
    return 1
  fi
  test -f "$DESCENDANT_STOPPED"
  printf 'stopped\n' >> "$TRACE"
}
acquire_controller_lock
PROVIDER_CLEANUP_REQUIRED=1
trap controller_cleanup EXIT
run_paid_stage train 60 train_stage
'''
    process = subprocess.Popen(
        ["bash", "-c", harness], cwd=ROOT,
        env={
            **env, "WRAPPER_PATH": str(WRAPPER), "TRACE": str(trace),
            "STAGE_ENTERED": str(stage_entered),
            "STAGE_PID": str(stage_pid),
            "DESCENDANT_PID": str(descendant_pid),
            "DESCENDANT_STOPPED": str(descendant_stopped),
            "LEHOME_N15_TEST_SETSID_WINDOW": str(window),
        },
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    lock_path = _controller_lock_path(env)
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if window.exists() and descendant_pid.exists() and lock_path.exists():
                raw_pid = descendant_pid.read_text(encoding="ascii").strip()
                if raw_pid.isdecimal():
                    child_pid = int(raw_pid)
                    break
            time.sleep(0.02)
        assert child_pid is not None
        assert not stage_entered.exists()

        os.kill(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode != 0, (stdout, stderr)
        assert descendant_stopped.is_file()
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert not stage_entered.exists()
        assert trace.read_text(encoding="utf-8").splitlines() == ["stopped"]
        assert not lock_path.exists()
    finally:
        if process.poll() is None:
            os.kill(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                for ready_file in Path(env["LEHOME_N15_PIPELINE_ROOT"]).glob(
                    ".stage-launcher-train.*/ready"
                ):
                    ready_pid = ready_file.read_text(encoding="ascii").strip()
                    if ready_pid.isdecimal():
                        try:
                            os.killpg(int(ready_pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=3)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if stage_pid.exists():
            try:
                os.killpg(int(stage_pid.read_text(encoding="ascii")), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_stale_controller_lock_is_reclaimed_without_killing_a_process(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    trace = tmp_path / "trace"
    provider = {
        "metadata": {"id": "computeinstance-u00t6xfqhadrcmssa2", "name": "lehome-rollout"},
        "status": {"state": "STOPPED"},
        "spec": {"boot_disk": {"managed_disk": {"spec": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}}}, "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}], "preemptible": {"on_preemption": "STOP"}},
    }
    (fake_bin / "nebius").write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\n"
        "Path(os.environ['FAKE_TRACE']).open('a').write(' '.join(sys.argv[1:4])+'\\n')\n"
        "print(json.dumps(" + repr(provider) + "))\n",
        encoding="utf-8",
    )
    (fake_bin / "ssh").write_text("#!/usr/bin/env bash\nexit 97\n", encoding="utf-8")
    (fake_bin / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    for command in (fake_bin / "nebius", fake_bin / "ssh", fake_bin / "sleep"): command.chmod(0o755)
    env = _wrapper_env(tmp_path, fake_bin, "n15-stale-lock")
    env["FAKE_TRACE"] = str(trace)
    pipeline = Path(env["LEHOME_N15_PIPELINE_ROOT"])
    stale = {
        "schema_version": 1, "kind": "lehome_public_n15_controller_lock_v1",
        "run_id": env["LEHOME_N15_RUN_ID"], "controller_pid": 99999998,
        "holder_pid": 99999999, "script_path": str(WRAPPER.resolve()),
    }
    lock_path = _controller_lock_path(env)
    lock_path.parent.mkdir(mode=0o700)
    lock_path.write_text(
        json.dumps(stale, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii"
    )
    lock_path.chmod(0o600)

    result = subprocess.run(
        ["bash", str(WRAPPER)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode != 0
    assert "compute instance start" in trace.read_text()
    assert not lock_path.exists()


def test_wrapper_dispatches_preemption_resume_completion_and_downstream_exactly_once(
    tmp_path: Path,
) -> None:
    """Exercise the wrapper's real post-runtime control flow with stage doubles."""
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, "n15-wrapper-resume")
    state = tmp_path / "state"; state.mkdir()
    trace = tmp_path / "trace.log"
    env.update({
        "FAKE_STATE_DIR": str(state),
        "FAKE_TRACE": str(trace),
        "LEHOME_N15_RESUME_PARTIAL": "1",
        "LEHOME_N15_RESUME_STEP": "1500",
        "LEHOME_N15_RESUME_ATTEMPT_ID": "attempt-integration",
        "LEHOME_N15_RESUME_CHECKPOINT": (
            "/mnt/source/outputs/train/groot_four_types_merged_batch64_lr2e-4/checkpoints/001500"
        ),
        "LEHOME_N15_PUBLIC_SOURCE_ROOT": "/mnt/source",
    })
    harness = r'''
source "$WRAPPER_PATH"
remote_file_exists() {
  case "$1" in
    "$TRAINING_IDENTITY_RECEIPT") test -f "$FAKE_STATE_DIR/training-012000" ;;
    "$TRAINING_PUBLICATION_RECEIPT") test -f "$FAKE_STATE_DIR/training-publication" ;;
    "$FOCUSED_PROMOTION_RECEIPT") test -f "$FAKE_STATE_DIR/focused-promotion" ;;
    *) return 1 ;;
  esac
}
run_paid_stage() {
  printf 'paid:%s\n' "$1" >> "$FAKE_TRACE"
  case "$1" in
    train)
      test "${ALLOW_TRAIN_COMPLETION:-0}" = 1 || return 17
      printf '012000\n' > "$FAKE_STATE_DIR/training-012000"
      ;;
    focused_gate) touch "$FAKE_STATE_DIR/focused-promotion" ;;
    harvest) touch "$FAKE_STATE_DIR/harvest-1000" ;;
  esac
}
verify_remote_training_chain() { printf 'verify:training\n' >> "$FAKE_TRACE"; test "$(cat "$FAKE_STATE_DIR/training-012000")" = 012000; }
publish_training_readback() { printf 'publish:training\n' >> "$FAKE_TRACE"; touch "$FAKE_STATE_DIR/training-publication"; }
verify_remote_training_publication() { printf 'verify:training-publication\n' >> "$FAKE_TRACE"; test -f "$FAKE_STATE_DIR/training-publication"; }
verify_remote_focused_chain() { printf 'verify:focused\n' >> "$FAKE_TRACE"; test -f "$FAKE_STATE_DIR/focused-promotion"; }
verify_remote_harvest_chain() { printf 'verify:harvest-1000\n' >> "$FAKE_TRACE"; test -f "$FAKE_STATE_DIR/harvest-1000"; }
fetch_remote_immutable() { printf 'fetch\n' >> "$FAKE_TRACE"; touch "$2"; }
stop_exact_vm() { printf 'stop\n' >> "$FAKE_TRACE"; }
finalize_host_harvest_terminal() { printf 'finalize\n' >> "$FAKE_TRACE"; touch "$HARVEST_TERMINAL_RECEIPT"; }
advance_paid_stage_admission_from_host_seals() { PRESTART_ADMITTED_STAGE=""; }
python3() { return 0; }
set +e
( set -e; run_pipeline_after_runtime )
first_status=$?
set -e
test "$first_status" = 17
test ! -e "$FAKE_STATE_DIR/training-012000"
ALLOW_TRAIN_COMPLETION=1
run_pipeline_after_runtime
run_pipeline_after_runtime
'''
    result = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={**env, "WRAPPER_PATH": str(WRAPPER)}, text=True, capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    lines = trace.read_text(encoding="utf-8").splitlines()
    assert lines.count("paid:train") == 2
    assert lines.count("publish:training") == 1
    assert lines.count("paid:focused_gate") == 1
    assert lines.count("paid:harvest") == 1
    assert lines.count("verify:harvest-1000") == 1
    assert (state / "training-012000").read_text(encoding="ascii") == "012000\n"


def test_paid_stage_waits_for_setsid_handshake_before_watchdog_monitoring(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    payload = tmp_path / "launcher.py"
    _write_executable(
        fake_bin / "python3",
        """#!/usr/bin/env bash
set -euo pipefail
payload="$FAKE_LAUNCHER_PAYLOAD"
cat > "$payload"
/bin/sleep 2
exec "$REAL_PYTHON" "$@" < "$payload"
""",
    )
    _write_executable(fake_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    env = _wrapper_env(tmp_path, fake_bin, "n15-setsid-handshake")
    harness = r'''
source "$WRAPPER_PATH"
initialize_deadline() { echo "$(( $(date +%s) + 60 ))"; }
initialize_stage_deadline() { echo "$(( $(date +%s) + 1 ))"; }
train_stage() { while :; do /bin/sleep 1; done; }
run_paid_stage train 60 train_stage
'''
    process = subprocess.Popen(
        ["bash", "-c", harness], cwd=ROOT,
        env={
            **env, "WRAPPER_PATH": str(WRAPPER), "REAL_PYTHON": sys.executable,
            "FAKE_LAUNCHER_PAYLOAD": str(payload),
        },
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=7)
        status = process.returncode
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate(timeout=3)
        status = None

    assert status is not None, (stdout, stderr)
    assert status != 0
    assert "handshake" in stderr.lower() or "timeout" in stderr.lower()


def test_paid_stage_reaps_normally_exiting_setsid_leader_without_watchdog_delay(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    sleep_trace = tmp_path / "sleep-trace"
    _write_executable(
        fake_bin / "sleep",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$SLEEP_TRACE"
exec /bin/sleep "$@"
""",
    )
    env = _wrapper_env(tmp_path, fake_bin, "n15-paid-finite-success")
    process = subprocess.Popen(
        ["bash", "-c", r'''
source "$WRAPPER_PATH"
initialize_deadline() { echo "$(( $(date +%s) + 30 ))"; }
initialize_stage_deadline() { echo "$(( $(date +%s) + 30 ))"; }
train_stage() { return 0; }
trap controller_cleanup EXIT TERM
run_paid_stage train 30 train_stage
printf 'complete\n'
'''],
        cwd=ROOT,
        env={
            **env,
            "WRAPPER_PATH": str(WRAPPER),
            "SLEEP_TRACE": str(sleep_trace),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        # The controller/launcher acknowledgement is bounded at five seconds,
        # and paid-stage cleanup gets a further five-second TERM grace period.
        # Keep the harness watchdog strictly outside both bounds so scheduler
        # load cannot kill only the controller and strand its detached child.
        stdout, stderr = process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        os.kill(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=3)
        pytest.fail(f"finite paid stage exceeded lifecycle bounds: {stdout!r} {stderr!r}")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=3)

    assert process.returncode == 0, stderr
    assert stdout == "complete\n"
    # A normally exited leader may briefly be a zombie until Bash waits it;
    # the watchdog must not mistake that for a live stage and sleep a second.
    sleep_calls = (
        sleep_trace.read_text(encoding="ascii").splitlines()
        if sleep_trace.exists()
        else []
    )
    assert "1" not in sleep_calls


def test_training_publication_adopts_verified_upload_after_local_receipt_crash(
    tmp_path: Path,
) -> None:
    from lehome import n15_reproduction as reproduction

    checkout, source_receipt = _FIXTURES._materialize_source(tmp_path)
    _, _, snapshots_receipt = _FIXTURES._materialize_snapshots(tmp_path, checkout)
    contract = _FIXTURES._fixture_contract(checkout)
    verified = reproduction.verify_inputs(
        checkout=checkout, source_receipt=source_receipt,
        resolved_snapshots_receipt=snapshots_receipt,
        vm_id=contract.vm_id, disk_id=contract.disk_id, contract=contract,
    )
    training = _FIXTURES._materialize_training_output(
        tmp_path, verified=verified, contract=contract
    )
    identity = reproduction.verify_training_output(
        verified=verified, training_root=training, contract=contract
    )
    (training / "training-identity.json").write_bytes(_FIXTURES._canonical(identity))
    token = tmp_path / "hf-token"; token.write_text("test-token\n", encoding="ascii")
    fake_site = tmp_path / "fake-site"; fake_site.mkdir()
    remote_store = tmp_path / "hf-store"; remote_store.mkdir()
    upload_log = tmp_path / "uploads.log"
    download_log = tmp_path / "downloads.log"
    training_cache = tmp_path / "training-hf-cache"; training_cache.mkdir(mode=0o700)
    (fake_site / "huggingface_hub.py").write_text(
        r'''import hashlib, os, shutil
from pathlib import Path
from types import SimpleNamespace

def _forbid_whole_file_reads(path):
    raise AssertionError(f"whole-file Path.read_bytes is forbidden: {path}")
Path.read_bytes = _forbid_whole_file_reads

class EntryNotFoundError(Exception): pass
class HfApi:
    def __init__(self, token=None): self.token = token
    def list_repo_commits(self, **kwargs):
        root = Path(os.environ["FAKE_HF_STORE"])
        return [SimpleNamespace(commit_id=item.name) for item in sorted(root.iterdir(), reverse=True) if item.is_dir()]
    def list_repo_tree(self, *, path_in_repo, revision, **kwargs):
        root = Path(os.environ["FAKE_HF_STORE"]) / revision / path_in_repo
        if not root.is_dir(): raise EntryNotFoundError(path_in_repo)
        return [SimpleNamespace(rfilename=f"{path_in_repo}/{p.relative_to(root).as_posix()}", size=p.stat().st_size) for p in sorted(root.rglob("*")) if p.is_file()]
    def upload_folder(self, *, folder_path, path_in_repo, **kwargs):
        log = Path(os.environ["FAKE_HF_UPLOAD_LOG"])
        with log.open("a") as stream: stream.write(f"upload\t{folder_path}\n")
        revision = "a" * 40
        destination = Path(os.environ["FAKE_HF_STORE"]) / revision / path_in_repo
        if destination.exists(): shutil.rmtree(destination)
        source = Path(folder_path)
        mutation = os.environ.get("FAKE_HF_MUTATE_SOURCE")
        if mutation:
            Path(mutation).write_bytes(b"concurrent mutation")
        for item in source.rglob("*"):
            if item.is_file() and item.name != "training-publication.json":
                target = destination / item.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item, target)
        crash = Path(os.environ["FAKE_HF_STORE"]) / ".crashed-once"
        if not crash.exists():
            crash.touch()
            raise RuntimeError("simulated response loss after upload")
        return SimpleNamespace(oid=revision)

def hf_hub_download(*, filename, revision, cache_dir, token, **kwargs):
    cache = Path(cache_dir)
    expected = Path(os.environ["FAKE_HF_DOWNLOAD_CACHE"])
    assert token is False
    assert cache == expected
    assert cache.is_dir() and not cache.is_symlink()
    sealed = Path(os.environ["FAKE_HF_SNAPSHOT"])
    assert not sealed.is_symlink()
    with Path(os.environ["FAKE_HF_DOWNLOAD_LOG"]).open("a") as stream:
        stream.write(f"{filename}\t{cache}\n")
    source = Path(os.environ["FAKE_HF_STORE"]) / revision / filename
    blob = cache / "blobs" / hashlib.sha256(filename.encode("utf-8")).hexdigest()
    blob.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copyfile(source, blob)
    snapshot = cache / "snapshots" / revision / filename
    snapshot.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    snapshot.unlink(missing_ok=True)
    snapshot.symlink_to(os.path.relpath(blob, snapshot.parent))
    return str(snapshot)
''',
        encoding="utf-8",
    )
    # The production code imports this public exception path.
    errors = fake_site / "huggingface_hub"; errors.mkdir()
    (errors / "__init__.py").write_text((fake_site / "huggingface_hub.py").read_text())
    (errors / "utils.py").write_text(
        "from huggingface_hub import EntryNotFoundError\n", encoding="utf-8"
    )
    env = _wrapper_env(tmp_path, tmp_path, "n15-publication-adoption")
    env.update({
        "LEHOME_N15_TRAINING_ROOT": str(training),
        "LEHOME_N15_HF_TOKEN_FILE": str(token),
        "FAKE_HF_STORE": str(remote_store),
        "FAKE_HF_UPLOAD_LOG": str(upload_log),
        "FAKE_HF_DOWNLOAD_LOG": str(download_log),
        "FAKE_HF_DOWNLOAD_CACHE": str(training_cache / "training-publication-readback"),
        "FAKE_HF_SNAPSHOT": str(training.parent / f".{training.name}.publication-snapshot"),
        "LEHOME_N15_TRAINING_HF_CACHE_ROOT": str(training_cache),
        "PYTHONPATH": f"{fake_site}:{ROOT / 'source/lehome'}",
    })
    harness = 'source "$WRAPPER_PATH"; remote() { command "$@"; }; publish_training_readback'

    publication_snapshot = training.parent / f".{training.name}.publication-snapshot"
    publication_snapshot.mkdir(mode=0o700)
    (publication_snapshot / "partial-copy").write_bytes(b"interrupted snapshot")

    last = training / "checkpoints/last"
    last.unlink()
    last.symlink_to("009000")
    wrong_last = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={**env, "WRAPPER_PATH": str(WRAPPER)}, text=True, capture_output=True,
    )
    assert wrong_last.returncode != 0
    assert not publication_snapshot.exists()
    assert not upload_log.exists()
    last.unlink()
    last.symlink_to("012000")

    external = tmp_path / "external-secret"
    external.write_bytes(b"must never be read or uploaded")
    external_link = training / "external-secret-link"
    external_link.symlink_to(external)
    unsafe = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={**env, "WRAPPER_PATH": str(WRAPPER)}, text=True, capture_output=True,
    )
    assert unsafe.returncode != 0
    assert not upload_log.exists()
    assert not any(
        path.is_file() and path.read_bytes() == external.read_bytes()
        for path in remote_store.rglob("*")
    )
    external_link.unlink()

    mutable_relative = "checkpoints/012000/pretrained_model/model.safetensors"
    mutable_artifact = training / mutable_relative
    original_mutable_artifact = mutable_artifact.read_bytes()
    first = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={
            **env,
            "WRAPPER_PATH": str(WRAPPER),
            "FAKE_HF_MUTATE_SOURCE": str(mutable_artifact),
        },
        text=True, capture_output=True,
    )
    assert first.returncode != 0
    assert not (training / "training-publication.json").exists()
    uploaded_mutable_artifact = (
        remote_store / ("a" * 40) / "n15-public/n15-publication-adoption/training"
        / mutable_relative
    )
    assert uploaded_mutable_artifact.exists(), first.stderr
    assert uploaded_mutable_artifact.read_bytes() == original_mutable_artifact
    assert mutable_artifact.read_bytes() == b"concurrent mutation"
    # The immutable remote prefix now contains the sealed pre-mutation tree.
    # A retry must adopt that exact tree from the durable local snapshot; it
    # must not require restoring the mutable source or attempt a second upload.
    assert publication_snapshot.exists()
    interrupted = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={
            **env,
            "WRAPPER_PATH": str(WRAPPER),
            "LEHOME_N15_TEST_PUBLICATION_RECEIPT_FAULT": "after-temporary",
        },
        text=True, capture_output=True,
    )
    assert interrupted.returncode != 0
    assert not (training / "training-publication.json").exists()
    assert list(training.glob(".training-publication.json.*")), interrupted.stderr

    second = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={**env, "WRAPPER_PATH": str(WRAPPER)}, text=True, capture_output=True,
    )

    assert second.returncode == 0, second.stderr
    upload_rows = upload_log.read_text().splitlines()
    assert len(upload_rows) == 1
    _, upload_source = upload_rows[0].split("\t", 1)
    assert Path(upload_source) != training
    assert Path(upload_source).parent == training.parent
    receipt = json.loads((training / "training-publication.json").read_text())
    assert receipt["immutable_revision"] == "a" * 40
    assert receipt["anonymous_byte_readback_verified"] is True
    assert mutable_artifact.read_bytes() == b"concurrent mutation"
    assert not publication_snapshot.exists()
    verifier = subprocess.run(
        ["bash", "-c", 'source "$WRAPPER_PATH"; remote() { command "$@"; }; verify_remote_training_publication'],
        cwd=ROOT, env={**env, "WRAPPER_PATH": str(WRAPPER)},
        text=True, capture_output=True,
    )
    # The receipt remains bound to the sealed/uploaded tree; a later mutation
    # of the canonical training root therefore fails closed rather than
    # silently treating the prefix as a publication of the new tree.
    assert verifier.returncode != 0
    mutable_artifact.write_bytes(original_mutable_artifact)
    verifier = subprocess.run(
        ["bash", "-c", 'source "$WRAPPER_PATH"; remote() { command "$@"; }; verify_remote_training_publication'],
        cwd=ROOT, env={**env, "WRAPPER_PATH": str(WRAPPER)},
        text=True, capture_output=True,
    )
    assert verifier.returncode == 0, verifier.stderr
    entries = receipt["entries"]
    download_rows = download_log.read_text(encoding="utf-8").splitlines()
    assert len(download_rows) == len(entries) * 3
    assert {
        row.split("\t", 1)[1] for row in download_rows
    } == {str(training_cache / "training-publication-readback")}

    verify_harness = (
        'source "$WRAPPER_PATH"; remote() { command "$@"; }; '
        "verify_remote_training_publication"
    )

    def verify_publication() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", verify_harness], cwd=ROOT,
            env={**env, "WRAPPER_PATH": str(WRAPPER)},
            text=True, capture_output=True,
        )

    assert verify_publication().returncode == 0
    receipt_path = training / "training-publication.json"
    original_receipt = receipt_path.read_bytes()

    receipt_path.chmod(0o644)
    assert verify_publication().returncode != 0
    receipt_path.chmod(0o444)

    forged = json.loads(original_receipt)
    forged["entries"][0]["extra"] = "forged"
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(_canonical_json_bytes(forged))
    receipt_path.chmod(0o444)
    assert verify_publication().returncode != 0

    forged = json.loads(original_receipt)
    forged["entries"][0]["sha256"] = "0" * 64
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(_canonical_json_bytes(forged))
    receipt_path.chmod(0o444)
    assert verify_publication().returncode != 0
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(original_receipt)
    receipt_path.chmod(0o444)

    relative = receipt["entries"][0]["path"]
    local_artifact = training / relative
    original_local = local_artifact.read_bytes()
    local_artifact.chmod(0o644)
    local_artifact.write_bytes(original_local + b"tampered")
    assert verify_publication().returncode != 0
    local_artifact.write_bytes(original_local)

    linked_copy = tmp_path / "linked-local-artifact"
    linked_copy.write_bytes(original_local)
    local_artifact.unlink()
    local_artifact.symlink_to(linked_copy)
    downloads_before_link = download_log.read_bytes()
    assert verify_publication().returncode != 0
    assert download_log.read_bytes() == downloads_before_link
    local_artifact.unlink()
    local_artifact.write_bytes(original_local)

    remote_artifact = remote_store / ("a" * 40) / receipt["remote_prefix"] / relative
    original_remote = remote_artifact.read_bytes()
    remote_artifact.write_bytes(original_remote + b"tampered")
    assert verify_publication().returncode != 0
    remote_artifact.write_bytes(original_remote)

    readback_cache = training_cache / "training-publication-readback"
    shutil.rmtree(readback_cache)
    symlink_target = tmp_path / "root-volume-cache"; symlink_target.mkdir()
    readback_cache.symlink_to(symlink_target, target_is_directory=True)
    before_unsafe = download_log.read_bytes()
    assert verify_publication().returncode != 0
    assert download_log.read_bytes() == before_unsafe

    assert os.stat("/dev").st_dev != training.stat().st_dev
    outside_mount = subprocess.run(
        ["bash", "-c", verify_harness], cwd=ROOT,
        env={
            **env,
            "WRAPPER_PATH": str(WRAPPER),
            "LEHOME_N15_TRAINING_HF_CACHE_ROOT": "/dev",
        },
        text=True, capture_output=True,
    )
    assert outside_mount.returncode != 0
    assert download_log.read_bytes() == before_unsafe


def test_fetch_remote_immutable_cleans_partial_transfer_and_retries(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, "n15-fetch-atomic")
    destination = tmp_path / "pipeline" / "fetched.json"
    harness = r'''
source "$WRAPPER_PATH"
remote() {
  attempt=0; [[ -f "$ATTEMPT_FILE" ]] && attempt=$(cat "$ATTEMPT_FILE")
  attempt=$((attempt + 1)); printf '%s' "$attempt" > "$ATTEMPT_FILE"
  if (( attempt == 1 )); then printf 'partial'; return 71; fi
  printf '{"complete":true}\n'
}
if fetch_remote_immutable /remote/receipt.json "$DESTINATION"; then exit 80; fi
[[ ! -e "$DESTINATION" && ! -L "$DESTINATION" ]]
fetch_remote_immutable /remote/receipt.json "$DESTINATION"
'''
    result = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={
            **env, "WRAPPER_PATH": str(WRAPPER),
            "DESTINATION": str(destination),
            "ATTEMPT_FILE": str(tmp_path / "fetch-attempt"),
        },
        text=True, capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert destination.read_bytes() == b'{"complete":true}\n'
    assert stat.S_IMODE(destination.stat().st_mode) == 0o444
    assert not list(destination.parent.glob(f".{destination.name}.*"))


@pytest.mark.parametrize("existing_kind", ["empty", "mutable", "mismatch"])
def test_fetch_remote_immutable_rejects_unsafe_or_mismatched_existing_receipt(
    tmp_path: Path, existing_kind: str,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, f"n15-fetch-existing-{existing_kind}")
    destination = tmp_path / "pipeline" / "fetched.json"
    expected = b'{"complete":true}\n'
    destination.write_bytes(b"" if existing_kind == "empty" else expected)
    if existing_kind == "mismatch":
        destination.write_bytes(b'{"complete":false}\n')
    destination.chmod(0o644 if existing_kind == "mutable" else 0o444)
    before = destination.read_bytes()
    harness = r'''
source "$WRAPPER_PATH"
remote() { printf '{"complete":true}\n'; }
fetch_remote_immutable /remote/receipt.json "$DESTINATION"
'''
    result = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={**env, "WRAPPER_PATH": str(WRAPPER), "DESTINATION": str(destination)},
        text=True, capture_output=True,
    )

    assert result.returncode != 0
    assert destination.read_bytes() == before


@pytest.mark.parametrize("interrupt_after", [1, 2])
def test_harvest_canonical_fetch_sequence_adopts_after_process_interruption(
    tmp_path: Path, interrupt_after: int,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, f"n15-harvest-fetch-{interrupt_after}")
    remote_store = tmp_path / "remote-harvest"; remote_store.mkdir()
    remote_payloads = {
        "manifest.json": b'{"episodes":1000}\n',
        "manifest-receipt.json": b'{"verified":true}\n',
        "harvest.publication.json": b'{"published":true}\n',
    }
    for name, payload in remote_payloads.items():
        (remote_store / name).write_bytes(payload)
    trace = tmp_path / "trace"
    harness = r'''
source "$WRAPPER_PATH"
remote_file_exists() {
  case "$1" in
    "$TRAINING_IDENTITY_RECEIPT"|"$TRAINING_PUBLICATION_RECEIPT"|"$FOCUSED_PROMOTION_RECEIPT"|"$REMOTE_PIPELINE_ROOT/harvest.publication.json") return 0 ;;
    *) return 1 ;;
  esac
}
verify_remote_training_chain() { :; }
verify_remote_training_publication() { :; }
verify_remote_focused_chain() { :; }
verify_remote_harvest_chain() { printf 'verify-harvest\n' >> "$TRACE"; }
record_host_stage_completion() { :; }
advance_paid_stage_admission_from_host_seals() { PRESTART_ADMITTED_STAGE=""; }
publish_training_readback() { return 90; }
run_paid_stage() { printf 'unexpected-paid:%s\n' "$1" >> "$TRACE"; return 91; }
remote() { local source="${!#}"; cat "$REMOTE_STORE/${source##*/}"; }
stop_exact_vm() { printf 'stop\n' >> "$TRACE"; }
finalize_host_harvest_terminal() { test -s "$HARVEST_MANIFEST" && test -s "$HARVEST_MANIFEST_RECEIPT" && test -s "$HARVEST_PUBLICATION_RECEIPT"; : > "$HARVEST_TERMINAL_RECEIPT"; }
run_pipeline_after_runtime
'''
    first = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={
            **env, "WRAPPER_PATH": str(WRAPPER), "REMOTE_STORE": str(remote_store),
            "TRACE": str(trace),
            "LEHOME_N15_TEST_HARVEST_FETCH_FAULT_AFTER": str(interrupt_after),
        },
        text=True, capture_output=True,
    )
    assert first.returncode == 130, first.stderr
    canonical = [
        Path(env["LEHOME_N15_PIPELINE_ROOT"]) / "harvest-manifest.json",
        Path(env["LEHOME_N15_PIPELINE_ROOT"]) / "harvest-manifest-receipt.json",
        Path(env["LEHOME_N15_PIPELINE_ROOT"]) / "harvest-publication.json",
    ]
    assert sum(path.exists() for path in canonical) == interrupt_after

    second = subprocess.run(
        ["bash", "-c", harness], cwd=ROOT,
        env={
            **env, "WRAPPER_PATH": str(WRAPPER), "REMOTE_STORE": str(remote_store),
            "TRACE": str(trace),
        },
        text=True, capture_output=True,
    )
    assert second.returncode == 0, second.stderr
    assert not [line for line in trace.read_text().splitlines() if line.startswith("unexpected-paid")]
    assert [path.read_bytes() for path in canonical] == list(remote_payloads.values())
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in canonical)
    assert not list(Path(env["LEHOME_N15_PIPELINE_ROOT"]).glob(".harvest-*.??????"))


@pytest.mark.parametrize("hang_point", ["runtime", "reconcile", "publication"])
def test_aggregate_watchdog_reaps_hung_post_start_lifecycle_before_stop_and_unlock(
    tmp_path: Path, hang_point: str,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, f"n15-aggregate-{hang_point}")
    pipeline = Path(env["LEHOME_N15_PIPELINE_ROOT"])
    trace = tmp_path / "trace"
    started_marker = tmp_path / "provider-started"
    stopped_marker = tmp_path / "provider-stopped"
    descendant_pid = tmp_path / "hung-descendant-pid"
    module = _load_cli()
    plan = pipeline / "lifecycle-plan.json"
    assert module.main([
        "lifecycle-plan", "--run-id", env["LEHOME_N15_RUN_ID"],
        "--repository", env["LEHOME_N15_PUBLIC_HF_REPOSITORY"],
        "--remote-pipeline-root", env["LEHOME_N15_REMOTE_PIPELINE_ROOT"],
        "--budget-usd", "100", "--estimated-cost-usd", "72",
        "--output", str(plan),
    ]) == 0
    started = int(time.time()) - 86_397
    paid_deadline = {
        "schema_version": 1,
        "kind": "lehome_public_n15_paid_deadline_v1",
        "run_id": env["LEHOME_N15_RUN_ID"],
        "lifecycle_plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "started_unix_seconds": started,
        "deadline_unix_seconds": started + 86_400,
    }
    deadline_path = pipeline / "paid-deadline.json"
    deadline_path.write_bytes(_canonical_json_bytes(paid_deadline)); deadline_path.chmod(0o444)
    deadline_before = deadline_path.read_bytes()
    # Keep the production deadline/termination code intact while making this
    # harness deterministic: once the isolated hung descendant is observable,
    # its test-only `date` shim advances the controller to its already sealed
    # aggregate deadline.  Scheduler delay can no longer consume the tiny
    # three-second real-time window before the watchdog is exercised.
    _write_executable(
        fake_bin / "date",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == +%s && -f "$FAKE_WATCHDOG_DESCENDANT_PID" ]]; then
  command python3 - "$FAKE_WATCHDOG_DEADLINE" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="ascii"))["deadline_unix_seconds"])
PY
  exit 0
fi
exec /bin/date "$@"
""",
    )
    harness = r'''
source "$WRAPPER_PATH"
acquire_controller_lock() { printf 'lock\n' >> "$TRACE"; }
release_controller_lock() {
  [[ -f "$STOPPED_MARKER" ]] || { printf 'unlock-before-stop\n' >> "$TRACE"; return 92; }
  printf 'unlock\n' >> "$TRACE"
}
capture_exact_provider_state() {
  case "$1" in
    STOPPED) [[ ! -f "$STARTED_MARKER" ]] || return 1 ;;
    RUNNING) [[ -f "$STARTED_MARKER" ]] || return 1 ;;
    *) return 1 ;;
  esac
  : > "$2"
}
nebius() {
  [[ "$*" == "compute instance start"* ]] || return 93
  printf 'start\n' >> "$TRACE"; : > "$STARTED_MARKER"
}
stop_exact_vm() {
  if [[ -f "$DESCENDANT_PID" ]]; then
    child="$(cat "$DESCENDANT_PID")"
    state="$(ps -o stat= -p "$child" 2>/dev/null || true)"
    [[ -z "$state" || "$state" == Z* ]] || { printf 'stop-before-reap\n' >> "$TRACE"; return 1; }
  fi
  printf 'stop\n' >> "$TRACE"; : > "$STOPPED_MARKER"
}
host_next_unfinished_stage() { printf 'train\n'; }
wait_for_ssh_readiness() { :; }
hang_lifecycle() {
  command python3 - "$DESCENDANT_PID" <<'PY' &
import os
from pathlib import Path
import signal
import sys

os.setsid()
Path(sys.argv[1]).write_text(f"{os.getpid()}\n", encoding="ascii")
signal.signal(signal.SIGTERM, lambda *_args: None)
while True:
    signal.pause()
PY
  child=$!
  while [[ ! -f "$DESCENDANT_PID" ]]; do /bin/sleep 0.01; done
  printf 'hang:%s\n' "$HANG_POINT" >> "$TRACE"
  wait "$child"
}
if [[ "$HANG_POINT" == runtime ]]; then
  wait_for_remote_runtime() { hang_lifecycle; }
else
  wait_for_remote_runtime() { :; }
fi
if [[ "$HANG_POINT" == reconcile ]]; then
  reconcile_remote_stage_seals() { hang_lifecycle; }
else
  reconcile_remote_stage_seals() { :; }
fi
remote_file_exists() { [[ "$1" == "$TRAINING_IDENTITY_RECEIPT" ]]; }
verify_remote_training_chain() { :; }
if [[ "$HANG_POINT" == publication ]]; then
  publish_training_readback() { hang_lifecycle; }
else
  publish_training_readback() { return 94; }
fi
main
'''
    process = subprocess.Popen(
        ["bash", "-c", harness], cwd=ROOT,
        env={
            **env,
            "WRAPPER_PATH": str(WRAPPER),
            "TRACE": str(trace),
            "STARTED_MARKER": str(started_marker),
            "STOPPED_MARKER": str(stopped_marker),
            "DESCENDANT_PID": str(descendant_pid),
            "FAKE_WATCHDOG_DESCENDANT_PID": str(descendant_pid),
            "FAKE_WATCHDOG_DEADLINE": str(deadline_path),
            "HANG_POINT": hang_point,
        },
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        if descendant_pid.exists():
            try:
                os.kill(int(descendant_pid.read_text(encoding="ascii")), signal.SIGKILL)
            except ProcessLookupError:
                pass
        stdout, stderr = process.communicate(timeout=3)
        pytest.fail(f"aggregate watchdog did not terminate {hang_point}: {stdout!r} {stderr!r}")

    assert process.returncode != 0
    lines = trace.read_text(encoding="ascii").splitlines()
    assert lines[:3] == ["lock", "start", f"hang:{hang_point}"]
    assert "stop-before-reap" not in lines
    assert lines[-2:] == ["stop", "unlock"]
    assert deadline_path.read_bytes() == deadline_before
    with pytest.raises(ProcessLookupError):
        os.kill(int(descendant_pid.read_text(encoding="ascii")), 0)


def test_running_observation_waits_for_cloud_init_after_ssh_is_ready(tmp_path: Path) -> None:
    """A transient runtime gate must not stop a guest that has already accepted SSH."""
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    state = tmp_path / "provider-state.txt"; state.write_text("STOPPED", encoding="utf-8")
    trace = tmp_path / "provider-trace.log"
    provider = {
        "metadata": {"id": "computeinstance-u00t6xfqhadrcmssa2", "name": "lehome-rollout"},
        "status": {"state": "STATE"},
        "spec": {"boot_disk": {"managed_disk": {"spec": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}}}, "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}], "preemptible": {"on_preemption": "STOP"}},
    }
    (fake_bin / "nebius").write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "state = Path(os.environ['FAKE_NEBIUS_STATE'])\n"
        "trace = Path(os.environ['FAKE_PROVIDER_TRACE'])\n"
        "command = sys.argv[1:4]\n"
        "if command == ['compute', 'instance', 'start']:\n"
        "    trace.open('a').write('start\\n')\n"
        "    if state.read_text().strip() != 'STOPPED': raise SystemExit(91)\n"
        "    state.write_text('RUNNING')\n"
        "elif command == ['compute', 'instance', 'stop']:\n"
        "    trace.open('a').write('stop\\n')\n"
        "    if state.read_text().strip() != 'RUNNING': raise SystemExit(92)\n"
        "    state.write_text('STOPPED')\n"
        "else:\n"
        "    value = " + repr(provider) + "; value['status']['state'] = state.read_text().strip(); print(json.dumps(value))\n",
        encoding="utf-8",
    )
    (fake_bin / "ssh").write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${!#}\" == true ]]; then\n"
        "  printf 'readiness\\n' >> \"$FAKE_PROVIDER_TRACE\"\n"
        "  attempts=0; [[ -f \"$FAKE_SSH_ATTEMPTS\" ]] && attempts=$(cat \"$FAKE_SSH_ATTEMPTS\")\n"
        "  attempts=$((attempts + 1)); printf '%s' \"$attempts\" > \"$FAKE_SSH_ATTEMPTS\"\n"
        "  (( attempts >= 7 )) && exit 0\n"
        "  exit 98\n"
        "fi\n"
        "attempts=0; [[ -f \"$FAKE_RUNTIME_ATTEMPTS\" ]] && attempts=$(cat \"$FAKE_RUNTIME_ATTEMPTS\")\n"
        "attempts=$((attempts + 1)); printf '%s' \"$attempts\" > \"$FAKE_RUNTIME_ATTEMPTS\"\n"
        "if (( attempts <= 3 )); then printf 'runtime\\n' >> \"$FAKE_PROVIDER_TRACE\"; (( attempts == 3 )) && exit 0; exit 97; fi\n"
        "if (( attempts == 4 )); then printf 'identity-check\\n' >> \"$FAKE_PROVIDER_TRACE\"; exit 97; fi\n"
        "printf 'train\\n' >> \"$FAKE_PROVIDER_TRACE\"\n"
        "exit 97\n",
        encoding="utf-8",
    )
    (fake_bin / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    for command in (fake_bin / "nebius", fake_bin / "ssh", fake_bin / "sleep"): command.chmod(0o755)
    pipeline = tmp_path / "pipeline"; pipeline.mkdir()
    env = {
        **os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_PROVIDER_TRACE": str(trace), "FAKE_NEBIUS_STATE": str(state), "FAKE_SSH_ATTEMPTS": str(tmp_path / "ssh-attempts"),
        "FAKE_RUNTIME_ATTEMPTS": str(tmp_path / "runtime-attempts"),
        "LEHOME_N15_RUN_ID": "n15-running-observation", "LEHOME_N15_PIPELINE_ROOT": str(pipeline),
        "LEHOME_N15_SSH_TARGET": "operator@example", "LEHOME_N15_REMOTE_ROOT": "/mnt/lehome/runtime",
        "LEHOME_N15_REMOTE_RUNS_BASE": "/mnt/lehome/runs", "LEHOME_N15_REMOTE_PIPELINE_ROOT": "/mnt/lehome/runs/n15-running-observation",
        "LEHOME_N15_PUBLIC_HF_REPOSITORY": "ryanjin333/public-n15", "LEHOME_OFFICIAL_ASSETS_ROOT": "/mnt/assets", "LEHOME_OFFICIAL_METADATA_ROOT": "/mnt/source",
        "LEHOME_N15_REFERENCE_CHECKPOINT": "/mnt/reference", "LEHOME_N15_REFERENCE_SANITIZED_CONFIG_ROOT": "/mnt/reference-config",
        "LEHOME_N15_REFERENCE_COMPATIBILITY_RECEIPT": "/mnt/reference-receipt", "LEHOME_N15_NATIVE_RUNTIME_EVIDENCE_ROOT": "/mnt/evidence",
        "LEHOME_N15_NATIVE_DEPENDENCIES_ROOT": "/mnt/deps", "LEHOME_N15_FOCUSED_HF_CACHE_ROOT": "/mnt/cache", "LEHOME_N15_ROLLOUT_IMAGE_RECEIPT": "/mnt/image.json",
        "LEHOME_N15_TRAINING_HF_CACHE_ROOT": "/mnt/train-cache", "LEHOME_N15_TRAINING_UV": "/mnt/uv", "LEHOME_N15_LEROBOT_WHEEL": "/mnt/lerobot.whl",
        "LEHOME_N15_TRAINING_ROOT": "/mnt/lehome/runs/n15-running-observation/training",
    }
    result = subprocess.run(["bash", str(WRAPPER)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    lines = trace.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "start" and lines[-1] == "stop"
    assert lines.count("readiness") >= 7
    assert lines.count("runtime") >= 3
    assert lines.count("identity-check") == 1
    assert lines.count("train") >= 1
    assert lines.index("runtime") > max(
        index for index, value in enumerate(lines) if value == "readiness"
    )
    assert lines.index("identity-check") > max(
        index for index, value in enumerate(lines) if value == "runtime"
    )
    assert lines.index("train") > lines.index("identity-check")
    assert state.read_text(encoding="utf-8") == "STOPPED"


def test_running_observation_hard_stops_a_hanging_ssh_readiness_probe(tmp_path: Path) -> None:
    """A hung readiness probe must not delay the controller's EXIT cleanup."""
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    state = tmp_path / "provider-state.txt"; state.write_text("STOPPED", encoding="utf-8")
    trace = tmp_path / "provider-trace.log"; ssh_attempts = tmp_path / "ssh-attempts"; ssh_pid = tmp_path / "hanging-ssh.pid"
    provider = {
        "metadata": {"id": "computeinstance-u00t6xfqhadrcmssa2", "name": "lehome-rollout"},
        "status": {"state": "STATE"},
        "spec": {"boot_disk": {"managed_disk": {"spec": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}}}, "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}], "preemptible": {"on_preemption": "STOP"}},
    }
    (fake_bin / "nebius").write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "state = Path(os.environ['FAKE_NEBIUS_STATE']); trace = Path(os.environ['FAKE_PROVIDER_TRACE'])\n"
        "command = sys.argv[1:4]\n"
        "if command == ['compute', 'instance', 'start']:\n"
        "    trace.open('a').write('start\\n'); state.read_text().strip() == 'STOPPED' or sys.exit(91); state.write_text('RUNNING')\n"
        "elif command == ['compute', 'instance', 'stop']:\n"
        "    trace.open('a').write('stop\\n'); state.read_text().strip() == 'RUNNING' or sys.exit(92); state.write_text('STOPPED')\n"
        "else:\n"
        "    value = " + repr(provider) + "; value['status']['state'] = state.read_text().strip(); print(json.dumps(value))\n",
        encoding="utf-8",
    )
    (fake_bin / "ssh").write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${!#}\" == true ]]; then\n"
        "  attempts=0; [[ -f \"$FAKE_SSH_ATTEMPTS\" ]] && attempts=$(cat \"$FAKE_SSH_ATTEMPTS\")\n"
        "  attempts=$((attempts + 1)); printf '%s' \"$attempts\" > \"$FAKE_SSH_ATTEMPTS\"\n"
        "  if (( attempts == 1 )); then printf 'readiness-hang\\n' >> \"$FAKE_PROVIDER_TRACE\"; printf '%s' \"$$\" > \"$FAKE_HANGING_SSH_PID\"; while :; do /bin/sleep 1; done; fi\n"
        "  printf 'readiness-fail\\n' >> \"$FAKE_PROVIDER_TRACE\"; exit 98\n"
        "fi\n"
        "printf 'runtime\\n' >> \"$FAKE_PROVIDER_TRACE\"; exit 97\n",
        encoding="utf-8",
    )
    (fake_bin / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    for command in (fake_bin / "nebius", fake_bin / "ssh", fake_bin / "sleep"): command.chmod(0o755)
    pipeline = tmp_path / "pipeline"; pipeline.mkdir()
    env = {
        **os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "FAKE_NEBIUS_STATE": str(state), "FAKE_PROVIDER_TRACE": str(trace),
        "FAKE_SSH_ATTEMPTS": str(ssh_attempts), "FAKE_HANGING_SSH_PID": str(ssh_pid), "LEHOME_N15_RUN_ID": "n15-hanging-readiness",
        "LEHOME_N15_PIPELINE_ROOT": str(pipeline), "LEHOME_N15_SSH_TARGET": "operator@example", "LEHOME_N15_REMOTE_ROOT": "/mnt/lehome/runtime",
        "LEHOME_N15_REMOTE_RUNS_BASE": "/mnt/lehome/runs", "LEHOME_N15_REMOTE_PIPELINE_ROOT": "/mnt/lehome/runs/n15-hanging-readiness",
        "LEHOME_N15_PUBLIC_HF_REPOSITORY": "ryanjin333/public-n15", "LEHOME_OFFICIAL_ASSETS_ROOT": "/mnt/assets", "LEHOME_OFFICIAL_METADATA_ROOT": "/mnt/source",
        "LEHOME_N15_REFERENCE_CHECKPOINT": "/mnt/reference", "LEHOME_N15_REFERENCE_SANITIZED_CONFIG_ROOT": "/mnt/reference-config",
        "LEHOME_N15_REFERENCE_COMPATIBILITY_RECEIPT": "/mnt/reference-receipt", "LEHOME_N15_NATIVE_RUNTIME_EVIDENCE_ROOT": "/mnt/evidence",
        "LEHOME_N15_NATIVE_DEPENDENCIES_ROOT": "/mnt/deps", "LEHOME_N15_FOCUSED_HF_CACHE_ROOT": "/mnt/cache", "LEHOME_N15_ROLLOUT_IMAGE_RECEIPT": "/mnt/image.json",
        "LEHOME_N15_TRAINING_HF_CACHE_ROOT": "/mnt/train-cache", "LEHOME_N15_TRAINING_UV": "/mnt/uv", "LEHOME_N15_LEROBOT_WHEEL": "/mnt/lerobot.whl",
        "LEHOME_N15_TRAINING_ROOT": "/mnt/lehome/runs/n15-hanging-readiness/training",
    }
    started = time.monotonic()
    process = subprocess.Popen(["bash", str(WRAPPER)], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        _, stderr = process.communicate(timeout=14)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL); process.communicate()
        pytest.fail("readiness probe exceeded the test wall-clock bound")
    assert time.monotonic() - started < 14
    assert process.returncode != 0
    assert "exact VM did not become SSH-ready" in stderr
    assert trace.read_text(encoding="utf-8").splitlines()[0] == "start"
    assert trace.read_text(encoding="utf-8").splitlines().count("stop") == 1
    assert state.read_text(encoding="utf-8") == "STOPPED"
    with pytest.raises(ProcessLookupError): os.kill(int(ssh_pid.read_text(encoding="utf-8")), 0)


def test_running_observation_reaps_a_term_ignoring_ssh_readiness_probe(tmp_path: Path) -> None:
    """Controller interruption must reap a TERM-ignoring readiness child."""
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    state = tmp_path / "provider-state.txt"; state.write_text("STOPPED", encoding="utf-8")
    trace = tmp_path / "provider-trace.log"; ssh_pid = tmp_path / "hanging-ssh.pid"
    provider = {
        "metadata": {"id": "computeinstance-u00t6xfqhadrcmssa2", "name": "lehome-rollout"},
        "status": {"state": "STATE"},
        "spec": {"boot_disk": {"managed_disk": {"spec": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}}}, "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}], "preemptible": {"on_preemption": "STOP"}},
    }
    (fake_bin / "nebius").write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "state = Path(os.environ['FAKE_NEBIUS_STATE']); trace = Path(os.environ['FAKE_PROVIDER_TRACE'])\n"
        "command = sys.argv[1:4]\n"
        "if command == ['compute', 'instance', 'start']:\n"
        "    trace.open('a').write('start\\n'); state.read_text().strip() == 'STOPPED' or sys.exit(91); state.write_text('RUNNING')\n"
        "elif command == ['compute', 'instance', 'stop']:\n"
        "    trace.open('a').write('stop\\n'); state.read_text().strip() == 'RUNNING' or sys.exit(92); state.write_text('STOPPED')\n"
        "else:\n"
        "    value = " + repr(provider) + "; value['status']['state'] = state.read_text().strip(); print(json.dumps(value))\n",
        encoding="utf-8",
    )
    (fake_bin / "ssh").write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${!#}\" == true ]]; then\n"
        "  printf 'readiness-term-ignore\\n' >> \"$FAKE_PROVIDER_TRACE\"; printf '%s' \"$$\" > \"$FAKE_HANGING_SSH_PID\"\n"
        "  trap '' TERM\n"
        "  while :; do /bin/sleep 1; done\n"
        "fi\n"
        "printf 'runtime\\n' >> \"$FAKE_PROVIDER_TRACE\"; exit 97\n",
        encoding="utf-8",
    )
    for command in (fake_bin / "nebius", fake_bin / "ssh"): command.chmod(0o755)
    pipeline = tmp_path / "pipeline"; pipeline.mkdir()
    env = {
        **os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "FAKE_NEBIUS_STATE": str(state), "FAKE_PROVIDER_TRACE": str(trace),
        "FAKE_HANGING_SSH_PID": str(ssh_pid), "LEHOME_N15_RUN_ID": "n15-interrupted-readiness", "LEHOME_N15_PIPELINE_ROOT": str(pipeline),
        "LEHOME_N15_SSH_TARGET": "operator@example", "LEHOME_N15_REMOTE_ROOT": "/mnt/lehome/runtime", "LEHOME_N15_REMOTE_RUNS_BASE": "/mnt/lehome/runs",
        "LEHOME_N15_REMOTE_PIPELINE_ROOT": "/mnt/lehome/runs/n15-interrupted-readiness", "LEHOME_N15_PUBLIC_HF_REPOSITORY": "ryanjin333/public-n15",
        "LEHOME_OFFICIAL_ASSETS_ROOT": "/mnt/assets", "LEHOME_OFFICIAL_METADATA_ROOT": "/mnt/source", "LEHOME_N15_REFERENCE_CHECKPOINT": "/mnt/reference",
        "LEHOME_N15_REFERENCE_SANITIZED_CONFIG_ROOT": "/mnt/reference-config", "LEHOME_N15_REFERENCE_COMPATIBILITY_RECEIPT": "/mnt/reference-receipt",
        "LEHOME_N15_NATIVE_RUNTIME_EVIDENCE_ROOT": "/mnt/evidence", "LEHOME_N15_NATIVE_DEPENDENCIES_ROOT": "/mnt/deps", "LEHOME_N15_FOCUSED_HF_CACHE_ROOT": "/mnt/cache",
        "LEHOME_N15_ROLLOUT_IMAGE_RECEIPT": "/mnt/image.json", "LEHOME_N15_TRAINING_HF_CACHE_ROOT": "/mnt/train-cache", "LEHOME_N15_TRAINING_UV": "/mnt/uv",
        "LEHOME_N15_LEROBOT_WHEEL": "/mnt/lerobot.whl", "LEHOME_N15_TRAINING_ROOT": "/mnt/lehome/runs/n15-interrupted-readiness/training",
    }
    process = subprocess.Popen(["bash", str(WRAPPER)], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    deadline = time.monotonic() + 3
    while not ssh_pid.exists() and time.monotonic() < deadline:
        if process.poll() is not None: pytest.fail("controller exited before readiness probe started")
        time.sleep(0.02)
    assert ssh_pid.exists()
    started = time.monotonic(); os.killpg(process.pid, signal.SIGTERM)
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL); process.communicate()
        pytest.fail("interrupted controller exceeded the test wall-clock bound")
    assert time.monotonic() - started < 5
    assert process.returncode != 0
    assert trace.read_text(encoding="utf-8").splitlines().count("stop") == 1
    assert state.read_text(encoding="utf-8") == "STOPPED"
    with pytest.raises(ProcessLookupError): os.kill(int(ssh_pid.read_text(encoding="utf-8")), 0)
    interrupted = WRAPPER.read_text(encoding="utf-8").split("def interrupted", 1)[1].split("signal.signal", 1)[0]
    assert interrupted.index("stop_probe_group(force=True)") < interrupted.rindex("probe.wait(timeout=reap_timeout)")
