"""Offline boundary tests for the bounded public N1.5 remote pipeline."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
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
    installed_site = tmp_path / "installed/site-packages"
    shutil.copytree(staging / "runtime/site-packages", installed_site)
    shutil.rmtree(staging / "runtime")
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
if [[ "${1:-}" == - && $# -eq 12 ]]; then
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
    runtime_value["lerobot_package_root"] = str(installed_site / "lerobot")
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
[[ "${1:-}" == -n ]] && shift
if [[ "${1:-}" == chown ]]; then exit 0; fi
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
        "FAKE_NATIVE_TRAINER": str(trainer),
        "FAKE_DATASET_BLOBS": str(
            Path(snapshots_value["dataset"]["snapshot_root"]).parents[1] / "blobs"
        ),
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
    assert '"$LEROBOT_WHEEL" "$RUNTIME_IMAGE_ID" "$RESUME_PARTIAL" "$RESUME_CHECKPOINT" "$RESUME_STEP" "$RESUME_ATTEMPT_ID" <<\'SH\'' in text
    assert 'runtime_image_id="${12}"' in text
    assert " LEROBOT_WHEEL RUNTIME_IMAGE_ID RESUME_PARTIAL RESUME_CHECKPOINT RESUME_STEP RESUME_ATTEMPT_ID ASSETS_ROOT" in text
    assert "LEHOME_N15_EXPECTED_IMAGE_ID" not in text
    assert "nebius compute instance start --id" in text
    assert "nebius compute instance stop --id" in text
    assert "compute instance create" not in text
    assert "compute disk create" not in text
    assert "compute image create" not in text
    assert "trap controller_cleanup EXIT" in text
    assert text.count("StrictHostKeyChecking=accept-new") == 2
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


@pytest.mark.parametrize("stage_deadline_kind", ["mismatch", "expired"])
def test_invalid_train_stage_deadline_fails_before_provider_start(
    tmp_path: Path, stage_deadline_kind: str,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    provider_log = tmp_path / "provider.log"
    (fake_bin / "nebius").write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_PROVIDER_LOG\"\nexit 97\n",
        encoding="utf-8",
    )
    (fake_bin / "ssh").write_text("#!/usr/bin/env bash\nexit 97\n", encoding="utf-8")
    for command in (fake_bin / "nebius", fake_bin / "ssh"): command.chmod(0o755)
    env = _wrapper_env(tmp_path, fake_bin, f"n15-stage-{stage_deadline_kind}")
    env["FAKE_PROVIDER_LOG"] = str(provider_log)
    module = _load_cli()
    pipeline = Path(env["LEHOME_N15_PIPELINE_ROOT"])
    plan_path = pipeline / "lifecycle-plan.json"
    assert module.main([
        "lifecycle-plan", "--run-id", env["LEHOME_N15_RUN_ID"],
        "--repository", env["LEHOME_N15_PUBLIC_HF_REPOSITORY"],
        "--remote-pipeline-root", env["LEHOME_N15_REMOTE_PIPELINE_ROOT"],
        "--budget-usd", "100", "--estimated-cost-usd", "72",
        "--output", str(plan_path),
    ]) == 0
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    paid_started = int(time.time())
    paid_deadline = paid_started + 86400
    paid = {
        "schema_version": 1, "kind": "lehome_public_n15_paid_deadline_v1",
        "run_id": env["LEHOME_N15_RUN_ID"], "lifecycle_plan_sha256": plan_sha,
        "started_unix_seconds": paid_started,
        "deadline_unix_seconds": paid_deadline,
    }
    (pipeline / "paid-deadline.json").write_text(
        json.dumps(paid, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii"
    )
    stage_started = 1 if stage_deadline_kind == "expired" else paid_started
    stage = {
        "schema_version": 1,
        "kind": "lehome_public_n15_stage_deadline_v1",
        "run_id": (
            env["LEHOME_N15_RUN_ID"] if stage_deadline_kind == "expired" else "wrong-run"
        ),
        "stage": "train",
        "lifecycle_plan_sha256": plan_sha,
        "started_unix_seconds": stage_started,
        "deadline_unix_seconds": min(stage_started + 43200, paid_deadline),
    }
    stage_path = pipeline / "stage-train-deadline.json"
    stage_path.write_text(
        json.dumps(stage, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii"
    )
    for path in (pipeline / "paid-deadline.json", stage_path):
        path.chmod(0o444)

    result = subprocess.run(
        ["bash", str(WRAPPER)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode != 0
    if stage_deadline_kind == "expired":
        assert "error: train deadline has expired" in result.stderr.lower()
    else:
        assert "train" in result.stderr.lower() and "deadline" in result.stderr.lower()
    assert not provider_log.exists()


def test_atomic_controller_singleton_rejects_second_live_controller_without_stopping_vm(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    state = tmp_path / "state"; state.write_text("STOPPED", encoding="ascii")
    trace = tmp_path / "trace"
    provider = {
        "metadata": {"id": "computeinstance-u00t6xfqhadrcmssa2", "name": "lehome-rollout"},
        "status": {"state": "STATE"},
        "spec": {"boot_disk": {"managed_disk": {"spec": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}}}, "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}]},
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
            "holder_pid": owner["holder_pid"],
            "script_path": str(WRAPPER.resolve()),
        }
        os.kill(owner["holder_pid"], 0)

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


def test_orphan_lock_holder_detects_abrupt_controller_death(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    env = _wrapper_env(tmp_path, fake_bin, "n15-orphan-lock")
    process = subprocess.Popen(
        ["bash", "-c", 'source "$WRAPPER_PATH"; acquire_controller_lock; while :; do /bin/sleep 1; done'],
        cwd=ROOT, env={**env, "WRAPPER_PATH": str(WRAPPER)},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    lock_path = _controller_lock_path(env)
    holder_pid: int | None = None
    try:
        deadline = time.monotonic() + 5
        while not lock_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert lock_path.is_file()
        holder_pid = json.loads(lock_path.read_text(encoding="ascii"))["holder_pid"]

        os.kill(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)
        deadline = time.monotonic() + 5
        while lock_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not lock_path.exists()
        with pytest.raises(ProcessLookupError):
            os.kill(holder_pid, 0)
    finally:
        if process.poll() is None:
            os.kill(process.pid, signal.SIGKILL)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
        if holder_pid is not None:
            try:
                os.kill(holder_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def test_stale_controller_lock_is_reclaimed_without_killing_a_process(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    trace = tmp_path / "trace"
    provider = {
        "metadata": {"id": "computeinstance-u00t6xfqhadrcmssa2", "name": "lehome-rollout"},
        "status": {"state": "STOPPED"},
        "spec": {"boot_disk": {"managed_disk": {"spec": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}}}, "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}]},
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


def test_running_observation_waits_for_cloud_init_after_ssh_is_ready(tmp_path: Path) -> None:
    """A transient runtime gate must not stop a guest that has already accepted SSH."""
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    state = tmp_path / "provider-state.txt"; state.write_text("STOPPED", encoding="utf-8")
    trace = tmp_path / "provider-trace.log"
    provider = {
        "metadata": {"id": "computeinstance-u00t6xfqhadrcmssa2", "name": "lehome-rollout"},
        "status": {"state": "STATE"},
        "spec": {"boot_disk": {"managed_disk": {"spec": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}}}, "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}]},
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
    assert trace.read_text(encoding="utf-8").splitlines() == ["start", *("readiness" for _ in range(7)), "runtime", "runtime", "runtime", "identity-check", "train", "stop"]
    assert state.read_text(encoding="utf-8") == "STOPPED"


def test_running_observation_hard_stops_a_hanging_ssh_readiness_probe(tmp_path: Path) -> None:
    """A hung readiness probe must not delay the controller's EXIT cleanup."""
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    state = tmp_path / "provider-state.txt"; state.write_text("STOPPED", encoding="utf-8")
    trace = tmp_path / "provider-trace.log"; ssh_attempts = tmp_path / "ssh-attempts"; ssh_pid = tmp_path / "hanging-ssh.pid"
    provider = {
        "metadata": {"id": "computeinstance-u00t6xfqhadrcmssa2", "name": "lehome-rollout"},
        "status": {"state": "STATE"},
        "spec": {"boot_disk": {"managed_disk": {"spec": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}}}, "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}]},
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
        _, stderr = process.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL); process.communicate()
        pytest.fail("readiness probe exceeded the test wall-clock bound")
    assert time.monotonic() - started < 8
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
        "spec": {"boot_disk": {"managed_disk": {"spec": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}}}, "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}]},
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
