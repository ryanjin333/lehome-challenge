"""Offline boundary tests for the bounded public N1.5 remote pipeline."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/run_public_n15_reproduction.py"
WRAPPER = ROOT / "rollout_appliance/run_public_n15_pipeline_remote.sh"


def _load_cli():
    spec = importlib.util.spec_from_file_location("public_n15_reproduction", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert '"$LEROBOT_WHEEL" "$RUNTIME_IMAGE_ID" <<\'SH\'' in text
    assert 'runtime_image_id="${12}"' in text
    assert " LEROBOT_WHEEL RUNTIME_IMAGE_ID ASSETS_ROOT" in text
    assert "LEHOME_N15_EXPECTED_IMAGE_ID" not in text
    assert "nebius compute instance start --id" in text
    assert "nebius compute instance stop --id" in text
    assert "compute instance create" not in text
    assert "compute disk create" not in text
    assert "compute image create" not in text
    assert "trap stop_exact_vm EXIT" in text
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
    assert " start " not in f" {log.read_text(encoding='utf-8')} "


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
