from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "rollout_appliance/run_official_lehome_comparison_container.sh"
RUNBOOK = ROOT / "docs/experiments/2026-08-30-official-lehome-comparison-runbook.md"


def test_container_wrapper_mounts_official_source_and_assets_read_only() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "a805ad2f7ab52a4583066fc4ee5180459a7f9d15" in text
    assert "bea65fd960ad5a1bb3bd3fa77164b28001c08ef9" in text
    assert 'src=$OFFICIAL_SOURCE_ROOT,dst=/official/lehome,readonly' in text
    assert 'src=$OFFICIAL_ASSETS_ROOT,dst=/official/assets,readonly' in text
    assert "--device cpu" in text
    assert "--gpus all" in text
    assert "--seed 42" in text
    assert "--n17-identity-receipt" in text
    assert "eval_groot_n17_public96_reference.json" in text
    assert 'src=$COMPETITOR_CHECKPOINT_ROOT,dst=$COMPETITOR_CHECKPOINT_ROOT,readonly' in text
    assert 'src=$SANITIZED_CONFIG_ROOT,dst=$SANITIZED_CONFIG_ROOT,readonly' in text
    assert "/official/competitor" not in text


def test_container_wrapper_binds_reviewed_runtime_images_and_real_policy_readiness() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "LEHOME_OFFICIAL_RUNTIME_REVISION" in text
    assert "git -C \"$REPO_ROOT\" status --porcelain" in text
    assert "runtime-identity.json" in text
    assert "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7" in text
    assert "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746" in text
    assert "run_groot_n17_public96_policy_server" in text
    assert "PYTHONPATH=/runtime/source/lehome:/runtime:/opt/isaac-groot" in text
    assert "N17_BASE_MODEL_ROOT" in text
    assert "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0" in text
    assert 'dst=/cache/models/nvidia/Cosmos-Reason2-2B,readonly' in text
    assert "--workdir /cache/models" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "policy-server-readiness.json" in text
    assert "policy-server-startup.log" in text
    assert "cuda-runtime.json" in text
    assert 'docker exec -i "$POLICY_CONTAINER"' in text
    assert 'local reference="$1" output="$2" mode="$3" raw\n' in text
    assert 'raw="$EVIDENCE_ROOT/image-inspect-$mode.json"' in text


def test_container_wrapper_reuses_exact_native_reference_dependency_boundary() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "/mnt/lehome/reference-native/dependencies/peft-0.18.1-py3-none-any.whl" in text
    assert "0bf06847a3551e3019fc58c440cffc9a6b73e6e2962c95b52e224f77bbdb50f1" in text
    assert "flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl" in text
    assert "dm_tree-0.1.9-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" in text
    assert "qwen_vl_utils-0.0.14-py3-none-any.whl" in text
    assert "torchdiffeq-0.2.5-py3-none-any.whl" in text
    assert "validate-peft-overlay" in text
    assert "validate-flash-attention-overlay" in text
    assert "validate-public-pyproject-dependencies-overlay" in text
    assert "uv pip install --offline --no-deps --python /opt/lehome-challenge/.venv/bin/python" in text
    assert "PYTHONEXE=/opt/lehome-challenge/.venv/bin/python" in text
    assert "--python-bin /opt/lehome-challenge/.venv/bin/python" in text
    assert "--competitor-runtime-evidence-root" in text
    assert text.count("/isaac-sim/python.sh -m scripts.serve_official_docker_policy_bridge") == 1
    assert 'then bridge_ready=1; break; fi' in text
    assert "CONTROLLER_WIRE_ROOT" in text
    assert "msgpack.__version__ != \"1.1.0\"" in text
    assert "zmq.__version__ != \"27.0.1\"" in text
    assert "third_party/IsaacLab/source/isaaclab_tasks" in text


def test_full_requires_smoke_receipt_and_smoke_rejects_one() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "LEHOME_OFFICIAL_SMOKE_RECEIPT" in text
    assert '[[ "$MODE" == full ]]' in text
    assert "full comparison requires a valid smoke receipt" in text
    assert "smoke mode does not accept a smoke receipt" in text


def test_container_wrapper_cleans_up_policy_bridge_and_exact_vm_is_operator_owned() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "trap cleanup EXIT INT TERM" in text
    assert "kill" in text
    assert "nebius" not in text.lower()
    assert "computeinstance-" not in text


def test_runbook_separates_execution_publication_and_provider_stop() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "infrastructure_invalid" in text
    assert "anonymous byte readback" in text
    assert "publication is explicit" in text.lower()
    assert "stop the exact VM" in text
    assert "1,000-rollout" in text
