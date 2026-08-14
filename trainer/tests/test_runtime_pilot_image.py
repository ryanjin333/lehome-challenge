from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "trainer" / "Dockerfile.runtime-pilot"
PROJECT = ROOT / "trainer" / "runtime-pilot" / "pyproject.toml"
LOCK = ROOT / "trainer" / "runtime-pilot" / "uv.lock"
WORKFLOW = ROOT / ".github" / "workflows" / "groot-trainer-image.yml"


def _workflow_job(workflow: str, job_name: str) -> str:
    match = re.search(
        rf"^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [a-z][a-z0-9-]+:\n|\Z)",
        workflow,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match, f"missing {job_name} job"
    return match.group(0)


def test_runtime_pilot_image_is_a_pinned_cpu_only_x86_contract() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    project = PROJECT.read_text(encoding="utf-8")
    lock = LOCK.read_text(encoding="utf-8")

    assert "FROM python:3.10.18-slim-bookworm@sha256:" in dockerfile
    assert "ARG UV_VERSION=" in dockerfile
    assert "ARG UV_SHA256=" in dockerfile
    assert "ARG DEBIAN_SNAPSHOT=" in dockerfile
    assert "snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}" in dockerfile
    assert "Acquire::Check-Valid-Until=false" in dockerfile
    assert "ARG ISAAC_GROOT_REVISION=23ace64f17aa5015259b8609d371eb61a357c776" in dockerfile
    assert "--no-install-recommends" in dockerfile
    assert all(package in dockerfile for package in ("ffmpeg", "git", "openssh-server", "tar"))
    assert "rm -f /etc/ssh/ssh_host_*_key" in dockerfile
    assert "lehome-sshd-wrapper" in dockerfile
    assert "lehome-entrypoint" in dockerfile
    assert "/opt/isaac-groot" in dockerfile
    assert "/opt/runtime" in dockerfile
    assert "torch==2.7.1+cpu" in project
    assert "torchvision==0.22.1+cpu" in project
    assert "torchcodec==0.4.0" in project
    assert re.search(r'name = "torch"\nversion = "2\.7\.1\+cpu"', lock)
    assert re.search(r'name = "torchvision"\nversion = "0\.22\.1\+cpu"', lock)
    assert re.search(r'name = "torchcodec"\nversion = "0\.4\.0"', lock)
    forbidden = ("nvidia/cuda", "isaac sim", "flash-attn", "tensorrt", "step-12000", "safetensors")
    haystack = f"{dockerfile}\n{project}\n{lock}".lower()
    assert not any(value in haystack for value in forbidden)


def test_runtime_pilot_publish_workflow_is_dispatch_only_and_preserves_gpu_tags() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runtime_job = _workflow_job(workflow, "publish-runtime-pilot")
    candidate_job = _workflow_job(workflow, "publish-candidate")
    gpu_job = _workflow_job(workflow, "gpu-structural")

    assert "publish_runtime_pilot:" in workflow
    assert 'description: "Publish the immutable CPU runtime-pilot image to GHCR"' in workflow
    assert "type: boolean" in workflow
    assert "default: false" in workflow
    assert "github.event_name == 'workflow_dispatch' && inputs.publish_runtime_pilot" in runtime_job
    assert "runs-on: ubuntu-22.04" in runtime_job
    assert "contents: read" in runtime_job
    assert "packages: write" in runtime_job
    assert "actions/checkout@11d5960a326750d5838078e36cf38b85af677262" in runtime_job
    assert "docker/setup-buildx-action@8d2750c68a42422c14e847fe6c8ac0403b4cbd6f" in runtime_job
    assert "docker/login-action@c94ce9fb468520275223c153574b00df6fe4bcc9" in runtime_job
    assert "file: trainer/Dockerfile.runtime-pilot" in runtime_job
    assert "context: ." in runtime_job
    assert "platforms: linux/amd64" in runtime_job
    assert "push: true" in runtime_job
    assert "type=raw,value=runtime-pilot-${{ github.sha }}" in runtime_job
    assert "REPOSITORY_COMMIT=${{ github.sha }}" in runtime_job
    assert "digest: ${{ steps.build.outputs.digest }}" in runtime_job
    assert "image: ${{ env.IMAGE_REPOSITORY }}" in runtime_job
    assert "docker pull \"${IMAGE_REPOSITORY}@${OCI_DIGEST}\"" in runtime_job
    assert "--entrypoint /bin/bash" in runtime_job
    assert 'dpkg --print-architecture)" = "amd64"' in runtime_job
    assert "sys.version_info[:3] == (3, 10, 18)" in runtime_job
    assert 'torch.__version__ == "2.7.1+cpu"' in runtime_job
    assert "torch.version.cuda is None" in runtime_job
    assert 'torchvision.__version__ == "0.22.1+cpu"' in runtime_job
    assert 'torchcodec.__version__ == "0.4.0"' in runtime_job
    assert "23ace64f17aa5015259b8609d371eb61a357c776" in runtime_job
    assert "bdc3cf8a6b9c92a0e3f46d79dc3de05b0a8c70c4289d44ac0aa1c75698a93f31" in runtime_job
    assert "ffmpeg -version" in runtime_job and "ffprobe -version" in runtime_job
    assert "lehome-train --help" in runtime_job
    assert "-name '*.safetensors'" in runtime_job
    assert "-name '*.ckpt'" in runtime_job
    assert "org.opencontainers.image.revision" in runtime_job
    assert "runtime-pilot-image-${{ github.sha }}" in runtime_job
    assert "runtime-pilot-image.txt" in runtime_job
    assert "runtime-pilot-${{ github.sha }}" in runtime_job

    assert "github.event_name == 'push'" in candidate_job
    assert "(github.event_name == 'workflow_dispatch' && inputs.publish)" in candidate_job
    assert "file: trainer/Dockerfile" in candidate_job
    assert "type=raw,value=${{ github.sha }}" in candidate_job
    assert "runtime-pilot-${{ github.sha }}" not in candidate_job
    assert "github.event_name == 'workflow_dispatch' && inputs.publish && inputs.run_gpu" in gpu_job
    assert "needs: publish-candidate" in gpu_job
