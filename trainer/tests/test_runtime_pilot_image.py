from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "trainer" / "Dockerfile.runtime-pilot"
PROJECT = ROOT / "trainer" / "runtime-pilot" / "pyproject.toml"
LOCK = ROOT / "trainer" / "runtime-pilot" / "uv.lock"


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
