#!/usr/bin/env python3
"""Remote half of immutable LeHome runtime-code staging (Linux only)."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


BASE = Path("/mnt/lehome/runtime-code")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _safe_stage(path: Path, *, base: Path) -> Path:
    expected_prefix = ".runtime-code-stage."
    if (
        not path.is_absolute() or path.parent != base or not path.name.startswith(expected_prefix)
        or not re.fullmatch(r"\.runtime-code-stage\.[A-Za-z0-9]{8,}", path.name)
        or path.is_symlink() or not path.is_dir()
    ):
        raise ValueError("remote staging directory is unsafe")
    return path


def _run(*argv: str) -> None:
    subprocess.run(argv, check=True, text=True, capture_output=True)


def _sha256(path: Path) -> str:
    if not _regular(path):
        raise ValueError("remote bundle is unsafe")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checkout(checkout: Path, *, revision: str) -> None:
    if checkout.is_symlink() or not checkout.is_dir() or (checkout / ".git").is_symlink() or not (checkout / ".git").is_dir():
        raise ValueError("runtime code checkout is unsafe")
    head = subprocess.run(("git", "-C", str(checkout), "rev-parse", "HEAD"), text=True, capture_output=True, check=True).stdout.strip()
    if head != revision:
        raise ValueError("runtime code final path already exists but is not exact")
    _run("git", "-C", str(checkout), "diff", "--quiet")
    if subprocess.run(("git", "-C", str(checkout), "status", "--porcelain"), text=True, capture_output=True, check=True).stdout:
        raise ValueError("runtime code final path already exists but is not exact")
    for relative in ("source/lehome", "trainer/src", "scripts", "rollout_appliance"):
        path = checkout / relative
        if path.is_symlink() or not path.is_dir():
            raise ValueError("runtime code checkout is incomplete")
    config = checkout / "configs/eval_groot_n17_public_280.json"
    if not _regular(config):
        raise ValueError("runtime code checkout is incomplete")


def rename_noreplace(source: Path, destination: Path) -> bool:
    """Atomically rename source only when destination does not exist.

    Returns False strictly for EEXIST. Ubuntu x86_64 provides syscall 316;
    any other platform fails closed rather than falling back to a racy rename.
    """
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        raise OSError(errno.ENOSYS, "renameat2 RENAME_NOREPLACE requires Linux x86_64")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(316, -100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return True
    code = ctypes.get_errno()
    if code == errno.EEXIST:
        return False
    raise OSError(code, os.strerror(code))


def stage_runtime_code(*, revision: str, bundle_sha256: str, stage: Path, base: Path = BASE, check_mount: bool = True) -> dict[str, str | int]:
    if _REVISION.fullmatch(revision) is None or _SHA256.fullmatch(bundle_sha256) is None:
        raise ValueError("remote identifiers are invalid")
    if check_mount:
        _run("mountpoint", "-q", "/mnt/lehome")
    if base.is_symlink() or not base.is_dir():
        raise ValueError("runtime-code base is unsafe")
    stage = _safe_stage(stage, base=base)
    final = base / revision
    bundle = stage / "code.bundle"
    if _sha256(bundle) != bundle_sha256:
        raise ValueError("remote bundle digest mismatch")
    try:
        if final.exists() or final.is_symlink():
            verify_checkout(final, revision=revision)
            status = "existing_verified"
        else:
            repository = stage / "repository"; checkout = stage / "checkout"
            _run("git", "init", "-q", str(repository))
            _run("git", "-C", str(repository), "bundle", "verify", str(bundle))
            _run("git", "-C", str(repository), "fetch", "-q", str(bundle), revision)
            _run("git", "clone", "-q", "--no-checkout", str(repository), str(checkout))
            _run("git", "-C", str(checkout), "checkout", "--detach", "-q", revision)
            verify_checkout(checkout, revision=revision)
            if rename_noreplace(checkout, final):
                status = "staged"
            else:
                verify_checkout(final, revision=revision)
                status = "existing_verified"
        verify_checkout(final, revision=revision)
        tree = subprocess.run(("git", "-C", str(final), "rev-parse", "HEAD^{tree}"), text=True, capture_output=True, check=True).stdout.strip()
        return {"schema_version": 1, "kind": "lehome_runtime_code_stage_v1", "revision": revision, "path": str(final), "bundle_sha256": bundle_sha256, "tree": tree, "status": status}
    finally:
        # The caller allocated this exact validated helper-owned directory.
        shutil.rmtree(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--stage", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(stage_runtime_code(revision=arguments.revision, bundle_sha256=arguments.bundle_sha256, stage=arguments.stage), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
