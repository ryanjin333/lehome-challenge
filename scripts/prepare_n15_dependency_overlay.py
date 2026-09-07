"""Load authenticated inference wheels into the container's executable RAM mount."""
from pathlib import Path
import subprocess
import zipfile

from scripts.verify_native_reference_evaluator_gate import (
    FLASH_ATTENTION_WHEEL_PATH,
    PUBLIC_PYPROJECT_DEPENDENCY_WHEELS,
    inspect_flash_attention_overlay,
    inspect_public_pyproject_dependencies_overlay,
)


def materialize(target: Path) -> None:
    # Authenticate bytes, zip members and metadata before creating output.
    inspect_flash_attention_overlay()
    inspect_public_pyproject_dependencies_overlay()
    target.mkdir(mode=0o700)
    wheels = [FLASH_ATTENTION_WHEEL_PATH, *(
        Path(spec["wheel_path"]) for spec in PUBLIC_PYPROJECT_DEPENDENCY_WHEELS
    )]
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(target)


if __name__ == "__main__":
    mount = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE,OPTIONS", "--mountpoint", "/flash"], text=True,
    ).split()
    if len(mount) != 2 or mount[0] != "tmpfs" or "noexec" in mount[1].split(","):
        raise SystemExit("dependency overlay requires an executable tmpfs at /flash")
    materialize(Path("/flash/site-packages"))
