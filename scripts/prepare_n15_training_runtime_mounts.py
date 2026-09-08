"""Resolve narrowly scoped, read-only mounts for a verified training Python."""
from pathlib import Path
import json
import subprocess
import sys


def runtime_mounts(
    executable: Path, prefix: Path, base_prefix: Path, *,
    allowed_root: Path = Path('/mnt/lehome/public-n15-tools'),
    allowed_base: Path = Path('/home/ubuntu/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu'),
) -> list[str]:
    roots = list(dict.fromkeys((prefix, base_prefix)))
    for root in roots:
        if (not root.is_absolute() or root == allowed_root
                or not (root.is_relative_to(allowed_root)
                        or (root == base_prefix == allowed_base and root != prefix))
                or any(char in str(root) for char in ',\n\r')):
            raise ValueError('training runtime mount is outside the allowed tools subtree')
        if root.is_symlink() or root.resolve(strict=True) != root or not root.is_dir():
            raise ValueError('training runtime mount must be a real canonical directory')
    if executable != prefix / 'bin/python':
        raise ValueError('training executable does not match the virtual environment')
    resolved = executable.resolve(strict=True)
    if not resolved.is_relative_to(base_prefix) or not resolved.is_file():
        raise ValueError('training executable target is outside the base runtime')
    return [f'type=bind,src={root},dst={root},readonly' for root in roots]


def mounts_for_interpreter(executable: Path) -> list[str]:
    if executable == Path('/opt/lehome-challenge/.venv/bin/python'):
        return []  # Already present in the pinned rollout image.
    if executable != Path('/mnt/lehome/public-n15-tools/venv/bin/python'):
        raise ValueError('unexpected training interpreter path')
    probe = subprocess.run(
        [str(executable), '-I', '-c',
         'import json,sys; print(json.dumps([sys.prefix,sys.base_prefix]))'],
        check=True, capture_output=True, text=True, timeout=10,
    )
    prefix, base_prefix = map(Path, json.loads(probe.stdout))
    return runtime_mounts(executable, prefix, base_prefix)


if __name__ == '__main__':
    # The caller first performs full host-side training identity validation.
    receipt = json.loads(Path(sys.argv[1]).read_text())
    for mount in mounts_for_interpreter(Path(receipt['python_executable'])):
        print(mount)
