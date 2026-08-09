"""No-follow, atomic Hugging Face token-file installation for trainer onstart."""

from __future__ import annotations

import argparse
import errno
import os
import stat
import sys
from pathlib import Path


DEFAULT_TOKEN_PATH = Path("/workspace/.cache/huggingface/token")


class TokenBootstrapError(RuntimeError):
    """The persistent token path cannot be safely replaced."""


def install_token(token: str, *, token_path: Path = DEFAULT_TOKEN_PATH, uid: int = 10001, gid: int = 10001) -> None:
    """Atomically replace a token path without following any path symlink."""

    if not token or not token_path.is_absolute() or token_path.name != "token":
        raise TokenBootstrapError("invalid production token path")
    parent_fd = _open_parent_no_follow(token_path.parent, uid=uid, gid=gid)
    temporary_name = f".token.{os.getpid()}.{os.urandom(8).hex()}"
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as writer:
                writer.write(token)
                writer.flush()
                os.fsync(writer.fileno())
            os.chown(temporary_name, uid, gid, dir_fd=parent_fd, follow_symlinks=False)
            os.replace(temporary_name, token_path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
    except OSError as error:
        raise TokenBootstrapError("unsafe token parent or atomic token install failure") from error
    finally:
        os.close(parent_fd)


def _open_parent_no_follow(path: Path, *, uid: int, gid: int) -> int:
    if not path.is_absolute():
        raise TokenBootstrapError("unsafe token parent")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                if error.errno != errno.ENOENT:
                    raise TokenBootstrapError("unsafe token parent") from error
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                    os.chown(component, uid, gid, dir_fd=descriptor, follow_symlinks=False)
                    next_descriptor = os.open(component, flags, dir_fd=descriptor)
                except OSError as create_error:
                    raise TokenBootstrapError("unsafe token parent") from create_error
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & 0o077:
            raise TokenBootstrapError("unsafe token parent")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(prog="b1k-trainer-token-bootstrap")
    parser.parse_args()
    try:
        install_token(sys.stdin.read())
    except TokenBootstrapError as error:
        print(f"b1k-trainer-token-bootstrap: {error}", file=sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
