from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_vast_ssh_base_stays_root_while_entrypoint_drops_to_trainer() -> None:
    dockerfile = (REPOSITORY_ROOT / "trainer" / "Dockerfile").read_text(encoding="utf-8")
    entrypoint = (REPOSITORY_ROOT / "trainer" / "docker" / "entrypoint.sh").read_text(
        encoding="utf-8"
    )

    users = [
        line.split(maxsplit=1)[1]
        for line in dockerfile.splitlines()
        if line.startswith("USER ")
    ]

    assert users[-1] == "root", "Vast must be able to derive its SSH layer as root"
    assert "--reuid=10001" in entrypoint
    assert "--regid=10001" in entrypoint
    assert 'exec "${drop_privileges[@]}" /usr/bin/env HOME=/nonexistent "$@"' in entrypoint
