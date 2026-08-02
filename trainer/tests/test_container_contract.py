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


def test_vast_ssh_privilege_separation_account_exists() -> None:
    dockerfile = (REPOSITORY_ROOT / "trainer" / "Dockerfile").read_text(encoding="utf-8")

    assert "useradd --system --user-group --home-dir /run/sshd" in dockerfile
    assert "install -d -o root -g root -m 0755 /run/sshd" in dockerfile


def test_vast_ssh_host_keys_are_generated_per_instance() -> None:
    dockerfile = (REPOSITORY_ROOT / "trainer" / "Dockerfile").read_text(encoding="utf-8")
    wrapper_path = REPOSITORY_ROOT / "trainer" / "docker" / "sshd-wrapper.sh"

    assert wrapper_path.is_file()
    wrapper = wrapper_path.read_text(encoding="utf-8")

    assert "openssh-server" in dockerfile
    assert "dpkg-divert --local --rename --add /usr/sbin/sshd" in dockerfile
    assert "rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub" in dockerfile
    assert "/workspace/.cache/lehome-ssh-hostkeys" in wrapper
    assert "ssh-keygen -q -t" in wrapper
    assert "-m 0600" in wrapper
