"""Static contracts for the controller image's provider-auth boundary."""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def test_controller_capacity_daemon_uses_only_systemd_credential_backed_nebius_auth() -> None:
    installer = (_ROOT / "infrastructure/nebius/packer/scripts/install-controller.sh").read_text(encoding="utf-8")
    unit = (_ROOT / "infrastructure/nebius/guest/systemd/lehome-experiment-capacity.service").read_text(encoding="utf-8")
    wrapper = (_ROOT / "infrastructure/nebius/guest/bin/lehome-experiment-capacity.sh").read_text(encoding="utf-8")

    assert "NEBIUS_CLI_VERSION=0.12.263" in installer
    assert "storage.eu-north1.nebius.cloud/cli/install.sh" in installer
    assert "lehome-experiment-capacity" in installer
    assert "LoadCredential=nebius-private-key:/etc/lehome/private/nebius-private-key" in unit
    assert "LoadCredential=controller-token:/etc/lehome/private/controller-token" in unit
    assert "User=root" in unit
    assert "ProtectSystem=strict" in unit
    assert "PrivateTmp=true" in unit
    assert "unset NEBIUS_CONFIG NEBIUS_PROFILE NEBIUS_TOKEN" in wrapper
    assert "nebius --config \"${LEHOME_CAPACITY_NEBIUS_CONFIG_FILE}\" profile create" in wrapper
    assert "--nebius-config-file \"${LEHOME_CAPACITY_NEBIUS_CONFIG_FILE}\"" in wrapper


def test_bootstrap_requires_a_root_only_nebius_private_key_for_capacity_mutations() -> None:
    bootstrap = (_ROOT / "infrastructure/nebius/tools/bootstrap-experiment-pool.sh").read_text(encoding="utf-8")

    assert "--nebius-private-key-file" in bootstrap
    assert "NEBIUS_PRIVATE_KEY_FILE" in bootstrap
    assert "/etc/lehome/private/nebius-private-key" in bootstrap
    assert "lehome-experiment-capacity.service" in bootstrap
    assert "LEHOME_CAPACITY_NEBIUS_CONFIG_FILE=/run/lehome-capacity/nebius-config.yaml" in bootstrap
