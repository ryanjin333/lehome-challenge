from __future__ import annotations

import pytest
from typer.testing import CliRunner

from lehome_train.cli import app
from lehome_train.constants import DEFAULT_SETTINGS


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "LeHome GR00T N1.7 trainer" in result.stdout


@pytest.mark.parametrize(
    "command",
    ["data", "prepare", "memorize", "smoke", "train", "report", "sync"],
)
def test_cli_registers_command_group(command: str) -> None:
    result = CliRunner().invoke(
        app,
        [command, "--help"],
        prog_name="lehome-train",
    )

    assert result.exit_code == 0
    assert f"Usage: lehome-train {command}" in result.stdout


def test_settings_use_exact_immutable_pins() -> None:
    settings = DEFAULT_SETTINGS

    assert settings.python_version == "3.10.18"
    assert (
        settings.cuda_base_digest
        == "sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719"
    )
    assert settings.isaac_groot_revision == "23ace64f17aa5015259b8609d371eb61a357c776"
    assert settings.model_revision == "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
    assert settings.data_repo == "ryanjin333/lehome-groot-n17-data"
    assert settings.model_repo == "ryanjin333/lehome-groot-n17-models"

    with pytest.raises(AttributeError):
        settings.model_revision = "explicit-override"
