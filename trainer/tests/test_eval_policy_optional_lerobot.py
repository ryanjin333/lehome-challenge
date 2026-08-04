from __future__ import annotations

import builtins
import importlib
from pathlib import Path
import sys
import types

import pytest


_PACKAGE_PREFIX = "scripts.eval_policy"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _clear_eval_policy_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in tuple(sys.modules):
        if (
            module_name == _PACKAGE_PREFIX
            or module_name.startswith(f"{_PACKAGE_PREFIX}.")
            or module_name == "lerobot"
            or module_name.startswith("lerobot.")
        ):
            monkeypatch.delitem(sys.modules, module_name, raising=False)


def _provide_minimal_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the LeRobot adapter reach its optional-package import in CPU CI."""

    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    monkeypatch.setitem(sys.modules, "torch", torch)


def _import_eval_policy_without_lerobot(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(_REPOSITORY_ROOT))
    _clear_eval_policy_modules(monkeypatch)
    _provide_minimal_torch(monkeypatch)
    original_import = builtins.__import__

    def import_without_lerobot(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "lerobot" or name.startswith("lerobot."):
            raise ModuleNotFoundError("No module named 'lerobot'", name="lerobot")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_lerobot)
    return importlib.import_module(_PACKAGE_PREFIX)


def test_groot_policy_remains_importable_when_lerobot_is_absent(monkeypatch):
    package = _import_eval_policy_without_lerobot(monkeypatch)
    groot_policy = importlib.import_module(f"{_PACKAGE_PREFIX}.groot_policy")

    assert package.PolicyRegistry.is_registered("custom")
    assert package.PolicyRegistry.is_registered("docker")
    assert package.PolicyRegistry.is_registered("groot")
    assert package.PolicyRegistry.is_registered("groot_server")
    assert groot_policy.GrootPolicy is package.GrootPolicy
    assert groot_policy.GrootServerPolicy is package.GrootServerPolicy
    assert groot_policy.PolicyServerClient is not None


def test_non_lerobot_module_not_found_error_propagates(monkeypatch):
    monkeypatch.syspath_prepend(str(_REPOSITORY_ROOT))
    _clear_eval_policy_modules(monkeypatch)
    _provide_minimal_torch(monkeypatch)
    original_import = builtins.__import__

    def import_with_broken_lerobot(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "lerobot" or name.startswith("lerobot."):
            raise ModuleNotFoundError("No module named 'transitive_dependency'", name="transitive_dependency")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_broken_lerobot)

    with pytest.raises(ModuleNotFoundError, match="transitive_dependency") as error:
        importlib.import_module(_PACKAGE_PREFIX)

    assert error.value.name == "transitive_dependency"
