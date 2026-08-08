import builtins
import importlib.util
import sys
import types
from pathlib import Path


DEVICES_INIT = (
    Path(__file__).parents[1] / "source" / "lehome" / "lehome" / "devices" / "__init__.py"
)


def test_devices_package_does_not_import_hardware_controllers(monkeypatch):
    """Headless task imports must not initialize pynput-backed controllers."""
    lehome_package = types.ModuleType("lehome")
    lehome_package.__path__ = [str(DEVICES_INIT.parents[1])]
    device_base_module = types.ModuleType("lehome.devices.device_base")
    device_base = object()
    device_base_module.DeviceBase = device_base

    monkeypatch.setitem(sys.modules, "lehome", lehome_package)
    monkeypatch.setitem(sys.modules, "lehome.devices.device_base", device_base_module)

    original_import = builtins.__import__

    def reject_hardware_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name in {"lerobot", "keyboard"}:
            raise AssertionError(f"eager hardware import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_hardware_import)

    spec = importlib.util.spec_from_file_location(
        "lehome.devices",
        DEVICES_INIT,
        submodule_search_locations=[str(DEVICES_INIT.parent)],
    )
    assert spec is not None and spec.loader is not None
    devices_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "lehome.devices", devices_module)

    spec.loader.exec_module(devices_module)

    assert devices_module.DeviceBase is device_base

