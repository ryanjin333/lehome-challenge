"""Teleoperation devices.

Hardware-backed controllers are loaded lazily so headless evaluation can import
``lehome.devices.action_process`` without initializing keyboard or X11 input.
"""

from importlib import import_module
import os

from .device_base import DeviceBase


_LAZY_IMPORTS = {
    "SO101Leader": (".lerobot", "SO101Leader"),
    "BiSO101Leader": (".lerobot", "BiSO101Leader"),
    "Se3Keyboard": (".keyboard", "Se3Keyboard"),
    "BiKeyboard": (".keyboard", "BiKeyboard"),
}
_KEYBOARD_DEVICES = {"Se3Keyboard", "BiKeyboard"}

__all__ = ["DeviceBase", *_LAZY_IMPORTS]


def __getattr__(name: str):
    """Load an optional controller only when a caller requests it."""
    lazy_import = _LAZY_IMPORTS.get(name)
    if lazy_import is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _KEYBOARD_DEVICES and os.environ.get("LEHOME_DISABLE_KEYBOARD") == "1":
        raise AttributeError(
            f"{name} is disabled because LEHOME_DISABLE_KEYBOARD=1"
        )

    module_name, attribute_name = lazy_import
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
