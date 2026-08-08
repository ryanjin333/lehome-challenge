import argparse
import importlib.util
import sys
import types
from pathlib import Path


COMMON_MODULE = Path(__file__).parents[1] / "scripts" / "utils" / "common.py"


def test_launch_app_caps_renderer_to_one_gpu(monkeypatch):
    launches = []
    simulation_app = object()

    class AppLauncher:
        def __init__(self, settings):
            launches.append(settings)
            self.app = simulation_app

    isaaclab = types.ModuleType("isaaclab")
    isaaclab_app = types.ModuleType("isaaclab.app")
    isaaclab_app.AppLauncher = AppLauncher
    isaacsim = types.ModuleType("isaacsim")
    isaacsim_app = types.ModuleType("isaacsim.simulation_app")
    isaacsim_app.SimulationApp = object
    torch = types.ModuleType("torch")
    torch.Tensor = object
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", isaaclab_app)
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
    monkeypatch.setitem(sys.modules, "isaacsim.simulation_app", isaacsim_app)
    monkeypatch.setitem(sys.modules, "torch", torch)

    spec = importlib.util.spec_from_file_location("common_under_test", COMMON_MODULE)
    assert spec is not None and spec.loader is not None
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)

    args = argparse.Namespace(headless=True, device="cuda:0")
    result = common.launch_app_from_args(args)

    assert result is simulation_app
    assert launches == [vars(args)]
    assert "--/renderer/multiGpu/maxGpuCount=1" in args.kit_args
    assert args.enable_cameras is True
    assert args.experience.endswith(
        "third_party/IsaacLab/apps/isaaclab.python.headless.rendering.kit"
    )
