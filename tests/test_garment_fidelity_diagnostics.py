from __future__ import annotations

import ast
from pathlib import Path
import types

import numpy as np
import pytest

from lehome.flywheel.fidelity import ClothFidelityError, fidelity_receipt


def _cpu_reset_method():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_flywheel_reset_legacy_cpu_garment"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "np": np,
        "ClothFidelityError": ClothFidelityError,
        "fidelity_receipt": fidelity_receipt,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_flywheel_reset_legacy_cpu_garment"]


@pytest.mark.parametrize("configured_max_velocity", [float("nan"), float("inf"), 0.0, -1.0])
def test_cpu_reset_preserves_typed_fidelity_abort_for_invalid_velocity_limit(
    configured_max_velocity: float,
) -> None:
    reset = _cpu_reset_method()
    env = types.SimpleNamespace(
        _flywheel_legacy_cpu_reset_state=(
            np.asarray([[0.2, 0.1, 0.7]], dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        ),
        particle_config={
            "objects": {"particle_system": {"max_velocity": configured_max_velocity}}
        },
        _flywheel_cloth_arrays=lambda positions, velocities: (
            np.asarray(positions, dtype=np.float32),
            np.asarray(velocities, dtype=np.float32),
        ),
        object=types.SimpleNamespace(),
    )

    with pytest.raises(ClothFidelityError) as captured:
        reset(env)

    assert captured.value.code == "cloth_flight"
    assert captured.value.fidelity["cloth_flight"] is True
    assert captured.value.diagnostic == {"stage": "cached_reset_velocity"}
