from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


_REPOSITORY = Path(__file__).resolve().parents[1]
_LOADER_SOURCE = (
    _REPOSITORY
    / "source/lehome/lehome/tasks/bedroom/challenge_garment_loader.py"
)
_ENVIRONMENT_SOURCE = (
    _REPOSITORY / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
)


def _loader_class(monkeypatch: pytest.MonkeyPatch):
    class OmegaConf:
        @staticmethod
        def load(path: str | Path):
            return types.SimpleNamespace(**json.loads(Path(path).read_text(encoding="utf-8")))

    monkeypatch.setitem(
        sys.modules,
        "omegaconf",
        types.SimpleNamespace(OmegaConf=OmegaConf, DictConfig=types.SimpleNamespace),
    )
    spec = importlib.util.spec_from_file_location(
        "challenge_garment_loader_under_test", _LOADER_SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ChallengeGarmentLoader


def _write_garment_config(root: Path, *, asset_path: str, visual_paths: list[str]) -> None:
    garment = root / "Release/Top_Short/Top_Short_Seen_1"
    garment.mkdir(parents=True)
    (garment / "config.json").write_text(
        json.dumps(
            {
                "asset_path": asset_path,
                "visual_usd_paths": visual_paths,
            }
        ),
        encoding="utf-8",
    )


def test_loader_rejects_missing_primary_usd_before_isaac_can_open_a_file_picker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    asset_root = tmp_path / "Assets/objects/Challenge_Garment"
    _write_garment_config(
        asset_root,
        asset_path="/Assets/garments/missing.usd",
        visual_paths=[],
    )

    loader = _loader_class(monkeypatch)(str(asset_root))

    with pytest.raises(FileNotFoundError, match="garment asset"):
        loader.load_garment_config("Top_Short_Seen_1")


def test_loader_rejects_missing_visual_usd_before_isaac_asset_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    asset_root = tmp_path / "Assets/objects/Challenge_Garment"
    primary = tmp_path / "Assets/garments/top-short.usd"
    primary.parent.mkdir(parents=True)
    primary.write_text("#usda 1.0\n", encoding="utf-8")
    _write_garment_config(
        asset_root,
        asset_path="/Assets/garments/top-short.usd",
        visual_paths=["/Assets/materials/missing.usd"],
    )

    loader = _loader_class(monkeypatch)(str(asset_root))

    with pytest.raises(FileNotFoundError, match="visual USD"):
        loader.load_garment_config("Top_Short_Seen_1")


def test_loader_accepts_existing_primary_and_visual_usds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    asset_root = tmp_path / "Assets/objects/Challenge_Garment"
    for relative in ("garments/top-short.usd", "materials/top-short.usd"):
        path = tmp_path / "Assets" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#usda 1.0\n", encoding="utf-8")
    _write_garment_config(
        asset_root,
        asset_path="/Assets/garments/top-short.usd",
        visual_paths=["/Assets/materials/top-short.usd"],
    )

    config = _loader_class(monkeypatch)(str(asset_root)).load_garment_config(
        "Top_Short_Seen_1"
    )

    assert config.asset_path == "/Assets/garments/top-short.usd"


def test_switch_garment_validates_replacement_before_deleting_current_object() -> None:
    source = _ENVIRONMENT_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "switch_garment"
    )
    calls = sorted(
        (
            node.lineno,
            node.func.attr,
        )
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"load_garment_config", "_delete_garment_object"}
    )

    assert [name for _, name in calls][:2] == [
        "load_garment_config",
        "_delete_garment_object",
    ]
