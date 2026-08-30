from __future__ import annotations

from pathlib import Path


def _included_by_context_rules(path: str, rules: list[str]) -> bool:
    """Resolve this intentionally small, anchored B1K context allow-list."""

    included = True
    for rule in rules:
        allow = rule.startswith("!")
        pattern = rule[1:] if allow else rule
        if pattern == "**":
            matched = True
        elif pattern.endswith("/**"):
            prefix = pattern.removesuffix("/**")
            matched = path == prefix or path.startswith(prefix + "/")
        elif pattern.endswith("/"):
            prefix = pattern.removesuffix("/")
            if prefix.startswith("**/"):
                matched = prefix.removeprefix("**/") in path.split("/")
            else:
                matched = path == prefix or path.startswith(prefix + "/")
        elif pattern == "**/*.py[cod]":
            matched = path.endswith((".pyc", ".pyo", ".pyd"))
        else:
            matched = path == pattern
        if matched:
            included = allow
    return included


def test_root_build_context_includes_exact_runtime_assets_and_excludes_unrelated_trainer_content() -> None:
    root = Path(__file__).parents[3]
    ignored = (root / ".dockerignore").read_text(encoding="utf-8")
    dockerfile = (root / "trainer" / "Dockerfile").read_text(encoding="utf-8")

    for path in (
        "!trainer/Dockerfile",
        "!trainer/pyproject.toml",
        "!trainer/uv.lock",
        "!trainer/src/**",
        "!trainer/config/**",
        "!trainer/scripts/verify-b1k-cli.py",
        "!trainer/b1k_launchkit/onstart.sh",
        "!trainer/b1k_launchkit/token_bootstrap.py",
        "!trainer/b1k_launchkit/training_smoke.py",
        "!trainer/b1k_launchkit/bucket_helper/b1k-bucket-helper",
        "!trainer/docker/entrypoint.sh",
    ):
        assert path in ignored
    assert "!trainer/tests/**" not in ignored
    for source in (
        "COPY trainer/src ./src",
        "COPY trainer/config ./config",
        "COPY trainer/scripts/verify-b1k-cli.py /usr/local/bin/verify-b1k-cli",
        "COPY trainer/b1k_launchkit/onstart.sh /opt/b1k-launchkit/onstart.sh",
        "COPY trainer/b1k_launchkit/token_bootstrap.py /opt/b1k-launchkit/token_bootstrap.py",
        "COPY trainer/b1k_launchkit/training_smoke.py /opt/b1k-launchkit/training_smoke.py",
        "COPY trainer/b1k_launchkit/bucket_helper/pyproject.toml trainer/b1k_launchkit/bucket_helper/uv.lock /opt/b1k-bucket-helper-src/",
        "COPY trainer/b1k_launchkit/bucket_helper/b1k-bucket-helper /opt/b1k-bucket-helper-src/b1k-bucket-helper",
        "COPY trainer/docker/entrypoint.sh /usr/local/bin/lehome-entrypoint",
    ):
        assert source in dockerfile


def test_pinned_official_source_contract_requires_the_episode_loader_module() -> None:
    root = Path(__file__).parents[3]
    required = "gr00t/data/dataset/lerobot_episode_loader.py"
    obsolete = "gr00t/data/dataset/lerobot_v3_dataset.py"

    for path in (
        root / "trainer" / "Dockerfile",
        root / "trainer" / "src" / "lehome_train" / "b1k" / "launch.py",
        root / "trainer" / "scripts" / "verify-image.sh",
    ):
        source = path.read_text(encoding="utf-8")
        assert required in source
        assert obsolete not in source


def test_context_allow_list_is_explicit_and_rejects_local_cache_or_bytecode_at_every_depth() -> None:
    root = Path(__file__).parents[3]
    rules = [line for line in (root / ".dockerignore").read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    runtime_inputs = {
        "trainer/Dockerfile",
        "trainer/pyproject.toml",
        "trainer/uv.lock",
        "trainer/src/lehome_train/b1k/launch.py",
        "trainer/config/lehome_four_types_mapping.json",
        "trainer/scripts/verify-b1k-cli.py",
        "trainer/b1k_launchkit/onstart.sh",
        "trainer/b1k_launchkit/token_bootstrap.py",
        "trainer/b1k_launchkit/training_smoke.py",
        "trainer/b1k_launchkit/bucket_helper/b1k-bucket-helper",
        "trainer/b1k_launchkit/bucket_helper/pyproject.toml",
        "trainer/b1k_launchkit/bucket_helper/uv.lock",
        "trainer/docker/entrypoint.sh",
    }
    assert all(_included_by_context_rules(path, rules) for path in runtime_inputs)
    excluded = {
        "trainer/tests/b1k/test_launch.py",
        "trainer/scripts/build-image.sh",
        "trainer/b1k_launchkit/README.md",
        "trainer/src/lehome_train/b1k/__pycache__/launch.cpython-310.pyc",
        "trainer/b1k_launchkit/__pycache__/onstart.cpython-310.pyc",
        "trainer/b1k_launchkit/bucket_helper/.pytest_cache/v/cache/nodeids",
        "trainer/b1k_launchkit/bucket_helper/local-garbage.pyc",
    }
    assert not any(_included_by_context_rules(path, rules) for path in excluded)
