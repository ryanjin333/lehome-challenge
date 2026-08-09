"""One-step, no-op-policy OmniGibson infrastructure smoke for the pinned image."""

from __future__ import annotations

import argparse
import io
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


_PROBE_BYTES = b'{"purpose":"b1k-private-release-bootstrap"}\n'
_TASK = "turning_on_radio"
_ROBOT_CONFIG = "/behavior-src/OmniGibson/omnigibson/eval/r1pro.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="b1k-rollout smoke-runtime")
    parser.add_argument("--success-prefix", required=True)
    parser.add_argument("--failure-prefix", required=True)
    args = parser.parse_args(argv)
    _prefix(args.success_prefix); _prefix(args.failure_prefix)
    if os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES":
        raise RuntimeError("Omni EULA is not accepted")
    token_file = Path(os.environ.get("B1K_HF_TOKEN_FILE", ""))
    metadata = token_file.stat()
    if token_file.is_symlink() or not token_file.is_file() or stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_uid != os.getuid():
        raise RuntimeError("runtime token file is invalid")

    from omegaconf import OmegaConf
    from omnigibson.eval.evaluator import Evaluator, resolve_instance_ids
    from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
    from omnigibson.macros import gm

    gm.HEADLESS = True
    robot = OmegaConf.load(_ROBOT_CONFIG)
    cfg = OmegaConf.create({
        "env_wrapper": {"_target_": "omnigibson.eval.wrappers.DefaultWrapper"},
        "policy_name": "local",
        "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
        "headless": True, "partial_scene_load": True, "max_steps": 1, "write_video": False,
        "mode": "public_test", "seed": seed_everything(DEFAULT_EVAL_SEED), "task": {"name": _TASK}, "robot": robot,
    })
    instance_id = resolve_instance_ids(_TASK, [0], mode="public_test")[0]
    with Evaluator(cfg) as evaluator:
        evaluator.reset()
        evaluator.load_task_instance(int(instance_id))
        evaluator.reset()
        rgb_count = sum(1 for key in evaluator.obs if isinstance(key, str) and key.endswith("::rgb"))
        if rgb_count < 1:
            raise RuntimeError("official evaluator reset returned no RGB observation")
        terminated, truncated = evaluator.step()
        action_count = _action_dimension(evaluator.robot_action)
        if action_count < 1:
            raise RuntimeError("official local policy produced no mapped robot action")
        if not (terminated or truncated):
            raise RuntimeError("one-step official evaluator did not terminate or truncate")

    import warp
    try:
        Path(warp.__file__).resolve().relative_to(Path(sys.prefix).resolve())
    except (AttributeError, ValueError):
        raise RuntimeError("Warp did not resolve from the bundled behavior runtime") from None

    token = token_file.read_text(encoding="utf-8").strip()
    from huggingface_hub import CommitOperationAdd, HfApi
    hub = HfApi(token=token)
    commit = _upload_fixtures_atomically(hub, args.success_prefix, args.failure_prefix, token, CommitOperationAdd)
    commits = {"success-fixture": commit, "failure-fixture": commit}
    print(json.dumps({
        "gpu_count": _cuda_count(), "eula_environment": "OMNI_KIT_ACCEPT_EULA=YES", "warp_runtime": "bundled-compatible",
        "headless_loads": 1, "resets": 2, "rgb_observation_count": rgb_count, "action_mapping_count": action_count,
        "evaluator_outcome": "terminal" if terminated or truncated else "quarantined", "remote_probe_upload_commits": commits,
        "infrastructure_smoke": True, "container_digest": os.environ.get("CONTAINER_DIGEST", ""),
    }, sort_keys=True, separators=(",", ":")))
    return 0


def _upload_fixtures_atomically(hub: Any, success_prefix: str, failure_prefix: str, token: str, operation_add: Any) -> str:
    result = hub.create_commit(
        repo_id="ryanjin333/behavior1k-groot-n17-rollouts", repo_type="dataset", token=token,
        commit_message="b1k rollout infrastructure smoke fixtures",
        operations=[
            operation_add(path_in_repo=f"{success_prefix}/probe.json", path_or_fileobj=io.BytesIO(_PROBE_BYTES)),
            operation_add(path_in_repo=f"{failure_prefix}/probe.json", path_or_fileobj=io.BytesIO(_PROBE_BYTES)),
        ],
    )
    commit = getattr(result, "oid", getattr(result, "commit_id", None))
    if not isinstance(commit, str) or len(commit) not in {40, 64}:
        raise RuntimeError("atomic remote fixture upload did not return an immutable commit")
    return commit


def _action_dimension(action: object) -> int:
    shape = getattr(action, "numel", None)
    if callable(shape):
        return int(shape())
    if isinstance(action, dict):
        return sum(_action_dimension(value) for value in action.values())
    return 0


def _cuda_count() -> int:
    import torch
    return int(torch.cuda.device_count())


def _prefix(value: object) -> None:
    import re
    if not isinstance(value, str) or re.fullmatch(r"b1k-bootstrap-[0-9a-f]{32}-(?:success-fixture|failure-fixture)", value) is None:
        raise RuntimeError("runtime probe prefix is invalid")
