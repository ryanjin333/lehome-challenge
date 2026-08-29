"""Stage entrypoint that installs the raw checker before importing Isaac evaluation."""

from __future__ import annotations

import argparse
import json
import os

from scripts.groot_n17_public96_raw_checker import RAW_CHECKER_OVERLAY_ID, install_raw_checker_overlay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--public96_raw_checker_overlay", required=True)
    parser.add_argument("--policy_server_endpoint", required=True)
    parser.add_argument("--policy_server_token_env", required=True)
    parser.add_argument("--policy_server_request_timeout", type=float, required=True)
    parser.add_argument("--public96_runtime_policy_sha256", required=True)
    parsed, remaining = parser.parse_known_args(argv)
    if parsed.public96_raw_checker_overlay != RAW_CHECKER_OVERLAY_ID:
        raise SystemExit("public96 raw checker overlay identity is invalid")
    if parsed.public96_runtime_policy_sha256 != "e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa":
        raise SystemExit("public96 runtime policy identity is invalid")
    installed = install_raw_checker_overlay()
    from scripts import eval as evaluator

    parser = evaluator.setup_eval_parser()
    evaluator.AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(
        policy_server_endpoint=parsed.policy_server_endpoint,
        policy_server_token_env=parsed.policy_server_token_env,
        policy_server_request_timeout=parsed.policy_server_request_timeout,
    )
    args = parser.parse_args(remaining)
    simulation_app = evaluator.launch_app_from_args(args)
    try:
        import lehome.tasks.bedroom
        from scripts.utils import evaluation as evaluation_module

        if getattr(args, "headless", False):
            os.environ["LEHOME_DISABLE_KEYBOARD"] = "1"
        evaluation_module.eval(args, simulation_app)
        print("PUBLIC96_STAGE_COMPLETE " + json.dumps({"raw_checker_overlay": installed, "runtime_policy_sha256": parsed.public96_runtime_policy_sha256}, sort_keys=True))
    finally:
        evaluator.common.close_app(simulation_app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
