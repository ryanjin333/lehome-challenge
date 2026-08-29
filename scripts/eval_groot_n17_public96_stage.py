"""Stage entrypoint that installs the raw checker before importing Isaac evaluation."""

from __future__ import annotations

import argparse
import json
import sys

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
    original_setup = evaluator.setup_eval_parser

    def scoped_setup():
        value = original_setup()
        value.set_defaults(
            policy_server_endpoint=parsed.policy_server_endpoint,
            policy_server_token_env=parsed.policy_server_token_env,
            policy_server_request_timeout=parsed.policy_server_request_timeout,
        )
        return value

    evaluator.setup_eval_parser = scoped_setup
    completed = False
    from scripts.utils import evaluation as evaluation_module
    original_eval = evaluation_module.eval

    def checked_eval(*args, **kwargs):
        nonlocal completed
        result = original_eval(*args, **kwargs)
        completed = True
        return result

    evaluation_module.eval = checked_eval

    previous = sys.argv
    try:
        sys.argv = [previous[0], *remaining]
        evaluator.main()
        if not completed:
            raise SystemExit("public96 evaluator did not complete; swallowed evaluator error")
        print("PUBLIC96_STAGE_COMPLETE " + json.dumps({"raw_checker_overlay": installed, "runtime_policy_sha256": parsed.public96_runtime_policy_sha256}, sort_keys=True))
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
