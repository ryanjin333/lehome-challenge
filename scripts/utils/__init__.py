"""Utility functions for LeHome scripts."""

from importlib import import_module

from . import common
from . import parser

# Note: evaluation, dataset_record and dataset_replay are not imported at module level
# to avoid importing Isaac Sim modules before SimulationApp is launched.
# They should be imported lazily when needed (after SimulationApp is launched).

# Export commonly used functions for convenience
from .parser import (
    setup_record_parser,
    setup_replay_parser,
    setup_inspect_parser,
    setup_read_parser,
    setup_augment_parser,
    setup_merge_parser,
    setup_eval_parser,
)
from .common import launch_app, launch_app_from_args, close_app

# Dataset tooling is intentionally loaded only when it is used.  Rollouts import
# ``scripts.utils.common`` before Isaac launches and do not need pyarrow or
# LeRobot unless they explicitly select dataset operations.
_LAZY_DATASET_EXPORTS = {
    "dataset_inspection": ("dataset_inspection", None),
    "inspect": ("dataset_inspection", "inspect"),
    "read_states": ("dataset_inspection", "read_states"),
    "dataset_processing": ("dataset_processing", None),
    "augment_ee_pose": ("dataset_processing", "augment_ee_pose"),
    "merge_datasets": ("dataset_processing", "merge_datasets"),
    "merge_garment_info": ("dataset_processing", "merge_garment_info"),
}


def __getattr__(name: str):
    """Load optional dataset helpers without hiding their import failures."""
    try:
        module_name, attribute_name = _LAZY_DATASET_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    module = import_module(f".{module_name}", __name__)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value

# Note: evaluation functions are not imported at module level to avoid
# importing Isaac Sim modules before SimulationApp is launched.
# Import them lazily when needed: from .utils.evaluation import <function>

__all__ = [
    "setup_record_parser",
    "setup_replay_parser",
    "setup_inspect_parser",
    "setup_read_parser",
    "setup_augment_parser",
    "setup_merge_parser",
    "setup_eval_parser",
    "launch_app",
    "launch_app_from_args",
    "close_app",
    "inspect",
    "read_states",
    "augment_ee_pose",
    "merge_datasets",
    "merge_garment_info",
    # Note: evaluation functions, "replay" and "record_dataset" are not exported
    # at module level to avoid importing Isaac Sim modules before SimulationApp
    # is launched. Import them lazily when needed:
    #   from .utils.evaluation import <function>
    #   from .utils import dataset_replay, dataset_record
]
