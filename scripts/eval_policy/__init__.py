"""
LeHome Challenge Policy Module

This module provides the base policy interface and implementations
for the LeHome Challenge evaluation framework.
"""

from .base_policy import BasePolicy
from .registry import PolicyRegistry

# Import policy implementations (this will auto-register them)
try:
    from .lerobot_policy import LeRobotPolicy
except ModuleNotFoundError as error:
    # GR00T inference intentionally runs in a small environment without the
    # optional LeRobot stack.  Only that absent top-level package is optional;
    # a partial/broken LeRobot install must still fail loudly.
    if error.name != "lerobot":
        raise
    _LEROBOT_AVAILABLE = False
else:
    _LEROBOT_AVAILABLE = True
from .example_participant_policy import CustomPolicy
from .docker_policy import DockerPolicy
from .groot_policy import GrootPolicy, GrootServerPolicy

__all__ = [
    "BasePolicy",
    "PolicyRegistry",
    "CustomPolicy",
    "DockerPolicy",
    "GrootPolicy",
    "GrootServerPolicy",
]
if _LEROBOT_AVAILABLE:
    __all__.append("LeRobotPolicy")
