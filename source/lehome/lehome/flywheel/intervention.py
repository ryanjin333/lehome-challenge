"""Explicit, one-way full-expert and DAgger intervention transitions."""

from __future__ import annotations

import math
from typing import Iterable

from .quality import QualityResult


class TransitionError(RuntimeError):
    """Raised when an unsafe or invalid collection transition is requested."""


def _finite_12(values: Iterable[object], name: str) -> tuple[float, ...]:
    result = tuple(values)
    if len(result) != 12 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in result):
        raise TransitionError(f"{name} must contain 12 finite values")
    return tuple(float(value) for value in result)


class InterventionController:
    """Owns action-source segments and forbids policy re-entry after takeover."""

    def __init__(self, *, mode: str, sync_tolerance_rad: float = 0.08) -> None:
        if mode not in {"practice", "expert", "dagger"}:
            raise ValueError("collection mode must be practice, expert, or dagger")
        if not math.isfinite(sync_tolerance_rad) or sync_tolerance_rad <= 0:
            raise ValueError("synchronization tolerance must be positive and finite")
        self.mode = mode
        self.sync_tolerance_rad = sync_tolerance_rad
        self.state = "ready"
        self.segment = 0
        self.policy_queue_clear_requested = False
        self._took_over = False

    @property
    def action_source(self) -> str:
        if self.state == "policy":
            return "policy"
        if self.state == "expert":
            return "expert"
        return "hold"

    @property
    def export_enabled(self) -> bool:
        return self.mode in {"expert", "dagger"} and self.state == "accepted"

    def start_policy(self) -> None:
        if self._took_over:
            raise TransitionError("DAgger takeover is one-way within an episode")
        if self.mode != "dagger" or self.state != "ready":
            raise TransitionError("policy may start only from a ready DAgger episode")
        self.state = "policy"
        self.segment += 1

    def start_expert(self) -> None:
        if self.mode not in {"practice", "expert"} or self.state != "ready":
            raise TransitionError("full expert control may start only from a ready practice or expert episode")
        self.state = "expert"
        self.segment += 1

    def request_takeover(self) -> None:
        if self.mode != "dagger" or self.state != "policy":
            raise TransitionError("takeover requires an active DAgger policy segment")
        self.state = "takeover_pending"

    def accept_expert(self, *, current_robot: Iterable[object], leader_command: Iterable[object]) -> None:
        if self.mode != "dagger" or self.state != "takeover_pending":
            raise TransitionError("expert takeover requires a pending DAgger takeover")
        robot = _finite_12(current_robot, "robot state")
        leader = _finite_12(leader_command, "leader command")
        if max(abs(current - command) for current, command in zip(robot, leader)) > self.sync_tolerance_rad:
            raise TransitionError("leader synchronization is outside the takeover tolerance")
        self.policy_queue_clear_requested = True
        self._took_over = True
        self.state = "expert"
        self.segment += 1

    def accept(self, quality: QualityResult) -> bool:
        if self.state != "expert":
            raise TransitionError("accept requires an active expert segment")
        if self.mode == "practice" or not quality.trainable:
            self.state = "diagnostic"
            return False
        self.state = "accepted"
        return True

    def discard(self) -> None:
        if self.state not in {"policy", "takeover_pending", "expert"}:
            raise TransitionError("discard requires an active collection segment")
        self.state = "diagnostic"

    def reset(self) -> None:
        if self.state not in {"accepted", "diagnostic"}:
            raise TransitionError("reset requires a finalized episode")
        self.state = "ready"
        self.segment = 0
        self.policy_queue_clear_requested = False
        self._took_over = False
