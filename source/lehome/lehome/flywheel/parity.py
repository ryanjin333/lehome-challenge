"""Fail-closed contracts for the GR00T rollout parity ladder and safety gates."""

from __future__ import annotations

from dataclasses import dataclass
import math


HISTORICAL_CONTROL_IDS = (
    "top-long-seen-1-seed-42",
    "top-long-seen-2-seed-42",
    "top-long-seen-3-seed-42",
    "top-long-unseen-0-seed-42",
    "top-short-seen-0-seed-42",
    "top-short-seen-1-seed-42",
    "top-short-seen-2-seed-42",
    "pant-short-seen-0-seed-42",
    "pant-short-seen-1-seed-42",
    "pant-short-seen-2-seed-42",
    "pant-short-seen-3-seed-42",
    "pant-short-unseen-1-seed-42",
)


@dataclass(frozen=True, slots=True)
class HistoricalControlTrial:
    category: str
    garment_name: str
    release_stage: str
    seed: int
    trial_id: str


@dataclass(frozen=True, slots=True)
class EpisodeGateEvidence:
    official_success: bool
    visible_contact: bool
    reset_hash: str | None


@dataclass(frozen=True, slots=True)
class AbortGate:
    completed_trials: int = 12

    def __post_init__(self) -> None:
        if not isinstance(self.completed_trials, int) or isinstance(self.completed_trials, bool) or not 12 <= self.completed_trials <= 24:
            raise ValueError("early abort completed_trials must be an integer in 12..24")


@dataclass(frozen=True, slots=True)
class ParityDecision:
    allowed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResetDiversityReport:
    passed: bool
    completed_episodes: int
    unique_hashes: int
    required_unique_hashes: int
    missing_hashes: int


def historical_control_trials() -> tuple[HistoricalControlTrial, ...]:
    """Return the immutable, documented twelve-case historical control."""
    values = (
        ("top_long", "Top_Long_Seen_1", "seen"),
        ("top_long", "Top_Long_Seen_2", "seen"),
        ("top_long", "Top_Long_Seen_3", "seen"),
        ("top_long", "Top_Long_Unseen_0", "public_unseen"),
        ("top_short", "Top_Short_Seen_0", "seen"),
        ("top_short", "Top_Short_Seen_1", "seen"),
        ("top_short", "Top_Short_Seen_2", "seen"),
        ("pant_short", "Pant_Short_Seen_0", "seen"),
        ("pant_short", "Pant_Short_Seen_1", "seen"),
        ("pant_short", "Pant_Short_Seen_2", "seen"),
        ("pant_short", "Pant_Short_Seen_3", "seen"),
        ("pant_short", "Pant_Short_Unseen_1", "public_unseen"),
    )
    return tuple(
        HistoricalControlTrial(category, garment_name, release_stage, 42, trial_id)
        for trial_id, (category, garment_name, release_stage) in zip(HISTORICAL_CONTROL_IDS, values, strict=True)
    )


def evaluate_parity_ladder(*, legacy_server_cpu_successes: int, server_cpu_successes: int, server_cuda_successes: int) -> ParityDecision:
    """Permit scaling only after legacy-server CPU, new server CPU, then CUDA parity."""
    values = (legacy_server_cpu_successes, server_cpu_successes, server_cuda_successes)
    if any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 12 for value in values):
        raise ValueError("parity success counts must be integers in 0..12")
    reasons: list[str] = []
    if legacy_server_cpu_successes < 10:
        reasons.append("legacy_server_cpu_below_10_of_12")
    if server_cpu_successes < 10:
        reasons.append("server_cpu_below_10_of_12")
    if server_cuda_successes < 10:
        reasons.append("server_cuda_below_10_of_12")
    if server_cpu_successes < legacy_server_cpu_successes - 1:
        reasons.append("server_cpu_more_than_one_success_below_legacy_server_cpu")
    if server_cuda_successes < legacy_server_cpu_successes - 1:
        reasons.append("server_cuda_more_than_one_success_below_legacy_server_cpu")
    return ParityDecision(not reasons, tuple(reasons))


def evaluate_cpu_scale_ladder(
    *,
    legacy_server_cpu_successes: int,
    server_cpu_successes: int,
    server_cpu_visible_contacts: int,
    server_cpu_unique_resets: int,
    cuda_abort_successes: int,
    cuda_abort_terminal_trials: int,
) -> ParityDecision:
    """Authorize only the explicitly approved CPU-simulator production path.

    This deliberately does not alter :func:`evaluate_parity_ladder`: CUDA
    production still needs its normal passing parity receipt.  A typed CUDA
    diagnostic abort is rejection evidence here, not a substitute for CUDA
    parity.
    """
    values = (
        legacy_server_cpu_successes,
        server_cpu_successes,
        server_cpu_visible_contacts,
        server_cpu_unique_resets,
        cuda_abort_successes,
        cuda_abort_terminal_trials,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError("CPU scale evidence counts must be integers")
    reasons: list[str] = []
    if not 9 <= legacy_server_cpu_successes <= 12:
        reasons.append("legacy_server_cpu_below_9_of_12")
    if not 10 <= server_cpu_successes <= 12:
        reasons.append("server_cpu_not_10_to_12_of_12")
    if abs(server_cpu_successes - legacy_server_cpu_successes) > 1:
        reasons.append("server_cpu_more_than_one_success_from_legacy_server_cpu")
    if server_cpu_visible_contacts != 12:
        reasons.append("server_cpu_visible_contacts_not_12_of_12")
    if server_cpu_unique_resets != 12:
        reasons.append("server_cpu_unique_resets_not_12_of_12")
    if cuda_abort_successes != 0 or cuda_abort_terminal_trials != 12:
        reasons.append("server_cuda_abort_not_exactly_0_of_12")
    return ParityDecision(not reasons, tuple(reasons))


def evaluate_abort_gate(evidence: list[EpisodeGateEvidence] | tuple[EpisodeGateEvidence, ...], gate: AbortGate) -> dict[str, object] | None:
    """Return a durable-safe abort payload once the first completed cohort is known."""
    if len(evidence) < gate.completed_trials:
        return None
    cohort = evidence[:gate.completed_trials]
    successes = sum(item.official_success for item in cohort)
    contacts = sum(item.visible_contact for item in cohort)
    reason = None
    if successes == 0:
        reason = "zero_official_successes"
    elif contacts == 0:
        reason = "zero_visible_robot_garment_contact"
    if reason is None:
        return None
    return {
        "schema_version": 1,
        "status": "aborted",
        "reason": reason,
        "completed_trials": len(cohort),
        "official_successes": successes,
        "visible_robot_garment_contacts": contacts,
    }


def assess_reset_diversity(evidence: list[EpisodeGateEvidence] | tuple[EpisodeGateEvidence, ...], *, minimum_ratio: float) -> ResetDiversityReport:
    """Require independently recorded canonical reset hashes at campaign close."""
    if not isinstance(minimum_ratio, (int, float)) or not math.isfinite(minimum_ratio) or not 0.0 < minimum_ratio <= 1.0:
        raise ValueError("minimum reset uniqueness ratio must be finite in 0.0..1.0")
    hashes = [item.reset_hash for item in evidence if isinstance(item.reset_hash, str) and len(item.reset_hash) == 64]
    completed = len(evidence)
    required = math.ceil(completed * minimum_ratio)
    unique = len(set(hashes))
    missing = completed - len(hashes)
    return ResetDiversityReport(missing == 0 and unique >= required, completed, unique, required, missing)
