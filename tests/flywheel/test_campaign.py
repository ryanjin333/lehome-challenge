from __future__ import annotations

from lehome.flywheel.artifacts import EpisodeArtifactWriter
from scripts.run_groot_flywheel_campaign import CampaignState, pending_trial_ids


def campaign_state_with_completed_trial(tmp_path, trial_id: str) -> CampaignState:
    writer = EpisodeArtifactWriter(tmp_path, trial_id)
    writer.append_annotation({"step": 0, "action_source": "policy"})
    writer.finalize({"terminal_reason": "horizon", "outcome": "timeout"})
    return CampaignState(output_root=tmp_path, trial_ids=(trial_id, "trial-002"))


def test_campaign_resume_skips_checksum_verified_trials(tmp_path) -> None:
    state = campaign_state_with_completed_trial(tmp_path, "trial-001")
    assert pending_trial_ids(state) == ("trial-002",)
