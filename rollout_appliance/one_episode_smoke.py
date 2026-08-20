"""Pinned identities for the first paid rollout boot.

This module is the operator contract, not a live GPU launcher.  The next
preemptible RTX PRO 6000 boot may run only the one-episode smoke.  The
80-unseen matrix starts only after that smoke leases and finishes.
"""

EVAL_CANDIDATES = [
    "original_baseline",
    "new_step_2k",
]
SMOKE_GARMENT = "Top_Long_Unseen_0"
SMOKE_SEED = 601
POLICY_GATEWAY_PORT = 15555
POLICY_TIMEOUT_SECONDS = 180
ISAAC_UID = 1234
POLICY_SERVER_UID = 10001
NEW_STEP_2K_POLICY_SHA256 = "761b1caacc606466fdd5d5720b4b9da3f2baf1fe929ccc1f50ec7f82094861e5"
TRAINER_IMAGE = (
    "ghcr.io/ryanjin333/lehome-groot-n17-trainer"
    "@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746"
)
PARENT_12K_REVISION = "30ac1a84da67b099e115ad147bcd61e9d60046d3"
PARENT_12K_SUBPATH = "policies/step-12000"
PARENT_12K_ARTIFACT_SHA256 = "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06"
SKIP_OLD_CAMPAIGN = ("previous_step_1k", "previous_step_2k", "rft/c87b1861")


# Live 2026-08-17 2K smoke result on computeinstance-u00ytn0r4csts9608t:
# leased Top_Long_Unseen_0 / 601 and finished 600 steps. Policy metrics
# reached model_calls=38. Extra concurrent Isaac workers currently crash
# (omni.kit.usd / Kit extension cache). Isolated /kitcache per worker is
# required before four-worker eval is stable.
SMOKE_FINISHED = True
