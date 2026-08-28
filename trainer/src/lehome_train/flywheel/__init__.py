"""Fail-closed contracts for the offline GR00T data flywheel."""

# Do not import the parquet materializer at package import time.  The rollout
# round seal is deliberately dependency-light and must remain usable on the
# appliance/publisher image, which does not install the trainer-only pyarrow
# stack.
__all__ = ("MaterializationReport", "materialize_episode", "materialize_rft_episode")


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from lehome_train.flywheel import materialize

    return getattr(materialize, name)
