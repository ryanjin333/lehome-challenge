from lehome_train.groot.throughput_tuning import TrainingProbe, tune_on_host, tune_training


def probe(workers: int, batch: int, rate: float, *, oom: bool = False) -> TrainingProbe:
    return TrainingProbe(workers, batch, rate, not oom, not oom, 20.0, 0.5)


def test_tuner_selects_loader_but_keeps_first_run_at_verified_batch() -> None:
    report = tune_training(
        loader_results=[probe(4, 64, 10), probe(8, 64, 14), probe(12, 64, 13)],
        batch_results=[probe(8, 64, 640), probe(8, 96, 800), probe(8, 128, 780)],
    )
    assert report.selected_loader_workers == 8
    assert report.fastest_stable_physical_batch == 96
    assert report.production_physical_batch == 64


def test_tuner_never_tries_larger_batch_after_proven_oom() -> None:
    attempted: list[int] = []
    tune_on_host(run=lambda workers, batch: attempted.append(batch) or probe(workers, batch, 1, oom=batch == 96))
    assert attempted[-2:] == [64, 96]
    assert 128 not in attempted
