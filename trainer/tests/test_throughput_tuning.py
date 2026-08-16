from lehome_train.groot.throughput_tuning import LOADER_CANDIDATES, TrainingProbe, tune_on_host, tune_training


def probe(workers: int, batch: int, rate: float, *, oom: bool = False) -> TrainingProbe:
    return TrainingProbe(workers, batch, rate, not oom, not oom, 20.0, 0.5)


def test_tuner_selects_only_the_fixed_batch64_loader_candidates() -> None:
    report = tune_training(
        loader_results=[probe(0, 64, 10), probe(4, 64, 14), probe(8, 64, 13), probe(12, 64, 12), probe(16, 64, 11)],
        batch_results=[],
    )
    assert LOADER_CANDIDATES == (0, 4, 8, 12, 16)
    assert report.selected_loader_workers == 4
    assert report.fastest_stable_physical_batch == 64
    assert report.production_physical_batch == 64


def test_tuner_never_sweeps_batch_size() -> None:
    attempted: list[tuple[int, int]] = []
    tune_on_host(run=lambda workers, batch: attempted.append((workers, batch)) or probe(workers, batch, 1))
    assert attempted == [(workers, 64) for workers in (0, 4, 8, 12, 16)]
