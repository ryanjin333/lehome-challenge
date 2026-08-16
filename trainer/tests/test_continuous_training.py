from pathlib import Path
from concurrent.futures import Future
import json
import threading
from time import monotonic
from types import SimpleNamespace

import pytest

from lehome_train.groot.continuous_training import run_continuous_supervisor, snapshot_checkpoint


@pytest.fixture(autouse=True)
def _complete_official_checkpoint_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test-created official checkpoint carries the pinned resume files."""

    original_mkdir = Path.mkdir

    def mkdir(path: Path, *args: object, **kwargs: object) -> None:
        original_mkdir(path, *args, **kwargs)
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdecimal():
            for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                artifact = path / name
                if not artifact.exists():
                    artifact.write_bytes(name.encode("ascii"))

    monkeypatch.setattr(Path, "mkdir", mkdir)


@pytest.mark.parametrize(
    "admission",
    (
        {"already_published": (500,)},
        {"local_recovery_root": Path("/tmp/local-recovery")},
        {"initial_immutable_publication": {"optimizer_step": 1000}},
    ),
)
def test_invalid_supervisor_admission_creates_no_training_or_publisher_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, admission: dict[str, object],
) -> None:
    import lehome_train.groot.continuous_training as continuous

    launched: list[object] = []

    def no_thread(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid admission must not construct a worker")

    monkeypatch.setattr(continuous.threading, "Thread", no_thread)
    with pytest.raises(ValueError):
        run_continuous_supervisor(
            run_root=tmp_path, launch=lambda: launched.append("launch"),
            package=lambda item: item, publish=lambda _item: True, **admission,  # type: ignore[arg-type]
        )
    assert launched == []


@pytest.mark.parametrize(
    "admission",
    (
        {"already_published": (2000,)},
        {"already_published": (1000, 2000)},
        {
            "already_published": (1000,),
            "initial_immutable_publication": {"optimizer_step": 1000, "readback_verified": True},
            "initial_immutable_anchor": {"immutable_anchor_revision": "1" * 40, "anchor_sha256": "2" * 64},
        },
        {
            "initial_immutable_publication": {"optimizer_step": 1000, "readback_verified": True, "immutable_revision": "1" * 40},
            "initial_immutable_anchor": {"immutable_anchor_revision": "1" * 40, "anchor_sha256": "2" * 64},
        },
    ),
)
def test_recovery_admission_requires_the_exact_one_k_readback_state_before_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, admission: dict[str, object],
) -> None:
    import lehome_train.groot.continuous_training as continuous

    monkeypatch.setattr(
        continuous.threading, "Thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not construct worker")),
    )
    with pytest.raises(ValueError, match="admission|already-published|immutable"):
        run_continuous_supervisor(
            run_root=tmp_path, launch=lambda: None, package=lambda item: item,
            publish=lambda _item: True, **admission,  # type: ignore[arg-type]
        )


def test_recovery_admission_rejects_malformed_identity_and_unsafe_root_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lehome_train.groot.continuous_training as continuous

    root = tmp_path / "missing" / "local-recovery"
    bad_identity = {
        "experiment_manifest_sha256": "a" * 64,
        "parent_checkpoint_artifact_sha256": "b" * 64,
        "runtime_mixture_id": "c" * 64,
        "trainer_code_sha256": "not-a-digest",
        "trainer_code_revision": "e" * 40,
    }
    monkeypatch.setattr(
        continuous.threading, "Thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not construct worker")),
    )
    with pytest.raises(ValueError, match="identity"):
        run_continuous_supervisor(
            run_root=tmp_path, launch=lambda: None, package=lambda item: item, publish=lambda _item: True,
            local_recovery_root=root, local_identity=bad_identity,
        )
    assert not root.parent.exists()


@pytest.mark.parametrize("kind", ("symlink", "file"))
def test_recovery_admission_rejects_existing_unsafe_root_before_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    import os
    import lehome_train.groot.continuous_training as continuous

    root = tmp_path / "local-recovery"
    if kind == "symlink":
        target = tmp_path / "outside"; target.mkdir()
        os.symlink(target, root, target_is_directory=True)
    else:
        root.write_text("not a directory", encoding="utf-8")
    identity = {
        "experiment_manifest_sha256": "a" * 64,
        "parent_checkpoint_artifact_sha256": "b" * 64,
        "runtime_mixture_id": "c" * 64,
        "trainer_code_sha256": "d" * 64,
        "trainer_code_revision": "e" * 40,
    }
    monkeypatch.setattr(
        continuous.threading, "Thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not construct worker")),
    )
    with pytest.raises(ValueError, match="root"):
        run_continuous_supervisor(
            run_root=tmp_path, launch=lambda: None, package=lambda item: item, publish=lambda _item: True,
            local_recovery_root=root, local_identity=identity,
        )


def test_observer_never_packages_checkpoint_without_completion_marker(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="complete checkpoint"):
        snapshot_checkpoint(checkpoint, optimizer_step=1000)


def test_snapshot_is_independent_byte_copy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"original")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1000, "log_history": [{"step": 1000, "loss": 0.25}]}))
    snapshot = snapshot_checkpoint(checkpoint, optimizer_step=1000)
    (checkpoint / "weights.bin").write_bytes(b"changed")
    assert (snapshot.snapshot_root / "weights.bin").read_bytes() == b"original"


def test_snapshot_accepts_a_complete_indexed_safetensors_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").unlink(missing_ok=True)
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"first-shard")
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"second-shard")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": 22},
            "weight_map": {
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors",
            },
        }),
        encoding="utf-8",
    )
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    (checkpoint / "rng_state.pth").write_bytes(b"rng")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 1000, "log_history": [{"step": 1000, "loss": .25}]}),
        encoding="utf-8",
    )

    snapshot = snapshot_checkpoint(checkpoint, optimizer_step=1000)

    assert (snapshot.snapshot_root / "model.safetensors.index.json").is_file()
    assert (snapshot.snapshot_root / "model-00002-of-00002.safetensors").read_bytes() == b"second-shard"


def test_snapshot_cancellation_leaves_no_partial_copy_and_a_later_retry_recovers(tmp_path: Path) -> None:
    from lehome_train.groot.continuous_training import SnapshotCancelled

    checkpoint = tmp_path / "checkpoint-1000"; checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"x" * (2 * 1024 * 1024))
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 1000, "log_history": [{"step": 1000, "loss": .25}]}),
    )
    calls = 0

    def cancel_mid_copy() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 4

    with pytest.raises(SnapshotCancelled, match="cancelled"):
        snapshot_checkpoint(checkpoint, optimizer_step=1000, cancel_requested=cancel_mid_copy)
    assert not list(tmp_path.glob(".checkpoint-1000.snapshot-1000*.incomplete"))
    assert not (tmp_path / ".checkpoint-1000.snapshot-1000").exists()
    assert snapshot_checkpoint(checkpoint, optimizer_step=1000).snapshot_root.is_dir()


def test_supervisor_reads_completed_checkpoints_not_caller_steps(tmp_path: Path) -> None:
    seen: list[int] = []
    def launch() -> None:
        for step in (1000, 2000):
            checkpoint = tmp_path / f"checkpoint-{step}"; checkpoint.mkdir()
            (checkpoint / "weights.bin").write_bytes(b"weights")
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step, "log_history": [{"step": step, "loss": 0.2}]}))
    def publish(item: object) -> dict[str, object]:
        step = item.optimizer_step  # type: ignore[attr-defined]
        seen.append(step)
        return {
            "optimizer_step": step, "readback_verified": True,
            "runtime_checkpoint_anchor": {
                "immutable_anchor_revision": "a" * 40,
                "anchor_sha256": "b" * 64,
            },
        }

    assert [item["optimizer_step"] for item in run_continuous_supervisor(run_root=tmp_path, launch=launch, package=lambda item: item, publish=publish)] == [1000, 2000]
    assert seen == [1000, 2000]


def test_supervisor_returns_published_checkpoint_after_interrupt(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1000"; checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1000, "log_history": [{"step": 1000, "loss": 0.2}]}))
    assert [item["optimizer_step"] for item in run_continuous_supervisor(run_root=tmp_path, launch=lambda: (_ for _ in ()).throw(KeyboardInterrupt()), package=lambda item: item, publish=lambda item: {"optimizer_step": item.optimizer_step, "readback_verified": True})] == [1000]


def test_supervisor_does_not_call_data_failure_a_resumable_interrupt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bad dataset"):
        run_continuous_supervisor(
            run_root=tmp_path,
            launch=lambda: (_ for _ in ()).throw(ValueError("bad dataset")),
            package=lambda item: item,
            publish=lambda item: {"optimizer_step": item.optimizer_step, "readback_verified": True},
        )


def test_supervisor_attests_every_official_500_boundary_but_only_publishes_one_k(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import discover_local_recovery

    identity = {
        "experiment_manifest_sha256": "a" * 64,
        "parent_checkpoint_artifact_sha256": "b" * 64,
        "runtime_mixture_id": "c" * 64,
        "trainer_code_sha256": "d" * 64,
        "trainer_code_revision": "e" * 40,
    }
    seen: list[int] = []

    def launch() -> None:
        for step in (500, 1000):
            checkpoint = tmp_path / f"checkpoint-{step}"
            checkpoint.mkdir()
            (checkpoint / "weights.bin").write_bytes(b"weights")
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step, "log_history": [{"step": step, "loss": 0.25}]}))

    published = run_continuous_supervisor(
        run_root=tmp_path, launch=launch, package=lambda item: item,
        publish=lambda item: seen.append(item.optimizer_step) or {"optimizer_step": item.optimizer_step, "readback_verified": True},
        local_recovery_root=tmp_path / "shared", local_identity=identity,
    )

    assert [item["optimizer_step"] for item in published] == [1000]
    assert seen == [1000]
    assert discover_local_recovery(metadata_root=tmp_path / "shared", identity=identity).optimizer_step == 1000


def test_supervisor_defers_post_one_k_local_attestation_until_async_anchor_is_verified(tmp_path: Path) -> None:
    """A fast trainer must not make an unanchored 1500 receipt or abort."""
    from lehome_train.groot.local_recovery import discover_local_recovery

    identity = {
        "experiment_manifest_sha256": "a" * 64, "parent_checkpoint_artifact_sha256": "b" * 64,
        "runtime_mixture_id": "c" * 64, "trainer_code_sha256": "d" * 64,
        "trainer_code_revision": "e" * 40,
    }
    publication_started, release_one_k = threading.Event(), threading.Event()

    def launch() -> None:
        for step in (500, 1000, 1500, 2000):
            checkpoint = tmp_path / f"checkpoint-{step}"
            checkpoint.mkdir()
            (checkpoint / "weights.bin").write_bytes(f"weights-{step}".encode())
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": step, "log_history": [{"step": step, "loss": .25}]})
            )

    def publish(item: object) -> dict[str, object]:
        step = item.optimizer_step  # type: ignore[attr-defined]
        if step == 1000:
            publication_started.set()
            assert release_one_k.wait(timeout=2.0)
        return {
            "optimizer_step": step, "readback_verified": True,
            "immutable_revision": ("1" if step == 1000 else "2") * 40,
            "runtime_checkpoint_anchor": {
                "immutable_anchor_revision": ("1" if step == 1000 else "2") * 40,
                "anchor_sha256": ("3" if step == 1000 else "4") * 64,
            },
        }

    waited = False

    def wait() -> None:
        nonlocal waited
        if not waited and publication_started.wait(timeout=.05):
            waited = True
            release_one_k.set()

    published = run_continuous_supervisor(
        run_root=tmp_path, launch=launch, package=lambda item: item, publish=publish,
        wait=wait, local_recovery_root=tmp_path / "shared", local_identity=identity,
    )

    assert waited
    assert [item["optimizer_step"] for item in published] == [1000, 2000]
    local = discover_local_recovery(metadata_root=tmp_path / "shared", identity=identity)
    assert local.optimizer_step == 2000
    assert local.last_immutable_publication == {"optimizer_step": 1000, "readback_verified": True, "immutable_revision": "1" * 40}
    assert local.terminal_immutable_publication == {"optimizer_step": 2000, "readback_verified": True, "immutable_revision": "2" * 40}


def test_finished_supervisor_keeps_polling_pending_one_k_until_it_unlocks_two_k(tmp_path: Path) -> None:
    release_one_k = threading.Event()
    published: list[int] = []

    for step in (1000, 2000):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "weights.bin").write_bytes(f"weights-{step}".encode())
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step, "log_history": [{"step": step, "loss": .25}]}),
        )

    def publish(item: object) -> dict[str, object]:
        step = item.optimizer_step  # type: ignore[attr-defined]
        published.append(step)
        if step == 1000:
            assert release_one_k.wait(timeout=3.0)
        return {
            "optimizer_step": step, "readback_verified": True,
            "runtime_checkpoint_anchor": {
                "immutable_anchor_revision": ("1" if step == 1000 else "2") * 40,
                "anchor_sha256": ("3" if step == 1000 else "4") * 64,
            },
        }

    def release_after_old_finished_poll_budget() -> None:
        # The default observer waits 100 ms per finished poll; releasing after
        # 2.1 s guarantees the old 20-poll escape hatch has fired first.
        threading.Event().wait(2.1)
        release_one_k.set()

    threading.Thread(target=release_after_old_finished_poll_budget, daemon=True).start()

    result = run_continuous_supervisor(
        run_root=tmp_path, launch=lambda: None, package=lambda item: item,
        publish=publish,
    )

    assert published == [1000, 2000]
    assert [item["optimizer_step"] for item in result] == [1000, 2000]


def test_finished_poll_gate_keeps_a_done_but_unresolved_one_k_observable() -> None:
    from lehome_train.groot.continuous_training import _has_unresolved_submitted_publication

    future: Future[object] = Future()
    future.set_result({"optimizer_step": 1000})

    # This is the narrow race: the future completes after the loop's top
    # resolve pass but before the finished-poll cutoff is evaluated.
    assert _has_unresolved_submitted_publication({1000: future}, {}) is True
    assert _has_unresolved_submitted_publication({1000: future}, {1000: future.result()}) is False


def test_preemption_controller_finalizes_only_an_already_complete_local_boundary(tmp_path: Path) -> None:
    from lehome_train.groot.continuous_training import PreemptionController
    from lehome_train.groot.local_recovery import discover_local_recovery

    identity = {
        "experiment_manifest_sha256": "a" * 64, "parent_checkpoint_artifact_sha256": "b" * 64,
        "runtime_mixture_id": "c" * 64, "trainer_code_sha256": "d" * 64,
        "trainer_code_revision": "e" * 40,
    }
    checkpoint = tmp_path / "checkpoint-500"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 500, "log_history": [{"step": 500, "loss": .25}]}))
    controller = PreemptionController()
    controller.handler(15, None)

    result = run_continuous_supervisor(
        run_root=tmp_path, launch=lambda: None, package=lambda item: item, publish=lambda _item: pytest.fail("preemption must not publish"),
        local_recovery_root=tmp_path / "shared", local_identity=identity, preemption=controller,
    )

    assert result == ()
    assert controller.status() == {"requested": True, "signal_number": 15, "finalized_step": None}
    assert discover_local_recovery(metadata_root=tmp_path / "shared", identity=identity) is None


def test_preemption_already_requested_never_invokes_the_training_launch(tmp_path: Path) -> None:
    from lehome_train.groot.continuous_training import PreemptionController

    controller = PreemptionController()
    controller.request(15)
    launched: list[bool] = []

    result = run_continuous_supervisor(
        run_root=tmp_path, launch=lambda: launched.append(True), package=lambda item: item,
        publish=lambda _item: pytest.fail("preempted supervisor must not publish"),
        preemption=controller,
    )

    assert result == ()
    assert launched == []


def test_preemption_cancels_queued_daemon_work_before_package_or_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lehome_train.groot.continuous_training as continuous
    from concurrent.futures import Future

    checkpoint = tmp_path / "checkpoint-1000"; checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 1000, "log_history": [{"step": 1000, "loss": .25}]}),
    )
    controller = continuous.PreemptionController()
    queued: list[object] = []

    class DeferredDaemonPublisher:
        def __init__(self, *, cancel_requested: object = None) -> None:
            self.thread = SimpleNamespace(daemon=True)
            self.futures: list[Future[object]] = []

        def submit(self, work: object) -> Future[object]:
            assert callable(work)
            queued.append(work)
            future: Future[object] = Future()
            self.futures.append(future)
            return future

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            if cancel_futures:
                for future in self.futures:
                    future.cancel()

    monkeypatch.setattr(continuous, "_DaemonSerialPublisher", DeferredDaemonPublisher)

    def wait() -> None:
        assert len(queued) == 1
        controller.request(15)

    packaged: list[object] = []
    published: list[object] = []
    result = continuous.run_continuous_supervisor(
        run_root=tmp_path, launch=lambda: None,
        package=lambda item: packaged.append(item) or item,
        publish=lambda item: published.append(item) or True,
        wait=wait, preemption=controller,
    )

    assert result == ()
    assert packaged == published == []
    # Simulate a worker that had dequeued the callback as SIGTERM arrived.
    queued[0]()
    assert packaged == published == []


def test_preemption_during_stuck_two_k_future_never_blocks_normal_collection(tmp_path: Path) -> None:
    from lehome_train.groot.continuous_training import PreemptionController

    checkpoint = tmp_path / "checkpoint-2000"; checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 2000, "log_history": [{"step": 2000, "loss": .25}]}),
    )
    controller = PreemptionController()
    publishing, release = threading.Event(), threading.Event()

    def publish(_item: object) -> dict[str, object]:
        publishing.set()
        assert release.wait(timeout=2.0)
        return {"optimizer_step": 2000, "readback_verified": True}

    def request_after_two_k_starts() -> None:
        assert publishing.wait(timeout=2.0)
        threading.Event().wait(.02)
        controller.request(15)

    requester = threading.Thread(target=request_after_two_k_starts, daemon=True)
    requester.start()
    started = monotonic()
    result = run_continuous_supervisor(
        run_root=tmp_path, launch=lambda: None, package=lambda item: item, publish=publish,
        already_published=(1000,),
        initial_immutable_publication={"optimizer_step": 1000, "readback_verified": True, "immutable_revision": "1" * 40},
        initial_immutable_anchor={"immutable_anchor_revision": "1" * 40, "anchor_sha256": "2" * 64},
        preemption=controller,
    )
    elapsed = monotonic() - started

    assert result == ()
    assert elapsed < 1.0
    release.set()


def test_preemption_returns_without_waiting_for_a_blocked_hub_publication(tmp_path: Path) -> None:
    from lehome_train.groot.continuous_training import PreemptionController

    identity = {
        "experiment_manifest_sha256": "a" * 64, "parent_checkpoint_artifact_sha256": "b" * 64,
        "runtime_mixture_id": "c" * 64, "trainer_code_sha256": "d" * 64,
        "trainer_code_revision": "e" * 40,
    }
    checkpoint = tmp_path / "checkpoint-1000"; checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1000, "log_history": [{"step": 1000, "loss": .25}]}))
    controller = PreemptionController()
    publishing, release, publisher_finished = threading.Event(), threading.Event(), threading.Event()

    def publish(_item: object) -> dict[str, object]:
        publishing.set()
        assert release.wait(timeout=2.0)
        publisher_finished.set()
        return {"optimizer_step": 1000, "readback_verified": True}

    def wait() -> None:
        if publishing.wait(timeout=.05):
            controller.request(15)

    started = monotonic()
    result = run_continuous_supervisor(
        run_root=tmp_path, launch=lambda: None, package=lambda item: item, publish=publish,
        wait=wait, local_recovery_root=tmp_path / "shared", local_identity=identity,
        preemption=controller,
    )
    elapsed = monotonic() - started

    assert result == ()
    assert elapsed < 1.0
    assert controller.status() == {"requested": True, "signal_number": 15, "finalized_step": 1000}
    workers = [thread for thread in threading.enumerate() if thread.name == "checkpoint-publisher"]
    assert len(workers) == 1
    # A blocked Hub call must not be joined by Python executor shutdown hooks
    # while the preempted process is trying to exit.
    assert workers[0].daemon is True
    assert not publisher_finished.is_set()
    release.set()
    assert publisher_finished.wait(timeout=2.0)
    workers[0].join(timeout=2.0)
    assert not workers[0].is_alive()


def test_preemption_journals_a_completed_publication_without_restarting_attestation(tmp_path: Path) -> None:
    from lehome_train.groot.continuous_training import PreemptionController
    from lehome_train.groot.local_recovery import discover_local_recovery

    identity = {
        "experiment_manifest_sha256": "a" * 64, "parent_checkpoint_artifact_sha256": "b" * 64,
        "runtime_mixture_id": "c" * 64, "trainer_code_sha256": "d" * 64,
        "trainer_code_revision": "e" * 40,
    }
    checkpoint = tmp_path / "checkpoint-1000"; checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 1000, "log_history": [{"step": 1000, "loss": .25}]}),
    )
    controller = PreemptionController()
    publish_finished = threading.Event()
    waits = 0
    receipt = {
        "optimizer_step": 1000, "readback_verified": True, "immutable_revision": "1" * 40,
        "runtime_checkpoint_anchor": {
            "immutable_anchor_revision": "1" * 40, "anchor_sha256": "2" * 64,
        },
    }

    def publish(_item: object) -> dict[str, object]:
        publish_finished.set()
        return receipt

    def wait() -> None:
        nonlocal waits
        waits += 1
        if waits == 1:
            assert publish_finished.wait(timeout=2.0)
        else:
            controller.request(15)

    result = run_continuous_supervisor(
        run_root=tmp_path, launch=lambda: None, package=lambda item: item, publish=publish,
        wait=wait, local_recovery_root=tmp_path / "shared", local_identity=identity,
        preemption=controller,
    )

    assert result == (receipt,)
    assert controller.status()["finalized_step"] == 1000
    recovered = discover_local_recovery(metadata_root=tmp_path / "shared", identity=identity)
    assert recovered is not None
    assert recovered.last_immutable_publication == {
        "optimizer_step": 1000, "readback_verified": True, "immutable_revision": "1" * 40,
    }


def test_daemon_serial_publisher_propagates_normal_path_failures(tmp_path: Path) -> None:
    """The daemon escape hatch changes preemption only, not normal failures."""
    for step in (1000, 2000):
        checkpoint = tmp_path / f"checkpoint-{step}"; checkpoint.mkdir()
        (checkpoint / "weights.bin").write_bytes(b"weights")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step, "log_history": [{"step": step, "loss": .25}]}),
        )

    published: list[int] = []

    def publish(item: object) -> bool:
        step = item.optimizer_step  # type: ignore[attr-defined]
        published.append(step)
        if step == 1000:
            raise RuntimeError("Hub upload failed")
        return True

    with pytest.raises(RuntimeError, match="Hub upload failed"):
        run_continuous_supervisor(
            run_root=tmp_path, launch=lambda: None, package=lambda item: item,
            publish=publish,
        )
    assert published == [1000]


def test_preemption_recovers_local_1500_while_one_k_hub_publication_is_blocked(tmp_path: Path) -> None:
    from lehome_train.groot.continuous_training import PreemptionController
    from lehome_train.groot.local_recovery import discover_local_recovery

    identity = {
        "experiment_manifest_sha256": "a" * 64, "parent_checkpoint_artifact_sha256": "b" * 64,
        "runtime_mixture_id": "c" * 64, "trainer_code_sha256": "d" * 64,
        "trainer_code_revision": "e" * 40,
    }
    for step in (500, 1000, 1500):
        checkpoint = tmp_path / f"checkpoint-{step}"; checkpoint.mkdir()
        (checkpoint / "weights.bin").write_bytes(b"weights")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step, "log_history": [{"step": step, "loss": .25}]}),
        )
    controller = PreemptionController()
    publishing, release = threading.Event(), threading.Event()
    published: list[int] = []

    def publish(item: object) -> dict[str, object]:
        published.append(item.optimizer_step)  # type: ignore[attr-defined]
        publishing.set()
        assert release.wait(timeout=2.0)
        return {"optimizer_step": 1000, "readback_verified": True}

    def wait() -> None:
        if publishing.wait(timeout=.05):
            controller.request(15)

    result = run_continuous_supervisor(
        run_root=tmp_path, launch=lambda: None, package=lambda item: item, publish=publish,
        wait=wait, local_recovery_root=tmp_path / "shared", local_identity=identity,
        preemption=controller,
    )

    assert result == ()
    assert published == [1000]
    assert controller.status()["finalized_step"] == 1500
    recovered = discover_local_recovery(metadata_root=tmp_path / "shared", identity=identity)
    assert recovered is not None and recovered.optimizer_step == 1500
    assert recovered.last_immutable_anchor is None
    release.set()


def test_publication_only_supervisor_finishes_local_2k_without_launching_gradients(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import attest_local_checkpoint, discover_local_recovery

    identity = {
        "experiment_manifest_sha256": "a" * 64, "parent_checkpoint_artifact_sha256": "b" * 64,
        "runtime_mixture_id": "c" * 64, "trainer_code_sha256": "d" * 64,
        "trainer_code_revision": "e" * 40,
    }
    checkpoint = tmp_path / "checkpoint-2000"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 2000, "log_history": [{"step": 2000, "loss": .25}]}))
    metadata = tmp_path / "shared"
    attest_local_checkpoint(
        checkpoint=checkpoint, metadata_root=metadata, optimizer_step=2000, identity=identity,
    )
    publication = {
        "optimizer_step": 2000, "readback_verified": True, "immutable_revision": "1" * 40,
        "runtime_checkpoint_anchor": {"immutable_anchor_revision": "1" * 40, "anchor_sha256": "2" * 64},
    }

    published = run_continuous_supervisor(
        run_root=tmp_path, launch=lambda: None,
        package=lambda item: item, publish=lambda _item: publication,
        already_published=(1000,), local_recovery_root=metadata, local_identity=identity,
        initial_immutable_publication={"optimizer_step": 1000, "readback_verified": True, "immutable_revision": "f" * 40},
        initial_immutable_anchor={"immutable_anchor_revision": "f" * 40, "anchor_sha256": "0" * 64},
    )

    assert [item["optimizer_step"] for item in published] == [2000]
    assert discover_local_recovery(metadata_root=metadata, identity=identity).terminal_immutable_publication == {
        "optimizer_step": 2000, "readback_verified": True, "immutable_revision": "1" * 40,
    }
