# LeHome GR00T N1.7 trainer

This isolated Python 3.10 package prepares the organizer demonstrations and
runs NVIDIA GR00T N1.7 behavior-cloning training without Isaac Sim. Simulation
and rollout evaluation remain on a separate LeHome-capable machine.

The pinned environment is installed from this directory:

```bash
uv sync --locked
uv run lehome-train --help
```

Remote data is stored only in the approved private dataset repository
`ryanjin333/lehome-groot-n17-data`. Checkpoints, reports, redacted logs, and
provenance are stored only in the approved private model repository
`ryanjin333/lehome-groot-n17-models`. Remote operations require `HF_TOKEN` in
the current process environment. The trainer passes it explicitly to Hub calls,
does not invoke `hf auth login`, does not put it in a child environment, and
does not write it to a credential cache, report, manifest, or upload.

Repository creation is never implicit. An owner can explicitly create or
verify either approved private repository with the library API:

```bash
HF_TOKEN='<write-scoped-token>' uv run python - <<'PY'
from lehome_train.hub import (
    HuggingFaceHubTransport,
    ensure_approved_private_repository,
)

transport = HuggingFaceHubTransport(timeout_seconds=30.0)
for repository in (
    "ryanjin333/lehome-groot-n17-data",
    "ryanjin333/lehome-groot-n17-models",
):
    ensure_approved_private_repository(
        transport=transport,
        repository=repository,
        create=True,
        timeout_seconds=30.0,
    )
PY
```

Use `create=False` to verify existing repositories without creating them. Any
unapproved repository or repository that is not private is rejected.

The complete operator workflow, request schemas, normalization contract,
restore choices, and shutdown gate are in
[the GR00T N1.7 training runbook](../docs/groot_n17_training.md).

The accepted Task 12 image sets `LEHOME_TRAIN_RUNTIME_FACTORY=module:factory`.
Outside that image, pass `--runtime-factory module:factory` explicitly. The
factory must return the production adapter implementing `prepare`, `memorize`,
`smoke`, and `train`; missing GPU/runtime wiring is a hard error, never a
successful no-op.

Every stage consumes a checked JSON request:

```bash
lehome-train prepare --request /requests/prepare.json
lehome-train memorize --request /requests/memorize.json
lehome-train smoke --request /requests/smoke.json
lehome-train train --request /requests/train.json
lehome-train report --request /requests/report.json
lehome-train sync --request /requests/sync.json
```

Run the focused report and synchronization checks with:

```bash
uv run pytest tests/test_report.py tests/test_sync.py -q
```

These tests use injected runtimes and an in-memory transport. Creating
repositories or performing a real upload additionally requires a valid
`HF_TOKEN`; no token is bundled in this checkout.
