# BEHAVIOR-1K R1Pro rollout image

This package is the headless, private rollout half of the B1K deployment. It
uses the immutable 100-task R1Pro manifest with public-test instances `0..9`,
so one campaign represents exactly 1,000 requested evaluator episodes. It
does not enable DAgger, hard-state resets, RoboTTT, desktop services, VNC, or
noVNC.

The template consumes only a Docker Hub digest and an account-mounted token
file at `/workspace/.cache/huggingface/token`; it contains no registry or Hub
credential value. `AUTO_DESTROY=0` is mandatory. The image entrypoint rejects
missing simulator assets, unpinned identities, a weak token-file mode, and a
final manifest whose exact bytes do not match `CHECKPOINT_ARTIFACT_SHA256`.
Only nonempty `checkpoint/**` entries from strict manifest schema v1 are
materialized. Each file is streamed to same-filesystem staging, size/hash
checked, then atomically promoted with a marker bound to the model repository,
immutable commit, manifest hash, run ID, and exact checkpoint file set.

Render the secret-free template fixture locally:

```bash
uv run --project rollout python - <<'PY'
from b1k_rollout.template import render_vast_template_fixture
print(render_vast_template_fixture(image_digest="sha256:" + "0" * 64), end="")
PY
```

The image parent is deliberately a required build argument. It must be the
resolved digest of the official compatible BEHAVIOR/Isaac runtime, never a
tag:

```bash
docker buildx build --platform linux/amd64 -f rollout/Dockerfile \
  --build-arg BEHAVIOR_PARENT_IMAGE='registry.example/behavior@sha256:<64hex>' \
  --build-arg BEHAVIOR_PARENT_DIGEST='sha256:<64hex>' \
  --build-arg REPOSITORY_COMMIT='<40hex>' .
```

No image build, push, template publication, or Vast rental is performed by
these local acceptance assets. The production CLI delegates policy serving to
the pinned upstream B1K `serve_b1k.py` server with its R1Pro modality wrapper;
it does not reimplement action mapping or temporal ensembling. It still
requires the selected model commit/artifact hash and a verified parent-image
digest.
