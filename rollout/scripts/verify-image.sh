#!/usr/bin/env bash
set -euo pipefail

image_ref=${1:-}
if ! [[ "$image_ref" =~ ^docker\.io/ryanjin333/behavior1k-groot-n17@sha256:[0-9a-f]{64}$ ]]; then
  echo "image must be the canonical Docker Hub rollout digest reference" >&2
  exit 64
fi

actual_user=$(docker image inspect --format '{{.Config.User}}' "$image_ref")
if [[ "$actual_user" != "" ]]; then
  echo "image Config.User must remain root for token and licensed-asset bootstrap" >&2
  exit 1
fi
parent_digest=$(docker image inspect --format '{{index .Config.Labels "io.lehome.behavior-parent-digest"}}' "$image_ref")
if [[ "$parent_digest" != "sha256:b789b8d8efefda509b37404a676523d6cee81e2860558287cf6c34c2af3b79c7" ]]; then
  echo "image has an unexpected immutable BEHAVIOR parent digest label" >&2
  exit 1
fi
for label in io.lehome.behavior-revision io.lehome.isaac-groot-revision org.opencontainers.image.revision io.lehome.image-role; do
  value=$(docker image inspect --format "{{index .Config.Labels \"$label\"}}" "$image_ref")
  if [[ "$label" == io.lehome.image-role && "$value" != rollout ]]; then
    echo "image label $label must identify the rollout purpose" >&2
    exit 1
  fi
  if [[ "$label" != io.lehome.image-role && ! "$value" =~ ^[0-9a-f]{40}$ ]]; then
    echo "image label $label must be an immutable source revision" >&2
    exit 1
  fi
done
if [[ -z "$(docker image inspect --format '{{json .Config.Healthcheck}}' "$image_ref")" ]] \
  || ! docker image inspect --format '{{json .Config.Healthcheck}}' "$image_ref" | grep -Fq '/opt/conda/envs/behavior/bin/python -m b1k_rollout.cli healthcheck'; then
  echo "image must expose the rollout healthcheck" >&2
  exit 1
fi
if docker image inspect --format '{{json .Config.Env}}' "$image_ref" \
  | grep -Eq 'HF_TOKEN=|hf_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{30,}'; then
  echo "image configuration contains credential material" >&2
  exit 1
fi

docker run --rm --platform linux/amd64 --entrypoint /bin/bash "$image_ref" -euo pipefail -c '
  test "$(id -u)" = 0
  test "$OMNI_KIT_ACCEPT_EULA" = YES
  test "$HEADLESS" = 1
  test -f /behavior-src/OmniGibson/omnigibson/eval/r1pro.yaml
  test -f /opt/isaac-groot/scripts/b1k/serve_b1k.py
  test -f /opt/isaac-groot/gr00t/policy/websocket_b1k_server.py
  test -f /opt/isaac-groot/gr00t/eval/eval_b1k_wrapper.py
  test -f /opt/rollout/task-manifest.json
  /opt/conda/envs/behavior/bin/python -c "import b1k_rollout.cli; import b1k_rollout.policy_server"
  /opt/conda/envs/behavior/bin/python -m b1k_rollout.cli --help >/dev/null
  set +e
  # Scan the shipped release sources, but not generated dependency or Git
  # metadata trees. Third-party test utilities can contain synthetic token
  # fixtures that are not release credentials and are not executable inputs.
  grep -RIE --exclude-dir=.git --exclude-dir=.venv \
    "hf_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{30,}" \
    /opt/rollout /behavior-src /opt/isaac-groot >/dev/null
  secret_status=$?
  set -e
  [[ "$secret_status" -eq 1 ]]
'

# This executes the production entrypoint through its root-only token-file
# bootstrap and setpriv handoff, without fetching licensed assets during a CPU
# publication check. The test flag exits immediately after proving UID/GID and
# secret scrubbing in the unprivileged branch.
docker run --rm --platform linux/amd64 --entrypoint /usr/local/bin/b1k-rollout-entrypoint \
  -e AUTO_DESTROY=0 \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e B1K_ACCEPT_DATASET_TOS=YES \
  -e HF_TOKEN=rollout-verification-token \
  -e B1K_HF_TOKEN_FILE=/workspace/.cache/huggingface/token \
  -e B1K_ROLLOUT_VERIFY_PRIVILEGE_DROP=1 \
  "$image_ref"

echo "verified $image_ref (rollout structural gate)"
