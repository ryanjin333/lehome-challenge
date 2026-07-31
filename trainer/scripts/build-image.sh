#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"

repository_commit=$(git rev-parse HEAD)
if [[ ! "$repository_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "unable to resolve an immutable repository commit" >&2
  exit 65
fi
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" && "${ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "refusing to label a dirty checkout with immutable commit $repository_commit" >&2
  echo "commit the intended image inputs first, or set ALLOW_DIRTY=1 for a non-release diagnostic build" >&2
  exit 65
fi

image_repository=${IMAGE_REPOSITORY:-lehome-groot-n17-trainer}
image_ref="${image_repository}:${repository_commit}"

docker buildx build \
  --platform linux/amd64 \
  --load \
  --build-arg "REPOSITORY_COMMIT=${repository_commit}" \
  --tag "$image_ref" \
  -f trainer/Dockerfile \
  .

printf '%s\n' "$image_ref"

